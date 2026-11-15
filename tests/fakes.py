"""Test doubles shared by the agent tests: a scripted model instead of Gemini, a fake BigQuery runner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from retail_agent.agent.graph import build_graph
from retail_agent.agent.session import AgentSession
from retail_agent.config import Settings
from retail_agent.data.bq_client import QueryResult

ROOT = Path(__file__).resolve().parents[1]


class ScriptedModel:
    """Returns the scripted replies in order (raising those that are exceptions) and records its inputs."""

    def __init__(self, *replies: AIMessage) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):  # scripted failure
            raise reply
        return reply


class FakeRunner:
    def __init__(self, df: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.df = df if df is not None else pd.DataFrame({"department": ["Women"], "n": [3]})
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def run(self, sql, *, labels=None):
        self.calls.append((sql, dict(labels or {})))
        if self.error:
            raise self.error
        return QueryResult(
            df=self.df, total_rows=len(self.df), truncated=False, bytes_billed=10_485_760,
            bytes_processed=1, cache_hit=False, job_id="job", elapsed_ms=5,
        )  # fmt: skip


class SequenceRunner(FakeRunner):
    """Each call returns (DataFrame) or raises (Exception) the next outcome."""

    def __init__(self, *outcomes) -> None:
        super().__init__()
        self.outcomes = list(outcomes)

    def run(self, sql, *, labels=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            self.calls.append((sql, dict(labels or {})))
            raise outcome
        self.df = outcome
        return super().run(sql, labels=labels)


def sql_call(sql: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content=[], tool_calls=[{"name": "run_sql", "args": {"sql": sql, "purpose": "test"}, "id": call_id}]
    )


def make_agent(settings: Settings, model: ScriptedModel, runner: FakeRunner, user: str = "alice"):
    session = AgentSession.create(settings, user, runner=runner, use_embeddings=False)
    return session, build_graph(session, model, today=date(2026, 11, 15))


def ask(graph, text: str, thread: str = "t1", trace: str = "trace1") -> dict:
    return graph.invoke(
        {"messages": [HumanMessage(text)], "trace_id": trace}, {"configurable": {"thread_id": thread}}
    )


def tool_call(name: str, call_id: str = "c1", **args) -> AIMessage:
    return AIMessage(content=[], tool_calls=[{"name": name, "args": args, "id": call_id}])


def resume(graph, decisions: dict, thread: str = "t1") -> dict:
    """Answer the pending interrupts: {interrupt_id: value}."""
    return graph.invoke(Command(resume=decisions), {"configurable": {"thread_id": thread}})
