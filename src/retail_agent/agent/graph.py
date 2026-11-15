"""The agent as a LangGraph state machine.

    START -> input_guard --(blocked)--> END
                  | ok
                  v
           prepare_context        persona (re-read), analyst examples for this question, user scope
                  v
                agent  <------+   Gemini with tools
                  | tool calls |
                  v            |
                tools ---------+  run_sql (guard -> BigQuery -> mask), report library, preferences;
                  | (no tool calls)   delete_reports pauses the graph for human confirmation
                  v
               finalize           mask PII; flag figures not found in query results; save if asked
                  v
                 END

Each human message is one turn; `trace_id` identifies it in logs, BigQuery job labels and /trace.

Failure handling: no node lets an error escape. If every model is unavailable, or the turn exceeds
its token budget, `agent` answers with a plain explanation and marks the turn `degraded` (so nothing
is saved as a report); an unexpected tool error becomes a tool message; a storage error while saving
a report is reported in the answer.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError

from retail_agent.agent.grounding import caveat, ungrounded_figures
from retail_agent.agent.prompts import build_system_prompt
from retail_agent.agent.session import AgentSession
from retail_agent.resilience import InvokableModel, LLMUnavailableError
from retail_agent.safety.input_guard import check_input
from retail_agent.safety.pii_masker import PiiMasker

log = logging.getLogger(__name__)

TOKEN_BUDGET_REPLY = (
    "I stopped because this question was taking too much work in one go. Try a narrower question, "
    "for example one period or one metric at a time."
)
TOOL_FAILURE = (
    "This step failed because of an internal error, not because of your input. Do not retry it; "
    "continue without it if you can, and tell the user what could not be done."
)


class AgentState(MessagesState):
    trace_id: str
    blocked: bool
    input_verdict: dict[str, Any]
    system_prompt: str
    retrieval: dict[str, Any]
    output_masked: dict[str, int]
    pending_report: dict[str, Any] | None  # set by save_report; the final answer becomes the report
    saved_report: dict[str, Any] | None
    degraded: str | None  # why the turn ended early: llm_unavailable / llm_auth / token_budget ...
    ungrounded: list[str]  # figures in the answer not found in this turn's query results


def build_graph(
    session: AgentSession,
    model: InvokableModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    today: date | None = None,
) -> CompiledStateGraph:
    settings = session.settings

    def input_guard(state: AgentState) -> dict[str, Any]:
        verdict = check_input(_last_human_text(state["messages"]))
        update: dict[str, Any] = {
            "blocked": not verdict.allowed,
            "input_verdict": {"category": verdict.category, "rule": verdict.rule},
            "output_masked": {},
            "pending_report": None,
            "saved_report": None,
            "degraded": None,
            "ungrounded": [],  # every per-turn field is reset here, or a blocked turn inherits the last one
            "retrieval": None,
        }
        if not verdict.allowed:
            update["messages"] = [AIMessage(content=verdict.reply)]
        return update

    def prepare_context(state: AgentState) -> dict[str, Any]:
        question = _last_human_text(state["messages"])
        retrieval = session.golden.search(question, k=settings.golden_top_k)
        prompt = build_system_prompt(
            policy=session.policy,
            profile=session.profile,
            persona=session.persona.get(),
            examples=[trio for trio, _ in retrieval.trios],
            preferences=_preferences(session),
            today=today or date.today(),
            max_rows=settings.max_rows_to_llm,
        )
        return {
            "system_prompt": prompt,
            "retrieval": {
                "method": retrieval.method,
                "trios": [{"id": trio.id, "score": round(score, 3)} for trio, score in retrieval.trios],
                "error": retrieval.error,
                "persona_error": session.persona.last_error,
            },
        }

    def agent(state: AgentState) -> dict[str, Any]:
        if turn_tokens(state["messages"]) >= settings.max_tokens_per_turn:
            return {"messages": [AIMessage(TOKEN_BUDGET_REPLY)], "degraded": "token_budget"}
        messages = [SystemMessage(state["system_prompt"]), *model_view(state["messages"])]
        try:
            return {"messages": [model.invoke(messages)]}
        except LLMUnavailableError as exc:
            log.warning("agent: LLM unavailable (%s) trace=%s", exc, state.get("trace_id"))
            return {"messages": [AIMessage(exc.user_message)], "degraded": f"llm_{exc.kind}"}

    def finalize(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        original = state["messages"][-1]
        message, counts = mask_message(original, session.masker)
        update: dict[str, Any] = {"output_masked": dict(counts), "ungrounded": []}
        if not state.get("degraded") and (missing := ungrounded_figures(message.text, state["messages"])):
            log.warning("finalize: ungrounded figures %s (trace %s)", missing, state.get("trace_id"))
            message = append_text(message, caveat(missing))
            update["ungrounded"] = missing
        pending = state.get("pending_report")
        if pending and message.text.strip() and not state.get("degraded"):
            # The report is exactly what the user sees, already masked; a failed turn saves nothing.
            try:
                report = session.reports.save(
                    owner_id=session.user_id,
                    thread_id=config["configurable"]["thread_id"],
                    title=pending["title"],
                    body=message.text.strip(),
                    trace_id=state.get("trace_id"),
                )
            except (sqlite3.Error, OSError):
                log.exception("finalize: could not save report (trace %s)", state.get("trace_id"))
                note = "\n\n---\n*I could not save this report: the report library is unavailable right now.*"
                message = append_text(message, note)
            else:
                note = f"\n\n---\n*Saved to your reports as `{report.id}`: {report.title}*"
                message = append_text(message, note)
                update["saved_report"] = {"id": report.id, "title": report.title}
        update["pending_report"] = None
        if message is not original:
            update["messages"] = [message]
        return update

    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(session.tools, handle_tool_errors=_tool_error))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", lambda s: END if s["blocked"] else "prepare_context")
    graph.add_edge("prepare_context", "agent")
    graph.add_conditional_edges("agent", lambda s: "tools" if _tool_calls(s["messages"][-1]) else "finalize")
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def _tool_error(exc: Exception) -> str:
    """Bad arguments go back to the model to fix; anything else is logged and reported, never raised."""
    if isinstance(exc, ToolInvocationError):
        return exc.message
    log.error("tool failed unexpectedly", exc_info=exc)
    return TOOL_FAILURE


def _preferences(session: AgentSession) -> list[str]:
    try:
        return session.preferences.list(session.user_id)
    except sqlite3.Error:  # preferences improve answers; they must never block one
        log.exception("could not load preferences for %s", session.user_id)
        return []


def turn_tokens(messages: Sequence[AnyMessage]) -> int:
    """LLM tokens (input + output) spent on the current question so far."""
    total = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.usage_metadata:
            total += message.usage_metadata.get("total_tokens", 0)
    return total


def model_view(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """What the model sees: earlier turns as question + final answer, the current turn in full.

    Intermediate tool calls and results of past turns are dropped: final answers already contain the
    numbers, and resending every past result would grow each request by thousands of tokens.
    """
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    earlier = [
        m
        for m in messages[:last_human]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.text)
    ]
    return [*earlier, *messages[last_human:]]


def mask_message(message: AnyMessage, masker: PiiMasker) -> tuple[AnyMessage, Counter[str]]:
    """Mask the text of a message, keeping its id (so it replaces the original) and any metadata."""
    counts: Counter[str] = Counter()
    if isinstance(message.content, str):
        text, counts = masker.mask_text(message.content)
        content: str | list[Any] = text
    else:
        content = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text, found = masker.mask_text(block.get("text", ""))
                counts.update(found)
                block = {**block, "text": text}
            elif isinstance(block, str):
                block, found = masker.mask_text(block)
                counts.update(found)
            content.append(block)
    return (message.model_copy(update={"content": content}) if counts else message), counts


_SYSTEM_NOTES = re.compile(
    r"\n*(?:---\n\*(?:Saved to your reports|I could not save this report)[^\n]*\*"
    r"|> ⚠ \*Automatic check:[^\n]*\*)\s*$"
)


def strip_system_notes(text: str) -> str:
    """The answer without the notes `finalize` appends (saved-report line, grounding caveat)."""
    previous = None
    while previous != text:
        previous, text = text, _SYSTEM_NOTES.sub("", text).rstrip()
    return text


def append_text(message: AnyMessage, text: str) -> AnyMessage:
    """Add text at the end of a message, keeping its id and metadata."""
    if isinstance(message.content, str):
        content: str | list[Any] = message.content + text
    else:
        content = [*message.content, {"type": "text", "text": text}]
    return message.model_copy(update={"content": content})


def _last_human_text(messages: Sequence[AnyMessage]) -> str:
    return next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def _tool_calls(message: AnyMessage) -> bool:
    return isinstance(message, AIMessage) and bool(message.tool_calls)
