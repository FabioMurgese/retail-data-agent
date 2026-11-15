"""One user turn, end to end: stream the graph, record it, ask for confirmations, survive failures.

Shared by the interactive chat and the evaluation suite, so both exercise exactly the same path.
The caller decides how to show progress (`on_update`) and how a human answers a destructive-action
confirmation (`confirm`): the chat asks the person at the keyboard, the evals use a scripted answer.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from retail_agent.agent.session import AgentSession
from retail_agent.observability import TurnObserver, TurnStatus

log = logging.getLogger(__name__)

# (interrupt payload) -> (confirmed?, text typed instead of a confirmation, to handle as a new request)
Confirm = Callable[[dict[str, Any]], tuple[bool, str | None]]


@dataclass
class TurnOutcome:
    trace_id: str
    state: dict[str, Any] | None
    summary: dict[str, Any]
    failure: TurnStatus | None = None  # interrupted / stopped / error: the turn did not finish
    error: str | None = None
    follow_up: str | None = None
    confirmations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def answer(self) -> str:
        messages = (self.state or {}).get("messages", [])
        return messages[-1].text if messages else ""


def run_turn(
    session: AgentSession,
    graph: CompiledStateGraph,
    text: str,
    *,
    confirm: Confirm,
    on_update: Callable[[str, Any], None] | None = None,
    trace_id: str | None = None,
) -> TurnOutcome:
    trace_id = trace_id or uuid.uuid4().hex[:12]
    config: dict[str, Any] = {
        "configurable": {"thread_id": session.thread_id},
        "recursion_limit": session.settings.recursion_limit,
    }
    observer = TurnObserver(
        session.events,
        session.settings,
        session.masker,
        trace_id=trace_id,
        user_id=session.user_id,
        thread_id=session.thread_id,
        question=text,
    )
    graph_input: Any = {"messages": [HumanMessage(text)], "trace_id": trace_id}
    failure: TurnStatus | None = None
    error = follow_up = None
    confirmations: list[dict[str, Any]] = []
    while True:
        try:
            for update in graph.stream(graph_input, config, stream_mode="updates"):
                for node, values in update.items():
                    observer.on_update(node, values)
                    if on_update is not None:
                        on_update(node, values)
        except KeyboardInterrupt:
            failure, error = "interrupted", "interrupted by the user"
            break
        except GraphRecursionError as exc:
            failure, error = "stopped", f"recursion limit: {exc}"[:300]
            break
        except Exception as exc:  # a turn must never take the caller down
            log.exception("turn failed (trace %s)", trace_id)
            failure, error = "error", f"{type(exc).__name__}: {exc}"[:300]
            break
        interrupts = graph.get_state(config).interrupts
        if not interrupts:
            break
        decisions = {}
        for pending in interrupts:  # normally one: the prompt asks for one delete call per request
            confirmed, typed = confirm(pending.value)
            observer.on_confirmation(pending.value, confirmed)
            confirmations.append({"request": pending.value, "confirmed": confirmed})
            decisions[pending.id] = {"token": pending.value["token"], "confirmed": confirmed}
            follow_up = follow_up or typed
        graph_input = Command(resume=decisions)

    state = _state(graph, config)
    summary = observer.finish(state, failure=failure, error=error)
    return TurnOutcome(trace_id, state, summary, failure, error, follow_up, confirmations)


def _state(graph: CompiledStateGraph, config: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return dict(graph.get_state(config).values)
    except Exception:  # recording must not fail because the state cannot be read
        log.exception("could not read the graph state")
        return None
