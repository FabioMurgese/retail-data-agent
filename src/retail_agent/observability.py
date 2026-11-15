"""Observability: what happened in each turn, and how the agent is doing overall.

Recording. `TurnObserver` follows a turn while the CLI streams it from LangGraph. Each node update
becomes an event: the input-guard verdict, the golden retrieval, every LLM call (model, latency,
tokens, fallbacks, tools requested), every tool call with its artifact (for run_sql: the SQL as the
model wrote it and as the guard rewrote it, status, rows, bytes billed), delete confirmations, and
user feedback. When the turn ends a `turn` event stores its summary and the whole message
correspondence. Every text is PII-masked before it is stored.

Storage. Events go to SQLite (`events`, queried by /trace and /stats) and to `logs/agent.jsonl`
(one JSON object per line: what production ships to Cloud Logging and BigQuery). A failure to record
is logged and never breaks the chat.

Analysis. `compute_stats` turns `turn` and `feedback` events into the agent-level metrics:
outcome rates, latency percentiles, LLM usage and fallbacks, SQL failure and self-correction rates,
guard refusals, grounding and masking, feedback, estimated cost.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from retail_agent.config import Settings
from retail_agent.safety.pii_masker import PiiMasker
from retail_agent.storage import Database

log = logging.getLogger(__name__)

TurnStatus = Literal["ok", "blocked", "degraded", "error", "interrupted", "stopped"]
_TEXT_LIMIT = 4_000
_EVENT_COLUMNS = ("ts", "trace_id", "thread_id", "user_id", "kind", "name", "status", "duration_ms", "data")
_INSERT_EVENT = (
    f"INSERT INTO events ({', '.join(_EVENT_COLUMNS)}) VALUES ({', '.join(':' + c for c in _EVENT_COLUMNS)})"
)


class EventStore:
    def __init__(self, database: Database, jsonl_path: Path | None = None) -> None:
        self.db = database
        self.jsonl_path = jsonl_path

    def write(
        self,
        *,
        trace_id: str,
        kind: str,
        thread_id: str | None = None,
        user_id: str | None = None,
        name: str | None = None,
        status: str | None = None,
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "trace_id": trace_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "kind": kind,
            "name": name,
            "status": status,
            "duration_ms": duration_ms,
            "data": data or {},
        }
        try:
            with self.db.transaction() as db:
                db.execute(_INSERT_EVENT, {**row, "data": json.dumps(row["data"], default=str)})
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
        except (sqlite3.Error, OSError):  # observability must never take the assistant down
            log.exception("could not record %s event for trace %s", kind, trace_id)

    def events(
        self,
        *,
        kind: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        where, params = [], []
        for column, value in (("kind", kind), ("trace_id", trace_id), ("user_id", user_id)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        if since is not None:
            where.append("ts >= ?")
            params.append(since.isoformat(timespec="milliseconds"))
        sql = "SELECT * FROM events" + (f" WHERE {' AND '.join(where)}" if where else "")
        sql += " ORDER BY id DESC" if newest_first else " ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self.db.transaction() as db:
            rows = [dict(row) for row in db.execute(sql, params)]
        for row in rows:
            row["data"] = json.loads(row["data"])
        return rows

    def resolve_trace(self, prefix: str, *, user_id: str | None = None) -> str | None:
        """Full trace id from a prefix (as shown in the CLI), limited to one user's turns if given."""
        for turn in self.events(kind="turn", user_id=user_id, newest_first=True):
            if turn["trace_id"].startswith(prefix):
                return turn["trace_id"]
        return None


class TurnObserver:
    """Records one turn. Call `on_update` for each streamed node update, then `finish` once."""

    def __init__(
        self,
        store: EventStore,
        settings: Settings,
        masker: PiiMasker,
        *,
        trace_id: str,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> None:
        self.store, self.settings, self.masker = store, settings, masker
        self.trace_id, self.user_id, self.thread_id = trace_id, user_id, thread_id
        self.question = self._mask(question)
        self._started = self._last = time.monotonic()
        self._tool_args: dict[str, dict[str, Any]] = {}
        self._confirmations: list[dict[str, Any]] = []

    def on_update(self, node: str, values: Any) -> None:
        now = time.monotonic()
        step_ms = int((now - self._last) * 1000)
        self._last = now
        if not isinstance(values, dict):
            return
        if node == "input_guard":
            verdict = values.get("input_verdict", {})
            self._write(
                "input_guard", status="blocked" if values.get("blocked") else "ok", ms=step_ms, data=verdict
            )
        elif node == "prepare_context" and (retrieval := values.get("retrieval")):
            self._write("retrieval", name=retrieval.get("method"), ms=step_ms, data=retrieval)
        elif node == "agent":
            message = values["messages"][-1]
            meta = getattr(message, "response_metadata", {}) or {}
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                self._tool_args[call["id"]] = call["args"]
            self._write(
                "llm_call",
                name=meta.get("model_name"),
                status=values.get("degraded") or "ok",
                ms=meta.get("latency_ms", step_ms),
                data={
                    "usage": getattr(message, "usage_metadata", None),
                    "tool_calls": [c["name"] for c in calls],
                    "fallbacks": meta.get("fallbacks", []),
                },
            )
        elif node == "tools":
            for message in values.get("messages", []):
                if isinstance(message, ToolMessage):
                    artifact = message.artifact or {}
                    self._write(
                        "tool_call",
                        name=message.name,
                        status=artifact.get("status"),
                        ms=artifact.get("elapsed_ms", step_ms),
                        data={
                            "args": self._masked_json(self._tool_args.get(message.tool_call_id, {})),
                            "artifact": self._masked_json(artifact),
                            "result": self._mask(_text(message))[:_TEXT_LIMIT],
                        },
                    )
        elif node == "verify" and (revision := values.get("revision")):
            self._write("grounding_check", status="revise", ms=step_ms, data=revision)
        elif node == "finalize":
            self._write(
                "output",
                ms=step_ms,
                data={
                    k: values.get(k) for k in ("output_masked", "ungrounded", "saved_report") if k in values
                },
            )

    def on_confirmation(self, request: dict[str, Any], confirmed: bool) -> None:
        record = {
            "criteria": request.get("criteria"),
            "report_ids": [r["id"] for r in request.get("reports", [])],
            "decision": "confirmed" if confirmed else "cancelled",
        }
        self._confirmations.append(record)
        self._write("confirmation", name="delete_reports", status=record["decision"], data=record)

    def finish(
        self, state: dict[str, Any] | None, *, failure: TurnStatus | None = None, error: str | None = None
    ) -> dict[str, Any]:
        summary = summarize_turn(state or {}, self.settings, self.masker, failure=failure, error=error)
        summary["question"] = self.question
        summary["confirmations"] = self._confirmations
        duration_ms = int((time.monotonic() - self._started) * 1000)
        self._write("turn", status=summary["status"], ms=duration_ms, data=summary)
        return summary

    def _write(self, kind: str, *, name: str | None = None, status: str | None = None, ms: int | None = None,
               data: dict[str, Any] | None = None) -> None:  # fmt: skip
        self.store.write(
            trace_id=self.trace_id, thread_id=self.thread_id, user_id=self.user_id,
            kind=kind, name=name, status=status, duration_ms=ms, data=data,
        )  # fmt: skip

    def _mask(self, text: str) -> str:
        return self.masker.mask_text(text)[0]

    def _masked_json(self, value: Any) -> Any:
        return json.loads(self._mask(json.dumps(value, default=str)))


def summarize_turn(
    state: dict[str, Any],
    settings: Settings,
    masker: PiiMasker,
    *,
    failure: TurnStatus | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The `turn` event: outcome, usage, SQL, quality signals and the masked message correspondence."""
    messages: list[AnyMessage] = list(state.get("messages", []))
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    turn = messages[start:]
    model_calls = [m for m in turn if isinstance(m, AIMessage) and m.response_metadata.get("model_name")]
    tokens_in = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in model_calls)
    tokens_out = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in model_calls)
    sql = [m.artifact or {} for m in turn if isinstance(m, ToolMessage) and m.name == "run_sql"]
    sql_status = Counter(a.get("status") for a in sql)
    bytes_billed = sum(a.get("bytes_billed") or 0 for a in sql)
    if failure is not None:
        status: TurnStatus = failure
    elif state.get("blocked"):
        status = "blocked"
    elif state.get("degraded"):
        status = "degraded"
    else:
        status = "ok"
    answer = turn[-1] if turn and isinstance(turn[-1], AIMessage) and not turn[-1].tool_calls else None
    return {
        "status": status,
        "error": error,
        "answer": masker.mask_text(answer.text)[0][:_TEXT_LIMIT] if answer else None,
        "input_verdict": state.get("input_verdict"),
        "retrieval": state.get("retrieval"),
        "degraded": state.get("degraded"),
        "llm": {
            "calls": len(model_calls),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "models": dict(Counter(m.response_metadata["model_name"] for m in model_calls)),
            "fallbacks": [f for m in model_calls for f in m.response_metadata.get("fallbacks", [])],
        },
        "sql": {
            "queries": sum(sql_status[s] for s in ("ok", "empty", "error", "refused")),
            **{
                s: sql_status[s]
                for s in ("ok", "empty", "error", "refused", "unavailable", "budget_exhausted")
            },
            "refused_codes": dict(Counter(a.get("code") for a in sql if a.get("status") == "refused")),
            "bytes_billed": bytes_billed,
        },
        "tools": dict(Counter(m.name for m in turn if isinstance(m, ToolMessage))),
        "ungrounded": state.get("ungrounded") or [],
        "revision": state.get("revision"),
        "output_masked": state.get("output_masked") or {},
        "saved_report": state.get("saved_report"),
        "cost_usd": {
            "llm": round(
                tokens_in / 1e6 * settings.llm_price_input_per_mtok
                + tokens_out / 1e6 * settings.llm_price_output_per_mtok,
                6,
            ),
            "bigquery": round(bytes_billed / 2**40 * settings.bq_price_per_tib, 6),
        },
        "messages": [_serialize(m, masker) for m in turn],
    }


def _serialize(message: AnyMessage, masker: PiiMasker) -> dict[str, Any]:
    role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(message.type, message.type)
    record: dict[str, Any] = {"role": role, "text": masker.mask_text(_text(message))[0][:_TEXT_LIMIT]}
    if isinstance(message, AIMessage):
        if message.tool_calls:
            calls = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
            record["tool_calls"] = json.loads(masker.mask_text(json.dumps(calls, default=str))[0])
        if model := message.response_metadata.get("model_name"):
            record["model"] = model
    if isinstance(message, ToolMessage):
        record["tool"] = message.name
    return record


def _text(message: AnyMessage) -> str:
    return message.content if isinstance(message.content, str) else message.text


# --- metrics -------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def parse_since(text: str | None) -> datetime | None:
    """'24h', '7d', '30m' -> a UTC datetime that far back; None for everything."""
    if not text:
        return None
    units = {"m": "minutes", "h": "hours", "d": "days"}
    value, unit = text[:-1], text[-1:].lower()
    if unit not in units or not value.isdigit():
        raise ValueError(f"expected e.g. 30m, 24h or 7d, got {text!r}")
    return datetime.now(UTC) - timedelta(**{units[unit]: int(value)})


def compute_stats(turns: Sequence[dict[str, Any]], feedback: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Agent-level metrics from `turn` events (rows from EventStore.events(kind="turn"))."""
    data = [t["data"] for t in turns]
    n = len(data)

    def share(count: int, total: int) -> float | None:
        return count / total if total else None

    status = Counter(d["status"] for d in data)
    answered = [t for t in turns if t["data"]["status"] in ("ok", "degraded")]
    replies = [t["data"] for t in answered]  # quality rates are over turns that produced an answer
    with_sql = [d for d in data if d["sql"]["queries"]]
    with_failures = [d for d in with_sql if d["sql"]["error"] + d["sql"]["refused"] + d["sql"]["empty"]]
    queries = sum(d["sql"]["queries"] for d in data)
    llm_calls = sum(d["llm"]["calls"] for d in data)
    ratings = Counter(f["status"] for f in feedback)
    refused_codes: Counter[str] = Counter()
    models: Counter[str] = Counter()
    for d in data:
        refused_codes.update(d["sql"]["refused_codes"])
        models.update(d["llm"]["models"])
    return {
        "turns": n,
        "status": dict(status),
        "rates": {
            "success": share(status["ok"], n),
            "blocked": share(status["blocked"], n),
            "degraded": share(status["degraded"], n),
            "error": share(status["error"] + status["stopped"], n),
        },
        "blocked_by": dict(Counter(d["input_verdict"]["category"] for d in data if d["status"] == "blocked")),
        "degraded_by": dict(Counter(d["degraded"] for d in data if d["degraded"])),
        "latency_ms": {
            "p50": percentile([t["duration_ms"] for t in answered], 0.5),
            "p95": percentile([t["duration_ms"] for t in answered], 0.95),
        },
        "llm": {
            "calls_per_turn": share(llm_calls, len(answered)),
            "input_tokens": sum(d["llm"]["input_tokens"] for d in data),
            "output_tokens": sum(d["llm"]["output_tokens"] for d in data),
            "fallback_turn_rate": share(sum(1 for d in replies if d["llm"]["fallbacks"]), len(replies)),
            "models": dict(models),
        },
        "sql": {
            "queries": queries,
            "per_turn": share(queries, len(with_sql)),
            "failure_rate": share(sum(d["sql"]["error"] + d["sql"]["refused"] for d in data), queries),
            "empty_rate": share(sum(d["sql"]["empty"] for d in data), queries),
            "self_corrected": share(sum(1 for d in with_failures if d["status"] == "ok"), len(with_failures)),
            "refused_by": dict(refused_codes),
            "budget_stops": sum(d["sql"]["budget_exhausted"] for d in data),
            "unavailable": sum(d["sql"]["unavailable"] for d in data),
            "bytes_billed": sum(d["sql"]["bytes_billed"] for d in data),
        },
        "quality": {
            "ungrounded_rate": share(sum(1 for d in replies if d["ungrounded"]), len(replies)),
            "revised_rate": share(sum(1 for d in replies if d.get("revision")), len(replies)),
            "masked_output_turns": sum(1 for d in data if d["output_masked"]),
            "retrieval": dict(
                Counter((d["retrieval"] or {}).get("method") for d in replies if d["retrieval"])
            ),
        },
        "feedback": {
            "good": ratings["good"],
            "bad": ratings["bad"],
            "satisfaction": share(ratings["good"], len(feedback)),
        },
        "reports": {
            "saved": sum(1 for d in data if d["saved_report"]),
            "deletes_confirmed": sum(
                1 for d in data for c in d["confirmations"] if c["decision"] == "confirmed"
            ),
            "deletes_cancelled": sum(
                1 for d in data for c in d["confirmations"] if c["decision"] == "cancelled"
            ),
        },
        "cost_usd": {
            "llm": round(sum(d["cost_usd"]["llm"] for d in data), 4),
            "bigquery": round(sum(d["cost_usd"]["bigquery"] for d in data), 4),
        },
    }
