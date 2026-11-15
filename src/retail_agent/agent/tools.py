"""Agent tools, built per session with the user's SQL guard bound in.

Because the guard is captured when the session starts, nothing said in the conversation can change
which rows or columns a tool may read.

Each tool returns (content, artifact): `content` is what the LLM reads (compact CSV or an error it
can act on); `artifact` is structured metadata for the CLI and for observability (status, rows,
bytes, guarded SQL, masking counts). The LLM never sees the artifact.

Self-correction is bounded in code, not only in the prompt: per question, run_sql accepts at most
`max_failures` failed/refused/empty queries and `max_queries` queries in total; the counts come from
the turn's own tool messages. A BigQuery circuit breaker turns an outage into an immediate answer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Protocol

import pandas as pd
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from retail_agent.data.bq_client import BigQueryError, BigQueryUnavailableError, QueryResult
from retail_agent.resilience import CircuitBreaker
from retail_agent.safety.pii_masker import PiiMasker
from retail_agent.safety.sql_guard import SqlGuard, SqlGuardError

log = logging.getLogger(__name__)

EMPTY_RESULT = (
    "The query matched no data (no rows, or only zero/NULL totals). If you expected data, check the "
    "filter values (for example with SELECT DISTINCT) or widen the date range before concluding there "
    "is none. Never fill the gap with estimates: to give figures for a corrected value, query them."
)


FAILED_STATUSES = frozenset({"error", "refused", "empty"})
EXECUTED_STATUSES = FAILED_STATUSES | {"ok"}


class QueryRunner(Protocol):
    def run(self, sql: str, *, labels: Mapping[str, str] | None = None) -> QueryResult: ...


def make_run_sql(
    *,
    guard: SqlGuard,
    runner: QueryRunner,
    masker: PiiMasker,
    user_id: str,
    max_rows_shown: int,
    breaker: CircuitBreaker,
    max_failures: int,
    max_queries: int,
) -> BaseTool:
    @tool(response_format="content_and_artifact", parse_docstring=True)
    def run_sql(
        sql: str, purpose: str, state: Annotated[dict[str, Any], InjectedState]
    ) -> tuple[str, dict[str, Any]]:
        """Run one read-only BigQuery SELECT on the company data and return the result as CSV.

        Args:
            sql: BigQuery Standard SQL, a single SELECT. Use bare table names: orders, order_items,
                products, users.
            purpose: One short line saying what this query answers, e.g. "net revenue by month in 2026".
        """
        trace_id = state.get("trace_id", "")
        artifact: dict[str, Any] = {"purpose": purpose, "sql": sql}
        executed, failed = sql_stats(state.get("messages", []))
        if failed >= max_failures or executed >= max_queries:
            limit = f"{failed} queries failed" if failed >= max_failures else f"{executed} queries were run"
            artifact.update(
                status="budget_exhausted", code="failures" if failed >= max_failures else "queries"
            )
            return (
                f"Stop querying: {limit} for this question, which is the limit. Do not call run_sql again. "
                "Answer with what you have, or explain what went wrong and suggest how to rephrase.",
                artifact,
            )

        try:
            guarded = guard.check(sql)
        except SqlGuardError as exc:
            artifact.update(status="refused", code=exc.code, error=str(exc))
            return f"Query refused by the data-access guard ({exc.code}): {exc}", artifact

        if not breaker.allow():
            artifact.update(status="unavailable", code="circuit_open", fixable=False)
            return (
                f"BigQuery is unavailable right now (it failed repeatedly; next check in about "
                f"{breaker.retry_in_s():.0f} s). Do not retry: tell the user to try again in a minute.",
                artifact,
            )

        artifact["guarded_sql"] = guarded.sql
        try:
            result = runner.run(guarded.sql, labels={"trace_id": trace_id, "user": user_id})
        except BigQueryError as exc:
            if isinstance(exc, BigQueryUnavailableError):
                breaker.record_failure(open_now=not exc.transient)  # setup problems will not heal by retrying
            else:
                breaker.record_success()  # BigQuery answered: the SQL was the problem
            message, _ = masker.mask_text(str(exc))
            artifact.update(status="error", code=type(exc).__name__, error=message, fixable=exc.fixable)
            advice = (
                "Fix the SQL and try again."
                if exc.fixable
                else "This is a service problem, not a SQL problem: do not retry; tell the user."
            )
            return f"BigQuery error: {message} {advice}", artifact

        breaker.record_success()
        df, masked = masker.mask_frame(result.df)
        artifact.update(
            status="ok" if len(df) else "empty",
            rows=result.total_rows,
            truncated=result.truncated,
            bytes_billed=result.bytes_billed,
            elapsed_ms=result.elapsed_ms,
            job_id=result.job_id,
            masked=dict(masked),
        )
        if masked:
            log.warning("run_sql: masked PII in a query result %s (trace %s)", dict(masked), trace_id)
        if no_data(df):
            artifact["status"] = "empty"
            return EMPTY_RESULT, artifact
        return render_result(
            df, total_rows=result.total_rows, truncated=result.truncated, limit=max_rows_shown
        ), artifact

    return run_sql


def no_data(df: pd.DataFrame) -> bool:
    """No rows, or one row of only NULL/zero values (what SUM/COUNT return when nothing matched)."""
    if df.empty:
        return True
    if len(df) > 1:
        return False
    return all(pd.isna(value) or (pd.api.types.is_number(value) and value == 0) for value in df.iloc[0])


def sql_stats(messages: Sequence[AnyMessage]) -> tuple[int, int]:
    """(queries executed, queries failed) by run_sql since the last human message."""
    executed = failed = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name == "run_sql":
            status = (message.artifact or {}).get("status")
            executed += status in EXECUTED_STATUSES
            failed += status in FAILED_STATUSES
    return executed, failed


def render_result(df: pd.DataFrame, *, total_rows: int, truncated: bool, limit: int) -> str:
    shown = df.head(limit)
    if truncated:
        header = (
            f"Result: more than {len(df)} rows (cut at the row limit); first {len(shown)} shown. "
            "Aggregate further for complete figures."
        )
    elif len(shown) < total_rows:
        header = f"Result: {total_rows} rows; first {len(shown)} shown."
    else:
        header = f"Result: {total_rows} row{'s' if total_rows != 1 else ''}."
    numeric = shown.select_dtypes("number").columns
    rounded = shown.round(dict.fromkeys(numeric, 4))  # dates and text are left as they are
    return f"{header}\n{rounded.to_csv(index=False).strip()}"
