"""BigQuery access layer: every query runs under the same cost, time and size limits.

`BigQueryRunner.run()` always:
  1. dry-runs the query (free): catches syntax errors and returns a bytes estimate;
  2. checks BigQuery's own view of the query: it must be a single SELECT that reads only the
     allowed tables (a second line of defence behind `safety.sql_guard`, which uses another parser);
  3. refuses queries estimated above the bytes cap;
  4. executes with `maximum_bytes_billed`, a server-side job timeout and a row cap.

Transient failures (429, 5xx, connection resets) are retried by the client library's own retry
policy, bounded by a deadline. Whatever is left is translated into `BigQueryError` subclasses whose
messages are safe to hand to the LLM and the user (URLs, project and job ids stripped). `fixable`
tells the agent whether rewriting the SQL can help.

The runner does not decide what SQL is acceptable: callers must pass SQL produced by `sql_guard`.

Evolved from the `BigQueryRunner` provided with the assignment: its public surface is kept
(`BigQueryRunner(project_id, dataset_id)`, `execute_query()`, `get_table_schema()`), and every call
now goes through the limits above. `run()` returns the result with its cost and job metadata.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exc
from google.cloud import bigquery
from google.cloud.bigquery.retry import DEFAULT_JOB_RETRY, DEFAULT_RETRY
from requests import exceptions as requests_exc

if TYPE_CHECKING:
    from retail_agent.config import Settings

log = logging.getLogger(__name__)

APP_LABEL = "retail-agent"
DEFAULT_DATASET = "bigquery-public-data.thelook_ecommerce"
_ADC_HINT = "Run `gcloud auth application-default login` (see README)."


# --- Errors ---------------------------------------------------------------------------------------


class BigQueryError(Exception):
    """A query could not be run. `str(error)` is safe to show to the LLM and to the user."""

    #: True if the agent may succeed by rewriting the SQL; False for outages and setup problems.
    fixable: bool = False


class InvalidQueryError(BigQueryError):
    """Syntax error, unknown column or table, type mismatch, not a single SELECT..."""

    fixable = True


class QueryTooExpensiveError(BigQueryError):
    fixable = True


class QueryTimeoutError(BigQueryError):
    fixable = True


class BigQueryUnavailableError(BigQueryError):
    """BigQuery cannot be used right now: outage that outlived the retries, quota, auth or setup."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


# --- Results --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DryRunResult:
    bytes_processed: int
    statement_type: str | None
    referenced_tables: tuple[str, ...]  # "project.dataset.table"


@dataclass(frozen=True)
class QueryResult:
    df: pd.DataFrame
    total_rows: int  # rows the query produced; `df` holds at most `max_rows` of them
    truncated: bool
    bytes_processed: int | None
    bytes_billed: int | None
    cache_hit: bool | None
    job_id: str | None
    elapsed_ms: int


# --- Runner ---------------------------------------------------------------------------------------


class BigQueryRunner:
    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str = DEFAULT_DATASET,
        *,
        location: str = "US",
        max_bytes_billed: int = 1_000_000_000,
        timeout_s: float = 60.0,
        max_rows: int = 1_000,
        retry_deadline_s: float = 30.0,
        allowed_tables: Collection[str] | None = None,
        client: bigquery.Client | None = None,
    ) -> None:
        self.project = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.max_bytes_billed = max_bytes_billed
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.allowed_tables = frozenset(allowed_tables) if allowed_tables is not None else None
        self._retry = DEFAULT_RETRY.with_timeout(retry_deadline_s)
        self._job_retry = DEFAULT_JOB_RETRY.with_timeout(retry_deadline_s)
        self._client = client  # created lazily so the CLI can start (and explain) without credentials

    @classmethod
    def from_settings(
        cls, settings: Settings, *, allowed_tables: Collection[str] | None = None
    ) -> BigQueryRunner:
        return cls(
            settings.gcp_project,
            settings.bq_dataset,
            location=settings.bq_location,
            max_bytes_billed=settings.bq_max_bytes_billed,
            timeout_s=settings.bq_timeout_s,
            max_rows=settings.bq_max_rows,
            retry_deadline_s=settings.bq_retry_deadline_s,
            allowed_tables=allowed_tables,
        )

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            with self._translated_errors():
                try:
                    self._client = bigquery.Client(project=self.project, location=self.location)
                except OSError as exc:  # e.g. no project in the environment and none configured
                    if isinstance(exc, ConnectionError):
                        raise
                    raise BigQueryUnavailableError(
                        f"Could not create the BigQuery client ({exc}). Set GCP_PROJECT in .env.",
                        transient=False,
                    ) from exc
        return self._client

    def dry_run(self, sql: str, *, labels: Mapping[str, str] | None = None) -> DryRunResult:
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, labels=_labels(labels))
        with self._translated_errors():
            job = self.client.query(sql, job_config=config, retry=self._retry, timeout=self.timeout_s)
        return DryRunResult(
            bytes_processed=job.total_bytes_processed or 0,
            statement_type=job.statement_type,
            referenced_tables=tuple(
                f"{t.project}.{t.dataset_id}.{t.table_id}" for t in job.referenced_tables
            ),
        )

    def run(self, sql: str, *, labels: Mapping[str, str] | None = None) -> QueryResult:
        estimate = self.dry_run(sql, labels=labels)
        self._check_statement(estimate)
        if estimate.bytes_processed > self.max_bytes_billed:
            raise QueryTooExpensiveError(
                f"The query would scan {_human_bytes(estimate.bytes_processed)}, above the "
                f"{_human_bytes(self.max_bytes_billed)} limit. Select fewer columns or filter earlier."
            )

        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes_billed,
            job_timeout_ms=int(self.timeout_s * 1000),
            labels=_labels(labels),
        )
        started = time.monotonic()
        with self._translated_errors():
            job = self.client.query(
                sql, job_config=config, retry=self._retry, job_retry=self._job_retry, timeout=self.timeout_s
            )
            try:
                rows = job.result(
                    max_results=self.max_rows,
                    timeout=self.timeout_s,
                    retry=self._retry,
                    job_retry=self._job_retry,
                )
            except TimeoutError:
                with contextlib.suppress(Exception):  # best effort: the server-side timeout also applies
                    job.cancel()
                raise
            df = rows.to_dataframe(create_bqstorage_client=False)

        total_rows = rows.total_rows if rows.total_rows is not None else len(df)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "query done job=%s rows=%s billed=%s cache=%s ms=%s",
            job.job_id, total_rows, job.total_bytes_billed, job.cache_hit, elapsed_ms,
        )  # fmt: skip
        return QueryResult(
            df=df,
            total_rows=total_rows,
            truncated=total_rows > len(df),
            bytes_processed=job.total_bytes_processed,
            bytes_billed=job.total_bytes_billed,
            cache_hit=job.cache_hit,
            job_id=job.job_id,
            elapsed_ms=elapsed_ms,
        )

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """The original API: run a query and return its rows, now under all the limits of `run()`."""
        return self.run(sql_query).df

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Live schema of a table in `dataset_id` (used by `doctor` to detect schema drift).

        Returns the original API's shape: one dict per column with name, type, mode and description.
        """
        with self._translated_errors():
            table = self.client.get_table(
                f"{self.dataset_id}.{table_name}", retry=self._retry, timeout=self.timeout_s
            )
        return [
            {"name": f.name, "type": f.field_type, "mode": f.mode, "description": f.description or ""}
            for f in table.schema
        ]

    def _check_statement(self, estimate: DryRunResult) -> None:
        if estimate.statement_type != "SELECT":
            raise InvalidQueryError(
                f"Only a single read-only SELECT is allowed (got {estimate.statement_type or 'unknown'})."
            )
        if self.allowed_tables is not None and (
            outside := sorted(set(estimate.referenced_tables) - self.allowed_tables)
        ):
            raise InvalidQueryError(f"The query reads tables that are not allowed: {', '.join(outside)}.")

    @contextlib.contextmanager
    def _translated_errors(self) -> Iterator[None]:
        try:
            yield
        except BigQueryError:
            raise
        except Exception as exc:
            translated = translate_error(exc, project=self.project)
            if translated is None:
                raise
            raise translated from exc


# --- Error translation ----------------------------------------------------------------------------

_NOISE = (
    re.compile(r"(?:\b\d{3} )?(?:GET|POST|PUT|PATCH|DELETE) https?://\S+?: "),  # "400 POST https://...: "
    re.compile(r"\s*Location: \S*"),
    re.compile(r"\s*Job ID: \S*"),
    re.compile(r"https?://\S+"),
)


def translate_error(exc: Exception, *, project: str | None = None) -> BigQueryError | None:
    """Map a Google/transport exception to a `BigQueryError`; None if it is not a BigQuery failure."""
    if isinstance(exc, gexc.RetryError):  # retries exhausted: report the last underlying failure
        cause = exc.cause if isinstance(exc.cause, Exception) else exc
        return BigQueryUnavailableError(
            f"BigQuery is temporarily unavailable: {_clean_message(cause, project)}", transient=True
        )

    reason = _reason(exc)
    message = _clean_message(exc, project)

    if reason == "bytesBilledLimitExceeded":  # arrives as HTTP 500: must be checked before ServerError
        return QueryTooExpensiveError(f"The query exceeded the bytes-billed limit: {message}")
    if isinstance(exc, TimeoutError | gexc.DeadlineExceeded) or reason in {"timeout", "jobTimeout"}:
        return QueryTimeoutError(
            "The query did not finish in time and was cancelled. "
            "Simplify it: aggregate earlier, filter by date or avoid large joins."
        )
    if isinstance(exc, gexc.NotFound) or reason == "notFound":
        # The SQL guard only lets the allowed tables through, so "not found" means the billing project,
        # dataset or a table is missing: a setup problem that rewriting the SQL cannot fix.
        return BigQueryUnavailableError(
            f"BigQuery setup problem: {message}. Check GCP_PROJECT and BQ_DATASET (`retail-agent doctor`).",
            transient=False,
        )
    if isinstance(exc, gexc.BadRequest) or reason in {"invalidQuery", "invalid"}:
        return InvalidQueryError(message)
    if isinstance(exc, auth_exc.DefaultCredentialsError):
        return BigQueryUnavailableError(f"No Google Cloud credentials found. {_ADC_HINT}", transient=False)
    if isinstance(exc, auth_exc.RefreshError):
        return BigQueryUnavailableError(
            f"Google Cloud credentials are expired or revoked. {_ADC_HINT}", transient=False
        )
    if isinstance(exc, gexc.TooManyRequests) or reason in {"rateLimitExceeded", "quotaExceeded"}:
        return BigQueryUnavailableError(f"BigQuery quota or rate limit reached: {message}", transient=True)
    if isinstance(exc, gexc.Forbidden | gexc.Unauthorized):
        return BigQueryUnavailableError(f"BigQuery access denied: {message}", transient=False)
    if isinstance(
        exc, gexc.ServerError | auth_exc.TransportError | requests_exc.RequestException | ConnectionError
    ):
        return BigQueryUnavailableError(f"BigQuery is temporarily unavailable: {message}", transient=True)
    return None


def _reason(exc: BaseException) -> str | None:
    errors = getattr(exc, "errors", None)
    if errors and isinstance(errors[0], Mapping):
        return errors[0].get("reason")
    return None


def _clean_message(exc: BaseException, project: str | None) -> str:
    errors = getattr(exc, "errors", None)
    message = errors[0].get("message") if errors and isinstance(errors[0], Mapping) else None
    message = message or getattr(exc, "message", None) or str(exc) or type(exc).__name__
    for pattern in _NOISE:
        message = pattern.sub("", message)
    if project:
        message = message.replace(project, "<billing-project>")
    return " ".join(message.split())


def _labels(extra: Mapping[str, str] | None) -> dict[str, str]:
    """Job labels (e.g. trace_id) make every query attributable in INFORMATION_SCHEMA.JOBS and billing."""
    labels = {"app": APP_LABEL, **(extra or {})}
    return {_label_part(key): _label_part(value) for key, value in labels.items()}


def _label_part(value: str) -> str:
    # BigQuery labels: lowercase letters, digits, '_' and '-', at most 63 characters.
    return re.sub(r"[^a-z0-9_-]", "_", str(value).lower())[:63]


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"
