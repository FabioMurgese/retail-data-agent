"""BigQueryRunner against a fake client: limits, statement checks and error translation."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exc

from retail_agent.data import bq_client
from retail_agent.data.bq_client import (
    BigQueryRunner,
    BigQueryUnavailableError,
    InvalidQueryError,
    QueryTimeoutError,
    QueryTooExpensiveError,
    translate_error,
)

PRODUCTS = "bigquery-public-data.thelook_ecommerce.products"


class FakeRows:
    def __init__(self, df: pd.DataFrame, max_results: int | None) -> None:
        self._df = df
        self.total_rows = len(df)
        self._max = max_results

    def to_dataframe(self, create_bqstorage_client: bool = True) -> pd.DataFrame:
        assert create_bqstorage_client is False
        return self._df.head(self._max) if self._max is not None else self._df


class FakeJob:
    def __init__(self, client: FakeClient, *, dry: bool) -> None:
        self.client = client
        self.total_bytes_processed = client.bytes_processed
        self.total_bytes_billed = None if dry else 10_485_760
        self.statement_type = client.statement_type
        self.referenced_tables = [
            SimpleNamespace(project=p, dataset_id=d, table_id=t)
            for p, d, t in (table.split(".") for table in client.tables)
        ]
        self.cache_hit = False
        self.job_id = "job-1"
        self.cancelled = False

    def result(self, max_results=None, timeout=None, retry=None, job_retry=None):
        self.client.result_calls.append({"max_results": max_results, "timeout": timeout})
        if self.client.result_error:
            raise self.client.result_error
        return FakeRows(self.client.df, max_results)

    def cancel(self) -> None:
        self.cancelled = True


class FakeClient:
    def __init__(self, **overrides) -> None:
        self.bytes_processed = 1_000
        self.statement_type = "SELECT"
        self.tables = [PRODUCTS]
        self.df = pd.DataFrame({"n": [1, 2, 3]})
        self.dry_error: Exception | None = None
        self.exec_error: Exception | None = None
        self.result_error: Exception | None = None
        self.__dict__.update(overrides)
        self.query_calls: list[dict] = []
        self.result_calls: list[dict] = []
        self.jobs: list[FakeJob] = []

    def query(self, sql, job_config=None, **kwargs):
        self.query_calls.append({"sql": sql, "config": job_config, **kwargs})
        error = self.dry_error if job_config.dry_run else self.exec_error
        if error:
            raise error
        job = FakeJob(self, dry=bool(job_config.dry_run))
        self.jobs.append(job)
        return job


def make_runner(client: FakeClient, **kwargs) -> BigQueryRunner:
    defaults = {"project_id": "my-billing-proj", "max_bytes_billed": 50_000_000, "max_rows": 1_000}
    return BigQueryRunner(client=client, allowed_tables={PRODUCTS}, **(defaults | kwargs))


def test_run_dry_runs_first_then_executes_with_limits():
    client = FakeClient()
    result = make_runner(client, timeout_s=30).run("SELECT 1", labels={"trace_id": "Trace-ABC.1"})

    dry, real = (call["config"] for call in client.query_calls)
    assert dry.dry_run is True and dry.use_query_cache is False
    assert not real.dry_run
    assert real.maximum_bytes_billed == 50_000_000
    assert int(real.job_timeout_ms) == 30_000  # stored in API form (a string)
    assert real.labels == {"app": "retail-agent", "trace_id": "trace-abc_1"}  # sanitised for BigQuery
    assert client.result_calls == [{"max_results": 1_000, "timeout": 30}]
    assert result.total_rows == 3 and not result.truncated
    assert result.bytes_billed == 10_485_760 and result.job_id == "job-1"


def test_row_cap_truncates_and_reports_it():
    client = FakeClient(df=pd.DataFrame({"n": range(10)}))
    result = make_runner(client, max_rows=4).run("SELECT 1")
    assert len(result.df) == 4
    assert result.total_rows == 10 and result.truncated


def test_refuses_expensive_query_without_executing_it():
    client = FakeClient(bytes_processed=60_000_000)
    with pytest.raises(QueryTooExpensiveError, match="60.0 MB, above the 50.0 MB limit"):
        make_runner(client).run("SELECT 1")
    assert len(client.query_calls) == 1  # dry-run only


@pytest.mark.parametrize("statement_type", ["SCRIPT", "DELETE", "CREATE_TABLE", None])
def test_only_select_statements_run(statement_type):
    client = FakeClient(statement_type=statement_type)
    with pytest.raises(InvalidQueryError, match="Only a single read-only SELECT"):
        make_runner(client).run("...")
    assert len(client.query_calls) == 1


def test_tables_outside_allowlist_are_refused():
    client = FakeClient(tables=[PRODUCTS, "bigquery-public-data.thelook_ecommerce.events"])
    with pytest.raises(InvalidQueryError, match="not allowed: bigquery-public-data.thelook_ecommerce.events"):
        make_runner(client).run("...")
    assert len(client.query_calls) == 1


def test_syntax_error_is_fixable_and_message_is_clean():
    error = gexc.BadRequest(
        "POST https://bigquery.googleapis.com/bigquery/v2/projects/my-billing-proj/jobs: Syntax error",
        errors=[
            {"reason": "invalidQuery", "message": 'Syntax error: Unexpected identifier "SELEC" at [1:1]'}
        ],
    )
    with pytest.raises(InvalidQueryError) as info:
        make_runner(FakeClient(dry_error=error)).run("SELEC 1")
    assert info.value.fixable
    assert str(info.value) == 'Syntax error: Unexpected identifier "SELEC" at [1:1]'


def test_noise_is_stripped_when_no_structured_message():
    error = gexc.Forbidden(
        "POST https://bigquery.googleapis.com/bigquery/v2/projects/my-billing-proj/jobs?prettyPrint=false: "
        "Access Denied: Project my-billing-proj  Location: US Job ID: 0d95be53"
    )
    translated = translate_error(error, project="my-billing-proj")
    assert isinstance(translated, BigQueryUnavailableError) and not translated.transient
    assert str(translated) == "BigQuery access denied: Access Denied: Project <billing-project>"


def test_not_found_is_a_setup_problem_not_a_sql_error():
    error = gexc.NotFound(
        "Not found", errors=[{"reason": "notFound", "message": "Not found: Project my-proj"}]
    )
    translated = translate_error(error, project="my-proj")
    assert isinstance(translated, BigQueryUnavailableError)
    assert not translated.transient and not translated.fixable
    assert "Not found: Project <billing-project>" in str(translated) and "GCP_PROJECT" in str(translated)


def test_bytes_billed_limit_arrives_as_500_but_is_not_an_outage():
    error = gexc.InternalServerError(
        "Query exceeded limit for bytes billed",
        errors=[
            {"reason": "bytesBilledLimitExceeded", "message": "Query exceeded limit for bytes billed: 1000."}
        ],
    )
    with pytest.raises(QueryTooExpensiveError):
        make_runner(FakeClient(exec_error=error)).run("SELECT 1")


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (gexc.ServiceUnavailable("backend down"), True),
        (gexc.RetryError("deadline", cause=gexc.ServiceUnavailable("backend down")), True),
        (gexc.TooManyRequests("slow down"), True),
        (gexc.Forbidden("quota", errors=[{"reason": "quotaExceeded", "message": "Quota exceeded"}]), True),
        (auth_exc.TransportError("connection reset"), True),
        (gexc.Forbidden("nope", errors=[{"reason": "accessDenied", "message": "Access Denied"}]), False),
        (auth_exc.RefreshError("invalid_grant"), False),
    ],
)
def test_service_failures_are_not_fixable_by_the_agent(error, transient):
    with pytest.raises(BigQueryUnavailableError) as info:
        make_runner(FakeClient(dry_error=error)).run("SELECT 1")
    assert info.value.transient is transient
    assert not info.value.fixable


def test_client_side_timeout_cancels_the_job():
    client = FakeClient(result_error=TimeoutError())
    with pytest.raises(QueryTimeoutError, match="did not finish in time"):
        make_runner(client).run("SELECT 1")
    assert client.jobs[-1].cancelled


def test_programming_errors_are_not_swallowed():
    with pytest.raises(ValueError, match="bug"):
        make_runner(FakeClient(dry_error=ValueError("bug"))).run("SELECT 1")


def test_client_is_created_lazily_and_setup_errors_are_explained(monkeypatch):
    def no_credentials(**_kwargs):
        raise auth_exc.DefaultCredentialsError("no ADC")

    monkeypatch.setattr(bq_client.bigquery, "Client", no_credentials)
    runner = BigQueryRunner(project_id="p")  # must not fail here
    with pytest.raises(BigQueryUnavailableError, match="gcloud auth application-default login"):
        runner.dry_run("SELECT 1")


def test_missing_project_is_explained(monkeypatch):
    def no_project(**_kwargs):
        raise OSError("Project was not passed and could not be determined from the environment.")

    monkeypatch.setattr(bq_client.bigquery, "Client", no_project)
    with pytest.raises(BigQueryUnavailableError, match="Set GCP_PROJECT"):
        BigQueryRunner(project_id=None).dry_run("SELECT 1")


def test_original_api_is_kept_and_goes_through_the_same_limits():
    client = FakeClient(df=pd.DataFrame({"n": [1, 2]}))
    client.get_table = lambda table_id, **kw: SimpleNamespace(
        table_id=table_id,
        schema=[SimpleNamespace(name="id", field_type="INTEGER", mode="NULLABLE", description=None)],
    )
    runner = BigQueryRunner("my-billing-proj", "bigquery-public-data.thelook_ecommerce", client=client)

    assert runner.execute_query("SELECT 1").n.tolist() == [1, 2]
    assert client.query_calls[0]["config"].dry_run  # dry-run and caps apply here too
    assert runner.get_table_schema("users") == [
        {"name": "id", "type": "INTEGER", "mode": "NULLABLE", "description": ""}
    ]
