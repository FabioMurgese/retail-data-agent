"""`retail-agent doctor`: check configuration, credentials and policy files against live services.

Cheap by design: Gemini models are looked up, never called (no generation quota used), and
BigQuery runs one small query on `products` (about 10 MB billed, the per-table minimum).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sqlglot import exp

from retail_agent.config import Settings
from retail_agent.curation import load_glossary
from retail_agent.data.bq_client import BigQueryError, BigQueryRunner
from retail_agent.golden import load_trios
from retail_agent.persona import PERSONA_FILE, PersonaStore
from retail_agent.policy import (
    ConfigFileError,
    DataPolicy,
    UserDirectory,
    load_data_policy,
    load_user_directory,
)
from retail_agent.safety.sql_guard import SqlGuard, SqlGuardError
from retail_agent.storage import Database, ReportStore


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str = ""


def run_checks(settings: Settings, env_file: Path = Path(".env")) -> Iterator[Check]:
    """Yield checks one at a time so the CLI can show progress; later checks skip if a prerequisite failed."""
    if env_file.exists():
        yield Check("Environment", Status.OK, f"{env_file} found")
    else:
        yield Check(
            "Environment", Status.WARN, f"no {env_file} in the working directory; using process env only"
        )

    policy, users = None, None
    try:
        policy = load_data_policy(settings.config_dir)
        exposed = sum(len(t.columns) for t in policy.tables.values())
        pii = sum(len(t.pii) for t in policy.tables.values())
        yield Check(
            "Data policy", Status.OK, f"{len(policy.tables)} tables, {exposed} exposed and {pii} PII columns"
        )
    except ConfigFileError as exc:
        yield Check("Data policy", Status.FAIL, str(exc))
    try:
        users = load_user_directory(settings.config_dir)
        yield Check("Users", Status.OK, f"{len(users.users)} users: {', '.join(users.users)}")
    except ConfigFileError as exc:
        yield Check("Users", Status.FAIL, str(exc))
    yield from _content_checks(settings, policy, users)
    yield _storage_check(settings)

    yield from _bigquery_checks(settings, policy, users)
    yield from _gemini_checks(settings)


def _content_checks(
    settings: Settings, policy: DataPolicy | None, users: UserDirectory | None
) -> Iterator[Check]:
    """Persona and golden trios: business-owned files, so validate them like code."""
    try:
        persona = PersonaStore(settings.config_dir / PERSONA_FILE).get()
        yield Check("Persona", Status.OK, f"version {persona.version} by {persona.updated_by}")
    except ConfigFileError as exc:
        yield Check("Persona", Status.FAIL, str(exc))
    try:
        trios = load_trios(settings.golden_dir)
    except ConfigFileError as exc:
        yield Check("Golden trios", Status.FAIL, str(exc))
        return
    if not trios:
        yield Check("Golden trios", Status.WARN, f"none found in {settings.golden_dir}")
        return
    broken = []
    if policy is not None and users is not None:  # every trio must pass the SQL guard for every user
        for trio in trios:
            for user_id, profile in users.users.items():
                try:
                    SqlGuard(policy, profile.scope, max_rows=settings.bq_max_rows).check(trio.sql)
                except SqlGuardError as exc:
                    broken.append(f"{trio.id} ({user_id}: {exc.code})")
                    break
    if broken:
        yield Check("Golden trios", Status.FAIL, f"SQL refused by the guard: {', '.join(broken)}")
    else:
        probation = sum(t.status == "probation" for t in trios)
        detail = f"{len(trios)} trios" + (
            f" ({probation} on probation, approved automatically)" if probation else ""
        )
        yield Check("Golden trios", Status.OK, detail)
    try:
        glossary = load_glossary(settings.glossary_path)
        yield Check("Glossary", Status.OK, f"{len(glossary.metrics)} metrics, version {glossary.version}")
    except ConfigFileError as exc:
        yield Check("Glossary", Status.FAIL, str(exc))


def _storage_check(settings: Settings) -> Check:
    path = settings.data_dir / "agent.db"
    try:
        store = ReportStore(Database(path))
        live = len(store.live_report_ids())
        return Check("Storage", Status.OK, f"{path} ({live} saved reports)")
    except (OSError, sqlite3.Error) as exc:
        return Check("Storage", Status.FAIL, f"{path}: {exc}")


def _bigquery_checks(
    settings: Settings, policy: DataPolicy | None, users: UserDirectory | None
) -> Iterator[Check]:
    if not settings.gcp_project:
        yield Check("BigQuery", Status.FAIL, "GCP_PROJECT is not set (.env)")
        return
    if policy is None:
        yield Check("BigQuery", Status.SKIP, "needs a valid data policy")
        return

    runner = BigQueryRunner.from_settings(settings, allowed_tables=policy.table_ids)
    try:
        for table, table_policy in policy.tables.items():
            actual = [column["name"] for column in runner.get_table_schema(table)]
            declared = [*table_policy.columns, *table_policy.pii]
            if missing := [c for c in declared if c not in actual]:
                yield Check(
                    f"Schema {table}", Status.FAIL, f"policy lists columns BigQuery does not have: {missing}"
                )
            elif unclassified := policy.unclassified_columns(table, actual):
                yield Check(
                    f"Schema {table}", Status.WARN, f"new columns, hidden until classified: {unclassified}"
                )
            else:
                yield Check(
                    f"Schema {table}",
                    Status.OK,
                    f"{len(table_policy.columns)} exposed, {len(table_policy.pii)} PII",
                )
        if users is not None:
            yield from _scope_checks(runner, policy, users)
    except BigQueryError as exc:
        yield Check("BigQuery", Status.FAIL, str(exc))


def _scope_checks(runner: BigQueryRunner, policy: DataPolicy, users: UserDirectory) -> Iterator[Check]:
    """Every scope value must exist in `products`; a typo would silently match nothing."""
    wanted: dict[str, set[str]] = defaultdict(set)
    for profile in users.users.values():
        for column, values in profile.scope.filters.items():
            wanted[column].update(values)

    found: set[tuple[str, str]] = set()
    if wanted:
        products = policy.table_id("products")
        selects = [
            f"SELECT '{column}' AS field, {column} AS value FROM `{products}` "
            f"WHERE {column} IN ({', '.join(_sql_string(v) for v in sorted(values))}) GROUP BY value"
            for column, values in sorted(wanted.items())
        ]
        result = runner.run("\nUNION ALL\n".join(selects), labels={"purpose": "doctor"})
        found = set(zip(result.df["field"], result.df["value"], strict=True))

    for user_id, profile in users.users.items():
        scope = profile.scope
        unknown = [f"{c}={v!r}" for c, values in scope.filters.items() for v in values if (c, v) not in found]
        summary = (
            "all products"
            if scope.all_products
            else "; ".join(f"{column} in {list(values)}" for column, values in scope.filters.items())
        )
        if unknown:
            yield Check(
                f"Scope {user_id}", Status.FAIL, f"values not found in products: {', '.join(unknown)}"
            )
        else:
            yield Check(f"Scope {user_id}", Status.OK, f"{profile.role}: {summary}")


def _gemini_checks(settings: Settings) -> Iterator[Check]:
    if settings.gemini_api_key is None:
        yield Check("Gemini", Status.FAIL, "GEMINI_API_KEY is not set (.env)")
        return

    from google import genai  # slow import; only needed here

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    roles = [("agent", settings.agent_model, Status.FAIL)]
    roles += [("fallback", model, Status.WARN) for model in settings.model_chain[1:]]
    roles += [("embeddings", settings.embedding_model, Status.WARN)]  # retrieval falls back to BM25
    for role, model, status_if_missing in roles:
        try:
            info = client.models.get(model=model)
            yield Check(f"Gemini {role}", Status.OK, f"{model} ({info.display_name or 'available'})")
        except Exception as exc:  # diagnostic command: report anything, never crash
            yield Check(f"Gemini {role}", status_if_missing, f"{model}: {_short(exc)}")


def _sql_string(value: str) -> str:
    return exp.Literal.string(value).sql(dialect="bigquery")


def _short(exc: Exception, limit: int = 160) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"
