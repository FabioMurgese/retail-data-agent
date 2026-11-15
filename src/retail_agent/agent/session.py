"""One chat session = one user, with every dependency bound to that user's identity and scope."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

from retail_agent.agent.report_tools import make_report_tools
from retail_agent.agent.tools import QueryRunner, make_run_sql
from retail_agent.config import Settings
from retail_agent.data.bq_client import BigQueryRunner
from retail_agent.golden import Embedder, GeminiEmbedder, GoldenRetriever, load_trios
from retail_agent.observability import EventStore
from retail_agent.persona import PERSONA_FILE, PersonaStore
from retail_agent.policy import DataPolicy, UserProfile, load_data_policy, load_user_directory
from retail_agent.resilience import CircuitBreaker
from retail_agent.safety.pii_masker import PiiMasker
from retail_agent.safety.sql_guard import SqlGuard
from retail_agent.storage import Database, PreferenceStore, ReportStore

DB_FILE = "agent.db"


@dataclass
class AgentSession:
    user_id: str
    profile: UserProfile
    settings: Settings
    policy: DataPolicy
    masker: PiiMasker
    persona: PersonaStore
    golden: GoldenRetriever
    reports: ReportStore
    preferences: PreferenceStore
    events: EventStore
    tools: list[BaseTool]
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def create(
        cls,
        settings: Settings,
        user_id: str,
        *,
        runner: QueryRunner | None = None,
        embedder: Embedder | None = None,
        use_embeddings: bool = True,
    ) -> AgentSession:
        """Load config for `user_id` (raises UnknownUserError / ConfigFileError) and wire the tools.

        `runner` and `embedder` are injectable for tests; by default they talk to BigQuery and Gemini.
        """
        policy = load_data_policy(settings.config_dir)
        profile = load_user_directory(settings.config_dir).get(user_id)
        masker = PiiMasker.from_policy(policy)
        guard = SqlGuard(policy, profile.scope, max_rows=settings.bq_max_rows)
        if runner is None:
            runner = BigQueryRunner.from_settings(settings, allowed_tables=policy.table_ids)
        if embedder is None and use_embeddings and settings.gemini_api_key is not None:
            embedder = GeminiEmbedder(settings.gemini_api_key.get_secret_value(), settings.embedding_model)
        golden = GoldenRetriever(
            load_trios(settings.golden_dir),
            embedder=embedder,
            cache_path=settings.cache_dir / "golden_embeddings.json",
        )
        run_sql = make_run_sql(
            guard=guard,
            runner=runner,
            masker=masker,
            user_id=user_id,
            max_rows_shown=settings.max_rows_to_llm,
            breaker=CircuitBreaker("bigquery", failure_threshold=2, cooldown_s=settings.breaker_cooldown_s),
            max_failures=settings.max_sql_failures_per_turn,
            max_queries=settings.max_sql_per_turn,
        )
        database = Database(settings.data_dir / DB_FILE)
        reports = ReportStore(database, settings.reports_dir)
        preferences = PreferenceStore(database)
        report_tools = make_report_tools(
            reports=reports, preferences=preferences, masker=masker, user_id=user_id
        )
        persona = PersonaStore(settings.config_dir / PERSONA_FILE)
        persona.get()  # a broken persona file fails at startup with a clear message, not mid-turn
        return cls(
            user_id=user_id,
            profile=profile,
            settings=settings,
            policy=policy,
            masker=masker,
            persona=persona,
            golden=golden,
            reports=reports,
            preferences=preferences,
            events=EventStore(database, settings.logs_dir / "agent.jsonl"),
            tools=[run_sql, *report_tools],
        )

    def new_thread(self) -> None:
        self.thread_id = uuid.uuid4().hex[:12]
