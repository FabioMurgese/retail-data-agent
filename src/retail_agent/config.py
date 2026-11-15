"""Runtime settings, read from environment variables and `.env` (template: `.env.example`).

Relative paths resolve against the working directory: run the CLI from the repository root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gemini (Google AI Studio) ---
    gemini_api_key: SecretStr | None = None
    agent_model: str = "gemini-3.5-flash-lite"
    # Comma-separated in the environment, tried in order when the primary model fails or is rate limited.
    fallback_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["gemini-3.1-flash-lite", "gemini-3.6-flash"]
    )
    embedding_model: str = "gemini-embedding-001"
    # Free-tier requests per minute, per model (AI Studio). Env format: "model=rpm,model=rpm".
    default_model_rpm: int = Field(15, gt=0)
    model_rpm: Annotated[dict[str, int], NoDecode] = Field(
        default_factory=lambda: {"gemini-3-flash": 5, "gemini-3.6-flash": 5, "gemini-3.8-flash": 5}
    )
    llm_timeout_s: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(2, ge=1)  # attempts per model (1 = no retry) before falling back

    # --- Resilience and cost caps (per question) ---
    breaker_cooldown_s: float = Field(60.0, gt=0)  # how long a failing dependency is skipped
    max_sql_failures_per_turn: int = Field(3, gt=0)  # failed, refused or empty queries before giving up
    max_sql_per_turn: int = Field(10, gt=0)
    max_tokens_per_turn: int = Field(200_000, gt=0)  # LLM tokens (input + output) for one question

    # --- Agent ---
    recursion_limit: int = Field(30, gt=0)  # LangGraph steps per turn (each LLM call or tool run is one)
    max_rows_to_llm: int = Field(50, gt=0)  # rows of a query result the model reads
    golden_top_k: int = Field(3, ge=0)  # analyst examples added to the prompt

    # --- BigQuery (auth: Application Default Credentials) ---
    gcp_project: str | None = None  # billing/quota project; queries on the public dataset are charged here
    bq_dataset: str = "bigquery-public-data.thelook_ecommerce"
    bq_location: str = "US"
    # BigQuery bills at least 10 MB per table referenced, so keep this well above 40 MB.
    bq_max_bytes_billed: int = Field(1_000_000_000, gt=0)
    bq_timeout_s: float = Field(60.0, gt=0)
    bq_max_rows: int = Field(1_000, gt=0)
    bq_retry_deadline_s: float = Field(30.0, gt=0)  # total time spent retrying transient errors

    # --- Cost estimates in /stats (USD). The free tier costs nothing: set list prices to estimate. ---
    llm_price_input_per_mtok: float = Field(0.0, ge=0)
    llm_price_output_per_mtok: float = Field(0.0, ge=0)
    bq_price_per_tib: float = Field(6.25, ge=0)  # on-demand list price; the first TiB per month is free

    # --- Local files ---
    config_dir: Path = Path("config")
    golden_dir: Path = Path("golden/trios")
    data_dir: Path = Path("data")  # SQLite: reports, audit log, preferences
    reports_dir: Path = Path("reports")  # Markdown export of saved reports
    cache_dir: Path = Path(".cache")
    logs_dir: Path = Path("logs")
    golden_candidates_dir: Path = Path("golden/candidates")  # trios proposed via /good, for analyst review

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("model_rpm", mode="before")
    @classmethod
    def _parse_rpm(cls, value: object) -> object:
        if isinstance(value, str):
            pairs = (item.split("=", 1) for item in value.split(",") if item.strip())
            return {name.strip(): int(rpm) for name, rpm in pairs}
        return value

    @field_validator("bq_dataset")
    @classmethod
    def _check_dataset(cls, value: str) -> str:
        if value.count(".") != 1:
            raise ValueError("expected '<project>.<dataset>'")
        return value

    @property
    def model_chain(self) -> list[str]:
        """Primary model followed by the fallbacks, without duplicates."""
        return list(dict.fromkeys([self.agent_model, *self.fallback_models]))

    def rpm_for(self, model: str) -> int:
        return self.model_rpm.get(model, self.default_model_rpm)


@lru_cache
def get_settings() -> Settings:
    return Settings()
