"""Settings parsing and the data-access policy files (the real ones in config/ and broken variants)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_agent.config import Settings
from retail_agent.doctor import Status, _scope_checks
from retail_agent.policy import (
    PII_POLICY_FILE,
    USERS_FILE,
    ConfigFileError,
    ProductScope,
    UnknownUserError,
    load_data_policy,
    load_user_directory,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def clean_env(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return monkeypatch


# --- Settings -------------------------------------------------------------------------------------


def test_settings_defaults_do_not_need_a_dotenv(clean_env):
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key is None and settings.gcp_project is None
    assert settings.bq_dataset == "bigquery-public-data.thelook_ecommerce"
    assert settings.model_chain[0] == settings.agent_model


def test_fallback_models_are_comma_separated_and_deduplicated(clean_env):
    clean_env.setenv("AGENT_MODEL", "m1")
    clean_env.setenv("FALLBACK_MODELS", " m2, ,m1,m3 ")
    settings = Settings(_env_file=None)
    assert settings.fallback_models == ["m2", "m1", "m3"]
    assert settings.model_chain == ["m1", "m2", "m3"]


def test_dataset_must_be_project_dot_dataset(clean_env):
    clean_env.setenv("BQ_DATASET", "thelook_ecommerce")
    with pytest.raises(ValueError, match="project"):
        Settings(_env_file=None)


# --- Real policy files ----------------------------------------------------------------------------

KNOWN_PII = {
    "first_name",
    "last_name",
    "email",
    "street_address",
    "postal_code",
    "latitude",
    "longitude",
    "user_geom",
}


def test_repo_policy_hides_every_known_pii_column():
    users_table = load_data_policy(CONFIG_DIR).tables["users"]
    assert set(users_table.pii) == KNOWN_PII
    assert not KNOWN_PII & set(users_table.columns)


def test_repo_scopes_only_filter_on_exposed_product_columns():
    products = load_data_policy(CONFIG_DIR).tables["products"]
    for profile in load_user_directory(CONFIG_DIR).users.values():
        assert set(profile.scope.filters) <= set(products.columns)


def test_unclassified_columns_are_reported():
    policy = load_data_policy(CONFIG_DIR)
    assert policy.unclassified_columns("users", ["id", "email", "phone_number"]) == ["phone_number"]


# --- Broken policy files fail loudly --------------------------------------------------------------


def write(tmp_path: Path, name: str, text: str) -> Path:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_column_both_exposed_and_pii_is_rejected(tmp_path):
    write(
        tmp_path,
        PII_POLICY_FILE,
        """
dataset: p.d
tables:
  users:
    description: x
    columns: {id: id, email: oops}
    pii: {email: email}
""",
    )
    with pytest.raises(ConfigFileError, match=r"pii_policy.yaml.*both as exposed and as PII.*email"):
        load_data_policy(tmp_path)


def test_typo_in_policy_key_is_rejected(tmp_path):
    write(
        tmp_path,
        PII_POLICY_FILE,
        "dataset: p.d\ntables:\n  users:\n    description: x\n    colums: {id: id}\n",
    )
    with pytest.raises(ConfigFileError, match="colums"):
        load_data_policy(tmp_path)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ConfigFileError, match=r"users.yaml: file not found"):
        load_user_directory(tmp_path)


@pytest.mark.parametrize(
    "scope",
    [{}, {"all_products": False}, {"all_products": True, "brands": ["Levi's"]}],
    ids=["empty", "explicit-false", "all-plus-filter"],
)
def test_scopes_fail_closed(scope):
    with pytest.raises(ValueError, match="all_products"):
        ProductScope.model_validate(scope)


def test_scope_filters_map_to_the_brand_column():
    assert ProductScope(brands=("Levi's", "Carhartt")).filters == {"brand": ("Levi's", "Carhartt")}
    assert ProductScope(all_products=True).filters == {}


@pytest.mark.parametrize("field", ["departments", "categories"])
def test_brand_is_the_only_access_dimension(field):
    with pytest.raises(ValueError, match=f"access is by brand only: replace {field}"):
        ProductScope.model_validate({field: ["Women"]})


def test_unknown_user_lists_known_users(tmp_path):
    write(
        tmp_path,
        USERS_FILE,
        """
users:
  alice: {display_name: A, role: r, scope: {brands: [Levi's]}}
""",
    )
    with pytest.raises(UnknownUserError, match="Known users: alice"):
        load_user_directory(tmp_path).get("mallory")


# --- doctor: scope values must exist in BigQuery --------------------------------------------------


class FakeRunner:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows
        self.sql = ""

    def run(self, sql, labels=None):
        self.sql = sql
        return type("R", (), {"df": pd.DataFrame(self.rows, columns=["field", "value"])})()


def test_doctor_flags_scope_values_missing_from_products():
    policy, users = load_data_policy(CONFIG_DIR), load_user_directory(CONFIG_DIR)
    every_value = [(c, v) for p in users.users.values() for c, vs in p.scope.filters.items() for v in vs]
    runner = FakeRunner([row for row in every_value if row != ("brand", "Levi's")])

    checks = {c.name: c for c in _scope_checks(runner, policy, users)}

    assert r"'Levi\'s'" in runner.sql  # quotes in values are escaped, not concatenated raw
    assert checks["Scope carol"].status is Status.FAIL and "Levi's" in checks["Scope carol"].detail
    assert checks["Scope alice"].status is Status.OK
    assert checks["Scope ceo"].detail.endswith("all products")
