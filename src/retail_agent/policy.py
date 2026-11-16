"""Data-access policy: which columns the agent may see, and which products each user may analyse.

Both files are meant to be edited by data stewards, not developers:

- `config/pii_policy.yaml` is an **allowlist**. A column is usable only if it is listed under
  `columns`. PII columns are listed under `pii` so the output masker and the docs know what they
  are. A column in neither list (e.g. one added upstream later) stays hidden.
- `config/users.yaml` maps each user to the brands they are responsible for (or all products).
  Scopes fail closed: a user without an explicit scope cannot be loaded.

This module only validates and exposes the policy; `safety.sql_guard` enforces it on SQL.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PiiKind = Literal["name", "email", "address", "postal_code", "geo"]
ModelT = TypeVar("ModelT", bound=BaseModel)

PII_POLICY_FILE = "pii_policy.yaml"
USERS_FILE = "users.yaml"


class ConfigFileError(ValueError):
    """A YAML config file is missing or invalid. The message names the file and the offending field."""


class UnknownUserError(LookupError):
    def __init__(self, user_id: str, known: Iterable[str]) -> None:
        super().__init__(f"Unknown user {user_id!r}. Known users: {', '.join(sorted(known))}")
        self.user_id = user_id


class _Strict(BaseModel):
    # Typos in a policy file must fail loudly, not be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class TablePolicy(_Strict):
    description: str
    columns: dict[str, str] = Field(min_length=1)  # exposed column -> meaning (shown to the LLM)
    pii: dict[str, PiiKind] = Field(default_factory=dict)  # hidden column -> kind of PII

    @model_validator(mode="after")
    def _exposed_and_pii_are_disjoint(self) -> TablePolicy:
        if clash := sorted(self.columns.keys() & self.pii.keys()):
            raise ValueError(f"columns listed both as exposed and as PII: {clash}")
        return self


class DataPolicy(_Strict):
    dataset: str  # "<project>.<dataset>"
    tables: dict[str, TablePolicy] = Field(min_length=1)

    def table_id(self, table: str) -> str:
        return f"{self.dataset}.{table}"

    @property
    def table_ids(self) -> frozenset[str]:
        return frozenset(self.table_id(name) for name in self.tables)

    def unclassified_columns(self, table: str, actual_columns: Iterable[str]) -> list[str]:
        """Columns present in BigQuery but in neither list: hidden, and worth a steward's review."""
        policy = self.tables[table]
        known = policy.columns.keys() | policy.pii.keys()
        return [column for column in actual_columns if column not in known]


class ProductScope(_Strict):
    """Products a user may analyse. Access is by brand only: a manager is responsible for one or more
    brands; the CEO has `all_products`. Products without a brand are in no manager's scope."""

    all_products: bool = False
    brands: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _brand_is_the_only_dimension(cls, data: object) -> object:
        if isinstance(data, dict) and (other := sorted({"departments", "categories"} & data.keys())):
            raise ValueError(f"access is by brand only: replace {', '.join(other)} with `brands: [...]`")
        return data

    @model_validator(mode="after")
    def _explicit_scope(self) -> ProductScope:
        if self.all_products == bool(self.brands):
            raise ValueError("set either `all_products: true` or a non-empty `brands` list, not both")
        return self

    @property
    def filters(self) -> dict[str, tuple[str, ...]]:
        """Non-empty filters, keyed by the `products` column they apply to (only `brand`)."""
        return {"brand": self.brands} if self.brands else {}


class UserProfile(_Strict):
    display_name: str
    role: str
    scope: ProductScope


class UserDirectory(_Strict):
    users: dict[str, UserProfile] = Field(min_length=1)

    def get(self, user_id: str) -> UserProfile:
        try:
            return self.users[user_id]
        except KeyError:
            raise UnknownUserError(user_id, self.users) from None


def load_yaml_model(model: type[ModelT], path: Path) -> ModelT:
    """Parse a YAML file into a pydantic model; any problem becomes a one-line `ConfigFileError`."""
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return model.model_validate(raw)
    except FileNotFoundError:
        raise ConfigFileError(f"{path}: file not found") from None
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"{path}: invalid YAML: {exc}") from exc
    except ValidationError as exc:
        # One short line per problem, e.g. "tables.users.colums: Extra inputs are not permitted".
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())
        raise ConfigFileError(f"{path}: {problems}") from exc


def load_data_policy(config_dir: Path) -> DataPolicy:
    return load_yaml_model(DataPolicy, config_dir / PII_POLICY_FILE)


def load_user_directory(config_dir: Path) -> UserDirectory:
    return load_yaml_model(UserDirectory, config_dir / USERS_FILE)
