"""SQL guard: the only path from LLM-written SQL to BigQuery.

The LLM is untrusted. Whatever SQL it writes is parsed with sqlglot (BigQuery dialect) and must be a
single read-only query over the allowed tables. Then every table reference is replaced by a
subquery that

  (a) selects only the columns allowed by `config/pii_policy.yaml`, so PII columns do not exist for
      the query at all, not even through `SELECT *` or `TO_JSON_STRING(t)`;
  (b) keeps only the rows related to the current user's products (`config/users.yaml`).

For a user scoped to the Women department:

    SELECT u.country, COUNT(*) AS n FROM users AS u GROUP BY 1

becomes (abridged)

    SELECT u.country, COUNT(*) AS n FROM (
      SELECT id, age, gender, city, state, country, traffic_source, created_at
      FROM `bigquery-public-data`.`thelook_ecommerce`.`users`
      WHERE id IN (SELECT user_id FROM ...order_items WHERE product_id IN
                     (SELECT id FROM ...products WHERE brand IN ('Carhartt')))
    ) AS u GROUP BY 1 LIMIT 1001

Refusals raise `SqlGuardError` with a message written for the LLM, so it can fix its query, and a
stable `code` for metrics. `BigQueryRunner` re-checks the output with BigQuery's own parser
(dry-run) as a second line of defence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError

from retail_agent.policy import DataPolicy, ProductScope

DIALECT = "bigquery"

# sqlglot warns when it falls back to an opaque `Command` (e.g. EXECUTE IMMEDIATE); we reject those anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Never part of an analysis query. The root must also be a query, so this is defence in depth.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Export,
    exp.Declare,
    exp.Set,
    exp.Transaction,
    exp.Into,
)
# Functions that reveal who runs the query, or reach outside the dataset.
_FORBIDDEN_FUNCTIONS = frozenset({"SESSION_USER", "EXTERNAL_QUERY", "EXTERNAL_OBJECT_TRANSFORM"})


class SqlGuardError(Exception):
    """The query was refused. `str(error)` is written for the LLM; `code` is stable, for metrics."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GuardedQuery:
    sql: str  # what BigQuery will run
    tables: tuple[str, ...]  # policy tables read by the query
    limit_applied: bool  # True if the guard added or lowered the outer LIMIT


class SqlGuard:
    """Rewrites SQL for one user. Build one per session: the scoped sources are precomputed."""

    def __init__(self, policy: DataPolicy, scope: ProductScope, *, max_rows: int) -> None:
        self.policy = policy
        self.scope = scope
        self.row_cap = max_rows + 1  # one extra row lets the runner report truncation
        self._project, self._dataset = policy.dataset.split(".")
        self._pii_columns = {column.lower() for table in policy.tables.values() for column in table.pii}
        self._tables_hint = ", ".join(policy.tables)
        self._tables_by_lower = {name.lower(): name for name in policy.tables}
        self._sources = {name: self._scoped_source(name) for name in policy.tables}

    def check(self, sql: str) -> GuardedQuery:
        tree = self._parse_single_query(sql)
        self._reject_forbidden(tree)
        self._reject_pii_columns(tree)

        cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
        read: set[str] = set()
        for ref in list(tree.find_all(exp.Table)):  # materialised: injected subqueries are not revisited
            name = self._resolve(ref, cte_names)
            if name is None:
                continue
            alias = ref.args.get("alias") or exp.TableAlias(this=exp.to_identifier(name))
            ref.replace(exp.Subquery(this=self._sources[name].copy(), alias=alias.copy()))
            read.add(name)

        tree, limit_applied = self._enforce_limit(tree)
        return GuardedQuery(
            sql=tree.sql(dialect=DIALECT, comments=False, pretty=True),
            tables=tuple(sorted(read)),
            limit_applied=limit_applied,
        )

    # --- validation ---------------------------------------------------------------------------

    def _parse_single_query(self, sql: str) -> exp.Query:
        if not sql or not sql.strip():
            raise SqlGuardError("The SQL is empty.", code="parse_error")
        try:
            statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
        except ParseError as exc:
            first = exc.errors[0] if exc.errors else {}
            where = f" at line {first['line']}, column {first['col']}" if "line" in first else ""
            raise SqlGuardError(
                f"SQL syntax error: {first.get('description', 'cannot parse the query')}{where}.",
                code="parse_error",
            ) from None
        except SqlglotError as exc:
            raise SqlGuardError(f"SQL syntax error: {exc}.", code="parse_error") from None

        if len(statements) != 1:
            raise SqlGuardError(
                "Send exactly one SELECT statement: no scripts or multiple statements.",
                code="multiple_statements",
            )
        tree = statements[0]
        if not isinstance(tree, exp.Query):
            raise SqlGuardError("Only read-only SELECT queries are allowed.", code="not_select")
        return tree

    def _reject_forbidden(self, tree: exp.Expression) -> None:
        if node := tree.find(*_FORBIDDEN_NODES):
            raise SqlGuardError(
                f"Only read-only SELECT queries are allowed (found {node.key.upper()}).", code="not_select"
            )
        if tree.find(exp.Parameter, exp.SessionParameter, exp.Placeholder):
            raise SqlGuardError(
                "Query parameters and system variables (@name, @@name) are not allowed.",
                code="forbidden_function",
            )
        for func in tree.find_all(exp.SessionUser, exp.Anonymous):
            name = func.sql_name() if isinstance(func, exp.SessionUser) else str(func.this).upper()
            if name in _FORBIDDEN_FUNCTIONS:
                raise SqlGuardError(f"The function {name} is not allowed.", code="forbidden_function")

    def _reject_pii_columns(self, tree: exp.Expression) -> None:
        # The rewrite below already makes PII columns unreachable; this gives a clear, early answer.
        for column in tree.find_all(exp.Column):
            if column.name.lower() in self._pii_columns:
                raise SqlGuardError(
                    f"`{column.name}` is personal data (PII) and cannot be queried. Refer to customers by "
                    "users.id and describe them with age, gender, city, state, country or traffic_source.",
                    code="pii_column",
                )

    def _resolve(self, ref: exp.Table, cte_names: set[str]) -> str | None:
        """Policy table name for a reference, None for a CTE reference; raises for anything else."""
        if not isinstance(ref.this, exp.Identifier):  # table functions, e.g. ML.PREDICT(...)
            raise SqlGuardError(
                f"`{ref.sql(dialect=DIALECT)}` is not allowed. Query only these tables: {self._tables_hint}.",
                code="forbidden_table",
            )
        parts = [part for part in (ref.catalog, ref.db, ref.name) if part]
        if len(parts) == 1 and ref.name in cte_names:
            return None
        # Case-insensitive: models often write `TheLook_eCommerce.Orders`. Safe, because the reference
        # is replaced by the canonical table anyway; it saves a wasted correction round trip.
        name = self._tables_by_lower.get(ref.name.lower())
        if name:
            project, dataset, table = self._project.lower(), self._dataset.lower(), name.lower()
            if [part.lower() for part in parts] in ([table], [dataset, table], [project, dataset, table]):
                return name
        raise SqlGuardError(
            f"Table `{'.'.join(parts)}` is not available. Query only these tables: {self._tables_hint}.",
            code="forbidden_table",
        )

    # --- rewriting ----------------------------------------------------------------------------

    def _enforce_limit(self, tree: exp.Query) -> tuple[exp.Query, bool]:
        limit = tree.args.get("limit")
        value = limit.expression if isinstance(limit, exp.Limit) else None
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= self.row_cap:
            return tree, False
        return tree.limit(self.row_cap, copy=False), True

    def _base(self, table: str) -> exp.Table:
        return exp.table_(table, db=self._dataset, catalog=self._project, quoted=True)

    def _scoped_source(self, table: str) -> exp.Select:
        columns = [exp.column(name) for name in self.policy.tables[table].columns]
        source = exp.select(*columns).from_(self._base(table))
        condition = self._row_filter(table)
        return source.where(condition) if condition is not None else source

    def _row_filter(self, table: str) -> exp.Expression | None:
        """How the product scope restricts each table (None = no restriction).

        products     -> the scope filters themselves
        order_items  -> items of in-scope products
        orders       -> orders containing at least one in-scope item
        users        -> customers who bought at least one in-scope item
        """
        if self.scope.all_products:
            return None
        product_filter = exp.and_(
            *(exp.column(column).isin(*values) for column, values in self.scope.filters.items())
        )
        if table == "products":
            return product_filter

        scoped_products = exp.select("id").from_(self._base("products")).where(product_filter)
        items_of_scope = exp.column("product_id").isin(query=scoped_products)
        if table == "order_items":
            return items_of_scope

        def from_scoped_items(column: str) -> exp.Select:
            return exp.select(column).from_(self._base("order_items")).where(items_of_scope.copy())

        if table == "orders":
            return exp.column("order_id").isin(query=from_scoped_items("order_id"))
        if table == "users":
            return exp.column("id").isin(query=from_scoped_items("user_id"))
        # Fail closed: a new table needs an explicit rule before scoped users can query it.
        raise ValueError(f"no product-scope rule for table {table!r}; add one to SqlGuard._row_filter")
