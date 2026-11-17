"""SQL guard: what is refused, how tables are rewritten, and invariants on every accepted query."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

from retail_agent.policy import ProductScope, load_data_policy, load_user_directory
from retail_agent.safety.sql_guard import SqlGuard, SqlGuardError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
POLICY = load_data_policy(CONFIG_DIR)
USERS = load_user_directory(CONFIG_DIR)
PROJECT, DATASET = POLICY.dataset.split(".")
PII = {column for table in POLICY.tables.values() for column in table.pii}
SCOPE_KEYS = {"id", "order_id", "user_id"}  # columns selected by the generated scope subqueries
MAX_ROWS = 100


def guard_for(user_id: str) -> SqlGuard:
    return SqlGuard(POLICY, USERS.get(user_id).scope, max_rows=MAX_ROWS)


def assert_reads_are_guarded(sql: str, *, scoped: bool) -> exp.Expression:
    """Every read of a base table projects allowed columns only and, for scoped users, is filtered."""
    tree = sqlglot.parse_one(sql, dialect="bigquery")
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not table.db and table.name in cte_names:
            continue
        assert (table.catalog, table.db) == (PROJECT, DATASET), f"unqualified base table in {sql}"
        select = table.parent_select
        projected = {e.alias_or_name for e in select.expressions}
        assert projected <= set(POLICY.tables[table.name].columns) | SCOPE_KEYS
        assert not projected & PII
        if scoped:
            assert select.args.get("where") is not None, f"unfiltered read of {table.name}"
    return tree


# --- accepted queries -----------------------------------------------------------------------------

ACCEPTED = [
    "SELECT * FROM users",
    "SELECT TO_JSON_STRING(u) FROM users AS u",  # a whole row as JSON: still only safe columns
    "SELECT u FROM users u",
    "SELECT o.status, COUNT(*) FROM `bigquery-public-data.thelook_ecommerce.orders` AS o GROUP BY 1",
    "SELECT * FROM `bigquery-public-data`.thelook_ecommerce.order_items",
    "SELECT p.brand, SUM(oi.sale_price) FROM thelook_ecommerce.order_items oi "
    "JOIN products p ON p.id = oi.product_id GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    "WITH spend AS (SELECT user_id, SUM(sale_price) AS total FROM order_items GROUP BY 1) "
    "SELECT u.country, AVG(s.total) FROM spend s JOIN users u ON u.id = s.user_id GROUP BY 1",
    "SELECT id FROM users UNION ALL SELECT id FROM products",
    "SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE age > 60)",
    # a CTE named like a real table must not dodge the rewrite of the real table inside it
    "WITH products AS (SELECT * FROM `bigquery-public-data.thelook_ecommerce.products`) "
    "SELECT * FROM products",
    "FROM users |> WHERE age > 30 |> AGGREGATE COUNT(*) AS n",  # pipe syntax
    "SELECT CURRENT_DATE()",
]


@pytest.mark.parametrize("user_id", ["ceo", "alice", "carol", "dan"])
@pytest.mark.parametrize("sql", ACCEPTED)
def test_accepted_queries_only_read_through_guarded_sources(user_id, sql):
    guarded = guard_for(user_id).check(sql)
    assert_reads_are_guarded(guarded.sql, scoped=not USERS.get(user_id).scope.all_products)


def test_table_forms_and_aliases_are_preserved():
    sql = (
        guard_for("ceo")
        .check("SELECT o.order_id, users.age FROM orders o JOIN users ON users.id = o.user_id")
        .sql
    )
    tree = sqlglot.parse_one(sql, dialect="bigquery")
    aliases = {sub.alias for sub in tree.find_all(exp.Subquery)}
    assert aliases == {"o", "users"}  # explicit alias kept; implicit alias = table name


def test_users_subquery_exposes_exactly_the_allowlist():
    tree = sqlglot.parse_one(guard_for("ceo").check("SELECT * FROM users").sql, dialect="bigquery")
    inner = next(tree.find_all(exp.Subquery)).this
    assert [e.alias_or_name for e in inner.expressions] == list(POLICY.tables["users"].columns)


def test_reported_tables_and_comments_stripped():
    guarded = guard_for("ceo").check("SELECT 1 FROM orders JOIN order_items USING (order_id) -- secret note")
    assert guarded.tables == ("order_items", "orders")
    assert "secret note" not in guarded.sql


# --- product scope --------------------------------------------------------------------------------


def where_of(sql: str, table: str) -> str:
    tree = sqlglot.parse_one(sql, dialect="bigquery")
    allowlist = list(POLICY.tables[table].columns)
    for ref in tree.find_all(exp.Table):
        select = ref.parent_select
        if ref.name == table and [e.alias_or_name for e in select.expressions] == allowlist:
            return select.args["where"].sql(dialect="bigquery")
    raise AssertionError(f"no outer read of {table}")


def test_scope_filters_each_table_through_products():
    guard = guard_for("dan")
    carhartt = "brand IN ('Carhartt')"
    assert where_of(guard.check("SELECT * FROM products").sql, "products") == f"WHERE {carhartt}"
    assert carhartt in where_of(guard.check("SELECT * FROM order_items").sql, "order_items")
    orders_where = where_of(guard.check("SELECT * FROM orders").sql, "orders")
    assert orders_where.startswith("WHERE order_id IN (SELECT order_id FROM") and carhartt in orders_where
    users_where = where_of(guard.check("SELECT * FROM users").sql, "users")
    assert users_where.startswith("WHERE id IN (SELECT user_id FROM") and carhartt in users_where


def test_several_brands_are_or_ed():
    sql = guard_for("bob").check("SELECT * FROM products").sql
    assert where_of(sql, "products") == (
        "WHERE brand IN ('The North Face', 'Columbia', 'Arc\\'teryx', 'Canada Goose', 'Mountain Hardwear')"
    )


def test_scope_values_are_escaped():
    sql = guard_for("carol").check("SELECT * FROM products").sql
    assert r"'Levi\'s'" in sql
    sqlglot.parse_one(sql, dialect="bigquery")  # still valid SQL


def test_ansi_quote_escaping_gets_a_hint_the_model_can_act_on():
    with pytest.raises(SqlGuardError, match=r"quote inside a string is written with a backslash") as error:
        guard_for("bob").check("SELECT * FROM products WHERE brand IN ('Arc''teryx')")
    assert error.value.code == "parse_error"
    for fixed in (
        "SELECT * FROM products WHERE brand = 'Arc\\'teryx'",
        'SELECT * FROM products WHERE brand = "Arc\'teryx"',
    ):
        guard_for("bob").check(fixed)  # both forms the hint suggests are accepted


def test_unrestricted_user_has_no_row_filter():
    sql = guard_for("ceo").check("SELECT * FROM users").sql
    assert "WHERE" not in sql


def test_new_table_without_scope_rule_fails_closed():
    extended = POLICY.model_copy(update={"tables": {**POLICY.tables, "events": POLICY.tables["orders"]}})
    with pytest.raises(ValueError, match="no product-scope rule for table 'events'"):
        SqlGuard(extended, ProductScope(brands=("Carhartt",)), max_rows=10)


# --- LIMIT ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected", "applied"),
    [
        ("SELECT * FROM orders", MAX_ROWS + 1, True),
        ("SELECT * FROM orders LIMIT 5", 5, False),
        ("SELECT * FROM orders LIMIT 100000", MAX_ROWS + 1, True),
        ("SELECT id FROM users UNION ALL SELECT id FROM products", MAX_ROWS + 1, True),
    ],
)
def test_outer_limit_is_capped(sql, expected, applied):
    guarded = guard_for("ceo").check(sql)
    tree = sqlglot.parse_one(guarded.sql, dialect="bigquery")
    assert int(tree.args["limit"].expression.this) == expected
    assert guarded.limit_applied is applied


# --- refusals -------------------------------------------------------------------------------------

REFUSED = {
    "parse_error": ["", "SELEC 1", "SELECT FROM WHERE"],
    "multiple_statements": ["SELECT 1; SELECT 2", "SELECT 1; DROP TABLE users", "DECLARE x INT64; SELECT x"],
    "not_select": [
        "DELETE FROM users WHERE TRUE",
        "UPDATE orders SET status = 'x' WHERE TRUE",
        "INSERT INTO orders (order_id) VALUES (1)",
        "DROP TABLE users",
        "CREATE TABLE t AS SELECT 1",
        "MERGE orders o USING orders s ON FALSE WHEN NOT MATCHED THEN INSERT ROW",
        "EXECUTE IMMEDIATE 'SELECT 1'",
        "EXPORT DATA OPTIONS(uri='gs://bucket/*', format='CSV') AS SELECT * FROM users",
    ],
    "forbidden_table": [
        "SELECT * FROM `bigquery-public-data.thelook_ecommerce.events`",
        "SELECT * FROM inventory_items",
        "SELECT * FROM `other-project.thelook_ecommerce.users`",
        "SELECT * FROM other_dataset.users",
        "SELECT * FROM `region-us`.INFORMATION_SCHEMA.JOBS",
        "SELECT * FROM thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS",
        "SELECT * FROM `bigquery-public-data.thelook_ecommerce.*`",
        "SELECT * FROM ML.PREDICT(MODEL m, TABLE t)",
        "SELECT * FROM o, o.items",
    ],
    "forbidden_function": [
        "SELECT SESSION_USER()",
        "SELECT @@project_id",
        "SELECT * FROM users WHERE id = @id",
        "SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')",
    ],
    "pii_column": [
        "SELECT email FROM users",
        "SELECT u.first_name, u.last_name FROM users u",
        "SELECT COUNT(*) FROM users WHERE EMAIL LIKE '%@example.com'",
        "SELECT * EXCEPT (street_address) FROM users",
        "SELECT ST_ASTEXT(user_geom) FROM users",
        "SELECT AVG(latitude) FROM users",
    ],
}


@pytest.mark.parametrize(
    ("code", "sql"), [(code, sql) for code, queries in REFUSED.items() for sql in queries]
)
def test_refused(code, sql):
    with pytest.raises(SqlGuardError) as info:
        guard_for("ceo").check(sql)
    assert info.value.code == code, str(info.value)


def test_refusal_messages_guide_the_llm():
    with pytest.raises(SqlGuardError, match="Query only these tables: orders, order_items, products, users"):
        guard_for("ceo").check("SELECT * FROM events")
    with pytest.raises(SqlGuardError, match="Refer to customers by users.id"):
        guard_for("ceo").check("SELECT email FROM users")


def test_table_names_are_case_insensitive_and_canonicalised():
    sql = guard_for("ceo").check("SELECT COUNT(*) FROM `bigquery-public-data.TheLook_eCommerce.Orders`").sql
    assert "`bigquery-public-data`.`thelook_ecommerce`.`orders`" in sql
