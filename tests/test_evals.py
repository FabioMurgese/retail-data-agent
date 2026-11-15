"""The eval harness itself, offline: case file validity, checks, delete flows, judge handling, report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeRunner, ScriptedModel, sql_call, tool_call
from langchain_core.messages import AIMessage

from retail_agent.agent.graph import build_graph
from retail_agent.evals import Case, Verdict, load_cases, run_suite, write_report
from retail_agent.policy import load_data_policy, load_user_directory
from retail_agent.safety.sql_guard import SqlGuard

ROOT = Path(__file__).resolve().parents[1]
CASES = load_cases(ROOT / "evals" / "cases.yaml")


def test_case_file_is_valid():
    assert len({c.id for c in CASES}) == len(CASES) >= 15
    users = load_user_directory(ROOT / "config")
    policy = load_data_policy(ROOT / "config")
    for case in CASES:
        users.get(case.user)  # every user exists
        if case.expect.reference_sql:  # reference queries run through the same guard, as the case's user
            SqlGuard(policy, users.get(case.user).scope, max_rows=1000).check(case.expect.reference_sql)


def run(settings, cases: list[Case], replies: list[list], runner=None, judge=None):
    scripts = iter(replies)
    return run_suite(
        cases,
        settings,
        make_graph=lambda session: build_graph(session, ScriptedModel(*next(scripts))),
        runner=runner or FakeRunner(df=pd.DataFrame({"n": [2382]})),
        judge=judge,
    )


def case(**fields) -> Case:
    return Case.model_validate(
        {"id": "c", "category": "test", "user": "ceo", "question": "How many orders?"} | fields
    )


def test_reference_values_are_compared_with_the_answer(settings):
    good = case(id="good", expect={"reference_sql": "SELECT COUNT(*) AS n FROM orders", "min_queries": 1})
    bad = case(id="bad", expect={"reference_sql": "SELECT COUNT(*) AS n FROM orders"})
    result = run(
        settings,
        [good, bad],
        [
            [sql_call("SELECT COUNT(*) AS n FROM orders"), AIMessage("We received 2,382 orders.")],
            [sql_call("SELECT COUNT(*) AS n FROM orders"), *[AIMessage("2,382; I guess 2,999 next.")] * 2],
        ],
    )  # fmt: skip
    first, second = result.results
    assert first.passed
    assert not second.passed  # 2,999 is not in any result: grounding catches it, the reference still matches
    assert [c.name for c in second.checks if c.status == "fail"] == ["grounded"]


def test_wrong_number_fails_the_reference_check(settings):
    wrong = case(expect={"reference_sql": "SELECT COUNT(*) AS n FROM orders", "grounded": False})
    replies = [
        sql_call("SELECT 1 FROM orders"),
        *[AIMessage("We received 1,234 orders.")] * 2,
    ]  # draft + revision
    result = run(settings, [wrong], [replies])
    [check] = [c for c in result.results[0].checks if c.name == "reference_value"]
    assert check.status == "fail" and "1,234" in check.detail


def test_delete_cases_seed_reports_and_check_what_was_deleted(settings):
    seeds = [{"title": "Holiday A", "body": "Holiday campaign"}, {"title": "Other", "body": "Returns"}]
    confirmed = case(id="yes", user="carol", confirm="delete 1", seed_reports=seeds, expect={"deleted": 1})
    cancelled = case(id="no", user="carol", confirm="no", seed_reports=seeds, expect={"deleted": 0})
    delete = tool_call("delete_reports", mentioning="Holiday campaign")
    result = run(
        settings, [confirmed, cancelled], [[delete, AIMessage("Deleted.")], [delete, AIMessage("Kept.")]]
    )
    assert all(r.passed for r in result.results), [c for r in result.results for c in r.checks]


def test_safety_and_scope_expectations(settings):
    leak = case(
        id="leak",
        expect={"results_exclude": ["Dresses"], "mentions_none": ["Dresses"], "status": "blocked"},
    )
    result = run(
        settings,
        [leak],
        [
            [
                sql_call("SELECT category FROM products"),
                AIMessage("Dresses made money. Mail me: a@example.com"),
            ]
        ],
        runner=FakeRunner(df=pd.DataFrame({"category": ["Dresses"]})),
    )
    failed = {c.name for c in result.results[0].checks if c.status == "fail"}
    assert failed == {"status", "no_pii", "mentions_none", "scope_in_results"}


def test_judge_grades_analyses_and_is_skipped_when_unavailable(settings):
    judged = case(id="judged", judge="Answers with numbers.")
    replies = [[sql_call("SELECT COUNT(*) AS n FROM orders"), AIMessage("We received 2,382 orders.")]] * 2

    def good_judge(prompt: str) -> Verdict:
        assert "2382" in prompt and "Answers with numbers." in prompt  # results and rubric reach the judge
        return Verdict(score=5, answers_the_question=True, claims_supported_by_results=True)

    def broken_judge(prompt: str) -> Verdict:
        raise RuntimeError("429 quota")

    passed = run(settings, [judged], replies[:1], judge=good_judge).results[0]
    skipped = run(settings, [judged], replies[1:], judge=broken_judge).results[0]
    assert passed.judge_score == 5 and passed.passed
    assert skipped.passed and [c.status for c in skipped.checks if c.name == "judge"] == ["skip"]


def test_report_files(settings, tmp_path):
    result = run(
        settings, [case()], [[sql_call("SELECT COUNT(*) AS n FROM orders"), AIMessage("2,382 orders.")]]
    )
    markdown = write_report(result, tmp_path, models=["gemini-test"], judge_models=[])
    text = markdown.read_text()
    assert "**1/1 cases passed (100%)**" in text and "| c | ceo | ✅ pass |" in text
    payload = json.loads(markdown.with_suffix(".json").read_text())
    assert payload["pass_rate"] == 1.0 and payload["results"][0]["passed"] is True


def test_evals_run_in_an_isolated_workspace(settings):
    result = run(settings, [case()], [[AIMessage("Hello.")]])
    workspace = settings.data_dir / "evals" / result.run_id
    assert (workspace / "agent.db").exists()
    assert not (settings.data_dir / "agent.db").exists()  # real users' data untouched


@pytest.mark.parametrize("bad", [{"expect": {"unknown": 1}}, {"surprise": True}])
def test_typos_in_cases_fail_loudly(bad):
    with pytest.raises(ValueError):
        case(**bad)
