"""Grounding check and empty-result detection (both added after a live hallucination, see PLAN.md)."""

from __future__ import annotations

import pandas as pd
import pytest
from fakes import FakeRunner, ScriptedModel, SequenceRunner, ask, make_agent, sql_call, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from retail_agent.agent.grounding import figures, ungrounded_figures
from retail_agent.agent.tools import EMPTY_RESULT, no_data

CSV = "month,orders,net_revenue\n2026-07-01,42,2252.76\n2026-08-01,45,2231.37\n2026-09-01,52,2667.32"
RESULT = ToolMessage(
    f"Result: 3 rows.\n{CSV}",
    tool_call_id="c1",
    name="run_sql",
)


def turn(*tool_results: ToolMessage, question: str = "Levi's revenue by month in Q3?") -> list:
    return [HumanMessage(question), *tool_results]


@pytest.mark.parametrize(
    "text",
    [
        "July net revenue was $2,252.76 from 42 orders.",
        "September brought in about $2,667.",  # rounded
        "Q3 total: $7,151.45 across 139 orders.",  # column totals
        "Average monthly revenue was $2,383.82.",  # column average
        "Revenue was $7.2K in Q3.",  # scaled and rounded
        "Up 18.4% from July to September; see 2026-07-01 and the year 2026.",  # %, dates, years skipped
        "We have fewer than 100 rows of data for 12 stores.",  # small numbers are not checked
        "September grew $414.56 over July ($2,667.32 vs $2,252.76).",  # a difference of two results
        "July and August together brought $4,484.13.",  # a sum of two results
        "Q3 revenue was over $7,000 and nearly $7,200.",  # bounds, not claims of exact values
    ],
)
def test_figures_backed_by_results_pass(text):
    assert ungrounded_figures(text, turn(RESULT)) == []


def test_exact_looking_figures_after_a_bound_word_are_still_checked():
    assert ungrounded_figures("Revenue was nearly $7,151.99.", turn(RESULT)) == ["$7,151.99"]


def test_invented_figures_are_flagged():
    answer = "Customers there spent $4,068,574.68 across 33,656 orders (12,961 buyers); July was $2,252.76."
    assert ungrounded_figures(answer, turn(RESULT)) == ["$4,068,574.68", "33,656", "12,961"]


def test_numbers_from_the_question_count_as_grounded():
    assert (
        ungrounded_figures("Here are the top 250 customers.", turn(question="Show the top 250 customers"))
        == []
    )


def test_figure_parsing():
    found = [(w, v) for w, v, _ in figures("$4.1M, 2,382 orders, 57.7%, r-a4e00186, Q3 2026, top 500")]
    assert found == [("$4.1M", pytest.approx(4_100_000)), ("2,382", 2382.0)]


@pytest.mark.parametrize(
    ("df", "empty"),
    [
        (pd.DataFrame({"x": []}), True),
        (pd.DataFrame({"revenue": [None], "orders": [0]}), True),  # SUM/COUNT over no rows
        (pd.DataFrame({"n": pd.Series([0], dtype="int64")}), True),  # numpy ints too
        (pd.DataFrame({"revenue": [10.5], "orders": [0]}), False),
        (pd.DataFrame({"state": ["Texas"], "n": [0]}), False),  # a label means a real row
        (pd.DataFrame({"n": [0, 0]}), False),
    ],
)
def test_no_data_detection(df, empty):
    assert no_data(df) is empty


def test_aggregate_over_nothing_is_reported_as_empty(settings):
    model = ScriptedModel(
        sql_call("SELECT SUM(sale_price) AS revenue FROM order_items"), AIMessage("Nothing matched.")
    )
    _, graph = make_agent(settings, model, FakeRunner(df=pd.DataFrame({"revenue": [None]})))

    state = ask(graph, "Revenue in Atlantis?")

    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_message.artifact["status"] == "empty" and tool_message.content == EMPTY_RESULT


def test_unbacked_figures_get_a_visible_caveat_and_are_saved_with_it(settings):
    model = ScriptedModel(
        sql_call("SELECT COUNT(*) AS n FROM users"),
        tool_call("save_report", "c2", title="California"),
        AIMessage("California has 3677 users and spent $4,068,574.68 in 2025."),
        AIMessage("California has 3677 users and spent $4,068,574.68 in 2025."),  # the revision keeps it
    )
    session, graph = make_agent(settings, model, FakeRunner(df=pd.DataFrame({"n": [3677]})))

    state = ask(graph, "California report")

    assert state["ungrounded"] == ["$4,068,574.68"]
    assert "Automatic check: these figures do not appear" in state["messages"][-1].text
    [report] = session.reports.find("alice")
    assert "$4,068,574.68." in report.body and "Automatic check" in report.body


def test_unbacked_figures_trigger_one_revision_that_can_fix_the_answer(settings):
    model = ScriptedModel(
        sql_call("SELECT COUNT(*) AS n FROM orders"),
        AIMessage("You have 3 orders worth $9,999."),
        AIMessage("You have 3 orders; their value was not part of the query."),
    )
    _, graph = make_agent(settings, model, FakeRunner(df=pd.DataFrame({"n": [3]})))

    state = ask(graph, "How many orders?")

    feedback = model.calls[2][-1]
    assert isinstance(feedback, HumanMessage) and "[Automatic check, not from the user]" in feedback.text
    assert "$9,999" in feedback.text  # the model is told which figures and sees its draft
    assert state["revision"]["unbacked"] == ["$9,999"] and state["ungrounded"] == []
    assert state["messages"][-1].text == "You have 3 orders; their value was not part of the query."
    assert not any("$9,999" in m.text for m in state["messages"])  # the draft is not kept
    assert not any("Automatic check" in m.text for m in state["messages"])  # nor the feedback


def test_the_revision_may_query_the_missing_figure(settings):
    model = ScriptedModel(
        sql_call("SELECT COUNT(*) AS n FROM orders", "c1"),
        AIMessage("You have 3 orders worth $9,999."),
        sql_call("SELECT SUM(sale_price) AS revenue FROM order_items", "c2"),
        AIMessage("You have 3 orders worth $9,999."),
    )
    runner = SequenceRunner(pd.DataFrame({"n": [3]}), pd.DataFrame({"revenue": [9999.0]}))
    _, graph = make_agent(settings, model, runner)

    state = ask(graph, "How many orders and how much revenue?")

    assert state["ungrounded"] == [] and "Automatic check" not in state["messages"][-1].text
    assert len(runner.calls) == 2


def test_only_one_revision_per_turn(settings):
    model = ScriptedModel(
        sql_call("SELECT COUNT(*) AS n FROM orders"),
        AIMessage("Worth $9,999."),
        AIMessage("Worth $8,888."),
    )
    _, graph = make_agent(settings, model, FakeRunner(df=pd.DataFrame({"n": [3]})))

    state = ask(graph, "How many orders?")

    assert len(model.calls) == 3  # draft, one revision, no more
    assert state["ungrounded"] == ["$8,888"] and "Automatic check" in state["messages"][-1].text
