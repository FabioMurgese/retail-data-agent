"""Saved reports, confirmed deletes (high-stakes oversight) and user preferences."""

from __future__ import annotations

import pytest
from fakes import FakeRunner, ScriptedModel, ask, make_agent, resume, tool_call
from langchain_core.messages import AIMessage, ToolMessage

from retail_agent.chat import confirmation_decision
from retail_agent.storage import Database, PreferenceStore, ReportStore

# --- storage ----------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> ReportStore:
    return ReportStore(Database(tmp_path / "agent.db"), tmp_path / "reports")


def test_find_is_scoped_to_the_owner_and_supports_full_text(store):
    ck = store.save(
        owner_id="alice", thread_id="t1", title="Calvin Klein review", body="Sales of Calvin Klein jeans"
    )
    store.save(owner_id="alice", thread_id="t2", title="Q3 review", body="Levi's grew 10%")
    store.save(owner_id="bob", thread_id="t9", title="Bob on Calvin Klein", body="Calvin Klein in denim")

    assert [r.id for r in store.find("alice", mentioning="calvin klein")] == [
        ck.id
    ]  # case-insensitive phrase
    assert [r.title for r in store.find("alice", mentioning="Levi's")] == ["Q3 review"]
    assert [r.title for r in store.find("alice", thread_id="t2")] == ["Q3 review"]
    assert len(store.find("alice")) == 2 and len(store.find("bob")) == 1
    assert store.find("alice", mentioning='" OR body:*') == []  # FTS syntax cannot be injected


def test_soft_delete_only_touches_the_owners_live_reports(store, tmp_path):
    mine = store.save(owner_id="alice", thread_id="t1", title="A", body="x")
    theirs = store.save(owner_id="bob", thread_id="t1", title="B", body="x")
    assert (tmp_path / "reports" / f"{mine.id}.md").exists()

    assert store.soft_delete("alice", [mine.id, theirs.id]) == [mine.id]
    assert store.soft_delete("alice", [mine.id]) == []  # already deleted
    assert store.find("alice") == [] and len(store.find("bob")) == 1
    assert not (tmp_path / "reports" / f"{mine.id}.md").exists()


def test_preferences_are_per_user_and_deduplicated(tmp_path):
    prefs = PreferenceStore(Database(tmp_path / "agent.db"))
    prefs.add("alice", "Prefers tables")
    prefs.add("alice", "Prefers tables")
    prefs.add("alice", "Wants short answers")
    prefs.add("bob", "Prefers bullet points")
    assert prefs.list("alice") == ["Prefers tables", "Wants short answers"]
    assert prefs.list("bob") == ["Prefers bullet points"]
    assert prefs.clear("alice") == 2 and prefs.list("alice") == []


# --- saving a report through the agent ----------------------------------------------------------------


def test_the_final_answer_is_saved_as_the_report_after_masking(settings):
    report_text = "# Q3 review\n## Executive summary\n- Revenue up. Contact: jane@example.com"
    model = ScriptedModel(tool_call("save_report", title="Q3 review"), AIMessage(report_text))
    session, graph = make_agent(settings, model, FakeRunner())

    state = ask(graph, "Create a report for Q3", thread="th-1", trace="tr-1")

    [report] = session.reports.find("alice")
    assert report.title == "Q3 review" and report.thread_id == "th-1"
    assert report.body.endswith("Contact: [email redacted]")  # saved exactly what the user saw, masked
    assert state["saved_report"]["id"] == report.id
    assert state["messages"][-1].text.endswith(f"*Saved to your reports as `{report.id}`: Q3 review*")
    assert [e["action"] for e in session.reports.audit_entries("alice")] == ["report_saved"]


def test_nothing_is_saved_if_the_turn_does_not_reach_a_final_answer(settings):
    model = ScriptedModel(tool_call("save_report", title="Never finished"))
    session, graph = make_agent(settings, model, FakeRunner())
    with pytest.raises(IndexError):  # the scripted model runs out: stands in for a failed turn
        ask(graph, "Create a report")
    assert session.reports.find("alice") == []


# --- deleting with confirmation ------------------------------------------------------------------------


def seed(session) -> dict[str, str]:
    save = session.reports.save
    return {
        "ck1": save(owner_id="alice", thread_id="old", title="Calvin Klein Q1", body="Calvin Klein sales").id,
        "ck2": save(owner_id="alice", thread_id="old", title="Brands", body="Mostly Calvin Klein").id,
        "other": save(owner_id="alice", thread_id="old", title="Levi's", body="Levi's only").id,
        "bob": save(owner_id="bob", thread_id="old", title="Calvin Klein (Bob)", body="Calvin Klein").id,
    }


def pending_interrupt(graph, thread: str = "t1"):
    [pending] = graph.get_state({"configurable": {"thread_id": thread}}).interrupts
    return pending


def test_delete_pauses_for_confirmation_and_deletes_exactly_what_was_shown(settings):
    model = ScriptedModel(
        tool_call("delete_reports", mentioning="Calvin Klein"), AIMessage("Deleted 2 reports.")
    )
    session, graph = make_agent(settings, model, FakeRunner())
    ids = seed(session)

    ask(graph, "Delete all reports mentioning Calvin Klein")
    pending = pending_interrupt(graph)
    shown = {r["id"] for r in pending.value["reports"]}
    assert shown == {ids["ck1"], ids["ck2"]}  # bob's report is not even a candidate
    assert session.reports.find("alice", ids=[ids["ck1"]])  # nothing deleted while waiting

    state = resume(graph, {pending.id: {"token": pending.value["token"], "confirmed": True}})

    assert {r.id for r in session.reports.find("alice")} == {ids["other"]}
    assert len(session.reports.find("bob")) == 1
    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_message.artifact == {"status": "deleted", "ids": sorted(shown)}
    actions = [
        e["action"] for e in session.reports.audit_entries("alice") if e["action"].startswith("delete")
    ]
    assert actions == ["delete_requested", "delete_confirmed"]  # requested logged once despite the re-run


def test_cancelled_delete_changes_nothing(settings):
    model = ScriptedModel(tool_call("delete_reports", mentioning="Calvin Klein"), AIMessage("OK, kept them."))
    session, graph = make_agent(settings, model, FakeRunner())
    seed(session)

    ask(graph, "Delete the Calvin Klein reports")
    pending = pending_interrupt(graph)
    state = resume(graph, {pending.id: {"token": pending.value["token"], "confirmed": False}})

    assert len(session.reports.find("alice")) == 3
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).artifact["status"] == "cancelled"
    assert session.reports.audit_entries("alice")[-1]["action"] == "delete_cancelled"


def test_confirmation_is_bound_to_the_exact_set_shown(settings):
    model = ScriptedModel(
        tool_call("delete_reports", mentioning="Calvin Klein"), AIMessage("Nothing deleted.")
    )
    session, graph = make_agent(settings, model, FakeRunner())
    seed(session)

    ask(graph, "Delete the Calvin Klein reports")
    pending = pending_interrupt(graph)
    # another report matching the criteria appears while the user is reading the confirmation
    session.reports.save(owner_id="alice", thread_id="other", title="New CK", body="Calvin Klein again")
    state = resume(graph, {pending.id: {"token": pending.value["token"], "confirmed": True}})

    assert len(session.reports.find("alice")) == 4  # nothing deleted
    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).artifact["status"] == "changed"
    assert session.reports.audit_entries("alice")[-1]["action"] == "delete_aborted"


def test_delete_this_conversation_uses_the_current_thread(settings):
    model = ScriptedModel(tool_call("delete_reports", this_conversation=True), AIMessage("Done."))
    session, graph = make_agent(settings, model, FakeRunner())
    seed(session)
    current = session.reports.save(owner_id="alice", thread_id="t1", title="Made now", body="x").id

    ask(graph, "Delete all the reports we made in this conversation", thread="t1")
    pending = pending_interrupt(graph)
    assert [r["id"] for r in pending.value["reports"]] == [current]


def test_no_match_and_other_users_ids_never_pause_or_delete(settings):
    model = ScriptedModel(tool_call("delete_reports", report_ids=["r-nothere"]), AIMessage("None found."))
    session, graph = make_agent(settings, model, FakeRunner())
    ids = seed(session)
    model.replies.insert(0, tool_call("delete_reports", report_ids=[ids["bob"]]))
    model.replies.insert(1, AIMessage("That report is not yours."))

    ask(graph, "Delete Bob's report")  # alice asks for bob's id: invisible to her
    ask(graph, "Delete r-nothere")

    assert graph.get_state({"configurable": {"thread_id": "t1"}}).interrupts == ()
    assert len(session.reports.find("bob")) == 1


# --- preferences through the agent ---------------------------------------------------------------------


def test_remembered_preferences_reach_the_prompt_next_turn(settings):
    model = ScriptedModel(
        tool_call("remember_preference", preference="Prefers tables to bullet points"),
        AIMessage("Noted."),
        AIMessage("Here is the table."),
    )
    session, graph = make_agent(settings, model, FakeRunner())

    ask(graph, "Always use tables, please")
    ask(graph, "Revenue by month?")

    assert session.preferences.list("alice") == ["Prefers tables to bullet points"]
    assert "saved preferences" not in model.calls[0][0].content
    assert "- Prefers tables to bullet points" in model.calls[-1][0].content


# --- CLI confirmation wording -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("delete 3", "confirm"),
        ("  DELETE   3 ", "confirm"),
        ("delete 2", "other"),  # wrong count: not a confirmation
        ("yes", "vague"),
        ("", "cancel"),
        ("no", "cancel"),
        ("no wait, only the Q1 one", "other"),  # handled as a new request after cancelling
    ],
)
def test_only_the_exact_phrase_confirms(answer, expected):
    assert confirmation_decision(answer, 3) == expected
