"""Resilience: circuit breakers, LLM fallback policy, bounded self-correction, graceful degradation."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from fakes import FakeRunner, ScriptedModel, SequenceRunner, ask, make_agent, sql_call, tool_call
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import AIMessage, ToolMessage

from retail_agent.agent.graph import TOKEN_BUDGET_REPLY, TOOL_FAILURE
from retail_agent.data.bq_client import BigQueryUnavailableError, InvalidQueryError
from retail_agent.resilience import CircuitBreaker, FallbackChatModel, LLMUnavailableError, ModelSlot

# --- circuit breaker -------------------------------------------------------------------------------


def test_breaker_opens_at_threshold_and_probes_after_cooldown():
    now = [0.0]
    breaker = CircuitBreaker("bq", failure_threshold=2, cooldown_s=60, clock=lambda: now[0])
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow() and breaker.retry_in_s() == 60

    now[0] = 60
    assert breaker.state == "half_open" and breaker.allow()
    breaker.record_failure()  # the probe failed: open again for a full cooldown
    assert breaker.state == "open" and breaker.retry_in_s() == 60

    now[0] = 130
    breaker.record_success()
    assert breaker.state == "closed" and breaker.failures == 0


def test_breaker_can_open_immediately():
    breaker = CircuitBreaker("llm", failure_threshold=5)
    breaker.record_failure(open_now=True)
    assert breaker.state == "open"


# --- LLM fallback chain ----------------------------------------------------------------------------


class Model:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages, config=None, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Limiter:
    def __init__(self, free: bool = True):
        self.free = free
        self.blocking_waits = 0

    def acquire(self, *, blocking: bool = True) -> bool:
        if blocking:
            self.blocking_waits += 1
            return True
        return self.free


def chain(*models: Model, limiters: list[Limiter] | None = None) -> FallbackChatModel:
    limiters = limiters or [Limiter() for _ in models]
    return FallbackChatModel([ModelSlot(f"m{i}", m, limiters[i]) for i, m in enumerate(models)])


def test_rate_limited_model_is_skipped_and_its_breaker_opens():
    primary, backup = Model(ModelRateLimitError("429 quota")), Model(AIMessage("from backup"))
    llm = chain(primary, backup)

    first = llm.invoke([])
    second = llm.invoke([])

    assert first.text == second.text == "from backup"
    assert first.response_metadata["fallbacks"][0]["model"] == "m0"
    assert primary.calls == 1  # second call skipped m0 without trying it: circuit open after a 429
    assert second.response_metadata["fallbacks"] == [{"model": "m0", "reason": "circuit open"}]


def test_unknown_model_is_skipped_from_the_first_failure():
    primary = Model(ModelNotFoundError("404 model not found"))
    llm = chain(primary, Model(AIMessage("ok")))
    llm.invoke([])
    llm.invoke([])
    assert primary.calls == 1


@pytest.mark.parametrize("error", [ModelAPIError("503"), ModelTimeoutError("slow"), ConnectionError("reset")])
def test_server_network_and_timeout_errors_fall_back(error):
    assert chain(Model(error), Model(AIMessage("ok"))).invoke([]).text == "ok"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ModelAuthenticationError("401"), "auth"),
        (ModelInvalidRequestError("400 API key not valid. Please pass a valid API key."), "auth"),
        (ModelInvalidRequestError("400 bad request"), "invalid_request"),
    ],
)
def test_errors_another_model_cannot_fix_stop_immediately(error, kind):
    backup = Model(AIMessage("never"))
    with pytest.raises(LLMUnavailableError) as info:
        chain(Model(error), backup).invoke([])
    assert info.value.kind == kind and backup.calls == 0


def test_all_models_failing_gives_a_user_message():
    with pytest.raises(LLMUnavailableError) as info:
        chain(Model(ModelRateLimitError("429")), Model(ModelAPIError("500"))).invoke([])
    assert info.value.kind == "unavailable"
    assert "try again in about 60 seconds" in info.value.user_message


def test_client_side_quota_uses_the_next_model_instead_of_waiting():
    limiters = [Limiter(free=False), Limiter(free=True)]
    primary, backup = Model(AIMessage("primary")), Model(AIMessage("backup"))
    assert chain(primary, backup, limiters=limiters).invoke([]).text == "backup"
    assert primary.calls == 0 and limiters[0].blocking_waits == 0


def test_when_every_model_is_at_its_quota_the_first_one_is_awaited():
    limiters = [Limiter(free=False), Limiter(free=False)]
    primary = Model(AIMessage("primary after waiting"))
    assert chain(primary, Model(AIMessage("x")), limiters=limiters).invoke([]).text == "primary after waiting"
    assert limiters[0].blocking_waits == 1 and limiters[1].blocking_waits == 0


# --- graceful degradation in the graph ------------------------------------------------------------


def test_llm_outage_becomes_an_answer_and_saves_nothing(settings):
    outage = LLMUnavailableError("unavailable", "all down", retry_in_s=42)
    model = ScriptedModel(tool_call("save_report", title="Q3"), outage)
    session, graph = make_agent(settings, model, FakeRunner())

    state = ask(graph, "Create a Q3 report")

    assert state["degraded"] == "llm_unavailable"
    assert (
        "can't reach the AI service" in state["messages"][-1].text
        and "40 seconds" in state["messages"][-1].text
    )
    assert session.reports.find("alice") == []  # the apology was not saved as a report


def test_self_correction_stops_after_three_failed_queries(settings):
    calls = [sql_call(f"SELECT bad{i} FROM orders", f"c{i}") for i in range(4)]
    model = ScriptedModel(*calls, AIMessage("I could not answer this one."))
    runner = SequenceRunner(*[InvalidQueryError(f"Unrecognized name: bad{i}") for i in range(3)])
    _, graph = make_agent(settings, model, runner)

    state = ask(graph, "orders?")

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [m.artifact["status"] for m in tool_messages] == ["error", "error", "error", "budget_exhausted"]
    assert len(runner.calls) == 3  # the 4th query never reached BigQuery
    assert "Do not call run_sql again" in tool_messages[-1].content


def test_empty_results_count_as_failed_attempts(settings):
    model = ScriptedModel(
        *[sql_call("SELECT 1 FROM orders", f"c{i}") for i in range(4)], AIMessage("No data.")
    )
    runner = FakeRunner(df=pd.DataFrame({"x": []}))
    _, graph = make_agent(settings, model, runner)

    state = ask(graph, "orders in Atlantis?")

    statuses = [m.artifact["status"] for m in state["messages"] if isinstance(m, ToolMessage)]
    assert statuses == ["empty", "empty", "empty", "budget_exhausted"]


def test_query_budget_per_question_and_reset_on_the_next_question(settings):
    model = ScriptedModel(
        *[sql_call("SELECT 1 FROM orders", f"a{i}") for i in range(3)], AIMessage("done"),
        sql_call("SELECT 1 FROM orders", "b0"), AIMessage("done again"),
    )  # fmt: skip
    runner = FakeRunner()
    _, graph = make_agent(settings.model_copy(update={"max_sql_per_turn": 2}), model, runner)

    first = ask(graph, "q1")
    second = ask(graph, "q2")

    assert [m.artifact["status"] for m in first["messages"] if isinstance(m, ToolMessage)] == [
        "ok", "ok", "budget_exhausted",
    ]  # fmt: skip
    assert (
        next(m for m in reversed(second["messages"]) if isinstance(m, ToolMessage)).artifact["status"] == "ok"
    )
    assert len(runner.calls) == 3


def test_bigquery_breaker_fails_fast_after_repeated_outages(settings):
    model = ScriptedModel(
        *[sql_call("SELECT 1 FROM orders", f"c{i}") for i in range(3)], AIMessage("BQ down.")
    )
    outage = BigQueryUnavailableError("BigQuery is temporarily unavailable", transient=True)
    runner = SequenceRunner(outage, outage)
    _, graph = make_agent(settings, model, runner)

    state = ask(graph, "orders?")

    artifacts = [m.artifact for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [a["status"] for a in artifacts] == ["error", "error", "unavailable"]
    assert artifacts[-1]["code"] == "circuit_open" and len(runner.calls) == 2


def test_setup_errors_open_the_breaker_at_once(settings):
    model = ScriptedModel(
        sql_call("SELECT 1 FROM orders", "c1"), sql_call("SELECT 2 FROM orders", "c2"), AIMessage("x")
    )
    runner = SequenceRunner(BigQueryUnavailableError("access denied", transient=False))
    _, graph = make_agent(settings, model, runner)

    state = ask(graph, "orders?")

    assert [m.artifact["status"] for m in state["messages"] if isinstance(m, ToolMessage)] == [
        "error",
        "unavailable",
    ]


def test_unexpected_tool_errors_do_not_break_the_turn(settings):
    model = ScriptedModel(sql_call("SELECT 1 FROM orders"), AIMessage("Sorry, that step failed."))
    _, graph = make_agent(settings, model, FakeRunner(error=RuntimeError("disk on fire")))

    state = ask(graph, "orders?")

    assert next(m for m in state["messages"] if isinstance(m, ToolMessage)).content == TOOL_FAILURE
    assert state["messages"][-1].text == "Sorry, that step failed."


def test_report_storage_failure_is_reported_in_the_answer(settings, monkeypatch):
    model = ScriptedModel(tool_call("save_report", title="Q3"), AIMessage("# Q3\nAll good."))
    session, graph = make_agent(settings, model, FakeRunner())

    def broken(**kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(session.reports, "save", broken)
    state = ask(graph, "Create a Q3 report")

    assert "could not save this report" in state["messages"][-1].text
    assert state["messages"][-1].text.startswith("# Q3")


def test_token_budget_stops_an_expensive_turn(settings):
    expensive = sql_call("SELECT 1 FROM orders")
    expensive.usage_metadata = {"input_tokens": 240_000, "output_tokens": 10_000, "total_tokens": 250_000}
    model = ScriptedModel(expensive, AIMessage("never sent"))
    _, graph = make_agent(settings, model, FakeRunner())

    state = ask(graph, "Analyse everything")

    assert state["messages"][-1].text == TOKEN_BUDGET_REPLY and state["degraded"] == "token_budget"
    assert len(model.calls) == 1


def test_a_turn_interrupted_mid_tool_does_not_poison_the_next_one(settings):
    model = ScriptedModel(sql_call("SELECT 1 FROM orders"), AIMessage("Second question answered."))
    _, graph = make_agent(settings, model, FakeRunner(error=KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):  # Ctrl-C while BigQuery runs
        ask(graph, "first question")
    state = ask(graph, "second question")

    sent = model.calls[-1][1:]
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in sent)  # dangling call not resent
    assert state["messages"][-1].text == "Second question answered."
