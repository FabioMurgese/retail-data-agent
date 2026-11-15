"""Resilience primitives shared by the agent: circuit breakers and the LLM fallback chain.

Failure policy, in one place:

- BigQuery: the client library already retries transient errors (bounded by a deadline). If calls
  still fail, a circuit breaker opens, so the next queries fail fast with a clear message instead of
  waiting through the same retries again.
- Gemini: each model has its own requests-per-minute window and breaker. A call goes to the first
  model that is healthy and has quota left this minute; on a rate limit, server error, timeout or
  network error the next model is tried. Errors that another model cannot fix (bad API key, refused
  or malformed request, context too long) stop immediately. When nothing works the agent answers
  with a plain explanation: the chat never crashes and nothing is retried in a loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRefusalError,
)
from langchain_core.messages import AIMessage, BaseMessage

log = logging.getLogger(__name__)


class CircuitBreaker:
    """closed -> (failures reach threshold) -> open -> (cooldown elapsed) -> half-open probe."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 2,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def retry_in_s(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self._opened_at + self.cooldown_s - self._clock())

    def record_success(self) -> None:
        self.failures = 0
        self._opened_at = None

    def record_failure(self, *, open_now: bool = False) -> None:
        self.failures += 1
        if open_now or self.failures >= self.failure_threshold or self.state == "half_open":
            if self._opened_at is None or self.state == "half_open":
                log.warning("circuit %s opened after %d failure(s)", self.name, self.failures)
            self._opened_at = self._clock()


# --- LLM fallback chain ------------------------------------------------------------------------------

LLMFailure = Literal["unavailable", "auth", "refused", "invalid_request", "context_too_long"]

USER_MESSAGES: dict[LLMFailure, str] = {
    "unavailable": (
        "I can't reach the AI service right now: every model is busy, rate-limited or down. "
        "Please try again in about {retry} seconds. Your conversation is still here."
    ),
    "auth": (
        "The AI service rejected our credentials, so I can't answer. An administrator needs to check "
        "GEMINI_API_KEY (run `retail-agent doctor`)."
    ),
    "refused": "I can't help with that request.",
    "invalid_request": (
        "The AI service could not process this conversation. Try rephrasing, or start a new "
        "conversation with /new."
    ),
    "context_too_long": "This conversation is too long for me to process. Please start a new one with /new.",
}


class LLMUnavailableError(Exception):
    def __init__(self, kind: LLMFailure, detail: str, *, retry_in_s: float = 60.0) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.retry_in_s = retry_in_s

    @property
    def user_message(self) -> str:
        return USER_MESSAGES[self.kind].format(retry=max(10, round(self.retry_in_s / 10) * 10))


def classify_llm_error(exc: Exception) -> LLMFailure | None:
    """None = another model may succeed (rate limit, 5xx, timeout, network, unknown model...)."""
    if isinstance(exc, ModelAuthenticationError | ModelPermissionDeniedError) or _bad_api_key(exc):
        return "auth"
    if isinstance(exc, ModelRefusalError):
        return "refused"
    if isinstance(exc, ContextOverflowError):
        return "context_too_long"
    if isinstance(exc, ModelInvalidRequestError):
        return "invalid_request"
    return None


def _bad_api_key(exc: Exception) -> bool:
    # Gemini reports a bad key as HTTP 400 INVALID_ARGUMENT, not 401.
    return "api key not valid" in str(exc).lower() or "api_key_invalid" in str(exc).lower()


class Limiter(Protocol):
    def acquire(self, *, blocking: bool = True) -> bool: ...


class InvokableModel(Protocol):
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> BaseMessage: ...


@dataclass
class ModelSlot:
    name: str
    model: InvokableModel
    limiter: Limiter
    breaker: CircuitBreaker = field(init=False)
    cooldown_s: float = 60.0

    def __post_init__(self) -> None:
        self.breaker = CircuitBreaker(f"llm:{self.name}", failure_threshold=2, cooldown_s=self.cooldown_s)


class FallbackChatModel:
    """Tries the configured models in order; see the module docstring for the policy.

    The answer's `response_metadata` gets `latency_ms`, `model_name` and `fallbacks` (the models
    skipped and why), for the CLI and for observability.
    """

    def __init__(self, slots: Sequence[ModelSlot]) -> None:
        if not slots:
            raise ValueError("at least one model is required")
        self.slots = list(slots)

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> BaseMessage:
        skipped: list[dict[str, str]] = []
        waiting: list[ModelSlot] = []  # healthy, but out of client-side quota for this minute
        for slot in self.slots:
            if not slot.breaker.allow():
                skipped.append({"model": slot.name, "reason": "circuit open"})
            elif not slot.limiter.acquire(blocking=False):
                skipped.append({"model": slot.name, "reason": "rate limit (client)"})
                waiting.append(slot)
            elif (response := self._call(slot, messages, config, kwargs, skipped)) is not None:
                return response
        if waiting:  # every healthy model is at its per-minute limit: wait for the first, once
            waiting[0].limiter.acquire(blocking=True)
            if (response := self._call(waiting[0], messages, config, kwargs, skipped)) is not None:
                return response
        retry = min((s.breaker.retry_in_s() for s in self.slots if not s.breaker.allow()), default=60.0)
        reasons = "; ".join(f"{s['model']}: {s['reason']}" for s in skipped)
        raise LLMUnavailableError("unavailable", reasons, retry_in_s=retry or 60.0)

    @staticmethod
    def _call(
        slot: ModelSlot, messages: Any, config: Any, kwargs: dict[str, Any], skipped: list[dict[str, str]]
    ) -> BaseMessage | None:
        started = time.monotonic()
        try:
            response = slot.model.invoke(messages, config, **kwargs)
        except Exception as exc:  # classify: stop now, or let the next model try
            if (kind := classify_llm_error(exc)) is not None:
                raise LLMUnavailableError(kind, _short(exc)) from exc
            # quota exhausted or model unknown: will not heal within seconds, skip it for a cooldown
            slot.breaker.record_failure(open_now=isinstance(exc, ModelRateLimitError | ModelNotFoundError))
            skipped.append({"model": slot.name, "reason": _short(exc)})
            log.warning("LLM %s failed, trying the next model: %s", slot.name, _short(exc))
            return None
        slot.breaker.record_success()
        if isinstance(response, AIMessage):
            response.response_metadata["latency_ms"] = int((time.monotonic() - started) * 1000)
            response.response_metadata.setdefault("model_name", slot.name)
            if skipped:
                response.response_metadata["fallbacks"] = list(skipped)
        return response


def _short(exc: Exception, limit: int = 200) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:limit]
