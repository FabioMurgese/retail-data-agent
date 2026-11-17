"""Gemini chat models behind a fallback chain (policy: `retail_agent.resilience`).

The free tier limits requests per minute per model (15 for Flash-Lite, 5 for Flash), and quotas are
separate per model. Each model gets its own sliding-window limiter, checked by the chain without
blocking: if the primary model has used its minute, the next one answers instead of us waiting.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import TypeVar

from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from retail_agent.config import Settings
from retail_agent.resilience import CircuitBreaker, FallbackChatModel, ModelSlot

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class SlidingWindowRateLimiter(BaseRateLimiter):
    """At most `max_requests` per `window_s`; requests under the limit go out immediately.

    (langchain's InMemoryRateLimiter is a token bucket that starts empty and spaces every request,
    which adds seconds to each agent step even when we are far below the quota.)
    """

    def __init__(self, max_requests: int, window_s: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._sent: deque[float] = deque()
        self._lock = threading.Lock()

    def _wait_time(self) -> float:
        now = time.monotonic()
        while self._sent and now - self._sent[0] >= self.window_s:
            self._sent.popleft()
        if len(self._sent) < self.max_requests:
            self._sent.append(now)
            return 0.0
        return self._sent[0] + self.window_s - now

    def acquire(self, *, blocking: bool = True) -> bool:
        while True:
            with self._lock:
                wait = self._wait_time()
            if wait == 0.0:
                return True
            if not blocking:
                return False
            time.sleep(wait)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        while True:
            with self._lock:
                wait = self._wait_time()
            if wait == 0.0:
                return True
            if not blocking:
                return False
            await asyncio.sleep(wait)


# Quotas and outages are per model and per GCP project, not per chat session: share limiters and
# breakers across every session (and eval case) in the process.
_LIMITERS: dict[str, SlidingWindowRateLimiter] = {}
_BREAKERS: dict[str, CircuitBreaker] = {}


def limiter_for(model: str, rpm: int) -> SlidingWindowRateLimiter:
    return _LIMITERS.setdefault(model, SlidingWindowRateLimiter(rpm))


def breaker_for(model: str, cooldown_s: float) -> CircuitBreaker:
    return _BREAKERS.setdefault(
        model, CircuitBreaker(f"llm:{model}", failure_threshold=2, cooldown_s=cooldown_s)
    )


def gemini(settings: Settings, model: str, **kwargs: object) -> ChatGoogleGenerativeAI:
    if settings.gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY is not set (.env). Run `retail-agent doctor`.")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_s,
        **kwargs,
    )


def build_chat_model(settings: Settings, tools: Sequence[BaseTool]) -> FallbackChatModel:
    def slot(model: str) -> ModelSlot:
        return ModelSlot(
            name=model,
            model=gemini(settings, model).bind_tools(list(tools)),
            limiter=limiter_for(model, settings.rpm_for(model)),
            breaker=breaker_for(model, settings.breaker_cooldown_s),
        )

    return FallbackChatModel([slot(model) for model in settings.model_chain])


def structured_model(
    settings: Settings, models: Sequence[str], schema: type[SchemaT]
) -> Callable[[str], SchemaT]:
    """Models that answer with an instance of `schema`, behind the same fallback chain, limiters and
    breakers as the agent (used by the eval judge and by golden-trio curation)."""
    chain = FallbackChatModel(
        [
            ModelSlot(
                name=model,
                model=gemini(settings, model).with_structured_output(schema),
                limiter=limiter_for(model, settings.rpm_for(model)),
                breaker=breaker_for(model, settings.breaker_cooldown_s),
            )
            for model in models
        ]
    )
    return lambda prompt: chain.invoke(prompt)
