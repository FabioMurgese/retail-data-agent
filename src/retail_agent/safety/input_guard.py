"""Input guard: cheap, deterministic checks on the user's message before any LLM call.

It stops two kinds of request early, with a specific reply and a logged verdict:

- prompt injection and jailbreak phrasing ("ignore previous instructions", "reveal your prompt"),
  plus SQL statements typed into the chat;
- explicit requests for personal data (emails, phone numbers, addresses, customer names...).

This is not the security boundary: the SQL guard and the output masker are, and they hold even when
this check is bypassed, e.g. by paraphrasing. Its job is UX (an immediate, specific answer), cost
(no tokens spent on requests we will refuse) and observability (block rate per category).

Off-topic requests are left to the agent's instructions: they are harmless, and judging them needs
language understanding. The check is deterministic on purpose: the free Gemini tier allows few
calls per minute, and a classifier call on every turn would halve that budget. Production would add
Model Armor (prompt-injection and jailbreak detection) in front of the agent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Category = Literal["ok", "empty", "too_long", "injection", "pii_request"]

_PEOPLE = r"(?:customers?|users?|clients?|buyers?|shoppers?|spenders?|people)"
_DETERMINER = r"(?:\w+\s+){0,3}"  # "of our top 10 customers"

# (category, label, pattern). Patterns run on normalised text: NFKC, case-folded, single spaces.
_RULES: tuple[tuple[Category, str, re.Pattern[str]], ...] = tuple(
    (category, label, re.compile(pattern))
    for category, label, pattern in (
        (
            "injection",
            "override_instructions",
            # "ignore all previous instructions" yes; "ignore cancelled orders, usual rules" no
            r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:\w+\s+){0,3}?(?:previous|prior|above|earlier|all|"
            r"your|system|safety|any)\s+(?:\w+\s+){0,2}?(?:instructions?|prompts?|rules|guardrails?|"
            r"polic(?:y|ies)|guidelines?|restrictions?|constraints?)\b",
        ),
        (
            "injection",
            "reveal_prompt",
            r"\b(?:reveal|show|print|repeat|display|leak|output|tell me)\b.{0,30}\b(?:system prompt|"
            r"your (?:initial |hidden |original |system )?(?:instructions|prompt|rules))\b",
        ),
        (
            "injection",
            "role_override",
            r"\b(?:you are now|act as|pretend to be|pretend you are|roleplay as|from now on you are)\b.{0,30}"
            r"\b(?:dan|unrestricted|unfiltered|jailbroken|uncensored|admin|administrator|root|dba)\b",
        ),
        ("injection", "jailbreak", r"\bjailbreak\w*\b|\b(?:developer|god|debug|admin) mode\b"),
        (
            "injection",
            "sql_statement",
            r";\s*(?:drop|delete|update|insert|truncate|alter|create|grant|merge)\b|\b(?:drop|truncate|alter)\s+table\b",
        ),
        (
            "pii_request",
            "email",
            # "Email" is also an acquisition channel: "customer email campaigns" must pass
            rf"\be-?mail address(?:es)?\b|\b{_PEOPLE}'?\s+e-?mails\b"
            rf"|\be-?mails?\s+(?:of|for)\s+{_DETERMINER}{_PEOPLE}\b"
            r"|\b(?:their|his|her)\s+e-?mails?\b(?!\s+(?:campaigns?|marketing|channel|traffic|source))",
        ),
        (
            "pii_request",
            "phone",
            r"\b(?:phone|mobile|telephone|cell)\s+numbers?\b|\bcontact (?:details|info\w*)\b",
        ),
        (
            "pii_request",
            "address",
            r"\b(?:street|home|postal|mailing|shipping|billing|physical)\s+address(?:es)?\b"
            r"|\b(?:postal|zip|post)\s*codes?\b|\baddress(?:es)?\s+of\b",
        ),
        (
            "pii_request",
            "name",
            rf"\b(?:first|last|full|real|sur)\s*names?\b|\b{_PEOPLE}'?\s+names?\b"
            rf"|\bnames?\s+of\s+{_DETERMINER}{_PEOPLE}\b",
        ),
        (
            "pii_request",
            "location",
            r"\b(?:latitude|longitude|gps|geo-?coordinates|coordinates|exact location|precise location)\b",
        ),
        (
            "pii_request",
            "personal_data",
            r"\b(?:pii|personally identifiable|personal (?:data|information|details)|social security|ssn)\b",
        ),
    )
)

_REPLIES: dict[Category, str] = {
    "empty": "Please type a question.",
    "too_long": "That message is too long ({length} characters, limit {limit}). Please shorten it.",
    "injection": (
        "I can only help with analysing the company's sales data, and I can't change how I work or share "
        'my instructions. Try a question such as "What were our top 5 categories by revenue last month?"'
    ),
    "pii_request": (
        "I can't share personal data about customers (names, emails, phone numbers, addresses or exact "
        "locations). I can analyse customers by id, age, gender, city, state, country or acquisition "
        'channel, for example "top 10 customers by spend, with their country and age".'
    ),
}

_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍‎‏⁠﻿"))


@dataclass(frozen=True)
class InputVerdict:
    category: Category
    rule: str | None = None  # which rule matched, for logs and metrics
    reply: str | None = None  # what to tell the user when blocked

    @property
    def allowed(self) -> bool:
        return self.category == "ok"


def normalise(text: str) -> str:
    """Undo common evasion tricks: full-width letters, ligatures, zero-width characters, case."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    return " ".join(text.casefold().split())


def check_input(message: str, *, max_chars: int = 4_000) -> InputVerdict:
    if not message.strip():
        return InputVerdict("empty", reply=_REPLIES["empty"])
    if len(message) > max_chars:
        return InputVerdict(
            "too_long", rule="length", reply=_REPLIES["too_long"].format(length=len(message), limit=max_chars)
        )
    text = normalise(message)
    for category, label, pattern in _RULES:
        if pattern.search(text):
            return InputVerdict(category, rule=label, reply=_REPLIES[category])
    return InputVerdict("ok")
