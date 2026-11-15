"""Output masking: the last line of defence before text reaches the user.

The SQL guard makes PII columns unreachable, so in normal operation nothing here fires. The masker
catches what slips through other paths: personal data typed by the user and echoed back, contact
details the model makes up, or a future tool that returns free text. It works on

- DataFrames: a column whose name is a PII column in the policy is blanked entirely;
- any text (cell values, tool output, the final answer): patterns for emails, phone numbers, street
  addresses, coordinate pairs and payment card numbers.

Every replacement is counted. A non-zero count means a leak was stopped, which observability
reports and someone should investigate.

Limits: person names and postal codes cannot be detected reliably with patterns, so they are
protected structurally by the guard. Production adds Cloud DLP (Sensitive Data Protection) here.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import pandas as pd

from retail_agent.policy import DataPolicy

_STREET_SUFFIXES = (
    "Alley|Avenue|Ave|Bend|Boulevard|Blvd|Branch|Bridge|Brook|Bypass|Camp|Canyon|Causeway|Center|Circle|"
    "Cliff|Common|Corner|Course|Court|Ct|Cove|Creek|Crescent|Crest|Crossing|Dale|Drive|Dr|Estate|"
    "Expressway|Falls|Ferry|Field|Flat|Ford|Forest|Forge|Fork|Fort|Freeway|Garden|Gateway|Glen|Green|"
    "Grove|Harbor|Haven|Heights|Highway|Hill|Hollow|Island|Junction|Key|Knoll|Lake|Landing|Lane|Ln|"
    "Light|Lock|Lodge|Loop|Manor|Meadow|Mews|Mill|Mission|Motorway|Mount|Mountain|Orchard|Oval|Park|"
    "Parkway|Pass|Path|Pike|Pine|Place|Pl|Plain|Plaza|Point|Port|Prairie|Ranch|Rapid|Ridge|River|Road|"
    "Rd|Route|Row|Run|Shoal|Shore|Spring|Square|Sq|Station|Stravenue|Stream|Street|St|Summit|Terrace|"
    "Trace|Track|Trail|Tunnel|Turnpike|Valley|Via|Viaduct|View|Village|Ville|Vista|Walk|Way|Well"
)

# (kind, pattern). Order matters: emails before phones, so digits in an email are not seen as a phone.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "address",
        re.compile(
            rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z']+\s+){{1,3}}(?:{_STREET_SUFFIXES})s?\b\.?"
            r"(?:,?\s+(?:Apt\.?|Suite|Unit)\s*\d+)?"
            # must end the phrase: "403 Jefferson Walks, Baltimore" yes; "12 Steens Mountain jackets" no
            r"(?=\s*(?:$|[,.;:!?)\]|\n]))"
        ),
    ),
    ("coordinates", re.compile(r"(?<![\d.])-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}(?![\d.])")),
    (
        "phone",
        re.compile(
            r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?![\w.])"  # +39 06 1234 5678
            r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}\b"  # (415) 555-0100
            r"|\b\d{3}[.-]\d{3}[.-]\d{4}\b"  # 415-555-0100
        ),
    ),
    ("card_number", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
)


def _luhn_ok(candidate: str) -> bool:
    digits = [int(d) for d in candidate if d.isdigit()]
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        checksum += digit
    return checksum % 10 == 0


class PiiMasker:
    def __init__(self, pii_columns: Iterable[str]) -> None:
        self.pii_columns = frozenset(column.lower() for column in pii_columns)

    @classmethod
    def from_policy(cls, policy: DataPolicy) -> PiiMasker:
        return cls(column for table in policy.tables.values() for column in table.pii)

    def mask_text(self, text: str) -> tuple[str, Counter[str]]:
        counts: Counter[str] = Counter()
        for kind, pattern in _PATTERNS:

            def replace(match: re.Match[str], kind: str = kind) -> str:
                if kind == "card_number" and not _luhn_ok(match.group()):
                    return match.group()  # long numbers that are not cards (ids, totals) stay
                counts[kind] += 1
                return f"[{kind} redacted]"

            text = pattern.sub(replace, text)
        return text, counts

    def mask_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Counter[str]]:
        """Return a masked copy; the input frame is not modified."""
        counts: Counter[str] = Counter()
        masked = df.copy()
        for column in masked.columns:
            values = masked[column]
            if str(column).lower() in self.pii_columns:
                counts["pii_column"] += int(values.notna().sum())
                masked[column] = values.where(values.isna(), "[redacted]").astype(object)
            elif pd.api.types.is_string_dtype(values) or pd.api.types.is_object_dtype(values):
                masked[column] = values.map(lambda v: self._mask_cell(v, counts))
        return masked, counts

    def _mask_cell(self, value: object, counts: Counter[str]) -> object:
        if not isinstance(value, str):
            return value
        masked, found = self.mask_text(value)
        counts.update(found)
        return masked
