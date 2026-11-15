"""Grounding check: are the figures in an answer backed by this turn's query results?

Observed failure that motivated it: after a filter matched nothing, the model "helpfully" gave
revenue for the corrected value without querying it, and the figures changed between runs.

Deterministic and free (no LLM call). Every significant figure in the final answer (>= 100, not a
year, a date, a percentage or a round number like "top 100") must match a number seen this turn: a
value in a tool result, a column total or average, or a number in the user's question, within the
rounding the answer uses
("$7,700" matches 7700.26; "$4.1M" matches 4_068_574). Unmatched figures are listed at the end of
the answer, so an executive never reads an unbacked number as fact, and recorded for observability.

Derived figures: sums and differences of two seen numbers are accepted ("up $18,748.75" from Q3 and
Q2 totals), percentages and ratios are not checked, and a round figure after a bound word ("over
$100,000") is a threshold, not a claim of an exact value. A production version would add an
LLM-as-judge pass or a revise-on-failure loop (docs/TECHNICAL.md).
"""

from __future__ import annotations

import bisect
import csv
import io
import math
import re
from collections.abc import Iterable, Sequence

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

_SCALES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}
_FIGURE = re.compile(
    r"(?<![\w./-])(?P<currency>[$€£])?\s?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![\w/-]*\d)(?:\s?(?P<scale>k|thousand|mn|m|million|bn|b|billion)\b)?(?P<percent>\s?%)?",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BOUND = re.compile(
    r"\b(?:over|above|more than|nearly|almost|about|around|roughly|approximately|under|below|less than|"
    r"exceeding|up to|at least|at most)\s*$",
    re.IGNORECASE,
)


def figures(text: str) -> list[tuple[str, float, float]]:
    """(as written, value, rounding tolerance) for each figure worth checking."""
    found = []
    for match in _FIGURE.finditer(text):
        if match["percent"]:
            continue
        digits = match["number"].replace(",", "")
        scale = _SCALES.get((match["scale"] or "").lower(), 1.0)
        value = float(digits) * scale
        is_year = not match["currency"] and scale == 1 and "." not in digits and "," not in match["number"]
        round_written = "." not in digits and digits.endswith("00")
        is_round = round_written and not match["currency"]  # "top 100", "1,000 rows"
        is_bound = round_written and _BOUND.search(
            text[max(0, match.start() - 20) : match.start()]
        )  # "over $100,000"
        if abs(value) < 100 or is_round or is_bound or (is_year and 1900 <= value <= 2100):
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        tolerance = 0.5 * 10**-decimals * scale + 1e-9
        found.append((match.group().strip(), value, tolerance))
    return found


def source_numbers(messages: Sequence[AnyMessage]) -> list[float]:
    """Numbers the model saw this turn: tool results (+ column totals and averages) and the question."""
    numbers: list[float] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            numbers += [float(n) for n in _NUMBER.findall(message.text.replace(",", ""))]
            break
        if isinstance(message, ToolMessage) and isinstance(message.content, str):
            numbers += _numbers_in_tool_result(message.content)
    return numbers


def ungrounded_figures(answer: str, messages: Sequence[AnyMessage]) -> list[str]:
    sources = sorted({abs(n) for n in source_numbers(messages)})
    missing: list[str] = []
    for written, value, tolerance in figures(answer):
        if not _backed(abs(value), tolerance, sources) and written not in missing:
            missing.append(written)
    return missing


def _backed(value: float, tolerance: float, sources: list[float]) -> bool:
    """A seen number, or the sum or difference of two seen numbers, within the answer's rounding."""

    def near(target: float) -> bool:
        i = bisect.bisect_left(sources, target - tolerance)
        return i < len(sources) and sources[i] <= target + tolerance

    if near(value):
        return True
    return any(near(a - value) or near(value - a) or near(a + value) for a in sources)


def _numbers_in_tool_result(content: str) -> list[float]:
    numbers = [float(n) for n in _NUMBER.findall(content)]  # CSV: commas separate cells, not thousands
    lines = content.split("\n", 1)
    if len(lines) == 2 and lines[0].startswith("Result:"):  # run_sql CSV: add column totals and averages
        rows = list(csv.reader(io.StringIO(lines[1])))
        for column in zip(*rows[1:], strict=False):
            values = [v for v in (_to_float(cell) for cell in column) if v is not None]
            if len(values) > 1:
                numbers += [math.fsum(values), math.fsum(values) / len(values)]
    return numbers


def _to_float(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def caveat(missing: Iterable[str]) -> str:
    listed = "; ".join(missing)
    return (
        "\n\n> ⚠ *Automatic check: these figures do not appear in the query results behind this answer, "
        f"so treat them with caution: {listed}.*"
    )
