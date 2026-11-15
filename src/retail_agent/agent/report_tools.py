"""Report-library and preference tools, bound to the session user.

High-stakes oversight (requirement 3) lives in `delete_reports`:

1. candidates are resolved in SQL and always filtered by the session owner;
2. the graph pauses with `interrupt()` and the CLI shows the exact reports to the human, who must
   type "delete N" (the model cannot answer for them: only the CLI can resume the graph);
3. on resume LangGraph re-runs the tool from the top, so the candidates are resolved again and their
   fingerprint must equal the one the human confirmed; if anything changed, nothing is deleted;
4. the delete is soft, and every step (requested, confirmed, cancelled, aborted) is in the audit log.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from retail_agent.safety.pii_masker import PiiMasker
from retail_agent.storage import PreferenceStore, Report, ReportStore

PREVIEW_LIMIT = 50  # more matches than this: ask the user to narrow the request


def make_report_tools(
    *, reports: ReportStore, preferences: PreferenceStore, masker: PiiMasker, user_id: str
) -> list[BaseTool]:
    @tool(parse_docstring=True)
    def save_report(title: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Save the report you are about to write to the user's report library.

        Call it after gathering the data and right before writing the report: your next answer, the
        full report, is what gets saved.

        Args:
            title: Short descriptive title, e.g. "Q3 2026 performance review".
        """
        clean = masker.mask_text(" ".join(title.split()))[0][:120] or "Untitled report"
        message = ToolMessage(
            content=f'OK. Now write the full report as your answer; it will be saved as "{clean}".',
            name="save_report",
            tool_call_id=tool_call_id,
            artifact={"status": "pending_save", "title": clean},
        )
        return Command(update={"pending_report": {"title": clean}, "messages": [message]})

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def list_reports(
        config: RunnableConfig, mentioning: str = "", this_conversation: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """List the user's saved reports, newest first.

        Args:
            mentioning: Only reports whose title or text mention this (a brand, client, topic...).
            this_conversation: Only reports created in the current conversation.
        """
        found = reports.find(
            user_id, mentioning=mentioning or None, thread_id=_thread(config) if this_conversation else None
        )
        if not found:
            return "No saved reports match.", {"status": "none", "rows": 0}
        lines = [f"- {r.id} · {r.created_at[:10]} · {r.title}" for r in found]
        return f"{len(found)} report(s):\n" + "\n".join(lines), {"status": "ok", "rows": len(found)}

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def delete_reports(
        config: RunnableConfig,
        trace_id: Annotated[str, InjectedState("trace_id")],
        mentioning: str = "",
        this_conversation: bool = False,
        report_ids: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Delete some of the user's saved reports. The user always confirms in the interface first.

        Pass exactly the criteria the user gave; several criteria must all match. Call it once per request.

        Args:
            mentioning: Reports whose title or text mention this text, e.g. "Calvin Klein".
            this_conversation: Reports created in the current conversation.
            report_ids: Specific report ids, e.g. ["r-1a2b3c4d"].
        """
        mentioning = " ".join(mentioning.split())
        if not (mentioning or this_conversation or report_ids):
            return (
                "Ask the user which reports to delete (text they mention, this conversation, or ids).",
                {"status": "invalid"},
            )
        criteria = _criteria(mentioning, this_conversation, report_ids)
        candidates = reports.find(
            user_id,
            mentioning=mentioning or None,
            thread_id=_thread(config) if this_conversation else None,
            ids=report_ids,
            limit=PREVIEW_LIMIT + 1,
        )
        if not candidates:
            return f"None of the user's reports match {criteria}. Nothing was deleted.", {
                "status": "none",
                "criteria": criteria,
            }
        if len(candidates) > PREVIEW_LIMIT:
            return f"More than {PREVIEW_LIMIT} reports match {criteria}: ask the user to narrow it down.", {
                "status": "too_many",
                "criteria": criteria,
            }

        ids = sorted(r.id for r in candidates)
        token = _fingerprint(user_id, ids)
        reports.audit(user_id, "delete_requested", ids, criteria=criteria, trace_id=trace_id, token=token)

        decision = interrupt(
            {
                "kind": "confirm_delete",
                "token": token,
                "criteria": criteria,
                "reports": [_preview(r) for r in candidates],
            }
        )

        if not isinstance(decision, dict) or decision.get("token") != token:
            reports.audit(user_id, "delete_aborted", ids, criteria=criteria, trace_id=trace_id, token=token)
            return (
                "The matching reports changed while waiting for confirmation, so nothing was deleted. "
                "Tell the user and ask them to repeat the request.",
                {"status": "changed", "criteria": criteria},
            )
        if decision.get("confirmed") is not True:
            reports.audit(user_id, "delete_cancelled", ids, criteria=criteria, trace_id=trace_id, token=token)
            return (
                "Cancelled in the confirmation dialog: nothing was deleted. Tell the user in one sentence.",
                {"status": "cancelled", "ids": ids},
            )

        deleted = reports.soft_delete(user_id, ids, trace_id=trace_id)
        reports.audit(user_id, "delete_confirmed", deleted, criteria=criteria, trace_id=trace_id, token=token)
        titles = "; ".join(f"{r.id} {r.title}" for r in candidates if r.id in deleted)
        return (
            f"Deleted {len(deleted)} report(s): {titles}. Confirm this to the user in one sentence.",
            {"status": "deleted", "ids": deleted},
        )

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def remember_preference(preference: str) -> tuple[str, dict[str, Any]]:
        """Remember a lasting preference of this user about answers: format, length, depth, charts or text.

        Only for preferences the user stated or confirmed, not for one-off requests.

        Args:
            preference: One short sentence, e.g. "Prefers tables to bullet points".
        """
        text = masker.mask_text(" ".join(preference.split()))[0][:200]
        if not text:
            return "Nothing to remember.", {"status": "invalid"}
        preferences.add(user_id, text)
        return f"Saved. Applied from now on: {text}", {"status": "ok", "preference": text}

    return [save_report, list_reports, delete_reports, remember_preference]


def _thread(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _fingerprint(user_id: str, ids: list[str]) -> str:
    return hashlib.sha256(json.dumps([user_id, ids]).encode()).hexdigest()[:16]


def _criteria(mentioning: str, this_conversation: bool, report_ids: list[str] | None) -> str:
    parts = []
    if mentioning:
        parts.append(f'mentioning "{mentioning}"')
    if this_conversation:
        parts.append("created in this conversation")
    if report_ids:
        parts.append(f"with ids {', '.join(report_ids)}")
    return " and ".join(parts)


def _preview(report: Report) -> dict[str, str]:
    return {"id": report.id, "title": report.title, "created_at": report.created_at}
