"""Interactive chat loop (`retail-agent chat --user <id>`): progress lines while the agent works,
the answer rendered as Markdown, confirmation of destructive actions, and a loop that survives any
error in a turn."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from retail_agent.agent.graph import strip_system_notes
from retail_agent.agent.prompts import describe_scope
from retail_agent.agent.session import AgentSession
from retail_agent.agent.turns import run_turn
from retail_agent.golden import write_candidate
from retail_agent.observability import compute_stats
from retail_agent.views import render_stats, render_trace, render_turn_list

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (line editing and history for input())

log = logging.getLogger(__name__)

HELP = """\
Ask a question about sales, customers or products, e.g.
  What were our top 5 categories by revenue last month?
  Create a report for Q3 with action items for Q4.
Commands: /reports · /report <id> · /prefs [clear] · /good|/bad [comment] (rate the last answer)
          /trace [id] [full] · /traces · /stats · /new (new conversation) · /whoami · /help · /exit"""

_CANCEL_WORDS = frozenset({"", "n", "no", "nope", "cancel", "stop", "abort", "don't", "dont"})
_VAGUE_YES = frozenset({"y", "yes", "ok", "okay", "sure", "confirm", "go", "go ahead", "do it"})


def confirmation_decision(answer: str, count: int) -> Literal["confirm", "cancel", "vague", "other"]:
    """Only the exact phrase "delete <count>" confirms; a bare "yes" is asked again; other text cancels."""
    normalised = " ".join(answer.lower().split())
    if normalised == f"delete {count}":
        return "confirm"
    if normalised in _CANCEL_WORDS:
        return "cancel"
    if normalised in _VAGUE_YES:
        return "vague"
    return "other"


class ChatApp:
    def __init__(
        self, session: AgentSession, graph: CompiledStateGraph, console: Console | None = None
    ) -> None:
        self.session = session
        self.graph = graph
        self.console = console or Console()
        self.last_trace_id: str | None = None

    # --- loop --------------------------------------------------------------------------------------

    def run(self) -> None:
        profile = self.session.profile
        self.console.print(
            f"[bold]TheLook data assistant[/] · signed in as [cyan]{profile.display_name}[/] "
            f"({escape(profile.role)}) · scope: {escape(describe_scope(profile))}"
        )
        self.console.print(f"[dim]{escape(HELP)}[/]\n")
        while True:
            text = self._input(f"[bold cyan]{self.session.user_id}>[/] ")
            if text is None:
                self.console.print("\nBye.")
                return
            if not text:
                continue
            if text.startswith("/"):
                if not self.command(text):
                    return
                continue
            self.turn(text)

    def turn(self, text: str) -> None:
        with self.console.status("Thinking…", spinner="dots") as status:

            def confirm(request: dict[str, Any]) -> tuple[bool, str | None]:
                status.stop()  # the spinner would overwrite the confirmation prompt
                try:
                    decision, typed = self._confirm_delete(request)
                finally:
                    status.start()
                return decision["confirmed"], typed

            outcome = run_turn(
                self.session,
                self.graph,
                text,
                confirm=confirm,
                on_update=lambda node, values: self._progress({node: values}, status),
            )
        self.last_trace_id = outcome.trace_id
        trace = f"[dim](trace {outcome.trace_id})[/]"
        if outcome.failure == "interrupted":
            self.console.print("[yellow]Interrupted.[/] The conversation is still open.")
            return
        if outcome.failure == "stopped":
            self.console.print(
                f"[yellow]That question needed too many steps, so I stopped.[/] Try a narrower one. {trace}"
            )
            return
        if outcome.failure == "error":
            self.console.print(
                f"[red]Something went wrong while answering.[/] The conversation is still open. {trace}"
            )
            return
        self.console.print()
        self.console.print(Markdown(outcome.answer or "(no answer)"))
        self.console.print(f"[dim]trace {outcome.trace_id}[/]\n")
        if outcome.follow_up:  # the user answered the confirmation with a new request: handle it now
            self.turn(outcome.follow_up)

    # --- destructive-action confirmation -----------------------------------------------------------

    def _confirm_delete(self, request: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Show exactly what would be deleted and read the human's answer (never the model's)."""
        reports = request["reports"]
        count = len(reports)
        table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
        table.add_column("id")
        table.add_column("title")
        table.add_column("created")
        for report in reports:
            table.add_row(report["id"], report["title"], report["created_at"][:16].replace("T", " "))
        noun = "your report" if count == 1 else f"{count} of your reports"
        self.console.print(
            Panel(
                Group(
                    Text(f"Delete {noun} {request['criteria']}?", style="bold"),
                    Text(""),
                    table,
                    Text(""),
                    Text.from_markup(f"Type [bold]delete {count}[/] to confirm. Anything else cancels."),
                ),
                title="Confirm deletion",
                border_style="red",
            )
        )
        typed = None
        for attempt in range(2):
            answer = self._input("[bold red]confirm>[/] ")
            decision = confirmation_decision(answer or "", count)
            if decision == "vague" and attempt == 0:
                self.console.print(
                    f'[yellow]To confirm, type exactly "delete {count}". Press Enter to cancel.[/]'
                )
                continue
            if decision == "other":
                typed = answer
            break
        confirmed = decision == "confirm"
        self.console.print(
            "[red]Confirmed.[/]" if confirmed else "[dim]Cancelled: nothing will be deleted.[/]"
        )
        return {"token": request["token"], "confirmed": confirmed}, typed

    # --- slash commands ----------------------------------------------------------------------------

    def command(self, text: str) -> bool:
        """Handle a slash command; False means quit. Commands read local data: no LLM call."""
        name, _, argument = text.partition(" ")
        name, argument = name.lower(), argument.strip()
        if name in ("/exit", "/quit"):
            self.console.print("Bye.")
            return False
        if name == "/new":
            self.session.new_thread()
            self.console.print("[dim]New conversation started.[/]")
        elif name == "/whoami":
            p = self.session.profile
            self.console.print(f"{p.display_name}, {escape(p.role)} · scope: {escape(describe_scope(p))}")
        elif name == "/reports":
            self._show_reports()
        elif name == "/report":
            self._show_report(argument)
        elif name == "/prefs":
            self._preferences(argument)
        elif name in ("/good", "/bad"):
            self._feedback(name[1:], argument)
        elif name == "/trace":
            self._trace(argument)
        elif name == "/traces":
            turns = self.session.events.events(
                kind="turn", user_id=self.session.user_id, limit=10, newest_first=True
            )
            render_turn_list(self.console, turns)
        elif name == "/stats":
            user = self.session.user_id
            turns = self.session.events.events(kind="turn", user_id=user)
            feedback = self.session.events.events(kind="feedback", user_id=user)
            render_stats(self.console, compute_stats(turns, feedback), title=f"Your assistant usage ({user})")
        else:
            self.console.print(f"[dim]{escape(HELP)}[/]")
        return True

    def _show_reports(self) -> None:
        reports = self.session.reports.find(self.session.user_id)
        if not reports:
            self.console.print("[dim]No saved reports yet.[/]")
            return
        table = Table(header_style="bold")
        for column in ("id", "title", "created", "this conversation"):
            table.add_column(column)
        for r in reports:
            here = "✓" if r.thread_id == self.session.thread_id else ""
            table.add_row(r.id, r.title, r.created_at[:16].replace("T", " "), here)
        self.console.print(table)

    def _show_report(self, report_id: str) -> None:
        found = self.session.reports.find(self.session.user_id, ids=[report_id]) if report_id else []
        if not found:
            self.console.print("[dim]Usage: /report <id> (see /reports).[/]")
            return
        self.console.print(Markdown(found[0].body))

    def _trace(self, argument: str) -> None:
        """The user's own turns only; the ops command `retail-agent trace` sees everyone's."""
        parts = argument.split()
        full = "full" in parts
        wanted = next((p for p in parts if p != "full"), None) or self.last_trace_id
        trace_id = self.session.events.resolve_trace(wanted, user_id=self.session.user_id) if wanted else None
        if trace_id is None:
            self.console.print("[dim]No such trace among your turns. See /traces.[/]")
            return
        render_trace(self.console, self.session.events.events(trace_id=trace_id), full=full)

    def _feedback(self, rating: str, comment: str) -> None:
        if self.last_trace_id is None:
            self.console.print("[dim]Ask something first, then rate the answer.[/]")
            return
        trace_id, events = self.last_trace_id, self.session.events
        comment = self.session.masker.mask_text(comment)[0]
        events.write(
            trace_id=trace_id,
            kind="feedback",
            user_id=self.session.user_id,
            thread_id=self.session.thread_id,
            status=rating,
            data={"comment": comment},
        )
        message = (
            "Thanks, noted."
            if rating == "good"
            else "Thanks, noted. Tell me what was wrong and I'll try again."
        )
        turn = next(iter(events.events(kind="turn", trace_id=trace_id)), None)
        if rating == "good" and turn and (path := self._propose_trio(turn)):
            message += f" Proposed as an example for the assistant ({path.name}); it is checked before use."
        self.console.print(f"[dim]{escape(message)}[/]")

    def _propose_trio(self, turn: dict[str, Any]) -> Path | None:
        """System-level learning: a well-rated, query-backed answer becomes a candidate golden trio."""
        data = turn["data"]
        queries = [
            call["args"]["sql"]
            for message in data.get("messages", [])
            for call in message.get("tool_calls", [])
            if call["name"] == "run_sql" and "sql" in call["args"]
        ]
        if data["status"] != "ok" or not data["sql"]["ok"] or not queries or not data.get("answer"):
            return None
        return write_candidate(
            self.session.settings.golden_candidates_dir,
            trace_id=turn["trace_id"],
            user_id=self.session.user_id,
            question=data["question"],
            answer=strip_system_notes(data["answer"]),
            queries=queries,
            today=date.today(),
            warnings=[f"the automatic check could not verify: {'; '.join(data['ungrounded'])}"]
            if data.get("ungrounded")
            else [],
        )

    def _preferences(self, argument: str) -> None:
        store, user = self.session.preferences, self.session.user_id
        if argument == "clear":
            self.console.print(f"[dim]Forgot {store.clear(user)} preference(s).[/]")
            return
        saved = store.list(user)
        self.console.print("\n".join(f"- {p}" for p in saved) if saved else "[dim]No saved preferences.[/]")

    # --- helpers -----------------------------------------------------------------------------------

    def _input(self, prompt: str) -> str | None:
        try:
            return self.console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def _progress(self, update: dict[str, Any], status: Status) -> None:
        for node, values in update.items():
            if not isinstance(values, dict):
                continue
            if node == "prepare_context" and (retrieval := values.get("retrieval")):
                ids = ", ".join(t["id"] for t in retrieval["trios"]) or "none"
                self.console.print(f"[dim]  analyst examples: {ids} ({retrieval['method']})[/]")
            elif node == "agent":
                answer = values["messages"][-1]
                for skipped in getattr(answer, "response_metadata", {}).get("fallbacks", []):
                    reason = skipped["reason"].split(":")[0]
                    model = skipped["model"]
                    self.console.print(f"[yellow]  ⚠ {model} unavailable ({escape(reason)}): fell back[/]")
                for call in getattr(answer, "tool_calls", None) or []:
                    purpose = call["args"].get("purpose") or call["name"].replace("_", " ")
                    self.console.print(f"[dim]  ▸ {escape(purpose)}[/]")
                    status.update("Querying BigQuery…" if call["name"] == "run_sql" else "Working…")
            elif node == "tools":
                for message in values.get("messages", []):
                    if isinstance(message, ToolMessage):
                        self.console.print(f"[dim]    {escape(describe_tool_result(message))}[/]")
                status.update("Thinking…")


def describe_tool_result(message: ToolMessage) -> str:
    artifact: dict[str, Any] = message.artifact or {}
    status = artifact.get("status")
    if message.name == "run_sql":
        if status == "ok":
            rows = f"{artifact['rows']}+ rows" if artifact.get("truncated") else f"{artifact['rows']} rows"
            mb = (artifact.get("bytes_billed") or 0) / 1e6
            return f"✓ {rows} · {artifact.get('elapsed_ms', 0) / 1000:.1f}s · {mb:.0f} MB billed"
        if status == "empty":
            return "∅ no rows"
        if status == "refused":
            return f"✗ refused by the data guard ({artifact.get('code')})"
        if status == "error":
            return f"✗ {artifact.get('error', 'error')}"[:160]
        if status == "unavailable":
            return "✗ BigQuery unavailable: skipped (circuit open)"
        if status == "budget_exhausted":
            return "■ query limit for this question reached"
    labels = {
        ("save_report", "pending_save"): f'✎ the answer will be saved as "{artifact.get("title")}"',
        ("list_reports", "ok"): f"✓ {artifact.get('rows')} report(s)",
        ("list_reports", "none"): "∅ no matching reports",
        ("delete_reports", "deleted"): f"✓ deleted {len(artifact.get('ids', []))} report(s)",
        ("delete_reports", "cancelled"): "✗ deletion cancelled, nothing deleted",
        ("delete_reports", "none"): "∅ no matching reports, nothing deleted",
        ("delete_reports", "changed"): "! reports changed before confirmation, nothing deleted",
        ("remember_preference", "ok"): "✓ preference saved",
    }
    return labels.get((message.name, status), "· done")
