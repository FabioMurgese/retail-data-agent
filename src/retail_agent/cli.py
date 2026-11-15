"""Command-line entry point (`retail-agent`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from retail_agent.config import Settings, get_settings

if TYPE_CHECKING:
    from retail_agent.observability import EventStore

app = typer.Typer(
    help="Data-analysis chat agent for retail executives (BigQuery thelook_ecommerce + Gemini).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def main() -> None:
    """Retail data-analysis chat agent."""


def _settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        console.print(f"[red]✗ Invalid settings (.env / environment):[/]\n{escape(str(exc))}")
        raise typer.Exit(1) from None


@app.command()
def chat(
    user: str = typer.Option(..., "--user", "-u", help="Who you are: a user id from config/users.yaml."),
    embeddings: bool = typer.Option(True, help="Use Gemini embeddings to find analyst examples (else BM25)."),
) -> None:
    """Chat with the data assistant."""
    from retail_agent.agent.graph import build_graph
    from retail_agent.agent.llm import build_chat_model
    from retail_agent.agent.session import AgentSession
    from retail_agent.chat import ChatApp
    from retail_agent.policy import ConfigFileError, UnknownUserError

    settings = _settings()
    _log_to_file(settings)
    try:
        session = AgentSession.create(settings, user, use_embeddings=embeddings)
        model = build_chat_model(settings, session.tools)
    except (UnknownUserError, ConfigFileError, ValueError) as exc:
        console.print(f"[red]✗ {escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    ChatApp(session, build_graph(session, model), console).run()


def _log_to_file(settings: Settings) -> None:
    """Keep the chat screen clean: library and agent logs go to logs/agent.log (phase 6: JSON)."""
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=settings.logs_dir / "agent.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _event_store(settings: Settings, data_dir: Path | None = None) -> EventStore:
    from retail_agent.observability import EventStore  # lazy: keeps `--help` fast
    from retail_agent.storage import Database

    return EventStore(Database((data_dir or settings.data_dir) / "agent.db"))


DATA_DIR_OPTION = typer.Option(None, "--data-dir", help="Read another workspace, e.g. data/evals/<run>.")


@app.command()
def traces(
    user: str | None = typer.Option(None, "--user", "-u", help="Only this user's turns."),
    limit: int = typer.Option(20, help="How many recent turns."),
    data_dir: Path | None = DATA_DIR_OPTION,
) -> None:
    """Ops: list recent turns of all users (newest first)."""
    from retail_agent.views import render_turn_list

    store = _event_store(_settings(), data_dir)
    render_turn_list(console, store.events(kind="turn", user_id=user, limit=limit, newest_first=True))


@app.command()
def trace(
    trace_id: str = typer.Argument(..., help="Trace id (or its first characters)."),
    full: bool = typer.Option(False, "--full", help="Guarded SQL, query results and every message."),
    data_dir: Path | None = DATA_DIR_OPTION,
) -> None:
    """Ops: replay one turn: timeline, SQL before/after the guard, verdicts, fallbacks, messages."""
    from retail_agent.views import render_trace

    store = _event_store(_settings(), data_dir)
    resolved = store.resolve_trace(trace_id)
    if resolved is None:
        console.print(f"[red]No trace starting with {escape(trace_id)}.[/]")
        raise typer.Exit(1)
    render_trace(console, store.events(trace_id=resolved), full=full)


@app.command()
def stats(
    user: str | None = typer.Option(None, "--user", "-u", help="Only this user."),
    since: str | None = typer.Option(None, help="Time window, e.g. 30m, 24h, 7d (default: everything)."),
    data_dir: Path | None = DATA_DIR_OPTION,
) -> None:
    """Ops: agent-level metrics: outcomes, latency, LLM and SQL behaviour, quality, feedback, cost."""
    from retail_agent.observability import compute_stats, parse_since
    from retail_agent.views import render_stats

    try:
        window = parse_since(since)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    store = _event_store(_settings(), data_dir)
    turns = store.events(kind="turn", user_id=user, since=window)
    feedback = store.events(kind="feedback", user_id=user, since=window)
    scope = f"user {user}" if user else "all users"
    render_stats(
        console, compute_stats(turns, feedback), title=f"Agent metrics · {scope} · {since or 'all time'}"
    )


@app.command("eval")
def run_evals(
    case: Annotated[
        list[str] | None, typer.Option("--case", "-c", help="Only these case ids (repeatable).")
    ] = None,
    category: Annotated[str | None, typer.Option(help="Only this category.")] = None,
    judge: Annotated[bool, typer.Option(help="Grade analyses and reports with an LLM judge.")] = True,
    cases_file: Annotated[Path, typer.Option(help="Cases to run.")] = Path("evals/cases.yaml"),
    out: Annotated[Path, typer.Option(help="Where to write the Markdown and JSON report.")] = Path(
        "evals/results"
    ),
) -> None:
    """Run the live evaluation suite against Gemini and BigQuery (uses real quota)."""
    from retail_agent.agent.graph import build_graph
    from retail_agent.agent.llm import build_chat_model
    from retail_agent.evals import CaseResult, gemini_judge, load_cases, run_suite, write_report

    settings = _settings()
    _log_to_file(settings)
    cases = [
        c
        for c in load_cases(cases_file)
        if (not case or c.id in case) and (category is None or c.category == category)
    ]
    if not cases:
        console.print("[red]No matching cases.[/]")
        raise typer.Exit(1)
    console.print(f"Running {len(cases)} cases · agent {', '.join(settings.model_chain)} · "
                  f"judge {', '.join(settings.eval_judge_models) if judge else 'off'}")  # fmt: skip

    def show(result: CaseResult) -> None:
        mark = "[green]✓[/]" if result.passed else "[red]✗[/]"
        failed = "; ".join(f"{c.name}: {c.detail}" for c in result.checks if c.status == "fail")
        score = f" · judge {result.judge_score}/5" if result.judge_score else ""
        console.print(
            f"{mark} {result.case:<26} {result.turn_status:<8} {result.llm_calls} LLM · "
            f"{result.queries} SQL · {result.seconds:.0f}s{score} [dim]{escape(failed)}[/]",
            highlight=False,
        )

    run = run_suite(
        cases,
        settings,
        make_graph=lambda session: build_graph(session, build_chat_model(session.settings, session.tools)),
        judge=gemini_judge(settings) if judge else None,
        on_result=show,
    )
    report = write_report(
        run, out, models=settings.model_chain, judge_models=settings.eval_judge_models if judge else []
    )
    passed = sum(r.passed for r in run.results)
    console.print(
        f"\n{passed}/{len(run.results)} passed · report {report} · traces: --data-dir data/evals/{run.run_id}"
    )
    raise typer.Exit(0 if passed == len(run.results) else 1)


@app.command()
def doctor() -> None:
    """Check configuration, credentials and policy files against BigQuery and Gemini."""
    from retail_agent.doctor import Status, run_checks

    settings = _settings()

    style = {
        Status.OK: "[green]✓[/]",
        Status.WARN: "[yellow]![/]",
        Status.FAIL: "[red]✗[/]",
        Status.SKIP: "[dim]-[/]",
    }
    failed = False
    for check in run_checks(settings):
        failed |= check.status is Status.FAIL
        console.print(
            f"{style[check.status]} [bold]{check.name:<18}[/] {escape(check.detail)}", highlight=False
        )

    console.print("\n[red]Some checks failed.[/]" if failed else "\n[green]Ready.[/]")
    raise typer.Exit(1 if failed else 0)
