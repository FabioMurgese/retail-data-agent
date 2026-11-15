"""Rich renderings of observability data, shared by the chat (/trace, /traces, /stats) and the ops
commands (`retail-agent trace | traces | stats`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table

_STATUS_STYLE = {"ok": "green", "blocked": "yellow", "degraded": "yellow", "error": "red", "stopped": "red",
                 "interrupted": "yellow"}  # fmt: skip


def render_turn_list(console: Console, turns: Sequence[dict[str, Any]]) -> None:
    if not turns:
        console.print("[dim]No turns recorded yet.[/]")
        return
    table = Table(header_style="bold")
    for column in ("trace", "ended (UTC)", "user", "status", "time", "LLM", "SQL", "question"):
        table.add_column(column, overflow="fold" if column == "question" else "ellipsis")
    for turn in turns:
        data = turn["data"]
        table.add_row(
            turn["trace_id"],
            turn["ts"][:19].replace("T", " "),
            turn["user_id"] or "",
            f"[{_STATUS_STYLE.get(turn['status'], 'white')}]{turn['status']}[/]",
            f"{(turn['duration_ms'] or 0) / 1000:.1f}s",
            str(data["llm"]["calls"]),
            str(data["sql"]["queries"]),
            escape((data.get("question") or "")[:80]),
        )
    console.print(table)


def render_trace(console: Console, events: Sequence[dict[str, Any]], *, full: bool = False) -> None:
    """Timeline of one turn plus its message correspondence: the deep-dive view."""
    turn = next((e for e in events if e["kind"] == "turn"), None)
    if turn is None:
        console.print("[dim]Trace not found.[/]")
        return
    data = turn["data"]
    llm, sql = data["llm"], data["sql"]
    style = _STATUS_STYLE.get(turn["status"], "white")
    console.print(
        f"[bold]Trace {turn['trace_id']}[/] · {turn['user_id']} · {turn['ts'][:19].replace('T', ' ')} UTC · "
        f"[{style}]{turn['status']}[/] · {(turn['duration_ms'] or 0) / 1000:.1f}s"
    )
    models = ", ".join(f"{m} ×{n}" for m, n in llm["models"].items()) or "none"
    cost = data["cost_usd"]
    console.print(
        f"[dim]LLM: {models} · {llm['input_tokens']:,} tokens in / {llm['output_tokens']:,} out · "
        f"SQL: {sql['queries']} queries, {sql['bytes_billed'] / 1e6:.0f} MB billed · "
        f"est. cost ${cost['llm'] + cost['bigquery']:.4f}[/]"
    )
    if data.get("error"):
        console.print(f"[red]Error: {escape(data['error'])}[/]")
    console.print(f"\n[bold]Question[/] {escape(data.get('question') or '')}\n")

    start = events[0]["ts"]
    for event in events:
        if event["kind"] in ("turn",):
            continue
        offset = _seconds_between(start, event["ts"])
        console.print(f"[dim]+{offset:5.1f}s[/] {_describe(event)}", highlight=False)
        if event["kind"] == "tool_call" and event["name"] == "run_sql":
            artifact = event["data"].get("artifact", {})
            _sql_block(console, "model SQL", artifact.get("sql"), full)
            if full:
                _sql_block(console, "guarded SQL (what BigQuery ran)", artifact.get("guarded_sql"), full)
                console.print(f"[dim]{escape(event['data'].get('result', ''))}[/]")

    answer = data.get("answer") or ""
    shown = (
        answer
        if full
        else "\n".join(answer.splitlines()[:8]) + ("\n…" if len(answer.splitlines()) > 8 else "")
    )
    console.print(f"\n[bold]Answer[/]\n{escape(shown)}")
    if full:
        console.print("\n[bold]Message correspondence[/]")
        for message in data.get("messages", []):
            calls = ", ".join(c["name"] for c in message.get("tool_calls", []))
            who = (
                message["role"]
                + (f" → {calls}" if calls else "")
                + (f" ({message['tool']})" if message.get("tool") else "")
            )
            console.print(f"[cyan]{escape(who)}[/] {escape(message['text'][:600])}")
    else:
        console.print("[dim](add --full / `/trace <id> full` for guarded SQL, results and every message)[/]")


def render_stats(console: Console, stats: dict[str, Any], *, title: str) -> None:
    console.print(f"[bold]{escape(title)}[/] · {stats['turns']} turns")
    if not stats["turns"]:
        return
    rates, llm, sql, quality = stats["rates"], stats["llm"], stats["sql"], stats["quality"]
    feedback, reports, cost = stats["feedback"], stats["reports"], stats["cost_usd"]
    latency = f"p50 {_secs(stats['latency_ms']['p50'])} · p95 {_secs(stats['latency_ms']['p95'])}"
    satisfaction = (
        f"{_pct(feedback['satisfaction'])} satisfied" if feedback["satisfaction"] is not None else ""
    )
    deletes = f"deletes: {reports['deletes_confirmed']} confirmed, {reports['deletes_cancelled']} cancelled"
    rows = [
        ("Outcomes", _pct(rates["success"]) + " answered fully", _counts(stats["status"])),
        ("Blocked by input guard", _pct(rates["blocked"]), _counts(stats["blocked_by"])),
        ("Degraded answers", _pct(rates["degraded"]), _counts(stats["degraded_by"])),
        ("Errors", _pct(rates["error"]), ""),
        ("Latency (answered)", latency, ""),
        ("LLM calls per turn", _num(llm["calls_per_turn"]), _counts(llm["models"])),
        ("Tokens", f"{llm['input_tokens']:,} in · {llm['output_tokens']:,} out", ""),
        ("Turns needing a fallback model", _pct(llm["fallback_turn_rate"]), ""),
        ("SQL queries", f"{sql['queries']} ({_num(sql['per_turn'])} per turn with SQL)", ""),
        ("SQL failed / refused", _pct(sql["failure_rate"]), _counts(sql["refused_by"])),
        ("SQL empty results", _pct(sql["empty_rate"]), ""),
        ("Self-corrected after a failure", _pct(sql["self_corrected"]), ""),
        ("Query-limit stops / BigQuery down", f"{sql['budget_stops']} / {sql['unavailable']}", ""),
        ("BigQuery billed", f"{sql['bytes_billed'] / 1e6:,.0f} MB", ""),
        ("Answers with unbacked figures", _pct(quality["ungrounded_rate"]), ""),
        ("Answers where PII was masked", str(quality["masked_output_turns"]), "should stay 0"),
        ("Example retrieval", "", _counts(quality["retrieval"])),
        ("Feedback", f"{feedback['good']} good · {feedback['bad']} bad", satisfaction),
        ("Reports", f"{reports['saved']} saved", deletes),
        (
            "Estimated cost",
            f"${cost['llm'] + cost['bigquery']:.4f}",
            f"LLM ${cost['llm']:.4f} · BigQuery ${cost['bigquery']:.4f}",
        ),
    ]
    table = Table(show_header=False, box=None, padding=(0, 2))
    for label, value, detail in rows:
        table.add_row(label, value, f"[dim]{escape(detail)}[/]")
    console.print(table)


def _describe(event: dict[str, Any]) -> str:
    kind, name, status, data = event["kind"], event["name"], event["status"], event["data"]
    ms = f"{(event['duration_ms'] or 0) / 1000:.1f}s"
    if kind == "input_guard":
        return f"[bold]input guard[/]  {status} ({escape(str(data.get('category')))}"\
               + (f", rule {escape(str(data.get('rule')))})" if data.get("rule") else ")")  # fmt: skip
    if kind == "retrieval":
        trios = ", ".join(f"{t['id']} {t['score']}" for t in data.get("trios", [])) or "none"
        error = f" · fallback reason: {escape(data['error'][:80])}" if data.get("error") else ""
        return f"[bold]examples[/]     {name}: {escape(trios)}{error}"
    if kind == "llm_call":
        usage = data.get("usage") or {}
        tools = ", ".join(data.get("tool_calls", [])) or "answer"
        fallbacks = "".join(
            f" · [yellow]skipped {f['model']} ({escape(f['reason'][:40])})[/]"
            for f in data.get("fallbacks", [])
        )
        return (f"[bold]llm[/]          {name or '(no model)'} · {ms} · {usage.get('input_tokens', 0):,}→"
                f"{usage.get('output_tokens', 0):,} tok · → {escape(tools)}{fallbacks}")  # fmt: skip
    if kind == "tool_call":
        artifact = data.get("artifact", {})
        detail = artifact.get("purpose") or artifact.get("title") or artifact.get("criteria") or ""
        extra = f" · {artifact['rows']} rows" if "rows" in artifact else ""
        if artifact.get("code"):
            extra += f" · {artifact['code']}"
        return f"[bold]{escape(name or 'tool'):<12}[/] {status}{extra} · {ms} · {escape(str(detail)[:90])}"
    if kind == "confirmation":
        count = len(data.get("report_ids", []))
        return f"[bold]confirmation[/] {status}: {count} report(s) {escape(str(data.get('criteria')))}"
    if kind == "output":
        ungrounded = ", ".join(data.get("ungrounded") or []) or "none"
        saved = (data.get("saved_report") or {}).get("id", "no")
        masked = data.get("output_masked") or "none"
        return (
            f"[bold]output[/]       unbacked figures: {escape(ungrounded)} · PII masked: {masked} · "
            f"report saved: {saved}"
        )
    if kind == "feedback":
        return f"[bold]feedback[/]     {status} {escape(data.get('comment') or '')}"
    return f"{kind} {name or ''} {status or ''}"


def _sql_block(console: Console, label: str, sql: str | None, full: bool) -> None:
    if not sql:
        return
    lines = sql.strip().splitlines()
    text = "\n".join(lines if full else lines[:6]) + ("" if full or len(lines) <= 6 else "\n-- …")
    console.print(f"[dim]         {label}:[/]")
    console.print(Syntax(text, "sql", theme="ansi_dark", background_color="default", padding=(0, 0, 0, 9)))


def _seconds_between(start: str, ts: str) -> float:
    from datetime import datetime

    return (datetime.fromisoformat(ts) - datetime.fromisoformat(start)).total_seconds()


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}"


def _secs(ms: float | None) -> str:
    return "–" if ms is None else f"{ms / 1000:.1f}s"


def _counts(counter: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]) if k)
