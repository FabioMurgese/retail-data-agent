"""Local persistence (SQLite): the saved-reports library, its audit log, user preferences and
observability events (written by `observability.EventStore`).

One file (`data/agent.db`) keeps the prototype runnable anywhere; production uses Cloud SQL
(Postgres) with the same tables (docs/HLD.md). Every query takes the owner id from the session, so a
user can only see, and delete, their own reports.

Deletes are soft (`deleted_at`): the row stays for the audit trail and recovery; the report leaves
the user's library and its Markdown export is removed.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS reports (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    trace_id    TEXT,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS reports_by_owner ON reports (owner_id, deleted_at, created_at);

-- Full-text index for "delete all reports mentioning Client X". Rows are never updated or
-- hard-deleted, so an insert trigger keeps it in sync.
CREATE VIRTUAL TABLE IF NOT EXISTS reports_fts
    USING fts5 (title, body, content = 'reports', content_rowid = 'rowid');
CREATE TRIGGER IF NOT EXISTS reports_fts_insert AFTER INSERT ON reports BEGIN
    INSERT INTO reports_fts (rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    action      TEXT NOT NULL,
    report_ids  TEXT NOT NULL,  -- JSON array
    criteria    TEXT,
    trace_id    TEXT,
    token       TEXT,
    UNIQUE (trace_id, action, token)  -- a re-executed step logs once
);

-- Observability: one row per event (turn summary, LLM call, tool call, verdict, feedback...).
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    thread_id   TEXT,
    user_id     TEXT,
    kind        TEXT NOT NULL,
    name        TEXT,
    status      TEXT,
    duration_ms INTEGER,
    data        TEXT NOT NULL  -- JSON, PII-masked
);
CREATE INDEX IF NOT EXISTS events_by_trace ON events (trace_id);
CREATE INDEX IF NOT EXISTS events_by_kind ON events (kind, ts);

CREATE TABLE IF NOT EXISTS user_preferences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    preference  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (user_id, preference)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript(_SCHEMA)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection per operation: safe across the threads tools run in."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:  # commit on success, roll back on error
                yield db
        finally:
            db.close()


@dataclass(frozen=True)
class Report:
    id: str
    owner_id: str
    thread_id: str
    title: str
    body: str
    created_at: str


class ReportStore:
    def __init__(self, database: Database, files_dir: Path | None = None) -> None:
        self.db = database
        self.files_dir = files_dir

    def save(
        self, *, owner_id: str, thread_id: str, title: str, body: str, trace_id: str | None = None
    ) -> Report:
        report = Report(f"r-{uuid.uuid4().hex[:8]}", owner_id, thread_id, title, body, _now())
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO reports (id, owner_id, thread_id, trace_id, title, body, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (report.id, owner_id, thread_id, trace_id, title, body, report.created_at),
            )
            _audit(db, owner_id, "report_saved", [report.id], trace_id=trace_id)
        if self.files_dir is not None:  # a convenience export; the database is the source of truth
            self.files_dir.mkdir(parents=True, exist_ok=True)
            self._file(report.id).write_text(body, encoding="utf-8")
        return report

    def find(
        self,
        owner_id: str,
        *,
        mentioning: str | None = None,
        thread_id: str | None = None,
        ids: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[Report]:
        """The owner's live reports, newest first. Given criteria must all match."""
        where, params = ["owner_id = ?", "deleted_at IS NULL"], [owner_id]
        if mentioning and mentioning.strip():
            where.append("rowid IN (SELECT rowid FROM reports_fts WHERE reports_fts MATCH ?)")
            params.append(_fts_phrase(mentioning))
        if thread_id is not None:
            where.append("thread_id = ?")
            params.append(thread_id)
        if ids is not None:
            where.append(f"id IN ({', '.join('?' * len(ids))})" if ids else "0")
            params.extend(ids)
        sql = (
            "SELECT id, owner_id, thread_id, title, body, created_at FROM reports "
            f"WHERE {' AND '.join(where)} ORDER BY created_at DESC, rowid DESC LIMIT ?"
        )
        with self.db.transaction() as db:
            return [Report(**row) for row in db.execute(sql, [*params, limit])]

    def live_report_ids(self) -> list[str]:
        """Ids of all live reports, any owner (operational checks only; never exposed to a user)."""
        with self.db.transaction() as db:
            return [row["id"] for row in db.execute("SELECT id FROM reports WHERE deleted_at IS NULL")]

    def soft_delete(self, owner_id: str, ids: Sequence[str], *, trace_id: str | None = None) -> list[str]:
        """Delete exactly `ids` if they belong to `owner_id` and are live; returns what was deleted."""
        if not ids:
            return []
        marks = ", ".join("?" * len(ids))
        with self.db.transaction() as db:
            live = [
                row["id"]
                for row in db.execute(
                    f"SELECT id FROM reports WHERE owner_id = ? AND deleted_at IS NULL AND id IN ({marks})",
                    [owner_id, *ids],
                )
            ]
            if live:
                db.execute(
                    "UPDATE reports SET deleted_at = ? WHERE owner_id = ? "
                    f"AND id IN ({', '.join('?' * len(live))})",
                    [_now(), owner_id, *live],
                )
        for report_id in live:
            if self.files_dir is not None:
                self._file(report_id).unlink(missing_ok=True)
        return live

    def audit(
        self,
        actor_id: str,
        action: str,
        report_ids: Sequence[str],
        *,
        criteria: str | None = None,
        trace_id: str | None = None,
        token: str | None = None,
    ) -> None:
        with self.db.transaction() as db:
            _audit(db, actor_id, action, report_ids, criteria=criteria, trace_id=trace_id, token=token)

    def audit_entries(self, actor_id: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM audit_log", []
        if actor_id is not None:
            sql, params = sql + " WHERE actor_id = ?", [actor_id]
        with self.db.transaction() as db:
            rows = [dict(row) for row in db.execute(sql + " ORDER BY id", params)]
        for row in rows:
            row["report_ids"] = json.loads(row["report_ids"])
        return rows

    def _file(self, report_id: str) -> Path:
        assert self.files_dir is not None
        return self.files_dir / f"{report_id}.md"


class PreferenceStore:
    """What a user asked the assistant to remember about answers (format, depth...)."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def add(self, user_id: str, preference: str) -> None:
        with self.db.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO user_preferences (user_id, preference, created_at) VALUES (?, ?, ?)",
                (user_id, preference, _now()),
            )

    def list(self, user_id: str, limit: int = 10) -> list[str]:
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT preference FROM user_preferences WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            return [row["preference"] for row in rows][::-1]  # oldest first

    def clear(self, user_id: str) -> int:
        with self.db.transaction() as db:
            return db.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,)).rowcount


def _audit(
    db: sqlite3.Connection,
    actor_id: str,
    action: str,
    report_ids: Sequence[str],
    *,
    criteria: str | None = None,
    trace_id: str | None = None,
    token: str | None = None,
) -> None:
    db.execute(
        "INSERT OR IGNORE INTO audit_log (at, actor_id, action, report_ids, criteria, trace_id, token) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_now(), actor_id, action, json.dumps(list(report_ids)), criteria, trace_id, token),
    )


def _fts_phrase(text: str) -> str:
    """User or model text as one FTS5 phrase: no query syntax can be injected."""
    return '"' + " ".join(text.split()).replace('"', '""') + '"'
