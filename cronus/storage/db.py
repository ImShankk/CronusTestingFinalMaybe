"""SQLite storage.

One database file holds memories, the user profile, and scheduled tasks.
Connections are per-thread (the tool pool and the scheduler both touch the
database), and the schema is created on demand.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..errors import StorageError
from ..logging_setup import get_logger

log = get_logger("storage.db")

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL DEFAULT 'fact',
    content     TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '',
    importance  INTEGER NOT NULL DEFAULT 1,
    source      TEXT    NOT NULL DEFAULT 'user',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    last_used_at REAL,
    use_count   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, tags, content='memories', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    instruction  TEXT    NOT NULL DEFAULT '',
    kind         TEXT    NOT NULL DEFAULT 'reminder',
    status       TEXT    NOT NULL DEFAULT 'scheduled',
    next_run_at  REAL,
    recurrence   TEXT,
    created_at   REAL    NOT NULL,
    last_run_at  REAL,
    run_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(status, next_run_at);

-- The tail of the last conversation, so closing the terminal does not
-- reset who Cronus is talking to. One row: the running summary plus the last
-- few completed turns. Deliberately not a transcript archive.
CREATE TABLE IF NOT EXISTS conversation_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    summary    TEXT NOT NULL DEFAULT '',
    turns      TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thread-local SQLite connections over a single file (or ``:memory:``)."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None
        # Re-entrant so a read inside a write block cannot deadlock.
        self._lock = threading.RLock()
        # Every connection handed out, so close() really closes all of them
        # and not just the calling thread's.
        self._connections: list[sqlite3.Connection] = []

        if self.path != ":memory:":
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StorageError(
                    f"cannot create data directory for {self.path}: {exc}",
                    user_message="I couldn't open my database file.",
                ) from exc
        self._init_schema()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=10.0, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        # An in-memory database only exists inside one connection, so share it.
        if self.path == ":memory:":
            with self._lock:
                if self._shared is None:
                    self._shared = self._connect()
                    self._connections.append(self._shared)
                return self._shared
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = self._connect()
            self._local.connection = existing
            with self._lock:
                self._connections.append(existing)
        return existing

    def _init_schema(self) -> None:
        try:
            with self.write() as connection:
                connection.executescript(_SCHEMA)
                connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
        except sqlite3.Error as exc:
            raise StorageError(
                f"cannot initialise database at {self.path}: {exc}",
                user_message="I couldn't set up my database.",
            ) from exc
        log.debug("database ready at %s", self.path)

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that commits on success and rolls back on error."""
        connection = self.connection
        try:
            with self._lock:
                yield connection
                connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            log.error("database write failed: %s", exc)
            raise StorageError(str(exc)) from exc

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        # Reads take the lock too: with an in-memory database every thread
        # shares one connection, so an unsynchronised read can interleave with
        # another thread's open transaction.
        try:
            with self._lock:
                return list(self.connection.execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            log.error("database query failed: %s", exc)
            raise StorageError(str(exc)) from exc

    def close(self) -> None:
        with self._lock:
            for connection in self._connections:
                try:
                    connection.close()
                except sqlite3.Error:  # pragma: no cover - shutdown best effort
                    pass
            self._connections.clear()
        self._local = threading.local()
        self._shared = None
