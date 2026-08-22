"""Reminders and recurring tasks.

A single background thread wakes every few seconds, asks SQLite for anything
due, and hands it to a callback the interface supplies. Tasks survive restarts
because they live in the database, not in memory. Recurrence is deliberately
small: daily, weekly on given weekdays, hourly, or every N minutes.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from ..errors import CronusError
from ..logging_setup import get_logger
from ..storage.db import Database

log = get_logger("automation.scheduler")

_TICK_SECONDS = 5.0
_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
)


@dataclass
class Task:
    id: int
    title: str
    instruction: str
    kind: str
    status: str
    next_run_at: float | None
    recurrence: str | None
    created_at: float
    run_count: int = 0

    @property
    def due_description(self) -> str:
        if self.next_run_at is None:
            return "no scheduled time"
        when = datetime.fromtimestamp(self.next_run_at)
        return when.strftime("%a %d %b at %H:%M")

    def summary(self) -> str:
        recurring = f", repeats {self.recurrence}" if self.recurrence else ""
        return f"[{self.id}] {self.title} - {self.due_description}{recurring}"


class RecurrenceError(CronusError):
    default_user_message = "I didn't understand that repeat schedule."


def parse_recurrence(spec: str) -> str:
    """Validate a recurrence spec, returning its normalised form.

    Accepted: ``daily``, ``hourly``, ``weekly:monday,friday``,
    ``every:30m`` / ``every:2h``.
    """
    spec = (spec or "").strip().lower()
    if spec in ("daily", "hourly"):
        return spec
    if spec.startswith("weekly"):
        _, _, days = spec.partition(":")
        chosen = [d.strip() for d in days.split(",") if d.strip()]
        unknown = [d for d in chosen if d not in _WEEKDAYS]
        if not chosen or unknown:
            raise RecurrenceError(f"unknown weekday(s): {', '.join(unknown) or 'none given'}")
        return f"weekly:{','.join(chosen)}"
    match = re.fullmatch(r"every:(\d+)([mh])", spec)
    if match:
        amount = int(match.group(1))
        if amount < 1:
            raise RecurrenceError("interval must be at least 1")
        return f"every:{amount}{match.group(2)}"
    raise RecurrenceError(f"unsupported recurrence {spec!r}")


def next_occurrence(recurrence: str, after: datetime) -> datetime:
    """The next time a recurring task should run, strictly after ``after``."""
    recurrence = parse_recurrence(recurrence)
    if recurrence == "hourly":
        return after + timedelta(hours=1)
    if recurrence == "daily":
        return after + timedelta(days=1)
    if recurrence.startswith("weekly:"):
        wanted = {
            _WEEKDAYS.index(day) for day in recurrence.split(":", 1)[1].split(",")
        }
        candidate = after
        for _ in range(1, 8):
            candidate = candidate + timedelta(days=1)
            if candidate.weekday() in wanted:
                return candidate
        return after + timedelta(days=7)  # pragma: no cover
    match = re.fullmatch(r"every:(\d+)([mh])", recurrence)
    assert match is not None  # parse_recurrence guarantees this
    amount, unit = int(match.group(1)), match.group(2)
    delta = timedelta(minutes=amount) if unit == "m" else timedelta(hours=amount)
    return after + delta


class Scheduler:
    """Stores tasks and fires them when they come due."""

    def __init__(
        self,
        db: Database,
        on_due: Callable[[Task], None] | None = None,
        *,
        tick: float = _TICK_SECONDS,
    ) -> None:
        self.db = db
        self.on_due = on_due
        self.tick = tick
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="cronus-scheduler", daemon=True
        )
        self._thread.start()
        log.info("scheduler started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("scheduler stopped")

    def _run(self) -> None:
        while not self._stop.wait(self.tick):
            try:
                for task in self.due_tasks():
                    self._fire(task)
            except Exception:  # pragma: no cover - the thread must survive
                log.exception("scheduler tick failed")

    def _fire(self, task: Task) -> None:
        log.info("task due id=%s title=%r", task.id, task.title)
        if self.on_due is not None:
            try:
                self.on_due(task)
            except Exception:  # pragma: no cover - delivery is best-effort
                log.exception("task delivery failed id=%s", task.id)
        self.complete(task)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------
    def schedule(
        self,
        title: str,
        when: datetime | None,
        *,
        instruction: str = "",
        kind: str = "reminder",
        recurrence: str | None = None,
    ) -> Task:
        if recurrence:
            recurrence = parse_recurrence(recurrence)
            if when is None:
                when = next_occurrence(recurrence, datetime.now())
        if when is None:
            raise CronusError(
                "a task needs a time",
                user_message="I need to know when to do that.",
            )

        now = time.time()
        with self.db.write() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks(title, instruction, kind, status, next_run_at,"
                " recurrence, created_at) VALUES (?, ?, ?, 'scheduled', ?, ?, ?)",
                (title, instruction, kind, when.timestamp(), recurrence, now),
            )
            task_id = int(cursor.lastrowid)
        log.info("task scheduled id=%s at=%s recurrence=%s", task_id, when, recurrence)
        return Task(
            id=task_id,
            title=title,
            instruction=instruction,
            kind=kind,
            status="scheduled",
            next_run_at=when.timestamp(),
            recurrence=recurrence,
            created_at=now,
        )

    def due_tasks(self, now: float | None = None) -> list[Task]:
        now = now if now is not None else time.time()
        rows = self.db.query(
            "SELECT * FROM tasks WHERE status = 'scheduled' AND next_run_at <= ?"
            " ORDER BY next_run_at",
            (now,),
        )
        return [_to_task(row) for row in rows]

    def complete(self, task: Task) -> None:
        """Mark a task done, or roll a recurring one forward to its next slot."""
        now = time.time()
        if task.recurrence:
            upcoming = next_occurrence(task.recurrence, datetime.fromtimestamp(now))
            with self.db.write() as connection:
                connection.execute(
                    "UPDATE tasks SET next_run_at = ?, last_run_at = ?,"
                    " run_count = run_count + 1 WHERE id = ?",
                    (upcoming.timestamp(), now, task.id),
                )
            log.info("recurring task id=%s rescheduled for %s", task.id, upcoming)
            return
        with self.db.write() as connection:
            connection.execute(
                "UPDATE tasks SET status = 'done', last_run_at = ?,"
                " run_count = run_count + 1 WHERE id = ?",
                (now, task.id),
            )

    def cancel(self, task_id: int) -> bool:
        with self.db.write() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status = 'cancelled' WHERE id = ? AND status = 'scheduled'",
                (task_id,),
            )
            cancelled = cursor.rowcount > 0
        if cancelled:
            log.info("task cancelled id=%s", task_id)
        return cancelled

    def upcoming(self, limit: int = 20) -> list[Task]:
        rows = self.db.query(
            "SELECT * FROM tasks WHERE status = 'scheduled' ORDER BY next_run_at LIMIT ?",
            (limit,),
        )
        return [_to_task(row) for row in rows]

    def get(self, task_id: int) -> Task | None:
        rows = self.db.query("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return _to_task(rows[0]) if rows else None


def _to_task(row) -> Task:
    return Task(
        id=int(row["id"]),
        title=row["title"],
        instruction=row["instruction"],
        kind=row["kind"],
        status=row["status"],
        next_run_at=row["next_run_at"],
        recurrence=row["recurrence"],
        created_at=float(row["created_at"]),
        run_count=int(row["run_count"]),
    )
