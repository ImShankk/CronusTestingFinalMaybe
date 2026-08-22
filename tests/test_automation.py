"""Scheduler, recurrence, and reminder tools."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from cronus.automation.scheduler import (
    RecurrenceError,
    Scheduler,
    next_occurrence,
    parse_recurrence,
)
from cronus.storage.db import Database
from cronus.tools.base import ToolContext
from cronus.tools.tasks_tools import create_reminder, list_reminders, parse_when


@pytest.fixture
def scheduler(database: Database) -> Scheduler:
    return Scheduler(database, tick=0.05)


# ----------------------------------------------------------------------
# Recurrence
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec", ["daily", "hourly", "weekly:monday", "weekly:monday,friday", "every:30m", "every:2h"]
)
def test_valid_recurrence_specs(spec):
    assert parse_recurrence(spec) == spec


@pytest.mark.parametrize("spec", ["yearly", "weekly:funday", "every:0m", "every:soon", ""])
def test_invalid_recurrence_specs_are_rejected(spec):
    with pytest.raises(RecurrenceError):
        parse_recurrence(spec)


def test_next_occurrence_moves_forward():
    now = datetime(2026, 8, 21, 9, 0)
    assert next_occurrence("daily", now) == now + timedelta(days=1)
    assert next_occurrence("every:30m", now) == now + timedelta(minutes=30)


def test_weekly_lands_on_the_requested_weekday():
    friday = datetime(2026, 8, 21, 9, 0)  # a Friday
    upcoming = next_occurrence("weekly:monday", friday)
    assert upcoming.weekday() == 0
    assert friday < upcoming <= friday + timedelta(days=7)


# ----------------------------------------------------------------------
# Time parsing
# ----------------------------------------------------------------------
def test_iso_times_are_parsed():
    assert parse_when("2026-08-22T09:00") == datetime(2026, 8, 22, 9, 0)


def test_relative_times_are_parsed():
    now = datetime(2026, 8, 21, 9, 0)
    assert parse_when("in 30 minutes", now) == now + timedelta(minutes=30)
    assert parse_when("in 2 hours", now) == now + timedelta(hours=2)


def test_clock_times_are_parsed():
    now = datetime(2026, 8, 21, 9, 0)
    assert parse_when("tomorrow at 8", now) == datetime(2026, 8, 22, 8, 0)
    assert parse_when("today at 5pm", now) == datetime(2026, 8, 21, 17, 0)


def test_a_clock_time_already_past_rolls_to_tomorrow():
    now = datetime(2026, 8, 21, 20, 0)
    assert parse_when("today 8:00", now) == datetime(2026, 8, 22, 8, 0)


def test_unparseable_times_are_reported():
    from cronus.errors import CronusError

    with pytest.raises(CronusError) as info:
        parse_when("whenever you feel like it")
    # The technical detail stays in the exception; the user gets guidance.
    assert "couldn't read that time" in info.value.user_message
    assert "2026-08-22 09:00" in info.value.user_message


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------
def test_a_task_becomes_due_and_fires(scheduler: Scheduler):
    fired = []
    scheduler.on_due = fired.append
    scheduler.schedule("Stand up", datetime.now() - timedelta(seconds=1))

    scheduler.start()
    deadline = time.time() + 3
    while not fired and time.time() < deadline:
        time.sleep(0.05)
    scheduler.stop()

    assert [task.title for task in fired] == ["Stand up"]


def test_a_fired_task_is_not_repeated(scheduler: Scheduler):
    task = scheduler.schedule("Once only", datetime.now() - timedelta(seconds=1))
    scheduler.complete(task)
    assert scheduler.due_tasks() == []
    assert scheduler.get(task.id).status == "done"


def test_recurring_tasks_reschedule_themselves(scheduler: Scheduler):
    task = scheduler.schedule(
        "Weekly check", datetime.now() - timedelta(seconds=1), recurrence="daily"
    )
    scheduler.complete(task)
    refreshed = scheduler.get(task.id)
    assert refreshed.status == "scheduled"
    assert refreshed.next_run_at > time.time()
    assert refreshed.run_count == 1


def test_future_tasks_are_not_due_yet(scheduler: Scheduler):
    scheduler.schedule("Later", datetime.now() + timedelta(hours=1))
    assert scheduler.due_tasks() == []


def test_cancelling_removes_a_task_from_the_queue(scheduler: Scheduler):
    task = scheduler.schedule("Drop me", datetime.now() + timedelta(hours=1))
    assert scheduler.cancel(task.id) is True
    assert scheduler.upcoming() == []
    assert scheduler.cancel(task.id) is False


def test_tasks_survive_a_restart(tmp_path):
    path = tmp_path / "tasks.db"
    first = Database(path)
    Scheduler(first).schedule("Persisted", datetime.now() + timedelta(hours=1))
    first.close()

    second = Database(path)
    try:
        assert [t.title for t in Scheduler(second).upcoming()] == ["Persisted"]
    finally:
        second.close()


def test_a_failing_delivery_does_not_kill_the_scheduler(scheduler: Scheduler):
    def explode(task):
        raise RuntimeError("delivery failed")

    scheduler.on_due = explode
    task = scheduler.schedule("Boom", datetime.now() - timedelta(seconds=1))
    scheduler.start()
    time.sleep(0.3)
    scheduler.stop()
    # The task is still marked handled, and the thread survived to do it.
    assert scheduler.get(task.id).status == "done"


# ----------------------------------------------------------------------
# Reminder tools
# ----------------------------------------------------------------------
def test_create_reminder_tool_schedules(config, scheduler: Scheduler):
    context = ToolContext(config=config, scheduler=scheduler)
    result = create_reminder(
        "Bring an umbrella", when="2030-01-01T08:00", context=context
    )
    assert result.ok
    assert "Bring an umbrella" in list_reminders(context=context).content


def test_reminders_in_the_past_are_refused(config, scheduler: Scheduler):
    context = ToolContext(config=config, scheduler=scheduler)
    result = create_reminder("Too late", when="2020-01-01T08:00", context=context)
    assert not result.ok and "in the past" in result.content


def test_a_bad_repeat_spec_explains_the_valid_ones(config, scheduler: Scheduler):
    context = ToolContext(config=config, scheduler=scheduler)
    result = create_reminder("Sometimes", repeat="fortnightly", context=context)
    assert not result.ok and "weekly:monday" in result.content


def test_a_reminder_needs_a_time_or_a_repeat(config, scheduler: Scheduler):
    context = ToolContext(config=config, scheduler=scheduler)
    assert not create_reminder("When?", context=context).ok
