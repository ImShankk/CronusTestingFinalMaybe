"""Reminder and scheduled-task tools."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from ..automation.scheduler import RecurrenceError, parse_recurrence
from ..errors import CronusError
from ..logging_setup import get_logger
from .base import RiskLevel, Tool, ToolContext, ToolResult, object_schema

log = get_logger("tools.tasks")

_RELATIVE_RE = re.compile(
    r"^in\s+(\d+)\s*(minute|min|hour|hr|day|week)s?$", re.IGNORECASE
)
_CLOCK_RE = re.compile(
    r"^(today|tomorrow)(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.IGNORECASE
)


def parse_when(raw: str, now: datetime | None = None) -> datetime:
    """Turn a time expression into an absolute local datetime.

    ISO 8601 is the expected form -- the model is told the current time and can
    do the arithmetic -- but common relative phrasings are accepted too.
    """
    now = now or datetime.now()
    text = (raw or "").strip()
    if not text:
        raise CronusError("no time given", user_message="I need to know when.")

    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    relative = _RELATIVE_RE.match(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        delta = {
            "minute": timedelta(minutes=amount),
            "min": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "hr": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
        }[unit]
        return now + delta

    clock = _CLOCK_RE.match(text)
    if clock:
        day, hour, minute, meridiem = clock.groups()
        hour, minute = int(hour), int(minute or 0)
        if meridiem:
            meridiem = meridiem.lower()
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day.lower() == "tomorrow":
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target

    raise CronusError(
        f"unparseable time {raw!r}",
        user_message=(
            "I couldn't read that time. Give it as a date and time, like "
            "2026-08-22 09:00."
        ),
    )


def create_reminder(
    title: str,
    when: str | None = None,
    repeat: str | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    if context is None or context.scheduler is None:
        return ToolResult.failure("Scheduling isn't available right now.")

    moment = None
    if when:
        try:
            moment = parse_when(when)
        except CronusError as exc:
            return ToolResult.failure(exc.user_message)
        if moment < datetime.now() - timedelta(minutes=1):
            return ToolResult.failure(
                f"{moment:%Y-%m-%d %H:%M} is in the past. Ask the user which day they meant."
            )
    elif not repeat:
        return ToolResult.failure("Give either a time or a repeat schedule.")

    try:
        if repeat:
            repeat = parse_recurrence(repeat)
        task = context.scheduler.schedule(title, moment, recurrence=repeat)
    except RecurrenceError as exc:
        return ToolResult.failure(
            f"{exc}. Use daily, hourly, weekly:monday,friday, or every:30m."
        )
    except CronusError as exc:
        return ToolResult.failure(exc.user_message)

    log.info("reminder created id=%s", task.id)
    return ToolResult(
        content=f"Reminder {task.id} set: {task.title}, {task.due_description}"
        + (f", repeating {task.recurrence}." if task.recurrence else "."),
        display=f"reminder set for {task.due_description}",
        data={"id": task.id},
    )


def list_reminders(context: ToolContext | None = None) -> ToolResult:
    if context is None or context.scheduler is None:
        return ToolResult.failure("Scheduling isn't available right now.")
    tasks = context.scheduler.upcoming()
    if not tasks:
        return ToolResult(content="Nothing is scheduled.", display="no reminders")
    lines = "\n".join(task.summary() for task in tasks)
    return ToolResult(
        content=f"Upcoming reminders:\n{lines}", display=f"{len(tasks)} scheduled"
    )


def cancel_reminder(reminder_id: int, context: ToolContext | None = None) -> ToolResult:
    if context is None or context.scheduler is None:
        return ToolResult.failure("Scheduling isn't available right now.")
    if context.scheduler.cancel(int(reminder_id)):
        return ToolResult(content=f"Reminder {reminder_id} cancelled.", display="cancelled")
    return ToolResult.failure(f"There is no active reminder numbered {reminder_id}.")


def _cancel_preview(arguments: dict[str, Any]) -> str:
    return f"Cancel reminder {arguments.get('reminder_id')}?"


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="create_reminder",
            description=(
                "Set a reminder or a recurring task. Give when as an absolute "
                "date and time in ISO format, worked out from the current time "
                "you were told. For something recurring, set repeat to daily, "
                "hourly, weekly:monday,friday, or every:30m."
            ),
            parameters=object_schema(
                {
                    "title": {
                        "type": "string",
                        "description": "What to remind the user about, in their own words.",
                    },
                    "when": {
                        "type": "string",
                        "description": "Absolute time, e.g. 2026-08-22T09:00.",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "Repeat schedule, if this recurs.",
                    },
                },
                required=["title"],
            ),
            handler=create_reminder,
            risk=RiskLevel.LOW,
            category="productivity",
        ),
        Tool(
            name="list_reminders",
            description="List reminders and recurring tasks that are still scheduled.",
            parameters=object_schema({}),
            handler=list_reminders,
            risk=RiskLevel.SAFE,
            category="productivity",
        ),
        Tool(
            name="cancel_reminder",
            description="Cancel a scheduled reminder by its number.",
            parameters=object_schema(
                {"reminder_id": {"type": "integer", "description": "Number from the list."}},
                required=["reminder_id"],
            ),
            handler=cancel_reminder,
            risk=RiskLevel.CONFIRM,
            category="productivity",
            preview=_cancel_preview,
        ),
    ]
