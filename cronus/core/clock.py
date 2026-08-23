"""The user's clock.

Everything time-related resolves through here so "tomorrow at 9" means the
same thing to the context builder, the reminder parser, and the scheduler.

If ``CRONUS_TIMEZONE`` is unset, the system timezone is used. If it is set but
the IANA database is unavailable (Windows ships no tzdata), that is reported
once and the system timezone is used rather than failing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from functools import lru_cache

from ..logging_setup import get_logger

log = get_logger("core.clock")


@lru_cache(maxsize=8)
def resolve_timezone(name: str | None) -> tzinfo | None:
    """Return the configured zone, or None to mean "use the system zone"."""
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception as exc:
        log.warning(
            "cannot use timezone %r (%s); falling back to the system timezone. "
            "Install the tzdata package to enable named timezones.",
            name,
            type(exc).__name__,
        )
        return None


def now(timezone_name: str | None = None) -> datetime:
    """The current time as an aware datetime in the user's timezone."""
    zone = resolve_timezone(timezone_name)
    return datetime.now(zone) if zone else datetime.now().astimezone()


def to_user_time(moment: datetime, timezone_name: str | None = None) -> datetime:
    """Render an instant in the user's timezone."""
    zone = resolve_timezone(timezone_name)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(zone) if zone else moment.astimezone()


def localise(wall_clock: datetime, timezone_name: str | None = None) -> datetime:
    """Attach the user's timezone to a naive wall-clock time.

    "Remind me at 9" means 9 o'clock where the user is, which is not
    necessarily 9 o'clock where the machine thinks it is.
    """
    if wall_clock.tzinfo is not None:
        return wall_clock
    zone = resolve_timezone(timezone_name)
    return wall_clock.replace(tzinfo=zone) if zone else wall_clock.astimezone()


def describe(moment: datetime | None = None, timezone_name: str | None = None) -> str:
    """A human sentence for the current date and time."""
    moment = to_user_time(moment or now(timezone_name), timezone_name)
    return moment.strftime("%A %d %B %Y, %H:%M %Z").strip()


def describe_offset(days: int, timezone_name: str | None = None) -> str:
    """Name a day relative to today, e.g. tomorrow's date."""
    return (now(timezone_name) + timedelta(days=days)).strftime("%A %d %B %Y")


def describe_age(seconds: float) -> str:
    """How long ago something happened, in the words a person would use."""
    minutes = max(seconds, 0) / 60
    if minutes < 2:
        return "a moment ago"
    if minutes < 60:
        return f"{int(minutes)} minutes ago"
    hours = minutes / 60
    if hours < 24:
        return "about an hour ago" if hours < 2 else f"{int(hours)} hours ago"
    days = hours / 24
    if days < 2:
        return "yesterday"
    return f"{int(days)} days ago"
