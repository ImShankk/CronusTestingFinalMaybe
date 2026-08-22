"""Assistant events.

The runtime narrates what it is doing through these events instead of
printing. That keeps the core independent of the CLI: a GUI or an API server
subscribes the same way.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..logging_setup import get_logger

log = get_logger("core.events")


class EventType(enum.Enum):
    STATE = "state"              # lifecycle state changed
    PROGRESS = "progress"        # short human-readable status line
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    RESPONSE = "response"        # final assistant text for this turn


class AssistantState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


Listener = Callable[[Event], None]


class EventEmitter:
    """A tiny synchronous pub/sub. A broken listener never breaks a turn."""

    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self.state: AssistantState = AssistantState.IDLE

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def emit(self, event: Event) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - listeners are best-effort
                log.exception("event listener failed")

    # Convenience wrappers -------------------------------------------------
    def set_state(self, state: AssistantState) -> None:
        if state is self.state:
            return
        self.state = state
        self.emit(Event(EventType.STATE, state.value, {"state": state}))

    def progress(self, message: str, **data: Any) -> None:
        self.emit(Event(EventType.PROGRESS, message, data))

    def tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.emit(
            Event(EventType.TOOL_START, name, {"tool": name, "arguments": arguments})
        )

    def tool_end(self, name: str, ok: bool, summary: str = "") -> None:
        self.emit(
            Event(EventType.TOOL_END, summary, {"tool": name, "ok": ok})
        )

    def error(self, message: str, **data: Any) -> None:
        self.emit(Event(EventType.ERROR, message, data))

    def response(self, text: str, **data: Any) -> None:
        self.emit(Event(EventType.RESPONSE, text, data))
