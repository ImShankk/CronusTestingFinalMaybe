"""Tool definitions: what a tool is, what it receives, what it returns."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from ..llm.base import ToolSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config
    from ..core.events import EventEmitter
    from ..memory.store import MemoryStore
    from ..automation.scheduler import Scheduler
    from ..security.paths import PathGuard


class RiskLevel(enum.Enum):
    """How much damage a tool can do, decided by us and never by the model."""

    SAFE = "safe"          # read-only, no side effects outside this process
    LOW = "low"            # small, reversible local side effects
    CONFIRM = "confirm"    # visible to the outside world or destructive
    HIGH = "high"          # dangerous; allowed only with explicit opt-in
    BLOCKED = "blocked"    # never executed

    @property
    def order(self) -> int:
        return _RISK_ORDER[self]


_RISK_ORDER = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.CONFIRM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.BLOCKED: 4,
}


@dataclass
class ToolContext:
    """Services a handler may use.

    Handlers get their dependencies here rather than from module globals, so
    they stay testable and there is exactly one place that decides what a tool
    is allowed to reach.
    """

    config: "Config"
    emitter: "EventEmitter | None" = None
    memory: "MemoryStore | None" = None
    scheduler: "Scheduler | None" = None
    paths: "PathGuard | None" = None
    session: dict[str, Any] = field(default_factory=dict)

    def progress(self, message: str) -> None:
        if self.emitter is not None:
            self.emitter.progress(message)


@dataclass
class ToolResult:
    """What a tool hands back.

    ``content`` is what the model reads. ``display`` is an optional short line
    for the user. ``data`` keeps structured output for callers that want it.
    """

    content: str
    ok: bool = True
    display: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, content: str, **data: Any) -> "ToolResult":
        return cls(content=content, ok=False, data=data)


ToolHandler = Callable[..., Any]


class ConfirmationPreview(Protocol):
    """Renders the human-readable summary shown before a risky action."""

    def __call__(self, arguments: dict[str, Any]) -> str: ...


@dataclass
class Tool:
    """A capability Cronus can invoke on the model's behalf."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    risk: RiskLevel = RiskLevel.SAFE
    category: str = "general"
    timeout: float | None = None
    preview: ConfirmationPreview | None = None
    enabled: bool = True

    def schema(self) -> ToolSchema:
        """The model-facing view. Risk metadata deliberately stays internal."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def describe_call(self, arguments: dict[str, Any]) -> str:
        """A one-line summary of a pending call, for confirmation prompts."""
        if self.preview is not None:
            try:
                return self.preview(arguments)
            except Exception:  # pragma: no cover - preview must never break a call
                pass
        rendered = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"{self.name}({rendered})"


def object_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    """Shorthand for the object schema shape every tool uses."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
