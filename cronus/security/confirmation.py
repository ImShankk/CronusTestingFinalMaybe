"""Confirmation for consequential actions.

Confirmation is enforced in application code, not by asking the model nicely.
A pending request is a real object with state and an expiry; the interface
(CLI today, a GUI later) only supplies a handler that answers it.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..logging_setup import get_logger

log = get_logger("security.confirmation")


class ConfirmationStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ConfirmationRequest:
    """A consequential action waiting for a yes or no."""

    tool_name: str
    summary: str
    arguments: dict[str, Any]
    details: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    status: ConfirmationStatus = ConfirmationStatus.PENDING

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def render(self) -> str:
        lines = [self.summary]
        for label, value in self.details.items():
            lines.append(f"{label}: {value}")
        return "\n".join(lines)


class ConfirmationHandler(Protocol):
    """Answers a confirmation request. Interfaces implement this."""

    def __call__(self, request: ConfirmationRequest) -> bool: ...


def always_decline(request: ConfirmationRequest) -> bool:
    """Default handler: an interface that cannot ask must not act."""
    log.warning(
        "no confirmation handler available; declining %s", request.tool_name
    )
    return False


class ConfirmationManager:
    """Creates, tracks, and resolves confirmation requests."""

    def __init__(
        self,
        handler: ConfirmationHandler | None = None,
        *,
        timeout: float = 120.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.handler: ConfirmationHandler = handler or always_decline
        self.timeout = timeout
        self._clock = clock
        self._history: list[ConfirmationRequest] = []
        self.pending: ConfirmationRequest | None = None

    def set_handler(self, handler: ConfirmationHandler) -> None:
        self.handler = handler

    def request(
        self,
        tool_name: str,
        summary: str,
        arguments: dict[str, Any],
        details: dict[str, str] | None = None,
    ) -> ConfirmationRequest:
        request = ConfirmationRequest(
            tool_name=tool_name,
            summary=summary,
            arguments=arguments,
            details=details or {},
            created_at=self._clock(),
            expires_at=self._clock() + self.timeout if self.timeout else None,
        )
        self.pending = request
        self._history.append(request)
        log.info("confirmation requested id=%s tool=%s", request.id, tool_name)
        return request

    def resolve(self, request: ConfirmationRequest) -> bool:
        """Ask the handler and record the outcome. Returns True if approved."""
        if request.expired:
            request.status = ConfirmationStatus.EXPIRED
            self.pending = None
            log.info("confirmation expired id=%s", request.id)
            return False
        try:
            approved = bool(self.handler(request))
        except (KeyboardInterrupt, EOFError):
            request.status = ConfirmationStatus.CANCELLED
            self.pending = None
            log.info("confirmation cancelled id=%s", request.id)
            return False
        except Exception:
            log.exception("confirmation handler failed id=%s", request.id)
            request.status = ConfirmationStatus.DECLINED
            self.pending = None
            return False

        request.status = (
            ConfirmationStatus.APPROVED if approved else ConfirmationStatus.DECLINED
        )
        self.pending = None
        log.info(
            "confirmation resolved id=%s tool=%s status=%s",
            request.id,
            request.tool_name,
            request.status.value,
        )
        return approved

    def cancel_pending(self) -> None:
        if self.pending is not None:
            self.pending.status = ConfirmationStatus.CANCELLED
            log.info("confirmation cancelled id=%s", self.pending.id)
            self.pending = None

    @property
    def history(self) -> list[ConfirmationRequest]:
        return list(self._history)
