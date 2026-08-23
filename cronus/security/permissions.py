"""Permission policy.

The model is not a security boundary. Every tool call passes through here, and
the decision is derived from the tool's declared risk plus the user's explicit
configuration -- never from anything the model said.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..config import SecurityConfig
from ..logging_setup import get_logger
from ..tools.base import RiskLevel, Tool

log = get_logger("security.permissions")


class Decision(enum.Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def needs_confirmation(self) -> bool:
        return self.decision is Decision.CONFIRM


_OVERRIDE_DECISIONS = {
    "allow": Decision.ALLOW,
    "always": Decision.ALLOW,
    "confirm": Decision.CONFIRM,
    "ask": Decision.CONFIRM,
    "deny": Decision.DENY,
    "block": Decision.DENY,
    "blocked": Decision.DENY,
}

_BASE_DECISIONS = {
    RiskLevel.SAFE: Decision.ALLOW,
    RiskLevel.LOW: Decision.ALLOW,
    RiskLevel.CONFIRM: Decision.CONFIRM,
    # HIGH is off by default; the user must opt in per tool via configuration.
    RiskLevel.HIGH: Decision.DENY,
    RiskLevel.BLOCKED: Decision.DENY,
}


class PermissionPolicy:
    """Maps a tool to an allow / confirm / deny decision."""

    def __init__(self, config: SecurityConfig) -> None:
        self._overrides: dict[str, Decision] = {}
        for name, raw in config.permission_overrides.items():
            decision = _OVERRIDE_DECISIONS.get(raw)
            if decision is None:
                log.warning(
                    "ignoring unknown permission override %s=%s "
                    "(use allow, confirm, or block)",
                    name,
                    raw,
                )
                continue
            self._overrides[name] = decision

    def check(self, tool: Tool) -> PermissionResult:
        if tool.risk is RiskLevel.BLOCKED:
            # A blocked tool stays blocked; configuration cannot unblock it.
            return PermissionResult(
                Decision.DENY, f"{tool.name} is blocked and cannot be enabled."
            )

        override = self._overrides.get(tool.name)
        if override is not None:
            log.info(
                "permission override applied tool=%s decision=%s",
                tool.name,
                override.value,
            )
            return PermissionResult(
                override, f"{tool.name} follows a configured permission override."
            )

        decision = _BASE_DECISIONS[tool.risk]
        if decision is Decision.DENY and tool.risk is RiskLevel.HIGH:
            reason = (
                f"{tool.name} is high risk and is disabled. Enable it deliberately "
                f"with CRONUS_TOOL_PERMISSIONS={tool.name}=confirm."
            )
        elif decision is Decision.CONFIRM:
            reason = f"{tool.name} affects things outside this conversation."
        else:
            reason = f"{tool.name} is safe to run directly."
        return PermissionResult(decision, reason)
