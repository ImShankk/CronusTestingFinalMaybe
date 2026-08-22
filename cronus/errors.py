"""Exception hierarchy for Cronus.

Everything user-visible is derived from ``CronusError`` and carries a
``user_message`` that is safe to speak or print.  Raw exception text is for
logs only.
"""

from __future__ import annotations


class CronusError(Exception):
    """Base class for all Cronus errors."""

    default_user_message = "Something went wrong on my end."

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or self.default_user_message


class ConfigError(CronusError):
    """Configuration is missing or invalid."""

    default_user_message = "My configuration is incomplete."


class ProviderError(CronusError):
    """The LLM provider failed (network, auth, quota, malformed reply)."""

    default_user_message = "I couldn't reach my language model just now."


class ToolError(CronusError):
    """A tool failed in a way the model should hear about and can react to."""

    default_user_message = "That tool didn't work."


class ToolNotFound(ToolError):
    default_user_message = "I don't have a tool for that."


class ToolValidationError(ToolError):
    """The model supplied arguments that don't match the tool schema."""

    default_user_message = "I called that tool incorrectly."


class ToolTimeout(ToolError):
    default_user_message = "That took too long, so I stopped it."


class PermissionDenied(CronusError):
    """Application policy blocks this action. The model cannot override this."""

    default_user_message = "I'm not allowed to do that."


class PathNotAllowed(PermissionDenied):
    default_user_message = "That location is outside the folders I can touch."


class ConfirmationDeclined(CronusError):
    default_user_message = "Okay, I won't do that."


class StorageError(CronusError):
    default_user_message = "I had trouble reading my own memory."
