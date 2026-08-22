"""The tool registry: the only path from a model request to real execution."""

from __future__ import annotations

import asyncio
import inspect
import re
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Iterable, Iterator

from ..errors import CronusError, ToolNotFound, ToolValidationError
from ..llm.base import ToolSchema
from ..logging_setup import get_logger
from .base import Tool, ToolContext, ToolResult
from .schema import validate_arguments

log = get_logger("tools.registry")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")


class ToolRegistry:
    """Holds tools, validates calls against their schemas, and runs them.

    Handlers may be sync or async; both are executed on a worker thread so a
    hung tool cannot block the assistant past its timeout. A timed-out thread
    is abandoned rather than killed (Python cannot safely kill threads), so
    handlers should set their own network timeouts too.
    """

    def __init__(self, default_timeout: float = 30.0, max_workers: int = 4) -> None:
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cronus-tool"
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if not _NAME_RE.match(tool.name):
            raise CronusError(
                f"invalid tool name {tool.name!r}: use lower_snake_case",
            )
        if tool.name in self._tools and not replace:
            raise CronusError(f"tool {tool.name!r} is already registered")
        if tool.parameters.get("type") != "object":
            raise CronusError(f"tool {tool.name!r} must take an object parameter schema")
        self._tools[tool.name] = tool
        log.debug("registered tool %s (risk=%s)", tool.name, tool.risk.value)
        return tool

    def register_all(self, tools: Iterable[Tool], *, replace: bool = False) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None or not tool.enabled:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFound(
                f"unknown tool {name!r} (known: {known})",
                user_message=f"I don't have a tool called {name}.",
            )
        return tool

    def has(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.enabled

    def __iter__(self) -> Iterator[Tool]:
        return iter(sorted(self._tools.values(), key=lambda t: t.name))

    def __len__(self) -> int:
        return sum(1 for tool in self._tools.values() if tool.enabled)

    def schemas(self) -> list[ToolSchema]:
        """Model-facing declarations for every enabled tool."""
        return [tool.schema() for tool in self if tool.enabled]

    def by_category(self) -> dict[str, list[Tool]]:
        grouped: dict[str, list[Tool]] = {}
        for tool in self:
            if tool.enabled:
                grouped.setdefault(tool.category, []).append(tool)
        return grouped

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check arguments before anything is executed or confirmed."""
        tool = self.get(name)
        if not isinstance(arguments, dict):
            raise ToolValidationError(f"{name}: arguments must be an object")
        return validate_arguments(arguments, tool.parameters)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        validated: bool = False,
    ) -> ToolResult:
        """Run a tool and always come back with a :class:`ToolResult`.

        Failures are converted into unsuccessful results so the agent loop can
        show them to the model and let it recover, rather than crashing a turn.
        """
        tool = self.get(name)
        try:
            payload = arguments if validated else self.validate(name, arguments)
        except ToolValidationError as exc:
            log.warning("tool %s rejected arguments: %s", name, exc)
            return ToolResult.failure(f"Invalid arguments for {name}: {exc}")

        timeout = tool.timeout or self.default_timeout
        log.info("tool execute name=%s args=%s", name, _safe_args(payload))

        future: Future[Any] = self._pool.submit(_invoke, tool.handler, payload, context)
        try:
            raw = future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            log.error("tool %s timed out after %.1fs", name, timeout)
            return ToolResult.failure(
                f"{name} timed out after {timeout:.0f} seconds and was stopped."
            )
        except CronusError as exc:
            log.warning("tool %s failed: %s", name, exc)
            return ToolResult.failure(f"{name} failed: {exc.user_message}")
        except Exception as exc:
            log.exception("tool %s raised", name)
            return ToolResult.failure(f"{name} failed: {type(exc).__name__}: {exc}")

        return _normalise(raw)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _invoke(handler: Any, arguments: dict[str, Any], context: ToolContext) -> Any:
    """Call a handler, awaiting it on a private loop when it is async."""
    accepts_context = "context" in inspect.signature(handler).parameters
    result = handler(context=context, **arguments) if accepts_context else handler(**arguments)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _normalise(raw: Any) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if raw is None:
        return ToolResult(content="Done.")
    return ToolResult(content=str(raw))


def _safe_args(arguments: dict[str, Any]) -> str:
    """Render arguments compactly for the log, truncating long values."""
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > 60:
            text = f"{text[:57]}..."
        parts.append(f"{key}={text}")
    return " ".join(parts)
