"""The assistant runtime: the multi-step agent loop.

One turn is: build context -> ask the model -> if it asked for tools, check
permissions, confirm when required, execute, feed results back -> repeat until
the model answers or the iteration limit is reached.

The runtime knows nothing about individual tools, and nothing about the CLI.
It talks to the registry, the permission policy, and an event emitter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..errors import CronusError, ProviderError
from ..llm.base import LLMProvider, Message, ToolCall
from ..logging_setup import get_logger
from ..memory.store import MemoryStore
from ..profile import UserProfile
from ..security.confirmation import ConfirmationManager
from ..security.permissions import Decision, PermissionPolicy
from ..tools.base import ToolContext, ToolResult
from ..tools.registry import ToolRegistry
from .context import ContextBuilder
from .conversation import Conversation
from .events import AssistantState, EventEmitter

log = get_logger("core.runtime")

# A tool result longer than this is trimmed before it goes back to the model.
_MAX_TOOL_RESULT_CHARS = 6_000


@dataclass
class TurnResult:
    """What one user turn produced."""

    text: str
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 1
    error: str | None = None
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.cancelled


class Assistant:
    """Coordinates the model, tools, memory, and safety checks."""

    def __init__(
        self,
        config: Config,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        memory: MemoryStore | None = None,
        profile: UserProfile | None = None,
        scheduler: Any | None = None,
        paths: Any | None = None,
        emitter: EventEmitter | None = None,
        confirmations: ConfirmationManager | None = None,
        voice_mode: bool = False,
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.profile = profile
        self.scheduler = scheduler
        self.emitter = emitter or EventEmitter()
        self.permissions = PermissionPolicy(config.security)
        self.confirmations = confirmations or ConfirmationManager(
            timeout=config.security.confirmation_timeout
        )
        self.voice_mode = voice_mode
        self.conversation = Conversation()
        self.context = ContextBuilder(
            self.conversation,
            profile=profile,
            memory=memory,
            char_budget=config.context_char_budget,
            max_memories=config.memory.max_recall,
        )
        self.tool_context = ToolContext(
            config=config,
            emitter=self.emitter,
            memory=memory,
            scheduler=scheduler,
            paths=paths,
        )
        self._cancelled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self, user_text: str) -> TurnResult:
        """Run one full turn and return the assistant's reply."""
        user_text = (user_text or "").strip()
        if not user_text:
            return TurnResult(text="")

        self._cancelled = False
        log.info("user input received (%d chars)", len(user_text))
        self.conversation.begin_turn(user_text)

        try:
            result = self._run_loop(user_text)
        except ProviderError as exc:
            log.warning("turn failed: %s", exc)
            self.conversation.abandon_turn()
            self.emitter.set_state(AssistantState.ERROR)
            self.emitter.error(exc.user_message)
            return TurnResult(text=exc.user_message, error=str(exc))
        except CronusError as exc:
            log.warning("turn failed: %s", exc)
            self.conversation.abandon_turn()
            self.emitter.set_state(AssistantState.ERROR)
            return TurnResult(text=exc.user_message, error=str(exc))
        except KeyboardInterrupt:
            self.conversation.abandon_turn()
            self.emitter.set_state(AssistantState.IDLE)
            return TurnResult(text="", cancelled=True)
        except Exception as exc:  # last-resort guard; users never see tracebacks
            log.exception("unexpected failure during turn")
            self.conversation.abandon_turn()
            self.emitter.set_state(AssistantState.ERROR)
            message = "Something went wrong on my end. The details are in my log."
            self.emitter.error(message)
            return TurnResult(text=message, error=str(exc))

        self.emitter.set_state(AssistantState.IDLE)
        return result

    def cancel(self) -> None:
        """Ask the current turn to stop at the next safe point."""
        self._cancelled = True
        self.confirmations.cancel_pending()

    def reset_conversation(self) -> None:
        self.conversation.clear()

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------
    def _run_loop(self, user_text: str) -> TurnResult:
        tools_used: list[str] = []
        schemas = self.registry.schemas()
        started = time.time()

        for iteration in range(1, self.config.max_tool_iterations + 1):
            if self._cancelled:
                self.conversation.abandon_turn()
                return TurnResult(text="", tools_used=tools_used, cancelled=True)

            self.emitter.set_state(AssistantState.THINKING)
            built = self.context.build(voice_mode=self.voice_mode, query=user_text)
            response = self.provider.generate(
                built.messages,
                system_instruction=built.system_instruction,
                tools=schemas,
            )

            if not response.wants_tools:
                text = response.text.strip()
                if not text:
                    text = "I didn't get an answer together for that. Try rephrasing?"
                self.conversation.end_turn(text)
                self.context.maybe_summarise(self.provider)
                self.emitter.response(text)
                log.info(
                    "turn complete iterations=%d tools=%s elapsed=%.1fs",
                    iteration,
                    tools_used or "-",
                    time.time() - started,
                )
                return TurnResult(
                    text=text, tools_used=tools_used, iterations=iteration
                )

            # The model asked for tools: record its turn, then run them.
            self.conversation.add_working(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    provider_state=response.provider_state,
                )
            )
            if response.text.strip():
                self.emitter.progress(response.text.strip())

            self.emitter.set_state(AssistantState.EXECUTING)
            for call in response.tool_calls:
                outcome = self._handle_call(call)
                self.conversation.add_working(
                    Message(
                        role="tool",
                        content=_trim(outcome.content),
                        tool_name=call.name,
                        tool_call_id=call.id,
                    )
                )
                if self.registry.has(call.name):
                    tools_used.append(call.name)

                if self._cancelled:
                    break

        # Out of iterations: ask for a final answer with tools withheld.
        log.warning("hit tool iteration limit (%d)", self.config.max_tool_iterations)
        return self._force_final_answer(tools_used)

    def _handle_call(self, call: ToolCall) -> ToolResult:
        """Permission check, confirmation, then execution -- in that order."""
        try:
            tool = self.registry.get(call.name)
        except CronusError:
            log.warning("model requested unknown tool %s", call.name)
            return ToolResult.failure(
                f"There is no tool named {call.name}. Available tools: "
                f"{', '.join(t.name for t in self.registry)}."
            )

        try:
            arguments = self.registry.validate(call.name, call.arguments)
        except CronusError as exc:
            return ToolResult.failure(f"Invalid arguments for {call.name}: {exc}")

        verdict = self.permissions.check(tool)
        log.info(
            "permission check tool=%s risk=%s decision=%s",
            tool.name,
            tool.risk.value,
            verdict.decision.value,
        )
        if verdict.decision is Decision.DENY:
            self.emitter.tool_end(tool.name, ok=False, summary="blocked")
            return ToolResult.failure(
                f"Blocked by the user's permission settings: {verdict.reason}"
            )

        if verdict.decision is Decision.CONFIRM:
            request = self.confirmations.request(
                tool.name,
                tool.describe_call(arguments),
                arguments,
                details=_preview_details(tool.name, arguments),
            )
            self.emitter.set_state(AssistantState.WAITING_FOR_CONFIRMATION)
            approved = self.confirmations.resolve(request)
            self.emitter.set_state(AssistantState.EXECUTING)
            if not approved:
                self.emitter.tool_end(tool.name, ok=False, summary="declined")
                return ToolResult.failure(
                    f"The user did not approve {tool.name}, so it was not run. "
                    "Acknowledge that briefly and ask what they'd like instead."
                )

        self.emitter.tool_start(tool.name, arguments)
        result = self.registry.execute(
            tool.name, arguments, self.tool_context, validated=True
        )
        self.emitter.tool_end(
            tool.name, ok=result.ok, summary=result.display or ""
        )
        return result

    def _force_final_answer(self, tools_used: list[str]) -> TurnResult:
        """Wrap up with what we have when the loop runs long."""
        self.conversation.add_working(
            Message(
                role="user",
                content=(
                    "Stop calling tools now and answer with what you already have. "
                    "If the task is unfinished, say what is missing."
                ),
            )
        )
        built = self.context.build(voice_mode=self.voice_mode)
        try:
            response = self.provider.generate(
                built.messages, system_instruction=built.system_instruction
            )
            text = response.text.strip()
        except ProviderError as exc:
            text = exc.user_message

        if not text:
            text = "I worked through several steps but couldn't finish that one."
        self.conversation.end_turn(text)
        self.emitter.response(text)
        return TurnResult(
            text=text,
            tools_used=tools_used,
            iterations=self.config.max_tool_iterations,
            error="iteration_limit",
        )


def _trim(content: str) -> str:
    if len(content) <= _MAX_TOOL_RESULT_CHARS:
        return content
    return (
        content[:_MAX_TOOL_RESULT_CHARS]
        + f"\n[...truncated, {len(content) - _MAX_TOOL_RESULT_CHARS} more characters]"
    )


def _preview_details(tool_name: str, arguments: dict[str, Any]) -> dict[str, str]:
    """Show the fields a person needs to judge a pending action."""
    interesting = ("to", "to_email", "recipient", "subject", "path", "destination", "url")
    details = {
        key: str(value)
        for key, value in arguments.items()
        if key in interesting and value
    }
    body = arguments.get("body") or arguments.get("message")
    if body:
        details["body"] = str(body)
    return details
