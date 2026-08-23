"""Provider-neutral LLM types and interface.

Cronus core speaks only these types. A provider is responsible for translating
them to and from its own SDK, so swapping Gemini for another model does not
touch the runtime, tools, or memory.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    ``provider_state`` carries opaque backend data that must be echoed back
    verbatim on the next request (Gemini 3, for instance, rejects a follow-up
    whose function-call parts have lost their thought signature). Core code
    never reads it; only the provider that produced it does.
    """

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One entry of conversation history.

    ``tool`` messages carry ``tool_name``/``tool_call_id`` and put the tool's
    output in ``content``. ``assistant`` messages may carry ``tool_calls``.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A single model turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolSchema:
    """The subset of a tool's definition the model is allowed to see."""

    name: str
    description: str
    parameters: dict[str, Any]


class LLMProvider(ABC):
    """Minimal contract a language-model backend must satisfy."""

    name: str = "llm"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        system_instruction: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Produce the next assistant turn.

        Implementations must raise :class:`cronus.errors.ProviderError` for any
        backend failure rather than leaking SDK-specific exceptions.
        """
