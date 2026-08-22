"""Tools that let the assistant use its own memory.

The model proposes what to remember; this layer decides whether it is
storable, normalises it, and owns persistence. Deleting a memory needs
confirmation -- forgetting something the user asked you to keep is not a
recoverable mistake.
"""

from __future__ import annotations

from typing import Any

from ..logging_setup import get_logger
from ..memory.store import VALID_KINDS
from .base import RiskLevel, Tool, ToolContext, ToolResult, object_schema

log = get_logger("tools.memory")


def remember_this(
    content: str,
    kind: str = "fact",
    importance: int = 1,
    context: ToolContext | None = None,
) -> ToolResult:
    if context is None or context.memory is None:
        return ToolResult.failure("Memory is not available right now.")
    try:
        item = context.memory.remember(
            content, kind=kind, importance=importance, source="assistant"
        )
    except Exception as exc:
        return ToolResult.failure(f"I couldn't store that: {exc}")
    return ToolResult(
        content=f"Stored as memory {item.id}: {item.content}",
        display="remembered",
        data={"id": item.id},
    )


def recall_memories(query: str, limit: int = 5, context: ToolContext | None = None) -> ToolResult:
    if context is None or context.memory is None:
        return ToolResult.failure("Memory is not available right now.")
    items = context.memory.recall(query, limit=max(1, min(int(limit), 20)))
    if not items:
        return ToolResult(
            content=f"Nothing stored about {query!r}.", display="nothing found"
        )
    lines = "\n".join(item.summary() for item in items)
    return ToolResult(
        content=f"Stored memories about {query!r}:\n{lines}",
        display=f"{len(items)} memories",
    )


def list_memories(limit: int = 20, context: ToolContext | None = None) -> ToolResult:
    if context is None or context.memory is None:
        return ToolResult.failure("Memory is not available right now.")
    items = context.memory.all(limit=max(1, min(int(limit), 100)))
    if not items:
        return ToolResult(content="Nothing is stored yet.", display="empty")
    lines = "\n".join(item.summary() for item in items)
    return ToolResult(
        content=f"{len(items)} stored memories:\n{lines}",
        display=f"{len(items)} memories",
    )


def forget_memory(
    memory_id: int | None = None,
    description: str | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    if context is None or context.memory is None:
        return ToolResult.failure("Memory is not available right now.")
    if memory_id is not None:
        if context.memory.forget(int(memory_id)):
            return ToolResult(content=f"Memory {memory_id} deleted.", display="forgotten")
        return ToolResult.failure(f"There is no memory numbered {memory_id}.")
    if description:
        removed = context.memory.forget_matching(description)
        if removed:
            return ToolResult(
                content=f"Deleted {removed} memory item(s) matching {description!r}.",
                display="forgotten",
            )
        return ToolResult.failure(f"Nothing stored clearly matches {description!r}.")
    return ToolResult.failure("Give either a memory_id or a description to forget.")


def set_preference(key: str, value: str, context: ToolContext | None = None) -> ToolResult:
    """Update a profile field the user has stated directly."""
    if context is None:
        return ToolResult.failure("Profile is not available right now.")
    profile = context.session.get("profile")
    if profile is None:
        return ToolResult.failure("Profile is not available right now.")
    if profile.set(key, value):
        return ToolResult(content=f"Profile {key} set to {value!r}.", display="updated")
    from ..profile import KNOWN_KEYS

    return ToolResult.failure(
        f"{key!r} is not a profile field. Valid fields: {', '.join(sorted(KNOWN_KEYS))}."
    )


def _forget_preview(arguments: dict[str, Any]) -> str:
    target = arguments.get("description") or f"memory {arguments.get('memory_id')}"
    return f"Permanently forget {target}?"


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="remember_this",
            description=(
                "Save a durable fact or preference about the user. Use it when "
                "they ask you to remember something, or state a lasting "
                "preference. Do not use it for passing chat or task details."
            ),
            parameters=object_schema(
                {
                    "content": {
                        "type": "string",
                        "description": "The fact, written as a standalone statement.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "What sort of memory this is.",
                        "enum": list(VALID_KINDS),
                        "default": "fact",
                    },
                    "importance": {
                        "type": "integer",
                        "description": "1 for ordinary, up to 5 for things that shape every answer.",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 1,
                    },
                },
                required=["content"],
            ),
            handler=remember_this,
            risk=RiskLevel.LOW,
            category="memory",
        ),
        Tool(
            name="recall_memories",
            description="Look up what you have stored about a topic, person, or preference.",
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "What to look for."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                required=["query"],
            ),
            handler=recall_memories,
            risk=RiskLevel.SAFE,
            category="memory",
        ),
        Tool(
            name="list_memories",
            description="List everything you have stored, for when the user asks what you remember.",
            parameters=object_schema(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}}
            ),
            handler=list_memories,
            risk=RiskLevel.SAFE,
            category="memory",
        ),
        Tool(
            name="forget_memory",
            description="Delete a stored memory, by its number or by describing it.",
            parameters=object_schema(
                {
                    "memory_id": {"type": "integer", "description": "The number shown in a listing."},
                    "description": {
                        "type": "string",
                        "description": "Describe the memory instead, if the number is unknown.",
                    },
                }
            ),
            handler=forget_memory,
            risk=RiskLevel.CONFIRM,
            category="memory",
            preview=_forget_preview,
        ),
        Tool(
            name="set_preference",
            description=(
                "Record a profile setting the user stated: name, timezone, "
                "response_style, default_city, or email_signature."
            ),
            parameters=object_schema(
                {
                    "key": {"type": "string", "description": "Which profile field to set."},
                    "value": {"type": "string", "description": "The value to store."},
                },
                required=["key", "value"],
            ),
            handler=set_preference,
            risk=RiskLevel.LOW,
            category="memory",
        ),
    ]
