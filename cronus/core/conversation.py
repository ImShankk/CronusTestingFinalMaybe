"""Conversation state.

Two things live here: the durable turn history (user text and final assistant
text) and the scratch space for the turn currently being worked on (tool calls
and tool results). Keeping them apart matters -- provider-specific per-part
state attached to tool calls is only valid within the turn that produced it,
so old turns are replayed as plain text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..llm.base import Message


@dataclass
class Turn:
    """One completed exchange."""

    user: str
    assistant: str
    at: float = field(default_factory=time.time)
    tools_used: list[str] = field(default_factory=list)

    def char_size(self) -> int:
        return len(self.user) + len(self.assistant)


class Conversation:
    """Recent turns plus the in-flight working set."""

    def __init__(self, max_turns: int = 40) -> None:
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self.summary: str = ""
        self.working: list[Message] = []
        self._current_user: str = ""
        self._tools_used: list[str] = []

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------
    def begin_turn(self, user_text: str) -> None:
        self._current_user = user_text
        self._tools_used = []
        self.working = [Message(role="user", content=user_text)]

    def add_working(self, message: Message) -> None:
        self.working.append(message)
        if message.role == "assistant":
            self._tools_used.extend(call.name for call in message.tool_calls)

    def end_turn(self, assistant_text: str) -> Turn:
        turn = Turn(
            user=self._current_user,
            assistant=assistant_text,
            tools_used=list(dict.fromkeys(self._tools_used)),
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            del self.turns[: len(self.turns) - self.max_turns]
        self.working = []
        self._current_user = ""
        self._tools_used = []
        return turn

    def abandon_turn(self) -> None:
        """Drop in-flight state without recording a turn (cancel/failure)."""
        self.working = []
        self._current_user = ""
        self._tools_used = []

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def history_messages(self, char_budget: int) -> list[Message]:
        """Completed turns as plain messages, newest-first within a budget."""
        messages: list[Message] = []
        used = 0
        for turn in reversed(self.turns):
            size = turn.char_size()
            if used + size > char_budget and messages:
                break
            used += size
            messages.append(Message(role="assistant", content=turn.assistant))
            messages.append(Message(role="user", content=turn.user))
        messages.reverse()
        return messages

    def dropped_turns(self, char_budget: int) -> list[Turn]:
        """Turns that fall outside the budget and are candidates to summarise."""
        kept = len(self.history_messages(char_budget)) // 2
        return self.turns[: len(self.turns) - kept] if kept < len(self.turns) else []

    def last_exchange(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.abandon_turn()
