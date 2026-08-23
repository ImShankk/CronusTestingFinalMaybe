"""Conversation state.

Two things live here: the durable turn history (user text and final assistant
text) and the scratch space for the turn currently being worked on (tool calls
and tool results). Keeping them apart matters -- provider-specific per-part
state attached to tool calls is only valid within the turn that produced it,
so old turns are replayed as plain text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..llm.base import Message
from ..logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..storage.db import Database

log = get_logger("core.conversation")

# How many completed turns survive a restart alongside the summary. Enough to
# resolve "the one we talked about", not an archive of everything ever said.
PERSISTED_TURNS = 6


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
        #: When the turns below were last spoken, if they came from an earlier
        #: run. Cleared once this session has added a turn of its own, so the
        #: "we were talking N hours ago" note appears exactly once.
        self.resumed_at: float | None = None
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
        # This session has now said something of its own; the conversation is
        # no longer "resumed", it is simply in progress.
        self.resumed_at = None
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

    def recent_text(self, turns: int = 2) -> str:
        """The last few exchanges as plain text, for widening memory recall.

        Only what was actually said -- no summary, no tool output -- because
        this is used as search terms, not as context for the model.
        """
        if turns <= 0:
            return ""
        recent = self.turns[-turns:]
        return " ".join(f"{turn.user} {turn.assistant}" for turn in recent).strip()

    def dropped_turns(self, char_budget: int) -> list[Turn]:
        """Turns that fall outside the budget and are candidates to summarise."""
        kept = len(self.history_messages(char_budget)) // 2
        return self.turns[: len(self.turns) - kept] if kept < len(self.turns) else []

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.resumed_at = None
        self.abandon_turn()


class ConversationStore:
    """Carries the tail of a conversation across restarts.

    Only the running summary and the last few completed turns are kept, in a
    single row that is overwritten each turn. This is continuity, not a
    transcript archive: ``/clear`` wipes it, and nothing here feeds long-term
    memory, which is still written only when the model explicitly asks.
    """

    def __init__(self, db: "Database") -> None:
        self.db = db

    def save(self, conversation: Conversation) -> None:
        turns = [
            {
                "user": turn.user,
                "assistant": turn.assistant,
                "at": turn.at,
                "tools_used": turn.tools_used,
            }
            for turn in conversation.turns[-PERSISTED_TURNS:]
        ]
        try:
            with self.db.write() as connection:
                connection.execute(
                    "INSERT INTO conversation_state(id, summary, turns, updated_at)"
                    " VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET"
                    " summary = excluded.summary, turns = excluded.turns,"
                    " updated_at = excluded.updated_at",
                    (conversation.summary, json.dumps(turns), time.time()),
                )
        except Exception:  # pragma: no cover - continuity is never critical
            log.warning("could not persist conversation state", exc_info=True)

    def restore(self, conversation: Conversation) -> bool:
        """Load the previous session into ``conversation``. True if anything was."""
        try:
            rows = self.db.query("SELECT * FROM conversation_state WHERE id = 1")
        except Exception:  # pragma: no cover - a missing table is not an error
            log.warning("could not read conversation state", exc_info=True)
            return False
        if not rows:
            return False

        row = rows[0]
        try:
            stored = json.loads(row["turns"]) or []
        except (TypeError, ValueError):
            log.warning("stored conversation turns were unreadable; ignoring them")
            stored = []

        turns = [
            Turn(
                user=str(entry.get("user", "")),
                assistant=str(entry.get("assistant", "")),
                at=float(entry.get("at") or 0.0),
                tools_used=list(entry.get("tools_used") or []),
            )
            for entry in stored
            if isinstance(entry, dict) and entry.get("user")
        ]
        summary = row["summary"] or ""
        if not turns and not summary:
            return False

        conversation.turns = turns
        conversation.summary = summary
        conversation.resumed_at = float(row["updated_at"])
        log.info("resumed conversation with %d turns", len(turns))
        return True

    def clear(self) -> None:
        try:
            with self.db.write() as connection:
                connection.execute("DELETE FROM conversation_state WHERE id = 1")
        except Exception:  # pragma: no cover - best effort
            log.warning("could not clear conversation state", exc_info=True)
