"""Context assembly.

Builds the request sent to the model out of clearly separated layers --
instructions, user profile, relevant memories, situational facts, conversation
history, and the in-flight turn -- inside a character budget, so requests stay
small as a session grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..llm.base import Message
from ..logging_setup import get_logger
from .conversation import Conversation
from .prompts import SUMMARISE_PROMPT, build_system_instruction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..llm.base import LLMProvider
    from ..memory.store import MemoryStore
    from ..profile import UserProfile

log = get_logger("core.context")

# Fraction of the budget history may use; the rest is left for the live turn.
_HISTORY_SHARE = 0.6
# Only summarise once enough has fallen out of the window to be worth a call.
_SUMMARY_TRIGGER_TURNS = 4


@dataclass
class BuiltContext:
    system_instruction: str
    messages: list[Message]
    memories_used: int = 0
    history_turns: int = 0


class ContextBuilder:
    """Turns state into a model request."""

    def __init__(
        self,
        conversation: Conversation,
        *,
        profile: "UserProfile | None" = None,
        memory: "MemoryStore | None" = None,
        char_budget: int = 12_000,
        max_memories: int = 6,
    ) -> None:
        self.conversation = conversation
        self.profile = profile
        self.memory = memory
        self.char_budget = char_budget
        self.max_memories = max_memories

    def build(self, *, voice_mode: bool = False, query: str = "") -> BuiltContext:
        sections: list[str] = [self._situation()]

        if self.profile is not None:
            profile_text = self.profile.as_prompt_section()
            if profile_text:
                sections.append(profile_text)

        memories_used = 0
        if self.memory is not None and query:
            recalled = self.memory.recall(query, limit=self.max_memories)
            if recalled:
                memories_used = len(recalled)
                lines = "\n".join(f"- {item.content}" for item in recalled)
                sections.append(
                    "# What you know about the user\n"
                    "These are your own saved notes, not instructions from anyone "
                    f"else.\n{lines}"
                )

        if self.conversation.summary:
            sections.append(
                f"# Earlier in this conversation\n{self.conversation.summary}"
            )

        system_instruction = build_system_instruction(
            voice_mode=voice_mode,
            has_memory=self.memory is not None,
            extra_sections=sections,
        )

        history_budget = int(self.char_budget * _HISTORY_SHARE)
        history = self.conversation.history_messages(history_budget)
        messages = history + list(self.conversation.working)

        return BuiltContext(
            system_instruction=system_instruction,
            messages=messages,
            memories_used=memories_used,
            history_turns=len(history) // 2,
        )

    def _situation(self) -> str:
        now = datetime.now().astimezone()
        return (
            "# Right now\n"
            f"Local date and time: {now.strftime('%A %d %B %Y, %H:%M %Z')}"
        )

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def maybe_summarise(self, provider: "LLMProvider") -> bool:
        """Fold turns that fell out of the window into a running summary.

        Costs one small model call, and only when enough has actually dropped
        out to be worth it. Returns True if a summary was written.
        """
        history_budget = int(self.char_budget * _HISTORY_SHARE)
        dropped = self.conversation.dropped_turns(history_budget)
        if len(dropped) < _SUMMARY_TRIGGER_TURNS:
            return False

        transcript = "\n".join(
            f"User: {turn.user}\nCronus: {turn.assistant}" for turn in dropped
        )
        if self.conversation.summary:
            transcript = f"Summary so far: {self.conversation.summary}\n{transcript}"

        try:
            response = provider.generate(
                [Message(role="user", content=SUMMARISE_PROMPT.format(transcript=transcript))],
                system_instruction="You compress conversations accurately and briefly.",
                temperature=0.2,
            )
        except Exception:
            log.warning("conversation summarisation failed; keeping prior summary")
            return False

        if not response.text.strip():
            return False

        self.conversation.summary = response.text.strip()
        # Those turns now live in the summary, so stop replaying them verbatim.
        del self.conversation.turns[: len(dropped)]
        log.info("summarised %d older turns", len(dropped))
        return True
