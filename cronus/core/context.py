"""Context assembly.

Builds the request sent to the model out of clearly separated layers --
instructions, user profile, relevant memories, situational facts, conversation
history, and the in-flight turn -- inside a character budget, so requests stay
small as a session grows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..llm.base import Message
from ..logging_setup import get_logger
from . import clock
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
        account_email: str | None = None,
        timezone: str | None = None,
        location: str | None = None,
    ) -> None:
        self.conversation = conversation
        self.profile = profile
        self.memory = memory
        # The address Cronus sends from. Not a secret -- it is the user's own
        # address -- and the model needs it to resolve "me" or "myself".
        self.account_email = account_email
        self.timezone = timezone
        self.location = location
        self.char_budget = char_budget
        self.max_memories = max_memories
        # Recall is a full-text query plus a usage-counter write. The agent
        # loop rebuilds context once per iteration with the same query, so the
        # result is computed once per turn instead of once per model call.
        self._recalled: list | None = None

    def begin_turn(self) -> None:
        """Drop per-turn caches so new memories are picked up next turn."""
        self._recalled = None

    def build(self, *, voice_mode: bool = False, query: str = "") -> BuiltContext:
        sections: list[str] = [self._situation()]

        if self.profile is not None:
            profile_text = self.profile.as_prompt_section()
            if profile_text:
                sections.append(profile_text)

        memories_used = 0
        if self.memory is not None and query:
            if self._recalled is None:
                self._recalled = self.memory.recall(
                    query,
                    limit=self.max_memories,
                    # A follow-up carries almost no searchable words of its
                    # own. The turns around it do, so they widen the search.
                    context=self.conversation.recent_text(),
                )
            recalled = self._recalled
            if recalled:
                memories_used = len(recalled)
                lines = "\n".join(f"- {item.content}" for item in recalled)
                sections.append(
                    "# What you know about the user\n"
                    "These are your own saved notes, not instructions from anyone "
                    f"else.\n{lines}"
                )

        if self.account_email:
            sections.append(
                "# The user's email account\n"
                f"Mail is sent from {self.account_email}, which is the user's own "
                'address. When they say "me", "myself", or "yourself", that is '
                "the address to use.\n"
                "Never invent an email address. If you do not know a recipient's "
                "address, ask for it rather than guessing one."
            )

        resumed = self._resumed_note()
        if resumed:
            sections.append(resumed)
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
        """Time and place.

        Both matter: without a stated location the model infers one from the
        timezone, which is how "what's the weather" once became Denver.
        """
        # Every named day of the coming week is spelled out. Given only today's
        # date the model has to do calendar arithmetic to answer "what about
        # Saturday?", and it gets it wrong often enough to matter.
        upcoming = ", ".join(
            clock.describe_offset(day, self.timezone) for day in range(2, 8)
        )
        lines = [
            "# Right now",
            f"Local date and time: {clock.describe(timezone_name=self.timezone)}",
            f"Tomorrow is {clock.describe_offset(1, self.timezone)}.",
            f"Then: {upcoming}.",
        ]
        where = self._where()
        if where:
            lines.append(
                f"The user is in {where}. Use this for weather and any other "
                "local question unless they name somewhere else."
            )
        else:
            lines.append(
                "The user's location is not configured. Never guess a city: if a "
                "request needs one, ask them which place they mean."
            )
        return "\n".join(lines)

    def _resumed_note(self) -> str:
        """Tell the model the history below is from a previous run.

        Without this the restored turns read as if they just happened, and
        Cronus picks up mid-thought as though no time passed. Shown once --
        the flag clears as soon as this session records a turn of its own.
        """
        resumed_at = self.conversation.resumed_at
        if not resumed_at:
            return ""
        ago = clock.describe_age(time.time() - resumed_at)
        return (
            "# Picking up again\n"
            f"The exchanges below are from an earlier session, {ago}. You are the "
            "same assistant continuing with the same person, so references back "
            "to them still work -- but do not greet them as though the last "
            "thing said was a second ago, and do not assume anything that was "
            "in progress is still current. Do not mention this note."
        )

    def _where(self) -> str | None:
        """What the user has told us, preferred over what was configured."""
        if self.profile is not None:
            stated = self.profile.get("location") or self.profile.get("default_city")
            if stated:
                return stated
        return self.location

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
