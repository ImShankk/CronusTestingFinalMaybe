"""The experience layer.

These pin behaviour a user can actually perceive: that a follow-up resolves
without repeating yourself, that talking over Cronus is a request rather than
just a stop button, that closing the terminal is not the same as being
forgotten, and that tool names stay out of the conversation.
"""

from __future__ import annotations

import time

import pytest

from cronus.config import VoiceConfig
from cronus.core import clock
from cronus.core.context import ContextBuilder
from cronus.core.conversation import Conversation, ConversationStore
from cronus.core.prompts import build_system_instruction
from cronus.interfaces.cli import _ACTIVITY, interpret_answer
from cronus.memory.store import MemoryStore
from cronus.storage.db import Database
from cronus.voice.base import ListenOutcome, SpeechToText, TextToSpeech
from cronus.voice.session import VoiceSession


# ======================================================================
# Confirmation answered in ordinary words
# ======================================================================
@pytest.mark.parametrize(
    "spoken",
    ["yes", "Yes.", "yeah", "yep, go ahead", "sure, send it", "ok do it",
     "go ahead", "please send that", "y", "Yes -- send it."],
)
def test_ordinary_agreement_is_understood(spoken: str) -> None:
    assert interpret_answer(spoken) is True


@pytest.mark.parametrize(
    "spoken",
    ["no", "nope", "cancel", "stop", "don't", "no, don't send it",
     "actually no, cancel that", "do not send it", "wait, no"],
)
def test_ordinary_refusal_is_understood(spoken: str) -> None:
    assert interpret_answer(spoken) is False


@pytest.mark.parametrize(
    "spoken",
    ["", "   ", "maybe", "what did you say", "hmm", "the weather in Edmonton"],
)
def test_anything_unclear_is_not_taken_as_consent(spoken: str) -> None:
    """Silence and ambiguity must never approve a consequential action."""
    assert interpret_answer(spoken) is None


def test_a_refusal_containing_an_agreeing_word_is_still_a_refusal() -> None:
    # The old exact-match table listed "send it" as a yes; negation must win.
    assert interpret_answer("no, don't send it") is False
    assert interpret_answer("cancel, do not do it") is False


# ======================================================================
# Tool names stay out of the conversation
# ======================================================================
def test_status_lines_never_name_a_tool() -> None:
    for tool_name, activity in _ACTIVITY.items():
        assert tool_name not in activity
        assert "_" not in activity, f"{activity!r} reads like an identifier"


def test_the_model_is_told_not_to_narrate_its_tools() -> None:
    instruction = build_system_instruction()
    assert "search_web" in instruction, "the example is what makes the rule land"
    assert "Never name your tools" in instruction


def test_the_model_is_told_to_resolve_references_rather_than_re_ask() -> None:
    instruction = build_system_instruction()
    assert "What about Saturday?" in instruction
    assert "one continuing conversation" in instruction


# ======================================================================
# Follow-ups and the calendar
# ======================================================================
def test_the_coming_week_is_named_so_a_weekday_needs_no_arithmetic() -> None:
    builder = ContextBuilder(Conversation(), timezone="America/Edmonton")
    situation = builder._situation()
    today = clock.now("America/Edmonton")
    for offset in range(1, 8):
        day = clock.describe_offset(offset, "America/Edmonton")
        assert day in situation, f"{day} is not resolvable from the context"
    assert today.strftime("%A") in situation


def test_a_configured_location_is_stated_and_guessing_is_forbidden() -> None:
    builder = ContextBuilder(Conversation(), location="Edmonton, Alberta, Canada")
    assert "Edmonton, Alberta, Canada" in builder._situation()


# ======================================================================
# Memory that answers a short follow-up
# ======================================================================
@pytest.fixture
def store() -> MemoryStore:
    db = Database(":memory:")
    yield MemoryStore(db)
    db.close()


def test_a_follow_up_recalls_what_the_conversation_established(
    store: MemoryStore,
) -> None:
    """The §20 example: "My name is Shank." ... "What do you call me?"

    The question carries almost no searchable words; the exchange it sits in
    does. Without conversational context this memory is never retrieved.
    """
    store.remember("The user's name is Shank", kind="person")

    alone = store.recall("What do you call me?")
    assert not any("Shank" in item.content for item in alone)

    with_context = store.recall(
        "What do you call me?", context="My name is Shank. Good to meet you."
    )
    assert any("Shank" in item.content for item in with_context)


def test_context_can_only_help_never_demote(store: MemoryStore) -> None:
    """Widening the query must not push a direct hit below the threshold."""
    store.remember("The user drives a blue Subaru", kind="fact")
    direct = store.recall("what colour is my Subaru")
    widened = store.recall(
        "what colour is my Subaru",
        context="unrelated chatter about groceries and the weather forecast",
    )
    assert [item.id for item in direct] == [item.id for item in widened][: len(direct)]


def test_recall_without_context_is_unchanged(store: MemoryStore) -> None:
    store.remember("The user prefers concise answers", kind="preference")
    # Identities, not whole items: recall bumps a usage counter as it goes.
    assert [item.id for item in store.recall("anything at all")] == [
        item.id for item in store.recall("anything at all", context="")
    ]


# ======================================================================
# Continuity across restarts
# ======================================================================
@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


def test_the_tail_of_a_conversation_survives_a_restart(db: Database) -> None:
    store = ConversationStore(db)
    before = Conversation()
    before.begin_turn("I'm working on Cronus")
    before.end_turn("Good luck with it.")
    before.summary = "The user is building an assistant called Cronus."
    store.save(before)

    after = Conversation()
    assert store.restore(after) is True
    assert after.turns[-1].user == "I'm working on Cronus"
    assert after.summary == "The user is building an assistant called Cronus."
    assert after.resumed_at is not None


def test_only_the_last_few_turns_are_kept(db: Database) -> None:
    store = ConversationStore(db)
    conversation = Conversation()
    for index in range(20):
        conversation.begin_turn(f"question {index}")
        conversation.end_turn(f"answer {index}")
    store.save(conversation)

    restored = Conversation()
    store.restore(restored)
    assert len(restored.turns) == 6, "this is continuity, not a transcript archive"
    assert restored.turns[-1].user == "question 19"


def test_clearing_the_conversation_clears_it_for_good(db: Database) -> None:
    store = ConversationStore(db)
    conversation = Conversation()
    conversation.begin_turn("something private")
    conversation.end_turn("noted")
    store.save(conversation)

    store.clear()
    assert store.restore(Conversation()) is False


def test_a_fresh_database_simply_has_nothing_to_resume(db: Database) -> None:
    assert ConversationStore(db).restore(Conversation()) is False


def test_unreadable_stored_turns_do_not_break_startup(db: Database) -> None:
    with db.write() as connection:
        connection.execute(
            "INSERT INTO conversation_state(id, summary, turns, updated_at)"
            " VALUES (1, 'a summary survived', 'not json at all', ?)",
            (time.time(),),
        )
    restored = Conversation()
    assert ConversationStore(db).restore(restored) is True
    assert restored.turns == []
    assert restored.summary == "a summary survived"


def test_resuming_is_announced_once_and_then_dropped(db: Database) -> None:
    store = ConversationStore(db)
    earlier = Conversation()
    earlier.begin_turn("remind me about Barcelona later")
    earlier.end_turn("Will do.")
    store.save(earlier)
    # Backdate it so the note has something to say.
    with db.write() as connection:
        connection.execute(
            "UPDATE conversation_state SET updated_at = ? WHERE id = 1",
            (time.time() - 3 * 3600,),
        )

    resumed = Conversation()
    store.restore(resumed)
    builder = ContextBuilder(resumed)

    first = builder.build(query="what were you saying about Barcelona?")
    assert "Picking up again" in first.system_instruction
    assert "3 hours ago" in first.system_instruction

    resumed.begin_turn("what were you saying about Barcelona?")
    resumed.end_turn("You asked me to remind you about it.")
    second = builder.build(query="and the flights?")
    assert "Picking up again" not in second.system_instruction


@pytest.mark.parametrize(
    "seconds,expected",
    [(30, "a moment ago"), (600, "10 minutes ago"), (3600, "about an hour ago"),
     (5 * 3600, "5 hours ago"), (30 * 3600, "yesterday"), (5 * 86400, "5 days ago")],
)
def test_elapsed_time_is_described_the_way_a_person_would(
    seconds: float, expected: str
) -> None:
    assert clock.describe_age(seconds) == expected


# ======================================================================
# Talking over Cronus is a request, not just a stop button
# ======================================================================
class _InterruptingSTT(SpeechToText):
    """Interrupts immediately, then hands back what the user actually said."""

    name = "interrupting"

    def __init__(self, utterances: list[str]) -> None:
        self.utterances = list(utterances)
        self.listen_calls = 0

    @property
    def available(self) -> bool:
        return True

    @property
    def supports_barge_in(self) -> bool:
        return True

    def listen(self, timeout: float | None = None) -> str:
        self.listen_calls += 1
        if self.utterances:
            self.last_outcome = ListenOutcome.HEARD
            return self.utterances.pop(0)
        self.last_outcome = ListenOutcome.NO_SPEECH
        return ""

    def wait_for_speech_start(self, stop, sensitivity: float = 2.5) -> bool:
        return not stop.wait(0.02)


class _SlowTTS(TextToSpeech):
    def __init__(self) -> None:
        self.stops = 0
        import threading

        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        return True

    def speak(self, text: str) -> None:
        self._stop.clear()
        self._stop.wait(5.0)

    def stop(self) -> None:
        self.stops += 1
        self._stop.set()


def test_interrupting_speech_keeps_what_was_said() -> None:
    """§13/§20: "Wait, stop. What about Sunday?" must not need repeating."""
    stt = _InterruptingSTT(["wait stop what about Sunday"])
    tts = _SlowTTS()
    session = VoiceSession(VoiceConfig(), stt=stt, tts=tts)

    assert session.speak("The weather tomorrow will be") is True
    assert tts.stops >= 1
    assert session.has_pending, "the interrupting words were thrown away"

    utterance = session.next_utterance()
    assert utterance.text == "wait stop what about Sunday"
    assert stt.listen_calls == 1, "the user was asked to say it a second time"


def test_a_consumed_interruption_is_not_replayed() -> None:
    stt = _InterruptingSTT(["what about Sunday", "and Monday"])
    session = VoiceSession(VoiceConfig(), stt=stt, tts=_SlowTTS())

    session.speak("a long reply")
    assert session.next_utterance().text == "what about Sunday"
    assert session.has_pending is False
    assert session.next_utterance().text == "and Monday"


def test_an_interruption_that_captured_nothing_falls_back_to_listening() -> None:
    """A cough stops the speech but must not become an empty turn."""
    stt = _InterruptingSTT([])          # nothing intelligible follows
    session = VoiceSession(VoiceConfig(), stt=stt, tts=_SlowTTS())

    session.speak("a long reply")
    assert session.has_pending is False
    stt.utterances.append("sorry, go on")
    assert session.next_utterance().text == "sorry, go on"


def test_closing_a_session_drops_anything_pending() -> None:
    stt = _InterruptingSTT(["something half-said"])
    session = VoiceSession(VoiceConfig(), stt=stt, tts=_SlowTTS())
    session.speak("a long reply")
    assert session.has_pending is True
    session.close()
    assert session.has_pending is False
