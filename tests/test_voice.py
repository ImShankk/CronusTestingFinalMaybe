"""Voice abstractions, tested without a microphone or speakers."""

from __future__ import annotations

import pytest

from cronus.config import VoiceConfig
from cronus.voice.base import SpeechToText
from cronus.voice.tts import clean_for_speech
from cronus.voice.wake import KeywordWakeWord, PushToTalk, build_wake_word


class ScriptedSTT(SpeechToText):
    """A fake microphone that returns a fixed list of utterances."""

    name = "scripted"

    def __init__(self, utterances: list[str], available: bool = True) -> None:
        self.utterances = list(utterances)
        self._available = available
        self.listens = 0

    @property
    def available(self) -> bool:
        return self._available

    def listen(self, timeout: float | None = None) -> str:
        self.listens += 1
        return self.utterances.pop(0) if self.utterances else ""


# ----------------------------------------------------------------------
# Speech cleanup
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected_absent",
    [
        ("**bold** text", "*"),
        ("# Heading", "#"),
        ("- a bullet", "- "),
        ("`code`", "`"),
        ("See https://example.com/x for more", "https"),
        ("```\nprint(1)\n```", "print"),
    ],
)
def test_markup_is_stripped_before_speaking(raw, expected_absent):
    assert expected_absent not in clean_for_speech(raw)


def test_links_become_words():
    assert "a link" in clean_for_speech("Read https://example.com now")


def test_link_text_is_kept_from_markdown_links():
    assert clean_for_speech("[the docs](https://example.com)") == "the docs"


def test_whitespace_is_collapsed():
    assert clean_for_speech("one\n\n\n   two") == "one two"


# ----------------------------------------------------------------------
# Wake word
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "heard,expected",
    [
        ("hey cronus what is the weather", True),
        ("hey chronos what is the weather", True),   # STT mishears the name
        ("hey chronus", True),
        ("what is the weather", False),
        ("okay computer", False),
        ("", False),
    ],
)
def test_wake_phrase_matching_tolerates_mishearing(heard, expected):
    matched, _ = KeywordWakeWord.match(heard, "hey cronus")
    assert matched is expected


def test_the_rest_of_the_sentence_is_carried_through():
    _, remainder = KeywordWakeWord.match("hey cronus what is the weather", "hey cronus")
    assert remainder == "what is the weather"


def test_a_bare_wake_word_carries_nothing():
    _, remainder = KeywordWakeWord.match("hey cronus", "hey cronus")
    assert remainder == ""


def test_detection_ignores_speech_not_addressed_to_cronus():
    stt = ScriptedSTT(["some unrelated chatter", "hey cronus set a timer"])
    detector = KeywordWakeWord(stt, VoiceConfig(wake_word="hey cronus"))
    assert detector.wait_for_wake() is True
    assert detector.carried_text == "set a timer"
    assert stt.listens == 2


def test_carried_text_is_consumed_once():
    stt = ScriptedSTT(["hey cronus hello there"])
    detector = KeywordWakeWord(stt, VoiceConfig(wake_word="hey cronus"))
    detector.wait_for_wake()
    assert detector.carried_text == "hello there"
    assert detector.carried_text == ""


def test_detection_can_be_cancelled():
    detector = KeywordWakeWord(ScriptedSTT([]), VoiceConfig())
    detector.cancel()
    assert detector.wait_for_wake() is False


def test_a_cancel_survives_until_reset():
    """cancel() is called from another thread; wait_for_wake must not clear it."""
    detector = KeywordWakeWord(ScriptedSTT(["hey cronus hello"]), VoiceConfig())
    detector.cancel()
    assert detector.wait_for_wake() is False
    detector.reset()
    assert detector.wait_for_wake() is True


def test_a_dead_microphone_ends_the_wait_instead_of_spinning():
    detector = KeywordWakeWord(ScriptedSTT([], available=False), VoiceConfig())
    assert detector.wait_for_wake() is False


# ----------------------------------------------------------------------
# Provider selection
# ----------------------------------------------------------------------
def test_push_to_talk_is_the_fallback_when_the_mic_is_missing():
    detector = build_wake_word(None, VoiceConfig(wake_word_enabled=True))
    assert isinstance(detector, PushToTalk)


def test_wake_word_is_used_when_speech_input_exists():
    """The builder answers "is it possible"; the voice mode answers "is it wanted"."""
    detector = build_wake_word(ScriptedSTT([]), VoiceConfig())
    assert isinstance(detector, KeywordWakeWord)


def test_an_unusable_microphone_falls_back_to_press_enter():
    detector = build_wake_word(ScriptedSTT([], available=False), VoiceConfig())
    assert isinstance(detector, PushToTalk)


def test_tts_selection_reports_nothing_rather_than_faking_it(tmp_path, monkeypatch):
    """With no Piper and no SAPI, speech must be absent, not pretended."""
    from cronus.voice import tts

    monkeypatch.setattr(tts.PiperTTS, "available", property(lambda self: False))
    monkeypatch.setattr(tts.SapiTTS, "available", property(lambda self: False))
    assert tts.build_tts(VoiceConfig()) is None


def test_tts_falls_back_when_the_preferred_provider_is_missing(monkeypatch):
    from cronus.voice import tts

    monkeypatch.setattr(tts.PiperTTS, "available", property(lambda self: False))
    monkeypatch.setattr(tts.SapiTTS, "available", property(lambda self: True))
    provider = tts.build_tts(VoiceConfig(tts_provider="piper"))
    assert provider is not None and provider.name == "sapi"


def test_stt_reports_unavailable_rather_than_raising(monkeypatch):
    from cronus.voice import stt

    monkeypatch.setattr(
        stt.SpeechRecognitionSTT, "_setup", lambda self: setattr(self, "_error", "no mic")
    )
    assert stt.build_stt(VoiceConfig()) is None
