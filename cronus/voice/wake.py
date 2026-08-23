"""Wake-word activation.

The detector interface is provider-neutral. What ships is an honest
implementation built on the STT provider already in use: Cronus listens in
short bursts and activates when the transcript starts with the wake phrase.

Known limitation: this is not always-on, low-power keyword spotting like
Porcupine or openWakeWord. It sends short audio clips to the same speech
service used for dictation, so it costs a round trip per burst and needs a
network connection. It is real and it works; it is not free. A dedicated
detector can be added later by implementing this same interface.
"""

from __future__ import annotations

import difflib

from ..config import VoiceConfig
from ..logging_setup import get_logger
from .base import SpeechToText, WakeWordDetector

log = get_logger("voice.wake")

_BURST_SECONDS = 4.0
_SIMILARITY = 0.75


class KeywordWakeWord(WakeWordDetector):
    """Activates when a short capture begins with the configured phrase."""

    name = "keyword"

    def __init__(self, stt: SpeechToText, config: VoiceConfig) -> None:
        self.stt = stt
        self.phrase = (config.wake_word or "hey cronus").lower().strip()
        self._carried = ""
        self._cancelled = False

    @property
    def available(self) -> bool:
        return self.stt.available

    @property
    def carried_text(self) -> str:
        carried, self._carried = self._carried, ""
        return carried

    def cancel(self) -> None:
        """Stop listening. Callable from another thread; cleared by reset()."""
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False

    def wait_for_wake(self) -> bool:
        while not self._cancelled:
            if not self.stt.available:
                # The microphone went away; don't spin on an empty stream.
                log.warning("speech input became unavailable while listening")
                return False
            heard = self.stt.listen(timeout=_BURST_SECONDS)
            if not heard:
                continue
            matched, remainder = self.match(heard, self.phrase)
            if matched:
                log.info("wake word detected")
                self._carried = remainder
                return True
            log.debug("ignored speech that was not addressed to me")
        return False

    @staticmethod
    def match(heard: str, phrase: str) -> tuple[bool, str]:
        """Check whether ``heard`` opens with the wake phrase.

        Speech recognition mangles unusual names ("hey chronos", "hey chronus"),
        so the comparison is fuzzy over the same number of words.
        """
        words = heard.lower().split()
        phrase_words = phrase.split()
        if len(words) < len(phrase_words):
            return False, ""
        candidate = " ".join(words[: len(phrase_words)])
        ratio = difflib.SequenceMatcher(None, candidate, phrase).ratio()
        if ratio < _SIMILARITY:
            return False, ""
        remainder = " ".join(heard.split()[len(phrase_words):]).strip(" ,.!?")
        return True, remainder


class PushToTalk(WakeWordDetector):
    """The fallback when hands-free listening isn't wanted: press Enter."""

    name = "push_to_talk"

    def __init__(self, prompt: str = "Press Enter to speak (or type 'quit'): ") -> None:
        self.prompt = prompt

    @property
    def available(self) -> bool:
        return True

    def wait_for_wake(self) -> bool:
        try:
            answer = input(self.prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer not in ("quit", "exit", "q")


def build_wake_word(
    stt: SpeechToText | None, config: VoiceConfig
) -> WakeWordDetector:
    """Hands-free detection when possible, press-Enter otherwise.

    Whether wake-word detection is *wanted* is a question of voice mode; this
    only answers whether it is *possible*.
    """
    if stt is not None and stt.available:
        return KeywordWakeWord(stt, config)
    log.warning("wake word requested but speech input is unavailable")
    return PushToTalk()
