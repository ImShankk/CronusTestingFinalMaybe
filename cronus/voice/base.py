"""Voice provider interfaces.

Speech is an interface to Cronus, not a part of it. Everything provider
specific stays behind these three classes so the runtime never imports an
audio library, and tests never need a microphone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SpeechToText(ABC):
    """Turns microphone audio into text."""

    name: str = "stt"

    @property
    @abstractmethod
    def available(self) -> bool:
        """False when the hardware or dependency is missing."""

    @abstractmethod
    def listen(self, timeout: float | None = None) -> str:
        """Capture one utterance. Returns '' when nothing was understood."""

    def calibrate(self) -> None:
        """Optionally adjust to background noise before the first capture."""


class TextToSpeech(ABC):
    """Speaks text aloud."""

    name: str = "tts"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak, blocking until finished or until :meth:`stop` is called."""

    def stop(self) -> None:
        """Interrupt playback. Safe to call when nothing is playing."""

    def close(self) -> None:
        """Release audio resources."""


class WakeWordDetector(ABC):
    """Decides when the user is addressing Cronus."""

    name: str = "wake"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def wait_for_wake(self) -> bool:
        """Block until activated. Returns False if the user quit instead."""

    @property
    def carried_text(self) -> str:
        """Speech captured alongside the wake word, if the detector heard any.

        Lets "Hey Cronus, what's the weather" work in one breath instead of
        forcing the user to wait and repeat themselves.
        """
        return ""
