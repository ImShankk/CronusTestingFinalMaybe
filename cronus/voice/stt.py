"""Speech-to-text via the ``SpeechRecognition`` package.

Carried over from the original prototype (Google's free web endpoint), with
the microphone opened once instead of per utterance, real silence detection,
and failures that report rather than crash.
"""

from __future__ import annotations

from ..config import VoiceConfig
from ..logging_setup import get_logger
from .base import SpeechToText

log = get_logger("voice.stt")


class SpeechRecognitionSTT(SpeechToText):
    """Microphone capture with silence-based endpointing."""

    name = "google_web"

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._recognizer = None
        self._microphone = None
        self._calibrated = False
        self._error: str | None = None
        self._setup()

    def _setup(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self._error = "the SpeechRecognition package is not installed"
            return
        try:
            self._recognizer = sr.Recognizer()
            # Stop capturing after this much silence, so replies feel prompt.
            self._recognizer.pause_threshold = 0.8
            self._recognizer.non_speaking_duration = 0.4
            if self.config.energy_threshold:
                self._recognizer.energy_threshold = self.config.energy_threshold
                self._recognizer.dynamic_energy_threshold = False
            self._microphone = sr.Microphone(device_index=self.config.mic_index)
        except Exception as exc:
            self._error = f"no usable microphone ({exc})"
            log.warning("microphone unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._recognizer is not None and self._microphone is not None

    @property
    def error(self) -> str | None:
        return self._error

    def calibrate(self) -> None:
        if not self.available or self._calibrated:
            return
        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.6)
            self._calibrated = True
            log.info("microphone calibrated threshold=%.0f", self._recognizer.energy_threshold)
        except Exception as exc:
            log.warning("microphone calibration failed: %s", exc)

    def listen(self, timeout: float | None = None) -> str:
        if not self.available:
            return ""
        import speech_recognition as sr

        self.calibrate()
        try:
            with self._microphone as source:
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=self.config.phrase_time_limit,
                )
        except sr.WaitTimeoutError:
            return ""
        except Exception as exc:
            log.warning("microphone capture failed: %s", exc)
            return ""

        try:
            text = self._recognizer.recognize_google(audio)
            log.info("speech recognised (%d chars)", len(text))
            return text.strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            log.error("speech recognition service failed: %s", exc)
            return ""


def build_stt(config: VoiceConfig) -> SpeechToText | None:
    """Return the configured STT provider, or None if it cannot run."""
    provider = SpeechRecognitionSTT(config)
    if not provider.available:
        log.warning("speech input unavailable: %s", provider.error)
        return None
    return provider
