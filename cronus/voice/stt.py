"""Speech-to-text via the ``SpeechRecognition`` package.

Carried over from the original prototype (Google's free web endpoint), with
the microphone opened once instead of per utterance, real silence detection,
and failures that report rather than crash.
"""

from __future__ import annotations

import audioop
import threading

from ..config import VoiceConfig
from ..logging_setup import get_logger
from .base import ListenOutcome, SpeechToText

log = get_logger("voice.stt")

# Barge-in tuning that users should not have to think about.
_FLOOR_SECONDS = 0.4      # how long to sample the room before watching
_HANGOVER_SECONDS = 0.12  # dips shorter than this do not end an utterance
_MINIMUM_FLOOR = 60.0     # never trigger on near-silence


class SpeechRecognitionSTT(SpeechToText):
    """Microphone capture with silence-based endpointing."""

    name = "google_web"

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._recognizer = None
        self._microphone = None
        self._calibrated = False
        self._error: str | None = None
        self.last_outcome = ListenOutcome.NO_SPEECH
        self.last_error: str | None = None
        self._setup()

    def _setup(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self._error = "the SpeechRecognition package is not installed"
            return
        try:
            self._recognizer = sr.Recognizer()
            # How much silence ends a phrase. Too short and a mid-sentence
            # breath is treated as the end of the request.
            self._recognizer.pause_threshold = self.config.pause_threshold
            self._recognizer.non_speaking_duration = self.config.non_speaking_duration
            self._recognizer.phrase_threshold = self.config.min_speech_seconds
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
            # Not fatal: listening can still be attempted with the default
            # threshold, and listen() will report it if the device is truly gone.
            log.warning("microphone calibration failed: %s", exc)

    @property
    def supports_barge_in(self) -> bool:
        return self.available

    def wait_for_speech_start(
        self, stop: threading.Event, sensitivity: float = 2.5
    ) -> bool:
        """Watch the microphone for someone starting to talk.

        Used while Cronus is speaking, which makes the noise floor the hard
        part: through speakers the microphone also hears Cronus. Rather than
        scaling the ambient threshold from calibration -- which swings widely
        depending on what the room was doing at the time -- the floor is
        measured live over the first moments of playback, so it already
        includes Cronus's own voice. The user then has to be clearly louder
        than that.

        Brief dips below the floor between syllables are tolerated, otherwise
        ordinary speech never accumulates an unbroken run.
        """
        if not self.available:
            return False

        chunk_seconds = self._microphone.CHUNK / self._microphone.SAMPLE_RATE
        needed_chunks = max(int(self.config.min_speech_seconds / chunk_seconds), 1)
        hangover_chunks = max(int(_HANGOVER_SECONDS / chunk_seconds), 1)

        try:
            with self._microphone as source:
                floor = self._measure_floor(source, chunk_seconds, sensitivity, stop)
                if floor is None:
                    return False
                log.debug("barge-in floor set to %.0f", floor)

                voiced = silent = 0
                while not stop.is_set():
                    buffer = source.stream.read(source.CHUNK)
                    if not buffer:
                        continue
                    level = audioop.rms(buffer, source.SAMPLE_WIDTH)
                    if level >= floor:
                        voiced += 1
                        silent = 0
                        if voiced >= needed_chunks:
                            log.info(
                                "barge-in detected (level=%d floor=%.0f)", level, floor
                            )
                            return True
                    else:
                        silent += 1
                        if silent > hangover_chunks:
                            voiced = 0
        except Exception as exc:
            # Losing the monitor must never break speaking.
            log.warning("barge-in monitor stopped: %s: %s", type(exc).__name__, exc)
        return False

    def _measure_floor(
        self,
        source,
        chunk_seconds: float,
        sensitivity: float,
        stop: threading.Event,
    ) -> float | None:
        """The level the user has to beat, measured from the room right now.

        Counted in chunks rather than wall time so a slow or stalled stream
        still yields a full sample rather than a floor based on two reads.
        """
        wanted = max(int(_FLOOR_SECONDS / chunk_seconds), 1)
        samples: list[int] = []
        while len(samples) < wanted:
            if stop.is_set():
                return None
            buffer = source.stream.read(source.CHUNK)
            if buffer:
                samples.append(audioop.rms(buffer, source.SAMPLE_WIDTH))
        baseline = sum(samples) / len(samples)
        return max(baseline * sensitivity, _MINIMUM_FLOOR)

    def _record(self, outcome: ListenOutcome, detail: str | None = None) -> str:
        """Note how this attempt ended and return the empty transcript."""
        self.last_outcome = outcome
        self.last_error = detail
        if detail:
            log.warning("listen ended as %s: %s", outcome.value, detail)
        else:
            log.debug("listen ended as %s", outcome.value)
        return ""

    def listen(self, timeout: float | None = None) -> str:
        if not self.available:
            return self._record(ListenOutcome.MIC_ERROR, self._error or "no microphone")
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
            return self._record(ListenOutcome.NO_SPEECH)
        except Exception as exc:
            return self._record(
                ListenOutcome.MIC_ERROR, f"{type(exc).__name__}: {exc}"
            )

        try:
            text = self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return self._record(ListenOutcome.NOT_UNDERSTOOD)
        except sr.RequestError as exc:
            return self._record(
                ListenOutcome.SERVICE_ERROR, f"{type(exc).__name__}: {exc}"
            )
        except Exception as exc:
            return self._record(
                ListenOutcome.SERVICE_ERROR, f"{type(exc).__name__}: {exc}"
            )

        text = text.strip()
        if not text:
            return self._record(ListenOutcome.NOT_UNDERSTOOD)
        self.last_outcome = ListenOutcome.HEARD
        self.last_error = None
        log.info("speech recognised (%d chars)", len(text))
        return text


def build_stt(config: VoiceConfig) -> SpeechToText | None:
    """Return the configured STT provider, or None if it cannot run."""
    provider = SpeechRecognitionSTT(config)
    if not provider.available:
        log.warning("speech input unavailable: %s", provider.error)
        return None
    return provider
