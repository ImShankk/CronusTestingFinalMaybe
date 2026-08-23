"""The voice turn-taking runtime.

One object owns the audio side of a conversation: getting the next utterance,
and speaking a reply in a way the user can interrupt. All three voice modes
share this code -- they differ only in how a turn is allowed to begin.

Interruption is real, not simulated. Speech is synthesised and played on a
worker thread while the microphone is watched on this one; when sustained
speech is heard, playback is stopped through the provider's own ``stop()``
and the thread is joined before the next turn starts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import VoiceConfig, VoiceMode
from ..core.events import AssistantState, EventEmitter
from ..logging_setup import get_logger
from .base import ListenOutcome, SpeechToText, TextToSpeech, WakeWordDetector

log = get_logger("voice.session")

# How long to wait for a speaking thread to notice it has been stopped.
_STOP_GRACE = 3.0


@dataclass
class Utterance:
    """One attempt to hear the user."""

    text: str = ""
    outcome: ListenOutcome = ListenOutcome.NO_SPEECH
    quit_requested: bool = False

    @property
    def heard(self) -> bool:
        return bool(self.text)


class VoiceSession:
    """Turn-taking over a microphone and a speaker."""

    def __init__(
        self,
        config: VoiceConfig,
        *,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        wake: WakeWordDetector | None = None,
        emitter: EventEmitter | None = None,
        prompt_to_talk: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.stt = stt
        self.tts = tts
        self.wake = wake
        self.emitter = emitter or EventEmitter()
        #: Used only by push-to-talk, so the interface owns its own prompting.
        self.prompt_to_talk = prompt_to_talk
        self.interrupted = False
        #: What the user said to interrupt, captured the moment speech stops.
        #: Talking over Cronus is a request, not just a stop button, so it is
        #: kept and answered rather than making them say it a second time.
        self.pending: Utterance | None = None
        self._speaking = threading.Event()

    # ------------------------------------------------------------------
    @property
    def mode(self) -> VoiceMode:
        return self.config.mode

    @property
    def can_listen(self) -> bool:
        return self.stt is not None and self.stt.available

    @property
    def can_speak(self) -> bool:
        return self.tts is not None

    @property
    def barge_in_available(self) -> bool:
        """Whether a reply can actually be interrupted by talking over it."""
        return bool(
            self.config.barge_in
            and self.can_speak
            and self.stt is not None
            and self.stt.supports_barge_in
        )

    @property
    def has_pending(self) -> bool:
        """Whether an interrupting utterance is already waiting to be answered."""
        return self.pending is not None and self.pending.heard

    def describe_mode(self) -> str:
        return self.mode.value.replace("_", "-")

    def describe_voice(self) -> str:
        """Which speech provider and voice is actually in use, for status output.

        Names the model rather than just the provider, so a silent fallback to
        a different voice than the one configured is visible at a glance.
        """
        if self.tts is None:
            return "no speech output"
        parts = [self.tts.name]
        if self.tts.name == "piper" and self.config.piper_model:
            parts.append(Path(self.config.piper_model).stem)
        parts.append(f"{self.config.speech_rate:.2f}x")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Getting a turn
    # ------------------------------------------------------------------
    def next_utterance(self) -> Utterance:
        """Wait for the user to say something, according to the current mode."""
        # Whatever they said over the top of the last reply is already in
        # hand. Answer it instead of asking them to start the turn again --
        # including in wake-word mode, since interrupting is itself an address.
        pending, self.pending = self.pending, None
        if pending is not None and pending.heard:
            return pending

        if not self.can_listen:
            return Utterance(outcome=ListenOutcome.MIC_ERROR)

        if self.mode is VoiceMode.PUSH_TO_TALK:
            if self.prompt_to_talk is not None and not self.prompt_to_talk():
                return Utterance(quit_requested=True)
            return self._capture()

        if self.mode is VoiceMode.WAKE_WORD and self.wake is not None:
            self.emitter.set_state(AssistantState.IDLE)
            if not self.wake.wait_for_wake():
                return Utterance(quit_requested=True)
            carried = self.wake.carried_text
            if carried:
                # They said it all in one breath: no second capture needed.
                return Utterance(text=carried, outcome=ListenOutcome.HEARD)
            return self._capture()

        # Continuous: the microphone is simply open for the next turn.
        return self._capture()

    def _capture(self) -> Utterance:
        self.emitter.set_state(AssistantState.LISTENING)
        try:
            text = self.stt.listen(timeout=self.config.listen_timeout)
        finally:
            self.emitter.set_state(AssistantState.IDLE)
        outcome = getattr(self.stt, "last_outcome", ListenOutcome.NO_SPEECH)
        return Utterance(text=text, outcome=outcome)

    # ------------------------------------------------------------------
    # Speaking, interruptibly
    # ------------------------------------------------------------------
    def speak(self, text: str) -> bool:
        """Say something. Returns True if the user talked over it.

        Without barge-in support this simply speaks and returns False, so the
        caller needs no special case.
        """
        self.interrupted = False
        if not self.can_speak or not text.strip():
            return False

        if not self.barge_in_available:
            self.emitter.set_state(AssistantState.SPEAKING)
            try:
                self.tts.speak(text)
            except Exception as exc:
                log.error("speech failed: %s: %s", type(exc).__name__, exc)
            finally:
                self.emitter.set_state(AssistantState.IDLE)
            return False

        return self._speak_interruptible(text)

    def _speak_interruptible(self, text: str) -> bool:
        finished = threading.Event()

        def play() -> None:
            try:
                self.tts.speak(text)
            except Exception as exc:
                log.error("speech failed: %s: %s", type(exc).__name__, exc)
            finally:
                finished.set()

        self.emitter.set_state(AssistantState.SPEAKING)
        self._speaking.set()
        speaker = threading.Thread(target=play, name="cronus-speak", daemon=True)
        speaker.start()

        interrupted = False
        try:
            # Returns as soon as the user speaks, or when playback finishes.
            interrupted = self.stt.wait_for_speech_start(
                finished, self.config.barge_in_sensitivity
            )
        except Exception as exc:  # pragma: no cover - monitor is best-effort
            log.warning("barge-in monitor failed: %s: %s", type(exc).__name__, exc)

        if interrupted:
            log.info("user interrupted; stopping speech")
            self.emitter.set_state(AssistantState.INTERRUPTED)
            self.stop_speaking()

        # Always join: no playback thread outlives the turn that started it.
        finished.set()
        speaker.join(timeout=_STOP_GRACE)
        if speaker.is_alive():  # pragma: no cover - defensive
            log.warning("speech thread did not stop within %.0fs", _STOP_GRACE)
        self._speaking.clear()
        self.interrupted = interrupted

        if interrupted:
            # The monitor only measures loudness, so the words themselves were
            # never transcribed. Reopen the microphone straight away, while
            # the user is still mid-sentence, and keep what it hears. The
            # syllable or two that triggered the interrupt is genuinely lost;
            # everything after it is not.
            self.pending = self._capture()

        self.emitter.set_state(AssistantState.IDLE)
        return interrupted

    def stop_speaking(self) -> None:
        """Cut off whatever is being said right now."""
        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                log.warning("could not stop speech: %s", exc)

    def close(self) -> None:
        self.pending = None
        self.stop_speaking()
        if self.tts is not None:
            try:
                self.tts.close()
            except Exception:  # pragma: no cover - shutdown best effort
                log.debug("tts close failed", exc_info=True)


def build_session(
    config: VoiceConfig,
    *,
    emitter: EventEmitter | None = None,
    prompt_to_talk: Callable[[], bool] | None = None,
    listen: bool = True,
) -> VoiceSession:
    """Assemble a session from whichever providers are actually available."""
    from .stt import build_stt
    from .tts import build_tts
    from .wake import build_wake_word

    tts = build_tts(config)
    stt = build_stt(config) if listen else None
    wake = None
    if stt is not None and config.mode is VoiceMode.WAKE_WORD:
        # The wake detector reuses the same recogniser, so no second mic.
        wake = build_wake_word(stt, config)
    return VoiceSession(
        config,
        stt=stt,
        tts=tts,
        wake=wake,
        emitter=emitter,
        prompt_to_talk=prompt_to_talk,
    )
