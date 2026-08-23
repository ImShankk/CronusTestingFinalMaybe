"""Text-to-speech.

Piper (local, offline, what the original prototype used) is preferred; the
Windows SAPI voice is a genuine fallback when Piper is not installed. Playback
is asynchronous internally so :meth:`stop` can actually cut speech off
mid-sentence when the user interrupts.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from ..config import VoiceConfig
from ..logging_setup import get_logger
from .base import TextToSpeech

log = get_logger("voice.tts")

_SPEAK_TIMEOUT = 60
# A reminder can come due on the scheduler thread while a reply is being
# spoken on the main thread. Speech is a single shared output device, so only
# one utterance may hold it at a time.
_speech_lock = threading.Lock()
_utterance_ids = itertools.count()


def clean_for_speech(text: str) -> str:
    """Strip markup that sounds like noise when read aloud."""
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    return re.sub(r"\s+", " ", text).strip()


class PiperTTS(TextToSpeech):
    """Local neural speech via the Piper binary."""

    name = "piper"

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._stop = threading.Event()
        #: The synthesis process currently running, so it can be cut short.
        self._process: subprocess.Popen | None = None
        self._temp_dir = Path(tempfile.gettempdir()) / "cronus-voice"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.config.piper_available and _can_play_audio()

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return
        with _speech_lock:
            self._speak_locked(spoken)

    def _speak_locked(self, spoken: str) -> None:
        self._stop.clear()
        # A distinct file per utterance: two speakers must never write the
        # same wav while the other is reading it.
        wav_path = self._temp_dir / f"speech-{next(_utterance_ids)}.wav"

        # Piper slows down as length-scale rises, so invert the rate.
        length_scale = max(0.5, min(1.0 / max(self.config.speech_rate, 0.1), 2.0))
        command = [
            str(self.config.piper_exe),
            "-m",
            str(self.config.piper_model),
            "-f",
            str(wav_path),
            "--length-scale",
            f"{length_scale:.2f}",
        ]
        # Popen rather than run(): a higher-quality voice can take seconds to
        # synthesise, and barge-in has to be able to abandon that work rather
        # than make the user wait for a reply they already interrupted.
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            log.error("could not start piper: %s", exc)
            return

        self._process = process
        try:
            process.communicate(input=spoken, timeout=_SPEAK_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.error("piper synthesis timed out after %ss", _SPEAK_TIMEOUT)
            process.kill()
            process.communicate()
            wav_path.unlink(missing_ok=True)
            return
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("piper synthesis failed: %s", exc)
            wav_path.unlink(missing_ok=True)
            return
        finally:
            self._process = None

        if self._stop.is_set():
            # Interrupted while still synthesising: throw the audio away.
            log.debug("synthesis abandoned after interruption")
            wav_path.unlink(missing_ok=True)
            return
        if process.returncode != 0 or not wav_path.exists():
            log.error("piper exited with %s", process.returncode)
            wav_path.unlink(missing_ok=True)
            return
        try:
            _play_wav(wav_path, self._stop)
        finally:
            wav_path.unlink(missing_ok=True)

    def stop(self) -> None:
        self._stop.set()
        # Kill synthesis in flight as well as audio already playing, so an
        # interruption is prompt whichever stage the reply had reached.
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:  # pragma: no cover - already gone
                log.debug("piper process already exited")
        _stop_playback()


# SAPI flags.
_SVSF_ASYNC = 1
_SVSF_PURGE = 3          # async + purge before speak
_POLL_MS = 50            # how often to check for an interruption


class SapiTTS(TextToSpeech):
    """The built-in Windows voice. No extra downloads, always available here.

    SAPI is COM, and COM is per-thread: a voice created on one thread cannot
    be driven from another, and any thread using it must initialise COM
    first. Speech runs on a worker thread so it can be interrupted, so the
    voice is thread-local and :meth:`stop` only raises a flag -- the purge
    itself happens on whichever thread is actually speaking.
    """

    name = "sapi"

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._local = threading.local()
        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        import importlib.util

        return importlib.util.find_spec("win32com.client") is not None

    def _ensure_voice(self):
        """A SAPI voice belonging to the calling thread."""
        voice = getattr(self._local, "voice", None)
        if voice is not None:
            return voice

        import pythoncom
        import win32com.client

        try:
            pythoncom.CoInitialize()
        except Exception:  # pragma: no cover - already initialised is fine
            log.debug("COM already initialised on this thread")

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        # SAPI rate runs -10..10; map 1.0x to 0 and scale from there.
        voice.Rate = int(max(-10, min((self.config.speech_rate - 1) * 10, 10)))
        self._local.voice = voice
        return voice

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return
        with _speech_lock:
            try:
                voice = self._ensure_voice()
                self._stop.clear()
                voice.Speak(spoken, _SVSF_ASYNC)
                # WaitUntilDone returns False while still speaking. Polling
                # Status.RunningState instead races the start of playback and
                # returns immediately, leaving audio running unattended.
                while not voice.WaitUntilDone(_POLL_MS):
                    if self._stop.is_set():
                        # Purge here: this is the thread that owns the voice.
                        voice.Speak("", _SVSF_PURGE)
                        break
            except Exception as exc:
                log.error("sapi speech failed: %s", exc)

    def stop(self) -> None:
        # Only a flag: the speaking thread does the actual purge, because the
        # COM object cannot be touched from here.
        self._stop.set()


def _can_play_audio() -> bool:
    if sys.platform == "win32":
        return True
    return any(_which(player) for player in ("afplay", "aplay", "paplay"))


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


_playback_process: subprocess.Popen | None = None


def _play_wav(path: Path, stop_event: threading.Event) -> None:
    """Play a wav file, returning early if ``stop_event`` is set."""
    global _playback_process

    if sys.platform == "win32":
        import winsound

        # SND_ASYNC hands control back so an interrupt can purge playback.
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        duration = _wav_duration(path)
        if stop_event.wait(duration):
            winsound.PlaySound(None, winsound.SND_PURGE)
        return

    player = next(
        (p for p in ("afplay", "aplay", "paplay") if _which(p)), None
    )
    if player is None:
        log.warning("no audio player available on this platform")
        return
    try:
        _playback_process = subprocess.Popen(
            [player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        while _playback_process.poll() is None:
            if stop_event.wait(0.1):
                _playback_process.terminate()
                break
    except OSError as exc:
        log.error("audio playback failed: %s", exc)
    finally:
        _playback_process = None


def _stop_playback() -> None:
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)
        return
    if _playback_process is not None and _playback_process.poll() is None:
        _playback_process.terminate()


def _wav_duration(path: Path) -> float:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate() or 1)
    except Exception:  # pragma: no cover - malformed audio
        return float(_SPEAK_TIMEOUT)


def build_tts(config: VoiceConfig) -> TextToSpeech | None:
    """Pick the best available speech provider, honouring configuration."""
    preferred = (config.tts_provider or "piper").lower()
    candidates: list[TextToSpeech] = []
    if preferred == "sapi":
        candidates = [SapiTTS(config), PiperTTS(config)]
    else:
        candidates = [PiperTTS(config), SapiTTS(config)]

    for provider in candidates:
        if provider.available:
            if provider.name != preferred:
                log.warning(
                    "%s speech is unavailable; using %s instead", preferred, provider.name
                )
            log.info("speech output using %s", provider.name)
            return provider
    log.warning("no speech output provider is available")
    return None
