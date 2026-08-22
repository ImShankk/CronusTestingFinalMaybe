"""Text-to-speech.

Piper (local, offline, what the original prototype used) is preferred; the
Windows SAPI voice is a genuine fallback when Piper is not installed. Playback
is asynchronous internally so :meth:`stop` can actually cut speech off
mid-sentence when the user interrupts.
"""

from __future__ import annotations

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
        self._temp_dir = Path(tempfile.gettempdir()) / "cronus-voice"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.config.piper_available and _can_play_audio()

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return
        self._stop.clear()
        wav_path = self._temp_dir / "speech.wav"

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
        try:
            process = subprocess.run(
                command,
                input=spoken,
                text=True,
                capture_output=True,
                timeout=_SPEAK_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("piper synthesis failed: %s", exc)
            return
        if process.returncode != 0 or not wav_path.exists():
            log.error("piper exited with %s", process.returncode)
            return
        _play_wav(wav_path, self._stop)

    def stop(self) -> None:
        self._stop.set()
        _stop_playback()


class SapiTTS(TextToSpeech):
    """The built-in Windows voice. No extra downloads, always available here."""

    name = "sapi"

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._voice = None
        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import win32com.client
        except ImportError:
            return False
        return win32com.client is not None

    def _ensure_voice(self):
        if self._voice is None:
            import win32com.client

            self._voice = win32com.client.Dispatch("SAPI.SpVoice")
            # SAPI rate runs -10..10; map 1.0x to 0 and scale from there.
            self._voice.Rate = int(max(-10, min((self.config.speech_rate - 1) * 10, 10)))
        return self._voice

    def speak(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            return
        try:
            voice = self._ensure_voice()
            self._stop.clear()
            # 1 == SVSFlagsAsync, so stop() can interrupt mid-sentence.
            voice.Speak(spoken, 1)
            while voice.Status.RunningState == 2 and not self._stop.is_set():
                threading.Event().wait(0.05)
        except Exception as exc:
            log.error("sapi speech failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._voice is not None:
            try:
                self._voice.Speak("", 3)  # purge whatever is queued
            except Exception:  # pragma: no cover - shutdown best effort
                pass


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
