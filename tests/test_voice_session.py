"""Voice modes, endpointing, barge-in, and turn lifecycle.

Everything here runs on fake providers: no microphone, no speaker, no network.
Timing-sensitive tests use short sleeps against real threads, because the
whole point of barge-in is that two threads cooperate correctly.
"""

from __future__ import annotations

import threading
from pathlib import Path
import time

import pytest

from cronus.config import VoiceConfig, VoiceMode, load_config
from cronus.core.events import AssistantState, EventEmitter, EventType
from cronus.errors import ConfigError
from cronus.voice.base import ListenOutcome, SpeechToText, TextToSpeech
from cronus.voice.session import VoiceSession, build_session


# ======================================================================
# Fakes
# ======================================================================
class FakeSTT(SpeechToText):
    """Scripted speech input with controllable barge-in behaviour."""

    name = "fake"

    def __init__(
        self,
        utterances: list[str] | None = None,
        *,
        available: bool = True,
        outcome: ListenOutcome = ListenOutcome.HEARD,
        barge_in_after: float | None = None,
        supports_barge_in: bool = True,
    ) -> None:
        self.utterances = list(utterances or [])
        self._available = available
        self._outcome = outcome
        self.barge_in_after = barge_in_after
        self._supports = supports_barge_in
        self.listen_calls: list[float | None] = []
        self.monitor_calls = 0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def supports_barge_in(self) -> bool:
        return self._supports and self._available

    def listen(self, timeout: float | None = None) -> str:
        self.listen_calls.append(timeout)
        if self.utterances:
            self.last_outcome = ListenOutcome.HEARD
            return self.utterances.pop(0)
        self.last_outcome = self._outcome
        return ""

    def wait_for_speech_start(self, stop, sensitivity=2.5) -> bool:
        self.monitor_calls += 1
        if self.barge_in_after is None:
            stop.wait()          # never interrupts; waits for playback to end
            return False
        # Interrupt once the configured moment arrives, unless speech ends first.
        return not stop.wait(self.barge_in_after)


class FakeTTS(TextToSpeech):
    """Speech output that takes real time and can really be stopped."""

    name = "fake"

    def __init__(self, duration: float = 0.4, fail: bool = False) -> None:
        self.duration = duration
        self.fail = fail
        self.spoken: list[str] = []
        self.completed: list[str] = []
        self.stops = 0
        self.closed = False
        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        return True

    def speak(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("audio device exploded")
        self.spoken.append(text)
        self._stop.clear()
        if not self._stop.wait(self.duration):
            self.completed.append(text)

    def stop(self) -> None:
        self.stops += 1
        self._stop.set()

    def close(self) -> None:
        self.closed = True


class FakeWake:
    name = "keyword"

    def __init__(self, results: list[bool], carried: str = ""):
        self.results = list(results)
        self._carried = carried

    @property
    def available(self) -> bool:
        return True

    def wait_for_wake(self) -> bool:
        return self.results.pop(0) if self.results else False

    @property
    def carried_text(self) -> str:
        carried, self._carried = self._carried, ""
        return carried


def make_session(**kwargs) -> VoiceSession:
    config = kwargs.pop("config", VoiceConfig())
    return VoiceSession(config, **kwargs)


# ======================================================================
# Configuration
# ======================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("continuous", VoiceMode.CONTINUOUS),
        ("push_to_talk", VoiceMode.PUSH_TO_TALK),
        ("push-to-talk", VoiceMode.PUSH_TO_TALK),
        ("WAKE_WORD", VoiceMode.WAKE_WORD),
        (None, VoiceMode.CONTINUOUS),
    ],
)
def test_voice_mode_parsing(raw, expected):
    assert VoiceMode.parse(raw) is expected


def test_an_unknown_voice_mode_is_rejected_clearly():
    with pytest.raises(ConfigError) as info:
        VoiceMode.parse("shouting")
    assert "push_to_talk, continuous, or wake_word" in info.value.user_message


def test_continuous_is_the_default_mode(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert load_config(env_file=None).voice.mode is VoiceMode.CONTINUOUS


def test_voice_mode_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_VOICE_MODE", "push_to_talk")
    assert load_config(env_file=None).voice.mode is VoiceMode.PUSH_TO_TALK


def test_the_legacy_wake_word_flag_still_selects_wake_mode(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_WAKE_WORD_ENABLED", "true")
    assert load_config(env_file=None).voice.mode is VoiceMode.WAKE_WORD


def test_an_explicit_mode_beats_the_legacy_flag(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_WAKE_WORD_ENABLED", "true")
    monkeypatch.setenv("CRONUS_VOICE_MODE", "continuous")
    assert load_config(env_file=None).voice.mode is VoiceMode.CONTINUOUS


def test_endpointing_defaults_allow_natural_pauses(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    voice = load_config(env_file=None).voice
    assert voice.pause_threshold >= 1.0, "too short and mid-sentence pauses cut off"
    assert voice.phrase_time_limit >= 30, "long requests must fit"
    assert voice.non_speaking_duration <= voice.pause_threshold


def test_endpointing_is_tunable(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_PAUSE_THRESHOLD", "2.5")
    monkeypatch.setenv("CRONUS_NON_SPEAKING_DURATION", "0.8")
    monkeypatch.setenv("CRONUS_PHRASE_TIME_LIMIT", "45")
    voice = load_config(env_file=None).voice
    assert (voice.pause_threshold, voice.non_speaking_duration) == (2.5, 0.8)
    assert voice.phrase_time_limit == 45


def test_contradictory_endpointing_is_rejected(monkeypatch):
    """SpeechRecognition requires non_speaking <= pause; say so up front."""
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_PAUSE_THRESHOLD", "0.5")
    monkeypatch.setenv("CRONUS_NON_SPEAKING_DURATION", "2.0")
    with pytest.raises(ConfigError, match="NON_SPEAKING_DURATION"):
        load_config(env_file=None)


def test_the_recogniser_receives_the_configured_endpointing(monkeypatch):
    """Settings are worthless if they never reach SpeechRecognition."""
    import sys

    captured = {}

    class Recognizer:
        def __setattr__(self, name, value):
            captured[name] = value
            object.__setattr__(self, name, value)

    class Microphone:
        def __init__(self, device_index=None):
            pass

    stub = type("SR", (), {"Recognizer": Recognizer, "Microphone": Microphone})
    monkeypatch.setitem(sys.modules, "speech_recognition", stub)

    from cronus.voice.stt import SpeechRecognitionSTT

    SpeechRecognitionSTT(
        VoiceConfig(pause_threshold=1.7, non_speaking_duration=0.6,
                    min_speech_seconds=0.25)
    )
    assert captured["pause_threshold"] == 1.7
    assert captured["non_speaking_duration"] == 0.6
    assert captured["phrase_threshold"] == 0.25


# ======================================================================
# Turn taking per mode
# ======================================================================
def test_continuous_mode_listens_without_any_prompting():
    stt = FakeSTT(["what is the weather"])
    prompts = []
    session = make_session(
        config=VoiceConfig(mode=VoiceMode.CONTINUOUS),
        stt=stt,
        prompt_to_talk=lambda: prompts.append(1) or True,
    )
    utterance = session.next_utterance()
    assert utterance.text == "what is the weather"
    assert prompts == [], "continuous mode must not ask the user to press anything"


def test_continuous_mode_keeps_listening_turn_after_turn():
    stt = FakeSTT(["first thing", "second thing", "third thing"])
    session = make_session(config=VoiceConfig(mode=VoiceMode.CONTINUOUS), stt=stt)
    heard = [session.next_utterance().text for _ in range(3)]
    assert heard == ["first thing", "second thing", "third thing"]


def test_push_to_talk_waits_for_the_prompt():
    stt = FakeSTT(["hello"])
    prompts = []
    session = make_session(
        config=VoiceConfig(mode=VoiceMode.PUSH_TO_TALK),
        stt=stt,
        prompt_to_talk=lambda: prompts.append(1) or True,
    )
    assert session.next_utterance().text == "hello"
    assert prompts == [1], "push-to-talk must prompt before capturing"


def test_push_to_talk_can_quit():
    session = make_session(
        config=VoiceConfig(mode=VoiceMode.PUSH_TO_TALK),
        stt=FakeSTT(["never reached"]),
        prompt_to_talk=lambda: False,
    )
    utterance = session.next_utterance()
    assert utterance.quit_requested and not utterance.heard


def test_wake_word_mode_waits_for_the_phrase():
    stt = FakeSTT(["set a timer"])
    session = make_session(
        config=VoiceConfig(mode=VoiceMode.WAKE_WORD),
        stt=stt,
        wake=FakeWake([True]),
    )
    assert session.next_utterance().text == "set a timer"


def test_wake_word_uses_speech_carried_in_the_same_breath():
    stt = FakeSTT(["should not be needed"])
    session = make_session(
        config=VoiceConfig(mode=VoiceMode.WAKE_WORD),
        stt=stt,
        wake=FakeWake([True], carried="what is the weather"),
    )
    utterance = session.next_utterance()
    assert utterance.text == "what is the weather"
    assert stt.listen_calls == [], "no second capture needed"


def test_all_modes_share_one_capture_path():
    """Modes differ in how a turn begins, not in how speech is captured."""
    for mode in VoiceMode:
        stt = FakeSTT(["hello"])
        session = make_session(
            config=VoiceConfig(mode=mode, listen_timeout=9.0),
            stt=stt,
            wake=FakeWake([True]),
            prompt_to_talk=lambda: True,
        )
        session.next_utterance()
        assert stt.listen_calls == [9.0], f"{mode} used a different capture path"


def test_a_missing_microphone_is_reported_not_hidden():
    session = make_session(stt=FakeSTT(available=False))
    utterance = session.next_utterance()
    assert utterance.outcome is ListenOutcome.MIC_ERROR
    assert not utterance.heard


def test_silence_is_reported_as_no_speech():
    session = make_session(stt=FakeSTT([], outcome=ListenOutcome.NO_SPEECH))
    assert session.next_utterance().outcome is ListenOutcome.NO_SPEECH


def test_stt_failure_surfaces_its_outcome():
    session = make_session(stt=FakeSTT([], outcome=ListenOutcome.SERVICE_ERROR))
    assert session.next_utterance().outcome is ListenOutcome.SERVICE_ERROR


# ======================================================================
# Barge-in
# ======================================================================
def test_speaking_completes_when_nobody_interrupts():
    tts = FakeTTS(duration=0.1)
    session = make_session(stt=FakeSTT(barge_in_after=None), tts=tts)
    assert session.speak("a full sentence") is False
    assert tts.completed == ["a full sentence"]
    assert tts.stops == 0


def test_speaking_is_cut_off_when_the_user_talks_over_it():
    tts = FakeTTS(duration=5.0)
    stt = FakeSTT(barge_in_after=0.05)
    session = make_session(stt=stt, tts=tts)

    started = time.monotonic()
    interrupted = session.speak("a very long answer that should be cut short")
    elapsed = time.monotonic() - started

    assert interrupted is True
    assert tts.stops >= 1, "playback was never actually stopped"
    assert tts.completed == [], "speech must not have run to completion"
    assert elapsed < 2.0, f"took {elapsed:.1f}s; interruption was not immediate"


def test_interruption_reaches_the_interrupted_state():
    emitter = EventEmitter()
    states = []
    emitter.subscribe(
        lambda e: states.append(e.data.get("state"))
        if e.type is EventType.STATE else None
    )
    session = make_session(
        stt=FakeSTT(barge_in_after=0.05), tts=FakeTTS(duration=5.0), emitter=emitter
    )
    session.speak("something long")
    assert AssistantState.SPEAKING in states
    assert AssistantState.INTERRUPTED in states
    assert states[-1] is AssistantState.IDLE, "must return to idle to listen again"


def test_the_listener_resumes_after_an_interruption():
    stt = FakeSTT(["I meant Edmonton"], barge_in_after=0.05)
    session = make_session(stt=stt, tts=FakeTTS(duration=5.0))
    assert session.speak("the weather in Denver is") is True
    assert session.next_utterance().text == "I meant Edmonton"


def test_repeated_interruptions_stay_stable():
    tts = FakeTTS(duration=5.0)
    session = make_session(stt=FakeSTT(barge_in_after=0.03), tts=tts)
    for _ in range(5):
        assert session.speak("another long reply") is True
    assert tts.stops >= 5
    assert threading.active_count() < 20, "speech threads are accumulating"


def test_no_speech_thread_outlives_its_turn():
    before = threading.active_count()
    session = make_session(stt=FakeSTT(barge_in_after=0.03), tts=FakeTTS(duration=5.0))
    for _ in range(3):
        session.speak("long reply")
    time.sleep(0.2)
    assert threading.active_count() <= before + 1, "orphaned speech thread"


def test_barge_in_is_skipped_when_the_provider_cannot_monitor():
    stt = FakeSTT(supports_barge_in=False)
    session = make_session(stt=stt, tts=FakeTTS(duration=0.05))
    assert session.barge_in_available is False
    assert session.speak("hello") is False
    assert stt.monitor_calls == 0


def test_barge_in_can_be_switched_off():
    stt = FakeSTT(barge_in_after=0.01)
    session = make_session(
        config=VoiceConfig(barge_in=False), stt=stt, tts=FakeTTS(duration=0.05)
    )
    assert session.barge_in_available is False
    assert session.speak("hello") is False
    assert stt.monitor_calls == 0


def test_a_tts_failure_does_not_break_the_turn():
    session = make_session(stt=FakeSTT(barge_in_after=None), tts=FakeTTS(fail=True))
    assert session.speak("this will fail") is False


def test_speaking_without_a_provider_is_harmless():
    session = make_session(stt=FakeSTT())
    assert session.speak("nobody can hear this") is False


def test_empty_replies_are_not_spoken():
    tts = FakeTTS()
    session = make_session(stt=FakeSTT(), tts=tts)
    assert session.speak("   ") is False
    assert tts.spoken == []


def test_stopping_speech_is_safe_when_idle():
    session = make_session(tts=FakeTTS())
    session.stop_speaking()
    session.stop_speaking()


def test_closing_releases_the_speaker():
    tts = FakeTTS()
    session = make_session(tts=tts)
    session.close()
    assert tts.closed is True


# ======================================================================
# Assembly
# ======================================================================
def test_a_session_without_listening_has_no_microphone(monkeypatch):
    from cronus.voice import session as session_module

    monkeypatch.setattr(session_module, "build_session", build_session)
    built = build_session(VoiceConfig(), listen=False)
    assert built.stt is None
    assert built.can_listen is False


def test_mode_names_are_human_readable():
    for mode, expected in [
        (VoiceMode.CONTINUOUS, "continuous"),
        (VoiceMode.PUSH_TO_TALK, "push-to-talk"),
        (VoiceMode.WAKE_WORD, "wake-word"),
    ]:
        assert make_session(config=VoiceConfig(mode=mode)).describe_mode() == expected


# ======================================================================
# The barge-in detector itself, driven with synthetic audio levels
# ======================================================================
class _FakeSource:
    """A microphone whose chunk levels are scripted."""

    CHUNK = 1024
    SAMPLE_RATE = 44100
    SAMPLE_WIDTH = 2

    def __init__(self, levels: list[int]):
        self.levels = list(levels)
        self.stream = self
        self.reads = 0

    def read(self, size):
        self.reads += 1
        level = self.levels.pop(0) if self.levels else 0
        # audioop.rms of a constant-valued buffer is that value.
        return level.to_bytes(2, "little", signed=True) * (size // 2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _detector(levels: list[int], config: VoiceConfig | None = None):
    """A real SpeechRecognitionSTT wired to a scripted microphone."""
    from cronus.voice.stt import SpeechRecognitionSTT

    stt = SpeechRecognitionSTT.__new__(SpeechRecognitionSTT)
    stt.config = config or VoiceConfig()
    stt._microphone = _FakeSource(levels)
    # Only needs to be non-None: `available` checks for a recogniser, and the
    # detector reads raw levels rather than using it.
    stt._recognizer = object()
    stt._calibrated = True
    stt._error = None
    stt.last_outcome = ListenOutcome.NO_SPEECH
    stt.last_error = None
    return stt


_CHUNKS_PER_SECOND = 44100 / 1024  # ~43


def _seconds(count: float) -> int:
    return max(int(count * _CHUNKS_PER_SECOND), 1)


def test_the_detector_fires_on_sustained_loud_speech():
    quiet = [100] * _seconds(0.5)          # floor measurement window
    speech = [900] * _seconds(0.5)         # clearly above the floor
    stt = _detector(quiet + speech)
    assert stt.wait_for_speech_start(threading.Event(), sensitivity=2.5) is True


def test_the_detector_ignores_a_single_blip():
    """A cough or a door must not be treated as an interruption."""
    quiet = [100] * _seconds(0.5)
    blip = [900] * 2 + [100] * _seconds(1.0)
    stt = _detector(quiet + blip)
    stop = threading.Event()
    threading.Timer(0.8, stop.set).start()
    assert stt.wait_for_speech_start(stop, sensitivity=2.5) is False


def test_brief_dips_between_syllables_do_not_reset_detection():
    """Real speech dips between words; requiring an unbroken run never fires."""
    quiet = [100] * _seconds(0.5)
    syllables: list[int] = []
    for _ in range(8):
        syllables += [900] * 3 + [80] * 2   # ~70ms voiced, ~46ms gap
    stt = _detector(quiet + syllables)
    assert stt.wait_for_speech_start(threading.Event(), sensitivity=2.5) is True


def test_the_detector_stays_quiet_in_a_silent_room():
    """Near-silence must never clear the floor, however low the ambient level."""
    stt = _detector([2] * _seconds(3.0))
    stop = threading.Event()
    threading.Timer(0.8, stop.set).start()
    assert stt.wait_for_speech_start(stop, sensitivity=2.5) is False


def test_the_floor_adapts_to_a_loud_room():
    """In a noisy room the bar rises, so the room itself does not interrupt."""
    loud_room = [500] * _seconds(0.5)
    same_level = [500] * _seconds(1.0)
    stt = _detector(loud_room + same_level)
    stop = threading.Event()
    threading.Timer(0.8, stop.set).start()
    assert stt.wait_for_speech_start(stop, sensitivity=2.5) is False


def test_stopping_during_the_floor_measurement_returns_immediately():
    stt = _detector([100] * _seconds(5.0))
    stop = threading.Event()
    stop.set()
    assert stt.wait_for_speech_start(stop, sensitivity=2.5) is False


def test_an_unavailable_microphone_cannot_barge_in():
    stt = _detector([900] * 100)
    stt._microphone = None
    stt._error = "no microphone"
    assert stt.wait_for_speech_start(threading.Event()) is False


# ======================================================================
# Piper: interrupting while it is still synthesising
# ======================================================================
class _FakePiperProcess:
    """Stands in for piper.exe: slow to synthesise, killable."""

    def __init__(self, wav_path, duration=5.0):
        self.wav_path = wav_path
        self.duration = duration
        self.returncode = None
        self.terminated = False
        self._done = threading.Event()

    def communicate(self, input=None, timeout=None):
        # Finishes early if terminated, exactly as a killed process would.
        if self._done.wait(self.duration):
            self.returncode = 1
        else:
            self.returncode = 0
            self.wav_path.write_bytes(b"RIFF fake wav")
        return ("", "")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._done.set()


def test_interrupting_piper_mid_synthesis_kills_the_process(monkeypatch, tmp_path):
    """Regression: stop() waited for a slow voice to finish synthesising."""
    from cronus.voice import tts as tts_module

    started = threading.Event()
    created: list = []

    def fake_popen(command, **kwargs):
        wav = Path(command[command.index("-f") + 1])
        process = _FakePiperProcess(wav)
        created.append(process)
        started.set()
        return process

    monkeypatch.setattr(tts_module.subprocess, "Popen", fake_popen)
    played: list = []
    monkeypatch.setattr(tts_module, "_play_wav", lambda p, s: played.append(p))
    monkeypatch.setattr(tts_module, "_stop_playback", lambda: None)

    config = VoiceConfig(
        piper_exe=str(tmp_path / "piper.exe"), piper_model=str(tmp_path / "v.onnx")
    )
    provider = tts_module.PiperTTS(config)

    speaker = threading.Thread(target=provider.speak, args=("a long reply",))
    speaker.start()
    assert started.wait(2.0), "synthesis never started"

    begun = time.monotonic()
    provider.stop()
    speaker.join(timeout=3.0)
    elapsed = time.monotonic() - begun

    assert not speaker.is_alive(), "speech thread outlived the interruption"
    assert created[0].terminated, "the synthesis process was never killed"
    assert elapsed < 2.0, f"took {elapsed:.1f}s to abandon synthesis"
    assert played == [], "audio was played after the user interrupted"


def test_piper_discards_audio_that_finished_after_the_interruption(
    monkeypatch, tmp_path
):
    """If terminate() lands too late, the finished audio must still be dropped."""
    from cronus.voice import tts as tts_module

    holder: dict = {}

    class RacingProcess(_FakePiperProcess):
        def communicate(self, input=None, timeout=None):
            # The user interrupts, but synthesis completes anyway.
            holder["provider"].stop()
            self.returncode = 0
            self.wav_path.write_bytes(b"RIFF fake wav")
            return ("", "")

    monkeypatch.setattr(
        tts_module.subprocess,
        "Popen",
        lambda command, **kw: RacingProcess(Path(command[command.index("-f") + 1])),
    )
    played: list = []
    monkeypatch.setattr(tts_module, "_play_wav", lambda p, s: played.append(p))
    monkeypatch.setattr(tts_module, "_stop_playback", lambda: None)

    config = VoiceConfig(
        piper_exe=str(tmp_path / "piper.exe"), piper_model=str(tmp_path / "v.onnx")
    )
    provider = tts_module.PiperTTS(config)
    holder["provider"] = provider
    provider.speak("a reply nobody wants any more")
    assert played == [], "spoke a reply the user had already interrupted"


def test_the_active_voice_is_named_in_status(tmp_path):
    config = VoiceConfig(
        piper_exe=str(tmp_path / "piper.exe"),
        piper_model=str(tmp_path / "en_US-ryan-high.onnx"),
        speech_rate=1.25,
    )
    session = VoiceSession(config, tts=FakeTTS())
    session.tts.name = "piper"
    described = session.describe_voice()
    assert "piper" in described and "en_US-ryan-high" in described
    assert "1.25x" in described


def test_status_says_so_when_there_is_no_speech_output():
    assert VoiceSession(VoiceConfig()).describe_voice() == "no speech output"
