"""Voice-mode activation and speech-input diagnostics.

No microphone, no audio device, and no network: the STT provider is driven
through its interface and the CLI is exercised with fakes.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from cronus.config import VoiceConfig, load_config
from cronus.interfaces import cli as cli_module
from cronus.voice.base import ListenOutcome, SpeechToText


# ======================================================================
# Configuration
# ======================================================================
def test_voice_defaults_to_off(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert load_config(env_file=None).voice.enabled is False


@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False),
                                          ("1", True), ("0", False),
                                          ("yes", True), ("no", False)])
def test_cronus_voice_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_VOICE", raw)
    assert load_config(env_file=None).voice.enabled is expected


# ======================================================================
# Startup mode selection
# ======================================================================
class _FakeCLI:
    """Captures how main() decided to start, without touching audio."""

    instances: list["_FakeCLI"] = []

    def __init__(self, cronus, **kwargs):
        self.cronus = cronus
        self.enable_voice_calls: list[bool] = []
        self.responded: list[str] = []
        self.ran = False
        self.tts = None
        self.session = None
        _FakeCLI.instances.append(self)

    def enable_voice(self, *, listen: bool) -> None:
        self.enable_voice_calls.append(listen)

    def respond(self, text: str) -> None:
        self.responded.append(text)

    def run(self) -> int:
        self.ran = True
        return 0


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """Run main() with the assistant and console stubbed out."""
    _FakeCLI.instances.clear()
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_DATA_DIR", str(tmp_path))

    built: dict[str, Any] = {}

    class FakeCronus:
        def __init__(self):
            self.config = None
            self.scheduler = type("S", (), {"start": lambda self: None})()

        def start(self):
            pass

        def shutdown(self):
            pass

    def fake_build(config, *, voice_mode=False, provider=None):
        built["voice_mode"] = voice_mode
        cronus = FakeCronus()
        cronus.config = config
        return cronus

    monkeypatch.setattr(cli_module, "build", fake_build)
    monkeypatch.setattr(cli_module, "CronusCLI", _FakeCLI)

    def run(argv):
        cli_module.main(argv)
        return _FakeCLI.instances[-1], built

    return run


def test_no_flags_and_no_config_stays_in_text_mode(launch, monkeypatch):
    monkeypatch.setenv("CRONUS_VOICE", "false")
    cli, built = launch(["--no-scheduler"])
    assert cli.enable_voice_calls == []
    assert built["voice_mode"] is False


def test_the_voice_flag_still_enables_voice(launch, monkeypatch):
    monkeypatch.setenv("CRONUS_VOICE", "false")
    cli, built = launch(["--voice", "--no-scheduler"])
    assert cli.enable_voice_calls == [True]
    assert built["voice_mode"] is True


def test_cronus_voice_enables_voice_without_the_flag(launch, monkeypatch):
    """Regression: CRONUS_VOICE was parsed and then ignored entirely."""
    monkeypatch.setenv("CRONUS_VOICE", "true")
    cli, built = launch(["--no-scheduler"])
    assert cli.enable_voice_calls == [True], "voice input was not enabled"
    assert built["voice_mode"] is True


def test_the_speak_flag_gives_speech_output_but_typed_input(launch, monkeypatch):
    monkeypatch.setenv("CRONUS_VOICE", "false")
    cli, built = launch(["--speak", "--no-scheduler"])
    assert cli.enable_voice_calls == [False], "should speak but not listen"
    assert built["voice_mode"] is False


def test_the_flag_and_the_setting_together_do_not_conflict(launch, monkeypatch):
    monkeypatch.setenv("CRONUS_VOICE", "true")
    cli, built = launch(["--voice", "--no-scheduler"])
    assert cli.enable_voice_calls == [True]


def test_speak_flag_with_voice_configured_still_listens(launch, monkeypatch):
    """CRONUS_VOICE=true means voice input; --speak must not downgrade it."""
    monkeypatch.setenv("CRONUS_VOICE", "true")
    cli, built = launch(["--speak", "--no-scheduler"])
    assert cli.enable_voice_calls == [True]


def test_one_shot_messages_skip_the_interactive_loop(launch, monkeypatch):
    monkeypatch.setenv("CRONUS_VOICE", "true")
    cli, _ = launch(["-m", "hello", "--no-scheduler"])
    assert cli.responded == ["hello"] and cli.ran is False


# ======================================================================
# STT failure reporting
# ======================================================================
class _FailingSTT(SpeechToText):
    """An STT provider whose outcome can be scripted."""

    name = "fake"

    def __init__(self, outcome: ListenOutcome, detail: str | None = None):
        self._outcome = outcome
        self.last_outcome = outcome
        self.last_error = detail

    @property
    def available(self) -> bool:
        return True

    def listen(self, timeout: float | None = None) -> str:
        self.last_outcome = self._outcome
        return "hello" if self._outcome is ListenOutcome.HEARD else ""


def _build_stt_with(monkeypatch, sr_module, config=None):
    """Construct the real provider against a stubbed speech_recognition."""
    from cronus.voice import stt as stt_module

    monkeypatch.setitem(__import__("sys").modules, "speech_recognition", sr_module)
    provider = stt_module.SpeechRecognitionSTT(config or VoiceConfig())
    return provider


class _StubSR:
    """Minimal stand-in for the speech_recognition module."""

    class WaitTimeoutError(Exception):
        pass

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    def __init__(self, on_listen=None, on_recognize=None):
        self._on_listen = on_listen
        self._on_recognize = on_recognize
        outer = self

        class Recognizer:
            energy_threshold = 100
            dynamic_energy_threshold = True
            pause_threshold = 0.8
            non_speaking_duration = 0.4

            def adjust_for_ambient_noise(self, source, duration=0.5):
                pass

            def listen(self, source, timeout=None, phrase_time_limit=None):
                if outer._on_listen is not None:
                    raise outer._on_listen
                return object()

            def recognize_google(self, audio):
                if outer._on_recognize is not None:
                    raise outer._on_recognize
                return "hello there"

        class Microphone:
            def __init__(self, device_index=None):
                self.device_index = device_index

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        self.Recognizer = Recognizer
        self.Microphone = Microphone


def test_silence_is_reported_as_no_speech(monkeypatch):
    stub = _StubSR(on_listen=_StubSR.WaitTimeoutError())
    provider = _build_stt_with(monkeypatch, stub)
    assert provider.listen() == ""
    assert provider.last_outcome is ListenOutcome.NO_SPEECH
    assert provider.last_outcome.is_failure is False


def test_a_microphone_error_is_distinguished_from_silence(monkeypatch):
    stub = _StubSR(on_listen=OSError("device disconnected"))
    provider = _build_stt_with(monkeypatch, stub)
    assert provider.listen() == ""
    assert provider.last_outcome is ListenOutcome.MIC_ERROR
    assert provider.last_outcome.is_failure is True
    assert "device disconnected" in provider.last_error


def test_unintelligible_audio_is_not_a_failure(monkeypatch):
    stub = _StubSR(on_recognize=_StubSR.UnknownValueError())
    provider = _build_stt_with(monkeypatch, stub)
    assert provider.listen() == ""
    assert provider.last_outcome is ListenOutcome.NOT_UNDERSTOOD
    assert provider.last_outcome.is_failure is False


def test_an_unreachable_recogniser_is_a_service_error(monkeypatch):
    stub = _StubSR(on_recognize=_StubSR.RequestError("connection refused"))
    provider = _build_stt_with(monkeypatch, stub)
    assert provider.listen() == ""
    assert provider.last_outcome is ListenOutcome.SERVICE_ERROR
    assert "connection refused" in provider.last_error


def test_a_successful_transcription_clears_the_error(monkeypatch):
    stub = _StubSR()
    provider = _build_stt_with(monkeypatch, stub)
    provider.last_error = "stale"
    assert provider.listen() == "hello there"
    assert provider.last_outcome is ListenOutcome.HEARD
    assert provider.last_error is None


def test_an_unavailable_device_reports_rather_than_returning_silence():
    from cronus.voice.stt import SpeechRecognitionSTT

    provider = SpeechRecognitionSTT.__new__(SpeechRecognitionSTT)
    provider.config = VoiceConfig()
    provider._recognizer = None
    provider._microphone = None
    provider._calibrated = False
    provider._error = "no usable microphone"
    provider.last_outcome = ListenOutcome.NO_SPEECH
    provider.last_error = None

    assert provider.listen() == ""
    assert provider.last_outcome is ListenOutcome.MIC_ERROR


# ======================================================================
# What the user actually sees
# ======================================================================
@pytest.fixture
def voice_cli(config, monkeypatch, tmp_path):
    """A CronusCLI with a scriptable STT and a recording console."""
    from cronus.core.events import EventEmitter

    printed: list[str] = []

    class RecordingConsole:
        def print(self, *args, **kwargs):
            printed.append(" ".join(str(a) for a in args))

        def input(self, *args, **kwargs):
            return ""

        def status(self, *args, **kwargs):
            class Status:
                def start(self): pass
                def stop(self): pass
                def update(self, *a): pass

            return Status()

    class FakeScheduler:
        on_due = None

    class FakeAssistant:
        voice_mode = True

        def __init__(self):
            from cronus.security.confirmation import ConfirmationManager

            self.confirmations = ConfirmationManager()

    class FakeCronus:
        def __init__(self):
            self.config = config
            self.emitter = EventEmitter()
            self.assistant = FakeAssistant()
            self.scheduler = FakeScheduler()
            self.registry = []
            self.memory = None
            self.profile = None

    cli = cli_module.CronusCLI(FakeCronus())
    cli.console = RecordingConsole()
    cli.ui.console = cli.console
    cli.voice_input = True
    return cli, printed


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (ListenOutcome.NO_SPEECH, "No speech detected."),
        (ListenOutcome.NOT_UNDERSTOOD, "I didn't catch that."),
        (ListenOutcome.MIC_ERROR, "Microphone unavailable."),
        (ListenOutcome.SERVICE_ERROR, "Speech recognition failed."),
    ],
)
def test_each_outcome_gets_its_own_message(voice_cli, outcome, expected):
    cli, printed = voice_cli
    cli.stt = _FailingSTT(outcome)
    cli._report_listen_failure()
    assert any(expected in line for line in printed), printed


def test_hard_failures_come_with_an_actionable_hint(voice_cli):
    cli, printed = voice_cli
    cli.stt = _FailingSTT(ListenOutcome.MIC_ERROR)
    cli._report_listen_failure()
    assert any("CRONUS_MIC_INDEX" in line for line in printed)

    printed.clear()
    cli.stt = _FailingSTT(ListenOutcome.SERVICE_ERROR)
    cli._report_listen_failure()
    assert any("internet connection" in line for line in printed)


def test_internal_details_stay_hidden_unless_debugging(voice_cli, monkeypatch):
    cli, printed = voice_cli
    cli.stt = _FailingSTT(ListenOutcome.MIC_ERROR, detail="OSError: errno -9996")

    monkeypatch.setattr(cli_module.log, "isEnabledFor", lambda level: False)
    cli._report_listen_failure()
    assert not any("errno -9996" in line for line in printed)

    printed.clear()
    cli._listen_failures = 0
    monkeypatch.setattr(
        cli_module.log, "isEnabledFor", lambda level: level == logging.DEBUG
    )
    cli._report_listen_failure()
    assert any("errno -9996" in line for line in printed)


def test_repeated_hard_failures_fall_back_to_typing(voice_cli):
    """A dead microphone must not be reported forever in a loop."""
    cli, printed = voice_cli
    cli.stt = _FailingSTT(ListenOutcome.MIC_ERROR)
    for _ in range(cli_module._MAX_LISTEN_FAILURES):
        cli._report_listen_failure()

    assert cli.voice_input is False
    assert any("Switching to typed input" in line for line in printed)


def test_silence_does_not_count_towards_the_failure_limit(voice_cli):
    cli, printed = voice_cli
    cli.stt = _FailingSTT(ListenOutcome.NO_SPEECH)
    for _ in range(cli_module._MAX_LISTEN_FAILURES * 3):
        cli._report_listen_failure()
    assert cli.voice_input is True


def test_a_success_resets_the_failure_count(voice_cli):
    cli, printed = voice_cli
    cli.stt = _FailingSTT(ListenOutcome.MIC_ERROR)
    cli._report_listen_failure()
    assert cli._listen_failures == 1

    cli.stt = _FailingSTT(ListenOutcome.NO_SPEECH)
    cli._report_listen_failure()
    assert cli._listen_failures == 0


def test_dictation_waits_a_bounded_time(monkeypatch):
    """An unbounded wait can never report 'no speech detected'."""
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert load_config(env_file=None).voice.listen_timeout > 0

    monkeypatch.setenv("CRONUS_LISTEN_TIMEOUT", "4.5")
    assert load_config(env_file=None).voice.listen_timeout == 4.5


def test_the_listen_timeout_is_passed_to_the_provider(config):
    """Regression: dictation called listen() with no timeout at all."""
    from dataclasses import replace

    from cronus.voice.session import VoiceSession

    captured: list = []

    class Recording(_FailingSTT):
        def listen(self, timeout=None):
            captured.append(timeout)
            return ""

    voice = replace(config.voice, listen_timeout=7.0)
    session = VoiceSession(voice, stt=Recording(ListenOutcome.NO_SPEECH))
    session.next_utterance()
    assert captured == [7.0]
