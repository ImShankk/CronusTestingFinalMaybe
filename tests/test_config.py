"""Configuration loading and validation."""

from __future__ import annotations

import pytest

from cronus.config import load_config
from cronus.errors import ConfigError
from cronus.logging_setup import RedactionFilter


def test_missing_api_key_is_an_actionable_error(monkeypatch):
    with pytest.raises(ConfigError) as info:
        load_config(env_file=None)
    assert "GOOGLE_API_KEY" in info.value.user_message


def test_offline_loading_skips_the_key_requirement():
    assert load_config(env_file=None, require_api_key=False).llm.api_key == ""


def test_defaults_are_sensible(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    config = load_config(env_file=None)
    assert config.llm.model == "gemini-flash-latest"
    assert config.max_tool_iterations > 1
    assert config.voice.enabled is False


def test_environment_overrides_are_read(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_MODEL", "some-other-model")
    monkeypatch.setenv("CRONUS_TEMPERATURE", "0.2")
    monkeypatch.setenv("CRONUS_VOICE", "true")
    monkeypatch.setenv("CRONUS_DATA_DIR", str(tmp_path))
    config = load_config(env_file=None)
    assert config.llm.model == "some-other-model"
    assert config.llm.temperature == 0.2
    assert config.voice.enabled is True
    assert config.db_path == tmp_path / "cronus.db"


def test_gemini_api_key_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert load_config(env_file=None).llm.api_key == "k"


def test_a_bad_boolean_is_rejected_clearly(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_VOICE", "sometimes")
    with pytest.raises(ConfigError, match="true/false"):
        load_config(env_file=None)


def test_a_bad_number_is_rejected_clearly(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_MAX_TOOL_ITERATIONS", "lots")
    with pytest.raises(ConfigError, match="integer"):
        load_config(env_file=None)


def test_permission_overrides_are_parsed(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_TOOL_PERMISSIONS", "send_email=allow, delete_file=block")
    overrides = load_config(env_file=None).security.permission_overrides
    assert overrides == {"send_email": "allow", "delete_file": "block"}


def test_malformed_permission_overrides_are_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_TOOL_PERMISSIONS", "send_email")
    with pytest.raises(ConfigError, match="key=value"):
        load_config(env_file=None)


def test_file_roots_are_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_FILE_ROOTS", str(tmp_path))
    assert load_config(env_file=None).security.file_roots == (tmp_path.resolve(),)


def test_piper_availability_reflects_the_filesystem(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CRONUS_PIPER_EXE", str(tmp_path / "piper.exe"))
    monkeypatch.setenv("CRONUS_PIPER_MODEL", str(tmp_path / "voice.onnx"))
    assert load_config(env_file=None).voice.piper_available is False

    (tmp_path / "piper.exe").write_text("")
    (tmp_path / "voice.onnx").write_text("")
    assert load_config(env_file=None).voice.piper_available is True


# ----------------------------------------------------------------------
# Secret handling
# ----------------------------------------------------------------------
def test_known_secrets_are_scrubbed_from_logs(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyVerySecretValue123")
    scrubbed = RedactionFilter.redact("calling with AIzaSyVerySecretValue123 now")
    assert "AIzaSyVerySecretValue123" not in scrubbed
    assert "<redacted>" in scrubbed


def test_key_shaped_text_is_scrubbed():
    assert "hunter2" not in RedactionFilter.redact("password=hunter2")
    assert "abc123" not in RedactionFilter.redact('"api_key": abc123')
