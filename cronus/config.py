"""Environment-driven configuration.

All configuration enters the process here. Secrets live in the environment
(loaded from ``.env``) and are never written to logs, prompts, or the model.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSEY:
        return False
    raise ConfigError(
        f"{name} must be true/false, got {raw!r}",
        user_message=f"The setting {name} should be true or false.",
    )


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{name} must be an integer, got {raw!r}",
            user_message=f"The setting {name} should be a whole number.",
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{name} must be a number, got {raw!r}",
            user_message=f"The setting {name} should be a number.",
        ) from exc


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    if value < 1:
        raise ConfigError(
            f"{name} must be at least 1, got {value}",
            user_message=f"The setting {name} has to be 1 or more.",
        )
    return value


def _env_opt_int(name: str) -> int | None:
    raw = _env(name)
    if raw is None:
        return None
    return _env_int(name, 0)


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = _env(name)
    if raw is None:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _default_file_roots() -> list[str]:
    home = Path.home()
    candidates = [home / "Documents", home / "Desktop", home / "Downloads"]
    return [str(p) for p in candidates if p.is_dir()]


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = "gemini-flash-latest"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    request_timeout: float = 45.0


class VoiceMode(enum.Enum):
    """How Cronus decides the user is talking to it."""

    PUSH_TO_TALK = "push_to_talk"   # press Enter before each turn
    CONTINUOUS = "continuous"       # listen again after every reply
    WAKE_WORD = "wake_word"         # stay idle until the wake phrase

    @classmethod
    def parse(cls, raw: str | None) -> "VoiceMode":
        if not raw:
            return cls.CONTINUOUS
        key = raw.strip().lower().replace("-", "_").replace(" ", "_")
        for mode in cls:
            if mode.value == key:
                return mode
        raise ConfigError(
            f"CRONUS_VOICE_MODE must be one of "
            f"{', '.join(m.value for m in cls)}, got {raw!r}",
            user_message=(
                "CRONUS_VOICE_MODE should be push_to_talk, continuous, or wake_word."
            ),
        )


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False
    mode: VoiceMode = VoiceMode.CONTINUOUS
    stt_provider: str = "google_web"
    tts_provider: str = "piper"
    piper_exe: str | None = None
    piper_model: str | None = None
    speech_rate: float = 1.0
    mic_index: int | None = None
    energy_threshold: int | None = None
    # --- endpointing: when Cronus decides the user has finished speaking ---
    #: Longest single utterance it will record.
    phrase_time_limit: float = 30.0
    #: How long to wait for the user to start speaking at all.
    listen_timeout: float = 15.0
    #: Silence needed to treat a phrase as finished. Raising this lets people
    #: pause mid-sentence without being cut off.
    pause_threshold: float = 1.0
    #: Silence kept on each side of the captured phrase.
    non_speaking_duration: float = 0.5
    #: Ignore blips shorter than this; stops coughs becoming requests.
    min_speech_seconds: float = 0.3

    # --- barge-in ---
    barge_in: bool = True
    #: How much louder than the ambient threshold speech must be to interrupt.
    #: Above 1.0 so Cronus's own voice through speakers is less likely to
    #: interrupt itself.
    barge_in_sensitivity: float = 2.5

    wake_word_enabled: bool = False
    wake_word: str = "hey cronus"

    @property
    def piper_available(self) -> bool:
        return bool(
            self.piper_exe
            and self.piper_model
            and Path(self.piper_exe).is_file()
            and Path(self.piper_model).is_file()
        )


@dataclass(frozen=True)
class EmailConfig:
    user: str | None = None
    app_password: str | None = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    timeout: float = 20.0

    @property
    def configured(self) -> bool:
        return bool(self.user and self.app_password)


@dataclass(frozen=True)
class SecurityConfig:
    file_roots: tuple[Path, ...] = ()
    max_read_bytes: int = 200_000
    permission_overrides: dict[str, str] = field(default_factory=dict)
    confirmation_timeout: float = 120.0
    allowed_apps: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryConfig:
    max_recall: int = 6
    min_relevance: float = 0.15
    max_stored: int = 500


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    voice: VoiceConfig
    email: EmailConfig
    security: SecurityConfig
    memory: MemoryConfig
    data_dir: Path
    log_level: str = "INFO"
    max_tool_iterations: int = 8
    tool_timeout: float = 30.0
    context_char_budget: int = 12_000
    user_name: str | None = None
    timezone: str | None = None
    location: str | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cronus.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "cronus.log"


def _parse_pairs(raw: list[str], setting: str) -> dict[str, str]:
    """Parse ``key=value`` pairs out of a comma-separated setting."""
    pairs: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ConfigError(
                f"{setting} entry {item!r} must look like key=value",
                user_message=f"The setting {setting} is malformed.",
            )
        key, _, value = item.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            pairs[key] = value
    return pairs


def load_config(
    env_file: str | os.PathLike[str] | None = ".env",
    *,
    require_api_key: bool = True,
) -> Config:
    """Build a :class:`Config` from the environment.

    Raises :class:`ConfigError` with an actionable message when something
    required is missing. Pass ``require_api_key=False`` for offline tooling
    and tests.
    """
    if env_file is not None and Path(env_file).is_file():
        load_dotenv(env_file, override=False)

    api_key = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
    if not api_key and require_api_key:
        raise ConfigError(
            "GOOGLE_API_KEY is not set",
            user_message=(
                "I need a Gemini API key. Copy .env.example to .env, set "
                "GOOGLE_API_KEY, then start me again."
            ),
        )

    data_dir = Path(_env("CRONUS_DATA_DIR") or (Path.home() / ".cronus")).expanduser()

    resolved_roots: list[Path] = []
    for root in _env_list("CRONUS_FILE_ROOTS") or _default_file_roots():
        try:
            resolved_roots.append(Path(root).expanduser().resolve(strict=False))
        except OSError:
            continue

    llm = LLMConfig(
        api_key=api_key or "",
        model=_env("CRONUS_MODEL", "gemini-flash-latest"),
        temperature=_env_float("CRONUS_TEMPERATURE", 0.7),
        max_output_tokens=_env_int("CRONUS_MAX_OUTPUT_TOKENS", 2048),
        request_timeout=_env_float("CRONUS_LLM_TIMEOUT", 45.0),
    )

    wake_word_enabled = _env_bool("CRONUS_WAKE_WORD_ENABLED", False)
    raw_mode = _env("CRONUS_VOICE_MODE")
    if raw_mode is None and wake_word_enabled:
        # Predates CRONUS_VOICE_MODE; honour it when no mode is stated.
        voice_mode = VoiceMode.WAKE_WORD
    else:
        voice_mode = VoiceMode.parse(raw_mode)
    pause_threshold = _env_float("CRONUS_PAUSE_THRESHOLD", 1.0)
    non_speaking = _env_float("CRONUS_NON_SPEAKING_DURATION", 0.5)
    if non_speaking > pause_threshold:
        # SpeechRecognition requires this ordering; a silent clamp here beats
        # an obscure failure inside the recogniser later.
        raise ConfigError(
            f"CRONUS_NON_SPEAKING_DURATION ({non_speaking}) cannot exceed "
            f"CRONUS_PAUSE_THRESHOLD ({pause_threshold})",
            user_message=(
                "CRONUS_NON_SPEAKING_DURATION must be less than or equal to "
                "CRONUS_PAUSE_THRESHOLD."
            ),
        )

    voice = VoiceConfig(
        enabled=_env_bool("CRONUS_VOICE", False),
        mode=voice_mode,
        stt_provider=_env("CRONUS_STT_PROVIDER", "google_web"),
        tts_provider=_env("CRONUS_TTS_PROVIDER", "piper"),
        piper_exe=_env("CRONUS_PIPER_EXE"),
        piper_model=_env("CRONUS_PIPER_MODEL"),
        speech_rate=_env_float("CRONUS_SPEECH_RATE", 1.0),
        mic_index=_env_opt_int("CRONUS_MIC_INDEX"),
        energy_threshold=_env_opt_int("CRONUS_MIC_ENERGY_THRESHOLD"),
        phrase_time_limit=_env_float("CRONUS_PHRASE_TIME_LIMIT", 30.0),
        listen_timeout=_env_float("CRONUS_LISTEN_TIMEOUT", 15.0),
        pause_threshold=pause_threshold,
        non_speaking_duration=non_speaking,
        min_speech_seconds=_env_float("CRONUS_MIN_SPEECH_SECONDS", 0.3),
        barge_in=_env_bool("CRONUS_BARGE_IN", True),
        barge_in_sensitivity=_env_float("CRONUS_BARGE_IN_SENSITIVITY", 2.5),
        wake_word_enabled=wake_word_enabled or voice_mode is VoiceMode.WAKE_WORD,
        wake_word=(_env("CRONUS_WAKE_WORD", "hey cronus") or "hey cronus").lower(),
    )

    email = EmailConfig(
        user=_env("GMAIL_USER"),
        app_password=_env("GMAIL_APP_PASSWORD"),
        smtp_host=_env("CRONUS_SMTP_HOST", "smtp.gmail.com"),
        smtp_port=_env_int("CRONUS_SMTP_PORT", 587),
        timeout=_env_float("CRONUS_SMTP_TIMEOUT", 20.0),
    )

    security = SecurityConfig(
        file_roots=tuple(resolved_roots),
        max_read_bytes=_env_int("CRONUS_MAX_READ_BYTES", 200_000),
        permission_overrides=_parse_pairs(
            _env_list("CRONUS_TOOL_PERMISSIONS"), "CRONUS_TOOL_PERMISSIONS"
        ),
        confirmation_timeout=_env_float("CRONUS_CONFIRMATION_TIMEOUT", 120.0),
        allowed_apps=_parse_pairs(_env_list("CRONUS_ALLOWED_APPS"), "CRONUS_ALLOWED_APPS"),
    )

    memory = MemoryConfig(
        max_recall=_env_positive_int("CRONUS_MEMORY_RECALL", 6),
        min_relevance=_env_float("CRONUS_MEMORY_MIN_RELEVANCE", 0.15),
        max_stored=_env_int("CRONUS_MEMORY_MAX", 500),
    )

    return Config(
        llm=llm,
        voice=voice,
        email=email,
        security=security,
        memory=memory,
        data_dir=data_dir,
        log_level=(_env("CRONUS_LOG_LEVEL", "INFO") or "INFO").upper(),
        max_tool_iterations=_env_positive_int("CRONUS_MAX_TOOL_ITERATIONS", 8),
        tool_timeout=_env_float("CRONUS_TOOL_TIMEOUT", 30.0),
        context_char_budget=_env_positive_int("CRONUS_CONTEXT_BUDGET", 12_000),
        user_name=_env("CRONUS_USER_NAME"),
        timezone=_env("CRONUS_TIMEZONE"),
        location=_env("CRONUS_LOCATION"),
    )
