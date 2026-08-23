"""Shared fixtures.

Every test here runs without an API key, without a network, and without a
microphone. The model is a scripted fake, so the agent loop is exercised
deterministically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from cronus.config import (
    Config,
    EmailConfig,
    LLMConfig,
    MemoryConfig,
    SecurityConfig,
    VoiceConfig,
)
from cronus.core.runtime import Assistant
from cronus.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSchema
from cronus.memory.store import MemoryStore
from cronus.profile import UserProfile
from cronus.security.confirmation import ConfirmationManager
from cronus.security.paths import PathGuard
from cronus.storage.db import Database
from cronus.tools.base import RiskLevel, Tool, ToolResult, object_schema
from cronus.tools.registry import ToolRegistry


class FakeProvider(LLMProvider):
    """Replays a scripted list of responses and records what it was asked."""

    name = "fake"

    def __init__(self, script: Sequence[LLMResponse] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Message],
        *,
        system_instruction: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system_instruction": system_instruction or "",
                "tools": list(tools or []),
            }
        )
        if not self.script:
            return LLMResponse(text="(no more scripted responses)")
        return self.script.pop(0)

    @property
    def last_system_instruction(self) -> str:
        return self.calls[-1]["system_instruction"] if self.calls else ""


def text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def tool_response(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        llm=LLMConfig(api_key="test-key", model="fake-model"),
        voice=VoiceConfig(),
        email=EmailConfig(user="me@example.com", app_password="secret"),
        security=SecurityConfig(
            file_roots=(tmp_path / "workspace",),
            max_read_bytes=10_000,
            confirmation_timeout=5.0,
        ),
        memory=MemoryConfig(),
        data_dir=tmp_path / "data",
        max_tool_iterations=4,
        tool_timeout=5.0,
    )


@pytest.fixture
def workspace(config: Config) -> Path:
    root = config.security.file_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def database() -> Database:
    db = Database(":memory:")
    yield db
    db.close()


@pytest.fixture
def memory(database: Database, config: Config) -> MemoryStore:
    return MemoryStore(database, config.memory)


@pytest.fixture
def profile(database: Database) -> UserProfile:
    return UserProfile(database)


@pytest.fixture
def registry(config: Config) -> ToolRegistry:
    reg = ToolRegistry(default_timeout=config.tool_timeout)
    yield reg
    reg.shutdown()


def echo_tool(text: str = "hi") -> ToolResult:
    return ToolResult(content=f"echo: {text}")


@pytest.fixture
def echo() -> Tool:
    return Tool(
        name="echo",
        description="Echo text back.",
        parameters=object_schema({"text": {"type": "string"}}, required=["text"]),
        handler=echo_tool,
        risk=RiskLevel.SAFE,
    )


@pytest.fixture
def make_assistant(config: Config, registry: ToolRegistry, memory: MemoryStore,
                   profile: UserProfile):
    """Builds an Assistant wired to a scripted provider."""

    def _build(script: Sequence[LLMResponse], *, approve: bool = True, **kwargs: Any):
        confirmations = ConfirmationManager(
            handler=lambda request: approve, timeout=5.0
        )
        assistant = Assistant(
            config,
            FakeProvider(script),
            registry,
            memory=memory,
            profile=profile,
            paths=PathGuard(config.security.file_roots, config.security.max_read_bytes),
            confirmations=confirmations,
            **kwargs,
        )
        return assistant

    return _build


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    """Keep the developer's real .env out of every test."""
    for name in (
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GMAIL_USER", "GMAIL_APP_PASSWORD",
        "CRONUS_MODEL", "CRONUS_DATA_DIR", "CRONUS_FILE_ROOTS", "CRONUS_VOICE",
        "CRONUS_TOOL_PERMISSIONS", "CRONUS_LOG_LEVEL", "CRONUS_TEMPERATURE",
        "CRONUS_TIMEZONE", "CRONUS_LOCATION", "CRONUS_VOICE_MODE",
        "CRONUS_PHRASE_TIME_LIMIT", "CRONUS_LISTEN_TIMEOUT",
        "CRONUS_PAUSE_THRESHOLD", "CRONUS_NON_SPEAKING_DURATION",
        "CRONUS_MIN_SPEECH_SECONDS", "CRONUS_BARGE_IN",
        "CRONUS_BARGE_IN_SENSITIVITY", "CRONUS_WAKE_WORD_ENABLED",
        "CRONUS_MAX_TOOL_ITERATIONS", "CRONUS_CONTEXT_BUDGET",
        "CRONUS_MEMORY_RECALL", "CRONUS_PIPER_EXE", "CRONUS_PIPER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
