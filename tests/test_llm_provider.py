"""The Gemini provider's translation layer, without touching the network."""

from __future__ import annotations

import pytest

from cronus.config import LLMConfig
from cronus.errors import ProviderError
from cronus.llm.base import Message, ToolSchema
from cronus.llm.gemini import GeminiProvider, _is_retryable, _retry_delay


class FakeModels:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, models):
        self.models = models


def make_part(text=None, call=None, signature=None, thought=False):
    from google.genai import types

    if call is not None:
        part = types.Part.from_function_call(name=call[0], args=call[1])
    else:
        part = types.Part(text=text)
    if signature:
        part.thought_signature = signature
    part.thought = thought
    return part


def make_response(parts, finish_reason="STOP"):
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ]
    )


@pytest.fixture
def llm_config():
    return LLMConfig(api_key="k", model="test-model")


def test_text_responses_are_translated(llm_config):
    models = FakeModels([make_response([make_part(text="Hello.")])])
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    response = provider.generate([Message(role="user", content="hi")])
    assert response.text == "Hello." and not response.wants_tools


def test_tool_calls_are_translated(llm_config):
    models = FakeModels(
        [make_response([make_part(call=("get_weather", {"city": "Oslo"}))])]
    )
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    response = provider.generate([Message(role="user", content="weather?")])
    assert response.wants_tools
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Oslo"}


def test_thought_signatures_survive_the_round_trip(llm_config):
    """Gemini rejects a follow-up whose function calls lost their signature."""
    signature = b"opaque-signature-bytes"
    models = FakeModels(
        [
            make_response([make_part(call=("echo", {"x": 1}), signature=signature)]),
            make_response([make_part(text="done")]),
        ]
    )
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    first = provider.generate([Message(role="user", content="go")])
    assert first.tool_calls[0].provider_state["thought_signature"] == signature

    provider.generate(
        [
            Message(role="user", content="go"),
            Message(role="assistant", tool_calls=first.tool_calls),
            Message(role="tool", tool_name="echo", content="ok"),
        ]
    )
    sent = models.requests[1]["contents"]
    assert sent[1].parts[0].thought_signature == signature


def test_raw_reasoning_is_never_exposed(llm_config):
    models = FakeModels(
        [
            make_response(
                [
                    make_part(text="internal deliberation", thought=True),
                    make_part(text="The answer is 4."),
                ]
            )
        ]
    )
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    response = provider.generate([Message(role="user", content="2+2")])
    assert response.text == "The answer is 4."
    assert "deliberation" not in response.text


def test_tool_results_are_sent_as_function_responses(llm_config):
    models = FakeModels([make_response([make_part(text="ok")])])
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    provider.generate(
        [
            Message(role="user", content="go"),
            Message(role="tool", tool_name="get_weather", content="Oslo: 4C"),
        ]
    )
    parts = models.requests[0]["contents"][1].parts
    assert parts[0].function_response.name == "get_weather"
    assert parts[0].function_response.response == {"result": "Oslo: 4C"}


def test_tools_are_declared_with_automatic_calling_disabled(llm_config):
    """Cronus executes tools itself; the SDK must never call a function."""
    models = FakeModels([make_response([make_part(text="ok")])])
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    provider.generate(
        [Message(role="user", content="hi")],
        tools=[ToolSchema(name="echo", description="Echo.", parameters={"type": "object"})],
    )
    sent = models.requests[0]["config"]
    assert sent.automatic_function_calling.disable is True
    assert sent.tools[0].function_declarations[0].name == "echo"


def test_an_empty_candidate_list_does_not_crash(llm_config):
    from google.genai import types

    models = FakeModels([types.GenerateContentResponse(candidates=[])])
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    assert provider.generate([Message(role="user", content="hi")]).text == ""


def test_errors_become_provider_errors_with_plain_messages(llm_config):
    models = FakeModels(error=RuntimeError("401 UNAUTHENTICATED: bad API key"))
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    with pytest.raises(ProviderError) as info:
        provider.generate([Message(role="user", content="hi")])
    assert "GOOGLE_API_KEY" in info.value.user_message


def test_transient_failures_are_retried(llm_config, monkeypatch):
    monkeypatch.setattr("cronus.llm.gemini.time.sleep", lambda seconds: None)

    class Flaky(FakeModels):
        def generate_content(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return make_response([make_part(text="recovered")])

    provider = GeminiProvider(llm_config, client=FakeClient(Flaky()))
    assert provider.generate([Message(role="user", content="hi")]).text == "recovered"


def test_a_daily_quota_is_not_retried(llm_config, monkeypatch):
    """A per-day limit will not clear inside a request, so fail fast and say so."""
    slept = []
    monkeypatch.setattr("cronus.llm.gemini.time.sleep", slept.append)
    error = RuntimeError(
        "429 RESOURCE_EXHAUSTED {'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'retryDelay': '24s'}"
    )
    models = FakeModels(error=error)
    provider = GeminiProvider(llm_config, client=FakeClient(models))
    with pytest.raises(ProviderError) as info:
        provider.generate([Message(role="user", content="hi")])
    assert slept == []
    assert len(models.requests) == 1
    assert "free Gemini quota" in info.value.user_message


def test_the_servers_retry_hint_is_preferred_over_backoff():
    error = RuntimeError("429 {'retryDelay': '24s'}")
    assert _retry_delay(error, attempt=1) == pytest.approx(25.0)
    assert _retry_delay(RuntimeError("503"), attempt=1) < 2.0


def test_retry_hint_is_capped():
    assert _retry_delay(RuntimeError("{'retryDelay': '9999s'}"), 1) <= 35.0


@pytest.mark.parametrize(
    "message,retryable",
    [
        ("503 UNAVAILABLE", True),
        ("500 internal", True),
        ("400 INVALID_ARGUMENT", False),
        ("429 PerMinute rate limit", True),
    ],
)
def test_retryable_classification(message, retryable):
    assert _is_retryable(RuntimeError(message)) is retryable
