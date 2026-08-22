"""Gemini implementation of :class:`~cronus.llm.base.LLMProvider`."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Sequence

from ..config import LLMConfig
from ..errors import ProviderError
from ..logging_setup import get_logger
from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolSchema

log = get_logger("llm.gemini")

# Errors worth telling the user about in different words.
_AUTH_HINTS = ("api key", "unauthenticated", "permission denied", "401", "403")
_QUOTA_HINTS = ("quota", "rate limit", "resource_exhausted", "429")
# Transient server-side conditions worth one or two automatic retries.
_RETRY_HINTS = ("503", "unavailable", "500", "internal", "deadline", "timeout", "429")
_MAX_ATTEMPTS = 3
# Gemini tells us how long to wait when we are rate limited; honour it up to
# a bound so a free-tier per-minute limit recovers instead of failing fast.
_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")
_MAX_RETRY_WAIT = 35.0
# A per-day quota will not recover inside a request, so never wait on one.
_DAILY_QUOTA_HINT = "perday"


class GeminiProvider(LLMProvider):
    """Wraps ``google-genai``. The client is created once and reused."""

    name = "gemini"

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self.config = config
        if client is not None:
            self._client = client
        else:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ProviderError(
                    "google-genai is not installed",
                    user_message="My Gemini library is missing. Run pip install -r requirements.txt.",
                ) from exc
            self._client = genai.Client(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message],
        *,
        system_instruction: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        from google.genai import types

        contents = self._to_contents(messages)
        config_kwargs: dict[str, Any] = {
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_output_tokens": self.config.max_output_tokens,
            "http_options": types.HttpOptions(
                timeout=int(self.config.request_timeout * 1000)
            ),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
            # Cronus executes tools itself, through the registry and the
            # permission layer. The SDK must never call a Python function.
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        log.debug(
            "model request model=%s messages=%d tools=%d",
            self.config.model,
            len(contents),
            len(tools or ()),
        )
        request_config = types.GenerateContentConfig(**config_kwargs)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                raw = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=request_config,
                )
            except Exception as exc:
                if attempt < _MAX_ATTEMPTS and _is_retryable(exc):
                    delay = _retry_delay(exc, attempt)
                    # A transient hiccup we can absorb: note it, don't alarm.
                    log.info(
                        "model request failed (%s); retrying in %.1fs (%d/%d)",
                        type(exc).__name__,
                        delay,
                        attempt,
                        _MAX_ATTEMPTS,
                    )
                    time.sleep(delay)
                    continue
                raise self._translate(exc) from exc
            return self._to_response(raw)
        raise ProviderError("exhausted model retries")  # pragma: no cover

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def _to_contents(self, messages: Sequence[Message]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for message in messages:
            if message.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=message.tool_name or "tool",
                                response=_as_response_dict(message.content),
                            )
                        ],
                    )
                )
                continue

            parts: list[Any] = []
            if message.content:
                text_part = types.Part(text=message.content)
                _restore_signature(text_part, message.provider_state)
                parts.append(text_part)
            for call in message.tool_calls:
                call_part = types.Part.from_function_call(
                    name=call.name, args=call.arguments
                )
                _restore_signature(call_part, call.provider_state)
                parts.append(call_part)
            if not parts:
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
        return contents

    def _to_response(self, raw: Any) -> LLMResponse:
        candidates = getattr(raw, "candidates", None) or []
        if not candidates:
            feedback = getattr(raw, "prompt_feedback", None)
            log.warning("model returned no candidates (feedback=%s)", feedback)
            return LLMResponse(
                text="", finish_reason=str(getattr(feedback, "block_reason", "empty"))
            )

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        text_state: dict[str, Any] = {}
        for part in parts:
            if getattr(part, "thought", False):
                # Never surface raw model reasoning; the signature still
                # travels back with whatever part carried it.
                continue
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                tool_call = ToolCall(
                    name=call.name,
                    arguments=dict(getattr(call, "args", None) or {}),
                    provider_state=_capture_signature(part),
                )
                call_id = getattr(call, "id", None)
                if call_id:
                    tool_call.id = str(call_id)
                tool_calls.append(tool_call)
                continue
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)
                text_state.update(_capture_signature(part))

        usage = getattr(raw, "usage_metadata", None)
        return LLMResponse(
            text="".join(text_chunks).strip(),
            tool_calls=tool_calls,
            finish_reason=str(getattr(candidate, "finish_reason", "") or "") or None,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            provider_state=text_state,
        )

    def _translate(self, exc: Exception) -> ProviderError:
        detail = f"{type(exc).__name__}: {exc}"
        lowered = detail.lower()
        if any(hint in lowered for hint in _AUTH_HINTS):
            message = "My Gemini API key was rejected. Check GOOGLE_API_KEY in .env."
        elif _is_daily_quota(exc):
            message = (
                "I've used up today's free Gemini quota. It resets tomorrow, or you "
                "can switch CRONUS_MODEL to a different one in .env."
            )
        elif any(hint in lowered for hint in _QUOTA_HINTS):
            message = "I've hit the Gemini rate limit. Give it a moment and try again."
        elif "timeout" in lowered or "deadline" in lowered:
            message = "Gemini took too long to answer."
        else:
            message = "I couldn't reach Gemini just now."
        # Handled: the caller turns this into a plain sentence for the user,
        # so the console stays clean and the detail lives in the log file.
        log.warning("gemini request failed: %s", detail)
        return ProviderError(detail, user_message=message)




def _is_daily_quota(exc: Exception) -> bool:
    return _DAILY_QUOTA_HINT in str(exc).lower().replace("_", "")


def _is_retryable(exc: Exception) -> bool:
    if _is_daily_quota(exc):
        return False
    detail = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in detail for hint in _RETRY_HINTS)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Prefer the server's own retry hint over blind exponential backoff."""
    requested = _RETRY_DELAY_RE.search(str(exc))
    if requested:
        return min(float(requested.group(1)) + 1.0, _MAX_RETRY_WAIT)
    return min(2 ** (attempt - 1), 4) + random.uniform(0, 0.4)


def _capture_signature(part: Any) -> dict[str, Any]:
    """Preserve Gemini's opaque per-part thought signature, if present."""
    signature = getattr(part, "thought_signature", None)
    return {"thought_signature": signature} if signature else {}


def _restore_signature(part: Any, state: dict[str, Any]) -> None:
    signature = state.get("thought_signature")
    if signature:
        part.thought_signature = signature


def _as_response_dict(content: str) -> dict[str, Any]:
    """Gemini wants function responses as an object, so normalise to one."""
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return {"result": content}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}
