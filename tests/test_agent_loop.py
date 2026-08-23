"""The multi-step agent loop: chaining, recovery, limits, and cancellation."""

from __future__ import annotations

import pytest

from cronus.core.events import EventType
from cronus.llm.base import LLMResponse, ToolCall
from cronus.tools.base import RiskLevel, Tool, ToolResult, object_schema
from tests.conftest import text_response, tool_response


@pytest.fixture(autouse=True)
def _tools(registry, echo):
    registry.register(echo)

    def add(a: int, b: int) -> ToolResult:
        return ToolResult(content=str(a + b))

    def flaky(fail: bool = True) -> ToolResult:
        if fail:
            return ToolResult.failure("the service is down")
        return ToolResult(content="worked this time")

    registry.register(
        Tool(name="add", description="Adds numbers.",
             parameters=object_schema(
                 {"a": {"type": "integer"}, "b": {"type": "integer"}},
                 required=["a", "b"]),
             handler=add)
    )
    registry.register(
        Tool(name="flaky", description="Fails sometimes.",
             parameters=object_schema({"fail": {"type": "boolean", "default": True}}),
             handler=flaky)
    )
    registry.register(
        Tool(name="risky", description="Needs approval.",
             parameters=object_schema({"target": {"type": "string"}}, required=["target"]),
             handler=lambda target: ToolResult(content=f"did it to {target}"),
             risk=RiskLevel.CONFIRM)
    )
    return registry


def test_plain_answer_needs_one_model_call(make_assistant):
    assistant = make_assistant([text_response("Hello there.")])
    result = assistant.send("hi")
    assert result.text == "Hello there."
    assert result.iterations == 1 and result.tools_used == []


def test_single_tool_call_feeds_result_back(make_assistant):
    assistant = make_assistant(
        [tool_response("echo", text="ping"), text_response("It said ping.")]
    )
    result = assistant.send("echo ping")
    assert result.text == "It said ping."
    assert result.tools_used == ["echo"]

    # The follow-up request must contain the tool's output.
    second = assistant.provider.calls[1]["messages"]
    assert any(m.role == "tool" and "echo: ping" in m.content for m in second)


def test_tools_chain_across_iterations(make_assistant):
    assistant = make_assistant(
        [
            tool_response("add", a=2, b=3),
            tool_response("add", a=5, b=10),
            text_response("The total is 15."),
        ]
    )
    result = assistant.send("add some numbers")
    assert result.tools_used == ["add", "add"]
    assert result.iterations == 3


def test_parallel_tool_calls_in_one_turn(make_assistant):
    assistant = make_assistant(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="echo", arguments={"text": "a"}),
                    ToolCall(name="echo", arguments={"text": "b"}),
                ]
            ),
            text_response("Both done."),
        ]
    )
    result = assistant.send("echo twice")
    assert result.tools_used == ["echo", "echo"]
    tool_messages = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"]
    assert len(tool_messages) == 2


def test_failed_tool_is_reported_to_the_model_not_raised(make_assistant):
    assistant = make_assistant(
        [
            tool_response("flaky", fail=True),
            tool_response("flaky", fail=False),
            text_response("It worked on the second try."),
        ]
    )
    result = assistant.send("try the flaky thing")
    assert result.ok
    first_result = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "the service is down" in first_result.content


def test_unknown_tool_is_reported_and_recoverable(make_assistant):
    assistant = make_assistant(
        [tool_response("does_not_exist", x=1), text_response("I don't have that tool.")]
    )
    result = assistant.send("use a made up tool")
    assert result.ok
    tool_message = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "There is no tool named does_not_exist" in tool_message.content
    assert "does_not_exist" not in result.tools_used


def test_invalid_arguments_are_reported_to_the_model(make_assistant):
    assistant = make_assistant(
        [tool_response("add", a="not a number", b=1), text_response("I got that wrong.")]
    )
    assistant.send("add badly")
    tool_message = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "Invalid arguments" in tool_message.content


def test_iteration_limit_forces_a_final_answer(make_assistant, config):
    # More tool calls than max_tool_iterations allows.
    script = [tool_response("echo", text=str(i)) for i in range(config.max_tool_iterations)]
    script.append(text_response("Here's what I found so far."))
    assistant = make_assistant(script)
    result = assistant.send("loop forever")
    assert result.error == "iteration_limit"
    assert result.text == "Here's what I found so far."
    assert len(assistant.provider.calls) == config.max_tool_iterations + 1


def test_declined_confirmation_stops_the_tool(make_assistant):
    assistant = make_assistant(
        [tool_response("risky", target="the thing"), text_response("Left it alone.")],
        approve=False,
    )
    result = assistant.send("do the risky thing")
    tool_message = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    # The model must read this as a refusal, not as "not confirmed yet" --
    # the latter makes it offer the same action again.
    assert "said no" in tool_message.content
    assert "must not be" in tool_message.content
    assert "did it to" not in tool_message.content
    assert result.text == "Left it alone."


def test_approved_confirmation_runs_the_tool(make_assistant):
    assistant = make_assistant(
        [tool_response("risky", target="the thing"), text_response("Done.")],
        approve=True,
    )
    assistant.send("do the risky thing")
    tool_message = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert "did it to the thing" in tool_message.content


def test_provider_failure_becomes_a_plain_message(make_assistant):
    from cronus.errors import ProviderError

    assistant = make_assistant([])

    def fail(*args, **kwargs):
        raise ProviderError("500 internal", user_message="I couldn't reach Gemini.")

    assistant.provider.generate = fail
    result = assistant.send("hello")
    assert result.text == "I couldn't reach Gemini."
    assert not result.ok
    assert assistant.conversation.turns == []  # the broken turn isn't recorded


def test_unexpected_error_never_surfaces_a_traceback(make_assistant):
    assistant = make_assistant([])

    def explode(*args, **kwargs):
        raise ValueError("internal detail the user must not see")

    assistant.provider.generate = explode
    result = assistant.send("hello")
    assert "internal detail" not in result.text
    assert "went wrong" in result.text


def test_empty_input_is_ignored(make_assistant):
    assistant = make_assistant([text_response("should not be used")])
    assert assistant.send("   ").text == ""
    assert assistant.provider.calls == []


def test_cancelling_mid_turn_stops_the_loop(make_assistant):
    """A cancel during tool execution ends the turn before the next model call."""
    assistant = make_assistant(
        [tool_response("echo", text="x"), text_response("should never be reached")]
    )
    assistant.emitter.subscribe(
        lambda event: assistant.cancel() if event.type is EventType.TOOL_END else None
    )
    result = assistant.send("do something")
    assert result.cancelled
    assert len(assistant.provider.calls) == 1
    assert assistant.conversation.turns == []


def test_a_new_turn_clears_a_stale_cancel(make_assistant):
    assistant = make_assistant([text_response("done")])
    assistant.cancel()
    assert assistant.send("do something").text == "done"


def test_events_narrate_the_turn(make_assistant):
    assistant = make_assistant([tool_response("echo", text="hi"), text_response("Said hi.")])
    seen = []
    assistant.emitter.subscribe(lambda event: seen.append(event))
    assistant.send("say hi")

    kinds = [event.type for event in seen]
    assert EventType.TOOL_START in kinds and EventType.RESPONSE in kinds
    started = next(e for e in seen if e.type is EventType.TOOL_START)
    assert started.data["tool"] == "echo"


def test_long_tool_output_is_truncated_before_the_model_sees_it(make_assistant, registry):
    registry.register(
        Tool(name="firehose", description="Lots of text.", parameters=object_schema({}),
             handler=lambda: ToolResult(content="x" * 50_000))
    )
    assistant = make_assistant([tool_response("firehose"), text_response("ok")])
    assistant.send("flood me")
    tool_message = [m for m in assistant.provider.calls[1]["messages"] if m.role == "tool"][0]
    assert len(tool_message.content) < 10_000
    assert "truncated" in tool_message.content
