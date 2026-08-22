"""Tool registry, schema validation, execution, and failure handling."""

from __future__ import annotations

import time

import pytest

from cronus.errors import ToolNotFound, ToolValidationError
from cronus.tools.base import RiskLevel, Tool, ToolContext, ToolResult, object_schema
from cronus.tools.registry import ToolRegistry
from cronus.tools.schema import validate_arguments


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------
def test_defaults_are_applied():
    schema = object_schema(
        {"a": {"type": "string"}, "b": {"type": "integer", "default": 3}},
        required=["a"],
    )
    assert validate_arguments({"a": "x"}, schema) == {"a": "x", "b": 3}


def test_missing_required_argument_is_rejected():
    schema = object_schema({"a": {"type": "string"}}, required=["a"])
    with pytest.raises(ToolValidationError, match="missing required"):
        validate_arguments({}, schema)


def test_unknown_arguments_are_rejected():
    schema = object_schema({"a": {"type": "string"}})
    with pytest.raises(ToolValidationError, match="unexpected"):
        validate_arguments({"a": "x", "sneaky": "y"}, schema)


def test_wrong_type_is_rejected():
    schema = object_schema({"n": {"type": "integer"}})
    with pytest.raises(ToolValidationError, match="expected integer"):
        validate_arguments({"n": [1, 2]}, schema)


def test_booleans_never_satisfy_numeric_fields():
    schema = object_schema({"n": {"type": "integer"}})
    with pytest.raises(ToolValidationError):
        validate_arguments({"n": True}, schema)


def test_numeric_strings_are_coerced():
    schema = object_schema({"n": {"type": "integer"}})
    assert validate_arguments({"n": "7"}, schema) == {"n": 7}


def test_enum_and_bounds_are_enforced():
    schema = object_schema(
        {"kind": {"type": "string", "enum": ["a", "b"]}, "n": {"type": "integer", "maximum": 5}}
    )
    with pytest.raises(ToolValidationError, match="one of"):
        validate_arguments({"kind": "c"}, schema)
    with pytest.raises(ToolValidationError, match="<= 5"):
        validate_arguments({"n": 9}, schema)


def test_array_items_are_validated():
    schema = object_schema({"xs": {"type": "array", "items": {"type": "integer"}}})
    assert validate_arguments({"xs": [1, "2"]}, schema) == {"xs": [1, 2]}


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def test_register_and_discover(registry: ToolRegistry, echo: Tool):
    registry.register(echo)
    assert registry.has("echo")
    assert len(registry) == 1
    assert [schema.name for schema in registry.schemas()] == ["echo"]


def test_duplicate_registration_is_refused(registry: ToolRegistry, echo: Tool):
    registry.register(echo)
    with pytest.raises(Exception, match="already registered"):
        registry.register(echo)
    registry.register(echo, replace=True)  # explicit replacement is fine


def test_invalid_tool_name_is_refused(registry: ToolRegistry, echo: Tool):
    echo.name = "Bad Name"
    with pytest.raises(Exception, match="invalid tool name"):
        registry.register(echo)


def test_unknown_tool_raises(registry: ToolRegistry):
    with pytest.raises(ToolNotFound):
        registry.get("nope")


def test_schemas_never_leak_risk_metadata(registry: ToolRegistry, echo: Tool):
    echo.risk = RiskLevel.CONFIRM
    registry.register(echo)
    assert not hasattr(registry.schemas()[0], "risk")


def test_execute_returns_result(registry: ToolRegistry, echo: Tool, config):
    registry.register(echo)
    result = registry.execute("echo", {"text": "hello"}, ToolContext(config=config))
    assert result.ok and result.content == "echo: hello"


def test_execute_with_bad_arguments_fails_gracefully(registry, echo, config):
    registry.register(echo)
    result = registry.execute("echo", {"wrong": 1}, ToolContext(config=config))
    assert not result.ok
    assert "Invalid arguments" in result.content


def test_handler_exception_becomes_failed_result(registry: ToolRegistry, config):
    def explode() -> str:
        raise RuntimeError("boom")

    registry.register(
        Tool(name="boom", description="Explodes.", parameters=object_schema({}),
             handler=explode)
    )
    result = registry.execute("boom", {}, ToolContext(config=config))
    assert not result.ok and "boom" in result.content


def test_slow_handler_times_out(registry: ToolRegistry, config):
    def slow() -> str:
        time.sleep(2)
        return "done"

    registry.register(
        Tool(name="slow", description="Sleeps.", parameters=object_schema({}),
             handler=slow, timeout=0.2)
    )
    result = registry.execute("slow", {}, ToolContext(config=config))
    assert not result.ok and "timed out" in result.content


def test_async_handlers_are_supported(registry: ToolRegistry, config):
    async def fetch() -> ToolResult:
        return ToolResult(content="async ok")

    registry.register(
        Tool(name="fetch", description="Async.", parameters=object_schema({}),
             handler=fetch)
    )
    assert registry.execute("fetch", {}, ToolContext(config=config)).content == "async ok"


def test_handlers_receive_context_only_when_declared(registry: ToolRegistry, config):
    def with_context(context: ToolContext) -> str:
        return f"model={context.config.llm.model}"

    registry.register(
        Tool(name="ctx", description="Uses context.", parameters=object_schema({}),
             handler=with_context)
    )
    assert "fake-model" in registry.execute("ctx", {}, ToolContext(config=config)).content
