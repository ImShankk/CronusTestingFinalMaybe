"""A small JSON-Schema validator for tool parameters.

Model output is untrusted input, so every tool call is validated before a
handler sees it. Only the subset of JSON Schema that tool definitions actually
use is supported, which keeps this dependency-free and easy to audit:

``type`` (object/string/number/integer/boolean/array/null), ``properties``,
``required``, ``enum``, ``default``, ``items``, ``minimum``/``maximum``,
``minLength``/``maxLength``, and rejection of unknown properties.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolValidationError

_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    checks = _TYPE_CHECKS.get(expected)
    if checks is None:
        return True
    # bool is an int subclass; never let True satisfy a number/integer field.
    if expected in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, checks)


def _coerce(value: Any, expected: str) -> Any:
    """Repair the few harmless type slips models actually make."""
    if expected == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if expected == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected in ("integer", "number") and isinstance(value, str):
        try:
            return int(value) if expected == "integer" else float(value)
        except ValueError:
            return value
    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    if expected == "array" and isinstance(value, tuple):
        return list(value)
    return value


def _fail(path: str, detail: str) -> None:
    where = path or "arguments"
    raise ToolValidationError(
        f"{where}: {detail}",
        user_message="I called that tool with the wrong arguments.",
    )


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> Any:
    expected = schema.get("type")
    if isinstance(expected, list):
        for option in expected:
            if _type_ok(value, option):
                expected = option
                break
        else:
            expected = expected[0] if expected else None

    if expected:
        value = _coerce(value, expected)
        if not _type_ok(value, expected):
            _fail(path, f"expected {expected}, got {type(value).__name__}")

    choices = schema.get("enum")
    if choices is not None and value not in choices:
        _fail(path, f"must be one of {choices}")

    if expected == "string":
        min_len, max_len = schema.get("minLength"), schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            _fail(path, f"must be at least {min_len} characters")
        if max_len is not None and len(value) > max_len:
            _fail(path, f"must be at most {max_len} characters")

    if expected in ("number", "integer"):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if minimum is not None and value < minimum:
            _fail(path, f"must be >= {minimum}")
        if maximum is not None and value > maximum:
            _fail(path, f"must be <= {maximum}")

    if expected == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            value = [
                _validate_value(item, item_schema, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            _fail(path, f"must have at most {max_items} items")

    if expected == "object" and "properties" in schema:
        value = validate_arguments(value, schema, path=path)

    return value


def validate_arguments(
    arguments: dict[str, Any], schema: dict[str, Any], *, path: str = ""
) -> dict[str, Any]:
    """Validate and normalise ``arguments`` against an object schema.

    Returns a new dict with defaults applied and light type coercion done.
    Raises :class:`~cronus.errors.ToolValidationError` on anything else.
    """
    if not isinstance(arguments, dict):
        _fail(path, f"expected an object, got {type(arguments).__name__}")

    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = list(schema.get("required", []) or [])
    allow_extra = schema.get("additionalProperties", False)

    unknown = set(arguments) - set(properties)
    if unknown and not allow_extra:
        _fail(path, f"unexpected argument(s): {', '.join(sorted(unknown))}")

    missing = [key for key in required if arguments.get(key) is None]
    if missing:
        _fail(path, f"missing required argument(s): {', '.join(missing)}")

    result: dict[str, Any] = {}
    for key, sub_schema in properties.items():
        sub_path = f"{path}.{key}" if path else key
        if key in arguments and arguments[key] is not None:
            result[key] = _validate_value(arguments[key], sub_schema or {}, sub_path)
        elif "default" in (sub_schema or {}):
            result[key] = sub_schema["default"]

    if allow_extra:
        for key in unknown:
            result[key] = arguments[key]

    return result
