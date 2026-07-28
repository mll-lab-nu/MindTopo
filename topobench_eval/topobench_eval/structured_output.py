"""Small, dependency-free helpers for structured model outputs.

This module deliberately knows nothing about TopoBench tasks.  It provides
the shared mechanics used by both reasoning-answer and planning-action
parsers: strict integer coercion, code-fence removal, and ordered extraction
of embedded JSON values.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator


_FENCED_BLOCK_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```$",
    re.IGNORECASE | re.DOTALL,
)


def strict_int(value: Any) -> int | None:
    """Return an integer without accepting booleans or lossy numerics."""
    if type(value) is int:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
    return None


def strip_code_fence(text: str) -> str:
    """Strip one outer Markdown JSON fence, leaving other prose untouched."""
    stripped = str(text).strip()
    match = _FENCED_BLOCK_RE.fullmatch(stripped)
    return match.group(1).strip() if match else stripped


def iter_json_values(text: str) -> Iterator[Any]:
    """Yield decodable JSON values in source order.

    The exact full response is yielded once when it is valid JSON. Otherwise,
    every embedded object/array/string/number that starts at a JSON-looking
    character is attempted with ``JSONDecoder.raw_decode``. Duplicate spans
    are suppressed. Consumers can use ``list(...)[-1]`` when the final
    structured answer should win.
    """
    stripped = strip_code_fence(text)
    if not stripped:
        return

    decoder = json.JSONDecoder()
    try:
        yield json.loads(stripped)
        return
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    consumed_until = 0
    for start, char in enumerate(stripped):
        if start < consumed_until:
            continue
        if char not in '{["-0123456789':
            continue
        try:
            value, length = decoder.raw_decode(stripped[start:])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        consumed_until = start + length
        yield value


def answer_objects(text: str) -> list[dict[str, Any]]:
    """Return embedded JSON objects containing an ``answer`` key, in order."""
    return [
        value
        for value in iter_json_values(text)
        if isinstance(value, dict) and "answer" in value
    ]


def last_json_value(text: str) -> Any:
    """Return the last decoded JSON value, or ``None`` when none is present."""
    values = list(iter_json_values(text))
    return values[-1] if values else None


def last_structured_value(text: str) -> Any:
    """Return full JSON, or the last embedded object/array.

    Embedded scalar-looking fragments are ignored so incidental prose numbers
    cannot become planning actions.  A response that is itself a JSON scalar
    remains valid for legacy clients that return a bare action.
    """
    stripped = strip_code_fence(text)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        containers = [value for value in iter_json_values(stripped) if isinstance(value, (dict, list))]
        return containers[-1] if containers else None
