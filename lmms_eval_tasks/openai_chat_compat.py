"""Runtime compatibility patch for lmms-eval's OpenAI chat backend.

Some current OpenAI reasoning/chat models reject the legacy Chat Completions
``max_tokens`` field and only accept ``max_completion_tokens``. lmms-eval still
emits ``max_tokens`` for most models, so TopoBench installs this patch in shard
subprocesses via ``sitecustomize``.
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import threading
import time
from functools import wraps
from typing import Any, Callable, Mapping


_PATCH_ATTR = "_topobench_openai_chat_compat"
_MAX_TOKENS_ERROR_MARKERS = (
    "unsupported parameter",
    "not supported",
)
_TEMPERATURE_ERROR_MARKERS = (
    "unsupported parameter",
    "unsupported value",
    "does not support",
    "only the default",
)
_REASONING_MODEL_PREFIXES = (
    "gpt-5",
    "gpt5",
    "o1",
    "o3",
    "o4",
)
_LOW_REASONING_EFFORT_MODEL_PREFIXES = (
    "gpt-5.5",
    "gpt5.5",
)
_GEMINI_MODEL_PREFIXES = (
    "gemini-",
)
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "too frequent",
    "请求过于频繁",
    "请求过快",
)
_REASONING_CONTENT_FIELDS = (
    "reasoning",
    "reasoning_content",
)

_THROTTLE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0
_ASYNC_THROTTLE_LOCK: asyncio.Lock | None = None


def prefers_max_completion_tokens(model: Any) -> bool:
    normalized = str(model or "").lower()
    return normalized.startswith(_REASONING_MODEL_PREFIXES)


def default_reasoning_effort(model: Any) -> str | None:
    normalized = str(model or "").lower()
    if normalized.startswith(_LOW_REASONING_EFFORT_MODEL_PREFIXES):
        return "low"
    return None


def wants_gemini_thought_summaries(model: Any) -> bool:
    provider = os.environ.get("TOPOBENCH_OPENAI_PROVIDER", "").strip().lower()
    normalized = str(model or "").lower()
    return provider == "google" or normalized.startswith(_GEMINI_MODEL_PREFIXES)


def _add_gemini_thought_summary_request(kwargs: dict[str, Any]) -> None:
    if not wants_gemini_thought_summaries(kwargs.get("model")):
        return
    extra_body = copy.deepcopy(kwargs.get("extra_body") or {})
    if not isinstance(extra_body, dict):
        return
    request_extra_body = extra_body.setdefault("extra_body", {})
    if not isinstance(request_extra_body, dict):
        return
    google_cfg = request_extra_body.setdefault("google", {})
    if not isinstance(google_cfg, dict):
        return
    thinking_cfg = google_cfg.setdefault("thinking_config", {})
    if not isinstance(thinking_cfg, dict):
        return
    thinking_cfg.setdefault("include_thoughts", True)
    kwargs["extra_body"] = extra_body


def normalize_chat_completion_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    _add_gemini_thought_summary_request(out)
    if not prefers_max_completion_tokens(out.get("model")):
        return out

    if "max_tokens" in out and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    else:
        out.pop("max_tokens", None)

    # Reasoning models commonly reject non-default temperature. Omitting the
    # field lets the API use the model-supported default.
    out.pop("temperature", None)
    default_effort = default_reasoning_effort(out.get("model"))
    if default_effort is not None:
        out.setdefault("reasoning_effort", default_effort)
    return out


def repair_chat_completion_kwargs_for_error(
    kwargs: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any] | None:
    message = str(error).lower()
    out = dict(kwargs)
    changed = False

    if (
        "max_tokens" in out
        and "max_completion_tokens" not in out
        and "max_tokens" in message
        and any(marker in message for marker in _MAX_TOKENS_ERROR_MARKERS)
    ):
        out["max_completion_tokens"] = out.pop("max_tokens")
        changed = True

    if (
        "temperature" in out
        and "temperature" in message
        and any(marker in message for marker in _TEMPERATURE_ERROR_MARKERS)
    ):
        out.pop("temperature", None)
        changed = True

    return out if changed else None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _rate_limit_retries() -> int:
    return max(0, _env_int("TOPOBENCH_OPENAI_RATE_LIMIT_RETRIES", 0))


def _rate_limit_initial_sleep() -> float:
    return max(0.0, _env_float("TOPOBENCH_OPENAI_RATE_LIMIT_INITIAL_SLEEP", 16.0))


def _rate_limit_max_sleep() -> float:
    return max(
        _rate_limit_initial_sleep(),
        _env_float("TOPOBENCH_OPENAI_RATE_LIMIT_MAX_SLEEP", 300.0),
    )


def _request_min_interval() -> float:
    return max(0.0, _env_float("TOPOBENCH_OPENAI_REQUEST_MIN_INTERVAL_SECONDS", 0.0))


def _is_rate_limit_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    message = str(error).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def _retry_after_seconds(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, value)


def _field_value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _set_field_value(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
        return
    try:
        setattr(obj, name, value)
    except Exception:
        return


def _nonempty_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def backfill_content_from_reasoning(response: Any) -> Any:
    """Make OpenAI-compatible servers with reasoning-only messages usable.

    Some vLLM reasoning deployments return the generated text in
    ``message.reasoning`` while leaving ``message.content`` as ``None``. The
    lmms-eval OpenAI backend reads only ``content``, so these responses look
    empty unless we copy the reasoning text over.
    """

    choices = _field_value(response, "choices")
    if not choices:
        return response

    for choice in choices:
        message = _field_value(choice, "message")
        if message is None:
            continue
        content = _field_value(message, "content")
        if _nonempty_text(content):
            continue
        for field_name in _REASONING_CONTENT_FIELDS:
            reasoning_text = _nonempty_text(_field_value(message, field_name))
            if reasoning_text:
                _set_field_value(message, "content", reasoning_text)
                break
    return response


def _rate_limit_sleep_seconds(error: BaseException, retry_index: int) -> float:
    exponential = min(
        _rate_limit_initial_sleep() * (2 ** retry_index),
        _rate_limit_max_sleep(),
    )
    retry_after = _retry_after_seconds(error)
    return max(exponential, retry_after or 0.0)


def _log_rate_limit_sleep(
    seconds: float,
    retry_index: int,
    retries: int,
    error: BaseException,
) -> None:
    preview = str(error).replace("\n", " ")[:200]
    print(
        "[topobench-openai-backoff] "
        f"rate limited; sleeping {seconds:.1f}s before retry "
        f"{retry_index + 1}/{retries}. Last error: {preview}",
        file=sys.stderr,
        flush=True,
    )


def _throttle_before_request() -> None:
    interval = _request_min_interval()
    if interval <= 0:
        return
    global _NEXT_REQUEST_AT
    with _THROTTLE_LOCK:
        now = time.monotonic()
        sleep_for = max(0.0, _NEXT_REQUEST_AT - now)
        _NEXT_REQUEST_AT = max(now, _NEXT_REQUEST_AT) + interval
    if sleep_for > 0:
        time.sleep(sleep_for)


async def _async_throttle_before_request() -> None:
    interval = _request_min_interval()
    if interval <= 0:
        return
    global _ASYNC_THROTTLE_LOCK, _NEXT_REQUEST_AT
    if _ASYNC_THROTTLE_LOCK is None:
        _ASYNC_THROTTLE_LOCK = asyncio.Lock()
    async with _ASYNC_THROTTLE_LOCK:
        now = time.monotonic()
        sleep_for = max(0.0, _NEXT_REQUEST_AT - now)
        _NEXT_REQUEST_AT = max(now, _NEXT_REQUEST_AT) + interval
    if sleep_for > 0:
        await asyncio.sleep(sleep_for)


def _call_with_rate_limit_backoff(
    original: Callable[..., Any],
    self: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    retries = _rate_limit_retries()
    for retry_index in range(retries + 1):
        _throttle_before_request()
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or retry_index >= retries:
                raise
            sleep_for = _rate_limit_sleep_seconds(exc, retry_index)
            _log_rate_limit_sleep(sleep_for, retry_index, retries, exc)
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")


async def _async_call_with_rate_limit_backoff(
    original: Callable[..., Any],
    self: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    retries = _rate_limit_retries()
    for retry_index in range(retries + 1):
        await _async_throttle_before_request()
        try:
            return await original(self, *args, **kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or retry_index >= retries:
                raise
            sleep_for = _rate_limit_sleep_seconds(exc, retry_index)
            _log_rate_limit_sleep(sleep_for, retry_index, retries, exc)
            await asyncio.sleep(sleep_for)
    raise RuntimeError("unreachable")


def _wrap_sync_create(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        normalized = normalize_chat_completion_kwargs(kwargs)
        try:
            return backfill_content_from_reasoning(
                _call_with_rate_limit_backoff(original, self, args, normalized)
            )
        except Exception as exc:
            repaired = repair_chat_completion_kwargs_for_error(normalized, exc)
            if repaired is None:
                raise
            return backfill_content_from_reasoning(
                _call_with_rate_limit_backoff(original, self, args, repaired)
            )

    setattr(create, _PATCH_ATTR, True)
    return create


def _wrap_async_create(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        normalized = normalize_chat_completion_kwargs(kwargs)
        try:
            return backfill_content_from_reasoning(
                await _async_call_with_rate_limit_backoff(
                    original,
                    self,
                    args,
                    normalized,
                )
            )
        except Exception as exc:
            repaired = repair_chat_completion_kwargs_for_error(normalized, exc)
            if repaired is None:
                raise
            return backfill_content_from_reasoning(
                await _async_call_with_rate_limit_backoff(
                    original,
                    self,
                    args,
                    repaired,
                )
            )

    setattr(create, _PATCH_ATTR, True)
    return create


def install() -> None:
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return

    if not getattr(Completions.create, _PATCH_ATTR, False):
        Completions.create = _wrap_sync_create(Completions.create)  # type: ignore[method-assign]

    if not getattr(AsyncCompletions.create, _PATCH_ATTR, False):
        AsyncCompletions.create = _wrap_async_create(AsyncCompletions.create)  # type: ignore[method-assign]
