"""Runtime guardrails for lmms-eval response caching.

TopoBench treats provider/API failures as "not run". Some lmms-eval versions
return a textual failure sentinel after exhausting request retries, and some
providers occasionally return "no image was provided" refusals for visual
prompts. If those responses reach ResponseCache, they can be replayed forever.
This patch keeps retryable failures out of cache DBs and audit logs.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

try:
    from _retryable_response import is_retryable_failure_text
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from lmms_eval_tasks._retryable_response import is_retryable_failure_text  # type: ignore

_PATCH_ATTR = "_topobench_failed_response_cache_guard"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _is_failed_response(value: Any) -> bool:
    return is_retryable_failure_text(_text(value))


def _wrap_store(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def wrapped(
        self: Any,
        cache_key: str,
        request_type: str,
        task_name: str,
        doc_id: Any,
        idx: int,
        gen_kwargs: dict,
        response: Any,
    ) -> Any:
        if _is_failed_response(response):
            return None
        return original(self, cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response)

    setattr(wrapped, _PATCH_ATTR, True)
    return wrapped


def _wrap_log_to_audit(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def wrapped(
        self: Any,
        request_type: str,
        task_name: str,
        doc_id: Any,
        idx: int,
        gen_kwargs: dict,
        response: Any,
        **kwargs: Any,
    ) -> Any:
        if _is_failed_response(response):
            return None
        return original(self, request_type, task_name, doc_id, idx, gen_kwargs, response, **kwargs)

    setattr(wrapped, _PATCH_ATTR, True)
    return wrapped


def install() -> None:
    try:
        from lmms_eval.caching.response_cache import ResponseCache
    except Exception:
        return

    if not getattr(ResponseCache._store, _PATCH_ATTR, False):
        ResponseCache._store = _wrap_store(ResponseCache._store)  # type: ignore[method-assign]
    if not getattr(ResponseCache._log_to_audit, _PATCH_ATTR, False):
        ResponseCache._log_to_audit = _wrap_log_to_audit(ResponseCache._log_to_audit)  # type: ignore[method-assign]
