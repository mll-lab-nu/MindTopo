"""Shared retryable-response detection for lmms-eval wrappers."""

from __future__ import annotations

FAILED_RESPONSE_PREFIX = "[LMMS_EVAL_REQUEST_FAILED"

# SQL LIKE patterns are lowercase because callers compare against LOWER(response).
RETRYABLE_FAILURE_SQL_LIKE_PATTERNS = (
    f"%{FAILED_RESPONSE_PREFIX.lower()}%",
    "%no image was provided%",
    "%no images were provided%",
    "%no image provided%",
    "%image was not provided%",
    "%image is not provided%",
    "%please provide the image%",
    "%without an image%",
)


def is_retryable_failure_text(text: object) -> bool:
    """Return True for provider failures that should not be cached as answers."""
    if not isinstance(text, str):
        text = str(text or "")
    stripped = text.strip()
    if FAILED_RESPONSE_PREFIX in stripped:
        return True

    lowered = stripped.lower()
    if "no image" in lowered and (
        "provided" in lowered or "attached" in lowered or "prompt" in lowered
    ):
        return True
    if "image was not provided" in lowered or "image is not provided" in lowered:
        return True
    if "please provide the image" in lowered:
        return True
    if "without an image" in lowered and (
        "cannot" in lowered or "can't" in lowered or "unable" in lowered
    ):
        return True
    return False
