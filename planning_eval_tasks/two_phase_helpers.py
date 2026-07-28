"""Shared helpers for two-phase generative eval clients.

`InterleavedModelClient` (interleaved + image-gen) and `VLMVideoModelClient`
(video-gen + frames) both run a phase-1 → side-task → phase-2 cycle. The
mechanics around stripping the env adapter's `[Answer Format]` block,
nudging phase-1 with a reminder, extracting a tagged prompt from phase-1
output, picking a reference image, naming the peer artifact, and resolving
a base URL are identical modulo artifact extension and tag name. This
module factors them out so both clients share one implementation.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from omegaconf import DictConfig

from model_adapter import (
    BaseModelClient,
    OpenAICompatibleModelClient,
    build_key_pool,
    load_keys_from_env,
    resolve_slots_per_key,
)
from topobench_eval.provider_registry import resolve_model_registry_entry
from unified_types import Message, SavedImage


_ANSWER_FORMAT_BLOCK_PATTERN = re.compile(
    r"\[Answer Format\][\s\S]*?(?=\n\[[^\]\n]+\]|\Z)",
)
_TRAILING_ANSWER_LINE_PATTERN = re.compile(
    r"^Output ONLY a single JSON object[^\n]*\n?",
    re.MULTILINE,
)

_FALLBACK_PROMPT_CHAR_LIMIT = 1000


def strip_answer_format_text(text: str) -> str:
    cleaned = _ANSWER_FORMAT_BLOCK_PATTERN.sub("", text)
    cleaned = _TRAILING_ANSWER_LINE_PATTERN.sub("", cleaned)
    return cleaned.rstrip()


def strip_answer_format_from_messages(messages: Sequence[Message]) -> List[Message]:
    """Return a copy of `messages` with the env adapter's answer-format text removed.

    Phase 1 should only emit the side-task tag; the env adapter's single-shot
    `{"answer": ...}` block conflicts with that. Phase 2 sees the un-stripped
    original messages so the canonical answer JSON is still produced there.
    """
    cleaned: List[Message] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            cleaned.append({**msg, "content": strip_answer_format_text(content)})
            continue
        if isinstance(content, list):
            new_blocks: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    new_blocks.append({**block, "text": strip_answer_format_text(block["text"])})
                else:
                    new_blocks.append(block)
            cleaned.append({**msg, "content": new_blocks})
            continue
        cleaned.append(dict(msg))
    return cleaned


def append_phase1_reminder(messages: Sequence[Message], reminder: str) -> List[Message]:
    """Append `reminder` as a trailing text block on the last user message."""
    if not messages:
        return list(messages)
    out = list(messages)
    last = dict(out[-1])
    if last.get("role") != "user":
        out.append({"role": "user", "content": [{"type": "text", "text": reminder}]})
        return out
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = (content.rstrip() + "\n\n" + reminder).strip()
    elif isinstance(content, list):
        last["content"] = list(content) + [{"type": "text", "text": reminder}]
    else:
        last["content"] = [{"type": "text", "text": reminder}]
    out[-1] = last
    return out


def extract_tagged_prompt(
    raw_text: str,
    reasoning_content: Optional[str],
    pattern: Pattern[str],
) -> Optional[str]:
    """Pull the first `pattern` match's inner body from raw_text or reasoning_content.

    Reasoning-style models (Nemotron Omni, OpenAI o-series, DeepSeek-R1, etc.)
    often route the chain-of-thought to `reasoning_content` and leave
    `message.content` empty. The literal value "none" (case-insensitive) is
    preserved so callers can distinguish "model declined" from "didn't follow
    protocol".
    """
    for source in (raw_text, reasoning_content):
        if not source:
            continue
        match = pattern.search(source)
        if match is None:
            continue
        inner = match.group(1).strip()
        if inner:
            return inner
    return None


def fallback_phase1_prompt(
    phase1_text: str,
    phase1_reasoning: Optional[str],
    *,
    default: str,
) -> Tuple[str, str]:
    """Synthesize a side-task prompt when phase-1 omitted the expected tag.

    Prefers phase-1 visible text, falls back to phase-1 reasoning_content for
    reasoning-style models that route output there, otherwise returns
    `default`. Truncated to keep request sizes sane. Returns
    (prompt, source_label) where source_label is one of "phase1_text" /
    "phase1_reasoning_content" / "static".
    """
    for source, label in ((phase1_text, "phase1_text"), (phase1_reasoning, "phase1_reasoning_content")):
        if source and source.strip():
            return source.strip()[:_FALLBACK_PROMPT_CHAR_LIMIT], label
    return default, "static"


def pick_current_reference_image(images: Sequence[SavedImage]) -> Optional[SavedImage]:
    """Return the most recent observation `SavedImage`, or None.

    Drives both the side-task's reference (multipart image / i2v frame) and
    the peer artifact's filename. Prefers a `current`-labelled image; falls
    back to the most recent saved image.
    """
    if not images:
        return None
    for img in reversed(images):
        if img.label == "current":
            return img
    return images[-1]


def peer_artifact_target(
    reference: Optional[SavedImage],
    *,
    output_dir: Path,
    suffix: str,
) -> Optional[Tuple[Path, str]]:
    """Return (abs_path, rel_path) for an artifact saved next to `reference`.

    When `reference.abs_path` lives under `output_dir` (the typical case for
    gym `step_<N>_current.png` and reasoning copied source PNGs), the peer
    artifact is saved there with `_imagined<suffix>` substituted for
    `_current.<ext>` (or appended when no `_current` suffix exists). Returns
    None when `reference is None` or its path is outside `output_dir` —
    callers handle that by computing a counter-based fallback.
    """
    if reference is None:
        return None
    ref_abs = reference.abs_path
    try:
        ref_abs.relative_to(output_dir)
    except ValueError:
        return None
    stem = ref_abs.stem
    new_stem = (
        stem[: -len("_current")] + "_imagined"
        if stem.endswith("_current")
        else f"{stem}_imagined"
    )
    out = ref_abs.parent / (new_stem + suffix)
    return out, str(out.relative_to(output_dir))


def build_inner_text_client(*, text_cfg: DictConfig, system_prompt: str) -> BaseModelClient:
    """Construct the inner text VLM client from a `model.text` cfg block.

    The text block must reference an entry in providers.yaml via
    `text_model_id`. All transport options (temperature, max_tokens, retries,
    etc.) live alongside.
    """
    text_model_id = str(text_cfg.text_model_id)
    resolved = resolve_model_registry_entry(text_model_id)
    keys = load_keys_from_env(resolved.api_key_env)
    rpm = float(text_cfg.requests_per_minute)
    pool = build_key_pool(
        provider=resolved.provider,
        keys=keys,
        rpm=rpm,
        slots_per_key=resolve_slots_per_key(text_cfg),
    )
    if resolved.adapter in ("openai", "gemini_api"):
        return OpenAICompatibleModelClient(
            model_cfg=text_cfg,
            system_prompt=system_prompt,
            model_id=resolved.model_id,
            model_name=resolved.upstream_model_name,
            provider_name=resolved.provider,
            base_url=resolved.base_url or "https://api.openai.com/v1",
            pool=pool,
        )
    raise ValueError(
        f"Unsupported adapter {resolved.adapter!r} for inner text model "
        f"{resolved.model_id!r}. Supported: 'openai', 'gemini_api'."
    )


def resolve_base_url_from_cfg(cfg: DictConfig, *, label: str) -> str:
    """Resolve a `base_url`, honoring `base_url_env` first then literal `base_url`.

    `label` is used only in the error message so callers (image-gen,
    video-gen) get a clear hint when neither env var nor literal URL is set.
    """
    base_url_env = str(getattr(cfg, "base_url_env", "") or "").strip()
    base_url = ""
    if base_url_env:
        base_url = os.environ.get(base_url_env, "").strip()
    if not base_url:
        base_url = str(getattr(cfg, "base_url", "") or "").strip()
    if not base_url:
        raise ValueError(
            f"{label} config requires either a non-empty `base_url` or a "
            "`base_url_env` pointing to an env var that resolves to a URL."
        )
    return base_url
