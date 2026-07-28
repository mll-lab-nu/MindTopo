"""Helpers for reading per-env metadata.json into model-facing prompts."""
from __future__ import annotations

import json
from pathlib import Path


def load_task_intro_line(
    project_root: Path,
    *,
    default: str = "You are working on this task.",
) -> str:
    """Compose a two-line task intro from `<project_root>/metadata.json`.

    Output format::

        You are solving {display_name}.
        In this task, you must {description}.

    The description's first letter is lowercased and any trailing period is
    stripped so it reads cleanly inside the template body. metadata.json's
    `description` should stay a proper standalone sentence (capitalized,
    punctuated) — the normalization happens here.

    Falls back to `default` if metadata.json is missing or malformed.
    """
    try:
        meta = json.loads((project_root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    display_name = str(meta.get("display_name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not display_name and not description:
        return default
    body = description.rstrip(".").strip()
    if body and body[:1].isupper():
        body = body[:1].lower() + body[1:]
    if display_name and body:
        return f"You are solving {display_name}.\nIn this task, you must {body}."
    if display_name:
        return f"You are solving {display_name}."
    return f"In this task, you must {body}."
