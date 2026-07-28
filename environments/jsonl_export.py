from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


def to_relative_path(path: str | Path, *, base_dir: str | Path) -> str:
    return os.path.relpath(Path(path), Path(base_dir))


def resolve_existing_path(path: str | Path, *bases: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for base in bases:
        resolved = Path(base) / candidate
        if resolved.exists():
            return resolved
    if bases:
        return Path(bases[0]) / candidate
    return candidate


def read_text_if_exists(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return ""
    return candidate.read_text(encoding="utf-8").strip()


_IMAGE_MARKER_RE = re.compile(r"^\[Image(?:\s+\d+)?\]$")


def _format_image_blocks(image_paths: Iterable[str | Path]) -> str:
    paths = [str(image_path) for image_path in image_paths]
    if not paths:
        return ""
    if len(paths) == 1:
        return f"[Image]\n{paths[0]}"
    return "\n".join(f"[Image {index}]\n{path}" for index, path in enumerate(paths, start=1))


def compose_pure_prompt(before_images: str, image_paths: Iterable[str | Path], after_images: str) -> str:
    parts: list[str] = []
    before_text = str(before_images).strip()
    after_text = str(after_images).strip()
    image_block = _format_image_blocks(image_paths)
    if before_text and image_block:
        parts.append(f"{before_text}\n{image_block}")
    elif before_text:
        parts.append(before_text)
    elif image_block:
        parts.append(image_block)
    if after_text:
        parts.append(after_text)
    return "\n\n".join(parts)


def is_meta_prompt(question: Any) -> bool:
    if not isinstance(question, str):
        return False
    text = question.lstrip()
    return text.startswith("[Task]") and "[Answer Format]" in text


def split_pure_prompt(pure_prompt: str, *, image_count: int) -> tuple[str, str]:
    text = str(pure_prompt).strip()
    if not text:
        return "", ""
    if not is_meta_prompt(text):
        return text, ""

    lines = text.splitlines()
    first_image_index: int | None = None
    for index, line in enumerate(lines):
        if _IMAGE_MARKER_RE.fullmatch(line.strip()):
            first_image_index = index
            break
    if first_image_index is None:
        for index, line in enumerate(lines):
            if line.strip() == "[Rules]":
                return "\n".join(lines[:index]).strip(), "\n".join(lines[index:]).strip()
        return text, ""

    index = first_image_index
    for _ in range(max(0, int(image_count))):
        if index >= len(lines) or _IMAGE_MARKER_RE.fullmatch(lines[index].strip()) is None:
            return text, ""
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            return text, ""
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1

    return "\n".join(lines[:first_image_index]).strip(), "\n".join(lines[index:]).strip()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def flatten_answer_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "answer" in value and len(value) == 1:
            return flatten_answer_value(value["answer"])
        if "slot_index" in value and len(value) == 1:
            return flatten_answer_value(value["slot_index"])
        if "cell_index" in value and len(value) == 1:
            return flatten_answer_value(value["cell_index"])
        if "action_index" in value and len(value) == 1:
            return flatten_answer_value(value["action_index"])
        if {"src_row", "src_col", "tgt_row", "tgt_col"} <= set(value.keys()) and len(value) == 4:
            return [
                flatten_answer_value(value.get("src_row")),
                flatten_answer_value(value.get("src_col")),
                flatten_answer_value(value.get("tgt_row")),
                flatten_answer_value(value.get("tgt_col")),
            ]
        if {"from_peg", "to_peg"} <= set(value.keys()) and len(value) == 2:
            return [
                flatten_answer_value(value.get("from_peg")),
                flatten_answer_value(value.get("to_peg")),
            ]
        if len(value) == 1:
            return flatten_answer_value(next(iter(value.values())))
        return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, tuple):
        return [flatten_answer_value(item) for item in value]
    if isinstance(value, list):
        return [flatten_answer_value(item) for item in value]
    return value


def interactive_trajectory_state(*, trajectory_index: int, total_steps: int, success: bool) -> str:
    if total_steps <= 0:
        return "in progress"
    if trajectory_index < total_steps - 1:
        return "in progress"
    return "success" if success else "failed"


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def reorganize_images_by_question_id(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    image_root_name: str = "images",
    remove_unreferenced_dirs: bool = True,
) -> None:
    output_dir = Path(output_dir)
    image_root = output_dir / image_root_name

    for row in rows:
        row_id = row.get("id")
        if not row_id:
            continue
        old_paths = list(row.get("images") or [])
        new_paths: list[str] = []
        for old_rel in old_paths:
            old_abs = (output_dir / old_rel).resolve()
            basename = Path(old_rel).name
            new_rel = f"{image_root_name}/{row_id}/{basename}"
            new_abs = (output_dir / new_rel).resolve()
            if new_abs == old_abs:
                new_paths.append(new_rel)
                continue
            new_abs.parent.mkdir(parents=True, exist_ok=True)
            if not new_abs.exists():
                if old_abs.exists():
                    shutil.copy2(old_abs, new_abs)
                else:
                    new_paths.append(old_rel)
                    continue
            new_paths.append(new_rel)
        row["images"] = new_paths

    valid_dirs = {row["id"] for row in rows if row.get("id")}
    if remove_unreferenced_dirs and image_root.exists():
        for child in image_root.iterdir():
            if child.is_dir() and child.name not in valid_dirs:
                shutil.rmtree(child)
