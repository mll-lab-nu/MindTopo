"""Canonical Core/Extended release paths and preflight checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("core", "extended")
REASONING_TASKS = (
    "continuity_2d_maze", "continuity_3d_maze", "separation_objects",
    "order_bead_string", "order_origami", "enclosure_sheep",
    "enclosure_hole_detection", "knots_static",
)
PLANNING_TASKS = (
    "continuity_pipe", "separation_one_stroke", "order_swap_2d_puzzle",
    "enclosure_chat_noir", "knots_untangle",
)
TASKS = REASONING_TASKS + PLANNING_TASKS


def selected_split(value: str | None = None) -> str:
    value = value if value is not None else os.environ.get("TOPOBENCH_DATA_SPLIT", "core")
    if value not in SPLITS:
        raise ValueError(f"Unknown data split {value!r}; choose core or extended")
    return value


def question_path(task: str, split: str | None = None) -> Path:
    if task not in TASKS:
        raise ValueError(f"Unknown benchmark task {task!r}")
    return REPO_ROOT / "data" / selected_split(split) / task / "question.jsonl"


def data_root(task: str, split: str | None = None) -> Path:
    return question_path(task, split).parent


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs in {path}")
    return rows


def provenance(path: Path, split: str | None, protocol: str = "benchmark") -> dict[str, Any]:
    release_path = REPO_ROOT / "data" / "manifests" / "release_counts.json"
    release = json.loads(release_path.read_text()) if release_path.exists() else {}
    return {
        "dataset_split": split,
        "dataset_protocol": protocol,
        "dataset_release": release.get("paper_source_commit"),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def require_same_dataset(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject resuming results from a different or unrecorded input dataset."""
    for key in ("dataset_sha256", "dataset_split", "dataset_protocol"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Cannot resume results with a different dataset ({key}); choose a new output directory")


def validate_images(rows: list[dict[str, Any]], root: Path) -> None:
    from PIL import Image
    checked: set[Path] = set()
    for row in rows:
        if not row.get("images"):
            raise ValueError(f"Reasoning instance {row['id']} has no images")
        for relative in row["images"]:
            path = (root / relative).resolve()
            if path in checked:
                continue
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing release image {path}; restore the matching data/assets package. "
                    "Benchmark instances must not be silently filtered."
                )
            with Image.open(path) as image:
                image.verify()
            checked.add(path)


def validate_dataset(task: str, split: str | None = None, *, images: bool = True) -> list[dict[str, Any]]:
    split = selected_split(split)
    path = question_path(task, split)
    rows = read_rows(path)
    release = json.loads((REPO_ROOT / "data/manifests/release_counts.json").read_text())
    task_info = next(item for item in release["tasks"] if item["task"] == task)
    if len(rows) != task_info[split]:
        raise ValueError(f"{task}/{split}: expected {task_info[split]}, found {len(rows)}")
    if any(row.get("meta_info", {}).get("benchmark_split") != split for row in rows):
        raise ValueError(f"Mixed benchmark splits in {path}")
    if images and task in REASONING_TASKS:
        validate_images(rows, path.parent)
    return rows


def _custom_image_root(cfg: Any, rows: list[dict[str, Any]], path: Path) -> Path:
    """Honour env.image_root for custom data, else use the JSONL's directory."""
    configured = getattr(cfg.env, "image_root", None)
    if configured:
        root = Path(str(configured))
        root = root if root.is_absolute() else REPO_ROOT / root
        if rows and all((root / image).is_file() for image in rows[0].get("images") or []):
            return root
    return path.parent


def configure_run(cfg: Any, *, probing: bool = False) -> dict[str, Any]:
    """Resolve data before constructing a model client, writer or environment."""
    from omegaconf import OmegaConf
    task = str(cfg.env.name)
    split_value = getattr(getattr(cfg, "data", None), "split", None)
    probe_path = REPO_ROOT / "data/probing" / task / "question.jsonl"
    override = (getattr(cfg.env, "source_question_jsonl", None)
                or getattr(cfg.env, "question_jsonl", None))
    override_path = None
    if override:
        candidate = Path(str(override))
        candidate = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        canonical = {question_path(task, split) for split in SPLITS}
        if candidate in canonical and candidate != question_path(task, split_value):
            raise ValueError("The explicit source path conflicts with data.split")
        # The stock probing path is only the default of probing runners. Anywhere
        # else it is an explicit choice and must not fall through to Core.
        if candidate not in canonical and not (probing and candidate == probe_path):
            override_path = candidate
    image_root = None
    if override_path is not None:
        path = override_path
        rows = read_rows(path)
        if task in REASONING_TASKS:
            image_root = _custom_image_root(cfg, rows, path)
            validate_images(rows, image_root)
        split, protocol = None, "custom"
    elif probing and split_value is None:
        path = probe_path
        rows = read_rows(path)
        if task in REASONING_TASKS:
            validate_images(rows, path.parent)
        split, protocol = None, "legacy_probing"
    else:
        split, protocol = selected_split(split_value), "benchmark"
        path = question_path(task, split)
        validate_dataset(task, split)
        if probing and getattr(cfg.env, "episode_ids_path", None):
            raise ValueError("Explicit Core/Extended probing requires env.episode_ids_path=null; "
                             "the existing balanced_105 protocol must not be silently changed")
    OmegaConf.set_struct(cfg, False)
    info = provenance(path, split, protocol)
    if protocol == "custom":
        info["dataset_release"] = None
    cfg.data = {"split": split, "protocol": protocol,
                "label": split or protocol, **info}
    portable_path = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    cfg.env.source_question_jsonl = portable_path
    if task in REASONING_TASKS:
        cfg.env.question_jsonl = portable_path
        root = image_root or path.parent
        cfg.env.image_root = str(root.relative_to(REPO_ROOT)) if root.is_relative_to(REPO_ROOT) else str(root)
        cfg.env.data_root = cfg.env.image_root
    run = getattr(cfg, "run", None)
    summary_path = getattr(run, "output_json", None) if run is not None else None
    if summary_path:
        summary_path = Path(str(summary_path))
        summary_path = summary_path if summary_path.is_absolute() else REPO_ROOT / summary_path
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            require_same_dataset(previous, info)
    return info
