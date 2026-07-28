"""Sample loader for the reasoning path of the interleaved harness.

Reads a per-env `question.jsonl` (sheep) or `questions.jsonl` (2D maze) and
normalizes each row into a `ReasoningSample`. Field-level differences between
envs (e.g. `answer` vs `gt_answer`, where the legal-keys list lives) are
flattened here so the runner sees a single shape.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig

# Reuse planning's repo-relative path helper.
_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))
_TOPOBENCH_EVAL_DIR = _THIS_DIR.parent / "topobench_eval"
if str(_TOPOBENCH_EVAL_DIR) not in sys.path:
    sys.path.append(str(_TOPOBENCH_EVAL_DIR))

from evaluator_utils import resolve_repo_path  # noqa: E402
from topobench_eval.answer_contract import AnswerContract, contract_from_spec  # noqa: E402
from topobench_eval.contract_resolver import name_list_keys, resolve_answer_contract  # noqa: E402


@dataclass
class ReasoningSample:
    id: str
    question: str
    image_paths: List[Path]
    image_relpaths: List[str]
    gt_answer: str
    contract: AnswerContract
    meta_info: dict
    category: List[str]
    type: str
    raw_row: dict = field(default_factory=dict)

    @property
    def legal_spec(self) -> Optional[str]:
        return self.contract.legal_spec

    @property
    def answer_mode(self) -> str:
        return self.contract.legacy_mode


@dataclass
class ReasoningLoadResult:
    samples: List[ReasoningSample]
    rows_in_slice: int           # rows surviving manifest_offset/limit before any drops
    dropped_missing_images: int  # rows referenced PNGs that don't exist on disk
    dropped_malformed: int       # rows that failed schema normalization


def load_samples(
    *,
    env_cfg: DictConfig,
    manifest_offset: int = 0,
    manifest_limit: Optional[int] = None,
) -> ReasoningLoadResult:
    """Load and normalize samples for a reasoning env.

    Reads `env_cfg.question_jsonl`, resolves image paths against `env_cfg.image_root`,
    applies the offset/limit slice, and drops rows whose images are missing on disk.
    Raises FileNotFoundError if the question.jsonl file isn't present (the 2D maze
    case where samples haven't been generated yet). Raises RuntimeError if every
    row in a non-empty slice was dropped — a fresh checkout missing the gitignored
    PNGs would otherwise produce a vacuous "0 correct out of 0" summary that looks
    identical to a real model failure.
    """
    question_path = resolve_repo_path(str(env_cfg.question_jsonl))
    if not question_path.exists():
        raise FileNotFoundError(
            f"Reasoning sample file not found: {question_path}. "
            f"Generate samples first (see env README) before running the reasoning harness."
        )
    image_root = resolve_repo_path(str(env_cfg.image_root))

    rows = _read_jsonl(question_path)
    end = manifest_offset + manifest_limit if manifest_limit is not None else None
    sliced = rows[manifest_offset:end] if end is not None else rows[manifest_offset:]

    samples: List[ReasoningSample] = []
    dropped_missing_images = 0
    dropped_malformed = 0
    for row in sliced:
        sample, drop_reason = _normalize_row(row, image_root=image_root)
        if sample is None:
            if drop_reason == "missing_image":
                dropped_missing_images += 1
            else:
                dropped_malformed += 1
            continue
        samples.append(sample)
    if dropped_missing_images:
        print(
            f"[reasoning-loader] dropped {dropped_missing_images}/{len(sliced)} rows: "
            f"images missing under {image_root}. Did you regenerate the env's image files? "
            f"(question.jsonl is checked in; PNGs are .gitignored.)"
        )
    if dropped_malformed:
        print(f"[reasoning-loader] dropped {dropped_malformed}/{len(sliced)} rows: malformed schema.")
    if sliced and not samples:
        raise RuntimeError(
            f"reasoning loader read {len(sliced)} rows but kept 0 (missing_images={dropped_missing_images}, "
            f"malformed={dropped_malformed}). Refusing to score over an empty set — regenerate sample images "
            f"under {image_root} (see env README) and retry."
        )
    return ReasoningLoadResult(
        samples=samples,
        rows_in_slice=len(sliced),
        dropped_missing_images=dropped_missing_images,
        dropped_malformed=dropped_malformed,
    )


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _normalize_row(row: dict, *, image_root: Path) -> tuple[Optional[ReasoningSample], Optional[str]]:
    """Return (sample, drop_reason). drop_reason ∈ {None, "malformed", "missing_image"}."""
    raw_images = row.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return None, "malformed"
    image_relpaths: List[str] = []
    image_paths: List[Path] = []
    for rel in raw_images:
        rel_str = str(rel)
        abs_path = (image_root / rel_str).resolve() if not Path(rel_str).is_absolute() else Path(rel_str)
        if not abs_path.exists():
            return None, "missing_image"
        image_relpaths.append(rel_str)
        image_paths.append(abs_path)

    gt_value = row.get("gt_answer")
    if gt_value is None:
        gt_value = row.get("answer")
    if gt_value is None:
        return None, "malformed"

    meta_info = dict(row.get("meta_info") or {})
    # Per-env conventions vary on where qtype lives: 2D maze nests it under
    # meta_info.question_type while top-level type="reasoning"; sheep
    # encodes it as the row-level type field. Prefer the nested form first
    # (it's the most specific signal) and only fall back to row-level fields.
    qtype = (
        str(meta_info.get("question_type") or "").strip()
        or str(row.get("question_type") or "").strip()
        or str(row.get("type") or "").strip()
    )
    contract = resolve_answer_contract(row, qtype_override=qtype)

    category_raw = row.get("category")
    if isinstance(category_raw, list):
        category = [str(c) for c in category_raw]
    elif isinstance(category_raw, str):
        category = [category_raw]
    else:
        category = []

    sample = ReasoningSample(
        id=str(row.get("id") or ""),
        question=str(row.get("question") or ""),
        image_paths=image_paths,
        image_relpaths=image_relpaths,
        gt_answer=str(gt_value),
        contract=contract,
        meta_info=meta_info,
        category=category,
        type=str(row.get("type") or qtype or "reasoning_qa"),
        raw_row=dict(row),
    )
    return sample, None


def _infer_legal_spec(row: dict, meta_info: dict, qtype: str) -> tuple[Optional[str], str]:
    """Deprecated compatibility facade over the shared contract resolver."""
    doc = {**row, "meta_info": dict(meta_info)}
    contract = resolve_answer_contract(doc, qtype_override=qtype or None)
    question = str(row.get("question") or "")
    explicit = bool(
        meta_info.get("legal_values_spec")
        or row.get("legal_values_spec")
        or row.get("answer_type")
        or meta_info.get("answer_type")
        or row.get("_answer_type")
        or qtype in {"reachability_set", "bar_removal", "connected_point_list", "door_open"}
        or "[Answer Format]" in question
    )
    spec = contract.legal_spec if explicit else None
    if spec == "NAME_LIST:":
        spec = None
    return spec, contract.legacy_mode


def _name_list_keys_for(qtype: str, meta_info: dict) -> List[str]:
    """Deprecated compatibility facade; use ``name_list_keys`` directly."""
    return name_list_keys(qtype, meta_info)


def _mode_from_spec(spec: str) -> str:
    """Deprecated compatibility facade; use ``contract_from_spec`` directly."""
    return contract_from_spec(spec).legacy_mode


def _mode_from_gt(gt_value: Any) -> str:
    """Deprecated compatibility facade; use ``contract_from_spec`` directly."""
    return contract_from_spec(None, gt_value).legacy_mode
