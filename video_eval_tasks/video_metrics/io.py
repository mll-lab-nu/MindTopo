from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .types import EpisodeSpec


TASK_ALIASES = {
    "knots_untangle": "knots_untangle",
    "untangle": "knots_untangle",
    "separation_one_stroke": "separation_one_stroke",
    "one_stroke": "separation_one_stroke",
    "order_swap_2d_puzzle": "order_swap_2d_puzzle",
    "swap2d": "order_swap_2d_puzzle",
    "continuity_pipe": "continuity_pipe",
    "pipe": "continuity_pipe",
    "enclosure_chat_noir": "enclosure_chat_noir",
    "chat_noir": "enclosure_chat_noir",
}

DEFAULT_QUESTION_FILES = {
    task: Path("environments") / task / "output" / "question.jsonl"
    for task in (
        "knots_untangle",
        "separation_one_stroke",
        "order_swap_2d_puzzle",
        "continuity_pipe",
        "enclosure_chat_noir",
    )
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_run_path(run_dir: Path, value: str) -> Tuple[Path, str]:
    """Resolve a portable run-relative resource and reject path traversal."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed in a public run: {value}")
    resolved = (run_dir / candidate).resolve()
    root = run_dir.resolve()
    if not _inside(resolved, root):
        raise ValueError(f"run-relative path escapes the run directory: {value}")
    return resolved, resolved.relative_to(root).as_posix()


def _source_question_paths(
    repo_root: Path, run_dir: Path, rows: Iterable[Dict[str, Any]]
) -> List[Path]:
    configured_values: List[str] = []
    for row in rows:
        configured = (row.get("meta_info") or {}).get("source_question_jsonl")
        if configured and str(configured) not in configured_values:
            configured_values.append(str(configured))
    resolved_config = run_dir / "resolved_config.yaml"
    if resolved_config.exists():
        doc = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
        configured = (doc.get("env") or {}).get("source_question_jsonl")
        if configured and str(configured) not in configured_values:
            configured_values.append(str(configured))

    paths: List[Path] = []
    for configured in configured_values:
        path = Path(configured)
        if not path.is_absolute():
            run_candidate = (run_dir / path).resolve()
            path = run_candidate if run_candidate.exists() else (repo_root / path).resolve()
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def _question_indices(
    repo_root: Path, run_dir: Path, rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    paths = _source_question_paths(repo_root, run_dir, rows)
    paths.extend(repo_root / rel for rel in DEFAULT_QUESTION_FILES.values())
    seen: set[Path] = set()
    for path in paths:
        if path is None or path in seen or not path.exists():
            continue
        seen.add(path)
        for question in _read_jsonl(path):
            question_id = str(question.get("id") or "")
            meta = dict(question.get("meta_info") or {})
            task = TASK_ALIASES.get(str(meta.get("task_name") or ""))
            if question_id:
                by_id.setdefault(question_id, meta)
                if task:
                    by_task.setdefault(task, {}).setdefault(question_id, meta)
    return by_id, by_task


def _metadata_input_error(task: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Return a task-specific oracle metadata error without raising the batch."""
    missing: List[str] = []
    state = metadata.get("initial_state")
    if not isinstance(state, dict):
        if task == "order_swap_2d_puzzle":
            state = {}
        else:
            return "missing oracle metadata: initial_state"
    reset = state.get("reset_config")
    reset = reset if isinstance(reset, dict) else {}

    if metadata.get("seed") is None and reset.get("seed") is None:
        missing.append("seed")

    if task == "knots_untangle":
        if state.get("gridSize") is None:
            missing.append("initial_state.gridSize")
        ropes = state.get("ropes")
        if not isinstance(ropes, list) or not ropes:
            missing.append("initial_state.ropes")
        elif any(not isinstance(rope, dict) or rope.get("color") is None for rope in ropes):
            missing.append("initial_state.ropes[].color")
    elif task == "separation_one_stroke":
        level = state.get("level_json") or reset.get("levelJson")
        if not isinstance(level, dict) or not isinstance(level.get("cells"), list):
            missing.append("initial_state.level_json.cells")
        if metadata.get("board_size") is None and reset.get("boardSize") is None:
            missing.append("board_size")
    elif task == "order_swap_2d_puzzle":
        # This task's environment sampler is deterministic from rows/cols/seed,
        # so those three values are a complete oracle contract even when the
        # original source question file is unavailable.
        if reset.get("gridRows") is None and metadata.get("grid_rows") is None:
            missing.append("grid_rows")
        if reset.get("gridCols") is None and metadata.get("grid_cols") is None:
            missing.append("grid_cols")
    elif task == "continuity_pipe":
        for key in ("gridSize", "solvedMasks", "rotations"):
            if reset.get(key) is None:
                missing.append(f"initial_state.reset_config.{key}")
        if reset.get("sourceIndex") is None and state.get("sourceIndex") is None:
            missing.append("initial_state.sourceIndex")
        if metadata.get("difficulty") is None:
            missing.append("difficulty")
    elif task == "enclosure_chat_noir":
        if state.get("boardRadius") is None and metadata.get("board_radius") is None:
            missing.append("board_radius")
        if state.get("catIndex") is None:
            missing.append("initial_state.catIndex")
        if not isinstance(state.get("blockedIndices"), list):
            missing.append("initial_state.blockedIndices")
        policy = state.get("catPolicy") or {}
        if metadata.get("cat_policy_name") is None and not (
            isinstance(policy, dict) and policy.get("name")
        ):
            missing.append("cat_policy_name")

    return f"missing oracle metadata: {', '.join(missing)}" if missing else None


def _video_values(row: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for step in row.get("trajectory") or []:
        raw = step.get("imagined_video") or step.get("video_gen_rel_path")
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            if candidate and str(candidate) not in values:
                values.append(str(candidate))
    return values


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "episode"


def load_run(run_dir: Path) -> List[EpisodeSpec]:
    run_dir = run_dir.resolve()
    answer_path = run_dir / "model_answer.jsonl"
    if not answer_path.is_file():
        raise FileNotFoundError(f"missing standard video-run input: {answer_path}")
    rows = _read_jsonl(answer_path)
    repo_root = Path(__file__).resolve().parents[2]
    questions, questions_by_task = _question_indices(repo_root, run_dir, rows)
    seen_ids: set[str] = set()
    specs: List[EpisodeSpec] = []
    for row_index, row in enumerate(rows, 1):
        source_id = str(row.get("id") or row.get("episode_id") or f"row_{row_index:06d}")
        meta = dict(row.get("meta_info") or {})
        task_raw = str(meta.get("task_name") or row.get("task_name") or "")
        task = TASK_ALIASES.get(task_raw)
        if task is None:
            raise ValueError(f"unknown video metric task {task_raw!r} for episode {source_id}")
        question_meta = questions_by_task.get(task, {}).get(source_id) or questions.get(source_id) or {}
        merged_meta = {**question_meta, **meta}
        model_id = merged_meta.get("model_id") or merged_meta.get("model_name")
        vlm_model = merged_meta.get("vlm_text_model_id") or merged_meta.get("vlm_model_id")
        video_model = merged_meta.get("video_gen_model_name") or merged_meta.get("video_model_id")
        metadata_error = _metadata_input_error(task, merged_meta)
        videos = _video_values(row)
        if not videos:
            videos = [""]
        for video_index, value in enumerate(videos, 1):
            result_id = _safe_id(source_id if len(videos) == 1 else f"{source_id}__video_{video_index:03d}")
            if result_id in seen_ids:
                raise ValueError(f"duplicate video metric episode id: {result_id}")
            seen_ids.add(result_id)
            path: Optional[Path] = None
            rel: Optional[str] = None
            errors: List[str] = []
            if value:
                path, rel = resolve_run_path(run_dir, value)
                if not path.is_file():
                    errors.append(f"video file does not exist: {rel}")
            else:
                errors.append("episode has no imagined_video in trajectory")
            if metadata_error:
                errors.append(metadata_error)
            seed_value = merged_meta.get("seed")
            if seed_value is None:
                state = merged_meta.get("initial_state") or {}
                reset = state.get("reset_config") or {}
                seed_value = reset.get("seed", 0)
            try:
                seed = int(seed_value)
            except (TypeError, ValueError):
                seed = 0
                errors.append(f"invalid oracle metadata: seed={seed_value!r}")
            specs.append(
                EpisodeSpec(
                    episode_id=result_id,
                    source_episode_id=source_id,
                    task_name=task,
                    video_path=path,
                    video_rel_path=rel,
                    difficulty=str(merged_meta.get("difficulty") or merged_meta.get("level") or "unknown"),
                    seed=seed,
                    metadata=merged_meta,
                    model_id=str(model_id) if model_id is not None else None,
                    vlm_text_model_id=str(vlm_model) if vlm_model is not None else None,
                    video_gen_model_name=str(video_model) if video_model is not None else None,
                    input_error="; ".join(errors) or None,
                )
            )
    return specs
