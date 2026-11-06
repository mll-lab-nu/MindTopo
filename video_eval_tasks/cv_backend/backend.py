"""Internal compatibility facade for the production video-metrics evaluator.

The task-specific CV implementation was developed in ``scripts/video_cv_pipeline``.
Public callers use this module so that the legacy private function names and argument
shapes do not leak into ``video_metrics``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .scripts import video_cv_pipeline as pipeline


def set_overlay_root(path: Path) -> None:
    pipeline.set_overlay_root(path)


def _initial_state(spec: Any) -> Dict[str, Any]:
    return dict(spec.metadata.get("initial_state") or {})


def evaluate_episode(
    task_name: str,
    spec: Any,
    artifact_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
) -> Dict[str, Any]:
    """Evaluate one episode through the appropriate legacy detector."""
    state = _initial_state(spec)

    if task_name == "knots_untangle":
        grid_size = int(state["gridSize"])
        colors = [int(rope["color"]) for rope in state["ropes"]]
        return pipeline._process_untangle_video(
            spec.episode_id,
            spec.video_path,
            grid_size,
            colors,
            spec.metadata,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
            False,
        )

    if task_name == "separation_one_stroke":
        reset = dict(state.get("reset_config") or {})
        level = dict(state.get("level_json") or reset.get("levelJson") or {})
        size = int(spec.metadata.get("board_size") or reset["boardSize"])
        return pipeline._process_one_stroke_video(
            spec.episode_id,
            spec.video_path,
            f"{size}x{size}",
            spec.seed,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
            level["cells"],
        )

    if task_name == "order_swap_2d_puzzle":
        reset = dict(state.get("reset_config") or {})
        initial = state.get("initialGrid") or []
        rows = int(reset.get("gridRows") or len(initial) or spec.metadata["grid_rows"])
        cols = int(
            reset.get("gridCols")
            or (len(initial[0]) if initial else 0)
            or spec.metadata["grid_cols"]
        )
        return pipeline._process_swap2d_video(
            spec.episode_id,
            spec.video_path,
            rows,
            cols,
            spec.seed,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
            spec.metadata,
        )

    if task_name == "continuity_pipe":
        reset = dict(state.get("reset_config") or {})
        size = int(state.get("gridSize") or reset["gridSize"])
        level = str(spec.metadata.get("level") or f"grid_{size}x{size}")
        return pipeline._process_pipe_video(
            spec.episode_id,
            spec.video_path,
            level,
            spec.difficulty,
            spec.seed,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
            spec.metadata,
        )

    if task_name == "enclosure_chat_noir":
        radius = int(spec.metadata.get("board_radius") or state["boardRadius"])
        blocked = int(
            spec.metadata.get("initial_block_count")
            or len(state["blockedIndices"])
        )
        policy = str(
            spec.metadata.get("cat_policy_name")
            or (state.get("catPolicy") or {})["name"]
        )
        return pipeline._process_chat_noir_video(
            spec.episode_id,
            spec.video_path,
            radius,
            blocked,
            policy,
            spec.seed,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
            spec.metadata,
        )

    raise ValueError(f"unsupported video metric task: {task_name}")
