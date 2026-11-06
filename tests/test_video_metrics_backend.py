from pathlib import Path

import pytest

from video_eval_tasks.cv_backend import backend
from video_eval_tasks.cv_backend.detectors import (
    chat_noir_oracle,
    pipe_oracle,
    swap_oracle,
)
from video_eval_tasks.video_metrics.types import EpisodeSpec


def _spec(task: str, metadata: dict, tmp_path: Path) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode",
        source_episode_id="episode",
        task_name=task,
        video_path=tmp_path / "video.mp4",
        video_rel_path="video.mp4",
        difficulty=str(metadata.get("difficulty") or "easy"),
        seed=int(metadata.get("seed") or 1),
        metadata=metadata,
        model_id=None,
        vlm_text_model_id=None,
        video_gen_model_name=None,
    )


CASES = [
    (
        "knots_untangle",
        "_process_untangle_video",
        {"seed": 1, "initial_state": {"gridSize": 5, "ropes": [{"color": 123}]}},
    ),
    (
        "separation_one_stroke",
        "_process_one_stroke_video",
        {
            "seed": 1,
            "board_size": 2,
            "initial_state": {
                "level_json": {"cells": [["red", None], [None, "blue"]]},
                "reset_config": {"boardSize": 2},
            },
        },
    ),
    (
        "order_swap_2d_puzzle",
        "_process_swap2d_video",
        {
            "seed": 1,
            "initial_state": {
                "initialGrid": [["_", "A"], ["B", "C"]],
                "reset_config": {"gridRows": 2, "gridCols": 2},
            },
        },
    ),
    (
        "continuity_pipe",
        "_process_pipe_video",
        {
            "seed": 1,
            "difficulty": "easy",
            "level": "grid_2x2",
            "initial_state": {"gridSize": 2, "reset_config": {"gridSize": 2}},
        },
    ),
    (
        "enclosure_chat_noir",
        "_process_chat_noir_video",
        {
            "seed": 1,
            "board_radius": 3,
            "initial_block_count": 1,
            "cat_policy_name": "easy",
            "initial_state": {"boardRadius": 3, "blockedIndices": [1]},
        },
    ),
]


@pytest.mark.parametrize("task,function_name,metadata", CASES)
def test_backend_facade_is_the_only_legacy_dispatch_boundary(
    task: str, function_name: str, metadata: dict, tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def fake(*args):
        calls.append(args)
        return {"task": task}

    monkeypatch.setattr(backend.pipeline, function_name, fake)
    spec = _spec(task, metadata, tmp_path)
    result = backend.evaluate_episode(task, spec, tmp_path / "artifacts", "sample_41", True)
    assert result == {"task": task}
    assert calls[0][0] == "episode"
    assert calls[0][1] == spec.video_path
    if task in {"order_swap_2d_puzzle", "continuity_pipe", "enclosure_chat_noir"}:
        assert calls[0][-1] is spec.metadata


def test_pipe_oracle_prefers_episode_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        pipe_oracle,
        "_load_index",
        lambda: (_ for _ in ()).throw(AssertionError("default question lookup used")),
    )
    metadata = {
        "difficulty": "easy",
        "initial_state": {
            "gridSize": 2,
            "sourceIndex": 1,
            "reset_config": {
                "gridSize": 2,
                "sourceIndex": 1,
                "solvedMasks": [2, 8, 0, 0],
                "rotations": [1, 2, 0, 0],
            },
        },
    }
    gt = pipe_oracle.load_pipe_gt("grid_2x2", "easy", 7, metadata=metadata)
    assert gt.grid_size == 2
    assert gt.source_index == 1
    assert gt.initial_rotations == (1, 2, 0, 0)


def test_chat_noir_oracle_prefers_episode_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_noir_oracle,
        "_load_index",
        lambda: (_ for _ in ()).throw(AssertionError("default question lookup used")),
    )
    metadata = {
        "board_radius": 3,
        "initial_block_count": 2,
        "cat_policy_name": "medium",
        "initial_state": {
            "boardRadius": 3,
            "catIndex": 4,
            "blockedIndices": [1, 2],
        },
    }
    gt = chat_noir_oracle.load_chat_noir_gt(
        3, 2, "medium", 7, metadata=metadata
    )
    assert gt.cat_index == 4
    assert gt.initial_blocked_indices == (1, 2)


def test_swap_oracle_prefers_episode_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        swap_oracle.swap_env,
        "sample_swap_2d_puzzle_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic fallback used")
        ),
    )
    metadata = {
        "initial_state": {
            "reset_config": {
                "gridRows": 2,
                "gridCols": 2,
                "seed": 7,
                "initialArrangement": ["_", "A", "B", "C"],
                "goalArrangement": ["A", "B", "C", "_"],
                "selectedBlockIds": ["A", "B", "C"],
                "theoreticalMinSteps": 3,
            }
        }
    }
    gt = swap_oracle.load_swap_gt(2, 2, 7, metadata=metadata)
    assert gt.initial_arrangement == ("_", "A", "B", "C")
    assert gt.theoretical_min_steps == 3
