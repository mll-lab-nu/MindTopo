from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_eval_tasks.video_metrics.io import load_run


def _write_run(run: Path, rows: list[dict]) -> None:
    run.mkdir(parents=True)
    (run / "model_answer.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(episode_id: str = "episode-1", video: str = "videos/result.mp4", task: str = "knots_untangle") -> dict:
    return {
        "id": episode_id,
        "meta_info": {
            "task_name": task,
            "difficulty": "easy",
            "seed": 1,
            "initial_state": {
                "gridSize": 3,
                "ropes": [{"color": 0xE53935}],
                "reset_config": {"seed": 1},
            },
        },
        "trajectory": [{"imagined_video": video}],
    }


def test_load_run_uses_only_run_relative_video_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, [_row()])
    (run / "videos").mkdir()
    (run / "videos" / "result.mp4").touch()
    spec = load_run(run)[0]
    assert spec.video_rel_path == "videos/result.mp4"
    assert spec.input_error is None


def test_load_run_reports_missing_video_without_dropping_episode(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, [_row()])
    spec = load_run(run)[0]
    assert spec.input_error == "video file does not exist: videos/result.mp4"


def test_load_run_reports_missing_oracle_metadata_without_dropping_episode(tmp_path: Path) -> None:
    run = tmp_path / "run"
    row = _row(episode_id="custom-episode")
    row["meta_info"].pop("initial_state")
    _write_run(run, [row])
    (run / "videos").mkdir()
    (run / "videos" / "result.mp4").touch()
    spec = load_run(run)[0]
    assert spec.input_error == "missing oracle metadata: initial_state"


def test_load_run_reports_invalid_seed_without_crashing_batch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    row = _row()
    row["meta_info"]["seed"] = "not-an-integer"
    _write_run(run, [row])
    (run / "videos").mkdir()
    (run / "videos" / "result.mp4").touch()
    spec = load_run(run)[0]
    assert spec.seed == 0
    assert spec.input_error == "invalid oracle metadata: seed='not-an-integer'"


def test_load_run_prefers_configured_question_metadata(tmp_path: Path) -> None:
    run = tmp_path / "run"
    source = run / "questions.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "id": "custom-episode",
                "meta_info": {
                    "task_name": "knots_untangle",
                    "difficulty": "hard",
                    "seed": 9,
                    "initial_state": {
                        "gridSize": 7,
                        "ropes": [{"color": 0x43A047}],
                        "reset_config": {"seed": 9},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = _row(episode_id="custom-episode")
    row["meta_info"] = {
        "task_name": "knots_untangle",
        "source_question_jsonl": "questions.jsonl",
    }
    (run / "model_answer.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (run / "videos").mkdir()
    (run / "videos" / "result.mp4").touch()
    spec = load_run(run)[0]
    assert spec.input_error is None
    assert spec.seed == 9
    assert spec.metadata["initial_state"]["gridSize"] == 7


@pytest.mark.parametrize("video", ["../outside.mp4", "/tmp/outside.mp4"])
def test_load_run_rejects_nonportable_paths(tmp_path: Path, video: str) -> None:
    run = tmp_path / "run"
    _write_run(run, [_row(video=video)])
    with pytest.raises(ValueError, match="absolute paths|escapes"):
        load_run(run)


def test_load_run_rejects_duplicate_ids_unknown_tasks_and_corrupt_jsonl(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    _write_run(duplicate, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_run(duplicate)

    unknown = tmp_path / "unknown"
    _write_run(unknown, [_row(task="not_a_task")])
    with pytest.raises(ValueError, match="unknown"):
        load_run(unknown)

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "model_answer.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        load_run(corrupt)


def test_load_run_accepts_legacy_video_gen_rel_path_and_splits_multiple_videos(tmp_path: Path) -> None:
    run = tmp_path / "run"
    row = _row()
    row["trajectory"] = [
        {"video_gen_rel_path": "videos/a.mp4"},
        {"imagined_video": "videos/b.mp4"},
    ]
    _write_run(run, [row])
    (run / "videos").mkdir()
    (run / "videos" / "a.mp4").touch()
    (run / "videos" / "b.mp4").touch()
    specs = load_run(run)
    assert [spec.episode_id for spec in specs] == ["episode-1__video_001", "episode-1__video_002"]
