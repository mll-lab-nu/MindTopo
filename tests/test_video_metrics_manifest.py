import json
from pathlib import Path

import pytest

from video_eval_tasks.video_metrics.io import load_run
from video_eval_tasks.video_metrics.manifest import import_manifest


def test_explicit_manifest_import_creates_portable_standard_run(tmp_path: Path) -> None:
    source = tmp_path / "loose.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "task_name": "pipe",
                "video": "loose.mp4",
                "meta_info": {
                    "seed": 1,
                    "difficulty": "easy",
                    "level": "grid_2x2",
                    "initial_state": {
                        "gridSize": 2,
                        "sourceIndex": 0,
                        "reset_config": {
                            "gridSize": 2,
                            "seed": 1,
                            "sourceIndex": 0,
                            "solvedMasks": [2, 8, 0, 0],
                            "rotations": [0, 0, 0, 0],
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = import_manifest(manifest, tmp_path / "run")
    spec = load_run(output)[0]
    assert spec.task_name == "continuity_pipe"
    assert spec.video_rel_path == "videos/sample.mp4"
    assert spec.input_error is None
    assert str(tmp_path) not in (output / "model_answer.jsonl").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        import_manifest(manifest, output)


def test_manifest_without_oracle_metadata_imports_as_explicit_input_error(tmp_path: Path) -> None:
    source = tmp_path / "loose.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps({"id": "custom", "task_name": "pipe", "video": "loose.mp4"})
        + "\n",
        encoding="utf-8",
    )
    spec = load_run(import_manifest(manifest, tmp_path / "run"))[0]
    assert spec.input_error == "missing oracle metadata: initial_state"
