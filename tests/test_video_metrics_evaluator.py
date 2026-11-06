from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_eval_tasks.video_metrics import evaluator
from video_eval_tasks.video_metrics.types import EpisodeSpec


class _FakeAdapter:
    def evaluate(self, legacy, spec, artifact_root, sampling_mode, overlay_failures_only):
        for directory in ("detections", "static_metrics", "dynamic_metrics", "outcome_metrics", "reports"):
            (artifact_root / directory).mkdir(parents=True, exist_ok=True)
        (artifact_root / "detections" / f"{spec.episode_id}_per_frame_detection.json").write_text(
            json.dumps({"frames": [{"parser_status": "ok"}, {"parser_status": "ok"}]}), encoding="utf-8"
        )
        for kind in ("static", "dynamic", "outcome"):
            suffix = "metrics" if kind == "outcome" else "metrics"
            (artifact_root / f"{kind}_metrics" / f"{spec.episode_id}_{kind}_metrics.json").write_text("{}\n", encoding="utf-8")
        return {
            "num_transitions": 1,
            "static_video_accuracy": 1.0,
            "dynamic_video_accuracy": 1.0,
            "final_confidence": 0.9,
            "outcome_metric_breakdown": {"final_solution_valid": 1.0},
        }


class _FakeLegacy:
    def set_overlay_root(self, path):
        self.overlay_root = path


class _OverlayAdapter(_FakeAdapter):
    def evaluate(self, legacy, spec, artifact_root, sampling_mode, overlay_failures_only):
        raw = super().evaluate(
            legacy, spec, artifact_root, sampling_mode, overlay_failures_only
        )
        overlay = legacy.overlay_root / "fake_detector" / "static"
        overlay.mkdir(parents=True, exist_ok=True)
        (overlay / f"{spec.episode_id}_frame_000_static.png").write_bytes(b"png")
        return raw


class _FailingOverlayAdapter(_OverlayAdapter):
    def evaluate(self, legacy, spec, artifact_root, sampling_mode, overlay_failures_only):
        raw = super().evaluate(
            legacy, spec, artifact_root, sampling_mode, overlay_failures_only
        )
        raw["static_video_accuracy"] = 0.0
        return raw


def _set_run(run: Path, episode_id: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    video = run / "videos" / f"{episode_id}.mp4"
    video.parent.mkdir(exist_ok=True)
    video.touch()
    row = {
        "id": episode_id,
        "meta_info": {
            "task_name": "knots_untangle",
            "difficulty": "easy",
            "seed": 1,
            "initial_state": {
                "gridSize": 3,
                "ropes": [{"color": 0xE53935}],
                "reset_config": {"seed": 1},
            },
        },
        "trajectory": [{"imagined_video": f"videos/{episode_id}.mp4"}],
    }
    (run / "model_answer.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_evaluate_atomically_replaces_old_episode_outputs(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    output = tmp_path / "public-report"
    monkeypatch.setattr(evaluator, "adapter_for", lambda task: _FakeAdapter())
    monkeypatch.setattr(evaluator, "import_module", lambda name: _FakeLegacy())

    _set_run(run, "old")
    evaluator.evaluate_run(run, output, overlays="none")
    assert (output / "episodes" / "old.json").is_file()

    _set_run(run, "new")
    evaluator.evaluate_run(run, output, overlays="none")
    assert not (output / "episodes" / "old.json").exists()
    assert (output / "episodes" / "new.json").is_file()
    assert not any(path.name.startswith(".public-report.tmp-") for path in tmp_path.iterdir())


def test_input_error_is_isolated_and_has_no_phantom_overlay(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    _set_run(run, "valid")
    rows = (run / "model_answer.jsonl").read_text(encoding="utf-8")
    invalid = {
        "id": "invalid",
        "meta_info": {"task_name": "knots_untangle", "seed": 2},
        "trajectory": [{"imagined_video": "videos/missing.mp4"}],
    }
    (run / "model_answer.jsonl").write_text(
        rows + json.dumps(invalid) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(evaluator, "adapter_for", lambda task: _FakeAdapter())
    monkeypatch.setattr(evaluator, "import_module", lambda name: _FakeLegacy())
    report = evaluator.evaluate_run(run, overlays="none")
    assert report["metrics"]["sample_count"] == 2
    assert report["errors"]["by_type"] == {"input_error": 1}
    episode = json.loads(
        (run / "reports/video_metrics/episodes/invalid.json").read_text(encoding="utf-8")
    )
    assert episode["evidence"] == {}


@pytest.mark.parametrize(
    "mode,expected", [("none", False), ("failures", False), ("all", True)]
)
def test_overlay_evidence_only_points_to_existing_output(
    tmp_path: Path, monkeypatch, mode: str, expected: bool
) -> None:
    run = tmp_path / "run"
    _set_run(run, "episode")
    monkeypatch.setattr(evaluator, "adapter_for", lambda task: _OverlayAdapter())
    monkeypatch.setattr(evaluator, "import_module", lambda name: _FakeLegacy())
    evaluator.evaluate_run(run, overlays=mode)
    episode = json.loads(
        (run / "reports/video_metrics/episodes/episode.json").read_text(encoding="utf-8")
    )
    assert ("overlays" in episode["evidence"]) is expected
    if expected:
        assert (run / "reports/video_metrics" / episode["evidence"]["overlays"]).is_dir()


def test_failure_overlay_is_retained_in_failures_mode(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    _set_run(run, "episode")
    monkeypatch.setattr(evaluator, "adapter_for", lambda task: _FailingOverlayAdapter())
    monkeypatch.setattr(evaluator, "import_module", lambda name: _FakeLegacy())
    evaluator.evaluate_run(run, overlays="failures")
    episode = json.loads(
        (run / "reports/video_metrics/episodes/episode.json").read_text(encoding="utf-8")
    )
    assert episode["verdict"]["overall"] == "fail"
    assert episode["evidence"]["overlays"] == "overlays/episode"


def test_missing_required_dynamic_checks_fail_conservatively(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    detection_dir = artifact_root / "detections"
    detection_dir.mkdir(parents=True)
    (detection_dir / "episode_per_frame_detection.json").write_text(
        json.dumps({"frames": [{"parser_status": "ok"}]}), encoding="utf-8"
    )
    spec = EpisodeSpec(
        episode_id="episode",
        source_episode_id="episode",
        task_name="continuity_pipe",
        video_path=tmp_path / "video.mp4",
        video_rel_path="video.mp4",
        difficulty="easy",
        seed=1,
        metadata={},
        model_id=None,
        vlm_text_model_id=None,
        video_gen_model_name=None,
    )
    result = evaluator._result_from_legacy(
        spec,
        {
            "static_video_accuracy": 1.0,
            "dynamic_video_accuracy": None,
            "outcome_metric_breakdown": {"final_solution_valid": 1.0},
        },
        artifact_root,
    )
    assert result.dynamic_valid is False
    assert result.verdict["dynamic"] == "fail"
    assert result.overall_valid is False


def test_custom_output_does_not_replace_foreign_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _set_run(run, "episode")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="not owned"):
        evaluator.evaluate_run(run, foreign)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_outcome_is_reported_separately_from_static_structure(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    for directory in ("detections", "static_metrics", "dynamic_metrics"):
        (artifact_root / directory).mkdir(parents=True, exist_ok=True)
    episode = "episode"
    (artifact_root / "detections" / f"{episode}_per_frame_detection.json").write_text(
        json.dumps({"frames": [{"parser_status": "ok"}]}), encoding="utf-8"
    )
    (artifact_root / "static_metrics" / f"{episode}_static_metrics.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "metrics": {
                            "board_structure_valid": {"pass": True},
                            "final_solution_valid": {"pass": False, "score": 0.0},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "dynamic_metrics" / f"{episode}_dynamic_metrics.json").write_text(
        json.dumps({"transitions": [{"metrics": {"legal_change": {"pass": True}}}]}),
        encoding="utf-8",
    )
    spec = EpisodeSpec(
        episode_id=episode,
        source_episode_id=episode,
        task_name="continuity_pipe",
        video_path=tmp_path / "video.mp4",
        video_rel_path="video.mp4",
        difficulty="easy",
        seed=1,
        metadata={},
        model_id=None,
        vlm_text_model_id=None,
        video_gen_model_name=None,
    )
    result = evaluator._result_from_legacy(spec, {}, artifact_root)
    assert result.task_success is False
    assert result.static_valid is True
    assert result.dynamic_valid is True
    assert result.overall_valid is False
    outcome = json.loads(
        (artifact_root / "outcome_metrics" / f"{episode}_outcome_metrics.json").read_text(encoding="utf-8")
    )
    assert outcome["metrics"]["final_solution_valid"]["pass"] is False
