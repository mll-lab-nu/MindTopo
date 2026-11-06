from __future__ import annotations

import csv
import json
from pathlib import Path

from video_eval_tasks.video_metrics.reporting import write_reports
from video_eval_tasks.video_metrics.types import EpisodeResult


def _result(episode_id: str, *, scorable: bool = True, passed: bool = True) -> EpisodeResult:
    valid = scorable and passed
    verdict = "pass" if valid else ("fail" if scorable else "unscorable")
    return EpisodeResult(
        episode_id=episode_id,
        source_episode_id=episode_id,
        task="knots_untangle",
        difficulty="easy",
        seed=1,
        source_video=f"images/{episode_id}.mp4",
        model_id="internvl",
        vlm_text_model_id="internvl3.5",
        video_gen_model_name="wan",
        scorable=scorable,
        task_success=valid,
        static_valid=valid,
        dynamic_valid=valid,
        overall_valid=valid,
        confidence=0.9 if scorable else None,
        verdict={key: verdict for key in ("outcome", "static", "dynamic", "overall")},
        evidence={},
        errors=[] if scorable else [{"type": "detector_failure", "message": "no grid"}],
    )


def test_reports_share_one_count_and_public_schema(tmp_path: Path) -> None:
    results = [_result("pass"), _result("unscorable", scorable=False)]
    report = write_reports(tmp_path, "../..", results)
    assert report["schema_version"] == "video_metrics/v1"
    assert report["metrics"]["sample_count"] == 2
    assert report["metrics"]["overall_video_accuracy"] == 0.5
    assert report["metrics"]["conditional_overall_accuracy"] == 1.0
    assert report["metrics"]["scorable_rate"] == 0.5
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 2
    with (tmp_path / "report.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert "Overall Video Accuracy" in (tmp_path / "report.md").read_text(encoding="utf-8")
    review = (tmp_path / "manual_review.csv").read_text(encoding="utf-8")
    assert "detector_failure" in review
    assert "/u/" not in json.dumps(report)
