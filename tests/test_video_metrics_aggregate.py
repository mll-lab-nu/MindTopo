import json
from pathlib import Path

from video_eval_tasks.video_metrics.aggregate import aggregate_runs


def test_cross_run_report_keeps_planning_and_video_in_separate_partitions(tmp_path: Path) -> None:
    planning = tmp_path / "planning-run"
    planning.mkdir()
    (planning / "summary.json").write_text(
        json.dumps({"total_episodes": 10, "success_rate": 0.6}), encoding="utf-8"
    )
    video = tmp_path / "video-run"
    report_dir = video / "reports" / "video_metrics"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "video_metrics/v1",
                "metrics": {
                    "sample_count": 8,
                    "overall_video_accuracy": 0.25,
                    "scorable_rate": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark"
    report = aggregate_runs([planning, video], output)
    assert set(report["pipelines"]) == {"planning", "video"}
    assert "combined" not in report
    assert report["pipelines"]["planning"][0]["primary_metric"] == "success_rate"
    assert report["pipelines"]["video"][0]["primary_metric"] == "overall_video_accuracy"
    assert (output / "benchmark_report.csv").is_file()
    assert (output / "benchmark_report.md").is_file()
