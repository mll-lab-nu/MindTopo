from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


SCHEMA_VERSION = "topobench_reports/v1"


def _locate(run: Path) -> tuple[str, Path]:
    video = run / "reports" / "video_metrics" / "report.json"
    if video.is_file():
        return "video", video
    summary = run / "summary.json"
    if summary.is_file():
        return "planning", summary
    raise FileNotFoundError(f"no video report or planning summary found under {run}")


def aggregate_runs(runs: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for run_value in runs:
        run = run_value.resolve()
        pipeline, path = _locate(run)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if pipeline == "video":
            metrics = payload.get("metrics") or {}
            record = {
                "pipeline": pipeline,
                "run": run.name,
                "sample_count": metrics.get("sample_count", 0),
                "primary_metric": "overall_video_accuracy",
                "primary_value": metrics.get("overall_video_accuracy", 0.0),
                "scorable_rate": metrics.get("scorable_rate"),
                "source": "reports/video_metrics/report.json",
            }
        else:
            record = {
                "pipeline": pipeline,
                "run": run.name,
                "sample_count": payload.get("total_episodes", payload.get("episode_count", 0)),
                "primary_metric": "success_rate",
                "primary_value": payload.get("success_rate", 0.0),
                "scorable_rate": None,
                "source": "summary.json",
            }
        records.append(record)
    report = {
        "schema_version": SCHEMA_VERSION,
        "note": "Planning success and video fidelity are reported in separate pipeline partitions and are not combined.",
        "pipelines": {
            pipeline: [record for record in records if record["pipeline"] == pipeline]
            for pipeline in ("planning", "video")
            if any(record["pipeline"] == pipeline for record in records)
        },
    }
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["pipeline", "run", "sample_count", "primary_metric", "primary_value", "scorable_rate", "source"]
    with (output_dir / "benchmark_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    lines = ["# TopoBench Report", "", report["note"], ""]
    for pipeline, rows in report["pipelines"].items():
        lines.extend([f"## {pipeline.title()}", "", "| Run | Samples | Primary metric | Value | Scorable rate |", "|---|---:|---|---:|---:|"])
        for row in rows:
            scorable = "—" if row["scorable_rate"] is None else f"{100 * row['scorable_rate']:.1f}%"
            lines.append(f"| {row['run']} | {row['sample_count']} | {row['primary_metric']} | {100 * row['primary_value']:.1f}% | {scorable} |")
        lines.append("")
    (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
