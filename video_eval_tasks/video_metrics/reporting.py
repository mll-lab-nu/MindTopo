from __future__ import annotations

import csv
import json
from pathlib import PureWindowsPath
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import EpisodeResult


SCHEMA_VERSION = "video_metrics/v1"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _rate(results: Sequence[EpisodeResult], field: str, conditional: bool = False) -> float:
    rows = [row for row in results if row.scorable] if conditional else list(results)
    if not rows:
        return 0.0
    return sum(bool(getattr(row, field)) for row in rows) / len(rows)


def _applicable_rate(results: Sequence[EpisodeResult], field: str) -> float:
    rows = [row for row in results if getattr(row, field) is not None]
    if not rows:
        return 0.0
    return sum(bool(getattr(row, field)) for row in rows) / len(rows)


def _identity(results: Sequence[EpisodeResult], field: str) -> List[str]:
    return sorted({value for row in results if (value := getattr(row, field))})


def _summary(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    return {
        "sample_count": len(results),
        "scorable_count": sum(row.scorable for row in results),
        "overall_video_accuracy": _rate(results, "overall_valid"),
        "conditional_overall_accuracy": _rate(results, "overall_valid", conditional=True),
        "scorable_rate": _rate(results, "scorable"),
        "task_success_rate": _rate(results, "task_success"),
        "static_validity_rate": _applicable_rate(results, "static_valid"),
        "dynamic_validity_rate": _applicable_rate(results, "dynamic_valid"),
        "mean_verifier_confidence": _mean(row.confidence for row in results if row.scorable),
    }


def build_report(run_source: str, results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    by_task: Dict[str, List[EpisodeResult]] = defaultdict(list)
    by_difficulty: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for result in results:
        by_task[result.task].append(result)
        by_difficulty[result.difficulty].append(result)
    error_counts = Counter(
        error.get("type", "unknown")
        for result in results
        for error in result.errors
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "run_dir": run_source,
            "model_answer": "model_answer.jsonl",
        },
        "identity": {
            "tasks": sorted(by_task),
            "model_id": _identity(results, "model_id"),
            "vlm_text_model_id": _identity(results, "vlm_text_model_id"),
            "video_gen_model_name": _identity(results, "video_gen_model_name"),
        },
        "metrics": _summary(results),
        "by_task": {key: _summary(value) for key, value in sorted(by_task.items())},
        "by_difficulty": {key: _summary(value) for key, value in sorted(by_difficulty.items())},
        "definitions": {
            "overall_video_accuracy": "Fraction of all videos passing every required outcome, static, and dynamic check; unscorable videos count as failures.",
            "conditional_overall_accuracy": "Overall accuracy restricted to videos that the visual verifier can score.",
            "scorable_rate": "Fraction of videos whose required frames can be parsed by the visual verifier.",
            "task_success_rate": "Fraction of all videos whose final state satisfies the task oracle.",
            "static_validity_rate": "Pass rate among videos for which static checks are applicable; unscorable videos remain failures.",
            "dynamic_validity_rate": "Pass rate among videos for which dynamic checks are applicable; unscorable videos remain failures.",
            "verifier_confidence": "Diagnostic ranking signal for manual review; it is not part of any accuracy score.",
        },
        "errors": {
            "count": sum(error_counts.values()),
            "by_type": dict(sorted(error_counts.items())),
        },
        "episodes": [f"episodes/{row.episode_id}.json" for row in results],
    }


CSV_FIELDS = [
    "task",
    "difficulty",
    "episode",
    "source_episode",
    "source_video",
    "scorable",
    "task_success",
    "static_verdict",
    "dynamic_verdict",
    "overall_verdict",
    "confidence",
    "model_id",
    "vlm_text_model_id",
    "video_gen_model_name",
    "evidence",
]


def write_csv(path: Path, results: Sequence[EpisodeResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "task": row.task,
                    "difficulty": row.difficulty,
                    "episode": row.episode_id,
                    "source_episode": row.source_episode_id,
                    "source_video": row.source_video or "",
                    "scorable": row.scorable,
                    "task_success": row.task_success,
                    "static_verdict": row.verdict["static"],
                    "dynamic_verdict": row.verdict["dynamic"],
                    "overall_verdict": row.verdict["overall"],
                    "confidence": "" if row.confidence is None else f"{row.confidence:.6f}",
                    "model_id": row.model_id or "",
                    "vlm_text_model_id": row.vlm_text_model_id or "",
                    "video_gen_model_name": row.video_gen_model_name or "",
                    "evidence": f"episodes/{row.episode_id}.json",
                }
            )


def write_manual_review(path: Path, results: Sequence[EpisodeResult]) -> None:
    fields = ["task", "episode", "reason", "confidence", "evidence", "human_verdict", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            reasons: List[str] = []
            if not row.scorable:
                reasons.append("detector_failure")
            if row.confidence is not None and row.confidence < 0.6:
                reasons.append("low_confidence")
            if row.scorable and not row.overall_valid:
                reasons.append("metric_failure")
            if not reasons:
                continue
            writer.writerow(
                {
                    "task": row.task,
                    "episode": row.episode_id,
                    "reason": ";".join(reasons),
                    "confidence": "" if row.confidence is None else f"{row.confidence:.6f}",
                    "evidence": f"episodes/{row.episode_id}.json",
                    "human_verdict": "",
                    "note": "",
                }
            )


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _table(title: str, groups: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [f"## {title}", "", "| Group | Videos | Scorable | Overall | Outcome | Static | Dynamic |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, metrics in groups.items():
        lines.append(
            f"| {key} | {metrics['sample_count']} | {_pct(metrics['scorable_rate'])} | "
            f"{_pct(metrics['overall_video_accuracy'])} | {_pct(metrics['task_success_rate'])} | "
            f"{_pct(metrics['static_validity_rate'])} | {_pct(metrics['dynamic_validity_rate'])} |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, report: Dict[str, Any], results: Sequence[EpisodeResult]) -> None:
    metrics = report["metrics"]
    lines = [
        "# Video Metrics Report",
        "",
        f"Schema: `{report['schema_version']}`  ",
        f"Input: `{report['input']['model_answer']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Videos | {metrics['sample_count']} |",
        f"| Overall Video Accuracy | {_pct(metrics['overall_video_accuracy'])} |",
        f"| Conditional Overall Accuracy | {_pct(metrics['conditional_overall_accuracy'])} |",
        f"| Outcome / Task Success | {_pct(metrics['task_success_rate'])} |",
        f"| Static Validity | {_pct(metrics['static_validity_rate'])} |",
        f"| Dynamic Validity | {_pct(metrics['dynamic_validity_rate'])} |",
        f"| Scorable Rate | {_pct(metrics['scorable_rate'])} |",
        "",
    ]
    lines.extend(_table("By Task", report["by_task"]))
    lines.extend(_table("By Difficulty", report["by_difficulty"]))
    failures = [row for row in results if not row.overall_valid]
    lines.extend(["## Failure Cases", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- [{row.episode_id}](episodes/{row.episode_id}.json) — "
                f"{row.task}, {row.verdict['overall']}"
            )
    else:
        lines.append("No failures.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(root: Path, run_source: str, results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    episodes = root / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for result in results:
        (episodes / f"{result.episode_id}.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = build_report(run_source, results)
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(root / "report.csv", results)
    write_manual_review(root / "manual_review.csv", results)
    write_markdown(root / "report.md", report, results)
    validate_report(root, report, results)
    return report


def validate_report(root: Path, report: Dict[str, Any], results: Sequence[EpisodeResult]) -> None:
    expected = len(results)
    episode_files = list((root / "episodes").glob("*.json"))
    with (root / "report.csv").open(encoding="utf-8", newline="") as handle:
        csv_count = sum(1 for _ in csv.DictReader(handle))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("video metrics schema validation failed")
    if report["metrics"]["sample_count"] != expected or len(episode_files) != expected or csv_count != expected:
        raise ValueError(
            f"inconsistent report counts: results={expected}, json={len(episode_files)}, csv={csv_count}"
        )
    markdown = (root / "report.md").read_text(encoding="utf-8")
    if f"| Videos | {expected} |" not in markdown:
        raise ValueError("Markdown sample count does not match the machine report")
    def find_absolute(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            return next((found for item in value.values() if (found := find_absolute(item))), None)
        if isinstance(value, list):
            return next((found for item in value if (found := find_absolute(item))), None)
        if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            return value
        return None

    for path in root.rglob("*.json"):
        absolute = find_absolute(json.loads(path.read_text(encoding="utf-8")))
        if absolute:
            raise ValueError(f"non-portable absolute path {absolute!r} leaked into {path}")
    for result in results:
        for label, relative in result.evidence.items():
            evidence_path = root / relative
            if not evidence_path.exists():
                raise ValueError(
                    f"episode {result.episode_id} has missing {label} evidence: {relative}"
                )
