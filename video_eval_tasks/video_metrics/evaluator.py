from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import adapter_for
from .adapters.base import load_detection
from .compat import METRIC_ALIASES, normalize_legacy_report
from .io import load_run
from .reporting import SCHEMA_VERSION, write_reports
from .types import EpisodeResult, EpisodeSpec, READABLE_PARSER_STATUSES


REQUIRED_CHECKS = {
    task: {"outcome", "static", "dynamic"}
    for task in (
        "knots_untangle",
        "separation_one_stroke",
        "order_swap_2d_puzzle",
        "continuity_pipe",
        "enclosure_chat_noir",
    )
}


def _portable_run_source(run_dir: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(run_dir, output_dir)).as_posix()
    except ValueError:
        return "."


def _relative_to(value: Path, root: Path) -> Optional[str]:
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return None


def _rewrite_portable_json_paths(staging: Path, run_dir: Path) -> None:
    """Remove machine-specific absolute paths from detector artifacts."""
    repo_root = Path(__file__).resolve().parents[2]

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, str) or not Path(value).is_absolute():
            return value
        path = Path(value).resolve(strict=False)
        for root in (staging, run_dir, repo_root):
            relative = _relative_to(path, root)
            if relative is not None:
                return relative
        return path.name

    for path in staging.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(rewrite(payload), indent=2) + "\n", encoding="utf-8")


def _bool_score(value: Any) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return bool(value)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_document_rows(rows: List[Dict[str, Any]], *, exclude_outcome: bool = False) -> Optional[bool]:
    applicable: List[Dict[str, Any]] = []
    for row in rows:
        for raw_name, metric in (row.get("metrics") or {}).items():
            name = METRIC_ALIASES.get(raw_name, raw_name)
            if exclude_outcome and name == "final_solution_valid":
                continue
            if metric.get("status") != "not_applicable":
                applicable.append(metric)
    if not applicable:
        return None
    return all(bool(metric.get("pass")) for metric in applicable)


def _outcome_metric(
    spec: EpisodeSpec,
    legacy_report: Dict[str, Any],
    artifact_root: Path,
) -> Dict[str, Any]:
    outcome_path = artifact_root / "outcome_metrics" / f"{spec.episode_id}_outcome_metrics.json"
    outcome_doc = _read_json(outcome_path)
    metrics = outcome_doc.get("metrics") or {}
    metric = metrics.get("final_solution_valid") or metrics.get("goal_reached")
    if not metric:
        static_path = artifact_root / "static_metrics" / f"{spec.episode_id}_static_metrics.json"
        static_doc = _read_json(static_path)
        for frame in reversed(static_doc.get("frames") or []):
            frame_metrics = frame.get("metrics") or {}
            metric = (
                frame_metrics.get("final_solution_valid")
                or frame_metrics.get("crossing_or_solution_state")
                or frame_metrics.get("goal_reached")
            )
            if metric and metric.get("status") != "not_applicable":
                break
    if not metric:
        breakdown = legacy_report.get("outcome_metric_breakdown") or {}
        value = breakdown.get("final_solution_valid", breakdown.get("goal_reached"))
        if value is None:
            value = (legacy_report.get("metric_breakdown") or {}).get("final_solution_valid")
        metric = {"score": value, "pass": _bool_score(value), "status": "ok" if value is not None else "unavailable"}
    metric_score = metric.get("score")
    if metric_score is None and "pass" in metric:
        metric_score = 1.0 if metric.get("pass") else 0.0
    normalized = {
        "score": metric_score,
        "pass": bool(metric.get("pass")),
        "status": metric.get("status", "ok"),
        "confidence": metric.get("confidence"),
        "reason": metric.get("reason"),
    }
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_doc = dict(outcome_doc)
    normalized_metrics = dict(metrics)
    normalized_metrics.pop("goal_reached", None)
    normalized_metrics["final_solution_valid"] = normalized
    normalized_doc.update(
        {
            "schema_version": "video_metrics/v1",
            "video_id": spec.episode_id,
            "task": spec.task_name,
            "metrics": normalized_metrics,
        }
    )
    outcome_path.write_text(
        json.dumps(normalized_doc, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return normalized


def _result_from_legacy(spec: EpisodeSpec, legacy_report: Dict[str, Any], artifact_root: Path) -> EpisodeResult:
    legacy_report = normalize_legacy_report(legacy_report)
    detection = load_detection(artifact_root, spec.episode_id)
    frames = detection.get("frames") or []
    statuses = [str(frame.get("parser_status") or "") for frame in frames]
    final_status = statuses[-1] if statuses else "no_frames"
    scorable = bool(frames) and final_status in READABLE_PARSER_STATUSES
    outcome_metric = _outcome_metric(spec, legacy_report, artifact_root)
    outcome = {"final_solution_valid": outcome_metric.get("score")}
    task_success = scorable and bool(outcome_metric.get("pass"))
    static_doc = _read_json(
        artifact_root / "static_metrics" / f"{spec.episode_id}_static_metrics.json"
    )
    dynamic_doc = _read_json(
        artifact_root / "dynamic_metrics" / f"{spec.episode_id}_dynamic_metrics.json"
    )
    raw_static = _strict_document_rows(
        list(static_doc.get("frames") or []), exclude_outcome=True
    )
    raw_dynamic = _strict_document_rows(list(dynamic_doc.get("transitions") or []))
    if not static_doc:
        raw_static = legacy_report.get("static_video_accuracy")
    if not dynamic_doc:
        raw_dynamic = legacy_report.get("dynamic_video_accuracy")
    required = REQUIRED_CHECKS[spec.task_name]
    static_valid = (
        None if scorable and raw_static is None and "static" not in required
        else scorable and bool(raw_static)
    )
    dynamic_valid = (
        None if scorable and raw_dynamic is None and "dynamic" not in required
        else scorable and bool(raw_dynamic)
    )
    overall_valid = (
        scorable
        and task_success
        and static_valid is not False
        and dynamic_valid is not False
    )
    confidence = legacy_report.get("final_confidence")
    if confidence is None:
        values = [legacy_report.get("static_confidence"), legacy_report.get("dynamic_confidence")]
        usable = [float(value) for value in values if value is not None]
        confidence = sum(usable) / len(usable) if usable else None
    errors: List[Dict[str, str]] = []
    if not scorable:
        errors.append({"type": "detector_failure", "message": f"final parser status: {final_status}"})
    return EpisodeResult(
        episode_id=spec.episode_id,
        source_episode_id=spec.source_episode_id,
        task=spec.task_name,
        difficulty=spec.difficulty,
        seed=spec.seed,
        source_video=spec.video_rel_path,
        model_id=spec.model_id,
        vlm_text_model_id=spec.vlm_text_model_id,
        video_gen_model_name=spec.video_gen_model_name,
        scorable=scorable,
        task_success=task_success,
        static_valid=static_valid,
        dynamic_valid=dynamic_valid,
        overall_valid=overall_valid,
        confidence=float(confidence) if confidence is not None else None,
        verdict={
            "outcome": "pass" if task_success else ("fail" if scorable else "unscorable"),
            "static": "not_applicable" if static_valid is None else ("pass" if static_valid else ("fail" if scorable else "unscorable")),
            "dynamic": "not_applicable" if dynamic_valid is None else ("pass" if dynamic_valid else ("fail" if scorable else "unscorable")),
            "overall": "pass" if overall_valid else ("fail" if scorable else "unscorable"),
        },
        evidence={
            "detection": f"artifacts/detections/{spec.episode_id}_per_frame_detection.json",
            "static": f"artifacts/static_metrics/{spec.episode_id}_static_metrics.json",
            "dynamic": f"artifacts/dynamic_metrics/{spec.episode_id}_dynamic_metrics.json",
            "outcome": f"artifacts/outcome_metrics/{spec.episode_id}_outcome_metrics.json",
        },
        diagnostics={
            "frame_count": len(frames),
            "transition_count": legacy_report.get("num_transitions"),
            "parser_status_counts": dict(sorted(Counter(statuses).items())),
            "failed_frames": legacy_report.get("failed_frames") or [],
            "failed_transitions": legacy_report.get("failed_transitions") or [],
            "failed_outcomes": legacy_report.get("failed_outcomes") or [],
            "low_confidence_frames": legacy_report.get("low_confidence_frames") or [],
            "low_confidence_transitions": legacy_report.get("low_confidence_transitions") or [],
            "process_metrics": legacy_report.get("process_metric_breakdown") or {},
        },
        metric_breakdown={
            "outcome": outcome,
            "static_dynamic": legacy_report.get("metric_breakdown") or {},
        },
        errors=errors,
    )


def _error_result(spec: EpisodeSpec, error_type: str, message: str) -> EpisodeResult:
    # Exceptions from OpenCV/ffmpeg may embed local absolute paths. Keep the
    # useful error type while preserving the report's portable-path contract.
    message = re.sub(r"/(?:[^\s:]+/)+[^\s:]+", "<absolute-path>", message)
    return EpisodeResult(
        episode_id=spec.episode_id,
        source_episode_id=spec.source_episode_id,
        task=spec.task_name,
        difficulty=spec.difficulty,
        seed=spec.seed,
        source_video=spec.video_rel_path,
        model_id=spec.model_id,
        vlm_text_model_id=spec.vlm_text_model_id,
        video_gen_model_name=spec.video_gen_model_name,
        scorable=False,
        task_success=False,
        static_valid=False,
        dynamic_valid=False,
        overall_valid=False,
        confidence=None,
        verdict={key: "unscorable" for key in ("outcome", "static", "dynamic", "overall")},
        evidence={},
        diagnostics={},
        errors=[{"type": error_type, "message": message}],
    )


def _organize_overlays(staging: Path, episode_ids: List[str]) -> None:
    root = staging / "overlays"
    if not root.exists():
        return
    for source in list(root.glob("*/static/*")) + list(root.glob("*/dynamic/*")):
        if not source.is_file():
            continue
        episode_id = next((value for value in sorted(episode_ids, key=len, reverse=True) if source.name.startswith(value + "_")), None)
        if episode_id is None:
            continue
        kind = source.parent.name
        destination = root / episode_id / kind / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    for detector_dir in [path for path in root.iterdir() if path.is_dir() and path.name not in episode_ids]:
        shutil.rmtree(detector_dir)


def _prune_success_overlays(staging: Path, results: List[EpisodeResult]) -> None:
    root = staging / "overlays"
    for result in results:
        needs_review = (
            not result.scorable
            or not result.overall_valid
            or (result.confidence is not None and result.confidence < 0.6)
        )
        episode_dir = root / result.episode_id
        if not needs_review and episode_dir.exists():
            shutil.rmtree(episode_dir)


def _attach_existing_overlay_evidence(staging: Path, results: List[EpisodeResult]) -> None:
    """Publish overlay evidence only when at least one evidence file exists."""
    root = staging / "overlays"
    for result in results:
        episode_dir = root / result.episode_id
        if episode_dir.is_dir() and any(path.is_file() for path in episode_dir.rglob("*")):
            result.evidence["overlays"] = f"overlays/{result.episode_id}"
        else:
            result.evidence.pop("overlays", None)


def _replace_tree(staging: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex}"
    if target.exists():
        target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def evaluate_run(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    overlays: str = "failures",
    sampling_mode: str = "sample_41",
) -> Dict[str, Any]:
    if overlays not in {"none", "failures", "all"}:
        raise ValueError("overlays must be one of: none, failures, all")
    run_dir = run_dir.resolve()
    target = (output_dir or (run_dir / "reports" / "video_metrics")).resolve()
    broad_targets = {Path("/"), Path.home().resolve(), run_dir, Path(__file__).resolve().parents[2]}
    if target in broad_targets:
        raise ValueError(f"refusing to replace broad output directory: {target}")
    if target.exists():
        marker = target / "report.json"
        try:
            marker_schema = json.loads(marker.read_text(encoding="utf-8")).get("schema_version")
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            marker_schema = None
        if marker_schema != SCHEMA_VERSION:
            raise ValueError(
                f"refusing to replace an output directory not owned by {SCHEMA_VERSION}: {target}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    artifact_root = staging / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    results: List[EpisodeResult] = []
    try:
        specs = load_run(run_dir)
        legacy = import_module("video_eval_tasks.cv_backend.backend")
        overlay_root = staging / (".discard_overlays" if overlays == "none" else "overlays")
        legacy.set_overlay_root(overlay_root)
        for spec in specs:
            if spec.input_error:
                results.append(_error_result(spec, "input_error", spec.input_error))
                continue
            try:
                raw = adapter_for(spec.task_name).evaluate(
                    legacy,
                    spec,
                    artifact_root,
                    sampling_mode,
                    overlays != "all",
                )
                results.append(_result_from_legacy(spec, raw, artifact_root))
            except Exception as exc:
                results.append(_error_result(spec, "metric_error", f"{type(exc).__name__}: {exc}"))
        if overlays == "none" and overlay_root.exists():
            shutil.rmtree(overlay_root)
        else:
            _organize_overlays(staging, [spec.episode_id for spec in specs])
            if overlays == "failures":
                _prune_success_overlays(staging, results)
            _attach_existing_overlay_evidence(staging, results)
        # The compatibility layer writes an old research report containing a
        # weighted composite. Public v1 exposes only the normalized episode
        # and aggregate reports, so do not publish that legacy score file.
        legacy_reports = artifact_root / "reports"
        if legacy_reports.exists():
            shutil.rmtree(legacy_reports)
        _rewrite_portable_json_paths(staging, run_dir)
        report = write_reports(staging, _portable_run_source(run_dir, target), results)
        _replace_tree(staging, target)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
