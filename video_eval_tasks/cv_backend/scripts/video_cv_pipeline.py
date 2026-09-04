"""Video-level CV recognition pipeline.

This is the execution-plan adapter for the topo metrics spike.  It keeps the
existing task-specific detectors intact, then writes a common set of artifacts:

  frames/ -> manifests/ -> detections/ -> static_metrics/ -> dynamic_metrics/
  -> reports/summary.csv

Untangle uses topo_spike.parsing.  One Stroke reuses the detector from
one_stroke_overlays.py, then fits a board homography and snaps white stroke
pixels onto legal grid edges for normalized path dynamics.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

# Kept internal to the detector backend. The public evaluator owns its report
# schema and deliberately does not need to be importable by this legacy module.
READABLE_PARSER_STATUSES = frozenset({"ok", "partial", "degraded", "invalid_state", ""})

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - the spike venv includes scipy; keep bare envs usable.
    linear_sum_assignment = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo_spike import config
from topo_spike.decode import decode_bgr, sample_indices
from topo_spike.gt_loader import Episode, UntangledGT, iter_episodes
from topo_spike.overlay import draw_overlay as draw_untangle_overlay
from topo_spike.parsing import (
    ParsedFrame,
    _clean_mask,
    assign_colors,
    parse_frame as parse_untangle_frame,
    rope_pixel_mask,
)
from topo_spike import swap2d, swap_oracle
from topo_spike import pipe_cv, pipe_oracle
from topo_spike import chat_noir_cv, chat_noir_oracle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import one_stroke_overlays as os_overlay  # noqa: E402

# One Stroke's real win condition ("does this cut separate every same-colored
# cell into its own region") already has a ground-truth implementation in the
# level generator/solver -- reuse it instead of judging the final frame with
# pixel heuristics (see `_one_stroke_final_solution_valid` below).
ONE_STROKE_GYM_DIR = config.REPO_ROOT / "environments" / "separation_one_stroke" / "gym"
if str(ONE_STROKE_GYM_DIR) not in sys.path:
    sys.path.insert(0, str(ONE_STROKE_GYM_DIR))

import solver as one_stroke_solver  # noqa: E402


DEFAULT_OUTPUT_ROOT = config.SCRATCH_ROOT / "video_cv_eval_outputs"
ABLATION_OVERLAY_DIR = config.REPORTS_DIR / "overlays"
UNTANGLE_DETECTOR_NAME = "untangle_video_cv"
ONE_STROKE_DETECTOR_NAME = "one_stroke_video_cv"
SWAP2D_DETECTOR_NAME = "swap2d_video_cv"
PIPE_DETECTOR_NAME = "pipe_video_cv"
CHAT_NOIR_DETECTOR_NAME = "chat_noir_video_cv"
PIPE_DEFAULT_DIR = config.REPO_ROOT / "logs" / "video_eval" / "continuity_pipe"
PIPE_ROTATION_ANGLE_LIMIT_DEG = 90.0
CHAT_NOIR_DEFAULT_DIR = config.REPO_ROOT / "logs" / "video_eval" / "enclosure_chat_noir"
ONE_STROKE_DEFAULT_DIR = (
    config.REPO_ROOT / "logs" / "video_eval" / "separation_one_stroke"
)
# The e2e eval folder is the source of truth: it nests one imagined mp4 per episode
# under images/grid_RxC/repeat_N_seed_S/ alongside the episode metadata.
SWAP2D_DEFAULT_DIR = (
    config.REPO_ROOT / "logs" / "video_eval" / "order_swap_2d_puzzle"
)

Metric = Dict[str, object]


def set_overlay_root(path: Path) -> None:
    """Route generated overlays to a caller-owned output tree.

    The public ``video_metrics`` evaluator uses this compatibility hook while
    the task-specific detectors are migrated out of this research module.
    Keeping the routing explicit prevents a benchmark run from writing into
    the historical ``logs/cv_backend`` directory.
    """
    global ABLATION_OVERLAY_DIR
    ABLATION_OVERLAY_DIR = Path(path)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _overlay_dirs(detector_name: str) -> Tuple[Path, Path]:
    root = ABLATION_OVERLAY_DIR / detector_name
    static_dir = root / "static"
    dynamic_dir = root / "dynamic"
    static_dir.mkdir(parents=True, exist_ok=True)
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    return static_dir, dynamic_dir


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fmt_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_overlay_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _default_metric_confidence(status: Optional[object]) -> Optional[float]:
    if status == "not_applicable":
        return None
    if status == "low_confidence":
        return 0.45
    return 0.85


def _metric(
    score: float,
    passed: bool,
    reason: str,
    *,
    evidence: Optional[Dict[str, object]] = None,
    status: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Metric:
    out: Metric = {
        "score": _clamp01(score),
        "pass": bool(passed),
        "reason": reason,
    }
    if confidence is None:
        confidence = _default_metric_confidence(status)
    if confidence is not None:
        out["confidence"] = _clamp01(confidence)
    if evidence:
        out["evidence"] = evidence
    if status:
        out["status"] = status
    return out


def _metric_score(metric: Metric) -> Optional[float]:
    if metric.get("status") in {"not_applicable"}:
        return None
    value = metric.get("score")
    return float(value) if value is not None else None


def _metric_confidence(metric: Metric) -> Optional[float]:
    if metric.get("status") == "not_applicable":
        return None
    value = metric.get("confidence")
    if value is None:
        value = _default_metric_confidence(metric.get("status"))
    return float(value) if value is not None else None


def _metrics_score(metrics: Dict[str, Metric]) -> Optional[float]:
    return _mean(_metric_score(m) for m in metrics.values())


def _metrics_confidence(metrics: Dict[str, Metric]) -> Optional[float]:
    return _mean(_metric_confidence(m) for m in metrics.values())


def _metrics_pass(metrics: Dict[str, Metric]) -> bool:
    applicable = [m for m in metrics.values() if m.get("status") != "not_applicable"]
    return all(bool(m.get("pass")) for m in applicable)


def _auto_goal_achieved(detections: Sequence[Dict[str, object]], goal_metric: Metric) -> bool:
    if not detections:
        return False
    final_status = str(detections[-1].get("parser_status") or "")
    return final_status in READABLE_PARSER_STATUSES and bool(goal_metric.get("pass"))


def _draw_overlay_label(
    img_bgr: np.ndarray,
    label: str,
    *,
    top: int = 0,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.rectangle(img_bgr, (0, top), (img_bgr.shape[1], top + 24), (25, 25, 25), -1)
    cv2.putText(img_bgr, label, (6, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _metrics_overlay_label(
    kind: str,
    score: Optional[float],
    confidence: Optional[float],
    passed: bool,
    auto_goal_achieved: Optional[bool] = None,
) -> str:
    goal = "" if auto_goal_achieved is None else f"auto_goal={int(auto_goal_achieved)} "
    return goal + (
        f"{kind} score={_fmt_overlay_score(score)} "
        f"conf={_fmt_overlay_score(confidence)} pass={passed}"
    )


def _add_confidence(values: List[float], value: object) -> None:
    if value is not None:
        values.append(_clamp01(float(value)))


def _detection_confidence(record: Dict[str, object]) -> Optional[float]:
    values: List[float] = []
    task_type = record.get("task_type")
    if task_type == "untangle":
        _add_confidence(values, (record.get("board") or {}).get("confidence"))
        for rope in record.get("ropes") or []:
            _add_confidence(values, rope.get("confidence"))
            for endpoint in rope.get("endpoints") or []:
                _add_confidence(values, endpoint.get("confidence"))
    elif task_type == "one_stroke":
        _add_confidence(values, (record.get("board") or {}).get("model_confidence"))
        _add_confidence(values, (record.get("start") or {}).get("confidence"))
        _add_confidence(values, (record.get("end") or {}).get("confidence"))
        _add_confidence(values, (record.get("raw_path") or {}).get("confidence"))
        _add_confidence(values, (record.get("normalized_path") or {}).get("snap_confidence"))
    elif task_type == "swap2d":
        # The record-level value includes the legal-board gate.  Averaging raw cell
        # colours alone made duplicate/missing-token motion frames look highly
        # confident even though they were not valid discrete states.
        _add_confidence(values, record.get("confidence"))
    elif task_type in ("pipe", "chat_noir"):
        _add_confidence(values, record.get("confidence"))

    if record.get("parser_status") not in (None, "ok", "partial"):
        values.append(0.45)
    return _mean(values)


def _static_confidence(metrics: Dict[str, Metric], record: Dict[str, object]) -> Optional[float]:
    return _mean((_metrics_confidence(metrics), _detection_confidence(record)))


def _dynamic_confidence(
    metrics: Dict[str, Metric],
    prev_record: Dict[str, object],
    cur_record: Dict[str, object],
) -> Optional[float]:
    return _mean(
        (
            _metrics_confidence(metrics),
            _detection_confidence(prev_record),
            _detection_confidence(cur_record),
        )
    )


def _combine_static_dynamic(static_value: Optional[float], dynamic_value: Optional[float]) -> Optional[float]:
    if static_value is None and dynamic_value is None:
        return None
    if static_value is None:
        return dynamic_value
    if dynamic_value is None:
        return static_value
    return 0.4 * static_value + 0.6 * dynamic_value


def _video_metadata(video_path: Path) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = float(n_frames / fps) if fps > 0 else None
    return {
        "fps": fps,
        "duration_seconds": duration,
        "total_frame_count": n_frames,
        "width": width,
        "height": height,
    }


def _frame_timestamps(indices: Sequence[int], fps: float) -> List[Optional[float]]:
    if fps <= 0:
        return [None for _ in indices]
    return [float(i / fps) for i in indices]


def extract_frames(
    video_path: Path,
    video_id: str,
    task_type: str,
    output_root: Path,
    sampling_mode: str,
    requested_num_frames: int = 41,
) -> Tuple[Dict[str, object], List[np.ndarray]]:
    """Decode, sample, save PNG frames, and write a stable manifest."""
    meta = _video_metadata(video_path)
    n_total = int(meta.get("total_frame_count") or 0)
    if sampling_mode == "all_frames":
        sampled = decode_bgr(video_path)
        indices = list(range(len(sampled)))
    elif sampling_mode == "sample_41":
        if n_total <= 0:
            raise ValueError(f"video contains no readable frames: {video_path}")
        indices = sample_indices(n_total, requested_num_frames)
        sampled = decode_bgr(video_path, indices)
    else:
        raise ValueError(f"unknown sampling mode: {sampling_mode}")

    fps = float(meta.get("fps") or 0.0)
    out_dir = output_root / "frames" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_records: List[Dict[str, object]] = []
    timestamps = _frame_timestamps(indices, fps)
    for out_idx, (src_idx, ts, frame) in enumerate(zip(indices, timestamps, sampled)):
        out_path = out_dir / f"frame_{out_idx:03d}.png"
        cv2.imwrite(str(out_path), frame)
        frame_records.append(
            {
                "frame_id": out_idx,
                "source_frame_index": int(src_idx),
                "timestamp_seconds": ts,
                "path": _rel(out_path),
            }
        )

    manifest = {
        "video_id": video_id,
        "task_type": task_type,
        "source_video_path": _rel(video_path),
        "sampling_mode": sampling_mode,
        "requested_num_frames": requested_num_frames,
        "actual_num_frames": len(sampled),
        "video_metadata": meta,
        "frames": frame_records,
    }
    _write_json(output_root / "manifests" / f"{video_id}_frame_manifest.json", manifest)
    return manifest, sampled


def _connected_components(mask: np.ndarray) -> List[Dict[str, object]]:
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: List[Dict[str, object]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        components.append(
            {
                "component_id": f"cc_{i - 1}",
                "area": area,
                "bbox_xyxy": [x, y, x + w, y + h],
                "centroid_xy": [float(centroids[i, 0]), float(centroids[i, 1])],
            }
        )
    components.sort(key=lambda c: int(c["area"]), reverse=True)
    return components


def _lattice_holes(parsed: ParsedFrame) -> List[Dict[str, object]]:
    lat = parsed.lattice
    if lat is None:
        return []
    holes = []
    step = (float(lat.step_x) + float(lat.step_y)) / 2.0
    for r in range(lat.grid_size):
        for c in range(lat.grid_size):
            x, y = lat.cell_to_px(r, c)
            holes.append(
                {
                    "id": f"hole_{r}_{c}",
                    "cell": [int(r), int(c)],
                    "center_xy": [float(x), float(y)],
                    "radius": float(0.14 * step),
                    "confidence": 1.0,
                }
            )
    return holes


def _rope_endpoints(parsed: ParsedFrame, rope) -> List[Dict[str, object]]:
    endpoints: List[Dict[str, object]] = []
    lat = parsed.lattice
    selected_px = list(getattr(rope, "selected_endpoints_px", []) or [])
    if rope.resolved and lat is not None:
        for idx, cell in enumerate(rope.endpoints_rc):
            if cell is None:
                continue
            if idx < len(selected_px):
                x, y = selected_px[idx]
            else:
                x, y = lat.cell_to_px(*cell)
            endpoints.append(
                {
                    "id": f"{rope.color_name}_ep_{idx}",
                    "xy": [float(x), float(y)],
                    "cell": [int(cell[0]), int(cell[1])],
                    "nearest_hole_id": f"hole_{cell[0]}_{cell[1]}",
                    "confidence": 0.95 if not rope.ambiguous else 0.65,
                }
            )
        return endpoints

    for idx, (x, y) in enumerate(list(rope.endpoint_candidates_px)[:2]):
        endpoints.append(
            {
                "id": f"{rope.color_name}_ep_{idx}",
                "xy": [float(x), float(y)],
                "cell": None,
                "nearest_hole_id": None,
                "confidence": 0.45,
            }
        )
    return endpoints


def _point_in_bbox(xy: Sequence[float], bbox: Sequence[object], margin: float = 4.0) -> bool:
    x, y = float(xy[0]), float(xy[1])
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


def _add_component_endpoint_fallbacks(
    rope_id: str,
    endpoints: List[Dict[str, object]],
    components: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Recover visible free endpoints from disconnected rope components.

    The cap-at-lattice detector misses dragged/free rope ends because those ends
    are not sitting on a detected hole.  When a same-color rope splits into two
    visible components and one component has no selected endpoint, its centroid
    is a useful low-confidence endpoint candidate for temporal tracking.
    """
    if len(endpoints) >= 2:
        return endpoints

    out = list(endpoints)
    for component in components:
        centroid = component.get("centroid_xy")
        bbox = component.get("bbox_xyxy")
        if centroid is None or bbox is None:
            continue
        if any(_point_in_bbox(ep.get("xy") or [math.nan, math.nan], bbox) for ep in out):
            continue
        out.append(
            {
                "id": f"{rope_id}_ep_component_{len(out)}",
                "xy": [float(centroid[0]), float(centroid[1])],
                "cell": None,
                "nearest_hole_id": None,
                "confidence": 0.35,
                "source": "component_centroid_fallback",
                "component_id": component.get("component_id"),
            }
        )
        if len(out) >= 2:
            break
    return out


def _untangle_detection_record(
    frame_id: int,
    image_path: str,
    img_bgr: np.ndarray,
    grid_size: int,
    expected_hex: Sequence[int],
    detect_crossing_candidates: bool,
    lattice_hint: Optional[object] = None,
) -> Tuple[Dict[str, object], ParsedFrame]:
    parsed = parse_untangle_frame(
        img_bgr,
        grid_size,
        list(expected_hex),
        detect_crossings=detect_crossing_candidates,
        lattice_hint=lattice_hint,
    )
    h, w = img_bgr.shape[:2]
    min_area = int(config.CV.rope_min_area_frac * h * w)
    fg = rope_pixel_mask(img_bgr)
    color_masks = assign_colors(img_bgr, fg, list(expected_hex))

    ropes = []
    warnings: List[str] = []
    for rope in parsed.ropes:
        clean = _clean_mask(color_masks[rope.color_hex], min_area)
        components = _connected_components(clean)
        if rope.mask_area < min_area:
            warnings.append(f"{rope.color_name}: missing_or_too_small_mask")
        if rope.n_endpoint_px != 2:
            warnings.append(f"{rope.color_name}: endpoint_count={rope.n_endpoint_px}")
        if rope.n_unconnected_endpoint_candidates:
            warnings.append(f"{rope.color_name}: isolated_endpoint_caps={rope.n_unconnected_endpoint_candidates}")
        endpoints = _add_component_endpoint_fallbacks(
            rope.color_name,
            _rope_endpoints(parsed, rope),
            components,
        )
        fallback_candidates = [
            ep["xy"]
            for ep in endpoints
            if ep.get("source") == "component_centroid_fallback"
        ]
        ropes.append(
            {
                "rope_id": rope.color_name,
                "color": rope.color_name,
                "color_hex": f"#{rope.color_hex:06x}",
                "mask_path": None,
                "segments": [],
                "endpoints": endpoints,
                "connected_components": components,
                "endpoint_candidates_xy": [
                    [float(x), float(y)] for x, y in rope.endpoint_candidates_px
                ],
                "fallback_endpoint_candidates_xy": fallback_candidates,
                "n_endpoint_candidates": int(rope.n_endpoint_candidates),
                "n_selected_endpoints": int(rope.n_endpoint_px),
                "n_unconnected_endpoint_candidates": int(rope.n_unconnected_endpoint_candidates),
                "ambiguous": bool(rope.ambiguous),
                "mask_area": int(rope.mask_area),
                "crossing_candidates": [
                    {
                        "xy": [float(c.x), float(c.y)],
                        "with_rope_id": config.COLOR_NAME_BY_HEX.get(c.color_b, f"#{c.color_b:06x}")
                        if c.color_a == rope.color_hex
                        else config.COLOR_NAME_BY_HEX.get(c.color_a, f"#{c.color_a:06x}"),
                        "confidence": 0.7,
                    }
                    for c in parsed.crossing_points_px
                    if c.color_a == rope.color_hex or c.color_b == rope.color_hex
                ],
                "confidence": 0.9 if rope.resolved and not rope.ambiguous else 0.55,
            }
        )

    record = {
        "frame_id": frame_id,
        "task_type": "untangle",
        "image_path": image_path,
        "board": {
            "bbox_xyxy": [0, 0, int(w), int(h)],
            "grid_size": int(grid_size),
            "confidence": 1.0 if parsed.lattice is not None else 0.0,
        },
        "holes": _lattice_holes(parsed),
        "ropes": ropes,
        "parser_status": parsed.status,
        "checklist": dict(parsed.checklist),
        "crossings_visual": parsed.crossings_visual,
        "crossings_logical": parsed.crossings_logical,
        "warnings": warnings,
    }
    return record, parsed


def _score_bool_list(checks: Sequence[bool]) -> float:
    return sum(1.0 for x in checks if x) / len(checks) if checks else 0.0


def _untangle_static_metrics(record: Dict[str, object], *, is_final_frame: bool) -> Dict[str, Metric]:
    ropes = list(record.get("ropes") or [])
    endpoint_checks = []
    endpoint_failures = []
    integrity_checks = []
    integrity_failures = []

    for rope in ropes:
        rid = str(rope["rope_id"])
        n_end = int(rope.get("n_selected_endpoints") or 0)
        n_cand = int(rope.get("n_endpoint_candidates") or 0)
        isolated = int(rope.get("n_unconnected_endpoint_candidates") or 0)
        endpoints = list(rope.get("endpoints") or [])
        cc = list(rope.get("connected_components") or [])
        area = int(rope.get("mask_area") or 0)

        checks = [
            area > 0,
            n_end == 2,
            len(endpoints) == 2,
            n_cand <= 2,
            isolated == 0,
        ]
        endpoint_checks.extend(checks)
        if not all(checks):
            endpoint_failures.append(
                {
                    "rope_id": rid,
                    "mask_area": area,
                    "selected_endpoints": n_end,
                    "endpoint_records": len(endpoints),
                    "endpoint_candidates": n_cand,
                    "isolated_endpoint_caps": isolated,
                }
            )

        total_cc_area = sum(int(c.get("area") or 0) for c in cc)
        largest = int(cc[0].get("area") or 0) if cc else 0
        largest_ratio = largest / float(total_cc_area) if total_cc_area else 0.0
        n_large_components = sum(
            1 for c in cc if total_cc_area and int(c.get("area") or 0) >= 0.12 * total_cc_area
        )
        ok_integrity = area > 0 and (largest_ratio >= 0.45 or n_large_components <= 2)
        integrity_checks.append(ok_integrity)
        if not ok_integrity:
            integrity_failures.append(
                {
                    "rope_id": rid,
                    "largest_component_ratio": round(largest_ratio, 3),
                    "n_large_components": n_large_components,
                }
            )

    endpoint_score = _score_bool_list(endpoint_checks)
    endpoint_metric = _metric(
        endpoint_score,
        endpoint_score == 1.0,
        "all ropes have two connected endpoint candidates"
        if endpoint_score == 1.0
        else "some ropes have missing, extra, ambiguous, or isolated endpoints",
        evidence={"failures": endpoint_failures} if endpoint_failures else None,
    )

    integrity_score = _score_bool_list(integrity_checks)
    integrity_metric = _metric(
        integrity_score,
        integrity_score >= 0.9,
        "no obvious rope fragmentation detected"
        if integrity_score >= 0.9
        else "one or more rope masks are fragmented or too small",
        evidence={"failures": integrity_failures} if integrity_failures else None,
    )

    crossings = record.get("crossings_visual")
    parser_ok = record.get("parser_status") != "failed"
    if crossings is None:
        crossing_metric = _metric(
            0.5 if parser_ok else 0.0,
            False,
            "crossing state unavailable because parser has no complete solver state",
            evidence={"parser_status": record.get("parser_status")},
            status="low_confidence" if parser_ok else None,
        )
    elif is_final_frame:
        crossing_metric = _metric(
            1.0 if int(crossings) == 0 else 0.0,
            int(crossings) == 0,
            "final frame is untangled"
            if int(crossings) == 0
            else "ropes remain tangled in final-state frame",
            evidence={"crossings_visual": crossings},
        )
    else:
        crossing_metric = _metric(
            1.0,
            True,
            "crossing state is readable; intermediate frames are not required to be untangled",
            evidence={"crossings_visual": crossings},
        )

    return {
        "endpoint_configuration": endpoint_metric,
        "rope_integrity": integrity_metric,
        "final_solution_valid": crossing_metric,
    }


def _endpoint_xy(endpoint: Dict[str, object]) -> Tuple[float, float]:
    xy = endpoint.get("xy") or [math.nan, math.nan]
    return float(xy[0]), float(xy[1])


def _endpoint_jump_threshold(record: Dict[str, object]) -> float:
    holes = record.get("holes") or []
    if len(holes) >= 2:
        pts = [h["center_xy"] for h in holes]
        return 0.5 * min(
            math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            for i in range(len(pts)) for j in range(i + 1, len(pts))
        )
    bbox = (record.get("board") or {}).get("bbox_xyxy") or [0, 0, 1, 1]
    return 0.28 * max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]), 1.0)


def _fallback_min_cost_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return []
    if rows <= cols:
        best = None
        for perm in itertools.permutations(range(cols), rows):
            total = float(sum(cost[i, j] for i, j in enumerate(perm)))
            if best is None or total < best[0]:
                best = (total, [(i, j) for i, j in enumerate(perm)])
        return best[1] if best is not None else []

    best = None
    for row_perm in itertools.permutations(range(rows), cols):
        total = float(sum(cost[i, j] for j, i in enumerate(row_perm)))
        if best is None or total < best[0]:
            best = (total, [(i, j) for j, i in enumerate(row_perm)])
    return best[1] if best is not None else []


def _match_endpoint_sets(
    endpoints_a: Sequence[Dict[str, object]],
    endpoints_b: Sequence[Dict[str, object]],
    *,
    max_distance: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    if not endpoints_a or not endpoints_b:
        return []

    reject_cost = 1e9
    cost = np.full((len(endpoints_a), len(endpoints_b)), reject_cost, dtype=float)
    raw_dist = np.full_like(cost, math.inf)
    for i, ep_a in enumerate(endpoints_a):
        ax, ay = _endpoint_xy(ep_a)
        for j, ep_b in enumerate(endpoints_b):
            bx, by = _endpoint_xy(ep_b)
            d = float(math.hypot(ax - bx, ay - by))
            raw_dist[i, j] = d
            if math.isfinite(d) and (max_distance is None or d <= max_distance):
                cost[i, j] = d

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost)
        assignments = list(zip(row_idx.tolist(), col_idx.tolist()))
    else:
        assignments = _fallback_min_cost_assignment(cost)

    pairs: List[Tuple[int, int, float]] = []
    for i, j in assignments:
        d = float(raw_dist[i, j])
        if not math.isfinite(d):
            continue
        if max_distance is not None and d > max_distance:
            continue
        pairs.append((int(i), int(j), d))
    return sorted(pairs, key=lambda item: item[0])


def _untangle_endpoint_dynamic(
    prev: Dict[str, object],
    cur: Dict[str, object],
) -> Metric:
    prev_ropes = {r["rope_id"]: r for r in prev.get("ropes") or []}
    cur_ropes = {r["rope_id"]: r for r in cur.get("ropes") or []}
    jump_thresh = _endpoint_jump_threshold(prev)

    rope_ids = sorted(set(prev_ropes) | set(cur_ropes))
    if not rope_ids:
        return _metric(1.0, True, "no ropes to track between adjacent frames", status="not_applicable")

    # A transition with N ropes used to fail 0/1 as a single AND over all N --
    # one noisy rope zeroed the whole transition even when the other N-1
    # tracked cleanly, which is why DynamicAccuracy read so low.  Score by
    # the fraction of ropes that actually failed instead, and let ropes the
    # parser itself flagged `ambiguous` (a known detector-confidence
    # limitation, not evidence the video moved the rope illegally) fall into
    # a low-confidence bucket rather than a hard fail.
    hard_failures: List[Dict[str, object]] = []
    low_confidence_ropes: List[Dict[str, object]] = []
    for rid in rope_ids:
        a = prev_ropes.get(rid)
        b = cur_ropes.get(rid)
        if a is None or b is None:
            hard_failures.append({"rope_id": rid, "issue": "rope_missing_in_adjacent_frame"})
            continue
        ambiguous = bool(a.get("ambiguous")) or bool(b.get("ambiguous"))
        eps_a = list(a.get("endpoints") or [])
        eps_b = list(b.get("endpoints") or [])
        if len(eps_a) != 2 or len(eps_b) != 2:
            issue = {
                "rope_id": rid,
                "issue": "endpoint_count_changed_or_unreadable",
                "from_count": len(eps_a),
                "to_count": len(eps_b),
            }
            (low_confidence_ropes if ambiguous else hard_failures).append(issue)
            continue
        effective_thresh = jump_thresh * (2.0 if ambiguous else 1.0)
        matches = _match_endpoint_sets(eps_a, eps_b, max_distance=effective_thresh)
        if len(matches) != 2:
            matched_a = {i for i, _, _ in matches}
            matched_b = {j for _, j, _ in matches}
            issue = {
                "rope_id": rid,
                "issue": "endpoint_matching_failed_or_over_distance_gate",
                "matched_pairs": len(matches),
                "from_unmatched": len(eps_a) - len(matched_a),
                "to_unmatched": len(eps_b) - len(matched_b),
                "distance_gate_px": round(effective_thresh, 1),
            }
            (low_confidence_ropes if ambiguous else hard_failures).append(issue)
            continue
        max_disp = max(d for _, _, d in matches)
        if max_disp > effective_thresh:
            hard_failures.append(
                {
                    "rope_id": rid,
                    "issue": "endpoint_jump_too_far",
                    "max_displacement_px": round(max_disp, 1),
                    "threshold_px": round(effective_thresh, 1),
                    "ambiguous": ambiguous,
                }
            )
        elif ambiguous and max_disp > jump_thresh:
            low_confidence_ropes.append(
                {
                    "rope_id": rid,
                    "issue": "endpoint_jump_near_threshold_ambiguous_rope",
                    "max_displacement_px": round(max_disp, 1),
                    "threshold_px": round(jump_thresh, 1),
                }
            )

    total = len(rope_ids)
    score = (total - len(hard_failures) - 0.5 * len(low_confidence_ropes)) / total
    passed = not hard_failures
    status = "low_confidence" if (not hard_failures and low_confidence_ropes) else None
    if hard_failures:
        reason = f"{len(hard_failures)}/{total} rope(s) disappeared, appeared, or jumped too far between adjacent frames"
    elif low_confidence_ropes:
        reason = f"all ropes trackable, but {len(low_confidence_ropes)}/{total} rely on ambiguous parser state"
    else:
        reason = "all rope endpoints are trackable between adjacent frames"
    return _metric(
        score,
        passed,
        reason,
        evidence={
            "hard_failures": hard_failures,
            "low_confidence_ropes": low_confidence_ropes,
            "total_ropes": total,
        }
        if (hard_failures or low_confidence_ropes)
        else None,
        status=status,
    )


def _untangle_endpoint_hole_signature(record: Dict[str, object]) -> Tuple[Optional[Dict[str, Tuple[str, str]]], List[str]]:
    signature: Dict[str, Tuple[str, str]] = {}
    missing: List[str] = []
    for rope in record.get("ropes") or []:
        rid = str(rope.get("rope_id"))
        holes = []
        for ep in rope.get("endpoints") or []:
            hole = ep.get("nearest_hole_id")
            if hole is None and ep.get("cell") is not None:
                cell = ep.get("cell")
                hole = f"hole_{int(cell[0])}_{int(cell[1])}"
            if hole is not None:
                holes.append(str(hole))
        if len(holes) == 2:
            signature[rid] = tuple(sorted(holes))  # type: ignore[assignment]
        else:
            missing.append(rid)
    return (signature if signature else None), missing


def _untangle_endpoint_motion_summary(prev: Dict[str, object], cur: Dict[str, object]) -> Dict[str, object]:
    prev_ropes = {r["rope_id"]: r for r in prev.get("ropes") or []}
    cur_ropes = {r["rope_id"]: r for r in cur.get("ropes") or []}
    dists: List[float] = []
    unmatched = 0
    for rid in sorted(set(prev_ropes) & set(cur_ropes)):
        eps_a = list(prev_ropes[rid].get("endpoints") or [])
        eps_b = list(cur_ropes[rid].get("endpoints") or [])
        if len(eps_a) != 2 or len(eps_b) != 2:
            unmatched += 2
            continue
        matches = _match_endpoint_sets(eps_a, eps_b)
        dists.extend(float(d) for _, _, d in matches)
        unmatched += max(0, 2 - len(matches))
    if not dists:
        return {
            "matched_endpoint_count": 0,
            "unmatched_endpoint_count": unmatched,
            "max_displacement_px": None,
            "mean_displacement_px": None,
        }
    return {
        "matched_endpoint_count": len(dists),
        "unmatched_endpoint_count": unmatched,
        "max_displacement_px": max(dists),
        "mean_displacement_px": sum(dists) / len(dists),
    }


def _untangle_topology_flicker_consistency(prev: Dict[str, object], cur: Dict[str, object]) -> Metric:
    prev_sig, prev_missing = _untangle_endpoint_hole_signature(prev)
    cur_sig, cur_missing = _untangle_endpoint_hole_signature(cur)
    motion = _untangle_endpoint_motion_summary(prev, cur)
    jump_thresh = _endpoint_jump_threshold(prev)
    stationary_thresh = max(8.0, 0.45 * jump_thresh)
    max_disp = motion.get("max_displacement_px")
    matched = int(motion.get("matched_endpoint_count") or 0)
    unmatched = int(motion.get("unmatched_endpoint_count") or 0)
    stationary = (
        max_disp is not None
        and float(max_disp) <= stationary_thresh
        and unmatched == 0
        and matched > 0
    )

    prev_cross = prev.get("crossings_visual")
    cur_cross = cur.get("crossings_visual")
    crossings_readable = prev_cross is not None and cur_cross is not None
    same_crossings = (not crossings_readable) or int(prev_cross) == int(cur_cross)
    same_signature = prev_sig is not None and cur_sig is not None and prev_sig == cur_sig

    if prev_sig is None and cur_sig is None and not crossings_readable:
        return _metric(
            1.0,
            True,
            "topology flicker cannot be scored because neither endpoint signatures nor crossing counts are readable",
            evidence={
                "prev_missing_signature_ropes": prev_missing,
                "cur_missing_signature_ropes": cur_missing,
            },
            status="not_applicable",
        )

    flicker_reasons = []
    if same_signature and not same_crossings:
        flicker_reasons.append("crossing count changed while endpoint-hole signature stayed fixed")
    if stationary and prev_sig is not None and cur_sig is not None and prev_sig != cur_sig:
        flicker_reasons.append("endpoint-hole signature changed while endpoints were nearly stationary")
    if stationary and not same_crossings:
        flicker_reasons.append("crossing count changed while endpoints were nearly stationary")

    if flicker_reasons:
        return _metric(
            0.0,
            False,
            "; ".join(flicker_reasons),
            evidence={
                "prev_signature": prev_sig,
                "cur_signature": cur_sig,
                "prev_crossings_visual": prev_cross,
                "cur_crossings_visual": cur_cross,
                "stationary_endpoint_threshold_px": round(stationary_thresh, 1),
                "endpoint_motion": {
                    k: (None if v is None else round(float(v), 1) if isinstance(v, float) else v)
                    for k, v in motion.items()
                },
            },
        )

    if not stationary and not same_signature:
        return _metric(
            1.0,
            True,
            "topology changed during visible endpoint motion; flicker check is not applicable to this transition",
            evidence={
                "prev_crossings_visual": prev_cross,
                "cur_crossings_visual": cur_cross,
                "stationary_endpoint_threshold_px": round(stationary_thresh, 1),
                "endpoint_motion": {
                    k: (None if v is None else round(float(v), 1) if isinstance(v, float) else v)
                    for k, v in motion.items()
                },
            },
            status="not_applicable",
        )

    status = None if crossings_readable and prev_sig is not None and cur_sig is not None else "low_confidence"
    return _metric(
        1.0,
        True,
        "endpoint-hole signature and crossing count do not flicker across adjacent frames",
        evidence={
            "same_signature": same_signature,
            "same_crossings": same_crossings,
            "prev_crossings_visual": prev_cross,
            "cur_crossings_visual": cur_cross,
            "stationary": stationary,
            "prev_missing_signature_ropes": prev_missing,
            "cur_missing_signature_ropes": cur_missing,
        },
        status=status,
    )


def _untangle_dynamic_metrics(
    prev: Dict[str, object],
    cur: Dict[str, object],
) -> Dict[str, Metric]:
    return {
        "endpoint_track_consistency": _untangle_endpoint_dynamic(prev, cur),
        "topology_flicker_consistency": _untangle_topology_flicker_consistency(prev, cur),
    }


def _draw_untangle_transition_overlay(
    img_a: np.ndarray,
    img_b: np.ndarray,
    prev: Dict[str, object],
    cur: Dict[str, object],
    metrics: Dict[str, Metric],
    confidence: Optional[float],
    auto_goal_achieved: bool,
    out_path: Path,
) -> None:
    left = img_a.copy()
    right = img_b.copy()
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    offset = left.shape[1]
    canvas = np.hstack([left, right])
    colors = {
        "red": (0, 0, 220),
        "green": (0, 180, 0),
        "blue": (220, 80, 0),
        "yellow": (0, 210, 210),
        "purple": (170, 40, 170),
        "orange": (0, 140, 255),
        "cyan": (210, 210, 0),
        "lime": (60, 200, 150),
    }
    prev_ropes = {r["rope_id"]: r for r in prev.get("ropes") or []}
    cur_ropes = {r["rope_id"]: r for r in cur.get("ropes") or []}
    for rid in sorted(set(prev_ropes) & set(cur_ropes)):
        bgr = colors.get(rid, (255, 255, 255))
        a_eps = list(prev_ropes[rid].get("endpoints") or [])
        b_eps = list(cur_ropes[rid].get("endpoints") or [])
        ambiguous = bool(prev_ropes[rid].get("ambiguous")) or bool(cur_ropes[rid].get("ambiguous"))
        distance_gate = _endpoint_jump_threshold(prev) * (2.0 if ambiguous else 1.0)
        matches = _match_endpoint_sets(a_eps, b_eps, max_distance=distance_gate)
        matched_a = {i for i, _, _ in matches}
        matched_b = {j for _, j, _ in matches}
        for i, j, _ in matches:
            ax, ay = _endpoint_xy(a_eps[i])
            bx, by = _endpoint_xy(b_eps[j])
            p0 = (int(round(ax)), int(round(ay)))
            p1 = (int(round(bx)) + offset, int(round(by)))
            cv2.circle(canvas, p0, 6, bgr, 2)
            cv2.circle(canvas, p1, 6, bgr, 2)
            cv2.line(canvas, p0, p1, bgr, 1)
        for i, ep in enumerate(a_eps):
            if i in matched_a:
                continue
            ax, ay = _endpoint_xy(ep)
            if not (math.isfinite(ax) and math.isfinite(ay)):
                continue
            p = (int(round(ax)), int(round(ay)))
            cv2.drawMarker(canvas, p, bgr, cv2.MARKER_TILTED_CROSS, 14, 2)
        for j, ep in enumerate(b_eps):
            if j in matched_b:
                continue
            bx, by = _endpoint_xy(ep)
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue
            p = (int(round(bx)) + offset, int(round(by)))
            cv2.drawMarker(canvas, p, bgr, cv2.MARKER_TILTED_CROSS, 14, 2)
    endpoint_metric = metrics["endpoint_track_consistency"]
    flicker_metric = metrics.get("topology_flicker_consistency")
    label = (
        f"{_metrics_overlay_label('dynamic', _metrics_score(metrics), confidence, _metrics_pass(metrics), auto_goal_achieved)} "
        f"endpoint={_fmt_overlay_score(endpoint_metric.get('score'))} "
        f"flicker={_fmt_overlay_score(flicker_metric.get('score') if flicker_metric else None)}"
    )
    _draw_overlay_label(canvas, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _safe_video_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _untangle_episodes(
    limit: Optional[int],
    video_dir: Optional[Path] = None,
) -> List[Tuple[str, Path, int, List[int], Dict[str, object]]]:
    """Return (video_id, mp4, grid_size, init_colors, meta)."""
    gt = UntangledGT()
    episodes: List[Tuple[str, Path, int, List[int], Dict[str, object]]] = []
    if video_dir is None:
        for ep in iter_episodes(config.DEFAULT_RUN_DIR):
            if ep.mp4_path is None:
                continue
            g = gt.get(ep.level, ep.seed)
            grid_size = g.grid_size if g is not None else (5 if ep.level == "easy" else 6)
            init_colors = list(g.color_state.keys()) if g is not None and g.color_state else []
            if not init_colors and ep.current_pngs:
                clean = cv2.imread(str(ep.current_pngs[0]))
                if clean is not None:
                    init_colors = [
                        r.color_hex for r in parse_untangle_frame(clean, grid_size, list(config.COLOR_NAME_BY_HEX)).ropes
                        if r.resolved
                    ]
            if not init_colors:
                init_colors = list(config.COLOR_NAME_BY_HEX.keys())
            video_id = _safe_video_id(ep.episode_id)
            episodes.append(
                (
                    video_id,
                    ep.mp4_path,
                    grid_size,
                    init_colors,
                    {"level": ep.level, "seed": ep.seed, "episode_id": ep.episode_id},
                )
            )
    else:
        level_pat = re.compile(r"difficulty_(easy|medium|hard)")
        seed_pat = re.compile(r"seed_(\d+)")
        for mp4 in sorted(video_dir.glob("*.mp4")):
            level_match = level_pat.search(mp4.name)
            seed_matches = seed_pat.findall(mp4.name)
            if not level_match or not seed_matches:
                continue
            level = level_match.group(1)
            # Some regenerated filenames contain both the source seed and the
            # actual output seed, e.g. "...seed_08_seed_09...".  Use the last
            # seed token so each mp4 maps to the intended unique episode id.
            seed = int(seed_matches[-1])
            g = gt.get(level, seed)
            grid_size = g.grid_size if g is not None else (5 if level == "easy" else 6)
            init_colors = list(g.color_state.keys()) if g is not None and g.color_state else list(config.COLOR_NAME_BY_HEX)
            video_id = _safe_video_id(f"{level}_seed_{seed:03d}")
            episodes.append(
                (video_id, mp4, grid_size, init_colors, {"level": level, "seed": seed, "episode_id": video_id})
            )
    return episodes[:limit] if limit is not None else episodes


def _process_untangle_video(
    video_id: str,
    video_path: Path,
    grid_size: int,
    init_colors: Sequence[int],
    meta: Dict[str, object],
    output_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
    detect_crossing_candidates: bool,
) -> Dict[str, object]:
    manifest, frames = extract_frames(video_path, video_id, "untangle", output_root, sampling_mode)
    detections = []
    parsed_frames: List[ParsedFrame] = []
    static_overlay_dir, dynamic_overlay_dir = _overlay_dirs(UNTANGLE_DETECTOR_NAME)
    for stale in itertools.chain(
        static_overlay_dir.glob(f"{video_id}_frame_*_static.png"),
        dynamic_overlay_dir.glob(f"{video_id}_transition_*_dynamic.png"),
    ):
        stale.unlink()
    lattice_hint = None
    for idx, (frame, frame_rec) in enumerate(zip(frames, manifest["frames"])):
        det, parsed = _untangle_detection_record(
            idx,
            str(frame_rec["path"]),
            frame,
            grid_size,
            init_colors,
            detect_crossing_candidates,
            lattice_hint,
        )
        detections.append(det)
        parsed_frames.append(parsed)
        if lattice_hint is None and parsed.lattice is not None:
            lattice_hint = parsed.lattice

    detection_doc = {
        "video_id": video_id,
        "task_type": "untangle",
        "metadata": meta,
        "frames": detections,
    }
    _write_json(output_root / "detections" / f"{video_id}_per_frame_detection.json", detection_doc)

    final_goal_metric = (
        _untangle_static_metrics(detections[-1], is_final_frame=True)["final_solution_valid"]
        if detections
        else {}
    )
    auto_goal_achieved = _auto_goal_achieved(detections, final_goal_metric)

    static_frame_ids = sample_indices(len(detections), 4)
    static_frames = []
    for idx in static_frame_ids:
        det = detections[idx]
        metrics = _untangle_static_metrics(det, is_final_frame=idx == len(detections) - 1)
        static_score = _metrics_score(metrics)
        static_confidence = _static_confidence(metrics, det)
        static_pass = _metrics_pass(metrics)
        static_frames.append(
            {
                "frame_id": idx,
                "metrics": metrics,
                "static_score": static_score,
                "static_confidence": static_confidence,
            }
        )
        static_failed = not static_pass
        if (not overlay_failures_only) or static_failed:
            suffix = " static fail" if static_failed else ""
            overlay = draw_untangle_overlay(frames[idx], parsed_frames[idx], f"{video_id} frame {idx:03d}{suffix}")
            label = _metrics_overlay_label(
                "static", static_score, static_confidence, static_pass, auto_goal_achieved
            )
            _draw_overlay_label(overlay, label, top=24)
            cv2.imwrite(str(static_overlay_dir / f"{video_id}_frame_{idx:03d}_static.png"), overlay)

    static_doc = {
        "video_id": video_id,
        "task_type": "untangle",
        "frames": static_frames,
    }
    _write_json(output_root / "static_metrics" / f"{video_id}_static_metrics.json", static_doc)

    dynamic_transitions = []
    for idx, (prev, cur) in enumerate(zip(detections, detections[1:])):
        metrics = _untangle_dynamic_metrics(prev, cur)
        dynamic_score = _metrics_score(metrics)
        dynamic_confidence = _dynamic_confidence(metrics, prev, cur)
        row = {
            "from_frame_id": idx,
            "to_frame_id": idx + 1,
            "metrics": metrics,
            "dynamic_score": dynamic_score,
            "dynamic_confidence": dynamic_confidence,
        }
        dynamic_transitions.append(row)
        if (not overlay_failures_only) or any(not bool(m.get("pass")) for m in metrics.values()):
            _draw_untangle_transition_overlay(
                frames[idx],
                frames[idx + 1],
                prev,
                cur,
                metrics,
                dynamic_confidence,
                auto_goal_achieved,
                dynamic_overlay_dir / f"{video_id}_transition_{idx:03d}_{idx + 1:03d}_dynamic.png",
            )

    dynamic_doc = {
        "video_id": video_id,
        "task_type": "untangle",
        "transitions": dynamic_transitions,
    }
    _write_json(output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json", dynamic_doc)
    return _write_video_report(video_id, "untangle", manifest, static_doc, dynamic_doc, output_root)


def _parse_one_stroke_name(path: Path) -> Optional[Tuple[str, int]]:
    m = re.search(r"size_(\d+x\d+)_seed_(\d+)", path.name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _one_stroke_videos(limit: Optional[int], video_dir: Optional[Path]) -> List[Tuple[str, Path, str, int]]:
    root = video_dir or ONE_STROKE_DEFAULT_DIR
    out: List[Tuple[str, Path, str, int]] = []
    for mp4 in sorted(root.glob("*.mp4")):
        parsed = _parse_one_stroke_name(mp4)
        if parsed is None:
            continue
        size, seed = parsed
        out.append((_safe_video_id(f"{size}_seed_{seed:03d}"), mp4, size, seed))
    return out[:limit] if limit is not None else out


def _one_stroke_board_size(size: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", size)
    if not m:
        return 0, 0
    return int(m.group(2)), int(m.group(1))


Node = Tuple[int, int]
Edge = Tuple[Node, Node]


@dataclass
class OneStrokeBoardModel:
    rows: int
    cols: int
    homography: np.ndarray
    inv_homography: np.ndarray
    grid_nodes: Dict[Node, Tuple[float, float]]
    quality: float
    warnings: List[str]


def _kmeans_1d(values: Sequence[float], k: int, n_iter: int = 20) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == 0 or k <= 0:
        return []
    if len(vals) < k:
        return sorted(float(v) for v in vals)
    lo, hi = float(vals.min()), float(vals.max())
    centers = np.linspace(lo, hi, k)
    for _ in range(n_iter):
        dist = np.abs(vals[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            group = vals[labels == idx]
            if len(group):
                new_centers[idx] = float(np.median(group))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return sorted(float(c) for c in centers)


def _nearest_index(value: float, centers: Sequence[float]) -> int:
    return int(min(range(len(centers)), key=lambda idx: abs(float(value) - centers[idx])))


def _fit_one_stroke_board_model(
    parsed: os_overlay.ParseResult,
    rows: int,
    cols: int,
) -> Optional[OneStrokeBoardModel]:
    """Fit board-coordinate -> image homography from detected physical cell centers.

    The old detector emits colored connected components.  Some are marker blobs
    or stroke-split fragments, so this fitter assigns components to an expected
    rows x cols grid and keeps one strong representative per slot.
    """
    warnings: List[str] = []
    cells = list(parsed.cells)
    if rows <= 0 or cols <= 0 or len(cells) < 4:
        return None

    xs = [float(c.center[0]) for c in cells]
    ys = [float(c.center[1]) for c in cells]
    x_centers = _kmeans_1d(xs, cols)
    y_centers = _kmeans_1d(ys, rows)
    if len(x_centers) < cols or len(y_centers) < rows:
        return None

    slot_cells: Dict[Tuple[int, int], os_overlay.Blob] = {}
    for cell in cells:
        row = _nearest_index(float(cell.center[1]), y_centers)
        col = _nearest_index(float(cell.center[0]), x_centers)
        old = slot_cells.get((row, col))
        if old is None or int(cell.area) > int(old.area):
            slot_cells[(row, col)] = cell

    if len(slot_cells) < 4:
        return None
    if len(slot_cells) < rows * cols:
        warnings.append(f"board_model_missing_slots={rows * cols - len(slot_cells)}")

    src = []
    dst = []
    for (row, col), cell in sorted(slot_cells.items()):
        src.append([float(col) + 0.5, float(row) + 0.5])
        dst.append([float(cell.center[0]), float(cell.center[1])])
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 1, 2)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 1, 2)
    method = cv2.RANSAC if len(src) >= 5 else 0
    cv2.setRNGSeed(0)
    homography, inliers = cv2.findHomography(src_arr, dst_arr, method, 12.0)
    if homography is None:
        return None
    inv = np.linalg.inv(homography)
    projected = cv2.perspectiveTransform(src_arr, homography).reshape(-1, 2)
    target = dst_arr.reshape(-1, 2)
    residual = np.linalg.norm(projected - target, axis=1)
    quality = float(1.0 / (1.0 + np.median(residual) / 20.0))
    if inliers is not None and int(inliers.sum()) < len(src):
        warnings.append(f"board_model_ransac_inliers={int(inliers.sum())}/{len(src)}")

    grid_nodes: Dict[Node, Tuple[float, float]] = {}
    node_src = []
    node_keys = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            node_keys.append((row, col))
            node_src.append([float(col), float(row)])
    node_dst = cv2.perspectiveTransform(
        np.asarray(node_src, dtype=np.float64).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    for key, (x, y) in zip(node_keys, node_dst):
        grid_nodes[key] = (float(x), float(y))

    return OneStrokeBoardModel(rows, cols, homography, inv, grid_nodes, quality, warnings)


def _project_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=float)
    pts = points_xy.astype(np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _edge_label(edge: Edge) -> str:
    a, b = edge
    return f"{a[0]},{a[1]}-{b[0]},{b[1]}"


def _nearest_legal_grid_edge(
    col_f: float,
    row_f: float,
    model: OneStrokeBoardModel,
    segment_margin: float,
) -> Optional[Tuple[float, Edge, float]]:
    best: Optional[Tuple[float, Edge, float]] = None

    col_line = int(round(float(col_f)))
    if 0 <= col_line <= model.cols and -segment_margin <= row_f <= model.rows + segment_margin:
        row_seg = int(math.floor(max(0.0, min(float(row_f), model.rows - 1e-6))))
        row_seg = max(0, min(model.rows - 1, row_seg))
        edge = _canonical_edge((row_seg, col_line), (row_seg + 1, col_line))
        best = (
            abs(float(col_f) - col_line),
            edge,
            max(0.0, min(1.0, float(row_f) - row_seg)),
        )

    row_line = int(round(float(row_f)))
    if 0 <= row_line <= model.rows and -segment_margin <= col_f <= model.cols + segment_margin:
        col_seg = int(math.floor(max(0.0, min(float(col_f), model.cols - 1e-6))))
        col_seg = max(0, min(model.cols - 1, col_seg))
        edge = _canonical_edge((row_line, col_seg), (row_line, col_seg + 1))
        dist = abs(float(row_f) - row_line)
        if best is None or dist < best[0]:
            best = (
                dist,
                edge,
                max(0.0, min(1.0, float(col_f) - col_seg)),
            )

    return best


def _stroke_centerline_points_xy(stroke_mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(stroke_mask > 0)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.stack([xs, ys], axis=1).astype(float)


def _one_stroke_model_step_px(model: OneStrokeBoardModel) -> float:
    lengths = []
    for (row, col), (x, y) in model.grid_nodes.items():
        for nxt in ((row, col + 1), (row + 1, col)):
            if nxt in model.grid_nodes:
                nx, ny = model.grid_nodes[nxt]
                lengths.append(math.hypot(nx - x, ny - y))
    return float(np.median(lengths)) if lengths else 60.0


def _skeleton_endpoint_points(stroke_mask: np.ndarray) -> List[Tuple[float, float]]:
    skel = skeletonize(stroke_mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []
    padded = np.pad(skel, 1)
    neighbor_count = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    ey, ex = np.where((skel > 0) & (neighbor_count == 1))
    return [(float(x), float(y)) for x, y in zip(ex, ey)]


def _dedupe_points(points: Sequence[Tuple[float, float]], min_distance: float = 8.0) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        if any(math.hypot(x - ox, y - oy) <= min_distance for ox, oy in out):
            continue
        out.append((float(x), float(y)))
    return out


def _dominant_direction_label(delta_uv: np.ndarray) -> str:
    if float(np.linalg.norm(delta_uv)) < 1e-6:
        return "unknown"
    dcol, drow = float(delta_uv[0]), float(delta_uv[1])
    if abs(dcol) >= abs(drow):
        return "R" if dcol > 0 else "L"
    # Board row increases downward in the fitted image grid; keep the label
    # explicit so the evidence is useful even if a caller uses a different
    # game-coordinate convention.
    return "D" if drow > 0 else "U"


def _arrow_candidate_from_tip(
    parsed: os_overlay.ParseResult,
    model: OneStrokeBoardModel,
    tip_xy: Tuple[float, float],
    *,
    step_px: float,
    interior_hint_xy: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, object]]:
    center_xy = _stroke_centerline_points_xy(parsed.stroke_mask)
    if len(center_xy) < 4:
        return None

    tip = np.asarray(tip_xy, dtype=float)
    radius = max(22.0, 0.55 * step_px)
    rel_center = center_xy - tip[None, :]
    center_dist = np.linalg.norm(rel_center, axis=1)
    local_center = center_xy[(center_dist <= radius) & (center_dist >= 2.0)]
    if len(local_center) < 3:
        return None

    if interior_hint_xy is not None:
        interior_vec = np.asarray(interior_hint_xy, dtype=float) - tip
    else:
        interior_vec = local_center.mean(axis=0) - tip
    norm = float(np.linalg.norm(interior_vec))
    if norm < 1e-6:
        centered = local_center - local_center.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        interior_vec = vh[0]
        if float((local_center.mean(axis=0) - tip) @ interior_vec) < 0:
            interior_vec = -interior_vec
        norm = float(np.linalg.norm(interior_vec))
    if norm < 1e-6:
        return None

    interior_dir = interior_vec / norm
    normal = np.asarray([-interior_dir[1], interior_dir[0]], dtype=float)

    ys, xs = np.where(parsed.stroke_mask > 0)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(float)
    rel = pts - tip[None, :]
    t = rel @ interior_dir
    q = rel @ normal
    local = (t >= -2.0) & (t <= radius) & (np.abs(q) <= radius)
    if int(local.sum()) < 12:
        return None

    t_local = t[local]
    q_local = q[local]
    n_bins = 8
    widths: List[float] = []
    for lo, hi in zip(np.linspace(0.0, radius, n_bins, endpoint=False), np.linspace(radius / n_bins, radius, n_bins)):
        vals = q_local[(t_local >= lo) & (t_local < hi)]
        if len(vals) < 3:
            widths.append(0.0)
            continue
        widths.append(float(np.percentile(vals, 95) - np.percentile(vals, 5)))

    nonzero_widths = [w for w in widths if w > 0]
    if not nonzero_widths:
        return None
    max_width = max(nonzero_widths)
    tail_widths = [w for idx, w in enumerate(widths) if idx >= n_bins // 2 and w > 0]
    tip_widths = [w for idx, w in enumerate(widths) if idx < 2 and w > 0]
    shaft_width = float(np.median(tail_widths or nonzero_widths))
    tip_width = float(np.median(tip_widths or nonzero_widths))
    width_ratio = max_width / max(shaft_width, 1.0)
    pointed_growth = max_width / max(tip_width, 1.0)
    width_frac = max_width / max(step_px, 1.0)
    area_density = int(local.sum()) / max(radius * max(shaft_width, 3.0), 1.0)
    if width_ratio < 1.35 and pointed_growth < 2.2:
        return None

    # Arrowheads are the only terminal feature that rapidly flares wider than
    # the shaft.  A line cap stays close to ratio 1.0, while the rendered
    # triangular head usually lands well above 2.0.
    flare_score = _clamp01((width_ratio - 1.25) / 1.6)
    growth_score = _clamp01((pointed_growth - 1.8) / 2.5)
    width_score = _clamp01((width_frac - 0.12) / 0.22)
    density_score = _clamp01((area_density - 1.0) / 1.5)
    score = 0.42 * growth_score + 0.28 * width_score + 0.20 * flare_score + 0.10 * density_score
    if score < 0.32:
        return None

    head_len = max(14.0, min(radius, 0.42 * step_px))
    base = tip + interior_dir * head_len
    direction = -interior_dir
    uv = _project_points(np.asarray([base, tip], dtype=float), model.inv_homography)
    delta_uv = uv[1] - uv[0]
    confidence = _clamp01(0.35 + 0.65 * score)
    return {
        "tip_xy": [float(tip[0]), float(tip[1])],
        "base_xy": [float(base[0]), float(base[1])],
        "direction_xy": [float(direction[0]), float(direction[1])],
        "direction_uv": [float(delta_uv[0]), float(delta_uv[1])],
        "direction_label": _dominant_direction_label(delta_uv),
        "confidence": float(confidence),
        "width_ratio": float(width_ratio),
        "pointed_growth": float(pointed_growth),
        "max_width_px": float(max_width),
        "shaft_width_px": float(shaft_width),
        "head_length_px": float(head_len),
    }


def _detect_one_stroke_arrows(
    parsed: os_overlay.ParseResult,
    model: Optional[OneStrokeBoardModel],
) -> Tuple[List[Dict[str, object]], Optional[np.ndarray]]:
    min_arrow_area = max(900, int(0.003 * parsed.stroke_mask.size))
    if model is None or parsed.stroke_area < min_arrow_area:
        return [], None

    step_px = _one_stroke_model_step_px(model)
    seed_points: List[Tuple[Tuple[float, float], Optional[Tuple[float, float]]]] = []
    if parsed.stroke_tip_a is not None and parsed.stroke_tip_b is not None:
        tip_a = (float(parsed.stroke_tip_a[0]), float(parsed.stroke_tip_a[1]))
        tip_b = (float(parsed.stroke_tip_b[0]), float(parsed.stroke_tip_b[1]))
        seed_points.extend([(tip_a, tip_b), (tip_b, tip_a)])
    for tip in (parsed.stroke_start_tip, parsed.stroke_end_tip):
        if tip is not None:
            seed_points.append(((float(tip[0]), float(tip[1])), None))
    seed_points.extend((pt, None) for pt in _skeleton_endpoint_points(parsed.stroke_mask))

    deduped: List[Tuple[Tuple[float, float], Optional[Tuple[float, float]]]] = []
    for pt, hint in seed_points:
        if any(math.hypot(pt[0] - old_pt[0], pt[1] - old_pt[1]) <= 8.0 for old_pt, _ in deduped):
            continue
        deduped.append((pt, hint))
    candidates = []
    for pt, hint in deduped:
        candidate = _arrow_candidate_from_tip(parsed, model, pt, step_px=step_px, interior_hint_xy=hint)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    candidates = candidates[:3]
    if not candidates or float(candidates[0].get("confidence") or 0.0) < 0.55:
        return candidates, None

    best = candidates[0]
    tip = np.asarray(best["tip_xy"], dtype=float)
    direction = np.asarray(best["direction_xy"], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return candidates, None
    direction = direction / norm
    interior = -direction
    normal = np.asarray([-interior[1], interior[0]], dtype=float)
    head_len = float(best.get("head_length_px") or max(14.0, 0.42 * step_px))
    half_width = max(6.0, 0.62 * float(best.get("max_width_px") or 0.0))

    ys, xs = np.where(parsed.stroke_mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(float) if len(xs) else np.zeros((0, 2), dtype=float)
    rel = pts - tip[None, :]
    t = rel @ interior
    q = np.abs(rel @ normal)
    remove = (t >= -2.0) & (t <= head_len * 1.08) & (q <= half_width)
    exclusion = np.zeros_like(parsed.stroke_mask)
    if len(pts) and np.any(remove):
        rm = pts[remove].round().astype(int)
        rm[:, 0] = np.clip(rm[:, 0], 0, exclusion.shape[1] - 1)
        rm[:, 1] = np.clip(rm[:, 1], 0, exclusion.shape[0] - 1)
        exclusion[rm[:, 1], rm[:, 0]] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        exclusion = cv2.dilate(exclusion, k)
    return candidates, exclusion


def _max_consecutive_run(values: Iterable[int]) -> int:
    best = 0
    cur = 0
    prev: Optional[int] = None
    for value in sorted(set(int(v) for v in values)):
        cur = cur + 1 if prev is not None and value == prev + 1 else 1
        best = max(best, cur)
        prev = value
    return best


def _assigned_points_for_edges(
    points_xy: np.ndarray,
    points_uv: np.ndarray,
    model: OneStrokeBoardModel,
    edges: Sequence[Edge],
    snap_thresh: float,
    segment_margin: float,
) -> Dict[Edge, np.ndarray]:
    wanted = set(edges)
    by_edge: Dict[Edge, List[np.ndarray]] = defaultdict(list)
    for xy, (col_f, row_f) in zip(points_xy, points_uv):
        best = _nearest_legal_grid_edge(float(col_f), float(row_f), model, segment_margin)
        if best is not None and best[0] <= snap_thresh and best[1] in wanted:
            by_edge[best[1]].append(np.asarray(xy, dtype=float))
    return {edge: np.asarray(pts, dtype=float) for edge, pts in by_edge.items()}


def _fit_segment_for_edge(
    edge: Edge,
    points_xy: np.ndarray,
    model: OneStrokeBoardModel,
) -> Optional[Tuple[List[List[float]], Dict[str, object]]]:
    if len(points_xy) < 2:
        return None
    pa = np.asarray(model.grid_nodes.get(edge[0]), dtype=float)
    pb = np.asarray(model.grid_nodes.get(edge[1]), dtype=float)
    if pa.shape != (2,) or pb.shape != (2,):
        return None
    vec = pb - pa
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None

    direction = vec / length
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    rel = points_xy - pa[None, :]
    along = rel @ direction
    offset_vals = rel @ normal

    if len(offset_vals) >= 8:
        lo, hi = np.percentile(offset_vals, [10, 90])
        trimmed = (offset_vals >= lo) & (offset_vals <= hi)
        if int(trimmed.sum()) >= 4:
            offset_vals = offset_vals[trimmed]
            along = along[trimmed]

    offset = float(np.median(offset_vals))
    along_lo = float(np.percentile(along, 5))
    along_hi = float(np.percentile(along, 95))
    coverage = max(0.0, min(1.0, (along_hi - along_lo) / length))

    # Full edges stay full; partial active tips are drawn only where the fitted
    # centerline has support, which keeps the overlay from overstating the path.
    if coverage >= 0.50:
        seg_lo, seg_hi = 0.0, length
    else:
        pad = max(6.0, 0.06 * length)
        seg_lo = max(0.0, along_lo - pad)
        seg_hi = min(length, along_hi + pad)
        if seg_hi - seg_lo < min(12.0, 0.25 * length):
            mid = max(0.0, min(length, 0.5 * (along_lo + along_hi)))
            half = min(0.5 * length, max(6.0, 0.12 * length))
            seg_lo = max(0.0, mid - half)
            seg_hi = min(length, mid + half)

    # `edge_fit_segments_xy` represents the normalized board edge, not the
    # visible stroke centerline.  Keep the measured normal offset as evidence,
    # but draw/score the fitted segment on the legal grid line itself.
    fit_a = pa + direction * seg_lo
    fit_b = pa + direction * seg_hi
    segment = [
        [float(fit_a[0]), float(fit_a[1])],
        [float(fit_b[0]), float(fit_b[1])],
    ]
    stats = {
        "support_points": int(len(points_xy)),
        "coverage": float(coverage),
        "normal_offset_px": float(offset),
        "drawn_fraction": float((seg_hi - seg_lo) / length),
    }
    return segment, stats


def _fit_snapped_edge_segments(
    stroke_mask: np.ndarray,
    all_points_xy: np.ndarray,
    all_points_uv: np.ndarray,
    model: OneStrokeBoardModel,
    edges: Sequence[Edge],
    snap_thresh: float,
    segment_margin: float,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, Dict[str, object]]]:
    if not edges:
        return {}, {}

    center_xy = _stroke_centerline_points_xy(stroke_mask)
    center_assignments: Dict[Edge, np.ndarray] = {}
    if len(center_xy) >= 2:
        center_uv = _project_points(center_xy, model.inv_homography)
        # The rendered stroke center is often shifted from the ideal grid line by
        # perspective/line thickness.  Use a looser gate for centerline fitting
        # than for metric snapping so visualization can still fit the visible ink.
        center_assignments = _assigned_points_for_edges(
            center_xy,
            center_uv,
            model,
            edges,
            max(snap_thresh + 0.16, snap_thresh * 1.75),
            segment_margin + 0.08,
        )

    fallback_assignments = _assigned_points_for_edges(
        all_points_xy,
        all_points_uv,
        model,
        edges,
        snap_thresh,
        segment_margin,
    )

    segments: Dict[str, List[List[float]]] = {}
    stats: Dict[str, Dict[str, object]] = {}
    node_to_edge_refs: Dict[Node, List[Tuple[Edge, int]]] = defaultdict(list)
    for edge in edges:
        pts = center_assignments.get(edge)
        source = "centerline"
        if pts is None or len(pts) < 4:
            pts = fallback_assignments.get(edge)
            source = "mask"
        if pts is None or len(pts) < 2:
            continue
        fit = _fit_segment_for_edge(edge, pts, model)
        if fit is None:
            continue
        segment, edge_stats = fit
        edge_stats["source"] = source
        key = _edge_label(edge)
        segments[key] = segment
        stats[key] = edge_stats
        node_to_edge_refs[edge[0]].append((edge, 0))
        node_to_edge_refs[edge[1]].append((edge, 1))

    # Each edge above is fit independently (own perpendicular offset, own
    # partial-coverage window), so adjacent retained edges can fail to meet at
    # their shared board node when one side is clipped as a short partial edge.
    # These segments visualize the normalized legal-edge path, not the visible
    # stroke centerline; once two retained edges share a legal node, draw them
    # as a connected polyline.  Real fragmentation is still handled by the
    # stroke-mask and skeleton component checks in path continuity.
    for node, refs in node_to_edge_refs.items():
        if len(refs) < 2:
            continue
        keys_idx = [(_edge_label(e), idx) for e, idx in refs]
        node_pt = np.asarray(model.grid_nodes[node], dtype=float)
        for k, idx in keys_idx:
            segments[k][idx] = [float(node_pt[0]), float(node_pt[1])]

    return segments, stats


def _stroke_line_fit_stats(points_uv: np.ndarray) -> Dict[str, object]:
    if len(points_uv) < 20:
        return {
            "is_straight_diagonal": False,
            "reason": "too_few_points",
        }
    centered = points_uv - points_uv.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "is_straight_diagonal": False,
            "reason": "svd_failed",
        }
    direction = vh[0]
    perp = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
    along = centered @ direction
    angle = abs(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    if angle > 90.0:
        angle = 180.0 - angle
    median_perp = float(np.median(perp))
    p90_perp = float(np.percentile(perp, 90))
    p95_perp = float(np.percentile(perp, 95))
    span = float(along.max() - along.min())
    is_diagonal_angle = 20.0 <= angle <= 70.0
    is_straight = span >= 1.4 and median_perp <= 0.08 and p90_perp <= 0.16 and p95_perp <= 0.30
    is_slight_center_diagonal = (
        span >= 1.4
        and is_diagonal_angle
        and not is_straight
        and median_perp <= 0.28
        and p90_perp <= 0.55
        and p95_perp <= 0.85
    )
    return {
        "is_straight_diagonal": bool(is_straight and is_diagonal_angle),
        "is_slight_center_diagonal": bool(is_slight_center_diagonal),
        "angle_degrees": float(angle),
        "median_perpendicular_error_cells": median_perp,
        "p90_perpendicular_error_cells": p90_perp,
        "p95_perpendicular_error_cells": p95_perp,
        "span_cells": span,
    }


def _canonical_edge(a: Node, b: Node) -> Edge:
    return (a, b) if a <= b else (b, a)


def _edge_to_json(edge: Edge) -> List[List[int]]:
    return [[int(edge[0][0]), int(edge[0][1])], [int(edge[1][0]), int(edge[1][1])]]


def _edge_key(edge: object) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    a, b = edge  # type: ignore[misc]
    aa = (int(a[0]), int(a[1]))
    bb = (int(b[0]), int(b[1]))
    return (aa, bb) if aa <= bb else (bb, aa)


def _nearest_board_node(model: OneStrokeBoardModel, xy: Optional[Tuple[float, float]]) -> Optional[Node]:
    if xy is None:
        return None
    uv = _project_points(np.asarray([[xy[0], xy[1]]], dtype=float), model.inv_homography)[0]
    col = int(round(float(uv[0])))
    row = int(round(float(uv[1])))
    row = max(0, min(model.rows, row))
    col = max(0, min(model.cols, col))
    return row, col


def _ordered_nodes_from_edges(
    edges: Sequence[Edge],
    start_node: Optional[Node],
    end_node: Optional[Node],
) -> List[Node]:
    if not edges:
        return []
    graph: Dict[Node, List[Node]] = defaultdict(list)
    nodes = set()
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
        nodes.add(a)
        nodes.add(b)

    starts = []
    if start_node is not None:
        starts.append(min(nodes, key=lambda n: abs(n[0] - start_node[0]) + abs(n[1] - start_node[1])))
    starts.extend(sorted(n for n in nodes if len(graph[n]) == 1))
    starts.extend(sorted(nodes))
    start = starts[0]

    if end_node is not None:
        target = min(nodes, key=lambda n: abs(n[0] - end_node[0]) + abs(n[1] - end_node[1]))
        queue = [(start, [start])]
        seen = {start}
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return path
            for nxt in sorted(graph[node]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))

    # Fallback: walk unused edges greedily so the JSON still has a stable node sequence.
    used = set()
    path = [start]
    cur = start
    while True:
        nxt = None
        for candidate in sorted(graph[cur]):
            key = _canonical_edge(cur, candidate)
            if key not in used:
                nxt = candidate
                break
        if nxt is None:
            break
        used.add(_canonical_edge(cur, nxt))
        path.append(nxt)
        cur = nxt
    return path


def _normalize_one_stroke_path(
    parsed: os_overlay.ParseResult,
    model: Optional[OneStrokeBoardModel],
    arrow_exclusion_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if model is None:
        return {
            "nodes": [],
            "edges": [],
            "snap_confidence": 0.0,
            "edge_pixel_coverage": {},
            "warnings": ["board_model_unavailable"],
        }
    if parsed.stroke_area <= 0:
        return {
            "nodes": [],
            "edges": [],
            "snap_confidence": 1.0,
            "edge_pixel_coverage": {},
            "warnings": ["no_stroke_pixels"],
        }

    snap_mask = parsed.stroke_mask
    arrow_exclusion_px = 0
    if arrow_exclusion_mask is not None:
        candidate = cv2.bitwise_and(parsed.stroke_mask, cv2.bitwise_not(arrow_exclusion_mask))
        candidate_area = int((candidate > 0).sum())
        # Keep enough shaft pixels for a meaningful path fit.  If the arrow
        # detector fires on a tiny/ambiguous stroke and would delete most of
        # the mask, fall back to the raw stroke rather than manufacturing an
        # empty normalized path.
        if candidate_area >= max(20, int(0.35 * parsed.stroke_area)):
            snap_mask = candidate
            arrow_exclusion_px = int(((parsed.stroke_mask > 0) & (arrow_exclusion_mask > 0)).sum())

    ys, xs = np.where(snap_mask > 0)
    pts_xy = np.stack([xs, ys], axis=1).astype(float)
    center_xy = _stroke_centerline_points_xy(snap_mask)
    if len(center_xy) >= 2:
        snap_pts_xy = center_xy
        snap_point_source = "centerline"
    else:
        snap_pts_xy = pts_xy
        snap_point_source = "mask"
    mask_uv = _project_points(pts_xy, model.inv_homography)
    snap_uv = _project_points(snap_pts_xy, model.inv_homography)

    line_fit = _stroke_line_fit_stats(snap_uv)
    total = len(snap_pts_xy)
    start_node = _nearest_board_node(model, parsed.start_pos)
    end_node = _nearest_board_node(model, parsed.end_pos)
    if bool(line_fit.get("is_straight_diagonal")) or bool(line_fit.get("is_slight_center_diagonal")):
        return {
            "nodes": [],
            "edges": [],
            "path_model": "straight_diagonal",
            "snap_confidence": 0.0,
            "raw_snap_confidence": 0.0,
            "mean_snap_distance_cells": None,
            "raw_mean_snap_distance_cells": None,
            "start_node": list(start_node) if start_node is not None else None,
            "end_node": list(end_node) if end_node is not None else None,
            "edge_pixel_coverage": {},
            "snap_point_source": snap_point_source,
            "snap_point_count": int(total),
            "stroke_mask_pixel_count": int(len(pts_xy)),
            "stroke_line_fit": line_fit,
            "warnings": list(model.warnings) + ["straight_diagonal_not_snapped_to_edges"],
        }

    snap_thresh = 0.23
    segment_margin = 0.18

    edge_counts: Counter = Counter()
    edge_dist_sums: Dict[Edge, float] = defaultdict(float)
    edge_bins: Dict[Edge, set] = defaultdict(set)
    assigned_dist_sum = 0.0
    assigned = 0
    longitudinal_bins = 24
    for col_f, row_f in snap_uv:
        best = _nearest_legal_grid_edge(float(col_f), float(row_f), model, segment_margin)

        if best is not None and best[0] <= snap_thresh:
            edge_counts[best[1]] += 1
            edge_dist_sums[best[1]] += best[0]
            bin_idx = int(max(0, min(longitudinal_bins - 1, math.floor(best[2] * longitudinal_bins))))
            edge_bins[best[1]].add(bin_idx)
            assigned += 1
            assigned_dist_sum += best[0]

    raw_snap_conf = assigned / float(total) if total else 0.0
    near_edge_frac = raw_snap_conf
    min_count = max(24, int(0.02 * max(assigned, 1)))
    # Intermediate One Stroke frames often contain only a short prefix of a
    # legal board edge.  Requiring half-edge coverage drops those prefixes and
    # leaves the ordered path empty even when the stroke is close to grid lines.
    min_bins = max(8, int(0.34 * longitudinal_bins))
    edges: List[Edge] = sorted(
        edge
        for edge, count in edge_counts.items()
        if count >= min_count and _max_consecutive_run(edge_bins.get(edge, set())) >= min_bins
    )
    retained_assigned = sum(edge_counts[edge] for edge in edges)
    retained_dist_sum = sum(edge_dist_sums[edge] for edge in edges)
    snap_conf = retained_assigned / float(total) if total else 0.0

    nodes = _ordered_nodes_from_edges(edges, start_node, end_node)
    fit_segments, fit_stats = _fit_snapped_edge_segments(
        snap_mask,
        pts_xy,
        mask_uv,
        model,
        edges,
        snap_thresh,
        segment_margin,
    )
    warnings = list(model.warnings)
    path_model = "legal_edges" if edges else "unresolved"
    if raw_snap_conf >= 0.35 and snap_conf < 0.35:
        warnings.append("sparse_legal_edge_coverage")
    if snap_conf < 0.45:
        warnings.append("low_snap_confidence")
    if not edges and parsed.stroke_area > 0:
        warnings.append("no_legal_grid_edges_snapped")
    if arrow_exclusion_px:
        warnings.append("arrowhead_excluded_from_snap")

    return {
        "nodes": [[int(r), int(c)] for r, c in nodes],
        "edges": [_edge_to_json(e) for e in edges],
        "path_model": path_model,
        "snap_confidence": float(snap_conf),
        "raw_snap_confidence": float(raw_snap_conf),
        "mean_snap_distance_cells": float(retained_dist_sum / retained_assigned) if retained_assigned else None,
        "raw_mean_snap_distance_cells": float(assigned_dist_sum / assigned) if assigned else None,
        "start_node": list(start_node) if start_node is not None else None,
        "end_node": list(end_node) if end_node is not None else None,
        "edge_pixel_coverage": {
            _edge_label((a, b)): int(count)
            for (a, b), count in sorted(edge_counts.items())
            if (a, b) in edges
        },
        "candidate_edge_pixel_coverage": {
            _edge_label((a, b)): int(count)
            for (a, b), count in sorted(edge_counts.items())
        },
        "edge_longitudinal_coverage": {
            _edge_label((a, b)): round(len(edge_bins.get((a, b), set())) / longitudinal_bins, 3)
            for (a, b), count in sorted(edge_counts.items())
        },
        "edge_longitudinal_run_coverage": {
            _edge_label((a, b)): round(_max_consecutive_run(edge_bins.get((a, b), set())) / longitudinal_bins, 3)
            for (a, b), count in sorted(edge_counts.items())
        },
        "near_legal_edge_fraction": float(near_edge_frac),
        "snap_point_source": snap_point_source,
        "snap_point_count": int(total),
        "stroke_mask_pixel_count": int(len(pts_xy)),
        "edge_fit_segments_xy": fit_segments,
        "edge_fit_stats": fit_stats,
        "stroke_line_fit": line_fit,
        "arrow_exclusion_pixel_count": int(arrow_exclusion_px),
        "warnings": warnings,
    }


def _one_stroke_detection_record(
    frame_id: int,
    image_path: str,
    img_bgr: np.ndarray,
    size: str,
    parsed: Optional[os_overlay.ParseResult] = None,
    board_model: Optional[OneStrokeBoardModel] = None,
) -> Tuple[Dict[str, object], os_overlay.ParseResult]:
    parsed = parsed or os_overlay.parse_frame(img_bgr)
    rows, cols = _one_stroke_board_size(size)
    arrow_candidates, arrow_exclusion = _detect_one_stroke_arrows(parsed, board_model)
    normalized_path = _normalize_one_stroke_path(parsed, board_model, arrow_exclusion)
    record = {
        "frame_id": frame_id,
        "task_type": "one_stroke",
        "image_path": image_path,
        "board": {
            "rows": rows,
            "cols": cols,
            "bbox_xyxy": [0, 0, int(img_bgr.shape[1]), int(img_bgr.shape[0])],
            "model_confidence": board_model.quality if board_model is not None else 0.0,
            "grid_nodes": [
                {"node": [int(r), int(c)], "xy": [float(x), float(y)]}
                for (r, c), (x, y) in sorted((board_model.grid_nodes if board_model else {}).items())
            ],
            "cell_bboxes": [
                {
                    "color": b.color,
                    "center_xy": [float(b.center[0]), float(b.center[1])],
                    "bbox_xyxy": [int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[0] + b.bbox[2]), int(b.bbox[1] + b.bbox[3])],
                    "area": int(b.area),
                }
                for b in parsed.cells
            ],
        },
        "start": {
            "cell": None,
            "xy": list(parsed.start_pos) if parsed.start_pos else None,
            "confidence": 0.9 if parsed.start_pos else 0.0,
        },
        "end": {
            "cell": None,
            "xy": list(parsed.end_pos) if parsed.end_pos else None,
            "confidence": 0.9 if parsed.end_pos else 0.0,
        },
        "raw_path": {
            "polyline_xy": [
                list(parsed.stroke_start_tip) if parsed.stroke_start_tip else None,
                list(parsed.stroke_end_tip) if parsed.stroke_end_tip else None,
            ],
            "stroke_area": int(parsed.stroke_area),
            "stroke_connected": bool(parsed.stroke_connected),
            "stroke_near_start": parsed.stroke_near_start,
            "stroke_near_end": parsed.stroke_near_end,
            "stroke_on_cell_px": int(parsed.stroke_on_cell_px),
            "mask_path": None,
            "arrow_exclusion_pixel_count": int(normalized_path.get("arrow_exclusion_pixel_count") or 0),
            "confidence": 0.8 if parsed.stroke_area > 0 and parsed.stroke_connected else 0.45,
        },
        "normalized_path": normalized_path,
        "arrow_candidates": arrow_candidates,
        "parser_status": parsed.status,
        "warnings": [] if parsed.status == "ok" else [f"parser_status={parsed.status}"],
    }
    return record, parsed


def _one_stroke_ref_count(ref_record: Dict[str, object]) -> Counter:
    cells = ref_record.get("board", {}).get("cell_bboxes", [])
    return Counter(str(c.get("color")) for c in cells)


def _one_stroke_cur_count(record: Dict[str, object]) -> Counter:
    cells = record.get("board", {}).get("cell_bboxes", [])
    return Counter(str(c.get("color")) for c in cells)


def _one_stroke_empty_stroke_metric(is_early_frame: bool, applicable_reason: str) -> Metric:
    """Shared branch for `stroke_area == 0`.

    Only vacuously pass early in the episode, when there's genuinely no path
    drawn yet.  A mid/late-episode frame with no detected stroke is more
    likely a detection failure (segmentation dropped the whole mask) than a
    legitimate empty state, so it should surface as low-confidence instead of
    silently inflating the accuracy the way a blanket `not_applicable` pass
    would.
    """
    if is_early_frame:
        return _metric(1.0, True, applicable_reason, status="not_applicable")
    return _metric(
        0.5,
        False,
        "stroke area is unexpectedly empty mid/late episode; likely a detection failure rather than a legitimate not-yet-drawn state",
        status="low_confidence",
    )


def _one_stroke_path_validity(record: Dict[str, object], ref_count: Counter, *, is_early_frame: bool) -> Metric:
    raw = record.get("raw_path") or {}
    normalized = record.get("normalized_path") or {}
    if not ref_count:
        return _metric(
            0.0,
            False,
            "reference grid is unavailable; cannot judge path validity",
            status="low_confidence",
        )
    cur = _one_stroke_cur_count(record)
    n_total = sum(ref_count.values())
    n_split = sum(max(0, cur.get(color, 0) - n) for color, n in ref_count.items())
    stroke_area = int(raw.get("stroke_area") or 0)
    if stroke_area == 0:
        return _one_stroke_empty_stroke_metric(
            is_early_frame, "no drawn path yet; path validity is vacuously valid for this frame"
        )
    path_model = str(normalized.get("path_model") or "")
    if path_model == "straight_diagonal":
        return _metric(
            0.0,
            False,
            "drawn path is a straight diagonal that cuts through cells; One Stroke only allows U/D/L/R grid-edge moves",
            evidence={
                "path_model": path_model,
                "stroke_line_fit": normalized.get("stroke_line_fit") or {},
                "normalized_warnings": normalized.get("warnings") or [],
            },
        )
    fragment_score = max(0.0, (n_total - n_split) / n_total) if n_total else 0.0
    snap_conf = float(normalized.get("snap_confidence") or 0.0)
    n_edges = len(normalized.get("edges") or [])
    score = 0.65 * snap_conf + 0.35 * fragment_score
    connected = bool(raw.get("stroke_connected"))
    if not connected:
        score *= 0.7
    ok = snap_conf >= 0.55 and fragment_score >= 0.65 and connected and n_edges > 0
    return _metric(
        score,
        ok,
        "path snaps to legal board edges with acceptable cell-fragment evidence"
        if ok
        else "path does not reliably snap to legal board edges or appears to cut through cells",
        evidence={
            "split_cell_count": int(n_split),
            "stroke_connected": connected,
            "snap_confidence": round(snap_conf, 3),
            "normalized_edge_count": n_edges,
            "normalized_warnings": normalized.get("warnings") or [],
        },
    )


def _one_stroke_start_end_validity(
    record: Dict[str, object], *, is_final_frame: bool, is_early_frame: bool
) -> Metric:
    raw = record.get("raw_path") or {}
    stroke_area = int(raw.get("stroke_area") or 0)
    near_start = raw.get("stroke_near_start")
    near_end = raw.get("stroke_near_end")
    if stroke_area == 0:
        return _one_stroke_empty_stroke_metric(
            is_early_frame, "no drawn path yet; start/end reachability is not applicable"
        )
    if is_final_frame:
        ok = near_start is True and near_end is True and bool(raw.get("stroke_connected"))
        score = sum([near_start is True, near_end is True, bool(raw.get("stroke_connected"))]) / 3.0
        return _metric(
            score,
            ok,
            "final path touches both start and end markers"
            if ok
            else "final path does not connect both start and end markers",
            evidence={
                "stroke_near_start": near_start,
                "stroke_near_end": near_end,
                "stroke_connected": bool(raw.get("stroke_connected")),
            },
        )
    ok = near_start is True or near_start is None
    return _metric(
        1.0 if ok else 0.5,
        ok,
        "intermediate path starts near the start marker or marker evidence is unavailable"
        if ok
        else "intermediate path is not anchored near the start marker",
        evidence={"stroke_near_start": near_start},
    )


def _one_stroke_direction_validity(record: Dict[str, object], *, is_early_frame: bool) -> Metric:
    raw = record.get("raw_path") or {}
    polyline = [p for p in raw.get("polyline_xy") or [] if p is not None]
    stroke_area = int(raw.get("stroke_area") or 0)
    if stroke_area == 0:
        return _one_stroke_empty_stroke_metric(is_early_frame, "no drawn path yet; direction is not applicable")
    arrows = list(record.get("arrow_candidates") or [])
    best_arrow = arrows[0] if arrows else None
    if best_arrow is not None and float(best_arrow.get("confidence") or 0.0) >= 0.55:
        tip_xy = best_arrow.get("tip_xy")
        if not (isinstance(tip_xy, list) and len(tip_xy) >= 2):
            return _metric(
                0.4,
                False,
                "arrow detector found a candidate but its tip geometry is malformed",
                evidence={"arrow_candidates": len(arrows)},
                status="low_confidence",
            )
        tip = np.asarray([float(tip_xy[0]), float(tip_xy[1])], dtype=float)
        endpoint_pts = [np.asarray([float(p[0]), float(p[1])], dtype=float) for p in polyline if len(p) >= 2]
        opposite = None
        if endpoint_pts:
            opposite = max(endpoint_pts, key=lambda p: float(np.linalg.norm(p - tip)))

        start_xy = (record.get("start") or {}).get("xy")
        end_xy = (record.get("end") or {}).get("xy")
        if start_xy is None or end_xy is None or opposite is None:
            return _metric(
                0.55,
                True,
                "arrow direction is detected, but start/end or opposite-tip evidence is incomplete",
                evidence={
                    "arrow_candidates": len(arrows),
                    "best_arrow": best_arrow,
                    "has_start": start_xy is not None,
                    "has_end": end_xy is not None,
                    "has_opposite_tip": opposite is not None,
                },
                status="low_confidence",
            )

        start = np.asarray([float(start_xy[0]), float(start_xy[1])], dtype=float)
        end = np.asarray([float(end_xy[0]), float(end_xy[1])], dtype=float)
        board_scale = _one_stroke_board_scale(record)
        margin = max(16.0, 0.06 * board_scale)
        tip_start_dist = float(np.linalg.norm(tip - start))
        opp_start_dist = float(np.linalg.norm(opposite - start))
        tip_end_dist = float(np.linalg.norm(tip - end))
        opp_end_dist = float(np.linalg.norm(opposite - end))
        front_from_start = tip_start_dist + margin >= opp_start_dist
        front_toward_end = tip_end_dist <= opp_end_dist + margin

        arrow_dir = np.asarray(best_arrow.get("direction_xy") or [0.0, 0.0], dtype=float)
        expected = end - start
        denom = float(np.linalg.norm(arrow_dir) * np.linalg.norm(expected))
        global_alignment = float((arrow_dir @ expected) / denom) if denom > 1e-6 else None
        score = 0.55 * float(front_from_start) + 0.30 * float(front_toward_end)
        if global_alignment is not None:
            score += 0.15 * _clamp01((global_alignment + 1.0) / 2.0)
        else:
            score += 0.075
        ok = front_from_start and front_toward_end
        return _metric(
            score,
            ok,
            "detected arrowhead is on the path front, away from the start and toward the end"
            if ok
            else "detected arrowhead appears reversed or attached to the wrong path endpoint",
            evidence={
                "arrow_candidates": len(arrows),
                "best_arrow": best_arrow,
                "tip_distance_to_start_px": round(tip_start_dist, 1),
                "opposite_tip_distance_to_start_px": round(opp_start_dist, 1),
                "tip_distance_to_end_px": round(tip_end_dist, 1),
                "opposite_tip_distance_to_end_px": round(opp_end_dist, 1),
                "global_start_to_end_alignment": None if global_alignment is None else round(global_alignment, 3),
            },
        )
    board = record.get("board") or {}
    bbox = board.get("bbox_xyxy") or [0, 0, 640, 480]
    frame_area = max(1.0, float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1]))
    min_direction_area = max(900, int(0.003 * frame_area))
    if stroke_area < min_direction_area and not is_early_frame:
        return _metric(
            0.4,
            False,
            "stroke evidence is too small for reliable direction detection; likely marker highlight or residual pixels",
            evidence={
                "stroke_area": stroke_area,
                "min_direction_area": min_direction_area,
                "arrow_candidates": len(arrows),
            },
            status="low_confidence",
        )
    if len(polyline) >= 2:
        return _metric(
            0.7,
            True,
            "stroke tips can be oriented from start side to end side, but arrowhead evidence is unavailable",
            evidence={"arrow_candidates": len(arrows)},
            status="low_confidence",
        )
    return _metric(
        0.4,
        False,
        "stroke direction cannot be inferred from geometric tips",
        evidence={"arrow_candidates": len(arrows)},
        status="low_confidence",
    )


def _one_stroke_cell_color_grid(record: Dict[str, object]) -> Optional[List[List[Optional[str]]]]:
    """Snap this frame's detected colored-cell blobs onto (row, col) board
    slots using the already-fitted grid node positions, so the oracle check
    below can be handed a real color grid instead of raw pixel blobs.
    """
    board = record.get("board") or {}
    rows = int(board.get("rows") or 0)
    cols = int(board.get("cols") or 0)
    expected = record.get("expected_cell_grid")
    if isinstance(expected, list) and len(expected) == rows and all(
        isinstance(row, list) and len(row) == cols for row in expected
    ):
        return expected
    node_xy = _node_xy_from_record(record)
    if rows <= 0 or cols <= 0 or not node_xy:
        return None

    cell_centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for row in range(rows):
        for col in range(cols):
            corners = [
                node_xy.get((row, col)),
                node_xy.get((row, col + 1)),
                node_xy.get((row + 1, col)),
                node_xy.get((row + 1, col + 1)),
            ]
            if any(c is None for c in corners):
                continue
            cell_centers[(row, col)] = (
                sum(c[0] for c in corners) / 4.0,
                sum(c[1] for c in corners) / 4.0,
            )
    if not cell_centers:
        return None

    slot_best: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for blob in board.get("cell_bboxes") or []:
        color = blob.get("color")
        center = blob.get("center_xy")
        area = int(blob.get("area") or 0)
        if not color or center is None:
            continue
        best_slot = min(
            cell_centers,
            key=lambda slot: math.hypot(center[0] - cell_centers[slot][0], center[1] - cell_centers[slot][1]),
        )
        prev = slot_best.get(best_slot)
        if prev is None or area > prev[0]:
            slot_best[best_slot] = (area, color)

    grid: List[List[Optional[str]]] = [[None for _ in range(cols)] for _ in range(rows)]
    for (row, col), (_, color) in slot_best.items():
        grid[row][col] = color
    return grid


def _one_stroke_solver_edges(record: Dict[str, object]) -> set:
    normalized = record.get("normalized_path") or {}
    edges_rc = normalized.get("edges") or []
    return {
        one_stroke_solver.edge_key((int(c1), int(r1)), (int(c2), int(r2))) for (r1, c1), (r2, c2) in edges_rc
    }


def _one_stroke_different_color_separation_valid(record: Dict[str, object], *, is_final_frame: bool) -> Metric:
    """Final-frame check for the user-visible color-separation property.

    `final_solution_valid` already checks the full solver win condition.  This
    metric exposes one half of that condition directly: no connected region may
    contain two different colors.  It intentionally does not require each
    same-colored group to remain connected; that is left to the full oracle.
    """
    if not is_final_frame:
        return _metric(
            1.0,
            True,
            "different-color separation check only applies to the final frame",
            status="not_applicable",
        )

    board = record.get("board") or {}
    rows = int(board.get("rows") or 0)
    cols = int(board.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        return _metric(0.0, False, "board dimensions unavailable; cannot evaluate color separation", status="low_confidence")

    cell_grid = _one_stroke_cell_color_grid(record)
    if cell_grid is None:
        return _metric(
            0.0,
            False,
            "could not snap detected cells onto board slots; cannot evaluate color separation",
            status="low_confidence",
        )

    solver_edges = _one_stroke_solver_edges(record)
    stroke_area = int((record.get("raw_path") or {}).get("stroke_area") or 0)
    if not solver_edges:
        return _metric(
            0.0,
            False,
            "no normalized path edges could be resolved; different colors are not separated by a valid cut",
            evidence={
                "edge_count": 0,
                "stroke_area": stroke_area,
                "path_model": (record.get("normalized_path") or {}).get("path_model"),
            },
        )

    regions = one_stroke_solver.calculate_regions(cols + 1, rows + 1, solver_edges)
    region_colors: Dict[int, set] = {}
    for row in range(rows):
        for col in range(cols):
            color = cell_grid[row][col]
            if color is None:
                continue
            rid = int(regions[row][col])
            region_colors.setdefault(rid, set()).add(str(color))
    mixed_regions = {
        int(rid): sorted(colors)
        for rid, colors in region_colors.items()
        if len(colors) > 1
    }
    ok = not mixed_regions
    return _metric(
        1.0 if ok else 0.0,
        ok,
        "every connected region contains at most one color"
        if ok
        else "at least one connected region still contains multiple colors",
        evidence={
            "mixed_regions": mixed_regions,
            "edge_count": len(solver_edges),
        },
    )


def _one_stroke_final_solution_valid(record: Dict[str, object], *, is_final_frame: bool) -> Metric:
    """Ground-truth oracle check for the terminal frame.

    Reuses the real level solver
    (`environments/separation_one_stroke/gym/solver.py::evaluate_constraints`,
    the same function the level generator uses to confirm a puzzle is
    solvable) instead of judging the ending with pixel heuristics like
    `direction_validity`.  A solution must connect start to end and satisfy
    the solver's same/different-color region constraints.
    """
    if not is_final_frame:
        return _metric(1.0, True, "oracle solution check only applies to the final frame", status="not_applicable")

    board = record.get("board") or {}
    rows = int(board.get("rows") or 0)
    cols = int(board.get("cols") or 0)
    if rows <= 0 or cols <= 0:
        return _metric(
            0.0, False, "board dimensions unavailable; cannot evaluate the oracle", status="low_confidence"
        )

    raw = record.get("raw_path") or {}
    if not (
        raw.get("stroke_near_start") is True
        and raw.get("stroke_near_end") is True
        and bool(raw.get("stroke_connected"))
    ):
        return _metric(
            0.0,
            False,
            "final path does not form one connected stroke from start to end",
            evidence={
                "stroke_near_start": raw.get("stroke_near_start"),
                "stroke_near_end": raw.get("stroke_near_end"),
                "stroke_connected": bool(raw.get("stroke_connected")),
            },
        )

    cell_grid = _one_stroke_cell_color_grid(record)
    if cell_grid is None:
        return _metric(
            0.0,
            False,
            "could not snap detected cells onto board slots; cannot evaluate the oracle",
            status="low_confidence",
        )

    normalized = record.get("normalized_path") or {}
    edges_rc = normalized.get("edges") or []
    stroke_area = int((record.get("raw_path") or {}).get("stroke_area") or 0)
    if not edges_rc:
        path_model = str(normalized.get("path_model") or "")
        if stroke_area <= 0:
            reason = "no path was drawn by the final frame; the puzzle was never completed"
        elif path_model == "straight_diagonal":
            # The game's action space is strictly U/D/L/R (grid-edge moves
            # only, see environments/separation_one_stroke/README.md) -- a
            # stroke the detector fits as an un-snapped diagonal is not a
            # legal cut regardless of how visually close it looks to the
            # start/end line, so this is a real oracle failure rather than a
            # detection gap.
            reason = "drawn path is a straight diagonal, not a grid-edge cut; the game only allows U/D/L/R moves so this cannot be a legal solution"
        else:
            reason = "no normalized path edges could be resolved onto legal grid edges; final cut is not a valid solution"
        return _metric(0.0, False, reason, evidence={"edge_count": 0, "path_model": path_model})

    solver_edges = _one_stroke_solver_edges(record)
    result = one_stroke_solver.evaluate_constraints(cols + 1, rows + 1, cell_grid, solver_edges)
    ok = bool(result.get("success"))
    return _metric(
        1.0 if ok else 0.0,
        ok,
        "final cut separates every same-colored cell into its own region (oracle-verified)"
        if ok
        else f"final cut does not satisfy the solver's win condition: {result.get('reason')}",
        evidence={"oracle_reason": result.get("reason"), "edge_count": len(solver_edges)},
    )


def _one_stroke_static_metrics(
    record: Dict[str, object],
    ref_count: Counter,
    parsed: os_overlay.ParseResult,
    *,
    is_final_frame: bool,
    is_early_frame: bool = False,
) -> Dict[str, Metric]:
    return {
        "path_validity": _one_stroke_path_validity(record, ref_count, is_early_frame=is_early_frame),
        "start_end_validity": _one_stroke_start_end_validity(
            record, is_final_frame=is_final_frame, is_early_frame=is_early_frame
        ),
        "direction_validity": _one_stroke_direction_validity(record, is_early_frame=is_early_frame),
        "path_continuity": _one_stroke_path_continuity(record, parsed, is_early_frame=is_early_frame),
        "different_color_separation_valid": _one_stroke_different_color_separation_valid(
            record, is_final_frame=is_final_frame
        ),
        "final_solution_valid": _one_stroke_final_solution_valid(record, is_final_frame=is_final_frame),
    }


def _mask_overlap(prev_mask: np.ndarray, cur_mask: np.ndarray) -> float:
    prev_area = int((prev_mask > 0).sum())
    if prev_area <= 0:
        return 1.0
    inter = int(((prev_mask > 0) & (cur_mask > 0)).sum())
    return inter / float(prev_area)


def _normalized_edge_set(record: Dict[str, object]) -> set:
    normalized = record.get("normalized_path") or {}
    return {_edge_key(edge) for edge in normalized.get("edges") or []}


def _edge_nodes(edges: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]) -> set:
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    return nodes


def _edge_set_json(edges: Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[List[List[int]]]:
    return [_edge_to_json((a, b)) for a, b in sorted(edges)]


def _one_stroke_node_xy_float(record: Dict[str, object]) -> Dict[Node, Tuple[float, float]]:
    out: Dict[Node, Tuple[float, float]] = {}
    for item in (record.get("board") or {}).get("grid_nodes") or []:
        node = item.get("node")
        xy = item.get("xy")
        if node is None or xy is None:
            continue
        out[(int(node[0]), int(node[1]))] = (float(xy[0]), float(xy[1]))
    return out


def _one_stroke_cell_step_px(record: Dict[str, object]) -> float:
    nodes = _one_stroke_node_xy_float(record)
    lengths = []
    for (row, col), (x, y) in nodes.items():
        for nxt in ((row, col + 1), (row + 1, col)):
            if nxt in nodes:
                nx, ny = nodes[nxt]
                lengths.append(math.hypot(nx - x, ny - y))
    if lengths:
        return float(np.median(lengths))
    return max(30.0, _one_stroke_board_scale(record) / 3.0)


def _edge_graph_component_count(edges: Iterable[Edge]) -> int:
    graph: Dict[Node, List[Node]] = defaultdict(list)
    nodes = set()
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
        nodes.add(a)
        nodes.add(b)
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            cur = stack.pop()
            for nxt in graph[cur]:
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _significant_component_stats(mask: np.ndarray) -> Dict[str, object]:
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area <= 0:
        return {
            "area": 0,
            "largest_ratio": 1.0,
            "num_significant": 0,
            "significant_areas": [],
            "threshold_px": 0,
        }

    # Close only sub-pixel/anti-aliased pinholes; real mid-stroke gaps survive.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    largest = areas[0] if areas else 0
    threshold = max(64, int(0.025 * max(area, 1)))
    significant = [a for a in areas if a >= threshold]
    return {
        "area": area,
        "largest_ratio": min(1.0, float(largest / max(area, 1))),
        "num_significant": int(len(significant)),
        "significant_areas": [int(a) for a in significant[:8]],
        "threshold_px": int(threshold),
    }


def _skeleton_component_stats(mask: np.ndarray) -> Dict[str, object]:
    skel = skeletonize(mask > 0).astype(np.uint8)
    total = int(skel.sum())
    if total <= 0:
        return {
            "skeleton_px": 0,
            "num_significant": 0,
            "significant_lengths": [],
            "threshold_px": 0,
        }
    n, _, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    lengths = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    threshold = max(8, int(0.035 * max(total, 1)))
    significant = [length for length in lengths if length >= threshold]
    return {
        "skeleton_px": total,
        "num_significant": int(len(significant)),
        "significant_lengths": [int(length) for length in significant[:8]],
        "threshold_px": int(threshold),
    }


def _fit_segment_endpoint_gap_px(record: Dict[str, object]) -> Optional[float]:
    normalized = record.get("normalized_path") or {}
    fit_segments = normalized.get("edge_fit_segments_xy") or {}
    edges = [_edge_key(edge) for edge in normalized.get("edges") or []]
    if len(edges) < 2 or not fit_segments:
        return None

    endpoints: Dict[Edge, Tuple[np.ndarray, np.ndarray]] = {}
    for edge in edges:
        segment = fit_segments.get(_edge_label(edge))
        if (
            isinstance(segment, list)
            and len(segment) == 2
            and isinstance(segment[0], list)
            and isinstance(segment[1], list)
            and len(segment[0]) >= 2
            and len(segment[1]) >= 2
        ):
            endpoints[edge] = (
                np.asarray([float(segment[0][0]), float(segment[0][1])], dtype=float),
                np.asarray([float(segment[1][0]), float(segment[1][1])], dtype=float),
            )

    max_gap: Optional[float] = None
    for i, edge_a in enumerate(edges):
        if edge_a not in endpoints:
            continue
        for edge_b in edges[i + 1 :]:
            if edge_b not in endpoints or not (set(edge_a) & set(edge_b)):
                continue
            pa = endpoints[edge_a]
            pb = endpoints[edge_b]
            gap = min(float(np.linalg.norm(a - b)) for a in pa for b in pb)
            max_gap = gap if max_gap is None else max(max_gap, gap)
    return max_gap


def _one_stroke_path_continuity(
    record: Dict[str, object],
    parsed: os_overlay.ParseResult,
    *,
    is_early_frame: bool = False,
) -> Metric:
    raw = record.get("raw_path") or {}
    stroke_area = int(raw.get("stroke_area") or 0)
    if stroke_area <= 0:
        return _one_stroke_empty_stroke_metric(is_early_frame, "no drawn path yet; path continuity is not applicable")

    normalized = record.get("normalized_path") or {}
    edges = {_edge_key(edge) for edge in normalized.get("edges") or []}
    component_stats = _significant_component_stats(parsed.stroke_mask)
    skeleton_stats = _skeleton_component_stats(parsed.stroke_mask)
    largest_ratio = float(component_stats["largest_ratio"])
    significant_components = int(component_stats["num_significant"])
    significant_skeleton_components = int(skeleton_stats["num_significant"])

    edge_graph_components = _edge_graph_component_count(edges) if edges else 0
    max_fit_gap = _fit_segment_endpoint_gap_px(record)
    cell_step = _one_stroke_cell_step_px(record)
    max_fit_gap_threshold = max(24.0, 0.18 * cell_step)

    mask_area_conf = min(1.0, stroke_area / max(500.0, 0.004 * parsed.stroke_mask.size))
    board_conf = float((record.get("board") or {}).get("model_confidence") or 0.0)
    snap_conf = float(normalized.get("snap_confidence") or normalized.get("raw_snap_confidence") or 0.0)
    path_model = str(normalized.get("path_model") or "")
    if path_model == "legal_edges" and edges:
        geometry_conf = max(0.35, min(1.0, snap_conf))
    elif path_model == "straight_diagonal":
        geometry_conf = 0.75
    else:
        geometry_conf = 0.45
    parser_conf = 1.0 if getattr(parsed, "status", "") == "ok" else 0.55
    confidence = (
        0.45 * mask_area_conf
        + 0.25 * board_conf
        + 0.20 * geometry_conf
        + 0.10 * parser_conf
    )

    largest_ratio_threshold = 0.92
    confidence_threshold = 0.60
    pixel_ok = significant_components <= 1 and largest_ratio >= largest_ratio_threshold
    skeleton_ok = significant_skeleton_components <= 1
    graph_ok = edge_graph_components <= 1
    fit_gap_ok = max_fit_gap is None or max_fit_gap <= max_fit_gap_threshold

    largest_score = min(1.0, largest_ratio / largest_ratio_threshold)
    component_score = 1.0 if significant_components <= 1 else 1.0 / significant_components
    skeleton_score = 1.0 if significant_skeleton_components <= 1 else 1.0 / significant_skeleton_components
    graph_score = 1.0 if edge_graph_components <= 1 else 1.0 / edge_graph_components
    if max_fit_gap is None:
        fit_gap_score = 1.0
    elif max_fit_gap <= max_fit_gap_threshold:
        fit_gap_score = 1.0
    else:
        fit_gap_score = max(0.0, min(1.0, max_fit_gap_threshold / max_fit_gap))

    score = (
        0.32 * largest_score
        + 0.24 * component_score
        + 0.18 * skeleton_score
        + 0.18 * graph_score
        + 0.08 * fit_gap_score
    )
    score *= 0.75 + 0.25 * confidence

    ok = (
        pixel_ok
        and skeleton_ok
        and graph_ok
        and fit_gap_ok
        and confidence >= confidence_threshold
    )
    if ok:
        reason = "stroke mask and normalized path form one continuous path with sufficient confidence"
        status = None
    elif confidence < confidence_threshold:
        reason = "path continuity evidence is too low-confidence"
        status = "low_confidence"
    elif not pixel_ok or not skeleton_ok:
        reason = "stroke mask breaks into multiple significant components"
        status = None
    elif not graph_ok:
        reason = "normalized path edges form disconnected components"
        status = None
    else:
        reason = "adjacent fitted path segments have an excessive endpoint gap"
        status = None

    return _metric(
        score,
        ok,
        reason,
        confidence=confidence,
        status=status,
        evidence={
            "stroke_area": stroke_area,
            "largest_component_ratio": round(largest_ratio, 3),
            "largest_component_ratio_threshold": largest_ratio_threshold,
            "significant_component_count": significant_components,
            "significant_component_areas": component_stats["significant_areas"],
            "significant_component_threshold_px": component_stats["threshold_px"],
            "significant_skeleton_component_count": significant_skeleton_components,
            "significant_skeleton_lengths": skeleton_stats["significant_lengths"],
            "edge_graph_component_count": edge_graph_components,
            "max_fit_endpoint_gap_px": None if max_fit_gap is None else round(max_fit_gap, 1),
            "max_fit_endpoint_gap_threshold_px": round(max_fit_gap_threshold, 1),
            "confidence_threshold": confidence_threshold,
            "confidence_inputs": {
                "mask_area_confidence": round(mask_area_conf, 3),
                "board_model_confidence": round(board_conf, 3),
                "geometry_confidence": round(geometry_conf, 3),
                "parser_confidence": round(parser_conf, 3),
                "path_model": path_model or None,
                "snap_confidence": round(snap_conf, 3),
            },
        },
    )


def _one_stroke_cells(record: Dict[str, object]) -> List[Dict[str, object]]:
    cells = []
    for cell in (record.get("board") or {}).get("cell_bboxes") or []:
        xy = cell.get("center_xy")
        color = cell.get("color")
        if not xy or color is None:
            continue
        cells.append(
            {
                "color": str(color),
                "xy": (float(xy[0]), float(xy[1])),
            }
        )
    return cells


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def _one_stroke_board_scale(record: Dict[str, object]) -> float:
    nodes = []
    for item in (record.get("board") or {}).get("grid_nodes") or []:
        xy = item.get("xy")
        if xy is not None:
            nodes.append((float(xy[0]), float(xy[1])))
    if nodes:
        xs = [xy[0] for xy in nodes]
        ys = [xy[1] for xy in nodes]
        return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    bbox = (record.get("board") or {}).get("bbox_xyxy") or [0, 0, 1, 1]
    return max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]), 1.0)


def _one_stroke_grid_cell_temporal_consistency(
    prev_record: Dict[str, object],
    cur_record: Dict[str, object],
) -> Metric:
    prev_cells = _one_stroke_cells(prev_record)
    cur_cells = _one_stroke_cells(cur_record)
    prev_counts = Counter(cell["color"] for cell in prev_cells)
    cur_counts = Counter(cell["color"] for cell in cur_cells)
    colors = sorted(set(prev_counts) | set(cur_counts))
    count_delta = {color: int(cur_counts.get(color, 0) - prev_counts.get(color, 0)) for color in colors}
    count_errors = sum(abs(delta) for delta in count_delta.values())
    count_score = 1.0 - count_errors / float(max(1, len(prev_cells) + len(cur_cells)))

    if not prev_cells or not cur_cells:
        return _metric(
            0.0,
            False,
            "grid cells are unreadable in one of the adjacent frames",
            evidence={
                "prev_color_counts": _counter_dict(prev_counts),
                "cur_color_counts": _counter_dict(cur_counts),
            },
            status="low_confidence",
        )

    board_scale = max(_one_stroke_board_scale(prev_record), _one_stroke_board_scale(cur_record))
    match_threshold = max(45.0, 0.15 * board_scale)
    unused_cur = set(range(len(cur_cells)))
    matches = []
    unmatched_prev = []
    color_changes = []
    for prev_idx, prev_cell in enumerate(prev_cells):
        best_idx = None
        best_dist = None
        px, py = prev_cell["xy"]
        for cur_idx in unused_cur:
            cx, cy = cur_cells[cur_idx]["xy"]
            dist = math.hypot(px - cx, py - cy)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = cur_idx
        if best_idx is None or best_dist is None or best_dist > match_threshold:
            unmatched_prev.append(
                {
                    "prev_index": prev_idx,
                    "color": prev_cell["color"],
                    "xy": [round(px, 1), round(py, 1)],
                }
            )
            continue
        unused_cur.remove(best_idx)
        cur_cell = cur_cells[best_idx]
        match = {
            "prev_index": prev_idx,
            "cur_index": best_idx,
            "prev_color": prev_cell["color"],
            "cur_color": cur_cell["color"],
            "distance_px": round(best_dist, 1),
        }
        matches.append(match)
        if prev_cell["color"] != cur_cell["color"]:
            color_changes.append(match)

    extra_cur = [
        {
            "cur_index": idx,
            "color": cur_cells[idx]["color"],
            "xy": [round(cur_cells[idx]["xy"][0], 1), round(cur_cells[idx]["xy"][1], 1)],
        }
        for idx in sorted(unused_cur)
    ]
    stable_matches = len(matches) - len(color_changes)
    match_score = stable_matches / float(max(1, len(prev_cells)))
    score = 0.6 * count_score + 0.4 * match_score
    ok = (
        count_errors == 0
        and not color_changes
        and not unmatched_prev
        and not extra_cur
    )
    return _metric(
        score,
        ok,
        "adjacent frames preserve grid cell counts, colors, and matched positions"
        if ok
        else "grid cell count, color, or matched position changes between adjacent frames",
        evidence={
            "prev_color_counts": _counter_dict(prev_counts),
            "cur_color_counts": _counter_dict(cur_counts),
            "count_delta_cur_minus_prev": count_delta,
            "match_threshold_px": round(match_threshold, 1),
            "matched_cell_count": len(matches),
            "unmatched_prev_cells": unmatched_prev[:8],
            "extra_cur_cells": extra_cur[:8],
            "color_changed_matches": color_changes[:8],
        },
    )


def _gate_path_consistency_by_continuity(
    metric: Metric,
    prev_continuity: Metric,
    cur_continuity: Metric,
) -> Metric:
    """Fold each frame's own fragmentation verdict into the transition score.

    The edge-set / pixel-mask comparisons above only check whether prev and
    cur *agree*, which can look perfectly consistent even when both frames'
    stroke detection is itself broken into pieces (e.g. a forked
    arrowhead).  `path_continuity` is what actually notices that breakage,
    but until now it only affected the per-frame static score.
    """
    if metric.get("status") == "not_applicable":
        return metric
    worst_continuity = min(float(prev_continuity["score"]), float(cur_continuity["score"]))
    broken = next(
        (
            c
            for c in (prev_continuity, cur_continuity)
            if c.get("status") not in ("not_applicable", "low_confidence") and not c["pass"]
        ),
        None,
    )
    low_conf = any(c.get("status") == "low_confidence" for c in (prev_continuity, cur_continuity))
    score = min(float(metric["score"]), worst_continuity)
    passed = bool(metric["pass"]) and broken is None
    status = metric.get("status")
    reason = metric["reason"]
    if broken is not None:
        reason = f"{reason}; adjacent-frame stroke detection is fragmented ({broken['reason']})"
    elif low_conf and status is None:
        status = "low_confidence"
    return _metric(score, passed, reason, evidence=metric.get("evidence"), status=status)


def _one_stroke_dynamic_metrics(
    prev_record: Dict[str, object],
    cur_record: Dict[str, object],
    prev_parse: os_overlay.ParseResult,
    cur_parse: os_overlay.ParseResult,
    *,
    prev_is_early: bool = False,
    cur_is_early: bool = False,
) -> Dict[str, Metric]:
    prev_edges = _normalized_edge_set(prev_record)
    cur_edges = _normalized_edge_set(cur_record)
    prev_conf = float((prev_record.get("normalized_path") or {}).get("snap_confidence") or 0.0)
    cur_conf = float((cur_record.get("normalized_path") or {}).get("snap_confidence") or 0.0)
    prev_area = int(prev_record.get("raw_path", {}).get("stroke_area") or 0)
    cur_area = int(cur_record.get("raw_path", {}).get("stroke_area") or 0)
    grid_consistency = _one_stroke_grid_cell_temporal_consistency(prev_record, cur_record)
    prev_model = str((prev_record.get("normalized_path") or {}).get("path_model") or "")
    cur_model = str((cur_record.get("normalized_path") or {}).get("path_model") or "")
    if "straight_diagonal" in {prev_model, cur_model}:
        return {
            "grid_cell_temporal_consistency": grid_consistency,
            "path_temporal_consistency": _metric(
                0.0,
                False,
                "transition contains a straight diagonal path; One Stroke only allows U/D/L/R grid-edge moves",
                evidence={
                    "prev_path_model": prev_model,
                    "cur_path_model": cur_model,
                    "prev_snap_confidence": round(prev_conf, 3),
                    "cur_snap_confidence": round(cur_conf, 3),
                },
            ),
        }
    prev_continuity = _one_stroke_path_continuity(prev_record, prev_parse, is_early_frame=prev_is_early)
    cur_continuity = _one_stroke_path_continuity(cur_record, cur_parse, is_early_frame=cur_is_early)

    if prev_edges and cur_edges and min(prev_conf, cur_conf) >= 0.35:
        shared = prev_edges & cur_edges
        removed = prev_edges - cur_edges
        new = cur_edges - prev_edges
        overlap_ratio = len(shared) / float(max(1, len(prev_edges)))
        allowed_removed = min(1, len(removed))
        growth_score = (len(shared) + allowed_removed) / float(max(1, len(prev_edges)))
        prev_nodes = _edge_nodes(prev_edges)
        new_connected = not new or any(a in prev_nodes or b in prev_nodes for a, b in new)
        growth_ok = (len(removed) <= 1 or overlap_ratio >= 0.85) and new_connected
        growth = _metric(
            growth_score,
            growth_ok,
            "normalized path preserves the previous edge set and extends from it"
            if growth_ok
            else "normalized path removes too many previous edges or adds a disconnected branch",
            evidence={
                "prev_edge_count": len(prev_edges),
                "cur_edge_count": len(cur_edges),
                "shared_edge_count": len(shared),
                "removed_edges": _edge_set_json(removed)[:12],
                "new_edges": _edge_set_json(new)[:12],
                "new_edges_connected_to_previous": new_connected,
                "prev_snap_confidence": round(prev_conf, 3),
                "cur_snap_confidence": round(cur_conf, 3),
            },
        )

        overlap_threshold = 0.45 if len(prev_edges) <= 2 else 0.65
        overlap = _metric(
            overlap_ratio,
            overlap_ratio >= overlap_threshold,
            "adjacent normalized paths share enough board edges"
            if overlap_ratio >= overlap_threshold
            else "adjacent normalized paths have low legal-edge overlap",
            evidence={
                "overlap_ratio": round(overlap_ratio, 3),
                "threshold": overlap_threshold,
                "shared_edges": _edge_set_json(shared)[:12],
            },
        )
        path_consistency = _metric(
            0.5 * (float(growth["score"]) + float(overlap["score"])),
            bool(growth["pass"]) and bool(overlap["pass"]),
            "normalized path retains previous edges and keeps enough adjacent-frame overlap"
            if bool(growth["pass"]) and bool(overlap["pass"])
            else "normalized path either drops previous edges or has low adjacent-frame overlap",
            evidence={
                "prev_edge_count": len(prev_edges),
                "cur_edge_count": len(cur_edges),
                "shared_edge_count": len(shared),
                "removed_edges": _edge_set_json(removed)[:12],
                "new_edges": _edge_set_json(new)[:12],
                "new_edges_connected_to_previous": new_connected,
                "growth_score": round(float(growth["score"]), 3),
                "growth_pass": bool(growth["pass"]),
                "overlap_ratio": round(overlap_ratio, 3),
                "overlap_threshold": overlap_threshold,
                "overlap_pass": bool(overlap["pass"]),
                "prev_snap_confidence": round(prev_conf, 3),
                "cur_snap_confidence": round(cur_conf, 3),
            },
        )
        path_consistency = _gate_path_consistency_by_continuity(path_consistency, prev_continuity, cur_continuity)
        return {
            "grid_cell_temporal_consistency": grid_consistency,
            "path_temporal_consistency": path_consistency,
        }

    if prev_area < 50 and prev_is_early:
        path_consistency = _metric(
            1.0,
            True,
            "previous frame has no substantial path; transition path consistency is not applicable",
            evidence={
                "prev_stroke_area": prev_area,
                "cur_stroke_area": cur_area,
                "fallback": "pixel_mask",
            },
            status="not_applicable",
        )
    elif prev_area < 50:
        path_consistency = _metric(
            0.5,
            False,
            "previous frame has no substantial path mid/late episode; likely a detection failure rather than a legitimate empty state",
            evidence={
                "prev_stroke_area": prev_area,
                "cur_stroke_area": cur_area,
                "fallback": "pixel_mask",
            },
            status="low_confidence",
        )
    else:
        retention = cur_area / float(max(prev_area, 1))
        retention_ok = retention >= 0.75
        growth = _metric(
            min(1.0, retention),
            retention_ok,
            "path area is retained or grows between adjacent frames"
            if retention_ok
            else "current frame drops a large fraction of the previous path area",
            evidence={
                "prev_stroke_area": prev_area,
                "cur_stroke_area": cur_area,
                "area_retention": round(retention, 3),
                "fallback": "pixel_mask",
                "prev_snap_confidence": round(prev_conf, 3),
                "cur_snap_confidence": round(cur_conf, 3),
            },
        )

        overlap_ratio = _mask_overlap(prev_parse.stroke_mask, cur_parse.stroke_mask)
        overlap_ok = overlap_ratio >= 0.45
        overlap = _metric(
            overlap_ratio,
            overlap_ok,
            "current stroke overlaps enough of the previous stroke"
            if overlap_ok
            else "adjacent paths have low pixel overlap",
            evidence={
                "mask_overlap_ratio": round(overlap_ratio, 3),
                "fallback": "pixel_mask",
                "prev_snap_confidence": round(prev_conf, 3),
                "cur_snap_confidence": round(cur_conf, 3),
            },
        )
        path_consistency = _metric(
            0.5 * (float(growth["score"]) + float(overlap["score"])),
            bool(growth["pass"]) and bool(overlap["pass"]),
            "path area is retained and current stroke overlaps enough of previous stroke"
            if bool(growth["pass"]) and bool(overlap["pass"])
            else "path area retention or pixel overlap fails between adjacent frames",
            evidence={
                "prev_stroke_area": prev_area,
                "cur_stroke_area": cur_area,
                "area_retention": round(retention, 3),
                "retention_pass": bool(growth["pass"]),
                "mask_overlap_ratio": round(overlap_ratio, 3),
                "overlap_threshold": 0.45,
                "overlap_pass": bool(overlap["pass"]),
                "fallback": "pixel_mask",
                "prev_snap_confidence": round(prev_conf, 3),
                "cur_snap_confidence": round(cur_conf, 3),
            },
        )

    path_consistency = _gate_path_consistency_by_continuity(path_consistency, prev_continuity, cur_continuity)
    return {
        "grid_cell_temporal_consistency": grid_consistency,
        "path_temporal_consistency": path_consistency,
    }


def _draw_one_stroke_transition_overlay(
    img_a: np.ndarray,
    img_b: np.ndarray,
    prev_record: Dict[str, object],
    cur_record: Dict[str, object],
    prev: os_overlay.ParseResult,
    cur: os_overlay.ParseResult,
    metrics: Dict[str, Metric],
    confidence: Optional[float],
    auto_goal_achieved: bool,
    out_path: Path,
) -> None:
    left = img_a.copy()
    right = img_b.copy()
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    _tint_mask(left, prev.stroke_mask, (230, 210, 0), alpha=0.42)
    _tint_mask(right, cur.stroke_mask, (0, 220, 220), alpha=0.42)
    canvas = np.hstack([left, right])
    _draw_normalized_grid_and_path(canvas[:, : left.shape[1]], prev_record, (255, 180, 40))
    _draw_normalized_grid_and_path(canvas[:, left.shape[1] :], cur_record, (40, 230, 255))
    label = (
        f"{_metrics_overlay_label('dynamic', _metrics_score(metrics), confidence, _metrics_pass(metrics), auto_goal_achieved)} "
        f"grid={metrics['grid_cell_temporal_consistency']['score']:.2f} "
        f"path={metrics['path_temporal_consistency']['score']:.2f} "
    )
    _draw_overlay_label(canvas, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _tint_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    pixels = mask > 0
    if not np.any(pixels):
        return
    color = np.asarray(color_bgr, dtype=np.float32)
    base = img_bgr[pixels].astype(np.float32)
    img_bgr[pixels] = np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _node_xy_from_record(record: Dict[str, object]) -> Dict[Node, Tuple[int, int]]:
    out: Dict[Node, Tuple[int, int]] = {}
    for item in (record.get("board") or {}).get("grid_nodes") or []:
        node = item.get("node")
        xy = item.get("xy")
        if node is None or xy is None:
            continue
        out[(int(node[0]), int(node[1]))] = (int(round(float(xy[0]))), int(round(float(xy[1]))))
    return out


def _draw_normalized_grid_and_path(
    img_bgr: np.ndarray,
    record: Dict[str, object],
    path_color: Tuple[int, int, int],
) -> None:
    nodes = _node_xy_from_record(record)
    if not nodes:
        return
    rows = int((record.get("board") or {}).get("rows") or 0)
    cols = int((record.get("board") or {}).get("cols") or 0)
    for row in range(rows + 1):
        for col in range(cols):
            a = nodes.get((row, col))
            b = nodes.get((row, col + 1))
            if a and b:
                cv2.line(img_bgr, a, b, (95, 95, 95), 1)
    for row in range(rows):
        for col in range(cols + 1):
            a = nodes.get((row, col))
            b = nodes.get((row + 1, col))
            if a and b:
                cv2.line(img_bgr, a, b, (95, 95, 95), 1)
    fit_segments = (record.get("normalized_path") or {}).get("edge_fit_segments_xy") or {}
    for edge in (record.get("normalized_path") or {}).get("edges") or []:
        a, b = _edge_key(edge)
        segment = fit_segments.get(_edge_label((a, b)))
        if (
            isinstance(segment, list)
            and len(segment) == 2
            and isinstance(segment[0], list)
            and isinstance(segment[1], list)
            and len(segment[0]) >= 2
            and len(segment[1]) >= 2
        ):
            pa = (int(round(float(segment[0][0]))), int(round(float(segment[0][1]))))
            pb = (int(round(float(segment[1][0]))), int(round(float(segment[1][1]))))
        else:
            pa = nodes.get(a)
            pb = nodes.get(b)
        if pa and pb:
            cv2.line(img_bgr, pa, pb, (0, 0, 0), 5)
            cv2.line(img_bgr, pa, pb, path_color, 3)


def _draw_one_stroke_detection_overlay(
    img_bgr: np.ndarray,
    parsed: os_overlay.ParseResult,
    record: Dict[str, object],
    title: str,
    metrics: Dict[str, Metric],
    confidence: Optional[float],
    auto_goal_achieved: bool,
) -> np.ndarray:
    out = os_overlay.draw_overlay(img_bgr, parsed, title)
    _draw_normalized_grid_and_path(out, record, (0, 255, 255))
    normalized = record.get("normalized_path") or {}
    snap = normalized.get("snap_confidence")
    edges = len(normalized.get("edges") or [])
    model_label = {
        "straight_diagonal": "diag",
        "legal_edges": "edges",
        "unresolved": "unres",
    }.get(str(normalized.get("path_model") or ""), str(normalized.get("path_model") or "-"))
    label = (
        f"{_metrics_overlay_label('static', _metrics_score(metrics), confidence, _metrics_pass(metrics), auto_goal_achieved)} "
        f"snap={float(snap or 0.0):.2f} edges={edges} model={model_label}"
    )
    _draw_overlay_label(out, label, top=24, color=(0, 255, 255))
    return out


def _is_early_one_stroke_frame(idx: int, total_frames: int) -> bool:
    """First ~5% of sampled frames (at least the first 2) count as "early
    episode" for the purposes of vacuously passing an empty-stroke check --
    everything after that is expected to have a path underway, so an empty
    detection is more likely a segmentation failure than a legitimate state.
    """
    return idx <= max(1, round(0.05 * max(0, total_frames - 1)))


def _process_one_stroke_video(
    video_id: str,
    video_path: Path,
    size: str,
    seed: int,
    output_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
    expected_cell_grid: Optional[List[List[Optional[str]]]] = None,
) -> Dict[str, object]:
    manifest, frames = extract_frames(video_path, video_id, "one_stroke", output_root, sampling_mode)
    detections = []
    parsed_frames: List[os_overlay.ParseResult] = [os_overlay.parse_frame(frame) for frame in frames]
    rows, cols = _one_stroke_board_size(size)
    model_source = next(
        (p for p in parsed_frames if p.stroke_area == 0 and len(p.cells) >= max(4, rows * cols - 1)),
        parsed_frames[0] if parsed_frames else None,
    )
    board_model = _fit_one_stroke_board_model(model_source, rows, cols) if model_source is not None else None
    static_overlay_dir, dynamic_overlay_dir = _overlay_dirs(ONE_STROKE_DETECTOR_NAME)
    for stale in itertools.chain(
        static_overlay_dir.glob(f"{video_id}_frame_*_static.png"),
        dynamic_overlay_dir.glob(f"{video_id}_transition_*_dynamic.png"),
    ):
        stale.unlink()
    for idx, (frame, frame_rec) in enumerate(zip(frames, manifest["frames"])):
        parsed = parsed_frames[idx]
        det, parsed = _one_stroke_detection_record(
            idx,
            str(frame_rec["path"]),
            frame,
            size,
            parsed,
            board_model,
        )
        if expected_cell_grid is not None:
            det["expected_cell_grid"] = expected_cell_grid
        detections.append(det)

    detection_doc = {
        "video_id": video_id,
        "task_type": "one_stroke",
        "metadata": {"size": size, "seed": seed},
        "frames": detections,
    }
    _write_json(output_root / "detections" / f"{video_id}_per_frame_detection.json", detection_doc)

    ref_count = _one_stroke_ref_count(detections[0]) if detections else Counter()
    final_goal_metric = (
        _one_stroke_final_solution_valid(detections[-1], is_final_frame=True)
        if detections
        else {}
    )
    auto_goal_achieved = _auto_goal_achieved(detections, final_goal_metric)
    static_frame_ids = sample_indices(len(detections), 4)
    static_frames = []
    for idx in static_frame_ids:
        det = detections[idx]
        metrics = _one_stroke_static_metrics(
            det,
            ref_count,
            parsed_frames[idx],
            is_final_frame=idx == len(detections) - 1,
            is_early_frame=_is_early_one_stroke_frame(idx, len(detections)),
        )
        static_score = _metrics_score(metrics)
        static_confidence = _static_confidence(metrics, det)
        static_pass = _metrics_pass(metrics)
        static_frames.append(
            {
                "frame_id": idx,
                "metrics": metrics,
                "static_score": static_score,
                "static_confidence": static_confidence,
            }
        )
        static_failed = not static_pass
        if (not overlay_failures_only) or static_failed:
            suffix = " static fail" if static_failed else ""
            overlay = _draw_one_stroke_detection_overlay(
                frames[idx],
                parsed_frames[idx],
                det,
                f"{video_id} frame {idx:03d}{suffix}",
                metrics,
                static_confidence,
                auto_goal_achieved,
            )
            cv2.imwrite(str(static_overlay_dir / f"{video_id}_frame_{idx:03d}_static.png"), overlay)

    static_doc = {
        "video_id": video_id,
        "task_type": "one_stroke",
        "frames": static_frames,
    }
    _write_json(output_root / "static_metrics" / f"{video_id}_static_metrics.json", static_doc)

    dynamic_transitions = []
    for idx in range(max(0, len(detections) - 1)):
        metrics = _one_stroke_dynamic_metrics(
            detections[idx],
            detections[idx + 1],
            parsed_frames[idx],
            parsed_frames[idx + 1],
            prev_is_early=_is_early_one_stroke_frame(idx, len(detections)),
            cur_is_early=_is_early_one_stroke_frame(idx + 1, len(detections)),
        )
        dynamic_score = _metrics_score(metrics)
        dynamic_confidence = _dynamic_confidence(
            metrics,
            detections[idx],
            detections[idx + 1],
        )
        row = {
            "from_frame_id": idx,
            "to_frame_id": idx + 1,
            "metrics": metrics,
            "dynamic_score": dynamic_score,
            "dynamic_confidence": dynamic_confidence,
        }
        dynamic_transitions.append(row)
        if (not overlay_failures_only) or any(not bool(m.get("pass")) for m in metrics.values()):
            _draw_one_stroke_transition_overlay(
                frames[idx],
                frames[idx + 1],
                detections[idx],
                detections[idx + 1],
                parsed_frames[idx],
                parsed_frames[idx + 1],
                metrics,
                dynamic_confidence,
                auto_goal_achieved,
                dynamic_overlay_dir / f"{video_id}_transition_{idx:03d}_{idx + 1:03d}_dynamic.png",
            )

    dynamic_doc = {
        "video_id": video_id,
        "task_type": "one_stroke",
        "transitions": dynamic_transitions,
    }
    _write_json(output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json", dynamic_doc)
    return _write_video_report(video_id, "one_stroke", manifest, static_doc, dynamic_doc, output_root)


# =====================================================================================
# order_swap_2d_puzzle ("2D swap") — sliding/swap tile puzzle.
#
# Discrete state is a row-major arrangement of block tokens (one blank).  We parse each
# frame with topo_spike.swap2d, then score against env-oracle GT (topo_spike.swap_oracle).
# Metrics follow reports/findings_swap_puzzle.md and are split into observation quality,
# task outcome, process validity/progress, and temporal coverage.
# =====================================================================================


def _swap2d_videos(limit: Optional[int], video_dir: Optional[Path]) -> List[Tuple[str, Path, int, int, int]]:
    """Discover imagined mp4s + (rows, cols, seed).

    Supports both the nested e2e layout (images/grid_RxC/repeat_N_seed_S/*_imagined.mp4)
    and a flat directory of grid_RxC__repeat_N_seed_S__*_imagined.mp4 files.
    """
    root = video_dir or SWAP2D_DEFAULT_DIR
    out: List[Tuple[str, Path, int, int, int]] = []
    seen: set = set()
    patt = re.compile(r"grid_(\d+)x(\d+).*?seed_(\d+)")
    candidates = sorted(root.rglob("*imagined.mp4")) + sorted(root.glob("*imagined.mp4"))
    for mp4 in candidates:
        rel = str(mp4)
        if rel in seen:
            continue
        m = patt.search(mp4.as_posix())
        if not m:
            continue
        rows, cols, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        seen.add(rel)
        out.append((_safe_video_id(f"grid_{rows}x{cols}_seed_{seed:03d}"), mp4, rows, cols, seed))
    out.sort(key=lambda t: t[0])
    return out[:limit] if limit is not None else out


def _swap2d_rollout_metadata(
    video_path: Path,
    rows: int,
    cols: int,
    seed: int,
    theoretical_min_steps: int,
) -> Dict[str, object]:
    """Load the actual runner budget/reference result when the batch JSONL is present."""
    model_answer = None
    for parent in (video_path.parent, *video_path.parents):
        candidate = parent / "model_answer.jsonl"
        if candidate.exists():
            model_answer = candidate
            break

    if model_answer is not None:
        try:
            with model_answer.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    meta = row.get("meta_info") or {}
                    if (
                        int(meta.get("grid_rows", -1)) == rows
                        and int(meta.get("grid_cols", -1)) == cols
                        and int(meta.get("seed", -1)) == seed
                    ):
                        budget = meta.get("step_budget")
                        return {
                            "step_budget": int(budget) if budget is not None else None,
                            "step_budget_source": "model_answer.meta_info",
                            "rollout_total_steps": meta.get("total_steps"),
                            "rollout_success": meta.get("success"),
                            "rollout_final_reason": meta.get("final_reason"),
                            "rollout_illegal_moves": meta.get("illegal_moves"),
                            "rollout_invalid_actions": meta.get("invalid_actions"),
                        }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return {
        "step_budget": max(1, int(math.ceil(theoretical_min_steps * swap_oracle.swap_env.DEFAULT_BUDGET_MULTIPLIER))),
        "step_budget_source": "environment_default",
        "rollout_total_steps": None,
        "rollout_success": None,
        "rollout_final_reason": None,
        "rollout_illegal_moves": None,
        "rollout_invalid_actions": None,
    }


def _swap2d_detection_record(
    idx: int,
    frame_path: str,
    parsed: "swap2d.ParsedSwapFrame",
    tile_centers: Optional[Dict[str, Tuple[float, float]]] = None,
    text_signatures: Optional[Dict[int, Dict[str, str]]] = None,
) -> Dict[str, object]:
    cells = [
        {
            "row": cell.row,
            "col": cell.col,
            "cell_index": cell.cell_index,
            "token": cell.token,
            "is_blank": cell.is_blank,
            "color_id": cell.color_id,
            "confidence": cell.confidence,
            "white_frac": cell.white_frac,
            "hue": cell.hue,
        }
        for cell in parsed.cells
    ]
    non_blank = [c["confidence"] for c in cells if not c["is_blank"] and c["confidence"] is not None]
    raw_confidence = (sum(non_blank) / len(non_blank)) if non_blank else 0.0
    status_caps = {
        "ok": 1.0,
        "partial": 0.45,
        "invalid_state": 0.25,
        "grid_not_found": 0.0,
    }
    confidence = min(raw_confidence, status_caps.get(parsed.parser_status, 0.35))
    return {
        "frame_id": idx,
        "path": frame_path,
        "task_type": "swap2d",
        "parser_status": parsed.parser_status,
        "state_valid": parsed.parser_status == "ok",
        "arrangement": list(parsed.arrangement),
        "n_blanks": parsed.n_blanks,
        "cells": cells,
        "tile_centers": {
            token: [round(center[0], 2), round(center[1], 2)]
            for token, center in (tile_centers or {}).items()
        },
        "text_signatures": {
            str(index): signature for index, signature in (text_signatures or {}).items()
        },
        "confidence": confidence,
        "warnings": list(parsed.warnings),
    }


def _swap2d_non_blank_tokens(arrangement: Sequence[Optional[str]]) -> List[Optional[str]]:
    return [t for t in arrangement if t != swap_oracle.BLANK_TOKEN]


def _swap2d_static_frame_ids(n_frames: int, n_checkpoints: int = 4) -> List[int]:
    """Fallback checkpoints when no settled Swap2D key-state can be recovered.

    For the default 41-frame sampling this is exactly {0, 13, 27, 40}; it generalises
    to other frame counts and always includes the first and last frame.
    """
    if n_frames <= 0:
        return []
    if n_frames <= n_checkpoints:
        return list(range(n_frames))
    return sorted({int(round(k * (n_frames - 1) / (n_checkpoints - 1))) for k in range(n_checkpoints)})


def _swap2d_static_key_frame_ids(
    key_states: Sequence[Dict[str, object]],
    n_frames: int,
    n_checkpoints: int = 4,
) -> List[int]:
    """Choose static checkpoints from settled key-states, retaining the final frame."""
    if not key_states:
        return _swap2d_static_frame_ids(n_frames, n_checkpoints)
    ids = [int(state["frame_id"]) for state in key_states]
    last_idx = n_frames - 1
    if last_idx >= 0 and last_idx not in ids:
        ids.append(last_idx)
    ids = sorted(set(ids))
    if len(ids) <= n_checkpoints:
        return ids
    return sorted({ids[int(round(k * (len(ids) - 1) / (n_checkpoints - 1)))] for k in range(n_checkpoints)})


def _swap2d_grid_structure_metric(record: Dict[str, object], gt: "swap_oracle.SwapGT") -> Metric:
    if record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "grid could not be resolved (board unreadable)", status="low_confidence")
    n_blanks = int(record.get("n_blanks") or 0)
    n_cells = len(record.get("arrangement") or [])
    if n_cells != gt.num_cells:
        return _metric(0.0, False, f"parsed {n_cells} cells, expected {gt.num_cells}")
    if n_blanks == 1:
        return _metric(1.0, True, f"{gt.grid_rows}x{gt.grid_cols} grid readable with exactly one blank")
    return _metric(0.0, False, f"expected exactly one blank, found {n_blanks}")


def _swap2d_block_identity_metric(record: Dict[str, object], gt: "swap_oracle.SwapGT") -> Metric:
    arrangement = list(record.get("arrangement") or [])
    if any(t is None for t in arrangement):
        n = sum(1 for t in arrangement if t is None)
        return _metric(0.0, False, f"{n} cell(s) unreadable — identity not verifiable", status="low_confidence")
    got = sorted(t for t in arrangement if t != swap_oracle.BLANK_TOKEN)
    expected = sorted(gt.selected_block_ids)
    if got == expected:
        return _metric(1.0, True, "block multiset matches the goal set (no dup/missing/extra)")
    got_c, exp_c = Counter(got), Counter(expected)
    extra = list((got_c - exp_c).elements())
    missing = list((exp_c - got_c).elements())
    return _metric(
        0.0, False,
        f"block multiset mismatch (extra={extra or '-'}, missing={missing or '-'})",
        evidence={"extra": extra, "missing": missing},
    )


def _swap2d_final_solution_metric(record: Dict[str, object], gt: "swap_oracle.SwapGT", is_final_frame: bool) -> Metric:
    if not is_final_frame:
        return _metric(0.0, True, "final-state check only applies to the last frame", status="not_applicable")
    arrangement = list(record.get("arrangement") or [])
    if any(t is None for t in arrangement):
        return _metric(0.0, False, "final frame unreadable — cannot confirm solved state", status="low_confidence")
    if arrangement == list(gt.goal_arrangement):
        return _metric(1.0, True, "final arrangement equals the goal grid (solved)")
    diffs = [i for i, (a, b) in enumerate(zip(arrangement, gt.goal_arrangement)) if a != b]
    return _metric(
        0.0, False,
        f"final arrangement differs from goal in {len(diffs)} cell(s)",
        evidence={"diff_cells": diffs, "parsed": arrangement, "goal": list(gt.goal_arrangement)},
    )


def _swap2d_static_metrics(record: Dict[str, object], gt: "swap_oracle.SwapGT", *, is_final_frame: bool) -> Dict[str, Metric]:
    return {
        "grid_structure_validity": _swap2d_grid_structure_metric(record, gt),
        "block_identity_validity": _swap2d_block_identity_metric(record, gt),
        "final_solution_valid": _swap2d_final_solution_metric(record, gt, is_final_frame),
    }


def _swap2d_grid_consistency_metric(
    records: Sequence[Dict[str, object]],
    expected_grid_count: int,
) -> Metric:
    """Video-level static consistency of visible grid count and colour identities."""
    signatures = []
    for record in records:
        colors = tuple(sorted((record.get("tile_centers") or {}).keys()))
        blank_count = 1 if int(record.get("n_blanks") or 0) > 0 else 0
        signatures.append({"grid_count": len(colors) + blank_count, "colors": colors})
    if len(signatures) < 2:
        return _metric(0.0, True, "grid consistency needs at least two static checkpoints", status="not_applicable")
    reference = signatures[0]
    consistent = all(signature == reference for signature in signatures[1:])
    count_ok = all(signature["grid_count"] == expected_grid_count for signature in signatures)
    passed = consistent and count_ok
    return _metric(
        1.0 if passed else 0.0,
        passed,
        "grid count and visible colour set stay constant across static checkpoints"
        if passed else "grid count or visible colour set jumps across static checkpoints",
        evidence={"checkpoint_signatures": signatures, "expected_grid_count": expected_grid_count},
    )


def _swap2d_token_positions(record: Dict[str, object]) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = defaultdict(list)
    for index, token in enumerate(record.get("arrangement") or []):
        if token not in (None, swap_oracle.BLANK_TOKEN):
            positions[str(token)].append(index)
    return positions


def _swap2d_adjacent_frame_metrics(
    prev: Dict[str, object],
    cur: Dict[str, object],
    expected_ids: Sequence[str],
    motion_threshold_px: float,
) -> Tuple[Dict[str, Metric], Dict[str, object]]:
    """Score continuous rendering between adjacent sampled frames.

    Tile centres provide physical motion evidence; fixed-cell token assignments check
    that every identity except the one moving tile stays in its original grid cell.
    """
    prev_centers = {str(k): v for k, v in (prev.get("tile_centers") or {}).items()}
    cur_centers = {str(k): v for k, v in (cur.get("tile_centers") or {}).items()}
    expected = [str(token) for token in expected_ids]
    common = [token for token in expected if token in prev_centers and token in cur_centers]
    missing = sorted(set(expected) - set(common))
    threshold = max(2.0, float(motion_threshold_px))

    if len(common) < max(2, len(expected) - 1):
        reason = f"tile motion is not observable ({len(common)}/{len(expected)} identities tracked)"
        unavailable = _metric(
            0.0, True, reason,
            evidence={"tracked_tokens": common, "missing_tokens": missing},
            status="not_applicable",
        )
        return {
            "alpha_number_consistency": dict(unavailable),
            "grid_move_consistency": dict(unavailable),
        }, {"moving_tokens": [], "motion_by_token_px": {}, "motion_threshold_px": threshold}

    displacements = {
        token: (
            float(cur_centers[token][0]) - float(prev_centers[token][0]),
            float(cur_centers[token][1]) - float(prev_centers[token][1]),
        )
        for token in common
    }
    # Remove global video/board jitter before deciding which tile actually moved.
    global_dx = float(np.median([delta[0] for delta in displacements.values()]))
    global_dy = float(np.median([delta[1] for delta in displacements.values()]))
    motion = {
        token: float(math.hypot(delta[0] - global_dx, delta[1] - global_dy))
        for token, delta in displacements.items()
    }
    moving = sorted(token for token, distance in motion.items() if distance > threshold)
    grid_ok = len(moving) <= 1
    grid_metric = _metric(
        1.0 if grid_ok else 0.0,
        grid_ok,
        f"{len(moving)} tile(s) moving; expected 0 or 1",
        evidence={
            "moving_tokens": moving,
            "motion_by_token_px": {token: round(distance, 3) for token, distance in motion.items()},
            "motion_threshold_px": round(threshold, 3),
            "global_shift_px": [round(global_dx, 3), round(global_dy, 3)],
        },
    )

    # Exclude at most one physically moving identity.  Any other token that changes
    # cell, disappears, or duplicates is a stationary-letter consistency failure.
    excluded = max(motion, key=motion.get) if moving else None
    prev_positions = _swap2d_token_positions(prev)
    cur_positions = _swap2d_token_positions(cur)
    inconsistent = []
    for token in expected:
        if token == excluded:
            continue
        if prev_positions.get(token) != cur_positions.get(token):
            inconsistent.append(token)
    excluded_cells = set(prev_positions.get(excluded, []) + cur_positions.get(excluded, [])) if excluded else set()
    prev_text = prev.get("text_signatures") or {}
    cur_text = cur.get("text_signatures") or {}

    def signature_distance(kind: str, cell_index: int) -> Optional[float]:
        a = (prev_text.get(str(cell_index)) or {}).get(kind)
        b = (cur_text.get(str(cell_index)) or {}).get(kind)
        if not a or not b or len(a) != len(b):
            return None
        return bin(int(a, 16) ^ int(b, 16)).count("1") / (4.0 * len(a))

    letter_changes = {
        index: distance
        for index in range(len(prev.get("arrangement") or []))
        if index not in excluded_cells
        for distance in [signature_distance("letter", index)]
        if distance is not None and distance > 0.12
    }
    number_changes = {
        index: distance
        for index in range(len(prev.get("arrangement") or []))
        for distance in [signature_distance("number", index)]
        if distance is not None and distance > 0.08
    }
    text_observable = bool(prev_text and cur_text)
    alpha_number_ok = not inconsistent and not letter_changes and not number_changes
    alpha_number_metric = _metric(
        1.0 if alpha_number_ok else 0.0,
        alpha_number_ok,
        "all non-moving letters and fixed cell numbers remain visually stable"
        if alpha_number_ok else "letter/number content changed outside the moving tile",
        evidence={
            "excluded_moving_token": excluded,
            "excluded_letter_cells": sorted(excluded_cells),
            "inconsistent_tokens": inconsistent,
            "letter_change_fraction_by_cell": {str(k): round(v, 4) for k, v in letter_changes.items()},
            "number_change_fraction_by_cell": {str(k): round(v, 4) for k, v in number_changes.items()},
        },
        status=None if text_observable else "not_applicable",
    )
    analysis = {
        "moving_tokens": moving,
        "motion_by_token_px": {token: round(distance, 3) for token, distance in motion.items()},
        "motion_threshold_px": round(threshold, 3),
    }
    return {
        "alpha_number_consistency": alpha_number_metric,
        "grid_move_consistency": grid_metric,
    }, analysis


def _swap2d_key_states(
    detections: Sequence[Dict[str, object]],
    min_stable_frames: int = 2,
) -> List[Dict[str, object]]:
    """Collapse settled legal runs into a discrete key-state sequence.

    A legal arrangement must persist for at least two sampled frames.  A valid first
    or last frame is retained even as a singleton so short endpoint holds do not lose
    the initial/final state.  Invalid and low-confidence motion frames delimit runs.
    """
    runs: List[Dict[str, object]] = []
    n = len(detections)
    i = 0
    while i < n:
        det = detections[i]
        if det.get("parser_status") != "ok" or not det.get("state_valid"):
            i += 1
            continue
        arrangement = tuple(det.get("arrangement") or [])
        j = i + 1
        while j < n:
            nxt = detections[j]
            if nxt.get("parser_status") != "ok" or not nxt.get("state_valid"):
                break
            if tuple(nxt.get("arrangement") or []) != arrangement:
                break
            j += 1

        is_endpoint = i == 0 or j == n
        if (j - i) >= max(1, int(min_stable_frames)) or is_endpoint:
            center = 0.5 * (i + j - 1)
            if i == 0:
                representative = i
            elif j == n:
                representative = j - 1
            else:
                representative = max(
                    range(i, j),
                    key=lambda k: (float(detections[k].get("confidence") or 0.0), -abs(k - center)),
                )
            run = {
                "frame_id": representative,
                "start_frame_id": i,
                "end_frame_id": j - 1,
                "arrangement": list(arrangement),
                "confidence": detections[representative].get("confidence"),
            }
            # An occlusion can split one settled state into two readable runs.  It is
            # still one key-state, not an idle transition through an unreadable gap.
            if runs and runs[-1]["arrangement"] == run["arrangement"]:
                runs[-1]["end_frame_id"] = run["end_frame_id"]
                if float(run.get("confidence") or 0.0) > float(runs[-1].get("confidence") or 0.0):
                    runs[-1]["frame_id"] = run["frame_id"]
                    runs[-1]["confidence"] = run["confidence"]
            else:
                runs.append(run)
        i = j
    return runs


def _swap2d_outcome_metrics(
    detections: Sequence[Dict[str, object]],
    key_states: Sequence[Dict[str, object]],
    gt: "swap_oracle.SwapGT",
    rollout_metadata: Dict[str, object],
) -> Dict[str, object]:
    goal = list(gt.goal_arrangement)
    initial = list(gt.initial_arrangement)
    first_state = list(key_states[0]["arrangement"]) if key_states else None
    last_state = list(key_states[-1]["arrangement"]) if key_states else None
    final_record = detections[-1] if detections else {}
    final_arrangement = list(final_record.get("arrangement") or [])
    final_readable = final_record.get("parser_status") == "ok"

    if first_state is None:
        initial_metric = _metric(
            0.0, False, "no readable stable initial state", status="low_confidence"
        )
    else:
        initial_ok = first_state == initial
        initial_metric = _metric(
            1.0 if initial_ok else 0.0,
            initial_ok,
            "first stable state matches the environment initial arrangement"
            if initial_ok else "first stable state does not match the environment initial arrangement",
            evidence={"observed": first_state, "expected": initial},
        )

    goal_hits = [idx for idx, state in enumerate(key_states) if list(state["arrangement"]) == goal]
    goal_ever_reached = bool(goal_hits)

    terminal_ok = bool(final_readable and final_arrangement == goal)
    goal_metric = _metric(
        1.0 if terminal_ok else 0.0,
        terminal_ok,
        "final video frame is a readable exact goal state"
        if terminal_ok else "final video frame is not a readable exact goal state",
        evidence={
            "final_parser_status": final_record.get("parser_status"),
            "observed": final_arrangement,
            "goal": goal,
        },
        status="low_confidence" if not final_readable else None,
        confidence=final_record.get("confidence"),
    )

    post_goal_states = key_states[goal_hits[0]:] if goal_hits else []
    goal_held = bool(goal_ever_reached and terminal_ok and all(list(s["arrangement"]) == goal for s in post_goal_states))
    hold_metric = _metric(
        1.0 if goal_held else 0.0,
        goal_held,
        "goal remains unchanged through the end of the video"
        if goal_held else (
            "goal was reached but was not preserved through the final frame"
            if goal_ever_reached else "goal was never reached, so terminal hold failed"
        ),
        evidence={
            "post_goal_key_states": len(post_goal_states),
            "goal_reached": terminal_ok,
            "goal_ever_observed_internally": goal_ever_reached,
        },
    )

    if len(final_arrangement) == gt.num_cells:
        correct_cells = sum(1 for observed, expected in zip(final_arrangement, goal) if observed == expected)
        cell_score = correct_cells / gt.num_cells
        cell_metric = _metric(
            cell_score,
            correct_cells == gt.num_cells,
            f"{correct_cells}/{gt.num_cells} final cells match the goal",
            evidence={"correct_cells": correct_cells, "total_cells": gt.num_cells},
            status="low_confidence" if not final_readable else None,
            confidence=final_record.get("confidence"),
        )
    else:
        cell_metric = _metric(0.0, False, "final grid cell count is unreadable", status="low_confidence")

    last_distance = swap_oracle.swap_distance(last_state, goal) if last_state is not None else None
    if last_distance is None:
        progress_metric = _metric(
            0.0, False, "no stable state is available for remaining-distance scoring", status="low_confidence"
        )
    else:
        progress_score = _clamp01(1.0 - last_distance / max(1, gt.theoretical_min_steps))
        progress_metric = _metric(
            progress_score,
            last_distance == 0,
            f"last stable state is {last_distance} optimal move(s) from the goal",
            evidence={
                "remaining_distance": last_distance,
                "initial_theoretical_min_steps": gt.theoretical_min_steps,
            },
        )

    budget = rollout_metadata.get("step_budget")
    feasible = budget is not None and int(budget) >= gt.theoretical_min_steps
    feasibility_metric = _metric(
        1.0 if feasible else 0.0,
        feasible,
        f"step budget {budget} {'covers' if feasible else 'is below'} theoretical minimum {gt.theoretical_min_steps}",
        evidence={
            "step_budget": budget,
            "theoretical_min_steps": gt.theoretical_min_steps,
            "source": rollout_metadata.get("step_budget_source"),
        },
        status="not_applicable" if budget is None else None,
    )

    metrics = {
        "initial_state_match": initial_metric,
        "final_solution_valid": goal_metric,
        "goal_hold_consistency": hold_metric,
        "final_goal_cell_accuracy": cell_metric,
        "last_stable_goal_progress": progress_metric,
        "budget_feasibility": feasibility_metric,
    }
    return {
        "metrics": metrics,
        "task_success": bool(goal_metric.get("pass")),
        "task_success_score": float(goal_metric.get("score") or 0.0),
        "outcome_confidence": goal_metric.get("confidence"),
    }


def _swap2d_process_summary(
    key_states: Sequence[Dict[str, object]],
    gt: "swap_oracle.SwapGT",
    step_budget: Optional[object],
) -> Dict[str, object]:
    states = [list(state["arrangement"]) for state in key_states]
    distances = [swap_oracle.swap_distance(state, gt.goal_arrangement) for state in states]
    spans = [swap_oracle.swap_distance(a, b) for a, b in zip(states, states[1:])]
    comparable_spans = [int(span) for span in spans if span is not None]
    unit_count = sum(1 for span in comparable_spans if span == 1)
    transition_count = len(spans)
    observability = unit_count / transition_count if transition_count else 0.0
    observability_metric = _metric(
        observability,
        bool(transition_count and unit_count == transition_count),
        f"{unit_count}/{transition_count} key-state transition(s) resolve to one action",
        evidence={"unit_transitions": unit_count, "total_transitions": transition_count},
        status="low_confidence" if not transition_count else None,
    )

    distance_pairs = [(a, b) for a, b in zip(distances, distances[1:]) if a is not None and b is not None]
    progressive = sum(1 for a, b in distance_pairs if b < a)
    regressions = sum(1 for a, b in distance_pairs if b > a)
    monotonic_rate = progressive / len(distance_pairs) if distance_pairs else 0.0
    monotonic_metric = _metric(
        monotonic_rate,
        bool(distance_pairs and regressions == 0),
        f"{progressive}/{len(distance_pairs)} transition(s) reduce goal distance; regressions={regressions}",
        evidence={"progressive_transitions": progressive, "regression_count": regressions},
        status="low_confidence" if not distance_pairs else None,
    )

    if distances and distances[0] is not None and distances[-1] is not None:
        progress_score = _clamp01(
            (int(distances[0]) - int(distances[-1])) / max(1, int(distances[0]))
        )
        normalized_metric = _metric(
            progress_score,
            int(distances[-1]) == 0,
            f"stable-state goal distance changed {distances[0]}->{distances[-1]}",
            evidence={"initial_distance": distances[0], "final_distance": distances[-1]},
        )
    else:
        normalized_metric = _metric(
            0.0, False, "insufficient stable states for normalized progress", status="low_confidence"
        )

    minimum_actions = (
        sum(comparable_spans)
        if spans and len(comparable_spans) == len(spans) else None
    )
    reached_goal = bool(states and states[-1] == list(gt.goal_arrangement))
    if reached_goal and minimum_actions:
        efficiency = _clamp01(gt.theoretical_min_steps / minimum_actions)
        efficiency_metric = _metric(
            efficiency,
            minimum_actions == gt.theoretical_min_steps,
            f"state-path lower bound uses {minimum_actions} action(s) vs theoretical {gt.theoretical_min_steps}",
            evidence={
                "minimum_required_actions_between_observed_states": minimum_actions,
                "theoretical_min_steps": gt.theoretical_min_steps,
                "is_lower_bound_only": any(span != 1 for span in comparable_spans),
            },
        )
    else:
        efficiency_metric = _metric(
            0.0, True, "path efficiency requires an observed goal state", status="not_applicable"
        )

    if step_budget is not None and minimum_actions is not None:
        lower_bound_within_budget = minimum_actions <= int(step_budget)
        budget_metric = _metric(
            1.0 if lower_bound_within_budget else 0.0,
            lower_bound_within_budget,
            f"observed state-path lower bound {minimum_actions} vs budget {step_budget}",
            evidence={"minimum_required_actions": minimum_actions, "step_budget": step_budget},
        )
    else:
        budget_metric = _metric(
            0.0, True, "budget lower-bound comparison is unavailable", status="not_applicable"
        )

    return {
        "metrics": {
            "transition_observability": observability_metric,
            "monotonic_goal_progress": monotonic_metric,
            "normalized_goal_progress": normalized_metric,
            "path_lower_bound_efficiency": efficiency_metric,
            "path_lower_bound_within_budget": budget_metric,
        },
        "distance_to_goal_curve": distances,
        "transition_minimum_action_spans": spans,
        "minimum_required_actions_between_observed_states": minimum_actions,
        "regression_count": regressions,
    }


def _process_swap2d_video(
    video_id: str,
    video_path: Path,
    rows: int,
    cols: int,
    seed: int,
    output_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gt = swap_oracle.load_swap_gt(rows, cols, seed, metadata=metadata)
    expected_ids = list(gt.selected_block_ids)
    rollout_metadata = _swap2d_rollout_metadata(
        video_path, rows, cols, seed, gt.theoretical_min_steps
    )
    manifest, frames = extract_frames(video_path, video_id, "swap2d", output_root, sampling_mode)

    geometry = swap2d.estimate_swap_grid(frames, rows, cols)
    parsed_frames: List[swap2d.ParsedSwapFrame] = [
        swap2d.parse_swap_frame(frame, rows, cols, expected_ids, geometry=geometry) for frame in frames
    ]
    tile_centers = [
        swap2d.detect_tile_centers(frame, expected_ids, geometry=geometry) for frame in frames
    ]
    text_signatures = [
        swap2d.text_region_signatures(frame, geometry, rows, cols) for frame in frames
    ]
    detections = [
        _swap2d_detection_record(
            idx, str(frame_rec["path"]), parsed_frames[idx], tile_centers[idx], text_signatures[idx]
        )
        for idx, frame_rec in enumerate(manifest["frames"])
    ]
    detection_doc = {
        "video_id": video_id,
        "task_type": "swap2d",
        "metadata": {
            "grid_rows": rows,
            "grid_cols": cols,
            "seed": seed,
            "difficulty": gt.difficulty,
            "selected_block_ids": expected_ids,
            "initial_arrangement": list(gt.initial_arrangement),
            "goal_arrangement": list(gt.goal_arrangement),
            "theoretical_min_steps": gt.theoretical_min_steps,
            **rollout_metadata,
            "grid_geometry": (
                {
                    "col_centers": list(geometry.col_centers),
                    "row_centers": list(geometry.row_centers),
                    "tile_width": geometry.tile_width,
                    "tile_height": geometry.tile_height,
                    "board_bbox_xyxy": list(geometry.board_bbox_xyxy),
                    "reference_frame_ids": list(geometry.reference_frame_ids),
                }
                if geometry is not None else None
            ),
        },
        "frames": detections,
    }
    _write_json(output_root / "detections" / f"{video_id}_per_frame_detection.json", detection_doc)
    key_states = _swap2d_key_states(detections)
    outcome_doc = {
        "video_id": video_id,
        "task_type": "swap2d",
        **_swap2d_outcome_metrics(detections, key_states, gt, rollout_metadata),
    }
    _write_json(output_root / "outcome_metrics" / f"{video_id}_outcome_metrics.json", outcome_doc)
    process_summary = _swap2d_process_summary(
        key_states, gt, rollout_metadata.get("step_budget")
    )

    static_overlay_dir, dynamic_overlay_dir = _overlay_dirs(SWAP2D_DETECTOR_NAME)
    # A key-state rerun may emit different frame pairs than an older adjacent-frame
    # run.  Remove only this video's generated overlays so stale failures (for example
    # 018->019) cannot be mistaken for current output.
    for stale in itertools.chain(
        static_overlay_dir.glob(f"{video_id}_frame_*_static.png"),
        dynamic_overlay_dir.glob(f"{video_id}_transition_*_dynamic.png"),
    ):
        stale.unlink()
    last_idx = len(detections) - 1
    static_frame_ids = _swap2d_static_key_frame_ids(key_states, len(detections))
    grid_consistency = _swap2d_grid_consistency_metric(
        [detections[index] for index in static_frame_ids], gt.num_cells
    )
    static_frames = []
    for idx in static_frame_ids:
        det = detections[idx]
        metrics = _swap2d_static_metrics(det, gt, is_final_frame=idx == last_idx)
        metrics["grid_consistency"] = grid_consistency
        static_score = _metrics_score(metrics)
        static_confidence = _static_confidence(metrics, det)
        static_pass = _metrics_pass(metrics)
        static_frames.append(
            {
                "frame_id": idx,
                "metrics": metrics,
                "static_score": static_score,
                "static_confidence": static_confidence,
            }
        )
        if (not overlay_failures_only) or (not static_pass):
            suffix = " static fail" if not static_pass else ""
            overlay = swap2d.draw_swap_overlay(
                frames[idx], parsed_frames[idx],
                f"{video_id} frame {idx:03d} {_metrics_overlay_label('static', static_score, static_confidence, static_pass)}{suffix}",
            )
            cv2.imwrite(str(static_overlay_dir / f"{video_id}_frame_{idx:03d}_static.png"), overlay)

    static_doc = {"video_id": video_id, "task_type": "swap2d", "frames": static_frames}
    _write_json(output_root / "static_metrics" / f"{video_id}_static_metrics.json", static_doc)

    dynamic_transitions = []
    motion_threshold_px = (
        0.08 * min(geometry.tile_width, geometry.tile_height)
        if geometry is not None else 8.0
    )
    for idx in range(len(detections) - 1):
        metrics, analysis = _swap2d_adjacent_frame_metrics(
            detections[idx], detections[idx + 1], expected_ids, motion_threshold_px
        )
        dynamic_transitions.append(
            {
                "from_frame_id": idx,
                "to_frame_id": idx + 1,
                "transition_kind": "adjacent_frame",
                **analysis,
                "metrics": metrics,
                "dynamic_score": _metrics_score(metrics),
                "dynamic_confidence": _mean(
                    (_metrics_confidence(metrics), detections[idx].get("confidence"), detections[idx + 1].get("confidence"))
                ),
            }
        )

    dynamic_doc = {
        "video_id": video_id,
        "task_type": "swap2d",
        "temporal_mode": "adjacent_frames",
        "key_states": key_states,
        "process_summary": process_summary,
        "transitions": dynamic_transitions,
    }
    _write_json(output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json", dynamic_doc)
    return _write_video_report(
        video_id, "swap2d", manifest, static_doc, dynamic_doc, output_root,
        outcome_doc=outcome_doc,
    )


# =====================================================================================
# continuity_pipe ("pipe") — rotate-the-pipes connectivity puzzle.
#
# Discrete state is a per-cell current_mask (which arms N/E/S/W a pipe shows) plus its
# green/blue connectivity colour.  We parse each frame with topo_spike.pipe_cv against a
# clip-level lattice, then score against env-oracle GT (topo_spike.pipe_oracle).  Static
# metrics look at 4 checkpoint frames {0,13,27,40}; dynamic metrics first map
# each frame/cell to a registered legal pipe pattern, skip only bracketed
# unreadable/rotating -1 spans, and fail readable pipe shapes outside the
# registered pattern table.
# =====================================================================================

_PIPE_NAME_RE = re.compile(r"grid_(\d+)x(\d+).*?difficulty_(easy|medium|hard)")
PIPE_INTERMEDIATE_PATTERN = -1


def _pipe_videos(limit: Optional[int], video_dir: Optional[Path]) -> List[Tuple[str, Path, str, str, int]]:
    """Discover imagined mp4s + (video_id, mp4, level, difficulty, seed)."""
    root = video_dir or PIPE_DEFAULT_DIR
    seed_re = re.compile(r"seed_(\d+)")
    out: List[Tuple[str, Path, str, str, int]] = []
    for mp4 in sorted(root.rglob("*imagined.mp4")):
        m = _PIPE_NAME_RE.search(mp4.name)
        seeds = seed_re.findall(mp4.name)
        if not m or not seeds:
            continue
        grid = int(m.group(1))
        difficulty = m.group(3)
        seed = int(seeds[-1])
        level = f"grid_{grid}x{grid}"
        video_id = _safe_video_id(f"{level}_{difficulty}_seed_{seed:03d}")
        out.append((video_id, mp4, level, difficulty, seed))
    out.sort(key=lambda t: t[0])
    return out[:limit] if limit is not None else out


def _pipe_masks_full(record: Dict[str, object], total: int) -> List[int]:
    masks = record.get("masks") or {}
    return [int(masks.get(i, 0)) for i in range(total)]


def _pipe_detection_record(
    idx: int,
    frame_path: str,
    parsed: "pipe_cv.ParsedPipeFrame",
    gt: "pipe_oracle.PipeGT",
) -> Dict[str, object]:
    cells = [
        {
            "index": cell.index,
            "gx": cell.gx,
            "gy": cell.gy,
            "center_xy": [round(cell.center_xy[0], 1), round(cell.center_xy[1], 1)],
            "active": cell.active,
            "mask": cell.mask,
            "arms": cell.arms,
            "shape": cell.shape,
            "color": cell.color,
            "confidence": cell.confidence,
        }
        for cell in parsed.cells
    ]
    masks = {int(i): int(parsed.masks[i]) for i in parsed.active_indices}
    colors = {int(i): parsed.colors[i] for i in parsed.active_indices}
    confs = [cell.confidence for cell in parsed.cells if cell.active]
    raw_conf = (sum(confs) / len(confs)) if confs else 0.0
    status_cap = {"ok": 1.0, "grid_not_found": 0.0}.get(parsed.parser_status, 0.4)
    confidence = min(raw_conf, status_cap) if parsed.active_indices else 0.0
    full = [int(masks.get(i, 0)) for i in range(gt.total_cells)]
    connected = sorted(set(pipe_oracle.connected_indices(gt.grid_size, gt.source_index, full)))
    return {
        "frame_id": idx,
        "path": frame_path,
        "task_type": "pipe",
        "parser_status": parsed.parser_status,
        "active_indices": list(parsed.active_indices),
        "masks": masks,
        "colors": colors,
        "connected_indices": connected,
        "source_index": gt.source_index,
        "cells": cells,
        "n_active": len(parsed.active_indices),
        "confidence": confidence,
        "warnings": list(parsed.warnings),
    }


def _pipe_color_matches_connectivity(record: Dict[str, object], gt: "pipe_oracle.PipeGT") -> Metric:
    if record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "board unreadable — connectivity colouring not verifiable", status="low_confidence")
    active = [int(i) for i in record.get("active_indices") or []]
    colors = {int(i): c for i, c in (record.get("colors") or {}).items()}
    connected = set(int(i) for i in record.get("connected_indices") or [])
    mism = [i for i in active if (colors.get(i) == "green") != (i in connected)]
    source_green = colors.get(gt.source_index) == "green"
    consistent = not mism and source_green
    passed = consistent
    reason = (
        "green/blue colouring matches source connectivity computed from the parsed pipes"
        if passed else
        f"colouring inconsistent with connectivity ({len(mism)} cell(s); source_green={source_green})"
    )
    return _metric(1.0 if passed else 0.0, passed, reason, evidence={"inconsistent_cells": mism})


def _pipe_final_solution_valid(record: Dict[str, object], gt: "pipe_oracle.PipeGT", is_final_frame: bool) -> Metric:
    if not is_final_frame:
        return _metric(0.0, True, "final-solution check only applies to the last frame", status="not_applicable")
    if record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "final frame unreadable — cannot confirm solved state", status="low_confidence")
    full = _pipe_masks_full(record, gt.total_cells)
    connected = set(pipe_oracle.connected_indices(gt.grid_size, gt.source_index, full))
    solved = gt.active_indices.issubset(connected)
    n_connected = len(gt.active_indices & connected)
    return _metric(
        1.0 if solved else 0.0, solved,
        "all pipes connect to the source (puzzle solved)" if solved
        else f"{n_connected}/{len(gt.active_indices)} pipes connect to the source (unsolved)",
        evidence={"connected_active": n_connected, "total_active": len(gt.active_indices)},
    )


def _pipe_static_metrics(
    record: Dict[str, object], gt: "pipe_oracle.PipeGT", *, is_final_frame: bool
) -> Dict[str, Metric]:
    return {
        "color_matches_connectivity": _pipe_color_matches_connectivity(record, gt),
        "final_solution_valid": _pipe_final_solution_valid(record, gt, is_final_frame),
    }


def _pipe_shape_of(mask: int) -> str:
    return pipe_oracle.pipe_shape(int(mask))


def _pipe_allowed_masks(gt: "pipe_oracle.PipeGT", cell_index: int) -> List[int]:
    solved = int(gt.solved_masks[int(cell_index)]) & 15
    if solved == 0:
        return [0]
    return sorted({pipe_oracle.rotate_mask(solved, ticks) for ticks in range(4)})


def _pipe_quarter_turn_distance(prev_mask: int, cur_mask: int) -> Optional[int]:
    prev_m = int(prev_mask) & 15
    cur_m = int(cur_mask) & 15
    matches = [
        ticks for ticks in range(4)
        if pipe_oracle.rotate_mask(prev_m, ticks) == cur_m
    ]
    if not matches:
        return None
    return min(min(ticks, 4 - ticks) for ticks in matches)


def _pipe_pattern_state(
    record: Dict[str, object],
    gt: "pipe_oracle.PipeGT",
) -> Dict[str, object]:
    """Map one parsed frame to registered legal pipe patterns or -1.

    A frame is legal only when every GT-active pipe cell is present and its
    current mask is one of that cell's legal rotations. Dynamic scoring still
    compares every adjacent sampled frame; this per-frame state is kept as
    evidence for interpreting parser and pattern failures.
    """
    frame_id = int(record.get("frame_id") or 0)
    parser_status = str(record.get("parser_status") or "")
    active = set(int(i) for i in record.get("active_indices") or [])
    masks = {int(i): int(m) for i, m in (record.get("masks") or {}).items()}

    patterns: Dict[int, int] = {}
    missing_cells: List[int] = []
    illegal_pattern_cells: List[int] = []
    illegal_pattern_details: List[Dict[str, object]] = []
    for index in sorted(gt.active_indices):
        mask = int(masks.get(index, 0)) & 15
        allowed = _pipe_allowed_masks(gt, index)
        if index in active and mask in allowed:
            patterns[index] = mask
        else:
            patterns[index] = PIPE_INTERMEDIATE_PATTERN
            if index not in active:
                missing_cells.append(index)
            else:
                illegal_pattern_cells.append(index)
                illegal_pattern_details.append(
                    {
                        "cell": int(index),
                        "observed_mask": mask,
                        "observed_shape": _pipe_shape_of(mask),
                        "expected_shape": _pipe_shape_of(gt.solved_masks[index]),
                        "allowed_masks": allowed,
                    }
                )

    extra_cells = sorted(active - set(gt.active_indices))
    extra_cell_details = [
        {
            "cell": int(index),
            "observed_mask": int(masks.get(index, 0)) & 15,
            "observed_shape": _pipe_shape_of(masks.get(index, 0)),
        }
        for index in extra_cells
    ]
    legal = (
        parser_status == "ok"
        and not missing_cells
        and not extra_cells
        and not illegal_pattern_cells
    )
    reasons: List[str] = []
    if parser_status != "ok":
        reasons.append(f"parser_status={parser_status}")
    if missing_cells:
        reasons.append(f"missing={missing_cells}")
    if extra_cells:
        reasons.append(f"extra={extra_cells}")
    if illegal_pattern_cells:
        reasons.append(f"illegal_pattern={illegal_pattern_cells}")
    if not reasons:
        reasons.append("all active cells match registered legal patterns")
    return {
        "frame_id": frame_id,
        "legal": legal,
        "patterns": patterns,
        "parser_status": parser_status,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "extra_cell_details": extra_cell_details,
        "illegal_pattern_cells": illegal_pattern_cells,
        "illegal_pattern_details": illegal_pattern_details,
        "reason": "; ".join(reasons),
        "confidence": record.get("confidence"),
    }


def _pipe_cell_occupancy_consistency(prev: Dict[str, object], cur: Dict[str, object]) -> Metric:
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "a frame is unreadable — cell occupancy not verifiable", status="low_confidence")
    pa = set(int(i) for i in prev.get("active_indices") or [])
    ca = set(int(i) for i in cur.get("active_indices") or [])
    if pa != ca:
        appeared = sorted(ca - pa)
        vanished = sorted(pa - ca)
        return _metric(
            0.0, False,
            f"pipe-cell set changed (appeared={appeared or '-'}, vanished={vanished or '-'})",
            evidence={"appeared": appeared, "vanished": vanished},
        )
    return _metric(1.0, True, "occupied pipe cells are unchanged")


def _pipe_type_consistency(prev: Dict[str, object], cur: Dict[str, object]) -> Metric:
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "a frame is unreadable — pipe type not verifiable", status="low_confidence")
    pa = set(int(i) for i in prev.get("active_indices") or [])
    ca = set(int(i) for i in cur.get("active_indices") or [])
    checked = sorted(pa & ca)
    if not checked:
        return _metric(
            0.0,
            True,
            "no shared occupied cells, so pipe type consistency is not applicable",
            status="not_applicable",
        )
    pm = {int(i): int(m) for i, m in (prev.get("masks") or {}).items()}
    cm = {int(i): int(m) for i, m in (cur.get("masks") or {}).items()}
    shape_changed = [i for i in checked if _pipe_shape_of(pm.get(i, 0)) != _pipe_shape_of(cm.get(i, 0))]
    if shape_changed:
        return _metric(
            0.0, False,
            f"{len(shape_changed)} cell(s) changed pipe shape (only orientation may change): {shape_changed}",
            evidence={"checked_cells": checked, "shape_changed_cells": shape_changed},
        )
    return _metric(
        1.0,
        True,
        "pipe types are unchanged on shared occupied cells (only orientation may differ)",
        evidence={"checked_cells": checked},
    )


def _pipe_orientation_changes(prev: Dict[str, object], cur: Dict[str, object]) -> List[int]:
    pm = {int(i): int(m) for i, m in (prev.get("masks") or {}).items()}
    cm = {int(i): int(m) for i, m in (cur.get("masks") or {}).items()}
    common = set(pm) & set(cm)
    return sorted(i for i in common if pm[i] != cm[i])


def _pipe_rotation_num_valid(prev: Dict[str, object], cur: Dict[str, object]) -> Tuple[Metric, List[int]]:
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "a frame is unreadable — rotation count not verifiable", status="low_confidence"), []
    changed = _pipe_orientation_changes(prev, cur)
    if not changed:
        return _metric(1.0, True, "no pipe orientation changed between these frames"), []
    if len(changed) > 1:
        return _metric(
            0.0, False,
            f"{len(changed)} pipes changed orientation in one transition (at most one is allowed): {changed}",
            evidence={"changed_cells": changed},
        ), changed
    return _metric(1.0, True, f"exactly one pipe (cell {changed[0]}) changed orientation"), changed


def _pipe_cell_arm_mask(
    frame_bgr: np.ndarray,
    geometry: Optional["pipe_cv.PipeGeometry"],
    cell_index: int,
) -> Optional[np.ndarray]:
    if geometry is None:
        return None
    grid_size = int(geometry.grid_size)
    gx, gy = int(cell_index) % grid_size, int(cell_index) // grid_size
    cx, cy = geometry.center_xy(gx, gy)
    pitch = float(geometry.pitch)
    half = max(4, int(round(0.48 * pitch)))
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half + 1
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half + 1
    h, w = frame_bgr.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    crop = frame_bgr[y0c:y1c, x0c:x1c]
    if crop.size == 0:
        return None
    color, _green, _blue = pipe_cv._color_masks(crop)
    mask = color.astype(np.uint8)

    # The filled centre dot is rotation-invariant and can dominate IoU.  Remove it
    # so the angle estimate is driven by the arms that actually rotate.
    local_cx = int(round(cx)) - x0c
    local_cy = int(round(cy)) - y0c
    rr = max(2, int(round(0.16 * pitch)))
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    center = (xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= rr ** 2
    mask[center] = 0
    if int(mask.sum()) < 8:
        return None
    return mask


def _rotate_binary_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return (rotated > 0).astype(np.uint8)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / union)


def _estimate_pipe_rotation_angle(
    prev_frame_bgr: np.ndarray,
    cur_frame_bgr: np.ndarray,
    geometry: Optional["pipe_cv.PipeGeometry"],
    cell_index: int,
) -> Optional[Tuple[float, float]]:
    prev_mask = _pipe_cell_arm_mask(prev_frame_bgr, geometry, cell_index)
    cur_mask = _pipe_cell_arm_mask(cur_frame_bgr, geometry, cell_index)
    if prev_mask is None or cur_mask is None or prev_mask.shape != cur_mask.shape:
        return None

    best_angle = 0.0
    best_iou = -1.0
    for angle in np.arange(-120.0, 120.0 + 1e-6, 2.0):
        iou = _mask_iou(_rotate_binary_mask(prev_mask, float(angle)), cur_mask)
        if iou > best_iou:
            best_angle = float(angle)
            best_iou = float(iou)
    return best_angle, best_iou


def _pipe_rotation_angle_valid(
    prev: Dict[str, object],
    cur: Dict[str, object],
    prev_frame_bgr: Optional[np.ndarray] = None,
    cur_frame_bgr: Optional[np.ndarray] = None,
    geometry: Optional["pipe_cv.PipeGeometry"] = None,
    changed: Optional[Sequence[int]] = None,
) -> Metric:
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "a frame is unreadable — rotation angle not verifiable", status="low_confidence")
    changed_cells = list(changed) if changed is not None else _pipe_orientation_changes(prev, cur)
    if not changed_cells:
        return _metric(1.0, True, "no parsed pipe rotation; angle bound is satisfied")
    if len(changed_cells) > 1:
        return _metric(
            0.0, False,
            "multiple pipes changed orientation, so no single rotation angle can be verified",
            evidence={"changed_cells": changed_cells},
        )
    if prev_frame_bgr is None or cur_frame_bgr is None or geometry is None:
        return _metric(0.0, False, "rotation angle requires frame pixels and pipe geometry", status="low_confidence")

    cell = int(changed_cells[0])
    estimate = _estimate_pipe_rotation_angle(prev_frame_bgr, cur_frame_bgr, geometry, cell)
    if estimate is None:
        return _metric(
            0.0, False,
            f"cell {cell} rotation angle could not be estimated from local pipe masks",
            status="low_confidence",
            evidence={"cell": cell},
        )
    angle_deg, best_iou = estimate
    abs_angle = abs(float(angle_deg))
    passed = abs_angle < PIPE_ROTATION_ANGLE_LIMIT_DEG
    return _metric(
        1.0 if passed else 0.0,
        passed,
        (
            f"cell {cell} rotated by about {abs_angle:.1f} degrees (< {PIPE_ROTATION_ANGLE_LIMIT_DEG:.0f})"
            if passed else
            f"cell {cell} rotated by about {abs_angle:.1f} degrees (>= {PIPE_ROTATION_ANGLE_LIMIT_DEG:.0f})"
        ),
        confidence=max(0.45, min(0.95, float(best_iou))),
        evidence={"cell": cell, "angle_degrees": angle_deg, "abs_angle_degrees": abs_angle, "mask_iou": best_iou},
    )


def _pipe_rotation_num_angle_valid(
    prev: Dict[str, object],
    cur: Dict[str, object],
    prev_frame_bgr: Optional[np.ndarray] = None,
    cur_frame_bgr: Optional[np.ndarray] = None,
    geometry: Optional["pipe_cv.PipeGeometry"] = None,
    gt: Optional["pipe_oracle.PipeGT"] = None,
) -> Tuple[Metric, List[int]]:
    if gt is not None:
        if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
            return _metric(
                0.0,
                False,
                "an adjacent frame is unreadable — registered pattern transition not verifiable",
                status="low_confidence",
            ), []

        occupancy_metric = _pipe_cell_occupancy_consistency(prev, cur)
        if not occupancy_metric.get("pass"):
            return _metric(
                0.0,
                False,
                f"pattern transition invalid; {occupancy_metric.get('reason', '')}",
                status=occupancy_metric.get("status"),
                confidence=float(occupancy_metric.get("confidence", 0.85)),
                evidence=occupancy_metric.get("evidence"),
            ), []

        type_metric = _pipe_type_consistency(prev, cur)
        if not type_metric.get("pass"):
            return _metric(
                0.0,
                False,
                f"pattern transition changed pipe shape; {type_metric.get('reason', '')}",
                status=type_metric.get("status"),
                confidence=float(type_metric.get("confidence", 0.85)),
                evidence=type_metric.get("evidence"),
            ), []

        changed = _pipe_orientation_changes(prev, cur)
        if not changed:
            return _metric(
                1.0,
                True,
                "registered legal pattern is unchanged",
            ), []
        if len(changed) > 1:
            return _metric(
                0.0,
                False,
                f"{len(changed)} pipes changed legal pattern in one adjacent transition (at most one is allowed): {changed}",
                evidence={"changed_cells": changed},
            ), changed

        cell = int(changed[0])
        pm = {int(i): int(m) for i, m in (prev.get("masks") or {}).items()}
        cm = {int(i): int(m) for i, m in (cur.get("masks") or {}).items()}
        prev_mask = int(pm.get(cell, 0))
        cur_mask = int(cm.get(cell, 0))
        quarter_turns = _pipe_quarter_turn_distance(prev_mask, cur_mask)
        legal = quarter_turns is not None and quarter_turns <= 1
        evidence = {
            "changed_cells": changed,
            "cell": cell,
            "from_mask": prev_mask,
            "to_mask": cur_mask,
            "from_shape": _pipe_shape_of(prev_mask),
            "to_shape": _pipe_shape_of(cur_mask),
            "quarter_turn_distance": quarter_turns,
            "allowed_masks": _pipe_allowed_masks(gt, cell),
        }
        return _metric(
            1.0 if legal else 0.0,
            bool(legal),
            (
                "one pipe changed to an adjacent registered legal orientation"
                if legal else
                "pipe legal pattern jumped more than one quarter-turn or changed to an incompatible orientation"
            ),
            evidence=evidence,
        ), changed

    rotation_num_metric, changed = _pipe_rotation_num_valid(prev, cur)
    if not rotation_num_metric.get("pass"):
        return _metric(
            0.0,
            False,
            f"rotation count invalid; {rotation_num_metric.get('reason', '')}",
            status=rotation_num_metric.get("status"),
            confidence=float(rotation_num_metric.get("confidence", 0.85)),
            evidence=rotation_num_metric.get("evidence"),
        ), changed

    rotation_angle_metric = _pipe_rotation_angle_valid(
        prev, cur, prev_frame_bgr, cur_frame_bgr, geometry, changed
    )
    if not rotation_angle_metric.get("pass"):
        return _metric(
            0.0,
            False,
            f"rotation angle invalid; {rotation_angle_metric.get('reason', '')}",
            status=rotation_angle_metric.get("status"),
            confidence=float(rotation_angle_metric.get("confidence", 0.85)),
            evidence=rotation_angle_metric.get("evidence"),
        ), changed

    if changed:
        reason = "one pipe changed orientation and its rotation angle is valid"
    else:
        reason = "no pipe orientation changed; rotation count and angle bound are valid"
    confidence = min(
        float(rotation_num_metric.get("confidence", 0.85)),
        float(rotation_angle_metric.get("confidence", 0.85)),
    )
    evidence = {}
    if changed:
        evidence["changed_cells"] = changed
    if rotation_angle_metric.get("evidence"):
        evidence.update(rotation_angle_metric["evidence"])
    return _metric(
        1.0,
        True,
        reason,
        confidence=confidence,
        evidence=evidence or None,
    ), changed


def _pipe_dynamic_metrics(
    prev: Dict[str, object],
    cur: Dict[str, object],
    prev_frame_bgr: Optional[np.ndarray] = None,
    cur_frame_bgr: Optional[np.ndarray] = None,
    geometry: Optional["pipe_cv.PipeGeometry"] = None,
    gt: Optional["pipe_oracle.PipeGT"] = None,
) -> Tuple[Dict[str, Metric], List[int]]:
    rotation_metric, changed = _pipe_rotation_num_angle_valid(
        prev, cur, prev_frame_bgr, cur_frame_bgr, geometry, gt,
    )
    metrics = {
        "cell_occupancy_consistency": _pipe_cell_occupancy_consistency(prev, cur),
        "pipe_type_consistency": _pipe_type_consistency(prev, cur),
        "rotation_num_angle_valid": rotation_metric,
    }
    return metrics, changed


def _process_pipe_video(
    video_id: str,
    video_path: Path,
    level: str,
    difficulty: str,
    seed: int,
    output_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gt = pipe_oracle.load_pipe_gt(level, difficulty, seed, metadata=metadata)
    manifest, frames = extract_frames(video_path, video_id, "pipe", output_root, sampling_mode)
    geometry = pipe_cv.estimate_pipe_geometry(frames, gt.grid_size)
    parsed_frames = [pipe_cv.parse_pipe_frame(frame, gt.grid_size, geometry) for frame in frames]
    detections = [
        _pipe_detection_record(idx, str(frame_rec["path"]), parsed_frames[idx], gt)
        for idx, frame_rec in enumerate(manifest["frames"])
    ]
    detection_doc = {
        "video_id": video_id,
        "task_type": "pipe",
        "metadata": {
            "level": level,
            "difficulty": difficulty,
            "seed": seed,
            "grid_size": gt.grid_size,
            "source_index": gt.source_index,
            "active_indices": sorted(gt.active_indices),
            "shape_by_index": gt.shape_by_index,
            "grid_geometry": (
                {
                    "col_centers": list(geometry.col_centers),
                    "row_centers": list(geometry.row_centers),
                    "pitch": geometry.pitch,
                    "board_bbox_xyxy": list(geometry.board_bbox_xyxy),
                    "reference_frame_ids": list(geometry.reference_frame_ids),
                }
                if geometry is not None else None
            ),
        },
        "frames": detections,
    }
    _write_json(output_root / "detections" / f"{video_id}_per_frame_detection.json", detection_doc)

    static_overlay_dir, dynamic_overlay_dir = _overlay_dirs(PIPE_DETECTOR_NAME)
    for stale in itertools.chain(
        static_overlay_dir.glob(f"{video_id}_frame_*_static.png"),
        dynamic_overlay_dir.glob(f"{video_id}_transition_*_dynamic.png"),
    ):
        stale.unlink()

    final_goal_metric = (
        _pipe_final_solution_valid(detections[-1], gt, True) if detections else {}
    )
    auto_goal_achieved = _auto_goal_achieved(detections, final_goal_metric)

    last_idx = len(detections) - 1
    static_frame_ids = sample_indices(len(detections), 4)
    static_frames = []
    for idx in static_frame_ids:
        det = detections[idx]
        metrics = _pipe_static_metrics(det, gt, is_final_frame=idx == last_idx)
        static_score = _metrics_score(metrics)
        static_confidence = _static_confidence(metrics, det)
        static_pass = _metrics_pass(metrics)
        static_frames.append(
            {"frame_id": idx, "metrics": metrics, "static_score": static_score, "static_confidence": static_confidence}
        )
        if (not overlay_failures_only) or (not static_pass):
            suffix = " static fail" if not static_pass else ""
            overlay = pipe_cv.draw_pipe_overlay(
                frames[idx], parsed_frames[idx], geometry,
                f"{video_id} frame {idx:03d} {_metrics_overlay_label('static', static_score, static_confidence, static_pass, auto_goal_achieved)}{suffix}",
                source_index=gt.source_index,
            )
            cv2.imwrite(str(static_overlay_dir / f"{video_id}_frame_{idx:03d}_static.png"), overlay)

    static_doc = {"video_id": video_id, "task_type": "pipe", "frames": static_frames}
    _write_json(output_root / "static_metrics" / f"{video_id}_static_metrics.json", static_doc)

    dynamic_transitions = []
    pattern_states = [_pipe_pattern_state(det, gt) for det in detections]
    for idx in range(max(0, len(detections) - 1)):
        metrics, changed = _pipe_dynamic_metrics(
            detections[idx], detections[idx + 1], frames[idx], frames[idx + 1], geometry, gt
        )
        dynamic_score = _metrics_score(metrics)
        dynamic_confidence = _mean(
            (
                _metrics_confidence(metrics),
                detections[idx].get("confidence"),
                detections[idx + 1].get("confidence"),
            )
        )
        dynamic_pass = _metrics_pass(metrics)
        dynamic_transitions.append(
            {
                "from_frame_id": idx,
                "to_frame_id": idx + 1,
                "transition_kind": "adjacent_sampled_frames",
                "changed_cells": changed,
                "metrics": metrics,
                "dynamic_score": dynamic_score,
                "dynamic_confidence": dynamic_confidence,
            }
        )
        if (not overlay_failures_only) or (not dynamic_pass):
            overlay = pipe_cv.draw_pipe_transition_overlay(
                frames[idx], frames[idx + 1], parsed_frames[idx], parsed_frames[idx + 1], geometry,
                f"{video_id} {idx:03d}->{idx + 1:03d} {_metrics_overlay_label('dynamic', dynamic_score, dynamic_confidence, dynamic_pass, auto_goal_achieved)}",
                source_index=gt.source_index, changed_indices=changed,
            )
            cv2.imwrite(
                str(dynamic_overlay_dir / f"{video_id}_transition_{idx:03d}_{idx + 1:03d}_dynamic.png"),
                overlay,
            )

    dynamic_doc = {
        "video_id": video_id,
        "task_type": "pipe",
        "temporal_mode": "adjacent_sampled_frames",
        "pattern_states": pattern_states,
        "transitions": dynamic_transitions,
    }
    _write_json(output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json", dynamic_doc)
    return _write_video_report(video_id, "pipe", manifest, static_doc, dynamic_doc, output_root)


# =====================================================================================
# enclosure_chat_noir ("chat noir") — trap-the-cat hex-board puzzle.
#
# Discrete state is (cat_cell, blocked_set) on a static hex board.  We parse each frame with
# topo_spike.chat_noir_cv against a clip-level lattice, then score against env-oracle GT
# (topo_spike.chat_noir_oracle).  Static metrics look at 4 checkpoint frames {0,13,27,40};
# dynamic metrics look at all 40 adjacent transitions, per reports/findings_chat_noir.md.
# =====================================================================================

_CHAT_NOIR_NAME_RE = re.compile(r"radius_(\d+).*?blocked_(\d+).*?policy_(easy|medium|hard)")


def _chat_noir_videos(limit: Optional[int], video_dir: Optional[Path]) -> List[Tuple[str, Path, int, int, str, int]]:
    """Discover imagined mp4s + (video_id, mp4, radius, blocked, policy, seed)."""
    root = video_dir or CHAT_NOIR_DEFAULT_DIR
    seed_re = re.compile(r"seed_(\d+)")
    out: List[Tuple[str, Path, int, int, str, int]] = []
    for mp4 in sorted(root.rglob("*imagined.mp4")):
        m = _CHAT_NOIR_NAME_RE.search(mp4.name)
        seeds = seed_re.findall(mp4.name)
        if not m or not seeds:
            continue
        radius = int(m.group(1))
        blocked = int(m.group(2))
        policy = m.group(3)
        seed = int(seeds[-1])
        video_id = _safe_video_id(f"radius_{radius}_blocked_{blocked}_policy_{policy}_seed_{seed:03d}")
        out.append((video_id, mp4, radius, blocked, policy, seed))
    out.sort(key=lambda t: t[0])
    return out[:limit] if limit is not None else out


def _chat_noir_detection_record(
    idx: int,
    frame_path: str,
    parsed: "chat_noir_cv.ParsedChatNoirFrame",
    geometry: Optional["chat_noir_cv.ChatNoirGeometry"],
    gt: "chat_noir_oracle.ChatNoirGT",
    number_match: Optional[Tuple[List[int], List[int], Dict[int, float]]] = None,
) -> Dict[str, object]:
    cells = [
        {
            "index": cell.index,
            "center_xy": [round(cell.center_xy[0], 1), round(cell.center_xy[1], 1)],
            "state": cell.state,
            "dark_frac": round(cell.dark_frac, 3),
            "orange_frac": round(cell.orange_frac, 3),
            "confidence": round(cell.confidence, 3),
        }
        for cell in parsed.cells
    ]
    confs = [cell.confidence for cell in parsed.cells]
    raw_conf = (sum(confs) / len(confs)) if confs else 0.0
    status_cap = {"ok": 1.0, "cat_not_found": 0.4, "grid_not_found": 0.0}.get(parsed.parser_status, 0.4)
    confidence = min(raw_conf, status_cap)
    number_matched, number_mismatched, number_scores = number_match or ([], [], {})
    return {
        "frame_id": idx,
        "path": frame_path,
        "task_type": "chat_noir",
        "parser_status": parsed.parser_status,
        "cat_index": parsed.cat_index,
        "cat_center_xy": (
            [round(parsed.cat_center_xy[0], 1), round(parsed.cat_center_xy[1], 1)]
            if parsed.cat_center_xy is not None else None
        ),
        "cat_count": parsed.cat_count,
        "blocked": sorted(parsed.blocked),
        "n_blocked": len(parsed.blocked),
        "n_cells": gt.cell_count,
        "board_residual": (geometry.residual if geometry is not None else None),
        "board_pitch": (geometry.pitch if geometry is not None else None),
        "cells": cells,
        "confidence": confidence,
        "warnings": list(parsed.warnings),
        "number_matched_indices": number_matched,
        "number_mismatched_indices": number_mismatched,
        "number_match_scores": {str(i): round(score, 3) for i, score in number_scores.items()},
    }


def _chat_noir_initial_blocker_valid(
    record: Dict[str, object], gt: "chat_noir_oracle.ChatNoirGT"
) -> Metric:
    if record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "board unreadable — blocked set not verifiable", status="low_confidence")
    blocked = set(int(i) for i in record.get("blocked") or [])
    expected_blocked = set(gt.initial_blocked_indices)
    missing = sorted(expected_blocked - blocked)
    added = sorted(blocked - expected_blocked)
    passed = not missing
    return _metric(
        1.0 if passed else 0.0,
        passed,
        "all environment-initial blocked cells are still present"
        if passed else f"environment-initial blocked cell(s) missing: {missing}",
        evidence={"missing_initial_blocked": missing, "subsequently_added_blocked": added},
    )


def _chat_noir_number_layout_match(record: Dict[str, object], gt: "chat_noir_oracle.ChatNoirGT") -> Metric:
    if record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "board unreadable — number set not verifiable", status="low_confidence")
    mismatched = [int(i) for i in record.get("number_mismatched_indices") or []]
    matched = [int(i) for i in record.get("number_matched_indices") or []]
    passed = not mismatched and len(matched) == gt.cell_count
    return _metric(
        1.0 if passed else 0.0,
        passed,
        f"all {gt.cell_count} fixed cell numbers are complete and unchanged"
        if passed else f"cell number(s) missing or altered: {mismatched}",
        evidence={"matched_count": len(matched), "expected_count": gt.cell_count, "mismatched": mismatched},
    )


def _chat_noir_final_solution_valid(record: Dict[str, object], gt: "chat_noir_oracle.ChatNoirGT", is_final_frame: bool) -> Metric:
    if not is_final_frame:
        return _metric(0.0, True, "final-state check only applies to the last frame", status="not_applicable")
    cat_index = record.get("cat_index")
    if cat_index is None or record.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "final frame unreadable — cannot confirm the cat is trapped", status="low_confidence")
    blocked = set(int(i) for i in record.get("blocked") or [])
    trapped = chat_noir_oracle.is_trapped(gt.board, int(cat_index), blocked)
    return _metric(
        1.0 if trapped else 0.0, trapped,
        "cat is trapped: no open path from the cat cell reaches the boundary (puzzle won)" if trapped
        else "cat still has an open escape path to the boundary (not trapped)",
        evidence={"cat_index": int(cat_index), "n_blocked": len(blocked)},
    )


def _chat_noir_static_metrics(
    record: Dict[str, object], gt: "chat_noir_oracle.ChatNoirGT", *, is_initial_frame: bool, is_final_frame: bool
) -> Dict[str, Metric]:
    return {
        "initial_blocker_valid": _chat_noir_initial_blocker_valid(record, gt),
        "number_layout_match": _chat_noir_number_layout_match(record, gt),
        "final_solution_valid": _chat_noir_final_solution_valid(record, gt, is_final_frame),
    }


def _chat_noir_cat_move_valid(
    prev: Dict[str, object], cur: Dict[str, object], gt: "chat_noir_oracle.ChatNoirGT"
) -> Tuple[Metric, Tuple[Optional[int], Optional[int]]]:
    pc = prev.get("cat_index")
    cc = cur.get("cat_index")
    if pc is None or cc is None:
        return _metric(0.0, False, "a frame's cat is unreadable — move legality not verifiable", status="low_confidence"), (pc, cc)
    pc, cc = int(pc), int(cc)
    if pc == cc:
        return _metric(1.0, True, f"cat stayed on cell {cc}"), (pc, cc)
    cur_blocked = set(int(i) for i in cur.get("blocked") or [])
    adjacent = chat_noir_oracle.are_adjacent(gt.board, pc, cc)
    if adjacent and cc not in cur_blocked:
        return _metric(1.0, True, f"cat moved to adjacent open cell {pc}->{cc}"), (pc, cc)
    reason = (
        f"cat moved onto a blocked cell {cc}" if cc in cur_blocked
        else f"cat jumped between non-adjacent cells {pc}->{cc}"
    )
    return _metric(0.0, False, reason, evidence={"from": pc, "to": cc}), (pc, cc)


def _chat_noir_cat_motion_smoothness(prev: Dict[str, object], cur: Dict[str, object]) -> Metric:
    pxy, cxy = prev.get("cat_center_xy"), cur.get("cat_center_xy")
    pitches = [float(v) for v in (prev.get("board_pitch"), cur.get("board_pitch")) if v is not None]
    if pxy is None or cxy is None or not pitches:
        return _metric(
            0.0, True, "cat centre unreadable — pixel-motion check not applicable",
            status="not_applicable",
        )
    pitch = sum(pitches) / len(pitches)
    distance = math.hypot(float(cxy[0]) - float(pxy[0]), float(cxy[1]) - float(pxy[1]))
    normalized = distance / pitch if pitch > 0 else float("inf")
    passed = normalized <= 0.5
    return _metric(
        1.0 if passed else 0.0,
        passed,
        f"cat visual displacement is {normalized:.3f} cell pitches (limit 0.500)"
        if passed else f"cat visually jumped {normalized:.3f} cell pitches (> 0.500)",
        evidence={"distance_px": round(distance, 3), "board_pitch_px": round(pitch, 3), "distance_in_cells": round(normalized, 4)},
    )


def _chat_noir_blocker_update_valid(
    prev: Dict[str, object], cur: Dict[str, object]
) -> Tuple[Metric, List[int]]:
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(0.0, False, "a frame is unreadable — blocker growth not verifiable", status="low_confidence"), []
    pb = set(int(i) for i in prev.get("blocked") or [])
    cb = set(int(i) for i in cur.get("blocked") or [])
    added = sorted(cb - pb)
    removed = sorted(pb - cb)
    if removed:
        return _metric(
            0.0, False,
            f"blocked cell(s) reverted to open: {removed}",
            evidence={"reverted": removed},
        ), added
    if len(added) > 1:
        return _metric(
            0.0, False,
            f"{len(added)} new blocked cells in one sampled visual transition (expected at most one): {added}",
            evidence={"added": added},
        ), added
    prev_cat = prev.get("cat_index")
    if added and prev_cat is not None and added[0] == int(prev_cat):
        return _metric(0.0, False, f"new block placed on the cat cell {added[0]}", evidence={"added": added}), added
    if added:
        return _metric(1.0, True, f"exactly one new blocked cell {added[0]}, monotonic growth"), added
    return _metric(1.0, True, "blocked set unchanged (monotonic)"), added


def _chat_noir_grid_consistency(
    prev: Dict[str, object],
    cur: Dict[str, object],
    gt: "chat_noir_oracle.ChatNoirGT",
    number_match: Optional[Tuple[List[int], List[int], Dict[int, float]]] = None,
) -> Metric:
    """Check visual continuity of blocker count and printed cell numbers.

    A legal blocker addition may change the count by one, so only a decrease or a
    magnitude larger than one is a count jump.  Blocker identity and placement
    legality remain the responsibility of ``blocker_update_valid``.
    """
    if prev.get("parser_status") == "grid_not_found" or cur.get("parser_status") == "grid_not_found":
        return _metric(
            0.0,
            False,
            "a frame is unreadable — blocker/number consistency not verifiable",
            status="low_confidence",
        )

    prev_count = int(prev.get("n_blocked", len(prev.get("blocked") or [])))
    cur_count = int(cur.get("n_blocked", len(cur.get("blocked") or [])))
    count_delta = cur_count - prev_count
    count_consistent = count_delta in (0, 1)

    if number_match is None:
        # Fallback for callers that only have serialized detections.  Production
        # evaluation supplies a direct adjacent-frame glyph comparison below.
        prev_mismatched = set(int(i) for i in prev.get("number_mismatched_indices") or [])
        cur_mismatched = set(int(i) for i in cur.get("number_mismatched_indices") or [])
        number_mismatched = sorted(prev_mismatched ^ cur_mismatched)
        number_consistent = not number_mismatched
        number_matched_count = gt.cell_count if number_consistent else gt.cell_count - len(number_mismatched)
        number_scores: Dict[int, float] = {}
    else:
        number_matched, number_mismatched, number_scores = number_match
        number_mismatched = [int(i) for i in number_mismatched]
        number_matched_count = len(number_matched)
        number_consistent = not number_mismatched and number_matched_count == gt.cell_count

    passed = count_consistent and number_consistent
    failures = []
    if not count_consistent:
        failures.append(f"blocker count jumped {prev_count}->{cur_count}")
    if not number_consistent:
        failures.append(f"printed number(s) changed at cell(s) {number_mismatched}")
    return _metric(
        1.0 if passed else 0.0,
        passed,
        "blocker count changed by at most one and all printed numbers stayed unchanged"
        if passed else "; ".join(failures),
        evidence={
            "previous_blocker_count": prev_count,
            "current_blocker_count": cur_count,
            "blocker_count_delta": count_delta,
            "number_matched_count": number_matched_count,
            "number_mismatched_indices": number_mismatched,
            "number_match_scores": {str(i): round(score, 3) for i, score in number_scores.items()},
        },
    )


def _chat_noir_dynamic_metrics(
    prev: Dict[str, object],
    cur: Dict[str, object],
    gt: "chat_noir_oracle.ChatNoirGT",
    number_match: Optional[Tuple[List[int], List[int], Dict[int, float]]] = None,
) -> Tuple[Dict[str, Metric], Tuple[Optional[int], Optional[int]], List[int]]:
    cat_metric, cat_move = _chat_noir_cat_move_valid(prev, cur, gt)
    blocker_metric, added = _chat_noir_blocker_update_valid(prev, cur)
    metrics = {
        "cat_move_valid": cat_metric,
        "cat_motion_smoothness": _chat_noir_cat_motion_smoothness(prev, cur),
        "blocker_update_valid": blocker_metric,
        "grid_consistency": _chat_noir_grid_consistency(prev, cur, gt, number_match),
    }
    return metrics, cat_move, added


def _process_chat_noir_video(
    video_id: str,
    video_path: Path,
    radius: int,
    blocked: int,
    policy: str,
    seed: int,
    output_root: Path,
    sampling_mode: str,
    overlay_failures_only: bool,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gt = chat_noir_oracle.load_chat_noir_gt(
        radius, blocked, policy, seed, metadata=metadata
    )
    manifest, frames = extract_frames(video_path, video_id, "chat_noir", output_root, sampling_mode)
    geometry = chat_noir_cv.estimate_chat_noir_geometry(frames, gt.radius)
    parsed_frames = [chat_noir_cv.parse_chat_noir_frame(frame, gt.radius, geometry) for frame in frames]
    number_matches = [chat_noir_cv.compare_number_set(frames[0], frame, geometry) for frame in frames]
    transition_number_matches = [
        chat_noir_cv.compare_number_set(prev, cur, geometry)
        for prev, cur in zip(frames, frames[1:])
    ]
    detections = [
        _chat_noir_detection_record(
            idx, str(frame_rec["path"]), parsed_frames[idx], geometry, gt, number_matches[idx]
        )
        for idx, frame_rec in enumerate(manifest["frames"])
    ]
    detection_doc = {
        "video_id": video_id,
        "task_type": "chat_noir",
        "metadata": {
            "radius": gt.radius,
            "policy": policy,
            "seed": seed,
            "cat_index": gt.cat_index,
            "initial_block_count": gt.initial_block_count,
            "initial_blocked_indices": list(gt.initial_blocked_indices),
            "cell_count": gt.cell_count,
            "grid_geometry": (
                {
                    "scale": geometry.scale,
                    "pitch": geometry.pitch,
                    "residual": geometry.residual,
                    "reference_frame_ids": list(geometry.reference_frame_ids),
                }
                if geometry is not None else None
            ),
        },
        "frames": detections,
    }
    _write_json(output_root / "detections" / f"{video_id}_per_frame_detection.json", detection_doc)

    static_overlay_dir, dynamic_overlay_dir = _overlay_dirs(CHAT_NOIR_DETECTOR_NAME)
    for stale in itertools.chain(
        static_overlay_dir.glob(f"{video_id}_frame_*_static.png"),
        dynamic_overlay_dir.glob(f"{video_id}_transition_*_dynamic.png"),
    ):
        stale.unlink()

    last_idx = len(detections) - 1
    static_frame_ids = _swap2d_static_frame_ids(len(detections), 4)
    static_frames = []
    for idx in static_frame_ids:
        det = detections[idx]
        metrics = _chat_noir_static_metrics(
            det, gt, is_initial_frame=idx == 0, is_final_frame=idx == last_idx
        )
        static_score = _metrics_score(metrics)
        static_confidence = _static_confidence(metrics, det)
        static_pass = _metrics_pass(metrics)
        static_frames.append(
            {"frame_id": idx, "metrics": metrics, "static_score": static_score, "static_confidence": static_confidence}
        )
        if (not overlay_failures_only) or (not static_pass):
            suffix = " static fail" if not static_pass else ""
            overlay = chat_noir_cv.draw_chat_noir_overlay(
                frames[idx], parsed_frames[idx], geometry,
                f"{video_id} frame {idx:03d} {_metrics_overlay_label('static', static_score, static_confidence, static_pass)}{suffix}",
            )
            cv2.imwrite(str(static_overlay_dir / f"{video_id}_frame_{idx:03d}_static.png"), overlay)

    static_doc = {"video_id": video_id, "task_type": "chat_noir", "frames": static_frames}
    _write_json(output_root / "static_metrics" / f"{video_id}_static_metrics.json", static_doc)

    dynamic_transitions = []
    for idx, (prev, cur) in enumerate(zip(detections, detections[1:])):
        metrics, cat_move, added = _chat_noir_dynamic_metrics(
            prev, cur, gt, transition_number_matches[idx]
        )
        dynamic_score = _metrics_score(metrics)
        dynamic_confidence = _dynamic_confidence(metrics, prev, cur)
        dynamic_pass = _metrics_pass(metrics)
        dynamic_transitions.append(
            {
                "from_frame_id": idx,
                "to_frame_id": idx + 1,
                "cat_move": list(cat_move),
                "new_blocked": added,
                "metrics": metrics,
                "dynamic_score": dynamic_score,
                "dynamic_confidence": dynamic_confidence,
            }
        )
        if (not overlay_failures_only) or (not dynamic_pass):
            overlay = chat_noir_cv.draw_chat_noir_transition_overlay(
                frames[idx], frames[idx + 1], parsed_frames[idx], parsed_frames[idx + 1], geometry,
                f"{video_id} {idx:03d}->{idx + 1:03d} {_metrics_overlay_label('dynamic', dynamic_score, dynamic_confidence, dynamic_pass)}",
                cat_move=cat_move, new_blocked=added,
            )
            cv2.imwrite(str(dynamic_overlay_dir / f"{video_id}_transition_{idx:03d}_{idx + 1:03d}_dynamic.png"), overlay)

    dynamic_doc = {"video_id": video_id, "task_type": "chat_noir", "transitions": dynamic_transitions}
    _write_json(output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json", dynamic_doc)
    return _write_video_report(video_id, "chat_noir", manifest, static_doc, dynamic_doc, output_root)


def _failed_items(static_doc: Dict[str, object], dynamic_doc: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    failed_frames: List[Dict[str, object]] = []
    failed_transitions: List[Dict[str, object]] = []
    for frame in static_doc.get("frames") or []:
        for name, metric in (frame.get("metrics") or {}).items():
            if not bool(metric.get("pass")) and metric.get("status") != "not_applicable":
                failed_frames.append(
                    {
                        "frame_id": frame.get("frame_id"),
                        "metric": name,
                        "confidence": metric.get("confidence"),
                        "reason": metric.get("reason"),
                    }
                )
    for tr in dynamic_doc.get("transitions") or []:
        for name, metric in (tr.get("metrics") or {}).items():
            if not bool(metric.get("pass")) and metric.get("status") != "not_applicable":
                failed_transitions.append(
                    {
                        "from_frame_id": tr.get("from_frame_id"),
                        "to_frame_id": tr.get("to_frame_id"),
                        "metric": name,
                        "confidence": metric.get("confidence"),
                        "reason": metric.get("reason"),
                    }
                )
    return failed_frames, failed_transitions


def _metric_breakdown(
    static_doc: Dict[str, object],
    dynamic_doc: Dict[str, object],
    value_fn=_metric_score,
) -> Dict[str, Optional[float]]:
    values: Dict[str, List[Optional[float]]] = defaultdict(list)
    for frame in static_doc.get("frames") or []:
        for name, metric in (frame.get("metrics") or {}).items():
            values[name].append(value_fn(metric))
    for tr in dynamic_doc.get("transitions") or []:
        for name, metric in (tr.get("metrics") or {}).items():
            values[name].append(value_fn(metric))
    return {name: _mean(xs) for name, xs in sorted(values.items())}


def _strict_rows_accuracy(rows: Sequence[Dict[str, object]]) -> Optional[float]:
    applicable = [
        metric
        for row in rows
        for metric in (row.get("metrics") or {}).values()
        if metric.get("status") != "not_applicable"
    ]
    if not applicable:
        return None
    return 1.0 if all(bool(metric.get("pass")) for metric in applicable) else 0.0


def _strict_all_metrics_accuracy(
    static_accuracy: Optional[float],
    dynamic_accuracy: Optional[float],
) -> Optional[float]:
    accuracies = [x for x in (static_accuracy, dynamic_accuracy) if x is not None]
    if not accuracies:
        return None
    return 1.0 if all(x == 1.0 for x in accuracies) else 0.0


def _write_video_report(
    video_id: str,
    task_type: str,
    manifest: Dict[str, object],
    static_doc: Dict[str, object],
    dynamic_doc: Dict[str, object],
    output_root: Path,
    outcome_doc: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    static_score = _mean(frame.get("static_score") for frame in static_doc.get("frames") or [])
    dynamic_score = _mean(tr.get("dynamic_score") for tr in dynamic_doc.get("transitions") or [])
    static_confidence = _mean(frame.get("static_confidence") for frame in static_doc.get("frames") or [])
    dynamic_confidence = _mean(tr.get("dynamic_confidence") for tr in dynamic_doc.get("transitions") or [])
    legacy_composite_score = _combine_static_dynamic(static_score, dynamic_score)
    legacy_composite_confidence = _combine_static_dynamic(static_confidence, dynamic_confidence)
    outcome_metrics = (outcome_doc or {}).get("metrics") or {}
    process_metrics = (dynamic_doc.get("process_summary") or {}).get("metrics") or {}
    process_score = _metrics_score(process_metrics) if process_metrics else None
    process_confidence = _metrics_confidence(process_metrics) if process_metrics else None
    task_success_metric = (
        outcome_metrics.get("final_solution_valid")
        or outcome_metrics.get("goal_reached")  # backward compatibility for existing JSON
        or {}
    )
    if task_type == "swap2d" and task_success_metric:
        final_score = float(task_success_metric.get("score") or 0.0)
        final_confidence = _metric_confidence(task_success_metric)
    else:
        final_score = legacy_composite_score
        final_confidence = legacy_composite_confidence
    static_video_accuracy = _strict_rows_accuracy(static_doc.get("frames") or [])
    dynamic_video_accuracy = _strict_rows_accuracy(dynamic_doc.get("transitions") or [])
    strict_video_accuracy = dynamic_video_accuracy
    strict_all_metrics_accuracy = _strict_all_metrics_accuracy(static_video_accuracy, dynamic_video_accuracy)
    task_success = None
    task_success_feasible = None
    strict_valid_success = None
    if task_type == "swap2d" and task_success_metric:
        task_success = 1.0 if bool(task_success_metric.get("pass")) else 0.0
        budget_metric = outcome_metrics.get("budget_feasibility") or {}
        budget_feasible = bool(budget_metric.get("pass"))
        task_success_feasible = task_success if budget_feasible else None
        initial_ok = bool((outcome_metrics.get("initial_state_match") or {}).get("pass"))
        grid_consistency_rows = [
            (frame.get("metrics") or {}).get("grid_consistency")
            for frame in static_doc.get("frames") or []
            if "grid_consistency" in (frame.get("metrics") or {})
        ]
        grid_consistency_rows = [
            metric for metric in grid_consistency_rows
            if metric and metric.get("status") != "not_applicable"
        ]
        grid_consistency_ok = bool(grid_consistency_rows) and all(
            bool(metric.get("pass")) for metric in grid_consistency_rows
        )
        strict_valid_success = 1.0 if (
            task_success == 1.0
            and initial_ok
            and grid_consistency_ok
            and dynamic_video_accuracy == 1.0
        ) else 0.0
        strict_all_metrics_accuracy = strict_valid_success

    failed_frames, failed_transitions = _failed_items(static_doc, dynamic_doc)
    failed_outcomes = [
        {
            "metric": name,
            "confidence": _metric_confidence(metric),
            "reason": metric.get("reason"),
        }
        for name, metric in outcome_metrics.items()
        if not bool(metric.get("pass")) and metric.get("status") != "not_applicable"
    ]
    report = {
        "video_id": video_id,
        "task_type": task_type,
        "num_frames": manifest.get("actual_num_frames"),
        "num_transitions": len(dynamic_doc.get("transitions") or []),
        "num_key_states": (
            len(dynamic_doc.get("key_states") or []) if task_type == "swap2d" else None
        ),
        "temporal_mode": dynamic_doc.get("temporal_mode"),
        "static_score": static_score,
        "static_confidence": static_confidence,
        "dynamic_score": dynamic_score,
        "dynamic_confidence": dynamic_confidence,
        "process_score": process_score,
        "process_confidence": process_confidence,
        "legacy_composite_score": legacy_composite_score,
        "legacy_composite_confidence": legacy_composite_confidence,
        "final_score": final_score,
        "final_confidence": final_confidence,
        "static_video_accuracy": static_video_accuracy,
        "dynamic_video_accuracy": dynamic_video_accuracy,
        "strict_video_accuracy": strict_video_accuracy,
        "strict_all_metrics_accuracy": strict_all_metrics_accuracy,
        "task_success": task_success,
        "task_success_feasible": task_success_feasible,
        "strict_valid_success": strict_valid_success,
        "outcome_metric_breakdown": {
            name: _metric_score(metric) for name, metric in sorted(outcome_metrics.items())
        },
        "outcome_confidence_breakdown": {
            name: _metric_confidence(metric) for name, metric in sorted(outcome_metrics.items())
        },
        "process_metric_breakdown": {
            name: _metric_score(metric) for name, metric in sorted(process_metrics.items())
        },
        "metric_breakdown": _metric_breakdown(static_doc, dynamic_doc),
        "metric_confidence_breakdown": _metric_breakdown(static_doc, dynamic_doc, _metric_confidence),
        "failed_frames": failed_frames[:25],
        "failed_transitions": failed_transitions[:25],
        "failed_outcomes": failed_outcomes,
        "low_confidence_frames": [
            {
                "frame_id": frame.get("frame_id"),
                "metric": name,
                "confidence": metric.get("confidence"),
                "reason": metric.get("reason"),
            }
            for frame in static_doc.get("frames") or []
            for name, metric in (frame.get("metrics") or {}).items()
            if metric.get("status") == "low_confidence"
        ][:25],
        "low_confidence_transitions": [
            {
                "from_frame_id": tr.get("from_frame_id"),
                "to_frame_id": tr.get("to_frame_id"),
                "metric": name,
                "confidence": metric.get("confidence"),
                "reason": metric.get("reason"),
            }
            for tr in dynamic_doc.get("transitions") or []
            for name, metric in (tr.get("metrics") or {}).items()
            if metric.get("status") == "low_confidence"
        ][:25],
    }
    _write_json(output_root / "reports" / f"{video_id}_video_report.json", report)
    return report


def _append_summary(output_root: Path, reports: Sequence[Dict[str, object]]) -> None:
    path = output_root / "reports" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "task_type",
        "num_frames",
        "num_transitions",
        "num_key_states",
        "static_score",
        "static_confidence",
        "dynamic_score",
        "dynamic_confidence",
        "process_score",
        "process_confidence",
        "legacy_composite_score",
        "final_score",
        "final_confidence",
        "static_video_accuracy",
        "dynamic_video_accuracy",
        "strict_video_accuracy",
        "strict_all_metrics_accuracy",
        "task_success",
        "task_success_feasible",
        "strict_valid_success",
        "n_failed_frames",
        "n_failed_transitions",
        "n_failed_outcomes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "video_id": report.get("video_id"),
                    "task_type": report.get("task_type"),
                    "num_frames": report.get("num_frames"),
                    "num_transitions": report.get("num_transitions"),
                    "num_key_states": report.get("num_key_states"),
                    "static_score": report.get("static_score"),
                    "static_confidence": report.get("static_confidence"),
                    "dynamic_score": report.get("dynamic_score"),
                    "dynamic_confidence": report.get("dynamic_confidence"),
                    "process_score": report.get("process_score"),
                    "process_confidence": report.get("process_confidence"),
                    "legacy_composite_score": report.get("legacy_composite_score"),
                    "final_score": report.get("final_score"),
                    "final_confidence": report.get("final_confidence"),
                    "static_video_accuracy": report.get("static_video_accuracy"),
                    "dynamic_video_accuracy": report.get("dynamic_video_accuracy"),
                    "strict_video_accuracy": report.get("strict_video_accuracy"),
                    "strict_all_metrics_accuracy": report.get("strict_all_metrics_accuracy"),
                    "task_success": report.get("task_success"),
                    "task_success_feasible": report.get("task_success_feasible"),
                    "strict_valid_success": report.get("strict_valid_success"),
                    "n_failed_frames": len(report.get("failed_frames") or []),
                    "n_failed_transitions": len(report.get("failed_transitions") or []),
                    "n_failed_outcomes": len(report.get("failed_outcomes") or []),
                }
            )


def _metric_summary_bucket() -> Dict[str, object]:
    return {
        "seen_count": 0,
        "applicable_count": 0,
        "score_sum": 0.0,
        "pass_count": 0,
        "fail_count": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "low_confidence_count": 0,
        "not_applicable_count": 0,
    }


def _update_metric_summary(bucket: Dict[str, object], metric: Metric) -> None:
    bucket["seen_count"] = int(bucket["seen_count"]) + 1
    if metric.get("status") == "not_applicable":
        bucket["not_applicable_count"] = int(bucket["not_applicable_count"]) + 1
        return

    score = _metric_score(metric)
    if score is None:
        return
    bucket["applicable_count"] = int(bucket["applicable_count"]) + 1
    bucket["score_sum"] = float(bucket["score_sum"]) + score
    if bool(metric.get("pass")):
        bucket["pass_count"] = int(bucket["pass_count"]) + 1
    else:
        bucket["fail_count"] = int(bucket["fail_count"]) + 1
    if metric.get("status") == "low_confidence":
        bucket["low_confidence_count"] = int(bucket["low_confidence_count"]) + 1

    confidence = _metric_confidence(metric)
    if confidence is not None:
        bucket["confidence_sum"] = float(bucket["confidence_sum"]) + confidence
        bucket["confidence_count"] = int(bucket["confidence_count"]) + 1


def _finalize_metric_summary(bucket: Dict[str, object]) -> Dict[str, object]:
    applicable = int(bucket["applicable_count"])
    confidence_count = int(bucket["confidence_count"])
    return {
        "seen_count": int(bucket["seen_count"]),
        "applicable_count": applicable,
        "not_applicable_count": int(bucket["not_applicable_count"]),
        "mean_score": (float(bucket["score_sum"]) / applicable) if applicable else None,
        "pass_rate": (int(bucket["pass_count"]) / applicable) if applicable else None,
        "fail_rate": (int(bucket["fail_count"]) / applicable) if applicable else None,
        "mean_confidence": (float(bucket["confidence_sum"]) / confidence_count) if confidence_count else None,
        "low_confidence_rate": (int(bucket["low_confidence_count"]) / applicable) if applicable else None,
    }


def _finalize_metric_summaries(
    buckets: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    return {
        scope: {
            name: _finalize_metric_summary(bucket)
            for name, bucket in sorted(metrics.items())
        }
        for scope, metrics in sorted(buckets.items())
    }


def _report_score_summary(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video_count": len(reports),
        "mean_static_score": _mean(r.get("static_score") for r in reports),
        "mean_dynamic_score": _mean(r.get("dynamic_score") for r in reports),
        "mean_process_score": _mean(r.get("process_score") for r in reports),
        "mean_final_score": _mean(r.get("final_score") for r in reports),
        "mean_final_confidence": _mean(r.get("final_confidence") for r in reports),
        "mean_strict_video_accuracy": _mean(r.get("strict_video_accuracy") for r in reports),
        "mean_strict_all_metrics_accuracy": _mean(r.get("strict_all_metrics_accuracy") for r in reports),
        "mean_task_success": _mean(r.get("task_success") for r in reports),
        "mean_task_success_feasible": _mean(r.get("task_success_feasible") for r in reports),
        "mean_strict_valid_success": _mean(r.get("strict_valid_success") for r in reports),
    }


def _write_metric_summary(output_root: Path, reports: Sequence[Dict[str, object]]) -> None:
    bucket_by_task: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(_metric_summary_bucket))
    )
    reports_by_task: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for report in reports:
        task_type = str(report.get("task_type"))
        video_id = str(report.get("video_id"))
        reports_by_task[task_type].append(report)

        static_path = output_root / "static_metrics" / f"{video_id}_static_metrics.json"
        dynamic_path = output_root / "dynamic_metrics" / f"{video_id}_dynamic_metrics.json"
        outcome_path = output_root / "outcome_metrics" / f"{video_id}_outcome_metrics.json"
        static_doc = json.loads(static_path.read_text()) if static_path.exists() else {"frames": []}
        dynamic_doc = json.loads(dynamic_path.read_text()) if dynamic_path.exists() else {"transitions": []}
        outcome_doc = json.loads(outcome_path.read_text()) if outcome_path.exists() else {"metrics": {}}

        for scope, rows in (
            ("static", static_doc.get("frames") or []),
            ("dynamic", dynamic_doc.get("transitions") or []),
            ("process", [{"metrics": (dynamic_doc.get("process_summary") or {}).get("metrics") or {}}]),
            ("outcome", [{"metrics": outcome_doc.get("metrics") or {}}]),
        ):
            for row in rows:
                for name, metric in (row.get("metrics") or {}).items():
                    for task_key in ("all", task_type):
                        bucket = bucket_by_task[task_key][scope][name]
                        _update_metric_summary(bucket, metric)

    doc = {
        "definition": {
            "mean_score": "Average metric.score over applicable frame/transition observations.",
            "pass_rate": "Fraction of applicable observations whose metric pass field is true.",
            "mean_confidence": "Average verifier-confidence over applicable observations.",
            "mean_strict_video_accuracy": "Video-level hard gate: any failed applicable dynamic metric makes the video 0.",
            "mean_strict_all_metrics_accuracy": "Stricter video-level gate: any failed static or dynamic metric makes the video 0.",
            "not_applicable": "Observations with status=not_applicable are counted but excluded from rates.",
        },
        "overall": {
            "video_scores": _report_score_summary(reports),
            "metrics": _finalize_metric_summaries(bucket_by_task["all"]),
        },
        "by_task": {
            task: {
                "video_scores": _report_score_summary(task_reports),
                "metrics": _finalize_metric_summaries(bucket_by_task[task]),
            }
            for task, task_reports in sorted(reports_by_task.items())
        },
    }
    _write_json(output_root / "reports" / "metric_summary.json", doc)


def _write_manual_check_list(output_root: Path, reports: Sequence[Dict[str, object]]) -> None:
    path = output_root / "manual_checks" / "manual_check_list.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "task_type",
        "frame_id",
        "from_frame_id",
        "to_frame_id",
        "metric",
        "auto_score",
        "auto_confidence",
        "auto_reason",
        "human_correct",
        "note",
    ]
    rows = []
    for report in reports:
        static_path = output_root / "static_metrics" / f"{report['video_id']}_static_metrics.json"
        dynamic_path = output_root / "dynamic_metrics" / f"{report['video_id']}_dynamic_metrics.json"
        outcome_path = output_root / "outcome_metrics" / f"{report['video_id']}_outcome_metrics.json"
        static_doc = json.loads(static_path.read_text()) if static_path.exists() else {"frames": []}
        dynamic_doc = json.loads(dynamic_path.read_text()) if dynamic_path.exists() else {"transitions": []}
        outcome_doc = json.loads(outcome_path.read_text()) if outcome_path.exists() else {"metrics": {}}
        for name, metric in (outcome_doc.get("metrics") or {}).items():
            if not bool(metric.get("pass")) and metric.get("status") != "not_applicable":
                rows.append(
                    {
                        "video_id": report["video_id"],
                        "task_type": report["task_type"],
                        "frame_id": "",
                        "from_frame_id": "",
                        "to_frame_id": "",
                        "metric": name,
                        "auto_score": metric.get("score"),
                        "auto_confidence": metric.get("confidence"),
                        "auto_reason": metric.get("reason"),
                        "human_correct": "",
                        "note": "task outcome",
                    }
                )
        for frame in static_doc.get("frames") or []:
            for name, metric in (frame.get("metrics") or {}).items():
                if not bool(metric.get("pass")) and metric.get("status") != "not_applicable":
                    rows.append(
                        {
                            "video_id": report["video_id"],
                            "task_type": report["task_type"],
                            "frame_id": frame.get("frame_id"),
                            "from_frame_id": "",
                            "to_frame_id": "",
                            "metric": name,
                            "auto_score": metric.get("score"),
                            "auto_confidence": metric.get("confidence"),
                            "auto_reason": metric.get("reason"),
                            "human_correct": "",
                            "note": "",
                        }
                    )
        for tr in dynamic_doc.get("transitions") or []:
            for name, metric in (tr.get("metrics") or {}).items():
                if not bool(metric.get("pass")) and metric.get("status") != "not_applicable":
                    rows.append(
                        {
                            "video_id": report["video_id"],
                            "task_type": report["task_type"],
                            "frame_id": "",
                            "from_frame_id": tr.get("from_frame_id"),
                            "to_frame_id": tr.get("to_frame_id"),
                            "metric": name,
                            "auto_score": metric.get("score"),
                            "auto_confidence": metric.get("confidence"),
                            "auto_reason": metric.get("reason"),
                            "human_correct": "",
                            "note": "",
                        }
                    )
    rows.sort(key=lambda r: (r["task_type"], r["video_id"], str(r["frame_id"]), str(r["from_frame_id"]), r["metric"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_root: Path, reports: Sequence[Dict[str, object]]) -> None:
    by_task = Counter(str(r.get("task_type")) for r in reports)
    lines = [
        "# Video CV Eval Outputs",
        "",
        "Generated by `video_eval_tasks/cv_backend/scripts/video_cv_pipeline.py`.",
        "",
        "Artifacts follow the execution plan:",
        "",
        "- `frames/`: sampled frame PNGs.",
        "- `manifests/`: one manifest per video, with source frame indices and timestamps.",
        "- `detections/`: per-frame structured detector output.",
        "- `static_metrics/`: per-frame static verifier output.",
        "- `dynamic_metrics/`: task-specific temporal verifier output (adjacent sampled frames for video tasks; stable key states for Swap2D).",
        "- `outcome_metrics/`: task-level outcome predicates such as Swap2D terminal goal success.",
        "- `reports/`: per-video reports plus `summary.csv` and `metric_summary.json`.",
        "- `manual_checks/manual_check_list.csv`: failed frame/transition queue for human review.",
        "",
        "Debug overlays are written to:",
        "",
        f"- `{_rel(ABLATION_OVERLAY_DIR / UNTANGLE_DETECTOR_NAME / 'static')}`",
        f"- `{_rel(ABLATION_OVERLAY_DIR / UNTANGLE_DETECTOR_NAME / 'dynamic')}`",
        f"- `{_rel(ABLATION_OVERLAY_DIR / ONE_STROKE_DETECTOR_NAME / 'static')}`",
        f"- `{_rel(ABLATION_OVERLAY_DIR / ONE_STROKE_DETECTOR_NAME / 'dynamic')}`",
        "",
        "Current detector caveats:",
        "",
        "- Untangle uses the color-segmentation parser and exposes endpoint/crossing evidence; dynamic scoring checks endpoint tracking plus topology flicker when endpoints are stationary or the snapped endpoint signature is unchanged.",
        "- One Stroke uses the existing cell/stroke/marker detector plus a homography-based board-edge snapper. A geometric arrowhead detector records direction and excludes the arrowhead from legal-edge snapping so the head is not fit as a neighboring grid edge.",
        "- One Stroke final-frame static scoring exposes both the full solver win condition and the narrower different-color separation condition.",
        "- Swap2D final_score is the exact terminal-goal predicate; process validity, progress, efficiency lower bound, coverage, and budget feasibility are separate fields.",
        "- Pipe dynamic scoring compares all adjacent sampled frames, while retaining per-frame registered-pattern states as evidence for parser and shape failures.",
        "- Confidence estimates how trustworthy the automatic verifier verdict is; it is a heuristic triage signal, not a calibrated probability.",
        "- Strict video accuracy is a hard video-level gate: any failed applicable dynamic metric makes the whole video score 0.",
        "",
        "Run summary:",
        "",
    ]
    for task, count in sorted(by_task.items()):
        task_reports = [r for r in reports if r.get("task_type") == task]
        if task == "swap2d":
            lines.append(
                f"- `{task}`: {count} videos, raw task success={_fmt_score(_mean(r.get('task_success') for r in task_reports))}, "
                f"feasible-only success={_fmt_score(_mean(r.get('task_success_feasible') for r in task_reports))}, "
                f"strict valid success={_fmt_score(_mean(r.get('strict_valid_success') for r in task_reports))}, "
                f"mean process={_fmt_score(_mean(r.get('process_score') for r in task_reports))}"
            )
        else:
            lines.append(
                f"- `{task}`: {count} videos, mean static={_fmt_score(_mean(r.get('static_score') for r in task_reports))}, "
                f"mean dynamic={_fmt_score(_mean(r.get('dynamic_score') for r in task_reports))}, "
                f"mean final={_fmt_score(_mean(r.get('final_score') for r in task_reports))}, "
                f"mean confidence={_fmt_score(_mean(r.get('final_confidence') for r in task_reports))}, "
                f"strict video accuracy={_fmt_score(_mean(r.get('strict_video_accuracy') for r in task_reports))}"
            )
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def _slice_batch(items: Sequence, offset: int, limit: Optional[int]) -> Sequence:
    start = max(0, int(offset or 0))
    if limit is None:
        return items[start:]
    return items[start : start + max(0, int(limit))]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["untangle", "one_stroke", "swap2d", "pipe", "chat_noir", "all"],
        default="all",
    )
    parser.add_argument("--sampling-mode", choices=["sample_41", "all_frames"], default="sample_41")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N videos in each selected task.")
    parser.add_argument("--limit", type=int, default=None, help="Per-task limit for smoke runs.")
    parser.add_argument("--untangle-video-dir", type=Path, default=None)
    parser.add_argument("--one-stroke-video-dir", type=Path, default=None)
    parser.add_argument("--swap2d-video-dir", type=Path, default=None)
    parser.add_argument("--pipe-video-dir", type=Path, default=None)
    parser.add_argument("--chat-noir-video-dir", type=Path, default=None)
    parser.add_argument(
        "--overlay-failures-only",
        action="store_true",
        help="Write overlays only for static/dynamic failures instead of every sampled frame.",
    )
    parser.add_argument(
        "--no-crossing-candidates",
        action="store_true",
        help="Skip Untangle pixel-level crossing candidate markers. Oracle crossing scores still run when endpoints resolve.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.ensure_dirs()
    args.output_root.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, object]] = []
    if args.task in {"untangle", "all"}:
        episodes = list(_slice_batch(_untangle_episodes(None, args.untangle_video_dir), args.offset, args.limit))
        print(f"Untangle: processing {len(episodes)} videos")
        for video_id, mp4, grid_size, colors, meta in episodes:
            report = _process_untangle_video(
                video_id,
                mp4,
                grid_size,
                colors,
                meta,
                args.output_root,
                args.sampling_mode,
                args.overlay_failures_only,
                not args.no_crossing_candidates,
            )
            reports.append(report)
            print(
                f"  {video_id}: static={_fmt_score(report.get('static_score'))} "
                f"dynamic={_fmt_score(report.get('dynamic_score'))} "
                f"final={_fmt_score(report.get('final_score'))} "
                f"conf={_fmt_score(report.get('final_confidence'))}",
                flush=True,
            )

    if args.task in {"one_stroke", "all"}:
        videos = list(_slice_batch(_one_stroke_videos(None, args.one_stroke_video_dir), args.offset, args.limit))
        print(f"One Stroke: processing {len(videos)} videos")
        for video_id, mp4, size, seed in videos:
            report = _process_one_stroke_video(
                video_id,
                mp4,
                size,
                seed,
                args.output_root,
                args.sampling_mode,
                args.overlay_failures_only,
            )
            reports.append(report)
            print(
                f"  {video_id}: static={_fmt_score(report.get('static_score'))} "
                f"dynamic={_fmt_score(report.get('dynamic_score'))} "
                f"final={_fmt_score(report.get('final_score'))} "
                f"conf={_fmt_score(report.get('final_confidence'))}",
                flush=True,
            )

    if args.task in {"swap2d", "all"}:
        videos = list(_slice_batch(_swap2d_videos(None, args.swap2d_video_dir), args.offset, args.limit))
        print(f"2D Swap: processing {len(videos)} videos")
        for video_id, mp4, rows, cols, seed in videos:
            report = _process_swap2d_video(
                video_id,
                mp4,
                rows,
                cols,
                seed,
                args.output_root,
                args.sampling_mode,
                args.overlay_failures_only,
            )
            reports.append(report)
            print(
                f"  {video_id}: static={_fmt_score(report.get('static_score'))} "
                f"dynamic={_fmt_score(report.get('dynamic_score'))} "
                f"final={_fmt_score(report.get('final_score'))} "
                f"conf={_fmt_score(report.get('final_confidence'))}",
                flush=True,
            )

    if args.task in {"pipe", "all"}:
        videos = list(_slice_batch(_pipe_videos(None, args.pipe_video_dir), args.offset, args.limit))
        print(f"Pipe: processing {len(videos)} videos")
        for video_id, mp4, level, difficulty, seed in videos:
            report = _process_pipe_video(
                video_id,
                mp4,
                level,
                difficulty,
                seed,
                args.output_root,
                args.sampling_mode,
                args.overlay_failures_only,
            )
            reports.append(report)
            print(
                f"  {video_id}: static={_fmt_score(report.get('static_score'))} "
                f"dynamic={_fmt_score(report.get('dynamic_score'))} "
                f"overall={_fmt_score(report.get('strict_all_metrics_accuracy'))} "
                f"conf={_fmt_score(report.get('final_confidence'))}",
                flush=True,
            )

    if args.task in {"chat_noir", "all"}:
        videos = list(_slice_batch(_chat_noir_videos(None, args.chat_noir_video_dir), args.offset, args.limit))
        print(f"Chat Noir: processing {len(videos)} videos")
        for video_id, mp4, radius, blocked, policy, seed in videos:
            report = _process_chat_noir_video(
                video_id,
                mp4,
                radius,
                blocked,
                policy,
                seed,
                args.output_root,
                args.sampling_mode,
                args.overlay_failures_only,
            )
            reports.append(report)
            print(
                f"  {video_id}: static={_fmt_score(report.get('static_score'))} "
                f"dynamic={_fmt_score(report.get('dynamic_score'))} "
                f"overall={_fmt_score(report.get('strict_all_metrics_accuracy'))} "
                f"conf={_fmt_score(report.get('final_confidence'))}",
                flush=True,
            )

    _append_summary(args.output_root, reports)
    _write_metric_summary(args.output_root, reports)
    _write_manual_check_list(args.output_root, reports)
    _write_readme(args.output_root, reports)
    print(f"\nwrote video-level outputs to {args.output_root}")


if __name__ == "__main__":
    main()
