from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from vite_server import ViteFrontendServer


def _import_playwright():
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "generate_samples.py requires the `playwright` package. Install it with: pip install playwright"
        ) from exc
    return async_playwright


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_IMAGE_ROOT_NAME = "images"
DIFFICULTY_DIR = {"easy": "difficulty_1", "medium": "difficulty_2", "hard": "difficulty_3"}
DIFFICULTY_NUM = {"easy": 1, "medium": 2, "hard": 3}
SINGLE_SECTIONS = ("singles", "links", "multi_rope", "link_groups", "mixed")
LEGACY_SECTION_DIRS = SINGLE_SECTIONS + ("pairs", "other")
ROOT_OUTPUT_FILES = ("dataset_metadata.json", "summary.json", "question.jsonl")
T04_PALETTE_NAMES = ("red", "blue", "green", "brown", "white", "purple")
T04_ANSWER_PALETTE = (*T04_PALETTE_NAMES, "none")

from jsonl_export import reorganize_images_by_question_id, to_relative_path, write_jsonl  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the knots_static dataset through the frontend batch API."
    )
    parser.add_argument("--seed", default="12345")
    parser.add_argument(
        "--variants-per-type",
        type=int,
        default=4,
        help="Number of slackness/deformation variants to render per knot type.",
    )
    parser.add_argument(
        "--angles-per-variant",
        type=int,
        default=5,
        help="Number of camera angles to keep for each single-scene sample.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of equivalence/non-equivalence pair samples to render.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Square output image size in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where dataset_metadata.json, summary.json, and images will be written.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help="Remove previously generated files in the output directory before writing new data.",
    )
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=12,
        help="If the page evaluate crashes mid-run, reopen the page and resume up to this many times.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="If set, truncate question.jsonl to this many rows after building. Rows are interleaved across "
             "tasks (sample-major order), so head-truncation keeps a balanced task-type distribution.",
    )
    parser.add_argument(
        "--questions-per-difficulty",
        type=int,
        default=None,
        help=(
            "If set, emit this many final question rows for each of easy, medium, and hard. "
            "Rows within each difficulty are stratified across task type and knot type. "
            "This takes precedence over --max-questions."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of browser pages to use for sharded parallel rendering. Default: 1.",
    )
    parser.set_defaults(headless=True, clean=True)
    return parser


def data_url_to_png_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Expected a PNG data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file())


def clear_output(output_dir: Path) -> int:
    removed = 0
    for section in (DEFAULT_IMAGE_ROOT_NAME,) + LEGACY_SECTION_DIRS:
        path = output_dir / section
        removed += count_files(path)
        if path.exists():
            shutil.rmtree(path)
    for filename in ROOT_OUTPUT_FILES:
        path = output_dir / filename
        removed += count_files(path)
        if path.exists():
            path.unlink()
    return removed


def build_metadata_payload(
    *,
    samples: Sequence[Dict[str, Any]],
    stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "version": "4.0",
        "task": "knots_static",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "samples": list(samples),
        "stats": dict(stats or {}),
    }


def infer_single_section(meta: Dict[str, Any]) -> str:
    family = str(meta.get("family") or "")
    if family == "MULTI_ROPE":
        return "multi_rope"
    if family == "LINK_GROUP":
        return "link_groups"
    if family == "MIXED":
        return "mixed"
    if bool(meta.get("isLink")):
        return "links"
    return "singles"


def expected_single_filenames(meta: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for image in meta.get("images") or []:
        filename = image.get("filename") if isinstance(image, dict) else str(image)
        filename = str(filename or "").strip()
        if not filename:
            raise ValueError(f"Missing filename in sample metadata: {meta.get('id')}")
        names.append(filename)
    return names


def expected_pair_filenames(meta: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("image1", "image2"):
        filename = str(meta.get(key) or "").strip()
        if not filename:
            raise ValueError(f"Missing {key} in pair metadata: {meta.get('id')}")
        names.append(filename)
    return names


def file_payload_map(files: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    payloads: Dict[str, str] = {}
    for file_entry in files or []:
        filename = str(file_entry.get("filename") or "").strip()
        data_url = str(file_entry.get("dataUrl") or "").strip()
        if filename and data_url:
            payloads[filename] = data_url
    return payloads


def infer_sample_section(meta: Dict[str, Any]) -> str:
    if "label_equivalent" in meta:
        return "pairs"
    return infer_single_section(meta)


def write_sample_images(
    *,
    meta: Dict[str, Any],
    files: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    sample_id = str(meta.get("id") or "sample")
    image_root = output_dir / DEFAULT_IMAGE_ROOT_NAME
    diff_dir = DIFFICULTY_DIR.get(meta.get("difficulty", ""), f"difficulty_{meta.get('difficulty', 'unknown')}")
    payloads = file_payload_map(files)
    expected = (
        expected_pair_filenames(meta)
        if "label_equivalent" in meta
        else expected_single_filenames(meta)
    )
    missing = [filename for filename in expected if filename not in payloads]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing rendered image payload(s) for {sample_id}: {preview}")

    sample_folder = f"knots_static_{diff_dir}_{sample_id}"
    written_paths: List[Path] = []
    for filename in expected:
        destination = image_root / diff_dir / sample_folder / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data_url_to_png_bytes(payloads[filename]))
        written_paths.append(destination)

    if "label_equivalent" in meta:
        for key in ("image1", "image2"):
            if key in meta:
                meta[key] = f"{diff_dir}/{sample_folder}/{meta[key]}"
    else:
        for image in meta.get("images") or []:
            if isinstance(image, dict) and "filename" in image:
                image["filename"] = f"{diff_dir}/{sample_folder}/{image['filename']}"

    return written_paths


def _normalize_difficulty(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"easy", "medium", "hard"}:
        return candidate
    return "hard"


def _normalize_seed(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    return int(text) if text.lstrip("-").isdigit() else text


CAMERA_PREFERENCE_BY_DIFFICULTY = {
    "easy": ("iso_fr", "front", "oblique", "top_tilt", "low_side"),
    "medium": ("oblique", "top_tilt", "front", "iso_fr", "low_side"),
    "hard": ("oblique", "top_tilt", "front", "low_side", "iso_fr"),
}

HARD_VIEW_CAMERA_TASKS = {"T03_link_topology", "T04_link_property"}
HARD_VIEW_CAMERA_PREFERENCE_BY_DIFFICULTY = {
    # T03/T04 need difficult perspectives, but not views where rings collapse
    # into indistinguishable front/back overlap. Prefer oblique/top-tilt and
    # keep low_side only as a last fallback for old three-angle datasets.
    "easy": ("oblique", "front", "iso_fr", "top_tilt", "low_side"),
    "medium": ("oblique", "top_tilt", "front", "iso_fr", "low_side"),
    "hard": ("oblique", "top_tilt", "front", "iso_fr", "low_side"),
}


def _image_camera_angle(image: Any) -> str:
    if isinstance(image, dict):
        camera_angle = str(image.get("cameraAngle") or "").strip()
        if camera_angle:
            return camera_angle
        filename = str(image.get("filename") or "")
    else:
        filename = str(image or "")
    stem = Path(filename).stem
    if stem.endswith("_colored"):
        stem = stem[: -len("_colored")]
    for camera in ("top_tilt", "low_side", "iso_fr", "oblique", "front"):
        if stem.endswith(f"_{camera}") or f"_{camera}_" in stem:
            return camera
    return ""


def _pick_preferred_image(
    sample: Dict[str, Any],
    *,
    difficulty: str,
    colored: bool = False,
    task_id: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Pick a task image using difficulty-aware camera preferences.

    Easy rows keep the canonical clear view. Medium/hard rows prefer oblique,
    top, and low-side views when those were rendered, falling back cleanly for
    smaller one-angle candidate pools.
    """
    candidates: list[tuple[str, str]] = []
    for img in sample.get("images") or []:
        fname = img.get("filename") if isinstance(img, dict) else str(img)
        fname = str(fname or "").strip()
        if not fname:
            continue
        is_colored = fname.endswith("_colored.png")
        if colored != is_colored:
            continue
        candidates.append((fname, _image_camera_angle(img)))
    if not candidates:
        return None, None

    camera_preferences = (
        HARD_VIEW_CAMERA_PREFERENCE_BY_DIFFICULTY
        if task_id in HARD_VIEW_CAMERA_TASKS
        else CAMERA_PREFERENCE_BY_DIFFICULTY
    )
    for camera in camera_preferences.get(difficulty, camera_preferences["hard"]):
        for fname, candidate_camera in candidates:
            if candidate_camera == camera:
                return fname, candidate_camera
    return candidates[0]


def _question_image_exists(output_dir: Path, image_filename: str) -> bool:
    return (output_dir / DEFAULT_IMAGE_ROOT_NAME / image_filename).exists()


def _missing_question_images(rows: Sequence[Dict[str, Any]], *, output_dir: Path) -> List[str]:
    missing: List[str] = []
    for row in rows:
        for image in row.get("images") or []:
            if not (output_dir / str(image)).exists():
                missing.append(f"{row.get('id')}: {image}")
    return missing


def _t04_free_colors_after_removal(sample: Dict[str, Any], removed_idx: int) -> List[str]:
    ring_colors = sample.get("ringColors") or []
    n = len(ring_colors)
    if removed_idx < 0 or removed_idx >= n:
        return []
    remaining = {i for i in range(n) if i != removed_idx}
    not_free: set[int] = set()
    for pair in sample.get("linkageEdges") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if i in remaining and j in remaining:
            not_free.add(i)
            not_free.add(j)
    for group in sample.get("brunnianGroups") or []:
        if not isinstance(group, (list, tuple)):
            continue
        try:
            members = [int(g) for g in group]
        except (TypeError, ValueError):
            continue
        if members and all(g in remaining for g in members):
            not_free.update(members)
    return [ring_colors[i] for i in sorted(remaining) if i not in not_free]


def _t04_removed_ring_degree(sample: Dict[str, Any], removed_idx: int) -> int:
    degree = 0
    for pair in sample.get("linkageEdges") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if i == removed_idx or j == removed_idx:
            degree += 1
    for group in sample.get("brunnianGroups") or []:
        if not isinstance(group, (list, tuple)):
            continue
        try:
            members = [int(g) for g in group]
        except (TypeError, ValueError):
            continue
        if removed_idx in members:
            degree += max(1, len(members) - 1)
    return degree


def _pick_t04_removed_color(sample: Dict[str, Any]) -> Optional[str]:
    """Deterministically pick which ring's color to ask the model to remove.

    Prefer bridge/interior removals that make multiple remaining rings free.
    That makes T04 test link-reasoning instead of just color lookup, while
    remaining stable across regenerations.
    """
    ring_colors = sample.get("ringColors") or []
    if not ring_colors:
        return None
    import hashlib
    h = int(hashlib.sha256(str(sample.get("id", "")).encode()).hexdigest(), 16)
    offset = h % len(ring_colors)

    best_idx = 0
    best_score: tuple[int, int, int, int, int] | None = None
    for idx in range(len(ring_colors)):
        free_colors = _t04_free_colors_after_removal(sample, idx)
        free_count = len(set(free_colors))
        free_bucket = 2 if free_count >= 2 else 1 if free_count == 1 else 0
        degree = _t04_removed_ring_degree(sample, idx)
        stable_tiebreak = -((idx - offset) % len(ring_colors))
        score = (free_bucket, min(free_count, 4), degree, len(ring_colors), stable_tiebreak)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    return ring_colors[best_idx]


def build_question_rows(
    *,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Expand raw samples into question.jsonl rows for every single-image task in vlm_benchmark.TASKS.

    Rows are emitted in sample-major order (interleaved across tasks): for each sample, every
    applicable task is emitted in TASKS order before moving to the next sample. This keeps the task
    distribution balanced when downstream callers head-truncate via --max-questions.
    """
    from vlm_benchmark import TASKS, build_prompt, compute_difficulty, get_ground_truth, sample_applicable

    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    # T04's prompt depends on the per-sample removed_color so it's built per-row
    # below; other tasks share a single static prompt. If a task isn't present
    # in TASK_SPECS (e.g. it's been temporarily disabled in question_phrasings.py),
    # skip it gracefully rather than crashing every sample.
    def _build_static_prompt(task_id: str) -> Optional[str]:
        if task_id == "T04_link_property":
            return ""
        try:
            return build_prompt(task_id, 0)
        except (KeyError, IndexError):
            return None
    prompts = {task_id: _build_static_prompt(task_id) for task_id in TASKS}
    rows: List[Dict[str, Any]] = []
    per_task_index: Dict[str, int] = {task_id: 0 for task_id in TASKS}
    phrasing_idx = 0

    for sample in samples:
        # Skip pair samples (they have label_equivalent and image1/image2 instead of images list)
        if "label_equivalent" in sample:
            continue
        sample_difficulty = sample.get("difficulty") or compute_difficulty(sample).get("level")
        sample_difficulty = _normalize_difficulty(sample_difficulty)
        seed_value = _normalize_seed(sample.get("seed"))
        knot_type = str(sample.get("knotType", ""))

        for task_id in TASKS:
            if not sample_applicable(sample, task_id):
                continue
            # Skip tasks that don't have a phrasing entry (and aren't T04, which builds per-sample).
            if task_id != "T04_link_property" and prompts.get(task_id) is None:
                continue

            # T04 needs (a) the per-ring solid-color image variant and
            # (b) a per-sample removed_color baked into the prompt and GT.
            if task_id == "T04_link_property":
                colored_image, camera_angle = _pick_preferred_image(
                    sample,
                    difficulty=sample_difficulty,
                    colored=True,
                    task_id=task_id,
                )
                ring_colors = sample.get("ringColors") or []
                try:
                    num_components = int(sample.get("numComponents") or 0)
                except (TypeError, ValueError):
                    num_components = 0
                if not (3 <= num_components <= len(T04_PALETTE_NAMES)):
                    continue
                if len(ring_colors) != num_components:
                    continue
                if any(str(color).strip().lower() not in T04_PALETTE_NAMES for color in ring_colors):
                    continue
                removed_color = _pick_t04_removed_color(sample)
                if not colored_image or not removed_color:
                    continue
                if not _question_image_exists(output_dir, colored_image):
                    continue
                # Compute GT against the chosen removed_color. We pass the
                # color through metadata so get_ground_truth stays a pure
                # function of (metadata, task_id).
                t04_metadata = dict(sample)
                t04_metadata["removed_color"] = removed_color
                try:
                    gt = get_ground_truth(t04_metadata, task_id)
                except (KeyError, TypeError):
                    continue
                if gt is None:
                    continue
                try:
                    question = build_prompt(task_id, 0, removed_color=removed_color)
                except (KeyError, IndexError):
                    continue
                image_filename = colored_image
                meta_extra = {
                    "removed_color": removed_color,
                    "ring_colors": list(ring_colors),
                    "answer_palette": list(T04_ANSWER_PALETTE),
                    "camera_angle": camera_angle,
                }
                # GT is a canonical comma-separated string of color names, or
                # the special token "none" when no remaining ring is free.
                # Serialize as a JSON list so the downstream answer schema is
                # always a non-empty list of strings — ["none"] when no rings
                # are free, never an empty list.
                if gt == "none":
                    gt_serialized = json.dumps(["none"])
                else:
                    gt_serialized = json.dumps(
                        [c for c in gt.split(",") if c]
                    )
            else:
                try:
                    gt = get_ground_truth(sample, task_id)
                except (KeyError, TypeError):
                    continue
                if gt is None:
                    continue
                question = prompts[task_id]
                image_filename, camera_angle = _pick_preferred_image(
                    sample,
                    difficulty=sample_difficulty,
                    colored=False,
                    task_id=task_id,
                )
                if not image_filename:
                    continue
                if not _question_image_exists(output_dir, image_filename):
                    continue
                meta_extra = {"camera_angle": camera_angle}
                gt_serialized = str(gt)

            difficulty = _question_level_difficulty(
                sample=sample,
                task_id=task_id,
                gt_answer=gt_serialized,
                sample_difficulty=sample_difficulty,
            )
            difficulty_num = DIFFICULTY_NUM.get(difficulty, difficulty)
            sample_index = per_task_index[task_id]
            row_id = f"knots_static_difficulty_{difficulty_num}_{task_id}_{sample_index:04d}"
            rows.append(
                {
                    "id": row_id,
                    "category": ["knots", "knots_static", task_id],
                    "type": task_id,
                    "meta_info": {
                        "task_name": "knots_static",
                        "config": config_rel,
                        "seed": seed_value,
                        "repeat_index": 0,
                        "difficulty": difficulty,
                        "sample_difficulty": sample_difficulty,
                        "knot_type": knot_type,
                        "sample_id": sample.get("id"),
                        "num_components": sample.get("numComponents"),
                        "num_linked_components": sample.get("numLinkedComponents"),
                        "difficulty_score": sample.get("difficulty_score"),
                        "trap_type": sample.get("trap_type"),
                        "phrasing_index": phrasing_idx,
                        **meta_extra,
                    },
                    "question": question,
                    "images": [f"{DEFAULT_IMAGE_ROOT_NAME}/{image_filename}"],
                    "gt_answer": gt_serialized,
                }
            )
            per_task_index[task_id] = sample_index + 1

    return rows


def _normalized_question_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _row_image_sha256s(row: Dict[str, Any], *, output_dir: Path) -> tuple[str, ...]:
    hashes: List[str] = []
    for image_ref in row.get("images") or []:
        image_path = Path(str(image_ref))
        if not image_path.is_absolute():
            image_path = output_dir / image_path
        hashes.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
    return tuple(hashes)


def _deduplicate_question_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Drop exact duplicate questions before any count-based truncation.

    The key is intentionally narrow: task type, normalized question text,
    ground-truth answer, and the SHA256 digest(s) of the referenced image
    bytes. This removes duplicate rows for the same task/image/question while
    preserving distinct tasks that happen to share an image.
    """
    seen: set[tuple[Any, ...]] = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("type") or ""),
            _normalized_question_text(row.get("question")),
            str(row.get("gt_answer") or ""),
            _row_image_sha256s(row, output_dir=output_dir),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _stratified_truncate_by_task(rows: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    """Truncate to `max_count` rows while keeping every (task_type, knot_type) cell represented.

    Buckets rows by `(type, meta_info.knot_type)` (preserving first-seen order), then round-robins
    one row per non-empty bucket per pass. Small buckets (rare knot types like torus_3_5, or link-
    only tasks T04/T05) are fully covered first; remaining capacity is absorbed proportionally by
    the larger ones. Within each bucket the original sample order is preserved.
    """
    if max_count <= 0 or len(rows) <= max_count:
        return list(rows)

    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("meta_info") or {}
        key = (str(row.get("type")), str(meta.get("knot_type", "")))
        buckets.setdefault(key, []).append(row)

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_count:
        progressed = False
        for bucket in buckets.values():
            if not bucket:
                continue
            if len(selected) >= max_count:
                break
            selected.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return selected


def _count_bucket(value: Any, *, one_label: str, two_label: str, mid_label: str, high_label: str) -> str:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return "count:other"
    if count <= 1:
        return one_label
    if count == 2:
        return two_label
    if count <= 4:
        return mid_label
    return high_label


def _t04_free_bucket(value: Any) -> str:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = value
    if isinstance(parsed, list):
        colors = [
            str(item).strip().lower()
            for item in parsed
            if str(item).strip().lower() and str(item).strip().lower() != "none"
        ]
    else:
        text = str(parsed or "").strip().lower()
        colors = [] if text in {"", "none", "[]"} else [part.strip() for part in text.split(",") if part.strip()]
    if not colors:
        return "free:none"
    if len(set(colors)) == 1:
        return "free:single"
    return "free:multi"


def _question_level_difficulty(
    *,
    sample: Dict[str, Any],
    task_id: str,
    gt_answer: Any,
    sample_difficulty: str,
) -> str:
    """Calibrate difficulty per question, not just per rendered sample.

    A visually complex single knot can still yield a trivial T02 answer of
    "1", while visually modest open/knot classification can be much harder
    than its sample-level score suggests. This per-task mapping prevents large
    eval slices from being inflated by easy counting rows.
    """
    knot_type = str(sample.get("knotType") or "")
    try:
        num_components = int(sample.get("numComponents") or 0)
    except (TypeError, ValueError):
        num_components = 0

    if task_id == "T01_structure_classification":
        if knot_type in {"chain_plus_free", "hopf_plus_free", "double_hopf", "link_cluster"}:
            return "hard"
        if knot_type in {
            "ring_like_open_rope",
            "loose_open_knot",
            "occluded_knot",
            "ring_like_open_knot",
            "loose_cinquefoil",
            "torus_3_5",
            "torus_3_4",
            "torus_2_9",
            "torus_2_7",
        }:
            return "hard" if sample_difficulty == "hard" else "medium"
        return sample_difficulty

    if task_id == "T02_component_count":
        if num_components <= 2:
            return "easy"
        if num_components <= 6:
            return "medium"
        return "hard"

    if task_id == "T03_link_topology":
        if knot_type in {"chain_plus_free", "hopf_plus_free", "double_hopf", "link_cluster"}:
            return "hard"
        if knot_type in {"chain", "borromean"}:
            return "medium"
        return "easy"

    if task_id == "T04_link_property":
        try:
            parsed = json.loads(gt_answer) if isinstance(gt_answer, str) else gt_answer
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = gt_answer
        if isinstance(parsed, list):
            free_count = len({str(item).strip().lower() for item in parsed if str(item).strip().lower() != "none"})
        else:
            text = str(parsed or "").strip().lower()
            free_count = 0 if text in {"", "none", "[]"} else len({part.strip() for part in text.split(",") if part.strip()})

        if knot_type in {"chain_plus_free", "hopf_plus_free"} or num_components >= 6 or free_count >= 3:
            return "hard"
        if knot_type == "borromean" or num_components >= 4 or free_count >= 2:
            return "medium"
        return "easy"

    if task_id == "T05_linked_count":
        try:
            linked_count = int(str(gt_answer).strip())
        except (TypeError, ValueError):
            linked_count = 0
        if linked_count <= 2:
            return "easy"
        if linked_count <= 6:
            return "medium"
        return "hard"

    return sample_difficulty


def _gt_bucket(row: Dict[str, Any]) -> str:
    task_id = str(row.get("type") or "")
    gt = row.get("gt_answer")
    if task_id in {"T01_structure_classification", "T03_link_topology"}:
        return f"option:{str(gt).strip().upper()}"
    if task_id == "T02_component_count":
        return _count_bucket(
            gt,
            one_label="components:1",
            two_label="components:2",
            mid_label="components:3-4",
            high_label="components:5+",
        )
    if task_id == "T04_link_property":
        return _t04_free_bucket(gt)
    if task_id == "T05_linked_count":
        return _count_bucket(
            gt,
            one_label="linked:0-1",
            two_label="linked:2",
            mid_label="linked:3-4",
            high_label="linked:5+",
        )
    return f"gt:{str(gt)}"


GT_BUCKET_PRIORITY = {
    "T02_component_count": ("components:5+", "components:3-4", "components:2", "components:1"),
    "T03_link_topology": ("option:D", "option:C", "option:A", "option:B"),
    "T04_link_property": ("free:multi", "free:single", "free:none"),
    "T05_linked_count": ("linked:5+", "linked:3-4", "linked:2", "linked:0-1"),
}

HARD_KNOT_TYPE_PRIORITY = {
    "chain_plus_free": 0,
    "hopf_plus_free": 1,
    "link_cluster": 2,
    "double_hopf": 3,
    "unlinked_rings": 4,
    "chain": 5,
    "borromean": 6,
    "hopf_link": 7,
}

HARD_TRAP_PRIORITY = {
    "linked_plus_free": 0,
    "multiple_link_groups": 1,
    "near_miss_unlink": 2,
    "long_slack_chain": 3,
}

EASY_KNOT_TYPE_PRIORITY = {
    "trefoil": 0,
    "figure8": 1,
    "torus_2_5": 2,
    "hopf_link": 3,
    "chain": 4,
    "unlinked_rings": 5,
    "twisted_ring": 6,
    "kinky_unknot": 7,
    "spiral_disk": 8,
    "unknot": 9,
}

MEDIUM_KNOT_TYPE_PRIORITY = {
    "occluded_knot": 0,
    "loose_open_knot": 1,
    "ring_like_open_rope": 2,
    "torus_3_5": 3,
    "torus_3_4": 4,
    "torus_2_9": 5,
    "torus_2_7": 6,
    "borromean": 7,
    "chain": 8,
    "hopf_link": 9,
}

DIFFICULTY_TASK_WEIGHTS = {
    # Easy should still include basic counting/topology, but not be dominated by
    # trivial single-rope counts.
    "easy": {
        "T01_structure_classification": 7,
        "T02_component_count": 3,
        "T03_link_topology": 3,
        "T04_link_property": 2,
        "T05_linked_count": 5,
    },
    # Medium is calibrated toward perceptual/topological discrimination rather
    # than long but regular chains, which were too easy for strong VLMs.
    "medium": {
        "T01_structure_classification": 8,
        "T02_component_count": 2,
        "T03_link_topology": 3,
        "T04_link_property": 4,
        "T05_linked_count": 3,
    },
}

DIFFICULTY_FALLBACK_TASK_PRIORITY = {
    "easy": {
        "T01_structure_classification": 0,
        "T05_linked_count": 1,
        "T03_link_topology": 2,
        "T04_link_property": 3,
        "T02_component_count": 4,
    },
    "medium": {
        "T01_structure_classification": 0,
        "T05_linked_count": 1,
        "T04_link_property": 2,
        "T03_link_topology": 3,
        "T02_component_count": 4,
    },
}

MEDIUM_COMPLEX_LINK_TYPES = {
    "chain",
    "borromean",
    "double_hopf",
    "link_cluster",
    "hopf_plus_free",
    "chain_plus_free",
}

HARD_COMPLEX_LINK_TYPES = {
    "double_hopf",
    "link_cluster",
    "hopf_plus_free",
    "chain_plus_free",
}

HARD_CORE_LINK_TYPES = {
    "double_hopf",
    "link_cluster",
    "chain_plus_free",
}

HARD_T04_LINK_TYPES = {
    "chain_plus_free",
    "hopf_plus_free",
    "chain",
    "borromean",
}


def _ordered_gt_buckets(task_id: str, buckets: Dict[str, Any]) -> List[str]:
    priority = GT_BUCKET_PRIORITY.get(task_id, ())
    ordered = [bucket for bucket in priority if bucket in buckets]
    ordered.extend(bucket for bucket in buckets if bucket not in set(ordered))
    return ordered


def _row_num_components(row: Dict[str, Any]) -> int:
    meta = row.get("meta_info") or {}
    try:
        return int(meta.get("num_components") or 0)
    except (TypeError, ValueError):
        return 0


def _row_knot_type(row: Dict[str, Any]) -> str:
    meta = row.get("meta_info") or {}
    return str(meta.get("knot_type") or "")


def _task_types_present(rows: Sequence[Dict[str, Any]]) -> set[str]:
    return {str(row.get("type") or "") for row in rows if row.get("type")}


def _is_hard_dense_candidate(
    row: Dict[str, Any],
    *,
    min_components: int,
    t04_min_components: int,
    knot_types: set[str],
) -> bool:
    knot_type = _row_knot_type(row)
    num_components = _row_num_components(row)
    task_id = str(row.get("type") or "")
    if task_id == "T04_link_property":
        return num_components >= t04_min_components and knot_type in HARD_T04_LINK_TYPES
    return num_components >= min_components and knot_type in knot_types


def _difficulty_candidate_pool(
    diff_rows: List[Dict[str, Any]],
    *,
    difficulty: str,
    target: int,
) -> List[Dict[str, Any]]:
    """Prefer visually substantial rows for medium/hard small balanced sets.

    The full generated pool can contain simple single-rope rows marked medium/hard
    due to saliency, occlusion, or answer-bucket balancing. For small slices such
    as 10 or 20 rows per difficulty, those rows make the requested difficulty look
    too easy. Use strict multi-component pools first, relaxing only if a dataset
    does not have enough rows or task coverage.
    """
    if difficulty not in {"medium", "hard"} or target <= 0:
        return diff_rows

    required_tasks = _task_types_present(diff_rows)
    if difficulty == "medium":
        return diff_rows

    if difficulty == "hard":
        tiers = (
            lambda row: _is_hard_dense_candidate(
                row,
                min_components=8,
                t04_min_components=8,
                knot_types=HARD_CORE_LINK_TYPES,
            ),
            lambda row: _is_hard_dense_candidate(
                row,
                min_components=7,
                t04_min_components=8,
                knot_types=HARD_COMPLEX_LINK_TYPES,
            ),
            lambda row: _is_hard_dense_candidate(
                row,
                min_components=6,
                t04_min_components=6,
                knot_types=HARD_COMPLEX_LINK_TYPES,
            ),
            lambda row: _is_hard_dense_candidate(
                row,
                min_components=5,
                t04_min_components=5,
                knot_types=HARD_COMPLEX_LINK_TYPES,
            ),
        )
    for predicate in tiers:
        candidates = [row for row in diff_rows if predicate(row)]
        if len(candidates) >= target and required_tasks <= _task_types_present(candidates):
            return candidates
    return diff_rows


def _row_sample_preference(row: Dict[str, Any]) -> tuple:
    meta = row.get("meta_info") or {}
    difficulty = _normalize_difficulty(meta.get("difficulty"))
    knot_type = str(meta.get("knot_type") or "")
    trap_type = str(meta.get("trap_type") or "")
    try:
        score = float(meta.get("difficulty_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    if difficulty == "hard":
        return (
            0 if trap_type else 1,
            HARD_TRAP_PRIORITY.get(trap_type, 99),
            HARD_KNOT_TYPE_PRIORITY.get(knot_type, 99),
            -_row_num_components(row),
            -score,
            str(meta.get("sample_id") or row.get("id") or ""),
        )
    if difficulty == "medium":
        return (
            0,
            0,
            -score,
            MEDIUM_KNOT_TYPE_PRIORITY.get(knot_type, 99),
            -_row_num_components(row),
            str(meta.get("sample_id") or row.get("id") or ""),
        )
    return (
        0,
        0,
        EASY_KNOT_TYPE_PRIORITY.get(knot_type, 99),
        -score,
        -_row_num_components(row),
        str(meta.get("sample_id") or row.get("id") or ""),
    )


def _knot_bucket_preference(knot_type: str, rows: List[Dict[str, Any]]) -> tuple:
    if not rows:
        return (99, 99, 99, 0.0, knot_type)
    best = min(_row_sample_preference(row) for row in rows)
    return (*best, knot_type)


def _stratified_truncate_by_task_gt(rows: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    """Truncate one task slice while balancing GT buckets and knot types."""
    if max_count <= 0 or len(rows) <= max_count:
        return list(rows)

    task_id = str(rows[0].get("type") or "") if rows else ""
    by_gt: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        meta = row.get("meta_info") or {}
        gt_bucket = _gt_bucket(row)
        knot_type = str(meta.get("knot_type", ""))
        by_gt.setdefault(gt_bucket, {}).setdefault(knot_type, []).append(row)
    for knot_buckets in by_gt.values():
        for bucket in knot_buckets.values():
            bucket.sort(key=_row_sample_preference)

    selected: List[Dict[str, Any]] = []
    gt_order = _ordered_gt_buckets(task_id, by_gt)
    knot_orders = {
        gt: sorted(
            knot_buckets,
            key=lambda knot_type: _knot_bucket_preference(knot_type, knot_buckets[knot_type]),
        )
        for gt, knot_buckets in by_gt.items()
    }
    knot_positions = {gt: 0 for gt in knot_orders}
    while len(selected) < max_count:
        progressed = False
        for gt_bucket in gt_order:
            if len(selected) >= max_count:
                break
            knot_buckets = by_gt.get(gt_bucket) or {}
            knot_order = knot_orders.get(gt_bucket, [])
            if not knot_order:
                continue
            start = knot_positions.get(gt_bucket, 0) % len(knot_order)
            for offset in range(len(knot_order)):
                knot_type = knot_order[(start + offset) % len(knot_order)]
                bucket = knot_buckets.get(knot_type) or []
                if not bucket:
                    continue
                selected.append(bucket.pop(0))
                knot_positions[gt_bucket] = (start + offset + 1) % len(knot_order)
                progressed = True
                break
        if not progressed:
            break
    return selected


def _weighted_task_quotas(
    *,
    difficulty: str,
    present_tasks: Sequence[str],
    target: int,
) -> Dict[str, int]:
    weights = DIFFICULTY_TASK_WEIGHTS.get(difficulty)
    if not weights:
        base_quota = target // len(present_tasks)
        remainder = target % len(present_tasks)
        return {
            task_id: base_quota + (1 if index < remainder else 0)
            for index, task_id in enumerate(present_tasks)
        }

    raw_weights = {task_id: max(0, int(weights.get(task_id, 1))) for task_id in present_tasks}
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return {task_id: 0 for task_id in present_tasks}

    quotas: Dict[str, int] = {}
    fractions: List[tuple[float, str]] = []
    assigned = 0
    for task_id in present_tasks:
        exact = target * raw_weights[task_id] / total_weight
        quota = int(exact)
        quotas[task_id] = quota
        assigned += quota
        fractions.append((exact - quota, task_id))

    for _, task_id in sorted(fractions, reverse=True)[: max(0, target - assigned)]:
        quotas[task_id] += 1
    return quotas


def _difficulty_fallback_rows(
    rows: List[Dict[str, Any]],
    *,
    difficulty: str,
    max_count: int,
) -> List[Dict[str, Any]]:
    if max_count <= 0:
        return []
    priority = DIFFICULTY_FALLBACK_TASK_PRIORITY.get(difficulty)
    if not priority:
        return _stratified_truncate_by_task(rows, max_count)
    return sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("type") or ""), 99),
            _row_sample_preference(row),
        ),
    )[:max_count]


def _stratified_truncate_by_difficulty_task(
    rows: List[Dict[str, Any]],
    per_difficulty: int,
) -> List[Dict[str, Any]]:
    """Truncate to N rows per difficulty while preserving task/knot-type coverage.

    This is the small-eval counterpart to `_stratified_truncate_by_task`. For a
    30-row eval, `per_difficulty=10` gives exactly 10 easy, 10 medium, and
    10 hard rows. Within each difficulty, rows are first spread across task
    types, then across knot types inside each task.
    """
    target = max(0, int(per_difficulty))
    if target <= 0:
        return []

    task_order = (
        "T01_structure_classification",
        "T02_component_count",
        "T03_link_topology",
        "T04_link_property",
        "T05_linked_count",
    )
    selected_by_difficulty: List[List[Dict[str, Any]]] = []
    for difficulty in ("easy", "medium", "hard"):
        diff_rows = [
            row for row in rows
            if _normalize_difficulty((row.get("meta_info") or {}).get("difficulty")) == difficulty
        ]
        diff_rows = _difficulty_candidate_pool(
            diff_rows,
            difficulty=difficulty,
            target=target,
        )
        present_tasks = [task for task in task_order if any(row.get("type") == task for row in diff_rows)]
        present_tasks.extend(
            sorted({
                str(row.get("type"))
                for row in diff_rows
                if str(row.get("type")) not in task_order
            })
        )
        if not present_tasks:
            raise ValueError(f"No rows available for difficulty {difficulty!r}")

        task_quotas = _weighted_task_quotas(
            difficulty=difficulty,
            present_tasks=present_tasks,
            target=target,
        )
        selected_by_task: Dict[str, List[Dict[str, Any]]] = {}
        selected_ids: set[str] = set()
        for task_id in present_tasks:
            quota = min(task_quotas.get(task_id, 0), sum(1 for row in diff_rows if row.get("type") == task_id))
            if quota <= 0:
                continue
            task_rows = [row for row in diff_rows if row.get("type") == task_id]
            task_selected = _stratified_truncate_by_task_gt(task_rows, quota)
            for row in task_selected:
                row_id = str(row.get("id") or id(row))
                if row_id in selected_ids:
                    continue
                selected_ids.add(row_id)
                selected_by_task.setdefault(task_id, []).append(row)

        diff_selected: List[Dict[str, Any]] = []
        task_positions = {task_id: 0 for task_id in present_tasks}
        while len(diff_selected) < target:
            progressed = False
            for task_id in present_tasks:
                bucket = selected_by_task.get(task_id) or []
                pos = task_positions.get(task_id, 0)
                if pos >= len(bucket):
                    continue
                diff_selected.append(bucket[pos])
                task_positions[task_id] = pos + 1
                progressed = True
                if len(diff_selected) >= target:
                    break
            if not progressed:
                break

        if len(diff_selected) < target:
            remaining = [
                row for row in diff_rows
                if str(row.get("id") or id(row)) not in selected_ids
            ]
            diff_selected.extend(
                _difficulty_fallback_rows(
                    remaining,
                    difficulty=difficulty,
                    max_count=target - len(diff_selected),
                )
            )

        if len(diff_selected) < target:
            raise ValueError(
                f"Not enough {difficulty} rows for --questions-per-difficulty={target}: "
                f"available {len(diff_selected)}"
            )
        selected_by_difficulty.append(diff_selected[:target])

    interleaved: List[Dict[str, Any]] = []
    for index in range(target):
        for bucket in selected_by_difficulty:
            if index < len(bucket):
                interleaved.append(bucket[index])
    return interleaved


KNOT_TYPE_ORDER = (
    "unknot",
    "twisted_ring",
    "spiral_disk",
    "kinky_unknot",
    "ring_like_open_rope",
    "loose_open_knot",
    "ring_like_open_knot",
    "loose_cinquefoil",
    "occluded_knot",
    "trefoil",
    "figure8",
    "torus_2_5",
    "torus_2_7",
    "torus_2_9",
    "torus_3_4",
    "torus_3_5",
)
LINK_TYPE_ORDER = (
    "hopf_link",
    "unlinked_rings",
    "chain",
    "borromean",
    "double_hopf",
    "link_cluster",
    "hopf_plus_free",
    "chain_plus_free",
)
LINK_VARIANT_COUNTS = {
    "hopf_link": 4,
    "unlinked_rings": 12,
    "chain": 10,
    "borromean": 18,
    "double_hopf": 36,
    "link_cluster": 42,
    "hopf_plus_free": 28,
    "chain_plus_free": 48,
}


def _variant_index_from_sample_id(sample_id: str) -> int:
    match = re.search(r"_v(\d+)$", sample_id)
    return int(match.group(1)) if match else 0


def _sample_sort_key(sample: Dict[str, Any], *, variants_per_type: int, num_pairs: int) -> tuple:
    generation_index = sample.get("generationIndex")
    if isinstance(generation_index, int):
        return (0, generation_index)
    if isinstance(generation_index, float) and generation_index.is_integer():
        return (0, int(generation_index))

    sample_id = str(sample.get("id") or "")
    single_base = len(KNOT_TYPE_ORDER) * variants_per_type
    link_base_by_type: Dict[str, int] = {}
    cursor = single_base
    for knot_type in LINK_TYPE_ORDER:
        link_base_by_type[knot_type] = cursor
        cursor += LINK_VARIANT_COUNTS[knot_type]

    if "label_equivalent" in sample:
        match = re.match(r"pair(\d+)$", sample_id)
        pair_index = int(match.group(1)) - 1 if match else num_pairs
        return (0, cursor + max(0, pair_index))

    knot_type = str(sample.get("knotType") or "")
    variant_index = _variant_index_from_sample_id(sample_id)
    if knot_type in KNOT_TYPE_ORDER:
        return (0, KNOT_TYPE_ORDER.index(knot_type) * variants_per_type + variant_index)
    if knot_type in link_base_by_type:
        return (0, link_base_by_type[knot_type] + variant_index)

    return (1, sample_id)


def ordered_metadata_entries(
    samples: Sequence[Dict[str, Any]],
    *,
    variants_per_type: int,
    num_pairs: int,
) -> List[Dict[str, Any]]:
    return sorted(
        samples,
        key=lambda sample: _sample_sort_key(
            sample,
            variants_per_type=variants_per_type,
            num_pairs=num_pairs,
        ),
    )


def build_summary(
    *,
    metadata_payload: Dict[str, Any],
    seed: str,
    variants_per_type: int,
    angles_per_variant: int,
    num_pairs: int,
    image_size: int,
    total_images: int,
    other_files: int,
) -> Dict[str, Any]:
    samples = list(metadata_payload.get("samples") or [])
    section_counts = {section: 0 for section in SINGLE_SECTIONS}
    pair_count = 0
    for sample in samples:
        section = infer_sample_section(sample)
        if section == "pairs":
            pair_count += 1
        elif section in section_counts:
            section_counts[section] += 1
    return {
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "variantsPerType": int(variants_per_type),
        "anglesPerVariant": int(angles_per_variant),
        "numPairs": int(num_pairs),
        "imageSize": int(image_size),
        "totalImages": int(total_images),
        "totalMetadataRecords": int(len(samples)),
        "sections": {
            **section_counts,
            "pairs": pair_count,
            "other": int(other_files),
        },
        "stats": metadata_payload.get("stats", {}),
    }


async def generate_samples_async(args: argparse.Namespace) -> Dict[str, Any]:
    variants_per_type = max(1, int(args.variants_per_type))
    angles_per_variant = max(1, int(args.angles_per_variant))
    num_pairs = max(0, int(args.num_pairs))
    image_size = max(1, int(args.image_size))
    workers = max(1, int(getattr(args, "workers", 1)))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_files = clear_output(output_dir) if args.clean else 0

    metadata_path = output_dir / "dataset_metadata.json"
    summary_path = output_dir / "summary.json"
    metadata_entries: List[Dict[str, Any]] = []
    stats_state: Dict[str, Any] = {}
    progress_state: Dict[str, Any] = {"last_stage": None}
    completed_sample_ids = set()
    state_lock = asyncio.Lock()
    written_images = 0
    written_samples = 0
    expected_total_images = 0

    # Resume support: bootstrap from existing metadata (only when --no-clean was passed).
    if not args.clean and metadata_path.exists():
        try:
            prior = json.loads(metadata_path.read_text())
            for s in prior.get("samples", []):
                metadata_entries.append(s)
                sample_id = s.get("id")
                if sample_id:
                    completed_sample_ids.add(str(sample_id))
                if "label_equivalent" in s:
                    written_images += 2
                else:
                    written_images += len(s.get("images") or [])
            written_samples = len(metadata_entries)
            if metadata_entries:
                print(f"[resume] loaded {written_samples} prior samples, {written_images} prior images on disk", flush=True)
        except Exception as exc:
            print(f"[resume] failed to load prior metadata ({exc}); starting fresh", flush=True)
            metadata_entries.clear()
            completed_sample_ids.clear()
            written_images = 0
            written_samples = 0

    def flush_state() -> Dict[str, Any]:
        ordered_samples = ordered_metadata_entries(
            metadata_entries,
            variants_per_type=variants_per_type,
            num_pairs=num_pairs,
        )
        metadata_payload = build_metadata_payload(samples=ordered_samples, stats=stats_state)
        write_json(metadata_path, metadata_payload)
        summary = build_summary(
            metadata_payload=metadata_payload,
            seed=str(args.seed),
            variants_per_type=variants_per_type,
            angles_per_variant=angles_per_variant,
            num_pairs=num_pairs,
            image_size=image_size,
            total_images=written_images,
            other_files=0,
        )
        write_json(summary_path, summary)
        return metadata_payload

    print(
        "Starting streamed generation for knots_static "
        f"(variants={variants_per_type}, angles={angles_per_variant}, pairs={num_pairs}, "
        f"size={image_size}, workers={workers})",
        flush=True,
    )

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with _import_playwright()() as playwright:
            browser = await playwright.chromium.launch(headless=args.headless)

            def _worker_label(worker_index: int) -> str:
                return f"worker {worker_index + 1}/{workers}"

            async def _handle_stream_progress(worker_index: int, _source: Any, payload: Dict[str, Any]) -> None:
                nonlocal expected_total_images
                message = str(payload.get("message") or "").strip()
                total = int(payload.get("total") or 0)
                if total > 0:
                    expected_total_images = max(expected_total_images, total)
                if not message or message.startswith("["):
                    return
                progress_key = f"{worker_index}:{message}"
                if progress_key != progress_state["last_stage"]:
                    progress_state["last_stage"] = progress_key
                    print(f"[progress][{_worker_label(worker_index)}] {message}", flush=True)

            async def _handle_stream_sample(worker_index: int, _source: Any, payload: Dict[str, Any]) -> None:
                nonlocal written_images, written_samples
                kind = str(payload.get("kind") or "single").strip().lower()
                meta = dict(payload.get("metadata") or {})
                files = list(payload.get("files") or [])
                sample_id = str(meta.get("id") or "").strip()

                if kind not in {"single", "pair"}:
                    raise ValueError(f"Unsupported sample kind from frontend: {kind!r}")
                async with state_lock:
                    if sample_id and sample_id in completed_sample_ids:
                        return
                    written_paths = write_sample_images(meta=meta, files=files, output_dir=output_dir)
                    metadata_entries.append(meta)
                    if sample_id:
                        completed_sample_ids.add(sample_id)

                    written_samples += 1
                    for path in written_paths:
                        written_images += 1
                        total_label = (
                            expected_total_images
                            if expected_total_images > 0 and written_images <= expected_total_images
                            else "?"
                        )
                        print(
                            f"[{written_images}/{total_label}][{_worker_label(worker_index)}] "
                            f"wrote {path.relative_to(output_dir)}",
                            flush=True,
                        )

                    flush_state()

            async def _open_generation_page(worker_index: int) -> Any:
                page = await browser.new_page(viewport={"width": 1440, "height": 1400})
                page.set_default_timeout(0)

                async def _progress_binding(source: Any, payload: Dict[str, Any]) -> None:
                    await _handle_stream_progress(worker_index, source, payload)

                async def _sample_binding(source: Any, payload: Dict[str, Any]) -> None:
                    await _handle_stream_sample(worker_index, source, payload)

                await page.expose_binding(
                    "__codex_stream_progress",
                    _progress_binding,
                )
                await page.expose_binding(
                    "__codex_stream_sample",
                    _sample_binding,
                )
                await page.goto(server.base_url, wait_until="networkidle")
                await page.wait_for_function(
                    "() => !!window.topoBench && "
                    "typeof window.topoBench.generateFullDataset === 'function'"
                )
                # Vite can trigger a one-off reload right after initial readiness.
                # Give it a short settle window so the long-running evaluate is not interrupted.
                await page.wait_for_timeout(1000)
                await page.wait_for_function(
                    "() => !!window.topoBench && "
                    "typeof window.topoBench.generateFullDataset === 'function'"
                )
                return page

            max_retries = max(0, int(getattr(args, "max_retries", 12)))

            async def _run_generation_worker(worker_index: int) -> Dict[str, Any]:
                page = await _open_generation_page(worker_index)
                eval_config = {
                    "seed": args.seed,
                    "variantsPerType": variants_per_type,
                    "anglesPerVariant": angles_per_variant,
                    "numPairs": num_pairs,
                    "renderWidth": image_size,
                    "renderHeight": image_size,
                    "retainSamples": False,
                    "shardIndex": worker_index,
                    "shardCount": workers,
                }
                for attempt in range(max_retries + 1):
                    async with state_lock:
                        eval_config["skipSampleIds"] = sorted(completed_sample_ids)
                    try:
                        return await page.evaluate(
                            """async (config) => {
                                const progressCb = async (current, total, message) => {
                                    if (typeof window.__codex_stream_progress === 'function') {
                                        await window.__codex_stream_progress({ current, total, message });
                                    }
                                };
                                const sampleCb = async (sample) => {
                                    if (typeof window.__codex_stream_sample === 'function') {
                                        await window.__codex_stream_sample(sample);
                                    }
                                };
                                const dataset = await window.topoBench.generateFullDataset(config, progressCb, sampleCb);
                                return {
                                    stats: dataset.stats || {},
                                    singleCount: (dataset.singleMetadata || []).length,
                                    pairCount: (dataset.pairMetadata || []).length,
                                };
                            }""",
                            eval_config,
                        )
                    except Exception as exc:
                        if attempt >= max_retries:
                            raise
                        print(
                            f"[retry {attempt + 1}/{max_retries}][{_worker_label(worker_index)}] "
                            f"page evaluate failed after {written_samples} samples / {written_images} images; "
                            f"reopening page. cause: {str(exc).splitlines()[0]}",
                            flush=True,
                        )
                        try:
                            await page.close()
                        except Exception:
                            pass
                        page = await _open_generation_page(worker_index)
                raise RuntimeError("Frontend generation did not return a payload.")

            print("Generating dataset in the frontend...", flush=True)
            try:
                browser_payloads = await asyncio.gather(
                    *(_run_generation_worker(worker_index) for worker_index in range(workers))
                )
            finally:
                await browser.close()
            stats_state = dict((browser_payloads[0] or {}).get("stats") or {})

    expected_total_records = int(stats_state.get("expectedSingleSampleCount") or 0) + int(stats_state.get("expectedPairSampleCount") or 0)
    if expected_total_records and written_samples != expected_total_records:
        raise ValueError(
            f"Streamed sample count mismatch: wrote {written_samples}, expected {expected_total_records}"
        )

    expected_images_from_metadata = sum(
        2 if "label_equivalent" in sample else len(sample.get("images") or [])
        for sample in metadata_entries
    )
    if expected_images_from_metadata and written_images != expected_images_from_metadata:
        raise ValueError(
            f"Streamed image count mismatch: wrote {written_images}, expected {expected_images_from_metadata}"
        )

    if stats_state:
        stats_state["totalImages"] = written_images
        stats_state["grandTotalImages"] = written_images
        stats_state["workers"] = workers

    metadata_payload = flush_state()

    samples_for_questions = list(metadata_payload.get("samples") or [])
    question_rows = build_question_rows(samples=samples_for_questions, output_dir=output_dir)
    question_rows = _deduplicate_question_rows(question_rows, output_dir=output_dir)
    questions_per_difficulty = getattr(args, "questions_per_difficulty", None)
    max_questions = getattr(args, "max_questions", None)
    if questions_per_difficulty is not None and questions_per_difficulty > 0:
        question_rows = _stratified_truncate_by_difficulty_task(
            question_rows,
            questions_per_difficulty,
        )
    elif max_questions is not None and max_questions > 0:
        question_rows = _stratified_truncate_by_task(question_rows, max_questions)
    reorganize_images_by_question_id(
        question_rows,
        output_dir=output_dir,
        image_root_name=DEFAULT_IMAGE_ROOT_NAME,
        remove_unreferenced_dirs=False,
    )
    missing_question_images = _missing_question_images(question_rows, output_dir=output_dir)
    if missing_question_images:
        preview = "; ".join(missing_question_images[:5])
        raise ValueError(f"question.jsonl would reference missing image(s): {preview}")
    question_path = output_dir / "question.jsonl"
    write_jsonl(question_path, question_rows)

    return {
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "question_path": str(question_path),
        "question_count": len(question_rows),
        "image_count": written_images,
        "removed_files": removed_files,
        "other_files": 0,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = asyncio.run(generate_samples_async(args))
    print(f"Wrote {payload['image_count']} image(s) to {payload['output_dir']}")
    print(f"Metadata: {payload['metadata_path']}")
    print(f"Summary: {payload['summary_path']}")
    print(f"Question dataset: {payload['question_path']} ({payload['question_count']} rows)")
    if args.clean:
        print(f"Removed {payload['removed_files']} previous generated file(s)")


if __name__ == "__main__":
    main()
