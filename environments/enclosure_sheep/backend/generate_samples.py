from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from vite_server import ViteFrontendServer

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "generate_samples.py requires the `playwright` package. Install it with: pip install playwright"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_IMAGE_ROOT_NAME = "images"
ALLOWED_DIFFICULTIES = ("easy", "medium", "hard")
ALLOWED_SCENE_TYPES = ("fence", "partitioned")
DIFFICULTY_DIR = {"easy": "difficulty_1", "medium": "difficulty_2", "hard": "difficulty_3"}
DIFFICULTY_NUM = {"easy": 1, "medium": 2, "hard": 3}

from jsonl_export import reorganize_images_by_question_id, to_relative_path, write_jsonl  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the enclosure_sheep dataset through the frontend batch API.")
    parser.add_argument("--seed", default="12345", help="Dataset generation seed.")
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of samples to generate for each requested difficulty and scene type.",
    )
    parser.add_argument(
        "--difficulty",
        default="easy,medium,hard",
        help="Comma-separated difficulty buckets.",
    )
    parser.add_argument(
        "--scene-types",
        default="fence,partitioned",
        help="Comma-separated scene types.",
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
        help="Directory where images, dataset_metadata.json, and summary.json will be written.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help="Remove previously generated PNGs in the output images directory before writing new data.",
    )
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help=(
            "Skip rendering and rebuild question.jsonl from an existing "
            "dataset_metadata.json in --output-dir. Useful for resuming after "
            "a failure during the final JSONL export step."
        ),
    )
    parser.add_argument(
        "--questions-per-difficulty",
        type=int,
        default=None,
        help=(
            "If set, stratified-sample this many final question rows per "
            "difficulty, balancing task type and ground-truth buckets."
        ),
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help=(
            "If set, stratified-sample exactly this many final question rows "
            "after any per-difficulty sampling."
        ),
    )
    parser.add_argument(
        "--task-quotas",
        default="",
        help=(
            "Optional comma-separated final question quotas by task, e.g. "
            "Q1_count_inside=450,Q2_escape_possibility=270,Q4_fence_repair=230,Q3_max_cell_count=50."
        ),
    )
    parser.add_argument(
        "--question-seed",
        type=int,
        default=12345,
        help="RNG seed used only for stratified final-question sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel Playwright pages used to render base samples. "
            "Each worker handles configs where `i %% numWorkers == workerIdx`."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "If > 0, render base samples in global-index chunks, reopening "
            "Playwright pages between chunks to avoid long-lived WebGL pages."
        ),
    )
    parser.set_defaults(headless=True, clean=True)
    return parser


def parse_choice_list(spec: str, *, allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    values: List[str] = []
    seen: set[str] = set()
    for chunk in spec.split(","):
        value = chunk.strip().lower()
        if not value:
            continue
        if value not in allowed_set:
            raise ValueError(f"Unsupported {label}: {value!r}. Allowed values: {', '.join(allowed)}")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"At least one {label} is required.")
    return values


def parse_task_quotas(spec: str) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid task quota {chunk!r}; expected TASK=COUNT.")
        key, value = chunk.split("=", 1)
        task_id = key.strip()
        if not task_id:
            raise ValueError(f"Invalid task quota {chunk!r}; empty task id.")
        count = int(value.strip())
        if count < 0:
            raise ValueError(f"Invalid task quota {chunk!r}; count must be non-negative.")
        quotas[task_id] = count
    return quotas


def data_url_to_png_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Expected a PNG data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


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


def build_question_rows(
    *,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Expand raw samples into question.jsonl rows for every task in vlm_benchmark_sheep.TASKS.

    Each (task, sample) pair is emitted only when the sample's sceneType is in the
    task's `scene_types` whitelist and ground truth is computable.
    """
    from vlm_benchmark_sheep import TASKS, build_prompt, get_ground_truth

    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    rows: List[Dict[str, Any]] = []
    sample_index = 0
    phrasing_idx = 0

    for task_id, task_def in TASKS.items():
        scene_whitelist = set(task_def.get("scene_types", []))
        prompt = build_prompt(task_id, phrasing_idx)
        for sample in samples:
            scene_type = str(sample.get("sceneType", ""))
            if scene_type not in scene_whitelist:
                continue
            try:
                gt = get_ground_truth(task_id, sample)
            except (KeyError, TypeError):
                continue
            if gt is None:
                continue
            difficulty = _normalize_difficulty(sample.get("difficulty"))
            difficulty_num = DIFFICULTY_NUM.get(difficulty, difficulty)
            row_id = f"enclosure_sheep_difficulty_{difficulty_num}_{task_id}_{sample_index:04d}"
            image_file = str(sample.get("imageFile") or "")
            if not image_file:
                continue
            rows.append(
                {
                    "id": row_id,
                    "category": ["enclosure", "enclosure_sheep", task_id],
                    "type": task_id,
                    "meta_info": {
                        "task_name": "enclosure_sheep",
                        "config": config_rel,
                        "seed": _normalize_seed(sample.get("seed")),
                        "repeat_index": int(sample.get("repeatIndex", 0)),
                        "difficulty": difficulty,
                        "scene_type": scene_type,
                        "phrasing_index": phrasing_idx,
                    },
                    "question": prompt,
                    "images": [f"{DEFAULT_IMAGE_ROOT_NAME}/{image_file}"],
                    "gt_answer": str(gt),
                }
            )
            sample_index += 1

    return rows


def _missing_question_images(rows: Sequence[Dict[str, Any]], *, output_dir: Path) -> List[str]:
    missing: List[str] = []
    for row in rows:
        for image in row.get("images") or []:
            if not (output_dir / str(image)).exists():
                missing.append(f"{row.get('id')}: {image}")
    return missing


def _id_list_length(value: Any) -> int:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "[]", "empty"}:
        return 0
    return len([part for part in text.split(",") if part.strip()])


def _gt_bucket(row: Dict[str, Any]) -> str:
    task_id = str(row.get("type") or "")
    gt = str(row.get("gt_answer") or "").strip()
    if task_id == "Q1_count_inside":
        try:
            count = int(gt)
        except ValueError:
            return "count:other"
        # Keep exact count buckets for Q1. Coarse buckets like 0 / 1-2 / 3+
        # made small balanced exports repeatedly choose easy answers near 0-2
        # even when the generated pool contained many higher-count scenes.
        return f"count:{count}"
    if task_id == "Q2_escape_possibility":
        length = _id_list_length(gt)
        if length == 0:
            return "ids:0"
        if length <= 2:
            return "ids:1-2"
        if length <= 5:
            return "ids:3-5"
        return "ids:6+"
    if task_id == "Q4_fence_repair":
        try:
            gaps = int(gt)
        except ValueError:
            return "gaps:other"
        if gaps == 0:
            return "gaps:0"
        if gaps <= 2:
            return "gaps:1-2"
        return "gaps:3+"
    return f"gt:{gt.upper()}"


def _round_robin_take(rows: Sequence[Dict[str, Any]], target: int, *, seed: int) -> List[Dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(_gt_bucket(row), []).append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda row: str(row.get("id") or ""))
        rng.shuffle(bucket_rows)
    buckets = sorted(by_bucket)
    rng.shuffle(buckets)
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < target:
        progressed = False
        for bucket in buckets:
            bucket_rows = by_bucket[bucket]
            while bucket_rows:
                row = bucket_rows.pop(0)
                row_id = str(row.get("id") or "")
                if row_id in seen:
                    continue
                selected.append(row)
                seen.add(row_id)
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break
    if len(selected) < target:
        remaining = [row for row in rows if str(row.get("id") or "") not in seen]
        remaining.sort(key=lambda row: str(row.get("id") or ""))
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])
    return selected[:target]


def stratified_question_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    per_difficulty: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    if per_difficulty is None:
        return list(rows)
    target = max(1, int(per_difficulty))
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    for index, difficulty in enumerate(ALLOWED_DIFFICULTIES):
        diff_rows = [
            row for row in rows
            if _normalize_difficulty((row.get("meta_info") or {}).get("difficulty")) == difficulty
        ]
        for row in _stratified_take_question_rows(
            diff_rows,
            target,
            seed=seed + index * 101,
        ):
            row_id = str(row.get("id") or "")
            if row_id and row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row_id)

    order = {str(row.get("id") or ""): index for index, row in enumerate(rows)}
    selected.sort(key=lambda row: order.get(str(row.get("id") or ""), 0))
    return selected


def _stratified_take_question_rows(
    rows: Sequence[Dict[str, Any]],
    target: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    task_order = ["Q2_escape_possibility", "Q1_count_inside", "Q3_max_cell_count", "Q4_fence_repair"]
    present_tasks = [task for task in task_order if any(row.get("type") == task for row in rows)]
    if not present_tasks:
        return []
    base_quota = target // len(present_tasks)
    remainder = target % len(present_tasks)
    selected: List[Dict[str, Any]] = []
    for index, task_id in enumerate(present_tasks):
        quota = base_quota + (1 if index < remainder else 0)
        task_rows = [row for row in rows if row.get("type") == task_id]
        selected.extend(
            _round_robin_take(task_rows, quota, seed=seed + index)
        )
    if len(selected) < target:
        used = {str(row.get("id") or "") for row in selected}
        remaining = [row for row in rows if str(row.get("id") or "") not in used]
        selected.extend(
            _round_robin_take(remaining, target - len(selected), seed=seed + 97)
        )
    return selected[:target]


def limit_question_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
    task_quotas: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    if limit is None:
        return list(rows)
    target = max(1, int(limit))
    if len(rows) < target:
        raise ValueError(f"Cannot sample {target} question rows from only {len(rows)} generated rows.")
    if task_quotas:
        quota_total = sum(task_quotas.values())
        if quota_total != target:
            raise ValueError(f"--task-quotas sum to {quota_total}, expected --question-limit {target}.")
        return _limit_question_rows_by_task_quota(
            rows,
            task_quotas=task_quotas,
            seed=seed,
        )

    # Put the remainder on harder buckets first so exact limits like 1000
    # become hard=334, medium=333, easy=333 rather than over-weighting easy.
    difficulty_order = ["hard", "medium", "easy"]
    by_difficulty = {
        difficulty: [
            row for row in rows
            if _normalize_difficulty((row.get("meta_info") or {}).get("difficulty")) == difficulty
        ]
        for difficulty in difficulty_order
    }
    present_difficulties = [difficulty for difficulty in difficulty_order if by_difficulty[difficulty]]
    if not present_difficulties:
        return list(rows)[:target]

    base_quota = target // len(present_difficulties)
    remainder = target % len(present_difficulties)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for index, difficulty in enumerate(present_difficulties):
        quota = base_quota + (1 if index < remainder else 0)
        for row in _stratified_take_question_rows(
            by_difficulty[difficulty],
            quota,
            seed=seed + index * 101,
        ):
            row_id = str(row.get("id") or "")
            if row_id and row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row_id)

    if len(selected) < target:
        remaining = [row for row in rows if str(row.get("id") or "") not in selected_ids]
        selected.extend(_round_robin_take(remaining, target - len(selected), seed=seed + 991))

    order = {str(row.get("id") or ""): index for index, row in enumerate(rows)}
    selected = selected[:target]
    selected.sort(key=lambda row: order.get(str(row.get("id") or ""), 0))
    return selected


def _take_single_task_rows(
    rows: Sequence[Dict[str, Any]],
    target: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    difficulty_order = ["hard", "medium", "easy"]
    by_difficulty = {
        difficulty: [
            row for row in rows
            if _normalize_difficulty((row.get("meta_info") or {}).get("difficulty")) == difficulty
        ]
        for difficulty in difficulty_order
    }
    present = [difficulty for difficulty in difficulty_order if by_difficulty[difficulty]]
    if not present:
        return []
    base_quota = target // len(present)
    remainder = target % len(present)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for index, difficulty in enumerate(present):
        quota = base_quota + (1 if index < remainder else 0)
        for row in _round_robin_take(by_difficulty[difficulty], quota, seed=seed + index * 101):
            row_id = str(row.get("id") or "")
            if row_id and row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row_id)
    if len(selected) < target:
        remaining = [row for row in rows if str(row.get("id") or "") not in selected_ids]
        selected.extend(_round_robin_take(remaining, target - len(selected), seed=seed + 991))
    return selected[:target]


def _limit_question_rows_by_task_quota(
    rows: Sequence[Dict[str, Any]],
    *,
    task_quotas: Dict[str, int],
    seed: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for index, task_id in enumerate(sorted(task_quotas)):
        quota = task_quotas[task_id]
        task_rows = [row for row in rows if row.get("type") == task_id]
        if len(task_rows) < quota:
            raise ValueError(f"Cannot sample {quota} rows for {task_id}; only {len(task_rows)} generated.")
        for row in _take_single_task_rows(task_rows, quota, seed=seed + index * 1009):
            row_id = str(row.get("id") or "")
            if row_id and row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row_id)

    target = sum(task_quotas.values())
    if len(selected) < target:
        remaining = [row for row in rows if str(row.get("id") or "") not in selected_ids]
        selected.extend(_round_robin_take(remaining, target - len(selected), seed=seed + 1999))

    order = {str(row.get("id") or ""): index for index, row in enumerate(rows)}
    selected = selected[:target]
    selected.sort(key=lambda row: order.get(str(row.get("id") or ""), 0))
    return selected


def build_summary(
    *,
    metadata_entries: Sequence[Dict[str, Any]],
    seed: str,
    count: int,
    difficulties: Sequence[str],
    scene_types: Sequence[str],
    image_size: int,
) -> Dict[str, Any]:
    return {
        "totalSamples": len(metadata_entries),
        "seed": str(seed),
        "sceneTypes": list(scene_types),
        "difficulties": list(difficulties),
        "samplesPerDifficulty": int(count),
        "imageSize": int(image_size),
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "breakdown": {
            difficulty: sum(1 for item in metadata_entries if item.get("difficulty") == difficulty)
            for difficulty in difficulties
        },
        "byType": {
            scene_type: sum(1 for item in metadata_entries if item.get("sceneType") == scene_type)
            for scene_type in scene_types
        },
    }


def clear_output(image_root: Path) -> int:
    removed = 0
    if not image_root.exists():
        return removed
    for image_path in image_root.rglob("*.png"):
        image_path.unlink()
        removed += 1
    return removed


SEEDED_RANDOM_SCRIPT = r"""
(() => {
  function hashSeed(value) {
    let h = 2166136261 >>> 0;
    const text = String(value);
    for (let i = 0; i < text.length; i += 1) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function random() {
      t = (t + 0x6D2B79F5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  window.__topobenchSeedMathRandom = (seedValue) => {
    Math.random = mulberry32(hashSeed(seedValue));
  };
})();
"""


async def _seed_page_random(page, seed: str) -> None:
    await page.evaluate(SEEDED_RANDOM_SCRIPT)
    await page.evaluate("(seedValue) => window.__topobenchSeedMathRandom(seedValue)", str(seed))


_SHARD_EVAL = """async (config) => {
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
    return await window.topoBench.generateBaseSamplesShard(config, progressCb, sampleCb);
}"""


async def generate_samples_async(args: argparse.Namespace) -> Dict[str, Any]:
    difficulties = parse_choice_list(args.difficulty, allowed=ALLOWED_DIFFICULTIES, label="difficulty")
    scene_types = parse_choice_list(args.scene_types, allowed=ALLOWED_SCENE_TYPES, label="scene type")
    dataset_seed = str(args.seed)
    count = max(1, int(args.count))
    image_size = max(1, int(args.image_size))
    task_quotas = parse_task_quotas(getattr(args, "task_quotas", ""))

    output_dir = Path(args.output_dir).resolve()
    image_root = output_dir / DEFAULT_IMAGE_ROOT_NAME
    metadata_path = output_dir / "dataset_metadata.json"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    if args.questions_only:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"--questions-only requires existing metadata at {metadata_path}"
            )
        metadata_entries = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata_entries, list):
            raise ValueError(f"Expected {metadata_path} to contain a JSON list.")

        question_rows = build_question_rows(samples=metadata_entries, output_dir=output_dir)
        question_rows = stratified_question_rows(
            question_rows,
            per_difficulty=args.questions_per_difficulty,
            seed=int(args.question_seed),
        )
        question_rows = limit_question_rows(
            question_rows,
            limit=args.question_limit,
            seed=int(args.question_seed),
            task_quotas=task_quotas,
        )
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
            "image_root": str(image_root),
            "metadata_path": str(metadata_path),
            "summary_path": str(summary_path),
            "question_path": str(question_path),
            "question_count": len(question_rows),
            "question_limit": args.question_limit,
            "sample_count": len(metadata_entries),
            "removed_images": 0,
            "difficulties": difficulties,
            "scene_types": scene_types,
            "seed": dataset_seed,
            "questions_only": True,
        }

    removed_images = clear_output(image_root) if args.clean else 0
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    requested_total = count * len(difficulties) * len(scene_types)
    workers = min(workers, requested_total)
    metadata_entries: List[Dict[str, Any]] = []
    progress_state: Dict[str, Any] = {"last_stage": None}
    state_lock = asyncio.Lock()

    print(
        f"Starting generation for ~{requested_total} sample(s) "
        f"(count={count} per difficulty×scene, difficulties={difficulties}, "
        f"scene_types={scene_types}, size={image_size}, workers={workers})",
        flush=True,
    )

    async def _handle_stream_sample(worker_idx: int, _source: Any, payload: Dict[str, Any]) -> None:
        sample = dict(payload)
        image_data_url = sample.pop("image_data_url", None)
        filename = sample.pop("filename", None)
        if not image_data_url or not filename:
            raise ValueError(f"Streamed sample missing image data or filename: {sample}")
        difficulty = sample.get("difficulty", "")
        diff_dir = DIFFICULTY_DIR.get(difficulty, f"difficulty_{difficulty}")
        sample_id = str(sample.get("id", "sample"))
        sample_folder = f"enclosure_sheep_{diff_dir}_{sample_id}"
        image_path = image_root / diff_dir / sample_folder / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data_url_to_png_bytes(image_data_url))
        sample["imageFile"] = f"{diff_dir}/{sample_folder}/{filename}"
        sample["seed"] = dataset_seed
        async with state_lock:
            metadata_entries.append(sample)
            written = len(metadata_entries)
            if written % 10 == 0 or written == requested_total:
                print(
                    f"[{written}/{requested_total}][worker {worker_idx + 1}/{workers}] "
                    f"wrote {sample['imageFile']}",
                    flush=True,
                )
                metadata_path.write_text(
                    json.dumps(metadata_entries, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    async def _handle_stream_progress(worker_idx: int, _source: Any, payload: Dict[str, Any]) -> None:
        message = str(payload.get("message") or "").strip()
        if not message:
            return
        key = f"{worker_idx}:{message}"
        if key != progress_state["last_stage"]:
            progress_state["last_stage"] = key
            current = int(payload.get("current") or 0)
            total = int(payload.get("total") or 0)
            suffix = f" ({current}/{total})" if total else ""
            print(f"[progress][worker {worker_idx + 1}/{workers}] {message}{suffix}", flush=True)

    async def _setup_page(browser, worker_idx: int):
        page = await browser.new_page(viewport={"width": 1400, "height": 1400})
        page.set_default_timeout(0)

        async def _sample_binding(source, payload):
            await _handle_stream_sample(worker_idx, source, payload)

        async def _progress_binding(source, payload):
            await _handle_stream_progress(worker_idx, source, payload)

        await page.expose_binding("__codex_stream_sample", _sample_binding)
        await page.expose_binding("__codex_stream_progress", _progress_binding)
        await page.goto(server.base_url, wait_until="networkidle")
        await page.wait_for_function(
            "() => !!window.topoBench && "
            "typeof window.topoBench.generateBaseSamplesShard === 'function'"
        )
        # Inject seeded-random helper so the shard can re-seed deterministically.
        await page.evaluate(SEEDED_RANDOM_SCRIPT)
        return page

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=args.headless)
            try:
                chunk_size = int(getattr(args, "chunk_size", 0) or 0)
                if chunk_size <= 0:
                    chunk_size = requested_total
                chunk_size = max(workers, chunk_size)
                for chunk_start in range(0, requested_total, chunk_size):
                    chunk_end = min(requested_total, chunk_start + chunk_size)
                    print(
                        f"[chunk] rendering global configs {chunk_start}-{chunk_end} of {requested_total}",
                        flush=True,
                    )
                    pages = await asyncio.gather(*(_setup_page(browser, idx) for idx in range(workers)))
                    shard_tasks = [
                        pages[idx].evaluate(
                            _SHARD_EVAL,
                            {
                                "seed": dataset_seed,
                                "count": count,
                                "difficulties": difficulties,
                                "sceneTypes": scene_types,
                                "imageSize": image_size,
                                "workerIdx": idx,
                                "numWorkers": workers,
                                "startIndex": chunk_start,
                                "endIndex": chunk_end,
                            },
                        )
                        for idx in range(workers)
                    ]
                    try:
                        await asyncio.gather(*shard_tasks)
                    finally:
                        await asyncio.gather(*(page.close() for page in pages), return_exceptions=True)
            finally:
                await browser.close()

    metadata_path.write_text(json.dumps(metadata_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = build_summary(
        metadata_entries=metadata_entries,
        seed=dataset_seed,
        count=count,
        difficulties=difficulties,
        scene_types=scene_types,
        image_size=image_size,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    question_rows = build_question_rows(samples=metadata_entries, output_dir=output_dir)
    question_rows = stratified_question_rows(
        question_rows,
        per_difficulty=args.questions_per_difficulty,
        seed=int(args.question_seed),
    )
    question_rows = limit_question_rows(
        question_rows,
        limit=args.question_limit,
        seed=int(args.question_seed),
        task_quotas=task_quotas,
    )
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
        "image_root": str(image_root),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "question_path": str(question_path),
        "question_count": len(question_rows),
        "question_limit": args.question_limit,
        "sample_count": len(metadata_entries),
        "removed_images": removed_images,
        "difficulties": difficulties,
        "scene_types": scene_types,
        "seed": dataset_seed,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = asyncio.run(generate_samples_async(args))
    if payload.get("questions_only"):
        print(f"Rebuilt questions from existing metadata in {payload['output_dir']}")
        print(f"Metadata: {payload['metadata_path']}")
        print(f"Question dataset: {payload['question_path']} ({payload['question_count']} rows)")
        return
    print(f"Wrote {payload['sample_count']} samples to {payload['output_dir']}")
    print(f"Metadata: {payload['metadata_path']}")
    print(f"Summary: {payload['summary_path']}")
    print(f"Question dataset: {payload['question_path']} ({payload['question_count']} rows)")
    if args.clean:
        print(f"Removed {payload['removed_images']} previous image(s)")


if __name__ == "__main__":
    main()
