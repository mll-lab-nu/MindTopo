from __future__ import annotations

import argparse
import asyncio
import base64
import json
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
DIFFICULTY_DIR = {"easy": "difficulty_1", "medium": "difficulty_2", "hard": "difficulty_3"}
DIFFICULTY_NUM = {"easy": 1, "medium": 2, "hard": 3}
DEFAULT_PAIR_SEED = 12345

from jsonl_export import reorganize_images_by_question_id, to_relative_path, write_jsonl  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the order_bead_string dataset through the frontend batch API."
    )
    parser.add_argument("--seed", default="12345")
    parser.add_argument(
        "--samples-per-difficulty",
        type=int,
        default=50,
        help="Number of base samples to generate for each requested difficulty bucket.",
    )
    parser.add_argument(
        "--difficulty",
        default="easy,medium,hard",
        help="Comma-separated difficulty buckets.",
    )
    parser.add_argument(
        "--cyclic-pairs",
        type=int,
        default=15,
        help="Number of additional cyclic-rotation samples to synthesize.",
    )
    parser.add_argument(
        "--pair-variants-per-relation",
        type=int,
        default=0,
        help=(
            "Number of source-linked synthetic images to generate for each "
            "pair-relation class per difficulty. These images are used only "
            "for pair questions."
        ),
    )
    parser.add_argument(
        "--pair-rows-per-difficulty",
        type=int,
        default=None,
        help=(
            "If set, emit this many T_BS02 pair questions per difficulty, "
            "balanced across IDENTICAL/REVERSED/CYCLIC_ROTATION/DIFFERENT."
        ),
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
        "--pair-seed",
        type=int,
        default=DEFAULT_PAIR_SEED,
        help="Seed for deterministic pair-task sampling in question.jsonl.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Playwright pages used to render base samples. "
             "Cyclic-pair generation always runs on a single page afterwards.",
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


def data_url_to_png_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Expected a PNG data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


def clear_output(image_root: Path) -> int:
    removed = 0
    if not image_root.exists():
        return removed
    for image_path in image_root.rglob("*.png"):
        image_path.unlink()
        removed += 1
    return removed


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
    pair_seed: int = DEFAULT_PAIR_SEED,
    pair_rows_per_difficulty: int | None = None,
) -> List[Dict[str, Any]]:
    """Expand raw samples into question.jsonl rows for every task in vlm_benchmark_bead.TASKS.

    Single-image tasks → 1 row per sample.
    Pair tasks → rows from generate_pairs(seed=pair_seed) (deterministic).
    """
    from vlm_benchmark_bead import TASKS, build_prompt, generate_pairs, get_ground_truth

    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    rows: List[Dict[str, Any]] = []
    sample_index = 0
    phrasing_idx = 0

    for task_id, task_def in TASKS.items():
        if task_def.get("input_type") != "single":
            continue
        prompt = build_prompt(task_id, phrasing_idx)
        for sample in samples:
            if sample.get("_pair_variant"):
                continue
            gt = get_ground_truth(task_id, sample)
            if gt is None:
                continue
            difficulty = _normalize_difficulty(sample.get("difficulty"))
            difficulty_num = DIFFICULTY_NUM.get(difficulty, difficulty)
            row_id = f"order_bead_string_difficulty_{difficulty_num}_{task_id}_{sample_index:04d}"
            rows.append(
                {
                    "id": row_id,
                    "category": ["order", "order_bead_string", task_id],
                    "type": task_id,
                    "meta_info": {
                        "task_name": "order_bead_string",
                        "config": config_rel,
                        "seed": _normalize_seed(sample.get("seed")),
                        "repeat_index": 0,
                        "difficulty": difficulty,
                        "phrasing_index": phrasing_idx,
                        "num_beads": int(sample.get("num_beads", 0)),
                    },
                    "question": prompt,
                    "images": [f"{DEFAULT_IMAGE_ROOT_NAME}/{sample['filename']}"],
                    "gt_answer": str(gt),
                }
            )
            sample_index += 1

    for task_id, task_def in TASKS.items():
        if task_def.get("input_type") != "pair":
            continue
        prompt = build_prompt(task_id, phrasing_idx)
        pairs = generate_pairs(
            list(samples),
            task_id,
            seed=pair_seed,
            max_pairs_per_difficulty=pair_rows_per_difficulty,
            balance_by_relation=pair_rows_per_difficulty is not None,
        )
        for sa, sb, pair_info in pairs:
            gt = get_ground_truth(task_id, sa, pair_sample=sb, pair_info=pair_info)
            if gt is None:
                continue
            difficulty = _normalize_difficulty(sa.get("difficulty"))
            difficulty_num = DIFFICULTY_NUM.get(difficulty, difficulty)
            row_id = f"order_bead_string_difficulty_{difficulty_num}_{task_id}_{sample_index:04d}"
            rows.append(
                {
                    "id": row_id,
                    "category": ["order", "order_bead_string", task_id],
                    "type": task_id,
                    "meta_info": {
                        "task_name": "order_bead_string",
                        "config": config_rel,
                        "seed": _normalize_seed(sa.get("seed")),
                        "repeat_index": 0,
                        "difficulty": difficulty,
                        "phrasing_index": phrasing_idx,
                        "num_beads": int(sa.get("num_beads", 0)),
                        "pair_partner_seed": _normalize_seed(sb.get("seed")),
                        "pair_relation": str(gt),
                        "pair_variant": bool(pair_info.get("_pair_variant")),
                    },
                    "question": prompt,
                    "images": [
                        f"{DEFAULT_IMAGE_ROOT_NAME}/{sa['filename']}",
                        f"{DEFAULT_IMAGE_ROOT_NAME}/{sb['filename']}",
                    ],
                    "gt_answer": str(gt),
                }
            )
            sample_index += 1

    return rows


def build_summary(
    *,
    metadata_entries: Sequence[Dict[str, Any]],
    dataset_payload: Dict[str, Any],
    seed: str,
    difficulties: Sequence[str],
    image_size: int,
    samples_per_difficulty: int,
    cyclic_pairs: int,
    pair_variants_per_relation: int,
    pair_rows_per_difficulty: int | None,
) -> Dict[str, Any]:
    return {
        "totalSamples": len(metadata_entries),
        "seed": str(seed),
        "difficulties": list(difficulties),
        "samplesPerDifficulty": int(samples_per_difficulty),
        "cyclicPairsRequested": int(cyclic_pairs),
        "pairVariantsPerRelation": int(pair_variants_per_relation),
        "pairRowsPerDifficulty": pair_rows_per_difficulty,
        "imageSize": int(image_size),
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stats": dataset_payload.get("stats", {}),
        "breakdown": {
            difficulty: sum(1 for item in metadata_entries if item.get("difficulty") == difficulty)
            for difficulty in difficulties
        },
    }


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

_CYCLIC_EVAL = """async (input) => {
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
    return await window.topoBench.generateCyclicPairs(input.config, input.sources, progressCb, sampleCb);
}"""

_PAIR_VARIANT_EVAL = """async (input) => {
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
    return await window.topoBench.generatePairVariants(input.config, input.sources, progressCb, sampleCb);
}"""


async def generate_samples_async(args: argparse.Namespace) -> Dict[str, Any]:
    difficulties = parse_choice_list(args.difficulty, allowed=ALLOWED_DIFFICULTIES, label="difficulty")
    samples_per_difficulty = max(1, int(args.samples_per_difficulty))
    cyclic_pairs = max(0, int(args.cyclic_pairs))
    pair_variants_per_relation = max(0, int(args.pair_variants_per_relation))
    pair_rows_per_difficulty = (
        None if args.pair_rows_per_difficulty is None else max(0, int(args.pair_rows_per_difficulty))
    )
    image_size = max(1, int(args.image_size))
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    # Can't usefully shard finer than the total work — each shard renders
    # `i % numWorkers == workerIdx` per difficulty, so extra workers just sit idle.
    workers = min(workers, samples_per_difficulty)
    requested_pair_variants = pair_variants_per_relation * len(difficulties) * 4
    requested_total = samples_per_difficulty * len(difficulties) + cyclic_pairs + requested_pair_variants

    output_dir = Path(args.output_dir).resolve()
    image_root = output_dir / DEFAULT_IMAGE_ROOT_NAME
    metadata_path = output_dir / "dataset_metadata.json"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    removed_images = clear_output(image_root) if args.clean else 0
    metadata_entries: List[Dict[str, Any]] = []
    progress_state: dict[str, Any] = {"last_stage": None}

    print(
        f"Starting streamed generation for ~{requested_total} sample(s) "
        f"({samples_per_difficulty} per difficulty, cyclic={cyclic_pairs}, "
        f"pair_variants={requested_pair_variants}, "
        f"size={image_size}, workers={workers})",
        flush=True,
    )

    async def _handle_stream_sample(_source: Any, payload: dict[str, Any]) -> None:
        sample = dict(payload)
        image_data_url = sample.pop("image_data_url", None)
        filename = sample.get("filename")
        if not image_data_url or not filename:
            raise ValueError(f"Streamed sample is missing image data or filename: {sample}")
        sample["raw_filename"] = filename
        difficulty = sample.get("difficulty", "")
        diff_dir = DIFFICULTY_DIR.get(difficulty, f"difficulty_{difficulty}")
        file_stem = Path(filename).stem
        sample_folder = f"order_bead_string_{diff_dir}_{file_stem}"
        image_path = image_root / diff_dir / sample_folder / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data_url_to_png_bytes(image_data_url))
        sample["filename"] = f"{diff_dir}/{sample_folder}/{filename}"
        metadata_entries.append(sample)
        print(f"[{len(metadata_entries)}/{requested_total}] wrote {sample['filename']}", flush=True)

    async def _handle_stream_progress(_source: Any, payload: dict[str, Any]) -> None:
        message = str(payload.get("message") or "").strip()
        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        if not message:
            return
        stage = message
        last_stage = progress_state["last_stage"]
        should_print = stage != last_stage
        if current and (current == 1 or current % 25 == 0):
            should_print = True
        if current and total and current == total:
            should_print = True
        if should_print:
            progress_state["last_stage"] = stage
            suffix = f" ({current}/{total})" if total else ""
            print(f"[progress] {message}{suffix}", flush=True)

    async def _setup_page(browser: Any) -> Any:
        page = await browser.new_page(viewport={"width": 1440, "height": 1400})
        await page.expose_binding("__codex_stream_sample", _handle_stream_sample)
        await page.expose_binding("__codex_stream_progress", _handle_stream_progress)
        await page.goto(server.base_url, wait_until="networkidle")
        await page.wait_for_function(
            "() => !!window.topoBench "
            "&& typeof window.topoBench.generateBaseSamplesShard === 'function' "
            "&& typeof window.topoBench.generateCyclicPairs === 'function' "
            "&& typeof window.topoBench.generatePairVariants === 'function'"
        )
        return page

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=args.headless)

            # Open one page per worker. Each page has its own three.js scene
            # and its own WebGL context, so they render independently.
            pages = await asyncio.gather(*(_setup_page(browser) for _ in range(workers)))

            base_cfg_common = {
                "seed": args.seed,
                "samplesPerDifficulty": samples_per_difficulty,
                "renderWidth": image_size,
                "renderHeight": image_size,
                "difficulties": difficulties,
                "numWorkers": workers,
            }
            shard_tasks = [
                pages[idx].evaluate(_SHARD_EVAL, {**base_cfg_common, "workerIdx": idx})
                for idx in range(workers)
            ]
            shard_results = await asyncio.gather(*shard_tasks)

            sample_summaries: List[Dict[str, Any]] = []
            difficulty_counts = {level: 0 for level in ALLOWED_DIFFICULTIES}
            for result in shard_results:
                sample_summaries.extend(result.get("sampleSummaries", []) or [])
                stats = result.get("stats") or {}
                for level in difficulty_counts:
                    difficulty_counts[level] += int(stats.get(level, 0) or 0)

            cyclic_emitted = 0
            if cyclic_pairs > 0 and sample_summaries:
                cyclic_payload = await pages[0].evaluate(
                    _CYCLIC_EVAL,
                    {
                        "config": {
                            "seed": args.seed,
                            "cyclicPairs": cyclic_pairs,
                            "renderWidth": image_size,
                            "renderHeight": image_size,
                        },
                        "sources": sample_summaries,
                    },
                )
                cyclic_emitted = int((cyclic_payload or {}).get("cyclic_pairs", 0) or 0)

            pair_variants_emitted = 0
            pair_variant_payload: Dict[str, Any] = {}
            if pair_variants_per_relation > 0 and sample_summaries:
                pair_variant_payload = await pages[0].evaluate(
                    _PAIR_VARIANT_EVAL,
                    {
                        "config": {
                            "seed": args.seed,
                            "variantsPerRelation": pair_variants_per_relation,
                            "renderWidth": image_size,
                            "renderHeight": image_size,
                            "difficulties": difficulties,
                        },
                        "sources": sample_summaries,
                    },
                )
                pair_variants_emitted = int((pair_variant_payload or {}).get("pair_variants", 0) or 0)

            await asyncio.gather(*(page.close() for page in pages))
            await browser.close()

    base_total = sum(difficulty_counts.values())
    dataset_payload: Dict[str, Any] = {
        "version": 2,
        "task": "bead_string",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stats": {
            "total": base_total + cyclic_emitted + pair_variants_emitted,
            **{level: difficulty_counts.get(level, 0) for level in ALLOWED_DIFFICULTIES},
            "cyclic_pairs": cyclic_emitted,
            "pair_variants": pair_variants_emitted,
            "pair_variants_by_relation": pair_variant_payload.get("by_relation", {}),
        },
    }

    actual_total = len(metadata_entries)
    reported_total = int(dataset_payload["stats"]["total"])
    if actual_total != reported_total:
        raise ValueError(
            f"Streamed sample count mismatch: wrote {actual_total}, "
            f"but frontend reported {reported_total}"
        )

    metadata_payload = {
        **{k: v for k, v in dataset_payload.items() if k != "samples"},
        "samples": metadata_entries,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = build_summary(
        metadata_entries=metadata_entries,
        dataset_payload=metadata_payload,
        seed=str(args.seed),
        difficulties=difficulties,
        image_size=image_size,
        samples_per_difficulty=samples_per_difficulty,
        cyclic_pairs=cyclic_pairs,
        pair_variants_per_relation=pair_variants_per_relation,
        pair_rows_per_difficulty=pair_rows_per_difficulty,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    question_rows = build_question_rows(
        samples=metadata_entries,
        output_dir=output_dir,
        pair_seed=int(args.pair_seed),
        pair_rows_per_difficulty=pair_rows_per_difficulty,
    )
    reorganize_images_by_question_id(
        question_rows,
        output_dir=output_dir,
        image_root_name=DEFAULT_IMAGE_ROOT_NAME,
    )
    question_path = output_dir / "question.jsonl"
    write_jsonl(question_path, question_rows)

    return {
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "question_path": str(question_path),
        "question_count": len(question_rows),
        "sample_count": len(metadata_entries),
        "removed_images": removed_images,
        "seed": str(args.seed),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = asyncio.run(generate_samples_async(args))
    print(f"Wrote {payload['sample_count']} samples to {payload['output_dir']}")
    print(f"Metadata: {payload['metadata_path']}")
    print(f"Summary: {payload['summary_path']}")
    print(f"Question dataset: {payload['question_path']} ({payload['question_count']} rows)")
    if args.clean:
        print(f"Removed {payload['removed_images']} previous image(s)")


if __name__ == "__main__":
    main()
