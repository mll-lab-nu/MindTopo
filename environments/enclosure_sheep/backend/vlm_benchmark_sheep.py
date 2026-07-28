#!/usr/bin/env python3
"""
vlm_benchmark_beam.py — Sheep Fence Enclosure VLM Benchmark (v2).

4 tasks:
  - Q1: Count sheep that cannot escape     (fence, INTEGER)
  - Q2: Escape possibility                 (fence, ID_LIST)
  - Q3: Max cell count                     (partitioned, CELL_LABEL)
  - Q4: Fence repair                       (fence, INTEGER)

Usage:
  python vlm_benchmark_beam.py --data_dir ./dataset --tasks all --limit 5
  python vlm_benchmark_beam.py --data_dir ./dataset --tasks Q1_count_inside
  python vlm_benchmark_beam.py --data_dir ./dataset --tasks all --all-phrasings --limit 3

Dependencies:
  pip install openai  # only needed when calling the VLM API

Required env var:
  OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from question_phrasings_sheep import PHRASINGS


# ═══════════════════════════════════════════════════════════════════
# TASK DEFINITIONS — 4 tasks
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    "Q1_count_inside": {
        "input_type": "single",
        "parse_key": "INTEGER",
        "max_tokens": 2048,
        "scene_types": ["fence", "partitioned"],
    },
    "Q2_escape_possibility": {
        "input_type": "single",
        "parse_key": "ID_LIST",
        "max_tokens": 2048,
        "scene_types": ["fence"],
    },
    "Q3_max_cell_count": {
        "input_type": "single",
        "parse_key": "CELL_LABEL",
        "max_tokens": 2048,
        "scene_types": ["partitioned"],
    },
    "Q4_fence_repair": {
        "input_type": "single",
        "parse_key": "INTEGER",
        "max_tokens": 2048,
        "scene_types": ["fence"],
    },
}

assert len(TASKS) == 4, f"Expected 4 tasks, got {len(TASKS)}"

TASK_PROMPT_SPECS = {
    "Q1_count_inside": {
        "task_name": "enclosure.count_cannot_escape",
        "task_category": "enclosure",
        "detailed_description": (
            "look at a 3D scene with fences and numbered sheep, and count how "
            "many sheep cannot escape to the outside"
        ),
        "definition": (
            "A sheep cannot escape if it is trapped inside fences and there "
            "is no continuous path from the sheep to the outside without "
            "crossing a fence segment. Sheep already outside all fences do "
            "not count. Sheep inside a fence with a usable gap to the outside "
            "also do not count."
        ),
        "legal_answer_space_description": (
            "a single JSON integer representing the count of sheep that "
            "cannot escape to the outside, for example 3"
        ),
        "parse_key": "INTEGER",
        "legal_values": None,  # open-ended integer
    },
    "Q2_escape_possibility": {
        "task_name": "enclosure.escape_possibility",
        "task_category": "enclosure",
        "detailed_description": (
            "look at a 3D scene with fences and numbered sheep, and identify "
            "which sheep can escape to the outside through fence gaps"
        ),
        "definition": (
            "An enclosure is a closed region formed by connected fence segments. "
            "A gap is a missing fence segment that creates an opening. A sheep "
            "can escape if there exists a continuous path from that sheep to "
            "the exterior that passes only through gaps and does not cross any "
            "fence segment."
        ),
        "legal_answer_space_description": (
            'the JSON string "NONE" or a single JSON string containing the '
            'escaping sheep IDs in ascending order, separated by commas, for '
            'example "1, 3, 5"'
        ),
        "parse_key": "ID_LIST",
        "legal_values": None,
    },
    "Q3_max_cell_count": {
        "task_name": "enclosure.max_cell_count",
        "task_category": "enclosure",
        "detailed_description": (
            "look at a partitioned enclosure scene where inner fences divide "
            "the enclosure into labeled regions, and determine which region "
            "contains the most sheep"
        ),
        "definition": (
            "A partitioned enclosure is divided into multiple labeled cells by "
            "inner fences. If multiple regions tie for the largest count, any "
            "tied label is acceptable."
        ),
        "legal_answer_space_description": (
            'a single JSON string containing one region label visible in the '
            'image, such as "A", "B", or "C"'
        ),
        "parse_key": "CELL_LABEL",
        "legal_values": None,  # dynamic per image
    },
    "Q4_fence_repair": {
        "task_name": "enclosure.fence_repair",
        "task_category": "enclosure",
        "detailed_description": (
            "look at a 3D scene with a partially broken outermost fence and "
            "determine the minimum number of fence gaps that must be repaired "
            "to fully close it"
        ),
        "definition": (
            "An enclosure is a closed region formed by connected fence segments. "
            "A gap is a missing fence segment that breaks the outermost fence. "
            "Each distinct gap counts as one repair, and the answer is the "
            "minimum number of repairs needed so that the outermost fence "
            "becomes fully closed."
        ),
        "legal_answer_space_description": (
            "a single JSON integer representing the minimum number of fence "
            "gaps that must be repaired, for example 2"
        ),
        "parse_key": "INTEGER",
        "legal_values": None,
    },
}


# ═══════════════════════════════════════════════════════════════════
# PHRASING HELPERS
# ═══════════════════════════════════════════════════════════════════

def num_phrasings(task_id: str) -> int:
    return len(PHRASINGS.get(task_id, []))


def _build_image_block(image_count: int) -> str:
    return "\n".join(
        f"[Image {index}]\nAttached image."
        for index in range(1, image_count + 1)
    )


def build_prompt(task_id: str, phrasing_idx: int, params: dict | None = None) -> str:
    """Build a complete meta prompt for the task."""
    phrasings = PHRASINGS.get(task_id, [])
    if not phrasings:
        return ""
    idx = phrasing_idx % len(phrasings)
    question_text = phrasings[idx]

    if params:
        try:
            question_text = question_text.format(**params)
        except KeyError:
            pass

    spec = TASK_PROMPT_SPECS[task_id]
    return "\n".join(
        [
            "[Task]",
            f"You are solving the {spec['task_category']} task.",
            f"In this task, you must {spec['detailed_description']}.",
            (
                "If this task depends on a specific visual definition, use this "
                f"definition exactly: {spec['definition']}"
            ),
            "The visual evidence for this question is provided below.",
            _build_image_block(1),
            "",
            "[Rules]",
            "1. Use only the images and text provided in this prompt.",
            "2. If answer options are provided, choose only from the provided options.",
            "3. Do not output explanation beyond the required final answer.",
            "",
            "[Question]",
            question_text,
            "",
            "[Answer Format]",
            'Output exactly one JSON object: {"answer": {value}} and nothing else.',
            (
                "Replace {value} with the single legal answer for this task, "
                f"chosen from {spec['legal_answer_space_description']}."
            ),
        ]
    )


# ═══════════════════════════════════════════════════════════════════
# GROUND TRUTH & PARAMETER GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_task_params(task_id: str, sample: dict, seed: int) -> dict:
    """Generate task-specific parameters for prompt substitution and ground truth."""
    return {}


def _count_sheep_that_cannot_escape(gt: dict[str, Any]) -> int:
    if "cannotEscapeCount" in gt:
        return int(gt["cannotEscapeCount"])

    details = gt.get("sheepDetails")
    if isinstance(details, list):
        explicit = [
            item
            for item in details
            if isinstance(item, dict) and "cannotEscape" in item
        ]
        if explicit:
            return sum(1 for item in explicit if item.get("cannotEscape", False))

        with_escape_flags = [
            item
            for item in details
            if isinstance(item, dict) and "canEscape" in item
        ]
        if with_escape_flags:
            return sum(
                1
                for item in with_escape_flags
                if item.get("isInside", False) and not item.get("canEscape", False)
            )

    return int(gt.get("insideCount", 0))


def get_ground_truth(task_id: str, sample: dict, params: dict | None = None) -> str | None:
    """Derive ground truth answer for a task."""
    gt = sample["groundTruth"]

    if task_id == "Q1_count_inside":
        return str(_count_sheep_that_cannot_escape(gt))

    if task_id == "Q2_escape_possibility":
        # Use per-sheep canEscape field (only sheep in outermost layer with gaps)
        details = gt["sheepDetails"]
        escapable = [str(s["id"]) for s in details if s.get("canEscape", False)]
        return ", ".join(sorted(escapable, key=int)) if escapable else "NONE"

    if task_id == "Q3_max_cell_count":
        cell_counts = gt.get("cellCounts", {})
        if not cell_counts:
            return None
        max_label = max(cell_counts, key=cell_counts.get)
        return max_label

    if task_id == "Q4_fence_repair":
        return str(gt.get("numGaps", 0))

    return None


# ═══════════════════════════════════════════════════════════════════
# ANSWER PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_answer(raw: str, task_id: str) -> tuple[str, str]:
    """Extract the answer from VLM response using 3-layer fallback.

    Returns (answer, parse_method) where parse_method is one of:
      "json"      — Layer 1: extracted from {"answer": "..."} JSON
      "scan"      — Layer 2: found legal value by scanning full output
      "tail"      — Layer 3: last matching token scanning from end
      "unclear"   — all layers failed
    """
    if not raw or not raw.strip():
        return "unclear", "unclear"

    task = TASKS.get(task_id, {})
    parse_key = task.get("parse_key", "")

    # ── Layer 1: JSON {"answer": "..."} extraction ──
    json_val = _try_json_answer(raw)
    if json_val is not None:
        result = _coerce_json_value(json_val, parse_key)
        if result is not None:
            return result, "json"

    # ── Layer 2: scan full output for legal values ──
    text_upper = raw.strip().upper()
    result = _scan_for_legal_value(text_upper, parse_key)
    if result is not None:
        return result, "scan"

    # ── Layer 3: scan from end for last matching token ──
    result = _scan_tail_for_value(text_upper, parse_key)
    if result is not None:
        return result, "tail"

    return "unclear", "unclear"


def _coerce_json_value(val: Any, parse_key: str) -> str | None:
    """Try to convert a JSON-extracted value into the expected answer format."""
    s = str(val).strip()
    s_upper = s.upper()
    if parse_key == "INTEGER":
        try:
            return str(int(val))
        except (ValueError, TypeError):
            nums = re.findall(r'\d+', s)
            return nums[0] if nums else None
    if parse_key == "ID_LIST":
        if s_upper == "NONE":
            return "NONE"
        ids = re.findall(r'\b(\d+)\b', s)
        if ids:
            return ", ".join(sorted(set(ids), key=int))
        return None
    if parse_key == "CELL_LABEL":
        m = re.search(r'\b([A-Z])\b', s_upper)
        return m.group(1) if m else None
    if parse_key == "OPTION":
        m = re.search(r'\b([A-E])\b', s_upper)
        return m.group(1) if m else None
    return s if s else None


def _scan_for_legal_value(text_upper: str, parse_key: str) -> str | None:
    """Layer 2: search the full output for any legal value (first match)."""
    if parse_key == "INTEGER":
        nums = re.findall(r'\b(\d+)\b', text_upper)
        return nums[0] if nums else None
    if parse_key == "ID_LIST":
        if "NONE" in text_upper and not re.search(r'\d', text_upper):
            return "NONE"
        ids = re.findall(r'\b(\d+)\b', text_upper)
        if ids:
            return ", ".join(sorted(set(ids), key=int))
        return None
    if parse_key == "CELL_LABEL":
        m = re.search(r'\b([A-Z])\b', text_upper)
        return m.group(1) if m else None
    if parse_key == "OPTION":
        m = re.search(r'\b([A-E])\b', text_upper)
        return m.group(1) if m else None
    return None


def _scan_tail_for_value(text_upper: str, parse_key: str) -> str | None:
    """Layer 3: scan from end of output for last matching token."""
    if parse_key == "INTEGER":
        nums = re.findall(r'\b(\d+)\b', text_upper)
        return nums[-1] if nums else None
    if parse_key == "ID_LIST":
        # For ID_LIST, scanning tail doesn't make sense — use scan result
        return None
    if parse_key in ("CELL_LABEL", "OPTION"):
        pattern = r'\b([A-E])\b' if parse_key == "OPTION" else r'\b([A-Z])\b'
        matches = re.findall(pattern, text_upper)
        return matches[-1] if matches else None
    return None


def _try_json_answer(raw: str) -> Any:
    """Try to extract the 'answer' field from a JSON response."""
    text = raw.strip()
    if not text:
        return None

    # Try regex for {"answer": "..."} or {"answer":"..."} first
    m = re.search(r'\{\s*"answer"\s*:\s*"([^"]*)"\s*\}', text)
    if m:
        return m.group(1)
    m = re.search(r"\{\s*['\"]answer['\"]\s*:\s*'([^']*)'\s*\}", text)
    if m:
        return m.group(1)
    # Try {"answer": <number>}
    m = re.search(r'\{\s*"answer"\s*:\s*(\d+)\s*\}', text)
    if m:
        return int(m.group(1))

    # Fall back to full JSON parse
    candidates = [text]
    fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    candidates.extend(match.group(1) for match in fence_pattern.finditer(text))

    object_pattern = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)
    candidates.extend(match.group(0) for match in object_pattern.finditer(text))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    return None


# ═══════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════

def score_answer(parsed: str, gt: str, task_id: str) -> dict[str, Any]:
    """Score a parsed answer against ground truth."""
    if gt is None or parsed == "unclear":
        return {"correct": False, "partial": False, "detail": "missing"}

    parse_key = TASKS.get(task_id, {}).get("parse_key", "")

    # ── ID_LIST — set comparison ──
    if parse_key == "ID_LIST":
        p_set = set(x.strip() for x in parsed.split(",") if x.strip()) if parsed != "NONE" else set()
        g_set = set(x.strip() for x in gt.split(",") if x.strip()) if gt != "NONE" else set()
        exact = p_set == g_set
        # Partial: at least half correct, no more than 1 extra
        overlap = len(p_set & g_set)
        partial = (not exact and overlap >= len(g_set) / 2 and len(p_set - g_set) <= 1) if g_set else (len(p_set) <= 1)
        return {
            "correct": exact,
            "partial": partial,
            "detail": f"pred={parsed},gt={gt}",
        }

    # ── INTEGER comparison (allow ±1 partial credit) ──
    if parse_key == "INTEGER":
        try:
            p, g = int(parsed), int(gt)
            return {
                "correct": p == g,
                "partial": abs(p - g) == 1,
                "detail": f"pred={parsed},gt={gt}",
            }
        except ValueError:
            return {"correct": False, "partial": False, "detail": "parse_fail"}

    # ── Default: exact keyword match ──
    return {
        "correct": parsed.upper() == gt.upper(),
        "partial": False,
        "detail": "exact" if parsed.upper() == gt.upper() else f"pred={parsed},gt={gt}",
    }


# ═══════════════════════════════════════════════════════════════════
# VLM API INTERACTION
# ═══════════════════════════════════════════════════════════════════

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_vlm(client: Any, model: str, prompt: str, image_paths: list[str],
            max_tokens: int = 2048, max_retries: int = 5) -> str:
    """Send image(s) + prompt to VLM and return raw text response."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    prompt_has_inline_image_labels = "[Image " in prompt
    for i, path in enumerate(image_paths):
        if not prompt_has_inline_image_labels:
            content.append({"type": "text", "text": f"[Image {i + 1}]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(path)}",
            },
        })

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0,
            )
            text = response.choices[0].message.content
            return (text or "").strip()
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "频繁" in err_str or "rate" in err_str.lower() or "-20048" in err_str
            if attempt < max_retries - 1:
                wait = (5 * (attempt + 1)) if is_rate_limit else (2 ** (attempt + 1))
                print(f"  [RETRY] attempt {attempt+1}/{max_retries}, waiting {wait}s: {exc}", flush=True)
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════
# IMAGE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def resolve_image(data_dir: Path, filename: str) -> Path | None:
    """Find image file in dataset directory."""
    candidates = [
        data_dir / "images" / filename,
        data_dir / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def print_report(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("SHEEP FENCE ENCLOSURE VLM BENCHMARK RESULTS")
    print("=" * 70)

    if not results:
        print("\nNo valid results to report.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    partial = sum(1 for r in results if r.get("partial") and not r.get("correct"))
    print(f"\nOverall: {correct}/{total} = {correct/total:.1%}"
          f"  (partial: {partial})")

    # By task
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    print(f"\n{'Task':<30s} {'Acc':>7s}  {'N':>4s}  {'easy':>6s}  {'med':>6s}  {'hard':>6s}")
    print("-" * 65)

    for task_id in TASKS:
        task_results = by_task.get(task_id, [])
        if not task_results:
            continue
        n = len(task_results)
        c = sum(1 for r in task_results if r.get("correct"))
        acc = c / n if n else 0

        diff_str = {}
        for level in ("easy", "medium", "hard"):
            lr = [r for r in task_results if r.get("difficulty_level") == level]
            if lr:
                lc = sum(1 for r in lr if r.get("correct"))
                diff_str[level] = f"{lc/len(lr):.0%}"
            else:
                diff_str[level] = "—"

        print(f"  {task_id:<28s} {acc:6.1%}  {n:4d}  "
              f"{diff_str['easy']:>6s}  {diff_str['medium']:>6s}  {diff_str['hard']:>6s}")

    # By difficulty overall
    print("\n-- Difficulty Breakdown --")
    for level in ("easy", "medium", "hard"):
        lr = [r for r in results if r.get("difficulty_level") == level]
        if lr:
            lc = sum(1 for r in lr if r.get("correct"))
            print(f"  {level:8s}: {lc/len(lr):.1%} ({lc}/{len(lr)})")

    # By nesting depth
    print("\n-- By Nesting Depth --")
    by_depth: dict[int, list] = {}
    for r in results:
        nd = r.get("nesting_depth", 0)
        by_depth.setdefault(nd, []).append(r)
    for nd in sorted(by_depth):
        dr = by_depth[nd]
        dc = sum(1 for r in dr if r.get("correct"))
        print(f"  depth {nd}: {dc/len(dr):.1%} ({dc}/{len(dr)})")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def _task_list_from_arg(tasks_arg: str) -> list[str]:
    if tasks_arg.strip().lower() == "all":
        return list(TASKS.keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Sheep Fence Benchmark v2")
    parser.add_argument("--data_dir", default="./dataset",
                        help="Directory containing images and dataset_metadata.json")
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated task IDs or 'all'")
    parser.add_argument("--model", default="gpt-4o",
                        help="VLM model name (default: gpt-4o)")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"],
                        help="Filter by difficulty level")
    parser.add_argument("--phrasing", type=int, default=None,
                        help="Use a specific phrasing index")
    parser.add_argument("--all-phrasings", action="store_true",
                        help="Run all available phrasings per task")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling API")
    parser.add_argument("--base-url", default=None,
                        help="Override OpenAI base URL (for Gemini/Claude)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Env var name for API key")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API calls (default: 1)")
    args = parser.parse_args()

    # ── Load dataset ──
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "dataset_metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] dataset_metadata.json not found in {data_dir}")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        samples = raw_data
    else:
        samples = raw_data.get("samples", raw_data)
    print(f"Loaded {len(samples)} samples from {meta_path}")

    # Filter by difficulty
    if args.difficulty:
        samples = [s for s in samples if s.get("difficulty") == args.difficulty]
        print(f"  Filtered to {len(samples)} {args.difficulty} samples")

    # Apply limit
    if args.limit:
        samples = samples[:args.limit]
        print(f"  Limited to {len(samples)} samples")

    if not samples:
        print("[ERROR] No samples to evaluate.")
        sys.exit(1)

    # ── API setup ──
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        print(f"[ERROR] {args.api_key_env} is not set.")
        sys.exit(1)

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "[ERROR] The `openai` package is required for VLM API calls, "
                "but not for dataset generation. Install it with `pip install openai` "
                "or rerun with --dry-run."
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 300.0}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)

    # ── Task selection ──
    task_ids = _task_list_from_arg(args.tasks)
    valid_task_ids = [t for t in task_ids if t in TASKS]
    for t in task_ids:
        if t not in TASKS:
            print(f"[WARN] Unknown task: {t}")
    if not valid_task_ids:
        print("[ERROR] No valid tasks selected.")
        sys.exit(1)

    print(f"Tasks: {len(valid_task_ids)} selected")

    # ── Build job list ──
    jobs: list[dict[str, Any]] = []
    for task_id in valid_task_ids:
        task = TASKS[task_id]
        max_tokens = task.get("max_tokens", 2048)

        if args.all_phrasings:
            phrasing_indices = list(range(num_phrasings(task_id)))
        elif args.phrasing is not None:
            phrasing_indices = [args.phrasing]
        else:
            phrasing_indices = [0]

        allowed_scenes = task.get("scene_types")

        for si, sample in enumerate(samples):
            if allowed_scenes and sample.get("sceneType", "fence") not in allowed_scenes:
                continue

            params = generate_task_params(task_id, sample, seed=si * 1000 + hash(task_id) % 1000)

            gt_val = get_ground_truth(task_id, sample, params)
            if gt_val is None:
                continue

            img_file = sample.get("imageFile", f"{sample['id']}.png")
            img_path = resolve_image(data_dir, img_file)
            if not img_path:
                continue

            for pidx in phrasing_indices:
                prompt = build_prompt(task_id, pidx, params)
                question_id = f"{task_id}_P{pidx:02d}"

                if args.dry_run:
                    print(f"  [DRY] {question_id} | {img_file} | gt={gt_val}")
                    continue

                jobs.append({
                    "task_id": task_id,
                    "question_id": question_id,
                    "phrasing_index": pidx,
                    "sample": sample,
                    "img_file": img_file,
                    "img_path": str(img_path),
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "gt_val": gt_val,
                    "params": params,
                })

    if args.dry_run:
        print(f"\n[DRY RUN] Would evaluate {len(jobs)} questions "
              f"across {len(valid_task_ids)} tasks")
        return

    print(f"Total jobs: {len(jobs)}, concurrency: {args.concurrency}")

    # ── Evaluate (concurrent) ──
    all_results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    completed = [0]

    def run_job(job: dict) -> dict | None:
        try:
            raw = ask_vlm(client, args.model, job["prompt"],
                          [job["img_path"]], job["max_tokens"])
            parsed = parse_answer(raw, job["task_id"])
            scoring = score_answer(parsed, job["gt_val"], job["task_id"])
            sample = job["sample"]

            result = {
                "task": job["task_id"],
                "question_id": job["question_id"],
                "phrasing_index": job["phrasing_index"],
                "sample_id": sample["id"],
                "filename": job["img_file"],
                "difficulty_level": sample.get("difficulty"),
                "nesting_depth": sample["groundTruth"].get("nestingDepth", 0),
                "num_sheep": sample["groundTruth"]["totalSheep"],
                "is_fence_closed": sample["groundTruth"].get("isFenceClosed", True),
                "num_gaps": sample["groundTruth"].get("numGaps", 0),
                "scene_type": sample.get("sceneType", "fence"),
                "ground_truth": job["gt_val"],
                "vlm_answer": parsed,
                "vlm_raw": raw,
                "correct": scoring["correct"],
                "partial": scoring.get("partial", False),
                "score_detail": scoring.get("detail", ""),
            }

            icon = "+" if scoring["correct"] else (
                "~" if scoring.get("partial") else "-")
            with results_lock:
                completed[0] += 1
                print(f"[{icon}] ({completed[0]}/{len(jobs)}) {job['question_id']} | "
                      f"{job['img_file']} | gt={job['gt_val']} | ans={parsed} | "
                      f"{sample.get('difficulty', '?')}", flush=True)
            return result
        except Exception as exc:
            with results_lock:
                completed[0] += 1
                print(f"[ERROR] ({completed[0]}/{len(jobs)}) {job['question_id']} on "
                      f"{job['img_file']}: {exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    # ── Save results ──
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_results)} results to {output_path}")
        print_report(all_results)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
