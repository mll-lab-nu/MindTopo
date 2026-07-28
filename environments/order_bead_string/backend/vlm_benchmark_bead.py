#!/usr/bin/env python3
"""
vlm_benchmark_bead.py — Bead String VLM Benchmark (v4).

2 Static Reasoning tasks (T_BS01–T_BS02):
  T_BS01_describe_sequence   — read bead sequence from single image
  T_BS02_pair_relationship   — classify pair relationship (IDENTICAL/REVERSED/CYCLIC_ROTATION/DIFFERENT)

Usage:
  python vlm_benchmark_bead.py --data_dir ./dataset --tasks all --limit 5
  python vlm_benchmark_bead.py --data_dir ./dataset --tasks T_BS01_describe_sequence
  python vlm_benchmark_bead.py --data_dir ./dataset --tasks all --all-phrasings --limit 3

  # InternVL via SiliconFlow (OpenAI-compatible):
  python vlm_benchmark_bead.py --model Pro/OpenGVLab/InternVL2.5-78B \\
      --base-url https://api.siliconflow.cn/v1 --api-key-env SILICONFLOW_API_KEY

  # Anthropic Claude:
  python vlm_benchmark_bead.py --provider anthropic --model claude-sonnet-4-20250514 \\
      --api-key-env ANTHROPIC_API_KEY

Dependencies:
  pip install openai          # for OpenAI / OpenAI-compatible providers
  pip install anthropic       # (optional) for Anthropic provider

Required env var:
  OPENAI_API_KEY (default) or use --api-key-env to specify
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

# Provider imports — lazy-loaded for Anthropic
from question_phrasings_bead import PHRASINGS


# ═══════════════════════════════════════════════════════════════════
# TASK DEFINITIONS — 2 Static Reasoning tasks
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    "T_BS01_describe_sequence": {
        "input_type": "single",
        "parse_key": "SEQUENCE",
        "max_tokens": 300,
    },
    "T_BS02_pair_relationship": {
        "input_type": "pair",
        "parse_key": "RELATIONSHIP",
        "max_tokens": 200,
    },
}

assert len(TASKS) == 2, f"Expected 2 tasks, got {len(TASKS)}"

TASK_PROMPT_SPECS = {
    "T_BS01_describe_sequence": {
        "task_name": "order.describe_sequence",
        "task_category": "order",
        "image_count": 1,
        "detailed_description": (
            "trace a bead string continuously from one endpoint to the other and "
            "report the bead colors in encounter order"
        ),
        "definition": (
            "A bead string is a rope threaded through colored beads. The bead "
            "sequence is determined by following the rope continuously from one "
            "visible endpoint to the other and listing bead colors in the exact "
            "order encountered along that path."
        ),
        "legal_answer_space_description": (
            'a single JSON string containing the comma-separated bead-color '
            'sequence, for example "RED, BLUE, GREEN"'
        ),
        "parse_key": "SEQUENCE",
        "legal_values": ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "WHITE", "BROWN"],
    },
    "T_BS02_pair_relationship": {
        "task_name": "order.pair_relationship",
        "task_category": "order",
        "image_count": 2,
        "detailed_description": (
            "compare the bead-color sequences shown in two bead-string images "
            "and classify their relationship"
        ),
        "definition": (
            "A bead string is a rope threaded through colored beads. IDENTICAL "
            "means the two sequences match exactly; REVERSED means one is the "
            "exact reverse of the other, such as RED, GREEN, BLUE, WHITE versus "
            "WHITE, BLUE, GREEN, RED; CYCLIC_ROTATION means one is a cyclic shift "
            "of the other without reversing the order, such as RED, GREEN, BLUE, "
            "WHITE versus BLUE, WHITE, RED, GREEN; "
            "DIFFERENT means none of the above."
        ),
        "legal_answer_space_description": (
            'one of the JSON strings "IDENTICAL", "REVERSED", '
            '"CYCLIC_ROTATION", or "DIFFERENT"'
        ),
        "parse_key": "RELATIONSHIP",
        "legal_values": ["IDENTICAL", "REVERSED", "CYCLIC_ROTATION", "DIFFERENT"],
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

    # Substitute parameters like {target}, {pos1}, {pos2}, etc.
    if params:
        try:
            question_text = question_text.format(**params)
        except KeyError:
            pass

    spec = TASK_PROMPT_SPECS[task_id]
    image_block = _build_image_block(spec["image_count"])
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
            image_block,
            "",
            "[Rules]",
            "1. Use only the images and text provided in this prompt.",
            "2. If answer options are provided, choose only from the provided options.",
            "3. Do not output explanation beyond the required final answer.",
            "4. Bead colors in these images are exactly: RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, WHITE, BROWN. Use these exact uppercase names; do not invent synonyms (e.g. write 'PURPLE', not 'VIOLET' or 'MAGENTA').",
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
# GROUND TRUTH COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# GROUND TRUTH — derive correct answer from sample/pair
# ═══════════════════════════════════════════════════════════════════

def get_ground_truth(task_id: str, sample: dict, params: dict | None = None,
                     pair_sample: dict | None = None,
                     pair_info: dict | None = None) -> str | None:
    """Derive ground truth answer for a task."""
    seq = sample["bead_sequence"]

    # ── T_BS01: describe sequence ──
    if task_id == "T_BS01_describe_sequence":
        return ", ".join(seq)

    # ── Pair tasks: check pre-computed GT from pair_info ──
    if pair_info and "_gt" in pair_info:
        return pair_info["_gt"]

    # ── Pair tasks: compute GT from sequences ──
    if pair_sample is not None:
        if task_id == "T_BS02_pair_relationship":
            return classify_pair_relationship(seq, pair_sample["bead_sequence"])

    return None


def classify_pair_relationship(seq_a: list[str], seq_b: list[str]) -> str:
    """Classify the relationship between two bead sequences."""
    if seq_a == seq_b:
        return "IDENTICAL"
    if seq_a == seq_b[::-1]:
        return "REVERSED"
    if len(seq_a) == len(seq_b):
        doubled = seq_a + seq_a
        n = len(seq_a)
        for k in range(1, n):
            if doubled[k:k + n] == seq_b:
                return "CYCLIC_ROTATION"
    return "DIFFERENT"


# ═══════════════════════════════════════════════════════════════════
# ANSWER PARSING
# ═══════════════════════════════════════════════════════════════════

_COLOR_RE = r'\b(RED|BLUE|GREEN|YELLOW|ORANGE|PURPLE|WHITE|BROWN)\b'
_RELATIONSHIP_KEYWORDS = ["CYCLIC_ROTATION", "IDENTICAL", "REVERSED", "DIFFERENT"]


def parse_answer(raw: str, task_id: str) -> tuple[str, str]:
    """Extract the answer using 3-layer fallback.

    Returns (answer, parse_method) where parse_method is one of:
      "json", "scan", "tail", "unclear".
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

    text_upper = raw.strip().upper()

    # ── Layer 2: scan full output for legal values ──
    result = _scan_for_legal_value(text_upper, parse_key)
    if result is not None:
        return result, "scan"

    # ── Layer 3: scan from end for last matching token ──
    result = _scan_tail_for_value(text_upper, parse_key)
    if result is not None:
        return result, "tail"

    return "unclear", "unclear"


def _coerce_json_value(val: Any, parse_key: str) -> str | None:
    s = str(val).strip().upper()
    if parse_key == "SEQUENCE":
        colors = re.findall(_COLOR_RE, s)
        return ", ".join(colors) if colors else None
    if parse_key == "RELATIONSHIP":
        for kw in _RELATIONSHIP_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', s):
                return kw
        if re.search(r'\bCYCLIC[\s_]ROTATION\b', s):
            return "CYCLIC_ROTATION"
        return None
    return None


def _scan_for_legal_value(text_upper: str, parse_key: str) -> str | None:
    if parse_key == "SEQUENCE":
        colors = re.findall(_COLOR_RE, text_upper)
        return ", ".join(colors) if colors else None
    if parse_key == "RELATIONSHIP":
        for kw in _RELATIONSHIP_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_upper):
                return kw
        if re.search(r'\bCYCLIC[\s_]ROTATION\b', text_upper):
            return "CYCLIC_ROTATION"
        return None
    return None


def _scan_tail_for_value(text_upper: str, parse_key: str) -> str | None:
    if parse_key == "RELATIONSHIP":
        # Scan from end: find the last keyword match
        last_pos, last_kw = -1, None
        for kw in _RELATIONSHIP_KEYWORDS:
            m = None
            for m in re.finditer(r'\b' + re.escape(kw) + r'\b', text_upper):
                pass
            if m and m.start() > last_pos:
                last_pos, last_kw = m.start(), kw
        # Also check CYCLIC ROTATION with space
        for m in re.finditer(r'\bCYCLIC[\s_]ROTATION\b', text_upper):
            if m.start() > last_pos:
                last_pos, last_kw = m.start(), "CYCLIC_ROTATION"
        return last_kw
    # For SEQUENCE, tail scan doesn't really help — use scan result
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

def score_answer(parsed: str, gt: str, task_id: str, extra: dict | None = None) -> dict[str, Any]:
    """Score a parsed answer against ground truth."""
    if gt is None or parsed == "unclear":
        return {"correct": False, "partial": False, "detail": "missing"}

    parse_key = TASKS.get(task_id, {}).get("parse_key", "")

    # ── Sequence comparison (T_BS01) ──
    if parse_key == "SEQUENCE":
        parsed_colors = [c.strip() for c in parsed.split(",") if c.strip()]
        gt_colors = [c.strip() for c in gt.split(",") if c.strip()]
        exact = parsed_colors == gt_colors
        # Partial: all colors present but order differs
        partial = (not exact and sorted(parsed_colors) == sorted(gt_colors))
        return {
            "correct": exact,
            "partial": partial,
            "detail": "exact" if exact else f"pred={parsed},gt={gt}",
        }

    # ── RELATIONSHIP: exact keyword match ──
    if parse_key == "RELATIONSHIP":
        correct = parsed.upper() == gt.upper()
        return {
            "correct": correct,
            "partial": False,
            "detail": "exact" if correct else f"pred={parsed},gt={gt}",
        }

    # ── Default: exact keyword match ──
    return {
        "correct": parsed.upper() == gt.upper(),
        "partial": False,
        "detail": "exact" if parsed.upper() == gt.upper() else f"pred={parsed},gt={gt}",
    }


# ═══════════════════════════════════════════════════════════════════
# PAIR GENERATION — create sample pairs for pair-type tasks
# ═══════════════════════════════════════════════════════════════════

def generate_pairs(
    samples: list[dict],
    task_id: str,
    seed: int = 42,
    max_pairs: int | None = None,
    max_pairs_per_difficulty: int | None = None,
    balance_by_relation: bool = False,
) -> list[tuple[dict, dict, dict]]:
    """Generate pair-task rows.

    When max_pairs is unset, default to the smallest budget that can cover every
    image at least once in a pair question.
    """
    rng = random.Random(seed)
    pairs: list[tuple[dict, dict, dict]] = []

    if task_id == "T_BS02_pair_relationship":
        unique_samples = [s for s in samples if s.get("filename") and s.get("bead_sequence")]
        base_samples = [s for s in unique_samples if not s.get("_pair_variant")]
        if len(unique_samples) < 2 or len(base_samples) < 1:
            return []

        coverage_min_pairs = math.ceil(len(base_samples) / 2)
        target_pairs = coverage_min_pairs if max_pairs is None else max(0, int(max_pairs))
        if target_pairs == 0:
            return []

        relation_priority = {
            "IDENTICAL": 3,
            "REVERSED": 2,
            "CYCLIC_ROTATION": 1,
            "DIFFERENT": 0,
        }
        by_filename: dict[str, dict] = {}
        for sample in unique_samples:
            for key in (sample.get("filename"), sample.get("raw_filename")):
                if key:
                    by_filename[str(key)] = sample

        candidate_pairs: list[tuple[dict, dict, dict]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for variant in unique_samples:
            source_name = variant.get("_pair_source")
            if not variant.get("_pair_variant") or not source_name:
                continue
            source = by_filename.get(str(source_name))
            if source is None or source.get("filename") == variant.get("filename"):
                continue
            gt = classify_pair_relationship(source["bead_sequence"], variant["bead_sequence"])
            pair_key = (str(source["filename"]), str(variant["filename"]))
            seen_pairs.add(pair_key)
            candidate_pairs.append((source, variant, {"_gt": gt, "_pair_variant": True}))

        for i, sa in enumerate(base_samples):
            for sb in base_samples[i + 1:]:
                if sa["filename"] == sb["filename"]:
                    continue
                pair_key = (str(sa["filename"]), str(sb["filename"]))
                if pair_key in seen_pairs:
                    continue
                gt = classify_pair_relationship(sa["bead_sequence"], sb["bead_sequence"])
                candidate_pairs.append((sa, sb, {"_gt": gt}))

        if not candidate_pairs:
            return []

        rng.shuffle(candidate_pairs)

        if max_pairs_per_difficulty is not None:
            target_per_diff = max(0, int(max_pairs_per_difficulty))
            if target_per_diff == 0:
                return []
            by_diff: dict[str, list[tuple[dict, dict, dict]]] = {}
            for pair in candidate_pairs:
                difficulty = str(pair[0].get("difficulty") or "easy").lower()
                by_diff.setdefault(difficulty, []).append(pair)

            selected: list[tuple[dict, dict, dict]] = []
            relation_order = ["IDENTICAL", "REVERSED", "CYCLIC_ROTATION", "DIFFERENT"]
            for difficulty in ("easy", "medium", "hard"):
                pairs_for_diff = by_diff.get(difficulty, [])
                if not pairs_for_diff:
                    continue
                by_relation: dict[str, list[tuple[dict, dict, dict]]] = {
                    relation: [] for relation in relation_order
                }
                for pair in pairs_for_diff:
                    by_relation.setdefault(pair[2]["_gt"], []).append(pair)
                for bucket in by_relation.values():
                    bucket.sort(key=lambda pair: (not bool(pair[2].get("_pair_variant")), rng.random()))

                chosen: list[tuple[dict, dict, dict]] = []
                used: set[tuple[str, str]] = set()
                while len(chosen) < target_per_diff:
                    progressed = False
                    ordered_relations = relation_order if balance_by_relation else sorted(
                        relation_order,
                        key=lambda relation: relation_priority[relation],
                        reverse=True,
                    )
                    for relation in ordered_relations:
                        bucket = by_relation.get(relation) or []
                        while bucket:
                            pair = bucket.pop(0)
                            key = (str(pair[0]["filename"]), str(pair[1]["filename"]))
                            if key in used:
                                continue
                            chosen.append(pair)
                            used.add(key)
                            progressed = True
                            break
                        if len(chosen) >= target_per_diff:
                            break
                    if not progressed:
                        break

                if len(chosen) < target_per_diff:
                    remaining = [
                        pair for pair in pairs_for_diff
                        if (str(pair[0]["filename"]), str(pair[1]["filename"])) not in used
                    ]
                    remaining.sort(key=lambda pair: (not bool(pair[2].get("_pair_variant")), rng.random()))
                    chosen.extend(remaining[:target_per_diff - len(chosen)])
                selected.extend(chosen[:target_per_diff])

            rng.shuffle(selected)
            return selected

        uncovered = {sample["filename"] for sample in base_samples}
        selected_indices: list[int] = []
        selected_set: set[int] = set()

        while uncovered and len(selected_indices) < target_pairs:
            best_index = None
            best_score = None
            for index, (sa, sb, info) in enumerate(candidate_pairs):
                if index in selected_set:
                    continue
                gain = int(sa["filename"] in uncovered) + int(sb["filename"] in uncovered)
                if gain == 0:
                    continue
                score = (gain, relation_priority[info["_gt"]])
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                break
            selected_indices.append(best_index)
            selected_set.add(best_index)
            sa, sb, _ = candidate_pairs[best_index]
            uncovered.discard(sa["filename"])
            uncovered.discard(sb["filename"])

        remaining_indices = [index for index in range(len(candidate_pairs)) if index not in selected_set]
        remaining_indices.sort(
            key=lambda index: relation_priority[candidate_pairs[index][2]["_gt"]],
            reverse=True,
        )
        for index in remaining_indices:
            if len(selected_indices) >= target_pairs:
                break
            selected_indices.append(index)

        pairs = [candidate_pairs[index] for index in selected_indices]
        rng.shuffle(pairs)
        return pairs[:target_pairs]

    return []


# ═══════════════════════════════════════════════════════════════════
# PAIR VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_pairs(pairs: list[tuple[dict, dict, dict]], task_id: str) -> dict:
    """Validate pair integrity for a task. Returns stats dict."""
    total = len(pairs)
    same_image = sum(1 for a, b, _ in pairs if a["filename"] == b["filename"])
    synthetic = sum(1 for a, b, _ in pairs
                    if a.get("_synthetic_subsequence") or a.get("_synthetic_swap") or a.get("_synthetic_deletion")
                    or b.get("_synthetic_subsequence") or b.get("_synthetic_swap") or b.get("_synthetic_deletion"))
    covered_images = {a["filename"] for a, _, _ in pairs} | {b["filename"] for _, b, _ in pairs}

    # GT distribution
    gt_counts: dict[str, int] = {}
    for a, b, info in pairs:
        gt = get_ground_truth(task_id, a, pair_sample=b, pair_info=info)
        gt_counts[gt] = gt_counts.get(gt, 0) + 1

    return {
        "total": total,
        "same_image": same_image,
        "synthetic": synthetic,
        "covered_images": len(covered_images),
        "gt_distribution": gt_counts,
    }


# ═══════════════════════════════════════════════════════════════════
# VLM API INTERACTION
# ═══════════════════════════════════════════════════════════════════

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_vlm(client, model: str, prompt: str, image_paths: list[str],
            max_tokens: int = 200, provider: str = "openai",
            max_retries: int = 5) -> str:
    """Send image(s) + prompt to VLM and return raw text response.

    Supports providers: 'openai' (+ any OpenAI-compatible API), 'anthropic'.
    Retries with exponential backoff on rate-limit / transient errors.
    """
    if provider == "anthropic":
        return _ask_anthropic(client, model, prompt, image_paths, max_tokens)

    # OpenAI / OpenAI-compatible (InternVL, Qwen-VL, etc.)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    prompt_has_inline_image_labels = "[Image " in prompt
    for i, path in enumerate(image_paths):
        if not prompt_has_inline_image_labels:
            content.append({"type": "text", "text": f"[Image {i + 1}]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(path)}",
                "detail": "high",
            },
        })

    # Use max_tokens for OpenAI-compatible APIs; max_completion_tokens for native OpenAI
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    # Try max_completion_tokens first (native OpenAI), fall back to max_tokens
    try:
        create_kwargs["max_completion_tokens"] = max_tokens
        response = client.chat.completions.create(**create_kwargs)
    except Exception:
        del create_kwargs["max_completion_tokens"]
        create_kwargs["max_tokens"] = max_tokens
        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                err_str = str(e)
                is_retryable = "过于频繁" in err_str or "rate" in err_str.lower() or "429" in err_str
                if is_retryable and attempt < max_retries:
                    wait = 2 ** attempt + random.random() * 2
                    time.sleep(wait)
                    continue
                raise

    text = response.choices[0].message.content
    return (text or "").strip()


def _ask_anthropic(client, model: str, prompt: str, image_paths: list[str],
                   max_tokens: int = 200) -> str:
    """Send image(s) + prompt via Anthropic Messages API."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    prompt_has_inline_image_labels = "[Image " in prompt
    for i, path in enumerate(image_paths):
        if not prompt_has_inline_image_labels:
            content.append({"type": "text", "text": f"[Image {i + 1}]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image(path),
            },
        })

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text.strip()


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
    print("STATIC REASONING BENCHMARK RESULTS")
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

    print(f"\n{'Task':<35s} {'Acc':>7s}  {'N':>4s}  {'easy':>6s}  {'med':>6s}  {'hard':>6s}")
    print("-" * 70)

    for task_id in TASKS:
        task_results = by_task.get(task_id, [])
        if not task_results:
            continue
        n = len(task_results)
        c = sum(1 for r in task_results if r.get("correct"))
        acc = c / n if n else 0

        # By difficulty
        diff_str = {}
        for level in ("easy", "medium", "hard"):
            lr = [r for r in task_results if r.get("difficulty_level") == level]
            if lr:
                lc = sum(1 for r in lr if r.get("correct"))
                diff_str[level] = f"{lc/len(lr):.0%}"
            else:
                diff_str[level] = "—"

        print(f"  {task_id:<33s} {acc:6.1%}  {n:4d}  "
              f"{diff_str['easy']:>6s}  {diff_str['medium']:>6s}  {diff_str['hard']:>6s}")

    # By difficulty overall
    print("\n-- Difficulty Breakdown --")
    for level in ("easy", "medium", "hard"):
        lr = [r for r in results if r.get("difficulty_level") == level]
        if lr:
            lc = sum(1 for r in lr if r.get("correct"))
            print(f"  {level:8s}: {lc/len(lr):.1%} ({lc}/{len(lr)})")

    # By num_beads
    print("\n-- By Num Beads --")
    by_beads: dict[int, list] = {}
    for r in results:
        nb = r.get("num_beads", 0)
        by_beads.setdefault(nb, []).append(r)
    for nb in sorted(by_beads):
        br = by_beads[nb]
        bc = sum(1 for r in br if r.get("correct"))
        print(f"  {nb:2d} beads: {bc/len(br):.1%} ({bc}/{len(br)})")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def _task_list_from_arg(tasks_arg: str) -> list[str]:
    if tasks_arg.strip().lower() == "all":
        return list(TASKS.keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Bead String Benchmark — Static Reasoning")
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
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "anthropic"],
                        help="API provider (default: openai). Use 'openai' for OpenAI-compatible APIs too (InternVL, Qwen-VL, etc.)")
    parser.add_argument("--base-url", default=None,
                        help="Override OpenAI base URL (for InternVL/Qwen-VL/Gemini)")
    parser.add_argument("--api-key-env", default=None,
                        help="Env var name for API key (default: auto-detect by provider)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API requests (default: 1)")
    args = parser.parse_args()

    # ── Load dataset ──
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "dataset_metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] dataset_metadata.json not found in {data_dir}")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset.get("samples", [])
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
    # Auto-detect API key env var if not specified
    if args.api_key_env is None:
        if args.provider == "anthropic":
            args.api_key_env = "ANTHROPIC_API_KEY"
        else:
            args.api_key_env = "OPENAI_API_KEY"

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        print(f"[ERROR] {args.api_key_env} is not set.")
        sys.exit(1)

    client = None
    if not args.dry_run:
        if args.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
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
        input_type = task["input_type"]
        max_tokens = task.get("max_tokens", 200)
        # Reasoning models (e.g. InternVL3.5) need more tokens for chain-of-thought
        if "internvl3" in args.model.lower() or "internvl-3" in args.model.lower():
            max_tokens = max(max_tokens, 2000)

        # Determine phrasing indices
        if args.all_phrasings:
            phrasing_indices = list(range(num_phrasings(task_id)))
        elif args.phrasing is not None:
            phrasing_indices = [args.phrasing]
        else:
            phrasing_indices = [0]  # default: first phrasing

        # ── Single-image tasks ──
        if input_type == "single":
            for si, sample in enumerate(samples):
                gt = get_ground_truth(task_id, sample)
                if gt is None:
                    continue

                img_path = resolve_image(data_dir, sample["filename"])
                if not img_path:
                    continue

                for pidx in phrasing_indices:
                    prompt = build_prompt(task_id, pidx)
                    question_id = f"{task_id}_Q{pidx:02d}"

                    if args.dry_run:
                        print(f"  [DRY] {question_id} | {sample['filename']} | gt={gt}")
                        continue

                    jobs.append({
                        "type": "single",
                        "task_id": task_id,
                        "question_id": question_id,
                        "pidx": pidx,
                        "prompt": prompt,
                        "image_paths": [str(img_path)],
                        "max_tokens": max_tokens,
                        "gt": gt,
                        "sample": sample,
                    })

        # ── Pair-image tasks ──
        elif input_type.startswith("pair"):
            pairs = generate_pairs(samples, task_id, seed=42, max_pairs=args.limit)
            if not pairs:
                print(f"  [{task_id}] No valid pairs generated, skipping.")
                continue

            # Print validation stats in dry-run mode
            if args.dry_run:
                stats = validate_pairs(pairs, task_id)
                print(f"  [{task_id}] Pair validation: total={stats['total']}, "
                      f"same_image={stats['same_image']}, synthetic={stats['synthetic']}, "
                      f"covered_images={stats['covered_images']}, "
                      f"gt_distribution={stats['gt_distribution']}")

            for pi, (sa, sb, pair_info) in enumerate(pairs):
                gt = get_ground_truth(task_id, sa, pair_sample=sb, pair_info=pair_info)
                if gt is None:
                    continue

                img_a = resolve_image(data_dir, sa["filename"])
                img_b = resolve_image(data_dir, sb["filename"])
                if not img_a or not img_b:
                    continue

                for pidx in phrasing_indices:
                    prompt = build_prompt(task_id, pidx)
                    question_id = f"{task_id}_Q{pidx:02d}"

                    if args.dry_run:
                        print(f"  [DRY] {question_id} | "
                              f"{sa['filename']}+{sb['filename']} | gt={gt}")
                        continue

                    jobs.append({
                        "type": "pair",
                        "task_id": task_id,
                        "question_id": question_id,
                        "pidx": pidx,
                        "pi": pi,
                        "prompt": prompt,
                        "image_paths": [str(img_a), str(img_b)],
                        "max_tokens": max_tokens,
                        "gt": gt,
                        "sa": sa,
                        "sb": sb,
                        "pair_info": pair_info,
                    })

    if args.dry_run:
        print(f"\n[DRY RUN] Would evaluate {len(jobs)} questions "
              f"across {len(valid_task_ids)} tasks")
        return

    # ── Execute jobs (with concurrency) ──
    all_results: list[dict[str, Any]] = []
    result_lock = Lock()
    completed = [0]  # mutable counter for thread-safe progress

    def _run_job(job: dict) -> dict | None:
        task_id = job["task_id"]
        question_id = job["question_id"]
        gt = job["gt"]
        try:
            raw = ask_vlm(client, args.model, job["prompt"],
                          job["image_paths"], job["max_tokens"], args.provider)
            parsed = parse_answer(raw, task_id)

            if job["type"] == "single":
                scoring = score_answer(parsed, gt, task_id)
                sample = job["sample"]
                result = {
                    "task": task_id,
                    "question_id": question_id,
                    "phrasing_index": job["pidx"],
                    "sample_id": sample.get("seed", sample["filename"]),
                    "filename": sample["filename"],
                    "bead_sequence": sample["bead_sequence"],
                    "num_beads": sample["num_beads"],
                    "is_ring": sample.get("is_ring", False),
                    "curve_type": sample.get("curve_type"),
                    "occlusion_level": sample.get("occlusion_level"),
                    "difficulty_score": sample.get("difficulty_score"),
                    "difficulty_level": sample.get("difficulty"),
                    "ground_truth": gt,
                    "vlm_answer": parsed,
                    "vlm_raw": raw,
                    "correct": scoring["correct"],
                    "partial": scoring.get("partial", False),
                    "score_detail": scoring.get("detail", ""),
                }
                icon = "+" if scoring["correct"] else (
                    "~" if scoring.get("partial") else "-")
                with result_lock:
                    completed[0] += 1
                    print(f"[{icon}] ({completed[0]}/{len(jobs)}) {question_id} | {sample['filename']} | "
                          f"gt={gt} | ans={parsed} | {sample.get('difficulty', '?')}",
                          flush=True)
            else:  # pair
                sa, sb, pair_info = job["sa"], job["sb"], job["pair_info"]
                scoring = score_answer(parsed, gt, task_id, extra=pair_info)
                result = {
                    "task": task_id,
                    "question_id": question_id,
                    "phrasing_index": job["pidx"],
                    "sample_id": f"{sa.get('seed', sa['filename'])}+{sb.get('seed', sb['filename'])}",
                    "filename": f"{sa['filename']}+{sb['filename']}",
                    "bead_sequence_a": sa["bead_sequence"],
                    "bead_sequence_b": sb["bead_sequence"],
                    "num_beads": sa["num_beads"],
                    "difficulty_level": sa.get("difficulty"),
                    "ground_truth": gt,
                    "vlm_answer": parsed,
                    "vlm_raw": raw,
                    "correct": scoring["correct"],
                    "partial": scoring.get("partial", False),
                    "score_detail": scoring.get("detail", ""),
                }
                icon = "+" if scoring["correct"] else (
                    "~" if scoring.get("partial") else "-")
                with result_lock:
                    completed[0] += 1
                    print(f"[{icon}] ({completed[0]}/{len(jobs)}) {question_id} | pair#{job['pi']} | "
                          f"gt={gt} | ans={parsed}", flush=True)
            return result
        except Exception as exc:
            with result_lock:
                completed[0] += 1
                label = job["sample"]["filename"] if job["type"] == "single" else f"pair#{job['pi']}"
                print(f"[ERROR] ({completed[0]}/{len(jobs)}) {question_id} on {label}: {exc}",
                      flush=True)
            return None

    print(f"Running {len(jobs)} jobs with concurrency={args.concurrency} ...")
    if args.concurrency <= 1:
        for job in jobs:
            r = _run_job(job)
            if r:
                all_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    with result_lock:
                        all_results.append(r)

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
