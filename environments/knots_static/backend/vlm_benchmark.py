#!/usr/bin/env python3
"""
vlm_benchmark.py - Knot topology VLM benchmark.

Five single-image tasks are exposed by the current standardized benchmark:
  - T01_structure_classification: topological structure type (all, 6-way: closed loop/open rope/knot/link/unlinked/mixed)
  - T02_component_count:          component reasoning (all)
  - T03_link_topology:            link structure identification (all)
  - T04_link_property:            link removal reasoning (link_removable)
  - T05_linked_count:             linked component counting for multi-component scenes

Usage:
  python vlm_benchmark.py --data_dir ./dataset --tasks all --limit 5
  python vlm_benchmark.py --data_dir ./dataset --tasks T01_structure_classification --difficulty hard

Dependencies:
  pip install openai

Required env var:
  OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Support both `python backend/vlm_benchmark.py` and `from backend.vlm_benchmark import ...`
try:
    from backend.question_phrasings import (
        build_default_prompt,
        build_prompt,
        num_phrasings,
    )
except ImportError:
    from question_phrasings import (
        build_default_prompt,
        build_prompt,
        num_phrasings,
    )


# ═══════════════════════════════════════════════════════════════════
# TASK DEFINITIONS — standardized benchmark task set
# ═══════════════════════════════════════════════════════════════════

TASKS = {

    "T01_structure_classification": {
        "applicable": "all_single_image",
        "prompt": (
            "Classify the topological structure in this image.\n\n"
            "Answer options:\n"
            "  A) Simple closed ring — one closed ring with no knot tied\n"
            "  B) Knot — a single rope tied into a knot\n"
            "  C) Link — multiple closed ropes that are topologically linked (cannot be separated without cutting)\n"
            "  D) Multiple ropes (unlinked) — more than one rope, but they can be separated without cutting\n"
            "  E) Open-ended rope with no knot — one open rope whose ends are not joined and that is not knotted\n"
            "  F) Mixed — a combination of different rope structure categories\n\n"
            'Return JSON only: {"answer": "<letter>"}'
        ),
        "parse_key": "A|B|C|D|E|F",
        "answer_fn": "structure_classification",
    },

    "T02_component_count": {
        "applicable": "all_single_image",
        "prompt": (
            "Count the total number of separate rope components in this image.\n\n"
            "Rules:\n"
            "- Each ring or strand counts as one component.\n"
            "- Linked or tangled components are still separate — count each one.\n\n"
            'Return JSON only: {"answer": <integer>}'
        ),
        "parse_key": "INTEGER",
        "answer_fn": "component_count",
    },

    "T03_link_topology": {
        "applicable": "multi_and_rope",
        "prompt": (
            "Look at the rings in this image. How are they connected?\n\n"
            "Answer options:\n"
            "  A) Not connected — the rings are separate, none passes through another\n"
            "  B) Chain — two or more rings are linked one after another, including a two-ring paired link\n"
            "  C) All interlocked — three rings are locked together as a group, while any two alone are unlinked\n"
            "  D) Mixed — chain-style links combined with free rings or all-interlocked groups\n\n"
            'Return JSON only: {"answer": "<letter>"}'
        ),
        "parse_key": "A|B|C|D",
        "answer_fn": "link_topology",
    },


    "T04_link_property": {
        "applicable": "link_removable",
        # Prompt is built per-sample via question_phrasings.build_prompt(... removed_color=...)
        # because the question text varies with which colored ring is removed.
        "prompt": "",
        "parse_key": "COLOR_LIST",
        "answer_fn": "link_property",
    },

    "T05_linked_count": {
        "applicable": "multi_and_rope",
        "prompt": (
            "Count the number of rope components that are topologically linked "
            "to at least one other component.\n\n"
            "Rules:\n"
            "- Linked = passes through another component, inseparable without cutting.\n"
            "- Tangled but separable = NOT linked.\n"
            "- If no components are linked, answer 0.\n\n"
            'Return JSON only: {"answer": <integer>}'
        ),
        "parse_key": "INTEGER",
        "answer_fn": "linked_count",
    },
}

# Legacy pair/anchor task branches remain below, but only these 5 task IDs are
# exposed through the current standardized benchmark.
assert len(TASKS) == 5, f"Expected 5 single-image tasks, got {len(TASKS)}"


# ═══════════════════════════════════════════════════════════════════
# DIFFICULTY COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_difficulty(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compute difficulty score (0-1) and level from metadata."""
    crossing = _safe_float(metadata.get("crossingNumber",
                           metadata.get("crossing_number", 0)))
    slackness = max(0.0, min(1.0, _safe_float(metadata.get("slackness", 0))))
    trap = metadata.get("trap_type")
    is_deceptive = metadata.get("isDeceptive", False)

    score_topology = min(crossing / 10.0, 1.0)
    score_saliency = slackness
    score_trap = 0.3 if (trap or is_deceptive) else 0.0

    difficulty = 0.35 * score_topology + 0.45 * score_saliency + 0.20 * score_trap

    if difficulty < 0.25:
        level = "easy"
    elif difficulty < 0.45:
        level = "medium"
    else:
        level = "hard"

    return {"score": round(difficulty, 3), "level": level}


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# ANSWER PARSING — robust extraction from VLM responses
# ═══════════════════════════════════════════════════════════════════

_T04_COLOR_PALETTE = (
    "red",
    "blue",
    "green",
    "brown",
    "white",
    "purple",
)
# Special token returned when no remaining ring is free. T04 answers MUST
# always be a non-empty list — empty `[]` is forbidden, ["none"] is required.
_T04_NONE_TOKEN = "none"


def _canonicalize_color_list(colors: list[str]) -> str:
    """Normalize a list of color strings into a canonical comma-separated form.

    Lowercases, drops unknown colors, dedups, and sorts so that
    ["Red", "blue"] and ["blue", "RED", "blue"] both collapse to "blue,red".
    A list with no valid colors (whether empty, or containing only "none",
    or whitespace) collapses to the special token "none".
    A list mixing "none" with real colors drops the "none" and keeps the colors.
    """
    seen = set()
    for c in colors:
        if not isinstance(c, str):
            continue
        name = c.strip().lower()
        if name in _T04_COLOR_PALETTE:
            seen.add(name)
    if not seen:
        return _T04_NONE_TOKEN
    return ",".join(sorted(seen))


def _canonicalize_color_answer_value(value: Any) -> str:
    """Canonicalize a T04 answer value from JSONL, oracle, or model output."""
    if isinstance(value, list):
        return _canonicalize_color_list([str(x) for x in value])
    text = str(value or "").strip()
    if not text:
        return _T04_NONE_TOKEN
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return _canonicalize_color_list([str(x) for x in parsed])
    return _canonicalize_color_list([part.strip() for part in text.split(",")])


def _parse_color_list_from_text(raw: str) -> str | None:
    """Find a JSON array of color strings inside *raw* and canonicalize it.

    Returns the canonical comma-separated form, or None if no list was found.
    An empty list "[]" or a list containing only "none" both canonicalize to
    the special token "none".
    """
    # Look for the FIRST JSON array in the text (greedy match across newlines).
    m = re.search(r'\[[^\[\]]*\]', raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return _canonicalize_color_list([str(x) for x in data])


def parse_answer(raw: str, task_id: str) -> tuple[str, str]:
    """Extract the answer using 3-layer fallback.

    Returns (answer, parse_method) where parse_method is one of:
      "json", "scan", "tail", "unclear".
    """
    if not raw or not raw.strip():
        return "unclear", "unclear"

    task = TASKS.get(task_id, {})
    parse_key = task.get("parse_key", "")
    is_integer = parse_key == "INTEGER"
    is_color_list = parse_key == "COLOR_LIST"
    legal_keys = [] if (is_integer or is_color_list) else [
        k for k in parse_key.split("|") if k
    ]

    # ── COLOR_LIST: T04 free-color list answer ──
    if is_color_list:
        # Layer 1: a JSON object like {"answer": [...]}
        json_val = _try_json_answer(raw)
        if isinstance(json_val, list):
            return _canonicalize_color_list([str(x) for x in json_val]), "json"
        # Layer 2: scan for any JSON array literal in the text
        canon = _parse_color_list_from_text(raw)
        if canon is not None:
            return canon, "scan"
        # Layer 3: scrape any palette color names mentioned in the response
        text_lower = raw.lower()
        found = [c for c in _T04_COLOR_PALETTE
                 if re.search(r'\b' + re.escape(c) + r'\b', text_lower)]
        if found:
            return _canonicalize_color_list(found), "tail"
        # Last resort: an explicit "none" mention with no colors
        if re.search(r'\bnone\b', text_lower):
            return _T04_NONE_TOKEN, "tail"
        return "unclear", "unclear"

    # ── Layer 1: JSON {"answer": "..."} extraction ──
    json_val = _try_json_answer(raw)
    if json_val is not None:
        if is_integer:
            try:
                return str(int(json_val)), "json"
            except (ValueError, TypeError):
                nums = re.findall(r'\d+', str(json_val))
                if nums:
                    return nums[0], "json"
        else:
            val = str(json_val).strip().upper()
            for key in legal_keys:
                if val == key or val.startswith(key):
                    return key, "json"

    text_upper = raw.strip().upper()

    # ── Layer 2: scan full output for legal values ──
    if is_integer:
        nums = re.findall(r'\b(\d+)\b', text_upper)
        if nums:
            return nums[0], "scan"
    else:
        for key in legal_keys:
            if re.search(r'\b' + re.escape(key) + r'\b', text_upper):
                return key, "scan"

    # ── Layer 3: scan from end for last matching token ──
    if is_integer:
        nums = re.findall(r'\b(\d+)\b', text_upper)
        if nums:
            return nums[-1], "tail"
    else:
        # Find the last occurrence of any legal key
        last_pos, last_key = -1, None
        for key in legal_keys:
            for m in re.finditer(r'\b' + re.escape(key) + r'\b', text_upper):
                if m.start() > last_pos:
                    last_pos, last_key = m.start(), key
        if last_key is not None:
            return last_key, "tail"

    return "unclear", "unclear"


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

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*"answer"\s*:\s*("[^"]*"|\d+)[^{}]*\}', text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "answer" in obj:
                return obj["answer"]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ═══════════════════════════════════════════════════════════════════
# GROUND TRUTH — derive correct answer from metadata
# ═══════════════════════════════════════════════════════════════════

# Knot type → family mapping (used by T01_structure_classification)
_FAMILY_MAP = {
    "unknot": "UNKNOT", "twisted_ring": "UNKNOT", "kinky_unknot": "UNKNOT",
    "ring_like_open_rope": "UNKNOT",
    "spiral_disk": "UNKNOT",
    "trefoil": "TORUS", "loose_open_knot": "TORUS",
    "loose_cinquefoil": "TORUS",
    "ring_like_open_knot": "TORUS", "occluded_knot": "TORUS",
    "torus_2_5": "TORUS", "torus_2_7": "TORUS",
    "torus_2_9": "TORUS", "torus_3_4": "TORUS", "torus_3_5": "TORUS",
    "figure8": "TWIST",
}

# topologicalId → family mapping (used by T05_pair_relation)
_TOPOID_FAMILY = {
    "unknot": "UNKNOT",
    "trefoil_3_1": "TORUS", "cinquefoil_5_1": "TORUS",
    "torus_7_1": "TORUS", "torus_9_1": "TORUS",
    "torus_3_4": "TORUS", "torus_3_5": "TORUS",
    "figure8_4_1": "TWIST",
}

# topologicalId → crossing number (used by T07_complexity_comparison)
_TOPOID_CROSSING = {
    "unknot": 0,
    "trefoil_3_1": 3, "cinquefoil_5_1": 5, "figure8_4_1": 4,
    "torus_7_1": 7, "torus_9_1": 9, "torus_3_4": 8, "torus_3_5": 10,
}


def _linked_component_count_from_graph(metadata: dict[str, Any]) -> int | None:
    """Count components linked to another component from recorded link graph metadata."""
    edges = metadata.get("linkageEdges")
    bgroups = metadata.get("brunnianGroups")
    if not edges and not bgroups:
        return None

    linked: set[int] = set()
    if isinstance(edges, list):
        for pair in edges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                i, j = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            linked.add(i)
            linked.add(j)

    if isinstance(bgroups, list):
        for group in bgroups:
            if not isinstance(group, (list, tuple)):
                continue
            for idx in group:
                try:
                    linked.add(int(idx))
                except (TypeError, ValueError):
                    continue

    return len(linked)


def _link_topology_from_graph(metadata: dict[str, Any]) -> str | None:
    """Classify T03 from recorded link graph metadata when available."""
    raw_edges = metadata.get("linkageEdges")
    raw_groups = metadata.get("brunnianGroups")
    has_graph = isinstance(raw_edges, list) or isinstance(raw_groups, list)
    if not has_graph:
        return None

    edges: set[tuple[int, int]] = set()
    linked: set[int] = set()
    if isinstance(raw_edges, list):
        for pair in raw_edges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                i, j = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            if i == j:
                continue
            edge = (i, j) if i < j else (j, i)
            edges.add(edge)
            linked.update(edge)

    groups: list[set[int]] = []
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if not isinstance(group, (list, tuple)):
                continue
            parsed: set[int] = set()
            for idx in group:
                try:
                    parsed.add(int(idx))
                except (TypeError, ValueError):
                    continue
            if parsed:
                groups.append(parsed)
                linked.update(parsed)

    try:
        num_components = int(metadata["numComponents"])
    except (KeyError, TypeError, ValueError):
        num_components = max(linked) + 1 if linked else 0

    if not linked:
        return "A"
    if 0 < num_components and len(linked) < num_components:
        return "D"

    if groups:
        if len(groups) == 1 and len(groups[0]) == 3 and not edges and len(linked) == 3:
            return "C"
        return "D"

    if not edges:
        return "A"

    degree: dict[int, int] = {idx: 0 for idx in linked}
    adjacency: dict[int, set[int]] = {idx: set() for idx in linked}
    for i, j in edges:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    if any(count > 2 for count in degree.values()):
        return "D"

    seen: set[int] = set()
    for start in linked:
        if start in seen:
            continue
        stack = [start]
        vertices: set[int] = set()
        component_edges = 0
        while stack:
            node = stack.pop()
            if node in vertices:
                continue
            vertices.add(node)
            seen.add(node)
            neighbours = adjacency.get(node, set())
            component_edges += len(neighbours)
            stack.extend(neighbour for neighbour in neighbours if neighbour not in vertices)
        if component_edges // 2 != len(vertices) - 1:
            return "D"

    return "B"


def get_ground_truth(metadata: dict, task_id: str) -> str | None:
    """Derive ground truth answer from metadata for a given task."""
    kt = metadata.get("knotType", "")
    family = metadata.get("family", "")
    crossing = metadata.get("crossingNumber") or 0
    is_link = metadata.get("isLink", False) or (kt in _LINK_TYPES)

    if task_id == "T01_structure_classification":
        # A=closed unknotted ring, B=knot, C=open-ended unknotted rope,
        # D=link, E=multiple unlinked ropes, F=mixed
        if family in ("MIXED", "mixed"):
            return "F"
        if family in ("MULTI_ROPE", "multi_rope"):
            return "E"
        if family in ("LINK_GROUP", "link_group"):
            return "D"
        if is_link or family in ("link", "LINK"):
            if kt == "unlinked_rings":
                return "E"
            return "D"
        knot_family = _FAMILY_MAP.get(kt)
        if knot_family is None:
            return None
        if knot_family == "UNKNOT":
            return "C" if metadata.get("isOpenRope") else "A"
        return "B"

    if task_id == "T02_component_count":
        num = metadata.get("numComponents")
        if num is None:
            if kt in ("hopf_link", "unlinked_rings"):
                num = 2
            elif kt == "chain":
                num = metadata.get("chainLinks", 3)
            else:
                num = 1
        return str(int(num))

    if task_id == "T03_link_topology":
        # A=unlinked, B=chain (including two-ring paired links),
        # C=all-interlocked (Borromean), D=mixed.
        # Legacy open/tangled multi-rope samples are separable, so they score as unlinked.
        if family in ("MULTI_ROPE", "multi_rope"):
            return "A"
        graph_topology = _link_topology_from_graph(metadata)
        if graph_topology is not None:
            return graph_topology
        if kt == "unlinked_rings":
            return "A"
        if kt in ("hopf_link", "chain", "double_hopf", "link_cluster"):
            return "B"
        if kt == "borromean":
            return "C"
        if kt in ("chain_plus_free", "hopf_plus_free") or family in ("MIXED", "mixed"):
            return "D"
        if family in ("LINK_GROUP", "link_group"):
            return "B"
        return None

    if task_id == "T05_pair_relation":
        equiv = metadata.get("label_equivalent", False)
        if equiv:
            return "A"
        tid_a = metadata.get("topologicalIdA", "")
        tid_b = metadata.get("topologicalIdB", "")
        fam_a = _TOPOID_FAMILY.get(tid_a, "OTHER")
        fam_b = _TOPOID_FAMILY.get(tid_b, "OTHER")
        is_knot_a = fam_a != "UNKNOT"
        is_knot_b = fam_b != "UNKNOT"

        if is_knot_a != is_knot_b:
            return "D"
        if fam_a == fam_b:
            return "B"
        return "C"

    if task_id == "T06_anchor_match":
        return metadata.get("correct_letter")

    if task_id == "T07_complexity_comparison":
        # Works for both knot pairs and multi_rope pairs
        tid_a = metadata.get("topologicalIdA", "")
        tid_b = metadata.get("topologicalIdB", "")
        c_a = _TOPOID_CROSSING.get(tid_a)
        c_b = _TOPOID_CROSSING.get(tid_b)
        # Fallback: use crossingNumberA/B or numComponentsA/B for multi_rope
        if c_a is None:
            c_a = metadata.get("crossingNumberA") or metadata.get("numComponentsA")
        if c_b is None:
            c_b = metadata.get("crossingNumberB") or metadata.get("numComponentsB")
        if c_a is None or c_b is None:
            return None
        c_a, c_b = int(c_a), int(c_b)
        if c_a > c_b:
            return "A"
        if c_a == c_b:
            return "B"
        return "C"

    if task_id == "T04_link_property":
        # New T04 schema: given metadata.removed_color, return the canonical
        # comma-separated list of color names of remaining rings that are now
        # FREE (not linked to any other remaining ring), based on the
        # linkageEdges + brunnianGroups + ringColors recorded by the batch
        # generator. The empty list canonicalizes to the empty string "".
        ring_colors = metadata.get("ringColors") or []
        edges = metadata.get("linkageEdges") or []
        bgroups = metadata.get("brunnianGroups") or []
        removed_color = metadata.get("removed_color")
        if not ring_colors or removed_color is None:
            return None
        if removed_color not in ring_colors:
            return None
        removed_idx = ring_colors.index(removed_color)
        n = len(ring_colors)
        remaining = {i for i in range(n) if i != removed_idx}
        not_free: set[int] = set()
        # Pairwise edges: each ring is non-free if some incident edge survives.
        for pair in edges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            i, j = int(pair[0]), int(pair[1])
            if i in remaining and j in remaining:
                not_free.add(i)
                not_free.add(j)
        # Brunnian groups: if every ring of the group is still present, the
        # whole group remains locked together (so none of its members is free).
        for group in bgroups:
            if not isinstance(group, (list, tuple)):
                continue
            members = [int(g) for g in group]
            if all(g in remaining for g in members):
                not_free.update(members)
        free_colors = [ring_colors[i] for i in sorted(remaining) if i not in not_free]
        return _canonicalize_color_list(free_colors)

    if task_id == "T05_linked_count":
        graph_count = _linked_component_count_from_graph(metadata)
        if graph_count is not None:
            return str(graph_count)
        # For mixed scenes: use numLinkedComponents
        n = metadata.get("numLinkedComponents")
        if n is not None:
            return str(int(n))
        # Legacy open/tangled multi-rope samples were not treated as linked.
        if family in ("MULTI_ROPE", "multi_rope"):
            return "0"
        # For links: all components are linked
        if is_link and kt != "unlinked_rings":
            num = metadata.get("numComponents")
            if num is not None:
                return str(int(num))
        # For unlinked_rings or single: 0
        if kt == "unlinked_rings" or not is_link:
            return "0"
        return None

    return None


# ═══════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════

def score_answer(parsed: str, gt: str, task_id: str) -> dict[str, Any]:
    """Score a parsed answer against ground truth."""
    if gt is None or parsed == "unclear":
        return {"correct": False, "partial": False, "detail": "missing"}

    if task_id in ("T02_component_count", "T05_linked_count"):
        try:
            return {
                "correct": int(parsed) == int(gt),
                "partial": abs(int(parsed) - int(gt)) == 1,
                "detail": f"pred={parsed},gt={gt}",
            }
        except ValueError:
            return {"correct": False, "partial": False, "detail": "parse_fail"}

    if task_id == "T04_link_property":
        pred_canon = _canonicalize_color_answer_value(parsed)
        gt_canon = _canonicalize_color_answer_value(gt)
        pred_set = set(pred_canon.split(",")) if pred_canon else set()
        gt_set = set(gt_canon.split(",")) if gt_canon else set()
        # Partial credit: Jaccard >= 0.5 over the union of colors mentioned
        union = pred_set | gt_set
        jaccard = (len(pred_set & gt_set) / len(union)) if union else 1.0
        return {
            "correct": pred_set == gt_set,
            "partial": (pred_set != gt_set) and jaccard >= 0.5,
            "detail": f"pred={pred_canon},gt={gt_canon},jaccard={jaccard:.2f}",
        }

    return {
        "correct": parsed == gt,
        "partial": False,
        "detail": "exact" if parsed == gt else f"pred={parsed},gt={gt}",
    }


# ═══════════════════════════════════════════════════════════════════
# SAMPLE FILTERING
# ═══════════════════════════════════════════════════════════════════

_LINK_TYPES = {"hopf_link", "unlinked_rings", "chain", "borromean",
               "double_hopf", "link_cluster", "chain_plus_free", "hopf_plus_free"}
_T04_RELIABLE_LINK_TYPES = {"chain", "borromean", "chain_plus_free", "hopf_plus_free"}


def sample_applicable(metadata: dict, task_id: str) -> bool:
    """Check if this sample should be tested with this task."""
    task = TASKS.get(task_id)
    if not task:
        return False

    applicable = task["applicable"]
    kt = metadata.get("knotType", "")
    family = metadata.get("family", "")
    is_link = metadata.get("isLink", False) or (kt in _LINK_TYPES)

    is_pair = "label_equivalent" in metadata
    is_anchor_query = metadata.get("_query_type") == "anchor_match"

    # all_single_image: applies to all non-pair, non-anchor samples
    if applicable == "all_single_image":
        return not is_pair and not is_anchor_query

    if applicable == "pair" and not is_pair:
        return False
    if applicable == "pair":
        return True

    if applicable == "anchor_match" and not is_anchor_query:
        return False
    if applicable == "anchor_match":
        return True

    # T04: link_removable — applies only to scenes that have ≥3 rings AND
    # the batch generator marked t04Colored=True (i.e. a per-ring solid-color
    # variant was rendered and ringColors / linkageEdges metadata is present).
    if applicable == "link_removable":
        if is_pair or is_anchor_query:
            return False
        if kt not in _T04_RELIABLE_LINK_TYPES:
            return False
        if not metadata.get("t04Colored"):
            return False
        ring_colors = metadata.get("ringColors") or []
        try:
            num_components = int(metadata.get("numComponents") or 0)
        except (TypeError, ValueError):
            num_components = 0
        if not (3 <= num_components <= len(_T04_COLOR_PALETTE)):
            return False
        if len(ring_colors) != num_components:
            return False
        return all(str(color).strip().lower() in _T04_COLOR_PALETTE for color in ring_colors)

    # T09: multi_and_rope — applies to multi-component and multi_rope samples (not pairs)
    if applicable == "multi_and_rope":
        if is_pair or is_anchor_query:
            return False
        return (is_link or family in ("MULTI_ROPE", "multi_rope",
                                       "MIXED", "mixed"))

    # T15: multi_rope — applies to MULTI_ROPE family samples
    if applicable == "multi_rope":
        return family in ("MULTI_ROPE", "multi_rope")

    return True


# ═══════════════════════════════════════════════════════════════════
# T06 ANCHOR MATCH — query construction
# ═══════════════════════════════════════════════════════════════════

def build_anchor_match_queries(
    all_samples: list[dict[str, Any]],
    data_dir: Path,
) -> list[dict[str, Any]]:
    """Build deterministic anchor-match queries from all dataset samples.

    Groups samples by topologicalId (the true topological equivalence class).
    For each group, constructs queries where the anchor and correct candidate
    are from the same topologicalId but different visual renderings, and
    3 distractors come from different topologicalIds.

    Returns a list of query dicts.
    """
    rng = random.Random(42)  # deterministic seed

    by_topoid: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        tid = s.get("topologicalId")
        if tid:
            by_topoid[tid].append(s)

    all_topoids = sorted(by_topoid.keys())
    queries: list[dict[str, Any]] = []
    query_num = 0

    for topoid in all_topoids:
        samples = by_topoid[topoid]
        if len(samples) < 2:
            continue

        distractor_pool = [s for s in all_samples
                           if s.get("topologicalId") and s["topologicalId"] != topoid]
        if len(distractor_pool) < 3:
            continue

        # Create up to n//2 queries per topologicalId
        for round_idx in range(len(samples) // 2):
            anchor = samples[round_idx * 2]
            correct = samples[round_idx * 2 + 1]

            # Pick 3 distractors from 3 different topologicalIds
            distractor_by_topoid = defaultdict(list)
            for s in distractor_pool:
                distractor_by_topoid[s["topologicalId"]].append(s)
            other_topoids = [t for t in distractor_by_topoid if t != topoid]
            rng.shuffle(other_topoids)
            distractors = []
            for dt in other_topoids:
                if len(distractors) >= 3:
                    break
                distractors.append(rng.choice(distractor_by_topoid[dt]))

            if len(distractors) < 3:
                continue

            # Place correct answer at deterministic random position
            candidates = distractors[:]
            correct_pos = rng.randint(0, 3)
            candidates.insert(correct_pos, correct)
            letters = ["A", "B", "C", "D"]

            # Resolve image paths
            anchor_img = _resolve_sample_image(anchor, data_dir)
            cand_imgs = [_resolve_sample_image(c, data_dir) for c in candidates]

            if anchor_img is None or any(ci is None for ci in cand_imgs):
                continue

            query = {
                "_query_type": "anchor_match",
                "query_id": f"anchor_{query_num:03d}",
                "id": f"anchor_{query_num:03d}",
                "correct_letter": letters[correct_pos],
                "topologicalId": topoid,
                "anchor_knotType": anchor["knotType"],
                "correct_knotType": correct["knotType"],
                "anchor_image": str(anchor_img),
                "candidate_images": [str(ci) for ci in cand_imgs],
                "anchor_id": anchor["id"],
                "correct_id": correct["id"],
                "distractor_ids": [d["id"] for d in distractors],
                "difficulty": "medium",
                "difficulty_score": 0.35,
            }
            queries.append(query)
            query_num += 1

    return queries


# Image roots for generated datasets.
IMAGE_ROOT_NAME = "images"
LEGACY_IMAGE_SUBDIRS = ("singles", "links", "multi_rope", "link_groups", "mixed")


def _resolve_sample_image(sample_meta: dict, data_dir: Path) -> Path | None:
    """Resolve the iso_fr image path for a sample from any dataset category."""
    kt = sample_meta.get("knotType", "")
    images = sample_meta.get("images", [])

    # Find iso_fr image
    for img in images:
        fname = img.get("filename", "") if isinstance(img, dict) else str(img)
        if "iso_fr" in fname:
            direct = data_dir / IMAGE_ROOT_NAME / fname
            if direct.exists():
                return direct
            # Try all category subdirectories
            for subdir in LEGACY_IMAGE_SUBDIRS:
                c = data_dir / subdir / kt / fname
                if c.exists():
                    return c
            # Also try flat paths
            for prefix in [data_dir / kt, data_dir]:
                c = prefix / fname
                if c.exists():
                    return c

    # Fallback: try any image
    if images:
        fname = images[0].get("filename", "") if isinstance(images[0], dict) else str(images[0])
        direct = data_dir / IMAGE_ROOT_NAME / fname
        if direct.exists():
            return direct
        for subdir in LEGACY_IMAGE_SUBDIRS:
            c = data_dir / subdir / kt / fname
            if c.exists():
                return c
        for prefix in [data_dir / kt, data_dir]:
            c = prefix / fname
            if c.exists():
                return c

    return None


# ═══════════════════════════════════════════════════════════════════
# VLM API INTERACTION
# ═══════════════════════════════════════════════════════════════════

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_vlm(client: OpenAI, model: str, prompt: str, image_paths: list[str],
            image_labels: list[str] | None = None,
            system_prompt: str | None = None) -> str:
    """Send image(s) + prompt to VLM and return raw text response."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    prompt_has_inline_image_labels = "[Image " in prompt
    for i, path in enumerate(image_paths):
        if image_labels:
            label = image_labels[i]
        elif len(image_paths) > 1:
            label = f"[Image {i + 1}]"
        else:
            label = None

        if label and not prompt_has_inline_image_labels:
            content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(path)}",
                    "detail": "high",
                },
            }
        )
    is_reasoning = any(model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5-mini"))
    is_internvl = "internvl" in model.lower()

    messages: list[dict[str, Any]] = []
    # Add system prompt (reasoning models don't support system role — use developer)
    if system_prompt:
        if is_reasoning:
            messages.append({"role": "developer", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if is_internvl:
        kwargs["max_tokens"] = 2048
    elif is_reasoning:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_completion_tokens"] = 200
        kwargs["temperature"] = 0
    for attempt in range(8):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            return (text or "").strip()
        except Exception as e:
            err = str(e)
            if any(kw in err for kw in ("请求过于频繁", "限流", "rate", "throttl")):
                wait = 3 * (2 ** attempt)
                time.sleep(wait)
                continue
            if "配额" in err or "quota" in err.lower():
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


# ═══════════════════════════════════════════════════════════════════
# IMAGE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def _extract_image_name(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("filename", "path", "file", "image", "image_path"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _extract_sample_records(node: Any) -> list[dict]:
    if isinstance(node, list):
        records: list[dict] = []
        for item in node:
            records.extend(_extract_sample_records(item))
        return records

    if not isinstance(node, dict):
        return []

    if any(key in node for key in ("filename", "images", "candidate_images", "label_equivalent")):
        return [node]

    records: list[dict] = []
    for key, value in node.items():
        if key in {"version", "generatedAt", "generated_at", "stats", "task"}:
            continue
        records.extend(_extract_sample_records(value))
    return records


def load_dataset(data_dir: Path, difficulty: str | None, limit: int | None) -> list[dict]:
    dataset_meta_path = data_dir / "dataset_metadata.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"dataset_metadata.json not found in {data_dir}")

    with dataset_meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        samples = raw
    elif isinstance(raw, dict):
        nested = raw.get("samples")
        samples = nested if isinstance(nested, list) else _extract_sample_records(raw)
    else:
        samples = []

    if difficulty:
        samples = [
            sample for sample in samples
            if (sample.get("difficulty") or compute_difficulty(sample).get("level")) == difficulty
        ]
    if limit is not None:
        samples = samples[:limit]
    return samples


def _resolve_image_path(data_dir: Path, metadata: dict[str, Any], image_name: str) -> Path | None:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path if image_path.exists() else None

    candidates: list[Path] = [(data_dir / IMAGE_ROOT_NAME / image_name).resolve()]
    knot_type = str(metadata.get("knotType") or "").strip()
    if knot_type:
        candidates.extend(
            (data_dir / subdir / knot_type / image_name).resolve()
            for subdir in LEGACY_IMAGE_SUBDIRS
        )
        candidates.append((data_dir / knot_type / image_name).resolve())
    candidates.extend([
        (data_dir / "pairs" / image_name).resolve(),
        (data_dir / image_name).resolve(),
    ])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return None


def _pick_single_image(metadata: dict[str, Any], data_dir: Path) -> Path | None:
    """Select one image for single-image tasks. Prefer iso_fr angle."""
    images = metadata.get("images", [])
    if isinstance(images, list) and images:
        names = [n for n in (_extract_image_name(i) for i in images) if n]
        if names:
            preferred = next((n for n in names if "iso_fr" in n), None)
            if not preferred:
                preferred = next((n for n in names if "front" in n), names[0])
            path = _resolve_image_path(data_dir, metadata, preferred)
            if path:
                return path

    for key in ("image", "image_path", "filename"):
        name = _extract_image_name(metadata.get(key))
        if name:
            path = _resolve_image_path(data_dir, metadata, name)
            if path:
                return path

    return None


def _pick_pair_images(metadata: dict[str, Any], data_dir: Path) -> list[Path]:
    """Select two images for pair comparison tasks."""
    paired_keys = [
        ("image1", "image2"), ("image_1", "image_2"),
        ("img1", "img2"), ("left_image", "right_image"),
    ]
    for k1, k2 in paired_keys:
        n1 = _extract_image_name(metadata.get(k1))
        n2 = _extract_image_name(metadata.get(k2))
        if n1 and n2:
            p1 = _resolve_image_path(data_dir, metadata, n1)
            p2 = _resolve_image_path(data_dir, metadata, n2)
            if p1 and p2:
                return [p1, p2]

    for key in ("pair_images", "pairImages", "image_pair", "imagePair"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) >= 2:
            names = [n for n in (_extract_image_name(v) for v in values) if n]
            if len(names) >= 2:
                p1 = _resolve_image_path(data_dir, metadata, names[0])
                p2 = _resolve_image_path(data_dir, metadata, names[1])
                if p1 and p2:
                    return [p1, p2]

    if "label_equivalent" in metadata:
        images = metadata.get("images")
        if isinstance(images, list) and len(images) >= 2:
            names = [n for n in (_extract_image_name(v) for v in images) if n]
            if len(names) >= 2:
                p1 = _resolve_image_path(data_dir, metadata, names[0])
                p2 = _resolve_image_path(data_dir, metadata, names[1])
                if p1 and p2:
                    return [p1, p2]

    return []


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def print_report(results: list[dict[str, Any]]) -> None:
    """Print accuracy report grouped by task, difficulty, and trap type."""
    print("\n" + "=" * 70)
    print("VLM BENCHMARK RESULTS")
    print("=" * 70)

    if not results:
        print("\nNo valid results to report.")
        return

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    print(f"\nOverall: {correct}/{total} = {correct/total:.1%}")

    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    for task_id in TASKS:
        task_results = by_task.get(task_id, [])
        if not task_results:
            continue

        n = len(task_results)
        c = sum(1 for r in task_results if r.get("correct"))
        acc = c / n if n > 0 else 0.0
        print(f"\n[{task_id}] {acc:.1%} ({c}/{n})")

        for level in ("easy", "medium", "hard"):
            level_results = [r for r in task_results if r.get("difficulty_level") == level]
            if level_results:
                lc = sum(1 for r in level_results if r.get("correct"))
                ln = len(level_results)
                print(f"  {level:8s}: {lc/ln:.1%} ({lc}/{ln})")

    # Deception split analysis
    print("\n-- Deception Split --")
    for label, filt in [("deceptive", True), ("standard", False)]:
        subset = [r for r in results if r.get("is_deceptive", False) == filt]
        if subset:
            c = sum(1 for r in subset if r.get("correct"))
            print(f"  {label:15s}: {c/len(subset):.1%} ({c}/{len(subset)})")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def _task_list_from_arg(tasks_arg: str) -> list[str]:
    if tasks_arg.strip().lower() == "all":
        return list(TASKS.keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Knot Topology Benchmark v10")
    parser.add_argument("--data_dir", default="./dataset",
                        help="Directory containing images and metadata")
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated task IDs or 'all'")
    parser.add_argument("--model", default="gpt-4o",
                        help="VLM model name (default: gpt-4o)")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max metadata samples to evaluate")
    parser.add_argument("--difficulty", default=None,
                        choices=["easy", "medium", "hard"],
                        help="Filter by difficulty level")
    parser.add_argument("--phrasing", type=int, default=None,
                        help="Use a specific phrasing index.")
    parser.add_argument("--all-phrasings", action="store_true",
                        help="Run all available phrasings for each task")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling API")
    parser.add_argument("--base-url", default=None,
                        help="Custom API base URL")
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable name for API key (default: OPENAI_API_KEY)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API calls (default: 1)")
    args = parser.parse_args()

    api_key_env = args.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    is_local = args.base_url is not None
    if not api_key and not args.dry_run and not is_local:
        print(f"[ERROR] {api_key_env} is not set.")
        sys.exit(1)
    if OpenAI is None and not args.dry_run:
        print("[ERROR] Missing optional dependency: openai. Install it with: pip install openai")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data_dir does not exist: {data_dir}")
        sys.exit(1)

    client_kwargs: dict[str, Any] = {
        "api_key": api_key or "local",
        "timeout": 120.0 if is_local else 60.0,
    }
    if is_local:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs) if not args.dry_run else None
    task_ids = _task_list_from_arg(args.tasks)
    valid_task_ids = [t for t in task_ids if t in TASKS]
    for t in task_ids:
        if t not in TASKS:
            print(f"[WARN] Unknown task: {t}")
    if not valid_task_ids:
        print("[ERROR] No valid tasks selected.")
        sys.exit(1)

    samples = load_dataset(data_dir, args.difficulty, args.limit)
    print(f"Loaded {len(samples)} samples from {data_dir}")

    all_results: list[dict[str, Any]] = []
    scanned = 0

    # ── Collect all jobs ──
    jobs: list[dict[str, Any]] = []
    metadata_path = data_dir / "dataset_metadata.json"
    regular_tasks = list(valid_task_ids)

    for metadata in samples:
        diff_info = compute_difficulty(metadata)
        scanned += 1
        sample_name = str(
            metadata.get("id") or metadata.get("sample_id") or f"sample_{scanned:04d}"
        )
        single_image = _pick_single_image(metadata, data_dir)
        pair_images = _pick_pair_images(metadata, data_dir)

        for task_id in regular_tasks:
            if not sample_applicable(metadata, task_id):
                continue

            task = TASKS[task_id]
            input_type = task["applicable"]

            image_paths: list[Path] = []
            if input_type in ("all_single_image", "link_removable",
                              "multi_and_rope", "multi_rope"):
                if not single_image:
                    continue
                image_paths = [single_image]
            elif input_type == "pair":
                if len(pair_images) != 2:
                    continue
                image_paths = pair_images
            else:
                continue

            gt = get_ground_truth(metadata, task_id)
            if gt is None:
                continue

            if args.all_phrasings:
                phrasing_indices = list(range(num_phrasings(task_id)))
            elif args.phrasing is not None:
                phrasing_indices = [args.phrasing]
            else:
                phrasing_indices = [None]

            for pidx in phrasing_indices:
                if pidx is not None:
                    prompt = build_prompt(task_id, pidx)
                    question_id = f"{task_id}_Q{pidx + 1:02d}"
                else:
                    prompt = build_default_prompt(task_id)
                    question_id = f"{task_id}_Q00"

                if args.dry_run:
                    print(f"  [DRY] {question_id} | sample={sample_name} | "
                          f"gt={gt} | images={[p.name for p in image_paths]}")
                    continue

                jobs.append({
                    "kind": "regular",
                    "task_id": task_id,
                    "question_id": question_id,
                    "phrasing_index": pidx,
                    "sample_name": sample_name,
                    "meta_path": str(metadata_path),
                    "metadata": metadata,
                    "input_type": input_type,
                    "image_paths": [str(p) for p in image_paths],
                    "image_labels": None,
                    "gt": gt,
                    "prompt": prompt,
                    "system_prompt": None,
                    "diff_info": diff_info,
                })

    # ── Execute jobs (with optional concurrency) ──
    def _run_job(job: dict[str, Any]) -> dict[str, Any] | None:
        try:
            raw = ask_vlm(client, args.model, job["prompt"],
                         job["image_paths"], job["image_labels"],
                         system_prompt=job.get("system_prompt"))
            parsed = parse_answer(raw, job["task_id"])
            scoring = score_answer(parsed, job["gt"], job["task_id"])

            md = job["metadata"]
            result = {
                "task": job["task_id"],
                "question_id": job["question_id"],
                "phrasing_index": job["phrasing_index"],
                "sample_id": job["sample_name"],
                "metadata_path": job["meta_path"],
                "input_type": job["input_type"],
                "image_paths": job["image_paths"],
                "knot_type": md.get("knotType"),
                "topological_id": md.get("topologicalId"),
                "crossing_number": md.get("crossingNumber"),
                "slackness": md.get("slackness", 0),
                "difficulty_score": job["diff_info"]["score"],
                "difficulty_level": job["diff_info"]["level"],
                "trap_type": md.get("trap_type"),
                "is_deceptive": md.get("isDeceptive", False),
                "ground_truth": job["gt"],
                "vlm_answer": parsed,
                "vlm_raw": raw,
                "prompt_used": job["prompt"],
                "correct": scoring["correct"],
                "partial": scoring.get("partial", False),
                "score_detail": scoring.get("detail", ""),
            }

            icon = "+" if scoring["correct"] else ("~" if scoring.get("partial") else "-")
            print(
                f"[{icon}] {job['question_id']} | {job['sample_name']} | "
                f"gt={job['gt']} | ans={parsed} | {job['diff_info']['level']}",
                flush=True,
            )
            return result
        except Exception as exc:
            print(f"[ERROR] {job['question_id']} on {job['sample_name']}: {exc}", flush=True)
            return None

    concurrency = args.concurrency
    if not args.dry_run and jobs:
        print(f"\nRunning {len(jobs)} evaluations (concurrency={concurrency})...\n", flush=True)
        if concurrency <= 1:
            for job in jobs:
                result = _run_job(job)
                if result:
                    all_results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_run_job, job): job for job in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)

    if not args.dry_run:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_results)} results to {args.output}")
        print_report(all_results)
    else:
        print(f"\n[DRY RUN] Scanned {scanned} samples × {len(valid_task_ids)} tasks")


if __name__ == "__main__":
    main()
