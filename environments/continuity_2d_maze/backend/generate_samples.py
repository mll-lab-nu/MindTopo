from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts import (
    build_bar_removal_question,
    build_reachability_set_question,
    formulate_bar_removal_prompt,
    formulate_reachability_set_prompt,
)
from vite_server import ViteFrontendServer

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "generate_samples.py requires the `playwright` package. "
        "Install it with: pip install -r backend/requirements.txt && playwright install chromium"
    ) from exc


TASK_NAME = "continuity_2d_maze"
CATEGORY = "continuity"
LEVEL = "reasoning"
QUESTION_TYPES = ("reachability_set", "bar_removal")
# Accepted choices for --question-type. "all" expands to each concrete qtype
# in `QUESTION_TYPES` run sequentially (convenience for producing a
# balanced multi-qtype batch in one command). Output files are suffixed
# per qtype so the two runs don't overwrite each other.
ALL_QUESTION_TYPES_SENTINEL = "all"
QUESTION_TYPE_CHOICES = QUESTION_TYPES + (ALL_QUESTION_TYPES_SENTINEL,)
# Default target point for reachability_set: first sampled point gets label "A".
DEFAULT_TARGET_POINT = "A"

Q2_DEFAULT_BAR_COUNT = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "generate_samples"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "samples.json"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
METADATA_JSON = PROJECT_ROOT / "metadata.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Generate reasoning samples for {TASK_NAME}.")
    parser.add_argument("--count", type=int, default=2, help="Number of scenes per tier (uniform across all selected tiers). Ignored if --count-per-tier is given.")
    parser.add_argument(
        "--count-per-tier",
        default=None,
        help=(
            "Override per-tier counts non-uniformly, e.g. "
            "'easy=167,medium=167,hard=166' for an exact 500-scene total "
            "per qtype. Keys must match --difficulty-tier exactly. When "
            "set, --count is ignored. Useful when the desired total isn't "
            "evenly divisible by len(tiers) × len(qtypes)."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base seed used to derive per-scene seeds.")
    parser.add_argument(
        "--question-type",
        default="reachability_set",
        choices=QUESTION_TYPE_CHOICES,
        help=(
            "Question formulation. "
            "'reachability_set' (default, Q1): for each scene emit ONE "
            "list-generation question asking which other points are connected "
            "to the target (fixed 'A'); gt_answer is a bracket-enclosed "
            "comma-separated list like '[B, D]'. "
            "'bar_removal' (Q2): scene has 2 points guaranteed disconnected "
            "plus N colored bars (recolored Mode-A walls); question asks "
            "which single-bar removals reconnect the pair; answer is a "
            "color list. "
            "'all': run each concrete question_type sequentially with the "
            "same --count/--difficulty-tier, writing per-qtype suffixed "
            "output files (e.g. samples.reachability_set.json + "
            "samples.bar_removal.json). Use this to produce a balanced "
            "multi-qtype batch in one command."
        ),
    )
    parser.add_argument(
        "--difficulty-tier",
        default="easy,medium,hard",
        help=(
            "Metric-based difficulty: comma-separated tiers from "
            "{easy, medium, hard}. Rejection-samples jittered configs until "
            "--count scenes in each target tier are collected. Tier "
            "thresholds differ per question_type (see metric.ts "
            "DIFFICULTY_TIER_THRESHOLDS / Q2_DIFFICULTY_TIER_THRESHOLDS)."
        ),
    )
    parser.add_argument(
        "--bar-count",
        type=int,
        default=Q2_DEFAULT_BAR_COUNT,
        help="Q2 bar_removal only: number of colored bars per scene (default 5, max 6).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto-pick free port.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument(
        "--output-question-jsonl",
        default=str(DEFAULT_QUESTION_JSONL),
        help="Per-sample meta_jsonl file (one question per line).",
    )
    parser.add_argument(
        "--max-attempts-per-scene",
        type=int,
        default=200,
        help=(
            "Safety cap for rejection sampling: the batch aborts if more "
            "than (total_required * this) attempts elapse without filling "
            "all tiers. Default 200."
        ),
    )
    parser.add_argument(
        "--dump-metric-distribution",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Calibration helper. Generate N random jittered configs "
            "(box depends on --question-type), compute the difficulty "
            "scene_score for each, and print a histogram + P33/P66 "
            "quantiles. Does NOT write samples.json or question.jsonl. "
            "Use to (re)calibrate T_low/T_high in metric.ts."
        ),
    )
    return parser


# iter D15 (2026-04-24): tier is now defined DIRECTLY by the maze
# configuration (grid size + which wall types are allowed + density),
# NOT by a post-hoc score. No score, no rejection sampling on score.
#
# Per-meeting decision (Qineng/Jianwen 2026-04-24):
#   easy   = small (4×4), only standard full H/V walls, sparse density
#   medium = 5×5, full + partial walls allowed, moderate density
#   hard   = 6×6, full + partial + diagonal walls, denser
#
# A scene is accepted as long as the underlying generator can produce
# a valid scene (Q2: at least one bar's single removal connects A-B).
# Tier is attached directly from the target tier passed in, with no
# score-based override.
#
# MUST stay in sync with frontend/src/main.ts (TIER_CONFIGS_Q1/Q2).

# wall_density is a per-tier RANGE (lo, hi); each scene's density is sampled
# deterministically from `seed` via _jitter_uniform below. The grid + wall
# type allowances are still fixed per tier (geometry IS the difficulty).
Q1_TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy":   {"grid_size": 4, "wd_lo": 0.35, "wd_hi": 0.45, "diagonal_ratio": 0.0,  "partial_ratio": 0.0},
    "medium": {"grid_size": 5, "wd_lo": 0.40, "wd_hi": 0.50, "diagonal_ratio": 0.0,  "partial_ratio": 0.30},
    "hard":   {"grid_size": 6, "wd_lo": 0.45, "wd_hi": 0.55, "diagonal_ratio": 0.25, "partial_ratio": 0.30},
}

Q2_TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy":   {"grid_size": 4, "wd_lo": 0.35, "wd_hi": 0.45, "diagonal_ratio": 0.0,  "partial_ratio": 0.0},
    "medium": {"grid_size": 5, "wd_lo": 0.40, "wd_hi": 0.50, "diagonal_ratio": 0.0,  "partial_ratio": 0.30},
    "hard":   {"grid_size": 6, "wd_lo": 0.45, "wd_hi": 0.55, "diagonal_ratio": 0.25, "partial_ratio": 0.30},
}

# Q1 point_count jitters in [Q1_POINT_COUNT_LO, Q1_POINT_COUNT_HI] (inclusive)
# per scene, deterministically derived from seed.
Q1_POINT_COUNT_LO = 3
Q1_POINT_COUNT_HI = 5

# iter D16/D17 extra rules. easy stays unconstrained; medium/hard reject
# trivial-looking scenes (close points / single-answer Q2 / one side
# trapped in a tiny pocket).
Q1_MIN_PAIRWISE_DIST_BY_TIER: Dict[str, int] = {
    "easy": 0,
    "medium": 3,
    "hard": 3,
}
Q2_MIN_PAIRWISE_DIST_BY_TIER: Dict[str, int] = {
    "easy": 0,
    "medium": 3,
    "hard": 5,
}
Q2_MIN_CORRECT_REMOVALS_BY_TIER: Dict[str, int] = {
    "easy": 1,
    "medium": 2,
    "hard": 2,
}
Q2_MIN_AREA_FRACTION_BY_TIER: Dict[str, float] = {
    "easy": 0.0,
    "medium": 1.0 / 4.0,
    "hard": 1.0 / 3.0,
}

PARTIAL_BC_SPLIT = 0.5


def _lcg_step(seed: int, salt: int = 0) -> int:
    """Tiny shared LCG. MUST match frontend/src/main.ts::lcgStep exactly so a
    given (seed, salt) yields the same jitter on both sides — the UI tier
    button and the CLI tier_config both call this to derive wall_density /
    point_count from `seed` without dragging in a real RNG library."""
    s = (int(seed) ^ int(salt)) & 0xFFFFFFFF
    s = (s * 1103515245 + 12345) & 0xFFFFFFFF
    return s


def _jitter_uniform(seed: int, lo: float, hi: float, salt: int = 0) -> float:
    s = _lcg_step(seed, salt)
    return float(lo) + (s % 10000) / 10000.0 * (float(hi) - float(lo))


def _jitter_int_inclusive(seed: int, lo: int, hi: int, salt: int = 1) -> int:
    """Returns int in [lo, hi] inclusive."""
    s = _lcg_step(seed, salt)
    span = int(hi) - int(lo) + 1
    return int(lo) + (s % span)

_VALID_TIERS = ("easy", "medium", "hard")


def parse_difficulty_tier(spec: str) -> List[str]:
    """Parse 'easy,medium,hard' → ['easy', 'medium', 'hard'] with dedup,
    preserving order."""
    result: List[str] = []
    for chunk in spec.split(","):
        part = chunk.strip().lower()
        if not part:
            continue
        if part not in _VALID_TIERS:
            raise ValueError(
                f"--difficulty-tier must be from {_VALID_TIERS}, got '{part}'"
            )
        if part not in result:
            result.append(part)
    if not result:
        raise ValueError("--difficulty-tier requires at least one tier")
    return result


def resolve_per_tier_counts(
    args: argparse.Namespace,
    target_tiers: List[str],
) -> Dict[str, int]:
    """Return {tier: count} for the run. If --count-per-tier is set, parse
    it and validate keys match target_tiers exactly; otherwise fall back
    to the uniform --count. Counts must be positive ints."""
    if not getattr(args, "count_per_tier", None):
        return {t: int(args.count) for t in target_tiers}
    parsed: Dict[str, int] = {}
    for chunk in str(args.count_per_tier).split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"--count-per-tier entries must be 'tier=count', got '{part}'"
            )
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k not in _VALID_TIERS:
            raise ValueError(
                f"--count-per-tier tier must be from {_VALID_TIERS}, got '{k}'"
            )
        try:
            n = int(v.strip())
        except ValueError:
            raise ValueError(f"--count-per-tier count for '{k}' is not int: '{v}'")
        if n <= 0:
            raise ValueError(f"--count-per-tier count for '{k}' must be > 0, got {n}")
        parsed[k] = n
    missing = [t for t in target_tiers if t not in parsed]
    extra = [t for t in parsed if t not in target_tiers]
    if missing or extra:
        raise ValueError(
            f"--count-per-tier keys must match --difficulty-tier exactly. "
            f"Missing: {missing}, extra: {extra}, target_tiers: {target_tiers}"
        )
    return parsed


def tier_config(
    seed: int,
    qtype: str,
    tier: str,
    bar_count: int = Q2_DEFAULT_BAR_COUNT,
) -> Dict[str, Any]:
    """Build a per-scene config from the tier's `TIER_CONFIGS` entry
    (iter D15). Grid + wall types are fixed per tier; wall_density jitters
    inside the tier's [wd_lo, wd_hi] band; Q1 also jitters point_count in
    [3, 5]. All jitter is seed-driven via the shared LCG so the CLI and UI
    produce identical scenes for the same seed."""
    base = (Q2_TIER_CONFIGS if qtype == "bar_removal" else Q1_TIER_CONFIGS)[tier]
    cfg: Dict[str, Any] = {
        "seed": seed,
        "grid_size": int(base["grid_size"]),
        "wall_density": _jitter_uniform(seed, base["wd_lo"], base["wd_hi"], salt=1),
        "diagonal_ratio": float(base["diagonal_ratio"]),
        "partial_ratio": float(base["partial_ratio"]),
        "partial_bc_split": PARTIAL_BC_SPLIT,
    }
    if qtype == "bar_removal":
        cfg["question_type"] = "bar_removal"
        cfg["bar_count"] = int(bar_count)
        cfg["min_correct_removals"] = Q2_MIN_CORRECT_REMOVALS_BY_TIER[tier]
        cfg["min_pairwise_distance"] = Q2_MIN_PAIRWISE_DIST_BY_TIER[tier]
        cfg["min_area_fraction"] = Q2_MIN_AREA_FRACTION_BY_TIER[tier]
    else:
        cfg["point_count"] = _jitter_int_inclusive(
            seed, Q1_POINT_COUNT_LO, Q1_POINT_COUNT_HI, salt=2
        )
        cfg["min_pairwise_distance"] = Q1_MIN_PAIRWISE_DIST_BY_TIER[tier]
    return cfg


def connected_set_answer(target: str, pairs: List[Dict[str, Any]]) -> str:
    """Given a list of pair dicts (each with a_name, b_name, connected), return
    a JSON-array string like `["A", "D"]` listing the OTHER points connected
    to `target`. Empty set → `[]`. Lex-sorted for determinism; scorer is
    order-insensitive."""
    names: List[str] = []
    for pair in pairs:
        if not pair.get("connected"):
            continue
        a, b = pair.get("a_name"), pair.get("b_name")
        if a == target and b is not None:
            names.append(b)
        elif b == target and a is not None:
            names.append(a)
    names_sorted = sorted(set(names))
    return json.dumps(names_sorted)


def sample_to_question_row(sample: Dict[str, Any], config_rel: str) -> Dict[str, Any]:
    """Convert an internal sample dict into a meta_jsonl question row.

    Field layout follows `environments/meta_jsonl/reasoning_question.meta.jsonl`.

    iter E1 (dev_trace_5): top-level `type` is uniformly `"reasoning"`
    regardless of qtype; the concrete question_type is preserved in
    `category[2]` and `meta_info.question_type`.
    """
    metadata = sample.get("metadata", {})
    qtype = sample.get("question_type", "reachability_set")
    meta_info: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "question_type": qtype,
        "config": config_rel,
        "seed": metadata.get("seed"),
        "repeat_index": metadata.get("repeat_index"),
        "difficulty": sample.get("difficulty"),
        "grid_size": metadata.get("grid_size"),
        "wall_density": metadata.get("wall_density"),
        "point_count": metadata.get("point_count"),
        "points": metadata.get("points", []),
    }
    if qtype == "reachability_set":
        meta_info["target"] = metadata.get("target")
    elif qtype == "bar_removal":
        meta_info["target_pair"] = metadata.get("target_pair")
        meta_info["correct_removals"] = metadata.get("correct_removals")
        meta_info["bars"] = metadata.get("bars")
    return {
        "id": sample["id"],
        "category": ["continuity", TASK_NAME, qtype],
        "type": "reasoning",
        "meta_info": meta_info,
        "question": sample["question"],
        "images": sample["images"],
        "gt_answer": sample["answer"],
    }


def write_question_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


async def fetch_scene(page: Any, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    metadata = await page.evaluate(
        """async (config) => {
            return window.topoBench.generate(config);
        }""",
        config,
    )
    image_data_url = await page.evaluate("() => window.topoBench.screenshot()")
    if not isinstance(image_data_url, str) or "," not in image_data_url:
        raise RuntimeError("Frontend screenshot() did not return a valid data URL.")
    png_bytes = base64.b64decode(image_data_url.split(",", 1)[1])
    return metadata, png_bytes


def record_scene_and_samples(
    *,
    samples: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    sample_index: int,
    image_root: Path,
    output_json: Path,
    args: argparse.Namespace,
    seed: int,
    base_seed: int,
    repeat_index: int,
    difficulty_label: str,
    grid_size: int,
    wall_density: float,
    metadata: Dict[str, Any],
    png_bytes: bytes,
) -> int:
    """Write PNG, append to scenes/samples lists, return new sample_index.

    iter E1 (dev_trace_5, 2026-04-24): question_id is flat
    `{TASK_NAME}_{tier}_seed_{seed}_{qtype}_{NNNN}`. The 4-digit suffix
    is `repeat_index + 1` (1-based per-tier).

    iter E2 (2026-04-24): image is written as `image_root/<qid>/image.png`
    — one PNG per per-question sub-folder; the filename `image.png` is
    fixed and irrelevant since the sub-folder name carries the qid. This
    makes the layout uniform across question_types (bar_removal /
    reachability_set share the same image_root with no per-qtype split)
    and aligns with the cross-environment convention of one image per
    question_id sub-folder.

    iter E3 (2026-04-24): question_id now embeds the per-scene `seed`
    (= base_seed + global_attempt) instead of the batch-level `base_seed`.
    Previously every id in a batch carried the same `seed_12345` token
    even though `meta_info.seed` already varied; the id is now actually
    informative and reproducible per-scene."""
    question_id = (
        f"{TASK_NAME}_{difficulty_label}_seed_{seed}"
        f"_{args.question_type}_{repeat_index + 1:04d}"
    )
    image_path = image_root / question_id / "image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(png_bytes)
    relative_image = str(image_path.relative_to(output_json.parent))

    diff_metric = metadata.get("difficulty") or {}

    scenes.append(
        {
            "scene_index": len(scenes),
            "seed": seed,
            "difficulty": difficulty_label,
            "grid_size": grid_size,
            "wall_density": wall_density,
            "point_count": int(metadata.get("point_count", 4)),
            "points": metadata.get("points", []),
            "pairs": metadata.get("pairs", []),
            "wall_shapes": metadata.get("maze", {}).get("shapes", {}),
            "wall_diagonals": metadata.get("maze", {}).get("diagonals", {}),
            "difficulty_score": diff_metric.get("scene_score"),
            "difficulty_tier": diff_metric.get("tier"),
            "difficulty_detour_total": diff_metric.get("detour_total"),
            "difficulty_iso_total": diff_metric.get("iso_total"),
            "difficulty_min_path": diff_metric.get("min_path"),
            "difficulty_area_a": diff_metric.get("area_a"),
            "difficulty_area_b": diff_metric.get("area_b"),
            "images": [relative_image],
        }
    )

    base_metadata = {
        "seed": seed,
        "repeat_index": repeat_index,
        "grid_size": grid_size,
        "wall_density": wall_density,
        "point_count": int(metadata.get("point_count", 4)),
        "points": metadata.get("points", []),
        "difficulty_score": diff_metric.get("scene_score"),
        "difficulty_tier": diff_metric.get("tier"),
        "difficulty_min_path": diff_metric.get("min_path"),
        "difficulty_area_a": diff_metric.get("area_a"),
        "difficulty_area_b": diff_metric.get("area_b"),
    }

    if args.question_type == "reachability_set":
        target = DEFAULT_TARGET_POINT
        answer = connected_set_answer(target, metadata.get("pairs", []))
        point_names = [p.get("name") for p in metadata.get("points", []) if p.get("name")]
        short_question = build_reachability_set_question(target, point_names)
        formulated_prompt = formulate_reachability_set_prompt(short_question)
        sample = {
            "id": question_id,
            "question": formulated_prompt,
            "answer": answer,
            "images": [relative_image],
            "task": TASK_NAME,
            "category": CATEGORY,
            "level": LEVEL,
            "question_type": args.question_type,
            "answer_type": "name_list",
            "difficulty": difficulty_label,
            "metadata": {**base_metadata, "target": target},
        }
        samples.append(sample)
        sample_index += 1
    else:  # bar_removal
        question_info = metadata.get("question", {}) or {}
        correct_removals = list(question_info.get("correct_removals") or [])
        target_pair = list(question_info.get("target_pair") or [])
        bars = list(metadata.get("bars") or [])
        bar_colors = [b.get("color") for b in bars if b.get("color")]
        # correct_removals already lex-sorted by question_gen.ts.
        answer = json.dumps(correct_removals)
        short_question = build_bar_removal_question(target_pair, bar_colors)
        formulated_prompt = formulate_bar_removal_prompt(short_question)
        sample = {
            "id": question_id,
            "question": formulated_prompt,
            "answer": answer,
            "images": [relative_image],
            "task": TASK_NAME,
            "category": CATEGORY,
            "level": LEVEL,
            "question_type": args.question_type,
            "answer_type": "name_list",
            "difficulty": difficulty_label,
            "metadata": {
                **base_metadata,
                "target_pair": target_pair,
                "correct_removals": correct_removals,
                "bars": bars,
            },
        }
        samples.append(sample)
        sample_index += 1

    return sample_index


async def generate_samples_async(args: argparse.Namespace) -> Dict[str, Any]:
    """Tier-based rejection sampling is the only supported mode. Parse
    --difficulty-tier and delegate; defaults are set by build_arg_parser."""
    target_tiers = parse_difficulty_tier(args.difficulty_tier)
    return await generate_samples_tier_mode(args, target_tiers)


async def generate_samples_tier_mode(
    args: argparse.Namespace,
    target_tiers: List[str],
) -> Dict[str, Any]:
    """iter D15: each tier gets a FIXED config (TIER_CONFIGS[tier]); the
    only retry path is when the inner generator throws (e.g. Q2 fails to
    find a valid bar layout for this seed). Re-running with the same
    --seed yields identical batches."""
    output_json = Path(args.output_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    per_tier_counts = resolve_per_tier_counts(args, target_tiers)
    total_required = sum(per_tier_counts.values())
    attempts_cap = total_required * int(args.max_attempts_per_scene)
    base_seed = int(args.seed)

    tier_needed: Dict[str, int] = dict(per_tier_counts)
    tier_collected: Dict[str, int] = {t: 0 for t in target_tiers}
    # iter D15: tracks per-scene diagnostics (min_path / min_area), not a
    # tier-determining score. Used only for the run summary printout.
    tier_scores: Dict[str, List[Dict[str, Any]]] = {t: [] for t in target_tiers}

    samples: List[Dict[str, Any]] = []
    scenes: List[Dict[str, Any]] = []
    sample_index = 0
    global_attempt = 0

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=args.headless,
                args=[
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--disable-gpu-sandbox",
                ],
            )
            page = await browser.new_page(viewport={"width": 1100, "height": 1100})
            await page.goto(server.base_url, wait_until="load")
            await page.wait_for_function(
                "() => !!window.topoBench && typeof window.topoBench.generate === 'function'"
            )

            # iter D15: per-tier loop. Each tier has a FIXED config
            # (TIER_CONFIGS[tier]); a scene is rejected only if the
            # generator throws (e.g. Q2 can't find a valid bar layout
            # for these specific seeds). No score, no tier filtering —
            # the geometry constants ARE the difficulty.
            for target_tier in target_tiers:
                while tier_collected[target_tier] < per_tier_counts[target_tier]:
                    if global_attempt > attempts_cap:
                        raise RuntimeError(
                            f"Generation failed: {global_attempt} attempts "
                            f"exhausted (cap={attempts_cap}). Still needed: {tier_needed}. "
                            f"Likely the {target_tier} TIER_CONFIG produces too "
                            f"many invalid Q2 scenes (no correct_removals); "
                            f"raise its wall_density or grid_size."
                        )
                    seed = base_seed + global_attempt
                    global_attempt += 1
                    cfg = tier_config(
                        seed, args.question_type, target_tier, int(args.bar_count)
                    )
                    try:
                        metadata, png_bytes = await fetch_scene(page, cfg)
                    except Exception:
                        continue

                    diff = metadata.get("difficulty") or {}
                    # iter D15 diagnostic: track min_path / min_area for
                    # the run summary; these are NOT used for tier rejection.
                    aa = diff.get("area_a")
                    ab = diff.get("area_b")
                    diag_record = {
                        "min_path": diff.get("min_path"),
                        "min_area": min(aa, ab) if aa is not None and ab is not None else None,
                    }
                    tier_scores[target_tier].append(diag_record)
                    repeat_index = tier_collected[target_tier]
                    sample_index = record_scene_and_samples(
                        samples=samples,
                        scenes=scenes,
                        sample_index=sample_index,
                        image_root=image_root,
                        output_json=output_json,
                        args=args,
                        seed=seed,
                        base_seed=base_seed,
                        repeat_index=repeat_index,
                        difficulty_label=target_tier,
                        grid_size=int(cfg["grid_size"]),
                        wall_density=float(cfg["wall_density"]),
                        metadata=metadata,
                        png_bytes=png_bytes,
                    )
                    tier_collected[target_tier] += 1
                    tier_needed[target_tier] -= 1

            await browser.close()

    payload: Dict[str, Any] = {
        "task": TASK_NAME,
        "category": CATEGORY,
        "level": LEVEL,
        "schema_version": 5,
        "wall_density_semantics": "absolute_v2_2026_04_22",
        "question_type": args.question_type,
        "difficulty_mode": "metric_based",
        "difficulty_tiers": target_tiers,
        "scenes_per_tier": per_tier_counts,
        "tier_configs": (
            {t: dict(c) for t, c in Q2_TIER_CONFIGS.items()}
            if args.question_type == "bar_removal"
            else {t: dict(c) for t, c in Q1_TIER_CONFIGS.items()}
        ),
        "extra_config": (
            {
                "partial_bc_split": PARTIAL_BC_SPLIT,
                "bar_count": int(args.bar_count),
            }
            if args.question_type == "bar_removal"
            else {
                "partial_bc_split": PARTIAL_BC_SPLIT,
                "point_count_range": [Q1_POINT_COUNT_LO, Q1_POINT_COUNT_HI],
            }
        ),
        "total_attempts": global_attempt,
        "accept_rate": len(scenes) / global_attempt if global_attempt else 0.0,
        "scene_count": len(scenes),
        "sample_count": len(samples),
        "samples": samples,
        "scenes": scenes,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    question_jsonl_path = Path(args.output_question_jsonl).resolve()
    try:
        config_rel = METADATA_JSON.resolve().relative_to(question_jsonl_path.parent).as_posix()
    except ValueError:
        import os
        config_rel = os.path.relpath(METADATA_JSON.resolve(), question_jsonl_path.parent)
    question_rows = [sample_to_question_row(sample, config_rel) for sample in samples]
    write_question_jsonl(question_jsonl_path, question_rows)
    payload["question_jsonl"] = str(question_jsonl_path)
    payload["_tier_scores"] = tier_scores  # stashed for main() summary print
    return payload


async def dump_metric_distribution_async(args: argparse.Namespace) -> None:
    """iter D8b: generate N configs with jittered params, compute the
    difficulty diagnostics (min_path / min_area) for each, print per-tier
    averages. Does not write samples.json / question.jsonl. iter D15: no
    longer suggests T_low/T_high since tier is config-driven."""
    N = int(args.dump_metric_distribution)
    base_seed = int(args.seed)
    rows: List[Dict[str, Any]] = []

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=args.headless,
                args=[
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--disable-gpu-sandbox",
                ],
            )
            page = await browser.new_page(viewport={"width": 1100, "height": 1100})
            await page.goto(server.base_url, wait_until="load")
            await page.wait_for_function(
                "() => !!window.topoBench && typeof window.topoBench.generate === 'function'"
            )

            # iter D15: cycle through easy/medium/hard tier configs in
            # round-robin to get a per-tier sample of (min_path, min_area)
            # for diagnostic. No score / no threshold suggestion — tier is
            # config-driven now.
            for i in range(N):
                tier = _VALID_TIERS[i % len(_VALID_TIERS)]
                config = tier_config(
                    base_seed + i,
                    args.question_type,
                    tier,
                    int(args.bar_count),
                )
                try:
                    metadata = await page.evaluate(
                        """async (config) => window.topoBench.generate(config)""",
                        config,
                    )
                except Exception:
                    continue
                diff = metadata.get("difficulty") or {}
                aa = diff.get("area_a")
                ab = diff.get("area_b")
                rows.append({
                    "i": i,
                    "tier": tier,
                    "grid_size": config["grid_size"],
                    "wd": round(config["wall_density"], 3),
                    "min_path": diff.get("min_path"),
                    "min_area": min(aa, ab) if aa is not None and ab is not None else None,
                    "area_a": aa,
                    "area_b": ab,
                })

            await browser.close()

    n = len(rows)
    if n == 0:
        print("No successful generations (all configs failed).")
        return

    print(f"Collected {n} successful generations (out of {N} attempts).")
    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    for tier in _VALID_TIERS:
        recs = by_tier.get(tier, [])
        if not recs:
            print(f"  [{tier}] no successful samples")
            continue
        mps = [r["min_path"] for r in recs if r.get("min_path") is not None]
        mas = [r["min_area"] for r in recs if r.get("min_area") is not None]
        line = f"  [{tier}] n={len(recs)}  grid={recs[0]['grid_size']}  wd={recs[0]['wd']}"
        if mps:
            line += f"  min_path: avg={sum(mps)/len(mps):.1f} min={min(mps)} max={max(mps)}"
        if mas:
            line += f"  min_area: avg={sum(mas)/len(mas):.1f} min={min(mas)} max={max(mas)}"
        print(line)


def _merge_all_payloads(
    payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine per-qtype payloads from `--question-type all` into a single
    unified samples.json. Samples + scenes are concatenated (scene_index
    renumbered); per-qtype fields (tier_thresholds, jitter_ranges, attempts)
    are kept under `per_qtype` for traceability. Image paths are unchanged
    — every question_id already encodes the qtype, so they coexist in a
    flat image_root without collision."""
    if not payloads:
        raise ValueError("_merge_all_payloads: no payloads")
    merged: Dict[str, Any] = {
        k: v for k, v in payloads[0].items() if k not in {
            "question_type", "tier_thresholds", "jitter_ranges",
            "total_attempts", "accept_rate", "scene_count",
            "sample_count", "samples", "scenes",
            "_tier_scores", "question_jsonl",
        }
    }
    merged["question_type"] = ALL_QUESTION_TYPES_SENTINEL
    merged["per_qtype"] = {}
    merged_samples: List[Dict[str, Any]] = []
    merged_scenes: List[Dict[str, Any]] = []
    total_attempts = 0
    for p in payloads:
        qt = p["question_type"]
        merged["per_qtype"][qt] = {
            "tier_thresholds": p.get("tier_thresholds"),
            "jitter_ranges": p.get("jitter_ranges"),
            "total_attempts": p.get("total_attempts"),
            "accept_rate": p.get("accept_rate"),
            "scene_count": p.get("scene_count"),
            "sample_count": p.get("sample_count"),
        }
        total_attempts += int(p.get("total_attempts", 0) or 0)
        for sc in p.get("scenes", []):
            sc = dict(sc)
            sc["scene_index"] = len(merged_scenes)
            sc["question_type"] = qt
            merged_scenes.append(sc)
        merged_samples.extend(p.get("samples", []))
    merged["samples"] = merged_samples
    merged["scenes"] = merged_scenes
    merged["sample_count"] = len(merged_samples)
    merged["scene_count"] = len(merged_scenes)
    merged["total_attempts"] = total_attempts
    merged["accept_rate"] = (
        len(merged_scenes) / total_attempts if total_attempts else 0.0
    )
    return merged


def _print_run_summary(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    print(
        f"Wrote {payload['sample_count']} samples "
        f"({payload['scene_count']} scenes) to {Path(args.output_json).resolve()}"
    )
    if "question_jsonl" in payload:
        print(f"Wrote {payload['sample_count']} questions to {payload['question_jsonl']}")
    qtype = payload.get("question_type")

    tier_scores = payload.get("_tier_scores", {})
    print(
        f"  total_attempts={payload['total_attempts']}  "
        f"accept_rate={payload['accept_rate']:.2%}"
    )
    # iter D15 diagnostic: print per-tier averages of min_path / min_area
    # (Q2 only fills these; Q1 has no min_path and area_a/b come from a
    # different code path).
    for tier, records in tier_scores.items():
        if not records:
            continue
        mps = [r["min_path"] for r in records if isinstance(r, dict) and r.get("min_path") is not None]
        mas = [r["min_area"] for r in records if isinstance(r, dict) and r.get("min_area") is not None]
        line = f"  [{tier}] scenes={len(records)}"
        if mps:
            line += f"  min_path_avg={sum(mps)/len(mps):.1f}"
        if mas:
            line += f"  min_area_avg={sum(mas)/len(mas):.1f}"
        print(line)

    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for sample in payload["samples"]:
        by_tier.setdefault(sample["difficulty"], []).append(sample)
    if qtype in QUESTION_TYPES:
        for tier, items in by_tier.items():
            lengths = []
            for s in items:
                ans = s["answer"].strip("[]").strip()
                lengths.append(0 if not ans else len([t for t in ans.split(",") if t.strip()]))
            total = len(lengths)
            avg = sum(lengths) / total if total else 0.0
            key = "avg_connected" if qtype == "reachability_set" else "avg_correct_removals"
            print(f"  [{tier}] {qtype}  scenes={total}  {key}={avg:.2f}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dump_metric_distribution is not None:
        asyncio.run(dump_metric_distribution_async(args))
        return

    if args.question_type == ALL_QUESTION_TYPES_SENTINEL:
        # Run each concrete qtype against the SAME output_json /
        # output_question_jsonl / image_root paths, then merge in memory
        # and rewrite both files with the unified payload. Per-qtype runs
        # overwrite the same json/jsonl during their pass; we only care
        # about the in-memory payloads, which we combine via
        # `_merge_all_payloads`. Images already coexist safely in the
        # shared image_root because question_id encodes the qtype.
        payloads: List[Dict[str, Any]] = []
        for qt in QUESTION_TYPES:
            print(f"\n=== Running --question-type {qt} ===")
            sub_args = argparse.Namespace(**vars(args))
            sub_args.question_type = qt
            payload = asyncio.run(generate_samples_async(sub_args))
            _print_run_summary(sub_args, payload)
            payloads.append(payload)

        merged = _merge_all_payloads(payloads)
        output_json_path = Path(args.output_json).resolve()
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        question_jsonl_path = Path(args.output_question_jsonl).resolve()
        try:
            config_rel = METADATA_JSON.resolve().relative_to(
                question_jsonl_path.parent
            ).as_posix()
        except ValueError:
            import os
            config_rel = os.path.relpath(
                METADATA_JSON.resolve(), question_jsonl_path.parent
            )
        all_rows = [
            sample_to_question_row(sample, config_rel)
            for sample in merged["samples"]
        ]
        write_question_jsonl(question_jsonl_path, all_rows)

        print(
            f"\n=== Merged: {merged['sample_count']} samples "
            f"({merged['scene_count']} scenes) "
            f"→ {output_json_path}, {question_jsonl_path} ==="
        )
        return

    payload = asyncio.run(generate_samples_async(args))
    _print_run_summary(args, payload)


if __name__ == "__main__":
    main()
