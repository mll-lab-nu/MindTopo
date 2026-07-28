from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GYM_ROOT = PROJECT_ROOT / "gym"
REPO_ROOT = PROJECT_ROOT.parent
for path in (GYM_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from env import (  # noqa: E402
    AUTO_SETUP_VALUE,
    DEFAULT_BOARD_RADIUS_SPEC,
    DEFAULT_INITIAL_BLOCK_COUNT_SPEC,
    DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
    DEFAULT_MIN_WINNING_FIRST_ACTIONS,
    DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
    is_auto_setup_value,
    sample_chat_noir_instance,
    validate_board_radius,
)
from jsonl_export import to_relative_path, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_BOARD_RADIUS = DEFAULT_BOARD_RADIUS_SPEC
DEFAULT_INITIAL_BLOCK_COUNT = DEFAULT_INITIAL_BLOCK_COUNT_SPEC
DEFAULT_CAT_POLICIES = "easy,medium,hard"
DEFAULT_REPEATS = 1
DEFAULT_SEED_START = 1
GENERATION_CAT_POLICIES = ("easy", "medium", "hard")
TASK_NAME = "enclosure_chat_noir"


def parse_int_range_list(spec: str) -> List[int]:
    values: List[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    if not values:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return values


def parse_board_radius_specs(spec: str) -> List[Any]:
    if is_auto_setup_value(spec):
        return [AUTO_SETUP_VALUE]
    return [validate_board_radius(value) for value in parse_int_range_list(spec)]


def parse_initial_block_count_specs(spec: str) -> List[Any]:
    if is_auto_setup_value(spec):
        return [AUTO_SETUP_VALUE]
    return parse_int_range_list(spec)


def difficulty_label_for_policy(cat_policy_name: str) -> str:
    value = str(cat_policy_name).strip().lower()
    if value not in GENERATION_CAT_POLICIES:
        raise ValueError(f"Unsupported generation cat policy: {value}")
    return value


def parse_policy_list(spec: str) -> List[str]:
    policies: List[str] = []
    for raw in str(spec).split(","):
        value = raw.strip().lower()
        if not value:
            continue
        if value not in GENERATION_CAT_POLICIES:
            raise ValueError(
                f"Unsupported generation cat policy: {value}. "
                f"Use one of: {', '.join(GENERATION_CAT_POLICIES)}."
            )
        if value not in policies:
            policies.append(value)
    if not policies:
        raise ValueError("At least one cat policy is required.")
    return policies


def build_episode_id(*, board_radius: int, initial_block_count: int, cat_policy_name: str, seed: int) -> str:
    return (
        f"enclosure_chat_noir_radius_{board_radius}_blocked_{initial_block_count}"
        f"_policy_{cat_policy_name}_seed_{seed}"
    )


def max_actions_per_traj(*, board_cell_count: int, initial_block_count: int) -> int:
    return max(0, int(board_cell_count) - int(initial_block_count) - 1)


def build_question_row(
    *,
    output_dir: Path,
    board_radius_spec: Any,
    initial_block_count_spec: Any,
    cat_policy_name: str,
    seed: int,
    repeat_index: int,
    min_winning_first_actions: Any,
    min_adjacent_winning_first_actions: int,
) -> Dict[str, Any]:
    sampled = sample_chat_noir_instance(
        board_radius_spec,
        initial_block_count=initial_block_count_spec,
        cat_policy=cat_policy_name,
        seed=seed,
        min_winning_first_actions=min_winning_first_actions,
        min_adjacent_winning_first_actions=min_adjacent_winning_first_actions,
    )
    policy = sampled["cat_policy"]
    difficulty = difficulty_label_for_policy(cat_policy_name)
    board_radius = int(sampled["board_radius"])
    initial_block_count = int(sampled["initial_block_count"])
    resolved_min_winning_first_actions = int(sampled["min_winning_first_actions"])
    resolved_min_adjacent_winning_first_actions = int(sampled["min_adjacent_winning_first_actions"])
    setup_level = f"radius_{board_radius}_blocked_{initial_block_count}_policy_{cat_policy_name}"
    reset_config = {
        "boardRadius": board_radius,
        "initialBlockedCount": initial_block_count,
        "catPolicy": cat_policy_name,
        "seed": int(seed),
        "rngSeed": int(seed),
        "minWinningFirstActions": resolved_min_winning_first_actions,
        "minAdjacentWinningFirstActions": resolved_min_adjacent_winning_first_actions,
        "catIndex": int(sampled["cat_index"]),
        "initialBlockedIndices": list(sampled["initial_blocked_indices"]),
    }
    board_cell_count = len(sampled["board"]["cells"])
    actual_initial_block_count = len(sampled["initial_blocked_indices"])
    return {
        "id": build_episode_id(
            board_radius=board_radius,
            initial_block_count=initial_block_count,
            cat_policy_name=cat_policy_name,
            seed=seed,
        ),
        "category": ["enclosure", TASK_NAME, "interactive"],
        "type": "interactive",
        "meta_info": {
            "task_name": TASK_NAME,
            "config": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir),
            "level": difficulty,
            "setup_level": setup_level,
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "difficulty": difficulty,
            "board_radius": board_radius,
            "initial_block_count": initial_block_count,
            "board_radius_spec": str(board_radius_spec),
            "initial_block_count_spec": str(initial_block_count_spec),
            "cat_policy_name": cat_policy_name,
            "cat_policy_level": int(policy["level"]),
            "difficulty_score": float(sampled["difficulty"]["overall_score"]),
            "sample_attempts": int(sampled.get("sample_attempts", 1)),
            "min_winning_first_actions": resolved_min_winning_first_actions,
            "min_adjacent_winning_first_actions": resolved_min_adjacent_winning_first_actions,
            "winning_first_action_count": int(
                (sampled.get("solvability", {}) or {}).get("winning_first_action_count", 0)
            ),
            "winning_first_actions": list(
                (sampled.get("solvability", {}) or {}).get("winning_first_actions", [])
            ),
            "adjacent_winning_first_action_count": int(
                (sampled.get("solvability", {}) or {}).get("adjacent_winning_first_action_count", 0)
            ),
            "adjacent_winning_first_actions": list(
                (sampled.get("solvability", {}) or {}).get("adjacent_winning_first_actions", [])
            ),
            "solvability_filter": dict(sampled.get("solvability", {})),
            "max_actions_per_traj": max_actions_per_traj(
                board_cell_count=board_cell_count,
                initial_block_count=actual_initial_block_count,
            ),
            "initial_state": {
                "boardRadius": board_radius,
                "catIndex": int(sampled["cat_index"]),
                "blockedIndices": list(sampled["initial_blocked_indices"]),
                "catPolicy": policy,
                "initialEscapeDistance": int(sampled["initial_escape_distance"]),
                "reset_config": reset_config,
            },
            "legal_action_format": '{"answer":{cell_index}}',
        },
        "images": [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate enclosure_chat_noir interactive question JSONL.")
    parser.add_argument("--board-radius", "--radii", dest="board_radius", default=DEFAULT_BOARD_RADIUS)
    parser.add_argument("--initial-block-count", "--blocked", dest="initial_block_count", default=DEFAULT_INITIAL_BLOCK_COUNT)
    parser.add_argument("--cat-policies", default=DEFAULT_CAT_POLICIES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--min-winning-first-actions",
        default=DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
        help=(
            'Use "auto" for policy defaults '
            f"(easy={DEFAULT_MIN_WINNING_FIRST_ACTIONS}, medium=5, hard=5), or pass one positive integer override."
        ),
    )
    parser.add_argument(
        "--min-adjacent-winning-first-actions",
        type=int,
        default=DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
        help="Minimum winning first block actions that must be adjacent to the initial cat cell.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_QUESTION_JSONL))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if not is_auto_setup_value(args.min_winning_first_actions) and int(args.min_winning_first_actions) <= 0:
        raise ValueError("--min-winning-first-actions must be positive.")
    if args.min_adjacent_winning_first_actions < 0:
        raise ValueError("--min-adjacent-winning-first-actions must be non-negative.")
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    radii = parse_board_radius_specs(args.board_radius)
    blocked_counts = parse_initial_block_count_specs(args.initial_block_count)
    policies = parse_policy_list(args.cat_policies)

    rows: List[Dict[str, Any]] = []
    next_seed = int(args.seed_start)
    total_rows = len(radii) * len(blocked_counts) * len(policies) * int(args.repeats)
    progress = tqdm(total=total_rows, desc="Chat Noir generate", unit="sample")
    try:
        for radius in radii:
            for blocked_count in blocked_counts:
                for policy in policies:
                    for repeat_index in range(int(args.repeats)):
                        rows.append(
                            build_question_row(
                                output_dir=output_json.parent,
                                board_radius_spec=radius,
                                initial_block_count_spec=blocked_count,
                                cat_policy_name=policy,
                                seed=next_seed,
                                repeat_index=repeat_index,
                                min_winning_first_actions=args.min_winning_first_actions,
                                min_adjacent_winning_first_actions=int(args.min_adjacent_winning_first_actions),
                            )
                        )
                        next_seed += 1
                        progress.update(1)
    finally:
        progress.close()
    write_jsonl(output_json, rows)
    print(f"Wrote {len(rows)} question rows to {output_json}")


if __name__ == "__main__":
    main()
