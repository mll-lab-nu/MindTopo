from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GYM_ROOT = PROJECT_ROOT / "gym"
REPO_ROOT = PROJECT_ROOT.parent
for path in (GYM_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from env import (  # noqa: E402
    arrangement_to_grid,
    difficulty_label_for_grid,
    grid_shape_for_difficulty,
    normalize_difficulty,
    sample_swap_2d_puzzle_instance,
    validate_grid_shape,
)
from jsonl_export import to_relative_path, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_DIFFICULTIES = "easy,medium,hard"
DEFAULT_REPEATS = 3
DEFAULT_SEED_START = 1
TASK_NAME = "order_swap_2d_puzzle"


def parse_grid_list(spec: str) -> List[Tuple[int, int]]:
    values: List[Tuple[int, int]] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Grid spec must use ROWSxCOLS, got: {part!r}")
        row_text, col_text = part.split("x", 1)
        values.append(validate_grid_shape(int(row_text), int(col_text)))
    if not values:
        raise ValueError("--grids must contain at least one grid shape.")
    return values


def parse_difficulty_list(spec: str) -> List[str]:
    values: List[str] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        difficulty = normalize_difficulty(part)
        if difficulty is None:
            raise ValueError(f"Difficulty must be easy, medium, or hard, got: {part!r}")
        values.append(difficulty)
    if not values:
        raise ValueError("--difficulties must contain at least one difficulty.")
    return values


def build_episode_id(*, grid_rows: int, grid_cols: int, seed: int) -> str:
    return f"order_swap_2d_puzzle_grid_{grid_rows}x{grid_cols}_seed_{seed}"


def build_question_row(
    *,
    output_dir: Path,
    grid_rows: int,
    grid_cols: int,
    seed: int,
    repeat_index: int,
    budget_multiplier: float,
    target_difficulty: str | None = None,
) -> Dict[str, Any]:
    resolved_difficulty = target_difficulty or difficulty_label_for_grid(grid_rows, grid_cols)
    sampled = sample_swap_2d_puzzle_instance(grid_rows, grid_cols, seed=seed)
    step_budget = max(1, math.ceil(int(sampled["theoretical_min_steps"]) * float(budget_multiplier)))
    reset_config = {
        "gridRows": int(grid_rows),
        "gridCols": int(grid_cols),
        "difficulty": resolved_difficulty,
        "seed": int(seed),
        "initialArrangement": list(sampled["initial_arrangement"]),
        "goalArrangement": list(sampled["goal_arrangement"]),
        "selectedBlockIds": list(sampled["selected_block_ids"]),
        "theoreticalMinSteps": int(sampled["theoretical_min_steps"]),
        "stepBudget": int(step_budget),
        "budgetMultiplier": float(budget_multiplier),
    }
    if target_difficulty is not None:
        reset_config["targetDifficulty"] = target_difficulty
    return {
        "id": build_episode_id(grid_rows=grid_rows, grid_cols=grid_cols, seed=seed),
        "category": ["order", TASK_NAME, "interactive"],
        "type": "interactive",
        "meta_info": {
            "task_name": TASK_NAME,
            "config": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir),
            "level": f"grid_{grid_rows}x{grid_cols}",
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "difficulty": resolved_difficulty,
            "grid_shape": f"{grid_rows}x{grid_cols}",
            "max_actions_per_traj": int(step_budget),
            "initial_state": {
                "initialGrid": arrangement_to_grid(sampled["initial_arrangement"], grid_rows, grid_cols),
                "goalGrid": arrangement_to_grid(sampled["goal_arrangement"], grid_rows, grid_cols),
                "reset_config": reset_config,
            },
            "legal_action_format": '{"answer": {"row": row, "col": col}}',
        },
        "images": [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate order_swap_2d_puzzle interactive question JSONL.")
    parser.add_argument(
        "--difficulties",
        default=DEFAULT_DIFFICULTIES,
        help="Comma-separated difficulties. Each difficulty maps to a seed-determined grid shape.",
    )
    parser.add_argument(
        "--grids",
        default="",
        help="Optional explicit comma-separated grid shapes, e.g. 3x3,3x4,4x4. Overrides --difficulties.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Rows to generate for each difficulty or explicit grid shape.",
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="First deterministic seed. Defaults to 1.")
    parser.add_argument("--budget-multiplier", type=float, default=1.2)
    parser.add_argument("--output-json", default=str(DEFAULT_QUESTION_JSONL))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.budget_multiplier <= 0:
        raise ValueError("--budget-multiplier must be positive.")
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    next_seed = int(args.seed_start)
    explicit_grid_shapes = parse_grid_list(args.grids) if str(args.grids).strip() else []
    difficulties = [] if explicit_grid_shapes else parse_difficulty_list(args.difficulties)
    total_rows = (len(explicit_grid_shapes) or len(difficulties)) * int(args.repeats)
    progress = tqdm(total=total_rows, desc="2D Swap Puzzle generate", unit="sample")
    try:
        if explicit_grid_shapes:
            for grid_rows, grid_cols in explicit_grid_shapes:
                for repeat_index in range(int(args.repeats)):
                    rows.append(
                        build_question_row(
                            output_dir=output_json.parent,
                            grid_rows=grid_rows,
                            grid_cols=grid_cols,
                            seed=next_seed,
                            repeat_index=repeat_index,
                            budget_multiplier=float(args.budget_multiplier),
                        )
                    )
                    next_seed += 1
                    progress.update(1)
        else:
            for difficulty in difficulties:
                for repeat_index in range(int(args.repeats)):
                    grid_rows, grid_cols = grid_shape_for_difficulty(difficulty, next_seed)
                    rows.append(
                        build_question_row(
                            output_dir=output_json.parent,
                            grid_rows=grid_rows,
                            grid_cols=grid_cols,
                            seed=next_seed,
                            repeat_index=repeat_index,
                            budget_multiplier=float(args.budget_multiplier),
                            target_difficulty=difficulty,
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
