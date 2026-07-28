from __future__ import annotations

import argparse
import math
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

from generator import generate_level_for_board_size, parse_board_size_list, parse_difficulty_list  # noqa: E402
from jsonl_export import to_relative_path, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_BOARD_SIZES = "4-6"
DEFAULT_REPEATS = 1
DEFAULT_SEED_START = 1
TASK_NAME = "separation_one_stroke"
DIFFICULTY_BOARD_SIZE = {"easy": 4, "medium": 5, "hard": 6}


def build_episode_id(*, board_size: int, seed: int) -> str:
    return f"separation_one_stroke_size_{board_size}x{board_size}_seed_{seed}"


def build_question_row(
    *,
    output_dir: Path,
    level_number: int,
    board_size: int,
    seed: int,
    repeat_index: int,
    max_attempts: int,
    solver_timeout: float,
) -> Dict[str, Any]:
    spec = generate_level_for_board_size(
        board_size=board_size,
        seed=seed,
        setup_index=0,
        max_attempts=max_attempts,
        solver_timeout_seconds=solver_timeout,
    )
    setup_id = f"generated_size_{board_size}x{board_size}_seed_{seed:03d}"
    level_json = dict(spec.level_json)
    return {
        "id": build_episode_id(board_size=board_size, seed=seed),
        "category": ["separation", TASK_NAME, "interactive"],
        "type": "interactive",
        "meta_info": {
            "task_name": TASK_NAME,
            "config": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir),
            "level": int(level_number),
            "setup_id": setup_id,
            "board_size": int(board_size),
            "generated": True,
            "setup_seed": int(spec.setup_seed),
            "generation_attempts": int(spec.generation_attempts),
            "construction_path_length": int(spec.construction_path_length),
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "difficulty": spec.difficulty,
            "solution_length": int(spec.solution_length),
            "max_actions_per_traj": max(1, int(math.ceil(int(spec.construction_path_length) * 1.2))),
            "initial_state": {
                "level_json": level_json,
                "reset_config": {
                    "levelJson": level_json,
                    "boardSize": int(board_size),
                    "seed": int(seed),
                },
            },
            "level_json": level_json,
            "legal_action_format": '{"answer":"U"}',
        },
        "images": [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate separation_one_stroke interactive question JSONL.")
    parser.add_argument("--board-sizes", default=DEFAULT_BOARD_SIZES)
    parser.add_argument(
        "--difficulty",
        "--difficulties",
        dest="difficulty",
        default="",
        help="Generate by difficulty label instead of board size, e.g. easy, medium, hard, easy,hard, or all.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--generation-max-attempts", type=int, default=1000)
    parser.add_argument("--generation-solver-timeout", type=float, default=2.0)
    parser.add_argument("--output-json", default=str(DEFAULT_QUESTION_JSONL))
    return parser


def selected_board_sizes(args: argparse.Namespace) -> List[int]:
    difficulty_spec = str(args.difficulty or "").strip()
    if difficulty_spec:
        return [DIFFICULTY_BOARD_SIZE[difficulty] for difficulty in parse_difficulty_list(difficulty_spec)]
    return parse_board_size_list(args.board_sizes)


def main() -> None:
    args = build_arg_parser().parse_args()
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive.")
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    board_sizes = selected_board_sizes(args)
    rows: List[Dict[str, Any]] = []
    next_seed = int(args.seed_start)
    total_rows = len(board_sizes) * int(args.repeats)
    progress = tqdm(total=total_rows, desc="One Stroke generate", unit="sample")
    try:
        for board_size in board_sizes:
            for repeat_index in range(int(args.repeats)):
                rows.append(
                    build_question_row(
                        output_dir=output_json.parent,
                        level_number=len(rows) + 1,
                        board_size=board_size,
                        seed=next_seed,
                        repeat_index=repeat_index,
                        max_attempts=int(args.generation_max_attempts),
                        solver_timeout=float(args.generation_solver_timeout),
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
