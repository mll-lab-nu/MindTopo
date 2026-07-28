from __future__ import annotations

import argparse
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

from env import DIFFICULTY_SETUPS, difficulty_label_for_solution_ticks, grid_size_for_difficulty, sample_pipe_instance  # noqa: E402
from jsonl_export import to_relative_path, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_REPEATS = 1
DEFAULT_SEED_START = 1
DEFAULT_DIFFICULTIES = "easy,medium,hard"
DEFAULT_SOLUTION_STEPS = ""
TASK_NAME = "continuity_pipe"


def parse_int_range_list(spec: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = [int(part)]
        for value in candidates:
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return values


def parse_solution_step_specs(spec: str) -> List[Tuple[int, int | None, str]]:
    text = str(spec).strip().lower()
    if not text:
        text = DEFAULT_SOLUTION_STEPS
    specs: List[Tuple[int, int | None, str]] = []
    seen: set[str] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("+"):
            minimum = int(part[:-1])
            label = f"{minimum}+"
            candidates: List[Tuple[int, int | None, str]] = [(minimum, None, label)]
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            minimum = int(start_text)
            maximum = int(end_text) if end_text.strip() else None
            if maximum is None:
                label = f"{minimum}+"
                candidates = [(minimum, None, label)]
            else:
                if maximum < minimum:
                    raise ValueError(f"Invalid --solution-steps range: {spec!r}")
                candidates = [(value, value, str(value)) for value in range(minimum, maximum + 1)]
        else:
            value = int(part)
            candidates = [(value, value, str(value))]
        for minimum, maximum, label in candidates:
            if label in seen:
                continue
            specs.append((minimum, maximum, label))
            seen.add(label)
    if not specs:
        raise ValueError(f"Empty --solution-steps spec: {spec!r}")
    return specs


def parse_difficulty_list(spec: str) -> List[str]:
    difficulties: List[str] = []
    seen: set[str] = set()
    for raw_part in str(spec).split(","):
        difficulty = raw_part.strip().lower()
        if not difficulty:
            continue
        if difficulty not in DIFFICULTY_SETUPS:
            raise ValueError(f"Unsupported --difficulties value: {difficulty!r}")
        if difficulty not in seen:
            difficulties.append(difficulty)
            seen.add(difficulty)
    if not difficulties:
        raise ValueError(f"Empty --difficulties spec: {spec!r}")
    return difficulties


def build_episode_id(*, grid_size: int, solution_steps_spec: str, seed: int) -> str:
    return f"continuity_pipe_grid_{grid_size}_solution_steps_{solution_steps_spec}_seed_{seed}"


def build_question_row(
    *,
    output_dir: Path,
    grid_size: int,
    seed: int,
    repeat_index: int,
    min_solution_steps: int | None,
    max_solution_steps: int | None,
    solution_steps_spec: str,
    difficulty: str | None = None,
) -> Dict[str, Any]:
    sampled = sample_pipe_instance(
        grid_size=grid_size,
        seed=seed,
        difficulty=difficulty,
        min_solution_ticks=min_solution_steps,
        max_solution_ticks=max_solution_steps,
    )
    actual_difficulty = str(difficulty or difficulty_label_for_solution_ticks(int(sampled["solution_ticks"])))
    setup = DIFFICULTY_SETUPS.get(actual_difficulty)
    reset_config = {
        "gridSize": int(grid_size),
        "seed": int(seed),
        "sourceIndex": int(sampled["source_index"]),
        "solvedMasks": list(sampled["solved_masks"]),
        "rotations": list(sampled["rotations"]),
        "maxSteps": int(sampled["max_steps"]),
        "difficulty": actual_difficulty,
        "targetDifficulty": actual_difficulty,
        "minSolutionTicks": int(setup["solution_ticks"][0] if setup else min_solution_steps),
        "maxSolutionTicks": None if max_solution_steps is None else int(max_solution_steps),
    }
    if setup:
        reset_config.update(
            {
                "maxSolutionTicks": int(setup["solution_ticks"][1]),
                "minActivePipeCount": int(setup["active_pipe_count"][0]),
                "maxActivePipeCount": int(setup["active_pipe_count"][1]),
                "minJunctionCount": int(setup["junction_count"][0]),
                "maxJunctionCount": int(setup["junction_count"][1]),
            }
        )
    return {
        "id": build_episode_id(
            grid_size=grid_size,
            solution_steps_spec=solution_steps_spec,
            seed=seed,
        ),
        "category": ["continuity", TASK_NAME, "interactive"],
        "type": "interactive",
        "meta_info": {
            "task_name": TASK_NAME,
            "config": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir),
            "level": f"grid_{grid_size}x{grid_size}",
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "difficulty": actual_difficulty,
            "solution_steps_spec": solution_steps_spec,
            "active_pipe_count": int(sampled["total_pipes"]),
            "junction_count": int(sampled["junction_count"]),
            "max_actions_per_traj": int(sampled["max_steps"]),
            "initial_state": {
                "gridSize": int(grid_size),
                "sourceIndex": int(sampled["source_index"]),
                "solutionTicks": int(sampled["solution_ticks"]),
                "solutionSteps": int(sampled["solution_ticks"]),
                "difficulty": actual_difficulty,
                "totalPipes": int(sampled["total_pipes"]),
                "junctionCount": int(sampled["junction_count"]),
                "reset_config": reset_config,
            },
            "legal_action_format": '{"answer":{"x": {x}, "y": {y}}}',
        },
        "images": [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate continuity_pipe interactive question JSONL.")
    parser.add_argument(
        "--difficulties",
        default=DEFAULT_DIFFICULTIES,
        help="Comma list of difficulty setups to generate when --grid-sizes/--solution-steps are omitted.",
    )
    parser.add_argument("--grid-sizes", default="", help="Comma/range grid sizes for legacy step-target generation. Supported values are 3-6.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Rows to generate for each difficulty setup or legacy grid/solution-step spec.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="First per-sample seed. Seeds increment by 1 for each generated row.",
    )
    parser.add_argument(
        "--solution-steps",
        default=DEFAULT_SOLUTION_STEPS,
        help=(
            "Target oracle solution steps. Bounded ranges expand to each exact step, "
            "e.g. 10-20 means every exact count in that range. When omitted, --difficulties presets are used."
        ),
    )
    parser.add_argument("--output-json", default=str(DEFAULT_QUESTION_JSONL))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    use_difficulty_setups = not args.grid_sizes.strip() and not args.solution_steps.strip()
    if use_difficulty_setups:
        difficulties = parse_difficulty_list(args.difficulties)
        generation_specs = [
            {
                "difficulty": difficulty,
                "grid_size": grid_size_for_difficulty(difficulty),
                "min_solution_steps": None,
                "max_solution_steps": None,
                "solution_steps_spec": difficulty,
            }
            for difficulty in difficulties
        ]
    else:
        grid_sizes = parse_int_range_list(args.grid_sizes or "5,6")
        solution_step_specs = parse_solution_step_specs(args.solution_steps or "10-20")
        for grid_size in grid_sizes:
            if grid_size < 3 or grid_size > 6:
                raise ValueError(f"Unsupported continuity_pipe grid size: {grid_size}. Supported values are 3 through 6.")
        for min_solution_steps, _, _ in solution_step_specs:
            if min_solution_steps < 1:
                raise ValueError("--solution-steps must be positive.")
        generation_specs = [
            {
                "difficulty": None,
                "grid_size": grid_size,
                "min_solution_steps": min_solution_steps,
                "max_solution_steps": max_solution_steps,
                "solution_steps_spec": solution_steps_spec,
            }
            for grid_size in grid_sizes
            for min_solution_steps, max_solution_steps, solution_steps_spec in solution_step_specs
        ]

    rows: List[Dict[str, Any]] = []
    next_seed = int(args.seed_start)
    total_rows = len(generation_specs) * int(args.repeats)
    progress = tqdm(total=total_rows, desc="Continuity Pipe generate", unit="sample")
    try:
        for spec in generation_specs:
            for repeat_index in range(int(args.repeats)):
                seed = next_seed
                next_seed += 1
                rows.append(
                    build_question_row(
                        output_dir=output_json.parent,
                        grid_size=int(spec["grid_size"]),
                        seed=seed,
                        repeat_index=repeat_index,
                        min_solution_steps=spec["min_solution_steps"],
                        max_solution_steps=spec["max_solution_steps"],
                        solution_steps_spec=str(spec["solution_steps_spec"]),
                        difficulty=None if spec["difficulty"] is None else str(spec["difficulty"]),
                    )
                )
                progress.update(1)
    finally:
        progress.close()
    write_jsonl(output_json, rows)
    print(f"Wrote {len(rows)} question rows to {output_json}")


if __name__ == "__main__":
    main()
