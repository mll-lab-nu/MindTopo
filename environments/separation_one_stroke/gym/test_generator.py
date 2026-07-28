from __future__ import annotations

import argparse

from generator import (
    difficulty_label_for_board_size,
    generate_level_for_board_size,
    level_signature,
)
from solver import shortest_solution


def _initial_state(level_json: dict) -> dict:
    return {
        "W": int(level_json["W"]),
        "H": int(level_json["H"]),
        "cells": [[cell for cell in row] for row in level_json["cells"]],
        "stroke": [{"x": 0, "y": 0}],
        "currentPos": {"x": 0, "y": 0},
        "done": False,
        "success": False,
        "failureReason": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Procedural generator smoke test for Separation One Stroke.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--solver-timeout", type=float, default=2.0)
    args = parser.parse_args()

    signatures = set()
    for board_size in (4, 5, 6):
        spec = generate_level_for_board_size(
            board_size=board_size,
            seed=args.seed,
            setup_index=0,
            max_attempts=args.max_attempts,
            solver_timeout_seconds=args.solver_timeout,
            exclude_signatures=signatures,
        )
        signatures.add(level_signature(spec.level_json))
        solution = shortest_solution(
            _initial_state(spec.level_json),
            timeout_seconds=args.solver_timeout,
            max_depth=spec.solution_length,
        )
        assert solution is not None
        assert len(solution) == spec.solution_length
        assert difficulty_label_for_board_size(board_size) == spec.difficulty
        assert spec.board_size == board_size
        assert spec.level_json["W"] == board_size + 1
        assert spec.level_json["H"] == board_size + 1
        assert len(spec.level_json["cells"]) == board_size
        assert all(len(row) == board_size for row in spec.level_json["cells"])
        print(
            f"[generated] board_size={board_size}x{board_size} difficulty={spec.difficulty} setup_id={spec.setup_id} "
            f"grid={spec.level_json['W']}x{spec.level_json['H']} "
            f"solution_length={spec.solution_length} attempts={spec.generation_attempts}"
        )


if __name__ == "__main__":
    main()
