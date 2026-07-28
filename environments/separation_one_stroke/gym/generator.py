from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from solver import DIR_TO_DELTA, calculate_regions, edge_key, evaluate_constraints, shortest_solution


Direction = str
Vertex = Tuple[int, int]
Cell = Tuple[int, int]

VALID_DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")
DIFFICULTY_INDEX: Dict[str, int] = {name: index for index, name in enumerate(VALID_DIFFICULTIES)}
VALID_BOARD_SIZES: Tuple[int, ...] = (4, 5, 6)
BOARD_SIZE_INDEX: Dict[int, int] = {size: index for index, size in enumerate(VALID_BOARD_SIZES)}
BOARD_SIZE_DIFFICULTY: Dict[int, str] = {4: "easy", 5: "medium", 6: "hard"}
COLORS: Tuple[str, ...] = ("red", "blue", "green", "yellow", "purple", "cyan", "gray")


@dataclass(frozen=True)
class DifficultyConfig:
    board_sizes: Tuple[int, ...]
    path_lengths: Tuple[int, ...]
    color_counts: Tuple[int, ...]
    shortest_steps: Tuple[int, int]
    min_turns: int
    min_regions: int
    fill_fraction: Tuple[float, float]


DIFFICULTY_CONFIGS: Dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        board_sizes=(4,),
        path_lengths=(10, 12, 14, 16),
        color_counts=(3,),
        shortest_steps=(9, 14),
        min_turns=5,
        min_regions=3,
        fill_fraction=(0.60, 1.0),
    ),
    "medium": DifficultyConfig(
        board_sizes=(5,),
        path_lengths=(14, 16, 18, 20, 22),
        color_counts=(3, 4, 5),
        shortest_steps=(11, 18),
        min_turns=7,
        min_regions=4,
        fill_fraction=(0.60, 1.0),
    ),
    "hard": DifficultyConfig(
        board_sizes=(6,),
        path_lengths=(20, 22, 24, 26, 28),
        color_counts=(5, 6, 7),
        shortest_steps=(19, 20),
        min_turns=8,
        min_regions=5,
        fill_fraction=(0.65, 1.0),
    ),
}


@dataclass(frozen=True)
class GeneratedLevelSpec:
    setup_id: str
    board_size: int
    difficulty: str
    setup_index: int
    setup_seed: int
    generation_attempts: int
    construction_path_length: int
    solution_length: int
    level_json: Dict[str, Any]


def difficulty_label_for_solution_length(solution_length: int) -> str:
    value = int(solution_length)
    if value <= 14:
        return "easy"
    if value <= 18:
        return "medium"
    return "hard"


def difficulty_label_for_board_size(board_size: int) -> str:
    value = int(board_size)
    if value not in BOARD_SIZE_DIFFICULTY:
        valid = ", ".join(f"{size}x{size}" for size in VALID_BOARD_SIZES)
        raise ValueError(f"Unsupported board size '{value}'. Expected one of: {valid}.")
    return BOARD_SIZE_DIFFICULTY[value]


def parse_difficulty_list(spec: Optional[str]) -> List[str]:
    if spec is None or not spec.strip():
        return list(VALID_DIFFICULTIES)

    difficulties: List[str] = []
    for raw_token in spec.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token == "all":
            candidates = VALID_DIFFICULTIES
        else:
            candidates = (token,)
        for candidate in candidates:
            if candidate not in DIFFICULTY_INDEX:
                valid = ", ".join(VALID_DIFFICULTIES)
                raise ValueError(f"Unknown difficulty '{candidate}'. Expected one of: {valid}.")
            if candidate not in difficulties:
                difficulties.append(candidate)

    if not difficulties:
        raise ValueError("No valid difficulties were selected.")
    return difficulties


def parse_board_size_list(spec: Optional[str]) -> List[int]:
    if spec is None or not spec.strip():
        return list(VALID_BOARD_SIZES)

    sizes: List[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip().lower().replace("x", "*")
        if not token:
            continue
        if token == "all":
            candidates = VALID_BOARD_SIZES
        elif "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            candidates = tuple(range(start, end + step, step))
        else:
            size_text = token.split("*", 1)[0]
            try:
                candidates = (int(size_text),)
            except ValueError as exc:
                valid = ", ".join(f"{size}x{size}" for size in VALID_BOARD_SIZES)
                raise ValueError(f"Unknown board size '{raw_token}'. Expected one of: {valid}.") from exc
        for candidate in candidates:
            if candidate not in BOARD_SIZE_INDEX:
                valid = ", ".join(f"{size}x{size}" for size in VALID_BOARD_SIZES)
                raise ValueError(f"Unsupported board size '{candidate}'. Expected one of: {valid}.")
            if candidate not in sizes:
                sizes.append(candidate)

    if not sizes:
        raise ValueError("No valid board sizes were selected.")
    return sizes


def level_signature(level_json: Dict[str, Any]) -> str:
    comparable = {
        "W": int(level_json["W"]),
        "H": int(level_json["H"]),
        "cells": level_json["cells"],
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def _initial_state(level_json: Dict[str, Any]) -> Dict[str, Any]:
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


def _mix_seed(base_seed: int, difficulty: str, setup_index: int) -> int:
    mask = (1 << 64) - 1
    value = int(base_seed) & mask
    value ^= ((setup_index + 1) * 0x9E3779B97F4A7C15) & mask
    value ^= ((DIFFICULTY_INDEX[difficulty] + 1) * 0xBF58476D1CE4E5B9) & mask
    return value & mask


def _mix_board_seed(base_seed: int, board_size: int, setup_index: int) -> int:
    mask = (1 << 64) - 1
    value = int(base_seed) & mask
    value ^= ((setup_index + 1) * 0x9E3779B97F4A7C15) & mask
    value ^= ((BOARD_SIZE_INDEX[int(board_size)] + 1) * 0x94D049BB133111EB) & mask
    return value & mask


def _manhattan(left: Vertex, right: Vertex) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _neighbors(vertex: Vertex, W: int, H: int) -> List[Tuple[Direction, Vertex]]:
    x, y = vertex
    result: List[Tuple[Direction, Vertex]] = []
    for direction, (dx, dy) in DIR_TO_DELTA.items():
        nxt = (x + dx, y + dy)
        if 0 <= nxt[0] < W and 0 <= nxt[1] < H:
            result.append((direction, nxt))
    return result


def _path_to_edges(directions: Sequence[Direction]) -> Set[str]:
    current = (0, 0)
    edges: Set[str] = set()
    for direction in directions:
        dx, dy = DIR_TO_DELTA[direction]
        nxt = (current[0] + dx, current[1] + dy)
        edges.add(edge_key(current, nxt))
        current = nxt
    return edges


def _turn_count(directions: Sequence[Direction]) -> int:
    return sum(1 for index in range(1, len(directions)) if directions[index] != directions[index - 1])


def _random_simple_path(
    *,
    W: int,
    H: int,
    target_length: int,
    rng: random.Random,
    max_nodes: int = 20000,
) -> Optional[List[Direction]]:
    start = (0, 0)
    goal = (W - 1, H - 1)
    visited: Set[Vertex] = {start}
    path: List[Direction] = []
    explored_nodes = 0

    def dfs(current: Vertex, remaining: int) -> bool:
        nonlocal explored_nodes
        explored_nodes += 1
        if explored_nodes > max_nodes:
            return False

        distance = _manhattan(current, goal)
        if distance > remaining or (remaining - distance) % 2 != 0:
            return False
        if remaining == 0:
            return current == goal

        choices = _neighbors(current, W, H)
        rng.shuffle(choices)
        choices.sort(key=lambda item: (_manhattan(item[1], goal) > remaining - 1, rng.random()))
        for direction, nxt in choices:
            if nxt in visited:
                continue
            if nxt == goal and remaining != 1:
                continue
            next_distance = _manhattan(nxt, goal)
            if next_distance > remaining - 1 or (remaining - 1 - next_distance) % 2 != 0:
                continue
            visited.add(nxt)
            path.append(direction)
            if dfs(nxt, remaining - 1):
                return True
            path.pop()
            visited.remove(nxt)
        return False

    if dfs(start, int(target_length)):
        return list(path)
    return None


def _regions_by_id(W: int, H: int, edges: Set[str]) -> Dict[int, List[Cell]]:
    regions = calculate_regions(W, H, edges)
    by_id: Dict[int, List[Cell]] = {}
    for j, row in enumerate(regions):
        for i, region_id in enumerate(row):
            by_id.setdefault(int(region_id), []).append((i, j))
    return by_id


def _build_cells_from_regions(
    *,
    W: int,
    H: int,
    regions_by_id: Dict[int, List[Cell]],
    color_counts: Sequence[int],
    fill_fraction: Tuple[float, float],
    rng: random.Random,
) -> Optional[List[List[Optional[str]]]]:
    available_regions = [region_id for region_id, cells in regions_by_id.items() if cells]
    possible_color_counts = [
        count for count in color_counts if count <= len(available_regions) and count <= len(COLORS)
    ]
    if not possible_color_counts:
        return None

    num_colors = rng.choice(possible_color_counts)
    rng.shuffle(available_regions)
    selected_regions = available_regions[:num_colors]
    colors = list(COLORS)
    rng.shuffle(colors)
    colors = colors[:num_colors]
    min_fraction, max_fraction = fill_fraction

    cells: List[List[Optional[str]]] = [[None for _ in range(W - 1)] for _ in range(H - 1)]
    for color, region_id in zip(colors, selected_regions):
        region_cells = list(regions_by_id[region_id])
        if not region_cells:
            return None
        rng.shuffle(region_cells)
        if len(region_cells) <= 2:
            colored_count = len(region_cells)
        else:
            min_count = max(1, math.ceil(len(region_cells) * min_fraction))
            max_count = max(min_count, min(len(region_cells), math.ceil(len(region_cells) * max_fraction)))
            colored_count = rng.randint(min_count, max_count)
        for i, j in region_cells[:colored_count]:
            cells[j][i] = color
    return cells


def _color_count(cells: Sequence[Sequence[Optional[str]]]) -> int:
    return len({str(cell) for row in cells for cell in row if cell is not None})


def _shortest_steps_in_range(solution_length: int, config: DifficultyConfig) -> bool:
    lower, upper = config.shortest_steps
    return lower <= int(solution_length) <= upper


def _valid_path_lengths_for_board_size(board_size: int, config: DifficultyConfig) -> List[int]:
    vertex_size = int(board_size) + 1
    min_distance = int(board_size) * 2
    return [
        length
        for length in config.path_lengths
        if min_distance <= length < vertex_size * vertex_size and (length - min_distance) % 2 == 0
    ]


def _generate_level_from_config(
    *,
    difficulty: str,
    setup_seed: int,
    config: DifficultyConfig,
    max_attempts: int = 1000,
    solver_timeout_seconds: float = 2.0,
    exclude_signatures: Optional[Set[str]] = None,
) -> GeneratedLevelSpec:
    rng = random.Random(setup_seed)
    excluded = exclude_signatures or set()

    for attempt in range(1, int(max_attempts) + 1):
        board_size = int(rng.choice(config.board_sizes))
        W = board_size + 1
        H = board_size + 1
        target_lengths = _valid_path_lengths_for_board_size(board_size, config)
        if not target_lengths:
            continue

        target_length = rng.choice(target_lengths)
        construction_path = _random_simple_path(
            W=W,
            H=H,
            target_length=target_length,
            rng=rng,
            max_nodes=50000,
        )
        if construction_path is None:
            continue
        if _turn_count(construction_path) < config.min_turns:
            continue

        edges = _path_to_edges(construction_path)
        regions_by_id = _regions_by_id(W, H, edges)
        if len(regions_by_id) < config.min_regions:
            continue

        candidate_cells = _build_cells_from_regions(
            W=W,
            H=H,
            regions_by_id=regions_by_id,
            color_counts=config.color_counts,
            fill_fraction=config.fill_fraction,
            rng=rng,
        )
        if candidate_cells is None:
            continue
        if _color_count(candidate_cells) not in config.color_counts:
            continue

        evaluation = evaluate_constraints(W, H, candidate_cells, edges)
        if not evaluation.get("success"):
            continue

        level_json: Dict[str, Any] = {
            "W": W,
            "H": H,
            "cells": candidate_cells,
            "solution": list(construction_path),
        }
        signature = level_signature(level_json)
        if signature in excluded:
            continue

        shortest = shortest_solution(
            _initial_state(level_json),
            timeout_seconds=solver_timeout_seconds,
            max_depth=config.shortest_steps[1],
        )
        if shortest is None:
            continue
        solution_length = len(shortest)
        if not _shortest_steps_in_range(solution_length, config):
            continue

        level_json["solution"] = list(shortest)
        return GeneratedLevelSpec(
            setup_id=f"generated_{difficulty}",
            board_size=board_size,
            difficulty=difficulty,
            setup_index=0,
            setup_seed=setup_seed,
            generation_attempts=attempt,
            construction_path_length=len(construction_path),
            solution_length=solution_length,
            level_json=level_json,
        )

    raise RuntimeError(
        f"Could not generate a {difficulty} one-stroke level after {max_attempts} attempts. "
        "Increase --generation-max-attempts or --generation-solver-timeout."
    )


def generate_level_for_difficulty(
    *,
    difficulty: str,
    seed: int,
    setup_index: int,
    max_attempts: int = 1000,
    solver_timeout_seconds: float = 2.0,
    exclude_signatures: Optional[Set[str]] = None,
) -> GeneratedLevelSpec:
    difficulty = difficulty.strip().lower()
    if difficulty not in DIFFICULTY_CONFIGS:
        valid = ", ".join(VALID_DIFFICULTIES)
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of: {valid}.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if solver_timeout_seconds <= 0:
        raise ValueError("solver_timeout_seconds must be positive.")

    setup_seed = _mix_seed(seed, difficulty, setup_index)
    spec = _generate_level_from_config(
        difficulty=difficulty,
        setup_seed=setup_seed,
        config=DIFFICULTY_CONFIGS[difficulty],
        max_attempts=max_attempts,
        solver_timeout_seconds=solver_timeout_seconds,
        exclude_signatures=exclude_signatures,
    )
    return GeneratedLevelSpec(
        setup_id=f"generated_{difficulty}_{setup_index + 1:03d}",
        board_size=spec.board_size,
        difficulty=spec.difficulty,
        setup_index=setup_index,
        setup_seed=setup_seed,
        generation_attempts=spec.generation_attempts,
        construction_path_length=spec.construction_path_length,
        solution_length=spec.solution_length,
        level_json=spec.level_json,
    )


def generate_level_for_board_size(
    *,
    board_size: int,
    seed: int,
    setup_index: int,
    max_attempts: int = 1000,
    solver_timeout_seconds: float = 2.0,
    exclude_signatures: Optional[Set[str]] = None,
) -> GeneratedLevelSpec:
    board_size = int(board_size)
    if board_size not in BOARD_SIZE_INDEX:
        valid = ", ".join(f"{size}x{size}" for size in VALID_BOARD_SIZES)
        raise ValueError(f"Unsupported board size '{board_size}'. Expected one of: {valid}.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if solver_timeout_seconds <= 0:
        raise ValueError("solver_timeout_seconds must be positive.")

    setup_seed = _mix_board_seed(seed, board_size, setup_index)
    difficulty = difficulty_label_for_board_size(board_size)
    config = replace(DIFFICULTY_CONFIGS[difficulty], board_sizes=(board_size,))
    spec = _generate_level_from_config(
        difficulty=difficulty,
        setup_seed=setup_seed,
        config=config,
        max_attempts=max_attempts,
        solver_timeout_seconds=solver_timeout_seconds,
        exclude_signatures=exclude_signatures,
    )
    return GeneratedLevelSpec(
        setup_id=f"generated_size_{board_size}x{board_size}_{setup_index + 1:03d}",
        board_size=board_size,
        difficulty=spec.difficulty,
        setup_index=setup_index,
        setup_seed=setup_seed,
        generation_attempts=spec.generation_attempts,
        construction_path_length=spec.construction_path_length,
        solution_length=spec.solution_length,
        level_json=spec.level_json,
    )


def generate_levels_by_difficulty(
    *,
    difficulties: Sequence[str],
    setups_per_difficulty: int,
    seed: int,
    max_attempts: int = 1000,
    solver_timeout_seconds: float = 2.0,
) -> List[GeneratedLevelSpec]:
    if setups_per_difficulty <= 0:
        raise ValueError("setups_per_difficulty must be positive.")

    generated: List[GeneratedLevelSpec] = []
    seen_signatures: Set[str] = set()
    for difficulty in difficulties:
        normalized = difficulty.strip().lower()
        for setup_index in range(int(setups_per_difficulty)):
            spec = generate_level_for_difficulty(
                difficulty=normalized,
                seed=seed,
                setup_index=setup_index,
                max_attempts=max_attempts,
                solver_timeout_seconds=solver_timeout_seconds,
                exclude_signatures=seen_signatures,
            )
            seen_signatures.add(level_signature(spec.level_json))
            generated.append(spec)
    return generated


def generate_levels_by_board_size(
    *,
    board_sizes: Sequence[int],
    setups_per_board_size: int,
    seed: int,
    max_attempts: int = 1000,
    solver_timeout_seconds: float = 2.0,
) -> List[GeneratedLevelSpec]:
    if setups_per_board_size <= 0:
        raise ValueError("setups_per_board_size must be positive.")

    generated: List[GeneratedLevelSpec] = []
    seen_signatures: Set[str] = set()
    for board_size in board_sizes:
        for setup_index in range(int(setups_per_board_size)):
            spec = generate_level_for_board_size(
                board_size=int(board_size),
                seed=seed,
                setup_index=setup_index,
                max_attempts=max_attempts,
                solver_timeout_seconds=solver_timeout_seconds,
                exclude_signatures=seen_signatures,
            )
            seen_signatures.add(level_signature(spec.level_json))
            generated.append(spec)
    return generated
