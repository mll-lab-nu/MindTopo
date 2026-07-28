from __future__ import annotations

import asyncio
import base64
import io
import math
import random
import threading
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        class _BaseEnv:
            metadata: Dict[str, Any] = {}

            def reset(self, *, seed: Optional[int] = None):
                if seed is not None:
                    random.seed(seed)
                return None

        class _Discrete:
            def __init__(self, n: int) -> None:
                self.n = int(n)

            def sample(self) -> int:
                return random.randrange(self.n)

        class _Box:
            def __init__(self, low: int, high: int, shape: Tuple[int, ...], dtype: Any) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _Spaces:
            Discrete = _Discrete
            Box = _Box

        class _GymFallback:
            Env = _BaseEnv

        gym = _GymFallback()
        spaces = _Spaces()


HEX_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)

INVALID_ACTION = -999
ALLOWED_BOARD_RADII = (3, 4)
MIN_BOARD_RADIUS = min(ALLOWED_BOARD_RADII)
MAX_BOARD_RADIUS = max(ALLOWED_BOARD_RADII)
DEFAULT_CAT_POLICY = "easy"
SAMPLE_ATTEMPTS = 512
SOLVER_ACTION_BRANCH_LIMIT = 6
SOLVER_MAX_NODES = 1000
SOLVER_MAX_DEPTH_BY_RADIUS = {
    3: 7,
    4: 6,
}
AUTO_SETUP_VALUE = "auto"
DEFAULT_BOARD_RADIUS_SPEC = AUTO_SETUP_VALUE
DEFAULT_INITIAL_BLOCK_COUNT_SPEC = AUTO_SETUP_VALUE
DEFAULT_MIN_WINNING_FIRST_ACTIONS = 5
DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC = AUTO_SETUP_VALUE
DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS = 3
MIN_WINNING_FIRST_ACTIONS_BY_POLICY: Dict[str, int] = {
    "easy": 5,
    "medium": 5,
    "hard": 5,
}
INITIAL_BLOCK_COUNT_RANGES: Dict[str, Dict[int, Tuple[int, int]]] = {
    "easy": {
        3: (8, 10),
        4: (10, 13),
    },
    "medium": {
        3: (10, 12),
        4: (13, 16),
    },
    "hard": {
        3: (12, 14),
        4: (16, 19),
    },
}

CAT_POLICIES: Dict[str, Dict[str, Any]] = {
    "static": {
        "name": "static",
        "level": 0,
        "description": (
            "Stationary cat. It never moves after the player blocks a cell."
        ),
    },
    "easy": {
        "name": "easy",
        "level": 1,
        "description": (
            "Mixed walker. On each move it uses shortest-path greedy behavior with 50% probability "
            "and otherwise chooses uniformly from the currently legal adjacent open cells."
        ),
    },
    "medium": {
        "name": "medium",
        "level": 2,
        "description": (
            "Greedy shortest-path runner. It chooses an adjacent open cell that minimizes current "
            "distance to the boundary, with simple lowest-index tie-breaks."
        ),
    },
    "hard": {
        "name": "hard",
        "level": 3,
        "description": (
            "Connectivity-aware greedy runner. It avoids immediate one-move traps when possible, "
            "then prefers shorter boundary distance while preserving more escape connectivity."
        ),
    },
}


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def coord_key(q: int, r: int) -> Tuple[int, int]:
    return (int(q), int(r))


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def validate_board_radius(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("board_radius must be 3 or 4.") from exc
    if parsed not in ALLOWED_BOARD_RADII:
        raise ValueError("board_radius must be 3 or 4.")
    return parsed


def normalize_cat_policy(raw_policy: Any) -> Dict[str, Any]:
    key = str(raw_policy or "").strip().lower()
    return dict(CAT_POLICIES.get(key, CAT_POLICIES[DEFAULT_CAT_POLICY]))


def is_auto_setup_value(value: Any) -> bool:
    return str(value).strip().lower() in {"", AUTO_SETUP_VALUE, "random"}


def min_winning_first_actions_for_policy(cat_policy_name: str) -> int:
    policy_name = normalize_cat_policy(cat_policy_name)["name"]
    if policy_name == "static":
        policy_name = "easy"
    return int(MIN_WINNING_FIRST_ACTIONS_BY_POLICY.get(policy_name, DEFAULT_MIN_WINNING_FIRST_ACTIONS))


def resolve_min_winning_first_actions_spec(value: Any, cat_policy_name: str) -> int:
    if value is None or is_auto_setup_value(value):
        return min_winning_first_actions_for_policy(cat_policy_name)
    return max(1, int(value))


def initial_block_count_range(cat_policy_name: str, board_radius: int) -> Tuple[int, int]:
    policy_name = normalize_cat_policy(cat_policy_name)["name"]
    if policy_name == "static":
        policy_name = "easy"
    if policy_name not in INITIAL_BLOCK_COUNT_RANGES:
        policy_name = DEFAULT_CAT_POLICY
    return INITIAL_BLOCK_COUNT_RANGES[policy_name][validate_board_radius(board_radius)]


def resolve_board_radius_spec(value: Any, rng: random.Random) -> int:
    if is_auto_setup_value(value):
        return int(rng.choice(ALLOWED_BOARD_RADII))
    return validate_board_radius(value)


def resolve_initial_block_count_spec(
    value: Any,
    *,
    board_radius: int,
    cat_policy_name: str,
    rng: random.Random,
) -> int:
    board_radius = validate_board_radius(board_radius)
    if is_auto_setup_value(value):
        minimum, maximum = initial_block_count_range(cat_policy_name, board_radius)
        return int(rng.randint(minimum, maximum))
    board = build_hex_board(board_radius)
    max_block_count = len(board["cells"]) - 1
    return max(0, min(int(value), max_block_count))


def build_hex_board(board_radius: int) -> Dict[str, Any]:
    board_radius = validate_board_radius(board_radius)
    cells: List[Dict[str, Any]] = []
    index_by_coord: Dict[Tuple[int, int], int] = {}
    index = 0
    scale = 1.22

    for r in range(-board_radius, board_radius + 1):
        min_q = max(-board_radius, -r - board_radius)
        max_q = min(board_radius, -r + board_radius)
        for q in range(min_q, max_q + 1):
            s = -q - r
            is_boundary = max(abs(q), abs(r), abs(s)) == board_radius
            cell = {
                "index": index,
                "q": q,
                "r": r,
                "s": s,
                "x": math.sqrt(3) * (q + r / 2.0) * scale,
                "y": -1.5 * r * scale,
                "is_boundary": is_boundary,
                "neighbor_indices": [],
            }
            cells.append(cell)
            index_by_coord[coord_key(q, r)] = index
            index += 1

    for cell in cells:
        neighbors: List[int] = []
        for dq, dr in HEX_DIRECTIONS:
            neighbor_index = index_by_coord.get(coord_key(cell["q"] + dq, cell["r"] + dr))
            if neighbor_index is not None:
                neighbors.append(neighbor_index)
        cell["neighbor_indices"] = neighbors

    boundary_indices = [cell["index"] for cell in cells if cell["is_boundary"]]
    center_index = next(
        (cell["index"] for cell in cells if cell["q"] == 0 and cell["r"] == 0),
        cells[0]["index"],
    )
    return {
        "radius": board_radius,
        "cells": cells,
        "boundary_indices": boundary_indices,
        "center_index": center_index,
    }


def normalize_blocked_indices(
    board: Dict[str, Any],
    blocked_indices: Iterable[Any],
    *,
    cat_index: int,
) -> List[int]:
    valid: List[int] = []
    max_index = len(board["cells"]) - 1
    for raw_value in blocked_indices:
        try:
            index = int(raw_value)
        except (TypeError, ValueError):
            continue
        if index < 0 or index > max_index or index == cat_index:
            continue
        valid.append(index)
    return sorted(set(valid))


def legal_cat_moves(board: Dict[str, Any], cat_index: int, blocked_set: Collection[int]) -> List[int]:
    blocked = set(blocked_set)
    return [
        neighbor_index
        for neighbor_index in board["cells"][cat_index]["neighbor_indices"]
        if neighbor_index not in blocked
    ]


def legal_player_actions(board: Dict[str, Any], cat_index: int, blocked_set: Collection[int]) -> List[int]:
    blocked = set(blocked_set)
    return [
        cell["index"]
        for cell in board["cells"]
        if cell["index"] != cat_index and cell["index"] not in blocked
    ]


def shortest_boundary_path(
    board: Dict[str, Any],
    start_index: int,
    blocked_set: Collection[int],
) -> Optional[List[int]]:
    blocked = set(blocked_set)
    if start_index in blocked:
        return None
    start_cell = board["cells"][start_index]
    if start_cell["is_boundary"]:
        return [start_index]

    queue: deque[int] = deque([start_index])
    parents: Dict[int, Optional[int]] = {start_index: None}

    while queue:
        current = queue.popleft()
        current_cell = board["cells"][current]
        for neighbor_index in current_cell["neighbor_indices"]:
            if neighbor_index in blocked or neighbor_index in parents:
                continue
            parents[neighbor_index] = current
            if board["cells"][neighbor_index]["is_boundary"]:
                path = [neighbor_index]
                cursor: Optional[int] = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parents.get(cursor)
                path.reverse()
                return path
            queue.append(neighbor_index)
    return None


def shortest_boundary_distance(
    board: Dict[str, Any],
    start_index: int,
    blocked_set: Collection[int],
) -> Optional[int]:
    path = shortest_boundary_path(board, start_index, blocked_set)
    if path is None:
        return None
    return len(path) - 1


def hex_distance(board: Dict[str, Any], left_index: int, right_index: int) -> int:
    left = board["cells"][left_index]
    right = board["cells"][right_index]
    return max(
        abs(int(left["q"]) - int(right["q"])),
        abs(int(left["r"]) - int(right["r"])),
        abs(int(left["s"]) - int(right["s"])),
    )


def count_open_neighbor_degree(
    board: Dict[str, Any],
    cell_index: int,
    blocked_set: Collection[int],
) -> int:
    blocked = set(blocked_set)
    return sum(1 for neighbor_index in board["cells"][cell_index]["neighbor_indices"] if neighbor_index not in blocked)


def total_blocked_distance(
    board: Dict[str, Any],
    move_index: int,
    blocked_set: Collection[int],
) -> int:
    blocked = set(blocked_set)
    return sum(hex_distance(board, move_index, blocked_index) for blocked_index in blocked)


def reachable_region_stats(
    board: Dict[str, Any],
    start_index: int,
    blocked_set: Collection[int],
) -> Dict[str, Any]:
    blocked = set(blocked_set)
    if start_index in blocked:
        return {
            "distance": None,
            "reachable_open_count": 0,
            "boundary_count": 0,
            "shortest_path_count": 0,
        }

    queue: deque[int] = deque([start_index])
    distances: Dict[int, int] = {start_index: 0}
    path_counts: Dict[int, int] = {start_index: 1}

    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        for neighbor_index in board["cells"][current]["neighbor_indices"]:
            if neighbor_index in blocked:
                continue
            next_distance = current_distance + 1
            if neighbor_index not in distances:
                distances[neighbor_index] = next_distance
                path_counts[neighbor_index] = path_counts[current]
                queue.append(neighbor_index)
            elif distances[neighbor_index] == next_distance:
                path_counts[neighbor_index] += path_counts[current]

    reachable_boundaries = [
        index
        for index in distances
        if board["cells"][index]["is_boundary"]
    ]
    if not reachable_boundaries:
        return {
            "distance": None,
            "reachable_open_count": len(distances),
            "boundary_count": 0,
            "shortest_path_count": 0,
        }

    shortest_distance = min(distances[index] for index in reachable_boundaries)
    shortest_path_count = sum(
        path_counts[index]
        for index in reachable_boundaries
        if distances[index] == shortest_distance
    )
    return {
        "distance": shortest_distance,
        "reachable_open_count": len(distances),
        "boundary_count": len(reachable_boundaries),
        "shortest_path_count": shortest_path_count,
    }


def has_one_turn_capture_risk(
    board: Dict[str, Any],
    cat_index: int,
    blocked_set: Collection[int],
) -> bool:
    blocked = set(blocked_set)
    if board["cells"][cat_index]["is_boundary"]:
        return False
    if shortest_boundary_path(board, cat_index, blocked) is None:
        return True
    for action_index in legal_player_actions(board, cat_index, blocked):
        blocked_after = set(blocked)
        blocked_after.add(action_index)
        if shortest_boundary_path(board, cat_index, blocked_after) is None:
            return True
    return False


def choose_cat_move(
    board: Dict[str, Any],
    cat_index: int,
    blocked_set: Collection[int],
    cat_policy_name: str,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    blocked = set(blocked_set)
    policy = normalize_cat_policy(cat_policy_name)
    candidate_moves = legal_cat_moves(board, cat_index, blocked)
    if not candidate_moves:
        return {"chosen_move": None, "legal_moves": [], "evaluated_moves": []}

    if policy["name"] == "static":
        return {
            "chosen_move": cat_index,
            "legal_moves": candidate_moves,
            "evaluated_moves": [],
        }

    if policy["name"] == "easy":
        chooser = rng if rng is not None else random
        mode_draw = chooser.random()
        greedy_entries: List[Dict[str, Any]] = []
        for move_index in candidate_moves:
            distance = shortest_boundary_distance(board, move_index, blocked)
            greedy_entries.append(
                {
                    "move_index": move_index,
                    "distance": distance,
                    "score_tuple": (
                        float("inf") if distance is None else float(distance),
                        float(move_index),
                    ),
                }
            )
        greedy_entries.sort(key=lambda entry: tuple(entry["score_tuple"]))
        greedy_move = int(greedy_entries[0]["move_index"])
        mode = "greedy" if mode_draw < 0.5 else "random"
        chosen_move = greedy_move if mode == "greedy" else chooser.choice(candidate_moves)
        evaluated_moves = []
        greedy_by_move = {entry["move_index"]: entry for entry in greedy_entries}
        for move_index in candidate_moves:
            greedy_entry = greedy_by_move[move_index]
            evaluated_moves.append(
                {
                    "move_index": move_index,
                    "distance": greedy_entry["distance"],
                    "degree": count_open_neighbor_degree(board, move_index, blocked),
                    "blocked_distance": total_blocked_distance(board, move_index, blocked),
                    "mode": mode,
                    "mode_draw": mode_draw,
                    "greedy_move": greedy_move,
                    "score_tuple": [0.0 if move_index == chosen_move else 1.0, float(move_index)],
                }
            )
        return {
            "chosen_move": chosen_move,
            "legal_moves": candidate_moves,
            "evaluated_moves": evaluated_moves,
        }

    evaluated_moves: List[Dict[str, Any]] = []
    for move_index in candidate_moves:
        region_stats = reachable_region_stats(board, move_index, blocked)
        distance = region_stats["distance"]
        degree = count_open_neighbor_degree(board, move_index, blocked)
        blocked_distance = total_blocked_distance(board, move_index, blocked)
        one_turn_capture_risk = None
        if policy["name"] == "medium":
            score_tuple: Tuple[float, ...] = (
                float("inf") if distance is None else float(distance),
                float(move_index),
            )
        else:
            one_turn_capture_risk = has_one_turn_capture_risk(board, move_index, blocked)
            score_tuple = (
                0.0 if distance == 0 else 1.0,
                1.0 if one_turn_capture_risk else 0.0,
                float("inf") if distance is None else float(distance),
                -float(region_stats["boundary_count"]),
                -float(region_stats["shortest_path_count"]),
                -float(region_stats["reachable_open_count"]),
                -float(degree),
                float(move_index),
            )
        evaluated_moves.append(
            {
                "move_index": move_index,
                "distance": distance,
                "degree": degree,
                "blocked_distance": blocked_distance,
                "reachable_open_count": region_stats["reachable_open_count"],
                "boundary_count": region_stats["boundary_count"],
                "shortest_path_count": region_stats["shortest_path_count"],
                "one_turn_capture_risk": one_turn_capture_risk,
                "score_tuple": list(score_tuple),
            }
        )

    evaluated_moves.sort(key=lambda entry: tuple(entry["score_tuple"]))
    return {
        "chosen_move": evaluated_moves[0]["move_index"],
        "legal_moves": candidate_moves,
        "evaluated_moves": evaluated_moves,
    }


def compute_difficulty(
    *,
    board_radius: int,
    initial_block_count: int,
    cat_policy_level: int,
) -> Dict[str, Any]:
    return {
        "board_size": int(board_radius),
        "initial_block_count": int(initial_block_count),
        "cat_intelligence": int(cat_policy_level),
        "overall_score": float(int(cat_policy_level)),
    }


def append_unique(values: List[int], value: int) -> None:
    if value not in values:
        values.append(value)


def solver_cat_responses(
    board: Dict[str, Any],
    cat_index: int,
    blocked_set: Collection[int],
    cat_policy_name: str,
) -> List[int]:
    policy_name = normalize_cat_policy(cat_policy_name)["name"]
    if policy_name == "static":
        return [cat_index]
    legal_moves = legal_cat_moves(board, cat_index, blocked_set)
    if not legal_moves:
        return []
    if policy_name == "easy":
        decision = choose_cat_move(board, cat_index, blocked_set, "medium")
        chosen_move = decision.get("chosen_move")
        return [] if chosen_move is None else [int(chosen_move)]
    decision = choose_cat_move(board, cat_index, blocked_set, policy_name)
    chosen_move = decision.get("chosen_move")
    return [] if chosen_move is None else [int(chosen_move)]


def ranked_solver_actions(
    board: Dict[str, Any],
    cat_index: int,
    blocked_set: Collection[int],
    cat_policy_name: str,
    *,
    limit: int,
) -> List[int]:
    blocked = set(blocked_set)
    legal_actions = legal_player_actions(board, cat_index, blocked)
    if not legal_actions:
        return []

    current_path = shortest_boundary_path(board, cat_index, blocked) or []
    priority_cells: set[int] = set(current_path[1:])
    for cell_index in current_path[1:]:
        priority_cells.update(board["cells"][cell_index]["neighbor_indices"])
    cat_move_cells = set(legal_cat_moves(board, cat_index, blocked))
    legal_action_set = set(legal_actions)
    ordered_candidates: List[int] = []
    for cell_index in sorted(cat_move_cells):
        if cell_index in legal_action_set:
            append_unique(ordered_candidates, cell_index)
    for cell_index in current_path[1:]:
        if cell_index in legal_action_set:
            append_unique(ordered_candidates, cell_index)
    for cell_index in sorted(priority_cells):
        if cell_index in legal_action_set:
            append_unique(ordered_candidates, cell_index)
    for cell_index in sorted(legal_actions, key=lambda value: (hex_distance(board, cat_index, value), value)):
        append_unique(ordered_candidates, cell_index)
    actions_to_score = ordered_candidates[: min(len(ordered_candidates), max(limit * 2, 12))]

    scored_actions: List[Tuple[Tuple[float, ...], int]] = []
    for action_index in actions_to_score:
        blocked_after = set(blocked)
        blocked_after.add(action_index)
        path_before_cat_move = shortest_boundary_path(board, cat_index, blocked_after)
        cat_moves_after = legal_cat_moves(board, cat_index, blocked_after)
        immediate_trap = path_before_cat_move is None or not cat_moves_after
        responses = solver_cat_responses(board, cat_index, blocked_after, cat_policy_name)
        response_distances = [
            shortest_boundary_distance(board, response_index, blocked_after)
            for response_index in responses
            if not board["cells"][response_index]["is_boundary"]
        ]
        best_response_distance = (
            float("inf")
            if immediate_trap or not response_distances
            else min(float(distance) if distance is not None else float("inf") for distance in response_distances)
        )
        action_score = (
            0.0 if immediate_trap else 1.0,
            0.0 if action_index in cat_move_cells else 1.0,
            0.0 if action_index in priority_cells else 1.0,
            -best_response_distance,
            float(hex_distance(board, cat_index, action_index)),
            float(action_index),
        )
        scored_actions.append((action_score, action_index))

    scored_actions.sort(key=lambda entry: entry[0])
    return [action_index for _, action_index in scored_actions[: max(1, int(limit))]]


def solve_chat_noir_instance(
    board: Dict[str, Any],
    cat_index: int,
    blocked_indices: Sequence[int],
    cat_policy_name: str,
    *,
    max_steps: int,
    winning_first_action_limit: Optional[int] = None,
    adjacent_winning_first_action_limit: Optional[int] = None,
) -> Dict[str, Any]:
    search_depth = min(
        max(0, int(max_steps)),
        SOLVER_MAX_DEPTH_BY_RADIUS.get(int(board["radius"]), 10),
    )
    node_count = 0
    hit_node_limit = False
    memo: Dict[Tuple[int, Tuple[int, ...], int], bool] = {}
    first_action: Optional[int] = None
    winning_first_actions: List[int] = []
    adjacent_winning_first_actions: List[int] = []
    search_limit = None if winning_first_action_limit is None else max(1, int(winning_first_action_limit))
    adjacent_search_limit = (
        None
        if adjacent_winning_first_action_limit is None
        else max(0, int(adjacent_winning_first_action_limit))
    )

    def can_trap(current_cat_index: int, blocked_tuple: Tuple[int, ...], remaining_steps: int) -> bool:
        nonlocal node_count, hit_node_limit
        node_count += 1
        if node_count > SOLVER_MAX_NODES:
            hit_node_limit = True
            return False
        blocked = set(blocked_tuple)
        if shortest_boundary_path(board, current_cat_index, blocked) is None:
            return True
        if board["cells"][current_cat_index]["is_boundary"] or remaining_steps <= 0:
            return False

        key = (current_cat_index, blocked_tuple, remaining_steps)
        if key in memo:
            return memo[key]

        for action_index in ranked_solver_actions(
            board,
            current_cat_index,
            blocked,
            cat_policy_name,
            limit=SOLVER_ACTION_BRANCH_LIMIT,
        ):
            if action_can_trap(current_cat_index, blocked, action_index, remaining_steps):
                memo[key] = True
                return True

        memo[key] = False
        return False

    def action_can_trap(
        current_cat_index: int,
        blocked: Collection[int],
        action_index: int,
        remaining_steps: int,
    ) -> bool:
        blocked_after = set(blocked)
        blocked_after.add(int(action_index))
        if shortest_boundary_path(board, current_cat_index, blocked_after) is None:
            return True

        responses = solver_cat_responses(board, current_cat_index, blocked_after, cat_policy_name)
        if not responses:
            return True

        next_blocked_tuple = tuple(sorted(blocked_after))
        for next_cat_index in responses:
            if board["cells"][next_cat_index]["is_boundary"]:
                return False
            if not can_trap(next_cat_index, next_blocked_tuple, remaining_steps - 1):
                return False
        return True

    root_blocked_tuple = tuple(sorted(int(value) for value in blocked_indices))
    root_blocked = set(root_blocked_tuple)
    adjacent_action_set = set(legal_cat_moves(board, cat_index, root_blocked))
    if shortest_boundary_path(board, cat_index, root_blocked) is None:
        solvable = True
    elif board["cells"][cat_index]["is_boundary"] or search_depth <= 0:
        solvable = False
    else:
        root_actions = ranked_solver_actions(
            board,
            cat_index,
            root_blocked,
            cat_policy_name,
            limit=max(1, len(legal_player_actions(board, cat_index, root_blocked))),
        )
        for action_index in root_actions:
            if action_can_trap(cat_index, root_blocked, action_index, search_depth):
                winning_first_actions.append(int(action_index))
                if int(action_index) in adjacent_action_set:
                    adjacent_winning_first_actions.append(int(action_index))
                if first_action is None:
                    first_action = int(action_index)
                has_enough_total = search_limit is None or len(winning_first_actions) >= search_limit
                has_enough_adjacent = (
                    adjacent_search_limit is None
                    or len(adjacent_winning_first_actions) >= adjacent_search_limit
                )
                if has_enough_total and has_enough_adjacent:
                    break
        solvable = bool(winning_first_actions)

    return {
        "solvable": bool(solvable),
        "max_steps": int(max_steps),
        "search_depth": int(search_depth),
        "nodes_searched": int(node_count),
        "hit_node_limit": bool(hit_node_limit),
        "branch_limit": int(SOLVER_ACTION_BRANCH_LIMIT),
        "first_action": first_action,
        "winning_first_actions": winning_first_actions,
        "winning_first_action_count": len(winning_first_actions),
        "winning_first_action_search_limit": search_limit,
        "adjacent_winning_first_actions": adjacent_winning_first_actions,
        "adjacent_winning_first_action_count": len(adjacent_winning_first_actions),
        "adjacent_winning_first_action_search_limit": adjacent_search_limit,
    }


def sample_chat_noir_instance(
    board_radius: Any = DEFAULT_BOARD_RADIUS_SPEC,
    *,
    initial_block_count: Any = DEFAULT_INITIAL_BLOCK_COUNT_SPEC,
    cat_index: Optional[int] = None,
    cat_policy: str = DEFAULT_CAT_POLICY,
    seed: Optional[int] = None,
    min_winning_first_actions: Any = DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
    min_adjacent_winning_first_actions: int = DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    policy = normalize_cat_policy(cat_policy)
    requested_board_radius = board_radius
    requested_initial_block_count = initial_block_count
    board_radius = resolve_board_radius_spec(board_radius, rng)
    initial_block_count = resolve_initial_block_count_spec(
        initial_block_count,
        board_radius=board_radius,
        cat_policy_name=policy["name"],
        rng=rng,
    )
    board = build_hex_board(board_radius)
    chosen_cat_index = board["center_index"] if cat_index is None else int(cat_index)
    if chosen_cat_index < 0 or chosen_cat_index >= len(board["cells"]):
        raise ValueError(f"cat_index out of range for radius {board_radius}: {chosen_cat_index}")
    if board["cells"][chosen_cat_index]["is_boundary"]:
        raise ValueError("cat_index must be an interior cell.")

    candidates = [cell["index"] for cell in board["cells"] if cell["index"] != chosen_cat_index]
    capped_block_count = max(0, min(int(initial_block_count), len(candidates)))
    required_winning_first_actions = resolve_min_winning_first_actions_spec(min_winning_first_actions, policy["name"])
    required_adjacent_winning_first_actions = max(0, int(min_adjacent_winning_first_actions))

    for attempt_index in range(SAMPLE_ATTEMPTS):
        blocked_indices = sorted(rng.sample(candidates, capped_block_count))
        blocked_set = set(blocked_indices)
        if not legal_cat_moves(board, chosen_cat_index, blocked_set):
            continue
        escape_path = shortest_boundary_path(board, chosen_cat_index, blocked_set)
        if escape_path is None:
            continue
        max_steps = max(0, len(board["cells"]) - len(blocked_indices) - 1)
        solvability = solve_chat_noir_instance(
            board,
            chosen_cat_index,
            blocked_indices,
            policy["name"],
            max_steps=max_steps,
            winning_first_action_limit=required_winning_first_actions,
            adjacent_winning_first_action_limit=required_adjacent_winning_first_actions,
        )
        if not solvability["solvable"]:
            continue
        if int(solvability.get("winning_first_action_count", 0)) < required_winning_first_actions:
            continue
        if int(solvability.get("adjacent_winning_first_action_count", 0)) < required_adjacent_winning_first_actions:
            continue
        return {
            "board": board,
            "board_radius": board_radius,
            "requested_board_radius": requested_board_radius,
            "requested_initial_block_count": requested_initial_block_count,
            "cat_index": chosen_cat_index,
            "initial_block_count": capped_block_count,
            "initial_blocked_indices": blocked_indices,
            "sample_attempts": attempt_index + 1,
            "solvability": solvability,
            "min_winning_first_actions": required_winning_first_actions,
            "min_adjacent_winning_first_actions": required_adjacent_winning_first_actions,
            "cat_policy": policy,
            "initial_escape_path": escape_path,
            "initial_escape_distance": len(escape_path) - 1,
            "difficulty": compute_difficulty(
                board_radius=board_radius,
                initial_block_count=len(blocked_indices),
                cat_policy_level=policy["level"],
            ),
        }

    raise RuntimeError("Unable to sample a valid Chat Noir instance with the requested parameters.")


class EnclosureChatNoirEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board_radius: int = 3,
        initial_block_count: int = 5,
        cat_policy: str = DEFAULT_CAT_POLICY,
        frontend_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        headless: bool = True,
        animate: bool = False,
        illegal_reward: float = -1.0,
        move_reward: float = -0.02,
        trap_reward: float = 1.0,
        escape_penalty: float = -1.0,
        viewport_width: int = 1440,
        viewport_height: int = 1080,
    ) -> None:
        super().__init__()

        self.board_radius = validate_board_radius(board_radius)
        self.initial_block_count = max(0, int(initial_block_count))
        self.cat_policy = normalize_cat_policy(cat_policy)["name"]
        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.animate = bool(animate)
        self.illegal_reward = float(illegal_reward)
        self.move_reward = float(move_reward)
        self.trap_reward = float(trap_reward)
        self.escape_penalty = float(escape_penalty)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        default_frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else default_frontend.resolve()
        self.index_file = self.frontend_dir / "index.html"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {self.index_file}")

        self.current_board = build_hex_board(self.board_radius)
        self.action_map = list(range(len(self.current_board["cells"])))
        self.action_space = spaces.Discrete(len(self.action_map))

        self._episode_steps = 0
        self.current_cat_index = self.current_board["center_index"]
        self.current_initial_blocked_indices: List[int] = []
        self.current_cat_policy = normalize_cat_policy(self.cat_policy)
        self.current_initial_escape_distance = 0
        self.current_difficulty: Dict[str, Any] = {}
        self.current_max_steps: Optional[int] = None
        self.current_min_winning_first_actions = DEFAULT_MIN_WINNING_FIRST_ACTIONS
        self.current_min_adjacent_winning_first_actions = DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS
        self.current_solvability: Dict[str, Any] = {}

        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = ""
        self._entry_url = self.index_file.as_uri()

        self._loop = asyncio.new_event_loop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self._start_server()
        try:
            self._run(self._start_browser())
            initial_obs = self.render()
            self.observation_space = spaces.Box(low=0, high=255, shape=initial_obs.shape, dtype=np.uint8)
        except Exception:
            self.close()
            raise

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _start_server(self) -> None:
        handler = partial(_FrontendHandler, directory=str(self.frontend_dir))
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler)
        except PermissionError:
            self._server = None
            self._server_thread = None
            self._base_url = ""
            self._entry_url = self.index_file.as_uri()
            return

        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        self._base_url = f"http://{bound_host}:{bound_port}"
        self._entry_url = f"{self._base_url}/index.html"

    async def _start_browser(self) -> None:
        self._playwright = await async_playwright().start()
        launch_kwargs: Dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            "timeout": 120000,
        }
        chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if chrome_path.exists():
            try:
                self._browser = await self._playwright.chromium.launch(
                    **{**launch_kwargs, "channel": "chrome"}
                )
            except Exception:
                self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        else:
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1.0,
        )
        self._page = await self._context.new_page()
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                await self._page.goto(self._entry_url, wait_until="load", timeout=120000)
                await self._page.wait_for_function(
                    "() => window.topoBench && typeof window.topoBench.step === 'function'",
                    timeout=120000,
                )
                await self._page.locator("#scene-container").wait_for(
                    state="visible",
                    timeout=120000,
                )
                await self._page.locator("#board-card").wait_for(
                    state="visible",
                    timeout=120000,
                )
                return
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1.0)
                    continue
                raise TimeoutError(
                    "Chat Noir frontend did not become ready after 3 attempts"
                ) from last_error

    async def _evaluate(self, expression: str, arg: Any = None) -> Any:
        if arg is None:
            return await self._page.evaluate(expression)
        return await self._page.evaluate(expression, arg)

    async def _capture_locator_png(self, selector: str, path: Optional[str] = None) -> bytes:
        locator = self._page.locator(selector)
        await locator.wait_for(state="visible", timeout=30000)
        return await locator.screenshot(path=path, type="png")

    def _png_bytes_to_rgb(self, png_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    def _build_reset_config(self, *, seed: Optional[int], options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        options = options or {}
        frontend_config = dict(options.get("frontend_config") or {})

        cat_policy = normalize_cat_policy(options.get("cat_policy", frontend_config.get("catPolicy", self.cat_policy)))
        board_radius_spec = options.get("board_radius", frontend_config.get("boardRadius", self.board_radius))
        initial_block_count_spec = options.get(
            "initial_block_count",
            frontend_config.get("initialBlockedCount", self.initial_block_count),
        )
        min_winning_first_actions = resolve_min_winning_first_actions_spec(
            options.get(
                "min_winning_first_actions",
                frontend_config.get("minWinningFirstActions", DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC),
            ),
            cat_policy["name"],
        )
        min_adjacent_winning_first_actions = max(
            0,
            int(
                options.get(
                    "min_adjacent_winning_first_actions",
                    frontend_config.get(
                        "minAdjacentWinningFirstActions",
                        DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
                    ),
                )
            ),
        )
        cat_index_value = options.get("cat_index", frontend_config.get("catIndex"))
        sampled: Optional[Dict[str, Any]] = None

        if "initialBlockedIndices" in frontend_config or "initial_blocked_indices" in options:
            if is_auto_setup_value(board_radius_spec):
                raise ValueError("Explicit initialBlockedIndices require a concrete boardRadius.")
            board_radius = validate_board_radius(board_radius_spec)
            board = build_hex_board(board_radius)
            cat_index = board["center_index"] if cat_index_value is None else int(cat_index_value)
            if cat_index < 0 or cat_index >= len(board["cells"]):
                raise ValueError(f"cat_index out of range for radius {board_radius}: {cat_index}")
            if board["cells"][cat_index]["is_boundary"]:
                raise ValueError("cat_index must be an interior cell.")
            blocked_indices = normalize_blocked_indices(
                board,
                options.get("initial_blocked_indices", frontend_config.get("initialBlockedIndices", [])),
                cat_index=cat_index,
            )
        else:
            sampled = sample_chat_noir_instance(
                board_radius_spec,
                initial_block_count=initial_block_count_spec,
                cat_index=None if cat_index_value is None else int(cat_index_value),
                cat_policy=cat_policy["name"],
                seed=seed,
                min_winning_first_actions=min_winning_first_actions,
                min_adjacent_winning_first_actions=min_adjacent_winning_first_actions,
            )
            board = sampled["board"]
            board_radius = int(sampled["board_radius"])
            cat_index = int(sampled["cat_index"])
            blocked_indices = sampled["initial_blocked_indices"]

        blocked_set = set(blocked_indices)
        escape_path = shortest_boundary_path(board, cat_index, blocked_set)
        if escape_path is None:
            raise ValueError("Initial Chat Noir state must leave the cat a path to the boundary.")

        max_steps = max(0, len(board["cells"]) - len(blocked_indices) - 1)
        solvability = (
            dict(sampled.get("solvability", {}))
            if sampled is not None
            else solve_chat_noir_instance(
                board,
                cat_index,
                blocked_indices,
                cat_policy["name"],
                max_steps=max_steps,
                winning_first_action_limit=min_winning_first_actions,
                adjacent_winning_first_action_limit=min_adjacent_winning_first_actions,
            )
        )
        self.board_radius = board_radius
        self.current_board = board
        self.current_cat_index = cat_index
        self.current_initial_blocked_indices = list(blocked_indices)
        self.current_cat_policy = cat_policy
        self.current_initial_escape_distance = len(escape_path) - 1
        self.current_min_winning_first_actions = min_winning_first_actions
        self.current_min_adjacent_winning_first_actions = min_adjacent_winning_first_actions
        self.current_solvability = solvability
        self.current_difficulty = compute_difficulty(
            board_radius=board_radius,
            initial_block_count=len(blocked_indices),
            cat_policy_level=cat_policy["level"],
        )
        explicit_max_steps = options.get("max_steps")
        self.current_max_steps = None if explicit_max_steps is None else max(1, int(explicit_max_steps))
        self.action_map = list(range(len(board["cells"])))
        self.action_space = spaces.Discrete(len(self.action_map))

        return {
            "boardRadius": board_radius,
            "catIndex": cat_index,
            "initialBlockedCount": len(blocked_indices),
            "initialBlockedIndices": blocked_indices,
            "catPolicy": cat_policy["name"],
            "minWinningFirstActions": min_winning_first_actions,
            "minAdjacentWinningFirstActions": min_adjacent_winning_first_actions,
            "rngSeed": options.get(
                "rng_seed",
                frontend_config.get("rngSeed", seed if seed is not None else random.randrange(1, 2**32)),
            ),
            "animate": options.get("animate", frontend_config.get("animate", self.animate)),
            "illegalReward": options.get("illegal_reward", frontend_config.get("illegalReward", self.illegal_reward)),
            "moveReward": options.get("move_reward", frontend_config.get("moveReward", self.move_reward)),
            "trapReward": options.get("trap_reward", frontend_config.get("trapReward", self.trap_reward)),
            "escapePenalty": options.get(
                "escape_penalty",
                frontend_config.get("escapePenalty", self.escape_penalty),
            ),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover
            super().reset()

        self._episode_steps = 0
        frontend_cfg = self._build_reset_config(seed=seed, options=options)
        frontend_result = self._run(self._evaluate("(cfg) => window.topoBench.reset(cfg)", frontend_cfg))
        obs = self.render()

        info = dict(frontend_result.get("info", {}) or {})
        info["frontend_result"] = frontend_result
        info["symbolic_observation"] = frontend_result.get("observation")
        info["state"] = info.get("state", self.get_state())
        info["success"] = bool(frontend_result.get("success", False))
        info["initial_escape_distance"] = self.current_initial_escape_distance
        info["difficulty"] = dict(self.current_difficulty)
        info["cat_policy"] = dict(self.current_cat_policy)
        info["solvability"] = dict(self.current_solvability)
        info["min_winning_first_actions"] = self.current_min_winning_first_actions
        info["min_adjacent_winning_first_actions"] = self.current_min_adjacent_winning_first_actions
        info["maxSteps"] = self.current_max_steps
        return obs, info

    def step(self, action: int):
        self._episode_steps += 1
        action_int = int(action)
        frontend_result = self.evaluate_frontend_step(action_int)
        obs = self.render()

        reward = float(frontend_result.get("reward", 0.0))
        terminated = bool(frontend_result.get("done", False))
        truncated = False
        if self.current_max_steps is not None and self._episode_steps >= self.current_max_steps and not terminated:
            truncated = True

        info = dict(frontend_result.get("info", {}) or {})
        info["episode_steps"] = self._episode_steps
        info["symbolic_observation"] = frontend_result.get("observation")
        info["success"] = bool(frontend_result.get("success", False))
        info["state"] = info.get("state", self.get_state())
        info["initial_escape_distance"] = self.current_initial_escape_distance
        info["difficulty"] = dict(self.current_difficulty)
        info["cat_policy"] = dict(self.current_cat_policy)
        info["solvability"] = dict(self.current_solvability)
        info["min_winning_first_actions"] = self.current_min_winning_first_actions
        info["min_adjacent_winning_first_actions"] = self.current_min_adjacent_winning_first_actions
        info["maxSteps"] = self.current_max_steps
        if truncated:
            info["truncated_reason"] = "step_budget"
        return obs, reward, terminated, truncated, info

    def evaluate_frontend_step(self, action: int) -> Dict[str, Any]:
        return self._run(self._evaluate("(a) => window.topoBench.step(a)", int(action)))

    def get_frontend_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getState()"))

    def get_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getDebugState()"))

    def decode_action(self, action: int) -> int:
        action_int = int(action)
        if action_int < 0 or action_int >= len(self.action_map):
            raise ValueError(f"Invalid Chat Noir action index: {action_int}")
        return action_int

    def screenshot(self, *, scene_only: bool = True, path: Optional[str] = None) -> bytes:
        selector = "#scene-container" if scene_only else "body"
        return self._run(self._capture_locator_png(selector, path=path))

    def screenshot_board_card(self, *, path: Optional[str] = None) -> bytes:
        return self._run(self._capture_locator_png("#board-card", path=path))

    def screenshot_base64(self, *, scene_only: bool = True) -> str:
        return base64.b64encode(self.screenshot(scene_only=scene_only)).decode("ascii")

    def render(self):
        return self._png_bytes_to_rgb(self.screenshot(scene_only=True))

    async def _close_browser(self) -> None:
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._run(self._close_browser())
            except Exception:
                pass

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
