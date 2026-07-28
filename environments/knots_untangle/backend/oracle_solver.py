from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


SolverState = Tuple[Tuple[int, int], ...]
ExplicitMove = Dict[str, int]
HOLE_SPACING = 2.2
ROPE_RADIUS = 0.22
VISUAL_OVERLAP_THRESHOLD = ROPE_RADIUS * 2.0


@dataclass
class OracleSearchResult:
    min_steps: Optional[int]
    plan: Optional[List[ExplicitMove]]
    expanded_nodes: int
    timed_out: bool
    exceeded_limit: bool


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _hole_rc(entry: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(entry, dict):
        return None
    row = _coerce_int(entry.get("row"))
    col = _coerce_int(entry.get("col"))
    if row is None or col is None:
        return None
    return row, col


def _segments_intersect(
    a1: Tuple[int, int],
    a2: Tuple[int, int],
    b1: Tuple[int, int],
    b2: Tuple[int, int],
) -> bool:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-9:
        return False
    dx, dy = x3 - x1, y3 - y1
    t = (dx * d2y - dy * d2x) / cross
    u = (dx * d1y - dy * d1x) / cross
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def _segments_close(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
    threshold: float,
) -> bool:
    ux, uy = a2[0] - a1[0], a2[1] - a1[1]
    vx, vy = b2[0] - b1[0], b2[1] - b1[1]
    wx, wy = a1[0] - b1[0], a1[1] - b1[1]
    aa = ux * ux + uy * uy
    bb = ux * vx + uy * vy
    cc = vx * vx + vy * vy
    dd = ux * wx + uy * wy
    ee = vx * wx + vy * wy
    determinant = aa * cc - bb * bb
    if determinant < 1e-8:
        sc = 0.0
        tc = dd / bb if abs(bb) > cc and abs(bb) > 1e-8 else (ee / cc if abs(cc) > 1e-8 else 0.0)
    else:
        sc = (bb * ee - cc * dd) / determinant
        tc = (aa * ee - bb * dd) / determinant
    sc = max(0.0, min(1.0, sc))
    tc = max(0.0, min(1.0, tc))
    dx = a1[0] + ux * sc - (b1[0] + vx * tc)
    dy = a1[1] + uy * sc - (b1[1] + vy * tc)
    return dx * dx + dy * dy < threshold * threshold


def hole_id_from_rc(row: int, col: int, grid_size: int) -> int:
    return row * grid_size + col


def hole_rc_from_id(hole_id: int, grid_size: int) -> Tuple[int, int]:
    return hole_id // grid_size, hole_id % grid_size


def hole_xy_from_id(hole_id: int, grid_size: int) -> Tuple[float, float]:
    row, col = hole_rc_from_id(hole_id, grid_size)
    offset = ((grid_size - 1) * HOLE_SPACING) / 2.0
    x = row * HOLE_SPACING - offset
    y = col * HOLE_SPACING - offset
    return x, y


def normalize_state(state: Iterable[Tuple[int, int]]) -> SolverState:
    return tuple(tuple(sorted((int(a), int(b)))) for a, b in state)


def symbolic_to_state(symbolic: Optional[Dict[str, Any]], grid_size: int) -> Optional[SolverState]:
    if not isinstance(symbolic, dict):
        return None
    ropes = symbolic.get("ropes")
    if not isinstance(ropes, list):
        return None

    indexed: List[Tuple[int, Tuple[int, int]]] = []
    for fallback_index, rope in enumerate(ropes):
        if not isinstance(rope, dict):
            return None
        rope_id = _coerce_int(rope.get("id"))
        if rope_id is None:
            rope_id = fallback_index
        start_rc = _hole_rc(rope.get("startHole"))
        end_rc = _hole_rc(rope.get("endHole"))
        if start_rc is None or end_rc is None:
            return None
        indexed.append(
            (
                rope_id,
                (
                    hole_id_from_rc(start_rc[0], start_rc[1], grid_size),
                    hole_id_from_rc(end_rc[0], end_rc[1], grid_size),
                ),
            )
        )

    indexed.sort(key=lambda item: item[0])
    return normalize_state(pair for _, pair in indexed)


def state_logical_crossings(state: SolverState, grid_size: int) -> int:
    total = 0
    for i in range(len(state)):
        a1 = hole_rc_from_id(state[i][0], grid_size)
        a2 = hole_rc_from_id(state[i][1], grid_size)
        for j in range(i + 1, len(state)):
            b1 = hole_rc_from_id(state[j][0], grid_size)
            b2 = hole_rc_from_id(state[j][1], grid_size)
            if _segments_intersect(a1, a2, b1, b2):
                total += 1
    return total


def state_crossings(state: SolverState, grid_size: int) -> int:
    total = 0
    for i in range(len(state)):
        a1 = hole_xy_from_id(state[i][0], grid_size)
        a2 = hole_xy_from_id(state[i][1], grid_size)
        for j in range(i + 1, len(state)):
            b1 = hole_xy_from_id(state[j][0], grid_size)
            b2 = hole_xy_from_id(state[j][1], grid_size)
            if _segments_close(a1, a2, b1, b2, VISUAL_OVERLAP_THRESHOLD):
                total += 1
    return total


def enumerate_legal_moves(state: SolverState, grid_size: int) -> List[ExplicitMove]:
    occupied = {hole_id for rope in state for hole_id in rope}
    empty_holes = [hole_id for hole_id in range(grid_size * grid_size) if hole_id not in occupied]
    moves: List[ExplicitMove] = []
    for endpoint_a, endpoint_b in state:
        for source_hole in (endpoint_a, endpoint_b):
            src_row, src_col = hole_rc_from_id(source_hole, grid_size)
            for target_hole in empty_holes:
                tgt_row, tgt_col = hole_rc_from_id(target_hole, grid_size)
                moves.append(
                    {
                        "src_row": src_row,
                        "src_col": src_col,
                        "tgt_row": tgt_row,
                        "tgt_col": tgt_col,
                    }
                )
    return moves


def apply_move(state: SolverState, move: ExplicitMove, grid_size: int) -> Optional[SolverState]:
    src_row = _coerce_int(move.get("src_row"))
    src_col = _coerce_int(move.get("src_col"))
    tgt_row = _coerce_int(move.get("tgt_row"))
    tgt_col = _coerce_int(move.get("tgt_col"))
    if None in {src_row, src_col, tgt_row, tgt_col}:
        return None

    source_hole = hole_id_from_rc(int(src_row), int(src_col), grid_size)
    target_hole = hole_id_from_rc(int(tgt_row), int(tgt_col), grid_size)
    if source_hole == target_hole:
        return None

    occupied = {hole_id for rope in state for hole_id in rope}
    if target_hole in occupied:
        return None

    next_state = list(state)
    for rope_index, (endpoint_a, endpoint_b) in enumerate(next_state):
        if source_hole not in (endpoint_a, endpoint_b):
            continue
        updated = [endpoint_a, endpoint_b]
        if updated[0] == source_hole:
            updated[0] = target_hole
        else:
            updated[1] = target_hole
        next_state[rope_index] = tuple(sorted((updated[0], updated[1])))
        return tuple(next_state)
    return None


def choose_greedy_move(state: SolverState, grid_size: int) -> Optional[ExplicitMove]:
    legal_moves = enumerate_legal_moves(state, grid_size)
    if not legal_moves:
        return None

    best_move: Optional[ExplicitMove] = None
    best_score: Optional[Tuple[int, int, int, int]] = None
    for move in legal_moves:
        next_state = apply_move(state, move, grid_size)
        if next_state is None:
            continue
        score = (
            state_crossings(next_state, grid_size),
            move["src_row"],
            move["src_col"],
            move["tgt_row"] * grid_size + move["tgt_col"],
        )
        if best_score is None or score < best_score:
            best_score = score
            best_move = move
    return best_move


def bfs_shortest_plan(
    state: SolverState,
    *,
    grid_size: int,
    max_depth: Optional[int] = None,
    max_expansions: int = 200000,
    timeout_seconds: float = 10.0,
) -> OracleSearchResult:
    if state_crossings(state, grid_size) == 0:
        return OracleSearchResult(
            min_steps=0,
            plan=[],
            expanded_nodes=0,
            timed_out=False,
            exceeded_limit=False,
        )

    start_time = time.monotonic()
    queue: Deque[SolverState] = deque([state])
    depth_by_state: Dict[SolverState, int] = {state: 0}
    parent: Dict[SolverState, Tuple[Optional[SolverState], Optional[ExplicitMove]]] = {
        state: (None, None)
    }
    expanded_nodes = 0

    while queue:
        if time.monotonic() - start_time > timeout_seconds:
            return OracleSearchResult(
                min_steps=None,
                plan=None,
                expanded_nodes=expanded_nodes,
                timed_out=True,
                exceeded_limit=False,
            )
        if expanded_nodes >= max_expansions:
            return OracleSearchResult(
                min_steps=None,
                plan=None,
                expanded_nodes=expanded_nodes,
                timed_out=False,
                exceeded_limit=True,
            )

        current = queue.popleft()
        current_depth = depth_by_state[current]
        if max_depth is not None and current_depth >= max_depth:
            continue

        expanded_nodes += 1
        for move in enumerate_legal_moves(current, grid_size):
            next_state = apply_move(current, move, grid_size)
            if next_state is None or next_state in depth_by_state:
                continue
            depth_by_state[next_state] = current_depth + 1
            parent[next_state] = (current, move)
            if state_crossings(next_state, grid_size) == 0:
                plan: List[ExplicitMove] = []
                cursor: Optional[SolverState] = next_state
                while cursor is not None:
                    prev_state, prev_move = parent[cursor]
                    if prev_move is not None:
                        plan.append(prev_move)
                    cursor = prev_state
                plan.reverse()
                return OracleSearchResult(
                    min_steps=len(plan),
                    plan=plan,
                    expanded_nodes=expanded_nodes,
                    timed_out=False,
                    exceeded_limit=False,
                )
            queue.append(next_state)

    return OracleSearchResult(
        min_steps=None,
        plan=None,
        expanded_nodes=expanded_nodes,
        timed_out=False,
        exceeded_limit=False,
    )
