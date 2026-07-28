from __future__ import annotations

import time
from collections import deque
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


Direction = str
Vertex = Tuple[int, int]
EdgeKey = str

ACTION_DIRECTIONS: Tuple[Direction, ...] = ("U", "D", "L", "R")
DIR_TO_DELTA: Dict[Direction, Vertex] = {
    "U": (0, 1),
    "D": (0, -1),
    "L": (-1, 0),
    "R": (1, 0),
}


def edge_key(v1: Vertex, v2: Vertex) -> EdgeKey:
    if v1 < v2:
        return f"{v1[0]},{v1[1]}-{v2[0]},{v2[1]}"
    return f"{v2[0]},{v2[1]}-{v1[0]},{v1[1]}"


def parse_edge_key(key: EdgeKey) -> Tuple[Vertex, Vertex]:
    left, right = key.split("-")
    lx, ly = left.split(",")
    rx, ry = right.split(",")
    return (int(lx), int(ly)), (int(rx), int(ry))


def is_adjacent(v1: Vertex, v2: Vertex) -> bool:
    dx = abs(v1[0] - v2[0])
    dy = abs(v1[1] - v2[1])
    return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def extract_edges_from_stroke(stroke: Sequence[Dict[str, Any]]) -> Set[EdgeKey]:
    vertices = [(int(item["x"]), int(item["y"])) for item in stroke]
    return {edge_key(vertices[index], vertices[index + 1]) for index in range(len(vertices) - 1)}


def build_adjacency(edges: Sequence[EdgeKey]) -> Dict[Vertex, Set[Vertex]]:
    adjacency: Dict[Vertex, Set[Vertex]] = {}
    for key in edges:
        left, right = parse_edge_key(key)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def previous_vertex(current: Vertex, edges: Sequence[EdgeKey]) -> Optional[Vertex]:
    adjacency = build_adjacency(edges)
    neighbors = list(adjacency.get(current, set()))
    if not neighbors:
        return None
    if len(neighbors) == 1:
        return neighbors[0]
    return None


def would_create_cycle(existing_edges: Set[EdgeKey], v1: Vertex, v2: Vertex) -> bool:
    adjacency = build_adjacency(existing_edges)
    if v1 not in adjacency or v2 not in adjacency:
        return False

    queue = [v1]
    seen = {v1}
    while queue:
        current = queue.pop(0)
        if current == v2:
            return True
        for neighbor in adjacency.get(current, set()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return False


def calculate_regions(W: int, H: int, edges: Set[EdgeKey]) -> List[List[int]]:
    cell_w = W - 1
    cell_h = H - 1
    regions = [[-1 for _ in range(cell_w)] for _ in range(cell_h)]

    def is_separated(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> bool:
        i1, j1 = cell1
        i2, j2 = cell2
        if i2 == i1 + 1 and j2 == j1:
            return edge_key((i1 + 1, j1), (i1 + 1, j1 + 1)) in edges
        if i2 == i1 and j2 == j1 + 1:
            return edge_key((i1, j1 + 1), (i1 + 1, j1 + 1)) in edges
        return True

    region_id = 0
    for j in range(cell_h):
        for i in range(cell_w):
            if regions[j][i] != -1:
                continue
            queue = [(i, j)]
            regions[j][i] = region_id
            while queue:
                cell = queue.pop(0)
                neighbors = [
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                ]
                for ni, nj in neighbors:
                    if ni < 0 or ni >= cell_w or nj < 0 or nj >= cell_h:
                        continue
                    if regions[nj][ni] != -1:
                        continue
                    if ni == cell[0] + 1:
                        separated = is_separated(cell, (ni, nj))
                    elif ni == cell[0] - 1:
                        separated = is_separated((ni, nj), cell)
                    elif nj == cell[1] + 1:
                        separated = is_separated(cell, (ni, nj))
                    else:
                        separated = is_separated((ni, nj), cell)
                    if not separated:
                        regions[nj][ni] = region_id
                        queue.append((ni, nj))
            region_id += 1
    return regions


def evaluate_constraints(W: int, H: int, cells: Sequence[Sequence[Optional[str]]], edges: Set[EdgeKey]) -> Dict[str, Any]:
    regions = calculate_regions(W, H, edges)
    color_regions: Dict[str, Set[int]] = {}
    region_colors: Dict[int, Set[str]] = {}
    for j in range(H - 1):
        for i in range(W - 1):
            color = cells[j][i]
            if color is None:
                continue
            rid = regions[j][i]
            color_regions.setdefault(str(color), set()).add(rid)
            region_colors.setdefault(rid, set()).add(str(color))

    for color, region_ids in color_regions.items():
        if len(region_ids) > 1:
            return {
                "success": False,
                "reason": f'Color "{color}" is split across {len(region_ids)} regions',
                "regions": regions,
                "split_same_color": True,
            }

    for rid, colors in region_colors.items():
        if len(colors) > 1:
            return {
                "success": False,
                "reason": f'Region {rid} contains multiple colors: {", ".join(sorted(colors))}',
                "regions": regions,
                "split_same_color": False,
            }

    return {
        "success": True,
        "reason": "All constraints satisfied",
        "regions": regions,
        "split_same_color": False,
    }


def available_moves_from_state(state: Dict[str, Any]) -> List[Direction]:
    W = int(state["W"])
    H = int(state["H"])
    current = (int(state["currentPos"]["x"]), int(state["currentPos"]["y"]))
    edges = extract_edges_from_stroke(state["stroke"])
    prev = previous_vertex(current, edges)

    moves: List[Direction] = []
    for direction in ACTION_DIRECTIONS:
        dx, dy = DIR_TO_DELTA[direction]
        nxt = (current[0] + dx, current[1] + dy)
        if nxt[0] < 0 or nxt[0] >= W or nxt[1] < 0 or nxt[1] >= H:
            continue
        if prev is not None and nxt == prev:
            moves.append(direction)
            continue
        key = edge_key(current, nxt)
        if key in edges:
            continue
        if would_create_cycle(edges, current, nxt):
            continue
        moves.append(direction)
    return moves


def apply_move(state: Dict[str, Any], direction: Direction) -> Optional[Dict[str, Any]]:
    if direction not in DIR_TO_DELTA:
        return None

    W = int(state["W"])
    H = int(state["H"])
    cells = [[cell for cell in row] for row in state["cells"]]
    stroke = [{"x": int(item["x"]), "y": int(item["y"])} for item in state["stroke"]]
    current = (int(state["currentPos"]["x"]), int(state["currentPos"]["y"]))
    edges = extract_edges_from_stroke(stroke)
    prev = previous_vertex(current, edges)
    dx, dy = DIR_TO_DELTA[direction]
    nxt = (current[0] + dx, current[1] + dy)

    if nxt[0] < 0 or nxt[0] >= W or nxt[1] < 0 or nxt[1] >= H:
        return None
    if not is_adjacent(current, nxt):
        return None

    key = edge_key(current, nxt)
    if prev is not None and nxt == prev:
        stroke.pop()
        edges.discard(key)
    else:
        if key in edges:
            return None
        if would_create_cycle(edges, current, nxt):
            return None
        edges.add(key)
        stroke.append({"x": nxt[0], "y": nxt[1]})

    done = nxt == (W - 1, H - 1)
    success = False
    failure_reason = None
    if done:
        evaluation = evaluate_constraints(W, H, cells, edges)
        success = bool(evaluation["success"])
        if not success:
            failure_reason = "DONE_BUT_CONSTRAINT_FAIL"

    return {
        "W": W,
        "H": H,
        "cells": cells,
        "stroke": stroke,
        "currentPos": {"x": nxt[0], "y": nxt[1]},
        "done": done,
        "success": success,
        "failureReason": failure_reason,
    }


def _move_priority(current: Vertex, goal: Vertex, direction: Direction) -> Tuple[int, int]:
    dx, dy = DIR_TO_DELTA[direction]
    nxt = (current[0] + dx, current[1] + dy)
    manhattan = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
    direction_rank = {"R": 0, "U": 1, "L": 2, "D": 3}
    return (manhattan, direction_rank.get(direction, 99))


@lru_cache(maxsize=24)
def _enumerated_simple_path_records(W: int, H: int, max_depth: int) -> Tuple[Tuple[str, bytes], ...]:
    edge_ids: Dict[EdgeKey, int] = {}
    for y in range(H):
        for x in range(W):
            src = (x, y)
            for direction in ACTION_DIRECTIONS:
                dx, dy = DIR_TO_DELTA[direction]
                dst = (x + dx, y + dy)
                if dst[0] < 0 or dst[0] >= W or dst[1] < 0 or dst[1] >= H:
                    continue
                key = edge_key(src, dst)
                if key not in edge_ids:
                    edge_ids[key] = len(edge_ids)

    goal = (W - 1, H - 1)
    cell_w = W - 1
    cell_h = H - 1
    cell_count = cell_w * cell_h
    cell_neighbors: List[List[Tuple[int, int]]] = [[] for _ in range(cell_count)]
    for j in range(cell_h):
        for i in range(cell_w):
            src_cell = j * cell_w + i
            if i + 1 < cell_w:
                key = edge_key((i + 1, j), (i + 1, j + 1))
                bit = 1 << edge_ids[key]
                dst_cell = j * cell_w + i + 1
                cell_neighbors[src_cell].append((dst_cell, bit))
                cell_neighbors[dst_cell].append((src_cell, bit))
            if j + 1 < cell_h:
                key = edge_key((i, j + 1), (i + 1, j + 1))
                bit = 1 << edge_ids[key]
                dst_cell = (j + 1) * cell_w + i
                cell_neighbors[src_cell].append((dst_cell, bit))
                cell_neighbors[dst_cell].append((src_cell, bit))

    def vertex_id(vertex: Vertex) -> int:
        return vertex[1] * W + vertex[0]

    def region_bytes_for(edge_mask: int) -> bytes:
        labels = [255 for _ in range(cell_count)]
        region_index = 0
        for start_cell in range(cell_count):
            if labels[start_cell] != 255:
                continue
            labels[start_cell] = region_index
            queue = [start_cell]
            while queue:
                cell_id = queue.pop()
                for next_cell, separator_bit in cell_neighbors[cell_id]:
                    if edge_mask & separator_bit:
                        continue
                    if labels[next_cell] != 255:
                        continue
                    labels[next_cell] = region_index
                    queue.append(next_cell)
            region_index += 1
        return bytes(labels)

    records: List[Tuple[str, bytes]] = []
    path: List[Direction] = []

    def dfs(current: Vertex, vertex_mask: int, edge_mask: int) -> None:
        depth = len(path)
        distance = abs(goal[0] - current[0]) + abs(goal[1] - current[1])
        if distance > max_depth - depth:
            return
        if current == goal:
            records.append(("".join(path), region_bytes_for(edge_mask)))
            return
        if depth >= max_depth:
            return

        choices: List[Tuple[int, Direction, Vertex]] = []
        for direction in ACTION_DIRECTIONS:
            dx, dy = DIR_TO_DELTA[direction]
            nxt = (current[0] + dx, current[1] + dy)
            if nxt[0] < 0 or nxt[0] >= W or nxt[1] < 0 or nxt[1] >= H:
                continue
            if vertex_mask & (1 << vertex_id(nxt)):
                continue
            choices.append((abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1]), direction, nxt))

        for _distance, direction, nxt in sorted(choices):
            key = edge_key(current, nxt)
            path.append(direction)
            dfs(nxt, vertex_mask | (1 << vertex_id(nxt)), edge_mask | (1 << edge_ids[key]))
            path.pop()

    dfs((0, 0), 1, 0)
    records.sort(key=lambda item: (len(item[0]), item[0]))
    return tuple(records)


def solve_state(
    state: Dict[str, Any],
    *,
    max_depth: Optional[int] = None,
    timeout_seconds: float = 5.0,
) -> Optional[List[Direction]]:
    start_time = time.monotonic()
    W = int(state["W"])
    H = int(state["H"])
    goal = (W - 1, H - 1)
    cells = tuple(tuple(cell for cell in row) for row in state["cells"])

    def state_signature(current_state: Dict[str, Any]) -> Tuple[Vertex, FrozenSet[EdgeKey]]:
        current = (int(current_state["currentPos"]["x"]), int(current_state["currentPos"]["y"]))
        edges = frozenset(extract_edges_from_stroke(current_state["stroke"]))
        return current, edges

    seen: Set[Tuple[Vertex, FrozenSet[EdgeKey]]] = set()

    def dfs(current_state: Dict[str, Any], depth: int) -> Optional[List[Direction]]:
        if time.monotonic() - start_time > timeout_seconds:
            return None

        current = (int(current_state["currentPos"]["x"]), int(current_state["currentPos"]["y"]))
        edges = extract_edges_from_stroke(current_state["stroke"])
        evaluation = evaluate_constraints(W, H, cells, edges)
        if evaluation.get("split_same_color"):
            return None

        if current == goal:
            if evaluation["success"]:
                return []
            return None

        if max_depth is not None and depth >= max_depth:
            return None

        signature = state_signature(current_state)
        if signature in seen:
            return None
        seen.add(signature)

        remaining_limit = None if max_depth is None else max_depth - depth
        if remaining_limit is not None:
            min_distance = abs(goal[0] - current[0]) + abs(goal[1] - current[1])
            if min_distance > remaining_limit:
                return None

        moves = available_moves_from_state(current_state)
        moves.sort(key=lambda direction: _move_priority(current, goal, direction))
        for direction in moves:
            next_state = apply_move(current_state, direction)
            if next_state is None:
                continue
            suffix = dfs(next_state, depth + 1)
            if suffix is not None:
                return [direction, *suffix]
        return None

    initial_state = {
        "W": W,
        "H": H,
        "cells": [list(row) for row in cells],
        "stroke": [{"x": int(item["x"]), "y": int(item["y"])} for item in state["stroke"]],
        "currentPos": {
            "x": int(state["currentPos"]["x"]),
            "y": int(state["currentPos"]["y"]),
        },
        "done": bool(state.get("done", False)),
        "success": bool(state.get("success", False)),
        "failureReason": state.get("failureReason"),
    }
    return dfs(initial_state, depth=0)


def shortest_solution(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    max_depth: Optional[int] = None,
) -> Optional[List[Direction]]:
    start_time = time.monotonic()
    W = int(state["W"])
    H = int(state["H"])
    goal = (W - 1, H - 1)
    cells = tuple(tuple(cell for cell in row) for row in state["cells"])

    def vertex_id(vertex: Vertex) -> int:
        return vertex[1] * W + vertex[0]

    def vertex_from_id(value: int) -> Vertex:
        return value % W, value // W

    edge_ids: Dict[EdgeKey, int] = {}
    edge_keys: List[EdgeKey] = []
    moves_by_vertex: List[List[Tuple[Direction, int, int]]] = [[] for _ in range(W * H)]
    for y in range(H):
        for x in range(W):
            src = (x, y)
            src_id = vertex_id(src)
            for direction in ACTION_DIRECTIONS:
                dx, dy = DIR_TO_DELTA[direction]
                dst = (x + dx, y + dy)
                if dst[0] < 0 or dst[0] >= W or dst[1] < 0 or dst[1] >= H:
                    continue
                key = edge_key(src, dst)
                if key not in edge_ids:
                    edge_ids[key] = len(edge_keys)
                    edge_keys.append(key)
                moves_by_vertex[src_id].append((direction, vertex_id(dst), edge_ids[key]))

    stroke_vertices = [(int(item["x"]), int(item["y"])) for item in state["stroke"]]
    current = (int(state["currentPos"]["x"]), int(state["currentPos"]["y"]))
    current_id = vertex_id(current)
    edge_mask = 0
    vertex_mask = 0
    for vertex in stroke_vertices:
        vertex_mask |= 1 << vertex_id(vertex)
    for index in range(len(stroke_vertices) - 1):
        key = edge_key(stroke_vertices[index], stroke_vertices[index + 1])
        edge_mask |= 1 << edge_ids[key]

    cell_w = W - 1
    cell_h = H - 1
    cell_count = cell_w * cell_h
    color_ids: Dict[str, int] = {}
    cell_color_ids: List[int] = [-1 for _ in range(cell_count)]
    for j in range(cell_h):
        for i in range(cell_w):
            color = cells[j][i]
            if color is None:
                continue
            color_key = str(color)
            if color_key not in color_ids:
                color_ids[color_key] = len(color_ids)
            cell_color_ids[j * cell_w + i] = color_ids[color_key]

    cell_neighbors: List[List[Tuple[int, int]]] = [[] for _ in range(cell_count)]
    for j in range(cell_h):
        for i in range(cell_w):
            src_cell = j * cell_w + i
            if i + 1 < cell_w:
                key = edge_key((i + 1, j), (i + 1, j + 1))
                bit = 1 << edge_ids[key]
                dst_cell = j * cell_w + i + 1
                cell_neighbors[src_cell].append((dst_cell, bit))
                cell_neighbors[dst_cell].append((src_cell, bit))
            if j + 1 < cell_h:
                key = edge_key((i, j + 1), (i + 1, j + 1))
                bit = 1 << edge_ids[key]
                dst_cell = (j + 1) * cell_w + i
                cell_neighbors[src_cell].append((dst_cell, bit))
                cell_neighbors[dst_cell].append((src_cell, bit))

    evaluation_cache: Dict[int, Tuple[bool, bool]] = {}

    def evaluation_for(mask: int) -> Tuple[bool, bool]:
        cached = evaluation_cache.get(mask)
        if cached is not None:
            return cached

        visited_cells = 0
        color_region: List[int] = [-1 for _ in range(len(color_ids))]
        split_same_color = False
        mixed_region = False
        region_index = 0

        for start_cell in range(cell_count):
            if visited_cells & (1 << start_cell):
                continue

            queue = [start_cell]
            visited_cells |= 1 << start_cell
            region_color_mask = 0
            while queue:
                cell_id = queue.pop()
                color_id = cell_color_ids[cell_id]
                if color_id >= 0:
                    region_color_mask |= 1 << color_id

                for next_cell, separator_bit in cell_neighbors[cell_id]:
                    if mask & separator_bit:
                        continue
                    if visited_cells & (1 << next_cell):
                        continue
                    visited_cells |= 1 << next_cell
                    queue.append(next_cell)

            if region_color_mask & (region_color_mask - 1):
                mixed_region = True
            color_mask = region_color_mask
            while color_mask:
                color_bit = color_mask & -color_mask
                color_id = color_bit.bit_length() - 1
                previous_region = color_region[color_id]
                if previous_region == -1:
                    color_region[color_id] = region_index
                elif previous_region != region_index:
                    split_same_color = True
                    break
                color_mask ^= color_bit

            if split_same_color:
                break
            region_index += 1

        success = not split_same_color and not mixed_region
        result = (success, split_same_color)
        evaluation_cache[mask] = result
        return result

    if (
        max_depth is not None
        and max_depth <= 22
        and len(stroke_vertices) == 1
        and stroke_vertices[0] == (0, 0)
        and current == (0, 0)
        and edge_mask == 0
    ):
        colored_cells = [
            (cell_id, color_id)
            for cell_id, color_id in enumerate(cell_color_ids)
            if color_id >= 0
        ]

        def path_satisfies(regions: bytes) -> bool:
            color_region = [-1 for _ in range(len(color_ids))]
            region_color = [-1 for _ in range(cell_count)]
            for cell_id, color_id in colored_cells:
                region_id = regions[cell_id]
                previous_region = color_region[color_id]
                if previous_region == -1:
                    color_region[color_id] = region_id
                elif previous_region != region_id:
                    return False

                previous_color = region_color[region_id]
                if previous_color == -1:
                    region_color[region_id] = color_id
                elif previous_color != color_id:
                    return False
            return True

        records = _enumerated_simple_path_records(W, H, int(max_depth))
        record_scan_start = time.monotonic()
        for path_text, regions in records:
            if time.monotonic() - record_scan_start > timeout_seconds:
                return None
            if path_satisfies(regions):
                return list(path_text)
        return None

    initial_signature = (current_id, edge_mask, vertex_mask)
    frontier = deque([(initial_signature, 0)])
    seen: Set[Tuple[int, int, int]] = {initial_signature}
    parent: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], Direction]] = {}

    while frontier:
        if time.monotonic() - start_time > timeout_seconds:
            return None

        current_signature, depth = frontier.popleft()
        current_id, edge_mask, vertex_mask = current_signature
        current = vertex_from_id(current_id)
        success, split_same_color = evaluation_for(edge_mask)
        if current == goal:
            if success:
                path: List[Direction] = []
                cursor = current_signature
                while cursor in parent:
                    cursor, direction = parent[cursor]
                    path.append(direction)
                path.reverse()
                return path

        if split_same_color:
            continue
        if max_depth is not None and depth >= max_depth:
            continue
        if max_depth is not None:
            remaining = max_depth - depth
            if abs(goal[0] - current[0]) + abs(goal[1] - current[1]) > remaining:
                continue

        for direction, next_id, edge_id in moves_by_vertex[current_id]:
            if edge_mask & (1 << edge_id):
                continue
            if vertex_mask & (1 << next_id):
                continue
            next_signature = (
                next_id,
                edge_mask | (1 << edge_id),
                vertex_mask | (1 << next_id),
            )
            if next_signature in seen:
                continue
            seen.add(next_signature)
            parent[next_signature] = (current_signature, direction)
            frontier.append((next_signature, depth + 1))
    return None


def shortest_solution_length(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    max_depth: Optional[int] = None,
) -> Optional[int]:
    solution = shortest_solution(state, timeout_seconds=timeout_seconds, max_depth=max_depth)
    if solution is None:
        return None
    return len(solution)
