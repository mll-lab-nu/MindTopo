from __future__ import annotations

import heapq
import math
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont

from house_pipeline import (
    get_oblique_camera,
    get_topdown_camera,
    openable_doors_from_house,
    prepare_controller_for_export,
    render_camera_frame,
    sample_house_with_controller,
    set_door_states,
)
from maze_questions import MIN_POINT_COUNT, supports_connected_subset_choice

TASK_NAME = "continuity_3d_maze"
VIEW_ORDER: Sequence[str] = ("topdown", "front45", "right45", "rear45", "left45")
VIEW_LABELS: Dict[str, str] = {
    "topdown": "Top-Down View",
    "front45": "Front Oblique View",
    "right45": "Right Oblique View",
    "rear45": "Rear Oblique View",
    "left45": "Left Oblique View",
}

MAZE_ALLOWED_ERROR = 0.05
MAZE_VACANCY_MAX_RADIUS = 4
DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080
DEFAULT_RENDER_QUALITY = "Ultra"
DEFAULT_GENERATION_WIDTH = 1024
DEFAULT_GENERATION_HEIGHT = 1024
DEFAULT_GENERATION_QUALITY = "Low"

POINT_PALETTE: Sequence[Tuple[int, int, int, int]] = (
    (213, 74, 74, 245),
    (57, 159, 214, 245),
    (230, 164, 54, 245),
    (97, 176, 89, 245),
    (168, 98, 212, 245),
    (224, 112, 170, 245),
)
DOOR_COLOR_PALETTE: Sequence[Dict[str, Any]] = (
    {"color_name": "red", "rgba": (238, 48, 59, 170)},
    {"color_name": "blue", "rgba": (42, 121, 255, 170)},
    {"color_name": "green", "rgba": (40, 190, 96, 170)},
    {"color_name": "yellow", "rgba": (255, 218, 46, 175)},
    {"color_name": "purple", "rgba": (168, 85, 247, 170)},
    {"color_name": "cyan", "rgba": (31, 213, 225, 170)},
    {"color_name": "orange", "rgba": (255, 128, 32, 170)},
    {"color_name": "pink", "rgba": (255, 86, 172, 170)},
)
DOOR_OPEN_MIN_ANSWER_DOORS = 1
DOOR_OPEN_MAX_ANSWER_DOORS = 4
DOOR_OCCLUSION_SAMPLE_STRIDE = 4
TOPDOWN_DOOR_HIGHLIGHT_WIDTH_RATIO = 90
FONT_CANDIDATES: Sequence[str] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "Arial Bold.ttf",
    "Arial.ttf",
)


def polygon_area_xz(points: Sequence[Dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point["x"] * next_point["z"] - next_point["x"] * point["z"]
    return abs(area) * 0.5


def point_in_polygon_xz(point: Dict[str, float], polygon: Sequence[Dict[str, float]]) -> bool:
    x = point["x"]
    z = point["z"]
    inside = False

    if len(polygon) < 3:
        return False

    for index, vertex in enumerate(polygon):
        next_vertex = polygon[(index + 1) % len(polygon)]
        x0, z0 = vertex["x"], vertex["z"]
        x1, z1 = next_vertex["x"], next_vertex["z"]

        if ((z0 > z) != (z1 > z)) and (
            x < (x1 - x0) * (z - z0) / ((z1 - z0) or 1e-9) + x0
        ):
            inside = not inside

    return inside


def room_id_for_position(position: Dict[str, float], rooms: Sequence[Dict[str, Any]]) -> str | None:
    for room in rooms:
        if point_in_polygon_xz(position, room.get("floorPolygon", [])):
            return room["id"]
    return None


def distance_xz(point_a: Dict[str, float], point_b: Dict[str, float]) -> float:
    return math.hypot(point_a["x"] - point_b["x"], point_a["z"] - point_b["z"])


def polyline_length_xz(points: Sequence[Dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance_xz(point_a, point_b) for point_a, point_b in zip(points, points[1:]))


def distance_point_to_segment_xz(
    point: Dict[str, float],
    segment_start: Dict[str, float],
    segment_end: Dict[str, float],
) -> float:
    px = point["x"]
    pz = point["z"]
    x1 = segment_start["x"]
    z1 = segment_start["z"]
    x2 = segment_end["x"]
    z2 = segment_end["z"]

    dx = x2 - x1
    dz = z2 - z1
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-9:
        return math.hypot(px - x1, pz - z1)

    t = ((px - x1) * dx + (pz - z1) * dz) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_z = z1 + t * dz
    return math.hypot(px - closest_x, pz - closest_z)


def polygon_edge_clearance_xz(point: Dict[str, float], polygon: Sequence[Dict[str, float]]) -> float:
    if len(polygon) < 2:
        return 0.0

    return min(
        distance_point_to_segment_xz(point, vertex, polygon[(index + 1) % len(polygon)])
        for index, vertex in enumerate(polygon)
    )


def estimate_nav_step(positions: Sequence[Dict[str, float]]) -> float:
    unique_x = sorted({round(point["x"], 4) for point in positions})
    unique_z = sorted({round(point["z"], 4) for point in positions})
    diffs: List[float] = []

    for values in (unique_x, unique_z):
        for current, nxt in zip(values, values[1:]):
            delta = round(nxt - current, 4)
            if delta > 1e-4:
                diffs.append(delta)

    return min(diffs) if diffs else 0.25


def maze_point_name(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return f"P{index + 1}"


def initial_door_states(house_data: Dict[str, Any]) -> Dict[str, bool]:
    return {
        door["id"]: bool(door.get("openness", 0) >= 0.5)
        for door in openable_doors_from_house(house_data)
    }


def choose_target_door_states(
    house_data: Dict[str, Any],
    mode: str,
    rng: random.Random,
) -> Dict[str, bool]:
    states = initial_door_states(house_data)
    if mode == "generated":
        return dict(states)
    if mode == "open":
        return {door_id: True for door_id in states}
    if mode == "closed":
        return {door_id: False for door_id in states}
    if mode == "random":
        return {door_id: bool(rng.random() < 0.5) for door_id in states}
    raise ValueError(f"Unknown door state mode: {mode}")


def apply_door_states_to_house_data(house_data: Dict[str, Any], door_states: Dict[str, bool]) -> None:
    for door in house_data.get("doors", []):
        door_id = door.get("id")
        if door.get("openable") and door_id in door_states:
            door["openness"] = 1.0 if door_states[door_id] else 0.0


def assert_door_state_results(
    requested_states: Dict[str, bool],
    results: Dict[str, Dict[str, Any]],
) -> None:
    missing = sorted(set(requested_states) - set(results))
    failures = [
        f"{door_id}: {result.get('error') or 'unknown error'}"
        for door_id, result in sorted(results.items())
        if not result.get("success", False)
    ]
    if missing or failures:
        details = []
        if missing:
            details.append(f"missing results for {missing}")
        if failures:
            details.append("failed actions: " + "; ".join(failures))
        raise RuntimeError("Unable to apply requested door states: " + " | ".join(details))


def teleport_agent_to_point(controller, house_data: Dict[str, Any], position: Dict[str, float]) -> None:
    agent_pose = house_data["metadata"]["agent"]
    event = controller.step(
        action="TeleportFull",
        position=position,
        rotation=agent_pose["rotation"],
        horizon=agent_pose["horizon"],
        standing=agent_pose["standing"],
        renderImage=False,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(event.metadata.get("errorMessage") or f"Unable to teleport to point {position}.")


def restore_default_agent_pose(controller, house_data: Dict[str, Any]) -> None:
    controller.step(action="TeleportFull", **house_data["metadata"]["agent"], renderImage=False)


def get_reachable_positions(controller) -> List[Dict[str, float]]:
    event = controller.step(action="GetReachablePositions", renderImage=False)
    if not event.metadata.get("lastActionSuccess", True):
        raise RuntimeError(event.metadata.get("errorMessage") or "Unable to retrieve reachable positions.")
    return event.metadata.get("actionReturn") or []


def build_reachable_path(
    reachable_positions: Sequence[Dict[str, float]],
    start_point: Dict[str, float],
    goal_point: Dict[str, float],
) -> List[Dict[str, float]]:
    if not reachable_positions:
        return []

    step = estimate_nav_step(reachable_positions)
    search_radius = max(MAZE_ALLOWED_ERROR * 2.0, step * 0.6)

    def key_for(point: Dict[str, float]) -> Tuple[int, int]:
        return (int(round(point["x"] / step)), int(round(point["z"] / step)))

    nodes: Dict[Tuple[int, int], Dict[str, float]] = {}
    for point in reachable_positions:
        nodes[key_for(point)] = {"x": float(point["x"]), "y": float(point["y"]), "z": float(point["z"])}

    def nearest_key(target: Dict[str, float]) -> Tuple[int, int] | None:
        target_key = key_for(target)
        if target_key in nodes:
            return target_key

        best_key = None
        best_distance = float("inf")
        for candidate_key, candidate in nodes.items():
            distance = distance_xz(candidate, target)
            if distance < best_distance:
                best_distance = distance
                best_key = candidate_key
        if best_key is not None and best_distance <= search_radius:
            return best_key
        return None

    start_key = nearest_key(start_point)
    goal_key = nearest_key(goal_point)
    if start_key is None or goal_key is None:
        return []

    frontier = [(0.0, 0.0, start_key)]
    came_from: Dict[Tuple[int, int], Tuple[int, int] | None] = {start_key: None}
    costs = {start_key: 0.0}
    neighbor_offsets = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz]

    while frontier:
        _, cost_so_far, current = heapq.heappop(frontier)
        if current == goal_key:
            break
        if cost_so_far > costs.get(current, float("inf")):
            continue

        current_point = nodes[current]
        for dx, dz in neighbor_offsets:
            neighbor_key = (current[0] + dx, current[1] + dz)
            neighbor_point = nodes.get(neighbor_key)
            if neighbor_point is None:
                continue

            new_cost = cost_so_far + distance_xz(current_point, neighbor_point)
            if new_cost >= costs.get(neighbor_key, float("inf")):
                continue

            costs[neighbor_key] = new_cost
            came_from[neighbor_key] = current
            heuristic = distance_xz(neighbor_point, nodes[goal_key])
            heapq.heappush(frontier, (new_cost + heuristic, new_cost, neighbor_key))

    if goal_key not in came_from:
        return []

    keys = []
    current = goal_key
    while current is not None:
        keys.append(current)
        current = came_from[current]
    keys.reverse()

    path = [dict(nodes[key]) for key in keys]
    if path:
        path[0] = dict(start_point)
        path[-1] = dict(goal_point)
    return path


def rank_maze_candidates(house_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    step = estimate_nav_step([candidate["position"] for candidate in candidates])
    rooms_by_id = {room["id"]: room for room in house_data.get("rooms", [])}

    def key_for(point: Dict[str, float]) -> Tuple[int, int]:
        return (int(round(point["x"] / step)), int(round(point["z"] / step)))

    node_keys = {key_for(candidate["position"]) for candidate in candidates}
    for candidate in candidates:
        point = candidate["position"]
        node_key = key_for(point)
        immediate_neighbors = 0
        wider_neighbors = 0
        occupancy_ratios: Dict[int, float] = {}
        vacancy_radius = 0

        for radius in range(1, MAZE_VACANCY_MAX_RADIUS + 1):
            reachable_count = 0
            total_count = (2 * radius + 1) ** 2 - 1
            full_square_clear = True

            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dz == 0:
                        continue
                    if (node_key[0] + dx, node_key[1] + dz) not in node_keys:
                        full_square_clear = False
                        continue
                    reachable_count += 1

            occupancy_ratios[radius] = reachable_count / max(total_count, 1)
            if full_square_clear:
                vacancy_radius = radius

        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if dx == 0 and dz == 0:
                    continue
                if (node_key[0] + dx, node_key[1] + dz) not in node_keys:
                    continue
                wider_neighbors += 1
                if abs(dx) <= 1 and abs(dz) <= 1:
                    immediate_neighbors += 1

        cardinal_clearances = []
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            distance = 0
            for step_distance in range(1, MAZE_VACANCY_MAX_RADIUS + 1):
                if (node_key[0] + dx * step_distance, node_key[1] + dz * step_distance) not in node_keys:
                    break
                distance += 1
            cardinal_clearances.append(distance)
        min_cardinal_clearance = min(cardinal_clearances) if cardinal_clearances else 0

        room = rooms_by_id.get(candidate["room_id"])
        room_edge_clearance = polygon_edge_clearance_xz(point, room.get("floorPolygon", []) if room else [])
        nav_clearance = step * (0.45 + min_cardinal_clearance * 0.75)
        if room is None:
            edge_clearance = nav_clearance
        else:
            edge_clearance = min(max(room_edge_clearance, step * 0.35), nav_clearance + step * 0.75)
        openness_score = (
            occupancy_ratios.get(1, 0.0) * 0.15
            + occupancy_ratios.get(2, 0.0) * 0.2
            + occupancy_ratios.get(3, 0.0) * 0.3
            + occupancy_ratios.get(4, 0.0) * 0.35
        )
        transition_bonus = 65.0 if room is None else 0.0
        candidate["nav_step"] = step
        candidate["neighbor_count"] = immediate_neighbors
        candidate["wide_neighbor_count"] = wider_neighbors
        candidate["edge_clearance"] = edge_clearance
        candidate["room_edge_clearance"] = room_edge_clearance
        candidate["nav_clearance"] = nav_clearance
        candidate["vacancy_radius"] = vacancy_radius
        candidate["min_cardinal_clearance"] = min_cardinal_clearance
        candidate["openness_score"] = openness_score
        candidate["transition_bonus"] = transition_bonus
        candidate["score"] = (
            vacancy_radius * 135
            + min_cardinal_clearance * 78
            + openness_score * 120
            + immediate_neighbors * 8
            + wider_neighbors * 1.25
            + min(edge_clearance / max(step, 1e-6), 4.0) * 18
            + transition_bonus
        )

    candidates.sort(
        key=lambda candidate: (
            candidate["score"],
            candidate["vacancy_radius"],
            candidate["min_cardinal_clearance"],
            candidate["openness_score"],
            candidate["edge_clearance"],
            candidate["neighbor_count"],
            candidate["wide_neighbor_count"],
        ),
        reverse=True,
    )
    return candidates


def collect_maze_candidates(
    controller,
    house_data: Dict[str, Any],
    *,
    current_door_states: Dict[str, bool] | None = None,
) -> List[Dict[str, Any]]:
    original_door_states = dict(current_door_states or initial_door_states(house_data))
    all_open_door_states = {door_id: True for door_id in original_door_states}
    if all_open_door_states:
        set_door_states(controller, house_data, all_open_door_states)

    try:
        positions = get_reachable_positions(controller)
        candidates = []
        rooms = house_data.get("rooms", [])
        for position in positions:
            room_id = room_id_for_position(position, rooms)
            candidates.append(
                {
                    "position": {
                        "x": float(position["x"]),
                        "y": float(position["y"]),
                        "z": float(position["z"]),
                    },
                    "room_id": room_id,
                }
            )
    finally:
        if all_open_door_states:
            set_door_states(controller, house_data, original_door_states)

    if len(candidates) < 2:
        raise RuntimeError("Not enough valid navigation points for maze sampling.")
    return rank_maze_candidates(house_data, candidates)


def candidate_sampling_weight(candidate: Dict[str, Any], selected: Sequence[Dict[str, Any]] | None = None) -> float:
    selected = selected or []
    step = max(candidate.get("nav_step", 0.25), 1e-6)
    score = max(float(candidate.get("score", 0.0)), 0.0)
    openness = max(float(candidate.get("openness_score", 0.0)), 0.0)
    edge_clearance = max(float(candidate.get("edge_clearance", 0.0)), 0.0)
    min_cardinal_clearance = max(float(candidate.get("min_cardinal_clearance", 0.0)), 0.0)

    # Keep all reachable positions possible, while gently preferring visually unambiguous floor space.
    weight = (
        1.0
        + math.sqrt(score + 1.0) * 0.08
        + openness * 0.85
        + min(edge_clearance / step, 3.0) * 0.22
        + min(min_cardinal_clearance, 4.0) * 0.12
    )

    if candidate.get("room_id") is None:
        weight += 0.18

    if selected:
        min_distance = min(distance_xz(candidate["position"], existing["position"]) for existing in selected)
        distance_scale = max(step * 10.0, 1e-6)
        weight += min(min_distance / distance_scale, 1.15)
        if candidate.get("room_id") not in {item.get("room_id") for item in selected}:
            weight += 0.22

    return max(weight, 0.05)


def select_random_maze_candidates(
    controller,
    house_data: Dict[str, Any],
    *,
    count: int,
    rng: random.Random,
    current_door_states: Dict[str, bool] | None = None,
    require_teleportable: bool = True,
) -> List[Dict[str, Any]]:
    candidates = collect_maze_candidates(controller, house_data, current_door_states=current_door_states)
    if count > len(candidates):
        raise RuntimeError(f"Only {len(candidates)} valid maze points are available in this house.")

    def weighted_choice(pool: Sequence[Dict[str, Any]], selected_so_far: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [candidate_sampling_weight(candidate, selected_so_far) for candidate in pool]
        return rng.choices(list(pool), weights=weights, k=1)[0]

    selected: List[Dict[str, Any]] = []
    selected_keys: set[Tuple[float, float]] = set()
    rejected_keys: set[Tuple[float, float]] = set()

    try:
        while len(selected) < count:
            remaining = []
            for candidate in candidates:
                candidate_key = (
                    round(candidate["position"]["x"], 4),
                    round(candidate["position"]["z"], 4),
                )
                if candidate_key in selected_keys or candidate_key in rejected_keys:
                    continue
                remaining.append(candidate)
            if not remaining:
                break

            next_candidate = weighted_choice(remaining, selected)
            next_key = (
                round(next_candidate["position"]["x"], 4),
                round(next_candidate["position"]["z"], 4),
            )
            if require_teleportable:
                try:
                    teleport_agent_to_point(controller, house_data, next_candidate["position"])
                except RuntimeError as exc:
                    next_candidate["teleport_error"] = str(exc)
                    rejected_keys.add(next_key)
                    continue

            selected.append(next_candidate)
            selected_keys.add(next_key)
    finally:
        if require_teleportable:
            restore_default_agent_pose(controller, house_data)

    if len(selected) < count:
        raise RuntimeError(
            "Unable to sample enough distinct maze points that are teleportable "
            f"under the current door states. selected={len(selected)} "
            f"required={count} rejected={len(rejected_keys)}"
        )
    return selected


def evaluate_maze_points(controller, house_data: Dict[str, Any], sampled_points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    points = []
    for index, point in enumerate(sampled_points):
        position = dict(point["position"] if "position" in point else point)
        points.append(
            {
                "name": maze_point_name(index),
                "position": position,
                "room_id": room_id_for_position(position, house_data.get("rooms", [])),
            }
        )

    pair_results = []
    try:
        for source_index, source_point in enumerate(points):
            teleport_agent_to_point(controller, house_data, source_point["position"])
            reachable_positions = get_reachable_positions(controller)
            for target_point in points[source_index + 1 :]:
                path = build_reachable_path(reachable_positions, source_point["position"], target_point["position"])
                connected = bool(path)
                pair_results.append(
                    {
                        "a_name": source_point["name"],
                        "b_name": target_point["name"],
                        "pair_name": f"{source_point['name']}_{target_point['name']}",
                        "a_room_id": source_point["room_id"],
                        "b_room_id": target_point["room_id"],
                        "connected": connected,
                        "answer": "yes" if connected else "no",
                        "path_length": polyline_length_xz(path) if connected else None,
                        "path_point_count": len(path),
                        "error": None if connected else "No path exists for the current door states.",
                    }
                )
    finally:
        restore_default_agent_pose(controller, house_data)

    return {
        "points": points,
        "pairs": pair_results,
        "all_connected": all(pair["connected"] for pair in pair_results),
    }


def door_colors_for_house(house_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    openable_doors = sorted(
        openable_doors_from_house(house_data),
        key=lambda door: str(door.get("id")),
    )
    if len(openable_doors) > len(DOOR_COLOR_PALETTE):
        raise RuntimeError(
            f"door_open supports at most {len(DOOR_COLOR_PALETTE)} colored doors; "
            f"found {len(openable_doors)}."
        )
    return [
        {
            "door_id": str(door["id"]),
            "color_name": str(color_spec["color_name"]),
            "rgba": list(color_spec["rgba"]),
        }
        for door, color_spec in zip(openable_doors, DOOR_COLOR_PALETTE)
    ]


def room_door_edges_for_house(
    house_data: Dict[str, Any],
    current_door_states: Dict[str, bool],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for door in openable_doors_from_house(house_data):
        door_id = str(door.get("id", ""))
        room0 = door.get("room0")
        room1 = door.get("room1")
        if not door_id or room0 is None or room1 is None or room0 == room1:
            continue
        edges.append(
            {
                "door_id": door_id,
                "room0": str(room0),
                "room1": str(room1),
                "is_open": bool(current_door_states.get(door_id, False)),
            }
        )
    return edges


def _room_graph_adjacency(edges: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge["room0"]), []).append(edge)
        adjacency.setdefault(str(edge["room1"]), []).append(edge)
    return adjacency


def _other_edge_room(edge: Dict[str, Any], room_id: str) -> str:
    room0 = str(edge["room0"])
    room1 = str(edge["room1"])
    return room1 if room_id == room0 else room0


def _enumerate_simple_room_paths(
    adjacency: Dict[str, List[Dict[str, Any]]],
    source_room_id: str,
    target_room_id: str,
) -> List[List[Dict[str, Any]]]:
    paths: List[List[Dict[str, Any]]] = []
    stack: List[Tuple[str, List[Dict[str, Any]], set[str]]] = [
        (source_room_id, [], {source_room_id})
    ]
    while stack:
        room_id, path, visited = stack.pop()
        if room_id == target_room_id:
            paths.append(path)
            continue
        for edge in adjacency.get(room_id, []):
            next_room_id = _other_edge_room(edge, room_id)
            if next_room_id in visited:
                continue
            stack.append((next_room_id, [*path, edge], {*visited, next_room_id}))
    return paths


def _room_ids_for_edge_path(source_room_id: str, path: Sequence[Dict[str, Any]]) -> List[str]:
    room_ids = [source_room_id]
    current_room_id = source_room_id
    for edge in path:
        current_room_id = _other_edge_room(edge, current_room_id)
        room_ids.append(current_room_id)
    return room_ids


def build_door_open_graph_plans(
    house_data: Dict[str, Any],
    current_door_states: Dict[str, bool],
    *,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    edges = room_door_edges_for_house(house_data, current_door_states)
    if not any(not edge["is_open"] for edge in edges):
        return []

    adjacency = _room_graph_adjacency(edges)
    room_ids = sorted(adjacency)
    plans: List[Dict[str, Any]] = []
    for source_room_id, target_room_id in combinations(room_ids, 2):
        closed_sets_to_paths: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        for path in _enumerate_simple_room_paths(adjacency, source_room_id, target_room_id):
            closed_door_ids = tuple(
                sorted(str(edge["door_id"]) for edge in path if not edge["is_open"])
            )
            if len(closed_door_ids) > DOOR_OPEN_MAX_ANSWER_DOORS:
                continue
            closed_sets_to_paths.setdefault(closed_door_ids, path)

        # If an all-open path exists, the rooms are already connected under the graph abstraction.
        if not closed_sets_to_paths or () in closed_sets_to_paths:
            continue

        minimum_size = min(len(closed_set) for closed_set in closed_sets_to_paths)
        if minimum_size < DOOR_OPEN_MIN_ANSWER_DOORS or minimum_size > DOOR_OPEN_MAX_ANSWER_DOORS:
            continue

        minimum_sets = [
            closed_set
            for closed_set in closed_sets_to_paths
            if len(closed_set) == minimum_size
        ]
        if len(minimum_sets) != 1:
            continue

        answer_door_ids = list(minimum_sets[0])
        graph_path = closed_sets_to_paths[minimum_sets[0]]
        plans.append(
            {
                "source_room_id": source_room_id,
                "target_room_id": target_room_id,
                "answer_door_ids": answer_door_ids,
                "answer_size": len(answer_door_ids),
                "graph_path_room_ids": _room_ids_for_edge_path(source_room_id, graph_path),
                "graph_path_door_ids": [str(edge["door_id"]) for edge in graph_path],
            }
        )

    rng.shuffle(plans)
    return plans


def _candidate_key(candidate: Dict[str, Any]) -> Tuple[float, float]:
    return (
        round(float(candidate["position"]["x"]), 4),
        round(float(candidate["position"]["z"]), 4),
    )


def _choose_teleportable_candidate(
    controller,
    house_data: Dict[str, Any],
    pool: Sequence[Dict[str, Any]],
    *,
    rng: random.Random,
    selected: Sequence[Dict[str, Any]],
    rejected_keys: set[Tuple[float, float]],
) -> Dict[str, Any] | None:
    while True:
        selected_keys = {_candidate_key(candidate) for candidate in selected}
        remaining = [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
            and _candidate_key(candidate) not in rejected_keys
        ]
        if not remaining:
            return None

        weights = [candidate_sampling_weight(candidate, selected) for candidate in remaining]
        candidate = rng.choices(list(remaining), weights=weights, k=1)[0]
        try:
            teleport_agent_to_point(controller, house_data, candidate["position"])
        except RuntimeError:
            rejected_keys.add(_candidate_key(candidate))
            continue
        return candidate


def select_door_open_maze_candidates(
    controller,
    house_data: Dict[str, Any],
    *,
    count: int,
    graph_plan: Dict[str, Any],
    rng: random.Random,
    current_door_states: Dict[str, bool],
) -> List[Dict[str, Any]]:
    if count < 2:
        raise RuntimeError("door_open requires at least two labeled points.")

    candidates = collect_maze_candidates(
        controller,
        house_data,
        current_door_states=current_door_states,
    )
    source_room_id = str(graph_plan["source_room_id"])
    target_room_id = str(graph_plan["target_room_id"])
    source_pool = [candidate for candidate in candidates if candidate.get("room_id") == source_room_id]
    target_pool = [candidate for candidate in candidates if candidate.get("room_id") == target_room_id]
    if not source_pool or not target_pool:
        raise RuntimeError(
            f"Unable to find navigable candidate points in graph rooms {source_room_id!r} and {target_room_id!r}."
        )

    selected: List[Dict[str, Any]] = []
    rejected_keys: set[Tuple[float, float]] = set()
    try:
        source_candidate = _choose_teleportable_candidate(
            controller,
            house_data,
            source_pool,
            rng=rng,
            selected=selected,
            rejected_keys=rejected_keys,
        )
        if source_candidate is None:
            raise RuntimeError(f"No teleportable source point found in room {source_room_id!r}.")
        selected.append(source_candidate)

        target_candidate = _choose_teleportable_candidate(
            controller,
            house_data,
            target_pool,
            rng=rng,
            selected=selected,
            rejected_keys=rejected_keys,
        )
        if target_candidate is None:
            raise RuntimeError(f"No teleportable target point found in room {target_room_id!r}.")
        selected.append(target_candidate)

        while len(selected) < count:
            next_candidate = _choose_teleportable_candidate(
                controller,
                house_data,
                candidates,
                rng=rng,
                selected=selected,
                rejected_keys=rejected_keys,
            )
            if next_candidate is None:
                break
            selected.append(next_candidate)
    finally:
        restore_default_agent_pose(controller, house_data)

    if len(selected) < count:
        raise RuntimeError(
            "Unable to sample enough distinct maze points for door_open. "
            f"selected={len(selected)} required={count}"
        )
    return selected


def _pair_connected_under_door_states(
    controller,
    house_data: Dict[str, Any],
    *,
    source_point: Dict[str, Any],
    target_point: Dict[str, Any],
    base_door_states: Dict[str, bool],
    opened_door_ids: Sequence[str],
) -> bool:
    trial_states = dict(base_door_states)
    for door_id in opened_door_ids:
        trial_states[str(door_id)] = True

    apply_door_states_to_house_data(house_data, trial_states)
    door_results = set_door_states(controller, house_data, trial_states)
    assert_door_state_results(trial_states, door_results)
    teleport_agent_to_point(controller, house_data, source_point["position"])
    reachable_positions = get_reachable_positions(controller)
    path = build_reachable_path(
        reachable_positions,
        source_point["position"],
        target_point["position"],
    )
    return bool(path)


def _maze_state_pair_connected(maze_state: Dict[str, Any], source_name: str, target_name: str) -> bool | None:
    target_pair = frozenset((source_name, target_name))
    for pair in maze_state.get("pairs", []):
        if frozenset((str(pair.get("a_name")), str(pair.get("b_name")))) == target_pair:
            return bool(pair.get("connected"))
    return None


def build_unique_door_open_question(
    controller,
    house_data: Dict[str, Any],
    maze_state: Dict[str, Any],
    *,
    current_door_states: Dict[str, bool],
    rng: random.Random,
    graph_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    closed_door_ids = sorted(
        str(door_id)
        for door_id, is_open in current_door_states.items()
        if not is_open
    )
    if not closed_door_ids:
        return None

    all_door_colors = door_colors_for_house(house_data)
    door_colors = [
        entry
        for entry in all_door_colors
        if str(entry["door_id"]) in closed_door_ids
    ]
    color_by_door_id = {
        str(entry["door_id"]): str(entry["color_name"]) for entry in door_colors
    }
    point_by_name = {str(point["name"]): point for point in maze_state.get("points", [])}
    if graph_plan is None:
        candidate_pairs = [
            (str(pair["a_name"]), str(pair["b_name"]), None)
            for pair in maze_state.get("pairs", [])
            if not pair.get("connected")
            and str(pair.get("a_name")) in point_by_name
            and str(pair.get("b_name")) in point_by_name
        ]
        rng.shuffle(candidate_pairs)
    else:
        candidate_pairs = [
            (
                str(graph_plan.get("source_name", maze_point_name(0))),
                str(graph_plan.get("target_name", maze_point_name(1))),
                tuple(sorted(str(door_id) for door_id in graph_plan.get("answer_door_ids", []))),
            )
        ]

    try:
        for source_name, target_name, expected_answer_door_ids in candidate_pairs:
            if source_name not in point_by_name or target_name not in point_by_name:
                continue
            if _maze_state_pair_connected(maze_state, source_name, target_name) is not False:
                continue
            source_point = point_by_name[source_name]
            target_point = point_by_name[target_name]

            for answer_size in range(
                DOOR_OPEN_MIN_ANSWER_DOORS,
                min(DOOR_OPEN_MAX_ANSWER_DOORS, len(closed_door_ids)) + 1,
            ):
                successful_combos: List[Tuple[str, ...]] = []
                for combo in combinations(closed_door_ids, answer_size):
                    if _pair_connected_under_door_states(
                        controller,
                        house_data,
                        source_point=source_point,
                        target_point=target_point,
                        base_door_states=current_door_states,
                        opened_door_ids=combo,
                    ):
                        successful_combos.append(tuple(combo))
                        if len(successful_combos) > 1:
                            break

                if len(successful_combos) == 1:
                    answer_door_ids = sorted(str(door_id) for door_id in successful_combos[0])
                    if (
                        expected_answer_door_ids is not None
                        and tuple(answer_door_ids) != expected_answer_door_ids
                    ):
                        return None
                    answer_colors = sorted(
                        color_by_door_id[door_id] for door_id in answer_door_ids
                    )
                    payload: Dict[str, Any] = {}
                    if graph_plan is not None:
                        payload["graph_plan"] = dict(graph_plan)
                    return {
                        "source_name": source_name,
                        "target_name": target_name,
                        "answer_door_ids": answer_door_ids,
                        "answer_colors": answer_colors,
                        "answer_size": answer_size,
                        "door_colors": door_colors,
                        "current_door_states": dict(current_door_states),
                        "unique_minimum_solution": True,
                        **payload,
                    }
                if successful_combos:
                    break
    finally:
        apply_door_states_to_house_data(house_data, current_door_states)
        restore_results = set_door_states(controller, house_data, current_door_states)
        assert_door_state_results(current_door_states, restore_results)
        restore_default_agent_pose(controller, house_data)

    return None


def build_view_cameras(controller, house_data: Dict[str, Any], *, front_yaw: Optional[float] = None) -> Dict[str, Any]:
    if front_yaw is None:
        front_yaw = house_data["metadata"]["agent"]["rotation"]["y"]

    return {
        "topdown": get_topdown_camera(controller),
        "front45": get_oblique_camera(house_data, yaw_deg=front_yaw),
        "right45": get_oblique_camera(house_data, yaw_deg=(front_yaw + 90) % 360),
        "rear45": get_oblique_camera(house_data, yaw_deg=(front_yaw + 180) % 360),
        "left45": get_oblique_camera(house_data, yaw_deg=(front_yaw + 270) % 360),
    }


def render_view_frames(controller, cameras: Dict[str, Any]) -> Dict[str, Any]:
    return {view_name: render_camera_frame(controller, cameras[view_name]) for view_name in VIEW_ORDER}


def _load_font(size: int):
    for font_name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _camera_basis(camera: Dict[str, Any]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    rotation = camera.get("rotation", {})
    yaw = math.radians(float(rotation.get("y", 0.0)))
    pitch = math.radians(float(rotation.get("x", 0.0)))

    forward = (
        math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    up = (
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    )
    return right, up, forward


def project_world_point(
    point: Dict[str, float],
    camera: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[float, float] | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    position = camera["position"]
    right, up, forward = _camera_basis(camera)
    offset = (
        point["x"] - position["x"],
        point["y"] - position["y"],
        point["z"] - position["z"],
    )
    camera_x = sum(component * axis for component, axis in zip(offset, right))
    camera_y = sum(component * axis for component, axis in zip(offset, up))
    camera_z = sum(component * axis for component, axis in zip(offset, forward))

    aspect = width / height
    if camera.get("orthographic"):
        if camera_z <= 0.0:
            return None
        ortho_size = max(float(camera.get("orthographicSize", 1.0)), 1e-6)
        # AI2-THOR uses Unity-style orthographicSize: half of the vertical view span.
        ndc_x = camera_x / (ortho_size * aspect)
        ndc_y = camera_y / ortho_size
    else:
        if camera_z <= 0.05:
            return None
        tan_half_fov = math.tan(math.radians(float(camera.get("fieldOfView", 60.0))) / 2.0)
        ndc_x = camera_x / (camera_z * tan_half_fov * aspect)
        ndc_y = camera_y / (camera_z * tan_half_fov)

    if not all(math.isfinite(value) for value in (ndc_x, ndc_y)):
        return None
    return ((ndc_x + 1.0) * 0.5 * width, (1.0 - ndc_y) * 0.5 * height)


def house_floor_height(house_data: Optional[Dict[str, Any]]) -> float | None:
    if not house_data:
        return None

    floor_y = float("inf")
    for wall in house_data.get("walls", []):
        for point in wall.get("polygon", []):
            y = point.get("y")
            if y is not None:
                floor_y = min(floor_y, float(y))

    if math.isfinite(floor_y):
        return floor_y
    return None


def grounded_render_point(
    point: Dict[str, float],
    house_data: Optional[Dict[str, Any]],
    *,
    lift: float = 0.02,
) -> Dict[str, float]:
    floor_y = house_floor_height(house_data)
    base_y = floor_y if floor_y is not None else float(point.get("y", 0.0))
    return {
        "x": float(point["x"]),
        "y": base_y + lift,
        "z": float(point["z"]),
    }


def _vector_sub(point_a: Dict[str, float], point_b: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(point_a["x"]) - float(point_b["x"]),
        float(point_a["y"]) - float(point_b["y"]),
        float(point_a["z"]) - float(point_b["z"]),
    )


def _vector_cross(
    vector_a: Tuple[float, float, float],
    vector_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    return (
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
    )


def _vector_dot(vector_a: Tuple[float, float, float], vector_b: Tuple[float, float, float]) -> float:
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]


def _vector_add(
    vector_a: Tuple[float, float, float],
    vector_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    return (
        vector_a[0] + vector_b[0],
        vector_a[1] + vector_b[1],
        vector_a[2] + vector_b[2],
    )


def _vector_scale(vector: Tuple[float, float, float], scale: float) -> Tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _point_from_vector(vector: Tuple[float, float, float]) -> Dict[str, float]:
    return {"x": vector[0], "y": vector[1], "z": vector[2]}


def _segment_intersects_triangle(
    start: Dict[str, float],
    end: Dict[str, float],
    tri_a: Dict[str, float],
    tri_b: Dict[str, float],
    tri_c: Dict[str, float],
    *,
    epsilon: float = 1e-6,
) -> bool:
    direction = _vector_sub(end, start)
    edge_ab = _vector_sub(tri_b, tri_a)
    edge_ac = _vector_sub(tri_c, tri_a)
    p_vector = _vector_cross(direction, edge_ac)
    determinant = _vector_dot(edge_ab, p_vector)
    if abs(determinant) <= epsilon:
        return False

    inv_determinant = 1.0 / determinant
    t_vector = _vector_sub(start, tri_a)
    barycentric_u = _vector_dot(t_vector, p_vector) * inv_determinant
    if barycentric_u < -epsilon or barycentric_u > 1.0 + epsilon:
        return False

    q_vector = _vector_cross(t_vector, edge_ab)
    barycentric_v = _vector_dot(direction, q_vector) * inv_determinant
    if barycentric_v < -epsilon or barycentric_u + barycentric_v > 1.0 + epsilon:
        return False

    ray_distance = _vector_dot(edge_ac, q_vector) * inv_determinant
    return epsilon < ray_distance < 1.0 - epsilon


def _screen_point_world_ray(
    screen_point: Tuple[float, float],
    *,
    camera: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, float], Tuple[float, float, float]] | None:
    if camera.get("orthographic"):
        return None

    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    origin = camera.get("position")
    if not isinstance(origin, dict):
        return None

    x, y = screen_point
    aspect = width / height
    ndc_x = (x / width) * 2.0 - 1.0
    ndc_y = 1.0 - (y / height) * 2.0
    tan_half_fov = math.tan(math.radians(float(camera.get("fieldOfView", 60.0))) / 2.0)
    right, up, forward = _camera_basis(camera)
    direction = _vector_add(
        _vector_add(
            forward,
            _vector_scale(right, ndc_x * tan_half_fov * aspect),
        ),
        _vector_scale(up, ndc_y * tan_half_fov),
    )
    return origin, direction


def _screen_point_on_world_quad(
    screen_point: Tuple[float, float],
    *,
    camera: Dict[str, Any],
    image_size: Tuple[int, int],
    world_quad: Sequence[Dict[str, float]],
    epsilon: float = 1e-6,
) -> Dict[str, float] | None:
    if len(world_quad) < 3:
        return None

    ray = _screen_point_world_ray(screen_point, camera=camera, image_size=image_size)
    if ray is None:
        return None

    origin, direction = ray
    edge_a = _vector_sub(world_quad[1], world_quad[0])
    edge_b = _vector_sub(world_quad[2], world_quad[0])
    normal = _vector_cross(edge_a, edge_b)
    denominator = _vector_dot(normal, direction)
    if abs(denominator) <= epsilon:
        return None

    distance = _vector_dot(normal, _vector_sub(world_quad[0], origin)) / denominator
    if distance <= epsilon:
        return None

    origin_vector = (float(origin["x"]), float(origin["y"]), float(origin["z"]))
    return _point_from_vector(_vector_add(origin_vector, _vector_scale(direction, distance)))


def _wall_triangles(wall: Dict[str, Any]) -> List[Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]:
    polygon = wall.get("polygon") or []
    if len(polygon) < 3:
        return []
    if len(polygon) == 3:
        return [(polygon[0], polygon[1], polygon[2])]
    if len(polygon) == 4:
        indices = ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3))
        return [(polygon[a], polygon[b], polygon[c]) for a, b, c in indices]
    return [(polygon[0], polygon[index], polygon[index + 1]) for index in range(1, len(polygon) - 1)]


def point_occluded_by_walls(
    point: Dict[str, float],
    camera: Dict[str, Any],
    house_data: Optional[Dict[str, Any]],
    *,
    ignored_wall_ids: set[str] | None = None,
) -> bool:
    if not house_data or camera.get("orthographic"):
        return False

    camera_position = camera.get("position")
    if not isinstance(camera_position, dict):
        return False

    ignored_wall_ids = ignored_wall_ids or set()
    for wall in house_data.get("walls", []):
        wall_id = str(wall.get("id", ""))
        if wall_id in ignored_wall_ids:
            continue
        for triangle in _wall_triangles(wall):
            if _segment_intersects_triangle(camera_position, point, *triangle):
                return True
    return False


def door_visible_mask(
    *,
    image_size: Tuple[int, int],
    camera: Dict[str, Any],
    house_data: Optional[Dict[str, Any]],
    door: Dict[str, Any],
    door_quad: Sequence[Dict[str, float]],
    projected_quad: Sequence[Tuple[float, float]],
) -> Image.Image | None:
    if not house_data or camera.get("orthographic") or len(projected_quad) != 4:
        return None

    width, height = image_size
    polygon_mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(polygon_mask).polygon(projected_quad, fill=255)
    bbox = polygon_mask.getbbox()
    if bbox is None:
        return polygon_mask

    ignored_wall_ids = {
        str(wall_id)
        for wall_id in (door.get("wall0"), door.get("wall1"))
        if wall_id is not None
    }
    visible_mask = Image.new("L", image_size, 0)
    visible_draw = ImageDraw.Draw(visible_mask)
    stride = DOOR_OCCLUSION_SAMPLE_STRIDE
    left, top, right, bottom = bbox

    for y0 in range(top, bottom, stride):
        y1 = min(y0 + stride, height)
        sample_y = min(y0 + stride * 0.5, height - 0.5)
        for x0 in range(left, right, stride):
            x1 = min(x0 + stride, width)
            sample_x = min(x0 + stride * 0.5, width - 0.5)
            if polygon_mask.getpixel((int(sample_x), int(sample_y))) == 0:
                continue

            world_point = _screen_point_on_world_quad(
                (sample_x, sample_y),
                camera=camera,
                image_size=image_size,
                world_quad=door_quad,
            )
            if world_point is None:
                continue
            if point_occluded_by_walls(
                world_point,
                camera,
                house_data,
                ignored_wall_ids=ignored_wall_ids,
            ):
                continue
            visible_draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)

    return ImageChops.multiply(polygon_mask, visible_mask)


def door_world_quad(house_data: Dict[str, Any], door: Dict[str, Any]) -> List[Dict[str, float]] | None:
    walls = {wall.get("id"): wall for wall in house_data.get("walls", [])}
    wall = walls.get(door.get("wall0")) or walls.get(door.get("wall1"))
    if not wall:
        return None

    wall_polygon = wall.get("polygon") or []
    hole_polygon = door.get("holePolygon") or []
    if len(wall_polygon) < 2 or len(hole_polygon) < 2:
        return None

    wall_start = wall_polygon[0]
    wall_end = wall_polygon[1]
    wall_dx = float(wall_end["x"]) - float(wall_start["x"])
    wall_dz = float(wall_end["z"]) - float(wall_start["z"])
    wall_length = math.hypot(wall_dx, wall_dz)
    if wall_length <= 1e-6:
        return None

    unit_x = wall_dx / wall_length
    unit_z = wall_dz / wall_length
    local_x_values = [float(point.get("x", 0.0)) for point in hole_polygon]
    local_y_values = [float(point.get("y", 0.0)) for point in hole_polygon]
    left_x = min(local_x_values)
    right_x = max(local_x_values)
    bottom_y = min(local_y_values)
    top_y = max(local_y_values)
    wall_base_y = float(wall_start.get("y", 0.0))

    def world_point(local_x: float, local_y: float) -> Dict[str, float]:
        return {
            "x": float(wall_start["x"]) + unit_x * local_x,
            "y": wall_base_y + local_y,
            "z": float(wall_start["z"]) + unit_z * local_x,
        }

    return [
        world_point(left_x, bottom_y),
        world_point(right_x, bottom_y),
        world_point(right_x, top_y),
        world_point(left_x, top_y),
    ]


def draw_door_highlights(
    image: Image.Image,
    *,
    camera: Dict[str, Any],
    house_data: Optional[Dict[str, Any]],
    highlighted_doors: Sequence[Dict[str, Any]] | None,
) -> Image.Image:
    if not house_data or not highlighted_doors:
        return image

    image_size = image.size
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    doors_by_id = {
        str(door.get("id")): door for door in openable_doors_from_house(house_data)
    }
    for entry in highlighted_doors:
        door_id = str(entry.get("door_id", ""))
        door = doors_by_id.get(door_id)
        if door is None:
            continue
        door_quad = door_world_quad(house_data, door)
        if not door_quad:
            continue
        rgba = tuple(entry.get("rgba") or (255, 220, 64, 170))
        outline_rgba = (*rgba[:3], 245) if len(rgba) >= 3 else (255, 220, 64, 245)
        projected = [
            project_world_point(point, camera, image_size) for point in door_quad
        ]
        if any(
            point is None or not all(math.isfinite(value) for value in point)
            for point in projected
        ):
            continue

        projected_quad = [point for point in projected if point is not None]
        visible_mask = door_visible_mask(
            image_size=image_size,
            camera=camera,
            house_data=house_data,
            door=door,
            door_quad=door_quad,
            projected_quad=projected_quad,
        )
        if visible_mask is not None and visible_mask.getbbox() is None:
            continue

        door_layer = Image.new("RGBA", image_size, (0, 0, 0, 0))
        door_draw = ImageDraw.Draw(door_layer, "RGBA")
        if camera.get("orthographic"):
            threshold_width = max(10, min(image_size) // TOPDOWN_DOOR_HIGHLIGHT_WIDTH_RATIO)
            door_draw.line(projected_quad[:2], fill=rgba, width=threshold_width)
            door_draw.line(projected_quad[:2], fill=outline_rgba, width=max(3, threshold_width // 3))
        else:
            door_draw.polygon(projected_quad, fill=rgba)
            door_draw.line([*projected_quad, projected_quad[0]], fill=outline_rgba, width=4)
        if visible_mask is not None:
            door_layer.putalpha(ImageChops.multiply(door_layer.getchannel("A"), visible_mask))
        overlay = Image.alpha_composite(overlay, door_layer)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font) -> Tuple[int, int, int, int]:
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:  # pragma: no cover - old Pillow fallback
        width, height = draw.textsize(text, font=font)
        return (int(xy[0]), int(xy[1]), int(xy[0] + width), int(xy[1] + height))


def _rounded_rectangle(draw: ImageDraw.ImageDraw, box, *, radius: int, fill) -> None:
    try:
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    except AttributeError:  # pragma: no cover - old Pillow fallback
        draw.rectangle(box, fill=fill)


def annotate_maze_image(
    frame: Any,
    *,
    camera: Dict[str, Any],
    maze_state: Dict[str, Any],
    view_label: str,
    house_data: Optional[Dict[str, Any]] = None,
    highlighted_points: Sequence[str] | None = None,
    highlighted_doors: Sequence[Dict[str, Any]] | None = None,
) -> Image.Image:
    image = frame.copy() if isinstance(frame, Image.Image) else Image.fromarray(frame)
    image = image.convert("RGB")
    width, height = image.size
    label_font = _load_font(max(42, min(width, height) // 24))
    point_font = _load_font(max(30, min(width, height) // 30))
    radius = max(11, min(width, height) // 70)

    image = draw_door_highlights(
        image,
        camera=camera,
        house_data=house_data,
        highlighted_doors=highlighted_doors,
    )
    draw = ImageDraw.Draw(image, "RGBA")

    draw.text(
        (24, 20),
        view_label,
        fill=(255, 255, 255, 255),
        font=label_font,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 165),
    )

    highlighted = set(highlighted_points or [])
    for index, point_entry in enumerate(maze_state.get("points", [])):
        label = point_entry["name"]
        point = point_entry["position"]
        base_point = grounded_render_point(point, house_data, lift=0.02)
        base_screen_point = project_world_point(base_point, camera, (width, height))
        if base_screen_point is None:
            continue
        if point_occluded_by_walls(base_point, camera, house_data):
            continue

        x, y = base_screen_point
        margin = radius * 3
        if x < -margin or x > width + margin or y < -margin or y > height + margin:
            continue

        point_color = POINT_PALETTE[index % len(POINT_PALETTE)]
        is_highlighted = not highlighted or label in highlighted
        fill = point_color if is_highlighted else (*point_color[:3], 190)
        outline = (255, 255, 255, 245) if is_highlighted else (245, 245, 245, 210)
        local_radius = radius

        draw.ellipse(
            (x - local_radius - 3, y - local_radius - 3, x + local_radius + 3, y + local_radius + 3),
            fill=(0, 0, 0, 150),
        )
        draw.ellipse(
            (x - local_radius, y - local_radius, x + local_radius, y + local_radius),
            fill=fill,
            outline=outline,
            width=3,
        )

        text_bbox = _text_bbox(draw, (0, 0), label, font=point_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_offset_x = local_radius + 8
        text_offset_y = local_radius + 6 if camera.get("orthographic") else local_radius + 14
        text_x = min(max(x + text_offset_x, 4), width - text_w - 12)
        text_y = min(max(y - text_offset_y, 4), height - text_h - 10)
        draw.text(
            (text_x, text_y),
            label,
            fill=(0, 0, 0, 255),
            font=point_font,
            stroke_width=3,
            stroke_fill=(255, 255, 255, 235),
        )

    return image


def annotate_view_frames(
    frames: Dict[str, Any],
    cameras: Dict[str, Any],
    maze_state: Dict[str, Any],
    *,
    house_data: Optional[Dict[str, Any]] = None,
    highlighted_points: Sequence[str] | None = None,
    highlighted_doors: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Image.Image]:
    return {
        view_name: annotate_maze_image(
            frames[view_name],
            camera=cameras[view_name],
            maze_state=maze_state,
            view_label=VIEW_LABELS[view_name],
            house_data=house_data,
            highlighted_points=highlighted_points,
            highlighted_doors=highlighted_doors,
        )
        for view_name in VIEW_ORDER
    }


def generate_reachability_scene(
    *,
    scene_index: int,
    start_seed: int,
    split: str = "train",
    max_attempts: int = 20,
    allow_warnings: bool = False,
    room_count: Optional[int] = None,
    door_count: Optional[int] = None,
    room_spec_id: Optional[str] = None,
    point_count: int = MIN_POINT_COUNT,
    door_state_mode: str = "random",
    runtime_name: Optional[str] = None,
    platform: Optional[str] = None,
    generation_width: int = DEFAULT_GENERATION_WIDTH,
    generation_height: int = DEFAULT_GENERATION_HEIGHT,
    generation_quality: str = DEFAULT_GENERATION_QUALITY,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
    render_quality: str = DEFAULT_RENDER_QUALITY,
    require_connected_subset_choice: bool = True,
    randomize_point_list_source: bool = True,
    require_door_open: bool = False,
) -> Dict[str, Any]:
    if point_count < MIN_POINT_COUNT:
        raise ValueError(
            f"point_count must be at least {MIN_POINT_COUNT} for the current 3D maze question set."
        )

    rng = random.Random(start_seed + scene_index * 1_000_003)
    last_error: Exception | None = None

    for attempt_offset in range(max_attempts):
        controller = None
        attempt_seed = start_seed + attempt_offset
        attempt_runtime_name = runtime_name or f"topobench_maze_{scene_index:04d}_{start_seed}"
        if runtime_name is None and max_attempts > 1:
            attempt_runtime_name = f"{attempt_runtime_name}_attempt_{attempt_offset + 1:03d}"

        try:
            house, used_seed, warnings, controller = sample_house_with_controller(
                split=split,
                start_seed=attempt_seed,
                max_attempts=1,
                allow_warnings=allow_warnings,
                room_count=room_count,
                door_count=door_count,
                room_spec_id=room_spec_id,
                runtime_name=attempt_runtime_name,
                platform=platform,
                quality=generation_quality,
                width=generation_width,
                height=generation_height,
            )

            house_data = house.data
            starting_door_states = initial_door_states(house_data)
            effective_door_state_mode = "closed" if require_door_open else door_state_mode
            target_door_states = choose_target_door_states(
                house_data,
                effective_door_state_mode,
                rng,
            )
            apply_door_states_to_house_data(house_data, target_door_states)
            door_results = set_door_states(controller, house_data, target_door_states)
            assert_door_state_results(target_door_states, door_results)
            door_open_graph_plans: List[Dict[str, Any]] = []
            if require_door_open:
                door_open_graph_plans = build_door_open_graph_plans(
                    house_data,
                    target_door_states,
                    rng=rng,
                )
                if not door_open_graph_plans:
                    raise RuntimeError(
                        "Unable to find a room-door graph plan with a unique minimum closed-door path."
                    )

            max_point_sampling_attempts = (
                max(20, point_count * 8) if require_door_open else max(6, point_count * 3)
            )
            last_sampling_error: Exception | None = None
            maze_state = None
            for point_attempt in range(max_point_sampling_attempts):
                graph_plan = None
                try:
                    if require_door_open:
                        graph_plan = door_open_graph_plans[
                            point_attempt % len(door_open_graph_plans)
                        ]
                        sampled_points = select_door_open_maze_candidates(
                            controller,
                            house_data,
                            count=point_count,
                            graph_plan=graph_plan,
                            rng=rng,
                            current_door_states=target_door_states,
                        )
                    else:
                        sampled_points = select_random_maze_candidates(
                            controller,
                            house_data,
                            count=point_count,
                            rng=rng,
                            current_door_states=target_door_states,
                            require_teleportable=True,
                        )
                except RuntimeError as exc:
                    last_sampling_error = exc
                    continue

                maze_state = evaluate_maze_points(controller, house_data, sampled_points)
                if require_connected_subset_choice and not supports_connected_subset_choice(maze_state):
                    continue
                if require_door_open:
                    graph_plan_for_points = {
                        **dict(graph_plan or {}),
                        "source_name": maze_point_name(0),
                        "target_name": maze_point_name(1),
                    }
                    door_open_question = build_unique_door_open_question(
                        controller,
                        house_data,
                        maze_state,
                        current_door_states=target_door_states,
                        rng=rng,
                        graph_plan=graph_plan_for_points,
                    )
                    if door_open_question is None:
                        continue
                    maze_state["door_open_question"] = door_open_question
                break
            else:
                requirements = []
                if require_connected_subset_choice:
                    requirements.append("connected-subset multiple-choice")
                if require_door_open:
                    requirements.append("unique door-open")
                requirement_text = " and ".join(requirements) or "current"
                if last_sampling_error is not None:
                    requirement_text += f" (last sampling error: {last_sampling_error})"
                raise RuntimeError(
                    f"Unable to sample labeled maze points that support the {requirement_text} question."
                )
            if randomize_point_list_source:
                point_entries = list(maze_state.get("points", []))
                if point_entries:
                    maze_state["point_list_source_name"] = str(rng.choice(point_entries)["name"])
            if require_connected_subset_choice:
                maze_state["subset_choice_shuffle_seed"] = rng.randrange(0, 2**31 - 1)

            prepare_controller_for_export(
                controller,
                width=render_width,
                height=render_height,
                quality=render_quality,
            )
            restore_default_agent_pose(controller, house_data)
            cameras = build_view_cameras(controller, house_data)
            frames = render_view_frames(controller, cameras)

            metadata = {
                "scene_index": scene_index,
                "requested_seed": start_seed,
                "used_seed": used_seed,
                "split": split,
                "room_count_filter": room_count,
                "door_count_filter": door_count,
                "room_spec_id": room_spec_id,
                "point_count": point_count,
                "door_state_mode": effective_door_state_mode,
                "num_rooms": len(house_data.get("rooms", [])),
                "num_objects": len(house_data.get("objects", [])),
                "num_openable_doors": len(starting_door_states),
                "initial_door_states": starting_door_states,
                "door_states": target_door_states,
                "door_action_results": door_results,
                "warnings": warnings,
                "points": maze_state["points"],
                "pairs": maze_state["pairs"],
                "point_list_source_name": maze_state.get("point_list_source_name"),
                "all_connected": maze_state["all_connected"],
                "render": {
                    "width": render_width,
                    "height": render_height,
                    "quality": render_quality,
                    "views": list(VIEW_ORDER),
                },
            }
            if "door_open_question" in maze_state:
                metadata["door_open_question"] = dict(maze_state["door_open_question"])
                metadata["door_colors"] = list(maze_state["door_open_question"].get("door_colors", []))
            return {
                "house_data": house_data,
                "metadata": metadata,
                "maze_state": maze_state,
                "frames": frames,
                "cameras": cameras,
            }
        except Exception as exc:
            last_error = exc
        finally:
            if controller is not None:
                controller.stop()

    raise RuntimeError(
        f"unable to generate a valid maze scene in {max_attempts} attempts"
    ) from last_error
