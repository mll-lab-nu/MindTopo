import heapq
import json
import math
import queue
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk

from house_pipeline import (
    AVAILABLE_ROOM_COUNTS,
    create_scene_controller,
    get_oblique_camera,
    openable_doors_from_house,
    render_camera_frame,
    sample_house_with_controller,
    set_door_states,
)
from maze_task import (
    VIEW_ORDER,
    annotate_view_frames,
    build_view_cameras,
    grounded_render_point,
    project_world_point,
    render_view_frames,
)

PREVIEW_RENDER_WIDTH = 1600
PREVIEW_RENDER_HEIGHT = 1200
PREVIEW_DISPLAY_WIDTH = 980
PREVIEW_DISPLAY_HEIGHT = 735
PREVIEW_MIN_RENDER_WIDTH = 1600
PREVIEW_MIN_RENDER_HEIGHT = 1200
PREVIEW_MAX_RENDER_WIDTH = 2048
PREVIEW_MAX_RENDER_HEIGHT = 1536
PREVIEW_RENDER_RESIZE_MS = 180
PREVIEW_RENDER_SIZE_STEP = 64
RENDER_DEBOUNCE_MS = 24
POLL_RESULTS_MS = 16
PREVIEW_PLATFORM = "CloudRendering"
PREVIEW_QUALITY = "Ultra"
PREVIEW_CAMERA_PITCH = 42.0
PREVIEW_CAMERA_FOV = 52
PREVIEW_ANTI_ALIASING = "smaa"
DOOR_COUNT_OPTIONS = ["Any"] + [str(value) for value in range(1, 9)]
EXPORT_DIR = Path("preview_exports")
EXPORT_RENDER_WIDTH = 1536
EXPORT_RENDER_HEIGHT = 1536
EXPORT_RENDER_QUALITY = "Ultra"
MAZE_ALLOWED_ERROR = 0.05
MAZE_POINT_MARKER_RADIUS = 10
MAZE_PATH_WIDTH = 5
MAX_MAZE_POINTS = 6
MAZE_VACANCY_MAX_RADIUS = 4
DOOR_HIGHLIGHT_FILL = (255, 204, 32, 165)
DOOR_HIGHLIGHT_SHADOW = (0, 0, 0, 140)


def door_world_quad(house_data: dict, door: dict) -> list[dict] | None:
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

    def world_point(local_x: float, local_y: float) -> dict:
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


def overlay_door_highlight_on_image(
    image: Image.Image,
    *,
    camera: dict,
    house_data: dict,
    door_id: str | None,
) -> Image.Image:
    if not door_id:
        return image

    door = next(
        (candidate for candidate in openable_doors_from_house(house_data) if candidate.get("id") == door_id),
        None,
    )
    if door is None:
        return image

    door_quad = door_world_quad(house_data, door)
    if not door_quad:
        return image

    source_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    annotated = source_image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    width, height = annotated.size
    projected = [
        project_world_point(point, camera, (width, height)) for point in door_quad
    ]
    visible = [
        point
        for point in projected
        if point is not None and all(math.isfinite(value) for value in point)
    ]
    outline_rgba = (*DOOR_HIGHLIGHT_FILL[:3], 255)
    if len(visible) >= 3:
        draw.polygon(visible, fill=(255, 238, 120, 75))
        draw.polygon(visible, fill=DOOR_HIGHLIGHT_FILL)
        draw.line([*visible, visible[0]], fill=outline_rgba, width=4)
    elif len(visible) >= 2:
        draw.line(visible, fill=DOOR_HIGHLIGHT_SHADOW, width=12)
        draw.line(visible, fill=outline_rgba, width=8)
    return annotated


def polygon_area_xz(points) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point["x"] * next_point["z"] - next_point["x"] * point["z"]
    return abs(area) * 0.5


def point_in_polygon_xz(point, polygon) -> bool:
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


def room_id_for_position(position, rooms) -> str | None:
    for room in rooms:
        if point_in_polygon_xz(position, room.get("floorPolygon", [])):
            return room["id"]
    return None


def distance_xz(point_a, point_b) -> float:
    return math.hypot(point_a["x"] - point_b["x"], point_a["z"] - point_b["z"])


def polyline_length_xz(points) -> float:
    if len(points) < 2:
        return 0.0

    total = 0.0
    for point_a, point_b in zip(points, points[1:]):
        total += distance_xz(point_a, point_b)
    return total


def distance_point_to_segment_xz(point, segment_start, segment_end) -> float:
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


def polygon_edge_clearance_xz(point, polygon) -> float:
    if len(polygon) < 2:
        return 0.0

    return min(
        distance_point_to_segment_xz(
            point,
            vertex,
            polygon[(index + 1) % len(polygon)],
        )
        for index, vertex in enumerate(polygon)
    )


def estimate_nav_step(positions) -> float:
    unique_x = sorted({round(point["x"], 4) for point in positions})
    unique_z = sorted({round(point["z"], 4) for point in positions})
    diffs = []

    for values in (unique_x, unique_z):
        for current, nxt in zip(values, values[1:]):
            delta = round(nxt - current, 4)
            if delta > 1e-4:
                diffs.append(delta)

    return min(diffs) if diffs else 0.25


def quantize_render_size(value: int) -> int:
    return max(
        PREVIEW_RENDER_SIZE_STEP,
        int(round(value / PREVIEW_RENDER_SIZE_STEP) * PREVIEW_RENDER_SIZE_STEP),
    )


def maze_point_name(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return f"P{index + 1}"


class PreviewWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.tasks: "queue.Queue[dict]" = queue.Queue()
        self.results: "queue.Queue[dict]" = queue.Queue()
        self.controller = None
        self.house = None
        self.house_data = None
        self.runtime_name = "preview_app"
        self.render_width = PREVIEW_RENDER_WIDTH
        self.render_height = PREVIEW_RENDER_HEIGHT
        self.door_states = {}
        self.maze_candidates = None
        self.maze_state = None

    def run(self) -> None:
        while True:
            task = self.tasks.get()
            kind = task["type"]
            if kind == "shutdown":
                self._close_controller()
                return

            try:
                if kind == "generate":
                    self._generate(task)
                elif kind == "update_doors":
                    self._update_doors(task)
                elif kind == "sample_maze_points":
                    self._sample_maze_points(task)
                elif kind == "render":
                    self._render(task)
                elif kind == "export_views":
                    self._export_views(task)
                elif kind == "resize_preview":
                    self._resize_preview(task)
            except Exception as exc:
                self.results.put({"type": "error", "message": str(exc)})

    def _close_controller(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self.house = None
        self.house_data = None
        self.door_states = {}
        self.maze_candidates = None
        self.maze_state = None

    def _render_image(self, yaw: float, pitch: float, zoom: float):
        camera = get_oblique_camera(
            self.house_data,
            yaw_deg=yaw,
            pitch_deg=pitch,
            zoom=zoom,
        )
        camera["fieldOfView"] = PREVIEW_CAMERA_FOV
        camera["antiAliasing"] = PREVIEW_ANTI_ALIASING
        frame = render_camera_frame(self.controller, camera)
        return Image.fromarray(frame)

    def _set_door_states(self, door_states: dict) -> dict:
        result = set_door_states(self.controller, self.house_data, door_states)
        for door_id, is_open in door_states.items():
            if result.get(door_id, {}).get("success", False):
                self.door_states[door_id] = is_open
        return result

    def _teleport_agent_to_point(self, position: dict) -> dict:
        agent_pose = self.house_data["metadata"]["agent"]
        event = self.controller.step(
            action="TeleportFull",
            position=position,
            rotation=agent_pose["rotation"],
            horizon=agent_pose["horizon"],
            standing=agent_pose["standing"],
            renderImage=False,
        )
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                event.metadata.get("errorMessage")
                or f"Unable to teleport to point {position}."
            )
        return event

    def _restore_default_agent_pose(self) -> None:
        agent_pose = self.house_data["metadata"]["agent"]
        self.controller.step(
            action="TeleportFull",
            **agent_pose,
            renderImage=False,
        )

    def _build_reachable_path(
        self,
        reachable_positions: list,
        start_point: dict,
        goal_point: dict,
    ) -> list:
        if not reachable_positions:
            return []

        step = estimate_nav_step(reachable_positions)
        search_radius = max(MAZE_ALLOWED_ERROR * 2.0, step * 0.6)

        def key_for(point):
            return (
                int(round(point["x"] / step)),
                int(round(point["z"] / step)),
            )

        nodes = {}
        for point in reachable_positions:
            nodes[key_for(point)] = {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point["z"]),
            }

        def nearest_key(target):
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
        came_from = {start_key: None}
        costs = {start_key: 0.0}
        neighbor_offsets = [
            (dx, dz)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if dx or dz
        ]

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
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic, new_cost, neighbor_key),
                )

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

    def _rank_maze_candidates(self, candidates: list) -> list:
        if not candidates:
            return []

        step = estimate_nav_step([candidate["position"] for candidate in candidates])
        rooms_by_id = {
            room["id"]: room
            for room in self.house_data.get("rooms", [])
        }

        def key_for(point):
            return (
                int(round(point["x"] / step)),
                int(round(point["z"] / step)),
            )

        node_keys = {key_for(candidate["position"]) for candidate in candidates}
        for candidate in candidates:
            point = candidate["position"]
            node_key = key_for(point)
            immediate_neighbors = 0
            wider_neighbors = 0
            occupancy_ratios = {}
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
                    if (
                        node_key[0] + dx * step_distance,
                        node_key[1] + dz * step_distance,
                    ) not in node_keys:
                        break
                    distance += 1
                cardinal_clearances.append(distance)
            min_cardinal_clearance = min(cardinal_clearances) if cardinal_clearances else 0

            room = rooms_by_id.get(candidate["room_id"])
            room_edge_clearance = polygon_edge_clearance_xz(
                point,
                room.get("floorPolygon", []) if room else [],
            )
            nav_clearance = step * (0.45 + min_cardinal_clearance * 0.75)
            if room is None:
                edge_clearance = nav_clearance
            else:
                edge_clearance = min(
                    max(room_edge_clearance, step * 0.35),
                    nav_clearance + step * 0.75,
                )
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

    def _comfortable_candidate_pool(self, candidates: list) -> list:
        if not candidates:
            return []

        step = candidates[0].get("nav_step", 0.25)
        filters = [
            lambda candidate: (
                candidate["min_cardinal_clearance"] >= 1
                and candidate["openness_score"] >= 0.5
                and candidate["edge_clearance"] >= step * 0.55
            ),
            lambda candidate: (
                candidate["openness_score"] >= 0.38
                and candidate["edge_clearance"] >= step * 0.35
            ),
            lambda candidate: (
                candidate["edge_clearance"] >= step * 0.2
            ),
        ]

        for predicate in filters:
            filtered = [candidate for candidate in candidates if predicate(candidate)]
            if len(filtered) >= 2:
                return filtered

        cutoff = max(2, min(len(candidates), max(12, len(candidates) // 2)))
        return candidates[:cutoff]

    def _ensure_maze_candidates(self) -> list:
        if self.maze_candidates is not None:
            return self.maze_candidates
        if self.controller is None or self.house_data is None:
            raise RuntimeError("No house loaded.")

        original_door_states = dict(self.door_states)
        all_open_door_states = {
            door_id: True for door_id in original_door_states
        }
        if all_open_door_states:
            self._set_door_states(all_open_door_states)

        try:
            event = self.controller.step(
                action="GetReachablePositions",
                renderImage=False,
            )
            if not event.metadata.get("lastActionSuccess", True):
                raise RuntimeError(
                    event.metadata.get("errorMessage")
                    or "Unable to retrieve reachable positions."
                )

            candidates = []
            rooms = self.house_data.get("rooms", [])
            for position in event.metadata.get("actionReturn") or []:
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
                self._set_door_states(original_door_states)

        if len(candidates) < 2:
            raise RuntimeError("Not enough valid navigation points for maze sampling.")

        self.maze_candidates = self._rank_maze_candidates(candidates)
        return self.maze_candidates

    def _get_reachable_positions_from_point(self, point: dict) -> list:
        self._teleport_agent_to_point(point)
        event = self.controller.step(
            action="GetReachablePositions",
            renderImage=False,
        )
        if not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError(
                event.metadata.get("errorMessage")
                or "Unable to retrieve reachable positions."
            )
        return event.metadata.get("actionReturn") or []

    def _candidate_sampling_weight(
        self,
        candidate: dict,
        selected: list[dict] | None = None,
    ) -> float:
        selected = selected or []
        step = max(candidate.get("nav_step", 0.25), 1e-6)
        score = max(float(candidate.get("score", 0.0)), 0.0)
        openness = max(float(candidate.get("openness_score", 0.0)), 0.0)
        edge_clearance = max(float(candidate.get("edge_clearance", 0.0)), 0.0)
        min_cardinal_clearance = max(
            float(candidate.get("min_cardinal_clearance", 0.0)),
            0.0,
        )

        # Keep every reachable point sampleable while mildly favoring clearer space.
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
            min_distance = min(
                distance_xz(candidate["position"], existing["position"])
                for existing in selected
            )
            distance_scale = max(step * 10.0, 1e-6)
            weight += min(min_distance / distance_scale, 1.15)
            if candidate.get("room_id") not in {item.get("room_id") for item in selected}:
                weight += 0.22

        return max(weight, 0.05)

    def _select_random_maze_candidates(self, count: int) -> list:
        candidates = self._ensure_maze_candidates()
        if count > len(candidates):
            raise RuntimeError(
                f"Only {len(candidates)} valid maze points are available in this house."
            )

        def weighted_choice(pool: list[dict], selected_so_far: list[dict]) -> dict:
            weights = [
                self._candidate_sampling_weight(candidate, selected_so_far)
                for candidate in pool
            ]
            return random.choices(pool, weights=weights, k=1)[0]

        selected = [weighted_choice(candidates, [])]
        selected_keys = {
            (
                round(selected[0]["position"]["x"], 4),
                round(selected[0]["position"]["z"], 4),
            )
        }

        while len(selected) < count:
            remaining = [
                candidate
                for candidate in candidates
                if (
                    round(candidate["position"]["x"], 4),
                    round(candidate["position"]["z"], 4),
                )
                not in selected_keys
            ]
            if not remaining:
                break

            next_candidate = weighted_choice(remaining, selected)
            selected.append(next_candidate)
            selected_keys.add(
                (
                    round(next_candidate["position"]["x"], 4),
                    round(next_candidate["position"]["z"], 4),
                )
            )

        if len(selected) < count:
            raise RuntimeError("Unable to sample enough distinct maze points.")
        return selected

    def _evaluate_maze_points(self, sampled_points: list[dict]) -> dict:
        if self.controller is None or self.house_data is None:
            raise RuntimeError("No house loaded.")

        points = []
        for index, point in enumerate(sampled_points):
            position = (
                dict(point["position"])
                if "position" in point
                else dict(point)
            )
            points.append(
                {
                    "name": maze_point_name(index),
                    "position": position,
                    "room_id": room_id_for_position(
                        position,
                        self.house_data.get("rooms", []),
                    ),
                }
            )

        pair_results = []
        display_path = []
        try:
            for index, source_point in enumerate(points):
                reachable_positions = self._get_reachable_positions_from_point(
                    source_point["position"]
                )
                for target_point in points[index + 1 :]:
                    path = self._build_reachable_path(
                        reachable_positions,
                        source_point["position"],
                        target_point["position"],
                    )
                    connected = bool(path)
                    pair_results.append(
                        {
                            "a_name": source_point["name"],
                            "b_name": target_point["name"],
                            "a_room_id": source_point["room_id"],
                            "b_room_id": target_point["room_id"],
                            "connected": connected,
                            "path_length": polyline_length_xz(path) if connected else None,
                            "error": None
                            if connected
                            else "No path exists for the current door states.",
                        }
                    )
                    if len(points) == 2:
                        display_path = path
        finally:
            self._restore_default_agent_pose()

        maze_state = {
            "points": points,
            "pairs": pair_results,
            "display_path": display_path,
            "all_connected": all(pair["connected"] for pair in pair_results),
        }
        self.maze_state = maze_state
        return maze_state

    def _choose_random_maze_points(self, count: int) -> dict:
        return self._evaluate_maze_points(
            self._select_random_maze_candidates(count)
        )

    def _generate(self, task: dict) -> None:
        self._close_controller()

        started_at = time.perf_counter()
        room_count = task["room_count"]
        door_count = task["door_count"]
        house, used_seed, warnings, controller = sample_house_with_controller(
            split="train",
            start_seed=task["seed"],
            max_attempts=task["max_attempts"],
            allow_warnings=False,
            room_count=room_count,
            door_count=door_count,
            runtime_name=self.runtime_name,
            platform=PREVIEW_PLATFORM,
            quality=PREVIEW_QUALITY,
            width=PREVIEW_RENDER_WIDTH,
            height=PREVIEW_RENDER_HEIGHT,
            validate_reachability=False,
        )
        self.house = house
        self.house_data = house.data
        self.controller = controller
        self.render_width = PREVIEW_RENDER_WIDTH
        self.render_height = PREVIEW_RENDER_HEIGHT

        yaw = house.data["metadata"]["agent"]["rotation"]["y"]
        pitch = PREVIEW_CAMERA_PITCH
        zoom = 1.0
        image = self._render_image(yaw=yaw, pitch=pitch, zoom=zoom)
        generation_seconds = time.perf_counter() - started_at

        openable_doors = openable_doors_from_house(house.data)
        door_states = {
            door["id"]: bool(door.get("openness", 0) >= 0.5) for door in openable_doors
        }
        self.door_states = dict(door_states)
        self.maze_candidates = None
        self.maze_state = None
        self.results.put(
            {
                "type": "generated",
                "seed": used_seed,
                "warnings": warnings,
                "room_count": len(house.data.get("rooms", [])),
                "num_objects": len(house.data.get("objects", [])),
                "room_spec_id": house.data["metadata"].get("roomSpecId"),
                "house_data": house.data,
                "doors": openable_doors,
                "door_states": door_states,
                "generation_seconds": generation_seconds,
                "yaw": yaw,
                "pitch": pitch,
                "zoom": zoom,
                "render_width": self.render_width,
                "render_height": self.render_height,
                "image": image,
            }
        )

    def _update_doors(self, task: dict) -> None:
        result = self._set_door_states(task["door_states"])
        maze_state = None
        if self.maze_state is not None:
            maze_state = self._evaluate_maze_points(self.maze_state["points"])
        image = self._render_image(
            yaw=task["yaw"],
            pitch=task["pitch"],
            zoom=task["zoom"],
        )
        self.results.put(
            {
                "type": "doors_updated",
                "door_results": result,
                "maze_state": maze_state,
                "image": image,
            }
        )

    def _sample_maze_points(self, task: dict) -> None:
        maze_state = self._choose_random_maze_points(task["count"])
        image = self._render_image(
            yaw=task["yaw"],
            pitch=task["pitch"],
            zoom=task["zoom"],
        )
        self.results.put(
            {
                "type": "maze_points_sampled",
                "maze_state": maze_state,
                "image": image,
            }
        )

    def _render(self, task: dict) -> None:
        image = self._render_image(
            yaw=task["yaw"],
            pitch=task["pitch"],
            zoom=task["zoom"],
        )
        self.results.put({"type": "rendered", "image": image})

    def _resize_preview(self, task: dict) -> None:
        if self.controller is None or self.house_data is None:
            raise RuntimeError("No house loaded.")

        width = task["width"]
        height = task["height"]
        if (
            self.controller.last_event.screen_width != width
            or self.controller.last_event.screen_height != height
        ):
            self.controller.step(
                action="ChangeResolution",
                x=width,
                y=height,
                renderImage=False,
                raise_for_failure=True,
            )
            self.controller.width = width
            self.controller.height = height

        self.render_width = width
        self.render_height = height
        image = self._render_image(
            yaw=task["yaw"],
            pitch=task["pitch"],
            zoom=task["zoom"],
        )
        self.results.put(
            {
                "type": "preview_resized",
                "width": width,
                "height": height,
                "image": image,
            }
        )

    def _export_views(self, task: dict) -> None:
        if self.house_data is None:
            raise RuntimeError("No house loaded.")

        seed = task["seed"]
        stamp = task["stamp"]
        export_dir = Path(task["export_dir"])
        door_states = task["door_states"]
        highlighted_door_id = task.get("highlighted_door_id") or ""
        export_dir.mkdir(parents=True, exist_ok=True)

        export_controller = None
        try:
            export_controller = create_scene_controller(
                self.house_data,
                runtime_name=f"{self.runtime_name}_export",
                width=EXPORT_RENDER_WIDTH,
                height=EXPORT_RENDER_HEIGHT,
                quality=EXPORT_RENDER_QUALITY,
                platform=PREVIEW_PLATFORM,
            )
            export_controller.step(
                action="TeleportFull",
                **self.house_data["metadata"]["agent"],
                renderImage=False,
            )
            set_door_states(export_controller, self.house_data, door_states)
            cameras = build_view_cameras(export_controller, self.house_data)
            frames = render_view_frames(export_controller, cameras)
            if highlighted_door_id:
                frames = {
                    view_name: overlay_door_highlight_on_image(
                        frames[view_name],
                        camera=cameras[view_name],
                        house_data=self.house_data,
                        door_id=highlighted_door_id,
                    )
                    for view_name in VIEW_ORDER
                }
            maze_state = self.maze_state or {"points": [], "pairs": [], "all_connected": False}
            views = annotate_view_frames(frames, cameras, maze_state, house_data=self.house_data)

            saved_paths = {}
            for view_name in VIEW_ORDER:
                path = export_dir / f"preview_seed_{seed}_{stamp}_{view_name}.png"
                views[view_name].save(path)
                saved_paths[view_name] = str(path)

            self.results.put({"type": "views_exported", "paths": saved_paths})
        finally:
            if export_controller is not None:
                export_controller.stop()


class PreviewApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ProcTHOR Studio")
        self.geometry("1440x900")
        self.configure(bg="#ebe4d8")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("Root.TFrame", background="#ebe4d8")
        self.style.configure("Card.TLabelframe", background="#f6f0e5", borderwidth=1)
        self.style.configure("Card.TLabelframe.Label", background="#f6f0e5", foreground="#3b352d")
        self.style.configure("App.TLabel", background="#ebe4d8", foreground="#3b352d")
        self.style.configure("Muted.TLabel", background="#ebe4d8", foreground="#746b5f")
        self.style.configure("Panel.TFrame", background="#f6f0e5")
        self.style.configure(
            "Accent.TButton",
            padding=8,
            background="#2f5d50",
            foreground="#ffffff",
        )
        self.style.configure("Small.TButton", padding=4)
        self.style.map(
            "Accent.TButton",
            background=[("active", "#25493f"), ("pressed", "#1e3a33")],
        )

        self.worker = PreviewWorker()
        self.worker.start()

        self.seed_var = tk.StringVar(value="12345")
        self.room_count_var = tk.StringVar(value="Any")
        self.door_count_var = tk.StringVar(value="Any")
        self.status_var = tk.StringVar(value="Ready")
        self.summary_var = tk.StringVar(value="No house loaded")
        self.maze_point_count_var = tk.StringVar(value="2")
        self.maze_status_var = tk.StringVar(
            value="Random maze points and pairwise connectivity will appear here after a house is loaded."
        )
        self.highlight_door_var = tk.StringVar(value="")

        self.current_yaw = 0.0
        self.current_pitch = 45.0
        self.current_zoom = 1.0
        self.base_yaw = 0.0
        self.base_pitch = 45.0
        self.base_zoom = 1.0
        self.orbit_drag_origin = None
        self.orbit_drag_active = False
        self.render_after_id = None
        self.busy_reason = None
        self.pending_render = None
        self.photo = None
        self.last_image = None
        self.image_resize_after_id = None
        self.preview_resize_after_id = None
        self.door_vars = {}
        self.current_house_data = None
        self.current_seed = None
        self.current_doors = []
        self.current_maze_state = None
        self.preview_render_width = PREVIEW_RENDER_WIDTH
        self.preview_render_height = PREVIEW_RENDER_HEIGHT
        self.pending_preview_resolution = None

        self._build_ui()
        self.after(POLL_RESULTS_MS, self.poll_results)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar_shell = ttk.Frame(self, style="Root.TFrame")
        sidebar_shell.grid(row=0, column=0, sticky="ns")
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)

        self.sidebar_canvas = tk.Canvas(
            sidebar_shell,
            width=340,
            highlightthickness=0,
            bd=0,
            bg="#ebe4d8",
        )
        self.sidebar_canvas.grid(row=0, column=0, sticky="ns")
        self.sidebar_canvas.bind("<Configure>", self.on_sidebar_canvas_configure)
        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_shell,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar = ttk.Frame(self.sidebar_canvas, padding=16, style="Root.TFrame")
        sidebar.rowconfigure(5, weight=1)
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0),
            window=sidebar,
            anchor="nw",
        )
        sidebar.bind("<Configure>", self.on_sidebar_frame_configure)

        viewer = ttk.Frame(self, padding=16, style="Root.TFrame")
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        ttk.Label(
            sidebar,
            text="ProcTHOR Studio",
            style="App.TLabel",
            font=("TkDefaultFont", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            sidebar,
            text="Seeded house preview with room/door/point setup controls, room summaries, and live door controls.",
            style="Muted.TLabel",
            wraplength=260,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 14))

        controls = ttk.LabelFrame(sidebar, text="Generate", style="Card.TLabelframe", padding=12)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Seed", style="App.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.seed_var, width=16).grid(
            row=1, column=0, sticky="ew", pady=(0, 8)
        )

        ttk.Label(controls, text="Room Count", style="App.TLabel").grid(row=2, column=0, sticky="w")
        counts = ["Any"] + [str(v) for v in AVAILABLE_ROOM_COUNTS]
        room_count_box = ttk.Combobox(
            controls,
            textvariable=self.room_count_var,
            values=counts,
            state="readonly",
            width=14,
        )
        room_count_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(controls, text="Controllable Doors", style="App.TLabel").grid(
            row=4, column=0, sticky="w"
        )
        door_count_box = ttk.Combobox(
            controls,
            textvariable=self.door_count_var,
            values=DOOR_COUNT_OPTIONS,
            state="readonly",
            width=14,
        )
        door_count_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        seed_actions = ttk.Frame(controls, style="Panel.TFrame")
        seed_actions.grid(row=6, column=0, sticky="ew", pady=(0, 6))
        seed_actions.columnconfigure(0, weight=1)
        seed_actions.columnconfigure(1, weight=1)
        ttk.Button(
            seed_actions,
            text="Random Seed",
            command=self.random_seed,
            style="Small.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            seed_actions,
            text="Generate House",
            command=self.generate_house,
            style="Accent.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        scene_card = ttk.LabelFrame(sidebar, text="Scene", style="Card.TLabelframe", padding=12)
        scene_card.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        scene_card.columnconfigure(0, weight=1)
        ttk.Label(scene_card, textvariable=self.summary_var, style="App.TLabel", wraplength=260).grid(
            row=0, column=0, sticky="ew", pady=(0, 10)
        )
        ttk.Button(
            scene_card,
            text="Reset View",
            command=self.reset_view,
            style="Small.TButton",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            scene_card,
            text="Save 5 Views",
            command=self.save_screenshot,
            style="Small.TButton",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            scene_card,
            text="Save House JSON",
            command=self.save_house_json,
            style="Small.TButton",
        ).grid(row=3, column=0, sticky="ew")
        ttk.Label(
            scene_card,
            text="Exports are saved under preview_exports/ with the current seed in the filename.",
            style="Muted.TLabel",
            wraplength=260,
        ).grid(row=4, column=0, sticky="ew", pady=(10, 0))

        maze_card = ttk.LabelFrame(sidebar, text="Maze", style="Card.TLabelframe", padding=12)
        maze_card.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        maze_card.columnconfigure(0, weight=1)
        ttk.Label(
            maze_card,
            text=f"Random Point Count (2-{MAX_MAZE_POINTS})",
            style="App.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(
            maze_card,
            textvariable=self.maze_point_count_var,
            width=8,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Button(
            maze_card,
            text="Random Maze Points",
            command=self.random_maze_points,
            style="Accent.TButton",
        ).grid(row=2, column=0, sticky="ew")
        ttk.Label(
            maze_card,
            textvariable=self.maze_status_var,
            style="App.TLabel",
            wraplength=260,
            justify="left",
        ).grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(
            maze_card,
            text="The connectivity result updates automatically when you open or close doors.",
            style="Muted.TLabel",
            wraplength=260,
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        doors_card = ttk.LabelFrame(sidebar, text="Doors", style="Card.TLabelframe", padding=12)
        doors_card.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        doors_card.columnconfigure(0, weight=1)
        doors_card.rowconfigure(1, weight=1)

        door_actions = ttk.Frame(doors_card, style="Panel.TFrame")
        door_actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        door_actions.columnconfigure(0, weight=1)
        door_actions.columnconfigure(1, weight=1)
        door_actions.columnconfigure(2, weight=1)
        ttk.Button(
            door_actions,
            text="Open All",
            command=lambda: self.apply_all_doors(True),
            style="Small.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            door_actions,
            text="Close All",
            command=lambda: self.apply_all_doors(False),
            style="Small.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(
            door_actions,
            text="Clear Highlight",
            command=self.clear_door_highlight,
            style="Small.TButton",
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        self.door_canvas = tk.Canvas(
            doors_card,
            width=260,
            highlightthickness=0,
            bg="#f6f0e5",
            bd=0,
        )
        self.door_canvas.grid(row=1, column=0, sticky="nsew")
        door_scrollbar = ttk.Scrollbar(
            doors_card, orient="vertical", command=self.door_canvas.yview
        )
        door_scrollbar.grid(row=1, column=1, sticky="ns")
        self.door_canvas.configure(yscrollcommand=door_scrollbar.set)

        self.door_frame = ttk.Frame(self.door_canvas, style="Panel.TFrame")
        self.door_frame.bind(
            "<Configure>",
            lambda event: self.door_canvas.configure(
                scrollregion=self.door_canvas.bbox("all")
            ),
        )
        self.door_canvas.create_window((0, 0), window=self.door_frame, anchor="nw")

        ttk.Label(
            viewer,
            text="Left drag to orbit, W zooms in, S zooms out. Preview renders through CloudRendering, so only this tool window should remain once the build is ready.",
            style="Muted.TLabel",
            wraplength=860,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.image_panel = tk.Canvas(
            viewer,
            bg="#1d2128",
            highlightthickness=0,
            bd=0,
            takefocus=1,
        )
        self.image_panel.grid(row=1, column=0, sticky="nsew")
        self.image_panel.bind("<Configure>", self.on_image_panel_resize)
        self.image_panel.bind("<KeyPress-w>", self.on_zoom_in_key)
        self.image_panel.bind("<KeyPress-W>", self.on_zoom_in_key)
        self.image_panel.bind("<KeyPress-s>", self.on_zoom_out_key)
        self.image_panel.bind("<KeyPress-S>", self.on_zoom_out_key)
        self.image_panel.bind("<Enter>", lambda _event: self.image_panel.focus_force())
        self.bind_all("<KeyPress-w>", self.on_zoom_in_key)
        self.bind_all("<KeyPress-W>", self.on_zoom_in_key)
        self.bind_all("<KeyPress-s>", self.on_zoom_out_key)
        self.bind_all("<KeyPress-S>", self.on_zoom_out_key)
        self.bind_all("<ButtonPress-1>", self.on_global_button_press, add="+")
        self.bind_all("<ButtonRelease-1>", self.on_global_button_release, add="+")
        self.bind_all("<B1-Motion>", self.on_global_button_motion, add="+")
        self.bind_all("<MouseWheel>", self.on_global_mouse_wheel, add="+")
        self.bind_all("<MouseWheel>", self.on_global_sidebar_mouse_wheel, add="+")
        self.bind_all(
            "<Button-4>",
            lambda event: self.on_global_mouse_wheel_legacy(event, True),
            add="+",
        )
        self.bind_all(
            "<Button-4>",
            lambda event: self.on_global_sidebar_mouse_wheel_legacy(event, True),
            add="+",
        )
        self.bind_all(
            "<Button-5>",
            lambda event: self.on_global_mouse_wheel_legacy(event, False),
            add="+",
        )
        self.bind_all(
            "<Button-5>",
            lambda event: self.on_global_sidebar_mouse_wheel_legacy(event, False),
            add="+",
        )
        self.image_panel.create_text(
            PREVIEW_DISPLAY_WIDTH // 2,
            PREVIEW_DISPLAY_HEIGHT // 2,
            text="Generate a house to begin",
            fill="#e7e5e4",
            font=("TkDefaultFont", 13, "bold"),
            tags=("placeholder",),
        )

        status_frame = ttk.Frame(viewer, style="Root.TFrame")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(status_frame, textvariable=self.status_var, style="App.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def random_seed(self) -> None:
        self.seed_var.set(str(random.randint(0, 999999)))

    def generate_house(self) -> None:
        if self.busy_reason is not None:
            return

        try:
            seed = int(self.seed_var.get())
        except ValueError:
            self.status_var.set("Seed must be an integer.")
            return

        room_count = (
            None if self.room_count_var.get() == "Any" else int(self.room_count_var.get())
        )
        door_count = (
            None if self.door_count_var.get() == "Any" else int(self.door_count_var.get())
        )
        self.busy_reason = "generate"
        self.pending_render = None
        self.status_var.set("Generating house...")
        summary_suffix = ""
        if room_count is not None:
            summary_suffix += f" {room_count} rooms"
        if door_count is not None:
            summary_suffix += f" {door_count} controllable doors"
        self.summary_var.set(f"Sampling ProcTHOR house{summary_suffix}...".strip())
        self.clear_maze_state()
        self.clear_doors()
        self.worker.tasks.put(
            {
                "type": "generate",
                "seed": seed,
                "room_count": room_count,
                "door_count": door_count,
                "max_attempts": 20,
            }
        )

    def clear_maze_state(self) -> None:
        self.current_maze_state = None
        self.maze_status_var.set(
            "Random maze points and pairwise connectivity will appear here after a house is loaded."
        )
        self.refresh_display_image()

    def random_maze_points(self) -> None:
        if self.current_house_data is None:
            self.status_var.set("Generate a house before sampling maze points.")
            return
        if self.busy_reason is not None:
            self.status_var.set("Please wait for the current task to finish.")
            return

        try:
            point_count = int(self.maze_point_count_var.get())
        except ValueError:
            self.status_var.set("Maze point count must be an integer.")
            return

        if point_count < 2 or point_count > MAX_MAZE_POINTS:
            self.status_var.set(
                f"Maze point count must be between 2 and {MAX_MAZE_POINTS}."
            )
            return

        self.busy_reason = "maze_points"
        self.status_var.set(f"Sampling {point_count} valid maze points...")
        self.worker.tasks.put(
            {
                "type": "sample_maze_points",
                "count": point_count,
                "yaw": self.current_yaw,
                "pitch": self.current_pitch,
                "zoom": self.current_zoom,
            }
        )

    def clear_doors(self) -> None:
        self.door_vars.clear()
        self.current_doors = []
        self.highlight_door_var.set("")
        for child in self.door_frame.winfo_children():
            child.destroy()

    def clear_door_highlight(self) -> None:
        if not self.highlight_door_var.get():
            return
        self.highlight_door_var.set("")
        self.refresh_display_image()

    def on_highlight_door_change(self) -> None:
        self.refresh_display_image()

    def room_label(self, room_id: str | None) -> str:
        if room_id is None:
            return "Transition"
        if not self.current_house_data:
            return room_id
        room = next(
            (room for room in self.current_house_data.get("rooms", []) if room["id"] == room_id),
            None,
        )
        if room is None:
            return room_id
        numeric = room_id.split("|")[-1]
        return f"{room['roomType']} {numeric}"

    def point_label(self, name: str, point: dict, room_id: str | None) -> str:
        room_label = self.room_label(room_id)
        return f"{name}: {room_label} ({point['x']:.2f}, {point['z']:.2f})"

    def set_maze_state(self, maze_state: dict | None) -> None:
        self.current_maze_state = maze_state
        if maze_state is None:
            self.maze_status_var.set(
                "Random maze points and pairwise connectivity will appear here after a house is loaded."
            )
            self.refresh_display_image()
            return

        point_lines = [
            self.point_label(
                point["name"],
                point["position"],
                point.get("room_id"),
            )
            for point in maze_state.get("points", [])
        ]
        pair_lines = []
        for pair in maze_state.get("pairs", []):
            line = (
                f"{pair['a_name']}-{pair['b_name']}: "
                f"{'Connected' if pair['connected'] else 'Blocked'}"
            )
            if pair["connected"] and pair.get("path_length") is not None:
                line += f" ({pair['path_length']:.2f} m)"
            pair_lines.append(line)

        summary_line = (
            f"{len(maze_state.get('pairs', []))}/{len(maze_state.get('pairs', []))} pairs connected"
            if maze_state.get("all_connected")
            else (
                f"{sum(1 for pair in maze_state.get('pairs', []) if pair['connected'])}/"
                f"{len(maze_state.get('pairs', []))} pairs connected"
            )
        )
        status_sections = [
            f"Points ({len(point_lines)})",
            *point_lines,
            "",
            summary_line,
            *pair_lines,
        ]
        self.maze_status_var.set("\n".join(status_sections))
        self.refresh_display_image()

    def build_room_summary(self, result) -> str:
        house_data = result["house_data"]
        counts = {}
        for room in house_data.get("rooms", []):
            counts[room["roomType"]] = counts.get(room["roomType"], 0) + 1
        count_summary = ", ".join(
            f"{count} {room_type}" for room_type, count in sorted(counts.items())
        )
        return (
            f"Seed {result['seed']}\n"
            f"Room Spec {result['room_spec_id']}\n"
            f"Rooms {result['room_count']}  ({count_summary})\n"
            f"Objects {result['num_objects']}  Openable Doors {len(result['doors'])}\n"
            f"Warnings {len(result['warnings'])}  Generated {result['generation_seconds']:.1f}s"
        )

    def rebuild_door_controls(self, doors, door_states) -> None:
        selected_highlight = self.highlight_door_var.get()
        door_ids = {door["id"] for door in doors}
        if selected_highlight and selected_highlight not in door_ids:
            self.highlight_door_var.set("")

        self.clear_doors()
        self.current_doors = list(doors)
        if selected_highlight in door_ids:
            self.highlight_door_var.set(selected_highlight)
        grouped = {}
        for door in doors:
            room0 = self.room_label(door["room0"])
            room1 = self.room_label(door["room1"])
            group_label = " / ".join(sorted({room0, room1}))
            grouped.setdefault(group_label, []).append(door)

        row = 0
        for group_label in sorted(grouped):
            group_doors = sorted(grouped[group_label], key=lambda door: door["id"])
            ttk.Label(
                self.door_frame,
                text=f"{group_label}  ({len(group_doors)} door{'s' if len(group_doors) != 1 else ''})",
                style="Muted.TLabel",
            ).grid(row=row, column=0, sticky="w", pady=(8 if row else 0, 2))
            row += 1
            for index, door in enumerate(group_doors, start=1):
                door_id = door["id"]
                var = tk.BooleanVar(value=door_states.get(door_id, False))
                self.door_vars[door_id] = var
                label = f"Door {index}  {'Open' if var.get() else 'Closed'}"
                door_row = ttk.Frame(self.door_frame, style="Panel.TFrame")
                door_row.grid(row=row, column=0, sticky="ew", pady=2, padx=(10, 0))
                door_row.columnconfigure(0, weight=1)
                button = ttk.Checkbutton(
                    door_row,
                    text=label,
                    variable=var,
                    command=self.on_door_toggle,
                )
                button.grid(row=0, column=0, sticky="w")
                highlight_button = ttk.Radiobutton(
                    door_row,
                    text="Highlight",
                    value=door_id,
                    variable=self.highlight_door_var,
                    command=self.on_highlight_door_change,
                )
                highlight_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
                row += 1

    def refresh_door_labels(self) -> None:
        if self.current_house_data is None:
            return
        self.rebuild_door_controls(
            self.current_doors or openable_doors_from_house(self.current_house_data),
            {door_id: var.get() for door_id, var in self.door_vars.items()},
        )

    def set_preview_image(self, image: Image.Image) -> None:
        self.last_image = image.copy()
        self.refresh_display_image()

    def get_display_size(self) -> tuple[int, int]:
        width = self.image_panel.winfo_width() - 48
        height = self.image_panel.winfo_height() - 48
        if width <= 32 or height <= 32:
            return PREVIEW_DISPLAY_WIDTH, PREVIEW_DISPLAY_HEIGHT
        return max(64, width), max(64, height)

    def get_target_preview_render_size(self) -> tuple[int, int]:
        display_width, display_height = self.get_display_size()
        width = min(
            PREVIEW_MAX_RENDER_WIDTH,
            max(PREVIEW_MIN_RENDER_WIDTH, quantize_render_size(display_width)),
        )
        height = min(
            PREVIEW_MAX_RENDER_HEIGHT,
            max(PREVIEW_MIN_RENDER_HEIGHT, quantize_render_size(display_height)),
        )
        return width, height

    def build_preview_camera(self) -> dict | None:
        if self.current_house_data is None:
            return None

        camera = get_oblique_camera(
            self.current_house_data,
            yaw_deg=self.current_yaw,
            pitch_deg=self.current_pitch,
            zoom=self.current_zoom,
        )
        camera["fieldOfView"] = PREVIEW_CAMERA_FOV
        return camera

    def project_world_point(self, point: dict, camera: dict, image_size: tuple[int, int]):
        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        position = camera["position"]
        yaw = math.radians(camera["rotation"]["y"])
        pitch = math.radians(camera["rotation"]["x"])

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

        offset = (
            point["x"] - position["x"],
            point["y"] - position["y"],
            point["z"] - position["z"],
        )
        camera_x = sum(component * axis for component, axis in zip(offset, right))
        camera_y = sum(component * axis for component, axis in zip(offset, up))
        camera_z = sum(component * axis for component, axis in zip(offset, forward))
        if camera_z <= 0.05:
            return None

        tan_half_fov = math.tan(math.radians(camera["fieldOfView"]) / 2.0)
        aspect = width / height
        ndc_x = camera_x / (camera_z * tan_half_fov * aspect)
        ndc_y = camera_y / (camera_z * tan_half_fov)
        return (
            (ndc_x + 1.0) * 0.5 * width,
            (1.0 - ndc_y) * 0.5 * height,
        )

    def overlay_door_highlight(self, image: Image.Image) -> Image.Image:
        camera = self.build_preview_camera()
        if camera is None or self.current_house_data is None:
            return image

        return overlay_door_highlight_on_image(
            image,
            camera=camera,
            house_data=self.current_house_data,
            door_id=self.highlight_door_var.get(),
        )

    def overlay_maze_annotations(self, image: Image.Image) -> Image.Image:
        if self.current_maze_state is None:
            return image

        camera = self.build_preview_camera()
        if camera is None:
            return image

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated, "RGBA")
        width, height = annotated.size
        lifted_path = [
            grounded_render_point(point, self.current_house_data, lift=0.06)
            for point in self.current_maze_state.get("display_path", [])
        ]
        projected_path = [
            self.project_world_point(point, camera, (width, height))
            for point in lifted_path
        ]
        visible_path = [point for point in projected_path if point is not None]
        if len(visible_path) >= 2:
            draw.line(visible_path, fill=(0, 0, 0, 180), width=MAZE_PATH_WIDTH + 3)
            draw.line(visible_path, fill=(78, 167, 109, 220), width=MAZE_PATH_WIDTH)

        point_palette = [
            (213, 74, 74, 240),
            (57, 159, 214, 240),
            (230, 164, 54, 240),
            (97, 176, 89, 240),
            (168, 98, 212, 240),
            (224, 112, 170, 240),
        ]
        for index, point_entry in enumerate(self.current_maze_state.get("points", [])):
            label = point_entry["name"]
            point = point_entry["position"]
            fill = point_palette[index % len(point_palette)]
            base_point = grounded_render_point(point, self.current_house_data, lift=0.02)
            base_screen_point = self.project_world_point(base_point, camera, (width, height))
            if base_screen_point is None:
                continue

            x, y = base_screen_point
            radius = MAZE_POINT_MARKER_RADIUS
            draw.ellipse(
                (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
                fill=(0, 0, 0, 170),
            )
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline=(255, 255, 255, 235),
                width=2,
            )
            draw.text((x + radius + 6, y - radius - 4), label, fill=(255, 255, 255, 255))

        pair_count = len(self.current_maze_state.get("pairs", []))
        connected_count = sum(
            1 for pair in self.current_maze_state.get("pairs", []) if pair["connected"]
        )
        all_connected = bool(pair_count) and self.current_maze_state.get("all_connected", False)
        if pair_count == 1:
            status_text = "Maze: Connected" if all_connected else "Maze: Blocked"
        else:
            status_text = f"Maze: {connected_count}/{pair_count} pairs"
        status_fill = (47, 93, 77, 215) if all_connected else (121, 69, 41, 215)
        draw.rounded_rectangle((12, 12, 200, 40), radius=8, fill=status_fill)
        draw.text((24, 20), status_text, fill=(255, 255, 255, 255))
        return annotated

    def refresh_display_image(self) -> None:
        if self.last_image is None:
            self.image_panel.delete("all")
            width = max(320, self.image_panel.winfo_width())
            height = max(240, self.image_panel.winfo_height())
            self.image_panel.create_text(
                width / 2,
                height / 2,
                text="Generate a house to begin",
                fill="#e7e5e4",
                font=("TkDefaultFont", 13, "bold"),
                tags=("placeholder",),
            )
            return
        target_size = self.get_display_size()
        if (
            self.last_image.width <= target_size[0]
            and self.last_image.height <= target_size[1]
        ):
            display = self.last_image.copy()
        else:
            display = ImageOps.contain(
                self.last_image,
                target_size,
                method=Image.Resampling.LANCZOS,
            )
        display = self.overlay_door_highlight(display)
        display = self.overlay_maze_annotations(display)
        self.photo = ImageTk.PhotoImage(display)
        self.image_panel.delete("all")
        canvas_width = max(1, self.image_panel.winfo_width())
        canvas_height = max(1, self.image_panel.winfo_height())
        self.image_panel.create_image(
            canvas_width / 2,
            canvas_height / 2,
            image=self.photo,
            anchor="center",
            tags=("preview_image",),
        )

    def on_image_panel_resize(self, _event=None) -> None:
        if self.image_resize_after_id is not None:
            self.after_cancel(self.image_resize_after_id)
        self.image_resize_after_id = self.after(20, self.finish_image_panel_resize)
        self.schedule_preview_resolution_update()

    def on_sidebar_frame_configure(self, _event=None) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def on_sidebar_canvas_configure(self, event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def finish_image_panel_resize(self) -> None:
        self.image_resize_after_id = None
        self.refresh_display_image()

    def schedule_preview_resolution_update(self) -> None:
        if self.current_house_data is None:
            return
        if self.preview_resize_after_id is not None:
            self.after_cancel(self.preview_resize_after_id)
        self.preview_resize_after_id = self.after(
            PREVIEW_RENDER_RESIZE_MS,
            self.enqueue_preview_resolution_update,
        )

    def enqueue_preview_resolution_update(self) -> None:
        self.preview_resize_after_id = None
        if self.current_house_data is None:
            return

        target = self.get_target_preview_render_size()
        if target == (self.preview_render_width, self.preview_render_height):
            return

        self.pending_preview_resolution = target
        if self.busy_reason is not None:
            return

        width, height = self.pending_preview_resolution
        self.pending_preview_resolution = None
        self.pending_render = None
        self.busy_reason = "resize_preview"
        self.status_var.set(f"Refining preview to {width}x{height}...")
        self.worker.tasks.put(
            {
                "type": "resize_preview",
                "width": width,
                "height": height,
                "yaw": self.current_yaw,
                "pitch": self.current_pitch,
                "zoom": self.current_zoom,
            }
        )

    def on_door_toggle(self) -> None:
        if self.busy_reason is not None or not self.door_vars:
            return

        self.refresh_door_labels()
        self.busy_reason = "door"
        self.status_var.set("Applying door states...")
        self.worker.tasks.put(
            {
                "type": "update_doors",
                "door_states": {
                    door_id: var.get() for door_id, var in self.door_vars.items()
                },
                "yaw": self.current_yaw,
                "pitch": self.current_pitch,
                "zoom": self.current_zoom,
            }
        )

    def apply_all_doors(self, is_open: bool) -> None:
        if not self.door_vars or self.busy_reason is not None:
            return
        for var in self.door_vars.values():
            var.set(is_open)
        self.on_door_toggle()

    def reset_view(self) -> None:
        if self.busy_reason == "generate":
            return
        self.current_yaw = self.base_yaw
        self.current_pitch = self.base_pitch
        self.current_zoom = self.base_zoom
        self.schedule_render()

    def save_screenshot(self) -> None:
        if self.current_house_data is None:
            self.status_var.set("No house loaded yet.")
            return
        if self.busy_reason is not None:
            self.status_var.set("Please wait for the current task to finish.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.current_seed if self.current_seed is not None else "unknown"
        self.busy_reason = "export"
        self.status_var.set("Saving topdown, front45, right45, rear45, and left45 views...")
        self.worker.tasks.put(
            {
                "type": "export_views",
                "seed": seed,
                "stamp": stamp,
                "export_dir": str(EXPORT_DIR),
                "door_states": {
                    door_id: var.get() for door_id, var in self.door_vars.items()
                },
                "highlighted_door_id": self.highlight_door_var.get(),
            }
        )

    def save_house_json(self) -> None:
        if self.current_house_data is None:
            self.status_var.set("No house JSON to save yet.")
            return
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.current_seed if self.current_seed is not None else "unknown"
        path = EXPORT_DIR / f"preview_seed_{seed}_{stamp}.json"
        path.write_text(json.dumps(self.current_house_data, indent=2))
        self.status_var.set(f"Saved house JSON to {path}")

    def on_orbit_start(self, event) -> None:
        if self.current_house_data is None:
            return
        self.image_panel.focus_force()
        self.orbit_drag_active = True
        self.orbit_drag_origin = (event.x_root, event.y_root)

    def on_orbit_motion(self, event) -> None:
        if self.orbit_drag_origin is None or self.busy_reason == "generate":
            return

        dx = event.x_root - self.orbit_drag_origin[0]
        dy = event.y_root - self.orbit_drag_origin[1]
        self.orbit_drag_origin = (event.x_root, event.y_root)

        self.current_yaw = (self.current_yaw + dx * 0.6) % 360
        self.current_pitch = min(80, max(15, self.current_pitch + dy * 0.35))
        self.schedule_render()

    def on_orbit_end(self, _event=None) -> None:
        self.orbit_drag_active = False
        self.orbit_drag_origin = None

    def widget_is_preview_widget(self, widget) -> bool:
        current = widget
        while current is not None:
            if current is self.image_panel:
                return True
            current = getattr(current, "master", None)
        return False

    def widget_is_sidebar_widget(self, widget) -> bool:
        current = widget
        while current is not None:
            if current is self.sidebar_canvas:
                return True
            current = getattr(current, "master", None)
        return False

    def resolve_event_widget(self, event):
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            widget = None
        if widget is None:
            widget = getattr(event, "widget", None)
        return widget

    def event_targets_preview(self, event) -> bool:
        widget = self.resolve_event_widget(event)
        return self.widget_is_preview_widget(widget)

    def event_targets_sidebar(self, event) -> bool:
        widget = self.resolve_event_widget(event)
        return self.widget_is_sidebar_widget(widget)

    def on_global_button_press(self, event) -> None:
        if not self.event_targets_preview(event):
            return
        self.on_orbit_start(event)

    def on_global_button_motion(self, event) -> None:
        if not self.orbit_drag_active:
            return
        self.on_orbit_motion(event)

    def on_global_button_release(self, event) -> None:
        if not self.orbit_drag_active:
            return
        self.on_orbit_end(event)

    def on_global_mouse_wheel(self, event) -> None:
        if not self.event_targets_preview(event):
            return
        self.image_panel.focus_force()
        self.on_mouse_wheel(event)

    def on_global_mouse_wheel_legacy(self, event, zoom_in: bool) -> None:
        if not self.event_targets_preview(event):
            return
        self.image_panel.focus_force()
        self.on_mouse_wheel_legacy(zoom_in)

    def scroll_sidebar_units(self, units: int) -> None:
        self.sidebar_canvas.yview_scroll(units, "units")

    def on_global_sidebar_mouse_wheel(self, event) -> None:
        if not self.event_targets_sidebar(event):
            return
        if event.delta == 0:
            return
        units = -1 if event.delta > 0 else 1
        self.scroll_sidebar_units(units)

    def on_global_sidebar_mouse_wheel_legacy(self, event, scroll_up: bool) -> None:
        if not self.event_targets_sidebar(event):
            return
        self.scroll_sidebar_units(-1 if scroll_up else 1)

    def should_handle_zoom_key(self) -> bool:
        return self.current_house_data is not None and self.busy_reason != "generate"

    def step_zoom(self, factor: float) -> None:
        if not self.should_handle_zoom_key():
            return
        self.current_zoom *= factor
        self.current_zoom = min(2.5, max(0.55, self.current_zoom))
        self.schedule_render()

    def on_zoom_in_key(self, _event=None) -> None:
        self.step_zoom(0.92)

    def on_zoom_out_key(self, _event=None) -> None:
        self.step_zoom(1.08)

    def on_mouse_wheel(self, event) -> None:
        if self.busy_reason == "generate":
            return

        if event.delta > 0:
            self.current_zoom *= 0.92
        else:
            self.current_zoom *= 1.08
        self.current_zoom = min(2.5, max(0.55, self.current_zoom))
        self.schedule_render()

    def on_mouse_wheel_legacy(self, zoom_in: bool) -> None:
        if self.busy_reason == "generate":
            return
        if zoom_in:
            self.current_zoom *= 0.92
        else:
            self.current_zoom *= 1.08
        self.current_zoom = min(2.5, max(0.55, self.current_zoom))
        self.schedule_render()

    def schedule_render(self) -> None:
        if self.render_after_id is not None:
            self.after_cancel(self.render_after_id)
        self.pending_render = (self.current_yaw, self.current_pitch, self.current_zoom)
        self.render_after_id = self.after(RENDER_DEBOUNCE_MS, self.enqueue_render)

    def enqueue_render(self) -> None:
        self.render_after_id = None
        if self.pending_render is None:
            return

        if self.busy_reason is not None:
            return
        if self.pending_preview_resolution is not None:
            self.after(1, self.enqueue_preview_resolution_update)
            return

        yaw, pitch, zoom = self.pending_render
        self.pending_render = None
        self.busy_reason = "render"
        self.status_var.set(
            f"Rendering view yaw={yaw:.1f} pitch={pitch:.1f}"
        )
        self.worker.tasks.put(
            {
                "type": "render",
                "yaw": yaw,
                "pitch": pitch,
                "zoom": zoom,
            }
        )

    def maybe_dispatch_pending_render(self) -> None:
        if self.busy_reason is not None:
            return
        if self.pending_preview_resolution is not None:
            self.after(1, self.enqueue_preview_resolution_update)
            return
        if self.pending_render is not None:
            self.after(1, self.enqueue_render)

    def poll_results(self) -> None:
        while True:
            try:
                result = self.worker.results.get_nowait()
            except queue.Empty:
                break

            result_type = result["type"]
            if result_type == "generated":
                self.busy_reason = None
                self.current_house_data = result["house_data"]
                self.current_seed = result["seed"]
                self.current_yaw = result["yaw"]
                self.current_pitch = result["pitch"]
                self.current_zoom = result["zoom"]
                self.base_yaw = result["yaw"]
                self.base_pitch = result["pitch"]
                self.base_zoom = result["zoom"]
                self.preview_render_width = result["render_width"]
                self.preview_render_height = result["render_height"]
                self.set_preview_image(result["image"])
                self.seed_var.set(str(result["seed"]))
                self.summary_var.set(self.build_room_summary(result))
                self.set_maze_state(None)
                self.status_var.set(
                    f"House generated in {result['generation_seconds']:.1f}s."
                )
                self.rebuild_door_controls(result["doors"], result["door_states"])
                self.schedule_preview_resolution_update()
                self.maybe_dispatch_pending_render()

            elif result_type == "doors_updated":
                self.busy_reason = None
                self.set_preview_image(result["image"])
                if result.get("maze_state") is not None:
                    self.set_maze_state(result["maze_state"])
                self.refresh_door_labels()
                if self.current_maze_state is not None:
                    pair_count = len(self.current_maze_state.get("pairs", []))
                    connected_count = sum(
                        1
                        for pair in self.current_maze_state.get("pairs", [])
                        if pair["connected"]
                    )
                    self.status_var.set(
                        "Door states updated. "
                        + f"{connected_count}/{pair_count} maze pairs connected."
                    )
                else:
                    self.status_var.set("Door states updated.")
                self.maybe_dispatch_pending_render()

            elif result_type == "maze_points_sampled":
                self.busy_reason = None
                self.set_preview_image(result["image"])
                self.set_maze_state(result["maze_state"])
                pair_count = len(result["maze_state"].get("pairs", []))
                connected_count = sum(
                    1
                    for pair in result["maze_state"].get("pairs", [])
                    if pair["connected"]
                )
                self.status_var.set(
                    f"Maze points sampled. {connected_count}/{pair_count} pairs connected."
                )
                self.maybe_dispatch_pending_render()

            elif result_type == "rendered":
                self.busy_reason = None
                self.set_preview_image(result["image"])
                self.status_var.set(
                    f"View yaw={self.current_yaw:.1f} pitch={self.current_pitch:.1f} zoom={self.current_zoom:.2f}"
                )
                self.maybe_dispatch_pending_render()

            elif result_type == "preview_resized":
                self.busy_reason = None
                self.preview_render_width = result["width"]
                self.preview_render_height = result["height"]
                self.set_preview_image(result["image"])
                self.status_var.set(
                    f"View yaw={self.current_yaw:.1f} pitch={self.current_pitch:.1f} zoom={self.current_zoom:.2f}"
                )
                self.maybe_dispatch_pending_render()

            elif result_type == "views_exported":
                self.busy_reason = None
                self.status_var.set(
                    "Saved 5 views to "
                    + ", ".join(
                        Path(result["paths"][name]).name
                        for name in ("topdown", "front45", "right45", "rear45", "left45")
                    )
                )
                self.maybe_dispatch_pending_render()

            elif result_type == "error":
                self.busy_reason = None
                self.status_var.set(result["message"])
                self.maybe_dispatch_pending_render()

        self.after(POLL_RESULTS_MS, self.poll_results)

    def on_close(self) -> None:
        self.on_orbit_end()
        if self.image_resize_after_id is not None:
            self.after_cancel(self.image_resize_after_id)
        if self.preview_resize_after_id is not None:
            self.after_cancel(self.preview_resize_after_id)
        self.worker.tasks.put({"type": "shutdown"})
        self.destroy()


def main() -> None:
    app = PreviewApp()
    app.mainloop()


if __name__ == "__main__":
    main()
