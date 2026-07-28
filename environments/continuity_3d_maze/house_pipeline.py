import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from thor_runtime import (
    DEFAULT_BRANCH,
    prepare_ai2thor_runtime,
    resolve_local_ai2thor_commit,
)

prepare_ai2thor_runtime()

from ai2thor.controller import Controller
import ai2thor.platform as thor_platform
from procthor.constants import SCHEMA
from procthor.generation import HouseGenerator
from procthor.generation.house import House
from procthor.generation.room_specs import PROCTHOR10K_ROOM_SPEC_SAMPLER
from procthor.utils.upgrade_house_version import HouseUpgradeManager
from procthor_optimizations import apply_procthor_algorithm_optimizations


apply_procthor_algorithm_optimizations()


ROOM_SPECS_BY_COUNT: Dict[int, List[Any]] = {}
ROOM_SPEC_IDS_BY_COUNT: Dict[int, List[str]] = {}
for _room_spec in PROCTHOR10K_ROOM_SPEC_SAMPLER.room_specs:
    room_count = len(_room_spec.room_type_map)
    ROOM_SPECS_BY_COUNT.setdefault(room_count, []).append(_room_spec)
    ROOM_SPEC_IDS_BY_COUNT.setdefault(room_count, []).append(_room_spec.room_spec_id)
AVAILABLE_ROOM_COUNTS = sorted(ROOM_SPECS_BY_COUNT)
AGENT_MASK_MARKERS = (
    "agent",
    "robot",
    "locobot",
    "fpscontroller",
    "capsule",
    "drone",
)


def normalize_platform_arg(platform):
    if platform is None or not isinstance(platform, str):
        return platform

    platform_cls = getattr(thor_platform, platform, None)
    if platform_cls is None:
        raise ValueError(f"Unknown AI2-THOR platform: {platform}")
    return platform_cls


def is_legacy_house_schema(data: Dict[str, Any]) -> bool:
    if isinstance(data.get("proceduralParameters", {}).get("ceilingMaterial"), str):
        return True

    for room in data.get("rooms", []):
        if isinstance(room.get("floorMaterial"), str):
            return True
        for ceiling in room.get("ceilings", []):
            if isinstance(ceiling.get("material"), str):
                return True
            if "materialProperties" in ceiling:
                return True

    for wall in data.get("walls", []):
        if isinstance(wall.get("material"), str):
            return True
        if "materialId" in wall or "materialProperties" in wall:
            return True

    for hole in data.get("doors", []) + data.get("windows", []):
        if "color" in hole or "assetOffset" in hole or "boundingBox" in hole:
            return True

    for obj in data.get("objects", []):
        if "materialProperties" in obj or "color" in obj:
            return True

    return False


def normalize_house_schema(house: House) -> House:
    data = copy.deepcopy(house.data)
    metadata = data.setdefault("metadata", {})
    needs_upgrade = metadata.get("schema") != SCHEMA or is_legacy_house_schema(data)
    if not needs_upgrade:
        return house

    if is_legacy_house_schema(data):
        metadata["schema"] = "0.0.1"

    upgraded = HouseUpgradeManager.upgrade_to(data, SCHEMA)
    return House(
        data=upgraded,
        rooms=house.rooms,
        interior_boundary=house.interior_boundary,
        room_spec=house.room_spec,
    )


def room_spec_ids_for_count(room_count: Optional[int]) -> List[str]:
    if room_count is None:
        return [room_spec.room_spec_id for room_spec in PROCTHOR10K_ROOM_SPEC_SAMPLER.room_specs]
    return ROOM_SPEC_IDS_BY_COUNT.get(room_count, [])


def choose_room_spec(room_count: Optional[int], seed: int):
    if room_count is None:
        return None

    candidates = ROOM_SPECS_BY_COUNT.get(room_count)
    if not candidates:
        raise ValueError(
            f"Unsupported room_count={room_count}. Available counts: {AVAILABLE_ROOM_COUNTS}"
        )

    return random.Random(seed).choice(candidates)


def create_generation_controller(
    *,
    runtime_name: Optional[str] = None,
    width: int = 384,
    height: int = 384,
    quality: str = "Low",
    platform: Optional[str] = None,
) -> Controller:
    platform_cls = normalize_platform_arg(platform)
    prepare_ai2thor_runtime(runtime_name)
    local_commit = resolve_local_ai2thor_commit(platform_cls)
    build_reference = (
        {"commit_id": local_commit}
        if local_commit is not None
        else {"branch": DEFAULT_BRANCH}
    )
    return Controller(
        **build_reference,
        scene="Procedural",
        width=width,
        height=height,
        quality=quality,
        agentMode="default",
        makeAgentsVisible=False,
        platform=platform_cls,
    )


def create_scene_controller(
    house_data: Dict[str, Any],
    *,
    runtime_name: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    quality: str = "Ultra",
    platform: Optional[str] = None,
) -> Controller:
    platform_cls = normalize_platform_arg(platform)
    prepare_ai2thor_runtime(runtime_name)
    local_commit = resolve_local_ai2thor_commit(platform_cls)
    build_reference = (
        {"commit_id": local_commit}
        if local_commit is not None
        else {"branch": DEFAULT_BRANCH}
    )
    return Controller(
        **build_reference,
        scene=house_data,
        width=width,
        height=height,
        quality=quality,
        agentMode="default",
        makeAgentsVisible=False,
        platform=platform_cls,
    )


def sample_house(
    *,
    split: str,
    start_seed: int,
    max_attempts: int,
    allow_warnings: bool,
    room_count: Optional[int] = None,
    door_count: Optional[int] = None,
    room_spec_id: Optional[str] = None,
    runtime_name: Optional[str] = None,
    platform: Optional[str] = None,
    quality: str = "Low",
    progress_callback: Optional[Callable[[str], None]] = None,
):
    last_error = None
    with create_generation_controller(
        runtime_name=runtime_name,
        platform=platform,
        quality=quality,
    ) as controller:
        for offset in range(max_attempts):
            seed = start_seed + offset
            if progress_callback is not None:
                progress_callback(
                    f"sampling seed={seed} attempt={offset + 1}/{max_attempts}"
                )

            try:
                generator = HouseGenerator(
                    split=split,
                    seed=seed,
                    room_spec=(
                        PROCTHOR10K_ROOM_SPEC_SAMPLER[room_spec_id]
                        if room_spec_id is not None
                        else choose_room_spec(room_count, seed)
                    ),
                    controller=controller,
                )
                house, _ = generator.sample()
                house = normalize_house_schema(house)

                openable_doors = [
                    door for door in house.data.get("doors", []) if door.get("openable")
                ]
                if not openable_doors:
                    raise RuntimeError("No openable room door was generated.")
                if door_count is not None and len(openable_doors) != door_count:
                    raise RuntimeError(
                        f"Expected {door_count} openable doors, found {len(openable_doors)}."
                    )

                warnings = house.validate(controller)
                fatal_warnings = {
                    key: value
                    for key, value in warnings.items()
                    if key in {"CreateHouse", "TeleportFull", "GetReachablePositions"}
                }
                if fatal_warnings:
                    raise RuntimeError(str(fatal_warnings))
                if warnings and not allow_warnings:
                    raise RuntimeError(str(warnings))

                return house, seed, warnings
            except Exception as exc:
                last_error = exc

    raise RuntimeError(
        f"unable to sample a valid furnished house in {max_attempts} attempts"
    ) from last_error


def load_house_into_controller(
    controller: Controller,
    house: House,
    *,
    validate_reachability: bool = True,
) -> Dict[str, str]:
    if validate_reachability:
        return house.validate(controller)

    warnings: Dict[str, str] = {}
    controller.reset(renderImage=False)

    event = controller.step(action="CreateHouse", house=house.data, renderImage=False)
    if not event:
        warnings["CreateHouse"] = "Failed to create house."
        house.data["metadata"]["warnings"] = warnings
        return warnings

    event = controller.step(
        action="TeleportFull",
        **house.data["metadata"]["agent"],
        renderImage=False,
    )
    if not event:
        warnings["TeleportFull"] = "Unable to teleport to starting position."
        house.data["metadata"]["warnings"] = warnings
        return warnings

    house.data["metadata"]["warnings"] = warnings
    return warnings


def sample_house_with_controller(
    *,
    split: str,
    start_seed: int,
    max_attempts: int,
    allow_warnings: bool,
    room_count: Optional[int] = None,
    door_count: Optional[int] = None,
    room_spec_id: Optional[str] = None,
    runtime_name: Optional[str] = None,
    platform: Optional[str] = None,
    quality: str = "Low",
    width: int = 384,
    height: int = 384,
    validate_reachability: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    last_error = None
    controller = create_generation_controller(
        runtime_name=runtime_name,
        width=width,
        height=height,
        platform=platform,
        quality=quality,
    )
    for offset in range(max_attempts):
        seed = start_seed + offset
        if progress_callback is not None:
            progress_callback(
                f"sampling seed={seed} attempt={offset + 1}/{max_attempts}"
            )

        try:
            generator = HouseGenerator(
                split=split,
                seed=seed,
                room_spec=(
                    PROCTHOR10K_ROOM_SPEC_SAMPLER[room_spec_id]
                    if room_spec_id is not None
                    else choose_room_spec(room_count, seed)
                ),
                controller=controller,
            )
            house, _ = generator.sample()
            house = normalize_house_schema(house)

            openable_doors = [
                door for door in house.data.get("doors", []) if door.get("openable")
            ]
            if not openable_doors:
                raise RuntimeError("No openable room door was generated.")
            if door_count is not None and len(openable_doors) != door_count:
                raise RuntimeError(
                    f"Expected {door_count} openable doors, found {len(openable_doors)}."
                )

            warnings = load_house_into_controller(
                controller,
                house,
                validate_reachability=validate_reachability,
            )
            fatal_warnings = {
                key: value
                for key, value in warnings.items()
                if key in {"CreateHouse", "TeleportFull", "GetReachablePositions"}
            }
            if fatal_warnings:
                raise RuntimeError(str(fatal_warnings))
            if warnings and not allow_warnings:
                raise RuntimeError(str(warnings))

            return house, seed, warnings, controller
        except Exception as exc:
            last_error = exc

    controller.stop()

    raise RuntimeError(
        f"unable to sample a valid furnished house in {max_attempts} attempts"
    ) from last_error


def openable_doors_from_house(house_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [door for door in house_data.get("doors", []) if door.get("openable")]


def openable_door_ids_from_house(house_data: Dict[str, Any]) -> List[str]:
    return [door["id"] for door in openable_doors_from_house(house_data)]


def get_live_openable_door_ids(
    controller: Controller, house_data: Dict[str, Any]
) -> List[str]:
    metadata_ids = [
        obj["objectId"]
        for obj in controller.last_event.metadata.get("objects", [])
        if obj.get("objectType") == "Door" and obj.get("openable")
    ]
    if metadata_ids:
        return metadata_ids
    return openable_door_ids_from_house(house_data)


def set_door_open(
    controller: Controller,
    door_id: str,
    is_open: bool,
) -> Dict[str, Any]:
    if is_open:
        event = controller.step(
            action="OpenObject",
            objectId=door_id,
            openness=1.0,
            forceAction=True,
            renderImage=False,
        )
    else:
        event = controller.step(
            action="CloseObject",
            objectId=door_id,
            forceAction=True,
            renderImage=False,
        )

    success = bool(event and event.metadata.get("lastActionSuccess", True))
    return {
        "success": success,
        "error": None
        if success
        else (
            event.metadata.get("errorMessage")
            or event.metadata.get("actionReturn")
            or f"Unable to set door {door_id} to {'open' if is_open else 'closed'}."
        ),
    }


def set_door_states(
    controller: Controller,
    house_data: Dict[str, Any],
    door_states: Dict[str, bool],
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for door_id in get_live_openable_door_ids(controller, house_data):
        if door_id in door_states:
            results[door_id] = set_door_open(controller, door_id, door_states[door_id])
    return results


def compute_house_bounds(house_data: Dict[str, Any]) -> Dict[str, Any]:
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for wall in house_data.get("walls", []):
        for point in wall["polygon"]:
            min_x = min(min_x, point["x"])
            min_y = min(min_y, point["y"])
            min_z = min(min_z, point["z"])
            max_x = max(max_x, point["x"])
            max_y = max(max_y, point["y"])
            max_z = max(max_z, point["z"])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    return {
        "min": {"x": min_x, "y": min_y, "z": min_z},
        "max": {"x": max_x, "y": max_y, "z": max_z},
        "center": {"x": center_x, "y": center_y, "z": center_z},
        "size": {
            "x": max_x - min_x,
            "y": max_y - min_y,
            "z": max_z - min_z,
        },
    }


def get_topdown_camera(controller: Controller) -> Dict[str, Any]:
    event = controller.step(action="GetMapViewCameraProperties", renderImage=False)
    camera = copy.deepcopy(event.metadata["actionReturn"])
    camera["orthographic"] = True
    camera["orthographicSize"] *= 1.05
    return camera


def get_oblique_camera(
    house_data: Dict[str, Any],
    *,
    yaw_deg: float,
    pitch_deg: float = 45.0,
    zoom: float = 1.0,
) -> Dict[str, Any]:
    bounds = compute_house_bounds(house_data)
    span_x = bounds["size"]["x"]
    span_y = bounds["size"]["y"]
    span_z = bounds["size"]["z"]
    center = bounds["center"]

    horizontal_span = max(span_x, span_z)
    distance = max(horizontal_span * 1.3 * zoom, 3.0)
    target_y = bounds["min"]["y"] + span_y * 0.4

    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    position = {
        "x": center["x"] - distance * math.cos(pitch_rad) * math.sin(yaw_rad),
        "y": target_y + distance * math.sin(pitch_rad),
        "z": center["z"] - distance * math.cos(pitch_rad) * math.cos(yaw_rad),
    }

    return {
        "position": position,
        "rotation": {"x": pitch_deg, "y": yaw_deg, "z": 0},
        "fieldOfView": 60,
        "farClippingPlane": 80,
        "orthographic": False,
    }


def is_agent_like_label(label: str) -> bool:
    lowered = label.lower()
    return any(marker in lowered for marker in AGENT_MASK_MARKERS)


def collect_agent_mask(event) -> Optional[np.ndarray]:
    merged_mask = None

    if getattr(event, "third_party_instance_masks", None):
        for key in event.third_party_instance_masks[0]:
            if not is_agent_like_label(key):
                continue
            mask = event.third_party_instance_masks[0][key]
            merged_mask = mask if merged_mask is None else np.logical_or(merged_mask, mask)

    if merged_mask is None and getattr(event, "third_party_class_masks", None):
        for key in event.third_party_class_masks[0]:
            if not is_agent_like_label(key):
                continue
            mask = event.third_party_class_masks[0][key]
            merged_mask = mask if merged_mask is None else np.logical_or(merged_mask, mask)

    if merged_mask is None or not merged_mask.any():
        return None
    return merged_mask


def inpaint_masked_region(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return frame

    output = frame.copy()
    rgb = output[:, :, :3].astype(np.float32, copy=True)
    ys, xs = np.where(mask)
    pad = 12
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)

    region = rgb[y0:y1, x0:x1]
    region_mask = mask[y0:y1, x0:x1].copy()
    known = ~region_mask
    if not known.any():
        return frame

    work = region.copy()
    padded_shape = ((1, 1), (1, 1))
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    max_iterations = max(region.shape[0], region.shape[1])
    for _ in range(max_iterations):
        unresolved = ~known
        if not unresolved.any():
            break

        padded_work = np.pad(work, padded_shape + ((0, 0),), mode="edge")
        padded_known = np.pad(known, padded_shape, mode="constant", constant_values=False)
        accum = np.zeros_like(work)
        counts = np.zeros(region_mask.shape, dtype=np.float32)

        for dy, dx in neighbor_offsets:
            y_start = 1 + dy
            x_start = 1 + dx
            neighbor_known = padded_known[y_start:y_start + region.shape[0], x_start:x_start + region.shape[1]]
            neighbor_values = padded_work[y_start:y_start + region.shape[0], x_start:x_start + region.shape[1]]
            accum += neighbor_values * neighbor_known[:, :, None]
            counts += neighbor_known.astype(np.float32)

        fillable = unresolved & (counts > 0)
        if not fillable.any():
            break
        work[fillable] = accum[fillable] / counts[fillable, None]
        known[fillable] = True

    unresolved = ~known
    if unresolved.any():
        mean_color = work[known].mean(axis=0)
        work[unresolved] = mean_color

    output[y0:y1, x0:x1, :3] = np.clip(work, 0, 255).astype(output.dtype)
    return output


def remove_agent_from_frame(event) -> np.ndarray:
    frame = event.third_party_camera_frames[0]
    mask = collect_agent_mask(event)
    if mask is None:
        return frame
    return inpaint_masked_region(frame, mask)


def render_camera_frame(controller: Controller, camera: Dict[str, Any]):
    if controller.last_event.metadata.get("thirdPartyCameras"):
        update_camera = dict(camera)
        update_camera.pop("antiAliasing", None)
        action = dict(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            renderInstanceSegmentation=True,
            **update_camera,
        )
    else:
        action = dict(
            action="AddThirdPartyCamera",
            skyboxColor="black",
            renderInstanceSegmentation=True,
            **camera,
        )
    event = controller.step(**action)
    if not event.metadata.get("lastActionSuccess", True) or not event.third_party_camera_frames:
        action.pop("renderInstanceSegmentation", None)
        event = controller.step(**action)
    if not event.metadata.get("lastActionSuccess", True) or not event.third_party_camera_frames:
        raise RuntimeError(
            event.metadata.get("errorMessage")
            or "Unable to render third-party camera frame."
        )
    return remove_agent_from_frame(event)


def prepare_controller_for_export(
    controller: Controller,
    *,
    width: int,
    height: int,
    quality: str,
) -> None:
    if (
        controller.last_event.screen_width != width
        or controller.last_event.screen_height != height
    ):
        controller.step(
            action="ChangeResolution",
            x=width,
            y=height,
            raise_for_failure=True,
        )
        controller.width = width
        controller.height = height

    if controller.quality != quality:
        controller.step(
            action="ChangeQuality",
            quality=quality,
            renderImage=False,
            raise_for_failure=True,
        )
        controller.quality = quality


def render_named_views(
    controller: Controller,
    house_data: Dict[str, Any],
    *,
    front_yaw: Optional[float] = None,
) -> Dict[str, Any]:
    if front_yaw is None:
        front_yaw = house_data["metadata"]["agent"]["rotation"]["y"]

    return {
        "topdown": render_camera_frame(controller, get_topdown_camera(controller)),
        "front45": render_camera_frame(
            controller,
            get_oblique_camera(house_data, yaw_deg=front_yaw),
        ),
        "right45": render_camera_frame(
            controller,
            get_oblique_camera(house_data, yaw_deg=(front_yaw + 90) % 360),
        ),
        "rear45": render_camera_frame(
            controller,
            get_oblique_camera(house_data, yaw_deg=(front_yaw + 180) % 360),
        ),
        "left45": render_camera_frame(
            controller,
            get_oblique_camera(house_data, yaw_deg=(front_yaw + 270) % 360),
        ),
    }
