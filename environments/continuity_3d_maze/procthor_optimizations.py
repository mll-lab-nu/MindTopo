import copy
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from shapely.geometry import Point

try:
    from shapely import contains_xy as _contains_xy
except ImportError:  # pragma: no cover - depends on installed shapely version
    _contains_xy = None


_OPTIMIZATIONS_APPLIED = False


def _polygon_contains_xy(polygon, x: float, z: float) -> bool:
    if _contains_xy is not None:
        return bool(_contains_xy(polygon, x, z))
    return bool(polygon.contains(Point(x, z)))


def _clear_orthogonal_polygon_caches(self) -> None:
    self._maze_neighboring_rectangles = None
    self._maze_all_rectangles = None


def _optimized_set_attributes(original_set_attributes):
    def wrapper(self) -> None:
        _clear_orthogonal_polygon_caches(self)
        original_set_attributes(self)

    return wrapper


def _enumerate_neighboring_rectangles(self) -> Set[Tuple[float, float, float, float]]:
    cached = getattr(self, "_maze_neighboring_rectangles", None)
    if cached is not None:
        return cached.copy()

    rectangles: Set[Tuple[float, float, float, float]] = set()
    unique_xs = self.unique_xs
    unique_zs = self.unique_zs
    polygon = self.polygon

    for x0, x1 in zip(unique_xs, unique_xs[1:]):
        mid_x = (x0 + x1) / 2
        for z0, z1 in zip(unique_zs, unique_zs[1:]):
            mid_z = (z0 + z1) / 2
            if _polygon_contains_xy(polygon, mid_x, mid_z):
                rectangles.add((x0, z0, x1, z1))

    self._maze_neighboring_rectangles = rectangles
    return rectangles.copy()


def _enumerate_all_rectangles(self) -> Set[Tuple[float, float, float, float]]:
    cached = getattr(self, "_maze_all_rectangles", None)
    if cached is not None:
        return cached.copy()

    xs = self.unique_xs
    zs = self.unique_zs
    num_cols = max(0, len(xs) - 1)
    num_rows = max(0, len(zs) - 1)
    if num_cols == 0 or num_rows == 0:
        self._maze_all_rectangles = set()
        return set()

    occupied = [[False] * num_cols for _ in range(num_rows)]
    neighboring_rectangles: Set[Tuple[float, float, float, float]] = set()
    polygon = self.polygon
    for col, (x0, x1) in enumerate(zip(xs, xs[1:])):
        mid_x = (x0 + x1) / 2
        for row, (z0, z1) in enumerate(zip(zs, zs[1:])):
            mid_z = (z0 + z1) / 2
            is_occupied = _polygon_contains_xy(polygon, mid_x, mid_z)
            occupied[row][col] = is_occupied
            if is_occupied:
                neighboring_rectangles.add((x0, z0, x1, z1))

    rectangles: Set[Tuple[float, float, float, float]] = set()
    for top in range(num_rows):
        valid_columns = [True] * num_cols
        z0 = zs[top]
        for bottom in range(top, num_rows):
            row = occupied[bottom]
            for col in range(num_cols):
                valid_columns[col] = valid_columns[col] and row[col]

            col = 0
            z1 = zs[bottom + 1]
            while col < num_cols:
                while col < num_cols and not valid_columns[col]:
                    col += 1
                run_start = col
                while col < num_cols and valid_columns[col]:
                    col += 1
                run_end = col
                if run_start >= run_end:
                    continue

                for left in range(run_start, run_end):
                    x0 = xs[left]
                    for right in range(left + 1, run_end + 1):
                        rectangles.add((x0, z0, xs[right], z1))

    self._maze_all_rectangles = rectangles
    self._maze_neighboring_rectangles = neighboring_rectangles
    return rectangles.copy()


def _asset_candidates_for_type(
    pt_db,
    split: str,
    object_type: str,
    cache: Dict[Tuple[str, str], object],
):
    key = (split, object_type)
    if key not in cache:
        asset_candidates = pt_db.ASSETS_DF[
            (pt_db.ASSETS_DF["objectType"] == object_type)
            & pt_db.ASSETS_DF["split"].isin([split, None])
        ]
        if object_type == "HousePlant":
            from procthor.generation.small_objects import HOUSE_PLANT_MAX_HEIGHT

            asset_candidates = asset_candidates[
                asset_candidates["ySize"] < HOUSE_PLANT_MAX_HEIGHT
            ]
        cache[key] = asset_candidates
    return cache[key]


def _build_receptacle_object_ids(
    objects: Sequence[Dict[str, object]],
    receptacle_ids: Iterable[str],
) -> Dict[str, List[str]]:
    object_ids = [obj["objectId"] for obj in objects]
    return {
        receptacle_id: [
            object_id for object_id in object_ids if object_id.startswith(receptacle_id)
        ]
        for receptacle_id in receptacle_ids
    }


def _optimized_add_small_objects(
    partial_house,
    controller,
    pt_db,
    split,
    rooms,
    max_object_types_per_room: int = 10000,
) -> None:
    from procthor.constants import FLOOR_Y, OPENNESS_RANDOMIZATIONS
    from procthor.generation.objects import ProceduralRoom, sample_openness
    from procthor.generation.small_objects import (
        CHILD_BIAS,
        FLOOR_OBJECTS_TO_DROP,
        MAX_OF_TYPE_ON_RECEPTACLE,
        OBJECTS_TO_DROP,
        PARENT_BIAS,
        randomize_bias,
    )
    from procthor.utils.types import Object, Vector3

    controller.reset()
    controller.step(action="ResetObjectFilter")
    event = controller.step(
        action="CreateHouse", house=partial_house.to_house_dict(), renderImage=False
    )
    assert event, "Unable to CreateHouse!"

    objects = [
        obj
        for obj in event.metadata["objects"]
        if not any(
            obj["objectId"].startswith(prefix)
            for prefix in ["wall|", "room|", "Floor", "door|", "window|"]
        )
    ]
    objects_per_room = defaultdict(list)
    for obj in objects:
        object_id = obj["objectId"]
        room_id = int(object_id[: object_id.find("|")])
        objects_per_room[room_id].append(obj)
    objects_per_room = dict(objects_per_room)
    receptacles_per_room = {
        room_id: [
            obj
            for obj in room_objects
            if obj["objectType"] in pt_db.OBJECTS_IN_RECEPTACLES
        ]
        for room_id, room_objects in objects_per_room.items()
    }
    object_types_in_rooms = {
        room_id: set(obj["objectType"] for obj in room_objects)
        for room_id, room_objects in objects_per_room.items()
    }

    receptacle_object_ids = _build_receptacle_object_ids(
        objects,
        (
            receptacle_id
            for room_receptacles in receptacles_per_room.values()
            for receptacle_id in [obj["objectId"] for obj in room_receptacles]
        ),
    )
    asset_candidates_cache: Dict[Tuple[str, str], object] = {}
    objects_in_house = {obj["id"]: obj for obj in partial_house.objects}

    house_bias = randomize_bias()
    num_placed_object_instances = 0
    for room_id, room in rooms.items():
        if room_id not in receptacles_per_room:
            continue
        receptacles_in_room = receptacles_per_room[room_id]
        room_type = room.room_type
        spawnable_objects = []
        for receptacle in receptacles_in_room:
            objects_in_receptacle = pt_db.OBJECTS_IN_RECEPTACLES[
                receptacle["objectType"]
            ]
            for object_type, data in objects_in_receptacle.items():
                room_weight = pt_db.PLACEMENT_ANNOTATIONS.loc[object_type][
                    f"in{room_type}s"
                ]
                if room_weight == 0:
                    continue
                spawnable_objects.append(
                    {
                        "receptacleId": receptacle["objectId"],
                        "receptacleType": receptacle["objectType"],
                        "childObjectType": object_type,
                        "childRoomWeight": room_weight,
                        "pSpawn": data["p"],
                    }
                )

        filtered_spawnable_groups = [
            group
            for group in spawnable_objects
            if random.random()
            <= (
                group["pSpawn"]
                + PARENT_BIAS[group["receptacleType"]]
                + CHILD_BIAS[group["childObjectType"]]
                + house_bias
            )
        ]
        random.shuffle(filtered_spawnable_groups)
        objects_types_placed_in_room = set()
        for group in filtered_spawnable_groups:
            if len(objects_types_placed_in_room) >= max_object_types_per_room:
                break

            num_of_type = 1
            while random.random() <= group["pSpawn"]:
                num_of_type += 1
                if num_of_type >= MAX_OF_TYPE_ON_RECEPTACLE:
                    break

            for _ in range(num_of_type):
                if (
                    group["childObjectType"] in object_types_in_rooms[room_id]
                    and not pt_db.PLACEMENT_ANNOTATIONS.loc[group["childObjectType"]][
                        "multiplePerRoom"
                    ]
                ):
                    break

                asset_candidates = _asset_candidates_for_type(
                    pt_db=pt_db,
                    split=split,
                    object_type=group["childObjectType"],
                    cache=asset_candidates_cache,
                )
                chosen_asset_id = asset_candidates.sample()["assetId"].iloc[0]
                generated_object_id = f"small|{room_id}|{num_placed_object_instances}"

                event = controller.step(
                    action="SpawnAsset",
                    assetId=chosen_asset_id,
                    generatedId=generated_object_id,
                    position=Vector3(x=0, y=FLOOR_Y - 20, z=0),
                    renderImage=False,
                )
                assert event, (
                    f"SpawnAsset failed for {chosen_asset_id} "
                    f"with {event.metadata['actionReturn']}!"
                )

                obj_type = pt_db.ASSET_ID_DATABASE[chosen_asset_id]["objectType"]
                openness = None
                if (
                    obj_type in OPENNESS_RANDOMIZATIONS
                    and "CanOpen"
                    in pt_db.ASSET_ID_DATABASE[chosen_asset_id]["secondaryProperties"]
                ):
                    openness = sample_openness(obj_type)
                    controller.step(
                        action="OpenObject",
                        objectId=generated_object_id,
                        openness=openness,
                        forceAction=True,
                        raise_for_failure=True,
                        renderImage=False,
                    )

                event = controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random.randint(0, 1_000_000_000),
                    objectIds=[generated_object_id],
                    receptacleObjectIds=receptacle_object_ids[group["receptacleId"]],
                    forceVisible=False,
                    allowFloor=False,
                    renderImage=False,
                    allowMoveable=True,
                )
                obj = next(
                    obj
                    for obj in event.metadata["objects"]
                    if obj["objectId"] == generated_object_id
                )
                center_position = obj["axisAlignedBoundingBox"]["center"].copy()

                if event and center_position["y"] > FLOOR_Y:
                    if obj["breakable"]:
                        center_position["y"] += 0.05

                    states = {}
                    if openness is not None:
                        states["openness"] = openness

                    house_data_receptacle = group["receptacleId"]
                    if "___" in group["receptacleId"]:
                        house_data_receptacle = group["receptacleId"][
                            : group["receptacleId"].find("___")
                        ]
                    if "children" not in objects_in_house[house_data_receptacle]:
                        objects_in_house[house_data_receptacle]["children"] = []

                    objects_in_house[house_data_receptacle]["children"].append(
                        Object(
                            id=generated_object_id,
                            assetId=chosen_asset_id,
                            rotation=obj["rotation"],
                            position=center_position,
                            kinematic=bool(
                                pt_db.PLACEMENT_ANNOTATIONS.loc[
                                    group["childObjectType"]
                                ]["isKinematic"]
                            ),
                            **states,
                        )
                    )

                    num_placed_object_instances += 1
                    objects_types_placed_in_room.add(obj_type)
                    object_types_in_rooms[room_id].add(group["childObjectType"])
                else:
                    controller.step(
                        action="DisableObject",
                        objectId=generated_object_id,
                        renderImage=False,
                    )

    def _set_drop_heights(objects: List[Object], obj_types):
        for obj in objects:
            obj_type = pt_db.ASSET_ID_DATABASE[obj["assetId"]]["objectType"]
            if obj_type in obj_types and random.random() < obj_types[obj_type]["p"]:
                obj["position"]["y"] = 3
                obj["rotation"]["x"] = random.random() * 2 + 3
                obj["rotation"]["y"] = random.random() * 360
                changed_ids.add(obj["id"])
            if "children" in obj:
                _set_drop_heights(obj["children"], obj_types=OBJECTS_TO_DROP)

    def _save_new_heights(objects: List[Object]):
        for obj in objects:
            if obj["id"] in changed_ids:
                thor_obj = next(
                    o for o in event.metadata["objects"] if o["objectId"] == obj["id"]
                )
                obj["position"] = thor_obj["axisAlignedBoundingBox"]["center"].copy()
                obj["rotation"] = thor_obj["rotation"].copy()
            if "children" in obj:
                _save_new_heights(obj["children"])

    changed_ids = set()
    orig_objects = copy.deepcopy(partial_house.objects)
    _set_drop_heights(objects=partial_house.objects, obj_types=FLOOR_OBJECTS_TO_DROP)
    if changed_ids:
        controller.reset()
        event = controller.step(
            action="CreateHouse", house=partial_house.to_house_dict(), renderImage=False
        )
        assert event, "Unable to CreateHouse!"

        last_objs = [
            obj for obj in event.metadata["objects"] if obj["objectId"] in changed_ids
        ]
        i = 0
        failed = False
        while True:
            i += 1
            if i > 1000:
                failed = True
                print("Objects not settling!")
                break

            event = controller.step(
                action="AdvancePhysicsStep",
                timeStep=0.01,
                allowAutoSimulation=True,
                renderImage=False,
            )
            objs = [
                obj
                for obj in event.metadata["objects"]
                if obj["objectId"] in changed_ids
            ]

            if all(
                all(
                    abs(obj["position"][key] - last_obj["position"][key]) < 1e-3
                    and abs(obj["rotation"][key] - last_obj["rotation"][key]) < 1e-3
                    for key in ["x", "y", "z"]
                )
                for obj, last_obj in zip(objs, last_objs)
            ):
                break
            last_objs = objs

        if failed:
            partial_house.objects = orig_objects
        else:
            _save_new_heights(objects=partial_house.objects)


def apply_procthor_algorithm_optimizations() -> None:
    global _OPTIMIZATIONS_APPLIED
    if _OPTIMIZATIONS_APPLIED:
        return

    import procthor.generation as generation_module
    import procthor.generation.objects as objects_module
    import procthor.generation.small_objects as small_objects_module

    objects_module.OrthogonalPolygon._set_attributes = _optimized_set_attributes(
        objects_module.OrthogonalPolygon._set_attributes
    )
    objects_module.OrthogonalPolygon.get_neighboring_rectangles = (
        _enumerate_neighboring_rectangles
    )
    objects_module.OrthogonalPolygon.get_all_rectangles = _enumerate_all_rectangles

    small_objects_module.default_add_small_objects = _optimized_add_small_objects
    generation_module.default_add_small_objects = _optimized_add_small_objects

    _OPTIMIZATIONS_APPLIED = True
