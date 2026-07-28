import argparse
import json
import time
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from statistics import mean
from typing import Any, Dict, List, Optional

from house_pipeline import (
    choose_room_spec,
    create_generation_controller,
    load_house_into_controller,
    normalize_house_schema,
)
from procthor.generation import HouseGenerator


def wrap_generation_functions(base_gfs, stage_stats: Dict[str, Dict[str, Any]]):
    wrapped = {}
    for name in base_gfs.__attrs_attrs__:
        attr_name = name.name
        fn = getattr(base_gfs, attr_name)

        def make_wrapper(stage_name, inner_fn):
            def wrapper(*args, **kwargs):
                partial_house = kwargs.get("partial_house")
                pre_count = None if partial_house is None else len(partial_house.objects)
                start = time.perf_counter()
                result = inner_fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                post_count = None if partial_house is None else len(partial_house.objects)

                stat = stage_stats.setdefault(
                    stage_name,
                    {"seconds": 0.0, "calls": 0, "objects_added": 0},
                )
                stat["seconds"] += elapsed
                stat["calls"] += 1
                if pre_count is not None and post_count is not None:
                    stat["objects_added"] += max(0, post_count - pre_count)
                return result

            return wrapper

        wrapped[attr_name] = make_wrapper(attr_name, fn)

    return type(base_gfs)(**wrapped)


class DeepProfiler:
    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"seconds": 0.0, "calls": 0}
        )
        self._restore_stack = []

    def record(
        self, name: str, elapsed: float, *, result: Any = None, original: Any = None
    ) -> None:
        stat = self.stats[name]
        stat["seconds"] += elapsed
        stat["calls"] += 1

        if name == "sample_and_add_floor_asset":
            if result is None:
                stat["returned_none"] = stat.get("returned_none", 0) + 1
            elif "objects" in result:
                stat["returned_asset_group"] = stat.get("returned_asset_group", 0) + 1
            else:
                stat["returned_asset"] = stat.get("returned_asset", 0) + 1
        elif name == "place_asset_group":
            if result is None:
                stat["returned_none"] = stat.get("returned_none", 0) + 1
            else:
                stat["returned_group"] = stat.get("returned_group", 0) + 1
        elif name == "get_intersecting_objects":
            if isinstance(result, tuple) and result:
                if bool(result[0]):
                    stat["collisions_true"] = stat.get("collisions_true", 0) + 1
                else:
                    stat["collisions_false"] = stat.get("collisions_false", 0) + 1
        elif name == "get_spawnable_asset_group_info" and hasattr(original, "cache_info"):
            cache_info = original.cache_info()
            stat["cache_hits"] = cache_info.hits
            stat["cache_misses"] = cache_info.misses
            stat["cache_currsize"] = cache_info.currsize

    def wrap(self, owner: Any, attr_name: str, stat_name: Optional[str] = None) -> None:
        original = getattr(owner, attr_name)
        name = stat_name or attr_name

        @wraps(original)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.record(name, elapsed, result=result, original=original)
            return result

        self._restore_stack.append((owner, attr_name, original))
        setattr(owner, attr_name, wrapped)

    def install(self) -> None:
        import procthor.generation.asset_groups as asset_groups_module
        import procthor.generation.objects as objects_module

        # `default_add_floor_objects` calls the locally imported symbol in objects.py.
        self.wrap(
            objects_module,
            "get_spawnable_asset_group_info",
            stat_name="get_spawnable_asset_group_info",
        )
        self.wrap(
            objects_module,
            "sample_and_add_floor_asset",
            stat_name="sample_and_add_floor_asset",
        )
        self.wrap(
            objects_module.ProceduralRoom,
            "sample_next_rectangle",
            stat_name="sample_next_rectangle",
        )
        self.wrap(
            objects_module.ProceduralRoom,
            "sample_anchor_location",
            stat_name="sample_anchor_location",
        )
        self.wrap(
            objects_module.ProceduralRoom,
            "place_asset",
            stat_name="place_asset",
        )
        self.wrap(
            objects_module.ProceduralRoom,
            "place_asset_group",
            stat_name="place_asset_group",
        )
        self.wrap(
            asset_groups_module.AssetGroupGenerator,
            "_set_dimensions",
            stat_name="asset_group_set_dimensions",
        )
        self.wrap(
            asset_groups_module.AssetGroupGenerator,
            "sample_object_placement",
            stat_name="asset_group_sample_object_placement",
        )
        self.wrap(
            asset_groups_module.AssetGroupGenerator,
            "get_intersecting_objects",
            stat_name="asset_group_get_intersecting_objects",
        )

    def uninstall(self) -> None:
        while self._restore_stack:
            owner, attr_name, original = self._restore_stack.pop()
            setattr(owner, attr_name, original)


class StepTimer:
    def __init__(self, controller) -> None:
        self.controller = controller
        self.original_step = controller.step
        self.phase = "sampling"
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        )

    def install(self) -> None:
        def wrapped_step(*args, **kwargs):
            action = kwargs.get("action")
            if action is None and args:
                action = args[0]
            if not isinstance(action, str):
                action = "<unknown>"

            start = time.perf_counter()
            try:
                return self.original_step(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                stat = self.stats[self.phase][action]
                stat["seconds"] += elapsed
                stat["calls"] += 1

        self.controller.step = wrapped_step

    def uninstall(self) -> None:
        self.controller.step = self.original_step

    def set_phase(self, phase: str) -> None:
        self.phase = phase


def sort_nested_stats(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        key: stats[key]
        for key in sorted(stats, key=lambda item: stats[item]["seconds"], reverse=True)
    }


def sort_action_stats(stats: Dict[str, Dict[str, Dict[str, float]]]):
    sorted_phases = {}
    for phase, phase_stats in stats.items():
        sorted_phases[phase] = {
            action: phase_stats[action]
            for action in sorted(
                phase_stats,
                key=lambda item: phase_stats[item]["seconds"],
                reverse=True,
            )
        }
    return sorted_phases


def sort_generic_stats(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        key: stats[key]
        for key in sorted(stats, key=lambda item: stats[item]["seconds"], reverse=True)
    }


def profile_once(
    *,
    split: str,
    seed: int,
    room_count: Optional[int],
    platform: Optional[str],
    quality: str,
    width: int,
    height: int,
    validate_reachability: bool,
    runtime_name: str,
):
    stage_stats: Dict[str, Dict[str, Any]] = {}
    deep_profiler = DeepProfiler()
    with create_generation_controller(
        runtime_name=runtime_name,
        width=width,
        height=height,
        quality=quality,
        platform=platform,
    ) as controller:
        timer = StepTimer(controller)
        timer.install()
        deep_profiler.install()
        try:
            room_spec = choose_room_spec(room_count, seed)
            probe = HouseGenerator(
                split=split,
                seed=seed,
                room_spec=room_spec,
                controller=controller,
            )
            gfs = wrap_generation_functions(probe.generation_functions, stage_stats)
            generator = HouseGenerator(
                split=split,
                seed=seed,
                room_spec=room_spec,
                controller=controller,
                generation_functions=gfs,
            )

            sample_start = time.perf_counter()
            house, _ = generator.sample()
            sample_seconds = time.perf_counter() - sample_start
            house = normalize_house_schema(house)

            timer.set_phase("load_validate")
            load_start = time.perf_counter()
            warnings = load_house_into_controller(
                controller,
                house,
                validate_reachability=validate_reachability,
            )
            load_seconds = time.perf_counter() - load_start
        finally:
            deep_profiler.uninstall()
            timer.uninstall()

    return {
        "seed": seed,
        "room_count": len(house.data.get("rooms", [])),
        "openable_doors": len(
            [door for door in house.data.get("doors", []) if door.get("openable")]
        ),
        "num_objects": len(house.data.get("objects", [])),
        "sample_seconds": sample_seconds,
        "load_validate_seconds": load_seconds,
        "total_seconds": sample_seconds + load_seconds,
        "warnings": warnings,
        "stage_stats": sort_nested_stats(stage_stats),
        "deep_stats": sort_generic_stats(deep_profiler.stats),
        "controller_actions": sort_action_stats(timer.stats),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    averaged_stages: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"seconds": 0.0, "calls": 0.0, "objects_added": 0.0}
    )
    averaged_actions: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"seconds": 0.0, "calls": 0.0})
    )
    averaged_deep: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"seconds": 0.0, "calls": 0.0}
    )

    for run in runs:
        for stage, values in run["stage_stats"].items():
            averaged_stages[stage]["seconds"] += values["seconds"]
            averaged_stages[stage]["calls"] += values["calls"]
            averaged_stages[stage]["objects_added"] += values["objects_added"]

        for phase, phase_stats in run["controller_actions"].items():
            for action, values in phase_stats.items():
                averaged_actions[phase][action]["seconds"] += values["seconds"]
                averaged_actions[phase][action]["calls"] += values["calls"]

        for name, values in run.get("deep_stats", {}).items():
            for key, value in values.items():
                averaged_deep[name][key] = averaged_deep[name].get(key, 0.0) + value

    num_runs = len(runs)
    for values in averaged_stages.values():
        values["seconds"] /= num_runs
        values["calls"] /= num_runs
        values["objects_added"] /= num_runs

    for phase_stats in averaged_actions.values():
        for values in phase_stats.values():
            values["seconds"] /= num_runs
            values["calls"] /= num_runs

    for values in averaged_deep.values():
        for key in list(values.keys()):
            values[key] /= num_runs

    return {
        "num_runs": num_runs,
        "avg_sample_seconds": mean(run["sample_seconds"] for run in runs),
        "avg_load_validate_seconds": mean(run["load_validate_seconds"] for run in runs),
        "avg_total_seconds": mean(run["total_seconds"] for run in runs),
        "avg_objects": mean(run["num_objects"] for run in runs),
        "avg_openable_doors": mean(run["openable_doors"] for run in runs),
        "stage_stats": sort_nested_stats(averaged_stages),
        "deep_stats": sort_generic_stats(averaged_deep),
        "controller_actions": sort_action_stats(averaged_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ProcTHOR generation stages.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-step", type=int, default=12)
    parser.add_argument("--room-count", type=int, default=3)
    parser.add_argument("--platform", default="CloudRendering")
    parser.add_argument("--quality", default="Medium")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--validate-reachability", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    runs = []
    for run_index in range(args.runs):
        seed = args.seed + run_index * args.seed_step
        runtime_name = f"profile_run_{seed}"
        run = profile_once(
            split=args.split,
            seed=seed,
            room_count=args.room_count,
            platform=args.platform,
            quality=args.quality,
            width=args.width,
            height=args.height,
            validate_reachability=args.validate_reachability,
            runtime_name=runtime_name,
        )
        runs.append(run)

    report = {
        "config": {
            "split": args.split,
            "seed": args.seed,
            "runs": args.runs,
            "seed_step": args.seed_step,
            "room_count": args.room_count,
            "platform": args.platform,
            "quality": args.quality,
            "width": args.width,
            "height": args.height,
            "validate_reachability": args.validate_reachability,
        },
        "summary": summarize_runs(runs),
        "runs": runs,
    }

    output = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(output)
    print(output)


if __name__ == "__main__":
    main()
