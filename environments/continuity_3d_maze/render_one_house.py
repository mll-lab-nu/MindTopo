import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from house_pipeline import (
    openable_doors_from_house,
    prepare_controller_for_export,
    render_named_views,
    sample_house_with_controller,
    set_door_states,
)

EXPORT_RENDER_WIDTH = 1536
EXPORT_RENDER_HEIGHT = 1536
EXPORT_RENDER_QUALITY = "Ultra"


def save_image(arr, path: Path) -> None:
    Image.fromarray(arr).save(path)


def build_progress_bar(current: int, total: int, width: int = 28) -> str:
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


def print_progress(current: int, total: int, status: str, *, done: bool = False) -> None:
    line = f"{build_progress_bar(current, total)} {status}"
    end = "\n" if done else "\r"
    sys.stdout.write(line.ljust(120) + end)
    sys.stdout.flush()


def save_house_artifacts(
    *,
    house,
    controller,
    out_dir: Path,
    index: int,
    seed: int,
    warnings,
    door_state_mode: str,
):
    prefix = f"house_{index:03d}"
    json_path = out_dir / f"{prefix}.json"
    info_path = out_dir / f"{prefix}_info.json"
    topdown_path = out_dir / f"{prefix}_topdown.png"
    front45_path = out_dir / f"{prefix}_front45.png"
    right45_path = out_dir / f"{prefix}_right45.png"
    rear45_path = out_dir / f"{prefix}_rear45.png"
    left45_path = out_dir / f"{prefix}_left45.png"

    openable_doors = openable_doors_from_house(house.data)
    initial_door_states = {
        door["id"]: bool(door.get("openness", 0) >= 0.5) for door in openable_doors
    }

    target_door_states = dict(initial_door_states)
    if door_state_mode == "open":
        target_door_states = {door_id: True for door_id in target_door_states}
    elif door_state_mode == "closed":
        target_door_states = {door_id: False for door_id in target_door_states}
    elif door_state_mode == "random":
        rng = random.Random(seed)
        target_door_states = {
            door_id: bool(rng.random() < 0.5)
            for door_id in sorted(target_door_states)
        }

    for door in house.data.get("doors", []):
        if door.get("openable") and door["id"] in target_door_states:
            door["openness"] = 1.0 if target_door_states[door["id"]] else 0.0

    house.to_json(str(json_path))
    prepare_controller_for_export(
        controller,
        width=EXPORT_RENDER_WIDTH,
        height=EXPORT_RENDER_HEIGHT,
        quality=EXPORT_RENDER_QUALITY,
    )
    controller.step(action="TeleportFull", **house.data["metadata"]["agent"])
    door_results = set_door_states(controller, house.data, target_door_states)
    views = render_named_views(controller, house.data)

    save_image(views["topdown"], topdown_path)
    save_image(views["front45"], front45_path)
    save_image(views["right45"], right45_path)
    save_image(views["rear45"], rear45_path)
    save_image(views["left45"], left45_path)

    info = {
        "index": index,
        "seed": seed,
        "door_state_mode": door_state_mode,
        "num_doors": len(house.data.get("doors", [])),
        "num_openable_doors": len(openable_doors),
        "num_rooms": len(house.data.get("rooms", [])),
        "num_objects": len(house.data.get("objects", [])),
        "initial_door_states": initial_door_states,
        "final_door_states": target_door_states,
        "door_action_results": door_results,
        "warnings": warnings,
        "files": {
            "house_json": json_path.name,
            "topdown_view": topdown_path.name,
            "front45_view": front45_path.name,
            "right45_view": right45_path.name,
            "rear45_view": rear45_path.name,
            "left45_view": left45_path.name,
        },
    }
    info_path.write_text(json.dumps(info, indent=2))
    return info


def render_house_job(job):
    out_dir = Path(job["output_dir"])
    house, used_seed, warnings, controller = sample_house_with_controller(
        split=job["split"],
        start_seed=job["seed_start"],
        max_attempts=job["max_attempts"],
        allow_warnings=job["allow_warnings"],
        room_count=job["room_count"],
        door_count=job["door_count"],
        runtime_name=f"sample_{job['index']:03d}_{os.getpid()}",
        width=1024,
        height=1024,
    )
    try:
        return save_house_artifacts(
            house=house,
            controller=controller,
            out_dir=out_dir,
            index=job["index"],
            seed=used_seed,
            warnings=warnings,
            door_state_mode=job["door_state"],
        )
    finally:
        controller.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-seed", type=int, default=12345)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--room-count", type=int)
    parser.add_argument("--door-count", type=int)
    parser.add_argument(
        "--door-state",
        choices=("generated", "open", "closed", "random"),
        default="generated",
    )
    parser.add_argument("--allow-warnings", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for index in range(args.count):
        jobs.append(
            {
                "index": index,
                "split": args.split,
                "seed_start": args.start_seed + index * args.max_attempts,
                "max_attempts": args.max_attempts,
                "allow_warnings": args.allow_warnings,
                "room_count": args.room_count,
                "door_count": args.door_count,
                "door_state": args.door_state,
                "output_dir": str(out_dir),
            }
        )

    manifest = []
    total = len(jobs)
    workers = max(1, args.workers)

    print_progress(0, total, f"queued {total} houses with workers={workers}")
    if workers == 1:
        for completed, job in enumerate(jobs, start=1):
            info = render_house_job(job)
            manifest.append(info)
            print_progress(
                completed,
                total,
                (
                    f"finished house {completed}/{total} "
                    f"seed={info['seed']} rooms={info['num_rooms']} objects={info['num_objects']}"
                ),
                done=completed == total,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_to_job = {executor.submit(render_house_job, job): job for job in jobs}
            for completed, future in enumerate(as_completed(future_to_job), start=1):
                info = future.result()
                manifest.append(info)
                print_progress(
                    completed,
                    total,
                    (
                        f"finished house {info['index'] + 1}/{total} "
                        f"seed={info['seed']} rooms={info['num_rooms']} objects={info['num_objects']}"
                    ),
                    done=completed == total,
                )

    manifest.sort(key=lambda item: item["index"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} houses to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
