from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import queue
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maze_task import (  # noqa: E402
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_QUALITY,
    DEFAULT_RENDER_WIDTH,
    TASK_NAME,
    VIEW_ORDER,
    annotate_view_frames,
    generate_reachability_scene,
)
from maze_questions import (  # noqa: E402
    DEFAULT_QUESTION_TYPES_CLI,
    MIN_POINT_COUNT,
    QUESTION_TYPE_CONNECTED_POINT_LIST,
    QUESTION_TYPE_CONNECTED_SUBSET_CHOICE,
    QUESTION_TYPE_DOOR_OPEN,
    build_scene_question_specs,
    build_user_prompt_blocks,
    parse_question_types,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_REPEATS = 1
DEFAULT_SEED = 12345
DEFAULT_ROOM_COUNTS = "4"
DEFAULT_DOOR_COUNTS = "2"
DEFAULT_POINT_COUNTS = str(MIN_POINT_COUNT)
DEFAULT_QUESTION_TYPES = DEFAULT_QUESTION_TYPES_CLI
DIFFICULTY_SPECS: Tuple[Tuple[str, int, Tuple[Tuple[int, int, int], ...]], ...] = (
    ("easy", 1, ((3, 2, 4),)),
    ("medium", 2, ((4, 3, 4), (5, 4, 4))),
    ("hard", 3, ((6, 5, 5), (7, 6, 5))),
)
DOOR_OPEN_DIFFICULTY_SPECS: Tuple[Tuple[str, int, Tuple[Tuple[int, int, int], ...]], ...] = (
    ("easy", 1, ((4, 3, 4),)),
    ("medium", 2, ((5, 4, 4),)),
    ("hard", 3, ((6, 5, 5),)),
)

from jsonl_export import compose_pure_prompt, jsonable, to_relative_path  # noqa: E402


def python_hash_seed_for_base_seed(base_seed: int) -> str:
    return str(int(base_seed) % (2**32))


def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


def parse_int_range_list(spec: str, *, name: str, min_value: int | None = None) -> List[int]:
    values: List[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    deduped: List[int] = []
    seen = set()
    for value in values:
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} values must be >= {min_value}; got {value}.")
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    if not deduped:
        raise ValueError(f"Empty {name} spec: {spec!r}")
    return deduped


def parse_optional_int_range_list(spec: str, *, name: str, min_value: int | None = None) -> List[int | None]:
    text = str(spec).strip()
    if not text or text.casefold() == "any":
        return [None]
    values: List[int | None] = []
    seen: set[int | None] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.casefold() == "any":
            value: int | None = None
            if value not in seen:
                values.append(value)
                seen.add(value)
            continue
        for value in parse_int_range_list(part, name=name, min_value=min_value):
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError(f"Empty {name} spec: {spec!r}")
    return values


def difficulty_specs_for_question_types(
    question_types: str | Tuple[str, ...] | List[str] | None,
) -> Tuple[Tuple[str, int, Tuple[Tuple[int, int, int], ...]], ...]:
    selected_question_types = parse_question_types(question_types)
    if QUESTION_TYPE_DOOR_OPEN in selected_question_types:
        return DOOR_OPEN_DIFFICULTY_SPECS
    return DIFFICULTY_SPECS


def difficulty_setup_options(
    difficulty: str | None,
    *,
    question_types: str | Tuple[str, ...] | List[str] | None = None,
) -> Tuple[Tuple[int, int, int], ...] | None:
    normalized = str(difficulty or "").strip().casefold()
    if not normalized:
        return None
    specs = difficulty_specs_for_question_types(question_types)
    for name, _rank, setups in specs:
        if normalized == name:
            return setups
    legal = ", ".join(name for name, *_rest in specs)
    raise ValueError(f"Unsupported difficulty {difficulty!r}. Use one of: {legal}.")


def sample_difficulty_setups(
    *,
    difficulty: str,
    count: int,
    seed: int,
    question_types: str | Tuple[str, ...] | List[str] | None = None,
) -> List[Dict[str, int | None]]:
    options = difficulty_setup_options(difficulty, question_types=question_types)
    if options is None:
        return []
    rng = random.Random(int(seed))
    return [
        {
            "room_count": room_count,
            "door_count": door_count,
            "point_count": point_count,
        }
        for room_count, door_count, point_count in (
            rng.choice(options) for _ in range(max(0, int(count)))
        )
    ]


def build_setups(args: argparse.Namespace) -> List[Dict[str, int | None]]:
    difficulty = str(getattr(args, "difficulty", "") or "").strip()
    if difficulty:
        return sample_difficulty_setups(
            difficulty=difficulty,
            count=int(args.repeats),
            seed=int(args.seed),
            question_types=parse_question_types(args.question_types),
        )

    room_counts = parse_optional_int_range_list(args.room_counts, name="room-counts", min_value=1)
    door_counts = parse_optional_int_range_list(args.door_counts, name="door-counts", min_value=1)
    point_counts = parse_int_range_list(args.point_counts, name="point-counts", min_value=MIN_POINT_COUNT)
    setups: List[Dict[str, int | None]] = []
    for room_count in room_counts:
        for door_count in door_counts:
            for point_count in point_counts:
                setups.append(
                    {
                        "room_count": room_count,
                        "door_count": door_count,
                        "point_count": int(point_count),
                    }
                )
    return setups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reasoning samples for continuity_3d_maze.")
    parser.add_argument(
        "--repeats",
        "--scene-count",
        "--count",
        dest="repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of generated mazes per requested room/door/point setup; with --difficulty, total sampled scenes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed used to derive a deterministic per-maze seed sequence.")
    parser.add_argument(
        "--point-counts",
        "--point-count",
        dest="point_counts",
        default=DEFAULT_POINT_COUNTS,
        help=f"Comma/range labeled point counts, e.g. {MIN_POINT_COUNT} or {MIN_POINT_COUNT}-{MIN_POINT_COUNT + 2}.",
    )
    parser.add_argument(
        "--question-types",
        "--question-type",
        dest="question_types",
        default=DEFAULT_QUESTION_TYPES,
        help=(
            "Comma-separated question types to emit: subset_choice, point_list, door_open."
        ),
    )
    parser.add_argument(
        "--difficulty",
        choices=tuple(name for name, *_rest in DIFFICULTY_SPECS),
        default="",
        help="Optional difficulty level. When set, room/door/point setups are sampled from that level using --seed.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--room-counts",
        "--room-count",
        dest="room_counts",
        default=DEFAULT_ROOM_COUNTS,
        help="Comma/range ProcTHOR room-count filters, e.g. 4,6,8. Use 'any' to disable the filter.",
    )
    parser.add_argument(
        "--door-counts",
        "--door-count",
        dest="door_counts",
        default=DEFAULT_DOOR_COUNTS,
        help="Comma/range exact openable-door-count filters, e.g. 2,4. Use 'any' to disable the filter.",
    )
    parser.add_argument("--room-spec-id", help="Optional ProcTHOR room spec id.")
    parser.add_argument(
        "--door-state",
        choices=("generated", "open", "closed", "random"),
        default="random",
        help="Door state mode used when computing connectivity and rendering images.",
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--allow-warnings", action="store_true")
    parser.add_argument(
        "--platform",
        default="CloudRendering",
        help="Optional AI2-THOR platform name, e.g. CloudRendering.",
    )
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--render-quality", default=DEFAULT_RENDER_QUALITY)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes used for scene generation.")
    parser.add_argument(
        "--scene-timeout-sec",
        type=float,
        default=180.0,
        help="Optional per-scene timeout. When positive, each scene runs in an isolated subprocess.",
    )
    parser.add_argument(
        "--skip-failed-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip failed/timed-out scene attempts and keep sampling seeds until each setup reaches --repeats successes.",
    )
    parser.add_argument(
        "--max-seed-multiplier",
        type=int,
        default=10,
        help=(
            "Maximum candidate seeds per setup is --repeats times this value when "
            "--skip-failed-scenes is enabled; with --difficulty, this is the total candidate-scene multiplier."
        ),
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b"\n"
        if needs_newline:
            with path.open("ab") as handle:
                handle.write(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False))
            handle.write("\n")


def read_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL.") from exc
            if isinstance(row, dict) and row.get("id") is not None:
                sample_ids.add(str(row["id"]))
    return sample_ids


def sample_id(
    *,
    requested_room_count: int | None,
    requested_door_count: int | None,
    point_count: int,
    used_seed: int,
    question_suffix: str,
) -> str:
    room_label = "any" if requested_room_count is None else f"{int(requested_room_count):02d}"
    door_label = "any" if requested_door_count is None else f"{int(requested_door_count):02d}"
    return (
        f"{TASK_NAME}_rooms_{room_label}_doors_{door_label}_points_{point_count:02d}"
        f"_seed_{used_seed}_{question_suffix}"
    )


def build_difficulty_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    room_count = int(metadata["num_rooms"])
    door_count = int(metadata["num_openable_doors"])
    point_count = int(metadata["point_count"])
    difficulty = "custom"
    difficulty_rank = 0
    requested_difficulty = str(metadata.get("requested_difficulty") or "").strip()
    if requested_difficulty:
        for name, rank, _setups in DIFFICULTY_SPECS:
            if requested_difficulty == name:
                difficulty = name
                difficulty_rank = rank
                break
    if difficulty == "custom":
        for name, rank, setups in DIFFICULTY_SPECS:
            if (room_count, door_count, point_count) in setups:
                difficulty = name
                difficulty_rank = rank
                break
    if difficulty == "custom":
        for name, rank, setups in DOOR_OPEN_DIFFICULTY_SPECS:
            if (room_count, door_count, point_count) in setups:
                difficulty = name
                difficulty_rank = rank
                break

    group_dir = (
        f"difficulty_{difficulty_rank:02d}_{difficulty}/"
        f"rooms_{room_count:02d}/"
        f"doors_{door_count:02d}/"
        f"points_{point_count:02d}"
    )
    return {
        "difficulty": difficulty,
        "difficulty_score": difficulty_rank,
        "room_count": room_count,
        "door_count": door_count,
        "point_count": point_count,
        "pair_count": point_count * (point_count - 1) // 2,
        "group_dir": group_dir,
        "factors": {
            "rooms": room_count,
            "openable_doors": door_count,
            "points": point_count,
        },
    }


def enrich_metadata_with_difficulty(metadata: Dict[str, Any]) -> Dict[str, Any]:
    difficulty_info = build_difficulty_info(metadata)
    metadata["difficulty"] = difficulty_info["difficulty"]
    metadata["difficulty_score"] = difficulty_info["difficulty_score"]
    metadata["difficulty_factors"] = difficulty_info["factors"]
    metadata["output_group"] = difficulty_info["group_dir"]
    return difficulty_info


def question_sample_suffix(question_spec: Dict[str, Any]) -> str:
    question_type = str(question_spec["question_type"])
    payload = question_spec.get("question_payload") or {}
    if question_type == "connected_subset_choice":
        return "subset_choice"
    if question_type == "connected_point_list":
        return "point_list"
    if question_type == "door_open":
        return "door_open"
    return question_type


def save_question_images(
    *,
    scene: Dict[str, Any],
    highlighted_points: List[str],
    highlighted_doors: List[Dict[str, Any]],
    sample_dir: Path,
    output_base: Path,
) -> List[str]:
    annotated = annotate_view_frames(
        scene["frames"],
        scene["cameras"],
        scene["maze_state"],
        house_data=scene["house_data"],
        highlighted_points=highlighted_points,
        highlighted_doors=highlighted_doors,
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    for view_name in VIEW_ORDER:
        image_path = sample_dir / f"{view_name}.png"
        image = annotated[view_name]
        try:
            image.save(image_path)
        finally:
            image.close()
        image_paths.append(to_relative_path(image_path, base_dir=output_base))
    return image_paths


def scene_metadata_for_output(scene: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(scene["metadata"])
    metadata["points"] = list(scene["maze_state"].get("points", []))
    metadata["pairs"] = list(scene["maze_state"].get("pairs", []))
    return metadata


def export_scene_outputs(
    *,
    scene_index: int,
    setup_index: int,
    repeat_index: int,
    requested_room_count: int | None,
    requested_door_count: int | None,
    requested_point_count: int,
    scene: Dict[str, Any],
    output_base: Path,
    image_root: Path,
    config_rel: str,
    question_types: Tuple[str, ...],
    existing_sample_ids: set[str] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata = scene_metadata_for_output(scene)
    difficulty_info = enrich_metadata_with_difficulty(metadata)
    used_seed = int(metadata["used_seed"])
    requested_seed = int(metadata["requested_seed"])

    scene_record = {
        "scene_index": scene_index,
        "setup_index": setup_index,
        "repeat_index": repeat_index,
        "requested_seed": requested_seed,
        "used_seed": used_seed,
        "scene_metadata_json": None,
        "difficulty": difficulty_info["difficulty"],
        "difficulty_score": difficulty_info["difficulty_score"],
        "requested_room_count": requested_room_count,
        "requested_door_count": requested_door_count,
        "requested_point_count": requested_point_count,
        "output_group": difficulty_info["group_dir"],
        "metadata": metadata,
        "sample_ids": [],
    }

    sample_blueprints: List[Dict[str, Any]] = []
    existing_sample_ids = existing_sample_ids or set()
    question_specs = build_scene_question_specs(scene["maze_state"], question_types=question_types)
    for question_spec in question_specs:
        question_type = str(question_spec["question_type"])
        answer_type = str(question_spec["answer_type"])
        question_payload = dict(question_spec.get("question_payload") or {})
        current_sample_id = sample_id(
            requested_room_count=requested_room_count,
            requested_door_count=requested_door_count,
            point_count=requested_point_count,
            used_seed=used_seed,
            question_suffix=question_sample_suffix(question_spec),
        )
        if current_sample_id in existing_sample_ids:
            continue
        sample_dir = image_root / current_sample_id
        house_json_path = sample_dir / "house.json"
        scene_metadata_path = sample_dir / "scene_metadata.json"
        question_metadata = dict(metadata)
        question_metadata["generation_scene_index"] = metadata.get("scene_index")
        question_metadata["scene_index"] = scene_index
        question_metadata["setup_index"] = setup_index
        question_metadata["repeat_index"] = repeat_index
        question_metadata["requested_room_count"] = requested_room_count
        question_metadata["requested_door_count"] = requested_door_count
        question_metadata["requested_point_count"] = requested_point_count
        question_metadata["question_type"] = question_type
        question_metadata["answer_type"] = answer_type
        question_metadata["question_payload"] = question_payload
        write_json(house_json_path, scene["house_data"])
        question_metadata["house_json"] = to_relative_path(house_json_path, base_dir=output_base)
        write_json(scene_metadata_path, question_metadata)
        scene_metadata_rel = to_relative_path(scene_metadata_path, base_dir=output_base)
        question_metadata["scene_metadata_json"] = scene_metadata_rel
        if scene_record["scene_metadata_json"] is None:
            scene_record["scene_metadata_json"] = scene_metadata_rel
            scene_record["metadata"] = question_metadata
        scene_record["sample_ids"].append(current_sample_id)
        images = save_question_images(
            scene=scene,
            highlighted_points=list(question_spec.get("highlighted_points") or []),
            highlighted_doors=list(question_spec.get("highlighted_doors") or []),
            sample_dir=sample_dir,
            output_base=output_base,
        )
        question = str(question_spec["question"])
        answer = question_spec["ground_truth"]
        before_images, after_images = build_user_prompt_blocks(question, answer_type)
        pure_prompt = compose_pure_prompt(before_images, images, after_images)
        sample_blueprints.append(
            {
                "id": current_sample_id,
                "category": ["continuity", TASK_NAME, question_type],
                "type": "reasoning",
                "meta_info": {
                    "task_name": TASK_NAME,
                    "config": config_rel,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "seed": used_seed,
                    "setup_index": setup_index,
                    "repeat_index": repeat_index,
                    "difficulty": difficulty_info["difficulty"],
                    "scene_index": scene_index,
                    "difficulty_score": difficulty_info["difficulty_score"],
                    "requested_room_count": requested_room_count,
                    "requested_door_count": requested_door_count,
                    "requested_point_count": requested_point_count,
                    "num_rooms": difficulty_info["room_count"],
                    "num_openable_doors": difficulty_info["door_count"],
                    "point_count": difficulty_info["point_count"],
                    "question_payload": question_payload,
                },
                "question": pure_prompt,
                "images": images,
                "gt_answer": answer,
                "_answer_type": answer_type,
                "_metadata": question_metadata,
            }
        )

    return scene_record, sample_blueprints


def split_scene_indices(scene_count: int, workers: int) -> List[List[int]]:
    return [list(range(worker_slot, scene_count, workers)) for worker_slot in range(workers)]


def build_worker_config(
    *,
    args: argparse.Namespace,
    output_base: Path,
    image_root: Path,
    total_scenes: int,
) -> Dict[str, Any]:
    question_types = parse_question_types(args.question_types)
    return {
        "seed": int(args.seed),
        "scene_start_seeds": deterministic_seed_sequence(int(args.seed), int(total_scenes)),
        "config_rel": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_base),
        "question_types": list(question_types),
        "difficulty": str(getattr(args, "difficulty", "") or "").strip(),
        "split": args.split,
        "max_attempts": int(args.max_attempts),
        "allow_warnings": bool(args.allow_warnings),
        "room_spec_id": args.room_spec_id,
        "door_state": args.door_state,
        "platform": args.platform,
        "render_width": int(args.render_width),
        "render_height": int(args.render_height),
        "render_quality": args.render_quality,
        "output_base": str(output_base),
        "image_root": str(image_root),
    }


def generate_scene_output(task: Dict[str, Any]) -> Dict[str, Any]:
    output_base = Path(task["output_base"])
    image_root = Path(task["image_root"])
    scene_index = int(task["scene_index"])
    generation_scene_index = int(task.get("generation_scene_index", scene_index))
    if "start_seed" in task:
        start_seed = int(task["start_seed"])
    else:
        start_seed = int(task["scene_start_seeds"][scene_index])
    runtime_name = f"generate_samples_worker_{int(task['worker_slot']):02d}"
    question_types = parse_question_types(task.get("question_types"))
    scene = generate_reachability_scene(
        scene_index=generation_scene_index,
        start_seed=start_seed,
        split=task["split"],
        max_attempts=int(task["max_attempts"]),
        allow_warnings=bool(task["allow_warnings"]),
        room_count=task["room_count"],
        door_count=task["door_count"],
        room_spec_id=task["room_spec_id"],
        point_count=int(task["point_count"]),
        door_state_mode=task["door_state"],
        runtime_name=runtime_name,
        platform=task["platform"],
        render_width=int(task["render_width"]),
        render_height=int(task["render_height"]),
        render_quality=task["render_quality"],
        require_connected_subset_choice=QUESTION_TYPE_CONNECTED_SUBSET_CHOICE in question_types,
        randomize_point_list_source=QUESTION_TYPE_CONNECTED_POINT_LIST in question_types,
        require_door_open=QUESTION_TYPE_DOOR_OPEN in question_types,
    )
    requested_difficulty = str(task.get("difficulty") or "").strip()
    if requested_difficulty:
        scene["metadata"]["requested_difficulty"] = requested_difficulty
    scene_record, sample_blueprints = export_scene_outputs(
        scene_index=scene_index,
        setup_index=int(task["setup_index"]),
        repeat_index=int(task["repeat_index"]),
        requested_room_count=task["room_count"],
        requested_door_count=task["door_count"],
        requested_point_count=int(task["point_count"]),
        scene=scene,
        output_base=output_base,
        image_root=image_root,
        config_rel=str(task["config_rel"]),
        question_types=question_types,
        existing_sample_ids=set(task.get("existing_sample_ids") or []),
    )
    del scene
    gc.collect()
    return {
        "scene_index": scene_index,
        "scene_record": scene_record,
        "sample_blueprints": sample_blueprints,
    }


def _generate_scene_output_worker(task: Dict[str, Any], result_queue) -> None:
    try:
        os.setsid()
    except Exception:
        pass
    try:
        result_queue.put({"ok": True, "item": generate_scene_output(task)})
    except BaseException as exc:  # pragma: no cover - exercised by subprocess.
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        gc.collect()


def generate_scene_output_with_timeout(
    task: Dict[str, Any],
    *,
    timeout_sec: float,
) -> Dict[str, Any]:
    if timeout_sec <= 0:
        return generate_scene_output(task)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_generate_scene_output_worker, args=(task, result_queue))
    process.start()
    process.join(float(timeout_sec))

    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.terminate()
        process.join(10)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.join(10)
        raise TimeoutError(
            f"Scene generation timed out after {timeout_sec:g}s "
            f"for setup_index={task.get('setup_index')} "
            f"repeat_index={task.get('repeat_index')} "
            f"seed={task.get('start_seed')}"
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"Scene generation subprocess exited without a result. exitcode={process.exitcode}"
        ) from exc

    if payload.get("ok"):
        return payload["item"]
    raise RuntimeError(payload.get("traceback") or payload.get("error") or "Scene generation failed.")


def append_scene_output(
    *,
    item: Dict[str, Any],
    scene_records: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> None:
    if not item["sample_blueprints"]:
        return
    scene_records.append(item["scene_record"])
    for sample_blueprint in item["sample_blueprints"]:
        sample_entry = {
            key: value
            for key, value in sample_blueprint.items()
            if not key.startswith("_")
        }
        samples.append(sample_entry)


def _resolve_output_path(path_text: str, *, output_base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return output_base / path


def rewrite_scene_output_indices(
    *,
    item: Dict[str, Any],
    output_base: Path,
    scene_index: int,
    setup_index: int,
    repeat_index: int,
) -> None:
    """Assign final contiguous indices after a candidate has been selected."""
    item["scene_index"] = scene_index
    scene_record = item.get("scene_record") or {}
    scene_record["scene_index"] = scene_index
    scene_record["setup_index"] = setup_index
    scene_record["repeat_index"] = repeat_index

    first_metadata: Dict[str, Any] | None = None
    for sample_blueprint in item.get("sample_blueprints", []):
        meta_info = sample_blueprint.get("meta_info")
        if isinstance(meta_info, dict):
            meta_info["scene_index"] = scene_index
            meta_info["setup_index"] = setup_index
            meta_info["repeat_index"] = repeat_index

        metadata = sample_blueprint.get("_metadata")
        if isinstance(metadata, dict):
            metadata["scene_index"] = scene_index
            metadata["setup_index"] = setup_index
            metadata["repeat_index"] = repeat_index
            if first_metadata is None:
                first_metadata = metadata

            metadata_path_text = metadata.get("scene_metadata_json")
            if metadata_path_text:
                write_json(_resolve_output_path(str(metadata_path_text), output_base=output_base), metadata)

    if first_metadata is not None:
        scene_record["metadata"] = first_metadata


def run_generate_candidate_task(task: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    return generate_scene_output_with_timeout(task, timeout_sec=timeout_sec)


def generate_samples(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    if args.max_seed_multiplier < 1:
        raise ValueError("max-seed-multiplier must be at least 1.")
    use_resilient_generation = bool(args.skip_failed_scenes or args.scene_timeout_sec > 0)
    if args.scene_timeout_sec > 0:
        os.environ["PYTHONHASHSEED"] = python_hash_seed_for_base_seed(int(args.seed))
    setups = build_setups(args)
    if not setups:
        raise ValueError("At least one room/door/point setup is required.")

    output_json = Path(args.output_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_base = output_json.parent
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    difficulty = str(getattr(args, "difficulty", "") or "").strip()
    scene_tasks: List[Dict[str, Any]] = []
    if difficulty:
        for setup in setups:
            scene_index = len(scene_tasks)
            scene_tasks.append(
                {
                    "setup_index": scene_index,
                    "repeat_index": 0,
                    "scene_index": scene_index,
                    **setup,
                }
            )
    else:
        for setup_index, setup in enumerate(setups):
            for repeat_index in range(int(args.repeats)):
                scene_index = len(scene_tasks)
                scene_tasks.append(
                    {
                        "setup_index": setup_index,
                        "repeat_index": repeat_index,
                        "scene_index": scene_index,
                        **setup,
                    }
                )

    worker_config = build_worker_config(
        args=args,
        output_base=output_base,
        image_root=image_root,
        total_scenes=len(scene_tasks),
    )
    existing_sample_ids = read_existing_sample_ids(output_json)
    worker_config["existing_sample_ids"] = sorted(existing_sample_ids)
    selected_question_types = tuple(worker_config["question_types"])
    scene_records: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    skipped_scene_attempts = 0
    attempted_scene_count = 0

    if use_resilient_generation:
        if not args.skip_failed_scenes:
            progress = tqdm(total=len(scene_tasks), desc="3D Maze generate", unit="scene")
            try:
                for scene_task in scene_tasks:
                    attempted_scene_count += 1
                    item = generate_scene_output_with_timeout(
                        {
                            **worker_config,
                            "worker_slot": 0,
                            **scene_task,
                        },
                        timeout_sec=float(args.scene_timeout_sec),
                    )
                    append_scene_output(item=item, scene_records=scene_records, samples=samples)
                    progress.update(1)
            finally:
                progress.close()

            sample_count = len(samples)
            scene_count = len(scene_records)
            payload = {
                "task": TASK_NAME,
                "category": "continuity",
                "level": "reasoning",
                "question_types": list(selected_question_types),
                "scene_count": scene_count,
                "sample_count": sample_count,
                "setup_count": len(setups),
                "setups": setups,
                "repeats_per_setup": args.repeats,
                "door_state_mode": args.door_state,
                "workers": args.workers,
                "question_jsonl": str(output_json),
                "question_jsonl_write_mode": "append",
                "scene_timeout_sec": float(args.scene_timeout_sec),
                "skip_failed_scenes": False,
                "skipped_scene_attempts": skipped_scene_attempts,
                "attempted_scene_count": attempted_scene_count,
                "scenes": scene_records,
            }
            append_jsonl(output_json, samples)
            return payload

        target_scene_count = len(scene_tasks)
        candidate_limit = int(args.repeats) * int(args.max_seed_multiplier)
        seed_count = (1 if difficulty else len(setups)) * candidate_limit
        candidate_seeds = deterministic_seed_sequence(int(args.seed), seed_count)
        candidate_setups = (
            sample_difficulty_setups(
                difficulty=difficulty,
                count=candidate_limit,
                seed=int(args.seed),
                question_types=selected_question_types,
            )
            if difficulty
            else []
        )
        progress = tqdm(total=target_scene_count, desc="3D Maze generate", unit="scene")
        executor = (
            ProcessPoolExecutor(max_workers=int(args.workers))
            if int(args.workers) > 1
            else None
        )
        try:
            scene_index = 0
            difficulty_setup_indices = [0] if difficulty else list(range(len(setups)))
            for setup_index in difficulty_setup_indices:
                setup = {} if difficulty else setups[setup_index]
                success_count = 0
                candidate_index = 0
                setup_target_count = int(args.repeats) if difficulty else int(args.repeats)
                while success_count < setup_target_count and candidate_index < candidate_limit:
                    remaining_successes = setup_target_count - success_count
                    wave_size = min(
                        int(args.workers),
                        remaining_successes,
                        candidate_limit - candidate_index,
                    )
                    wave_tasks: List[Dict[str, Any]] = []
                    for wave_offset in range(wave_size):
                        current_candidate_index = candidate_index + wave_offset
                        seed_index = (
                            current_candidate_index
                            if difficulty
                            else setup_index * candidate_limit + current_candidate_index
                        )
                        start_seed = int(candidate_seeds[seed_index])
                        scene_setup = candidate_setups[current_candidate_index] if difficulty else setup
                        wave_tasks.append(
                            {
                                **worker_config,
                                "worker_slot": current_candidate_index % int(args.workers),
                                "setup_index": scene_index + wave_offset if difficulty else setup_index,
                                "repeat_index": 0 if difficulty else success_count + wave_offset,
                                "scene_index": seed_index,
                                "generation_scene_index": seed_index,
                                "candidate_index": current_candidate_index,
                                "start_seed": start_seed,
                                **scene_setup,
                            }
                        )

                    attempted_scene_count += len(wave_tasks)
                    wave_results: List[Tuple[int, Dict[str, Any], Dict[str, Any] | None, BaseException | None]] = []
                    if executor is None:
                        for scene_task in wave_tasks:
                            try:
                                item = run_generate_candidate_task(
                                    scene_task,
                                    float(args.scene_timeout_sec),
                                )
                                wave_results.append((int(scene_task["candidate_index"]), scene_task, item, None))
                            except BaseException as exc:
                                wave_results.append((int(scene_task["candidate_index"]), scene_task, None, exc))
                    else:
                        future_to_task = {
                            executor.submit(
                                run_generate_candidate_task,
                                scene_task,
                                float(args.scene_timeout_sec),
                            ): scene_task
                            for scene_task in wave_tasks
                        }
                        for future in as_completed(future_to_task):
                            scene_task = future_to_task[future]
                            try:
                                item = future.result()
                                wave_results.append((int(scene_task["candidate_index"]), scene_task, item, None))
                            except BaseException as exc:
                                wave_results.append((int(scene_task["candidate_index"]), scene_task, None, exc))

                    for _current_candidate_index, scene_task, item, exc in sorted(
                        wave_results,
                        key=lambda result: result[0],
                    ):
                        if exc is not None:
                            skipped_scene_attempts += 1
                            progress.set_postfix(
                                setup=setup_index,
                                skipped=skipped_scene_attempts,
                                seed=scene_task.get("start_seed"),
                            )
                            continue

                        final_setup_index = scene_index if difficulty else setup_index
                        final_repeat_index = 0 if difficulty else success_count
                        rewrite_scene_output_indices(
                            item=item or {},
                            output_base=output_base,
                            scene_index=scene_index,
                            setup_index=final_setup_index,
                            repeat_index=final_repeat_index,
                        )
                        append_scene_output(item=item or {}, scene_records=scene_records, samples=samples)
                        scene_index += 1
                        success_count += 1
                        progress.update(1)
                        progress.set_postfix(setup=setup_index, skipped=skipped_scene_attempts)

                    candidate_index += wave_size

                if success_count < int(args.repeats):
                    raise RuntimeError(
                        f"Only generated {success_count}/{args.repeats} scenes for setup_index={setup_index} "
                        f"after trying {candidate_limit} candidate seeds. Increase --max-seed-multiplier "
                        "or relax the room/door/point setup."
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            progress.close()

        sample_count = len(samples)
        scene_count = len(scene_records)
        payload = {
            "task": TASK_NAME,
            "category": "continuity",
            "level": "reasoning",
            "question_types": list(selected_question_types),
            "scene_count": scene_count,
            "sample_count": sample_count,
            "setup_count": len(setups),
            "setups": setups,
            "repeats_per_setup": args.repeats,
            "door_state_mode": args.door_state,
            "workers": args.workers,
            "question_jsonl": str(output_json),
            "question_jsonl_write_mode": "append",
            "scene_timeout_sec": float(args.scene_timeout_sec),
            "skip_failed_scenes": bool(args.skip_failed_scenes),
            "skipped_scene_attempts": skipped_scene_attempts,
            "attempted_scene_count": attempted_scene_count,
            "scenes": scene_records,
        }
        append_jsonl(output_json, samples)
        return payload

    scene_outputs_by_index: Dict[int, Dict[str, Any]] = {}
    next_scene_index = 0

    def flush_ready_scene_outputs() -> int:
        nonlocal next_scene_index
        flushed = 0
        while next_scene_index in scene_outputs_by_index:
            item = scene_outputs_by_index.pop(next_scene_index)
            append_scene_output(item=item, scene_records=scene_records, samples=samples)
            next_scene_index += 1
            flushed += 1
        if flushed:
            gc.collect()
        return flushed

    progress = tqdm(total=len(scene_tasks), desc="3D Maze generate", unit="scene")
    try:
        if args.workers == 1:
            for scene_task in scene_tasks:
                scene_outputs_by_index[int(scene_task["scene_index"])] = generate_scene_output(
                    {
                        **worker_config,
                        "worker_slot": 0,
                        **scene_task,
                    }
                )
                flush_ready_scene_outputs()
                progress.update(1)
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = [
                    executor.submit(
                        generate_scene_output,
                            {
                                **worker_config,
                                "worker_slot": int(scene_task["scene_index"]) % int(args.workers),
                                **scene_task,
                            },
                        )
                    for scene_task in scene_tasks
                ]
                for future in as_completed(futures):
                    item = future.result()
                    scene_outputs_by_index[int(item["scene_index"])] = item
                    flush_ready_scene_outputs()
                    progress.update(1)
    finally:
        progress.close()

    while flush_ready_scene_outputs():
        pass

    sample_count = len(samples)
    scene_count = len(scene_records)
    payload = {
        "task": TASK_NAME,
        "category": "continuity",
        "level": "reasoning",
        "question_types": list(selected_question_types),
        "scene_count": scene_count,
        "sample_count": sample_count,
        "setup_count": len(setups),
        "setups": setups,
        "repeats_per_setup": args.repeats,
        "door_state_mode": args.door_state,
        "workers": args.workers,
        "question_jsonl": str(output_json),
        "question_jsonl_write_mode": "append",
        "scenes": scene_records,
    }
    append_jsonl(output_json, samples)
    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = generate_samples(args)
    print(f"Appended {payload['sample_count']} question rows to {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
