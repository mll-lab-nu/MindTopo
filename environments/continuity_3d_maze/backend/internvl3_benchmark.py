from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gc
import io
import json
import multiprocessing as mp
from multiprocessing import current_process
import os
import queue
import random
import signal
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from internvl3_config import load_local_config, resolve_config_value
from tqdm.auto import tqdm

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. Install it with: pip install requests"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonl_export import compose_pure_prompt, is_meta_prompt, split_pure_prompt  # noqa: E402
from maze_questions import (  # noqa: E402
    DEFAULT_QUESTION_TYPES_CLI,
    MIN_POINT_COUNT,
    QUESTION_TYPES,
    QUESTION_TYPE_CONNECTED_POINT_LIST,
    QUESTION_TYPE_CONNECTED_SUBSET_CHOICE,
    QUESTION_TYPE_DOOR_OPEN,
    ANSWER_TYPE_MULTIPLE_CHOICE,
    ANSWER_TYPE_COLOR_LIST,
    ANSWER_TYPE_NAME_LIST,
    ANSWER_TYPE_YES_NO,
    answers_equal,
    build_scene_question_specs,
    build_user_prompt_blocks as build_task_user_prompt_blocks,
    parse_question_types,
    question_count_for_point_count,
    parse_answer as parse_task_answer,
)

_MAZE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from generate_samples import enrich_metadata_with_difficulty, question_sample_suffix  # noqa: E402
    from maze_task import (  # noqa: E402
        DEFAULT_RENDER_HEIGHT,
        DEFAULT_RENDER_QUALITY,
        DEFAULT_RENDER_WIDTH,
        TASK_NAME,
        VIEW_ORDER,
        annotate_view_frames,
        generate_reachability_scene,
    )
except Exception as exc:  # pragma: no cover - load mode can run without AI2-THOR assets.
    _MAZE_IMPORT_ERROR = exc
    DEFAULT_RENDER_WIDTH = 1920
    DEFAULT_RENDER_HEIGHT = 1080
    DEFAULT_RENDER_QUALITY = "Ultra"
    TASK_NAME = "continuity_3d_maze"
    VIEW_ORDER = ("topdown", "front45", "right45", "rear45", "left45")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


def python_hash_seed_for_base_seed(base_seed: int) -> str:
    return str(int(base_seed) % (2**32))


def read_question_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
    return rows


def loaded_jsonl_value(
    row: Dict[str, Any],
    meta_info: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in meta_info and meta_info[key] is not None:
        return meta_info[key]
    if key in row and row[key] is not None:
        return row[key]
    return default


def resolve_question_image_path(image_path: str, *, question_jsonl: Path) -> Path:
    raw_path = Path(image_path)
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidates = [
        question_jsonl.parent / raw_path,
        PROJECT_ROOT / raw_path,
        REPO_ROOT / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image path {image_path!r} from {question_jsonl}.")


def image_file_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_continuity_3d_maze.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_continuity_3d_maze.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_REPEATS = 1
DEFAULT_SEED = 12345
DEFAULT_ROOM_COUNTS = "4"
DEFAULT_DOOR_COUNTS = "2"
DEFAULT_POINT_COUNTS = str(MIN_POINT_COUNT)
DEFAULT_QUESTION_TYPES = DEFAULT_QUESTION_TYPES_CLI


@dataclass
class SampleResult:
    sample_index: int
    sample_id: str
    question: str
    scene_index: int
    setup_index: int
    repeat_index: int
    seed: int
    question_type: str
    answer_type: str
    difficulty: str
    difficulty_score: int
    requested_room_count: Optional[int]
    requested_door_count: Optional[int]
    requested_point_count: int
    num_rooms: int
    num_openable_doors: int
    point_count: int
    question_payload: Dict[str, Any]
    ground_truth: Any
    predicted_answer: Optional[Any]
    correct: bool
    invalid_response: bool
    api_error: bool
    api_error_message: Optional[str]
    raw_response_text: str
    image_paths: List[str]


def ensure_jsonl_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        needs_newline = handle.read(1) != b"\n"
    if needs_newline:
        with path.open("ab") as handle:
            handle.write(b"\n")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append:
        ensure_jsonl_append_boundary(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_jsonl_rows_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
    return rows


def upsert_jsonl_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    new_ids = {str(row.get("id")) for row in rows}
    retained_rows = [
        row
        for row in read_jsonl_rows_if_exists(path)
        if str(row.get("id")) not in new_ids
    ]
    write_jsonl(path, [*retained_rows, *rows], append=False)


def to_relative_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def build_sample_id(
    *,
    requested_room_count: int | None,
    requested_door_count: int | None,
    point_count: int,
    seed: int,
    question_suffix: str,
) -> str:
    room_label = "any" if requested_room_count is None else f"{int(requested_room_count):02d}"
    door_label = "any" if requested_door_count is None else f"{int(requested_door_count):02d}"
    return (
        f"{TASK_NAME}_rooms_{room_label}_doors_{door_label}_points_{point_count:02d}"
        f"_seed_{seed}_{question_suffix}"
    )


def write_meta_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    results: Sequence[SampleResult],
    mode: str = "write",
    write_question_jsonl: bool = True,
) -> Tuple[Path, Path]:
    if mode not in {"write", "append", "upsert"}:
        raise ValueError(f"Unsupported JSONL write mode: {mode!r}")

    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", output_dir)
    question_path = output_dir / "question.jsonl"
    answer_path = output_dir / "model_answer.jsonl"
    question_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []

    for result in results:
        base_meta = {
            "task_name": TASK_NAME,
            "config": config_rel,
            "question_type": result.question_type,
            "answer_type": result.answer_type,
            "seed": result.seed,
            "setup_index": result.setup_index,
            "repeat_index": result.repeat_index,
            "scene_index": result.scene_index,
            "sample_index": result.sample_index,
            "question_payload": result.question_payload,
            "difficulty": result.difficulty,
            "difficulty_score": result.difficulty_score,
            "requested_room_count": result.requested_room_count,
            "requested_door_count": result.requested_door_count,
            "requested_point_count": result.requested_point_count,
            "num_rooms": result.num_rooms,
            "num_openable_doors": result.num_openable_doors,
            "point_count": result.point_count,
        }
        question_rows.append(
            {
                "id": result.sample_id,
                "category": ["continuity", TASK_NAME, result.question_type],
                "type": "reasoning",
                "meta_info": base_meta,
                "question": result.question,
                "images": result.image_paths,
                "gt_answer": result.ground_truth,
            }
        )
        answer_rows.append(
            {
                "id": result.sample_id,
                "category": ["continuity", TASK_NAME, result.question_type],
                "type": "reasoning",
                "meta_info": {
                    **base_meta,
                    "model_id": model_id,
                },
                "question": result.question,
                "answer": result.predicted_answer,
                "correct": result.correct,
                "invalid_response": result.invalid_response,
                "api_error": result.api_error,
                "raw_response_text": result.raw_response_text,
            }
        )

    if mode == "upsert":
        if write_question_jsonl:
            upsert_jsonl_rows(question_path, question_rows)
        upsert_jsonl_rows(answer_path, answer_rows)
    else:
        if write_question_jsonl:
            write_jsonl(question_path, question_rows, append=mode == "append")
        write_jsonl(answer_path, answer_rows, append=mode == "append")
    return question_path, answer_path


def _collect_text_fragments(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if value is None:
        return []
    if isinstance(value, list):
        fragments: List[str] = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: List[str] = []
        for key in ("text", "content", "value", "output_text", "refusal"):
            if key in value:
                fragments.extend(_collect_text_fragments(value.get(key)))
        if fragments:
            return fragments
    return []


def normalize_message_content(content: Any) -> str:
    fragments = _collect_text_fragments(content)
    return "\n".join(fragments).strip() if fragments else ""


def extract_response_text(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    debug_info: Dict[str, Any] = {"response_payload": payload}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", debug_info, "Unexpected response payload: missing choices."
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    if not isinstance(message, dict):
        message = {}
    debug_info["finish_reason"] = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    debug_info["message_content_raw"] = message.get("content")
    debug_info["message_reasoning_content_raw"] = message.get("reasoning_content")
    text = (
        normalize_message_content(message.get("content"))
        or normalize_message_content(first_choice.get("text") if isinstance(first_choice, dict) else None)
        or normalize_message_content(first_choice.get("content") if isinstance(first_choice, dict) else None)
    )
    return text, debug_info, None


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_answer(text: str, answer_type: str) -> Optional[Any]:
    return parse_task_answer(text, answer_type)


def build_user_prompt_blocks(question: str, answer_type: str) -> Tuple[str, str]:
    return build_task_user_prompt_blocks(question, answer_type)


def request_prediction(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_prompt_before_images: str,
    user_prompt_after_images: str,
    image_data_urls: Sequence[str],
    max_tokens: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_before_images}]
    for image_data_url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    user_content.append({"type": "text", "text": user_prompt_after_images})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        return "", {"http_status": response.status_code, "response_text": response.text}, f"HTTP {response.status_code}: {response.text}"
    return extract_response_text(response.json())


def save_question_assets(
    *,
    scene: Dict[str, Any],
    highlighted_points: Sequence[str],
    highlighted_doors: Sequence[Dict[str, Any]],
    sample_dir: Path,
    output_base: Path,
    include_data_urls: bool,
) -> Tuple[List[str], List[str]]:
    annotated = annotate_view_frames(
        scene["frames"],
        scene["cameras"],
        scene["maze_state"],
        house_data=scene["house_data"],
        highlighted_points=highlighted_points,
        highlighted_doors=highlighted_doors,
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    relative_paths: List[str] = []
    image_data_urls: List[str] = []
    for view_name in VIEW_ORDER:
        image = annotated[view_name]
        image_path = sample_dir / f"{view_name}.png"
        try:
            image.save(image_path)
            relative_paths.append(to_relative_path(image_path, output_base))
            if include_data_urls:
                image_data_urls.append(image_to_data_url(image))
        finally:
            image.close()
    return relative_paths, image_data_urls


def write_prompt_files(
    *,
    pair_dir: Path,
    before_images: str,
    after_images: str,
    image_paths: List[str],
    ground_truth: Any,
    raw_response_text: str,
    predicted_answer: Optional[Any],
    api_error_message: Optional[str],
    response_debug: Optional[Dict[str, Any]],
) -> None:
    image_list = "\n".join(image_paths)
    prompt_debug = (
        "[user_prompt]\n"
        f"{before_images}\n\n[Images inserted here]\n\n{after_images}\n\n"
        "[images]\n"
        f"{image_list}\n\n"
        "[ground_truth]\n"
        f"{ground_truth}\n\n"
        "[model_response]\n"
        f"{raw_response_text or '<empty>'}\n\n"
        "[parsed_answer]\n"
        f"{('<none>' if predicted_answer is None else predicted_answer)}\n\n"
        "[api_error]\n"
        f"{api_error_message or '<none>'}\n\n"
        "[response_debug]\n"
        f"{json.dumps(response_debug, indent=2, ensure_ascii=False) if response_debug else '<none>'}\n"
    )
    (pair_dir / "prompt_debug.txt").write_text(prompt_debug, encoding="utf-8")

    pure_prompt = compose_pure_prompt(before_images, image_paths, after_images)
    (pair_dir / "pure_prompt.txt").write_text(pure_prompt + "\n", encoding="utf-8")


def write_pure_prompt_file(*, pair_dir: Path, before_images: str, image_paths: List[str], after_images: str) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    pure_prompt = compose_pure_prompt(before_images, image_paths, after_images)
    (pair_dir / "pure_prompt.txt").write_text(pure_prompt + "\n", encoding="utf-8")


def summarize_results(results: Sequence[SampleResult]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(result.correct for result in results)
    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_responses": sum(result.invalid_response for result in results),
        "api_errors": sum(result.api_error for result in results),
    }


def split_scene_indices(scene_count: int, workers: int) -> List[List[int]]:
    return [list(range(worker_slot, scene_count, workers)) for worker_slot in range(workers)]


def export_benchmark_scene(
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
    include_data_urls: bool,
    question_types: Tuple[str, ...],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata = dict(scene["metadata"])
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

    question_payloads: List[Dict[str, Any]] = []
    question_specs = build_scene_question_specs(scene["maze_state"], question_types=question_types)
    for question_spec in question_specs:
        question_type = str(question_spec["question_type"])
        answer_type = str(question_spec["answer_type"])
        question_payload = dict(question_spec.get("question_payload") or {})
        sample_id = build_sample_id(
            requested_room_count=requested_room_count,
            requested_door_count=requested_door_count,
            point_count=requested_point_count,
            seed=used_seed,
            question_suffix=question_sample_suffix(question_spec),
        )
        sample_dir = image_root / sample_id
        house_json_path = sample_dir / "house.json"
        scene_metadata_path = sample_dir / "scene_metadata.json"
        question_metadata = dict(metadata)
        question_metadata["setup_index"] = setup_index
        question_metadata["repeat_index"] = repeat_index
        question_metadata["requested_room_count"] = requested_room_count
        question_metadata["requested_door_count"] = requested_door_count
        question_metadata["requested_point_count"] = requested_point_count
        question_metadata["question_type"] = question_type
        question_metadata["answer_type"] = answer_type
        question_metadata["question_payload"] = question_payload
        write_json(house_json_path, scene["house_data"])
        question_metadata["house_json"] = to_relative_path(house_json_path, output_base)
        write_json(scene_metadata_path, question_metadata)
        if scene_record["scene_metadata_json"] is None:
            scene_record["scene_metadata_json"] = to_relative_path(scene_metadata_path, output_base)
            scene_record["metadata"] = question_metadata
        scene_record["sample_ids"].append(sample_id)
        image_paths, image_data_urls = save_question_assets(
            scene=scene,
            highlighted_points=list(question_spec.get("highlighted_points") or []),
            highlighted_doors=list(question_spec.get("highlighted_doors") or []),
            sample_dir=sample_dir,
            output_base=output_base,
            include_data_urls=include_data_urls,
        )
        question = str(question_spec["question"])
        answer = question_spec["ground_truth"]
        before_images, after_images = build_user_prompt_blocks(question, answer_type)
        pure_prompt = compose_pure_prompt(before_images, image_paths, after_images)
        question_payloads.append(
            {
                "scene_index": scene_index,
                "setup_index": setup_index,
                "repeat_index": repeat_index,
                "seed": used_seed,
                "question_type": question_type,
                "question": pure_prompt,
                "answer_type": answer_type,
                "difficulty": difficulty_info["difficulty"],
                "difficulty_score": difficulty_info["difficulty_score"],
                "requested_room_count": requested_room_count,
                "requested_door_count": requested_door_count,
                "requested_point_count": requested_point_count,
                "num_rooms": difficulty_info["room_count"],
                "num_openable_doors": difficulty_info["door_count"],
                "point_count": difficulty_info["point_count"],
                "question_payload": question_payload,
                "ground_truth": answer,
                "before_images": before_images,
                "after_images": after_images,
                "image_paths": image_paths,
                "image_data_urls": image_data_urls,
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
            }
        )

    return scene_record, question_payloads


def generate_benchmark_scene_output(task: Dict[str, Any]) -> Dict[str, Any]:
    output_base = Path(task["output_base"])
    image_root = Path(task["image_root"])
    scene_index = int(task["scene_index"])
    generation_scene_index = int(task.get("generation_scene_index", scene_index))
    if "start_seed" in task:
        start_seed = int(task["start_seed"])
    else:
        start_seed = int(task["scene_start_seeds"][scene_index])
    runtime_name = f"benchmark_worker_{int(task.get('worker_slot', 0)):02d}"
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
    scene_record, question_payloads = export_benchmark_scene(
        scene_index=scene_index,
        setup_index=int(task["setup_index"]),
        repeat_index=int(task["repeat_index"]),
        requested_room_count=task["room_count"],
        requested_door_count=task["door_count"],
        requested_point_count=int(task["point_count"]),
        scene=scene,
        output_base=output_base,
        image_root=image_root,
        include_data_urls=False,
        question_types=question_types,
    )
    del scene
    gc.collect()
    return {
        "scene_index": scene_index,
        "scene_record": scene_record,
        "question_payloads": question_payloads,
    }


def _generate_benchmark_scene_output_worker(task: Dict[str, Any], result_queue) -> None:
    try:
        os.setsid()
    except Exception:
        pass
    try:
        result_queue.put({"ok": True, "item": generate_benchmark_scene_output(task)})
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


def generate_benchmark_scene_output_with_timeout(
    task: Dict[str, Any],
    *,
    timeout_sec: float,
) -> Dict[str, Any]:
    if timeout_sec <= 0:
        return generate_benchmark_scene_output(task)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_generate_benchmark_scene_output_worker, args=(task, result_queue))
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


def _resolve_output_path(path_text: str, *, output_base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return output_base / path


def rewrite_benchmark_scene_output_indices(
    *,
    item: Dict[str, Any],
    output_base: Path,
    scene_index: int,
    setup_index: int,
    repeat_index: int,
) -> None:
    """Assign final contiguous indices after a generated candidate is selected."""
    item["scene_index"] = scene_index
    scene_record = item.get("scene_record") or {}
    scene_record["scene_index"] = scene_index
    scene_record["setup_index"] = setup_index
    scene_record["repeat_index"] = repeat_index
    metadata = scene_record.get("metadata")
    if isinstance(metadata, dict):
        metadata["scene_index"] = scene_index
        metadata["setup_index"] = setup_index
        metadata["repeat_index"] = repeat_index

    metadata_paths: set[Path] = set()
    scene_metadata_path_text = scene_record.get("scene_metadata_json")
    if scene_metadata_path_text:
        metadata_paths.add(_resolve_output_path(str(scene_metadata_path_text), output_base=output_base))

    for question_payload in item.get("question_payloads", []):
        question_payload["scene_index"] = scene_index
        question_payload["setup_index"] = setup_index
        question_payload["repeat_index"] = repeat_index
        sample_dir_text = question_payload.get("sample_dir")
        if sample_dir_text:
            metadata_paths.add(Path(str(sample_dir_text)) / "scene_metadata.json")

    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        try:
            file_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(file_metadata, dict):
            file_metadata["scene_index"] = scene_index
            file_metadata["setup_index"] = setup_index
            file_metadata["repeat_index"] = repeat_index
            write_json(metadata_path, file_metadata)


def run_benchmark_candidate_task(task: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    return generate_benchmark_scene_output_with_timeout(task, timeout_sec=timeout_sec)


def build_benchmark_worker_config(
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


def build_sample_result(
    *,
    sample_index: int,
    result_data: Dict[str, Any],
) -> SampleResult:
    seed = int(result_data["seed"])
    return SampleResult(
        sample_index=sample_index,
        sample_id=str(result_data["sample_id"]),
        question=result_data["question"],
        scene_index=int(result_data["scene_index"]),
        setup_index=int(result_data.get("setup_index", 0)),
        repeat_index=int(result_data.get("repeat_index", result_data["scene_index"])),
        seed=seed,
        question_type=str(result_data.get("question_type") or QUESTION_TYPES[0]),
        answer_type=result_data["answer_type"],
        difficulty=result_data["difficulty"],
        difficulty_score=int(result_data["difficulty_score"]),
        requested_room_count=result_data.get("requested_room_count"),
        requested_door_count=result_data.get("requested_door_count"),
        requested_point_count=int(result_data.get("requested_point_count", result_data["point_count"])),
        num_rooms=int(result_data["num_rooms"]),
        num_openable_doors=int(result_data["num_openable_doors"]),
        point_count=int(result_data["point_count"]),
        question_payload=dict(result_data.get("question_payload") or {}),
        ground_truth=result_data["ground_truth"],
        predicted_answer=result_data["predicted_answer"],
        correct=bool(result_data["correct"]),
        invalid_response=bool(result_data["invalid_response"]),
        api_error=bool(result_data["api_error"]),
        api_error_message=result_data["api_error_message"],
        raw_response_text=result_data["raw_response_text"],
        image_paths=list(result_data["image_paths"]),
    )


def infer_loaded_answer_type(
    row: Dict[str, Any],
    meta_info: Dict[str, Any],
) -> str:
    answer_type = str(loaded_jsonl_value(row, meta_info, "answer_type", "") or "").strip()
    if answer_type:
        return answer_type

    question_type = str(loaded_jsonl_value(row, meta_info, "question_type", "") or "").strip()
    if question_type == QUESTION_TYPE_DOOR_OPEN:
        return ANSWER_TYPE_COLOR_LIST

    answer = row.get("gt_answer")
    if isinstance(answer, list):
        return ANSWER_TYPE_NAME_LIST
    if isinstance(answer, str):
        normalized = answer.strip().casefold()
        if normalized in {"yes", "no"}:
            return ANSWER_TYPE_YES_NO
        if answer.strip() in {"1", "2", "3", "4", "5"}:
            return ANSWER_TYPE_MULTIPLE_CHOICE
    raise ValueError(
        f"Question row {row.get('id')!r} is missing answer_type and has unsupported gt_answer={answer!r}."
    )


def run_loaded_question_rows(
    *,
    args: argparse.Namespace,
    question_jsonl: Path,
    output_base: Path,
    image_root: Path,
    api_key: str,
    base_url: str,
    model: str,
    selected_question_types: Tuple[str, ...] | None,
) -> List[SampleResult]:
    rows = read_question_jsonl(question_jsonl)
    if not rows:
        raise ValueError(f"Question JSONL has no data rows: {question_jsonl}")
    if selected_question_types is None:
        filtered_rows = list(rows)
    else:
        filtered_rows = []
        for row in rows:
            meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
            question_type = str(
                loaded_jsonl_value(
                    row,
                    meta_info,
                    "question_type",
                    (row.get("category") or [None, None, QUESTION_TYPES[0]])[2],
                )
            )
            if question_type in selected_question_types:
                filtered_rows.append(row)
    if not filtered_rows:
        if selected_question_types is None:
            raise ValueError(f"Question JSONL has no data rows: {question_jsonl}")
        selected_text = ", ".join(selected_question_types)
        raise ValueError(f"Question JSONL has no rows matching --question-types {selected_text}: {question_jsonl}")

    results: List[SampleResult] = []
    progress = tqdm(total=len(filtered_rows), desc="3D maze loaded benchmark")

    for sample_index, row in enumerate(filtered_rows):
        meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} must contain at least one image path.")
        image_paths: List[str] = []
        image_data_urls: List[str] = []
        for raw_image_path in images:
            image_path = resolve_question_image_path(str(raw_image_path), question_jsonl=question_jsonl)
            image_paths.append(to_relative_path(image_path, output_base))
            image_data_urls.append(image_file_to_data_url(image_path))

        row_question = str(row.get("question") or "")
        if not row_question:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} is missing question text.")
        answer_type = infer_loaded_answer_type(row, meta_info)
        answer = row.get("gt_answer")
        if is_meta_prompt(row_question):
            before_images, after_images = split_pure_prompt(row_question, image_count=len(image_paths))
            question = row_question
        else:
            before_images, after_images = build_user_prompt_blocks(row_question, answer_type)
            question = compose_pure_prompt(before_images, image_paths, after_images)

        sample_id = str(row.get("id") or f"{TASK_NAME}_loaded_{sample_index:04d}")
        pair_dir = image_root / sample_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_pure_prompt_file(pair_dir=pair_dir, before_images=before_images, image_paths=image_paths, after_images=after_images)

        raw_response_text = ""
        parsed_answer: Optional[Any] = None
        invalid_response = False
        api_error = False
        api_error_message: Optional[str] = None
        response_debug: Optional[Dict[str, Any]] = None

        if args.oracle:
            raw_response_text = json.dumps({"answer": answer}, ensure_ascii=False)
            parsed_answer = answer
        else:
            try:
                raw_response_text, response_debug, api_error_message = request_prediction(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    user_prompt_before_images=before_images,
                    user_prompt_after_images=after_images,
                    image_data_urls=image_data_urls,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                api_error_message = f"{type(exc).__name__}: {exc}"
                response_debug = {"exception": type(exc).__name__}
            api_error = api_error_message is not None
            parsed_answer = parse_answer(raw_response_text, answer_type) if not api_error else None
            invalid_response = not api_error and parsed_answer is None

        correct = answers_equal(
            parsed_answer,
            answer,
            answer_type,
        )
        write_prompt_files(
            pair_dir=pair_dir,
            before_images=before_images,
            after_images=after_images,
            image_paths=image_paths,
            ground_truth=answer,
            raw_response_text=raw_response_text,
            predicted_answer=parsed_answer,
            api_error_message=api_error_message,
            response_debug=response_debug,
        )

        result_data = {
            "sample_id": sample_id,
            "scene_index": int(meta_info.get("scene_index", meta_info.get("repeat_index", sample_index))),
            "setup_index": int(meta_info.get("setup_index", 0)),
            "repeat_index": int(meta_info.get("repeat_index", meta_info.get("scene_index", sample_index))),
            "seed": int(meta_info.get("seed", sample_index)),
            "question_type": str(
                loaded_jsonl_value(
                    row,
                    meta_info,
                    "question_type",
                    (row.get("category") or [None, None, QUESTION_TYPES[0]])[2],
                )
            ),
            "question": question,
            "answer_type": answer_type,
            "difficulty": str(loaded_jsonl_value(row, meta_info, "difficulty", "loaded")),
            "difficulty_score": int(loaded_jsonl_value(row, meta_info, "difficulty_score", 0)),
            "requested_room_count": meta_info.get("requested_room_count"),
            "requested_door_count": meta_info.get("requested_door_count"),
            "requested_point_count": int(meta_info.get("requested_point_count", meta_info.get("point_count", 2))),
            "num_rooms": int(meta_info.get("num_rooms", 0)),
            "num_openable_doors": int(meta_info.get("num_openable_doors", 0)),
            "point_count": int(meta_info.get("point_count", 2)),
            "question_payload": loaded_jsonl_value(row, meta_info, "question_payload", {}),
            "ground_truth": answer,
            "predicted_answer": parsed_answer,
            "correct": correct,
            "invalid_response": invalid_response,
            "api_error": api_error,
            "api_error_message": api_error_message,
            "raw_response_text": raw_response_text,
            "image_paths": image_paths,
        }
        results.append(build_sample_result(sample_index=sample_index, result_data=result_data))
        progress.update(1)
        progress.set_postfix(
            accuracy=f"{sum(r.correct for r in results) / len(results):.3f}",
            invalid=sum(r.invalid_response for r in results),
            api_errors=sum(r.api_error for r in results),
        )

    progress.close()
    return results


def process_oracle_scene(task: Dict[str, Any]) -> Dict[str, Any]:
    identity = getattr(current_process(), "_identity", ())
    worker_index = int(identity[0] - 1) if identity else 0
    scene_output = generate_benchmark_scene_output({**task, "worker_slot": worker_index})

    result_blueprints: List[Dict[str, Any]] = []
    for question_payload in scene_output["question_payloads"]:
        raw_response_text = json.dumps({"answer": question_payload["ground_truth"]}, ensure_ascii=False)
        correct = answers_equal(
            question_payload["ground_truth"],
            question_payload["ground_truth"],
            question_payload["answer_type"],
        )
        write_prompt_files(
            pair_dir=Path(question_payload["sample_dir"]),
            before_images=question_payload["before_images"],
            after_images=question_payload["after_images"],
            image_paths=question_payload["image_paths"],
            ground_truth=question_payload["ground_truth"],
            raw_response_text=raw_response_text,
            predicted_answer=question_payload["ground_truth"],
            api_error_message=None,
            response_debug=None,
        )
        result_blueprints.append(
            {
                "sample_id": str(question_payload["sample_id"]),
                "scene_index": int(question_payload["scene_index"]),
                "setup_index": int(question_payload["setup_index"]),
                "repeat_index": int(question_payload["repeat_index"]),
                "seed": int(question_payload["seed"]),
                "question_type": question_payload["question_type"],
                "question": question_payload["question"],
                "answer_type": question_payload["answer_type"],
                "difficulty": question_payload["difficulty"],
                "difficulty_score": int(question_payload["difficulty_score"]),
                "requested_room_count": question_payload["requested_room_count"],
                "requested_door_count": question_payload["requested_door_count"],
                "requested_point_count": int(question_payload["requested_point_count"]),
                "num_rooms": int(question_payload["num_rooms"]),
                "num_openable_doors": int(question_payload["num_openable_doors"]),
                "point_count": int(question_payload["point_count"]),
                "question_payload": question_payload["question_payload"],
                "ground_truth": question_payload["ground_truth"],
                "predicted_answer": question_payload["ground_truth"],
                "correct": correct,
                "invalid_response": False,
                "api_error": False,
                "api_error_message": None,
                "raw_response_text": raw_response_text,
                "image_paths": question_payload["image_paths"],
            }
        )

    gc.collect()
    return {
        "scene_index": int(scene_output["scene_index"]),
        "scene_record": scene_output["scene_record"],
        "result_blueprints": result_blueprints,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark InternVL3 on continuity_3d_maze.")
    parser.add_argument("--config", help="Optional JSON config file with api_key/base_url/model.")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of generated mazes per requested room/door/point setup; with --difficulty, total sampled scenes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
            "Comma-separated question types to benchmark: subset_choice, point_list, door_open."
        ),
    )
    parser.add_argument(
        "--difficulty",
        choices=tuple(name for name, *_rest in DIFFICULTY_SPECS),
        default="",
        help="Optional difficulty level for direct generation. When set, room/door/point setups are sampled from that level using --seed.",
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
    parser.add_argument("--room-spec-id")
    parser.add_argument("--door-state", choices=("generated", "open", "closed", "random"), default="random")
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
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes used for direct scene generation.")
    parser.add_argument(
        "--scene-timeout-sec",
        type=float,
        default=180.0,
        help="Optional per-scene generation timeout in direct generation mode.",
    )
    parser.add_argument(
        "--skip-failed-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip failed/timed-out generated scenes and keep sampling seeds until the requested scene count is reached.",
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
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--question-jsonl", default="", help="Load existing question.jsonl rows and images instead of generating new scenes.")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--oracle", action="store_true")
    return parser


def question_types_were_explicit(argv: Sequence[str] | None = None) -> bool:
    argv = list(sys.argv[1:] if argv is None else argv)
    return any(
        arg == "--question-types"
        or arg == "--question-type"
        or arg.startswith("--question-types=")
        or arg.startswith("--question-type=")
        for arg in argv
    )


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    explicit_question_types = question_types_were_explicit()
    selected_question_types = (
        parse_question_types(args.question_types)
        if question_jsonl is None or explicit_question_types
        else None
    )
    generation_question_types = selected_question_types or parse_question_types(args.question_types)
    if question_jsonl is None and args.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    if question_jsonl is None and args.max_seed_multiplier < 1:
        raise ValueError("max-seed-multiplier must be at least 1.")
    use_resilient_generation = question_jsonl is None and bool(args.skip_failed_scenes or args.scene_timeout_sec > 0)
    if question_jsonl is None and args.workers > 1 and not args.oracle and not use_resilient_generation:
        raise ValueError("Parallel workers are currently supported only with --oracle.")
    if question_jsonl is None and _MAZE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "continuity_3d_maze generation requires AI2-THOR/maze assets. "
            "Use --question-jsonl to load existing rendered samples without generating scenes."
        ) from _MAZE_IMPORT_ERROR
    if question_jsonl is None and args.scene_timeout_sec > 0:
        os.environ["PYTHONHASHSEED"] = python_hash_seed_for_base_seed(int(args.seed))

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    image_root = Path(args.image_root).resolve()
    output_base = output_json.parent
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    config_path, local_config = load_local_config(args.config)
    api_key = "" if args.oracle else resolve_config_value(
        args.api_key,
        env_keys=("INTERNVL3_API_KEY", "OPENAI_API_KEY"),
        config=local_config,
        config_key="api_key",
        default="",
    )
    base_url = resolve_config_value(
        args.base_url,
        env_keys=("INTERNVL3_BASE_URL", "OPENAI_BASE_URL"),
        config=local_config,
        config_key="base_url",
        default="https://chat.intern-ai.org.cn/api/v1",
    )
    model = resolve_config_value(
        args.model,
        env_keys=("INTERNVL3_MODEL",),
        config=local_config,
        config_key="model",
        default="internvl3.5-latest",
    )
    if not args.oracle and not api_key:
        raise ValueError(f"Missing API key. Checked CLI, env vars, and config file {config_path}.")

    results: List[SampleResult] = []
    scenes: List[Dict[str, Any]] = []
    skipped_scene_attempts = 0
    attempted_scene_count = 0
    if question_jsonl is not None:
        setups: List[Dict[str, int | None]] = []
        results = run_loaded_question_rows(
            args=args,
            question_jsonl=question_jsonl,
            output_base=output_base,
            image_root=image_root,
            api_key=api_key,
            base_url=base_url,
            model=model,
            selected_question_types=selected_question_types,
        )
    else:
        difficulty = str(getattr(args, "difficulty", "") or "").strip()
        setups = build_setups(args)
        scene_tasks: List[Dict[str, Any]] = []
        total_samples = 0
        if difficulty:
            for setup in setups:
                point_count = int(setup["point_count"])
                question_count = question_count_for_point_count(point_count, generation_question_types)
                scene_index = len(scene_tasks)
                scene_tasks.append(
                    {
                        "setup_index": scene_index,
                        "repeat_index": 0,
                        "scene_index": scene_index,
                        **setup,
                    }
                )
                total_samples += question_count
        else:
            for setup_index, setup in enumerate(setups):
                point_count = int(setup["point_count"])
                question_count = question_count_for_point_count(point_count, generation_question_types)
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
                    total_samples += question_count
        worker_config = build_benchmark_worker_config(
            args=args,
            output_base=output_base,
            image_root=image_root,
            total_scenes=len(scene_tasks),
        )
        if args.oracle and args.workers > 1 and not use_resilient_generation:
            progress = tqdm(total=total_samples, desc="3D maze benchmark")
            scene_outputs_by_index: Dict[int, Dict[str, Any]] = {}
            next_scene_index = 0
            sample_index = 0

            def flush_ready_scene_outputs() -> int:
                nonlocal next_scene_index, sample_index
                flushed = 0
                while next_scene_index in scene_outputs_by_index:
                    item = scene_outputs_by_index.pop(next_scene_index)
                    scenes.append(item["scene_record"])
                    for result_data in item["result_blueprints"]:
                        results.append(build_sample_result(sample_index=sample_index, result_data=result_data))
                        sample_index += 1
                    next_scene_index += 1
                    flushed += 1
                if flushed:
                    gc.collect()
                return flushed

            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = [
                    executor.submit(
                        process_oracle_scene,
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
                    progress.update(len(item["result_blueprints"]))
                    flush_ready_scene_outputs()

            progress.close()
            while flush_ready_scene_outputs():
                pass
        else:
            progress = tqdm(total=total_samples, desc="3D maze benchmark")
            sample_index = 0
            correct_count = 0
            invalid_count = 0
            api_error_count = 0

            def generated_image_data_urls(question_payload: Dict[str, Any]) -> List[str]:
                existing_data_urls = list(question_payload.get("image_data_urls") or [])
                if existing_data_urls:
                    return existing_data_urls
                data_urls: List[str] = []
                for image_path_text in question_payload["image_paths"]:
                    image_path = Path(str(image_path_text))
                    if not image_path.is_absolute():
                        image_path = output_base / image_path
                    data_urls.append(image_file_to_data_url(image_path.resolve()))
                return data_urls

            def evaluate_scene_output(item: Dict[str, Any]) -> None:
                nonlocal sample_index, correct_count, invalid_count, api_error_count
                scenes.append(item["scene_record"])
                for question_payload in item["question_payloads"]:
                    raw_response_text = ""
                    parsed_answer: Optional[Any] = None
                    invalid_response = False
                    api_error = False
                    api_error_message: Optional[str] = None
                    response_debug: Optional[Dict[str, Any]] = None

                    if args.oracle:
                        raw_response_text = json.dumps({"answer": question_payload["ground_truth"]}, ensure_ascii=False)
                        parsed_answer = question_payload["ground_truth"]
                    else:
                        write_pure_prompt_file(
                            pair_dir=Path(question_payload["sample_dir"]),
                            before_images=question_payload["before_images"],
                            image_paths=question_payload["image_paths"],
                            after_images=question_payload["after_images"],
                        )
                        try:
                            raw_response_text, response_debug, api_error_message = request_prediction(
                                api_key=api_key,
                                base_url=base_url,
                                model=model,
                                user_prompt_before_images=question_payload["before_images"],
                                user_prompt_after_images=question_payload["after_images"],
                                image_data_urls=generated_image_data_urls(question_payload),
                                max_tokens=args.max_tokens,
                            )
                        except Exception as exc:
                            api_error_message = f"{type(exc).__name__}: {exc}"
                            response_debug = {"exception": type(exc).__name__}
                        api_error = api_error_message is not None
                        parsed_answer = parse_answer(raw_response_text, question_payload["answer_type"]) if not api_error else None
                        invalid_response = not api_error and parsed_answer is None

                    correct = answers_equal(
                        parsed_answer,
                        question_payload["ground_truth"],
                        question_payload["answer_type"],
                    )
                    write_prompt_files(
                        pair_dir=Path(question_payload["sample_dir"]),
                        before_images=question_payload["before_images"],
                        after_images=question_payload["after_images"],
                        image_paths=question_payload["image_paths"],
                        ground_truth=question_payload["ground_truth"],
                        raw_response_text=raw_response_text,
                        predicted_answer=parsed_answer,
                        api_error_message=api_error_message,
                        response_debug=response_debug,
                    )

                    result_data = {
                        "sample_id": str(question_payload["sample_id"]),
                        "scene_index": int(question_payload["scene_index"]),
                        "setup_index": int(question_payload["setup_index"]),
                        "repeat_index": int(question_payload["repeat_index"]),
                        "seed": int(question_payload["seed"]),
                        "question_type": question_payload["question_type"],
                        "question": question_payload["question"],
                        "answer_type": question_payload["answer_type"],
                        "difficulty": question_payload["difficulty"],
                        "difficulty_score": int(question_payload["difficulty_score"]),
                        "requested_room_count": question_payload["requested_room_count"],
                        "requested_door_count": question_payload["requested_door_count"],
                        "requested_point_count": int(question_payload["requested_point_count"]),
                        "num_rooms": int(question_payload["num_rooms"]),
                        "num_openable_doors": int(question_payload["num_openable_doors"]),
                        "point_count": int(question_payload["point_count"]),
                        "question_payload": question_payload["question_payload"],
                        "ground_truth": question_payload["ground_truth"],
                        "predicted_answer": parsed_answer,
                        "correct": correct,
                        "invalid_response": invalid_response,
                        "api_error": api_error,
                        "api_error_message": api_error_message,
                        "raw_response_text": raw_response_text,
                        "image_paths": question_payload["image_paths"],
                    }
                    results.append(build_sample_result(sample_index=sample_index, result_data=result_data))
                    sample_index += 1
                    correct_count += int(correct)
                    invalid_count += int(invalid_response)
                    api_error_count += int(api_error)
                    progress.update(1)
                    progress.set_postfix(
                        accuracy=f"{correct_count / len(results):.3f}",
                        invalid=invalid_count,
                        api_errors=api_error_count,
                    )

                gc.collect()

            try:
                if not args.skip_failed_scenes:
                    for scene_task in scene_tasks:
                        attempted_scene_count += 1
                        item = generate_benchmark_scene_output_with_timeout(
                            {
                                **worker_config,
                                "worker_slot": 0,
                                **scene_task,
                            },
                            timeout_sec=float(args.scene_timeout_sec),
                        )
                        evaluate_scene_output(item)
                else:
                    candidate_limit = int(args.repeats) * int(args.max_seed_multiplier)
                    seed_count = (1 if difficulty else len(setups)) * candidate_limit
                    candidate_seeds = deterministic_seed_sequence(int(args.seed), seed_count)
                    candidate_setups = (
                        sample_difficulty_setups(
                            difficulty=difficulty,
                            count=candidate_limit,
                            seed=int(args.seed),
                            question_types=generation_question_types,
                        )
                        if difficulty
                        else []
                    )
                    executor = (
                        ProcessPoolExecutor(max_workers=int(args.workers))
                        if int(args.workers) > 1
                        else None
                    )
                    scene_index = 0
                    try:
                        difficulty_setup_indices = [0] if difficulty else list(range(len(setups)))
                        for setup_index in difficulty_setup_indices:
                            setup = {} if difficulty else setups[setup_index]
                            success_count = 0
                            candidate_index = 0
                            setup_target_count = int(args.repeats)
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
                                            item = run_benchmark_candidate_task(
                                                scene_task,
                                                float(args.scene_timeout_sec),
                                            )
                                            wave_results.append((int(scene_task["candidate_index"]), scene_task, item, None))
                                        except BaseException as exc:
                                            wave_results.append((int(scene_task["candidate_index"]), scene_task, None, exc))
                                else:
                                    future_to_task = {
                                        executor.submit(
                                            run_benchmark_candidate_task,
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
                                    rewrite_benchmark_scene_output_indices(
                                        item=item or {},
                                        output_base=output_base,
                                        scene_index=scene_index,
                                        setup_index=final_setup_index,
                                        repeat_index=final_repeat_index,
                                    )
                                    evaluate_scene_output(item or {})
                                    scene_index += 1
                                    success_count += 1
                                    progress.set_postfix(setup=setup_index, skipped=skipped_scene_attempts)

                                candidate_index += wave_size

                            if success_count < setup_target_count:
                                raise RuntimeError(
                                    f"Only generated {success_count}/{setup_target_count} scenes for setup_index={setup_index} "
                                    f"after trying {candidate_limit} candidate seeds. Increase --max-seed-multiplier "
                                    "or relax the room/door/point setup."
                                )
                    finally:
                        if executor is not None:
                            executor.shutdown(wait=True, cancel_futures=True)
            finally:
                progress.close()

    question_jsonl_path, model_answer_jsonl_path = write_meta_jsonl(
        output_dir=output_json.parent,
        model_id="oracle" if args.oracle else model,
        results=results,
        mode="upsert" if question_jsonl is None else "write",
        write_question_jsonl=question_jsonl is None,
    )
    output_question_jsonl_path = question_jsonl_path if question_jsonl is None else question_jsonl

    sample_rows = [asdict(result) for result in results]
    payload = {
        "task": TASK_NAME,
        "category": "continuity",
        "level": "reasoning",
        "oracle": bool(args.oracle),
        "question_types": None if selected_question_types is None else list(selected_question_types),
        "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        "repeats_per_setup": args.repeats if question_jsonl is None else None,
        "setup_count": len(setups) if question_jsonl is None else None,
        "setups": setups if question_jsonl is None else None,
        "door_state_mode": args.door_state,
        "workers": args.workers,
        "scene_timeout_sec": float(args.scene_timeout_sec) if question_jsonl is None else None,
        "skip_failed_scenes": bool(args.skip_failed_scenes) if question_jsonl is None else None,
        "skipped_scene_attempts": skipped_scene_attempts if question_jsonl is None else None,
        "attempted_scene_count": attempted_scene_count if question_jsonl is None else None,
        "question_jsonl": to_relative_path(output_question_jsonl_path, output_json.parent),
        "model_answer_jsonl": to_relative_path(model_answer_jsonl_path, output_json.parent),
        "generated_jsonl_write_mode": "upsert" if question_jsonl is None else "load_only",
        "summary": summarize_results(results),
        "samples": sample_rows,
        "scenes": scenes,
    }
    write_json(output_json, payload)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(results[0]).keys()) if results else ["sample_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["question_payload"] = json.dumps(row["question_payload"], ensure_ascii=False)
            row["image_paths"] = json.dumps(row["image_paths"], ensure_ascii=False)
            writer.writerow(row)

    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_benchmark(args)
    summary = payload["summary"]
    print(
        f"accuracy={summary['accuracy']:.3f} "
        f"correct={summary['correct']}/{summary['total_samples']}"
    )
    print(f"Wrote JSON summary to {Path(args.output_json).resolve()}")
    print(f"Wrote CSV summary to {Path(args.output_csv).resolve()}")
    if payload.get("generated_jsonl_write_mode") == "load_only":
        print(f"Loaded question JSONL from {payload.get('load_question_jsonl')}")
        print(f"Wrote model-answer JSONL to {(Path(args.output_json).resolve().parent / 'model_answer.jsonl')}")
        return
    if payload.get("generated_jsonl_write_mode") == "upsert":
        jsonl_action = "Upserted"
    else:
        jsonl_action = "Wrote"
    print(f"{jsonl_action} question JSONL to {(Path(args.output_json).resolve().parent / 'question.jsonl')}")
    print(f"{jsonl_action} model-answer JSONL to {(Path(args.output_json).resolve().parent / 'model_answer.jsonl')}")


if __name__ == "__main__":
    main()
