from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOPOBENCH_EVAL_ROOT = REPO_ROOT.parent / "topobench_eval"
if str(TOPOBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPOBENCH_EVAL_ROOT))
GYM_DIR = PROJECT_ROOT / "gym"
if str(GYM_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_DIR))
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from env import (  # noqa: E402
    DIFFICULTY_SETUPS,
    INVALID_ACTION,
    ContinuityPipeEnv,
    action_to_xy,
    difficulty_label_for_solution_ticks,
    grid_size_for_difficulty,
    xy_to_action,
)
from internvl3_config import load_local_config, resolve_config_value  # noqa: E402
from topobench_eval.metadata_utils import load_task_intro_line  # noqa: E402
from jsonl_export import (  # noqa: E402
    flatten_answer_value,
    interactive_trajectory_state,
    resolve_existing_path,
    to_relative_path as jsonl_relpath,
    write_jsonl,
)
from tqdm.auto import tqdm  # noqa: E402

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. Install it with: pip install requests"
    ) from exc


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_continuity_pipe.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_continuity_pipe.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0
DEFAULT_SEED_START = 1
DEFAULT_DIFFICULTIES = "easy,medium,hard"
DEFAULT_SOLUTION_STEPS = ""
STATE_CAPTION_FONT_SIZE = 34
STATE_CAPTION_TOP_MARGIN = 14


@dataclass
class EpisodeResult:
    episode_index: int
    episode_id: str
    grid_size: int
    difficulty: str
    repeat_index: int
    seed: int
    source_index: int
    solution_ticks: int
    step_budget: int
    success: bool
    terminated: bool
    truncated: bool
    total_steps: int
    illegal_moves: int
    invalid_responses: int
    api_errors: int
    final_reward: float
    final_reason: Optional[str]
    steps: List[Dict[str, Any]]


def build_initial_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    return {
        "gridSize": state.get("gridSize"),
        "sourceIndex": state.get("sourceIndex"),
        "source": state.get("source"),
        "solvedMasks": list(state.get("solvedMasks") or []),
        "rotations": list(state.get("rotations") or []),
        "currentMasks": list(state.get("currentMasks") or []),
        "connectedIndices": list(state.get("connectedIndices") or []),
        "connectedCount": state.get("connectedCount"),
        "totalPipes": state.get("totalPipes"),
        "junctionCount": state.get("junctionCount"),
        "activePipeIndices": list(state.get("activePipeIndices") or []),
        "solutionTicks": state.get("solutionTicks"),
        "solutionSteps": state.get("solutionSteps", state.get("solutionTicks")),
        "difficulty": state.get("difficulty"),
        "maxSteps": state.get("maxSteps"),
    }


def build_episode_id(*, grid_size: int, solution_steps_spec: str, seed: int) -> str:
    return f"continuity_pipe_grid_{grid_size}_solution_steps_{solution_steps_spec}_seed_{seed}"


def parse_difficulty_list(spec: str) -> List[str]:
    difficulties: List[str] = []
    seen: set[str] = set()
    for raw_part in str(spec).split(","):
        difficulty = raw_part.strip().lower()
        if not difficulty:
            continue
        if difficulty not in DIFFICULTY_SETUPS:
            raise ValueError(f"Unsupported --difficulties value: {difficulty!r}")
        if difficulty not in seen:
            difficulties.append(difficulty)
            seen.add(difficulty)
    if not difficulties:
        raise ValueError(f"Empty --difficulties spec: {spec!r}")
    return difficulties


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


def load_episode_specs(question_jsonl: Path) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for row_index, row in enumerate(read_question_jsonl(question_jsonl)):
        meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        initial_state = meta_info.get("initial_state") if isinstance(meta_info.get("initial_state"), dict) else {}
        reset_config = initial_state.get("reset_config") if isinstance(initial_state.get("reset_config"), dict) else {}
        if not reset_config:
            raise ValueError(f"Question row {row.get('id', row_index)!r} is missing meta_info.initial_state.reset_config.")
        grid_size = int(reset_config.get("gridSize", initial_state.get("gridSize", 3)))
        seed = int(reset_config.get("seed", meta_info.get("seed", row_index)))
        repeat_index = int(meta_info.get("repeat_index", row_index))
        difficulty = str(meta_info.get("difficulty", initial_state.get("difficulty", "loaded")))
        solution_steps_spec = str(meta_info.get("solution_steps_spec") or initial_state.get("solutionSteps") or "loaded")
        specs.append(
            {
                "episode_id": str(row.get("id") or build_episode_id(
                    grid_size=grid_size,
                    solution_steps_spec=solution_steps_spec,
                    seed=seed,
                )),
                "repeat_index": repeat_index,
                "seed": seed,
                "grid_size": grid_size,
                "difficulty": difficulty,
                "solution_steps_spec": solution_steps_spec,
                "reset_options": {"frontend_config": reset_config},
            }
        )
    return specs


def write_meta_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    results: Sequence[EpisodeResult],
    write_question_jsonl: bool = True,
) -> Tuple[Path, Path]:
    config_rel = jsonl_relpath(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    question_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    for result in results:
        first_step = result.steps[0] if result.steps else None
        initial_state = build_initial_state_snapshot(first_step.get("state_before_action", {})) if first_step else {}
        if initial_state:
            reset_config = {
                "gridSize": int(result.grid_size),
                "seed": int(result.seed),
                "sourceIndex": int(result.source_index),
                "solvedMasks": list(initial_state.get("solvedMasks") or []),
                "rotations": list(initial_state.get("rotations") or []),
                "maxSteps": int(result.step_budget),
                "difficulty": result.difficulty,
                "targetDifficulty": result.difficulty,
            }
            setup = DIFFICULTY_SETUPS.get(result.difficulty)
            if setup:
                reset_config.update(
                    {
                        "minSolutionTicks": int(setup["solution_ticks"][0]),
                        "maxSolutionTicks": int(setup["solution_ticks"][1]),
                        "minActivePipeCount": int(setup["active_pipe_count"][0]),
                        "maxActivePipeCount": int(setup["active_pipe_count"][1]),
                        "minJunctionCount": int(setup["junction_count"][0]),
                        "maxJunctionCount": int(setup["junction_count"][1]),
                    }
                )
            initial_state["reset_config"] = reset_config
        initial_images = (
            [
                jsonl_relpath(
                    resolve_existing_path(
                        first_step["saved_current_image_path"],
                        output_dir,
                        PROJECT_ROOT,
                        DEFAULT_OUTPUT_ROOT,
                    ),
                    base_dir=output_dir,
                )
            ]
            if first_step
            else []
        )
        level = f"grid_{result.grid_size}x{result.grid_size}"
        question_rows.append(
            {
                "id": result.episode_id,
                "category": ["continuity", "continuity_pipe", "interactive"],
                "type": "interactive",
                "meta_info": {
                    "task_name": "continuity_pipe",
                    "config": config_rel,
                    "level": level,
                    "seed": result.seed,
                    "repeat_index": result.repeat_index,
                    "difficulty": result.difficulty,
                    "solution_steps_spec": result.difficulty if result.difficulty in DIFFICULTY_SETUPS else str(result.solution_ticks),
                    "active_pipe_count": initial_state.get("totalPipes"),
                    "junction_count": initial_state.get("junctionCount"),
                    "max_actions_per_traj": result.step_budget,
                    "initial_state": initial_state,
                    "legal_action_format": '{"answer":{"x": {x}, "y": {y}}}',
                },
                "images": initial_images,
            }
        )
        answer_rows.append(
            {
                "id": result.episode_id,
                "category": ["continuity", "continuity_pipe", "interactive"],
                "type": "interactive",
                "meta_info": {
                    "task_name": "continuity_pipe",
                    "config": config_rel,
                    "level": level,
                    "seed": result.seed,
                    "repeat_index": result.repeat_index,
                    "difficulty": result.difficulty,
                    "grid_size": result.grid_size,
                    "solution_ticks": result.solution_ticks,
                    "active_pipe_count": initial_state.get("totalPipes"),
                    "junction_count": initial_state.get("junctionCount"),
                    "model_id": model_id,
                    "success": bool(result.success),
                    "final_reason": result.final_reason,
                    "total_steps": int(result.total_steps),
                },
                "trajectory": [
                    {
                        "step_index": int(step["step_index"]),
                        "current_images": [
                            jsonl_relpath(
                                resolve_existing_path(
                                    step["saved_current_image_path"],
                                    output_dir,
                                    PROJECT_ROOT,
                                    DEFAULT_OUTPUT_ROOT,
                                ),
                                base_dir=output_dir,
                            )
                        ],
                        "state": interactive_trajectory_state(
                            trajectory_index=trajectory_index,
                            total_steps=len(result.steps),
                            success=bool(result.success),
                        ),
                        "answer": flatten_answer_value(step.get("predicted_action")),
                        "invalid_response": bool(step.get("invalid_response", False)),
                        "api_error": bool(step.get("api_error_message")),
                        "illegal": bool(step.get("illegal", False)),
                        "raw_response_text": step.get("raw_response_text", ""),
                    }
                    for trajectory_index, step in enumerate(result.steps)
                ],
            }
        )

    question_path = output_dir / "question.jsonl"
    answer_path = output_dir / "model_answer.jsonl"
    if write_question_jsonl:
        write_jsonl(question_path, question_rows)
    write_jsonl(answer_path, answer_rows)
    return question_path, answer_path


class StepImageWriter:
    def __init__(self, *, root_dir: Path, prompt_path_base: Optional[Path] = None) -> None:
        self.root_dir = root_dir
        self.prompt_path_base = prompt_path_base or root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def episode_dir(self, *, episode_id: str) -> Path:
        directory = self.root_dir / episode_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_png(
        self,
        *,
        episode_id: str,
        step_index: int,
        image_kind: str,
        png_bytes: bytes,
    ) -> Path:
        output_path = self.episode_dir(episode_id=episode_id) / (
            f"step_{step_index:04d}_{image_kind}.png"
        )
        output_path.write_bytes(png_bytes)
        return output_path

    @staticmethod
    def _serialize_message_content(content: Any, *, path_base: Path) -> List[str]:
        if isinstance(content, str):
            text = content.strip()
            return [text] if text else []
        if not isinstance(content, list):
            if content is None:
                return []
            return [json.dumps(content, ensure_ascii=False)]

        blocks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    blocks.append(text)
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    blocks.append(text)
            elif item_type == "image_path":
                raw_path = str(item.get("path", "")).strip()
                if raw_path:
                    blocks.extend(("[Image]", os.path.relpath(Path(raw_path), path_base)))
            else:
                serialized = json.dumps(item, ensure_ascii=False).strip()
                if serialized:
                    blocks.append(serialized)
        return blocks

    def save_prompt_debug(
        self,
        *,
        episode_id: str,
        step_index: int,
        prompt_messages: Sequence[Dict[str, Any]],
        ground_truth: Optional[List[int]],
        raw_response_text: str,
        parsed_answer: Optional[List[int]],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]],
    ) -> Path:
        output_path = self.episode_dir(episode_id=episode_id) / (
            f"step_{step_index:04d}_prompt_debug.txt"
        )
        ground_truth_text = "<none>" if ground_truth is None else _coord_answer_to_json(ground_truth)
        parsed_text = "<none>" if parsed_answer is None else _coord_answer_to_json(parsed_answer)
        blocks: List[str] = ["[prompt_messages]"]
        for message in prompt_messages:
            role = str(message.get("role", "message"))
            blocks.append(f"[{role}]")
            blocks.extend(self._serialize_message_content(message.get("content"), path_base=self.prompt_path_base))
        blocks.extend(
            [
                "[ground_truth]",
                ground_truth_text,
                "[model_response]",
                raw_response_text or "<empty>",
                "[parsed_answer]",
                parsed_text,
                "[api_error]",
                api_error_message or "<none>",
                "[response_debug]",
                json.dumps(response_debug, indent=2, ensure_ascii=False) if response_debug is not None else "<none>",
            ]
        )
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return output_path

    def save_pure_prompt(
        self,
        *,
        episode_id: str,
        step_index: int,
        prompt_messages: Sequence[Dict[str, Any]],
    ) -> Path:
        output_path = self.episode_dir(episode_id=episode_id) / (
            f"step_{step_index:04d}_pure_prompt.txt"
        )
        blocks: List[str] = []
        for message in prompt_messages:
            blocks.extend(self._serialize_message_content(message.get("content"), path_base=self.prompt_path_base))
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return output_path


def to_relative_path(path: Path, *, base: Path = PROJECT_ROOT) -> str:
    return os.path.relpath(path, base)


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = [int(part)]
        for value in candidates:
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return values


def parse_solution_step_specs(spec: str) -> List[Tuple[int, Optional[int], str]]:
    text = str(spec).strip().lower()
    if not text:
        text = DEFAULT_SOLUTION_STEPS
    if not text:
        return []
    specs: List[Tuple[int, Optional[int], str]] = []
    seen: set[str] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.endswith("+"):
            minimum = int(part[:-1])
            label = f"{minimum}+"
            candidates: List[Tuple[int, Optional[int], str]] = [(minimum, None, label)]
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            minimum = int(start_text)
            maximum = int(end_text) if end_text.strip() else None
            if maximum is None:
                label = f"{minimum}+"
                candidates = [(minimum, None, label)]
            else:
                if maximum < minimum:
                    raise ValueError(f"Invalid --solution-steps range: {spec!r}")
                candidates = [(value, value, str(value)) for value in range(minimum, maximum + 1)]
        else:
            value = int(part)
            candidates = [(value, value, str(value))]
        for minimum, maximum, label in candidates:
            if label in seen:
                continue
            specs.append((minimum, maximum, label))
            seen.add(label)
    if not specs:
        raise ValueError(f"Empty --solution-steps spec: {spec!r}")
    return specs


def default_grid_sizes() -> List[int]:
    return [4, 5]


def difficulty_definition_text() -> str:
    parts = []
    for difficulty in ("easy", "medium", "hard"):
        setup = DIFFICULTY_SETUPS[difficulty]
        grid_min, grid_max = setup["grid_size"]
        active_min, active_max = setup["active_pipe_count"]
        junction_min, junction_max = setup["junction_count"]
        tick_min, tick_max = setup["solution_ticks"]
        grid_text = f"{grid_min}x{grid_min}" if grid_min == grid_max else f"{grid_min}x{grid_min}-{grid_max}x{grid_max}"
        parts.append(
            f"{difficulty}: grid={grid_text}, pipes={active_min}-{active_max}, "
            f"junctions={junction_min}-{junction_max}, oracle_steps={tick_min}-{tick_max}"
        )
    return "; ".join(parts)


def load_state_caption_font(size: int):
    for font_name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial Bold.ttf",
        "Arial.ttf",
        "Arial Bold",
        "Arial",
        "Helvetica.ttc",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_state_caption(png_bytes: bytes, caption_text: str) -> bytes:
    with Image.open(io.BytesIO(png_bytes)) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = load_state_caption_font(STATE_CAPTION_FONT_SIZE)
        bbox = draw.textbbox((0, 0), caption_text, font=font)
        text_width = bbox[2] - bbox[0]
        x = max(0, (image.width - text_width) // 2)
        draw.text((x, STATE_CAPTION_TOP_MARGIN), caption_text, fill="white", font=font)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _coord_answer_to_json(answer: Sequence[int]) -> str:
    return json.dumps({"answer": {"x": int(answer[0]), "y": int(answer[1])}}, ensure_ascii=False, separators=(",", ":"))


def format_legal_actions_line(
    legal_actions: Sequence[int],
    grid_size: int,
    *,
    include_legal_actions_in_prompt: bool = True,
) -> str:
    if not include_legal_actions_in_prompt:
        return "Legal actions for this turn are not provided; infer one non-empty pipe cell coordinate from the current state."
    if not legal_actions:
        return "Legal actions for this turn: <none>."
    coords = [action_to_xy(action, grid_size) for action in legal_actions]
    return "Legal actions for this turn: " + ", ".join(f"(x={coord['x']}, y={coord['y']})" for coord in coords) + "."


def build_initial_prompt_texts(
    *,
    grid_size: int,
    legal_actions: Sequence[int],
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str]:
    legal_rule = (
        "3. A legal action must be one of the listed (x, y) coordinates for the current state."
        if include_legal_actions_in_prompt
        else "3. A legal action must choose one non-empty pipe cell visible in the current state."
    )
    before_image_lines = [
        "[Task]",
        load_task_intro_line(
            PROJECT_ROOT,
            default=(
                "You are solving Continuity Pipe.\n"
                "In this task, you must rotate pipe pieces 90 degrees clockwise "
                "per turn until every pipe connects back to the green source."
            ),
        ),
        "The board is a square grid. Some cells contain rotatable pipe pieces and some cells may be empty.",
        "The green source pipe is the starting source. Pipes connected to the source are green; pipes not connected to the source are blue.",
        "Your goal is to rotate every non-empty pipe cell until every pipe on the board is connected to the green source.",
        "The x coordinates are shown above the grid and the y coordinates are shown on the left side of the grid.",
        "",
        "[Rules]",
        "1. At each turn, choose exactly one non-empty pipe cell.",
        "2. The selected pipe rotates clockwise by 90 degrees.",
        legal_rule,
        "4. Empty cells are not legal actions.",
        "5. The task succeeds when every pipe is connected to the source.",
        "6. At each turn, output exactly one next action for the current state.",
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer":{"x": {x}, "y": {y}}} and nothing else.',
        'Replace {x} and {y} with the selected legal grid coordinate, wrapped inside the "answer" field.',
        "",
        "[Current Task]",
        "The current state image is shown below.",
    ]
    after_image_lines = [
        f"The board size is {grid_size}x{grid_size}.",
        format_legal_actions_line(
            legal_actions,
            grid_size,
            include_legal_actions_in_prompt=include_legal_actions_in_prompt,
        ),
        "To solve this task, output the next legal action for the current state.",
    ]
    return "\n".join(before_image_lines), "\n".join(after_image_lines)


def build_followup_prompt_texts(
    *,
    grid_size: int,
    legal_actions: Sequence[int],
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str]:
    before_image_lines = [
        "[Current Task]",
        "The updated current state image is shown below.",
    ]
    after_image_lines = [
        format_legal_actions_line(
            legal_actions,
            grid_size,
            include_legal_actions_in_prompt=include_legal_actions_in_prompt,
        ),
        "To solve this task, output the next legal action for the current state.",
    ]
    return "\n".join(before_image_lines), "\n".join(after_image_lines)


def build_user_content(*, before_image_text: str, image_data_url: str, after_image_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": before_image_text}]
    content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    if after_image_text:
        content.append({"type": "text", "text": after_image_text})
    return content


def build_debug_user_content(*, before_image_text: str, image_path: Path, after_image_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": before_image_text}]
    content.append({"type": "image_path", "path": str(image_path)})
    if after_image_text:
        content.append({"type": "text", "text": after_image_text})
    return content


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
        for key in ("text", "content", "value", "reasoning_content", "output_text", "refusal"):
            if key in value:
                fragments.extend(_collect_text_fragments(value.get(key)))
        if fragments:
            return fragments
        block_type = value.get("type")
        if isinstance(block_type, str) and "text" in block_type.lower():
            for nested_key, nested_value in value.items():
                if nested_key != "type":
                    fragments.extend(_collect_text_fragments(nested_value))
        return fragments
    return []


def normalize_message_content(content: Any) -> str:
    fragments = _collect_text_fragments(content)
    if fragments:
        return "\n".join(fragments).strip()
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return ""
    return str(content).strip()


def extract_response_text(payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[str]]:
    debug_info: Dict[str, Any] = {
        "response_payload": payload,
        "response_content_source": None,
        "finish_reason": None,
    }
    if not isinstance(payload, dict):
        return "", debug_info, "Unexpected response payload type."
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", debug_info, "Unexpected response payload: missing choices."
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        debug_info["first_choice_raw"] = first_choice
        return "", debug_info, "Unexpected response payload: first choice is not an object."

    message = first_choice.get("message")
    message = message if isinstance(message, dict) else {}
    delta = first_choice.get("delta")
    delta = delta if isinstance(delta, dict) else {}

    debug_info["finish_reason"] = first_choice.get("finish_reason")
    candidates = [
        ("message.content", message.get("content")),
        ("choice.text", first_choice.get("text")),
        ("choice.content", first_choice.get("content")),
        ("delta.content", delta.get("content")),
    ]
    normalized_candidates: Dict[str, str] = {}
    for source_name, candidate in candidates:
        normalized_text = normalize_message_content(candidate)
        normalized_candidates[source_name] = normalized_text
        if normalized_text and debug_info["response_content_source"] is None:
            debug_info["response_content_source"] = source_name
            debug_info["normalized_candidates"] = normalized_candidates
            return normalized_text, debug_info, None
    debug_info["normalized_candidates"] = normalized_candidates
    return "", debug_info, None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _validate_xy(x: Optional[int], y: Optional[int], grid_size: int) -> Optional[List[int]]:
    if x is None or y is None:
        return None
    if 0 <= int(x) < grid_size and 0 <= int(y) < grid_size:
        return [int(x), int(y)]
    return None


def _parse_action_value(value: Any, grid_size: int) -> Optional[List[int]]:
    if isinstance(value, dict):
        parsed = _validate_xy(_coerce_int(value.get("x")), _coerce_int(value.get("y")), grid_size)
        if parsed is not None:
            return parsed
        for key in ("action_index", "actionIndex", "cell_index", "cellIndex", "index", "action", "cell"):
            action = _coerce_int(value.get(key))
            if action is not None and 0 <= action < grid_size * grid_size:
                coord = action_to_xy(action, grid_size)
                return [coord["x"], coord["y"]]
        return None
    if isinstance(value, list):
        if len(value) >= 2:
            return _validate_xy(_coerce_int(value[0]), _coerce_int(value[1]), grid_size)
        if len(value) == 1:
            return _parse_action_value(value[0], grid_size)
    action = _coerce_int(value)
    if action is not None and 0 <= action < grid_size * grid_size:
        coord = action_to_xy(action, grid_size)
        return [coord["x"], coord["y"]]
    return None


def parse_action(text: str, grid_size: int) -> Optional[List[int]]:
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("answer", "move", "action", "response"):
            parsed = _parse_action_value(payload.get(key), grid_size)
            if parsed is not None:
                return parsed
        parsed = _parse_action_value(payload, grid_size)
        if parsed is not None:
            return parsed
    elif payload is not None:
        parsed = _parse_action_value(payload, grid_size)
        if parsed is not None:
            return parsed

    lowered = text.casefold()
    brace_matches = re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL)
    for brace_text in reversed(brace_matches):
        x_match = re.search(r'"?x"?\s*[:=\-]\s*(-?\d+)', brace_text)
        y_match = re.search(r'"?y"?\s*[:=\-]\s*(-?\d+)', brace_text)
        parsed = _validate_xy(
            int(x_match.group(1)) if x_match else None,
            int(y_match.group(1)) if y_match else None,
            grid_size,
        )
        if parsed is not None:
            return parsed

    pair_patterns = [
        r"x\s*[:=\-]\s*(-?\d+)[^\d-]+y\s*[:=\-]\s*(-?\d+)",
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    ]
    for pattern in pair_patterns:
        matches = re.findall(pattern, lowered, flags=re.DOTALL)
        for x_text, y_text in reversed(matches):
            parsed = _validate_xy(int(x_text), int(y_text), grid_size)
            if parsed is not None:
                return parsed

    for pattern in (
        r"(?:action(?:_index)?|cell(?:_index)?|index)\s*[:=\-]?\s*(-?\d+)",
        r"(?:choose|select|rotate)\s+(?:cell\s*)?(-?\d+)",
    ):
        matches = re.findall(pattern, lowered, flags=re.DOTALL)
        for candidate in reversed(matches):
            action = int(candidate)
            if 0 <= action < grid_size * grid_size:
                coord = action_to_xy(action, grid_size)
                return [coord["x"], coord["y"]]

    integer_matches = re.findall(r"-?\d+", lowered)
    if len(integer_matches) >= 2:
        parsed = _validate_xy(int(integer_matches[-2]), int(integer_matches[-1]), grid_size)
        if parsed is not None:
            return parsed
    return None


def build_oracle_action(state: Dict[str, Any]) -> Optional[List[int]]:
    ticks = [int(value) for value in state.get("targetRotationTicks", [])]
    grid_size = int(state.get("gridSize", 0))
    for action_index, remaining in enumerate(ticks):
        if remaining > 0:
            coord = action_to_xy(action_index, grid_size)
            return [coord["x"], coord["y"]]
    return None


class InternVL3Policy:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        api_retries: int,
        retry_sleep_seconds: float,
        requests_per_minute: float,
        request_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.api_retries = int(api_retries)
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.requests_per_minute = float(requests_per_minute)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._session = requests.Session()
        self._next_request_not_before = 0.0
        self._min_request_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0

    def _wait_rate_limit(self) -> None:
        if self._min_request_interval <= 0:
            return
        now = time.monotonic()
        sleep_seconds = max(0.0, self._next_request_not_before - now)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        self._next_request_not_before = time.monotonic() + self._min_request_interval

    def predict_messages(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        grid_size: int,
    ) -> tuple[Optional[List[int]], str, Optional[str], bool, Dict[str, Any]]:
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": list(messages),
        }
        if self.max_tokens > 0:
            normalized_model = self.model.strip().lower()
            token_field = "max_completion_tokens" if normalized_model.startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens"
            request_payload[token_field] = self.max_tokens
        endpoint = f"{self.base_url}/chat/completions"
        last_error: Optional[str] = None
        for attempt in range(self.api_retries + 1):
            try:
                self._wait_rate_limit()
                response = self._session.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.request_timeout_seconds,
                )
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text.strip() or 'request failed'}"
                    if response.status_code >= 500 and attempt < self.api_retries:
                        time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                        continue
                    return None, "", last_error, False, {"http_status": response.status_code}
                payload = response.json()
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.api_retries:
                    time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                    continue
                return None, "", last_error, False, {}

            response_text, response_debug, extraction_error = extract_response_text(payload)
            response_debug["request_message_count"] = len(messages)
            response_debug["request_roles"] = [str(message.get("role", "")) for message in messages]
            if extraction_error is not None:
                return None, response_text, None, True, response_debug
            parsed = parse_action(response_text, grid_size)
            if parsed is None:
                response_debug["parse_error"] = "Failed to parse model action."
                return None, response_text, None, True, response_debug
            return parsed, response_text, None, False, response_debug

        return None, "", last_error or "Unknown API failure.", False, {}


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def summarize_results(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    count = len(results)
    successes = sum(1 for result in results if result.success)
    success_steps = [result.total_steps for result in results if result.success]
    return {
        "total_episodes": count,
        "successes": successes,
        "success_rate": (successes / count) if count else 0.0,
        "avg_solution_ticks": mean_or_zero([result.solution_ticks for result in results]) if count else 0.0,
        "avg_steps_all": mean_or_zero([result.total_steps for result in results]) if count else 0.0,
        "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
        "avg_illegal_moves": mean_or_zero([result.illegal_moves for result in results]) if count else 0.0,
        "avg_invalid_responses": mean_or_zero([result.invalid_responses for result in results]) if count else 0.0,
        "avg_api_errors": mean_or_zero([result.api_errors for result in results]) if count else 0.0,
    }


def summarize_by_grid(results: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.grid_size, []).append(result)
    rows: List[Dict[str, Any]] = []
    for grid_size, grid_results in sorted(grouped.items()):
        summary = summarize_results(grid_results)
        difficulty_counts = {
            difficulty: sum(1 for result in grid_results if result.difficulty == difficulty)
            for difficulty in ("easy", "medium", "hard")
        }
        rows.append(
            {
                "grid_size": grid_size,
                "difficulty_counts": json.dumps(difficulty_counts, sort_keys=True),
                "repeats": len(grid_results),
                "successes": summary["successes"],
                "success_rate": summary["success_rate"],
                "avg_solution_ticks": summary["avg_solution_ticks"],
                "avg_steps_all": summary["avg_steps_all"],
                "avg_steps_on_success": summary["avg_steps_on_success"],
                "avg_illegal_moves": summary["avg_illegal_moves"],
                "avg_invalid_responses": summary["avg_invalid_responses"],
                "avg_api_errors": summary["avg_api_errors"],
            }
        )
    return rows


def write_episode_csv(output_path: Path, results: Sequence[EpisodeResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "episode_id",
        "grid_size",
        "difficulty",
        "repeat_index",
        "seed",
        "source_index",
        "solution_ticks",
        "step_budget",
        "success",
        "terminated",
        "truncated",
        "total_steps",
        "illegal_moves",
        "invalid_responses",
        "api_errors",
        "final_reward",
        "final_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: getattr(result, key) for key in fieldnames})


def write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_combo(
    *,
    env: ContinuityPipeEnv,
    policy: Optional[InternVL3Policy],
    image_writer: StepImageWriter,
    grid_size: int,
    min_solution_steps: int,
    max_solution_steps: Optional[int],
    solution_steps_spec: str,
    episode_seeds: Sequence[int],
    oracle: bool,
    episode_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    specs = list(episode_specs or [])
    if not specs:
        specs = [
            {
                "repeat_index": repeat_index,
                "seed": int(episode_seed),
                "grid_size": int(grid_size),
                "reset_options": {
                    "grid_size": int(grid_size),
                    "min_solution_ticks": int(min_solution_steps),
                    "max_solution_ticks": None if max_solution_steps is None else int(max_solution_steps),
                },
                "solution_steps_spec": solution_steps_spec,
            }
            for repeat_index, episode_seed in enumerate(episode_seeds)
        ]
    progress_label = (
        "loaded"
        if episode_specs and solution_steps_spec == "loaded"
        else "difficulty_presets"
        if episode_specs
        else f"grid={grid_size}x{grid_size} steps={solution_steps_spec}"
    )
    progress = tqdm(total=len(specs), desc=progress_label, leave=False, unit="episode")

    for spec in specs:
        repeat_index = int(spec.get("repeat_index", len(results)))
        episode_seed = int(spec.get("seed", repeat_index))
        grid_size = int(spec.get("grid_size", grid_size))
        _, info = env.reset(seed=episode_seed, options=dict(spec.get("reset_options") or {}))
        current_state = info.get("state") or env.get_state()
        episode_difficulty = str(
            spec.get("difficulty")
            or info.get("difficulty", difficulty_label_for_solution_ticks(int(info.get("solution_ticks", 0))))
        )
        episode_id = str(spec.get("episode_id") or build_episode_id(
            grid_size=grid_size,
            solution_steps_spec=str(spec.get("solution_steps_spec") or solution_steps_spec),
            seed=episode_seed,
        ))
        conversation_messages: List[Dict[str, Any]] = []
        conversation_debug_messages: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        illegal_moves = 0
        invalid_responses = 0
        api_errors = 0
        terminated = False
        truncated = False
        final_reward = 0.0
        final_reason: Optional[str] = None
        step_budget = int(info.get("max_steps", current_state.get("maxSteps", grid_size * grid_size * 4)))
        next_info = dict(info)

        for step_index in range(step_budget):
            legal_actions = [int(value) for value in current_state.get("legalActionIndices", [])]
            if not legal_actions:
                break
            raw_png_bytes = env.screenshot(scene_only=True)
            current_png_bytes = add_state_caption(raw_png_bytes, f"State {step_index}")
            saved_current_image_path = image_writer.save_png(
                episode_id=episode_id,
                step_index=step_index,
                image_kind="current",
                png_bytes=current_png_bytes,
            )

            current_image_data_url = png_bytes_to_data_url(current_png_bytes)
            if step_index == 0:
                before_image_text, after_image_text = build_initial_prompt_texts(
                    grid_size=grid_size,
                    legal_actions=legal_actions,
                )
            else:
                before_image_text, after_image_text = build_followup_prompt_texts(
                    grid_size=grid_size,
                    legal_actions=legal_actions,
                )

            user_content = build_user_content(
                before_image_text=before_image_text,
                image_data_url=current_image_data_url,
                after_image_text=after_image_text,
            )
            debug_user_content = build_debug_user_content(
                before_image_text=before_image_text,
                image_path=saved_current_image_path,
                after_image_text=after_image_text,
            )
            request_messages = [*conversation_messages, {"role": "user", "content": user_content}]
            request_debug_messages = [*conversation_debug_messages, {"role": "user", "content": debug_user_content}]
            saved_pure_prompt_path = image_writer.save_pure_prompt(
                episode_id=episode_id,
                step_index=step_index,
                prompt_messages=request_debug_messages,
            )

            oracle_answer = build_oracle_action(current_state)
            if oracle:
                predicted_answer = oracle_answer
                raw_response_text = "<no-legal-action>" if predicted_answer is None else _coord_answer_to_json(predicted_answer)
                api_error_message = None
                invalid_response = predicted_answer is None
                response_debug = {"mode": "oracle"}
            else:
                assert policy is not None
                try:
                    predicted_answer, raw_response_text, api_error_message, invalid_response, response_debug = policy.predict_messages(
                        messages=request_messages,
                        grid_size=grid_size,
                    )
                except Exception as exc:
                    predicted_answer = None
                    raw_response_text = ""
                    api_error_message = f"{type(exc).__name__}: {exc}"
                    invalid_response = False
                    response_debug = {"exception": type(exc).__name__}
                if api_error_message is not None:
                    api_errors += 1

            if invalid_response:
                invalid_responses += 1

            if predicted_answer is not None:
                predicted_action_index = xy_to_action(predicted_answer[0], predicted_answer[1], grid_size)
            else:
                predicted_action_index = INVALID_ACTION
            if predicted_action_index not in legal_actions:
                action_index = INVALID_ACTION
            else:
                action_index = predicted_action_index

            _, reward, terminated, truncated, next_info = env.step(action_index)
            if next_info.get("illegal"):
                illegal_moves += 1
            final_reward = float(reward)
            final_reason = next_info.get("reason")

            conversation_messages.append({"role": "user", "content": user_content})
            conversation_messages.append({"role": "assistant", "content": raw_response_text or ""})
            conversation_debug_messages.append({"role": "user", "content": debug_user_content})
            conversation_debug_messages.append({"role": "assistant", "content": raw_response_text or ""})

            saved_prompt_debug_path = image_writer.save_prompt_debug(
                episode_id=episode_id,
                step_index=step_index,
                prompt_messages=request_debug_messages,
                ground_truth=oracle_answer,
                raw_response_text=raw_response_text,
                parsed_answer=predicted_answer,
                api_error_message=api_error_message,
                response_debug=response_debug,
            )

            step_logs.append(
                {
                    "step_index": step_index,
                    "seed": episode_seed,
                    "state_before_action": current_state,
                    "legal_actions": [
                        {"x": action_to_xy(action, grid_size)["x"], "y": action_to_xy(action, grid_size)["y"]}
                        for action in legal_actions
                    ],
                    "ground_truth": None if oracle_answer is None else {"x": oracle_answer[0], "y": oracle_answer[1]},
                    "predicted_action": None if predicted_answer is None else [predicted_answer[0], predicted_answer[1]],
                    "raw_response_text": raw_response_text,
                    "invalid_response": invalid_response,
                    "api_error_message": api_error_message,
                    "illegal": bool(next_info.get("illegal", False)),
                    "reason": next_info.get("reason"),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "saved_current_image_path": to_relative_path(saved_current_image_path),
                    "saved_prompt_debug_path": to_relative_path(saved_prompt_debug_path),
                    "saved_pure_prompt_path": to_relative_path(saved_pure_prompt_path),
                }
            )

            current_state = next_info.get("state") or current_state
            if terminated or truncated:
                break

        result = EpisodeResult(
            episode_index=len(results),
            episode_id=episode_id,
            grid_size=grid_size,
            difficulty=episode_difficulty,
            repeat_index=repeat_index,
            seed=episode_seed,
            source_index=int((step_logs[0]["state_before_action"] if step_logs else current_state).get("sourceIndex", 0)),
            solution_ticks=int(info.get("solution_ticks", 0)),
            step_budget=step_budget,
            success=bool(next_info.get("success", False)),
            terminated=terminated,
            truncated=truncated,
            total_steps=int(next_info.get("episode_steps", len(step_logs))),
            illegal_moves=illegal_moves,
            invalid_responses=invalid_responses,
            api_errors=api_errors,
            final_reward=final_reward,
            final_reason=final_reason,
            steps=step_logs,
        )
        results.append(result)
        progress.update(1)
        summary = summarize_results(results)
        progress.set_postfix(
            success=f"{summary['success_rate']:.3f}",
            invalid=sum(item.invalid_responses for item in results),
            illegal=sum(item.illegal_moves for item in results),
        )

    progress.close()
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InternVL3 benchmark for continuity_pipe.")
    parser.add_argument("--config", default=None, help="Path to local JSON config with api_key/base_url/model.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--difficulties",
        default=DEFAULT_DIFFICULTIES,
        help="Comma list of difficulty setups to run when --question-jsonl/--grid-sizes/--solution-steps are omitted.",
    )
    parser.add_argument("--grid-sizes", default="", help="Optional comma/range list of grid sizes. Supported values: 3-6.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Episodes to run for each difficulty setup or legacy grid/solution-step spec.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="First per-episode seed. Seeds increment by 1 for each generated episode.",
    )
    parser.add_argument(
        "--solution-steps",
        default=DEFAULT_SOLUTION_STEPS,
        help=(
            "Target oracle solution steps. Bounded ranges expand to each exact step, "
            "e.g. 10-20 means 10 through 20. Open ranges like 44+ stay as one bucket. "
            "When omitted, --difficulties presets are used."
        ),
    )
    parser.add_argument("--question-jsonl", default="", help="Load existing interactive question.jsonl rows via reset_config.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--requests-per-minute", type=float, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--oracle", action="store_true", help="Use the internal solver instead of the API.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    _, local_config = load_local_config(args.config)
    api_key = resolve_config_value(
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
        default=DEFAULT_BASE_URL,
    )
    model = resolve_config_value(
        args.model,
        env_keys=("INTERNVL3_MODEL",),
        config=local_config,
        config_key="model",
        default=DEFAULT_MODEL,
    )

    if not args.oracle and not api_key:
        raise ValueError(
            "Missing API key. Provide --api-key, set INTERNVL3_API_KEY/OPENAI_API_KEY, "
            "or add api_key to the local config file."
        )

    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    loaded_specs = load_episode_specs(question_jsonl) if question_jsonl is not None else []
    use_difficulty_setups = not loaded_specs and not args.grid_sizes.strip() and not args.solution_steps.strip()
    direct_difficulty_specs: List[Dict[str, Any]] = []
    difficulties: List[str] = []
    if loaded_specs:
        grid_sizes = sorted({int(spec["grid_size"]) for spec in loaded_specs})
    elif use_difficulty_setups:
        difficulties = parse_difficulty_list(args.difficulties)
        grid_sizes = sorted({grid_size_for_difficulty(difficulty) for difficulty in difficulties})
        next_seed = int(args.seed_start)
        for difficulty in difficulties:
            grid_size = grid_size_for_difficulty(difficulty)
            for repeat_index in range(int(args.repeats)):
                seed = next_seed
                next_seed += 1
                direct_difficulty_specs.append(
                    {
                        "episode_id": build_episode_id(
                            grid_size=grid_size,
                            solution_steps_spec=difficulty,
                            seed=seed,
                        ),
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "grid_size": grid_size,
                        "difficulty": difficulty,
                        "solution_steps_spec": difficulty,
                        "reset_options": {"difficulty": difficulty},
                    }
                )
    else:
        if args.grid_sizes.strip():
            grid_sizes = parse_int_list(args.grid_sizes)
        else:
            grid_sizes = default_grid_sizes()
    solution_step_specs = [] if (loaded_specs or use_difficulty_setups) else parse_solution_step_specs(args.solution_steps or "10-20")
    for min_solution_steps, _, _ in solution_step_specs:
        if min_solution_steps < 1:
            raise ValueError("--solution-steps must be positive.")
    for grid_size in grid_sizes:
        if grid_size < 3 or grid_size > 6:
            raise ValueError(f"Unsupported continuity_pipe grid size: {grid_size}. Supported values are 3 through 6.")

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_by_grid_json = output_json.with_name(f"{output_json.stem}_by_grid.json")
    output_by_grid_csv = output_csv.with_name(f"{output_csv.stem}_by_grid.csv")

    image_writer = StepImageWriter(root_dir=Path(args.image_root).resolve(), prompt_path_base=output_json.parent)
    policy = None if args.oracle else InternVL3Policy(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_retries=args.api_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        requests_per_minute=args.requests_per_minute,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    env = ContinuityPipeEnv(
        grid_size=grid_sizes[0],
        headless=args.headless,
        host=args.host,
        port=args.port,
    )
    all_results: List[EpisodeResult] = []
    try:
        combos = (
            [(grid_sizes[0], 0, None, "loaded")]
            if loaded_specs
            else [(grid_sizes[0], 0, None, "difficulty_presets")]
            if direct_difficulty_specs
            else [
                (grid_size, min_solution_steps, max_solution_steps, solution_steps_spec)
                for grid_size in grid_sizes
                for min_solution_steps, max_solution_steps, solution_steps_spec in solution_step_specs
            ]
        )
        size_progress = tqdm(combos, desc="Continuity Pipe combos", unit="combo")
        next_seed = int(args.seed_start)
        for grid_size, min_solution_steps, max_solution_steps, solution_steps_spec in size_progress:
            if loaded_specs or direct_difficulty_specs:
                episode_seeds = []
            else:
                episode_seeds = list(range(next_seed, next_seed + int(args.repeats)))
                next_seed += int(args.repeats)
            combo_results = run_combo(
                env=env,
                policy=policy,
                image_writer=image_writer,
                grid_size=grid_size,
                min_solution_steps=min_solution_steps,
                max_solution_steps=max_solution_steps,
                solution_steps_spec=solution_steps_spec,
                episode_seeds=episode_seeds,
                oracle=args.oracle,
                episode_specs=loaded_specs if loaded_specs else direct_difficulty_specs if direct_difficulty_specs else None,
            )
            for local_index, result in enumerate(combo_results):
                result.episode_index = len(all_results) + local_index
            all_results.extend(combo_results)
            combo_summary = summarize_results(combo_results)
            size_progress.set_postfix(
                grid=f"{grid_size}x{grid_size}",
                success=f"{combo_summary['success_rate']:.3f}",
                invalid=f"{combo_summary['avg_invalid_responses']:.3f}",
                illegal=f"{combo_summary['avg_illegal_moves']:.3f}",
            )
    finally:
        env.close()

    overall_summary = summarize_results(all_results)
    by_grid_summary = summarize_by_grid(all_results)
    payload = {
        "config": {
            "grid_sizes": grid_sizes,
            "difficulty_definition": difficulty_definition_text(),
            "difficulties": difficulties if use_difficulty_setups else None,
            "solution_steps": [spec for _, _, spec in solution_step_specs] if solution_step_specs else None,
            "repeats_per_difficulty": int(args.repeats) if use_difficulty_setups else None,
            "repeats_per_grid_solution_step": int(args.repeats) if solution_step_specs else None,
            "repeats_per_grid": len(solution_step_specs) * int(args.repeats) if solution_step_specs else None,
            "seed_start": None if loaded_specs else args.seed_start,
            "model": model,
            "base_url": base_url,
            "oracle": args.oracle,
            "max_tokens": args.max_tokens,
            "headless": args.headless,
            "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        },
        "summary": overall_summary,
        "summary_by_grid": by_grid_summary,
        "episodes": [asdict(result) for result in all_results],
    }
    question_jsonl_path, model_answer_jsonl_path = write_meta_jsonl(
        output_dir=output_json.parent,
        model_id="oracle" if args.oracle else model,
        results=all_results,
        write_question_jsonl=question_jsonl is None,
    )
    output_question_jsonl_path = question_jsonl_path if question_jsonl is None else question_jsonl
    payload["question_jsonl"] = jsonl_relpath(output_question_jsonl_path, base_dir=output_json.parent)
    payload["model_answer_jsonl"] = str(model_answer_jsonl_path.relative_to(output_json.parent))

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_episode_csv(output_csv, all_results)
    output_by_grid_json.write_text(json.dumps(by_grid_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_csv(output_by_grid_csv, by_grid_summary)

    print(
        f"success_rate={overall_summary['success_rate']:.3f} | "
        f"successes={overall_summary['successes']}/{overall_summary['total_episodes']} | "
        f"illegal={overall_summary['avg_illegal_moves']:.3f} | "
        f"invalid={overall_summary['avg_invalid_responses']:.3f} | "
        f"api_errors={overall_summary['avg_api_errors']:.3f}"
    )
    print(f"Wrote JSON results to {output_json}")
    print(f"Wrote CSV results to {output_csv}")
    print(f"Wrote grid JSON summary to {output_by_grid_json}")
    print(f"Wrote grid CSV summary to {output_by_grid_csv}")
    if question_jsonl is None:
        print(f"Wrote question JSONL to {question_jsonl_path}")
    else:
        print(f"Loaded question JSONL from {question_jsonl}")
    print(f"Wrote model answer JSONL to {model_answer_jsonl_path}")


if __name__ == "__main__":
    main()
