from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

GYM_DIR = Path(__file__).resolve().parent.parent / "gym"
if str(GYM_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_DIR))
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from env import ACTION_DIRECTIONS, INVALID_ACTION, SeparationOneStrokeEnv
from internvl3_config import load_local_config, resolve_config_value
from generator import (
    GeneratedLevelSpec,
    difficulty_label_for_board_size,
    generate_level_for_board_size,
    parse_board_size_list,
    parse_difficulty_list,
)
from level_data import count_level_colors
from solver import DIR_TO_DELTA, shortest_solution, solve_state
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
TOPOBENCH_EVAL_ROOT = REPO_ROOT.parent / "topobench_eval"
if str(TOPOBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPOBENCH_EVAL_ROOT))

from topobench_eval.metadata_utils import load_task_intro_line

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_separation_one_stroke.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_separation_one_stroke.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0
DEFAULT_BOARD_SIZES = "4-6"
DEFAULT_REPEATS = 1
DEFAULT_SEED_START = 1
DIFFICULTY_BOARD_SIZE = {"easy": 4, "medium": 5, "hard": 6}
MIN_EFFECTIVE_MAX_TOKENS = 32
STATE_CAPTION_FONT_SIZE = 34
STATE_CAPTION_TOP_MARGIN = 14

from jsonl_export import interactive_trajectory_state, read_text_if_exists, resolve_existing_path, to_relative_path, write_jsonl


@dataclass
class EpisodeResult:
    episode_index: int
    episode_id: str
    repeat_index: int
    seed: int
    setup_id: str
    board_size: int
    difficulty: str
    generated: bool
    setup_seed: Optional[int]
    generation_attempts: Optional[int]
    construction_path_length: Optional[int]
    level_index: Optional[int]
    level_number: int
    width: int
    height: int
    num_colors: int
    solution_length: int
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
    level_json: Dict[str, Any]
    steps: List[Dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkSetup:
    setup_id: str
    board_size: int
    difficulty: str
    generated: bool
    setup_seed: Optional[int]
    generation_attempts: Optional[int]
    construction_path_length: Optional[int]
    level_index: Optional[int]
    level_number: int
    solution_length: int
    level_json: Dict[str, Any]
    question_row_id: Optional[str] = None
    question_repeat_index: Optional[int] = None
    question_seed: Optional[int] = None
    question_max_steps: Optional[int] = None
    reset_config: Optional[Dict[str, Any]] = None


def build_episode_id(*, board_size: int, seed: int) -> str:
    return f"separation_one_stroke_size_{board_size}x{board_size}_seed_{seed}"


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


def load_benchmark_setups_from_question_jsonl(question_jsonl: Path) -> List[BenchmarkSetup]:
    setups: List[BenchmarkSetup] = []
    for row_index, row in enumerate(read_question_jsonl(question_jsonl)):
        meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        initial_state = meta_info.get("initial_state") if isinstance(meta_info.get("initial_state"), dict) else {}
        reset_config = initial_state.get("reset_config") if isinstance(initial_state.get("reset_config"), dict) else {}
        level_json = reset_config.get("levelJson") or initial_state.get("level_json") or meta_info.get("level_json")
        if not isinstance(level_json, dict):
            raise ValueError(f"Question row {row.get('id', row_index)!r} is missing levelJson in reset_config.")
        board_size = int(meta_info.get("board_size", reset_config.get("boardSize", int(level_json["W"]) - 1)))
        solution_length = int(meta_info.get("solution_length", len(level_json.get("solution", []))))
        seed = int(reset_config.get("seed", meta_info.get("seed", row_index)))
        repeat_index = int(meta_info.get("repeat_index", row_index))
        setup_id = str(meta_info.get("setup_id") or f"loaded_size_{board_size}x{board_size}_row_{row_index:04d}")
        setups.append(
            BenchmarkSetup(
                setup_id=setup_id,
                board_size=board_size,
                difficulty=str(meta_info.get("difficulty", difficulty_label_for_board_size(board_size))),
                generated=bool(meta_info.get("generated", True)),
                setup_seed=meta_info.get("setup_seed"),
                generation_attempts=meta_info.get("generation_attempts"),
                construction_path_length=meta_info.get("construction_path_length"),
                level_index=None,
                level_number=int(meta_info.get("level", row_index + 1)),
                solution_length=solution_length,
                level_json=level_json,
                question_row_id=str(row.get("id") or build_episode_id(
                    board_size=board_size,
                    seed=seed,
                )),
                question_repeat_index=repeat_index,
                question_seed=seed,
                question_max_steps=meta_info.get("max_actions_per_traj"),
                reset_config=dict(reset_config),
            )
        )
    return setups


def write_meta_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    results: Sequence[EpisodeResult],
    write_question_jsonl: bool = True,
) -> Tuple[Path, Path]:
    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    question_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    for result in results:
        first_step = result.steps[0] if result.steps else None
        first_image = first_step.get("saved_image_path") if first_step else None
        initial_images: List[str] = []
        if first_image:
            resolved_first_image = resolve_existing_path(first_image, output_dir)
            initial_images = [to_relative_path(resolved_first_image, base_dir=output_dir)]
        initial_state = {
            "current_position": first_step.get("current_position") if first_step else {},
            "level_json": result.level_json,
            "reset_config": {
                "levelJson": result.level_json,
                "boardSize": int(result.board_size),
                "seed": int(result.seed),
            },
        }
        question_row = {
            "id": result.episode_id,
            "category": ["separation", "separation_one_stroke", "interactive"],
            "type": "interactive",
            "meta_info": {
                "task_name": "separation_one_stroke",
                "config": config_rel,
                "level": result.level_number,
                "setup_id": result.setup_id,
                "board_size": result.board_size,
                "generated": result.generated,
                "setup_seed": result.setup_seed,
                "generation_attempts": result.generation_attempts,
                "construction_path_length": result.construction_path_length,
                "seed": result.seed,
                "repeat_index": result.repeat_index,
                "difficulty": result.difficulty,
                "solution_length": result.solution_length,
                "max_actions_per_traj": result.step_budget,
                "initial_state": initial_state,
                "level_json": result.level_json,
                "legal_action_format": '{"answer":"U"}',
            },
            "images": initial_images,
        }
        question_rows.append(question_row)
        answer_rows.append(
            {
                "id": result.episode_id,
                "category": ["separation", "separation_one_stroke", "interactive"],
                "type": "interactive",
                "meta_info": {
                    "task_name": "separation_one_stroke",
                    "config": config_rel,
                    "level": result.level_number,
                    "setup_id": result.setup_id,
                    "board_size": result.board_size,
                    "generated": result.generated,
                    "setup_seed": result.setup_seed,
                    "seed": result.seed,
                    "repeat_index": result.repeat_index,
                    "difficulty": result.difficulty,
                    "model_id": model_id,
                    "success": bool(result.success),
                    "final_reason": result.final_reason,
                    "total_steps": int(result.total_steps),
                },
                "trajectory": [
                    {
                        "step_index": int(step["step_index"]),
                        "current_images": [
                            to_relative_path(
                                resolve_existing_path(step["saved_image_path"], output_dir),
                                base_dir=output_dir,
                            )
                        ],
                        "state": interactive_trajectory_state(
                            trajectory_index=trajectory_index,
                            total_steps=len(result.steps),
                            success=bool(result.success),
                        ),
                        "answer": step.get("predicted_direction"),
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


SUMMARY_FIELD_ORDER = [
    "setup_id",
    "board_size",
    "difficulty",
    "generated",
    "setup_seed",
    "level_number",
    "width",
    "height",
    "num_colors",
    "solution_length",
    "repeats",
    "successes",
    "success_rate",
    "avg_steps_all",
    "avg_steps_on_success",
    "avg_illegal_moves",
    "avg_invalid_responses",
    "avg_api_errors",
]


class StepImageWriter:
    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _episode_dir(self, *, episode_id: str) -> Path:
        episode_dir = self.root_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def save_png(
        self,
        *,
        episode_id: str,
        step_index: int,
        image_kind: str,
        png_bytes: bytes,
    ) -> Path:
        episode_dir = self._episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_{image_kind}.png"
        output_path.write_bytes(png_bytes)
        return output_path

    def _display_path_text(self, raw_path: str) -> str:
        path = Path(raw_path)
        try:
            return str(path.relative_to(self.root_dir.parent))
        except ValueError:
            return raw_path

    def _serialize_message_content(self, content: Any) -> List[str]:
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return [json.dumps(content, ensure_ascii=False)] if content is not None else []

        blocks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                blocks.append(str(item))
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                blocks.append(str(item.get("text", "")))
            elif item_type == "image_path":
                path_text = self._display_path_text(str(item.get("path", "")))
                blocks.extend(["[Image]", path_text])
            else:
                blocks.append(json.dumps(item, ensure_ascii=False))
        return blocks

    def _serialize_pure_prompt_messages(self, prompt_messages: Sequence[Dict[str, Any]]) -> List[str]:
        blocks: List[str] = []
        for message in prompt_messages:
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    blocks.append(text)
                continue
            if not isinstance(content, list):
                serialized = json.dumps(content, ensure_ascii=False) if content is not None else ""
                serialized = serialized.strip()
                if serialized:
                    blocks.append(serialized)
                continue
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
                    if not raw_path:
                        continue
                    display_path = self._display_path_text(raw_path)
                    blocks.extend(("[Image]", display_path))
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
        raw_response_text: str,
        parsed_answer: Optional[Any],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]] = None,
    ) -> Path:
        episode_dir = self._episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_prompt_debug.txt"
        parsed_text = (
            "<none>"
            if parsed_answer is None
            else (json.dumps(parsed_answer, ensure_ascii=False) if isinstance(parsed_answer, (list, dict)) else str(parsed_answer))
        )
        blocks: List[str] = ["[prompt_messages]"]
        for message in prompt_messages:
            role = str(message.get("role", "message"))
            blocks.append(f"[{role}]")
            blocks.extend(self._serialize_message_content(message.get("content")))
        blocks.extend(
            [
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
        episode_dir = self._episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_pure_prompt.txt"
        blocks = self._serialize_pure_prompt_messages(prompt_messages)
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return output_path


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
            continue
        values.append(int(part))
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return deduped


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
        y = STATE_CAPTION_TOP_MARGIN
        draw.text(
            (x, y),
            caption_text,
            fill="white",
            font=font,
        )
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


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
        preferred_keys = ("text", "content", "value", "reasoning_content", "output_text", "refusal")
        for key in preferred_keys:
            if key in value:
                fragments.extend(_collect_text_fragments(value.get(key)))
        if fragments:
            return fragments
        block_type = value.get("type")
        if isinstance(block_type, str) and "text" in block_type.lower():
            for nested_key, nested_value in value.items():
                if nested_key == "type":
                    continue
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


def extract_response_text(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
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
    if message is not None and not isinstance(message, dict):
        debug_info["message_raw"] = message
        message = {}
    elif message is None:
        message = {}

    delta = first_choice.get("delta")
    if delta is not None and not isinstance(delta, dict):
        debug_info["delta_raw"] = delta
        delta = {}
    elif delta is None:
        delta = {}

    debug_info["finish_reason"] = first_choice.get("finish_reason")
    debug_info["message_content_raw"] = message.get("content")
    debug_info["message_reasoning_content_raw"] = message.get("reasoning_content")
    debug_info["message_refusal_raw"] = message.get("refusal")
    debug_info["choice_text_raw"] = first_choice.get("text")
    debug_info["choice_content_raw"] = first_choice.get("content")
    debug_info["delta_content_raw"] = delta.get("content")

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


def parse_direction(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.casefold()
    try:
        payload = json.loads(lowered)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("answer", "direction", "move", "choice", "action"):
            value = payload.get(key)
            if isinstance(value, str):
                candidate = value.strip().upper()
                if candidate in ACTION_DIRECTIONS:
                    return candidate

    stripped = lowered.strip()
    if stripped.upper() in ACTION_DIRECTIONS:
        return stripped.upper()

    brace_matches = re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL)
    for brace_text in reversed(brace_matches):
        for pattern in (
            r'"(?:answer|direction|move|choice|action)"\s*:\s*"([udlr])"',
            r"(?:move|answer|direction|choice|action)\s*[:\-]?\s*([udlr])\b",
            r'"([udlr])"',
        ):
            match = re.search(pattern, brace_text)
            if match:
                return match.group(1).upper()

    explicit_tail_matches = list(
        re.finditer(r"(?:move|answer|direction|choice|action|go|choose)\s*[:\-]?\s*([udlr])\b", lowered)
    )
    if explicit_tail_matches:
        return explicit_tail_matches[-1].group(1).upper()

    quoted_tail_matches = list(re.finditer(r'"([udlr])"', lowered))
    if quoted_tail_matches:
        return quoted_tail_matches[-1].group(1).upper()

    standalone_tail_matches = list(re.finditer(r"\b([udlr])\b", lowered))
    if standalone_tail_matches:
        return standalone_tail_matches[-1].group(1).upper()
    return None


def format_legal_actions_line(legal_directions: Sequence[str]) -> str:
    legal = [direction.upper() for direction in legal_directions if direction.upper() in ACTION_DIRECTIONS]
    action_list = ", ".join(legal)
    if not legal:
        return "Legal actions for this turn: <none>."
    return f"Legal actions for this turn: {action_list}."


def build_interact_initial_user_prompt_parts(
    *,
    info: Dict[str, Any],
    include_legal_actions_in_prompt: bool = True,
) -> Tuple[str, str]:
    rule_2 = (
        "2. A legal action must be one of the legal actions listed for the current state. You must not reuse an edge or create a closed loop. Backtracking over the most recent edge is allowed and acts like undoing the last move."
        if include_legal_actions_in_prompt
        else
        "2. Infer one legal action from the current state. You must not reuse an edge or create a closed loop. Backtracking over the most recent edge is allowed and acts like undoing the last move."
    )
    answer_line = (
        'Replace {ans} with the single legal answer for this task, chosen from "U", "D", "L", "R", using the JSON shape {"answer":"{ans}"}.'
        if include_legal_actions_in_prompt
        else
        'Replace {ans} with one legal answer for this task, using exactly one of "U", "D", "L", or "R" and the JSON shape {"answer":"{ans}"}.'
    )
    before_image_lines = [
        "[Task]",
        load_task_intro_line(
            PROJECT_ROOT,
            default=(
                "You are solving One-Stroke Color Grouping.\n"
                "In this task, you must draw one continuous stroke from the "
                "bottom-left to the top-right so same-colored cells stay together "
                "and different colors are separated."
            ),
        ),
        (
            "In this task, you must extend the current stroke from the bottom-left start vertex to the top-right goal "
            "vertex so that same-colored cells end up in the same region and different-colored cells end up in different regions."
        ),
        "In each board image, the white stroke is the current path, the green peg marks the start, and the red peg marks the goal.",
        "",
        "[Rules]",
        "1. At each turn, choose exactly one move from U, D, L, or R to extend the current stroke by one edge. U means up, D means down, L means left, and R means right.",
        rule_2,
        "3. After each action, the environment returns the next state. Illegal actions keep the board state unchanged but still count as a step.",
        "4. The task is solved when the stroke reaches the top-right goal vertex and the final regions satisfy the color-separation rule.",
        "5. The episode ends when the puzzle reaches a done state or the step budget is exhausted.",
        "6. At each turn, output exactly one next action for the current state.",
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer":"{ans}"} and nothing else.',
        answer_line,
        "",
        "[Current Task]",
        "The current state image is shown below.",
    ]
    after_image_lines = (
        [
            format_legal_actions_line(info["legalDirections"]),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            'Legal actions for this turn: not provided; infer one legal action from the board state and the task rules.',
            "To solve this task, output the next legal action for the current state.",
        ]
    )
    return "\n".join(before_image_lines), "\n".join(after_image_lines)


def build_interact_followup_user_prompt_parts(
    *,
    info: Dict[str, Any],
    include_legal_actions_in_prompt: bool = True,
) -> Tuple[str, str]:
    before_image_lines = [
        "[Current Task]",
        "The updated current state image is shown below.",
    ]
    after_image_lines = (
        [
            format_legal_actions_line(info["legalDirections"]),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            'Legal actions for this turn: not provided; infer one legal action from the updated board state and the task rules.',
            "To solve this task, output the next legal action for the current state.",
        ]
    )
    return "\n".join(before_image_lines), "\n".join(after_image_lines)


def build_user_content(
    *,
    user_prompt_before_image: str,
    image_data_urls: Sequence[str],
    user_prompt_after_image: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_before_image}]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url}} for image_data_url in image_data_urls)
    if user_prompt_after_image:
        content.append({"type": "text", "text": user_prompt_after_image})
    return content


def build_debug_user_content(
    *,
    user_prompt_before_image: str,
    image_items: Sequence[Tuple[str, Path]],
    user_prompt_after_image: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_before_image}]
    content.extend({"type": "image_path", "path": str(path)} for _label, path in image_items)
    if user_prompt_after_image:
        content.append({"type": "text", "text": user_prompt_after_image})
    return content


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
    ) -> Tuple[Any, str, Optional[str], bool, Dict[str, Any]]:
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": list(messages),
        }

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
            parsed = parse_direction(response_text)
            if parsed is None:
                return None, response_text, None, True, response_debug
            return parsed, response_text, None, False, response_debug

        return None, "", last_error or "Unknown API failure.", False, {}


def compute_summary(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    count = len(results)
    successes = sum(1 for result in results if result.success)
    illegal = sum(result.illegal_moves for result in results)
    invalid = sum(result.invalid_responses for result in results)
    api_errors = sum(result.api_errors for result in results)
    success_steps = [result.total_steps for result in results if result.success]
    return {
        "total_episodes": count,
        "successes": successes,
        "success_rate": (successes / count) if count else 0.0,
        "avg_steps_all": (sum(result.total_steps for result in results) / count) if count else 0.0,
        "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
        "avg_illegal_moves": (illegal / count) if count else 0.0,
        "avg_invalid_responses": (invalid / count) if count else 0.0,
        "avg_api_errors": (api_errors / count) if count else 0.0,
    }


def compute_level_summaries(results: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.setup_id, []).append(result)

    rows: List[Dict[str, Any]] = []
    for _setup_id, rows_for_level in sorted(grouped.items(), key=lambda item: item[1][0].level_number):
        first = rows_for_level[0]
        summary = compute_summary(rows_for_level)
        rows.append(
            {
                "setup_id": first.setup_id,
                "board_size": first.board_size,
                "difficulty": first.difficulty,
                "generated": first.generated,
                "setup_seed": first.setup_seed,
                "generation_attempts": first.generation_attempts,
                "level_number": first.level_number,
                "width": first.width,
                "height": first.height,
                "num_colors": first.num_colors,
                "solution_length": first.solution_length,
                "repeats": len(rows_for_level),
                "successes": summary["successes"],
                "success_rate": summary["success_rate"],
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "episode_id",
                "repeat_index",
                "seed",
                "setup_id",
                "board_size",
                "difficulty",
                "generated",
                "setup_seed",
                "generation_attempts",
                "level_index",
                "level_number",
                "width",
                "height",
                "num_colors",
                "solution_length",
                "success",
                "terminated",
                "truncated",
                "total_steps",
                "illegal_moves",
                "invalid_responses",
                "api_errors",
                "final_reward",
                "final_reason",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: getattr(result, key) for key in writer.fieldnames})


def write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _reference_solution_next_direction(
    info: Dict[str, Any],
    reference_solution: Optional[Sequence[Any]],
) -> Optional[str]:
    if not isinstance(reference_solution, Sequence) or isinstance(reference_solution, (str, bytes)):
        return None
    state = info.get("state")
    if not isinstance(state, dict):
        return None
    stroke = state.get("stroke")
    if not isinstance(stroke, list) or not stroke:
        return None

    drawn_steps = len(stroke) - 1
    if drawn_steps < 0 or drawn_steps >= len(reference_solution):
        return None

    x = 0
    y = 0
    first = stroke[0]
    if not isinstance(first, dict) or int(first.get("x", -1)) != x or int(first.get("y", -1)) != y:
        return None
    for index, direction in enumerate(reference_solution[:drawn_steps], start=1):
        if direction not in DIR_TO_DELTA:
            return None
        dx, dy = DIR_TO_DELTA[str(direction)]
        x += dx
        y += dy
        actual = stroke[index]
        if not isinstance(actual, dict):
            return None
        if int(actual.get("x", -1)) != x or int(actual.get("y", -1)) != y:
            return None

    candidate = str(reference_solution[drawn_steps]).upper()
    if candidate in info.get("legalDirections", []):
        return candidate
    return None


def build_oracle_direction(
    info: Dict[str, Any],
    *,
    reference_solution: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    reference_direction = _reference_solution_next_direction(info, reference_solution)
    if reference_direction is not None:
        return reference_direction

    remaining_budget = max(0, int(info["maxSteps"]) - int(info["episode_steps"]))
    exact_solution = shortest_solution(
        info["state"],
        timeout_seconds=3.0,
        max_depth=remaining_budget,
    )
    if exact_solution and len(exact_solution) <= remaining_budget:
        return exact_solution[0]
    solution = solve_state(
        info["state"],
        max_depth=remaining_budget,
        timeout_seconds=5.0,
    )
    if solution:
        return solution[0]
    return None


def _setup_from_generated_spec(
    spec: GeneratedLevelSpec,
    *,
    level_number: int,
    seed: int,
    repeat_index: int,
) -> BenchmarkSetup:
    setup_id = f"generated_size_{spec.board_size}x{spec.board_size}_seed_{seed:03d}"
    question_max_steps = max(1, int(math.ceil(int(spec.construction_path_length) * 1.2)))
    return BenchmarkSetup(
        setup_id=setup_id,
        board_size=spec.board_size,
        difficulty=spec.difficulty,
        generated=True,
        setup_seed=spec.setup_seed,
        generation_attempts=spec.generation_attempts,
        construction_path_length=spec.construction_path_length,
        level_index=None,
        level_number=level_number,
        solution_length=spec.solution_length,
        level_json=spec.level_json,
        question_row_id=build_episode_id(board_size=spec.board_size, seed=seed),
        question_repeat_index=repeat_index,
        question_seed=seed,
        question_max_steps=question_max_steps,
        reset_config={
            "levelJson": spec.level_json,
            "boardSize": int(spec.board_size),
            "seed": int(seed),
        },
    )


def selected_board_sizes(args) -> List[int]:
    difficulty_spec = str(getattr(args, "difficulty", "") or "").strip()
    if difficulty_spec:
        return [DIFFICULTY_BOARD_SIZE[difficulty] for difficulty in parse_difficulty_list(difficulty_spec)]
    return parse_board_size_list(args.board_sizes)


def build_benchmark_setups(args) -> List[BenchmarkSetup]:
    setups: List[BenchmarkSetup] = []
    board_sizes = selected_board_sizes(args)
    next_seed = int(args.seed_start)
    for board_size in board_sizes:
        for repeat_index in range(int(args.repeats)):
            spec = generate_level_for_board_size(
                board_size=int(board_size),
                seed=next_seed,
                setup_index=0,
                max_attempts=int(args.generation_max_attempts),
                solver_timeout_seconds=float(args.generation_solver_timeout),
            )
            setups.append(
                _setup_from_generated_spec(
                    spec,
                    level_number=len(setups) + 1,
                    seed=next_seed,
                    repeat_index=repeat_index,
                )
            )
            next_seed += 1
    return setups


def run_benchmark(args) -> None:
    config_path, local_config = load_local_config(args.config)
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
            f"or add api_key to {config_path}."
        )

    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    setups = (
        load_benchmark_setups_from_question_jsonl(question_jsonl)
        if question_jsonl is not None
        else build_benchmark_setups(args)
    )
    if not setups:
        raise ValueError("No benchmark setups were selected.")
    effective_max_tokens = int(args.max_tokens)
    if not args.oracle and 0 < effective_max_tokens < MIN_EFFECTIVE_MAX_TOKENS:
        print(
            f"Requested --max-tokens {effective_max_tokens} is too small for this benchmark; "
            f"using {MIN_EFFECTIVE_MAX_TOKENS} instead so the model can emit a final direction after reasoning."
        )
        effective_max_tokens = MIN_EFFECTIVE_MAX_TOKENS

    policy = None if args.oracle else InternVL3Policy(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        max_tokens=effective_max_tokens,
        api_retries=args.api_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        requests_per_minute=args.requests_per_minute,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_by_level_json = output_json.with_name(f"{output_json.stem}_by_level.json")
    output_by_level_csv = output_csv.with_name(f"{output_csv.stem}_by_level.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_writer = StepImageWriter(root_dir=Path(args.image_root).resolve())

    def episode_seeds_for_setup(setup: BenchmarkSetup, episode_offset: int) -> List[int]:
        if setup.question_seed is not None:
            return [int(setup.question_seed)]
        return [int(args.seed_start) + int(episode_offset)]

    total_episodes = sum(len(episode_seeds_for_setup(setup, 0)) for setup in setups)
    env = SeparationOneStrokeEnv(
        headless=args.headless,
        max_steps=args.max_steps,
        port=args.port,
        host=args.host,
    )
    results: List[EpisodeResult] = []
    try:
        progress = tqdm(total=total_episodes, desc="One Stroke benchmark")
        episode_index = 0
        for setup in setups:
            level_json = setup.level_json
            setup_seeds = episode_seeds_for_setup(setup, episode_index)
            setup_repeat_total = len(setup_seeds)
            for repeat_index, episode_seed in enumerate(setup_seeds):
                result_repeat_index = setup.question_repeat_index if setup.question_repeat_index is not None else repeat_index
                episode_id = setup.question_row_id or build_episode_id(
                    board_size=setup.board_size,
                    seed=episode_seed,
                )
                frontend_config: Dict[str, Any]
                if setup.reset_config is not None:
                    frontend_config = dict(setup.reset_config)
                elif setup.level_index is None:
                    frontend_config = {"levelJson": level_json}
                else:
                    frontend_config = {"levelIndex": setup.level_index}
                reset_max_steps = int(setup.question_max_steps) if setup.question_max_steps is not None else args.max_steps
                _, info = env.reset(
                    seed=episode_seed,
                    options={
                        "frontend_config": frontend_config,
                        "max_steps": reset_max_steps,
                    },
                )
                reference_solution_length = int(
                    info.get("referenceSolutionLength")
                    or setup.solution_length
                    or len(level_json.get("solution", []))
                )
                terminated = False
                truncated = False
                illegal_moves = 0
                invalid_responses = 0
                api_errors = 0
                final_reward = 0.0
                final_reason: Optional[str] = None
                step_logs: List[Dict[str, Any]] = []
                step_index = 0

                conversation_messages: List[Dict[str, Any]] = []
                conversation_debug_messages: List[Dict[str, Any]] = []
                while not terminated and not truncated:
                    raw_png_bytes = env.capture_png_bytes()
                    png_bytes = add_state_caption(raw_png_bytes, f"State {step_index}")
                    saved_image_path = image_writer.save_png(
                        episode_id=episode_id,
                        step_index=step_index,
                        image_kind="current",
                        png_bytes=png_bytes,
                    )
                    current_image_data_url = png_bytes_to_data_url(png_bytes)
                    if step_index == 0:
                        user_prompt_before_image, user_prompt_after_image = build_interact_initial_user_prompt_parts(info=info)
                    else:
                        user_prompt_before_image, user_prompt_after_image = build_interact_followup_user_prompt_parts(info=info)
                    user_content = build_user_content(
                        user_prompt_before_image=user_prompt_before_image,
                        image_data_urls=[current_image_data_url],
                        user_prompt_after_image=user_prompt_after_image,
                    )
                    debug_user_content = build_debug_user_content(
                        user_prompt_before_image=user_prompt_before_image,
                        image_items=[("current", saved_image_path)],
                        user_prompt_after_image=user_prompt_after_image,
                    )
                    request_messages = [*conversation_messages, {"role": "user", "content": user_content}]
                    request_debug_messages = [*conversation_debug_messages, {"role": "user", "content": debug_user_content}]
                    image_writer.save_pure_prompt(
                        episode_id=episode_id,
                        step_index=step_index,
                        prompt_messages=request_debug_messages,
                    )

                    if args.oracle:
                        predicted_direction = build_oracle_direction(
                            info,
                            reference_solution=level_json.get("solution"),
                        )
                        raw_response_text = (
                            json.dumps({"answer": predicted_direction}, separators=(",", ":"))
                            if predicted_direction in ACTION_DIRECTIONS
                            else "<no-solution>"
                        )
                        api_error_message = None
                        invalid_response = predicted_direction is None
                        response_debug = {"mode": "oracle"}
                    else:
                        try:
                            (
                                predicted_direction,
                                raw_response_text,
                                api_error_message,
                                invalid_response,
                                response_debug,
                            ) = policy.predict_messages(
                                messages=request_messages,
                            )
                        except Exception as exc:
                            predicted_direction = None
                            raw_response_text = ""
                            api_error_message = f"{type(exc).__name__}: {exc}"
                            invalid_response = False
                            response_debug = {"exception": type(exc).__name__}
                        if api_error_message is not None:
                            api_errors += 1

                    if invalid_response:
                        invalid_responses += 1

                    action = env.encode_direction(predicted_direction) if predicted_direction in ACTION_DIRECTIONS else INVALID_ACTION
                    _, reward, terminated, truncated, next_info = env.step(action)
                    if next_info.get("illegal"):
                        illegal_moves += 1
                    final_reward = reward
                    final_reason = next_info.get("reason") or next_info.get("truncated_reason")
                    conversation_messages.append({"role": "user", "content": user_content})
                    conversation_messages.append({"role": "assistant", "content": raw_response_text or ""})
                    conversation_debug_messages.append({"role": "user", "content": debug_user_content})
                    conversation_debug_messages.append({"role": "assistant", "content": raw_response_text or ""})

                    image_writer.save_prompt_debug(
                        episode_id=episode_id,
                        step_index=step_index,
                        prompt_messages=request_debug_messages,
                        raw_response_text=raw_response_text,
                        parsed_answer=predicted_direction,
                        api_error_message=api_error_message,
                        response_debug=response_debug,
                    )

                    step_logs.append(
                        {
                            "step_index": step_index,
                            "current_position": dict(info["state"]["currentPos"]),
                            "legal_directions": list(info["legalDirections"]),
                            "predicted_direction": predicted_direction,
                            "action_index": action,
                            "raw_response_text": raw_response_text,
                            "invalid_response": invalid_response,
                            "api_error_message": api_error_message,
                            "illegal": next_info.get("illegal"),
                            "reason": next_info.get("reason"),
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "saved_image_path": to_relative_path(saved_image_path, base_dir=output_json.parent),
                        }
                    )
                    info = next_info
                    step_index += 1

                result = EpisodeResult(
                    episode_index=episode_index,
                    episode_id=episode_id,
                    repeat_index=result_repeat_index,
                    seed=episode_seed,
                    setup_id=setup.setup_id,
                    board_size=setup.board_size,
                    difficulty=setup.difficulty,
                    generated=setup.generated,
                    setup_seed=setup.setup_seed,
                    generation_attempts=setup.generation_attempts,
                    construction_path_length=setup.construction_path_length,
                    level_index=setup.level_index,
                    level_number=setup.level_number,
                    width=int(level_json["W"]),
                    height=int(level_json["H"]),
                    num_colors=count_level_colors(level_json),
                    solution_length=reference_solution_length,
                    step_budget=int(info.get("maxSteps", args.max_steps or 0)),
                    success=bool(info.get("success", False)),
                    terminated=terminated,
                    truncated=truncated,
                    total_steps=int(info.get("episode_steps", step_index)),
                    illegal_moves=illegal_moves,
                    invalid_responses=invalid_responses,
                    api_errors=api_errors,
                    final_reward=float(final_reward),
                    final_reason=final_reason,
                    level_json=level_json,
                    steps=step_logs,
                )
                results.append(result)
                progress.set_postfix(
                    success=f"{(sum(1 for item in results if item.success) / len(results)):.3f}",
                    invalid=sum(item.invalid_responses for item in results),
                    illegal=sum(item.illegal_moves for item in results),
                    setup=setup.setup_id,
                    difficulty=setup.difficulty,
                    repeat=f"{repeat_index + 1}/{setup_repeat_total}",
                )
                progress.update(1)
                episode_index += 1
    finally:
        env.close()

    summary = compute_summary(results)
    by_level_summary = compute_level_summaries(results)
    payload = {
        "config": {
            "mode": "loaded_question_jsonl" if question_jsonl is not None else "generated_board_size",
            "levels": [setup.level_number for setup in setups],
            "setup_ids": [setup.setup_id for setup in setups],
            "board_sizes": [setup.board_size for setup in setups],
            "difficulty_labels": [setup.difficulty for setup in setups],
            "difficulty": args.difficulty or None,
            "repeats_per_board_size": 1 if question_jsonl is not None else int(args.repeats),
            "seed_start": None if question_jsonl is not None else int(args.seed_start),
            "difficulty_rule": "4x4 easy, 5x5 medium, 6x6 hard; labels are filtered by BFS shortest solution length and color count",
            "generation_max_attempts": args.generation_max_attempts,
            "generation_solver_timeout": args.generation_solver_timeout,
            "model": model,
            "base_url": base_url,
            "oracle": args.oracle,
            "max_steps": args.max_steps,
            "requested_max_tokens": args.max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        },
        "summary": summary,
        "summary_by_level": by_level_summary,
        "episodes": [result.__dict__ for result in results],
    }
    question_jsonl_path, model_answer_jsonl_path = write_meta_jsonl(
        output_dir=output_json.parent,
        model_id="oracle" if args.oracle else model,
        results=results,
        write_question_jsonl=question_jsonl is None,
    )
    output_question_jsonl_path = question_jsonl_path if question_jsonl is None else question_jsonl
    payload["question_jsonl"] = to_relative_path(output_question_jsonl_path, base_dir=output_json.parent)
    payload["model_answer_jsonl"] = str(model_answer_jsonl_path.relative_to(output_json.parent))
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_episode_csv(output_csv, results)
    output_by_level_json.write_text(json.dumps(by_level_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_csv(output_by_level_csv, by_level_summary)

    print(
        f"success_rate={summary['success_rate']:.3f} | successes={summary['successes']}/{summary['total_episodes']} | "
        f"illegal={summary['avg_illegal_moves']:.3f} | invalid={summary['avg_invalid_responses']:.3f} | "
        f"api_errors={summary['avg_api_errors']:.3f}"
    )
    print(f"Wrote JSON results to {output_json}")
    print(f"Wrote CSV results to {output_csv}")
    print(f"Wrote level JSON summary to {output_by_level_json}")
    print(f"Wrote level CSV summary to {output_by_level_csv}")
    if question_jsonl is None:
        print(f"Wrote question JSONL to {question_jsonl_path}")
    else:
        print(f"Loaded question JSONL from {question_jsonl}")
    print(f"Wrote model answer JSONL to {model_answer_jsonl_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InternVL3 benchmark for separation_one_stroke.")
    parser.add_argument("--config", default=None, help="Path to local JSON config with api_key/base_url/model.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--requests-per-minute", type=float, default=DEFAULT_REQUESTS_PER_MINUTE)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--board-sizes",
        default=DEFAULT_BOARD_SIZES,
        help="Generate procedural setups for these board sizes, e.g. 4,5,6 or 4-6.",
    )
    parser.add_argument(
        "--difficulty",
        "--difficulties",
        dest="difficulty",
        default="",
        help="Generate procedural setups by difficulty label, e.g. easy, medium, hard, easy,hard, or all. Overrides --board-sizes.",
    )
    parser.add_argument("--generation-max-attempts", type=int, default=1000)
    parser.add_argument("--generation-solver-timeout", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--question-jsonl", default="", help="Load existing interactive question.jsonl rows via reset_config.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--oracle", action="store_true", help="Use the built-in solver instead of calling the API.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.generation_max_attempts <= 0:
        raise ValueError("--generation-max-attempts must be positive.")
    if args.generation_solver_timeout <= 0:
        raise ValueError("--generation-solver-timeout must be positive.")

    run_benchmark(args)


if __name__ == "__main__":
    main()
