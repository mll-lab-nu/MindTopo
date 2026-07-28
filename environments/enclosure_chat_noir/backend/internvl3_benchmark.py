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

from env import (
    AUTO_SETUP_VALUE,
    CAT_POLICIES,
    DEFAULT_BOARD_RADIUS_SPEC,
    DEFAULT_INITIAL_BLOCK_COUNT_SPEC,
    DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
    DEFAULT_MIN_WINNING_FIRST_ACTIONS,
    DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
    INVALID_ACTION,
    EnclosureChatNoirEnv,
    is_auto_setup_value,
    legal_cat_moves,
    solve_chat_noir_instance,
    shortest_boundary_distance,
    validate_board_radius,
)
from internvl3_config import load_local_config, resolve_config_value
from topobench_eval.metadata_utils import load_task_intro_line
from tqdm.auto import tqdm
from jsonl_export import (
    flatten_answer_value,
    interactive_trajectory_state,
    read_text_if_exists,
    resolve_existing_path,
    to_relative_path as jsonl_relpath,
    write_jsonl,
)

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. Install it with: pip install requests"
    ) from exc

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_enclosure_chat_noir.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_enclosure_chat_noir.csv"
DEFAULT_GENERATE_ROOT = PROJECT_ROOT / "output" / "generate_samples"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0
DEFAULT_BOARD_RADIUS = DEFAULT_BOARD_RADIUS_SPEC
DEFAULT_INITIAL_BLOCK_COUNT = DEFAULT_INITIAL_BLOCK_COUNT_SPEC
DEFAULT_CAT_POLICIES = "easy,medium,hard"
DEFAULT_REPEATS = 1
DEFAULT_SEED_START = 1
GENERATION_CAT_POLICIES = ("easy", "medium", "hard")
STATE_CAPTION_FONT_SIZE = 34
STATE_CAPTION_TOP_MARGIN = 14


@dataclass
class EpisodeResult:
    episode_index: int
    episode_id: str
    board_radius: int
    initial_block_count: int
    cat_policy_name: str
    cat_policy_level: int
    difficulty: str
    difficulty_score: float
    repeat_index: int
    seed: int
    initial_escape_distance: int
    min_winning_first_actions: int
    min_adjacent_winning_first_actions: int
    winning_first_action_count: int
    winning_first_actions: List[int]
    adjacent_winning_first_action_count: int
    adjacent_winning_first_actions: List[int]
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
    board_cells = state.get("boardCells")
    if isinstance(board_cells, list):
        board_cells = [
            {
                "index": cell.get("index"),
                "isBoundary": cell.get("isBoundary"),
                "neighborIndices": cell.get("neighborIndices"),
            }
            for cell in board_cells
            if isinstance(cell, dict)
        ]
    cat_policy = state.get("catPolicy")
    if isinstance(cat_policy, dict):
        cat_policy = cat_policy.get("name")
    return {
        "boardRadius": state.get("boardRadius"),
        "catIndex": state.get("catIndex"),
        "blockedIndices": state.get("blockedIndices"),
        "catPolicy": cat_policy,
        "boardCells": board_cells,
    }


def difficulty_label_for_score(score: float) -> str:
    value = int(score)
    if value <= 1:
        return "easy"
    if value == 2:
        return "medium"
    return "hard"


def difficulty_label_for_policy(cat_policy_name: str, difficulty_score: float = 1.0) -> str:
    value = str(cat_policy_name).strip().lower()
    if value in GENERATION_CAT_POLICIES:
        return value
    return difficulty_label_for_score(difficulty_score)


def build_episode_id(
    *,
    board_radius: int,
    initial_block_count: int,
    cat_policy_name: str,
    seed: int,
) -> str:
    return (
        f"enclosure_chat_noir_radius_{board_radius}_blocked_{initial_block_count}"
        f"_policy_{cat_policy_name}_seed_{seed}"
    )


def max_actions_per_traj(*, board_cell_count: int, initial_block_count: int) -> int:
    return max(0, int(board_cell_count) - int(initial_block_count) - 1)


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
        board_radius = validate_board_radius(reset_config.get("boardRadius", initial_state.get("boardRadius", 3)))
        initial_block_count = int(reset_config.get("initialBlockedCount", len(initial_state.get("blockedIndices") or [])))
        cat_policy_name = str(reset_config.get("catPolicy", initial_state.get("catPolicy", "medium"))).strip().lower()
        seed = int(reset_config.get("seed", meta_info.get("seed", row_index)))
        repeat_index = int(meta_info.get("repeat_index", row_index))
        max_steps_value = meta_info.get("max_actions_per_traj")
        specs.append(
            {
                "episode_id": str(row.get("id") or build_episode_id(
                    board_radius=board_radius,
                    initial_block_count=initial_block_count,
                    cat_policy_name=cat_policy_name,
                    seed=seed,
                )),
                "repeat_index": repeat_index,
                "seed": seed,
                "board_radius": board_radius,
                "initial_block_count": initial_block_count,
                "cat_policy_name": cat_policy_name,
                "difficulty": str(meta_info.get("difficulty", difficulty_label_for_policy(cat_policy_name))),
                "max_steps": int(max_steps_value) if max_steps_value is not None else None,
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
        difficulty = result.difficulty
        setup_level = f"radius_{result.board_radius}_blocked_{result.initial_block_count}_policy_{result.cat_policy_name}"
        first_step = result.steps[0] if result.steps else None
        initial_state = build_initial_state_snapshot(first_step.get("state_before_action", {})) if first_step else {}
        if initial_state:
            blocked_indices = list(initial_state.get("blockedIndices") or [])
            initial_state["reset_config"] = {
                "boardRadius": int(result.board_radius),
                "initialBlockedCount": int(result.initial_block_count),
                "catPolicy": result.cat_policy_name,
                "seed": int(result.seed),
                "rngSeed": int(result.seed),
                "minWinningFirstActions": int(result.min_winning_first_actions),
                "minAdjacentWinningFirstActions": int(result.min_adjacent_winning_first_actions),
                "catIndex": int(initial_state.get("catIndex") or 0),
                "initialBlockedIndices": blocked_indices,
            }
        initial_images = (
            [
                jsonl_relpath(
                    resolve_existing_path(
                        first_step["saved_current_image_path"],
                        output_dir,
                        PROJECT_ROOT,
                        DEFAULT_GENERATE_ROOT,
                        DEFAULT_OUTPUT_ROOT,
                    ),
                    base_dir=output_dir,
                )
            ]
            if first_step
            else []
        )
        question_row = {
            "id": result.episode_id,
            "category": ["enclosure", "enclosure_chat_noir", "interactive"],
            "type": "interactive",
            "meta_info": {
                "task_name": "enclosure_chat_noir",
                "config": config_rel,
                "level": difficulty,
                "setup_level": setup_level,
                "seed": result.seed,
                "repeat_index": result.repeat_index,
                "difficulty": difficulty,
                "board_radius": result.board_radius,
                "initial_block_count": result.initial_block_count,
                "cat_policy_name": result.cat_policy_name,
                "cat_policy_level": result.cat_policy_level,
                "difficulty_score": result.difficulty_score,
                "min_winning_first_actions": int(result.min_winning_first_actions),
                "min_adjacent_winning_first_actions": int(result.min_adjacent_winning_first_actions),
                "winning_first_action_count": int(result.winning_first_action_count),
                "winning_first_actions": list(result.winning_first_actions),
                "adjacent_winning_first_action_count": int(result.adjacent_winning_first_action_count),
                "adjacent_winning_first_actions": list(result.adjacent_winning_first_actions),
                "max_actions_per_traj": result.step_budget,
                "initial_state": initial_state,
                "legal_action_format": '{"answer":{cell_index}}',
            },
            "images": initial_images,
        }
        question_rows.append(question_row)
        answer_rows.append(
            {
                "id": result.episode_id,
                "category": ["enclosure", "enclosure_chat_noir", "interactive"],
                "type": "interactive",
                "meta_info": {
                    "task_name": "enclosure_chat_noir",
                    "config": config_rel,
                    "level": difficulty,
                    "setup_level": setup_level,
                    "seed": result.seed,
                    "repeat_index": result.repeat_index,
                    "difficulty": difficulty,
                    "board_radius": result.board_radius,
                    "initial_block_count": result.initial_block_count,
                    "cat_policy_name": result.cat_policy_name,
                    "cat_policy_level": result.cat_policy_level,
                    "difficulty_score": result.difficulty_score,
                    "min_winning_first_actions": int(result.min_winning_first_actions),
                    "min_adjacent_winning_first_actions": int(result.min_adjacent_winning_first_actions),
                    "winning_first_action_count": int(result.winning_first_action_count),
                    "winning_first_actions": list(result.winning_first_actions),
                    "adjacent_winning_first_action_count": int(result.adjacent_winning_first_action_count),
                    "adjacent_winning_first_actions": list(result.adjacent_winning_first_actions),
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
                                    DEFAULT_GENERATE_ROOT,
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
    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = root_dir
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
        episode_dir = self.episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_{image_kind}.png"
        output_path.write_bytes(png_bytes)
        return output_path

    @staticmethod
    def _serialize_message_content(content: Any, *, image_root: Path) -> List[str]:
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
                if not raw_path:
                    continue
                display_path = os.path.relpath(Path(raw_path), image_root)
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
        ground_truth: Optional[int],
        raw_response_text: str,
        parsed_answer: Optional[int],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]],
    ) -> Path:
        episode_dir = self.episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_prompt_debug.txt"
        ground_truth_text = "<none>" if ground_truth is None else _cell_answer_to_json(ground_truth)
        parsed_text = "<none>" if parsed_answer is None else _cell_answer_to_json(parsed_answer)
        blocks: List[str] = ["[prompt_messages]"]
        for message in prompt_messages:
            role = str(message.get("role", "message"))
            blocks.append(f"[{role}]")
            blocks.extend(self._serialize_message_content(message.get("content"), image_root=self.root_dir))
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
        episode_dir = self.episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_pure_prompt.txt"
        blocks: List[str] = []
        for message in prompt_messages:
            blocks.extend(self._serialize_message_content(message.get("content"), image_root=self.root_dir))
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return output_path


def to_relative_path(path: Path, *, base: Path = PROJECT_ROOT) -> str:
    return os.path.relpath(path, base)


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
        else:
            values.append(int(part))
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return deduped


def parse_board_radius_specs(spec: str) -> List[Any]:
    if is_auto_setup_value(spec):
        return [AUTO_SETUP_VALUE]
    return [validate_board_radius(value) for value in parse_int_list(spec)]


def parse_initial_block_count_specs(spec: str) -> List[Any]:
    if is_auto_setup_value(spec):
        return [AUTO_SETUP_VALUE]
    return parse_int_list(spec)


def parse_string_list(spec: str) -> List[str]:
    values = [part.strip() for part in spec.split(",") if part.strip()]
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    if not deduped:
        raise ValueError(f"Empty string list spec: {spec!r}")
    return deduped


def parse_policy_list(spec: str) -> List[str]:
    policies: List[str] = []
    for raw in str(spec).split(","):
        value = raw.strip().lower()
        if not value:
            continue
        if value not in GENERATION_CAT_POLICIES:
            raise ValueError(
                f"Unsupported generation cat policy: {value}. "
                f"Use one of: {', '.join(GENERATION_CAT_POLICIES)}."
            )
        if value not in policies:
            policies.append(value)
    if not policies:
        raise ValueError("At least one cat policy is required.")
    return policies


def build_direct_episode_specs(
    *,
    board_radii: Sequence[Any],
    blocked_counts: Sequence[Any],
    cat_policies: Sequence[str],
    repeats: int,
    seed_start: int,
    min_winning_first_actions: Any = DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
    min_adjacent_winning_first_actions: int = DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    next_seed = int(seed_start)
    for board_radius in board_radii:
        for initial_block_count in blocked_counts:
            for cat_policy_name in cat_policies:
                for repeat_index in range(int(repeats)):
                    specs.append(
                        {
                            "repeat_index": repeat_index,
                            "seed": next_seed,
                            "board_radius": board_radius,
                            "initial_block_count": initial_block_count,
                            "cat_policy_name": str(cat_policy_name),
                            "reset_options": {
                                "board_radius": board_radius,
                                "initial_block_count": initial_block_count,
                                "cat_policy": str(cat_policy_name),
                                "min_winning_first_actions": min_winning_first_actions,
                                "min_adjacent_winning_first_actions": int(min_adjacent_winning_first_actions),
                            },
                        }
                    )
                    next_seed += 1
    return specs


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
        draw.text((x, STATE_CAPTION_TOP_MARGIN), caption_text, fill="black", font=font)
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


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _validate_cell(cell_index: Optional[int], cell_count: int) -> Optional[int]:
    if cell_index is None:
        return None
    if 0 <= cell_index < cell_count:
        return cell_index
    return None


def _parse_cell_value(value: Any, cell_count: int) -> Optional[int]:
    if isinstance(value, dict):
        for key in ("cell_index", "cellIndex", "index", "action", "cell"):
            parsed = _validate_cell(_coerce_int(value.get(key)), cell_count)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, list) and value:
        return _validate_cell(_coerce_int(value[0]), cell_count)
    return _validate_cell(_coerce_int(value), cell_count)


def parse_action(text: str, cell_count: int) -> Optional[int]:
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("answer", "move", "action", "response"):
            parsed = _parse_cell_value(payload.get(key), cell_count)
            if parsed is not None:
                return parsed
        for key in ("cell_index", "cellIndex", "index", "action", "cell"):
            parsed = _validate_cell(_coerce_int(payload.get(key)), cell_count)
            if parsed is not None:
                return parsed

    if isinstance(payload, list) and payload:
        parsed = _validate_cell(_coerce_int(payload[0]), cell_count)
        if parsed is not None:
            return parsed

    lowered = text.casefold()
    brace_matches = re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL)
    for brace_text in reversed(brace_matches):
        for pattern in (
            r'"(?:cell_index|cellindex|cell|index|action)"\s*:\s*(-?\d+)',
            r"(?:cell(?:_index)?|index|action)\s*[:=\-]?\s*(-?\d+)",
        ):
            match = re.search(pattern, brace_text, flags=re.DOTALL)
            if match:
                parsed = _validate_cell(int(match.group(1)), cell_count)
                if parsed is not None:
                    return parsed

    textual_patterns = [
        r"(?:cell(?:_index)?|index|action)\s*[:=\-]?\s*(-?\d+)",
        r"(?:choose|select|block)\s+(?:cell\s*)?(-?\d+)",
    ]
    for pattern in textual_patterns:
        matches = re.findall(pattern, lowered, flags=re.DOTALL)
        for candidate in reversed(matches):
            parsed = _validate_cell(int(candidate), cell_count)
            if parsed is not None:
                return parsed

    quoted_matches = re.findall(r'"(-?\d+)"', lowered)
    for candidate in reversed(quoted_matches):
        parsed = _validate_cell(int(candidate), cell_count)
        if parsed is not None:
            return parsed

    integers = re.findall(r"-?\d+", lowered)
    for candidate in reversed(integers):
        parsed = _validate_cell(int(candidate), cell_count)
        if parsed is not None:
            return parsed
    return None


def _cell_answer_to_json(cell_index: int) -> str:
    return json.dumps({"answer": int(cell_index)}, ensure_ascii=False, separators=(",", ":"))


def format_legal_actions_line(legal_actions: Sequence[int]) -> str:
    if not legal_actions:
        return "Legal actions for this turn: <none>."
    return "Legal actions for this turn: " + ", ".join(str(cell_index) for cell_index in legal_actions) + "."


def build_initial_prompt_texts(
    *,
    board_radius: int,
    cat_index: int,
    legal_actions: Sequence[int],
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str]:
    rule_3 = (
        "3. A legal action must be one of the legal actions listed for the current state."
        if include_legal_actions_in_prompt
        else
        "3. A legal action is any currently open non-cat cell. No legal-action list will be provided."
    )
    answer_line = (
        'Replace {cell_index} with the single legal answer for this task, chosen from the legal cell indices listed for the current state, using the JSON shape {"answer":{cell_index}}.'
        if include_legal_actions_in_prompt
        else
        'Replace {cell_index} with one legal open non-cat cell index for the current state, using the JSON shape {"answer":{cell_index}}.'
    )
    before_image_lines = [
        "[Task]",
        load_task_intro_line(
            PROJECT_ROOT,
            default=(
                "You are solving Chat Noir / Encircle the Cat.\n"
                "In this task, you must block one hex cell per turn to surround "
                "the cat before it reaches the boundary."
            ),
        ),
        "In this task, you must trap the cat by blocking one open non-cat cell at each turn before the cat reaches any boundary cell.",
        "Each cell index is printed directly on the board image.",
        "",
        "[Rules]",
        "1. At each turn, choose exactly one open non-cat cell index to block.",
        "2. After your action, the environment applies the block and then the cat may stay still or move one hex step according to the configured cat policy.",
        rule_3,
        "4. Illegal actions keep the state unchanged but still count as a step.",
        "5. The task succeeds when the cat has no remaining path to any boundary cell.",
        "6. The task fails when the cat reaches a boundary cell.",
        "7. At each turn, output exactly one next action for the current state.",
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer":{cell_index}} and nothing else.',
        answer_line,
        "",
        "[Current Task]",
        "The current state image is shown below.",
    ]
    after_image_lines = (
        [
            f"The board radius is {board_radius}.",
            f"The current cat index is {cat_index}.",
            format_legal_actions_line(legal_actions),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            f"The board radius is {board_radius}.",
            f"The current cat index is {cat_index}.",
            "Legal actions for this turn: not provided; infer one legal open non-cat cell index from the board state and cell labels.",
            "To solve this task, output the next legal action for the current state.",
        ]
    )
    return "\n".join(before_image_lines), "\n".join(after_image_lines)


def build_followup_prompt_texts(
    *,
    cat_index: int,
    legal_actions: Sequence[int],
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str]:
    before_image_lines = [
        "[Current Task]",
        "The updated current state image is shown below.",
    ]
    after_image_lines = (
        [
            f"The current cat index is {cat_index}.",
            format_legal_actions_line(legal_actions),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            f"The current cat index is {cat_index}.",
            "Legal actions for this turn: not provided; infer one legal open non-cat cell index from the updated board state and cell labels.",
            "To solve this task, output the next legal action for the current state.",
        ]
    )
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


def build_oracle_action(*, board: Dict[str, Any], state: Dict[str, Any]) -> Optional[int]:
    legal_actions = [int(value) for value in state.get("legalActionIndices", [])]
    if not legal_actions:
        return None

    blocked_set = {int(value) for value in state.get("blockedIndices", [])}
    cat_index = int(state.get("catIndex", -1))
    cat_policy_name = str((state.get("catPolicy") or {}).get("name") or "medium").strip().lower()
    search_result = solve_chat_noir_instance(
        board,
        cat_index,
        sorted(blocked_set),
        cat_policy_name,
        max_steps=len(legal_actions),
        winning_first_action_limit=1,
    )
    search_action = search_result.get("first_action")
    if search_result.get("solvable") and search_action in legal_actions:
        return int(search_action)

    best_action: Optional[int] = None
    best_score: Optional[tuple[float, float, float, float, float]] = None
    for action in legal_actions:
        blocked_after = set(blocked_set)
        blocked_after.add(action)

        if cat_policy_name == "static":
            distances = [shortest_boundary_distance(board, cat_index, blocked_after)]
            cat_move_count = 0
        else:
            cat_moves = legal_cat_moves(board, cat_index, blocked_after)
            cat_move_count = len(cat_moves)
            distances = (
                [None]
                if not cat_moves
                else [shortest_boundary_distance(board, move_index, blocked_after) for move_index in cat_moves]
            )

        normalized_distances = [float("inf") if distance is None else float(distance) for distance in distances]
        all_trapped = bool(normalized_distances) and all(distance == float("inf") for distance in normalized_distances)
        worst_case_distance = min(normalized_distances) if normalized_distances else float("inf")
        mean_distance = (
            sum(normalized_distances) / len(normalized_distances) if normalized_distances else float("inf")
        )
        score = (
            1.0 if all_trapped else 0.0,
            worst_case_distance,
            mean_distance,
            -float(cat_move_count),
            -float(action),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = action
    return best_action


def build_answer_fallback_action(*, oracle_action: Optional[int], legal_actions: Sequence[int]) -> Optional[int]:
    legal_action_set = {int(action) for action in legal_actions}
    if oracle_action is not None and int(oracle_action) in legal_action_set:
        return int(oracle_action)
    if legal_actions:
        return int(legal_actions[0])
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
        cell_count: int,
    ) -> tuple[Optional[int], str, Optional[str], bool, Dict[str, Any]]:
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
            parsed = parse_action(response_text, cell_count)
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
        "avg_initial_escape_distance": mean_or_zero([result.initial_escape_distance for result in results]) if count else 0.0,
        "avg_steps_all": mean_or_zero([result.total_steps for result in results]) if count else 0.0,
        "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
        "avg_illegal_moves": mean_or_zero([result.illegal_moves for result in results]) if count else 0.0,
        "avg_invalid_responses": mean_or_zero([result.invalid_responses for result in results]) if count else 0.0,
        "avg_api_errors": mean_or_zero([result.api_errors for result in results]) if count else 0.0,
    }


def summarize_by_combo(results: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, int, str], List[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault((result.board_radius, result.initial_block_count, result.cat_policy_name), []).append(result)

    rows: List[Dict[str, Any]] = []
    for (board_radius, initial_block_count, cat_policy_name), combo_results in sorted(grouped.items()):
        summary = summarize_results(combo_results)
        rows.append(
            {
                "board_radius": board_radius,
                "initial_block_count": initial_block_count,
                "cat_policy_name": cat_policy_name,
                "cat_policy_level": combo_results[0].cat_policy_level,
                "difficulty_score": combo_results[0].difficulty_score,
                "avg_initial_escape_distance": summary["avg_initial_escape_distance"],
                "repeats": len(combo_results),
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
    fieldnames = [
        "episode_index",
        "episode_id",
        "board_radius",
        "initial_block_count",
        "cat_policy_name",
        "cat_policy_level",
        "difficulty",
        "difficulty_score",
        "repeat_index",
        "seed",
        "initial_escape_distance",
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
            row = {key: getattr(result, key) for key in fieldnames}
            writer.writerow(row)


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


def run_combo(
    *,
    env: EnclosureChatNoirEnv,
    policy: Optional[InternVL3Policy],
    image_writer: StepImageWriter,
    board_radius: int,
    initial_block_count: int,
    cat_policy_name: str,
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
                "board_radius": int(board_radius),
                "initial_block_count": int(initial_block_count),
                "cat_policy_name": cat_policy_name,
                "difficulty": difficulty_label_for_policy(cat_policy_name),
                "reset_options": {
                    "board_radius": int(board_radius),
                    "initial_block_count": int(initial_block_count),
                    "cat_policy": cat_policy_name,
                },
            }
            for repeat_index, episode_seed in enumerate(episode_seeds)
        ]
    progress = tqdm(total=len(specs), desc="Chat Noir episodes", leave=False, unit="episode")

    for spec in specs:
        repeat_index = int(spec.get("repeat_index", len(results)))
        episode_seed = int(spec.get("seed", repeat_index))
        cat_policy_name = str(spec.get("cat_policy_name", cat_policy_name))
        difficulty = str(spec.get("difficulty", difficulty_label_for_policy(cat_policy_name)))
        _, info = env.reset(seed=episode_seed, options=dict(spec.get("reset_options") or {}))

        current_state = info.get("state") or env.get_state()
        actual_board_radius = int(current_state.get("boardRadius", env.board_radius))
        actual_initial_block_count = len(current_state.get("blockedIndices") or [])
        episode_id = str(spec.get("episode_id") or build_episode_id(
            board_radius=actual_board_radius,
            initial_block_count=actual_initial_block_count,
            cat_policy_name=cat_policy_name,
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
        max_model_calls = max_actions_per_traj(
            board_cell_count=int(current_state.get("boardCellCount", 0)),
            initial_block_count=actual_initial_block_count,
        )
        next_info = dict(info)

        for step_index in range(max_model_calls):
            legal_actions = [int(value) for value in current_state.get("legalActionIndices", [])]
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
                    board_radius=actual_board_radius,
                    cat_index=int(current_state.get("catIndex", -1)),
                    legal_actions=legal_actions,
                )
            else:
                before_image_text, after_image_text = build_followup_prompt_texts(
                    cat_index=int(current_state.get("catIndex", -1)),
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

            oracle_action = build_oracle_action(board=env.current_board, state=current_state)
            if oracle:
                predicted_action = oracle_action
                raw_response_text = "<no-legal-action>" if predicted_action is None else _cell_answer_to_json(predicted_action)
                api_error_message = None
                invalid_response = predicted_action is None
                response_debug = {"mode": "oracle"}
            else:
                assert policy is not None
                try:
                    predicted_action, raw_response_text, api_error_message, invalid_response, response_debug = policy.predict_messages(
                        messages=request_messages,
                        cell_count=int(current_state.get("boardCellCount", 0)),
                    )
                except Exception as exc:  # keep prompt/debug artifacts even on unexpected API payloads
                    predicted_action = None
                    raw_response_text = ""
                    api_error_message = f"{type(exc).__name__}: {exc}"
                    invalid_response = False
                    response_debug = {"exception": type(exc).__name__}
                if api_error_message is not None:
                    api_errors += 1

            if invalid_response:
                invalid_responses += 1

            fallback_action: Optional[int] = None
            if predicted_action is not None and predicted_action in legal_actions:
                action_index = predicted_action
                assistant_history_text = raw_response_text or ""
            else:
                if invalid_response:
                    fallback_action = build_answer_fallback_action(
                        oracle_action=oracle_action,
                        legal_actions=legal_actions,
                    )
                if fallback_action is not None:
                    response_debug = dict(response_debug or {})
                    response_debug["answer_fallback"] = True
                    response_debug["fallback_reason"] = "invalid_response"
                    response_debug["fallback_action"] = {"cell_index": fallback_action}
                    predicted_action = fallback_action
                    action_index = fallback_action
                    assistant_history_text = _cell_answer_to_json(fallback_action)
                else:
                    action_index = INVALID_ACTION
                    assistant_history_text = raw_response_text or ""

            _, reward, terminated, truncated, next_info = env.step(action_index)
            if next_info.get("illegal"):
                illegal_moves += 1
            final_reward = float(reward)
            final_reason = next_info.get("reason")

            conversation_messages.append({"role": "user", "content": user_content})
            conversation_messages.append({"role": "assistant", "content": assistant_history_text})
            conversation_debug_messages.append({"role": "user", "content": debug_user_content})
            conversation_debug_messages.append({"role": "assistant", "content": assistant_history_text})

            saved_prompt_debug_path = image_writer.save_prompt_debug(
                episode_id=episode_id,
                step_index=step_index,
                prompt_messages=request_debug_messages,
                ground_truth=oracle_action,
                raw_response_text=raw_response_text,
                parsed_answer=predicted_action,
                api_error_message=api_error_message,
                response_debug=response_debug,
            )

            step_logs.append(
                {
                    "step_index": step_index,
                    "seed": episode_seed,
                    "state_before_action": current_state,
                    "legal_actions": [{"cell_index": cell_index} for cell_index in legal_actions],
                    "ground_truth": None if oracle_action is None else {"cell_index": oracle_action},
                    "predicted_action": None if predicted_action is None else {"cell_index": predicted_action},
                    "answer_fallback": fallback_action is not None,
                    "fallback_action": None if fallback_action is None else {"cell_index": fallback_action},
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

        if not terminated and not truncated and len(step_logs) >= max_model_calls:
            truncated = True
            final_reason = final_reason or "internal_step_limit"

        result = EpisodeResult(
            episode_index=len(results),
            episode_id=episode_id,
            board_radius=actual_board_radius,
            initial_block_count=actual_initial_block_count,
            cat_policy_name=cat_policy_name,
            cat_policy_level=int(CAT_POLICIES[cat_policy_name]["level"]),
            difficulty=difficulty,
            difficulty_score=float(int(CAT_POLICIES[cat_policy_name]["level"])),
            repeat_index=repeat_index,
            seed=episode_seed,
            initial_escape_distance=int(info.get("initial_escape_distance", 0)),
            min_winning_first_actions=int(info.get("min_winning_first_actions", 0)),
            min_adjacent_winning_first_actions=int(info.get("min_adjacent_winning_first_actions", 0)),
            winning_first_action_count=int((info.get("solvability", {}) or {}).get("winning_first_action_count", 0)),
            winning_first_actions=list((info.get("solvability", {}) or {}).get("winning_first_actions", [])),
            adjacent_winning_first_action_count=int(
                (info.get("solvability", {}) or {}).get("adjacent_winning_first_action_count", 0)
            ),
            adjacent_winning_first_actions=list(
                (info.get("solvability", {}) or {}).get("adjacent_winning_first_actions", [])
            ),
            step_budget=max_model_calls,
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
    parser = argparse.ArgumentParser(description="InternVL3 benchmark for enclosure_chat_noir.")
    parser.add_argument("--config", default=None, help="Path to local JSON config with api_key/base_url/model.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--board-radius", "--radii", dest="board_radius", default=DEFAULT_BOARD_RADIUS, help='Use "auto" or comma-separated ints/ranges, e.g. 3-4')
    parser.add_argument("--initial-block-count", "--blocked", dest="initial_block_count", default=DEFAULT_INITIAL_BLOCK_COUNT, help='Use "auto" or comma-separated ints/ranges, e.g. 10,13-17')
    parser.add_argument("--cat-policies", default=DEFAULT_CAT_POLICIES, help="Comma-separated policies from {easy,medium,hard}; this defines difficulty.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--min-winning-first-actions",
        default=DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
        help=(
            'Use "auto" for policy defaults '
            f"(easy={DEFAULT_MIN_WINNING_FIRST_ACTIONS}, medium=5, hard=5), or pass one positive integer override."
        ),
    )
    parser.add_argument(
        "--min-adjacent-winning-first-actions",
        type=int,
        default=DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
        help="Minimum winning first block actions that must be adjacent to the initial cat cell.",
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
    parser.add_argument("--oracle", action="store_true", help="Use the internal heuristic policy instead of the API.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if not is_auto_setup_value(args.min_winning_first_actions) and int(args.min_winning_first_actions) <= 0:
        raise ValueError("--min-winning-first-actions must be positive.")
    if args.min_adjacent_winning_first_actions < 0:
        raise ValueError("--min-adjacent-winning-first-actions must be non-negative.")

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
    if question_jsonl is not None:
        loaded_specs = load_episode_specs(question_jsonl)
        if not loaded_specs:
            raise ValueError(f"Question JSONL has no episode rows: {question_jsonl}")
        board_radii = sorted({validate_board_radius(spec["board_radius"]) for spec in loaded_specs})
        blocked_counts = sorted({int(spec["initial_block_count"]) for spec in loaded_specs})
        cat_policies = sorted({str(spec["cat_policy_name"]) for spec in loaded_specs})
        episode_specs = loaded_specs
        repeats_per_policy = None
        seed_start = None
    else:
        board_radii = parse_board_radius_specs(args.board_radius)
        blocked_counts = parse_initial_block_count_specs(args.initial_block_count)
        cat_policies = parse_policy_list(args.cat_policies)
        episode_specs = build_direct_episode_specs(
            board_radii=board_radii,
            blocked_counts=blocked_counts,
            cat_policies=cat_policies,
            repeats=args.repeats,
            seed_start=args.seed_start,
            min_winning_first_actions=args.min_winning_first_actions,
            min_adjacent_winning_first_actions=int(args.min_adjacent_winning_first_actions),
        )
        repeats_per_policy = args.repeats
        seed_start = args.seed_start
    for cat_policy_name in cat_policies:
        if cat_policy_name not in CAT_POLICIES:
            raise ValueError(f"Unsupported cat policy: {cat_policy_name}")
    if not episode_specs:
        raise ValueError("No Chat Noir episodes were configured.")

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_by_combo_json = output_json.with_name(f"{output_json.stem}_by_combo.json")
    output_by_combo_csv = output_csv.with_name(f"{output_csv.stem}_by_combo.csv")

    image_writer = StepImageWriter(root_dir=Path(args.image_root).resolve())
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

    first_spec = episode_specs[0]
    env_init_board_radius = (
        3
        if is_auto_setup_value(first_spec["board_radius"])
        else validate_board_radius(first_spec["board_radius"])
    )
    env_init_block_count = 0 if is_auto_setup_value(first_spec["initial_block_count"]) else int(first_spec["initial_block_count"])
    env = EnclosureChatNoirEnv(
        board_radius=env_init_board_radius,
        initial_block_count=env_init_block_count,
        cat_policy=str(first_spec["cat_policy_name"]),
        headless=args.headless,
        animate=args.animate,
        host=args.host,
        port=args.port,
    )
    all_results: List[EpisodeResult] = []
    try:
        all_results = run_combo(
            env=env,
            policy=policy,
            image_writer=image_writer,
            board_radius=env_init_board_radius,
            initial_block_count=env_init_block_count,
            cat_policy_name=str(first_spec["cat_policy_name"]),
            episode_seeds=[],
            oracle=args.oracle,
            episode_specs=episode_specs,
        )
        for episode_index, result in enumerate(all_results):
            result.episode_index = episode_index
    finally:
        env.close()

    overall_summary = summarize_results(all_results)
    by_combo_summary = summarize_by_combo(all_results)
    payload = {
        "config": {
            "board_radius": board_radii,
            "initial_block_count": blocked_counts,
            "cat_policies": cat_policies,
            "repeats_per_policy": repeats_per_policy,
            "seed_start": seed_start,
            "difficulty_rule": "cat_policy defines difficulty: easy=easy, medium=medium, hard=hard",
            "model": model,
            "base_url": base_url,
            "oracle": args.oracle,
            "max_tokens": args.max_tokens,
            "headless": args.headless,
            "animate": args.animate,
            "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        },
        "summary": overall_summary,
        "summary_by_combo": by_combo_summary,
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
    output_by_combo_json.write_text(json.dumps(by_combo_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_csv(output_by_combo_csv, by_combo_summary)

    print(
        f"success_rate={overall_summary['success_rate']:.3f} | "
        f"successes={overall_summary['successes']}/{overall_summary['total_episodes']} | "
        f"illegal={overall_summary['avg_illegal_moves']:.3f} | "
        f"invalid={overall_summary['avg_invalid_responses']:.3f} | "
        f"api_errors={overall_summary['avg_api_errors']:.3f}"
    )
    print(f"Wrote JSON results to {output_json}")
    print(f"Wrote CSV results to {output_csv}")
    print(f"Wrote combo JSON summary to {output_by_combo_json}")
    print(f"Wrote combo CSV summary to {output_by_combo_csv}")
    if question_jsonl is None:
        print(f"Wrote question JSONL to {question_jsonl_path}")
    else:
        print(f"Loaded question JSONL from {question_jsonl}")
    print(f"Wrote model answer JSONL to {model_answer_jsonl_path}")


if __name__ == "__main__":
    main()
