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

from env import (  # noqa: E402
    BLANK_TOKEN,
    OrderSwap2DPuzzleEnv,
    cell_index_to_row_col,
    difficulty_label_for_grid,
    grid_shape_for_difficulty,
    normalize_difficulty,
    row_col_to_cell_index,
    shortest_action_sequence,
)
from internvl3_config import load_local_config, resolve_config_value  # noqa: E402
from topobench_eval.metadata_utils import load_task_intro_line  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from jsonl_export import (  # noqa: E402
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


TASK_NAME = "order_swap_2d_puzzle"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_order_swap_2d_puzzle.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_order_swap_2d_puzzle.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_GENERATE_ROOT = PROJECT_ROOT / "output" / "generate_samples"
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0
DEFAULT_DIFFICULTIES = "easy,medium,hard"
DEFAULT_REPEATS = 3
DEFAULT_SEED_START = 1
INVALID_ACTION = -999
STATE_CAPTION_FONT_SIZE = 34
STATE_CAPTION_TOP_MARGIN = 14


@dataclass
class EpisodeResult:
    episode_index: int
    episode_id: str
    grid_rows: int
    grid_cols: int
    difficulty: str
    repeat_index: int
    seed: int
    theoretical_min_steps: int
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


def build_episode_id(*, grid_rows: int, grid_cols: int, seed: int) -> str:
    return f"order_swap_2d_puzzle_grid_{grid_rows}x{grid_cols}_seed_{seed}"


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
        seed = int(reset_config.get("seed", meta_info.get("seed", row_index)))
        has_grid_shape = (
            "gridRows" in reset_config
            or "gridCols" in reset_config
            or "gridRows" in initial_state
            or "gridCols" in initial_state
        )
        target_difficulty = normalize_difficulty(reset_config.get("targetDifficulty", None))
        if target_difficulty is None and not has_grid_shape:
            target_difficulty = normalize_difficulty(reset_config.get("difficulty", meta_info.get("difficulty", None)))
        if target_difficulty is not None:
            grid_rows, grid_cols = grid_shape_for_difficulty(target_difficulty, seed)
        else:
            grid_rows = int(reset_config.get("gridRows", initial_state.get("gridRows", 3)))
            grid_cols = int(reset_config.get("gridCols", initial_state.get("gridCols", 3)))
        repeat_index = int(meta_info.get("repeat_index", row_index))
        difficulty = str(meta_info.get("difficulty", reset_config.get("difficulty", difficulty_label_for_grid(grid_rows, grid_cols))))
        specs.append(
            {
                "episode_id": str(row.get("id") or build_episode_id(
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    seed=seed,
                )),
                "repeat_index": repeat_index,
                "seed": seed,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "difficulty": difficulty,
                "reset_options": {"frontend_config": reset_config},
            }
        )
    return specs


def build_initial_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    palette = state.get("blockPalette")
    if isinstance(palette, list):
        palette = [
            entry.get("id") if isinstance(entry, dict) else entry
            for entry in palette
        ]
    return {
        "gridRows": state.get("gridRows"),
        "gridCols": state.get("gridCols"),
        "blockPalette": palette,
        "currentArrangement": state.get("currentArrangement"),
        "currentGrid": state.get("currentGrid"),
        "goalArrangement": state.get("goalArrangement"),
        "goalGrid": state.get("goalGrid"),
        "blankCellIndex": state.get("blankCellIndex"),
        "blankRow": state.get("blankRow"),
        "blankCol": state.get("blankCol"),
    }


def action_answer_value(action: Any) -> Any:
    if not isinstance(action, dict):
        return None
    row = action.get("row")
    col = action.get("col")
    if row is None or col is None:
        return None
    return {"row": int(row), "col": int(col)}


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
        grid_shape = f"{result.grid_rows}x{result.grid_cols}"
        first_step = result.steps[0] if result.steps else None
        initial_state_snapshot = build_initial_state_snapshot(first_step.get("state_before_action", {})) if first_step else {}
        initial_state: Dict[str, Any] = {}
        if initial_state_snapshot:
            reset_config = {
                "gridRows": int(result.grid_rows),
                "gridCols": int(result.grid_cols),
                "difficulty": difficulty,
                "targetDifficulty": difficulty if difficulty in {"easy", "medium", "hard"} else None,
                "seed": int(result.seed),
                "selectedBlockIds": list(initial_state_snapshot.get("blockPalette") or []),
                "initialArrangement": list(initial_state_snapshot.get("currentArrangement") or []),
                "goalArrangement": list(initial_state_snapshot.get("goalArrangement") or []),
                "theoreticalMinSteps": int(result.theoretical_min_steps),
                "stepBudget": int(result.step_budget),
            }
            if reset_config["targetDifficulty"] is None:
                reset_config.pop("targetDifficulty", None)
            initial_state = {
                "initialGrid": initial_state_snapshot.get("currentGrid"),
                "goalGrid": initial_state_snapshot.get("goalGrid"),
                "reset_config": reset_config,
            }
        initial_images: List[str] = []
        if first_step:
            initial_images = [
                jsonl_relpath(
                    resolve_existing_path(
                        first_step["saved_current_image_path"],
                        output_dir,
                        PROJECT_ROOT,
                        DEFAULT_GENERATE_ROOT,
                        DEFAULT_OUTPUT_ROOT,
                    ),
                    base_dir=output_dir,
                ),
                jsonl_relpath(
                    resolve_existing_path(
                        first_step["saved_goal_image_path"],
                        output_dir,
                        PROJECT_ROOT,
                        DEFAULT_GENERATE_ROOT,
                        DEFAULT_OUTPUT_ROOT,
                    ),
                    base_dir=output_dir,
                ),
            ]
        question_row = {
            "id": result.episode_id,
            "category": ["order", TASK_NAME, "interactive"],
            "type": "interactive",
            "meta_info": {
                "task_name": TASK_NAME,
                "config": config_rel,
                "level": f"grid_{grid_shape}",
                "seed": result.seed,
                "repeat_index": result.repeat_index,
                "difficulty": difficulty,
                "grid_shape": grid_shape,
                "max_actions_per_traj": result.step_budget,
                "initial_state": initial_state,
                "legal_action_format": '{"answer": {"row": row, "col": col}}',
            },
            "images": initial_images,
        }
        question_rows.append(question_row)
        answer_rows.append(
            {
                "id": result.episode_id,
                "category": ["order", TASK_NAME, "interactive"],
                "type": "interactive",
                "meta_info": {
                    "task_name": TASK_NAME,
                    "config": config_rel,
                    "level": f"grid_{grid_shape}",
                    "seed": result.seed,
                    "repeat_index": result.repeat_index,
                    "difficulty": difficulty,
                    "grid_shape": grid_shape,
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
                            ),
                            jsonl_relpath(
                                resolve_existing_path(
                                    step["saved_goal_image_path"],
                                    output_dir,
                                    PROJECT_ROOT,
                                    DEFAULT_GENERATE_ROOT,
                                    DEFAULT_OUTPUT_ROOT,
                                ),
                                base_dir=output_dir,
                            ),
                        ],
                        "state": interactive_trajectory_state(
                            trajectory_index=trajectory_index,
                            total_steps=len(result.steps),
                            success=bool(result.success),
                        ),
                        "answer": action_answer_value(step.get("predicted_action")),
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
                display_path = os.path.relpath(Path(raw_path), image_root.parent)
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
        ground_truth: Optional[Dict[str, int]],
        raw_response_text: str,
        parsed_answer: Optional[Dict[str, int]],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]],
    ) -> Path:
        episode_dir = self.episode_dir(episode_id=episode_id)
        output_path = episode_dir / f"step_{step_index:04d}_prompt_debug.txt"
        ground_truth_text = "<none>" if ground_truth is None else _answer_to_json(ground_truth)
        parsed_text = "<none>" if parsed_answer is None else _answer_to_json(parsed_answer)
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


def add_image_caption(png_bytes: bytes, caption_text: str) -> bytes:
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


def compute_legal_actions(state: Dict[str, Any]) -> List[Dict[str, int]]:
    arrangement = state.get("currentArrangement")
    grid_cols = int(state.get("gridCols", 0) or 0)
    if not isinstance(arrangement, list) or grid_cols <= 0:
        return []
    actions: List[Dict[str, int]] = []
    for cell_index, token in enumerate(arrangement):
        if token == BLANK_TOKEN:
            continue
        row_col = cell_index_to_row_col(cell_index, grid_cols)
        actions.append({"row": int(row_col["row"]), "col": int(row_col["col"]), "cell_index": int(cell_index)})
    return actions


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _validate_cell(cell_index: Optional[int], grid_rows: int, grid_cols: int) -> Optional[Dict[str, int]]:
    if cell_index is None:
        return None
    cell_count = int(grid_rows) * int(grid_cols)
    if 0 <= int(cell_index) < cell_count:
        row_col = cell_index_to_row_col(int(cell_index), grid_cols)
        return {"row": int(row_col["row"]), "col": int(row_col["col"]), "cell_index": int(cell_index)}
    return None


def _validate_row_col(row: Optional[int], col: Optional[int], grid_rows: int, grid_cols: int) -> Optional[Dict[str, int]]:
    if row is None or col is None:
        return None
    if 0 <= int(row) < int(grid_rows) and 0 <= int(col) < int(grid_cols):
        cell_index = row_col_to_cell_index(int(row), int(col), int(grid_cols))
        return {"row": int(row), "col": int(col), "cell_index": int(cell_index)}
    return None


def _parse_action_value(value: Any, grid_rows: int, grid_cols: int) -> Optional[Dict[str, int]]:
    if isinstance(value, dict):
        parsed = _validate_row_col(
            _coerce_int(value.get("row", value.get("r"))),
            _coerce_int(value.get("col", value.get("column", value.get("c")))),
            grid_rows,
            grid_cols,
        )
        if parsed is not None:
            return parsed
        for key in ("cell_index", "cellIndex", "index", "slot_index", "slotIndex", "action"):
            parsed = _validate_cell(_coerce_int(value.get(key)), grid_rows, grid_cols)
            if parsed is not None:
                return parsed
    if isinstance(value, list) and len(value) >= 2:
        parsed = _validate_row_col(_coerce_int(value[0]), _coerce_int(value[1]), grid_rows, grid_cols)
        if parsed is not None:
            return parsed
    if isinstance(value, list) and value:
        parsed = _validate_cell(_coerce_int(value[0]), grid_rows, grid_cols)
        if parsed is not None:
            return parsed
    if isinstance(value, str):
        return parse_action(value, grid_rows, grid_cols)
    parsed = _validate_cell(_coerce_int(value), grid_rows, grid_cols)
    if parsed is not None:
        return parsed
    return None


def parse_action(text: str, grid_rows: int, grid_cols: int) -> Optional[Dict[str, int]]:
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for answer_key in ("answer", "move", "action"):
            parsed = _parse_action_value(payload.get(answer_key), grid_rows, grid_cols)
            if parsed is not None:
                return parsed
        parsed = _parse_action_value(payload, grid_rows, grid_cols)
        if parsed is not None:
            return parsed

    if isinstance(payload, list):
        parsed = _parse_action_value(payload, grid_rows, grid_cols)
        if parsed is not None:
            return parsed

    lowered = text.casefold()
    row_col_patterns = [
        r'"row"\s*:\s*(-?\d+).*?"(?:col|column)"\s*:\s*(-?\d+)',
        r"(?:row|r)\s*[:=\-]?\s*(-?\d+).*?(?:col|column|c)\s*[:=\-]?\s*(-?\d+)",
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
    ]
    for pattern in row_col_patterns:
        matches = re.findall(pattern, lowered, flags=re.DOTALL)
        for row_text, col_text in reversed(matches):
            parsed = _validate_row_col(int(row_text), int(col_text), grid_rows, grid_cols)
            if parsed is not None:
                return parsed

    cell_patterns = [
        r'"(?:cell_index|cellindex|index|slot_index|slotindex|action)"\s*:\s*(-?\d+)',
        r"(?:cell(?:_index)?|index|slot(?:_index)?|action)\s*[:=\-]?\s*(-?\d+)",
    ]
    for pattern in cell_patterns:
        matches = re.findall(pattern, lowered, flags=re.DOTALL)
        for candidate in reversed(matches):
            parsed = _validate_cell(int(candidate), grid_rows, grid_cols)
            if parsed is not None:
                return parsed

    integers = re.findall(r"-?\d+", lowered)
    if len(integers) >= 2:
        parsed = _validate_row_col(int(integers[-2]), int(integers[-1]), grid_rows, grid_cols)
        if parsed is not None:
            return parsed
    for candidate in reversed(integers):
        parsed = _validate_cell(int(candidate), grid_rows, grid_cols)
        if parsed is not None:
            return parsed
    return None


def action_to_index(action: Any, grid_rows: int, grid_cols: int) -> int:
    parsed = _parse_action_value(action, grid_rows, grid_cols)
    return INVALID_ACTION if parsed is None else int(parsed["cell_index"])


def _action_to_json(action: Dict[str, int]) -> str:
    return json.dumps({"row": int(action["row"]), "col": int(action["col"])}, separators=(",", ":"))


def _answer_to_json(action: Dict[str, int]) -> str:
    return json.dumps({"answer": {"row": int(action["row"]), "col": int(action["col"])}}, separators=(",", ":"))


def format_legal_actions_line(legal_actions: Sequence[Dict[str, int]]) -> str:
    if not legal_actions:
        return "Legal actions for this turn: <none>."
    return "Legal actions for this turn: " + ", ".join(_action_to_json(action) for action in legal_actions) + "."


def task_intro_text() -> str:
    return load_task_intro_line(
        PROJECT_ROOT,
        default=(
            "You need to solve a 2D grid swapping puzzle that requires ordering ability.\n"
            "In this task, you must swap one colored block with the empty "
            "cell per turn until the grid matches the target arrangement."
        ),
    ).replace(
        "You are solving Order Swap 2D Puzzle.",
        "You need to solve a 2D grid swapping puzzle that requires ordering ability.",
    )


def build_initial_prompt_texts(
    *,
    grid_rows: int,
    grid_cols: int,
    legal_actions: Sequence[Dict[str, int]],
    include_difficulty: bool = False,
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str, str]:
    rule_3 = (
        "3. A legal action must be one of the legal actions listed for the current state."
        if include_legal_actions_in_prompt
        else "3. A legal action is any non-empty cell in the current grid. No legal-action list will be provided."
    )
    answer_line = (
        "Replace row and col with the single legal answer for this task, chosen from the legal row/column cells listed for the current state."
        if include_legal_actions_in_prompt
        else "Replace row and col with one legal non-empty cell for the current state."
    )
    before_current_lines = [
        "[Task]",
        task_intro_text(),
        "Row and column coordinates are zero-based: rows increase from top to bottom, and columns increase from left to right.",
        "",
        "[Rules]",
        "1. At each turn, choose exactly one non-empty cell by row and column.",
        "2. The selected block swaps positions with the blank cell.",
        rule_3,
        "4. After each action, the environment returns the next state. Illegal actions keep the state unchanged but still count as a step.",
        "5. The task is solved when the current grid exactly matches the goal grid.",
        "6. The episode ends when the puzzle reaches a done state or the step budget is exhausted.",
        "7. At each turn, output exactly one next action for the current state.",
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer": {"row": row, "col": col}} and nothing else.',
        answer_line,
        "",
        "[Current Task]",
        "The current state image is shown below.",
    ]
    between_images_lines = [
        "The goal state image is shown below.",
    ]
    after_goal_lines = (
        [
            format_legal_actions_line(legal_actions),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            "Legal actions for this turn: not provided; infer one legal non-empty cell from the current grid.",
            "To solve this task, output the next legal action for the current state.",
        ]
    )
    return "\n".join(before_current_lines), "\n".join(between_images_lines), "\n".join(after_goal_lines)


def build_followup_prompt_texts(
    *,
    grid_rows: int,
    grid_cols: int,
    legal_actions: Sequence[Dict[str, int]],
    include_difficulty: bool = False,
    include_legal_actions_in_prompt: bool = True,
) -> tuple[str, str, str]:
    before_current_lines = [
        "[Current Task]",
        "The updated current state image is shown below.",
    ]
    between_images_lines = [
        "The goal state image is shown below.",
    ]
    after_goal_lines = (
        [
            format_legal_actions_line(legal_actions),
            "To solve this task, output the next legal action for the current state.",
        ]
        if include_legal_actions_in_prompt
        else
        [
            "Legal actions for this turn: not provided; infer one legal non-empty cell from the updated current grid.",
            "To solve this task, output the next legal action for the current state.",
        ]
    )
    return "\n".join(before_current_lines), "\n".join(between_images_lines), "\n".join(after_goal_lines)


def build_user_content(
    *,
    before_current_text: str,
    current_image_data_url: str,
    between_images_text: str,
    goal_image_data_url: str,
    after_goal_text: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": before_current_text}]
    content.append({"type": "image_url", "image_url": {"url": current_image_data_url}})
    if between_images_text:
        content.append({"type": "text", "text": between_images_text})
    content.append({"type": "image_url", "image_url": {"url": goal_image_data_url}})
    if after_goal_text:
        content.append({"type": "text", "text": after_goal_text})
    return content


def build_debug_user_content(
    *,
    before_current_text: str,
    current_image_path: Path,
    between_images_text: str,
    goal_image_path: Path,
    after_goal_text: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": before_current_text}]
    content.append({"type": "image_path", "path": str(current_image_path)})
    if between_images_text:
        content.append({"type": "text", "text": between_images_text})
    content.append({"type": "image_path", "path": str(goal_image_path)})
    if after_goal_text:
        content.append({"type": "text", "text": after_goal_text})
    return content


def build_pure_user_content(
    *,
    before_current_text: str,
    current_image_path: Path,
    between_images_text: str,
    goal_image_path: Path,
    after_goal_text: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": before_current_text}]
    content.append({"type": "image_path", "path": str(current_image_path)})
    if between_images_text:
        content.append({"type": "text", "text": between_images_text})
    content.append({"type": "image_path", "path": str(goal_image_path)})
    if after_goal_text:
        content.append({"type": "text", "text": after_goal_text})
    return content


def build_oracle_action(state: Dict[str, Any]) -> Optional[Dict[str, int]]:
    current_arrangement = state.get("currentArrangement")
    goal_arrangement = state.get("goalArrangement")
    grid_cols = int(state.get("gridCols", 0) or 0)
    if not isinstance(current_arrangement, list) or not isinstance(goal_arrangement, list) or grid_cols <= 0:
        return None
    actions = shortest_action_sequence(current_arrangement, goal_arrangement)
    if not actions:
        return None
    row_col = cell_index_to_row_col(int(actions[0]), grid_cols)
    return {"row": int(row_col["row"]), "col": int(row_col["col"]), "cell_index": int(actions[0])}


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
        grid_rows: int,
        grid_cols: int,
    ) -> tuple[Optional[Dict[str, int]], str, Optional[str], bool, Dict[str, Any]]:
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
            parsed = parse_action(response_text, grid_rows, grid_cols)
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
        "avg_theoretical_min_steps": mean_or_zero([result.theoretical_min_steps for result in results]) if count else 0.0,
        "avg_step_budget": mean_or_zero([result.step_budget for result in results]) if count else 0.0,
        "avg_steps_all": mean_or_zero([result.total_steps for result in results]) if count else 0.0,
        "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
        "avg_illegal_moves": mean_or_zero([result.illegal_moves for result in results]) if count else 0.0,
        "avg_invalid_responses": mean_or_zero([result.invalid_responses for result in results]) if count else 0.0,
        "avg_api_errors": mean_or_zero([result.api_errors for result in results]) if count else 0.0,
    }


def summarize_by_grid(results: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int], List[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault((result.difficulty, result.grid_rows, result.grid_cols), []).append(result)

    rows: List[Dict[str, Any]] = []
    for (difficulty, grid_rows, grid_cols), grid_results in sorted(grouped.items()):
        summary = summarize_results(grid_results)
        rows.append(
            {
                "difficulty": difficulty,
                "grid_shape": f"{grid_rows}x{grid_cols}",
                "repeats": len(grid_results),
                "successes": summary["successes"],
                "success_rate": summary["success_rate"],
                "avg_theoretical_min_steps": summary["avg_theoretical_min_steps"],
                "avg_step_budget": summary["avg_step_budget"],
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
        "grid_rows",
        "grid_cols",
        "difficulty",
        "repeat_index",
        "seed",
        "theoretical_min_steps",
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


def parse_difficulty_list(spec: str) -> List[str]:
    values: List[str] = []
    for chunk in str(spec).split(","):
        part = chunk.strip()
        if not part:
            continue
        difficulty = normalize_difficulty(part)
        if difficulty is None:
            raise ValueError(f"Difficulty must be easy, medium, or hard, got: {part!r}")
        values.append(difficulty)
    if not values:
        raise ValueError(f"Empty difficulty list spec: {spec!r}")
    return values


def run_combo(
    *,
    env: OrderSwap2DPuzzleEnv,
    policy: Optional[InternVL3Policy],
    image_writer: StepImageWriter,
    difficulty: str,
    episode_seeds: Sequence[int],
    budget_multiplier: float,
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
                "difficulty": difficulty,
                "reset_options": {
                    "frontend_config": {
                        "targetDifficulty": difficulty,
                        "seed": int(episode_seed),
                        "budgetMultiplier": budget_multiplier,
                    }
                },
            }
            for repeat_index, episode_seed in enumerate(episode_seeds)
        ]

    progress = tqdm(total=len(specs), desc=f"{difficulty}", leave=False, unit="episode")
    for spec in specs:
        repeat_index = int(spec.get("repeat_index", len(results)))
        episode_seed = int(spec.get("seed", repeat_index))
        difficulty = str(spec.get("difficulty", difficulty))
        _, info = env.reset(seed=episode_seed, options=dict(spec.get("reset_options") or {}))
        current_state = info.get("state") or env.get_state()
        grid_rows = int(current_state.get("gridRows", info.get("grid_rows", 3)))
        grid_cols = int(current_state.get("gridCols", info.get("grid_cols", 3)))
        episode_id = str(spec.get("episode_id") or build_episode_id(grid_rows=grid_rows, grid_cols=grid_cols, seed=episode_seed))
        conversation_messages: List[Dict[str, Any]] = []
        conversation_pure_messages: List[Dict[str, Any]] = []
        conversation_debug_messages: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        illegal_moves = 0
        invalid_responses = 0
        api_errors = 0
        terminated = False
        truncated = False
        final_reward = 0.0
        final_reason: Optional[str] = None
        theoretical_min_steps = int(info.get("theoretical_min_steps", env.current_theoretical_min_steps))
        step_budget = int(info.get("step_budget", env.current_step_budget))

        for step_index in range(step_budget):
            legal_actions = compute_legal_actions(current_state)
            current_png_bytes = add_image_caption(env.screenshot_current_card(), f"State {step_index}")
            goal_png_bytes = add_image_caption(env.screenshot_goal_card(), "Goal State")
            saved_current_image_path = image_writer.save_png(
                episode_id=episode_id,
                step_index=step_index,
                image_kind="current",
                png_bytes=current_png_bytes,
            )
            saved_goal_image_path = image_writer.save_png(
                episode_id=episode_id,
                step_index=step_index,
                image_kind="goal",
                png_bytes=goal_png_bytes,
            )

            current_image_data_url = png_bytes_to_data_url(current_png_bytes)
            goal_image_data_url = png_bytes_to_data_url(goal_png_bytes)

            if step_index == 0:
                before_current_text, between_images_text, after_goal_text = build_initial_prompt_texts(
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    legal_actions=legal_actions,
                )
                debug_before_current_text, debug_between_images_text, debug_after_goal_text = build_initial_prompt_texts(
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    legal_actions=legal_actions,
                    include_difficulty=True,
                )
            else:
                before_current_text, between_images_text, after_goal_text = build_followup_prompt_texts(
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    legal_actions=legal_actions,
                )
                debug_before_current_text, debug_between_images_text, debug_after_goal_text = build_followup_prompt_texts(
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    legal_actions=legal_actions,
                    include_difficulty=True,
                )

            user_content = build_user_content(
                before_current_text=before_current_text,
                current_image_data_url=current_image_data_url,
                between_images_text=between_images_text,
                goal_image_data_url=goal_image_data_url,
                after_goal_text=after_goal_text,
            )
            debug_user_content = build_debug_user_content(
                before_current_text=debug_before_current_text,
                current_image_path=saved_current_image_path,
                between_images_text=debug_between_images_text,
                goal_image_path=saved_goal_image_path,
                after_goal_text=debug_after_goal_text,
            )
            pure_user_content = build_pure_user_content(
                before_current_text=before_current_text,
                current_image_path=saved_current_image_path,
                between_images_text=between_images_text,
                goal_image_path=saved_goal_image_path,
                after_goal_text=after_goal_text,
            )
            request_messages = [*conversation_messages, {"role": "user", "content": user_content}]
            request_pure_messages = [*conversation_pure_messages, {"role": "user", "content": pure_user_content}]
            request_debug_messages = [*conversation_debug_messages, {"role": "user", "content": debug_user_content}]
            saved_pure_prompt_path = image_writer.save_pure_prompt(
                episode_id=episode_id,
                step_index=step_index,
                prompt_messages=request_pure_messages,
            )

            oracle_action = build_oracle_action(current_state)
            if oracle:
                predicted_action = oracle_action
                raw_response_text = "<no-solution>" if predicted_action is None else _answer_to_json(predicted_action)
                api_error_message = None
                invalid_response = predicted_action is None
                response_debug = {"mode": "oracle"}
            else:
                assert policy is not None
                try:
                    predicted_action, raw_response_text, api_error_message, invalid_response, response_debug = policy.predict_messages(
                        messages=request_messages,
                        grid_rows=grid_rows,
                        grid_cols=grid_cols,
                    )
                except Exception as exc:
                    predicted_action = None
                    raw_response_text = ""
                    api_error_message = f"{type(exc).__name__}: {exc}"
                    invalid_response = False
                    response_debug = {"exception": type(exc).__name__}
                if api_error_message is not None:
                    api_errors += 1

            if invalid_response:
                invalid_responses += 1

            if predicted_action is not None and predicted_action in legal_actions:
                action_index = int(predicted_action["cell_index"])
            else:
                action_index = INVALID_ACTION

            _, reward, terminated, truncated, next_info = env.step(action_index)
            if next_info.get("illegal"):
                illegal_moves += 1
            final_reward = float(reward)
            final_reason = next_info.get("reason") or next_info.get("truncated_reason")

            conversation_messages.append({"role": "user", "content": user_content})
            conversation_messages.append({"role": "assistant", "content": raw_response_text or ""})
            conversation_pure_messages.append({"role": "user", "content": pure_user_content})
            conversation_pure_messages.append({"role": "assistant", "content": raw_response_text or ""})
            conversation_debug_messages.append({"role": "user", "content": debug_user_content})
            conversation_debug_messages.append({"role": "assistant", "content": raw_response_text or ""})

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
                    "legal_actions": legal_actions,
                    "ground_truth": oracle_action,
                    "predicted_action": predicted_action,
                    "raw_response_text": raw_response_text,
                    "invalid_response": invalid_response,
                    "api_error_message": api_error_message,
                    "illegal": bool(next_info.get("illegal", False)),
                    "reason": next_info.get("reason"),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "saved_current_image_path": to_relative_path(saved_current_image_path),
                    "saved_goal_image_path": to_relative_path(saved_goal_image_path),
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
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            difficulty=difficulty,
            repeat_index=repeat_index,
            seed=episode_seed,
            theoretical_min_steps=theoretical_min_steps,
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
    parser = argparse.ArgumentParser(description="InternVL3 benchmark for order_swap_2d_puzzle.")
    parser.add_argument("--config", default=None, help="Path to local JSON config with api_key/base_url/model.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--difficulties", default=DEFAULT_DIFFICULTIES, help="Comma-separated difficulties, e.g. easy,medium,hard")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="First deterministic seed. Defaults to 1.")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--question-jsonl", default="", help="Load existing interactive question.jsonl rows via reset_config.")
    parser.add_argument("--budget-multiplier", type=float, default=1.2)
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
    parser.add_argument("--oracle", action="store_true", help="Use the internal shortest-path solver instead of the API.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--list", action="store_true", help="List parsed episode specs and exit.")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    seed_start = int(args.seed if args.seed is not None else args.seed_start)

    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    specs = load_episode_specs(question_jsonl) if question_jsonl is not None else []
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    if question_jsonl is not None and not specs:
        raise ValueError(f"--question-jsonl contains no loadable rows: {question_jsonl}")
    if specs:
        difficulties = []
        seen = set()
        specs_by_difficulty: Dict[str, List[Dict[str, Any]]] = {}
        for spec in specs:
            difficulty = str(spec.get("difficulty", ""))
            specs_by_difficulty.setdefault(difficulty, []).append(spec)
            if difficulty not in seen:
                difficulties.append(difficulty)
                seen.add(difficulty)
    else:
        difficulties = parse_difficulty_list(args.difficulties)
        specs_by_difficulty = {}
    if args.budget_multiplier <= 0:
        raise ValueError("--budget-multiplier must be positive.")

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

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_by_grid_json = output_json.with_name(f"{output_json.stem}_by_grid.json")
    output_by_grid_csv = output_csv.with_name(f"{output_csv.stem}_by_grid.csv")

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

    env = OrderSwap2DPuzzleEnv(
        grid_rows=4,
        grid_cols=4,
        headless=args.headless,
        animate=args.animate,
        max_steps=None,
        budget_multiplier=args.budget_multiplier,
        host=args.host,
        port=args.port,
    )
    all_results: List[EpisodeResult] = []
    try:
        combo_progress = tqdm(difficulties, desc="Order Swap 2D Puzzle difficulties", unit="combo")
        next_seed = seed_start
        for difficulty in combo_progress:
            if specs:
                episode_specs = specs_by_difficulty.get(difficulty, [])
                episode_seeds = []
            else:
                episode_specs = None
                episode_seeds = list(range(next_seed, next_seed + args.repeats))
                next_seed += args.repeats
            combo_results = run_combo(
                env=env,
                policy=policy,
                image_writer=image_writer,
                difficulty=difficulty,
                episode_seeds=episode_seeds,
                budget_multiplier=args.budget_multiplier,
                oracle=args.oracle,
                episode_specs=episode_specs,
            )
            for local_index, result in enumerate(combo_results):
                result.episode_index = len(all_results) + local_index
            all_results.extend(combo_results)
            combo_summary = summarize_results(combo_results)
            combo_progress.set_postfix(
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
            "difficulties": difficulties,
            "repeats_per_difficulty": (
                {difficulty: len(specs_by_difficulty.get(difficulty, [])) for difficulty in difficulties}
                if specs
                else args.repeats
            ),
            "seed_start": seed_start,
            "budget_multiplier": args.budget_multiplier,
            "model": model,
            "base_url": base_url,
            "oracle": args.oracle,
            "max_tokens": args.max_tokens,
            "headless": args.headless,
            "animate": args.animate,
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
