from __future__ import annotations

import argparse
import base64
from collections import deque
import io
import json
import os
import random
import re
import shutil
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

GYM_DIR = Path(__file__).resolve().parent.parent / "gym"
if str(GYM_DIR) not in sys.path:
    sys.path.insert(0, str(GYM_DIR))

from env import KnotsUntangleEnv
from internvl3_config import load_local_config, resolve_config_value
from oracle_solver import (
    OracleSearchResult,
    apply_move as apply_oracle_move,
    bfs_shortest_plan,
    choose_greedy_move,
    enumerate_legal_moves as enumerate_oracle_legal_moves,
    state_crossings as oracle_state_crossings,
    symbolic_to_state,
)
from tqdm.auto import tqdm

try:
    import requests
except ImportError as exc:  # pragma: no cover - runtime dependency check.
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. "
        "Install it with: pip install requests"
    ) from exc

INVALID_ACTION = -999
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOPOBENCH_EVAL_ROOT = REPO_ROOT.parent / "topobench_eval"
if str(TOPOBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPOBENCH_EVAL_ROOT))

from topobench_eval.metadata_utils import load_task_intro_line

DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "output" / "benchmark_output"
DEFAULT_OUTPUT_JSON = DEFAULT_BENCHMARK_ROOT / "internvl3_knots_untangle.json"
DEFAULT_OUTPUT_CSV = DEFAULT_BENCHMARK_ROOT / "internvl3_knots_untangle.csv"
DEFAULT_STEP_IMAGE_ROOT = DEFAULT_BENCHMARK_ROOT / "images"
DEFAULT_HTTP_IMAGE_DIR = DEFAULT_BENCHMARK_ROOT / "internvl3_frames"
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0

from jsonl_export import flatten_answer_value, interactive_trajectory_state, resolve_existing_path, to_relative_path, write_jsonl

DIFFICULTIES = ("easy", "medium", "hard")
POLICIES = ("internvl", "random", "greedy", "oracle")
DIFFICULTY_GRID = {"easy": 5, "medium": 6, "hard": 6}
DIFFICULTY_ROPES = {"easy": 4, "medium": 5, "hard": 6}


DEFAULT_MAX_STEPS_BY_DIFFICULTY: Dict[str, int] = {
    "easy": 15,
    "medium": 15,
    "hard": 15,
}


def make_episode_id(difficulty: str, seed: int) -> str:
    return f"{PROJECT_ROOT.name}_difficulty_{difficulty}_seed_{int(seed):02d}"

# Frontend rope palette (hex int -> human-readable name).
COLOR_NAME_BY_HEX: Dict[int, str] = {
    0xE53935: "red",
    0x43A047: "green",
    0x1E88E5: "blue",
    0xFDD835: "yellow",
    0x8E24AA: "purple",
    0xFB8C00: "orange",
    0x00ACC1: "cyan",
    0xC0CA33: "lime",
}


def build_initial_prompt(
    *,
    include_legal_moves_in_prompt: bool,
    difficulty: str,
    grid_size: int,
    rope_count: int,
    crossings: Optional[int],
    symbolic: Optional[Dict[str, Any]],
    legal_moves: Sequence[Dict[str, int]],
    step_budget: int,
) -> str:
    """Build the full meta-format prompt for the first turn of a multi-turn episode."""
    crossings_text = str(crossings) if crossings is not None else "unknown"
    rope_descriptions = describe_ropes(symbolic)
    rope_lines = rope_descriptions if rope_descriptions else ["(rope list unavailable)"]
    occupied_text = describe_occupied_holes(symbolic)

    if include_legal_moves_in_prompt:
        move_rule = (
            "1. At each turn, choose exactly one move from the legal actions "
            "listed for the current state."
        )
        legal_actions_text = format_legal_moves(legal_moves)
    else:
        move_rule = (
            "1. At each turn, infer one legal move from the current image and "
            "task information. No legal-action list will be provided."
        )
        legal_actions_text = (
            "not provided; infer one legal move from the current state image and task rules"
        )

    lines = [
        "[Task]",
        load_task_intro_line(
            PROJECT_ROOT,
            default=(
                "You are solving Knots Untangle.\n"
                "In this task, you must untangle the ropes by moving one endpoint "
                "at a time so no two ropes cross."
            ),
        ),
        (
            "The pegboard is a square grid of indexed holes. Each rope connects "
            "two holes. A move relocates one endpoint from its current hole to "
            "an empty hole. Two ropes count as crossing when their top-down "
            "projected paths overlap, including cases where rope bodies touch "
            "because of rope thickness."
        ),
        "",
        "[Rules]",
        move_rule,
        "2. A legal action must move one rope endpoint from an occupied source hole to an empty target hole.",
        "3. After each action, the environment returns the next state. "
        "Illegal actions keep the board state unchanged but still count as a step.",
        "4. The task is solved when zero crossings remain.",
        f"5. The episode ends when the puzzle is solved or when the step budget ({step_budget}) is exhausted.",
        "6. At each turn, output exactly one next action for the current state.",
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer": {value}} and nothing else.',
        (
            "Replace {value} with the single legal answer for this task, chosen "
            'from a JSON object with integer fields "src_row", "src_col", '
            '"tgt_row", and "tgt_col" describing one legal move for the current '
            'turn, for example {"src_row":0,"src_col":1,"tgt_row":2,"tgt_col":2}.'
        ),
        "",
        "[Current Task]",
        "The current state image is shown below.",
        "[Image 1]",
        "Attached image.",
        "",
        "Current state summary:",
        f"- State index: 0",
        f"- Difficulty: {difficulty}",
        f"- Pegboard: {grid_size}x{grid_size} grid (rows/columns 0..{grid_size - 1})",
        "- Coordinate convention: holes are addressed as (row, col). Row numbers are shown along the top edge and increase from left to right; column numbers are shown along the left edge and increase from top to bottom. The top-left hole is (row=0, col=0).",
        f"- Rope count: {rope_count}",
        f"- Crossings remaining: {crossings_text}",
        f"- Goal: reduce crossings to 0",
        "",
        "Ropes on the board:",
    ]
    lines.extend(f"- {line}" for line in rope_lines)
    lines.extend([
        "",
        f"Occupied holes: {occupied_text}",
        "",
        f"Legal actions for this turn: {legal_actions_text}",
        'Output ONLY a single JSON object in this exact format and nothing else: {"answer":{"src_row":R,"src_col":C,"tgt_row":R,"tgt_col":C}}',
    ])
    return "\n".join(lines)


def build_followup_prompt(
    *,
    step_index: int,
    grid_size: int,
    rope_count: int,
    crossings: Optional[int],
    symbolic: Optional[Dict[str, Any]],
    legal_moves: Sequence[Dict[str, int]],
    include_legal_moves_in_prompt: bool,
) -> str:
    """Build the [Current Task] prompt for subsequent turns (step >= 1)."""
    crossings_text = str(crossings) if crossings is not None else "unknown"
    rope_descriptions = describe_ropes(symbolic)
    rope_lines = rope_descriptions if rope_descriptions else ["(rope list unavailable)"]
    occupied_text = describe_occupied_holes(symbolic)
    if include_legal_moves_in_prompt:
        legal_actions_text = format_legal_moves(legal_moves)
    else:
        legal_actions_text = (
            "not provided; infer one legal move from the updated state image and task rules"
        )

    lines = [
        "[Current Task]",
        "The updated current state image is shown below.",
        "[Image 1]",
        "Attached image.",
        "",
        "Current state summary:",
        f"- State index: {step_index}",
        f"- Pegboard: {grid_size}x{grid_size} grid (rows/columns 0..{grid_size - 1})",
        "- Coordinate convention: holes are addressed as (row, col). Row numbers are shown along the top edge and increase from left to right; column numbers are shown along the left edge and increase from top to bottom. The top-left hole is (row=0, col=0).",
        f"- Rope count: {rope_count}",
        f"- Crossings remaining: {crossings_text}",
        "",
        "Ropes on the board:",
    ]
    lines.extend(f"- {line}" for line in rope_lines)
    lines.extend([
        "",
        f"Occupied holes: {occupied_text}",
        "",
        f"Legal actions for this turn: {legal_actions_text}",
        'Output ONLY a single JSON object in this exact format and nothing else: {"answer":{"src_row":R,"src_col":C,"tgt_row":R,"tgt_col":C}}',
    ])
    return "\n".join(lines)


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    success: bool
    terminated: bool
    truncated: bool
    total_steps: int
    illegal_moves: int
    invalid_responses: int
    api_errors: int
    initial_crossings: int
    final_crossings: Optional[int]
    theoretical_min_steps: Optional[int]
    final_reward: float
    final_reason: Optional[str]
    metadata: Dict[str, Any]
    steps: List[Dict[str, Any]]


def build_initial_state_snapshot(symbolic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(symbolic, dict):
        return {}
    return {
        "gridSize": symbolic.get("gridSize"),
        "ropeCount": symbolic.get("ropeCount"),
        "crossings": symbolic.get("crossings"),
        "occupiedHoles": symbolic.get("occupiedHoles"),
        "ropes": symbolic.get("ropes"),
    }


@dataclass(frozen=True)
class BenchmarkSetup:
    difficulty: str
    seed: int
    repeat_index: int
    episode_id: Optional[str] = None
    max_steps: Optional[int] = None
    reset_config: Optional[Dict[str, Any]] = None


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


def load_benchmark_setups_from_question_jsonl(path: Path) -> List[BenchmarkSetup]:
    setups: List[BenchmarkSetup] = []
    for row_index, row in enumerate(read_question_jsonl(path)):
        meta = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        initial_state = meta.get("initial_state") if isinstance(meta.get("initial_state"), dict) else {}
        reset_config = initial_state.get("reset_config") if isinstance(initial_state.get("reset_config"), dict) else {}
        difficulty = str(reset_config.get("difficulty") or meta.get("difficulty") or meta.get("level") or "easy")
        seed = int(reset_config.get("seed", meta.get("seed", row_index)))
        repeat_index = int(meta.get("repeat_index", row_index))
        max_steps_value = meta.get("max_actions_per_traj")
        setups.append(
            BenchmarkSetup(
                difficulty=difficulty,
                seed=seed,
                repeat_index=repeat_index,
                episode_id=str(row.get("id")) if row.get("id") else None,
                max_steps=int(max_steps_value) if max_steps_value is not None else None,
                reset_config=dict(reset_config) if reset_config else None,
            )
        )
    return setups


def write_meta_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    summaries: Sequence[Dict[str, Any]],
    write_question_jsonl: bool = True,
) -> Tuple[Path, Path]:
    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    question_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []

    for summary in summaries:
        difficulty = str(summary["difficulty"])
        grid_size = int(summary["grid_size"])
        rope_count = int(summary["rope_count"])
        step_budget = int(summary["step_budget"])
        episodes: Sequence[EpisodeResult] = summary.get("episodes", [])
        for episode in episodes:
            first_step = episode.steps[0] if episode.steps else None
            initial_images: List[str] = []
            if first_step:
                initial_images = [
                    to_relative_path(
                        resolve_existing_path(
                            first_step["saved_image_path"],
                            output_dir,
                            PROJECT_ROOT,
                            REPO_ROOT,
                            DEFAULT_BENCHMARK_ROOT,
                        ),
                        base_dir=output_dir,
                    )
                ]
            episode_id = make_episode_id(difficulty, episode.seed)
            question_rows.append(
                {
                    "id": episode_id,
                    "category": ["knots", PROJECT_ROOT.name, "interactive"],
                    "type": "episode_initialization",
                    "meta_info": {
                        "task_name": PROJECT_ROOT.name,
                        "config": config_rel,
                        "level": difficulty,
                        "seed": int(episode.seed),
                        "repeat_index": int(episode.episode_index),
                        "difficulty": difficulty,
                        "max_actions_per_traj": step_budget,
                        "grid_size": grid_size,
                        "rope_count": rope_count,
                        "initial_state": build_initial_state_snapshot(
                            first_step.get("state_before_action", {}) if first_step else {}
                        ),
                        "legal_action_format": '{"answer":{"src_row":0,"src_col":1,"tgt_row":2,"tgt_col":2}}',
                    },
                    "images": initial_images,
                }
            )
            answer_rows.append(
                {
                    "id": episode_id,
                    "category": ["knots", PROJECT_ROOT.name, "interactive"],
                    "type": "episode_rollout",
                    "meta_info": {
                        "task_name": PROJECT_ROOT.name,
                        "config": config_rel,
                        "level": difficulty,
                        "seed": int(episode.seed),
                        "repeat_index": int(episode.episode_index),
                        "difficulty": difficulty,
                        "model_id": model_id,
                        "success": bool(episode.success),
                        "final_reason": episode.final_reason,
                        "total_steps": int(episode.total_steps),
                    },
                    "trajectory": [
                        {
                            "step_index": int(step["step_index"]),
                            "current_images": [
                                to_relative_path(
                                    resolve_existing_path(
                                        step["saved_image_path"],
                                        output_dir,
                                        PROJECT_ROOT,
                                        REPO_ROOT,
                                        DEFAULT_BENCHMARK_ROOT,
                                    ),
                                    base_dir=output_dir,
                                )
                            ],
                            "state": interactive_trajectory_state(
                                trajectory_index=trajectory_index,
                                total_steps=len(episode.steps),
                                success=bool(episode.success),
                            ),
                            "answer": flatten_answer_value(step.get("predicted_action")),
                            "invalid_response": bool(step.get("invalid_response", False)),
                            "api_error": bool(step.get("api_error_message")),
                            "illegal": bool(step.get("illegal", False)),
                            "raw_response_text": step.get("raw_response_text", ""),
                        }
                        for trajectory_index, step in enumerate(episode.steps)
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
    "difficulty",
    "grid_size",
    "rope_count",
    "initial_crossings",
    "theoretical_min_steps",
    "step_budget",
    "repeats",
    "successes",
    "success_rate",
    "avg_steps_all",
    "avg_steps_on_success",
    "avg_illegal_moves",
    "avg_invalid_responses",
    "avg_api_errors",
]


class _ImageHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


class ScreenshotURLServer:
    def __init__(self, *, bind_host: str, port: int, public_host: str, image_dir: Path) -> None:
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()
        self._recent_images: deque[Path] = deque()
        self._max_retained = 256

        handler = partial(_ImageHandler, directory=str(self.image_dir))
        self.server = ThreadingHTTPServer((bind_host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        bound_port = self.server.server_address[1]
        self.base_url = f"http://{public_host}:{bound_port}"

    def publish_image(self, image_bytes: bytes, *, extension: str) -> str:
        with self._lock:
            self._counter += 1
            safe_ext = extension.lstrip(".") or "png"
            filename = f"frame_{self._counter:08d}_{time.time_ns()}.{safe_ext}"
            final_path = self.image_dir / filename
            tmp_path = self.image_dir / f".{filename}.tmp"
            tmp_path.write_bytes(image_bytes)
            tmp_path.replace(final_path)

            self._recent_images.append(final_path)
            while len(self._recent_images) > self._max_retained:
                stale_path = self._recent_images.popleft()
                try:
                    stale_path.unlink()
                except FileNotFoundError:
                    pass

            return f"{self.base_url}/{filename}"

    def publish_png(self, png_bytes: bytes) -> str:
        return self.publish_image(png_bytes, extension="png")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)


class StepImageWriter:
    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = root_dir
        if self.root_dir.exists():
            if self.root_dir.is_dir():
                shutil.rmtree(self.root_dir)
            else:
                self.root_dir.unlink()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _episode_dir(self, *, difficulty: str, seed: int) -> Path:
        episode_dir = self.root_dir / make_episode_id(difficulty, seed)
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def save_png(
        self,
        *,
        difficulty: str,
        seed: int,
        step_index: int,
        png_bytes: bytes,
        suffix: str = "",
    ) -> Path:
        episode_dir = self._episode_dir(difficulty=difficulty, seed=seed)
        output_path = episode_dir / f"step_{step_index:04d}{suffix}.png"
        output_path.write_bytes(png_bytes)
        return output_path

    def save_prompt_debug(
        self,
        *,
        difficulty: str,
        seed: int,
        step_index: int,
        user_prompt: str,
        saved_image_path: Path,
        raw_response_text: str,
        parsed_answer: Optional[Any],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]] = None,
    ) -> Path:
        episode_dir = self._episode_dir(difficulty=difficulty, seed=seed)
        output_path = episode_dir / f"step_{step_index:04d}_prompt_debug.txt"
        parsed_text = (
            "<none>"
            if parsed_answer is None
            else (
                json.dumps(parsed_answer, ensure_ascii=False)
                if isinstance(parsed_answer, (list, dict))
                else str(parsed_answer)
            )
        )
        output_path.write_text(
            "\n\n".join(
                [
                    "[user_prompt]",
                    user_prompt,
                    "[image]",
                    str(saved_image_path),
                    "[model_response]",
                    raw_response_text or "<empty>",
                    "[parsed_answer]",
                    parsed_text,
                    "[api_error]",
                    api_error_message or "<none>",
                    "[response_debug]",
                    json.dumps(response_debug, indent=2, ensure_ascii=False)
                    if response_debug is not None
                    else "<none>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    def save_pure_prompt(
        self,
        *,
        difficulty: str,
        seed: int,
        step_index: int,
        user_prompt: str,
        saved_image_path: Path,
    ) -> Path:
        episode_dir = self._episode_dir(difficulty=difficulty, seed=seed)
        output_path = episode_dir / f"step_{step_index:04d}_pure_prompt.txt"
        prompt_with_path = re.sub(
            r"\[Image \d+\]\nAttached image\.",
            f"[Image]\n{saved_image_path}",
            user_prompt,
            count=1,
        )
        output_path.write_text(prompt_with_path + "\n", encoding="utf-8")
        return output_path

    def save_episode_summary(
        self,
        *,
        difficulty: str,
        seed: int,
        payload: Dict[str, Any],
    ) -> Path:
        episode_dir = self._episode_dir(difficulty=difficulty, seed=seed)
        output_path = episode_dir / "episode_summary.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def save_demo_gif(
        self,
        *,
        difficulty: str,
        seed: int,
        frame_paths: Sequence[Path],
        frame_duration_ms: int,
        final_hold_ms: int,
    ) -> Optional[Path]:
        usable_paths: List[Path] = []
        for path in frame_paths:
            if not isinstance(path, Path):
                path = Path(path)
            if not path.exists():
                continue
            if usable_paths and usable_paths[-1] == path:
                continue
            usable_paths.append(path)
        if not usable_paths:
            return None

        frames: List[Image.Image] = []
        durations: List[int] = []
        try:
            for index, path in enumerate(usable_paths):
                with Image.open(path) as image:
                    frames.append(image.convert("P", palette=Image.ADAPTIVE))
                durations.append(final_hold_ms if index == len(usable_paths) - 1 else frame_duration_ms)
            if not frames:
                return None
            output_path = self._episode_dir(difficulty=difficulty, seed=seed) / "demo.gif"
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
                disposal=2,
            )
            return output_path
        finally:
            for frame in frames:
                frame.close()

STATE_CAPTION_FONT_SIZE = 34
STATE_CAPTION_TOP_MARGIN = 14


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


def prepare_model_image_bytes(
    png_bytes: bytes,
    *,
    max_side: int,
    state_label: Optional[str] = None,
) -> Tuple[bytes, str, str]:
    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
        longest_side = max(width, height)
        rgb = image.convert("RGB")
        if max_side > 0 and longest_side > max_side:
            scale = max_side / float(longest_side)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        if state_label:
            draw = ImageDraw.Draw(rgb)
            font = load_state_caption_font(STATE_CAPTION_FONT_SIZE)
            bbox = draw.textbbox((0, 0), state_label, font=font)
            text_width = bbox[2] - bbox[0]
            x = max(0, (rgb.width - text_width) // 2)
            y = STATE_CAPTION_TOP_MARGIN
            draw.text(
                (x, y),
                state_label,
                fill="white",
                font=font,
            )
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=72, optimize=True)
        return buffer.getvalue(), "image/jpeg", "jpg"


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


def parse_difficulty_list(spec: str) -> List[str]:
    values: List[str] = []
    for chunk in spec.split(","):
        part = chunk.strip().lower()
        if not part:
            continue
        if part not in DIFFICULTIES:
            raise ValueError(
                f"Unknown difficulty {part!r}. Allowed: {', '.join(DIFFICULTIES)}"
            )
        if part not in values:
            values.append(part)
    if not values:
        raise ValueError(f"Empty difficulty list spec: {spec!r}")
    # Preserve canonical ordering (easy, medium, hard).
    ordered = [d for d in DIFFICULTIES if d in values]
    return ordered


def color_name_from_hex(value: Any) -> str:
    if isinstance(value, int):
        key = int(value) & 0xFFFFFF
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        elif text.lower().startswith("0x"):
            text = text[2:]
        try:
            key = int(text, 16) & 0xFFFFFF
        except ValueError:
            return "unknown"
    else:
        return "unknown"

    if key in COLOR_NAME_BY_HEX:
        return COLOR_NAME_BY_HEX[key]
    return f"#{key:06x}"


def _collect_text_fragments(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if value is None:
        return []

    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.extend(_collect_text_fragments(item))
        return parts

    if isinstance(value, dict):
        parts: List[str] = []
        preferred_keys = ("text", "content", "value", "reasoning_content", "output_text", "refusal")
        for key in preferred_keys:
            if key in value:
                parts.extend(_collect_text_fragments(value.get(key)))

        if parts:
            return parts

        block_type = value.get("type")
        if isinstance(block_type, str) and "text" in block_type.lower():
            for nested_key, nested_value in value.items():
                if nested_key == "type":
                    continue
                parts.extend(_collect_text_fragments(nested_value))
        return parts

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

    debug_info["finish_reason"] = first_choice.get("finish_reason")
    debug_info["message_content_raw"] = message.get("content")
    debug_info["message_reasoning_content_raw"] = message.get("reasoning_content")
    debug_info["message_refusal_raw"] = message.get("refusal")
    debug_info["choice_text_raw"] = first_choice.get("text")
    debug_info["choice_content_raw"] = first_choice.get("content")

    candidates = [
        ("message.content", message.get("content")),
        ("choice.text", first_choice.get("text")),
        ("choice.content", first_choice.get("content")),
    ]
    normalized_candidates: Dict[str, str] = {}
    for source_name, candidate in candidates:
        normalized_text = normalize_message_content(candidate)
        normalized_candidates[source_name] = normalized_text
        if normalized_text and debug_info["response_content_source"] is None:
            debug_info["response_content_source"] = source_name
            debug_info["normalized_candidates"] = normalized_candidates
            return normalized_text, debug_info, None

    reasoning_candidates = [
        ("message.reasoning_content", message.get("reasoning_content")),
        ("message.refusal", message.get("refusal")),
    ]
    for source_name, candidate in reasoning_candidates:
        normalized_candidates[source_name] = normalize_message_content(candidate)

    debug_info["normalized_candidates"] = normalized_candidates
    return "", debug_info, None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (and optional language tag).
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try pulling the first balanced JSON object/array out of a response that
    # contains extra text. Avoid free-form integer extraction: truncated model
    # reasoning often mentions rope endpoints before giving the final action.
    for start, opener in ((idx, ch) for idx, ch in enumerate(text) if ch in "{["):
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _validate_explicit_move(
    src_row: Optional[int],
    src_col: Optional[int],
    tgt_row: Optional[int],
    tgt_col: Optional[int],
    *,
    grid_size: int,
) -> Optional[Dict[str, int]]:
    if any(v is None for v in (src_row, src_col, tgt_row, tgt_col)):
        return None
    if any(v < 0 or v >= grid_size for v in (src_row, src_col, tgt_row, tgt_col)):
        return None
    if (src_row, src_col) == (tgt_row, tgt_col):
        return None
    return {
        "src_row": int(src_row),
        "src_col": int(src_col),
        "tgt_row": int(tgt_row),
        "tgt_col": int(tgt_col),
    }


def parse_action(
    text: str,
    *,
    grid_size: int,
) -> Optional[Dict[str, int]]:
    if not text:
        return None

    cleaned = _strip_markdown_fences(text)
    payload = _try_json_loads(cleaned)

    if isinstance(payload, dict) and "answer" in payload:
        answer_payload = payload.get("answer")
        if isinstance(answer_payload, (dict, list)):
            payload = answer_payload
        elif isinstance(answer_payload, str):
            nested = _try_json_loads(_strip_markdown_fences(answer_payload))
            if nested is not None:
                payload = nested

    if isinstance(payload, dict):
        explicit_key_sets = (
            ("src_row", "src_col", "tgt_row", "tgt_col"),
            ("source_row", "source_col", "target_row", "target_col"),
        )
        for keys in explicit_key_sets:
            parsed = _validate_explicit_move(
                _coerce_int(payload.get(keys[0])),
                _coerce_int(payload.get(keys[1])),
                _coerce_int(payload.get(keys[2])),
                _coerce_int(payload.get(keys[3])),
                grid_size=grid_size,
            )
            if parsed is not None:
                return parsed

    if isinstance(payload, list) and len(payload) >= 4:
        parsed = _validate_explicit_move(
            _coerce_int(payload[0]),
            _coerce_int(payload[1]),
            _coerce_int(payload[2]),
            _coerce_int(payload[3]),
            grid_size=grid_size,
        )
        if parsed is not None:
            return parsed

    return None


def move_to_frontend_action(move: Dict[str, int]) -> Dict[str, int]:
    return {
        "src_row": int(move["src_row"]),
        "src_col": int(move["src_col"]),
        "tgt_row": int(move["tgt_row"]),
        "tgt_col": int(move["tgt_col"]),
    }


def move_key(move: Dict[str, int]) -> Tuple[int, int, int, int]:
    return (
        int(move["src_row"]),
        int(move["src_col"]),
        int(move["tgt_row"]),
        int(move["tgt_col"]),
    )


def reverse_move_key(move: Dict[str, int]) -> Tuple[int, int, int, int]:
    src_row, src_col, tgt_row, tgt_col = move_key(move)
    return (tgt_row, tgt_col, src_row, src_col)


def _move_tiebreak_value(move: Dict[str, int], *, seed: int, step_index: int) -> int:
    src_row, src_col, tgt_row, tgt_col = move_key(move)
    value = (
        (seed + 1) * 73856093
        ^ (step_index + 1) * 19349663
        ^ (src_row + 1) * 83492791
        ^ (src_col + 1) * 2654435761
        ^ (tgt_row + 1) * 97531
        ^ (tgt_col + 1) * 421
    )
    return value & 0xFFFFFFFF


def format_legal_moves(legal_moves: Sequence[Dict[str, int]]) -> str:
    return json.dumps(list(legal_moves), separators=(",", ":"))


def compute_legal_moves(symbolic: Optional[Dict[str, Any]], grid_size: int) -> List[Dict[str, int]]:
    state = symbolic_to_state(symbolic, grid_size)
    if state is None:
        return []
    return enumerate_oracle_legal_moves(state, grid_size)


def order_legal_moves_for_prompt(
    *,
    symbolic: Optional[Dict[str, Any]],
    legal_moves: Sequence[Dict[str, int]],
    action_history: Sequence[Dict[str, int]],
    grid_size: int,
    seed: int,
    step_index: int,
) -> List[Dict[str, int]]:
    moves = [dict(move) for move in legal_moves]
    if not moves:
        return []

    current_state = symbolic_to_state(symbolic, grid_size)
    current_logical_crossings: Optional[int] = None
    if current_state is not None:
        current_logical_crossings = oracle_state_crossings(current_state, grid_size)

    recent_history = [dict(move) for move in action_history[-4:]]
    recent_move_keys = {move_key(move) for move in recent_history}
    recent_reverse_keys = {reverse_move_key(move) for move in recent_history}
    last_move = recent_history[-1] if recent_history else None
    last_move_reverse = reverse_move_key(last_move) if last_move is not None else None

    def sort_key(move: Dict[str, int]) -> Tuple[int, int, int, int, int, int]:
        key = move_key(move)
        src = key[:2]
        same_source_recent = sum(1 for item in recent_history if move_key(item)[:2] == src)
        exact_undo_last = int(last_move_reverse == key)
        seen_recently = int(key in recent_move_keys)
        reverse_seen_recently = int(key in recent_reverse_keys)

        predicted_after = current_logical_crossings
        if current_state is not None and current_logical_crossings is not None:
            next_state = apply_oracle_move(current_state, move, grid_size)
            if next_state is not None:
                predicted_after = oracle_state_crossings(next_state, grid_size)
        predicted_after = 999 if predicted_after is None else int(predicted_after)
        predicted_delta = 0
        if current_logical_crossings is not None and predicted_after != 999:
            predicted_delta = int(current_logical_crossings - predicted_after)

        if predicted_delta > 0:
            benefit_bucket = 0
        elif predicted_delta == 0:
            benefit_bucket = 1
        else:
            benefit_bucket = 2

        return (
            benefit_bucket,
            exact_undo_last,
            same_source_recent,
            seen_recently + reverse_seen_recently,
            predicted_after,
            _move_tiebreak_value(move, seed=seed, step_index=step_index),
        )

    return sorted(moves, key=sort_key)


def _extract_rope_entries(symbolic: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(symbolic, dict):
        return []
    ropes = symbolic.get("ropes")
    if not isinstance(ropes, list):
        return []
    return [rope for rope in ropes if isinstance(rope, dict)]


def _hole_rc(entry: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(entry, dict):
        return None
    row = _coerce_int(entry.get("row"))
    col = _coerce_int(entry.get("col"))
    if row is None or col is None:
        return None
    return row, col


def _occupied_holes_list(symbolic: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
    if not isinstance(symbolic, dict):
        return []
    raw = symbolic.get("occupiedHoles")
    occupied: List[Tuple[int, int]] = []
    if isinstance(raw, list):
        for item in raw:
            rc = _hole_rc(item)
            if rc is not None and rc not in occupied:
                occupied.append(rc)
    if occupied:
        return occupied
    # Fallback: derive from rope endpoints.
    for rope in _extract_rope_entries(symbolic):
        for key in ("startHole", "endHole"):
            rc = _hole_rc(rope.get(key))
            if rc is not None and rc not in occupied:
                occupied.append(rc)
    return occupied


def _crossings_count(symbolic: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(symbolic, dict):
        return None
    for key in ("visualCrossings", "crossings"):
        crossings = symbolic.get(key)
        if isinstance(crossings, int):
            return crossings
        if isinstance(crossings, list):
            return len(crossings)
        if isinstance(crossings, str) and crossings.strip().isdigit():
            return int(crossings.strip())
    return None


def puzzle_signature(symbolic: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(symbolic, dict):
        return None
    generator_info = symbolic.get("generatorInfo")
    if isinstance(generator_info, dict):
        signature = generator_info.get("signature")
        if isinstance(signature, str) and signature:
            return signature
    rope_entries = _extract_rope_entries(symbolic)
    if not rope_entries:
        return None
    normalized: List[str] = []
    for fallback_index, rope in enumerate(rope_entries):
        rope_id = _coerce_int(rope.get("id"))
        if rope_id is None:
            rope_id = fallback_index
        start_rc = _hole_rc(rope.get("startHole"))
        end_rc = _hole_rc(rope.get("endHole"))
        if start_rc is None or end_rc is None:
            return None
        endpoints = sorted((start_rc, end_rc))
        normalized.append(
            f"{rope_id}:{endpoints[0][0]},{endpoints[0][1]}-{endpoints[1][0]},{endpoints[1][1]}"
        )
    return "|".join(normalized)


def describe_ropes(symbolic: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    ropes = _extract_rope_entries(symbolic)
    for index, rope in enumerate(ropes):
        rope_id = _coerce_int(rope.get("id"))
        if rope_id is None:
            rope_id = index
        color_name = color_name_from_hex(rope.get("color"))
        start_rc = _hole_rc(rope.get("startHole"))
        end_rc = _hole_rc(rope.get("endHole"))
        if start_rc is None or end_rc is None:
            continue
        lines.append(
            f"Rope {rope_id} ({color_name}) endpoints at "
            f"(row={start_rc[0]},col={start_rc[1]}) and "
            f"(row={end_rc[0]},col={end_rc[1]})"
        )
    return lines


def describe_occupied_holes(symbolic: Optional[Dict[str, Any]]) -> str:
    occupied = _occupied_holes_list(symbolic)
    if not occupied:
        return "(none)"
    return ", ".join(f"(row={r},col={c})" for r, c in occupied)


def format_recent_actions(recent: Sequence[Dict[str, int]]) -> str:
    return json.dumps(list(recent), separators=(",", ":"))


def simplify_state_for_prompt(symbolic: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(symbolic, dict):
        return None
    rope_entries: List[Dict[str, Any]] = []
    for index, rope in enumerate(_extract_rope_entries(symbolic)):
        rope_id = _coerce_int(rope.get("id"))
        if rope_id is None:
            rope_id = index
        color_name = color_name_from_hex(rope.get("color"))
        start_rc = _hole_rc(rope.get("startHole"))
        end_rc = _hole_rc(rope.get("endHole"))
        rope_entries.append(
            {
                "rope_id": rope_id,
                "color": color_name,
                "start": list(start_rc) if start_rc else None,
                "end": list(end_rc) if end_rc else None,
            }
        )
    return {
        "ropes": rope_entries,
        "occupied": [list(rc) for rc in _occupied_holes_list(symbolic)],
        "crossings": _crossings_count(symbolic),
    }


def format_prompt_trajectory(trajectory: Sequence[Dict[str, Any]]) -> str:
    serialized_entries: List[Dict[str, Any]] = []
    for entry in trajectory:
        if not isinstance(entry, dict):
            continue
        state_before_action = simplify_state_for_prompt(entry.get("symbolic_before_action"))
        action = entry.get("action")
        if state_before_action is None:
            continue
        serialized_entries.append(
            {
                "state": state_before_action,
                "action": action,
            }
        )
    return json.dumps(serialized_entries, separators=(",", ":"))


def build_user_prompt(
    *,
    difficulty: str,
    grid_size: int,
    rope_count: int,
    crossings: Optional[int],
    symbolic: Optional[Dict[str, Any]],
    legal_moves: Sequence[Dict[str, int]],
    recent_actions: Sequence[Dict[str, int]],
    include_legal_moves_in_prompt: bool,
    prompt_trajectory: Optional[Sequence[Dict[str, Any]]] = None,
    step_index: int = 0,
    step_budget: int = 32,
) -> str:
    """Build user prompt. Delegates to build_initial_prompt / build_followup_prompt."""
    if step_index == 0:
        return build_initial_prompt(
            include_legal_moves_in_prompt=include_legal_moves_in_prompt,
            difficulty=difficulty,
            grid_size=grid_size,
            rope_count=rope_count,
            crossings=crossings,
            symbolic=symbolic,
            legal_moves=legal_moves,
            step_budget=step_budget,
        )
    return build_followup_prompt(
        step_index=step_index,
        grid_size=grid_size,
        rope_count=rope_count,
        crossings=crossings,
        symbolic=symbolic,
        legal_moves=legal_moves,
        include_legal_moves_in_prompt=include_legal_moves_in_prompt,
    )


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
        self._request_gate = threading.Lock()
        self._next_request_not_before = 0.0
        self._min_request_interval = 0.0
        if self.requests_per_minute > 0:
            self._min_request_interval = 60.0 / self.requests_per_minute
        self._conversation_history: List[Dict[str, Any]] = []

    def _wait_for_rate_limit_slot(self) -> None:
        if self._min_request_interval <= 0:
            return

        with self._request_gate:
            now = time.monotonic()
            sleep_seconds = max(0.0, self._next_request_not_before - now)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            self._next_request_not_before = time.monotonic() + self._min_request_interval

    def _build_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_retry_after_seconds(self, response: requests.Response) -> float:
        raw_value = response.headers.get("Retry-After")
        if raw_value is None:
            return 0.0
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 0.0

    def _format_error_response(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("detail")
                if isinstance(message, str) and message.strip():
                    return f"HTTP {response.status_code}: {message.strip()}"
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return f"HTTP {response.status_code}: {message.strip()}"

        text = response.text.strip()
        if text:
            return f"HTTP {response.status_code}: {text}"
        return f"HTTP {response.status_code}"

    def _safe_response_payload(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    def reset_conversation(self) -> None:
        """Clear accumulated conversation history for a new episode."""
        self._conversation_history: List[Dict[str, Any]] = []

    def choose_action(
        self,
        *,
        image_url: str,
        user_prompt: str,
        grid_size: int,
    ) -> Tuple[Optional[Dict[str, int]], str, Optional[str], Dict[str, Any]]:
        # Build the new user message with text + image
        new_user_message: Dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
        # Multi-turn: accumulate conversation history
        self._conversation_history.append(new_user_message)

        last_error: Optional[str] = None
        endpoint = f"{self.base_url}/chat/completions"
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": list(self._conversation_history),
        }
        for attempt in range(self.api_retries + 1):
            try:
                self._wait_for_rate_limit_slot()
                response = self._session.post(
                    endpoint,
                    headers=self._build_request_headers(),
                    json=request_payload,
                    timeout=self.request_timeout_seconds,
                )

                if response.status_code == 429:
                    last_error = self._format_error_response(response)
                    if attempt < self.api_retries:
                        time.sleep(
                            max(
                                self.retry_sleep_seconds,
                                self._min_request_interval,
                                self._parse_retry_after_seconds(response),
                            )
                        )
                        continue
                    return None, "", last_error, {"response_payload": self._safe_response_payload(response)}

                if response.status_code >= 500:
                    last_error = self._format_error_response(response)
                    if attempt < self.api_retries:
                        time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                        continue
                    return None, "", last_error, {"response_payload": self._safe_response_payload(response)}

                response.raise_for_status()
                payload = response.json()
                raw_text, response_debug, extraction_error = extract_response_text(payload)
                if extraction_error is not None:
                    last_error = extraction_error
                    return None, "", last_error, response_debug
                # Append assistant response to conversation history for multi-turn
                self._conversation_history.append(
                    {"role": "assistant", "content": raw_text}
                )
                parsed = parse_action(
                    raw_text,
                    grid_size=grid_size,
                )
                return parsed, raw_text, None, response_debug
            except requests.RequestException as exc:  # pragma: no cover - network / remote server path.
                response = getattr(exc, "response", None)
                if response is not None:
                    last_error = self._format_error_response(response)
                    if attempt < self.api_retries and response.status_code >= 500:
                        time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                        continue
                    return None, "", last_error, {"response_payload": self._safe_response_payload(response)}

                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.api_retries:
                    time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.api_retries:
                    time.sleep(self.retry_sleep_seconds)

        return None, "", last_error, {}


def choose_local_action(
    *,
    policy_name: str,
    legal_moves: Sequence[Dict[str, int]],
    symbolic: Optional[Dict[str, Any]],
    grid_size: int,
    rng: random.Random,
    oracle_cache: Dict[Tuple[Tuple[int, int], ...], OracleSearchResult],
    oracle_timeout_seconds: float,
    oracle_max_expansions: int,
    step_budget: int,
) -> Tuple[Optional[Dict[str, int]], str, Optional[str], Dict[str, Any]]:
    if not legal_moves:
        return None, "", "No legal moves available.", {}

    if policy_name == "random":
        move = dict(rng.choice(list(legal_moves)))
        return move, json.dumps({"answer": move}, separators=(",", ":")), None, {"policy": "random"}

    state = symbolic_to_state(symbolic, grid_size)
    if state is None:
        return None, "", "Failed to derive symbolic state for local policy.", {}

    if policy_name == "greedy":
        move = choose_greedy_move(state, grid_size)
        if move is None:
            return None, "", "Greedy policy found no legal move.", {}
        return dict(move), json.dumps({"answer": move}, separators=(",", ":")), None, {"policy": "greedy"}

    if policy_name == "oracle":
        search = oracle_cache.get(state)
        if search is None:
            search = bfs_shortest_plan(
                state,
                grid_size=grid_size,
                max_depth=step_budget,
                max_expansions=oracle_max_expansions,
                timeout_seconds=oracle_timeout_seconds,
            )
            oracle_cache[state] = search
        debug = {
            "policy": "oracle",
            "oracle_min_steps": search.min_steps,
            "oracle_expanded_nodes": search.expanded_nodes,
            "oracle_timed_out": search.timed_out,
            "oracle_exceeded_limit": search.exceeded_limit,
        }
        if search.plan:
            move = dict(search.plan[0])
            return move, json.dumps({"answer": move}, separators=(",", ":")), None, debug
        fallback = choose_greedy_move(state, grid_size)
        if fallback is None:
            return None, "", "Oracle search found no plan and greedy fallback failed.", debug
        debug["fallback_policy"] = "greedy"
        move = dict(fallback)
        return move, json.dumps({"answer": move}, separators=(",", ":")), "Oracle search returned no plan.", debug

    raise ValueError(f"Unsupported local policy: {policy_name}")


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def format_optional_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def serialize_episode(episode: EpisodeResult, *, include_step_debug: bool) -> Dict[str, Any]:
    payload = {
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "success": episode.success,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "total_steps": episode.total_steps,
        "illegal_moves": episode.illegal_moves,
        "invalid_responses": episode.invalid_responses,
        "api_errors": episode.api_errors,
        "initial_crossings": episode.initial_crossings,
        "final_crossings": episode.final_crossings,
        "theoretical_min_steps": episode.theoretical_min_steps,
        "final_reward": episode.final_reward,
        "final_reason": episode.final_reason,
        "metadata": episode.metadata,
    }
    if include_step_debug:
        payload["steps"] = episode.steps
    return payload


def serialize_summary(
    summary: Dict[str, Any],
    *,
    include_episodes: bool,
    include_step_debug: bool,
) -> Dict[str, Any]:
    payload = {key: summary.get(key) for key in SUMMARY_FIELD_ORDER}
    if include_episodes:
        episodes = summary.get("episodes", [])
        payload["episodes"] = [
            serialize_episode(episode, include_step_debug=include_step_debug)
            for episode in episodes
        ]
    return payload


def detect_public_host(base_url: str, explicit_public_host: Optional[str]) -> str:
    if explicit_public_host:
        return explicit_public_host

    parsed = urlparse(base_url)
    target_host = parsed.hostname
    target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not target_host or target_host in {"127.0.0.1", "localhost", "0.0.0.0"}:
        return "127.0.0.1"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((target_host, target_port))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def is_official_intern_api_base_url(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname in {"chat.intern-ai.org.cn", "internlm.intern-ai.org.cn"}


def run_single_difficulty(
    *,
    policy_name: str,
    policy: Optional[InternVL3Policy],
    difficulty: str,
    repeats: int,
    max_steps: int,
    headless: bool,
    animate: bool,
    progress_every: int,
    model_image_max_side: int,
    image_transport: str,
    screenshot_server: Optional[ScreenshotURLServer],
    step_image_writer: StepImageWriter,
    collect_step_debug: bool,
    include_legal_moves_in_prompt: bool,
    include_prompt_trajectory: bool,
    save_demo_gifs: bool,
    demo_frame_duration_ms: int,
    demo_final_hold_ms: int,
    oracle_timeout_seconds: float,
    oracle_max_expansions: int,
    setups: Optional[Sequence[BenchmarkSetup]] = None,
) -> Dict[str, Any]:
    grid_size = DIFFICULTY_GRID[difficulty]
    rope_count = DIFFICULTY_ROPES[difficulty]
    step_budget = int(max_steps)
    iter_setups: Sequence[BenchmarkSetup] = (
        list(setups)
        if setups is not None
        else [BenchmarkSetup(difficulty=difficulty, seed=i, repeat_index=i) for i in range(repeats)]
    )

    env = KnotsUntangleEnv(
        difficulty=difficulty,
        headless=headless,
        animate=animate,
        max_steps=step_budget,
    )

    episodes: List[EpisodeResult] = []
    seen_puzzle_signatures: set[str] = set()
    oracle_cache: Dict[Tuple[Tuple[int, int], ...], OracleSearchResult] = {}
    try:
        desc = f"difficulty={difficulty}"
        with tqdm(total=len(iter_setups), desc=desc, unit="episode", leave=False) as progress_bar:
            for loop_index, setup in enumerate(iter_setups):
                # Reset multi-turn conversation history for each new episode
                if policy is not None and hasattr(policy, "reset_conversation"):
                    policy.reset_conversation()
                seed = setup.seed
                episode_index = setup.repeat_index
                generation_seed = seed
                duplicate_resamples = 0
                current_symbolic: Optional[Dict[str, Any]] = None
                reset_info: Dict[str, Any] = {}
                signature: Optional[str] = None
                while True:
                    _, reset_info = env.reset(seed=generation_seed)
                    current_symbolic = reset_info.get("symbolic_observation")
                    signature = puzzle_signature(current_symbolic)
                    if signature is None or signature not in seen_puzzle_signatures:
                        if signature is not None:
                            seen_puzzle_signatures.add(signature)
                        break
                    duplicate_resamples += 1
                    if duplicate_resamples >= 64:
                        break
                    generation_seed = seed + duplicate_resamples * 1000

                episode_rng = random.Random(generation_seed)
                initial_crossings = _crossings_count(current_symbolic) or 0
                initial_state = symbolic_to_state(current_symbolic, grid_size)
                initial_oracle: Optional[OracleSearchResult] = None
                if initial_state is not None:
                    initial_oracle = oracle_cache.get(initial_state)
                    if initial_oracle is None:
                        initial_oracle = bfs_shortest_plan(
                            initial_state,
                            grid_size=grid_size,
                            max_depth=step_budget,
                            max_expansions=oracle_max_expansions,
                            timeout_seconds=oracle_timeout_seconds,
                        )
                        oracle_cache[initial_state] = initial_oracle
                theoretical_min_steps = (
                    None if initial_oracle is None else initial_oracle.min_steps
                )
                episode_metadata: Dict[str, Any] = {
                    "policy": policy_name,
                    "requested_seed": seed,
                    "generation_seed": generation_seed,
                    "duplicate_resamples": duplicate_resamples,
                    "puzzle_signature": signature,
                    "crossing_metric": "visual_rope_overlap",
                    "theoretical_min_steps": theoretical_min_steps,
                    "theoretical_min_steps_metric": "straight_endpoint_visual_overlap",
                }
                if isinstance(current_symbolic, dict):
                    generator_info = current_symbolic.get("generatorInfo")
                    if isinstance(generator_info, dict):
                        template_name = generator_info.get("templateName")
                        if isinstance(template_name, str) and template_name:
                            episode_metadata["template_name"] = template_name
                if initial_oracle is not None:
                    episode_metadata.update(
                        {
                            "oracle_expanded_nodes": initial_oracle.expanded_nodes,
                            "oracle_timed_out": initial_oracle.timed_out,
                            "oracle_exceeded_limit": initial_oracle.exceeded_limit,
                        }
                    )

                executed_actions: List[Dict[str, int]] = []
                total_steps = 0
                illegal_moves = 0
                invalid_responses = 0
                api_errors = 0
                final_reward = 0.0
                final_reason: Optional[str] = None
                final_crossings: Optional[int] = initial_crossings
                step_records: List[Dict[str, Any]] = []
                prompt_trajectory: List[Dict[str, Any]] = []
                demo_frame_paths: List[Path] = []
                final_image_path: Optional[Path] = None
                terminated = False
                truncated = False

                for step_index in range(step_budget):
                    crossings = _crossings_count(current_symbolic)
                    recent_actions_before = executed_actions[-2:]
                    legal_moves = order_legal_moves_for_prompt(
                        symbolic=current_symbolic,
                        legal_moves=compute_legal_moves(current_symbolic, grid_size),
                        action_history=executed_actions,
                        grid_size=grid_size,
                        seed=generation_seed,
                        step_index=step_index,
                    )
                    raw_screenshot_png = env.screenshot(scene_only=True)
                    screenshot_png = add_state_caption(raw_screenshot_png, f"State {step_index}")
                    saved_image_path = step_image_writer.save_png(
                        difficulty=difficulty,
                        seed=seed,
                        step_index=step_index,
                        png_bytes=screenshot_png,
                    )
                    demo_frame_paths.append(saved_image_path)
                    final_image_path = saved_image_path
                    model_image_bytes, model_image_mime, model_image_extension = prepare_model_image_bytes(
                        screenshot_png,
                        max_side=model_image_max_side,
                    )
                    if image_transport == "data_url":
                        image_url = (
                            f"data:{model_image_mime};base64,"
                            f"{base64.b64encode(model_image_bytes).decode('ascii')}"
                        )
                    else:
                        if screenshot_server is None:
                            raise RuntimeError("HTTP image transport requires a screenshot server.")
                        image_url = screenshot_server.publish_image(
                            model_image_bytes,
                            extension=model_image_extension,
                        )
                    user_prompt = build_user_prompt(
                        difficulty=difficulty,
                        grid_size=grid_size,
                        rope_count=rope_count,
                        crossings=crossings,
                        symbolic=current_symbolic,
                        legal_moves=legal_moves,
                        recent_actions=recent_actions_before,
                        include_legal_moves_in_prompt=include_legal_moves_in_prompt,
                        step_index=step_index,
                        step_budget=step_budget,
                    )
                    step_image_writer.save_pure_prompt(
                        difficulty=difficulty,
                        seed=seed,
                        step_index=step_index,
                        user_prompt=user_prompt,
                        saved_image_path=saved_image_path,
                    )

                    if policy_name == "internvl":
                        if policy is None:
                            raise RuntimeError("InternVL policy requested but not initialized.")
                        parsed_action, _raw_response, api_error, response_debug = policy.choose_action(
                            image_url=image_url,
                            user_prompt=user_prompt,
                            grid_size=grid_size,
                        )
                    else:
                        parsed_action, _raw_response, api_error, response_debug = choose_local_action(
                            policy_name=policy_name,
                            legal_moves=legal_moves,
                            symbolic=current_symbolic,
                            grid_size=grid_size,
                            rng=episode_rng,
                            oracle_cache=oracle_cache,
                            oracle_timeout_seconds=oracle_timeout_seconds,
                            oracle_max_expansions=oracle_max_expansions,
                            step_budget=step_budget,
                        )

                    saved_prompt_debug_path = step_image_writer.save_prompt_debug(
                        difficulty=difficulty,
                        seed=seed,
                        step_index=step_index,
                        user_prompt=user_prompt,
                        saved_image_path=saved_image_path,
                        raw_response_text=_raw_response,
                        parsed_answer=parsed_action,
                        api_error_message=api_error,
                        response_debug=response_debug,
                    )
                    invalid_response = parsed_action is None
                    if invalid_response:
                        invalid_responses += 1
                        frontend_action: Any = INVALID_ACTION
                    else:
                        frontend_action = move_to_frontend_action(parsed_action)
                    if api_error and policy_name == "internvl":
                        api_errors += 1

                    _, reward, terminated, truncated, info = env.step(frontend_action)
                    total_steps += 1
                    final_reward = reward
                    final_reason = info.get("reason") if isinstance(info, dict) else None
                    illegal_flag = bool(info.get("illegal", False)) if isinstance(info, dict) else False
                    illegal_moves += int(illegal_flag)

                    next_symbolic: Optional[Dict[str, Any]] = None
                    if isinstance(info, dict):
                        next_symbolic = info.get("symbolic_observation")
                    if next_symbolic is None:
                        next_symbolic = current_symbolic
                    final_crossings = _crossings_count(next_symbolic)

                    step_record: Dict[str, Any] = {
                        "step_index": step_index,
                        "saved_image_path": str(saved_image_path),
                        "saved_prompt_debug_path": str(saved_prompt_debug_path),
                        "state_before_action": current_symbolic,
                        "crossings_before_action": crossings,
                        "legal_moves": legal_moves,
                        "predicted_action": None if parsed_action is None else dict(parsed_action),
                        "raw_response_text": _raw_response,
                        "invalid_response": invalid_response,
                        "api_error_message": api_error,
                        "illegal": illegal_flag,
                        "reason": final_reason,
                        "state_after_action": next_symbolic,
                        "crossings_after_action": final_crossings,
                    }
                    if collect_step_debug:
                        step_record.update(
                            {
                                "image_url": image_url,
                                "model_image_bytes": len(model_image_bytes),
                                "model_image_mime": model_image_mime,
                                "model_image_extension": model_image_extension,
                                "recent_actions_before_action": list(recent_actions_before),
                                "frontend_action": frontend_action,
                                "policy_debug": response_debug,
                                "response_content_source": response_debug.get("response_content_source"),
                                "finish_reason": response_debug.get("finish_reason"),
                                "normalized_candidates": response_debug.get("normalized_candidates"),
                                "message_content_raw": response_debug.get("message_content_raw"),
                                "message_reasoning_content_raw": response_debug.get("message_reasoning_content_raw"),
                                "message_refusal_raw": response_debug.get("message_refusal_raw"),
                                "choice_text_raw": response_debug.get("choice_text_raw"),
                                "choice_content_raw": response_debug.get("choice_content_raw"),
                                "response_payload": response_debug.get("response_payload"),
                                "reward": reward,
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": info if isinstance(info, dict) else None,
                            }
                        )
                    step_records.append(step_record)
                    prompt_trajectory.append(
                        {
                            "step_index": step_index,
                            "symbolic_before_action": current_symbolic,
                            "action": None if parsed_action is None else dict(parsed_action),
                        }
                    )
                    if parsed_action is not None and not illegal_flag:
                        executed_actions.append(dict(parsed_action))
                    current_symbolic = next_symbolic

                    if terminated or truncated:
                        break

                if total_steps > 0:
                    terminal_state_index = step_index + 1
                    raw_terminal_png = env.screenshot(scene_only=True)
                    terminal_png = add_state_caption(raw_terminal_png, f"State {terminal_state_index}")
                    saved_terminal_image_path = step_image_writer.save_png(
                        difficulty=difficulty,
                        seed=seed,
                        step_index=terminal_state_index,
                        png_bytes=terminal_png,
                    )
                    demo_frame_paths.append(saved_terminal_image_path)
                    final_image_path = saved_terminal_image_path

                success_flag = bool(terminated)
                if isinstance(reset_info, dict):
                    # nothing to reuse, kept for symmetry with hanoi reference
                    pass
                # success is reflected in the final info[success] too; prefer terminated for consistency.
                if final_crossings == 0:
                    success_flag = True
                if final_image_path is not None:
                    episode_metadata["final_image_path"] = str(final_image_path)
                episode_metadata["prompt_includes_legal_moves"] = include_legal_moves_in_prompt
                if save_demo_gifs:
                    demo_gif_path = step_image_writer.save_demo_gif(
                        difficulty=difficulty,
                        seed=seed,
                        frame_paths=demo_frame_paths,
                        frame_duration_ms=demo_frame_duration_ms,
                        final_hold_ms=demo_final_hold_ms,
                    )
                    if demo_gif_path is not None:
                        episode_metadata["demo_gif_path"] = str(demo_gif_path)

                episode = EpisodeResult(
                    episode_index=episode_index,
                    seed=seed,
                    success=success_flag,
                    terminated=terminated,
                    truncated=truncated,
                    total_steps=total_steps,
                    illegal_moves=illegal_moves,
                    invalid_responses=invalid_responses,
                    api_errors=api_errors,
                    initial_crossings=initial_crossings,
                    final_crossings=final_crossings,
                    theoretical_min_steps=theoretical_min_steps,
                    final_reward=final_reward,
                    final_reason=final_reason,
                    metadata=episode_metadata,
                    steps=step_records,
                )
                step_image_writer.save_episode_summary(
                    difficulty=difficulty,
                    seed=seed,
                    payload=serialize_episode(episode, include_step_debug=collect_step_debug),
                )
                episodes.append(episode)
                progress_bar.update(1)

                if progress_every > 0 and (
                    loop_index == 0
                    or (loop_index + 1) % progress_every == 0
                    or loop_index + 1 == len(iter_setups)
                ):
                    success_count = sum(int(item.success) for item in episodes)
                    progress_bar.set_postfix(
                        success_rate=f"{success_count / len(episodes):.3f}",
                        illegal_avg=format_optional_float(
                            mean_or_none([item.illegal_moves for item in episodes])
                        ),
                        invalid_avg=format_optional_float(
                            mean_or_none([item.invalid_responses for item in episodes])
                        ),
                        api_error_avg=format_optional_float(
                            mean_or_none([item.api_errors for item in episodes])
                        ),
                    )
    finally:
        env.close()

    successes = [item for item in episodes if item.success]
    success_count = len(successes)
    avg_initial_crossings = mean_or_none([item.initial_crossings for item in episodes])
    avg_theoretical_min_steps = mean_or_none(
        [item.theoretical_min_steps for item in episodes if item.theoretical_min_steps is not None]
    )
    summary = {
        "difficulty": difficulty,
        "grid_size": grid_size,
        "rope_count": rope_count,
        "initial_crossings": avg_initial_crossings,
        "theoretical_min_steps": avg_theoretical_min_steps,
        "step_budget": step_budget,
        "repeats": len(iter_setups),
        "successes": success_count,
        "success_rate": success_count / len(iter_setups) if iter_setups else 0.0,
        "avg_steps_all": mean_or_none([item.total_steps for item in episodes]),
        "avg_steps_on_success": mean_or_none([item.total_steps for item in successes]),
        "avg_illegal_moves": mean_or_none([item.illegal_moves for item in episodes]),
        "avg_invalid_responses": mean_or_none([item.invalid_responses for item in episodes]),
        "avg_api_errors": mean_or_none([item.api_errors for item in episodes]),
        "episodes": episodes,
    }
    return summary


def maybe_write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_write_csv(path: Optional[str], summaries: Sequence[Dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = SUMMARY_FIELD_ORDER
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for item in summaries:
            row = [str(item.get(key, "")) for key in header]
            handle.write(",".join(row) + "\n")


def parse_max_steps_override(spec: Optional[str]) -> Dict[str, int]:
    if not spec:
        return {}
    overrides: Dict[str, int] = {}
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"--max-steps entries must be key=value, got {part!r}"
            )
        key, value = part.split("=", 1)
        key_lower = key.strip().lower()
        value_text = value.strip()
        if key_lower not in DIFFICULTIES:
            raise ValueError(
                f"Unknown difficulty {key_lower!r} in --max-steps override."
            )
        try:
            overrides[key_lower] = int(value_text)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for --max-steps {part!r}") from exc
    return overrides


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark KnotsUntangle difficulties with an InternVL3 policy."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional local JSON config path. Defaults to auto-loading backend/internvl3_local_config.json if present.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--policy",
        default="internvl",
        choices=POLICIES,
        help="Action policy to run: remote InternVL or local random/greedy/oracle baselines. The oracle uses logical endpoint BFS, not full rope physics.",
    )
    parser.add_argument(
        "--difficulties",
        default="easy,medium,hard",
        help="Comma-separated difficulty list (easy,medium,hard).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of seeds per difficulty. Seed range = 0..repeats-1.",
    )
    parser.add_argument(
        "--question-jsonl",
        default="",
        help="Load existing interactive question.jsonl rows (difficulty + seed) instead of generating fresh.",
    )
    parser.add_argument(
        "--max-steps",
        default=None,
        help=(
            "Optional override for the per-episode step budget. "
            "Either a single integer applied to every difficulty, or a "
            "comma-separated difficulty=value list like 'easy=6,hard=32'."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Maximum completion tokens. Official API docs state 0 enables adaptive maximum length.",
    )
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=float(os.getenv("INTERN_API_REQUESTS_PER_MINUTE", str(DEFAULT_REQUESTS_PER_MINUTE))),
        help="Client-side throttle for chat/completions calls. Official API docs currently allow 30 req/min.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=float(os.getenv("INTERN_API_TIMEOUT_SECONDS", "120")),
        help="HTTP timeout for each API request.",
    )
    parser.add_argument("--headed", action="store_true", help="Show the Playwright browser window.")
    parser.add_argument(
        "--headless",
        dest="headed",
        action="store_false",
        help="Run the Playwright browser in headless mode (default).",
    )
    parser.set_defaults(headed=False)
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--model-image-max-side",
        type=int,
        default=768,
        help="Resize the screenshot sent to the model so its longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--image-transport",
        choices=("http_url", "data_url"),
        default="data_url",
        help="How to send screenshots to the model. Official Intern API should use data_url.",
    )
    parser.add_argument(
        "--image-server-bind-host",
        default="0.0.0.0",
        help="Bind host for the temporary screenshot HTTP server.",
    )
    parser.add_argument(
        "--image-public-host",
        default=None,
        help="Public host or IP that the remote model server can reach for screenshots.",
    )
    parser.add_argument(
        "--image-server-port",
        type=int,
        default=0,
        help="Port for the temporary screenshot HTTP server. Default picks a free port.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_HTTP_IMAGE_DIR),
        help="Directory used by the temporary screenshot HTTP server.",
    )
    parser.add_argument(
        "--step-image-root",
        default=str(DEFAULT_STEP_IMAGE_ROOT),
        help="Directory where every per-step model screenshot is saved. Cleared at the start of each run.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for JSON results. Default JSON is summary-only.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional path for summary CSV results.")
    parser.add_argument(
        "--include-episodes",
        action="store_true",
        help="Include per-episode aggregates in the JSON output.",
    )
    parser.add_argument(
        "--include-step-debug",
        action="store_true",
        help="Include full step-by-step debug records in the JSON output. Implies --include-episodes.",
    )
    parser.add_argument(
        "--include-prompt-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append past states and parsed actions to the user prompt at each step.",
    )
    parser.add_argument(
        "--include-legal-moves-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to include the full legal-action list in the user prompt. "
            "Keep this off for a harder, more image-grounded benchmark."
        ),
    )
    parser.add_argument(
        "--save-demo-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a per-episode demo.gif alongside the per-step PNGs.",
    )
    parser.add_argument(
        "--demo-frame-duration-ms",
        type=int,
        default=300,
        help="Frame duration for intermediate demo GIF frames.",
    )
    parser.add_argument(
        "--demo-final-hold-ms",
        type=int,
        default=900,
        help="Final-frame hold duration for demo GIFs.",
    )
    parser.add_argument(
        "--oracle-timeout-seconds",
        type=float,
        default=10.0,
        help="Per-state BFS oracle timeout used for metadata and the local oracle policy.",
    )
    parser.add_argument(
        "--oracle-max-expansions",
        type=int,
        default=200000,
        help="Maximum BFS node expansions per oracle search before aborting.",
    )
    return parser


def resolve_max_steps(
    difficulties: Sequence[str],
    spec: Optional[str],
) -> Dict[str, int]:
    result: Dict[str, int] = {d: DEFAULT_MAX_STEPS_BY_DIFFICULTY[d] for d in difficulties}
    if not spec:
        return result

    stripped = spec.strip()
    if stripped and "=" not in stripped:
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for --max-steps {stripped!r}") from exc
        for d in difficulties:
            result[d] = value
        return result

    overrides = parse_max_steps_override(spec)
    for key, value in overrides.items():
        if key in result:
            result[key] = value
    return result


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _, local_config = load_local_config(args.config)

    base_url = resolve_config_value(
        args.base_url,
        env_keys=("INTERNVL3_BASE_URL",),
        config=local_config,
        config_key="base_url",
        default=DEFAULT_BASE_URL,
    )
    api_key = resolve_config_value(
        args.api_key,
        env_keys=("INTERN_API_KEY", "OPENAI_API_KEY"),
        config=local_config,
        config_key="api_key",
        default="",
    )
    model = resolve_config_value(
        args.model,
        env_keys=("INTERNVL3_MODEL",),
        config=local_config,
        config_key="model",
        default=DEFAULT_MODEL,
    )

    difficulties = parse_difficulty_list(args.difficulties)
    max_steps_by_difficulty = resolve_max_steps(difficulties, args.max_steps)

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.policy == "internvl" and not api_key:
        raise ValueError(
            "Missing API key. Pass --api-key, set INTERN_API_KEY, or store api_key in backend/internvl3_local_config.json."
        )
    if args.requests_per_minute < 0:
        raise ValueError("--requests-per-minute must be non-negative.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be positive.")
    if args.model_image_max_side < 64:
        raise ValueError("--model-image-max-side must be at least 64.")
    if args.demo_frame_duration_ms <= 0:
        raise ValueError("--demo-frame-duration-ms must be positive.")
    if args.demo_final_hold_ms <= 0:
        raise ValueError("--demo-final-hold-ms must be positive.")
    if args.oracle_timeout_seconds <= 0:
        raise ValueError("--oracle-timeout-seconds must be positive.")
    if args.oracle_max_expansions < 1:
        raise ValueError("--oracle-max-expansions must be at least 1.")
    if (
        args.policy == "internvl"
        and is_official_intern_api_base_url(base_url)
        and args.image_transport != "data_url"
    ):
        raise ValueError("Official Intern API should use --image-transport data_url.")

    policy: Optional[InternVL3Policy] = None
    if args.policy == "internvl":
        policy = InternVL3Policy(
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

    screenshot_server: Optional[ScreenshotURLServer] = None
    step_image_writer = StepImageWriter(root_dir=Path(args.step_image_root))
    if args.image_transport == "http_url":
        public_host = detect_public_host(base_url, args.image_public_host)
        screenshot_server = ScreenshotURLServer(
            bind_host=args.image_server_bind_host,
            port=args.image_server_port,
            public_host=public_host,
            image_dir=Path(args.image_dir),
        )
        tqdm.write(f"Serving screenshots from {screenshot_server.base_url}/<unique-frame>.png")

    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    if question_jsonl is not None:
        loaded_setups = load_benchmark_setups_from_question_jsonl(question_jsonl)
        grouped: Dict[Tuple[str, Optional[int]], List[BenchmarkSetup]] = {}
        for setup in loaded_setups:
            grouped.setdefault((setup.difficulty, setup.max_steps), []).append(setup)
        work_items: List[Tuple[str, int, Optional[List[BenchmarkSetup]]]] = [
            (difficulty, override_max_steps if override_max_steps is not None else max_steps_by_difficulty.get(difficulty, DEFAULT_MAX_STEPS_BY_DIFFICULTY[difficulty]), group)
            for (difficulty, override_max_steps), group in grouped.items()
        ]
    else:
        work_items = [(d, max_steps_by_difficulty[d], None) for d in difficulties]

    combo_summaries: List[Dict[str, Any]] = []
    include_episodes = bool(args.include_episodes or args.include_step_debug)
    try:
        for difficulty, group_max_steps, group_setups in tqdm(work_items, desc="Difficulties", unit="difficulty"):
            summary = run_single_difficulty(
                policy_name=args.policy,
                policy=policy,
                difficulty=difficulty,
                repeats=len(group_setups) if group_setups is not None else args.repeats,
                max_steps=group_max_steps,
                headless=not args.headed,
                animate=args.animate,
                progress_every=args.progress_every,
                model_image_max_side=args.model_image_max_side,
                image_transport=args.image_transport,
                screenshot_server=screenshot_server,
                step_image_writer=step_image_writer,
                collect_step_debug=bool(args.include_step_debug),
                include_legal_moves_in_prompt=bool(args.include_legal_moves_in_prompt),
                include_prompt_trajectory=bool(args.include_prompt_trajectory),
                save_demo_gifs=bool(args.save_demo_gifs),
                demo_frame_duration_ms=args.demo_frame_duration_ms,
                demo_final_hold_ms=args.demo_final_hold_ms,
                oracle_timeout_seconds=args.oracle_timeout_seconds,
                oracle_max_expansions=args.oracle_max_expansions,
                setups=group_setups,
            )
            combo_summaries.append(summary)
            tqdm.write(
                f"difficulty={difficulty} | "
                f"grid={summary['grid_size']}x{summary['grid_size']} | "
                f"ropes={summary['rope_count']} | "
                f"init_crossings={format_optional_float(summary['initial_crossings'])} | "
                f"success={summary['success_rate']:.3f} | "
                f"illegal={format_optional_float(summary['avg_illegal_moves'])} | "
                f"invalid={format_optional_float(summary['avg_invalid_responses'])}"
            )
    finally:
        if screenshot_server is not None:
            screenshot_server.close()

    result_payload = {
        "config": {
            "base_url": base_url,
            "model": model,
            "policy": args.policy,
            "difficulties": difficulties,
            "repeats": args.repeats,
            "max_steps_by_difficulty": max_steps_by_difficulty,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "api_retries": args.api_retries,
            "retry_sleep_seconds": args.retry_sleep_seconds,
            "requests_per_minute": args.requests_per_minute,
            "request_timeout_seconds": args.request_timeout_seconds,
            "model_image_max_side": args.model_image_max_side,
            "image_transport": args.image_transport,
            "image_server_bind_host": args.image_server_bind_host,
            "image_public_host": args.image_public_host,
            "image_server_port": args.image_server_port,
            "image_dir": args.image_dir,
            "step_image_root": args.step_image_root,
            "headless": not args.headed,
            "animate": args.animate,
            "include_episodes": include_episodes,
            "include_step_debug": bool(args.include_step_debug),
            "include_prompt_trajectory": bool(args.include_prompt_trajectory),
            "include_legal_moves_in_prompt": bool(args.include_legal_moves_in_prompt),
            "save_demo_gifs": bool(args.save_demo_gifs),
            "demo_frame_duration_ms": args.demo_frame_duration_ms,
            "demo_final_hold_ms": args.demo_final_hold_ms,
            "oracle_timeout_seconds": args.oracle_timeout_seconds,
            "oracle_max_expansions": args.oracle_max_expansions,
            "crossing_metric": "visual_rope_overlap",
            "theoretical_min_steps_metric": "straight_endpoint_visual_overlap",
        },
        "summaries": [
            serialize_summary(
                summary,
                include_episodes=include_episodes,
                include_step_debug=bool(args.include_step_debug),
            )
            for summary in combo_summaries
        ],
    }

    maybe_write_csv(args.output_csv, combo_summaries)

    jsonl_output_dir = (
        Path(args.output_json).resolve().parent
        if args.output_json
        else Path(args.output_csv).resolve().parent
        if args.output_csv
        else DEFAULT_BENCHMARK_ROOT
    )
    question_jsonl_path, model_answer_jsonl_path = write_meta_jsonl(
        output_dir=jsonl_output_dir,
        model_id=model if args.policy == "internvl" else args.policy,
        summaries=combo_summaries,
        write_question_jsonl=question_jsonl is None,
    )
    output_question_jsonl_path = question_jsonl_path if question_jsonl is None else question_jsonl
    result_payload["question_jsonl"] = to_relative_path(output_question_jsonl_path, base_dir=jsonl_output_dir)
    result_payload["model_answer_jsonl"] = str(model_answer_jsonl_path.relative_to(jsonl_output_dir))
    if args.output_json:
        maybe_write_json(args.output_json, result_payload)

    if args.output_json:
        tqdm.write(f"Wrote JSON summary to {args.output_json}")
    if args.output_csv:
        tqdm.write(f"Wrote CSV summary to {args.output_csv}")
    if question_jsonl is None:
        tqdm.write(f"Wrote question JSONL to {question_jsonl_path}")
    else:
        tqdm.write(f"Loaded question JSONL from {question_jsonl}")
    tqdm.write(f"Wrote model answer JSONL to {model_answer_jsonl_path}")


if __name__ == "__main__":
    main()
