from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

from internvl3_config import load_local_config, resolve_config_value
from tqdm.auto import tqdm

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `pillow` package. Install it with: pip install pillow"
    ) from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. Install it with: pip install requests"
    ) from exc

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `playwright` package. Install it with: pip install playwright"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from jsonl_export import compose_pure_prompt, flatten_answer_value, is_meta_prompt, split_pure_prompt, to_relative_path, write_jsonl
ENTRY_PATH = "/separation_objects/frontend/index.html"
CATALOG_PATH = PROJECT_ROOT / "frontend" / "catalog.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_separation_objects.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_separation_objects.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0
DEFAULT_REPEATS = 1
DEFAULT_SEED = 12345
DEFAULT_OPTION_MODE = "revised"
MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS = 200
VALID_OPTION_MODES = {"revised"}
COMPLETE_VISIBILITY_EXCLUDED_OBJECTS = {
    ("Bench", "tjusig"),
    ("Chair", "applaro_2"),
    ("Chair", "dalfred"),
    ("Chair", "klappsta"),
    ("Chair", "lisabo"),
    ("Chair", "skogsta"),
    ("Chair", "stig"),
    ("Chair", "teodores"),
    ("Desk", "alex"),
    ("Desk", "fredrik"),
    ("Misc", "vaniljstang"),
    ("Table", "gladom"),
    ("Table", "tornviken"),
    ("Table", "vadholma"),
    ("Table", "voxlov"),
}
TITLE_FONT_SIZE = 30
CAPTION_FONT_SIZE = 24
TEXT_MARGIN_X = 16
TEXT_MARGIN_Y = 14
TEXT_FILL = "black"
TEXT_STROKE_FILL = "black"
TEXT_STROKE_WIDTH = 0
COMPLETE_OBJECT_PANEL_SHIFT_Y = 50

def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


def option_visibility_failure_summary(scene: Dict[str, Any]) -> str:
    failures = scene.get("optionVisibilityFailures") or []
    if not failures:
        return "no failure details returned"
    first = failures[0]
    return (
        f"{len(failures)} failed part(s); first is option {first.get('optionLetter')}, "
        f"subassembly {int(first.get('subassemblyIndex', 0)) + 1}, "
        f"{first.get('objectCategory')}/{first.get('objectName')} part {first.get('partIndex')} "
        f"visible={first.get('visiblePixels')} required={first.get('requiredPixels')}"
    )


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


def difficulty_part_count_proxy(value: Any) -> int:
    label = str(value or "").strip().casefold()
    if label == "hard":
        return 9
    if label == "medium":
        return 8
    return 5


def infer_option_letters(question: str, image_paths: Sequence[str], ground_truth: str) -> List[str]:
    letters: List[str] = []
    for image_path in image_paths:
        match = re.search(r"option[_\-\s]*([A-Za-z])", Path(str(image_path)).stem)
        if match:
            letter = match.group(1).upper()
            if letter not in letters:
                letters.append(letter)
    if not letters:
        option_count = max(1, len(image_paths) - 1)
        letters = [chr(ord("A") + index) for index in range(option_count)]
    if ground_truth and ground_truth.upper() not in letters:
        letters.append(ground_truth.upper())
    return letters


@dataclass
class SampleResult:
    sample_index: int
    sample_id: str
    question: str
    repeat_index: int
    correct: bool
    predicted_answer: Optional[str]
    ground_truth: str
    invalid_response: bool
    api_error: bool
    api_error_message: Optional[str]
    raw_response_text: str
    object_category: str
    object_name: str
    difficulty: str
    part_count: int
    subassembly_count: int
    image_count: int
    split_balance_gap: int
    seed: int
    option_visibility_pass: Optional[bool]
    option_visibility_resamples: int
    correct_option_type: str
    option_types: List[Dict[str, str]]
    image_paths: List[str]


@dataclass(frozen=True)
class BenchmarkTarget:
    category: str
    object_name: str


def difficulty_label_for_part_count(part_count: int) -> str:
    value = int(part_count)
    if value <= 5:
        return "easy"
    if value <= 10:
        return "medium"
    return "hard"


def _id_component(value: Any) -> str:
    text = str(value or "any").strip() or "any"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "any"


def normalize_option_mode(value: str) -> str:
    mode = str(value or DEFAULT_OPTION_MODE).strip().lower()
    return mode if mode in VALID_OPTION_MODES else DEFAULT_OPTION_MODE


def subassembly_count_for_option_mode(option_mode: str) -> int:
    return 2


def passes_complete_visibility_filter(entry: Dict[str, Any]) -> bool:
    return (str(entry.get("category", "")), str(entry.get("name", ""))) not in COMPLETE_VISIBILITY_EXCLUDED_OBJECTS


def build_sample_id(*, category: str, object_name: str, seed: int) -> str:
    return (
        f"separation_objects_category_{_id_component(category)}"
        f"_object_{_id_component(object_name)}_seed_{seed}"
    )


def write_meta_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    results: Sequence[SampleResult],
    write_question_jsonl: bool = True,
) -> Tuple[Path, Path]:
    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir)
    question_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    for result in results:
        base_row = {
            "id": result.sample_id,
            "category": ["separation", "separation_objects", "subassembly_option_mcq"],
            "type": "reasoning",
            "meta_info": {
                "task_name": "separation_objects",
                "config": config_rel,
                "seed": result.seed,
                "repeat_index": result.repeat_index,
                "difficulty": result.difficulty,
                "object_category": result.object_category,
                "object_name": result.object_name,
                "part_count": result.part_count,
                "subassembly_count": result.subassembly_count,
                "image_count": result.image_count,
                "split_balance_gap": result.split_balance_gap,
                "option_visibility_pass": result.option_visibility_pass,
                "option_visibility_resamples": result.option_visibility_resamples,
                "correct_option_type": result.correct_option_type,
                "option_types": result.option_types,
            },
            "question": result.question,
            "images": result.image_paths,
            "gt_answer": result.ground_truth,
        }
        question_rows.append(base_row)
        answer_rows.append(
            {
                "id": result.sample_id,
                "category": ["separation", "separation_objects", "subassembly_option_mcq"],
                "type": "reasoning",
                "meta_info": {
                    **base_row["meta_info"],
                    "model_id": model_id,
                },
                "question": result.question,
                "answer": flatten_answer_value(result.predicted_answer),
                "correct": result.correct,
                "invalid_response": result.invalid_response,
                "api_error": result.api_error,
                "raw_response_text": result.raw_response_text,
            }
        )

    question_path = output_dir / "question.jsonl"
    answer_path = output_dir / "model_answer.jsonl"
    if write_question_jsonl:
        write_jsonl(question_path, question_rows)
    write_jsonl(answer_path, answer_rows)
    return question_path, answer_path


class _RepoHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".obj": "text/plain",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class ImageWriter:
    def __init__(self, *, root_dir: Path, output_json_path: Path) -> None:
        self.root_dir = root_dir
        self.output_json_path = output_json_path
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _path_component(value: str) -> str:
        return value.strip().replace("/", "_")

    def sample_dir(self, *, sample_id: str) -> Path:
        directory = self.root_dir / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_png(
        self,
        *,
        sample_id: str,
        image_key: str,
        png_bytes: bytes,
    ) -> Tuple[Path, str]:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / f"{image_key}.png"
        output_path.write_bytes(png_bytes)
        relative_path = to_relative_path(output_path, base_dir=self.output_json_path.parent)
        return output_path, relative_path

    def save_prompt_debug(
        self,
        *,
        sample_id: str,
        user_prompt: str,
        image_paths: Sequence[str],
        ground_truth: str,
        raw_response_text: str,
        predicted_answer: Optional[str],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]] = None,
    ) -> Path:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / "prompt_debug.txt"
        blocks = [
            "[user_prompt]",
            user_prompt,
            "[images]",
            "\n".join(image_paths),
            "[ground_truth]",
            ground_truth,
            "[model_response]",
            raw_response_text or "<empty>",
            "[parsed_answer]",
            predicted_answer or "<none>",
            "[api_error]",
            api_error_message or "<none>",
        ]
        if response_debug:
            blocks.extend(
                [
                    "[response_debug]",
                    json.dumps(response_debug, indent=2, ensure_ascii=False),
                ]
        )
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return output_path

    def save_pure_prompt(
        self,
        *,
        sample_id: str,
        task_block: str,
        image_paths: Sequence[str],
        tail_block: str,
    ) -> Path:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / "pure_prompt.txt"
        pure_prompt = compose_pure_prompt(task_block, image_paths, tail_block)
        output_path.write_text(pure_prompt + "\n", encoding="utf-8")
        return output_path


def start_server(host: str, port: int) -> tuple[ThreadingHTTPServer, Thread, str]:
    handler = partial(_RepoHandler, directory=str(PROJECT_ROOT.parent))
    server = ThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{server.server_address[1]}"
    return server, thread, base_url


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_caption_font(size: int):
    for font_name in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial Bold.ttf",
        "Arial.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_top_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font,
    left: int,
    right: int,
    top: int = TEXT_MARGIN_Y,
) -> None:
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=TEXT_STROKE_WIDTH)
    text_width = text_bbox[2] - text_bbox[0]
    x = left + max(0, ((right - left) - text_width) / 2)
    draw.text(
        (x, top),
        text,
        fill=TEXT_FILL,
        font=font,
        stroke_width=TEXT_STROKE_WIDTH,
        stroke_fill=TEXT_STROKE_FILL,
    )


def _draw_bottom_right_label(draw: ImageDraw.ImageDraw, *, text: str, font, canvas_width: int, canvas_height: int) -> None:
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=TEXT_STROKE_WIDTH)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = max(0, canvas_width - text_width - TEXT_MARGIN_X)
    y = max(0, canvas_height - text_height - TEXT_MARGIN_Y)
    draw.text(
        (x, y),
        text,
        fill=TEXT_FILL,
        font=font,
        stroke_width=TEXT_STROKE_WIDTH,
        stroke_fill=TEXT_STROKE_FILL,
    )


def compose_captioned_image(
    *,
    image_bytes_list: Sequence[bytes],
    image_captions: Sequence[str],
    option_label: Optional[str] = None,
    top_label: Optional[str] = None,
) -> bytes:
    if not image_bytes_list:
        raise ValueError("At least one image is required to compose a captioned image.")

    if len(image_captions) != len(image_bytes_list):
        raise ValueError("image_captions must match image_bytes_list length.")

    panels: List[Image.Image] = []
    for png_bytes in image_bytes_list:
        with Image.open(io.BytesIO(png_bytes)) as image:
            panels.append(image.convert("RGB"))

    title_font = load_caption_font(TITLE_FONT_SIZE)
    caption_font = load_caption_font(CAPTION_FONT_SIZE)
    panel_shift_y = COMPLETE_OBJECT_PANEL_SHIFT_Y if top_label else 0
    canvas_width = sum(panel.width for panel in panels)
    canvas_height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, caption_text in zip(panels, image_captions):
        if panel_shift_y:
            draw.rectangle(
                (x, 0, x + panel.width, panel_shift_y),
                fill=panel.getpixel((0, 0)),
            )
            visible_panel = panel.crop((0, 0, panel.width, max(1, panel.height - panel_shift_y)))
            canvas.paste(visible_panel, (x, panel_shift_y))
        else:
            canvas.paste(panel, (x, 0))
        _draw_centered_top_label(
            draw,
            text=caption_text,
            font=caption_font,
            left=x,
            right=x + panel.width,
            top=(TITLE_FONT_SIZE + TEXT_MARGIN_Y + 8 if top_label else TEXT_MARGIN_Y),
        )
        x += panel.width

    if top_label:
        _draw_centered_top_label(draw, text=top_label.strip(), font=title_font, left=0, right=canvas_width)

    if option_label:
        _draw_bottom_right_label(
            draw,
            text=option_label.strip(),
            font=title_font,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def image_bytes_to_jpeg_data_url(image_bytes: bytes, *, max_side: int = 768, quality: int = 85) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def data_url_to_png_bytes(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Expected a PNG data URL.")
    return base64.b64decode(data_url[len(prefix) :])


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
        normalized = normalize_message_content(candidate)
        normalized_candidates[source_name] = normalized
        if normalized and debug_info["response_content_source"] is None:
            debug_info["response_content_source"] = source_name
            debug_info["normalized_candidates"] = normalized_candidates
            return normalized, debug_info, None

    debug_info["normalized_candidates"] = normalized_candidates
    return "", debug_info, None


def _normalize_option_letter(candidate: str, option_letters: Sequence[str]) -> Optional[str]:
    option_map = {letter.casefold(): letter for letter in option_letters}
    stripped = candidate.strip().casefold()
    if not stripped:
        return None
    exact = option_map.get(stripped)
    if exact is not None:
        return exact

    prefixed_match = re.fullmatch(r"(?:option|answer)\s*[:\-]?\s*([a-z])\.?", stripped)
    if prefixed_match:
        return option_map.get(prefixed_match.group(1))

    quoted_match = re.fullmatch(r'["\']([a-z])["\']', stripped)
    if quoted_match:
        return option_map.get(quoted_match.group(1))

    return None


def parse_choice(text: str, option_letters: Sequence[str]) -> Optional[str]:
    if not text:
        return None

    lowered = text.casefold()
    try:
        payload = json.loads(lowered)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for key in ("answer", "choice", "option", "label"):
            value = payload.get(key)
            if isinstance(value, str):
                letter = _normalize_option_letter(value, option_letters)
                if letter is not None:
                    return letter

    brace_matches = re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL)
    for brace_text in reversed(brace_matches):
        for pattern in (
            r'"(?:answer|choice|option|label)"\s*:\s*"([a-z])"',
            r"(?:option|answer)\s*[:\-]?\s*([a-z])\b",
            r'"([a-z])"',
        ):
            match = re.search(pattern, brace_text)
            if not match:
                continue
            letter = _normalize_option_letter(match.group(1), option_letters)
            if letter is not None:
                return letter

    explicit_tail_matches = list(re.finditer(r"(?:option|answer)\s*[:\-]?\s*([a-z])\b", lowered))
    for match in reversed(explicit_tail_matches):
        letter = _normalize_option_letter(match.group(1), option_letters)
        if letter is not None:
            return letter

    quoted_tail_matches = list(re.finditer(r'"([a-z])"', lowered))
    for match in reversed(quoted_tail_matches):
        letter = _normalize_option_letter(match.group(1), option_letters)
        if letter is not None:
            return letter

    standalone_tail_matches = list(re.finditer(r"\b([a-z])\b", lowered))
    for match in reversed(standalone_tail_matches):
        letter = _normalize_option_letter(match.group(1), option_letters)
        if letter is not None:
            return letter

    return None


def decomposition_part_label(subassembly_count: int) -> str:
    return "three-part" if int(subassembly_count) == 3 else "two-part"


def candidate_subassembly_label(subassembly_count: int) -> str:
    return "three candidate subassemblies" if int(subassembly_count) == 3 else "two candidate subassemblies"


def build_user_prompt_blocks(*, question: str, option_letters: Sequence[str], subassembly_count: int = 2) -> Tuple[str, str, str]:
    option_space = ", ".join(f'"{letter}"' for letter in option_letters)
    decomposition_label = decomposition_part_label(subassembly_count)
    subassembly_label = candidate_subassembly_label(subassembly_count)
    task_block_lines = [
        "[Task]",
        "You are solving a topological task called Separation Objects under the separation category.",
        (
            f"In this task, you must determine which candidate option shows the correct {decomposition_label} decomposition "
            "of the complete object."
        ),
        (
            f"The correct option must split the complete object into exactly {subassembly_label} that "
            "together form the complete object, with no missing parts, no extra parts, and no parts from another object."
        ),
        (
            "Image 1 contains two labeled views of the complete object: Front upper right 45-degree oblique and "
            f"Back upper left 45-degree oblique. Each option image contains the {subassembly_label} for one option; "
            "each subassembly is shown from the front upper right 45-degree oblique view."
        ),
        "The visual evidence for this question is provided below.",
    ]
    tail_block_lines = [
        "[Rules]",
        "1. Use only the images and text provided in this prompt.",
        "2. If answer options are provided, choose only from the provided options.",
        "3. Do not output explanation beyond the required final answer.",
        "",
        "[Question]",
        question,
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer":"{ans}"} and nothing else.',
        f'Replace {{ans}} with the single legal answer for this task, chosen from {option_space}.',
    ]
    task_block = "\n".join(task_block_lines)
    tail_block = "\n".join(tail_block_lines)
    debug_prompt = "\n\n".join([task_block, "[Images inserted here]", tail_block])
    return task_block, tail_block, debug_prompt


def resolve_user_prompt_blocks(
    *,
    question: str,
    option_letters: Sequence[str],
    image_count: int,
    subassembly_count: int = 2,
) -> Tuple[str, str, str, str]:
    if is_meta_prompt(question):
        task_block, tail_block = split_pure_prompt(question, image_count=image_count)
        user_prompt = question
        pure_prompt = question
    else:
        task_block, tail_block, user_prompt = build_user_prompt_blocks(
            question=question,
            option_letters=option_letters,
            subassembly_count=subassembly_count,
        )
        pure_prompt = ""
    return task_block, tail_block, user_prompt, pure_prompt


def build_question_text(option_letters: Sequence[str], *, subassembly_count: int = 2) -> str:
    options_line = ", ".join(option_letters)
    return (
        f"Which option shows the correct {decomposition_part_label(subassembly_count)} decomposition of the complete object? "
        f"Choose one option from {options_line}."
    )


def complete_object_captions(whole_views: Sequence[Dict[str, Any]]) -> List[str]:
    captions: List[str] = []
    for index, view in enumerate(whole_views):
        view_label = str(view.get("viewLabel") or f"View {index + 1}")
        captions.append(view_label)
    return captions


def load_eligible_targets(*, option_mode: str = DEFAULT_OPTION_MODE) -> List[BenchmarkTarget]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing catalog file: {CATALOG_PATH}. Run separation_objects/backend/generate_catalog.py first."
        )

    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    raw_entries: List[Dict[str, Any]] = []
    for category, rows in payload.get("categories", {}).items():
        if not isinstance(rows, list):
            continue
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            raw_entries.append(
                {
                    "category": category,
                    "name": str(entry.get("name", "")),
                    "part_count": int(entry.get("part_count", 0)),
                }
            )

    binary_eligible = [
        entry
        for entry in raw_entries
        if int(entry["part_count"]) >= 2 and passes_complete_visibility_filter(entry)
    ]
    if not binary_eligible:
        raise RuntimeError("No decomposition-eligible entries found in catalog.")

    category_counts: Dict[str, int] = {}
    for entry in binary_eligible:
        category_counts[entry["category"]] = category_counts.get(entry["category"], 0) + 1
    distinct_categories = {entry["category"] for entry in binary_eligible}

    normalized_mode = normalize_option_mode(option_mode)
    if normalized_mode == "revised":
        targets = [
            BenchmarkTarget(category=entry["category"], object_name=entry["name"])
            for entry in binary_eligible
            if int(entry["part_count"]) >= 3
            and any(
                other["category"] == entry["category"]
                and other["name"] != entry["name"]
                for other in binary_eligible
            )
        ]
    elif normalized_mode == "revise_3parts":
        targets = [
            BenchmarkTarget(category=entry["category"], object_name=entry["name"])
            for entry in binary_eligible
            if int(entry["part_count"]) >= 4
            and any(
                other["category"] == entry["category"]
                and other["name"] != entry["name"]
                for other in binary_eligible
            )
        ]
    else:
        targets = [
            BenchmarkTarget(category=entry["category"], object_name=entry["name"])
            for entry in binary_eligible
            if int(entry["part_count"]) >= 3
            and category_counts.get(entry["category"], 0) >= 2
            and len(distinct_categories) >= 2
        ]

    if not targets:
        raise RuntimeError(f"No valid targets found for the {normalized_mode} option benchmark.")
    return sorted(targets, key=lambda item: (item.category, item.object_name))


def parse_category_filter(category: str) -> List[str]:
    return [item.strip() for item in category.split(",") if item.strip()]


def select_targets(*, category: str, object_name: str, option_mode: str = DEFAULT_OPTION_MODE) -> List[BenchmarkTarget]:
    targets = load_eligible_targets(option_mode=option_mode)
    categories = parse_category_filter(category)

    if object_name and len(categories) > 1:
        raise ValueError("--object-name can only be used with a single --category value.")

    if categories and object_name:
        matches = [
            target
            for target in targets
            if target.category == categories[0] and target.object_name == object_name
        ]
        if not matches:
            raise ValueError(
                f"No eligible object found for category={categories[0]!r}, object_name={object_name!r}."
            )
        return matches

    if categories:
        category_set = set(categories)
        matches = [target for target in targets if target.category in category_set]
        if not matches:
            raise ValueError(f"No eligible objects found in categories={categories!r}.")
        return matches

    if object_name:
        raise ValueError("--object-name requires --category.")

    return targets


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

    def _safe_response_payload(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

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

    def predict(
        self,
        *,
        image_urls: Sequence[str],
        question: str,
        option_letters: Sequence[str],
        subassembly_count: int = 2,
    ) -> Tuple[Optional[str], str, Optional[str], bool, Dict[str, Any], str, str, str]:
        task_block, tail_block, user_prompt, _ = resolve_user_prompt_blocks(
            question=question,
            option_letters=option_letters,
            image_count=len(image_urls),
            subassembly_count=subassembly_count,
        )
        content_blocks = (
            [{"type": "text", "text": task_block}]
            + [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls]
        )
        if tail_block:
            content_blocks.append({"type": "text", "text": tail_block})
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
                },
            ],
        }

        endpoint = f"{self.base_url}/chat/completions"
        last_error: Optional[str] = None
        for attempt in range(self.api_retries + 1):
            try:
                self._wait_for_rate_limit_slot()
                response = self._session.post(
                    endpoint,
                    headers=self._build_request_headers(),
                    json=request_payload,
                    timeout=self.request_timeout_seconds,
                )
                if response.status_code >= 400:
                    last_error = self._format_error_response(response)
                    if response.status_code >= 500 and attempt < self.api_retries:
                        time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                        continue
                    return (
                        None,
                        "",
                        last_error,
                        False,
                        {"response_payload": self._safe_response_payload(response)},
                        user_prompt,
                        task_block,
                        tail_block,
                    )

                payload = response.json()
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.api_retries:
                    time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                    continue
                return None, "", last_error, False, {"request_exception": repr(exc)}, user_prompt, task_block, tail_block

            response_text, debug_info, extraction_error = extract_response_text(payload)
            parsed_choice = parse_choice(response_text, option_letters)
            debug_info["request_payload"] = request_payload
            if extraction_error is not None:
                return None, response_text, extraction_error, False, debug_info, user_prompt, task_block, tail_block
            if parsed_choice is None or parsed_choice not in set(option_letters):
                debug_info["parse_error"] = "Failed to parse model answer."
                return None, response_text, None, True, debug_info, user_prompt, task_block, tail_block
            return parsed_choice, response_text, None, False, debug_info, user_prompt, task_block, tail_block

        return None, "", last_error or "Unknown API failure.", False, {}, user_prompt, task_block, tail_block


async def load_scene_sample(
    *,
    page,
    sample_id: str,
    repeat_index: int,
    seed: int,
    category: str,
    object_name: str,
    option_mode: str,
    image_writer: ImageWriter,
    scene: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    if scene is None:
        scene = await page.evaluate(
            "(async (cfg) => await window.topoBench.generateTwoPartOptionTask(cfg))",
            {
                "category": category,
                "objectName": object_name,
                "seed": seed,
                "optionMode": option_mode,
            },
        )

    image_urls: List[str] = []
    image_paths: List[str] = []
    whole_view_bytes = [data_url_to_png_bytes(view["dataUrl"]) for view in scene["wholeViews"]]
    whole_image_bytes = compose_captioned_image(
        image_bytes_list=whole_view_bytes,
        image_captions=complete_object_captions(scene["wholeViews"]),
        top_label="Complete Object",
    )
    _, relative_path = image_writer.save_png(
        sample_id=sample_id,
        image_key="complete_object",
        png_bytes=whole_image_bytes,
    )
    image_urls.append(image_bytes_to_jpeg_data_url(whole_image_bytes))
    image_paths.append(relative_path)

    for option_index, option in enumerate(scene["options"], start=2):
        option_view_bytes = [
            data_url_to_png_bytes(subassembly["views"][0]["dataUrl"])
            for subassembly in option["subassemblies"]
        ]
        option_image_bytes = compose_captioned_image(
            image_bytes_list=option_view_bytes,
            image_captions=[
                f"Subassembly {index + 1}"
                for index in range(len(option_view_bytes))
            ],
            option_label=f"Option {option['letter']}",
        )
        _, relative_path = image_writer.save_png(
            sample_id=sample_id,
            image_key=f"option_{option['letter']}",
            png_bytes=option_image_bytes,
        )
        image_urls.append(image_bytes_to_jpeg_data_url(option_image_bytes))
        image_paths.append(relative_path)
    return scene, image_urls, image_paths


async def generate_option_visible_scene(
    *,
    page,
    seed_rng: random.Random,
    category: str,
    object_name: str,
    option_mode: str,
    max_attempts: int = MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS,
) -> Tuple[int, Dict[str, Any], int]:
    last_failure = ""
    for attempt in range(1, int(max_attempts) + 1):
        scene_seed = int(seed_rng.randrange(0, 2**31 - 1))
        scene = await page.evaluate(
            "(async (cfg) => await window.topoBench.generateTwoPartOptionTask(cfg))",
            {
                "category": category,
                "objectName": object_name,
                "seed": scene_seed,
                "optionMode": option_mode,
            },
        )
        if bool(scene.get("optionVisibilityPass")):
            scene["optionVisibilityRejectedScenes"] = attempt - 1
            return scene_seed, scene, attempt - 1
        last_failure = option_visibility_failure_summary(scene)

    raise RuntimeError(
        f"Could not generate an option-visible separation_objects scene for "
        f"{category}/{object_name} after {max_attempts} attempts. Last failure: {last_failure}"
    )


def compute_summary(results: Sequence[SampleResult]) -> Dict[str, Any]:
    count = len(results)
    correct = sum(1 for result in results if result.correct)
    invalid = sum(1 for result in results if result.invalid_response)
    api_errors = sum(1 for result in results if result.api_error)
    mean_parts = sum(result.part_count for result in results) / count if count else 0.0
    mean_subassemblies = sum(result.subassembly_count for result in results) / count if count else 0.0
    mean_images = sum(result.image_count for result in results) / count if count else 0.0
    mean_gap = sum(result.split_balance_gap for result in results) / count if count else 0.0
    return {
        "total_samples": count,
        "correct": correct,
        "accuracy": (correct / count) if count else 0.0,
        "invalid_responses": invalid,
        "api_errors": api_errors,
        "mean_part_count": mean_parts,
        "mean_subassembly_count": mean_subassemblies,
        "mean_image_count": mean_images,
        "mean_split_balance_gap": mean_gap,
    }


def compute_grouped_summaries(
    results: Sequence[SampleResult],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_category: Dict[str, List[SampleResult]] = {}
    by_object: Dict[Tuple[str, str], List[SampleResult]] = {}

    for result in results:
        by_category.setdefault(result.object_category, []).append(result)
        by_object.setdefault((result.object_category, result.object_name), []).append(result)

    category_rows: List[Dict[str, Any]] = []
    for category, rows in sorted(by_category.items()):
        summary = compute_summary(rows)
        category_rows.append({"object_category": category, **summary})

    object_rows: List[Dict[str, Any]] = []
    for (category, object_name), rows in sorted(by_object.items()):
        summary = compute_summary(rows)
        object_rows.append({"object_category": category, "object_name": object_name, **summary})

    return category_rows, object_rows


def write_csv(output_path: Path, results: Sequence[SampleResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "sample_id",
                "correct",
                "predicted_answer",
                "ground_truth",
                "invalid_response",
                "api_error",
                "api_error_message",
                "object_category",
                "object_name",
                "difficulty",
                "part_count",
                "subassembly_count",
                "image_count",
                "split_balance_gap",
                "seed",
                "option_visibility_pass",
                "option_visibility_resamples",
                "correct_option_type",
                "option_types_json",
                "image_paths_json",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_index": result.sample_index,
                    "sample_id": result.sample_id,
                    "correct": result.correct,
                    "predicted_answer": result.predicted_answer or "",
                    "ground_truth": result.ground_truth,
                    "invalid_response": result.invalid_response,
                    "api_error": result.api_error,
                    "api_error_message": result.api_error_message or "",
                    "object_category": result.object_category,
                    "object_name": result.object_name,
                    "difficulty": result.difficulty,
                    "part_count": result.part_count,
                    "subassembly_count": result.subassembly_count,
                    "image_count": result.image_count,
                    "split_balance_gap": result.split_balance_gap,
                    "seed": result.seed,
                    "option_visibility_pass": "" if result.option_visibility_pass is None else result.option_visibility_pass,
                    "option_visibility_resamples": result.option_visibility_resamples,
                    "correct_option_type": result.correct_option_type,
                    "option_types_json": json.dumps(result.option_types, ensure_ascii=False),
                    "image_paths_json": json.dumps(result.image_paths, ensure_ascii=False),
                }
            )


def write_summary_csv(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_loaded_question_rows(
    *,
    args: argparse.Namespace,
    question_jsonl: Path,
    output_json: Path,
    image_writer: ImageWriter,
    policy: Optional[InternVL3Policy],
) -> List[SampleResult]:
    rows = read_question_jsonl(question_jsonl)
    if not rows:
        raise ValueError(f"Question JSONL has no data rows: {question_jsonl}")
    results: List[SampleResult] = []
    progress = tqdm(total=len(rows), desc="Separation Objects loaded benchmark")

    for sample_index, row in enumerate(rows):
        meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        raw_image_paths = row.get("images")
        if not isinstance(raw_image_paths, list) or not raw_image_paths:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} must contain image paths.")

        image_paths: List[str] = []
        image_urls: List[str] = []
        for raw_image_path in raw_image_paths:
            image_path = resolve_question_image_path(str(raw_image_path), question_jsonl=question_jsonl)
            image_paths.append(to_relative_path(image_path, base_dir=output_json.parent))
            image_urls.append(image_bytes_to_jpeg_data_url(image_path.read_bytes()))

        question_text = str(row.get("question") or "")
        if not question_text:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} is missing question text.")
        ground_truth = str(row.get("gt_answer") or "").strip().upper()
        option_letters = infer_option_letters(question_text, image_paths, ground_truth)
        part_count = int(meta_info.get("part_count", difficulty_part_count_proxy(meta_info.get("difficulty"))))
        subassembly_count = int(meta_info.get("subassembly_count", 2))
        sample_id = str(row.get("id") or build_sample_id(
            category=str(meta_info.get("object_category", "any")),
            object_name=str(meta_info.get("object_name", "any")),
            seed=int(meta_info.get("seed", sample_index)),
        ))
        row_task_block, row_tail_block, row_user_prompt, row_pure_prompt = resolve_user_prompt_blocks(
            question=question_text,
            option_letters=option_letters,
            image_count=len(image_paths),
            subassembly_count=subassembly_count,
        )
        pure_prompt = row_pure_prompt or compose_pure_prompt(row_task_block, image_paths, row_tail_block)
        image_writer.save_pure_prompt(
            sample_id=sample_id,
            task_block=row_task_block,
            image_paths=image_paths,
            tail_block=row_tail_block,
        )

        if args.oracle:
            predicted_answer = ground_truth
            raw_response_text = json.dumps({"answer": predicted_answer}, separators=(",", ":"))
            api_error_message = None
            invalid_response = False
            task_block = row_task_block
            tail_block = row_tail_block
            user_prompt = row_user_prompt
            response_debug = {"oracle": True}
        else:
            assert policy is not None
            try:
                (
                    predicted_answer,
                    raw_response_text,
                    api_error_message,
                    invalid_response,
                    response_debug,
                    user_prompt,
                    task_block,
                    tail_block,
                ) = policy.predict(
                    image_urls=image_urls,
                    question=question_text,
                    option_letters=option_letters,
                    subassembly_count=subassembly_count,
                )
            except Exception as exc:
                predicted_answer = None
                raw_response_text = ""
                api_error_message = f"{type(exc).__name__}: {exc}"
                invalid_response = False
                response_debug = {"exception": type(exc).__name__}
                user_prompt = row_user_prompt
                task_block = row_task_block
                tail_block = row_tail_block

        loaded_option_visibility_pass = meta_info.get("option_visibility_pass")
        result = SampleResult(
            sample_index=sample_index,
            sample_id=sample_id,
            question=pure_prompt,
            repeat_index=int(meta_info.get("repeat_index", sample_index)),
            correct=predicted_answer == ground_truth,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            invalid_response=invalid_response,
            api_error=api_error_message is not None,
            api_error_message=api_error_message,
            raw_response_text=raw_response_text,
            object_category=str(meta_info.get("object_category", "loaded")),
            object_name=str(meta_info.get("object_name", sample_id)),
            difficulty=str(meta_info.get("difficulty", difficulty_label_for_part_count(part_count))),
            part_count=part_count,
            subassembly_count=subassembly_count,
            image_count=len(image_paths),
            split_balance_gap=int(meta_info.get("split_balance_gap", 0)),
            seed=int(meta_info.get("seed", sample_index)),
            option_visibility_pass=loaded_option_visibility_pass if isinstance(loaded_option_visibility_pass, bool) else None,
            option_visibility_resamples=int(meta_info.get("option_visibility_resamples", 0)),
            correct_option_type=str(meta_info.get("correct_option_type", "loaded")),
            option_types=list(meta_info.get("option_types", [])) if isinstance(meta_info.get("option_types"), list) else [],
            image_paths=image_paths,
        )
        results.append(result)

        image_writer.save_prompt_debug(
            sample_id=sample_id,
            user_prompt=user_prompt,
            image_paths=image_paths,
            ground_truth=ground_truth,
            raw_response_text=raw_response_text,
            predicted_answer=predicted_answer,
            api_error_message=api_error_message,
            response_debug=response_debug,
        )
        image_writer.save_pure_prompt(
            sample_id=sample_id,
            task_block=task_block,
            image_paths=image_paths,
            tail_block=tail_block,
        )

        accuracy = sum(1 for item in results if item.correct) / len(results)
        progress.set_postfix(
            accuracy=f"{accuracy:.3f}",
            invalid=sum(1 for item in results if item.invalid_response),
            api_errors=sum(1 for item in results if item.api_error),
        )
        progress.update(1)

    progress.close()
    return results


async def run_benchmark(args) -> None:
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
    option_mode = normalize_option_mode(args.option_mode)
    targets: List[BenchmarkTarget] = []
    if question_jsonl is None:
        targets = select_targets(category=args.category, object_name=args.object_name, option_mode=option_mode)
    total_samples = 0 if question_jsonl is not None else len(targets) * int(args.repeats)

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_by_category_json = output_json.with_name(f"{output_json.stem}_by_category.json")
    output_by_category_csv = output_csv.with_name(f"{output_csv.stem}_by_category.csv")
    output_by_object_json = output_json.with_name(f"{output_json.stem}_by_object.json")
    output_by_object_csv = output_csv.with_name(f"{output_csv.stem}_by_object.csv")
    image_root = Path(args.image_root).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_writer = ImageWriter(root_dir=image_root, output_json_path=output_json)
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

    results: List[SampleResult] = []
    seed_rng = random.Random(int(args.seed))
    if question_jsonl is not None:
        results = run_loaded_question_rows(
            args=args,
            question_jsonl=question_jsonl,
            output_json=output_json,
            image_writer=image_writer,
            policy=policy,
        )
    else:
        server, thread, base_url_local = start_server(args.host, args.port)
        page_url = f"{base_url_local}{ENTRY_PATH}"
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=args.headless,
                    args=["--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check"],
                )
                page = await browser.new_page(viewport={"width": 1800, "height": 1280}, device_scale_factor=1.0)
                await page.goto(page_url, wait_until="load")
                await page.wait_for_function(
                    "() => window.topoBench && typeof window.topoBench.generateTwoPartOptionTask === 'function'",
                    timeout=120000,
                )

                progress = tqdm(total=total_samples, desc="Separation Objects option benchmark")
                sample_index = 0
                for target in targets:
                    for repeat_index in range(int(args.repeats)):
                        seed, scene, rejected_scenes = await generate_option_visible_scene(
                            page=page,
                            seed_rng=seed_rng,
                            category=target.category,
                            object_name=target.object_name,
                            option_mode=option_mode,
                        )
                        sample_id = build_sample_id(
                            category=target.category,
                            object_name=target.object_name,
                            seed=seed,
                        )
                        scene, image_urls, image_paths = await load_scene_sample(
                            page=page,
                            sample_id=sample_id,
                            repeat_index=repeat_index,
                            seed=seed,
                            category=target.category,
                            object_name=target.object_name,
                            option_mode=option_mode,
                            image_writer=image_writer,
                            scene=scene,
                        )

                        option_letters = [str(option["letter"]) for option in scene["options"]]
                        subassembly_count = int(scene["subassemblyCount"])
                        question_text = build_question_text(option_letters, subassembly_count=subassembly_count)
                        task_block, tail_block, user_prompt = build_user_prompt_blocks(
                            question=question_text,
                            option_letters=option_letters,
                            subassembly_count=subassembly_count,
                        )
                        image_writer.save_pure_prompt(
                            sample_id=sample_id,
                            task_block=task_block,
                            image_paths=image_paths,
                            tail_block=tail_block,
                        )
                        if args.oracle:
                            predicted_answer = scene["correctOptionLetter"]
                            raw_response_text = json.dumps({"answer": predicted_answer}, separators=(",", ":"))
                            api_error_message = None
                            invalid_response = False
                            response_debug = {
                                "oracle": True,
                                "correct_option_letter": scene["correctOptionLetter"],
                            }
                        else:
                            assert policy is not None
                            try:
                                (
                                    predicted_answer,
                                    raw_response_text,
                                    api_error_message,
                                    invalid_response,
                                    response_debug,
                                    user_prompt,
                                    task_block,
                                    tail_block,
                                ) = policy.predict(
                                    image_urls=image_urls,
                                    question=question_text,
                                    option_letters=option_letters,
                                    subassembly_count=subassembly_count,
                                )
                            except Exception as exc:
                                predicted_answer = None
                                raw_response_text = ""
                                api_error_message = f"{type(exc).__name__}: {exc}"
                                invalid_response = False
                                response_debug = {"exception": type(exc).__name__}

                        pure_prompt = compose_pure_prompt(task_block, image_paths, tail_block)
                        correct = predicted_answer == scene["correctOptionLetter"]
                        result = SampleResult(
                            sample_index=sample_index,
                            sample_id=sample_id,
                            question=pure_prompt,
                            repeat_index=repeat_index,
                            correct=correct,
                            predicted_answer=predicted_answer,
                            ground_truth=scene["correctOptionLetter"],
                            invalid_response=invalid_response,
                            api_error=api_error_message is not None,
                            api_error_message=api_error_message,
                            raw_response_text=raw_response_text,
                            object_category=scene["objectCategory"],
                            object_name=scene["objectName"],
                            difficulty=difficulty_label_for_part_count(int(scene["partCount"])),
                            part_count=int(scene["partCount"]),
                            subassembly_count=subassembly_count,
                            image_count=len(image_paths),
                            split_balance_gap=int(scene["splitBalanceGap"]),
                            seed=seed,
                            option_visibility_pass=bool(scene.get("optionVisibilityPass")),
                            option_visibility_resamples=rejected_scenes,
                            correct_option_type=str(scene["correctOptionType"]),
                            option_types=[
                                {
                                    "letter": str(option["letter"]),
                                    "optionType": str(option["optionType"]),
                                    "sourceCategory": str(option["sourceCategory"]),
                                    "sourceObjectName": str(option["sourceObjectName"]),
                                    "optionDetail": str(option.get("optionDetail") or ""),
                                }
                                for option in scene["options"]
                            ],
                            image_paths=image_paths,
                        )
                        results.append(result)

                        image_writer.save_prompt_debug(
                            sample_id=sample_id,
                            user_prompt=user_prompt,
                            image_paths=image_paths,
                            ground_truth=scene["correctOptionLetter"],
                            raw_response_text=raw_response_text,
                            predicted_answer=predicted_answer,
                            api_error_message=api_error_message,
                            response_debug=response_debug,
                        )
                        image_writer.save_pure_prompt(
                            sample_id=sample_id,
                            task_block=task_block,
                            image_paths=image_paths,
                            tail_block=tail_block,
                        )

                        accuracy = sum(1 for item in results if item.correct) / len(results)
                        progress.set_postfix(
                            accuracy=f"{accuracy:.3f}",
                            invalid=sum(1 for item in results if item.invalid_response),
                            api_errors=sum(1 for item in results if item.api_error),
                            object=f"{scene['objectCategory']}/{scene['objectName']}",
                            part_count=scene["partCount"],
                            repeat=f"{repeat_index + 1}/{args.repeats}",
                            rejected=rejected_scenes,
                        )
                        progress.update(1)
                        sample_index += 1

                progress.close()
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

    summary = compute_summary(results)
    by_category_summary, by_object_summary = compute_grouped_summaries(results)
    payload = {
        "config": {
            "category": args.category,
            "object_name": args.object_name,
            "repeats_per_object": None if question_jsonl is not None else args.repeats,
            "target_count": len(results) if question_jsonl is not None else len(targets),
            "total_samples": len(results) if question_jsonl is not None else total_samples,
            "seed": args.seed,
            "difficulty_rule": "catalog object part_count defines difficulty: 3-5=easy, 6-10=medium, 11+=hard",
            "model": model,
            "base_url": base_url,
            "oracle": args.oracle,
            "task": "subassembly_option_mcq",
            "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        },
        "summary": summary,
        "summary_by_category": by_category_summary,
        "summary_by_object": by_object_summary,
        "samples": [result.__dict__ for result in results],
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
    write_csv(output_csv, results)
    output_by_category_json.write_text(
        json.dumps(by_category_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_by_object_json.write_text(
        json.dumps(by_object_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(output_by_category_csv, by_category_summary)
    write_summary_csv(output_by_object_csv, by_object_summary)

    print(
        f"accuracy={summary['accuracy']:.3f} | correct={summary['correct']}/{summary['total_samples']} | "
        f"invalid={summary['invalid_responses']} | api_errors={summary['api_errors']}"
    )
    print(f"Wrote JSON results to {output_json}")
    print(f"Wrote CSV results to {output_csv}")
    print(f"Wrote category JSON summary to {output_by_category_json}")
    print(f"Wrote category CSV summary to {output_by_category_csv}")
    print(f"Wrote object JSON summary to {output_by_object_json}")
    print(f"Wrote object CSV summary to {output_by_object_csv}")
    if question_jsonl is None:
        print(f"Wrote question JSONL to {question_jsonl_path}")
    else:
        print(f"Loaded question JSONL from {question_jsonl}")
    print(f"Wrote model answer JSONL to {model_answer_jsonl_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InternVL3 benchmark for separation_objects option task.")
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
    parser.add_argument("--category", default="", help="Optional single category or comma-separated categories.")
    parser.add_argument("--object-name", default="")
    parser.add_argument("--option-mode", choices=sorted(VALID_OPTION_MODES), default=DEFAULT_OPTION_MODE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--question-jsonl", default="", help="Load existing question.jsonl rows and images instead of generating new scenes.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--oracle", action="store_true", help="Use the ground-truth answer instead of calling the API.")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    import asyncio

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
