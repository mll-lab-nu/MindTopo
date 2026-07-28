from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Sequence

from playwright.async_api import async_playwright
from tqdm.auto import tqdm

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "generate_samples.py requires the `pillow` package. Install it with: pip install pillow"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENTRY_PATH = "/separation_objects/frontend/index.html"
CATALOG_PATH = PROJECT_ROOT / "frontend" / "catalog.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "output" / "question.jsonl"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "output" / "images"
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

from jsonl_export import compose_pure_prompt, to_relative_path, write_jsonl  # noqa: E402


@dataclass(frozen=True)
class GenerationTarget:
    category: str
    object_name: str


def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


def option_visibility_failure_summary(scene: dict) -> str:
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


class _RepoHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".obj": "text/plain",
    }

    def log_message(self, _format: str, *_args) -> None:
        return


def start_server(host: str, port: int) -> tuple[ThreadingHTTPServer, Thread, str]:
    handler = partial(_RepoHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{server.server_address[1]}"
    return server, thread, base_url


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate separation_objects option samples.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--repeats", "--count", dest="repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--category", default="", help="Optional single category or comma-separated categories.")
    parser.add_argument("--object-name", default="")
    parser.add_argument("--option-mode", choices=sorted(VALID_OPTION_MODES), default=DEFAULT_OPTION_MODE)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


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
    option_label: str | None = None,
    top_label: str | None = None,
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


def data_url_to_png_bytes(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Expected a PNG data URL.")
    return base64.b64decode(data_url[len(prefix) :])


async def generate_option_visible_scene(
    *,
    page,
    seed_rng: random.Random,
    category: str,
    object_name: str,
    option_mode: str,
    max_attempts: int = MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS,
) -> tuple[int, dict, int, int]:
    max_attempts = max(0, int(max_attempts))
    if max_attempts <= 0:
        raise RuntimeError(
            f"Could not generate an option-visible separation_objects scene for "
            f"{category}/{object_name}: no visibility resample attempts left."
        )

    last_failure = ""
    for attempt in range(1, max_attempts + 1):
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
            return scene_seed, scene, attempt - 1, attempt
        last_failure = option_visibility_failure_summary(scene)

    raise RuntimeError(
        f"Could not generate an option-visible separation_objects scene for "
        f"{category}/{object_name} after {max_attempts} attempts. Last failure: {last_failure}"
    )


def decomposition_part_label(subassembly_count: int) -> str:
    return "three-part" if int(subassembly_count) == 3 else "two-part"


def candidate_subassembly_label(subassembly_count: int) -> str:
    return "three candidate subassemblies" if int(subassembly_count) == 3 else "two candidate subassemblies"


def build_question_text(option_letters: Sequence[str], *, subassembly_count: int = 2) -> str:
    options_line = ", ".join(option_letters)
    return (
        f"Which option shows the correct {decomposition_part_label(subassembly_count)} decomposition of the complete object? "
        f"Choose one option from {options_line}."
    )


def build_user_prompt_blocks(*, question: str, option_letters: Sequence[str], subassembly_count: int = 2) -> tuple[str, str]:
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
    return "\n".join(task_block_lines), "\n".join(tail_block_lines)


def _path_component(value: str) -> str:
    return value.strip().replace("/", "_")


def parse_category_filter(category: str) -> List[str]:
    return [item.strip() for item in category.split(",") if item.strip()]


def normalize_option_mode(value: str) -> str:
    mode = str(value or DEFAULT_OPTION_MODE).strip().lower()
    return mode if mode in VALID_OPTION_MODES else DEFAULT_OPTION_MODE


def subassembly_count_for_option_mode(option_mode: str) -> int:
    return 2


def passes_complete_visibility_filter(entry: Dict[str, Any]) -> bool:
    return (str(entry.get("category", "")), str(entry.get("name", ""))) not in COMPLETE_VISIBILITY_EXCLUDED_OBJECTS


def load_eligible_targets(*, option_mode: str = DEFAULT_OPTION_MODE) -> List[GenerationTarget]:
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
            GenerationTarget(category=entry["category"], object_name=entry["name"])
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
            GenerationTarget(category=entry["category"], object_name=entry["name"])
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
            GenerationTarget(category=entry["category"], object_name=entry["name"])
            for entry in binary_eligible
            if int(entry["part_count"]) >= 3
            and category_counts.get(entry["category"], 0) >= 2
            and len(distinct_categories) >= 2
        ]
    if not targets:
        raise RuntimeError(f"No valid targets found for the {normalized_mode} option generator.")
    return sorted(targets, key=lambda item: (item.category, item.object_name))


def select_targets(*, category: str, object_name: str, option_mode: str = DEFAULT_OPTION_MODE) -> List[GenerationTarget]:
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


def sample_dir(images_dir: Path, *, sample_id: str) -> Path:
    directory = images_dir / _path_component(sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def difficulty_label_for_part_count(part_count: int) -> str:
    value = int(part_count)
    if value <= 5:
        return "easy"
    if value <= 10:
        return "medium"
    return "hard"


def _id_component(value: str) -> str:
    text = str(value or "any").strip() or "any"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "any"


def build_sample_id(*, category: str, object_name: str, seed: int) -> str:
    return (
        f"separation_objects_category_{_id_component(category)}"
        f"_object_{_id_component(object_name)}_seed_{seed}"
    )


def complete_object_captions(whole_views: Sequence[dict]) -> List[str]:
    captions: List[str] = []
    for index, view in enumerate(whole_views):
        view_label = str(view.get("viewLabel") or f"View {index + 1}")
        captions.append(view_label)
    return captions


def build_sample_record(*, sample_index: int, sample_id: str, scene: dict, image_paths: list[str], config_rel: str, repeat_index: int) -> dict:
    option_letters = [str(option["letter"]) for option in scene["options"]]
    subassembly_count = int(scene["subassemblyCount"])
    question_text = build_question_text(option_letters, subassembly_count=subassembly_count)
    task_block, tail_block = build_user_prompt_blocks(
        question=question_text,
        option_letters=option_letters,
        subassembly_count=subassembly_count,
    )
    pure_prompt = compose_pure_prompt(task_block, image_paths, tail_block)
    return {
        "id": sample_id,
        "category": ["separation", "separation_objects", "subassembly_option_mcq"],
        "type": "reasoning",
        "meta_info": {
            "task_name": "separation_objects",
            "config": config_rel,
            "seed": int(scene["seed"]),
            "repeat_index": repeat_index,
            "difficulty": difficulty_label_for_part_count(int(scene["partCount"])),
            "object_category": scene["objectCategory"],
            "object_name": scene["objectName"],
            "part_count": int(scene["partCount"]),
            "subassembly_count": int(scene["subassemblyCount"]),
            "image_count": len(image_paths),
            "split_balance_gap": int(scene["splitBalanceGap"]),
            "option_visibility_pass": bool(scene.get("optionVisibilityPass")),
            "option_visibility_resamples": int(scene.get("optionVisibilityRejectedScenes", 0)),
            "correct_option_type": scene["correctOptionType"],
            "option_types": [
                {
                    "letter": str(option["letter"]),
                    "optionType": str(option["optionType"]),
                    "sourceCategory": str(option["sourceCategory"]),
                    "sourceObjectName": str(option["sourceObjectName"]),
                    "optionDetail": str(option.get("optionDetail") or ""),
                }
                for option in scene["options"]
            ],
        },
        "question": pure_prompt,
        "images": image_paths,
        "gt_answer": scene["correctOptionLetter"],
    }


def build_scene_record(*, sample_index: int, sample_id: str, scene: dict, image_paths: list[str], repeat_index: int) -> dict:
    option_letters = [str(option["letter"]) for option in scene["options"]]
    return {
        "id": sample_id,
        "sample_index": sample_index,
        "repeat_index": repeat_index,
        "task": "separation_objects",
        "category": "separation",
        "level": "reasoning",
        "question_type": "subassembly_option_mcq",
        "answer_type": "multiple_choice",
        "options": option_letters,
        "metadata": {
            "object_name": scene["objectName"],
            "object_category": scene["objectCategory"],
            "part_count": scene["partCount"],
            "subassembly_count": scene["subassemblyCount"],
            "image_count": len(image_paths),
            "split_balance_gap": scene["splitBalanceGap"],
            "seed": scene["seed"],
            "option_visibility_pass": bool(scene.get("optionVisibilityPass")),
            "option_visibility_resamples": int(scene.get("optionVisibilityRejectedScenes", 0)),
            "correct_option_letter": scene["correctOptionLetter"],
            "correct_option_type": scene["correctOptionType"],
            "option_types": [
                {
                    "letter": str(option["letter"]),
                    "optionType": str(option["optionType"]),
                    "sourceCategory": str(option["sourceCategory"]),
                    "sourceObjectName": str(option["sourceObjectName"]),
                    "optionDetail": str(option.get("optionDetail") or ""),
                }
                for option in scene["options"]
            ],
        },
    }


async def generate_samples(args) -> None:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    option_mode = normalize_option_mode(args.option_mode)
    targets = select_targets(category=args.category, object_name=args.object_name, option_mode=option_mode)

    output_json = Path(args.output_json).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_root = output_json.parent
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_root)
    total_samples = len(targets) * int(args.repeats)
    seed_rng = random.Random(int(args.seed))

    server, thread, base_url = start_server(args.host, args.port)
    page_url = f"{base_url}{ENTRY_PATH}"

    question_rows: list[dict] = []
    scene_records: list[dict] = []
    skipped_requests: list[dict[str, Any]] = []
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

            sample_index = 0
            progress = tqdm(total=total_samples, desc="Separation Objects generate", unit="sample")
            try:
                for target in targets:
                    attempts_left = MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS
                    generated_for_target = 0
                    last_error = ""
                    while generated_for_target < int(args.repeats) and attempts_left > 0:
                        try:
                            scene_seed, scene, rejected_scenes, attempts_used = await generate_option_visible_scene(
                                page=page,
                                seed_rng=seed_rng,
                                category=target.category,
                                object_name=target.object_name,
                                option_mode=option_mode,
                                max_attempts=attempts_left,
                            )
                        except RuntimeError as exc:
                            last_error = str(exc)
                            attempts_left = 0
                            break
                        attempts_left -= attempts_used
                        repeat_index = generated_for_target
                        sample_id = build_sample_id(
                            category=target.category,
                            object_name=target.object_name,
                            seed=scene_seed,
                        )

                        current_sample_dir = sample_dir(
                            images_dir,
                            sample_id=sample_id,
                        )

                        image_paths: list[str] = []

                        whole_view_bytes = [data_url_to_png_bytes(view["dataUrl"]) for view in scene["wholeViews"]]
                        whole_image_bytes = compose_captioned_image(
                            image_bytes_list=whole_view_bytes,
                            image_captions=complete_object_captions(scene["wholeViews"]),
                            top_label="Complete Object",
                        )
                        whole_output_path = current_sample_dir / "complete_object.png"
                        whole_output_path.write_bytes(whole_image_bytes)
                        image_paths.append(to_relative_path(whole_output_path, base_dir=output_root))

                        for option in scene["options"]:
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
                            option_output_path = current_sample_dir / f"option_{option['letter']}.png"
                            option_output_path.write_bytes(option_image_bytes)
                            image_paths.append(to_relative_path(option_output_path, base_dir=output_root))

                        question_rows.append(
                            build_sample_record(
                                sample_index=sample_index,
                                sample_id=sample_id,
                                scene=scene,
                                image_paths=image_paths,
                                config_rel=config_rel,
                                repeat_index=repeat_index,
                            )
                        )
                        scene_records.append(
                            build_scene_record(
                                sample_index=sample_index,
                                sample_id=sample_id,
                                scene=scene,
                                image_paths=image_paths,
                                repeat_index=repeat_index,
                            )
                        )
                        sample_index += 1
                        generated_for_target += 1
                        progress.set_postfix(
                            object=f"{scene['objectCategory']}/{scene['objectName']}",
                            seed=scene_seed,
                            rejected=rejected_scenes,
                            left=attempts_left,
                        )
                        progress.update(1)
                    if generated_for_target < int(args.repeats):
                        skipped_count = int(args.repeats) - generated_for_target
                        skipped_requests.append(
                            {
                                "category": target.category,
                                "object_name": target.object_name,
                                "requested": int(args.repeats),
                                "generated": generated_for_target,
                                "skipped": skipped_count,
                                "max_attempts": MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS,
                                "reason": last_error or "visibility attempt budget exhausted",
                            }
                        )
                        progress.set_postfix(
                            object=f"{target.category}/{target.object_name}",
                            skipped=skipped_count,
                        )
                        progress.write(
                            f"[warn] {target.category}/{target.object_name}: generated "
                            f"{generated_for_target}/{int(args.repeats)} samples after "
                            f"{MAX_OPTION_VISIBILITY_RESAMPLE_ATTEMPTS} visibility attempts; "
                            f"skipping {skipped_count} requested sample(s)."
                        )
                        progress.update(skipped_count)
            finally:
                progress.close()

            await browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    write_jsonl(output_json, question_rows)
    print(f"Wrote {len(question_rows)} question rows to {output_json}")
    print(f"Saved images under {images_dir}")
    if skipped_requests:
        skipped_total = sum(int(item["skipped"]) for item in skipped_requests)
        print(
            f"Skipped {skipped_total} requested question row(s) across "
            f"{len(skipped_requests)} object(s) after the per-object visibility attempt cap."
        )
        for item in skipped_requests:
            print(
                f"  - {item['category']}/{item['object_name']}: "
                f"generated {item['generated']}/{item['requested']}, skipped {item['skipped']}"
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    import asyncio

    asyncio.run(generate_samples(args))


if __name__ == "__main__":
    main()
