from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from vite_server import ViteFrontendServer

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "generate_samples.py requires the `pillow` package. Install it with: pip install pillow"
    ) from exc

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "generate_samples.py requires the `playwright` package. Install it with: pip install playwright"
    ) from exc

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_DIFFICULTIES = "1,2,3"
DEFAULT_REPEATS = 1
DEFAULT_SEED = 12345
QUESTION_TYPE = "count_holes_topdown"
BOARD_SHAPES = (
    "rect",
    "circle",
    "polygon",
    # "rounded_rect",
    # "ellipse",
    # "star",
    # "cross",
    # "l_shape",
    # "irregular",
)
UNDER_BOARD_COLORS = (
    "#78d6ff",  # bright light blue
    "#ffe45c",  # bright light yellow
    "#ff8f8f",  # bright light red
)
VISIBLE_HOLE_COUNT_RANGES = {
    1: (5, 10),
    2: (7, 13),
    3: (9, 17),
}
STATE_LABEL = "Top View"
LABEL_FONT_SIZE = 40
LABEL_MARGIN_X = 18
LABEL_MARGIN_Y = 16
LABEL_FILL = "black"
LABEL_STROKE_FILL = "black"
LABEL_STROKE_WIDTH = 0
FONT_CANDIDATES: Sequence[str] = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "Arial Bold.ttf",
    "Arial.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
)

from jsonl_export import compose_pure_prompt, to_relative_path, write_jsonl  # noqa: E402


def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reasoning samples for enclosure_hole_detection.")
    parser.add_argument(
        "--repeats",
        "--count",
        dest="repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of seeds to generate for each requested difficulty.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed used to derive a deterministic per-sample seed sequence.")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTIES, help="Comma-separated difficulty buckets (1-3).")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        value = int(part)
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"Empty integer list spec: {spec!r}")
    return values


def board_shape_for_seed(seed: int) -> str:
    return BOARD_SHAPES[abs(int(seed)) % len(BOARD_SHAPES)]


def under_board_color_for_seed(seed: int) -> str:
    return UNDER_BOARD_COLORS[abs(int(seed)) % len(UNDER_BOARD_COLORS)]


def stable_u32(seed: int, salt: int) -> int:
    value = (int(seed) ^ int(salt)) & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (value ^ (value >> 16)) & 0xFFFFFFFF


def under_board_enabled_for_seed(seed: int) -> bool:
    return stable_u32(seed, 701) % 2 == 0


def deterministic_int(seed: int, salt: int, low: int, high: int) -> int:
    if high < low:
        return low
    return low + stable_u32(seed, salt) % (high - low + 1)


def deterministic_shuffle(values: Sequence[str], seed: int, salt: int) -> List[str]:
    decorated = [
        (stable_u32(seed, salt + index), index, value)
        for index, value in enumerate(values)
    ]
    decorated.sort()
    return [value for _, _, value in decorated]


def visible_opening_count_for_plan(type_plan: Sequence[str]) -> int:
    return sum(2 if hole_type == "mixed2" else 1 if hole_type in {"hole", "mixed"} else 0 for hole_type in type_plan)


def visible_hole_count_range_for_difficulty(difficulty: int) -> tuple[int, int]:
    return VISIBLE_HOLE_COUNT_RANGES.get(int(difficulty), VISIBLE_HOLE_COUNT_RANGES[3])


def type_plan_for_difficulty(seed: int, difficulty: int) -> List[str]:
    min_visible, max_visible = visible_hole_count_range_for_difficulty(difficulty)
    if difficulty == 1:
        visible_count = deterministic_int(seed, 101, min_visible, max_visible)
        pit_count = deterministic_int(seed, 201, 1, 3)
        return deterministic_shuffle(["hole"] * visible_count + ["pit"] * pit_count, seed, 301)

    if difficulty == 2:
        visible_count = deterministic_int(seed, 102, min_visible, max_visible)
        pit_count = deterministic_int(seed, 202, 1, 3)
        mixed_count = deterministic_int(seed, 402, 1, min(3, visible_count))
        hole_count = visible_count - mixed_count
        return deterministic_shuffle(["hole"] * hole_count + ["mixed"] * mixed_count + ["pit"] * pit_count, seed, 302)

    visible_count = deterministic_int(seed, 103, min_visible, max_visible)
    plan = ["hole", "pit", "mixed", "mixed2"]
    remaining = visible_count - visible_opening_count_for_plan(plan)
    fill_index = 0
    while remaining > 0:
        prefer_mixed2 = remaining >= 2 and stable_u32(seed, 403 + fill_index) % 3 == 0
        if prefer_mixed2:
            plan.append("mixed2")
            remaining -= 2
        else:
            plan.append("mixed" if stable_u32(seed, 503 + fill_index) % 2 else "hole")
            remaining -= 1
        fill_index += 1
    return deterministic_shuffle(plan, seed, 603)


def difficulty_config(seed: int, difficulty: int) -> Dict[str, Any]:
    board_shape = board_shape_for_seed(seed)
    type_plan = type_plan_for_difficulty(seed, difficulty)
    hole_types = list(dict.fromkeys(type_plan))
    config: Dict[str, Any] = {"seed": seed, "difficulty": difficulty}
    common_geometry = {
        "board": {"shape": board_shape, "size": {"width": 10, "height": 10}, "thickness": 0.5},
        "holes": {
            "count": {"min": len(type_plan), "max": len(type_plan)},
            "typePlan": type_plan,
            "sizeRange": {"min": 0.35, "max": 0.9},
            "minDistance": 0.45,
            "edgeBuffer": 0.85,
            "shapeVariant": "vertical",
        },
    }
    if difficulty == 1:
        config.update(
            {
                "geometry": {
                    **common_geometry,
                    "holes": {
                        **common_geometry["holes"],
                        "types": hole_types,
                    },
                }
            }
        )
    elif difficulty == 2:
        config.update(
            {
                "geometry": {
                    **common_geometry,
                    "holes": {
                        **common_geometry["holes"],
                        "types": hole_types,
                    },
                }
            }
        )
    else:
        config.update(
            {
                "geometry": {
                    **common_geometry,
                    "holes": {
                        **common_geometry["holes"],
                        "types": hole_types,
                    },
                }
            }
        )
    return config


def scene_config(seed: int, difficulty: int) -> Dict[str, Any]:
    config = difficulty_config(seed, difficulty)
    config["geometry"]["underBoard"] = {
        "enable": under_board_enabled_for_seed(seed),
        "offset": {
            "x": 0.2,
            "y": 0.1,
        },
        "coverage": "partial",
    }
    config["render"] = {
        "camera": {
            "pitch": 89,
            "yaw": 45,
        },
        "appearance": {
            "underBoardColor": under_board_color_for_seed(seed),
        },
    }
    return config


def build_question_and_answer(metadata: Dict[str, Any]) -> tuple[str, int, str]:
    return (
        "How many holes are visible in this top-down view of the board?",
        int(metadata["visible_opening_count"]),
        "integer",
    )


def build_user_prompt_blocks(question: str, answer_type: str, has_under_board: bool = False) -> tuple[str, str]:
    if answer_type == "integer":
        answer_format = (
            'Output exactly one JSON object: {"answer":{ans}} and nothing else.\n'
            "Replace {ans} with the single legal answer for this task, chosen from the valid integers for this task.\n"
        )
    else:
        answer_format = (
            'Output exactly one JSON object: {"answer":"{ans}"} and nothing else.\n'
            "Replace {ans} with the single legal answer for this task, chosen from the legal answers for this task.\n"
        )
    under_board_note = (
        "If another board is visible beneath it, evaluate only the top board and count holes on the top board only.\n"
        if has_under_board
        else ""
    )
    before_images = (
        "[Task]\n"
        "You are solving a topological task called Hole Detection under the enclosure category.\n"
        "In this task, you must determine how many holes are visible in a top-down image of a board. "
        "Count only openings that pass through the board to open space.\n"
        "A hole is an opening that connects through the board to open space. "
        "A pit or shallow depression that does not pass through the board is not a hole.\n"
        f"{under_board_note}"
        "The visual evidence for this question is provided below."
    )
    after_images = (
        "[Rules]\n"
        "1. Use only the images and text provided in this prompt.\n"
        "2. If answer options are provided, choose only from the provided options.\n"
        "3. Do not output explanation beyond the required final answer.\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Answer Format]\n"
        f"{answer_format}"
    )
    return before_images, after_images


def difficulty_label_for_bucket(difficulty: int) -> str:
    value = int(difficulty)
    if value == 1:
        return "easy"
    if value == 2:
        return "medium"
    if value == 3:
        return "hard"
    raise ValueError(f"Unsupported enclosure_hole_detection difficulty bucket: {difficulty}")


def build_sample_id(*, difficulty: int, seed: int) -> str:
    return f"enclosure_hole_detection_difficulty_{difficulty}_seed_{seed}"


def load_caption_font(size: int):
    for font_name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_state_image(image_bytes: bytes, label: str = STATE_LABEL) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = load_caption_font(LABEL_FONT_SIZE)
        draw.text(
            (LABEL_MARGIN_X, LABEL_MARGIN_Y),
            label,
            fill=LABEL_FILL,
            font=font,
            stroke_width=LABEL_STROKE_WIDTH,
            stroke_fill=LABEL_STROKE_FILL,
        )
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


async def generate_scene(page, config: Dict[str, Any]) -> tuple[Dict[str, Any], bytes]:
    metadata = await page.evaluate(
        """async (config) => {
            return await window.topoBench.generate(config);
        }""",
        config,
    )
    image_data_url = await page.evaluate("() => window.topoBench.screenshot()")
    if not isinstance(image_data_url, str) or "," not in image_data_url:
        raise RuntimeError("Frontend screenshot() did not return a valid data URL.")
    png_bytes = base64.b64decode(image_data_url.split(",", 1)[1])
    return metadata, label_state_image(png_bytes)


async def generate_samples_async(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    output_json = Path(args.output_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    config_rel = to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_json.parent)

    difficulties = [d for d in parse_int_list(args.difficulty) if 1 <= d <= 3]
    if not difficulties:
        raise ValueError("At least one difficulty in the range 1-3 is required.")
    seed_values = deterministic_seed_sequence(int(args.seed), len(difficulties) * int(args.repeats))

    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=args.headless)
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            await page.goto(server.base_url, wait_until="networkidle")
            await page.wait_for_function("() => !!window.topoBench && typeof window.topoBench.generate === 'function'")

            question_rows: List[Dict[str, Any]] = []
            scene_records: List[Dict[str, Any]] = []
            sample_index = 0
            progress = tqdm(total=len(difficulties) * int(args.repeats), desc="Hole Detection generate", unit="sample")

            try:
                for difficulty in difficulties:
                    for repeat_index in range(args.repeats):
                        seed = int(seed_values[sample_index])
                        sample_id = build_sample_id(difficulty=difficulty, seed=seed)
                        config = scene_config(seed, difficulty)
                        metadata, png_bytes = await generate_scene(page, config)
                        sample_dir = image_root / sample_id
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        image_path = sample_dir / "state_0.png"
                        image_path.write_bytes(png_bytes)
                        relative_image = to_relative_path(image_path, base_dir=output_json.parent)
                        question, answer, answer_type = build_question_and_answer(metadata)
                        before_images, after_images = build_user_prompt_blocks(
                            question,
                            answer_type,
                            has_under_board=bool(metadata.get("under_board_enabled", False)),
                        )
                        pure_prompt = compose_pure_prompt(before_images, [relative_image], after_images)

                        question_rows.append(
                            {
                                "id": sample_id,
                                "category": ["enclosure", "enclosure_hole_detection", QUESTION_TYPE],
                                "type": "reasoning",
                                "meta_info": {
                                    "task_name": "enclosure_hole_detection",
                                    "config": config_rel,
                                    "seed": seed,
                                    "repeat_index": repeat_index,
                                    "difficulty": difficulty_label_for_bucket(difficulty),
                                    "board_shape": metadata.get("board_shape", config["geometry"]["board"]["shape"]),
                                },
                                "question": pure_prompt,
                                "images": [relative_image],
                                "gt_answer": answer,
                            }
                        )
                        scene_records.append(
                            {
                                "scene_index": sample_index,
                                "repeat_index": repeat_index,
                                "seed": seed,
                                "difficulty": difficulty,
                                "answer_type": answer_type,
                                "images": [relative_image],
                                "metadata": metadata,
                            },
                        )
                        sample_index += 1
                        progress.update(1)
            finally:
                progress.close()

            await browser.close()

    payload = {
        "task": "enclosure_hole_detection",
        "category": "enclosure",
        "level": "reasoning",
        "question_type": QUESTION_TYPE,
        "requested_difficulties": difficulties,
        "repeats_per_difficulty": args.repeats,
        "scene_count": len(scene_records),
        "sample_count": len(question_rows),
        "question_jsonl": str(output_json),
        "scenes": scene_records,
    }
    write_jsonl(output_json, question_rows)
    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = asyncio.run(generate_samples_async(args))
    print(f"Wrote {payload['sample_count']} question rows to {Path(args.output_json).resolve()}")
    print(f"Saved images under {Path(args.image_root).resolve()}")


if __name__ == "__main__":
    main()
