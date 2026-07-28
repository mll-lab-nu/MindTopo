from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from generate_samples import (
    build_question_and_answer,
    difficulty_label_for_bucket,
    label_state_image,
    parse_int_list,
    scene_config,
)
from internvl3_config import load_local_config, resolve_config_value
from tqdm.auto import tqdm
from vite_server import ViteFrontendServer

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
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_enclosure_hole_detection.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_enclosure_hole_detection.csv"
DEFAULT_IMAGE_ROOT = DEFAULT_OUTPUT_ROOT / "images"
DEFAULT_DIFFICULTIES = "1,2,3"
DEFAULT_REPEATS = 1
DEFAULT_SEED = 12345
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
QUESTION_TYPE = "count_holes_topdown"

from jsonl_export import compose_pure_prompt, flatten_answer_value, is_meta_prompt, split_pure_prompt, to_relative_path, write_jsonl


def deterministic_seed_sequence(base_seed: int, count: int) -> List[int]:
    rng = random.Random(int(base_seed))
    return [rng.randrange(0, 2**31 - 1) for _ in range(max(0, int(count)))]


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


def image_file_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass
class SampleResult:
    sample_index: int
    sample_id: str
    question: str
    repeat_index: int
    seed: int
    difficulty: int
    difficulty_label: str
    board_shape: Optional[str]
    question_type: str
    answer_type: str
    ground_truth: Any
    predicted_answer: Optional[Any]
    correct: bool
    invalid_response: bool
    api_error: bool
    api_error_message: Optional[str]
    raw_response_text: str
    image_path: str


def difficulty_bucket_from_label(value: Any) -> int:
    if isinstance(value, int):
        return value
    label = str(value or "").strip().casefold()
    if label in {"1", "easy"}:
        return 1
    if label in {"2", "medium"}:
        return 2
    if label in {"3", "hard"}:
        return 3
    return 1


def build_sample_id(*, difficulty: int, seed: int) -> str:
    return f"enclosure_hole_detection_difficulty_{difficulty}_seed_{seed}"


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
        meta_info = {
            "task_name": "enclosure_hole_detection",
            "config": config_rel,
            "seed": result.seed,
            "repeat_index": result.repeat_index,
            "difficulty": result.difficulty_label,
        }
        if result.board_shape:
            meta_info["board_shape"] = result.board_shape
        base_row = {
            "id": result.sample_id,
            "category": ["enclosure", "enclosure_hole_detection", result.question_type],
            "type": "reasoning",
            "meta_info": meta_info,
            "question": result.question,
            "images": [result.image_path],
            "gt_answer": result.ground_truth,
        }
        question_rows.append(base_row)
        answer_rows.append(
            {
                "id": result.sample_id,
                "category": ["enclosure", "enclosure_hole_detection", result.question_type],
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


class ImageWriter:
    def __init__(self, *, root_dir: Path, output_json_path: Path) -> None:
        self.root_dir = root_dir
        self.output_json_path = output_json_path
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def sample_dir(self, *, sample_id: str) -> Path:
        directory = self.root_dir / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_png(
        self,
        *,
        sample_id: str,
        image_name: str,
        image_bytes: bytes,
    ) -> str:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / image_name
        output_path.write_bytes(image_bytes)
        return to_relative_path(output_path, base_dir=self.output_json_path.parent)

    def save_prompt_debug(
        self,
        *,
        sample_id: str,
        user_prompt: str,
        image_paths: List[str],
        ground_truth: Any,
        raw_response_text: str,
        predicted_answer: Optional[Any],
        api_error_message: Optional[str],
        response_debug: Optional[Dict[str, Any]] = None,
    ) -> Path:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / "prompt_debug.txt"
        output_path.write_text(
            "\n\n".join(
                [
                    "[user_prompt]",
                    user_prompt,
                    "[images]",
                    "\n".join(image_paths),
                    "[ground_truth]",
                    str(ground_truth),
                    "[model_response]",
                    raw_response_text or "<empty>",
                    "[parsed_answer]",
                    "<none>" if predicted_answer is None else str(predicted_answer),
                    "[api_error]",
                    api_error_message or "<none>",
                    "[response_debug]",
                    json.dumps(response_debug, indent=2, ensure_ascii=False) if response_debug else "<none>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    def save_pure_prompt(
        self,
        *,
        sample_id: str,
        user_prompt_before_images: str,
        image_paths: List[str],
        user_prompt_after_images: str,
    ) -> Path:
        sample_dir = self.sample_dir(sample_id=sample_id)
        output_path = sample_dir / "pure_prompt.txt"
        pure_prompt = compose_pure_prompt(user_prompt_before_images, image_paths, user_prompt_after_images)
        output_path.write_text(pure_prompt + "\n", encoding="utf-8")
        return output_path


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


def image_bytes_to_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_integer_value(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
    return None


def parse_answer(text: str, answer_type: str) -> Optional[Any]:
    if not text:
        return None
    if answer_type == "integer":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "answer" in payload:
            parsed = _parse_integer_value(payload["answer"])
            if parsed is not None:
                return parsed
        if payload is not None and not isinstance(payload, dict):
            parsed = _parse_integer_value(payload)
            if parsed is not None:
                return parsed
        brace_matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        for brace_text in reversed(brace_matches):
            brace_numbers = list(re.finditer(r"-?\d+", brace_text))
            if brace_numbers:
                return int(brace_numbers[-1].group(0))
        number_matches = list(re.finditer(r"-?\d+", text))
        if number_matches:
            return int(number_matches[-1].group(0))
        return None
    if answer_type == "yes_no":
        lowered = text.casefold()
        try:
            payload = json.loads(lowered)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "answer" in payload and isinstance(payload["answer"], str):
            answer = payload["answer"].strip().casefold()
            if answer in {"yes", "no"}:
                return answer
        brace_matches = re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL)
        for brace_text in reversed(brace_matches):
            brace_answer_match = re.search(r"\b(yes|no)\b", brace_text)
            if brace_answer_match:
                return brace_answer_match.group(1)
        answer_matches = list(re.finditer(r"\b(yes|no)\b", lowered))
        if answer_matches:
            return answer_matches[-1].group(1)
        return None
    return text.strip() or None


def summarize_results(results: Sequence[SampleResult], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[SampleResult]] = {}
    for result in results:
        key = tuple(getattr(result, field) for field in group_fields)
        groups.setdefault(key, []).append(result)

    rows: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        total = len(items)
        valid_items = [item for item in items if not item.api_error]
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                "repeats": total,
                "correct": sum(1 for item in items if item.correct),
                "accuracy": sum(1 for item in items if item.correct) / total if total else 0.0,
                "valid_repeats": len(valid_items),
                "valid_accuracy": (
                    sum(1 for item in valid_items if item.correct) / len(valid_items) if valid_items else 0.0
                ),
                "invalid_responses": sum(item.invalid_response for item in items),
                "api_errors": sum(item.api_error for item in items),
            }
        )
        rows.append(row)
    return rows


async def render_scene(page, config: Dict[str, Any]) -> tuple[Dict[str, Any], bytes]:
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


def build_user_prompt_blocks(question: str, answer_type: str, has_under_board: bool = False) -> Tuple[str, str]:
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


def request_prediction(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_prompt_before_images: str,
    user_prompt_after_images: str,
    image_data_url: str,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_before_images},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_prompt_after_images},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        return (
            "",
            {"http_status": response.status_code, "response_text": response.text},
            f"HTTP {response.status_code}: {response.text}",
        )
    response_payload = response.json()
    return extract_response_text(response_payload)


def run_loaded_question_rows(
    *,
    args: argparse.Namespace,
    question_jsonl: Path,
    output_json: Path,
    writer: ImageWriter,
    api_key: str,
    base_url: str,
    model: str,
) -> List[SampleResult]:
    rows = read_question_jsonl(question_jsonl)
    if not rows:
        raise ValueError(f"Question JSONL has no data rows: {question_jsonl}")
    results: List[SampleResult] = []
    progress = tqdm(total=len(rows), desc="Hole Detection loaded benchmark")

    for sample_index, row in enumerate(rows):
        meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} must contain at least one image path.")
        image_path = resolve_question_image_path(str(images[0]), question_jsonl=question_jsonl)
        image_relative_path = to_relative_path(image_path, base_dir=output_json.parent)
        image_data_url = image_file_to_data_url(image_path)

        row_question = str(row.get("question") or "")
        if not row_question:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} is missing question text.")
        if "gt_answer" not in row:
            raise ValueError(f"Question row {row.get('id', sample_index)!r} is missing gt_answer.")
        answer = row.get("gt_answer")
        answer_type = "integer" if _parse_integer_value(answer) is not None else "yes_no"
        if answer_type == "integer":
            answer = _parse_integer_value(answer)
        loaded_difficulty = meta_info.get("difficulty")
        difficulty = difficulty_bucket_from_label(loaded_difficulty)
        difficulty_label = str(loaded_difficulty if loaded_difficulty is not None else difficulty_label_for_bucket(difficulty))
        board_shape = meta_info.get("board_shape")
        sample_id = str(row.get("id") or build_sample_id(
            difficulty=difficulty,
            seed=int(meta_info.get("seed", sample_index)),
        ))

        if is_meta_prompt(row_question):
            user_prompt_before_images, user_prompt_after_images = split_pure_prompt(row_question, image_count=1)
            question = row_question
        else:
            user_prompt_before_images, user_prompt_after_images = build_user_prompt_blocks(
                row_question,
                answer_type,
                has_under_board=bool(meta_info.get("under_board_enabled", False)),
            )
            question = compose_pure_prompt(user_prompt_before_images, [image_relative_path], user_prompt_after_images)
        user_prompt_debug = f"{user_prompt_before_images}\n\n[Images inserted here]\n\n{user_prompt_after_images}"
        writer.save_pure_prompt(
            sample_id=sample_id,
            user_prompt_before_images=user_prompt_before_images,
            image_paths=[image_relative_path],
            user_prompt_after_images=user_prompt_after_images,
        )
        raw_response_text = ""
        parsed_answer: Optional[Any] = None
        invalid_response = False
        api_error = False
        api_error_message: Optional[str] = None
        response_debug: Dict[str, Any] | None = None

        if args.oracle:
            raw_response_text = json.dumps({"answer": answer}, ensure_ascii=False)
            parsed_answer = answer
        else:
            try:
                raw_response_text, response_debug, api_error_message = request_prediction(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    user_prompt_before_images=user_prompt_before_images,
                    user_prompt_after_images=user_prompt_after_images,
                    image_data_url=image_data_url,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                api_error_message = f"{type(exc).__name__}: {exc}"
                response_debug = {"exception": type(exc).__name__}
            api_error = api_error_message is not None
            parsed_answer = parse_answer(raw_response_text, answer_type) if not api_error else None
            invalid_response = not api_error and parsed_answer is None

        correct = parsed_answer == answer
        writer.save_prompt_debug(
            sample_id=sample_id,
            user_prompt=user_prompt_debug,
            image_paths=[image_relative_path],
            ground_truth=answer,
            raw_response_text=raw_response_text,
            predicted_answer=parsed_answer,
            api_error_message=api_error_message,
            response_debug=response_debug,
        )
        category = row.get("category")
        loaded_question_type = (
            str(category[2])
            if isinstance(category, list) and len(category) >= 3 and str(category[2]) not in {"reasoning", "interactive"}
            else str(row.get("type") or QUESTION_TYPE)
        )
        if loaded_question_type in {"reasoning", "interactive"}:
            loaded_question_type = QUESTION_TYPE

        results.append(
            SampleResult(
                sample_index=sample_index,
                sample_id=sample_id,
                question=question,
                repeat_index=int(meta_info.get("repeat_index", sample_index)),
                seed=int(meta_info.get("seed", sample_index)),
                difficulty=difficulty,
                difficulty_label=difficulty_label,
                board_shape=None if board_shape is None else str(board_shape),
                question_type=loaded_question_type,
                answer_type=answer_type,
                ground_truth=answer,
                predicted_answer=parsed_answer,
                correct=correct,
                invalid_response=invalid_response,
                api_error=api_error,
                api_error_message=api_error_message,
                raw_response_text=raw_response_text,
                image_path=image_relative_path,
            )
        )
        progress.update(1)
        progress.set_postfix(
            accuracy=f"{sum(r.correct for r in results) / len(results):.3f}",
            invalid=sum(r.invalid_response for r in results),
            api_errors=sum(r.api_error for r in results),
        )

    progress.close()
    return results


async def run_benchmark_async(args: argparse.Namespace) -> Dict[str, Any]:
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    image_root = Path(args.image_root).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    question_jsonl = Path(args.question_jsonl).resolve() if args.question_jsonl else None
    difficulties = []
    if question_jsonl is None:
        difficulties = [d for d in parse_int_list(args.difficulty) if 1 <= d <= 3]
        if not difficulties:
            raise ValueError("At least one difficulty in the range 1-3 is required.")

    config_path, local_config = load_local_config(args.config)
    api_key = "" if args.oracle else resolve_config_value(
        args.api_key,
        env_keys=("INTERNVL3_API_KEY", "INTERNVL_API_KEY", "OPENAI_API_KEY"),
        config=local_config,
        config_key="api_key",
        default="",
    )
    base_url = resolve_config_value(
        args.base_url,
        env_keys=("INTERNVL3_BASE_URL", "INTERNVL_BASE_URL", "OPENAI_BASE_URL"),
        config=local_config,
        config_key="base_url",
        default=DEFAULT_BASE_URL,
    )
    model = resolve_config_value(
        args.model,
        env_keys=("INTERNVL3_MODEL", "INTERNVL_MODEL"),
        config=local_config,
        config_key="model",
        default=DEFAULT_MODEL,
    )
    if not args.oracle and not api_key:
        raise ValueError(f"Missing API key. Checked CLI, env vars, and config file {config_path}.")

    writer = ImageWriter(root_dir=image_root, output_json_path=output_json)
    results: List[SampleResult] = []
    sample_index = 0

    if question_jsonl is not None:
        results = run_loaded_question_rows(
            args=args,
            question_jsonl=question_jsonl,
            output_json=output_json,
            writer=writer,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    else:
        with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=args.headless)
                page = await browser.new_page(viewport={"width": 1400, "height": 900})
                await page.goto(server.base_url, wait_until="networkidle")
                await page.wait_for_function("() => !!window.topoBench && typeof window.topoBench.generate === 'function'")

                total_steps = args.repeats * len(difficulties)
                seed_values = deterministic_seed_sequence(int(args.seed), total_steps)
                progress = tqdm(total=total_steps, desc="Hole Detection benchmark")

                for difficulty in difficulties:
                    for repeat_index in range(args.repeats):
                        seed = int(seed_values[sample_index])
                        sample_id = build_sample_id(difficulty=difficulty, seed=seed)
                        config = scene_config(seed, difficulty)
                        metadata, png_bytes = await render_scene(page, config)
                        image_relative_path = writer.save_png(
                            sample_id=sample_id,
                            image_name="state_0.png",
                            image_bytes=png_bytes,
                        )
                        image_data_url = image_bytes_to_data_url(png_bytes)
                        question, answer, answer_type = build_question_and_answer(metadata)
                        board_shape = str(metadata.get("board_shape", config["geometry"]["board"]["shape"]))
                        user_prompt_before_images, user_prompt_after_images = build_user_prompt_blocks(
                            question,
                            answer_type,
                            has_under_board=bool(metadata.get("under_board_enabled", False)),
                        )
                        user_prompt_debug = (
                            f"{user_prompt_before_images}\n\n[Images inserted here]\n\n{user_prompt_after_images}"
                        )
                        pure_prompt = compose_pure_prompt(
                            user_prompt_before_images,
                            [image_relative_path],
                            user_prompt_after_images,
                        )
                        writer.save_pure_prompt(
                            sample_id=sample_id,
                            user_prompt_before_images=user_prompt_before_images,
                            image_paths=[image_relative_path],
                            user_prompt_after_images=user_prompt_after_images,
                        )
                        raw_response_text = ""
                        parsed_answer: Optional[Any] = None
                        invalid_response = False
                        api_error = False
                        api_error_message: Optional[str] = None
                        response_debug: Dict[str, Any] | None = None

                        if args.oracle:
                            raw_response_text = json.dumps({"answer": answer}, ensure_ascii=False)
                            parsed_answer = answer
                        else:
                            try:
                                raw_response_text, response_debug, api_error_message = request_prediction(
                                    api_key=api_key,
                                    base_url=base_url,
                                    model=model,
                                    user_prompt_before_images=user_prompt_before_images,
                                    user_prompt_after_images=user_prompt_after_images,
                                    image_data_url=image_data_url,
                                    max_tokens=args.max_tokens,
                                )
                            except Exception as exc:
                                api_error_message = f"{type(exc).__name__}: {exc}"
                                response_debug = {"exception": type(exc).__name__}
                            api_error = api_error_message is not None
                            parsed_answer = parse_answer(raw_response_text, answer_type) if not api_error else None
                            invalid_response = not api_error and parsed_answer is None

                        correct = parsed_answer == answer
                        writer.save_prompt_debug(
                            sample_id=sample_id,
                            user_prompt=user_prompt_debug,
                            image_paths=[image_relative_path],
                            ground_truth=answer,
                            raw_response_text=raw_response_text,
                            predicted_answer=parsed_answer,
                            api_error_message=api_error_message,
                            response_debug=response_debug,
                        )
                        results.append(
                            SampleResult(
                                sample_index=sample_index,
                                sample_id=sample_id,
                                question=pure_prompt,
                                repeat_index=repeat_index,
                                seed=seed,
                                difficulty=difficulty,
                                difficulty_label=difficulty_label_for_bucket(difficulty),
                                board_shape=board_shape,
                                question_type=QUESTION_TYPE,
                                answer_type=answer_type,
                                ground_truth=answer,
                                predicted_answer=parsed_answer,
                                correct=correct,
                                invalid_response=invalid_response,
                                api_error=api_error,
                                api_error_message=api_error_message,
                                raw_response_text=raw_response_text,
                                image_path=image_relative_path,
                            )
                        )
                        sample_index += 1
                        progress.update(1)
                        progress.set_postfix(
                            accuracy=f"{sum(r.correct for r in results) / len(results):.3f}",
                            invalid=sum(r.invalid_response for r in results),
                            api_errors=sum(r.api_error for r in results),
                        )

                progress.close()
                await browser.close()

    by_difficulty_rows = summarize_results(results, ("difficulty",))
    payload = {
        "task": "enclosure_hole_detection",
        "category": "enclosure",
        "level": "reasoning",
        "oracle": bool(args.oracle),
        "load_question_jsonl": str(question_jsonl) if question_jsonl is not None else None,
        "question_type": QUESTION_TYPE,
        "requested_difficulties": difficulties if question_jsonl is None else sorted({r.difficulty for r in results}),
        "repeats_per_difficulty": args.repeats if question_jsonl is None else None,
        "summary": {
            "total_samples": len(results),
            "correct": sum(item.correct for item in results),
            "accuracy": (sum(item.correct for item in results) / len(results)) if results else 0.0,
            "invalid_responses": sum(item.invalid_response for item in results),
            "api_errors": sum(item.api_error for item in results),
        },
        "by_difficulty": by_difficulty_rows,
        "samples": [result.__dict__ for result in results],
    }
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()) if results else ["sample_index"])
        writer_csv.writeheader()
        for result in results:
            writer_csv.writerow(result.__dict__)

    by_difficulty_json = output_json.with_name(f"{output_json.stem}_by_difficulty.json")
    by_difficulty_csv = output_csv.with_name(f"{output_csv.stem}_by_difficulty.csv")
    by_difficulty_json.write_text(json.dumps(by_difficulty_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if by_difficulty_rows:
        with by_difficulty_csv.open("w", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(by_difficulty_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(by_difficulty_rows)

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

    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark InternVL3 on enclosure_hole_detection.")
    parser.add_argument("--config", help="Optional JSON config file with api_key/base_url/model.")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTIES, help="Comma-separated difficulty buckets.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--question-jsonl", default="", help="Load existing question.jsonl rows and images instead of generating new scenes.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    payload = asyncio.run(run_benchmark_async(args))
    print(
        f"accuracy={payload['summary']['accuracy']:.3f} "
        f"correct={payload['summary']['correct']}/{payload['summary']['total_samples']}"
    )
    print(f"Wrote JSON summary to {Path(args.output_json).resolve()}")
    print(f"Wrote CSV summary to {Path(args.output_csv).resolve()}")
    if payload.get("load_question_jsonl"):
        print(f"Loaded question JSONL from {payload['load_question_jsonl']}")
    else:
        print(f"Wrote question JSONL to {(Path(args.output_json).resolve().parent / 'question.jsonl')}")
    print(f"Wrote model-answer JSONL to {(Path(args.output_json).resolve().parent / 'model_answer.jsonl')}")


if __name__ == "__main__":
    main()
