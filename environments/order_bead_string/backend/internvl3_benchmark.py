"""
internvl3_benchmark.py — Bead String benchmark using InternVL3 API.

Reads pre-generated dataset (question.jsonl + images/) produced by
generate_samples.py, runs InternVL3 inference, and writes:
  - model_answer.jsonl
  - internvl3_order_bead_string.json / .csv
  - internvl3_order_bead_string_by_difficulty.json / .csv

Usage:
  python internvl3_benchmark.py --tasks all
  python internvl3_benchmark.py --tasks T_BS01_describe_sequence --limit 10
  python internvl3_benchmark.py --oracle
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from internvl3_config import DEFAULT_LOCAL_CONFIG_PATH, load_local_config, resolve_config_value
from tqdm.auto import tqdm

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. Install it with: pip install requests"
    ) from exc

from vlm_benchmark_bead import TASKS, parse_answer, score_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "internvl3_order_bead_string.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "internvl3_order_bead_string.csv"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_BASE_URL = "https://chat.intern-ai.org.cn/api/v1"
DEFAULT_MODEL = "internvl3.5-latest"
DEFAULT_REQUESTS_PER_MINUTE = 30.0

from jsonl_export import flatten_answer_value, write_jsonl


@dataclass
class SampleResult:
    sample_index: int
    sample_id: str
    seed: Any
    repeat_index: int
    task_id: str
    phrasing_index: int
    question: str
    correct: bool
    partial: bool
    predicted_answer: Optional[str]
    ground_truth: str
    invalid_response: bool
    api_error: bool
    api_error_message: Optional[str]
    raw_response_text: str
    difficulty: Optional[str]
    num_beads: int
    image_paths: List[str]
    parse_method: str = "unclear"


def _normalize_difficulty(value: Optional[str]) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"easy", "medium", "hard"}:
        return candidate
    return "hard"


# ───────────────────────────────────────────────────────────────────
# Dataset loading
# ───────────────────────────────────────────────────────────────────

def load_question_jsonl(
    path: Path,
    *,
    tasks: Optional[Sequence[str]],
    difficulty: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"question.jsonl not found at {path}. "
            "Run `python backend/generate_samples.py` first to generate the dataset."
        )
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if tasks:
        wanted = set(tasks)
        rows = [r for r in rows if r.get("type") in wanted]
    if difficulty:
        rows = [r for r in rows if (r.get("meta_info") or {}).get("difficulty") == difficulty]
    if limit:
        rows = rows[:limit]
    return rows


# ───────────────────────────────────────────────────────────────────
# Debug per-sample writers
# ───────────────────────────────────────────────────────────────────

class DebugWriter:
    """Writes prompt_debug.txt next to each sample's first image (for inspection)."""

    def __init__(self, *, benchmark_dir: Path) -> None:
        self.benchmark_dir = benchmark_dir

    def save(
        self,
        *,
        first_image_rel: str,
        task_id: str,
        sample_id: str,
        prompt: str,
        image_paths_rel: List[str],
        ground_truth: Any,
        raw_response_text: str,
        predicted_answer: Optional[Any],
        api_error_message: Optional[str],
    ) -> None:
        first_image_abs = (self.benchmark_dir / first_image_rel).resolve()
        debug_path = first_image_abs.parent / f"{task_id}_{sample_id}_prompt_debug.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            "\n\n".join(
                [
                    "[user_prompt]",
                    prompt,
                    "[images]",
                    "\n".join(image_paths_rel),
                    "[ground_truth]",
                    str(ground_truth),
                    "[model_response]",
                    raw_response_text or "<empty>",
                    "[parsed_answer]",
                    "<none>" if predicted_answer is None or predicted_answer == "unclear" else str(predicted_answer),
                    "[api_error]",
                    api_error_message or "<none>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


# ───────────────────────────────────────────────────────────────────
# model_answer.jsonl writer
# ───────────────────────────────────────────────────────────────────

def write_model_answer_jsonl(
    *,
    output_dir: Path,
    model_id: str,
    rows: Sequence[Dict[str, Any]],
    results: Sequence[SampleResult],
) -> Path:
    """Write model_answer.jsonl, mirroring the question.jsonl row structure plus prediction fields."""
    answer_rows: List[Dict[str, Any]] = []
    for row, result in zip(rows, results):
        meta_info = dict(row.get("meta_info") or {})
        meta_info["model_id"] = model_id
        answer_rows.append(
            {
                "id": row["id"],
                "category": row.get("category"),
                "type": row.get("type"),
                "meta_info": meta_info,
                "question": row.get("question"),
                "answer": flatten_answer_value(result.predicted_answer),
                "correct": bool(result.correct),
                "invalid_response": bool(result.invalid_response),
                "api_error": bool(result.api_error),
                "raw_response_text": result.raw_response_text,
            }
        )
    answer_path = output_dir / "model_answer.jsonl"
    write_jsonl(answer_path, answer_rows)
    return answer_path


# ───────────────────────────────────────────────────────────────────
# InternVL3 API helpers
# ───────────────────────────────────────────────────────────────────

def png_path_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def extract_response_text(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    if not isinstance(payload, dict):
        return "", "Unexpected response payload type."
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", "Missing choices in response."
    first = choices[0]
    if not isinstance(first, dict):
        return "", "First choice is not an object."
    message = first.get("message") or {}
    for source in [message.get("content"), first.get("text"), first.get("content")]:
        text = normalize_message_content(source)
        if text:
            return text, None
    return "", None


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
        self._min_request_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0

    def _wait_for_rate_limit_slot(self) -> None:
        if self._min_request_interval <= 0:
            return
        with self._request_gate:
            now = time.monotonic()
            sleep_seconds = max(0.0, self._next_request_not_before - now)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            self._next_request_not_before = time.monotonic() + self._min_request_interval

    def predict(
        self,
        *,
        image_urls: Sequence[str],
        prompt: str,
        max_tokens: int | None = None,
    ) -> Tuple[str, Optional[str]]:
        request_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                    + (
                        [
                            {"type": "image_url", "image_url": {"url": url}}
                            for url in image_urls
                        ]
                        if "[Image " in prompt
                        else [
                            item
                            for index, url in enumerate(image_urls, start=1)
                            for item in (
                                {"type": "text", "text": f"[Image {index}]"},
                                {"type": "image_url", "image_url": {"url": url}},
                            )
                        ]
                    ),
                },
            ],
        }
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[str] = None
        for attempt in range(self.api_retries + 1):
            try:
                self._wait_for_rate_limit_slot()
                response = self._session.post(
                    endpoint, headers=headers, json=request_payload,
                    timeout=self.request_timeout_seconds,
                )
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if response.status_code >= 500 and attempt < self.api_retries:
                        time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                        continue
                    return "", last_error
                text, err = extract_response_text(response.json())
                if err:
                    return "", err
                return text, None
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < self.api_retries:
                    time.sleep(max(self.retry_sleep_seconds, self._min_request_interval))
                    continue
                return "", last_error
        return "", last_error or "Unknown API failure."


# ───────────────────────────────────────────────────────────────────
# CSV / JSON output
# ───────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "sample_index", "sample_id", "task_id", "phrasing_index",
    "correct", "partial", "predicted_answer", "ground_truth",
    "invalid_response", "api_error", "api_error_message",
    "difficulty", "num_beads", "image_paths_json",
]


def write_csv(path: Path, results: Sequence[SampleResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k, "") for k in CSV_FIELDS if k != "image_paths_json"}
            row["image_paths_json"] = json.dumps(r.image_paths, ensure_ascii=False)
            writer.writerow(row)


def compute_summary(results: Sequence[SampleResult]) -> Dict[str, Any]:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    partial = sum(1 for r in results if r.partial)
    api_errors = sum(1 for r in results if r.api_error)
    invalid = sum(1 for r in results if r.invalid_response)
    return {
        "total_samples": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "partial": partial,
        "invalid_responses": invalid,
        "api_errors": api_errors,
    }


def _compute_by_difficulty(results: Sequence[SampleResult]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[SampleResult]] = {}
    for r in results:
        diff = _normalize_difficulty(r.difficulty)
        groups.setdefault(diff, []).append(r)
    rows: List[Dict[str, Any]] = []
    for diff in ["easy", "medium", "hard"]:
        if diff not in groups:
            continue
        s = compute_summary(groups[diff])
        rows.append({"difficulty": diff, **s})
    return rows


# ───────────────────────────────────────────────────────────────────
# Main benchmark loop
# ───────────────────────────────────────────────────────────────────

def _resolve_task_filter(arg: str) -> Optional[List[str]]:
    if arg.strip().lower() == "all":
        return None
    requested = [t.strip() for t in arg.split(",") if t.strip()]
    valid = [t for t in requested if t in TASKS]
    if not valid:
        raise ValueError(f"No valid tasks in: {requested!r}. Known: {list(TASKS)}")
    return valid


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    config_path, local_config = load_local_config(args.config)
    api_key = resolve_config_value(
        args.api_key, env_keys=("INTERNVL3_API_KEY", "INTERNVL_API_KEY", "OPENAI_API_KEY"),
        config=local_config, config_key="api_key", default="",
    )
    base_url = resolve_config_value(
        args.base_url, env_keys=("INTERNVL3_BASE_URL", "INTERNVL_BASE_URL", "OPENAI_BASE_URL"),
        config=local_config, config_key="base_url", default=DEFAULT_BASE_URL,
    )
    model = resolve_config_value(
        args.model, env_keys=("INTERNVL3_MODEL", "INTERNVL_MODEL"),
        config=local_config, config_key="model", default=DEFAULT_MODEL,
    )
    if not args.oracle and not api_key:
        message_parts = []
        if args.config and not config_path.exists():
            message_parts.append(
                f"Config file not found: {config_path}. The README string "
                "`path/to/internvl3_local_config.json` is a placeholder, not a real path."
            )
        message_parts.append(
            "Missing API key. Provide --api-key, set INTERNVL3_API_KEY/OPENAI_API_KEY, "
            f"or add api_key to {config_path}."
        )
        if not args.config and not DEFAULT_LOCAL_CONFIG_PATH.exists():
            message_parts.append(f"The default local config path is {DEFAULT_LOCAL_CONFIG_PATH}.")
        raise ValueError(" ".join(message_parts))

    question_jsonl = Path(args.question_jsonl).resolve()
    benchmark_dir = question_jsonl.parent
    task_filter = _resolve_task_filter(args.tasks)
    rows = load_question_jsonl(
        question_jsonl, tasks=task_filter, difficulty=args.difficulty, limit=args.limit,
    )
    if not rows:
        raise RuntimeError("No question rows after filtering.")
    print(f"Loaded {len(rows)} rows from {question_jsonl}")

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    policy = None if args.oracle else InternVL3Policy(
        base_url=base_url, api_key=api_key, model=model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        api_retries=args.api_retries, retry_sleep_seconds=args.retry_sleep_seconds,
        requests_per_minute=args.requests_per_minute,
        request_timeout_seconds=args.request_timeout_seconds,
    )

    debug_writer = DebugWriter(benchmark_dir=benchmark_dir)

    results: List[SampleResult] = []
    consumed_rows: List[Dict[str, Any]] = []
    image_data_url_cache: Dict[str, str] = {}

    progress = tqdm(total=len(rows), desc="Bead String benchmark")

    for sample_index, row in enumerate(rows):
        task_id = row["type"]
        prompt = row["question"]
        gt = row["gt_answer"]
        image_paths_rel: List[str] = list(row.get("images") or [])
        meta = row.get("meta_info") or {}

        image_paths_abs = [(benchmark_dir / p) for p in image_paths_rel]
        missing = [p for p, abs_p in zip(image_paths_rel, image_paths_abs) if not abs_p.exists()]
        if missing:
            progress.write(f"[skip] missing image(s): {missing}")
            progress.update(1)
            continue

        max_tokens_task = TASKS.get(task_id, {}).get("max_tokens", args.max_tokens or 300)

        if args.oracle:
            raw_text = json.dumps({"answer": gt}, ensure_ascii=False)
            parsed, p_method = gt, "oracle"
            api_err = None
        else:
            assert policy is not None
            urls: List[str] = []
            for abs_p in image_paths_abs:
                key = str(abs_p)
                if key not in image_data_url_cache:
                    image_data_url_cache[key] = png_path_to_data_url(key)
                urls.append(image_data_url_cache[key])
            raw_text, api_err = policy.predict(
                image_urls=urls, prompt=prompt, max_tokens=max_tokens_task,
            )
            if not api_err:
                parsed, p_method = parse_answer(raw_text, task_id)
            else:
                parsed, p_method = "unclear", "unclear"

        scoring = (
            score_answer(parsed, gt, task_id) if parsed != "unclear"
            else {"correct": False, "partial": False}
        )

        debug_writer.save(
            first_image_rel=image_paths_rel[0],
            task_id=task_id,
            sample_id=row["id"],
            prompt=prompt,
            image_paths_rel=image_paths_rel,
            ground_truth=gt,
            raw_response_text=raw_text,
            predicted_answer=parsed,
            api_error_message=api_err,
        )

        results.append(
            SampleResult(
                sample_index=sample_index,
                sample_id=row["id"],
                seed=meta.get("seed"),
                repeat_index=int(meta.get("repeat_index", 0)),
                task_id=task_id,
                phrasing_index=int(meta.get("phrasing_index", 0)),
                question=prompt,
                correct=scoring.get("correct", False),
                partial=scoring.get("partial", False),
                predicted_answer=parsed,
                ground_truth=str(gt),
                invalid_response=(parsed == "unclear" and api_err is None),
                api_error=api_err is not None,
                api_error_message=api_err,
                raw_response_text=raw_text,
                difficulty=meta.get("difficulty"),
                num_beads=int(meta.get("num_beads", 0)),
                image_paths=image_paths_rel,
                parse_method=p_method,
            )
        )
        consumed_rows.append(row)
        progress.update(1)
        if results:
            acc = sum(1 for r in results if r.correct) / len(results)
            progress.set_postfix(accuracy=f"{acc:.3f}", task=task_id)

    progress.close()

    # Successful tasks (those that produced ≥1 row)
    seen_tasks = []
    for r in results:
        if r.task_id not in seen_tasks:
            seen_tasks.append(r.task_id)

    summary = compute_summary(results)
    by_task: Dict[str, List[SampleResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    task_summaries = [{"task_id": tid, **compute_summary(by_task[tid])} for tid in seen_tasks]
    by_difficulty_data = _compute_by_difficulty(results)

    payload: Dict[str, Any] = {
        "task": "order_bead_string",
        "category": "order",
        "level": "reasoning",
        "oracle": bool(args.oracle),
        "question_types": seen_tasks,
        "question_jsonl": str(question_jsonl.relative_to(benchmark_dir)) if benchmark_dir == output_json.parent else str(question_jsonl),
        "summary": summary,
        "by_task": task_summaries,
        "by_difficulty": by_difficulty_data,
        "samples": [r.__dict__ for r in results],
    }

    answer_path = write_model_answer_jsonl(
        output_dir=output_json.parent,
        model_id="oracle" if args.oracle else model,
        rows=consumed_rows,
        results=results,
    )
    payload["model_answer_jsonl"] = str(answer_path.relative_to(output_json.parent))
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_csv, results)

    by_difficulty_json = output_json.with_name(f"{output_json.stem}_by_difficulty.json")
    by_difficulty_csv = output_csv.with_name(f"{output_csv.stem}_by_difficulty.csv")
    by_difficulty_json.write_text(json.dumps(by_difficulty_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if by_difficulty_data:
        with by_difficulty_csv.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(by_difficulty_data[0].keys()))
            csv_writer.writeheader()
            csv_writer.writerows(by_difficulty_data)

    print(
        f"accuracy={summary['accuracy']:.3f} | correct={summary['correct']}/{summary['total_samples']} | "
        f"invalid={summary['invalid_responses']} | api_errors={summary['api_errors']}"
    )
    print(f"Wrote JSON results to {output_json}")
    print(f"Wrote CSV results to {output_csv}")
    print(f"Wrote by-difficulty JSON to {by_difficulty_json}")
    print(f"Wrote model_answer JSONL to {answer_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="InternVL3 benchmark for order_bead_string.")
    p.add_argument("--config", default=None, help="Optional JSON config file with api_key/base_url/model.")
    p.add_argument("--api-key", default=None, help="Override API key from CLI.")
    p.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL.")
    p.add_argument("--model", default=None, help="Override model name, e.g. internvl3.5-latest.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--api-retries", type=int, default=2)
    p.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    p.add_argument("--requests-per-minute", type=float, default=DEFAULT_REQUESTS_PER_MINUTE)
    p.add_argument("--request-timeout-seconds", type=float, default=120.0)
    p.add_argument(
        "--question-jsonl",
        default=str(DEFAULT_QUESTION_JSONL),
        help="Path to question.jsonl produced by generate_samples.py.",
    )
    p.add_argument("--tasks", default="all", help="Comma-separated task IDs or 'all'.")
    p.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    p.add_argument("--limit", type=int, default=None, help="Limit total rows after filtering.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    p.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    p.add_argument("--oracle", action="store_true", help="Use ground-truth answer instead of calling the API.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
