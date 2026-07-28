"""InternVL3 inference runner for continuity_2d_maze.

iter2 (dev_trace_5, 2026-04-24): reads pre-generated `questions.jsonl`
(produced by `generate_samples.py` after iter E1), sends each prompt +
image to an InternVL3-compatible chat completions endpoint, parses the
model's JSON answer, and writes per-sample model_answer.jsonl + score.json
with overall / per-difficulty / per-question_type accuracy.

Differences from continuity_3d_maze/backend/internvl3_benchmark.py:
- Does NOT generate scenes; consumes pre-built questions.jsonl.
- One image per question (vs multi-view pair renders in 3d).
- Answer is a JSON list of names (vs yes/no in 3d).
- No multi-process oracle workers; single-pass loop is fast enough for
  the dataset sizes we expect (60-200 samples per run).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from internvl3_config import load_local_config, resolve_config_value

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "internvl3_benchmark.py requires the `requests` package. "
        "Install it with: pip install requests"
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "output" / "generate_samples" / "questions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "internvl3_benchmark"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_id: str
    question_type: str
    difficulty: str
    gt_answer: List[str]
    parsed_answer: Optional[List[str]]
    correct: bool
    invalid_response: bool
    api_error: bool
    api_error_message: Optional[str]
    raw_response_text: str
    image_path: str
    meta_info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"questions jsonl not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def resolve_image_path(image_rel: str, jsonl_path: Path) -> Path:
    p = Path(image_rel)
    if p.is_absolute():
        return p
    return (jsonl_path.parent / p).resolve()


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def split_prompt_at_images(prompt: str) -> Tuple[str, str]:
    """Split the prompt around the topobench-standard image marker
    `[Image 1]\\nAttached image.\\n` so the inference framework can insert
    the actual image content in place of that block. Single-image questions
    use just `[Image 1]`; the two-line marker (`[Image 1]` + `Attached
    image.`) is removed atomically. Falls back to the legacy `{images}`
    placeholder for jsonls produced before iter E2's format change."""
    marker = "[Image 1]\nAttached image.\n"
    if marker in prompt:
        before, after = prompt.split(marker, 1)
        return before.rstrip("\n"), after.lstrip("\n")
    legacy = "{images}"
    if legacy in prompt:
        before, after = prompt.split(legacy, 1)
        return before.rstrip("\n"), after.lstrip("\n")
    raise ValueError(
        "Prompt is missing the image marker '[Image 1]\\nAttached image.\\n' "
        "(or the legacy '{images}' placeholder). Re-run generate_samples.py "
        "to produce jsonl in the iter E2 standard format."
    )


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def _collect_text_fragments(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_collect_text_fragments(item))
        return out
    if isinstance(value, dict):
        out2: List[str] = []
        for key in ("text", "content", "value", "output_text", "refusal"):
            if key in value:
                out2.extend(_collect_text_fragments(value.get(key)))
        if out2:
            return out2
    return []


def normalize_message_content(content: Any) -> str:
    fragments = _collect_text_fragments(content)
    return "\n".join(fragments).strip() if fragments else ""


def extract_response_text(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    debug: Dict[str, Any] = {"response_payload": payload}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", debug, "Unexpected response payload: missing choices."
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        message = {}
    debug["finish_reason"] = first.get("finish_reason") if isinstance(first, dict) else None
    debug["message_content_raw"] = message.get("content")
    text = (
        normalize_message_content(message.get("content"))
        or normalize_message_content(first.get("text") if isinstance(first, dict) else None)
        or normalize_message_content(first.get("content") if isinstance(first, dict) else None)
    )
    return text, debug, None


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_BRACED_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# iter E3 fallback: InternVL3.5 often emits answers as `\boxed{X}` (LaTeX
# notation) instead of JSON, even when the prompt explicitly asks for JSON.
# Match each individual \boxed{...} occurrence.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _strip_markdown_fence(text: str) -> str:
    m = _MARKDOWN_FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _to_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x).strip() for x in value]
    if isinstance(value, str):
        s = value.strip()
        # Try inner JSON list first.
        try:
            inner = json.loads(s)
            if isinstance(inner, list):
                return [str(x).strip() for x in inner]
        except json.JSONDecodeError:
            pass
        # Legacy bracket-list "[A, D]" or quoted "[\"A\", \"D\"]".
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s.strip():
            return []
        tokens = [t.strip().strip('"').strip("'") for t in s.split(",")]
        return [t for t in tokens if t]
    return None


def parse_model_answer(raw_text: str) -> Optional[List[str]]:
    """Extract the answer list from a model response. Tolerates:
    - bare JSON object: `{"answer": ["A", "D"]}`
    - JSON wrapped in markdown fence: ```json\\n{"answer":...}\\n```
    - JSON object embedded in surrounding prose
    - answer value either as JSON list or as bracket-list string"""
    if not raw_text:
        return None
    text = _strip_markdown_fence(raw_text.strip())

    # Direct JSON parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return _to_string_list(obj["answer"])
    except json.JSONDecodeError:
        pass

    # Search embedded {...} objects, last one wins (model may write
    # commentary first, then the JSON answer).
    for m in reversed(_BRACED_OBJECT_RE.findall(text)):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return _to_string_list(obj["answer"])

    # Fallback: \boxed{X} scan. If the model emitted multiple \boxed{...}
    # (e.g. one per name) we union them. If a single \boxed contains a
    # bracket-list or comma-separated names, split further.
    boxed_hits = _BOXED_RE.findall(text)
    if boxed_hits:
        collected: List[str] = []
        for hit in boxed_hits:
            inner = hit.strip()
            # `\boxed{[A, D]}` or `\boxed{["A","D"]}` → split after stripping brackets.
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1]
            if "," in inner:
                parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
                collected.extend(parts)
            elif inner:
                collected.append(inner.strip('"').strip("'"))
        # Dedup while preserving order.
        seen = set()
        result: List[str] = []
        for name in collected:
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_messages(
    *,
    before: str,
    after: str,
    image_data_url: str,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": before},
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": after},
    ]
    return [{"role": "user", "content": user_content}]


def request_prediction(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    timeout: float = 180.0,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        return (
            "",
            {"http_status": response.status_code, "response_text": response.text},
            f"HTTP {response.status_code}: {response.text[:500]}",
        )
    return extract_response_text(response.json())


# ---------------------------------------------------------------------------
# Comparison & summarization
# ---------------------------------------------------------------------------

def lists_equal_as_set(a: Optional[Sequence[str]], b: Optional[Sequence[str]]) -> bool:
    if a is None or b is None:
        return False
    return frozenset(str(item).strip().casefold() for item in a) == frozenset(
        str(item).strip().casefold() for item in b
    )


def answer_correct(parsed: Optional[Sequence[str]], gt_answer: Optional[Sequence[str]]) -> bool:
    return lists_equal_as_set(parsed, gt_answer)


def summarize(results: Sequence[SampleResult]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(r.correct for r in results)
    by_diff: Dict[str, Dict[str, int]] = {}
    by_qtype: Dict[str, Dict[str, int]] = {}
    for r in results:
        for bucket, key in ((by_diff, r.difficulty), (by_qtype, r.question_type)):
            entry = bucket.setdefault(key, {"total": 0, "correct": 0, "invalid": 0, "api_errors": 0})
            entry["total"] += 1
            entry["correct"] += int(r.correct)
            entry["invalid"] += int(r.invalid_response)
            entry["api_errors"] += int(r.api_error)

    def with_acc(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in d.items():
            tot = v["total"]
            out[k] = {**v, "accuracy": v["correct"] / tot if tot else 0.0}
        return out

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_responses": sum(r.invalid_response for r in results),
        "api_errors": sum(r.api_error for r in results),
        "by_difficulty": with_acc(by_diff),
        "by_question_type": with_acc(by_qtype),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_model_answer_jsonl(
    path: Path,
    results: Sequence[SampleResult],
    model_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in results:
            row = {
                "id": r.sample_id,
                "category": ["continuity", "continuity_2d_maze", r.question_type],
                "type": "reasoning",
                "meta_info": {**r.meta_info, "model_id": model_id},
                "gt_answer": json.dumps(r.gt_answer),
                "answer": r.parsed_answer,
                "correct": r.correct,
                "invalid_response": r.invalid_response,
                "api_error": r.api_error,
                "api_error_message": r.api_error_message,
                "raw_response_text": r.raw_response_text,
                "images": [r.image_path],
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_score_json(
    path: Path,
    summary: Dict[str, Any],
    *,
    task: str,
    model_id: str,
    oracle: bool,
    questions_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "model_id": model_id,
        "oracle": oracle,
        "questions_source": str(questions_path),
        **summary,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_prompt_dump(
    dir_path: Path,
    sample_id: str,
    *,
    before: str,
    after: str,
    image_path: str,
    raw_response: str,
    parsed_answer: Optional[List[str]],
    gt_answer: List[str],
    correct: bool,
    api_error_message: Optional[str],
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    body = (
        f"[before_images]\n{before}\n\n"
        f"[image]\n{image_path}\n\n"
        f"[after_images]\n{after}\n\n"
        f"[gt_answer]\n{json.dumps(gt_answer)}\n\n"
        f"[raw_response]\n{raw_response or '<empty>'}\n\n"
        f"[parsed_answer]\n{json.dumps(parsed_answer)}\n\n"
        f"[correct]\n{correct}\n\n"
        f"[api_error]\n{api_error_message or '<none>'}\n"
    )
    (dir_path / f"{sample_id}.txt").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run InternVL3 inference on continuity_2d_maze questions.jsonl")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--config", help="Optional JSON config with api_key/base_url/model.")
    p.add_argument("--api-key")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help=(
            "Max output tokens. Default 2048 — InternVL3.5-241b-a28b emits "
            "a ~2300-token thinking preamble before the answer; a tight cap "
            "truncates during thinking and the answer never appears. "
            "Set 0 to uncap (server default, often unlimited)."
        ),
    )
    p.add_argument("--limit", type=int, default=0, help="Run only first N rows; 0 = all.")
    p.add_argument("--oracle", action="store_true", help="Use gt_answer as the model response (sanity check).")
    p.add_argument("--dump-prompts", action="store_true", help="Write per-sample prompt + raw_response under output-dir/prompts/.")
    p.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout per request, seconds.")
    return p


def resolve_api_config(args: argparse.Namespace) -> Tuple[str, str, str, Path]:
    config_path, local_config = load_local_config(args.config)
    api_key = "" if args.oracle else resolve_config_value(
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
        default="https://chat.intern-ai.org.cn/api/v1",
    )
    model = resolve_config_value(
        args.model,
        env_keys=("INTERNVL3_MODEL",),
        config=local_config,
        config_key="model",
        default="internvl3.5-latest",
    )
    if not args.oracle and not api_key:
        raise ValueError(
            f"Missing API key. Checked CLI, env vars (INTERNVL3_API_KEY, OPENAI_API_KEY), "
            f"and config file {config_path}."
        )
    return api_key, base_url, model, config_path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    api_key, base_url, model, _config_path = resolve_api_config(args)
    questions_path = Path(args.questions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts" if args.dump_prompts else None

    rows = read_questions(questions_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No questions found in {questions_path}")

    model_id = "oracle" if args.oracle else model
    results: List[SampleResult] = []

    for row in tqdm(rows, desc="2d_maze inference"):
        sample_id = row["id"]
        meta = row.get("meta_info", {}) or {}
        qtype = meta.get("question_type") or (row.get("category", []) or [None, None, None])[2] or "unknown"
        difficulty = meta.get("difficulty", "unknown")
        gt_answer = json.loads(row["gt_answer"])
        prompt = row["question"]
        image_rel = (row.get("images") or [None])[0]
        if image_rel is None:
            raise ValueError(f"Sample {sample_id} has no images.")
        image_path = resolve_image_path(image_rel, questions_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sample {sample_id} image missing: {image_path}")

        before, after = split_prompt_at_images(prompt)

        raw_text = ""
        parsed: Optional[List[str]] = None
        api_error_msg: Optional[str] = None
        api_error = False
        invalid = False

        if args.oracle:
            raw_text = json.dumps({"answer": gt_answer}, ensure_ascii=False)
            parsed = list(gt_answer)
        else:
            data_url = image_to_data_url(image_path)
            messages = build_messages(before=before, after=after, image_data_url=data_url)
            try:
                raw_text, _debug, api_error_msg = request_prediction(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            except requests.RequestException as exc:
                raw_text = ""
                api_error_msg = f"RequestException: {exc}"
            api_error = bool(api_error_msg)
            if not api_error:
                parsed = parse_model_answer(raw_text)
                invalid = parsed is None

        correct = answer_correct(parsed, gt_answer) if not api_error else False

        if prompts_dir is not None:
            write_prompt_dump(
                prompts_dir,
                sample_id,
                before=before,
                after=after,
                image_path=str(image_path),
                raw_response=raw_text,
                parsed_answer=parsed,
                gt_answer=gt_answer,
                correct=correct,
                api_error_message=api_error_msg,
            )

        results.append(
            SampleResult(
                sample_id=sample_id,
                question_type=qtype,
                difficulty=difficulty,
                gt_answer=list(gt_answer),
                parsed_answer=parsed,
                correct=correct,
                invalid_response=invalid,
                api_error=api_error,
                api_error_message=api_error_msg,
                raw_response_text=raw_text,
                image_path=image_rel,
                meta_info=meta,
            )
        )

    write_model_answer_jsonl(output_dir / "model_answer.jsonl", results, model_id)
    summary = summarize(results)
    write_score_json(
        output_dir / "score.json",
        summary,
        task="continuity_2d_maze",
        model_id=model_id,
        oracle=bool(args.oracle),
        questions_path=questions_path,
    )

    return {"results": [asdict(r) for r in results], "summary": summary, "output_dir": str(output_dir)}


def print_summary(summary: Dict[str, Any]) -> None:
    print(
        f"accuracy={summary['accuracy']:.3f}  "
        f"correct={summary['correct']}/{summary['total']}  "
        f"invalid={summary['invalid_responses']}  "
        f"api_errors={summary['api_errors']}"
    )
    for key in ("by_difficulty", "by_question_type"):
        print(f"  {key}:")
        for k, v in summary[key].items():
            print(
                f"    [{k}]  total={v['total']:3d}  "
                f"correct={v['correct']:3d}  acc={v['accuracy']:.3f}  "
                f"invalid={v['invalid']}  api_errors={v['api_errors']}"
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run(args)
    print_summary(payload["summary"])
    print(f"Wrote outputs to {payload['output_dir']}")


if __name__ == "__main__":
    main()
