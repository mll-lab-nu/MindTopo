"""Minimal benchmark runner for continuity_2d_maze (stub strategies).

Reads `question.jsonl` (meta_jsonl style) and produces:
  - `model_answer.jsonl`: one line per question, meta_jsonl shape
  - `score.json`: overall accuracy plus per-difficulty accuracy

No network calls. Strategies are stubs used to validate the pipeline end to end.
Real-model clients (iter 7+) can drop in by replacing `run_strategy` while
reusing `write_model_answer_jsonl` and `summarize_accuracy`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts import (
    build_bar_removal_prompt,
    build_reachability_set_prompt,
)


TASK_NAME = "continuity_2d_maze"
DEFAULT_QUESTION_TYPE = "reachability_set"


def infer_question_type(rows: List[Dict[str, Any]]) -> str:
    """Pick the question_type from the first row's `type` field. All rows in a
    benchmark run are assumed to share the same type (CLI generates one at a
    time). Falls back to the legacy default if the field is missing."""
    for row in rows:
        t = row.get("type")
        if t:
            return str(t)
    return DEFAULT_QUESTION_TYPE


def normalize_answer_for_type(qtype: str, raw: Optional[str]) -> Optional[str]:
    """Parse a raw model response into the canonical answer string. Both
    active question types (reachability_set, bar_removal) return a
    bracket-list; extract the JSON `answer` field if present, else return
    the stripped raw text and let `parse_name_list` normalize it into a
    set for comparison."""
    del qtype  # signature kept for future per-type parsing; currently uniform
    if not raw:
        return None
    text = raw.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "answer" in payload:
            v = payload["answer"]
            if isinstance(v, str):
                return v.strip()
    except json.JSONDecodeError:
        pass
    return text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "generate_samples"
DEFAULT_QUESTIONS = DEFAULT_OUTPUT_ROOT / "questions.jsonl"
DEFAULT_MODEL_ANSWER = DEFAULT_OUTPUT_ROOT / "model_answer.jsonl"
DEFAULT_SCORE = DEFAULT_OUTPUT_ROOT / "score.json"

STRATEGIES = ("oracle", "bar_random", "bar_empty")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stub benchmark runner for continuity_2d_maze.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_ANSWER))
    parser.add_argument("--score", default=str(DEFAULT_SCORE))
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="oracle",
        help="Stub answering strategy. 'oracle' copies gt_answer (expect accuracy=1.0).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --strategy random.")
    parser.add_argument(
        "--dump-prompt",
        action="store_true",
        help="Include the rendered prompt string in each model_answer.jsonl line.",
    )
    return parser


def read_questions(path: Path) -> List[Dict[str, Any]]:
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


def run_strategy(
    strategy: str,
    question_row: Dict[str, Any],
    rng: random.Random,
) -> str:
    gt_answer = question_row.get("gt_answer", "")
    if strategy == "oracle":
        return gt_answer
    if strategy == "bar_random":
        # Pick one random bar color from the scene's available bars. On Q1
        # (reachability_set) rows there are no bars, so fall back to `[]`.
        bars = (question_row.get("meta_info") or {}).get("bars") or []
        if not bars:
            return "[]"
        color = rng.choice([b.get("color") for b in bars if b.get("color")])
        return f"[{color}]" if color else "[]"
    if strategy == "bar_empty":
        return "[]"
    raise ValueError(f"Unknown strategy: {strategy}")


def parse_name_list(s: Optional[str]) -> Optional[frozenset]:
    """Parse a bracket-enclosed (or loose) comma-separated list of point names
    into a frozenset. iter E1 format `["A", "D"]` (legal JSON) and legacy
    `[A, D]` / `A, D` / `[D,A]` / `[]` / `  ` all handled. Returns None
    only if input is None."""
    if s is None:
        return None
    stripped = s.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return frozenset(str(x).strip().casefold() for x in parsed if str(x).strip())
    except json.JSONDecodeError:
        pass
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    tokens = [t.strip().strip('"').strip("'").casefold() for t in stripped.split(",") if t.strip()]
    return frozenset(tokens)


def answers_equal(gt: str, model: Optional[str]) -> bool:
    """Compare a model answer to gt_answer. Both active question types
    (reachability_set, bar_removal) produce a bracket-list; parse both
    and compare as sets order- and case-insensitively.
    """
    if model is None:
        return False
    return parse_name_list(gt) == parse_name_list(model)


def build_model_answer_row(
    *,
    question_row: Dict[str, Any],
    raw_response_text: str,
    parsed_answer: Optional[str],
    strategy: str,
    prompt: Optional[str],
) -> Dict[str, Any]:
    meta_info = dict(question_row.get("meta_info", {}))
    meta_info["model_id"] = f"stub:{strategy}"
    correct = answers_equal(
        question_row.get("gt_answer", ""),
        parsed_answer,
    )
    row: Dict[str, Any] = {
        "id": question_row["id"],
        "category": question_row.get("category"),
        "type": question_row.get("type"),
        "meta_info": meta_info,
        "question": question_row.get("question"),
        "answer": parsed_answer,
        "correct": bool(correct),
        "invalid_response": parsed_answer is None,
        "api_error": False,
        "raw_response_text": raw_response_text,
    }
    if prompt is not None:
        row["prompt"] = prompt
    return row


def write_model_answer_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def summarize_accuracy(
    question_rows: List[Dict[str, Any]],
    answer_rows: List[Dict[str, Any]],
    strategy: str,
    question_type: str = DEFAULT_QUESTION_TYPE,
) -> Dict[str, Any]:
    if len(question_rows) != len(answer_rows):
        raise ValueError("question and answer rows must be same length")
    total = len(question_rows)
    correct = sum(1 for row in answer_rows if row["correct"])
    invalid = sum(1 for row in answer_rows if row["invalid_response"])
    by_difficulty: Dict[str, Dict[str, int]] = {}
    for q_row, a_row in zip(question_rows, answer_rows):
        difficulty = q_row.get("meta_info", {}).get("difficulty", "unknown")
        bucket = by_difficulty.setdefault(difficulty, {"total": 0, "correct": 0, "invalid": 0})
        bucket["total"] += 1
        bucket["correct"] += int(a_row["correct"])
        bucket["invalid"] += int(a_row["invalid_response"])
    by_difficulty_out: Dict[str, Dict[str, Any]] = {}
    for difficulty, bucket in by_difficulty.items():
        t = bucket["total"]
        by_difficulty_out[difficulty] = {
            "total": t,
            "correct": bucket["correct"],
            "invalid": bucket["invalid"],
            "accuracy": bucket["correct"] / t if t else 0.0,
        }
    return {
        "task": TASK_NAME,
        "question_type": question_type,
        "strategy": strategy,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "by_difficulty": by_difficulty_out,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    questions_path = Path(args.questions).resolve()
    output_path = Path(args.output).resolve()
    score_path = Path(args.score).resolve()

    question_rows = read_questions(questions_path)
    if not question_rows:
        raise SystemExit(f"No questions found in {questions_path}")
    question_type = infer_question_type(question_rows)

    rng = random.Random(args.seed)
    answer_rows: List[Dict[str, Any]] = []
    for q_row in question_rows:
        raw_response_text = run_strategy(args.strategy, q_row, rng)
        parsed_answer = normalize_answer_for_type(
            q_row.get("type", question_type),
            raw_response_text,
        )
        prompt = None
        if args.dump_prompt:
            # iter E1 (dev_trace_5): top-level `type` is now uniformly
            # "reasoning"; the concrete qtype lives in meta_info.question_type
            # or category[2]. Also `question` is now the FULL formulated
            # prompt with literal `{images}`, so we just inline image refs
            # here instead of re-running the template.
            qtype = (
                q_row.get("meta_info", {}).get("question_type")
                or (q_row.get("category", []) or [None, None, None])[2]
                or question_type
            )
            question_text = q_row.get("question", "")
            images = q_row.get("images", [])
            if "{images}" in question_text:
                image_block = (
                    "\n".join(f"[Image]\n{p}" for p in images)
                    if images
                    else "[No images provided]"
                )
                prompt = question_text.replace("{images}", image_block)
            else:
                # Legacy: question is short text → fall back to template builder.
                builder = (
                    build_bar_removal_prompt
                    if qtype == "bar_removal"
                    else build_reachability_set_prompt
                )
                prompt = builder(question_text, images)
        answer_rows.append(
            build_model_answer_row(
                question_row=q_row,
                raw_response_text=raw_response_text,
                parsed_answer=parsed_answer,
                strategy=args.strategy,
                prompt=prompt,
            )
        )

    write_model_answer_jsonl(output_path, answer_rows)
    score = summarize_accuracy(question_rows, answer_rows, args.strategy, question_type)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"strategy={score['strategy']}  total={score['total']}  "
        f"correct={score['correct']}  accuracy={score['accuracy']:.3f}"
    )
    for difficulty, bucket in score["by_difficulty"].items():
        print(
            f"  [{difficulty}] total={bucket['total']}  correct={bucket['correct']}  "
            f"accuracy={bucket['accuracy']:.3f}"
        )
    print(f"Wrote model answers to {output_path}")
    print(f"Wrote score summary to {score_path}")


if __name__ == "__main__":
    main()
