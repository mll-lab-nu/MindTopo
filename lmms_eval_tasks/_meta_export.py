"""Convert lmms-eval per-sample shards into TopoBench's reasoning_model_answer.meta.jsonl.

Schema reference: environments/meta_jsonl/README.md (reasoning_model_answer section).

Source: each shard directory at lmms_eval_tasks/logs/<model>__<task>/shard_NN/
contains an lmms-eval-produced `<datetime>_samples_<task>.jsonl` whose rows
include a TopoBench metric payload (see lmms_eval_tasks/_common.py).

Sinks:
- environments/<task>/output/model_answer.jsonl (or output/benchmark_output/model_answer.jsonl on legacy envs)
- lmms_eval_tasks/logs/<model>__<task>/model_answer.jsonl

Each file has one row per question, in the same order as `question.jsonl`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOPBENCH_EVAL_ROOT = ROOT / "topobench_eval"
if str(TOPBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPBENCH_EVAL_ROOT))

_HOOKS_CACHE: Dict[str, Any] = {}

try:
    from _retryable_response import is_retryable_failure_text
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from lmms_eval_tasks._retryable_response import is_retryable_failure_text  # type: ignore


def _task_output_root(task_name: str) -> Path:
    """Prefer the new `environments/<task>/output/` layout, fall back to legacy."""
    new_layout = ROOT / "environments" / task_name / "output"
    if (new_layout / "question.jsonl").exists():
        return new_layout
    legacy = new_layout / "benchmark_output"
    if (legacy / "question.jsonl").exists():
        return legacy
    return new_layout


def _question_jsonl_path(task_name: str) -> Path:
    return _task_output_root(task_name) / "question.jsonl"


def _model_answer_path(task_name: str) -> Path:
    return _task_output_root(task_name) / "model_answer.jsonl"


def _shard_root(model_id: str, task_name: str) -> Path:
    safe_model = model_id.replace("/", "_")
    return HERE / "logs" / f"{safe_model}__{task_name}"


def _log_model_answer_path(model_id: str, task_name: str) -> Path:
    return _shard_root(model_id, task_name) / "model_answer.jsonl"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_questions(task_name: str) -> Dict[str, Dict[str, Any]]:
    qpath = _question_jsonl_path(task_name)
    if not qpath.exists():
        raise FileNotFoundError(f"missing question.jsonl for task {task_name}: {qpath}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(qpath):
        qid = row.get("id")
        if isinstance(qid, str):
            out[qid] = row
    if not out:
        raise RuntimeError(f"question.jsonl for task {task_name} is empty: {qpath}")
    return out


def _samples_files(model_id: str, task_name: str) -> List[Path]:
    root = _shard_root(model_id, task_name)
    if not root.exists():
        return []
    return sorted(root.rglob(f"*_samples_{task_name}.jsonl"))


def _meta_match(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # These metrics intentionally carry the same per-sample payload. Prefer the
    # main accuracy metric, but fall back to auxiliary metrics so model_answer
    # export does not depend on one displayed aggregate existing forever.
    for metric_name in ("topobench_exact_match", "json_parse_rate", "answer_format_hit_rate"):
        payload = sample.get(metric_name)
        if isinstance(payload, dict):
            return payload
        # lmms-eval may serialize the metric instance under a wrapping list.
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
    return None


def _raw_response(sample: Dict[str, Any], meta: Dict[str, Any]) -> str:
    raw = meta.get("raw")
    if raw is not None:
        return str(raw)
    raw = sample.get("resps")
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if raw is not None:
        return str(raw)
    raw = sample.get("filtered_resps")
    if raw is not None:
        return str(raw)
    return ""


def _is_api_failure_text(text: str) -> bool:
    return is_retryable_failure_text(text)


def _is_invalid_model_response(
    *,
    api_error: bool,
    parse_method: str,
    predicted: Any,
    semantic_invalid: Any = None,
) -> bool:
    if api_error:
        return False
    if semantic_invalid is not None:
        return bool(semantic_invalid)
    return parse_method == "unclear" or predicted in (None, "")


def _fresh_meta(task_name: str, sample: Dict[str, Any], qrow: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    raw = _raw_response(sample, fallback)
    try:
        try:
            from lmms_eval_tasks._common import make_hooks  # type: ignore
        except ModuleNotFoundError:
            from _common import make_hooks  # type: ignore

        hooks = _HOOKS_CACHE.get(task_name)
        if hooks is None:
            hooks = make_hooks(_question_jsonl_path(task_name).parent)
            _HOOKS_CACHE[task_name] = hooks
        result = hooks["process_results"](qrow, [raw])
        payload = result.get("topobench_exact_match")
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        # Never silently preserve stale correctness when reparsing fails.
        return {
            **fallback,
            "raw": raw,
            "correct": False,
            "predicted": "unclear",
            "parse_method": "unclear",
            "invalid_response": True,
            "reparse_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **fallback,
        "raw": raw,
        "correct": False,
        "predicted": "unclear",
        "parse_method": "unclear",
        "invalid_response": True,
        "reparse_error": "process_results returned no topobench_exact_match payload",
    }


def _convert_row(
    task_name: str,
    sample: Dict[str, Any],
    questions: Dict[str, Dict[str, Any]],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    meta = _meta_match(sample)
    if meta is None:
        return None
    qid = meta.get("id") or sample.get("doc_id_meta") or sample.get("id")
    if not isinstance(qid, str):
        return None
    qrow = questions.get(qid)
    if qrow is None:
        return None
    raw_text = _raw_response(sample, meta)
    api_error = _is_api_failure_text(raw_text)
    if not api_error:
        meta = _fresh_meta(task_name, sample, qrow, meta)
    parse_method = str(meta.get("parse_method", "unclear"))
    predicted = meta.get("predicted")
    raw_text = str(meta.get("raw", ""))
    api_error = api_error or _is_api_failure_text(raw_text)
    invalid = _is_invalid_model_response(
        api_error=api_error,
        parse_method=parse_method,
        predicted=predicted,
        semantic_invalid=meta.get("invalid_response"),
    )
    return {
        "id": qrow["id"],
        "category": qrow["category"],
        "type": qrow["type"],
        "meta_info": {**(qrow.get("meta_info") or {}), "model_id": model_name},
        "question": qrow.get("question", ""),
        "gt_answer": qrow.get("gt_answer"),
        "answer": predicted,
        "correct": bool(meta.get("correct", False)),
        "invalid_response": bool(invalid),
        "api_error": bool(api_error),
        "raw_response_text": raw_text,
    }


def _compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    invalid = sum(1 for r in rows if r.get("invalid_response"))
    api_err = sum(1 for r in rows if r.get("api_error"))
    by_type: Dict[str, List[int]] = {}
    by_diff: Dict[str, List[int]] = {}
    # [correct, n, invalid]
    by_qtype: Dict[str, List[int]] = {}
    by_qtype_diff: Dict[tuple, List[int]] = {}
    for r in rows:
        meta = r.get("meta_info") or {}
        t = str(r.get("type", "?"))
        d = str(meta.get("difficulty", "?"))
        q = meta.get("question_type")
        is_correct = bool(r.get("correct"))
        is_invalid = bool(r.get("invalid_response"))
        by_type.setdefault(t, [0, 0])[1] += 1
        by_diff.setdefault(d, [0, 0])[1] += 1
        if is_correct:
            by_type[t][0] += 1
            by_diff[d][0] += 1
        if q is not None:
            qs = str(q)
            by_qtype.setdefault(qs, [0, 0, 0])[1] += 1
            by_qtype_diff.setdefault((qs, d), [0, 0, 0])[1] += 1
            if is_correct:
                by_qtype[qs][0] += 1
                by_qtype_diff[(qs, d)][0] += 1
            if is_invalid:
                by_qtype[qs][2] += 1
                by_qtype_diff[(qs, d)][2] += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "invalid": invalid,
        "api_errors": api_err,
        "by_type": by_type,
        "by_difficulty": by_diff,
        "by_question_type": by_qtype,
        "by_question_type_difficulty": by_qtype_diff,
    }


def _should_replace_existing_row(
    existing: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> bool:
    """Prefer completed model outputs over later transient API failures."""
    if existing is None:
        return True
    existing_api_error = bool(existing.get("api_error"))
    candidate_api_error = bool(candidate.get("api_error"))
    if existing_api_error and not candidate_api_error:
        return True
    if not existing_api_error and candidate_api_error:
        return False
    return True


def _fmt_pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.2f}%" if d else "—"


def _render_task_summary_md(task_name: str, model_id: str, model_name: str, stats: Dict[str, Any]) -> str:
    total = stats["total"]
    correct = stats["correct"]
    lines = [
        f"# {task_name} — {model_name}",
        "",
        f"- Model ID: `{model_id}`",
        f"- Total samples: {total}",
        f"- Correct: {correct}",
        f"- Accuracy: **{_fmt_pct(correct, total)}**",
        f"- Invalid responses: {stats['invalid']}",
        f"- API errors: {stats['api_errors']}",
        "",
        "## By Subtype",
        "",
        "| Subtype | N | Accuracy |",
        "| --- | --- | --- |",
    ]
    for t, (c, n) in sorted(stats["by_type"].items()):
        lines.append(f"| {t} | {n} | {_fmt_pct(c, n)} |")
    lines += [
        "",
        "## By Difficulty",
        "",
        "| Difficulty | N | Accuracy |",
        "| --- | --- | --- |",
    ]
    for d, (c, n) in sorted(stats["by_difficulty"].items()):
        lines.append(f"| {d} | {n} | {_fmt_pct(c, n)} |")
    by_qtype = stats.get("by_question_type") or {}
    by_qtype_diff = stats.get("by_question_type_difficulty") or {}
    if by_qtype:
        lines += [
            "",
            "## By Question Type",
            "",
            "| Question Type | N | Accuracy | Invalid |",
            "| --- | --- | --- | --- |",
        ]
        for q, (c, n, inv) in sorted(by_qtype.items()):
            lines.append(f"| {q} | {n} | {_fmt_pct(c, n)} | {inv} |")
    if by_qtype_diff:
        diff_keys: List[str] = []
        for (_, d) in by_qtype_diff:
            if d not in diff_keys:
                diff_keys.append(d)
        preferred = [d for d in ("easy", "medium", "hard") if d in diff_keys]
        ordered_diffs = preferred + sorted(d for d in diff_keys if d not in preferred)
        lines += [
            "",
            "## By Question Type × Difficulty",
            "",
            "| Question Type | Difficulty | N | Accuracy | Invalid |",
            "| --- | --- | --- | --- | --- |",
        ]
        for q in sorted({qq for qq, _ in by_qtype_diff}):
            for d in ordered_diffs:
                cell = by_qtype_diff.get((q, d))
                if not cell:
                    continue
                c, n, inv = cell
                lines.append(f"| {q} | {d} | {n} | {_fmt_pct(c, n)} | {inv} |")
    lines.append("")
    return "\n".join(lines)


def _render_overall_summary_md(model_id: str, model_name: str, per_task: Dict[str, Dict[str, Any]]) -> str:
    total_n = sum(s["total"] for s in per_task.values())
    total_c = sum(s["correct"] for s in per_task.values())
    lines = [
        f"# Reasoning Summary — {model_name}",
        "",
        f"- Model ID: `{model_id}`",
        f"- Tasks: {len(per_task)}",
        f"- Total samples: {total_n}",
        f"- Aggregate accuracy: **{_fmt_pct(total_c, total_n)}**",
        "",
        "## Per-Task Accuracy",
        "",
        "| Task | N | Accuracy |",
        "| --- | --- | --- |",
    ]
    for t, s in sorted(per_task.items(), key=lambda kv: -kv[1]["accuracy"]):
        lines.append(f"| {t} | {s['total']} | {_fmt_pct(s['correct'], s['total'])} |")
    diff_keys: List[str] = []
    for s in per_task.values():
        for d in s["by_difficulty"]:
            if d not in diff_keys:
                diff_keys.append(d)
    preferred = [d for d in ("easy", "medium", "hard") if d in diff_keys]
    other = sorted(d for d in diff_keys if d not in preferred)
    diff_cols = preferred + other
    if diff_cols:
        header = "| Task | " + " | ".join(diff_cols) + " |"
        sep = "| --- |" + "".join(" --- |" for _ in diff_cols)
        lines += ["", "## Per-Task Difficulty Breakdown", "", header, sep]
        for t, s in sorted(per_task.items()):
            cells = []
            for d in diff_cols:
                v = s["by_difficulty"].get(d)
                cells.append(_fmt_pct(v[0], v[1]) if v else "—")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-Task Subtype Breakdown", ""]
    for t, s in sorted(per_task.items()):
        if not s["by_type"]:
            continue
        lines += [f"### {t}", "", "| Subtype | N | Accuracy |", "| --- | --- | --- |"]
        for typ, (c, n) in sorted(s["by_type"].items()):
            lines.append(f"| {typ} | {n} | {_fmt_pct(c, n)} |")
        lines.append("")
    return "\n".join(lines)


def _safe_model_filename(model_id: str) -> str:
    return model_id.replace("/", "_")


def export_task(task_name: str, *, model_id: str, model_name: str) -> Dict[str, Any]:
    """Merge all shard samples for `task_name` into model_answer.jsonl. Returns counts."""
    questions = _load_questions(task_name)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    samples_files = _samples_files(model_id, task_name)
    if not samples_files:
        print(
            f"[meta-export] no shard sample files for task={task_name!r} under "
            f"{_shard_root(model_id, task_name)}",
            file=sys.stderr,
        )
    for samples_file in samples_files:
        for sample in _read_jsonl(samples_file):
            row = _convert_row(task_name, sample, questions, model_name)
            if row is None:
                continue
            # If the same question id appears in multiple shards (shouldn't happen
            # with even slicing, but resume edge cases can produce duplicates),
            # the last-write wins except that transient API failures do not
            # overwrite an already completed model output.
            if _should_replace_existing_row(rows_by_id.get(row["id"]), row):
                rows_by_id[row["id"]] = row

    # Stable order: follow question.jsonl
    ordered_ids = list(questions.keys())
    rows_out = [rows_by_id[qid] for qid in ordered_ids if qid in rows_by_id]
    out_path = _model_answer_path(task_name)
    log_out_path = _log_model_answer_path(model_id, task_name)
    for path in (out_path, log_out_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    stats = _compute_stats(rows_out)
    summary_md = _render_task_summary_md(task_name, model_id, model_name, stats)
    summary_path = _shard_root(model_id, task_name) / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    print(
        f"[meta-export] task={task_name} wrote {len(rows_out)}/{len(questions)} "
        f"rows → {out_path} and {log_out_path}; summary → {summary_path}"
    )
    return {"questions": len(questions), "answered": len(rows_out), "stats": stats}


def export_all(model_id: str, tasks: List[str]) -> None:
    from topobench_eval.provider_registry import resolve_model_registry_entry  # type: ignore

    resolved = resolve_model_registry_entry(model_id)
    model_name = resolved.upstream_model_name
    per_task_stats: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        try:
            result = export_task(task, model_id=model_id, model_name=model_name)
        except FileNotFoundError as exc:
            print(f"[meta-export] skip task={task!r}: {exc}", file=sys.stderr)
            continue
        stats = result.get("stats")
        if isinstance(stats, dict) and stats.get("total"):
            per_task_stats[task] = stats

    if per_task_stats:
        overall_md = _render_overall_summary_md(model_id, model_name, per_task_stats)
        overall_path = HERE / "logs" / f"{_safe_model_filename(model_id)}_reasoning_summary.md"
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        overall_path.write_text(overall_md, encoding="utf-8")
        print(f"[meta-export] cross-task summary → {overall_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python _meta_export.py <model_id> <task1> [task2 ...]", file=sys.stderr)
        raise SystemExit(2)
    export_all(sys.argv[1], list(sys.argv[2:]))
