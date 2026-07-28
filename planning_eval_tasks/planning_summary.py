from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _fmt_pct(n: float, d: float) -> str:
    return f"{(n / d * 100):.2f}%" if d else "-"


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "success"}
    return bool(value)


def _count_step_flags(steps: Any, *flags: str) -> int:
    if not isinstance(steps, list):
        return 0
    total = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if any(_safe_bool(step.get(flag)) for flag in flags):
            total += 1
    return total


def _metric_from_row(row: Mapping[str, Any], meta: Mapping[str, Any], metric_field: str, *step_flags: str) -> int:
    value = _first_non_empty(row.get(metric_field), meta.get(metric_field))
    if value is not None:
        return _safe_int(value)
    return _count_step_flags(row.get("trajectory"), *step_flags)


def _compact_episode_from_answer_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    episode_id = _first_non_empty(row.get("id"), row.get("episode_id"))
    if episode_id is None:
        return None
    raw_meta = row.get("meta_info")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    trajectory = row.get("trajectory")
    trajectory_len = len(trajectory) if isinstance(trajectory, list) else 0
    total_steps = _safe_int(_first_non_empty(meta.get("total_steps"), row.get("total_steps"), trajectory_len))
    return {
        "episode_id": str(episode_id),
        "level": _first_non_empty(row.get("level"), meta.get("level")),
        "seed": _first_non_empty(row.get("seed"), meta.get("seed")),
        "repeat_index": _first_non_empty(row.get("repeat_index"), meta.get("repeat_index")),
        "success": _safe_bool(_first_non_empty(row.get("success"), meta.get("success"))),
        "total_steps": total_steps,
        "illegal_moves": _metric_from_row(row, meta, "illegal_moves", "illegal"),
        "invalid_responses": _metric_from_row(row, meta, "invalid_responses", "invalid_response"),
        "api_errors": _metric_from_row(row, meta, "api_errors", "api_error", "vlm_api_error", "video_gen_api_error"),
        "meta_info": meta,
    }


def _count_worker_errors(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.startswith("[worker-error]"))
    except OSError:
        return None


def _summary_from_answer_jsonl(path: Path, base_summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    if not rows:
        return None

    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        episode = _compact_episode_from_answer_row(row)
        if episode is None:
            continue
        unique[str(episode["episode_id"])] = episode
    episodes = list(unique.values())
    if not episodes:
        return None

    first_meta = episodes[0].get("meta_info") if isinstance(episodes[0].get("meta_info"), dict) else {}
    total = len(episodes)
    successes = sum(1 for episode in episodes if _safe_bool(episode.get("success")))
    success_steps = [
        _safe_int(episode.get("total_steps"))
        for episode in episodes
        if _safe_bool(episode.get("success"))
    ]
    worker_errors = _count_worker_errors(path.with_name("worker_errors.log"))

    summary = dict(base_summary)
    summary.update(
        {
            "env": str(_first_non_empty(base_summary.get("env"), first_meta.get("task_name"), path.parent.name)),
            "model_id": str(_first_non_empty(base_summary.get("model_id"), first_meta.get("model_id"), path.parent.parent.name)),
            "total_episodes": total,
            "successes": successes,
            "success_rate": (successes / total) if total else 0.0,
            "avg_steps_all": (sum(_safe_int(episode.get("total_steps")) for episode in episodes) / total) if total else 0.0,
            "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
            "avg_illegal_moves": (sum(_safe_int(episode.get("illegal_moves")) for episode in episodes) / total) if total else 0.0,
            "avg_invalid_responses": (sum(_safe_int(episode.get("invalid_responses")) for episode in episodes) / total) if total else 0.0,
            "avg_api_errors": (sum(_safe_int(episode.get("api_errors")) for episode in episodes) / total) if total else 0.0,
            "episodes": episodes,
            "model_answer_jsonl": path.name,
        }
    )
    if worker_errors is not None:
        summary["worker_errors"] = worker_errors
    return summary


def _summary_rank(path: Path, summary: Mapping[str, Any]) -> Tuple[int, int, int, float]:
    episodes = summary.get("episodes")
    has_final_episodes = 1 if isinstance(episodes, list) else 0
    has_markdown = 1 if (path.parent / "summary.md").exists() else 0
    total = _safe_int(summary.get("total_episodes"))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return has_final_episodes, has_markdown, total, mtime


def _iter_summary_files(model_root: Path) -> Iterable[Path]:
    for path in sorted(model_root.rglob("summary.json")):
        if path.name == "summary.json":
            yield path


def _select_env_summaries(model_root: Path) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Tuple[Tuple[int, int, int, float], Dict[str, Any]]] = {}
    for path in _iter_summary_files(model_root):
        summary = _read_json(path)
        if not summary:
            continue
        summary = _summary_from_answer_jsonl(path.parent / "model_answer.jsonl", summary) or summary
        env = summary.get("env")
        if not isinstance(env, str) or not env:
            continue
        summary = dict(summary)
        summary["_summary_path"] = str(path)
        rank = _summary_rank(path, summary)
        current = selected.get(env)
        if current is None or rank > current[0]:
            selected[env] = (rank, summary)
    return {env: item[1] for env, item in selected.items()}


def _difficulty_for_episode(episode: Mapping[str, Any]) -> str:
    meta_info = episode.get("meta_info")
    if isinstance(meta_info, dict):
        if meta_info.get("task_name") == "enclosure_chat_noir":
            policy = str(meta_info.get("cat_policy_name") or "").strip().lower()
            if policy in {"easy", "medium", "hard"}:
                return policy
        difficulty = meta_info.get("difficulty")
        if difficulty not in (None, ""):
            return str(difficulty)
        if meta_info.get("task_name") == "separation_one_stroke":
            board_size = _safe_int(meta_info.get("board_size"))
            board_size_difficulty = {4: "easy", 5: "medium", 6: "hard"}
            if board_size in board_size_difficulty:
                return board_size_difficulty[board_size]
    difficulty = episode.get("difficulty")
    if difficulty not in (None, ""):
        return str(difficulty)
    level = episode.get("level")
    return str(level if level not in (None, "") else "?")


def _weighted_average(summaries: Iterable[Mapping[str, Any]], field: str, weight_field: str) -> Optional[float]:
    numerator = 0.0
    denominator = 0
    for summary in summaries:
        weight = _safe_int(summary.get(weight_field))
        if weight <= 0:
            continue
        value = summary.get(field)
        if value is None:
            continue
        numerator += _safe_float(value) * weight
        denominator += weight
    return (numerator / denominator) if denominator else None


def render_model_planning_summary_md(model_root: Path) -> Optional[str]:
    per_env = _select_env_summaries(model_root)
    if not per_env:
        return None

    summaries = list(per_env.values())
    first = summaries[0]
    model_id = str(first.get("model_id") or model_root.name)
    model_name = str(first.get("model_name") or model_id)
    total = sum(_safe_int(s.get("total_episodes")) for s in summaries)
    successes = sum(_safe_int(s.get("successes")) for s in summaries)
    skipped = sum(len(s.get("skipped_episodes") or []) for s in summaries)
    worker_errors = sum(_safe_int(s.get("worker_errors")) for s in summaries)
    avg_steps_all = _weighted_average(summaries, "avg_steps_all", "total_episodes")
    avg_steps_success = _weighted_average(summaries, "avg_steps_on_success", "successes")
    avg_illegal = _weighted_average(summaries, "avg_illegal_moves", "total_episodes")
    avg_invalid = _weighted_average(summaries, "avg_invalid_responses", "total_episodes")
    avg_api = _weighted_average(summaries, "avg_api_errors", "total_episodes")

    lines = [
        f"# Planning Summary - {model_name}",
        "",
        f"- Model ID: `{model_id}`",
        f"- Tasks: {len(per_env)}",
        f"- Total episodes: {total}",
        f"- Successes: {successes}",
        f"- Aggregate success rate: **{_fmt_pct(successes, total)}**",
        f"- Avg steps (all): {_fmt_float(avg_steps_all)}",
        f"- Avg steps (on success): {_fmt_float(avg_steps_success)}",
        f"- Avg illegal moves: {_fmt_float(avg_illegal)}",
        f"- Avg invalid responses: {_fmt_float(avg_invalid)}",
        f"- Avg API errors: {_fmt_float(avg_api)}",
        f"- Skipped episodes: {skipped}",
        f"- Worker errors: {worker_errors}",
        "",
        "## Per-Task Success",
        "",
        "| Task | N | Successes | Success rate | Avg steps | Avg illegal | Avg invalid | Avg API errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for env, summary in sorted(
        per_env.items(),
        key=lambda item: (
            -_safe_float(item[1].get("success_rate")),
            item[0],
        ),
    ):
        env_total = _safe_int(summary.get("total_episodes"))
        env_successes = _safe_int(summary.get("successes"))
        lines.append(
            f"| {env} | {env_total} | {env_successes} | "
            f"{_fmt_pct(env_successes, env_total)} | "
            f"{_fmt_float(summary.get('avg_steps_all'))} | "
            f"{_fmt_float(summary.get('avg_illegal_moves'))} | "
            f"{_fmt_float(summary.get('avg_invalid_responses'))} | "
            f"{_fmt_float(summary.get('avg_api_errors'))} |"
        )

    diff_keys: List[str] = []
    per_env_diff: Dict[str, Dict[str, List[int]]] = {}
    for env, summary in per_env.items():
        diff_counts: Dict[str, List[int]] = {}
        episodes = summary.get("episodes")
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            difficulty = _difficulty_for_episode(episode)
            if difficulty not in diff_keys:
                diff_keys.append(difficulty)
            bucket = diff_counts.setdefault(difficulty, [0, 0])
            bucket[1] += 1
            if episode.get("success"):
                bucket[0] += 1
        if diff_counts:
            per_env_diff[env] = diff_counts

    preferred = [d for d in ("easy", "medium", "hard") if d in diff_keys]
    other = sorted(d for d in diff_keys if d not in preferred)
    diff_cols = preferred + other
    if diff_cols:
        lines += [
            "",
            "## Per-Task Difficulty Breakdown",
            "",
            "| Task | " + " | ".join(diff_cols) + " |",
            "| --- |" + "".join(" --- |" for _ in diff_cols),
        ]
        for env in sorted(per_env):
            counts = per_env_diff.get(env, {})
            cells = []
            for difficulty in diff_cols:
                value = counts.get(difficulty)
                cells.append(_fmt_pct(value[0], value[1]) if value else "-")
            lines.append(f"| {env} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def write_model_planning_summary(model_root: Path) -> Optional[Path]:
    model_root = model_root.resolve()
    markdown = render_model_planning_summary_md(model_root)
    if markdown is None:
        return None
    out_path = model_root / "planning_summary.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def model_root_for_summary(summary_path: Path, summary_payload: Mapping[str, Any]) -> Optional[Path]:
    env = summary_payload.get("env")
    if not isinstance(env, str) or not env:
        return None
    model_id = str(summary_payload.get("model_id") or "")
    model_names = {model_id, model_id.replace("/", "_")} if model_id else set()
    parent = summary_path.resolve().parent
    # Current runs use logs/planning_eval/<model>/<env>/summary.json, but a
    # few older examples used logs/planning_eval/<env>/<model>/summary.json.
    # In the old layout, returning the env's parent would accidentally scan
    # every model under logs/planning_eval, so keep aggregation local.
    if parent.parent.name == env and (not model_names or parent.name in model_names):
        return parent
    for candidate in (parent, *parent.parents):
        if candidate.name == env:
            if candidate.parent.name == "planning_eval":
                return None
            return candidate.parent
    return None


def write_model_planning_summary_for_run(
    *,
    summary_path: Path,
    summary_payload: Mapping[str, Any],
) -> Optional[Path]:
    model_root = model_root_for_summary(summary_path, summary_payload)
    if model_root is None:
        return None
    return write_model_planning_summary(model_root)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python planning_summary.py <model_root> [<model_root> ...]", file=sys.stderr)
        return 2
    for raw in argv[1:]:
        out_path = write_model_planning_summary(Path(raw))
        if out_path is None:
            print(f"[planning-summary] no summaries found under {raw}", file=sys.stderr)
            continue
        print(f"[planning-summary] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
