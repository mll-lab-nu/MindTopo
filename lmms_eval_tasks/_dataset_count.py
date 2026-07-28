"""Cache the post-`process_docs` filtered dataset length for a task.

The lmms-eval API pool wrapper uses this to compute even shard ranges across
parallel workers via `--offset/--limit`. Loading a task through TaskManager runs
the dataset's `process_docs` filter (which for topobench tasks opens every PNG
to verify it's loadable) — that's tens of seconds on cold start, but the result
is stable as long as `question.jsonl` hasn't changed, so we cache by mtime.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOPBENCH_EVAL_ROOT = ROOT / "topobench_eval"
if str(TOPBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPBENCH_EVAL_ROOT))


def _question_jsonl_path(task_name: str) -> Optional[Path]:
    """Best-effort lookup of the question.jsonl that backs the task config.

    Prefers the new `environments/<task>/output/question.jsonl` layout and
    falls back to the legacy `output/benchmark_output/` location.
    """
    new_layout = ROOT / "environments" / task_name / "output" / "question.jsonl"
    if new_layout.exists():
        return new_layout
    legacy = ROOT / "environments" / task_name / "output" / "benchmark_output" / "question.jsonl"
    if legacy.exists():
        return legacy
    return None


def _cache_file(task_name: str) -> Path:
    return HERE / "cache" / f".dataset_count__{task_name}.json"


def _read_cache(task_name: str, mtime: float) -> Optional[int]:
    cache = _cache_file(task_name)
    if not cache.exists():
        return None
    try:
        payload = json.loads(cache.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if float(payload.get("mtime", -1)) != mtime:
        return None
    size = payload.get("size")
    return int(size) if isinstance(size, int) else None


def _write_cache(task_name: str, mtime: float, size: int) -> None:
    cache = _cache_file(task_name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mtime": mtime, "size": size}, sort_keys=True))


def get_or_compute(task_name: str) -> int:
    """Return the post-filter sample count for `task_name`.

    Cache key = (task_name, question.jsonl mtime). Cold path imports lmms-eval
    and runs the task's `process_docs` filter — slow but only once per dataset
    revision. Caller is responsible for handling the lmms-eval import being
    expensive (≈10s).
    """
    qpath = _question_jsonl_path(task_name)
    mtime = qpath.stat().st_mtime if qpath else 0.0

    cached = _read_cache(task_name, mtime)
    if cached is not None:
        return cached

    # Cold path: load via TaskManager. Suppress the noisy default loguru output
    # from lmms-eval task discovery; users will see only our wrapper logs.
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    os.environ.setdefault("TOPOBENCH_LMMS_INCLUDE_DEFAULT_TASKS", "0")
    from lmms_eval.tasks import TaskManager, get_task_dict  # type: ignore

    tm = TaskManager(include_path=str(HERE), include_defaults=False)
    task_dict = get_task_dict([task_name], task_manager=tm)
    if task_name not in task_dict:
        raise RuntimeError(
            f"lmms-eval did not register task {task_name!r}; check include_path={HERE}"
        )
    task = task_dict[task_name]
    docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
    size = int(len(docs))
    _write_cache(task_name, mtime, size)
    return size


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python _dataset_count.py <task_name>", file=sys.stderr)
        raise SystemExit(2)
    print(get_or_compute(sys.argv[1]))
