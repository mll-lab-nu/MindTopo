"""Generate split-specific lmms task configs without changing task names."""
from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "topobench_eval"))
from topobench_eval.datasets import REASONING_TASKS, question_path, selected_split


def _write_atomic(path: Path, text: str) -> None:
    """Shards of a concurrent run may be reading these configs."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def include_path() -> Path:
    return _include_path(selected_split())


@lru_cache(maxsize=2)
def _include_path(split: str) -> Path:
    output = HERE / "cache/task_configs" / split
    for task in REASONING_TASKS:
        target = output / task
        target.mkdir(parents=True, exist_ok=True)
        config = (HERE / task / "task.yaml").read_text()
        config = config.replace(f"test: data/core/{task}/question.jsonl",
                                "test: " + json.dumps(str(question_path(task, split))))
        _write_atomic(target / "task.yaml", config)
        original = str(HERE / task / "utils.py")
        _write_atomic(
            target / "utils.py",
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location({('_release_' + split + '_' + task)!r}, {original!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "globals().update({name: value for name, value in vars(_module).items() if not name.startswith('__')})\n"
        )
    _write_atomic(output / "topobench.yaml", (HERE / "topobench.yaml").read_text())
    return output
