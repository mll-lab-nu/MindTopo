"""Static reasoning chain: task wiring and fixed benchmark data."""

from __future__ import annotations

import json

import yaml

from conftest import ROOT, STATIC_TASKS


class _TolerantLoader(yaml.SafeLoader):
    """lmms-eval task YAMLs use `!function ...` tags; ignore them when parsing."""


_TolerantLoader.add_multi_constructor(
    "!function", lambda loader, suffix, node: str(suffix)
)


def _load_task_yaml(path):
    return yaml.load(path.read_text(), Loader=_TolerantLoader)


def test_all_static_tasks_in_group_config():
    group = yaml.safe_load((ROOT / "lmms_eval_tasks" / "topobench.yaml").read_text())
    listed = set(group.get("task", []))
    assert set(STATIC_TASKS) <= listed, set(STATIC_TASKS) - listed


def test_each_static_task_is_wired():
    for task in STATIC_TASKS:
        tdir = ROOT / "lmms_eval_tasks" / task
        assert (tdir / "task.yaml").is_file(), f"{task}: missing task.yaml"
        assert (tdir / "utils.py").is_file(), f"{task}: missing utils.py"
        cfg = _load_task_yaml(tdir / "task.yaml")
        data_files = cfg["dataset_kwargs"]["data_files"]
        rel = data_files["test"] if isinstance(data_files, dict) else data_files
        assert (ROOT / rel).is_file(), f"{task}: dataset path {rel} missing"


def test_static_question_jsonl_is_valid_and_references_images():
    for task in STATIC_TASKS:
        path = ROOT / "environments" / task / "output" / "question.jsonl"
        assert path.is_file(), task
        with path.open() as fh:
            first = fh.readline()
            row = json.loads(first)
            for key in ("id", "images", "gt_answer"):
                assert key in row, f"{task}: row missing {key}"
            # Static tasks are image-based; the fixed rows must reference images.
            assert isinstance(row["images"], list) and row["images"], task
            # Validate the rest of the file parses as JSONL.
            for line in fh:
                if line.strip():
                    json.loads(line)
