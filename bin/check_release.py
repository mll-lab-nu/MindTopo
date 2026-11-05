#!/usr/bin/env python3
"""Validate release counts, IDs, Core overlap, assets and probing lists."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topobench_eval"))
from topobench_eval.datasets import TASKS, validate_dataset, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="Check manifests/JSONL only")
    args = parser.parse_args()
    records = {}
    for split, count in (("core", 3900), ("extended", 11035)):
        rows = []
        for task in TASKS:
            task_rows = validate_dataset(task, split, images=not args.skip_images)
            if split == "core":
                assert Counter(row["meta_info"].get("benchmark_level") for row in task_rows) == {"Easy": 150, "Hard": 150}, task
            for row in task_rows:
                assert all(not Path(image).is_absolute() for image in row.get("images", []))
            rows.extend((task, row) for row in task_rows)
        assert len(rows) == count
        manifest = read_rows(ROOT / f"data/manifests/{split}_manifest.jsonl")
        expected = {(entry["task"], entry["id"]): entry for entry in manifest}
        for task, row in rows:
            entry = expected[(task, row["id"])]
            assert row["meta_info"]["release_record_sha256"] == entry["record_sha256"]
            assert row["meta_info"]["release_input_sha256"] == entry["input_sha256"]
        records[split] = rows
        print(f"{split}: {len(rows)} records, {len(TASKS)} tasks")
    keys = lambda rows: {(task, row["id"]) for task, row in rows}
    inputs = lambda rows: {(task, row["meta_info"]["release_input_sha256"]) for task, row in rows}
    for split, count in (("core", 3900), ("extended", 11035)):
        assert len(keys(records[split])) == count
    # Extended is the unmodified original dataset, so it overlaps Core and keeps
    # its five repeated Swap inputs. IDs are unique within a split only.
    assert len(inputs(records["extended"])) == 11030
    shared = inputs(records["core"]) & inputs(records["extended"])
    flagged = {(entry["task"], entry["input_sha256"])
               for entry in read_rows(ROOT / "data/manifests/extended_manifest.jsonl") if entry["in_core"]}
    assert len(shared) == 2250 and shared == flagged
    core_inputs = {(task, row["id"]): row["meta_info"]["release_input_sha256"] for task, row in records["core"]}
    reused = [row for task, row in records["extended"]
              if (task, row["id"]) in core_inputs
              and core_inputs[(task, row["id"])] != row["meta_info"]["release_input_sha256"]]
    assert len(reused) == 150
    for path in (ROOT / "data/slices/balanced_105").glob("*.json"):
        ids = json.loads(path.read_text())
        available = {row["id"] for row in read_rows(ROOT / f"data/probing/{path.stem}/question.jsonl")}
        assert len(ids) == len(set(ids)) == 105 and set(ids) <= available, path
    if not args.skip_images:
        assets = json.loads((ROOT / "data/manifests/assets.json").read_text())
        for relative, info in assets.items():
            path = ROOT / relative
            assert not path.is_symlink() and path.stat().st_size == info["bytes"], path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], path
        print(f"{len(assets)} image hashes verified")
    print("PASS: fixed counts, flagged Core overlap, split-local IDs and unchanged balanced_105 lists")


if __name__ == "__main__":
    main()
