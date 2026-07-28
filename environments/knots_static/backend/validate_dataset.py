#!/usr/bin/env python3
"""
validate_dataset.py - 验证数据集 metadata 完整性和 ground truth 一致性。
在批量生成图片前运行此脚本，提前发现问题。

Usage:
  python backend/validate_dataset.py --data_dir ./dataset
"""

import json
import sys
from pathlib import Path
from collections import Counter

# 从 vlm_benchmark 复用 ground truth 逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.vlm_benchmark import (
    get_ground_truth, compute_difficulty, sample_applicable,
    TASKS, _FAMILY_MAP, _LINK_TYPES,
)

IMAGE_ROOT_NAME = "images"
LEGACY_IMAGE_SUBDIRS = ("pairs", "singles", "links", "multi_rope", "link_groups", "mixed")
_EXCLUDED_TYPES: set[str] = set()
REQUIRED_FIELDS = [
    "knotType", "topologicalId", "isKnot", "isUnknot", "isDeceptive",
    "crossingNumber", "slackness", "images",
]
LINK_REQUIRED_FIELDS = ["numComponents"]
PAIR_REQUIRED_FIELDS = ["label_equivalent", "topologicalIdA", "topologicalIdB", "image1", "image2"]

KNOWN_KNOT_TYPES = set(_FAMILY_MAP.keys()) | _LINK_TYPES


def _extract_sample_records(node: object) -> list[dict]:
    if isinstance(node, list):
        records: list[dict] = []
        for item in node:
            records.extend(_extract_sample_records(item))
        return records
    if not isinstance(node, dict):
        return []
    if any(key in node for key in ("filename", "images", "candidate_images", "label_equivalent")):
        return [node]
    records: list[dict] = []
    for key, value in node.items():
        if key in {"version", "generatedAt", "generated_at", "stats", "task"}:
            continue
        records.extend(_extract_sample_records(value))
    return records


def load_dataset_records(data_dir: Path) -> list[tuple[str, dict]]:
    dataset_meta_path = data_dir / "dataset_metadata.json"
    if dataset_meta_path.exists():
        with dataset_meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            samples = raw
        elif isinstance(raw, dict):
            nested = raw.get("samples")
            samples = nested if isinstance(nested, list) else _extract_sample_records(raw)
        else:
            samples = []
        return [(str(dataset_meta_path), sample) for sample in samples if isinstance(sample, dict)]

    meta_files = sorted(
        list(data_dir.glob("**/metadata*.json")) +
        list(data_dir.glob("**/*metadata*.json"))
    )
    seen = set()
    records: list[tuple[str, dict]] = []
    for meta_path in meta_files:
        key = str(meta_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                records.append((str(meta_path), json.load(f)))
        except Exception as exc:
            records.append((str(meta_path), {"_load_error": str(exc)}))
    return records


def resolve_image_path(data_dir: Path, metadata: dict, image_name: str) -> Path | None:
    image_path = Path(image_name)
    if image_path.is_absolute():
        return image_path if image_path.exists() else None

    candidates: list[Path] = [(data_dir / IMAGE_ROOT_NAME / image_name).resolve()]
    knot_type = str(metadata.get("knotType") or "").strip()
    if knot_type:
        candidates.extend(
            (data_dir / subdir / knot_type / image_name).resolve()
            for subdir in LEGACY_IMAGE_SUBDIRS
            if subdir != "pairs"
        )
        candidates.append((data_dir / knot_type / image_name).resolve())
    candidates.extend([
        (data_dir / "pairs" / image_name).resolve(),
        (data_dir / image_name).resolve(),
    ])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def validate_metadata(source_label: str, metadata: dict, data_dir: Path) -> list[str]:
    """Validate a single metadata file. Returns list of issues."""
    issues = []
    if "_load_error" in metadata:
        return [f"ERROR loading {source_label}: {metadata['_load_error']}"]
    if "label_equivalent" in metadata:
        for field in PAIR_REQUIRED_FIELDS:
            if field not in metadata:
                issues.append(f"MISSING pair field: {field}")
        for key in ("image1", "image2"):
            name = metadata.get(key)
            if isinstance(name, str) and name.strip():
                img_path = resolve_image_path(data_dir, metadata, name.strip())
                if img_path is None:
                    issues.append(f"MISSING image: {name}")
        return issues
    kt = metadata.get("knotType", "")

    # Skip excluded types
    if kt in _EXCLUDED_TYPES:
        return [f"SKIP: excluded type '{kt}'"]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in metadata:
            issues.append(f"MISSING field: {field}")

    # Check known type
    if kt and kt not in KNOWN_KNOT_TYPES:
        issues.append(f"UNKNOWN knotType: '{kt}'")

    # Link-specific checks
    if kt in _LINK_TYPES:
        if not metadata.get("isLink", False):
            issues.append(f"INCONSISTENT: knotType='{kt}' but isLink is not True")
        for field in LINK_REQUIRED_FIELDS:
            if field not in metadata:
                issues.append(f"MISSING link field: {field}")

    # Check isKnot consistency
    is_knot = metadata.get("isKnot")
    unknot_types = {"unknot", "twisted_ring", "spiral_disk", "kinky_unknot", "ring_like_open_rope"}
    if kt in unknot_types and is_knot is True:
        issues.append(f"INCONSISTENT: knotType='{kt}' should have isKnot=false")
    knot_types = {
        "trefoil", "figure8", "loose_open_knot", "ring_like_open_knot", "loose_cinquefoil", "occluded_knot",
        "torus_2_5", "torus_2_7", "torus_2_9", "torus_3_4", "torus_3_5",
    }
    if kt in knot_types and is_knot is not True:
        issues.append(f"INCONSISTENT: knotType='{kt}' should have isKnot=true")

    # Check images exist
    images = metadata.get("images", [])
    if not images:
        issues.append("NO images listed")
    else:
        for img in images:
            name = img.get("filename") if isinstance(img, dict) else img
            if name:
                img_path = resolve_image_path(data_dir, metadata, str(name))
                if img_path is None:
                    issues.append(f"MISSING image: {name}")

    # Check ground truth is computable for all applicable tasks
    for task_id in TASKS:
        if sample_applicable(metadata, task_id):
            if task_id == "T04_link_property":
                ring_colors = metadata.get("ringColors") or []
                if not ring_colors:
                    issues.append("GT=None for applicable task T04_link_property")
                    continue
                gt_metadata = dict(metadata)
                gt_metadata["removed_color"] = ring_colors[0]
                gt = get_ground_truth(gt_metadata, task_id)
            else:
                gt = get_ground_truth(metadata, task_id)
            if gt is None:
                issues.append(f"GT=None for applicable task {task_id}")

    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate dataset metadata")
    parser.add_argument("--data_dir", default="./dataset")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    records = load_dataset_records(data_dir)

    print(f"Found {len(records)} metadata records in {data_dir}\n")

    total_issues = 0
    total_ok = 0
    total_skipped = 0
    type_counter = Counter()
    difficulty_counter = Counter()
    task_coverage = Counter()

    for source_label, metadata in records:
        issues = validate_metadata(source_label, metadata, data_dir)
        kt = metadata.get("knotType", "unknown")
        type_counter[kt] += 1

        if issues and issues[0].startswith("SKIP:"):
            total_skipped += 1
            continue

        diff = compute_difficulty(metadata)
        difficulty_counter[diff["level"]] += 1

        # Count task coverage
        for task_id in TASKS:
            if sample_applicable(metadata, task_id):
                task_coverage[task_id] += 1

        if issues:
            label = metadata.get("id") or Path(source_label).name
            print(f"  [{kt}] {label}")
            for issue in issues:
                print(f"    - {issue}")
            total_issues += len(issues)
        else:
            total_ok += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total files:   {len(records)}")
    print(f"  OK:            {total_ok}")
    print(f"  Skipped:       {total_skipped}")
    print(f"  Issues found:  {total_issues}")

    print(f"\n-- Knot Type Distribution --")
    for kt, count in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {kt:25s}: {count}")

    print(f"\n-- Difficulty Distribution --")
    for level in ("easy", "medium", "hard"):
        print(f"  {level:10s}: {difficulty_counter[level]}")

    print(f"\n-- Task Coverage (samples per task) --")
    for task_id in TASKS:
        count = task_coverage.get(task_id, 0)
        flag = " ⚠️ LOW" if count < 3 else ""
        print(f"  {task_id:30s}: {count}{flag}")

    if total_issues > 0:
        print(f"\n⚠  {total_issues} issues found. Fix before running benchmark.")
        return 1
    else:
        print(f"\n✓  All metadata validated successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
