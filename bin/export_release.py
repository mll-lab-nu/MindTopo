#!/usr/bin/env python3
"""Materialize the paper's fixed release without regenerating scenes or labels."""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topobench_eval"))
from topobench_eval.datasets import REASONING_TASKS


def sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()
    paper, source = args.paper_root.resolve(), args.benchmark_root.resolve()
    manifests = paper / "arxiv/data"
    # paper_source_commit below must name a commit that contains these manifests.
    if subprocess.check_output(["git", "status", "--porcelain", "--", "arxiv/data"], cwd=paper, text=True).strip():
        raise SystemExit("Commit arxiv/data in the paper repository before exporting")
    target = ROOT / "data/manifests"
    target.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, tuple[Path, list[str]]] = {}
    asset_cache: dict[Path, Path] = {}
    assets: dict[str, dict] = {}
    clonefile = None
    if sys.platform == "darwin":
        clonefile = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).clonefile
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clonefile.restype = ctypes.c_int

    def read_source(name: str) -> tuple[Path, list[str]]:
        if name not in source_cache:
            if name.startswith("git:"):
                _, commit, relative = name.split(":", 2)
                text = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=source, text=True)
            else:
                relative, text = name, (source / name).read_text()
            source_cache[name] = (Path(relative).parent, text.splitlines())
        return source_cache[name]

    def copy_asset(path: Path, task: str, expected: str | None = None) -> Path:
        path = path.resolve()
        if path in asset_cache:
            destination = asset_cache[path]
            if expected and expected not in destination.name:
                raise ValueError(f"Image hash mismatch: {path}")
            return destination
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Image hash mismatch: {path}")
        destination = ROOT / "data/assets" / task / digest[:2] / (digest + path.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cloned = clonefile is not None and clonefile(os.fsencode(path), os.fsencode(destination), 0) == 0
            if not cloned:
                shutil.copy2(path, destination)
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Corrupt existing asset: {destination}")
        asset_cache[path] = destination
        assets[str(destination.relative_to(ROOT))] = {"sha256": digest, "bytes": path.stat().st_size}
        return destination

    def portable(row: dict, base: Path, output: Path, task: str, hashes: list[str] | None = None) -> dict:
        row = copy.deepcopy(row)
        original = row.get("images", [])
        row["images"] = [os.path.relpath(copy_asset(base / image, task, hashes[i] if hashes else None), output)
                         for i, image in enumerate(original)]
        meta = row.setdefault("meta_info", {})
        if "config" in meta:
            metadata = ROOT / "environments" / task / "metadata.json"
            if metadata.is_file():
                meta["config"] = os.path.relpath(metadata, output)
        return row

    for split in ("core", "extended"):
        manifest = manifests / f"{split}_manifest.jsonl"
        entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        grouped: dict[str, list[dict]] = {}
        for entry in entries:
            base, lines = read_source(entry["source"])
            original = json.loads(lines[entry["source_line"] - 1])
            if sha(original) != entry["record_sha256"]:
                raise ValueError(f"Source record changed: {entry['source']}:{entry['source_line']}")
            output = ROOT / "data" / split / entry["task"]
            row = portable(original, source / base, output, entry["task"], entry["image_sha256"])
            row["id"] = entry["id"]
            meta = row.setdefault("meta_info", {})
            meta.update(benchmark_split=split, release_record_sha256=entry["record_sha256"],
                        release_input_sha256=entry["input_sha256"])
            if entry.get("level"):
                meta["benchmark_level"] = entry["level"]
            grouped.setdefault(entry["task"], []).append(row)
        for task, rows in grouped.items():
            output = ROOT / "data" / split / task
            output.mkdir(parents=True, exist_ok=True)
            (output / "question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        shutil.copy2(manifest, target / manifest.name)
        print(f"Exported {split}: {len(entries)} records", flush=True)

    # Preserve the public snapshot's existing probing question text, IDs and
    # difficulty labels. Only relocate its image assets to portable paths.
    for task in ("continuity_2d_maze", "enclosure_sheep", "continuity_pipe",
                 "separation_one_stroke", "knots_untangle"):
        original_path = ROOT / "environments" / task / "output/question.jsonl"
        output = ROOT / "data/probing" / task
        rows = []
        for line in original_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            base = original_path.parent
            if row.get("images") and not (base / row["images"][0]).is_file():
                base = source / "environments" / task / "output_arxiv"
            row = portable(row, base, output, task)
            row.setdefault("meta_info", {})["dataset_protocol"] = "legacy_probing"
            rows.append(row)
        output.mkdir(parents=True, exist_ok=True)
        (output / "question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    release = json.loads((manifests / "release_counts.json").read_text())
    release["paper_manifest_sha256"] = {split: hashlib.sha256((manifests / f"{split}_manifest.jsonl").read_bytes()).hexdigest() for split in ("core", "extended")}
    release["paper_source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paper, text=True).strip()
    (target / "release_counts.json").write_text(json.dumps(release, indent=2) + "\n")
    (target / "assets.json").write_text(json.dumps(assets, indent=2, sort_keys=True) + "\n")
    print(f"Materialized {len(assets)} content-addressed images; no absolute symlinks", flush=True)


if __name__ == "__main__":
    main()
