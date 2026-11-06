from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .io import TASK_ALIASES


def import_manifest(manifest: Path, output_run_dir: Path) -> Path:
    """Convert an explicit loose-video JSONL manifest into a standard run.

    Each row must contain ``id``, ``task_name`` (or ``meta_info.task_name``),
    and ``video``. IDs absent from the repository's source questions also need
    complete task-specific oracle metadata in ``meta_info``; the run loader
    reports omissions as per-episode input errors. Video paths are resolved
    relative to the manifest and copied into the run.
    """
    manifest = manifest.resolve()
    target = output_run_dir.resolve()
    broad_targets = {Path("/"), Path.home().resolve(), Path(__file__).resolve().parents[2]}
    if target in broad_targets:
        raise ValueError(f"refusing to create a run at broad path: {target}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    answers: List[Dict[str, Any]] = []
    seen: set[str] = set()
    try:
        for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
            episode_id = str(row.get("id") or "")
            if not episode_id or episode_id in seen:
                raise ValueError(f"missing or duplicate id at manifest line {line_number}: {episode_id!r}")
            seen.add(episode_id)
            meta = {**dict((row.get("question") or {}).get("meta_info") or {}), **dict(row.get("meta_info") or {})}
            task_raw = str(row.get("task_name") or meta.get("task_name") or "")
            task = TASK_ALIASES.get(task_raw)
            if task is None:
                raise ValueError(f"unknown task at manifest line {line_number}: {task_raw!r}")
            source = Path(str(row.get("video") or ""))
            if not source.is_absolute():
                source = (manifest.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"manifest video does not exist at line {line_number}: {source}")
            suffix = source.suffix.lower() or ".mp4"
            safe_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in episode_id)
            rel_video = Path("videos") / f"{safe_id}{suffix}"
            destination = staging / rel_video
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            answers.append(
                {
                    "id": episode_id,
                    "type": "imported_video",
                    "meta_info": {**meta, "task_name": task},
                    "trajectory": [{"step_index": 0, "imagined_video": rel_video.as_posix()}],
                }
            )
        (staging / "model_answer.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answers), encoding="utf-8"
        )
        (staging / "import_manifest.json").write_text(
            json.dumps({"schema_version": "video_manifest_import/v1", "source_name": manifest.name, "episode_count": len(answers)}, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
        return target
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
