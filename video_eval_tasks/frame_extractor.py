"""Sample evenly spaced frames from an mp4 via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List


class FrameExtractionError(RuntimeError):
    pass


def _probe_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return 0.0


def extract_frames(video_path: Path, *, out_dir: Path, num_frames: int) -> List[Path]:
    if num_frames < 1:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(video_path)
    timestamps = (
        [duration * (index + 1) / (num_frames + 1) for index in range(num_frames)]
        if duration > 0
        else [0.0]
    )
    paths: List[Path] = []
    for index, timestamp in enumerate(timestamps):
        path = out_dir / f"frame_{index:02d}.png"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                    "-i", str(video_path), "-frames:v", "1", "-update", "1",
                    "-y", str(path),
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
        paths.append(path)
    if not paths:
        raise FrameExtractionError(f"No frames could be extracted from {video_path}")
    return paths
