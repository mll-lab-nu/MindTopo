"""MP4 decoding with imageio and its bundled ffmpeg.

The existing frame_extractor.py relies on the ffmpeg CLI (not available on this
machine), so this module uses imageio instead.
Returns BGR frames (matching cv2, for consistent downstream processing).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import imageio.v3 as iio
import numpy as np


def decode_bgr(
    mp4_path: Path,
    frame_indices: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Stream a video and return all frames or selected frames in BGR order."""
    if frame_indices is not None:
        indices = list(frame_indices)
        if any(index < 0 for index in indices):
            raise ValueError("frame indices must be non-negative")
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise ValueError("frame indices must be strictly increasing")
        if not indices:
            return []
    else:
        indices = None

    frames: List[np.ndarray] = []
    target_position = 0
    for source_index, frame_rgb in enumerate(iio.imiter(mp4_path)):
        if indices is not None:
            target_index = indices[target_position]
            if source_index < target_index:
                continue
        frames.append(frame_rgb[:, :, ::-1].copy())
        if indices is not None:
            target_position += 1
            if target_position == len(indices):
                break

    if indices is not None and len(frames) != len(indices):
        raise ValueError(
            f"video ended before requested frame {indices[len(frames)]}: {mp4_path}"
        )
    return frames


def sample_indices(n_frames: int, k: int) -> List[int]:
    """Pick k evenly spaced frames out of n (including the first and last), for lightweight sampling."""
    if n_frames <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]
