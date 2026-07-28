"""Small common surface for the supported video-generation backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig


@dataclass
class VideoGenResult:
    video_path: Path
    upstream_response: Dict[str, Any]
    api_error: bool = False
    api_error_message: Optional[str] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None
    latency_seconds: Optional[float] = None


def build_video_gen_client(video_cfg: DictConfig):
    backend = str(getattr(video_cfg, "backend", "ark") or "ark").lower()
    if backend not in {"ark", "volcengine"}:
        raise ValueError(
            f"Minimal video pipeline supports only backend='ark', got {backend!r}."
        )
    from video_gen_client_ark import build_ark_video_gen_client

    return build_ark_video_gen_client(video_cfg)
