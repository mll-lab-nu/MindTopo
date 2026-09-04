from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Sequence

from ..types import EpisodeSpec


class TaskAdapter(ABC):
    task_name: str
    legacy_task: str

    def evaluate(
        self,
        legacy: Any,
        spec: EpisodeSpec,
        artifact_root: Path,
        sampling_mode: str,
        overlay_failures_only: bool,
    ) -> Dict[str, Any]:
        return legacy.evaluate_episode(
            self.task_name,
            spec,
            artifact_root,
            sampling_mode,
            overlay_failures_only,
        )

    @staticmethod
    def initial_state(spec: EpisodeSpec) -> Dict[str, Any]:
        return dict(spec.metadata.get("initial_state") or {})


def _registry() -> Dict[str, TaskAdapter]:
    from .chat_noir import ChatNoirAdapter
    from .one_stroke import OneStrokeAdapter
    from .pipe import PipeAdapter
    from .swap2d import Swap2DAdapter
    from .untangle import UntangleAdapter

    adapters: Sequence[TaskAdapter] = (
        UntangleAdapter(),
        OneStrokeAdapter(),
        Swap2DAdapter(),
        PipeAdapter(),
        ChatNoirAdapter(),
    )
    return {adapter.task_name: adapter for adapter in adapters}


def adapter_for(task_name: str) -> TaskAdapter:
    try:
        return _registry()[task_name]
    except KeyError as exc:
        raise ValueError(f"unsupported video metric task: {task_name}") from exc


def load_detection(artifact_root: Path, episode_id: str) -> Dict[str, Any]:
    path = artifact_root / "detections" / f"{episode_id}_per_frame_detection.json"
    return json.loads(path.read_text(encoding="utf-8"))
