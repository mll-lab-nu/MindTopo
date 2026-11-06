from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


READABLE_PARSER_STATUSES = frozenset({"ok", "partial", "degraded", "invalid_state", ""})


@dataclass(frozen=True)
class EpisodeSpec:
    """One generated video and the metadata needed to evaluate it."""

    episode_id: str
    source_episode_id: str
    task_name: str
    video_path: Optional[Path]
    video_rel_path: Optional[str]
    difficulty: str
    seed: int
    metadata: Dict[str, Any]
    model_id: Optional[str]
    vlm_text_model_id: Optional[str]
    video_gen_model_name: Optional[str]
    input_error: Optional[str] = None


@dataclass
class EpisodeResult:
    episode_id: str
    source_episode_id: str
    task: str
    difficulty: str
    seed: int
    source_video: Optional[str]
    model_id: Optional[str]
    vlm_text_model_id: Optional[str]
    video_gen_model_name: Optional[str]
    scorable: bool
    task_success: bool
    static_valid: Optional[bool]
    dynamic_valid: Optional[bool]
    overall_valid: bool
    confidence: Optional[float]
    verdict: Dict[str, str]
    evidence: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metric_breakdown: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
