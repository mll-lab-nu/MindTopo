from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

PromptBlock = Dict[str, Any]
Message = Dict[str, Any]


@dataclass
class EpisodeSpec:
    episode_id: str
    level: Any
    seed: Any
    repeat_index: int
    category: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    reset_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SavedImage:
    ref_id: str
    label: str
    abs_path: Path
    rel_path: str


@dataclass
class TurnContext:
    obs_rgb: Any
    planner_state: Any
    symbolic_observation: Any
    legal_actions: List[Any]
    step_index: int
    step_budget: Optional[int]
    termination_state: Optional[str]
    image_views: List[Dict[str, Any]]
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptParts:
    model_blocks: List[PromptBlock]
    debug_blocks: Optional[List[PromptBlock]] = None
    pure_blocks: Optional[List[PromptBlock]] = None


@dataclass
class ParsedActionResult:
    parsed_action: Any = None
    raw_response_text: str = ""
    response_debug: Dict[str, Any] = field(default_factory=dict)
    invalid_response: bool = False


@dataclass
class TurnInput:
    spec: EpisodeSpec
    context: TurnContext
    history: Sequence["TurnRecord"]
    saved_images: List[SavedImage]


@dataclass
class TurnRecord:
    spec: EpisodeSpec
    step_index: int
    context: TurnContext
    saved_images: List[SavedImage]
    prompt_parts: PromptParts
    parsed_action: Any
    flattened_answer: Any
    raw_response_text: str
    api_error: bool
    api_error_message: Optional[str]
    invalid_response: bool
    illegal: bool
    reward: float
    terminated: bool
    truncated: bool
    reason: Optional[str]
    response_debug: Dict[str, Any] = field(default_factory=dict)
    next_info: Dict[str, Any] = field(default_factory=dict)
    after_images: List[SavedImage] = field(default_factory=list)


@dataclass
class EpisodeRunResult:
    spec: EpisodeSpec
    success: bool
    terminated: bool
    truncated: bool
    total_steps: int
    illegal_moves: int
    invalid_responses: int
    api_errors: int
    final_reward: float
    final_reason: Optional[str]
    turns: List[TurnRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)
    skip_reason: Optional[str] = None

    @property
    def first_turn(self) -> Optional[TurnRecord]:
        return self.turns[0] if self.turns else None


@dataclass
class ModelResponse:
    raw_response_text: str
    api_error: bool
    api_error_message: Optional[str]
    response_debug: Dict[str, Any] = field(default_factory=dict)


class PlanningEnvAdapter(Protocol):
    def env_name(self) -> str:
        ...

    def make_env(self, cfg: Any) -> Any:
        ...

    def enumerate_episodes(self, cfg: Any) -> List[EpisodeSpec]:
        ...

    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: Any) -> Dict[str, Any]:
        ...

    def extract_turn_context(
        self,
        env: Any,
        obs: Any,
        info: Dict[str, Any],
        step_index: int,
        history: Sequence[TurnRecord],
    ) -> TurnContext:
        ...

    def capture_step_images(
        self,
        env: Any,
        spec: EpisodeSpec,
        turn: TurnInput,
        writer: Any,
    ) -> List[SavedImage]:
        ...

    def capture_post_step_images(
        self,
        env: Any,
        spec: EpisodeSpec,
        turn: TurnInput,
        obs: Any,
        info: Dict[str, Any],
        writer: Any,
    ) -> List[SavedImage]:
        ...

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        ...

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        ...

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        ...

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        ...

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        ...

    def flatten_answer(self, parsed_action: Any) -> Any:
        ...

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        ...
