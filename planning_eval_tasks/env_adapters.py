from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig

from evaluator_utils import REPO_ROOT, resolve_repo_path
from topobench_eval.structured_output import strict_int
from unified_types import (
    EpisodeRunResult,
    EpisodeSpec,
    ParsedActionResult,
    PlanningEnvAdapter,
    PromptParts,
    SavedImage,
    TurnContext,
    TurnInput,
    TurnRecord,
)


def _coerce_int(value: Any) -> Optional[int]:
    return strict_int(value)


def _flatten_answer_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "answer" in value and len(value) == 1:
            return _flatten_answer_value(value["answer"])
        if "slot_index" in value and len(value) == 1:
            return _flatten_answer_value(value["slot_index"])
        if "cell_index" in value and len(value) == 1:
            return _flatten_answer_value(value["cell_index"])
        if "action_index" in value and len(value) == 1:
            return _flatten_answer_value(value["action_index"])
        if {"src_row", "src_col", "tgt_row", "tgt_col"} <= set(value.keys()) and len(value) == 4:
            return [
                _flatten_answer_value(value["src_row"]),
                _flatten_answer_value(value["src_col"]),
                _flatten_answer_value(value["tgt_row"]),
                _flatten_answer_value(value["tgt_col"]),
            ]
        if len(value) == 1:
            return _flatten_answer_value(next(iter(value.values())))
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, tuple):
        return [_flatten_answer_value(item) for item in value]
    if isinstance(value, list):
        return [_flatten_answer_value(item) for item in value]
    return value


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(ref_id: str) -> Dict[str, Any]:
    return {"type": "image_ref", "ref_id": ref_id}


def _current_task_text(text: str) -> str:
    """Remove stable instructions already supplied through the system prompt."""
    marker = "[Current Task]"
    _, found, suffix = str(text).partition(marker)
    if not found:
        raise ValueError(f"Initial planning prompt is missing the {marker!r} boundary.")
    return f"{found}{suffix}"


def _prompt_parts(
    *,
    model_blocks: List[Dict[str, Any]],
    debug_blocks: Optional[List[Dict[str, Any]]] = None,
    pure_blocks: Optional[List[Dict[str, Any]]] = None,
) -> PromptParts:
    return PromptParts(
        model_blocks=model_blocks,
        debug_blocks=debug_blocks if debug_blocks is not None else list(model_blocks),
        pure_blocks=pure_blocks if pure_blocks is not None else list(model_blocks),
    )


def _saved_image(turn: TurnInput, label: str) -> SavedImage:
    for image in turn.saved_images:
        if image.label == label:
            return image
    raise KeyError(f"Missing saved image {label!r} for episode {turn.spec.episode_id}")


def _cfg_output_dir(cfg: DictConfig) -> Path:
    return resolve_repo_path(str(cfg.run.output_dir))


def _config_relpath(cfg: DictConfig, env_name: str) -> str:
    output_dir = _cfg_output_dir(cfg)
    metadata_path = REPO_ROOT / "environments" / env_name / "metadata.json"
    if metadata_path.is_relative_to(output_dir):
        return str(metadata_path.relative_to(output_dir))
    return str(metadata_path.relative_to(REPO_ROOT))


def _cfg_optional_int(value: Any) -> Optional[int]:
    if value in (None, "null"):
        return None
    return int(value)


def _source_question_jsonl_relpath(env_name: str) -> str:
    return f"environments/{env_name}/output/question.jsonl"


class _BenchmarkAdapterBase:
    env_dir_name: str = ""
    cleanup_modules: Sequence[str] = (
        "internvl3_benchmark",
        "env",
        "internvl3_config",
        "solver",
        "level_data",
        "vite_server",
        "oracle_solver",
    )

    def __init__(self) -> None:
        self._benchmark: Any = None
        self._cached_specs: Optional[List[EpisodeSpec]] = None
        self._include_legal_moves_in_prompt = True

    def _configure_run_options(self, cfg: DictConfig) -> None:
        self._include_legal_moves_in_prompt = bool(getattr(cfg.run, "include_legal_moves_in_prompt", True))

    def _load_benchmark(self) -> Any:
        if self._benchmark is not None:
            return self._benchmark
        backend_dir = REPO_ROOT / "environments" / self.env_dir_name / "backend"
        gym_dir = REPO_ROOT / "environments" / self.env_dir_name / "gym"
        benchmark_path = backend_dir / "internvl3_benchmark.py"
        existing = sys.modules.get("internvl3_benchmark")
        existing_path = Path(getattr(existing, "__file__", "")) if existing is not None else None
        if existing_path and existing_path.resolve() != benchmark_path.resolve():
            for module_name in self.cleanup_modules:
                sys.modules.pop(module_name, None)
        for candidate in (str(backend_dir), str(gym_dir), str(backend_dir.parent), str(REPO_ROOT)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        importlib.invalidate_caches()
        self._benchmark = importlib.import_module("internvl3_benchmark")
        return self._benchmark

    def source_question_jsonl_relpath(self, cfg: Optional[DictConfig] = None) -> str:
        # Per-run override: env yaml or CLI sets `env.source_question_jsonl=...`
        # to point at a curated subset (e.g. interleaved_dataset_video20/<env>/
        # question.jsonl). Defaults to the env's bench output if unset.
        if cfg is not None:
            override = getattr(getattr(cfg, "env", None), "source_question_jsonl", None)
            if override is not None and str(override).strip() and str(override) != "null":
                return str(override)
        return _source_question_jsonl_relpath(self.env_dir_name)

    def source_question_jsonl_path(self, cfg: Optional[DictConfig] = None) -> Path:
        rel = self.source_question_jsonl_relpath(cfg)
        path = Path(rel) if Path(rel).is_absolute() else (REPO_ROOT / rel)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing question JSONL: {path}. "
                f"Generate {self.env_dir_name} samples or fix env.source_question_jsonl before running."
            )
        return path

    def _prepare_question_specs(self, cfg: DictConfig, specs: List[EpisodeSpec]) -> List[EpisodeSpec]:
        self._configure_run_options(cfg)
        source_rel = self.source_question_jsonl_relpath(cfg)
        for spec in specs:
            spec.metadata.setdefault("config_rel", _config_relpath(cfg, self.env_dir_name))
            spec.metadata.setdefault("source_question_jsonl", source_rel)
        self._cached_specs = specs
        return specs

    def _require_cached_specs(self, cfg: DictConfig) -> List[EpisodeSpec]:
        if self._cached_specs is None:
            return self.enumerate_episodes(cfg)  # type: ignore[attr-defined]
        return self._cached_specs

    def capture_post_step_images(
        self,
        env: Any,
        spec: EpisodeSpec,
        turn: TurnInput,
        obs: Any,
        info: Dict[str, Any],
        writer: Any,
    ) -> List[SavedImage]:
        if not bool(info.get("_topobench_episode_done", False)):
            return []
        try:
            final_context = self.extract_turn_context(
                env=env,
                obs=obs,
                info=info,
                step_index=int(turn.context.step_index) + 1,
                history=turn.history,
            )
            final_turn = TurnInput(
                spec=spec,
                context=final_context,
                history=turn.history,
                saved_images=[],
            )
            captured = self.capture_step_images(env=env, spec=spec, turn=final_turn, writer=writer)
        except Exception:
            return []
        return [
            SavedImage(
                ref_id=f"{image.ref_id}:after",
                label="after",
                abs_path=image.abs_path,
                rel_path=image.rel_path,
            )
            for image in captured
            if image.label != "goal"
        ]


class KnotsAdapter(_BenchmarkAdapterBase, PlanningEnvAdapter):
    env_dir_name = "knots_untangle"

    def env_name(self) -> str:
        return "knots_untangle"

    def make_env(self, cfg: DictConfig) -> Any:
        specs = self._require_cached_specs(cfg)
        return self.make_env_for_episode(cfg, specs[0])

    def _difficulty_for_spec(self, spec: EpisodeSpec) -> str:
        return str(spec.metadata.get("difficulty", spec.level))

    def env_cache_key(self, cfg: DictConfig, spec: EpisodeSpec) -> tuple[str, int]:
        bench = self._load_benchmark()
        difficulty = self._difficulty_for_spec(spec)
        step_budget = self._resolve_step_budget(cfg=cfg, difficulty=difficulty, bench=bench, spec=spec)
        return (difficulty, step_budget)

    def make_env_for_episode(self, cfg: DictConfig, spec: EpisodeSpec) -> Any:
        self._configure_run_options(cfg)
        bench = self._load_benchmark()
        difficulty = self._difficulty_for_spec(spec)
        step_budget = self._resolve_step_budget(cfg=cfg, difficulty=difficulty, bench=bench, spec=spec)
        frontend_dir = cfg.env.frontend_dir
        return bench.KnotsUntangleEnv(
            difficulty=difficulty,
            frontend_dir=None if frontend_dir in (None, "null") else str(resolve_repo_path(frontend_dir)),
            headless=bool(cfg.env.headless),
            animate=bool(cfg.env.animate),
            illegal_reward=float(cfg.env.illegal_reward),
            move_reward=float(cfg.env.move_reward),
            step_penalty=float(cfg.env.step_penalty),
            win_reward=float(cfg.env.win_reward),
            physics_frames_per_step=int(cfg.env.physics_frames_per_step),
            max_steps=step_budget,
            viewport_width=int(cfg.env.viewport_width),
            viewport_height=int(cfg.env.viewport_height),
        )

    def _resolve_step_budget(self, *, cfg: DictConfig, difficulty: str, bench: Any, spec: Optional[EpisodeSpec] = None) -> int:
        configured = _cfg_optional_int(cfg.run.step_budget)
        if configured is not None:
            return configured
        if spec is not None and spec.metadata.get("max_steps") is not None:
            return int(spec.metadata["max_steps"])
        defaults = getattr(bench, "DEFAULT_MAX_STEPS_BY_DIFFICULTY", {})
        if difficulty in defaults:
            return int(defaults[difficulty])
        return 15

    def enumerate_episodes(self, cfg: DictConfig) -> List[EpisodeSpec]:
        bench = self._load_benchmark()
        setups = bench.load_benchmark_setups_from_question_jsonl(self.source_question_jsonl_path(cfg))
        specs: List[EpisodeSpec] = []
        for setup in setups:
            reset_config = dict(setup.reset_config or {})
            seed = int(reset_config.get("seed", setup.seed))
            difficulty = str(reset_config.get("difficulty", setup.difficulty))
            metadata = {"difficulty": difficulty}
            if setup.max_steps is not None:
                metadata["max_steps"] = int(setup.max_steps)
            reset_kwargs: Dict[str, Any] = {"seed": seed}
            if reset_config:
                reset_kwargs["options"] = {"frontend_config": reset_config}
            specs.append(
                EpisodeSpec(
                    episode_id=str(setup.episode_id or f"knots_untangle_{difficulty}_seed_{seed}"),
                    level=difficulty,
                    seed=seed,
                    repeat_index=int(setup.repeat_index),
                    category=["knots", "knots_untangle", "interactive"],
                    metadata=metadata,
                    reset_kwargs=reset_kwargs,
                )
            )
        return self._prepare_question_specs(cfg, specs)

    # send seed to env
    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: DictConfig) -> Dict[str, Any]:
        del cfg
        return dict(spec.reset_kwargs)

    def extract_turn_context(self, env: Any, obs: Any, info: Dict[str, Any], step_index: int, history: Sequence[TurnRecord]) -> TurnContext:
        bench = self._load_benchmark()
        symbolic = info.get("symbolic_observation")
        legal_moves = bench.order_legal_moves_for_prompt(
            symbolic=symbolic,
            legal_moves=bench.compute_legal_moves(symbolic, int(env.grid_size)),
            action_history=[turn.parsed_action for turn in history if turn.parsed_action is not None],
            grid_size=int(env.grid_size),
            seed=int(info.get("seed", 0) or 0),
            step_index=step_index,
        )
        theoretical_min_steps: Optional[int] = None
        if step_index == 0:
            state = bench.symbolic_to_state(symbolic, int(env.grid_size))
            if state is not None:
                search = bench.bfs_shortest_plan(
                    state,
                    grid_size=int(env.grid_size),
                    max_depth=_coerce_int(env.max_steps),
                    max_expansions=50000,
                    timeout_seconds=2.0,
                )
                theoretical_min_steps = search.min_steps
        return TurnContext(
            obs_rgb=obs,
            planner_state=info.get("state"),
            symbolic_observation=symbolic,
            legal_actions=[dict(move) for move in legal_moves],
            step_index=step_index,
            step_budget=_coerce_int(env.max_steps),
            termination_state=info.get("reason"),
            image_views=[{"label": "current"}],
            prompt_metadata={
                "difficulty": getattr(env, "difficulty", "easy"),
                "grid_size": int(env.grid_size),
                "rope_count": int(env.rope_count),
                "crossings": bench._crossings_count(symbolic),
                "theoretical_min_steps": theoretical_min_steps,
            },
        )

    def capture_step_images(self, env: Any, spec: EpisodeSpec, turn: TurnInput, writer: Any) -> List[SavedImage]:
        bench = self._load_benchmark()
        png_bytes = bench.add_state_caption(env.screenshot(scene_only=True), f"State {turn.context.step_index}")
        return [
            writer.save_image(
                path_segments=["images", f"difficulty_{spec.metadata['difficulty']}", f"seed_{spec.seed:02d}"],
                filename=f"step_{turn.context.step_index:04d}_current.png",
                png_bytes=png_bytes,
                ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:current",
                label="current",
            )
        ]

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        prompt = bench.build_initial_prompt(
            include_legal_moves_in_prompt=self._include_legal_moves_in_prompt,
            difficulty=str(turn.context.prompt_metadata.get("difficulty", "easy")),
            grid_size=int(turn.context.prompt_metadata.get("grid_size", 0)),
            rope_count=int(turn.context.prompt_metadata.get("rope_count", 0)),
            crossings=turn.context.prompt_metadata.get("crossings"),
            symbolic=turn.context.symbolic_observation,
            legal_moves=turn.context.legal_actions,
            step_budget=int(turn.context.step_budget or 0),
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(_current_task_text(prompt)), _image_block(image.ref_id)])

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        prompt = bench.build_followup_prompt(
            step_index=int(turn.context.step_index),
            grid_size=int(turn.context.prompt_metadata.get("grid_size", 0)),
            rope_count=int(turn.context.prompt_metadata.get("rope_count", 0)),
            crossings=turn.context.prompt_metadata.get("crossings"),
            symbolic=turn.context.symbolic_observation,
            legal_moves=turn.context.legal_actions,
            include_legal_moves_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(prompt), _image_block(image.ref_id)])

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        bench = self._load_benchmark()
        parsed = bench.parse_action(raw_text, grid_size=int(turn.context.prompt_metadata.get("grid_size", 0)))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, invalid_response=parsed is None)

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        if policy_name not in {"oracle", "greedy"}:
            return None
        bench = self._load_benchmark()
        parsed, raw_text, api_error_message, response_debug = bench.choose_local_action(
            policy_name=policy_name,
            legal_moves=turn.context.legal_actions,
            symbolic=turn.context.symbolic_observation,
            grid_size=int(turn.context.prompt_metadata.get("grid_size", 0)),
            rng=random.Random(f"{turn.spec.episode_id}:{turn.context.step_index}:{policy_name}"),
            oracle_cache={},
            oracle_timeout_seconds=2.0,
            oracle_max_expansions=50000,
            step_budget=int(turn.context.step_budget or 0),
        )
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text if raw_text else (api_error_message or ""), response_debug=response_debug, invalid_response=parsed is None)

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        bench = self._load_benchmark()
        del turn
        return bench.INVALID_ACTION if parsed_action is None else bench.move_to_frontend_action(parsed_action)

    def flatten_answer(self, parsed_action: Any) -> Any:
        return _flatten_answer_value(parsed_action)

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        first_turn = result.first_turn
        return {
            "task_name": self.env_name(),
            "level": spec.level,
            "seed": spec.seed,
            "repeat_index": spec.repeat_index,
            "difficulty": spec.metadata["difficulty"],
            "initial_crossings": None if first_turn is None else first_turn.context.prompt_metadata.get("crossings"),
            "theoretical_min_steps": None if first_turn is None else first_turn.context.prompt_metadata.get("theoretical_min_steps"),
        }


class ContinuityPipeAdapter(_BenchmarkAdapterBase, PlanningEnvAdapter):
    env_dir_name = "continuity_pipe"

    def env_name(self) -> str:
        return "continuity_pipe"

    def make_env(self, cfg: DictConfig) -> Any:
        specs = self._require_cached_specs(cfg)
        return self.make_env_for_episode(cfg, specs[0])

    def make_env_for_episode(self, cfg: DictConfig, spec: EpisodeSpec) -> Any:
        self._configure_run_options(cfg)
        bench = self._load_benchmark()
        grid_size = int(spec.metadata.get("grid_size", 3))
        frontend_dir = getattr(cfg.env, "frontend_dir", None)
        return bench.ContinuityPipeEnv(
            grid_size=grid_size,
            frontend_dir=None if frontend_dir in (None, "null") else str(resolve_repo_path(frontend_dir)),
            host=str(cfg.env.host),
            port=int(cfg.env.port),
            headless=bool(cfg.env.headless),
            illegal_reward=float(cfg.env.illegal_reward),
            move_reward=float(cfg.env.move_reward),
            success_reward=float(cfg.env.success_reward),
            timeout_penalty=float(cfg.env.timeout_penalty),
            viewport_width=int(cfg.env.viewport_width),
            viewport_height=int(cfg.env.viewport_height),
        )

    def env_cache_key(self, cfg: DictConfig, spec: EpisodeSpec) -> tuple[int]:
        del cfg
        return (int(spec.metadata.get("grid_size", 3)),)

    def enumerate_episodes(self, cfg: DictConfig) -> List[EpisodeSpec]:
        bench = self._load_benchmark()
        loaded = bench.load_episode_specs(self.source_question_jsonl_path(cfg))
        specs: List[EpisodeSpec] = []
        for item in loaded:
            grid_size = int(item["grid_size"])
            reset_options = dict(item.get("reset_options") or {})
            specs.append(
                EpisodeSpec(
                    episode_id=str(item["episode_id"]),
                    level=f"grid_{grid_size}x{grid_size}",
                    seed=int(item["seed"]),
                    repeat_index=int(item["repeat_index"]),
                    category=["continuity", "continuity_pipe", "interactive"],
                    metadata={
                        "grid_size": grid_size,
                        "difficulty": str(item.get("difficulty", "loaded")),
                        "solution_steps_spec": str(item.get("solution_steps_spec", "loaded")),
                    },
                    reset_kwargs={"seed": int(item["seed"]), "options": reset_options},
                )
            )
        return self._prepare_question_specs(cfg, specs)

    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: DictConfig) -> Dict[str, Any]:
        reset_kwargs = dict(spec.reset_kwargs)
        options: Dict[str, Any] = dict(reset_kwargs.get("options") or {})
        frontend_config = dict(options.get("frontend_config") or {})
        if cfg.run.step_budget not in (None, "null"):
            options["max_steps"] = int(cfg.run.step_budget)
            frontend_config["maxSteps"] = int(cfg.run.step_budget)
        options["frontend_config"] = frontend_config
        reset_kwargs["seed"] = int(reset_kwargs.get("seed", spec.seed))
        reset_kwargs["options"] = options
        return reset_kwargs

    def extract_turn_context(self, env: Any, obs: Any, info: Dict[str, Any], step_index: int, history: Sequence[TurnRecord]) -> TurnContext:
        del history
        state = info.get("state") or env.get_state()
        grid_size = int(state.get("gridSize", getattr(env, "grid_size", 3)))
        solution_ticks = _coerce_int(info.get("solution_ticks", state.get("solutionTicks")))
        difficulty = str(info.get("difficulty") or state.get("difficulty") or "hard")
        return TurnContext(
            obs_rgb=obs,
            planner_state=state,
            symbolic_observation=info.get("symbolic_observation"),
            legal_actions=[int(value) for value in state.get("legalActionIndices", []) or []],
            step_index=step_index,
            step_budget=_coerce_int(info.get("max_steps", state.get("maxSteps"))),
            termination_state=info.get("reason") or info.get("truncated_reason"),
            image_views=[{"label": "current"}],
            prompt_metadata={
                "difficulty": difficulty,
                "grid_size": grid_size,
                "source_index": _coerce_int(state.get("sourceIndex")),
                "solution_ticks": solution_ticks,
                "connected_count": _coerce_int(state.get("connectedCount")),
                "total_pipes": _coerce_int(state.get("totalPipes")),
            },
        )

    def capture_step_images(self, env: Any, spec: EpisodeSpec, turn: TurnInput, writer: Any) -> List[SavedImage]:
        bench = self._load_benchmark()
        grid_size = int(turn.context.prompt_metadata["grid_size"])
        difficulty = str(turn.context.prompt_metadata["difficulty"])
        png_bytes = bench.add_state_caption(env.screenshot(scene_only=True), f"State {turn.context.step_index}")
        return [
            writer.save_image(
                path_segments=[
                    "images",
                    f"grid_{grid_size}x{grid_size}",
                    f"difficulty_{difficulty}",
                    f"repeat_{spec.repeat_index + 1}_seed_{spec.seed}",
                ],
                filename=f"step_{turn.context.step_index:04d}_current.png",
                png_bytes=png_bytes,
                ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:current",
                label="current",
            )
        ]

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        grid_size = int(turn.context.prompt_metadata["grid_size"])
        before_text, after_text = bench.build_initial_prompt_texts(
            grid_size=grid_size,
            legal_actions=[int(action) for action in turn.context.legal_actions],
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(_current_task_text(before_text)), _image_block(image.ref_id), _text_block(after_text)])

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        grid_size = int(turn.context.prompt_metadata["grid_size"])
        before_text, after_text = bench.build_followup_prompt_texts(
            grid_size=grid_size,
            legal_actions=[int(action) for action in turn.context.legal_actions],
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(before_text), _image_block(image.ref_id), _text_block(after_text)])

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        bench = self._load_benchmark()
        parsed = bench.parse_action(raw_text, int(turn.context.prompt_metadata["grid_size"]))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, invalid_response=parsed is None)

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        if policy_name != "oracle":
            return None
        bench = self._load_benchmark()
        parsed = bench.build_oracle_action(turn.context.planner_state or {})
        raw_text = "<no-legal-action>" if parsed is None else json.dumps({"answer": {"x": int(parsed[0]), "y": int(parsed[1])}}, separators=(",", ":"))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, response_debug={"mode": "oracle"}, invalid_response=parsed is None)

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        bench = self._load_benchmark()
        grid_size = int(turn.context.prompt_metadata["grid_size"])
        if parsed_action is None:
            return bench.INVALID_ACTION
        if isinstance(parsed_action, int):
            return int(parsed_action)
        if isinstance(parsed_action, list) and len(parsed_action) >= 2:
            return bench.xy_to_action(int(parsed_action[0]), int(parsed_action[1]), grid_size)
        if isinstance(parsed_action, dict):
            parsed = bench.parse_action(json.dumps(parsed_action), grid_size)
            if parsed is not None:
                return bench.xy_to_action(int(parsed[0]), int(parsed[1]), grid_size)
        flattened = _flatten_answer_value(parsed_action)
        return bench.INVALID_ACTION if flattened is None else int(flattened)

    def flatten_answer(self, parsed_action: Any) -> Any:
        return _flatten_answer_value(parsed_action) if parsed_action is not None else None

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        meta = result.first_turn.context.prompt_metadata if result.first_turn is not None else {}
        grid_size = meta.get("grid_size", spec.metadata.get("grid_size"))
        return {
            "task_name": self.env_name(),
            "level": f"grid_{grid_size}x{grid_size}" if grid_size is not None else spec.level,
            "seed": spec.seed,
            "repeat_index": spec.repeat_index,
            "difficulty": meta.get("difficulty", spec.metadata.get("difficulty")),
            "grid_size": grid_size,
            "source_index": meta.get("source_index"),
            "solution_ticks": meta.get("solution_ticks"),
            "max_steps": result.first_turn.context.step_budget if result.first_turn is not None else None,
            "total_pipes": meta.get("total_pipes"),
        }


class SeparationAdapter(_BenchmarkAdapterBase, PlanningEnvAdapter):
    env_dir_name = "separation_one_stroke"

    def env_name(self) -> str:
        return "separation_one_stroke"

    def make_env(self, cfg: DictConfig) -> Any:
        self._configure_run_options(cfg)
        bench = self._load_benchmark()
        return bench.SeparationOneStrokeEnv(
            headless=bool(cfg.env.headless),
            max_steps=None if cfg.run.step_budget in (None, "null") else int(cfg.run.step_budget),
            port=int(cfg.env.port),
            host=str(cfg.env.host),
        )

    def enumerate_episodes(self, cfg: DictConfig) -> List[EpisodeSpec]:
        bench = self._load_benchmark()
        setups = bench.load_benchmark_setups_from_question_jsonl(self.source_question_jsonl_path(cfg))
        specs: List[EpisodeSpec] = []
        for setup in setups:
            metadata = {
                "board_size": int(setup.board_size),
                "level_number": int(setup.level_number),
                "difficulty": str(setup.difficulty),
                "solution_length": int(setup.solution_length),
                "setup_id": str(setup.setup_id),
            }
            if setup.question_max_steps is not None:
                metadata["max_steps"] = int(setup.question_max_steps)
            specs.append(
                EpisodeSpec(
                    episode_id=str(setup.question_row_id),
                    level=int(setup.level_number),
                    seed=int(setup.question_seed),
                    repeat_index=int(setup.question_repeat_index),
                    category=["separation", "separation_one_stroke", "interactive"],
                    metadata=metadata,
                    reset_kwargs={
                        "seed": int(setup.question_seed),
                        "options": {"frontend_config": dict(setup.reset_config)},
                    },
                )
            )
        return self._prepare_question_specs(cfg, specs)

    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: DictConfig) -> Dict[str, Any]:
        reset_kwargs = dict(spec.reset_kwargs)
        options: Dict[str, Any] = dict(reset_kwargs.get("options") or {})
        if cfg.run.step_budget not in (None, "null"):
            options["max_steps"] = int(cfg.run.step_budget)
        reset_kwargs["seed"] = int(reset_kwargs.get("seed", spec.seed))
        reset_kwargs["options"] = options
        return reset_kwargs

    def extract_turn_context(self, env: Any, obs: Any, info: Dict[str, Any], step_index: int, history: Sequence[TurnRecord]) -> TurnContext:
        del history, env
        solution_length = int(info.get("referenceSolutionLength") or info.get("solutionLength") or 0)
        state = info.get("state") or {}
        width = _coerce_int(state.get("W"))
        height = _coerce_int(state.get("H"))
        board_size = (width - 1) if width and width == height else None
        if board_size == 4:
            difficulty = "easy"
        elif board_size == 5:
            difficulty = "medium"
        elif board_size == 6:
            difficulty = "hard"
        elif solution_length <= 0:
            difficulty = "hard"
        elif solution_length <= 6:
            difficulty = "easy"
        elif solution_length <= 8:
            difficulty = "medium"
        else:
            difficulty = "hard"
        return TurnContext(
            obs_rgb=obs,
            planner_state=state,
            symbolic_observation=info.get("symbolic_observation"),
            legal_actions=[{"direction": direction} for direction in info.get("legalDirections", [])],
            step_index=step_index,
            step_budget=_coerce_int(info.get("maxSteps")),
            termination_state=info.get("reason") or info.get("truncated_reason"),
            image_views=[{"label": "current"}],
            prompt_metadata={"difficulty": difficulty, "solution_length": solution_length, "width": width, "height": height, "num_colors": _coerce_int(info.get("numColors"))},
        )

    def capture_step_images(self, env: Any, spec: EpisodeSpec, turn: TurnInput, writer: Any) -> List[SavedImage]:
        bench = self._load_benchmark()
        board_size = int(spec.metadata.get("board_size") or 0)
        png_bytes = bench.add_state_caption(env.capture_png_bytes(), f"State {turn.context.step_index}")
        return [
            writer.save_image(
                path_segments=["images", f"size_{board_size}x{board_size}", f"seed_{int(spec.seed):03d}"],
                filename=f"step_{turn.context.step_index:04d}_current.png",
                png_bytes=png_bytes,
                ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:current",
                label="current",
            )
        ]

    def capture_post_step_images(
        self,
        env: Any,
        spec: EpisodeSpec,
        turn: TurnInput,
        obs: Any,
        info: Dict[str, Any],
        writer: Any,
    ) -> List[SavedImage]:
        del obs, info
        bench = self._load_benchmark()
        board_size = int(spec.metadata.get("board_size") or 0)
        next_state_index = int(turn.context.step_index) + 1
        png_bytes = bench.add_state_caption(env.capture_png_bytes(), f"State {next_state_index}")
        return [
            writer.save_image(
                path_segments=["images", f"size_{board_size}x{board_size}", f"seed_{int(spec.seed):03d}"],
                filename=f"step_{turn.context.step_index:04d}_after.png",
                png_bytes=png_bytes,
                ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:after",
                label="after",
            )
        ]

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        before_image, after_image = bench.build_interact_initial_user_prompt_parts(
            info={"legalDirections": [action["direction"] for action in turn.context.legal_actions]},
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        blocks = [_text_block(_current_task_text(before_image)), _image_block(image.ref_id), _text_block(after_image)]
        return _prompt_parts(model_blocks=blocks)

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        before_image, after_image = bench.build_interact_followup_user_prompt_parts(
            info={"legalDirections": [action["direction"] for action in turn.context.legal_actions]},
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        blocks = [_text_block(before_image), _image_block(image.ref_id), _text_block(after_image)]
        return _prompt_parts(model_blocks=blocks)

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        bench = self._load_benchmark()
        parsed = bench.parse_direction(raw_text)
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, invalid_response=parsed is None)

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        if policy_name != "oracle":
            return None
        bench = self._load_benchmark()
        info = {"state": turn.context.planner_state, "maxSteps": int(turn.context.step_budget or 0), "episode_steps": int(turn.context.step_index)}
        parsed = bench.build_oracle_direction(info)
        return ParsedActionResult(parsed_action=parsed, raw_response_text=json.dumps({"answer": parsed}, separators=(",", ":")) if parsed else "<no-solution>", response_debug={"mode": "oracle"}, invalid_response=parsed is None)

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        bench = self._load_benchmark()
        del turn
        flattened = _flatten_answer_value(parsed_action)
        direction = str(flattened).upper() if isinstance(flattened, str) else None
        return bench.INVALID_ACTION if direction not in getattr(bench, "ACTION_DIRECTIONS", ("U", "D", "L", "R")) else {"U": 0, "D": 1, "L": 2, "R": 3}[direction]

    def flatten_answer(self, parsed_action: Any) -> Any:
        return _flatten_answer_value(parsed_action)

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        meta = result.first_turn.context.prompt_metadata if result.first_turn is not None else {}
        return {"task_name": self.env_name(), "level": spec.level, "seed": spec.seed, "repeat_index": spec.repeat_index, "difficulty": spec.metadata.get("difficulty") or meta.get("difficulty", "hard"), "board_size": spec.metadata.get("board_size"), "width": meta.get("width"), "height": meta.get("height"), "num_colors": meta.get("num_colors"), "solution_length": meta.get("solution_length")}


class OrderSwap2DAdapter(_BenchmarkAdapterBase, PlanningEnvAdapter):
    env_dir_name = "order_swap_2d_puzzle"

    def env_name(self) -> str:
        return "order_swap_2d_puzzle"

    def make_env(self, cfg: DictConfig) -> Any:
        self._configure_run_options(cfg)
        bench = self._load_benchmark()
        specs = self._require_cached_specs(cfg)
        max_rows = max(int(spec.metadata["grid_rows"]) for spec in specs)
        max_cols = max(int(spec.metadata["grid_cols"]) for spec in specs)
        return bench.OrderSwap2DPuzzleEnv(
            grid_rows=max_rows,
            grid_cols=max_cols,
            headless=bool(cfg.env.headless),
            animate=bool(cfg.env.animate),
            max_steps=None,
            budget_multiplier=float(cfg.env.budget_multiplier),
            host=str(cfg.env.host),
            port=int(cfg.env.port),
            viewport_width=int(getattr(cfg.env, "viewport_width", 1440)),
            viewport_height=int(getattr(cfg.env, "viewport_height", 980)),
        )

    def enumerate_episodes(self, cfg: DictConfig) -> List[EpisodeSpec]:
        bench = self._load_benchmark()
        loaded = bench.load_episode_specs(self.source_question_jsonl_path(cfg))
        specs: List[EpisodeSpec] = []
        for item in loaded:
            grid_rows = int(item["grid_rows"])
            grid_cols = int(item["grid_cols"])
            specs.append(
                EpisodeSpec(
                    episode_id=str(item["episode_id"]),
                    level=f"grid_{grid_rows}x{grid_cols}",
                    seed=int(item["seed"]),
                    repeat_index=int(item["repeat_index"]),
                    category=["order", "order_swap_2d_puzzle", "interactive"],
                    metadata={
                        "grid_rows": grid_rows,
                        "grid_cols": grid_cols,
                        "difficulty": str(item.get("difficulty", "")),
                    },
                    reset_kwargs={
                        "seed": int(item["seed"]),
                        "options": dict(item.get("reset_options") or {}),
                    },
                )
            )
        return self._prepare_question_specs(cfg, specs)

    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: DictConfig) -> Dict[str, Any]:
        del cfg
        reset_kwargs = dict(spec.reset_kwargs)
        reset_kwargs["seed"] = int(reset_kwargs.get("seed", spec.seed))
        return reset_kwargs

    def extract_turn_context(self, env: Any, obs: Any, info: Dict[str, Any], step_index: int, history: Sequence[TurnRecord]) -> TurnContext:
        del history
        bench = self._load_benchmark()
        state = info.get("state") or {}
        grid_rows = int(state.get("gridRows", info.get("grid_rows", 0)) or 0)
        grid_cols = int(state.get("gridCols", info.get("grid_cols", 0)) or 0)
        legal_actions = bench.compute_legal_actions(state)
        return TurnContext(
            obs_rgb=obs,
            planner_state=state,
            symbolic_observation=info.get("symbolic_observation"),
            legal_actions=legal_actions,
            step_index=step_index,
            step_budget=_coerce_int(info.get("step_budget")) or _coerce_int(state.get("stepBudget")),
            termination_state=info.get("reason") or info.get("truncated_reason"),
            image_views=[{"label": "current"}, {"label": "goal"}],
            prompt_metadata={
                "difficulty": str(state.get("difficulty", info.get("difficulty", bench.difficulty_label_for_grid(grid_rows, grid_cols)))),
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "theoretical_min_steps": _coerce_int(info.get("theoretical_min_steps")) or _coerce_int(state.get("theoreticalMinSteps")),
            },
        )

    def capture_step_images(self, env: Any, spec: EpisodeSpec, turn: TurnInput, writer: Any) -> List[SavedImage]:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        grid_label = f"grid_{int(meta['grid_rows'])}x{int(meta['grid_cols'])}"
        path_segments = ["images", grid_label, f"repeat_{spec.repeat_index + 1}_seed_{spec.seed}"]
        current_png = bench.add_image_caption(env.screenshot_current_card(), f"State {turn.context.step_index}")
        goal_png = bench.add_image_caption(env.screenshot_goal_card(), "Goal State")
        return [
            writer.save_image(path_segments=path_segments, filename=f"step_{turn.context.step_index:04d}_current.png", png_bytes=current_png, ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:current", label="current"),
            writer.save_image(path_segments=path_segments, filename=f"step_{turn.context.step_index:04d}_goal.png", png_bytes=goal_png, ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:goal", label="goal"),
        ]

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        before_current, between_images, after_goal = bench.build_initial_prompt_texts(
            grid_rows=int(meta["grid_rows"]),
            grid_cols=int(meta["grid_cols"]),
            legal_actions=turn.context.legal_actions,
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        debug_before, debug_between, debug_after = bench.build_initial_prompt_texts(
            grid_rows=int(meta["grid_rows"]),
            grid_cols=int(meta["grid_cols"]),
            legal_actions=turn.context.legal_actions,
            include_difficulty=True,
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        current = _saved_image(turn, "current")
        goal = _saved_image(turn, "goal")
        model_blocks = [_text_block(_current_task_text(before_current)), _image_block(current.ref_id), _text_block(between_images), _image_block(goal.ref_id), _text_block(after_goal)]
        debug_blocks = [_text_block(_current_task_text(debug_before)), _image_block(current.ref_id), _text_block(debug_between), _image_block(goal.ref_id), _text_block(debug_after)]
        return _prompt_parts(model_blocks=model_blocks, debug_blocks=debug_blocks, pure_blocks=model_blocks)

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        before_current, between_images, after_goal = bench.build_followup_prompt_texts(
            grid_rows=int(meta["grid_rows"]),
            grid_cols=int(meta["grid_cols"]),
            legal_actions=turn.context.legal_actions,
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        debug_before, debug_between, debug_after = bench.build_followup_prompt_texts(
            grid_rows=int(meta["grid_rows"]),
            grid_cols=int(meta["grid_cols"]),
            legal_actions=turn.context.legal_actions,
            include_difficulty=True,
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        current = _saved_image(turn, "current")
        goal = _saved_image(turn, "goal")
        model_blocks = [_text_block(before_current), _image_block(current.ref_id), _text_block(between_images), _image_block(goal.ref_id), _text_block(after_goal)]
        debug_blocks = [_text_block(debug_before), _image_block(current.ref_id), _text_block(debug_between), _image_block(goal.ref_id), _text_block(debug_after)]
        return _prompt_parts(model_blocks=model_blocks, debug_blocks=debug_blocks, pure_blocks=model_blocks)

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        parsed = bench.parse_action(raw_text, int(meta["grid_rows"]), int(meta["grid_cols"]))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, invalid_response=parsed is None)

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        if policy_name != "oracle":
            return None
        bench = self._load_benchmark()
        parsed = bench.build_oracle_action(turn.context.planner_state or {})
        raw = "<no-solution>" if parsed is None else json.dumps({"answer": {"row": parsed["row"], "col": parsed["col"]}}, separators=(",", ":"))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw, response_debug={"mode": "oracle"}, invalid_response=parsed is None)

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        return bench.action_to_index(parsed_action, int(meta["grid_rows"]), int(meta["grid_cols"]))

    def flatten_answer(self, parsed_action: Any) -> Any:
        if not isinstance(parsed_action, dict):
            return None
        return {"row": int(parsed_action["row"]), "col": int(parsed_action["col"])}

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        meta = result.first_turn.context.prompt_metadata if result.first_turn is not None else {}
        return {
            "task_name": self.env_name(),
            "level": spec.level,
            "seed": spec.seed,
            "repeat_index": spec.repeat_index,
            "difficulty": meta.get("difficulty"),
            "grid_rows": meta.get("grid_rows"),
            "grid_cols": meta.get("grid_cols"),
            "theoretical_min_steps": meta.get("theoretical_min_steps"),
        }


class ChatNoirAdapter(_BenchmarkAdapterBase, PlanningEnvAdapter):
    env_dir_name = "enclosure_chat_noir"

    def env_name(self) -> str:
        return "enclosure_chat_noir"

    def make_env(self, cfg: DictConfig) -> Any:
        self._configure_run_options(cfg)
        bench = self._load_benchmark()
        specs = self._require_cached_specs(cfg)
        first_spec = specs[0]
        return bench.EnclosureChatNoirEnv(board_radius=int(first_spec.metadata["board_radius"]), initial_block_count=int(first_spec.metadata["initial_block_count"]), cat_policy=str(first_spec.metadata["cat_policy_name"]), headless=bool(cfg.env.headless), animate=bool(cfg.env.animate), host=str(cfg.env.host), port=int(cfg.env.port))

    def enumerate_episodes(self, cfg: DictConfig) -> List[EpisodeSpec]:
        bench = self._load_benchmark()
        loaded = bench.load_episode_specs(self.source_question_jsonl_path(cfg))
        specs: List[EpisodeSpec] = []
        for item in loaded:
            board_radius = int(item["board_radius"])
            initial_block_count = int(item["initial_block_count"])
            cat_policy_name = str(item["cat_policy_name"])
            metadata = {
                "board_radius": board_radius,
                "initial_block_count": initial_block_count,
                "cat_policy_name": cat_policy_name,
                "difficulty": str(item.get("difficulty", "")),
            }
            if item.get("max_steps") is not None:
                metadata["max_steps"] = int(item["max_steps"])
            specs.append(
                EpisodeSpec(
                    episode_id=str(item["episode_id"]),
                    level=f"radius_{board_radius}_blocked_{initial_block_count}_policy_{cat_policy_name}",
                    seed=int(item["seed"]),
                    repeat_index=int(item["repeat_index"]),
                    category=["enclosure", "enclosure_chat_noir", "interactive"],
                    metadata=metadata,
                    reset_kwargs={
                        "seed": int(item["seed"]),
                        "options": dict(item.get("reset_options") or {}),
                    },
                )
            )
        return self._prepare_question_specs(cfg, specs)

    def build_reset_kwargs(self, spec: EpisodeSpec, cfg: DictConfig) -> Dict[str, Any]:
        reset_kwargs = dict(spec.reset_kwargs)
        options: Dict[str, Any] = dict(reset_kwargs.get("options") or {})
        configured = _cfg_optional_int(cfg.run.step_budget)
        if configured is not None:
            options["max_steps"] = configured
        elif spec.metadata.get("max_steps") is not None:
            options["max_steps"] = int(spec.metadata["max_steps"])
        reset_kwargs["seed"] = int(reset_kwargs.get("seed", spec.seed))
        reset_kwargs["options"] = options
        return reset_kwargs

    def extract_turn_context(self, env: Any, obs: Any, info: Dict[str, Any], step_index: int, history: Sequence[TurnRecord]) -> TurnContext:
        del history
        bench = self._load_benchmark()
        state = info.get("state") or {}
        board_cell_count = int(state.get("boardCellCount", 0))
        step_budget = _coerce_int(info.get("maxSteps")) or max(64, board_cell_count * 8)
        cat_policy = state.get("catPolicy") or {}
        difficulty_info = info.get("difficulty") or {}
        difficulty_score = float(
            difficulty_info.get(
                "overall_score",
                difficulty_info.get("score", difficulty_info.get("level", difficulty_info.get("cat_intelligence", 0))),
            )
            or 0
        )
        return TurnContext(
            obs_rgb=obs,
            planner_state=state,
            symbolic_observation=info.get("symbolic_observation"),
            legal_actions=[{"cell_index": int(value)} for value in state.get("legalActionIndices", [])],
            step_index=step_index,
            step_budget=step_budget,
            termination_state=info.get("reason") or info.get("truncated_reason"),
            image_views=[{"label": "current"}],
            prompt_metadata={
                "difficulty": bench.difficulty_label_for_score(difficulty_score),
                "board_radius": int(state.get("boardRadius", 0)),
                "initial_block_count": int(state.get("initialBlockedCount", 0)),
                "cat_policy_name": str(cat_policy.get("name", info.get("cat_policy", {}).get("name", ""))),
                "cat_policy_description": str(cat_policy.get("description", "")),
                "cat_policy_level": int(info.get("cat_policy", {}).get("level", 0)),
                "difficulty_score": difficulty_score,
                "initial_escape_distance": _coerce_int(info.get("initial_escape_distance")),
                "current_board": getattr(env, "current_board", None),
            },
        )

    def capture_step_images(self, env: Any, spec: EpisodeSpec, turn: TurnInput, writer: Any) -> List[SavedImage]:
        bench = self._load_benchmark()
        png_bytes = bench.add_state_caption(env.screenshot(scene_only=True), f"State {turn.context.step_index}")
        return [
            writer.save_image(
                path_segments=["images", f"radius_{int(spec.metadata['board_radius'])}", f"blocked_{int(spec.metadata['initial_block_count'])}", f"policy_{spec.metadata['cat_policy_name']}", f"repeat_{spec.repeat_index + 1}_seed_{spec.seed}"],
                filename=f"step_{turn.context.step_index:04d}_current.png",
                png_bytes=png_bytes,
                ref_id=f"{spec.episode_id}:step_{turn.context.step_index}:current",
                label="current",
            )
        ]

    def build_initial_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        meta = turn.context.prompt_metadata
        before_text, after_text = bench.build_initial_prompt_texts(
            board_radius=int(meta["board_radius"]),
            cat_index=int((turn.context.planner_state or {}).get("catIndex", -1)),
            legal_actions=[int(action["cell_index"]) for action in turn.context.legal_actions],
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(_current_task_text(before_text)), _image_block(image.ref_id), _text_block(after_text)])

    def build_followup_prompt(self, turn: TurnInput) -> PromptParts:
        bench = self._load_benchmark()
        before_text, after_text = bench.build_followup_prompt_texts(
            cat_index=int((turn.context.planner_state or {}).get("catIndex", -1)),
            legal_actions=[int(action["cell_index"]) for action in turn.context.legal_actions],
            include_legal_actions_in_prompt=self._include_legal_moves_in_prompt,
        )
        image = _saved_image(turn, "current")
        return _prompt_parts(model_blocks=[_text_block(before_text), _image_block(image.ref_id), _text_block(after_text)])

    def parse_model_action(self, raw_text: str, turn: TurnInput) -> ParsedActionResult:
        bench = self._load_benchmark()
        parsed = bench.parse_action(raw_text, int((turn.context.planner_state or {}).get("boardCellCount", 0)))
        return ParsedActionResult(parsed_action=parsed, raw_response_text=raw_text, invalid_response=parsed is None)

    def choose_local_action(self, policy_name: str, turn: TurnInput) -> Optional[ParsedActionResult]:
        if policy_name != "oracle":
            return None
        bench = self._load_benchmark()
        parsed = bench.build_oracle_action(board=turn.context.prompt_metadata.get("current_board"), state=turn.context.planner_state or {})
        return ParsedActionResult(parsed_action=parsed, raw_response_text="<no-legal-action>" if parsed is None else json.dumps({"answer": {"cell_index": parsed}}, separators=(",", ":")), response_debug={"mode": "oracle"}, invalid_response=parsed is None)

    def to_env_action(self, parsed_action: Any, turn: TurnInput) -> Any:
        bench = self._load_benchmark()
        del turn
        flattened = _flatten_answer_value(parsed_action)
        return bench.INVALID_ACTION if flattened is None else int(flattened)

    def flatten_answer(self, parsed_action: Any) -> Any:
        return _flatten_answer_value(parsed_action) if parsed_action is not None else None

    def build_episode_meta(self, spec: EpisodeSpec, result: EpisodeRunResult) -> Dict[str, Any]:
        meta = result.first_turn.context.prompt_metadata if result.first_turn is not None else {}
        return {"task_name": self.env_name(), "level": spec.level, "seed": spec.seed, "repeat_index": spec.repeat_index, "difficulty": spec.metadata.get("difficulty") or meta.get("difficulty"), "board_radius": meta.get("board_radius"), "initial_block_count": meta.get("initial_block_count"), "cat_policy_name": meta.get("cat_policy_name"), "cat_policy_level": meta.get("cat_policy_level"), "difficulty_score": meta.get("difficulty_score"), "initial_escape_distance": meta.get("initial_escape_distance")}


def build_adapter(env_name: str) -> PlanningEnvAdapter:
    registry = {
        "knots_untangle": KnotsAdapter,
        "continuity_pipe": ContinuityPipeAdapter,
        "separation_one_stroke": SeparationAdapter,
        "order_swap_2d_puzzle": OrderSwap2DAdapter,
        "enclosure_chat_noir": ChatNoirAdapter,
    }
    if env_name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported env {env_name!r}. Known envs: {known}")
    return registry[env_name]()
