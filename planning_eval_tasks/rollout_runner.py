from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import random
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from artifact_writer import ArtifactWriter
from model_adapter import BaseModelClient
from unified_types import (
    EpisodeRunResult,
    Message,
    ModelResponse,
    ParsedActionResult,
    PlanningEnvAdapter,
    TurnInput,
    TurnRecord,
)


_SKIP_REASON_BUCKETS = {
    "no_image_prompt_tag",
    "model_declined",
    "max_image_gen_calls_reached",
    "step_deadline_exceeded",
}


def _skip_reason_bucket(reason: str) -> str:
    raw = (reason or "").strip()
    if raw in _SKIP_REASON_BUCKETS:
        return raw
    if raw.startswith("image_gen_failed"):
        return "image_gen_failed"
    if raw.startswith("step_deadline_exceeded"):
        return "step_deadline_exceeded"
    return raw or "unknown"


class RolloutRunner:
    def __init__(
        self,
        *,
        cfg: DictConfig,
        adapter: PlanningEnvAdapter,
        writer: ArtifactWriter,
        model_client: Optional[BaseModelClient],
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.writer = writer
        self.model_client = model_client
        # Tally of worker exceptions surfaced in summary.json so flaky episodes
        # don't silently disappear during multi-hour parallel sweeps.
        self._worker_error_count = 0

    def run(self) -> Dict[str, Any]:
        episode_specs = self._slice_episode_specs(self.adapter.enumerate_episodes(self.cfg))
        removed_api_error_rows = self.writer.scrub_api_error_rows(filename="model_answer.jsonl")
        if removed_api_error_rows:
            print(
                f"[api-failure-scrub] removed {removed_api_error_rows} api-error rows from model_answer.jsonl"
            )
        answer_written_ids = self.writer.load_jsonl_ids(filename="model_answer.jsonl")
        completed_episode_ids = set(answer_written_ids)
        episode_payloads, skipped_episodes = self._load_summary_state(completed_episode_ids)
        self._write_incremental_summary(episode_payloads=episode_payloads, skipped_episodes=skipped_episodes)
        parallel_sessions = self._parallel_sessions()

        if parallel_sessions > 1:
            self._run_parallel(
                episode_specs=episode_specs,
                parallel_sessions=parallel_sessions,
                episode_payloads=episode_payloads,
                skipped_episodes=skipped_episodes,
                completed_episode_ids=completed_episode_ids,
                answer_written_ids=answer_written_ids,
            )
        else:
            self._run_serial(
                episode_specs=episode_specs,
                episode_payloads=episode_payloads,
                skipped_episodes=skipped_episodes,
                completed_episode_ids=completed_episode_ids,
                answer_written_ids=answer_written_ids,
            )

        summary = self._build_summary(episode_payloads=episode_payloads, skipped_episodes=skipped_episodes)
        summary.update(self.writer.jsonl_artifact_paths())
        self.writer.write_summary(summary_payload=summary)
        return summary

    def _slice_episode_specs(self, episode_specs: Sequence[Any]) -> List[Any]:
        episode_ids_path = str(
            getattr(self.cfg.env, "episode_ids_path", "") or ""
        ).strip()
        if episode_ids_path:
            path = Path(episode_ids_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            raw_ids = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                loaded_ids = json.loads(raw_ids)
                if not isinstance(loaded_ids, list):
                    raise ValueError(f"Episode-id JSON must contain a list: {path}")
                requested_ids = [str(value).strip() for value in loaded_ids if str(value).strip()]
            else:
                requested_ids = [
                    line.strip()
                    for line in raw_ids.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            if len(requested_ids) != len(set(requested_ids)):
                raise ValueError(f"Duplicate episode ids in {path}")
            by_id = {str(getattr(spec, "episode_id", "")): spec for spec in episode_specs}
            missing = [episode_id for episode_id in requested_ids if episode_id not in by_id]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"{len(missing)} episode ids from {path} are absent from the source pool: {preview}"
                )
            episode_specs = [by_id[episode_id] for episode_id in requested_ids]

        only_episode_id = str(getattr(self.cfg.run, "only_episode_id", "") or "").strip()
        if only_episode_id:
            return [spec for spec in episode_specs if getattr(spec, "episode_id", "") == only_episode_id]

        # Accept either `sample_*` (legacy planning) or `manifest_*` (interleaved /
        # reasoning / video). The CLI in those harnesses uses `manifest_limit=N`,
        # which silently no-op'd on the gym path before this alias and let smokes
        # plow through the entire bench.
        offset_raw = getattr(self.cfg.run, "manifest_offset", None)
        if offset_raw is None:
            offset_raw = getattr(self.cfg.run, "sample_offset", 0)
        offset = int(offset_raw)
        if offset < 0:
            raise ValueError("run.manifest_offset / run.sample_offset must be >= 0.")
        limit = getattr(self.cfg.run, "manifest_limit", None)
        if limit in (None, "null"):
            limit = getattr(self.cfg.run, "sample_limit", None)
        if limit in (None, "null"):
            return list(episode_specs[offset:])
        sample_limit = int(limit)
        if sample_limit < 0:
            raise ValueError("run.manifest_limit / run.sample_limit must be >= 0.")
        return list(episode_specs[offset : offset + sample_limit])

    def _run_serial(
        self,
        *,
        episode_specs: Sequence[Any],
        episode_payloads: List[Dict[str, Any]],
        skipped_episodes: List[Dict[str, Any]],
        completed_episode_ids: set[str],
        answer_written_ids: set[str],
    ) -> None:
        env = None
        current_env_key: Any = None
        try:
            for spec in episode_specs:
                if self._is_resume_complete(spec, completed_episode_ids):
                    self._print_resume_skip(spec)
                    continue
                desired_env_key = self._env_cache_key(spec)
                if env is None or desired_env_key != current_env_key:
                    if env is not None:
                        env.close()
                    env = self._make_env(spec)
                    current_env_key = desired_env_key
                result = self._run_episode(env=env, spec=spec)
                self._commit_run_result(
                    spec=spec,
                    result=result,
                    episode_payloads=episode_payloads,
                    skipped_episodes=skipped_episodes,
                    completed_episode_ids=completed_episode_ids,
                    answer_written_ids=answer_written_ids,
                )
        finally:
            if env is not None:
                env.close()

    def _run_parallel(
        self,
        *,
        episode_specs: List[Any],
        parallel_sessions: int,
        episode_payloads: List[Dict[str, Any]],
        skipped_episodes: List[Dict[str, Any]],
        completed_episode_ids: set[str],
        answer_written_ids: set[str],
    ) -> None:
        # Resolve "[resume-skip]" up front and fan out only the unfinished specs.
        pending_specs = [
            spec for spec in episode_specs
            if not self._is_resume_complete(spec, completed_episode_ids)
        ]
        for spec in episode_specs:
            if self._is_resume_complete(spec, completed_episode_ids):
                self._print_resume_skip(spec)

        if not pending_specs:
            return

        # Multi-process workers, each pinned to one (or, with slots_per_key=2,
        # half) of an API key. Spawn is mandatory: Playwright's chromium handles
        # are not fork-safe. All shared JSONL/summary writes happen here in the
        # main process; workers only return picklable payload dicts.
        from configs.system_prompts import get_system_prompt  # local import to keep the leaf-module rule
        import _worker  # noqa: F401  — leaf worker module

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        system_prompt = get_system_prompt(str(self.cfg.env.system_prompt_key))
        ctx = multiprocessing.get_context("spawn")
        spec_dicts = [_worker.spec_to_dict(spec) for spec in pending_specs]
        spec_lookup = {spec.episode_id: spec for spec in pending_specs}

        # Spawned children inherit the parent's signal disposition while they
        # bootstrap. Temporarily ignoring SIGINT here prevents child processes
        # from printing noisy KeyboardInterrupt tracebacks; the parent restores
        # its handler immediately and owns interruption/shutdown below.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=parallel_sessions,
                mp_context=ctx,
                initializer=_worker.worker_init,
                initargs=(cfg_dict, system_prompt),
            )
            future_to_episode_id = {
                executor.submit(_worker.worker_run, spec_dict): spec_dict["episode_id"]
                for spec_dict in spec_dicts
            }
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        try:
            for future in concurrent.futures.as_completed(future_to_episode_id):
                episode_id = future_to_episode_id[future]
                spec = spec_lookup[episode_id]
                try:
                    result_payload = future.result()
                except Exception as exc:
                    self._record_worker_error(spec=spec, error=exc)
                    continue
                kind = result_payload.get("kind")
                if kind == "error":
                    self._record_worker_error(spec=spec, error=result_payload.get("error"))
                    continue
                if kind == "skip":
                    self._commit_skipped_episode(
                        spec=spec,
                        skip_reason=result_payload.get("skip_reason"),
                        episode_payloads=episode_payloads,
                        skipped_episodes=skipped_episodes,
                        completed_episode_ids=completed_episode_ids,
                    )
                    continue
                self._commit_episode_payload(
                    spec=spec,
                    episode_payload=result_payload["payload"],
                    answer_row=result_payload.get("answer_row"),
                    summary_log=str(result_payload.get("summary_log", "")),
                    episode_payloads=episode_payloads,
                    skipped_episodes=skipped_episodes,
                    completed_episode_ids=completed_episode_ids,
                    answer_written_ids=answer_written_ids,
                )
        except KeyboardInterrupt:
            print("[rollout-runner] interrupted; terminating workers...", flush=True)
            processes = list((getattr(executor, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                process.join(timeout=5.0)
            for process in processes:
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=1.0)
            raise
        else:
            executor.shutdown(wait=True)

    def _make_env(self, spec: Any) -> Any:
        make_env_for_episode = getattr(self.adapter, "make_env_for_episode", None)
        if callable(make_env_for_episode):
            return make_env_for_episode(self.cfg, spec)
        return self.adapter.make_env(self.cfg)

    def _env_cache_key(self, spec: Any) -> Any:
        env_cache_key = getattr(self.adapter, "env_cache_key", None)
        if callable(env_cache_key):
            return env_cache_key(self.cfg, spec)
        return ("default",)

    def _run_episode(self, *, env: Any, spec: Any) -> EpisodeRunResult:
        conversation_messages: List[Message] = []
        debug_messages: List[Message] = []
        pure_messages: List[Message] = []
        cached_initial_user_message: Optional[Message] = None
        cached_initial_debug_message: Optional[Message] = None
        cached_initial_pure_message: Optional[Message] = None
        all_images: Dict[str, Any] = {}
        turns: List[TurnRecord] = []
        invalid_responses = 0
        illegal_moves = 0
        api_errors = 0
        final_reward = 0.0
        final_reason: Optional[str] = None
        terminated = False
        truncated = False

        if self.model_client is not None:
            self.model_client.reset_episode()

        # core interact code
        try:
            obs, info = env.reset(**self.adapter.build_reset_kwargs(spec, self.cfg))
        except Exception as exc:
            if getattr(self.adapter, "should_skip_reset_error", None) is not None and self.adapter.should_skip_reset_error(spec, exc):
                return EpisodeRunResult(
                    spec=spec,
                    success=False,
                    terminated=False,
                    truncated=False,
                    total_steps=0,
                    illegal_moves=0,
                    invalid_responses=0,
                    api_errors=0,
                    final_reward=0.0,
                    final_reason=None,
                    turns=[],
                    metadata={},
                    skip_reason=str(exc),
                )
            raise

        step_index = 0
        while not terminated and not truncated:
            turn_context = self.adapter.extract_turn_context(env=env, obs=obs, info=info, step_index=step_index, history=turns)
            if step_index == 0 and self.model_client is not None:
                set_step_budget = getattr(self.model_client, "set_episode_step_budget", None)
                if callable(set_step_budget):
                    set_step_budget(turn_context.step_budget)
            turn_context.prompt_metadata.setdefault("env_ref", env)
            turn_context.prompt_metadata.setdefault("current_info", info)
            provisional_turn = TurnInput(spec=spec, context=turn_context, history=turns, saved_images=[])
            saved_images = self.adapter.capture_step_images(env=env, spec=spec, turn=provisional_turn, writer=self.writer)
            for image in saved_images:
                all_images[image.ref_id] = image
            turn_input = TurnInput(spec=spec, context=turn_context, history=turns, saved_images=saved_images)
            # Three message-construction modes:
            #   keep_message_history=true               → accumulate full conversation; initial at step 0, followup after.
            #   cache_task_prompt=true (history off)    → step 0 sends initial; step k≥1 sends [cached_initial, current_followup]
            #                                             (no assistant turns kept; initial state framing stays cached).
            #   default (history off, no cache)         → rebuild the initial state-shaped user prompt every step.
            # Stable task/rules/answer instructions live in the model client's system prompt in every mode.
            keep_history = self._keep_message_history()
            cache_task_prompt = self._cache_task_prompt() and not keep_history
            use_initial = step_index == 0 or (not keep_history and not cache_task_prompt)
            prompt_parts = self.adapter.build_initial_prompt(turn_input) if use_initial else self.adapter.build_followup_prompt(turn_input)

            current_user_message = {"role": "user", "content": prompt_parts.model_blocks}
            current_debug_message = {"role": "user", "content": prompt_parts.debug_blocks or prompt_parts.model_blocks}
            current_pure_message = {"role": "user", "content": prompt_parts.pure_blocks or prompt_parts.model_blocks}
            if cache_task_prompt and step_index == 0:
                cached_initial_user_message = current_user_message
                cached_initial_debug_message = current_debug_message
                cached_initial_pure_message = current_pure_message
            if keep_history:
                request_messages = self._windowed_request_messages(conversation_messages, current_user_message)
                request_debug_messages = self._windowed_request_messages(debug_messages, current_debug_message)
                request_pure_messages = self._windowed_request_messages(pure_messages, current_pure_message)
            elif cache_task_prompt and step_index > 0:
                assert cached_initial_user_message is not None
                request_messages = [cached_initial_user_message, current_user_message]
                request_debug_messages = [cached_initial_debug_message, current_debug_message]
                request_pure_messages = [cached_initial_pure_message, current_pure_message]
            else:
                request_messages = [current_user_message]
                request_debug_messages = [current_debug_message]
                request_pure_messages = [current_pure_message]

            # local models
            local_policy_name = str(self.cfg.model.policy_name) if str(self.cfg.model.kind) == "local_policy" else None
            if local_policy_name == "random":
                parse_result = self._choose_random_action(turn_input)
                response = ModelResponse(parse_result.raw_response_text, False, None, parse_result.response_debug)
            elif local_policy_name is not None:
                parse_result = self.adapter.choose_local_action(local_policy_name, turn_input)
                if parse_result is None:
                    raise ValueError(f"Local policy {local_policy_name!r} is unsupported by env {self.adapter.env_name()!r}.")
                response = ModelResponse(parse_result.raw_response_text, False, None, parse_result.response_debug)
            else:
                if self.model_client is None:
                    raise ValueError("OpenAI-compatible run requested without a model client.")
                request_images = self._request_images(all_images, request_messages)
                response = self.model_client.generate(messages=request_messages, images=request_images)
                if response.api_error:
                    # Transport/API failures are tracked separately from invalid model outputs
                    parse_result = ParsedActionResult(parsed_action=None, raw_response_text=response.raw_response_text, response_debug=response.response_debug, invalid_response=False)
                else:
                    # Remote models return raw text
                    parse_result = self.adapter.parse_model_action(response.raw_response_text, turn_input)
                    if not parse_result.response_debug:
                        parse_result.response_debug = dict(response.response_debug)

            api_error = bool(response.api_error)
            invalid_response = bool(parse_result.invalid_response) and not api_error
            if api_error:
                api_errors += 1
            if invalid_response:
                invalid_responses += 1

            if api_error:
                # Causal-progression gate: with no valid action we cannot commit
                # an env.step (which would have to invent a `None` action and
                # would burn a real step against `max_steps`). Terminate the
                # episode here. The next launch's `scrub_api_error_rows` drops
                # this row and the resume gate replays the episode from scratch.
                reward = 0.0
                terminated = True
                truncated = False
                next_info = {"reason": "api_error_aborted", "success": False}
                after_images = []
                illegal = False
                final_reward = float(reward)
                final_reason = "api_error_aborted"
            else:
                env_action = self.adapter.to_env_action(parse_result.parsed_action, turn_input)
                obs, reward, terminated, truncated, next_info = env.step(env_action)
                post_step_info = dict(next_info)
                post_step_info["_topobench_episode_done"] = bool(terminated or truncated)
                after_images = self.adapter.capture_post_step_images(
                    env=env,
                    spec=spec,
                    turn=turn_input,
                    obs=obs,
                    info=post_step_info,
                    writer=self.writer,
                )
                for image in after_images:
                    all_images[image.ref_id] = image
                illegal = bool(next_info.get("illegal", False))
                if illegal:
                    illegal_moves += 1
                final_reward = float(reward)
                final_reason = next_info.get("reason") or next_info.get("truncated_reason")

            turn_record = TurnRecord(
                spec=spec,
                step_index=step_index,
                context=turn_context,
                saved_images=saved_images,
                prompt_parts=prompt_parts,
                parsed_action=parse_result.parsed_action,
                flattened_answer=self.adapter.flatten_answer(parse_result.parsed_action),
                raw_response_text=response.raw_response_text,
                api_error=api_error,
                api_error_message=response.api_error_message,
                invalid_response=invalid_response,
                illegal=illegal,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                reason=final_reason,
                response_debug=parse_result.response_debug or response.response_debug,
                next_info=dict(next_info),
                after_images=after_images,
            )

            assistant_history_message = self._assistant_history_message(
                parse_result=parse_result,
                api_error=api_error,
            )
            conversation_messages.extend([current_user_message, assistant_history_message])
            debug_messages.extend([current_debug_message, dict(assistant_history_message)])
            pure_messages.extend([current_pure_message, dict(assistant_history_message)])
            artifact_debug_messages = list(request_debug_messages)
            if self.model_client is not None:
                sys_prompt = self.model_client.effective_system_prompt
                if sys_prompt:
                    artifact_debug_messages.insert(0, {"role": "system", "content": sys_prompt})
            prompt_artifacts = self.writer.write_prompt_artifacts(turn=turn_record, debug_messages=artifact_debug_messages, pure_messages=request_pure_messages, image_index=all_images)
            if prompt_artifacts:
                turn_record.next_info["prompt_artifacts"] = prompt_artifacts
            turns.append(turn_record)
            info = next_info
            step_index += 1

        success = bool(info.get("success", False) if isinstance(info, dict) else False)
        result = EpisodeRunResult(
            spec=spec,
            success=success,
            terminated=bool(terminated),
            truncated=bool(truncated),
            total_steps=int(info.get("episode_steps", len(turns))) if isinstance(info, dict) else len(turns),
            illegal_moves=illegal_moves,
            invalid_responses=invalid_responses,
            api_errors=api_errors,
            final_reward=final_reward,
            final_reason=final_reason,
            turns=turns,
        )
        result.metadata = self.adapter.build_episode_meta(spec, result)
        return result

    def _choose_random_action(self, turn: TurnInput) -> ParsedActionResult:
        legal_actions = list(turn.context.legal_actions)
        if not legal_actions:
            return ParsedActionResult(parsed_action=None, raw_response_text="<no-legal-action>", response_debug={"mode": "random"}, invalid_response=True)
        rng = random.Random(f"{turn.spec.episode_id}:{turn.context.step_index}:random")
        parsed_action = rng.choice(legal_actions)
        response_answer: Any = parsed_action
        if isinstance(parsed_action, dict) and set(parsed_action.keys()) == {"direction"}:
            parsed_action = parsed_action["direction"]
            response_answer = parsed_action
        elif self.adapter.env_name() == "enclosure_chat_noir" and isinstance(parsed_action, dict) and "cell_index" in parsed_action:
            parsed_action = int(parsed_action["cell_index"])
            response_answer = {"cell_index": parsed_action}
        elif self.adapter.env_name() == "continuity_pipe" and isinstance(parsed_action, int):
            grid_size = int(turn.context.prompt_metadata.get("grid_size", 0) or 0)
            if grid_size > 0:
                x = int(parsed_action) % grid_size
                y = int(parsed_action) // grid_size
                parsed_action = [x, y]
                response_answer = {"x": x, "y": y}
        raw_response_text = json.dumps({"answer": response_answer}, ensure_ascii=False, separators=(",", ":"))
        return ParsedActionResult(parsed_action=parsed_action, raw_response_text=raw_response_text, response_debug={"mode": "random"}, invalid_response=False)

    def _parallel_sessions(self) -> int:
        configured = int(getattr(self.cfg.run, "parallel_sessions", 0))
        if configured > 0:
            return configured
        if self.model_client is not None:
            return self.model_client.concurrency_hint
        return 1

    def _keep_message_history(self) -> bool:
        value = getattr(self.cfg.model, "keep_message_history", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _history_window(self) -> Optional[int]:
        value = getattr(self.cfg.run, "history_window", 5)
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"", "none", "null", "unlimited"}:
                return None
            value = stripped
        window = int(value)
        if window <= 0 or window % 2 == 0:
            raise ValueError(
                "run.history_window must be a positive odd message count "
                "(current user plus complete user/assistant history turns), or null for unlimited history."
            )
        return window

    def _windowed_request_messages(
        self,
        history_messages: Sequence[Message],
        current_message: Message,
    ) -> List[Message]:
        window = self._history_window()
        history_turns = self._history_turns(history_messages)
        if window is None:
            selected_turns = history_turns
        else:
            max_turns = (window - 1) // 2
            selected_turns = history_turns[-max_turns:] if max_turns > 0 else []

        image_limit = self._prompt_image_limit()
        while selected_turns and image_limit is not None:
            request = self._flatten_turns(selected_turns, current_message)
            if self._message_image_ref_count(request) <= image_limit:
                break
            selected_turns = selected_turns[1:]

        return self._flatten_turns(selected_turns, current_message)

    def _history_turns(self, history_messages: Sequence[Message]) -> List[List[Message]]:
        """Group completed history as user+assistant turns.

        Multimodal content lives inside the user message. Grouping by completed
        turns keeps each turn's text and image refs entering/leaving the window
        together instead of slicing a flat message list mid-turn.
        """
        turns: List[List[Message]] = []
        pending: List[Message] = []
        for message in history_messages:
            role = str(message.get("role", ""))
            if role == "user":
                if pending:
                    turns.append(pending)
                pending = [message]
            elif pending:
                pending.append(message)
                turns.append(pending)
                pending = []
            else:
                # Defensive fallback for malformed history. Keep non-user
                # messages attached to the next complete suffix only if they
                # are already part of a pending turn.
                continue
        return turns

    def _flatten_turns(self, turns: Sequence[Sequence[Message]], current_message: Message) -> List[Message]:
        flattened: List[Message] = []
        for turn in turns:
            flattened.extend(turn)
        flattened.append(current_message)
        return flattened

    def _prompt_image_limit(self) -> Optional[int]:
        configured = getattr(self.cfg.run, "max_prompt_images", None)
        if configured not in (None, "null", ""):
            limit = int(configured)
            if limit <= 0:
                return None
            return limit
        if self.model_client is not None:
            provider = str(self.model_client.describe().get("model_provider", "")).lower()
            if provider == "internvl":
                return 1
            if provider == "nvidia_nim":
                return 10
        return None

    def _compact_assistant_history(self) -> bool:
        value = getattr(self.cfg.run, "compact_assistant_history", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def _assistant_history_message(
        self,
        *,
        parse_result: ParsedActionResult,
        api_error: bool,
    ) -> Message:
        if not self._compact_assistant_history():
            return {"role": "assistant", "content": parse_result.raw_response_text or ""}

        payload: Dict[str, Any] = {"answer": self.adapter.flatten_answer(parse_result.parsed_action)}
        if api_error:
            payload["api_error"] = True
        elif parse_result.invalid_response:
            payload["invalid_response"] = True
        return {
            "role": "assistant",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }

    def _message_image_ref_ids(self, messages: Sequence[Message]) -> List[str]:
        ref_ids: List[str] = []
        seen: set[str] = set()
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_ref":
                    continue
                ref_id = str(block.get("ref_id", ""))
                if ref_id and ref_id not in seen:
                    seen.add(ref_id)
                    ref_ids.append(ref_id)
        return ref_ids

    def _message_image_ref_count(self, messages: Sequence[Message]) -> int:
        count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            count += sum(1 for block in content if isinstance(block, dict) and block.get("type") == "image_ref")
        return count

    def _request_images(self, all_images: Dict[str, Any], request_messages: Sequence[Message]) -> List[Any]:
        return [
            all_images[ref_id]
            for ref_id in self._message_image_ref_ids(request_messages)
            if ref_id in all_images
        ]

    def _cache_task_prompt(self) -> bool:
        value = getattr(self.cfg.run, "cache_task_prompt", False)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def _load_summary_state(self, completed_episode_ids: set[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        summary = self.writer.load_existing_summary()
        episode_payloads = summary.get("episodes") or []
        skipped_episodes = summary.get("skipped_episodes") or []
        if not isinstance(episode_payloads, list):
            raise ValueError("Existing summary field 'episodes' must be a list.")
        if not isinstance(skipped_episodes, list):
            raise ValueError("Existing summary field 'skipped_episodes' must be a list.")
        filtered_payloads: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for payload in episode_payloads:
            episode_id = payload.get("episode_id") if isinstance(payload, dict) else None
            if episode_id is None:
                continue
            episode_id = str(episode_id)
            if episode_id not in completed_episode_ids or episode_id in seen:
                continue
            filtered_payloads.append(payload)
            seen.add(episode_id)
        if len(seen) < len(completed_episode_ids):
            recovered = 0
            for path in sorted((self.writer.output_dir / "images").glob("**/episode_summary.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                episode_id = payload.get("episode_id")
                if episode_id is None:
                    continue
                episode_id = str(episode_id)
                if episode_id not in completed_episode_ids or episode_id in seen:
                    continue
                filtered_payloads.append(payload)
                seen.add(episode_id)
                recovered += 1
            if recovered:
                print(f"[resume-recover] restored {recovered} episode summaries from per-episode artifacts")
        # Incremental summary writes intentionally omit the full `episodes`
        # list to avoid O(N^2) serialization.  On a restarted run that means
        # summary.json alone cannot reconstruct the counters for rows that the
        # resume gate skips.  model_answer.jsonl is the durable source of truth
        # (append + fsync), so rebuild the compact episode payloads from it.
        # Without this recovery, a 340/600 run restarted at episode 341 would
        # finish with total_episodes=260 even though all 600 answer rows exist.
        if len(seen) < len(completed_episode_ids):
            answer_path = self.writer.output_dir / "model_answer.jsonl"
            recovered = 0
            if answer_path.exists():
                lines = answer_path.read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(lines):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        # Match ArtifactWriter.load_jsonl_ids: tolerate only a
                        # torn trailing append; corruption in the middle must
                        # remain visible instead of silently changing scores.
                        if index == len(lines) - 1:
                            continue
                        raise
                    if not isinstance(row, dict):
                        continue
                    episode_id = row.get("id")
                    if episode_id is None:
                        continue
                    episode_id = str(episode_id)
                    if episode_id not in completed_episode_ids or episode_id in seen:
                        continue
                    filtered_payloads.append(self._episode_payload_from_answer_row(row))
                    seen.add(episode_id)
                    recovered += 1
            if recovered:
                print(f"[resume-recover] restored {recovered} episode summaries from model_answer.jsonl")
        return filtered_payloads, list(skipped_episodes)

    @staticmethod
    def _episode_payload_from_answer_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Recreate the summary fields persisted in a durable answer row."""
        meta = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        trajectory = row.get("trajectory") if isinstance(row.get("trajectory"), list) else []
        steps = [step for step in trajectory if isinstance(step, dict)]
        return {
            "episode_id": str(row.get("id", "")),
            "level": meta.get("level"),
            "seed": meta.get("seed"),
            "repeat_index": meta.get("repeat_index"),
            "success": bool(meta.get("success", row.get("success", False))),
            "terminated": bool(meta.get("success", False)),
            "truncated": False,
            "total_steps": int(meta.get("total_steps", len(steps)) or 0),
            "illegal_moves": sum(1 for step in steps if bool(step.get("illegal", False))),
            "invalid_responses": sum(1 for step in steps if bool(step.get("invalid_response", False))),
            "api_errors": sum(1 for step in steps if bool(step.get("api_error", False))),
            "final_reward": float(meta.get("final_reward", 0.0) or 0.0),
            "final_reason": meta.get("final_reason"),
            "meta_info": dict(meta),
            "trajectory": steps,
        }

    @staticmethod
    def _is_resume_complete(spec: Any, completed_episode_ids: set[str]) -> bool:
        return spec.episode_id in completed_episode_ids

    def _print_resume_skip(self, spec: Any) -> None:
        print(
            f"[resume-skip] env={self.adapter.env_name()} level={spec.level} "
            f"seed={spec.seed} repeat={spec.repeat_index + 1} episode_id={spec.episode_id}"
        )

    def _commit_run_result(
        self,
        *,
        spec: Any,
        result: EpisodeRunResult,
        episode_payloads: List[Dict[str, Any]],
        skipped_episodes: List[Dict[str, Any]],
        completed_episode_ids: set[str],
        answer_written_ids: set[str],
    ) -> None:
        if result.skip_reason is not None:
            self._commit_skipped_episode(
                spec=spec,
                skip_reason=result.skip_reason,
                episode_payloads=episode_payloads,
                skipped_episodes=skipped_episodes,
                completed_episode_ids=completed_episode_ids,
            )
            return

        episode_payload = self._episode_summary_payload(result)
        if self.writer.write_step_images_enabled and result.turns and result.turns[0].saved_images:
            episode_base = Path(result.turns[0].saved_images[0].rel_path).parent
            self.writer.write_episode_summary_json(base_dir=episode_base, payload=episode_payload)
            self.writer.write_episode_demo_gif(result=result)
        answer_row = self._build_episode_jsonl_row(result)
        summary_log = (
            f"success={result.success} steps={result.total_steps} "
            f"illegal={result.illegal_moves} invalid={result.invalid_responses} "
            f"api_errors={result.api_errors}"
        )
        self._commit_episode_payload(
            spec=spec,
            episode_payload=episode_payload,
            answer_row=answer_row,
            summary_log=summary_log,
            episode_payloads=episode_payloads,
            skipped_episodes=skipped_episodes,
            completed_episode_ids=completed_episode_ids,
            answer_written_ids=answer_written_ids,
        )
        self.writer.discard_step_artifacts(result=result)

    def _record_worker_error(self, *, spec: Any, error: Any) -> None:
        """Persist worker traceback to <output_dir>/worker_errors.log so flaky
        episodes don't vanish from the run record.

        `error` may be an Exception (executor raise) or a stringified repr from
        the child process — we accept both. The summary's `worker_errors`
        counter increments and the per-line log captures spec metadata for
        triage. Episodes that raise here are NOT marked complete, so resume
        will re-run them on the next invocation.
        """
        self._worker_error_count += 1
        message_lines = [
            f"env={self.adapter.env_name()}",
            f"level={getattr(spec, 'level', '?')}",
            f"seed={getattr(spec, 'seed', '?')}",
            f"repeat={getattr(spec, 'repeat_index', 0) + 1}",
            f"episode_id={getattr(spec, 'episode_id', '?')}",
        ]
        header = " ".join(message_lines)
        if isinstance(error, BaseException):
            import traceback as _tb
            tb_text = "".join(_tb.format_exception(type(error), error, error.__traceback__))
            body = f"{error!r}\n{tb_text}"
        else:
            body = str(error)
        line = f"[worker-error] {header} error={body}"
        print(line)
        log_path = self.writer.output_dir / "worker_errors.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print(f"[worker-error] failed to persist worker_errors.log: {exc!r}")

    def _commit_skipped_episode(
        self,
        *,
        spec: Any,
        skip_reason: Any,
        episode_payloads: List[Dict[str, Any]],
        skipped_episodes: List[Dict[str, Any]],
        completed_episode_ids: set[str],
    ) -> None:
        self._upsert_skipped_episode(
            skipped_episodes,
            {
                "episode_id": spec.episode_id,
                "level": spec.level,
                "seed": spec.seed,
                "repeat_index": spec.repeat_index,
                "reason": skip_reason,
            },
        )
        completed_episode_ids.add(spec.episode_id)
        self._write_incremental_summary(episode_payloads=episode_payloads, skipped_episodes=skipped_episodes)
        print(
            f"[skip] env={self.adapter.env_name()} level={spec.level} "
            f"seed={spec.seed} repeat={spec.repeat_index + 1} reason={skip_reason}"
        )

    def _commit_episode_payload(
        self,
        *,
        spec: Any,
        episode_payload: Dict[str, Any],
        answer_row: Optional[Dict[str, Any]],
        summary_log: str,
        episode_payloads: List[Dict[str, Any]],
        skipped_episodes: List[Dict[str, Any]],
        completed_episode_ids: set[str],
        answer_written_ids: set[str],
    ) -> None:
        if answer_row is not None and spec.episode_id not in answer_written_ids:
            self.writer.append_jsonl_row(filename="model_answer.jsonl", row=answer_row)
            answer_written_ids.add(spec.episode_id)

        if spec.episode_id in answer_written_ids:
            episode_payloads.append(episode_payload)
            completed_episode_ids.add(spec.episode_id)
            self._write_incremental_summary(
                episode_payloads=episode_payloads,
                skipped_episodes=skipped_episodes,
            )
        print(
            f"[episode] env={self.adapter.env_name()} level={spec.level} "
            f"seed={spec.seed} repeat={spec.repeat_index + 1} {summary_log}"
        )

    def _upsert_skipped_episode(self, skipped_episodes: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
        episode_id = str(payload.get("episode_id", ""))
        for index, row in enumerate(skipped_episodes):
            if str(row.get("episode_id", "")) == episode_id:
                skipped_episodes[index] = payload
                return
        skipped_episodes.append(payload)

    def _episode_summary_payload(self, result: EpisodeRunResult) -> Dict[str, Any]:
        return {
            "episode_id": result.spec.episode_id,
            "level": result.spec.level,
            "seed": result.spec.seed,
            "repeat_index": result.spec.repeat_index,
            "success": result.success,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "total_steps": result.total_steps,
            "illegal_moves": result.illegal_moves,
            "invalid_responses": result.invalid_responses,
            "api_errors": result.api_errors,
            "final_reward": result.final_reward,
            "final_reason": result.final_reason,
            "meta_info": result.metadata,
            "steps": [
                {
                    "step_index": turn.step_index,
                    "current_images": [
                        image.rel_path
                        for image in turn.saved_images
                        if image.label == "current"
                    ] or [image.rel_path for image in turn.saved_images],
                    "post_action_images": [image.rel_path for image in turn.after_images],
                    "answer": turn.flattened_answer,
                    "invalid_response": turn.invalid_response,
                    "api_error": turn.api_error,
                    "illegal": turn.illegal,
                    "raw_response_text": turn.raw_response_text,
                    "reward": turn.reward,
                    "reason": turn.reason,
                }
                for turn in result.turns
            ],
        }

    def _write_incremental_summary(self, *, episode_payloads: Sequence[Dict[str, Any]], skipped_episodes: Sequence[Dict[str, Any]]) -> None:
        # Incremental writes omit the full `episodes` list — model_answer.jsonl
        # already records per-episode rows and the per-episode dirs hold the
        # full trajectory. Including the list here would make per-commit
        # serialization cost O(N²) on large runs.
        summary = self._build_summary(
            episode_payloads=episode_payloads,
            skipped_episodes=skipped_episodes,
            include_episodes=False,
        )
        summary.update(self.writer.jsonl_artifact_paths())
        self.writer.write_summary_only(summary_payload=summary)

    def _build_summary(
        self,
        *,
        episode_payloads: Sequence[Dict[str, Any]],
        skipped_episodes: Sequence[Dict[str, Any]],
        include_episodes: bool = True,
    ) -> Dict[str, Any]:
        total = len(episode_payloads)
        successes = sum(1 for payload in episode_payloads if bool(payload.get("success", False)))
        success_steps = [
            int(payload.get("total_steps", 0))
            for payload in episode_payloads
            if bool(payload.get("success", False))
        ]
        # Walk trajectory steps for interleaved-specific signals so an operator
        # can distinguish "model declined imagery 50 times" from "key pool
        # exhausted on day-rate-limit 50 times" — both look identical in the
        # raw api_errors counter.
        skip_reason_counts: Dict[str, int] = {}
        for payload in episode_payloads:
            for step in payload.get("trajectory") or []:
                reason = step.get("image_gen_skip_reason")
                if not reason:
                    continue
                bucket = _skip_reason_bucket(str(reason))
                skip_reason_counts[bucket] = skip_reason_counts.get(bucket, 0) + 1
        model_info = self.model_client.describe() if self.model_client is not None else {}
        summary: Dict[str, Any] = {
            "env": self.adapter.env_name(),
            "model_id": str(model_info.get("model_id", self.cfg.model.id)),
            "model_name": str(model_info.get("model_name", getattr(self.cfg.model, "name", self.cfg.model.id))),
            "model_provider": str(model_info.get("model_provider", getattr(self.cfg.model, "provider", "local"))),
            "model_base_url": str(model_info.get("model_base_url", getattr(self.cfg.model, "base_url", ""))),
            "total_episodes": total,
            "source_question_jsonl": self._source_question_jsonl(),
            "skipped_episodes": list(skipped_episodes),
            "successes": successes,
            "success_rate": (successes / total) if total else 0.0,
            "avg_steps_all": (sum(int(payload.get("total_steps", 0)) for payload in episode_payloads) / total) if total else 0.0,
            "avg_steps_on_success": (sum(success_steps) / len(success_steps)) if success_steps else None,
            "avg_illegal_moves": (sum(int(payload.get("illegal_moves", 0)) for payload in episode_payloads) / total) if total else 0.0,
            "avg_invalid_responses": (sum(int(payload.get("invalid_responses", 0)) for payload in episode_payloads) / total) if total else 0.0,
            "avg_api_errors": (sum(int(payload.get("api_errors", 0)) for payload in episode_payloads) / total) if total else 0.0,
            "image_gen_skip_reasons": skip_reason_counts,
            "worker_errors": int(self._worker_error_count),
            "config": {
                "env": OmegaConf.to_container(self.cfg.env, resolve=True),
                "model": OmegaConf.to_container(self.cfg.model, resolve=True),
                "run": OmegaConf.to_container(self.cfg.run, resolve=True),
            },
        }
        if include_episodes:
            summary["episodes"] = list(episode_payloads)
        return summary

    def _source_question_jsonl(self) -> Optional[str]:
        source_relpath = getattr(self.adapter, "source_question_jsonl_relpath", None)
        if callable(source_relpath):
            return str(source_relpath())
        return None

    def _build_episode_jsonl_row(self, result: EpisodeRunResult) -> Optional[Dict[str, Any]]:
        if result.first_turn is None:
            return None
        meta_info = {
            **result.metadata,
            "task_name": self.adapter.env_name(),
            "config": result.spec.metadata.get("config_rel"),
            "level": result.spec.level,
            "seed": result.spec.seed,
            "repeat_index": result.spec.repeat_index,
            "difficulty": result.metadata.get("difficulty", "hard"),
            "model_id": str(self.cfg.model.id),
            "success": bool(result.success),
            "final_reason": result.final_reason,
            "total_steps": int(result.total_steps),
        }
        answer_row = self.writer.build_answer_row(result=result, meta_info=meta_info)
        return answer_row
