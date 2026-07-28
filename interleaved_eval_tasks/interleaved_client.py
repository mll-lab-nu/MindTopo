from __future__ import annotations

import math
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make planning_eval_tasks importable so we can reuse env adapters, key pool,
# model_adapter, unified_types, etc. without duplicating them. Append (not
# insert) so that interleaved's own modules — particularly `configs.*` —
# shadow planning's where the names collide.
_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))

from omegaconf import DictConfig

from evaluator_utils import resolve_repo_path  # noqa: E402
from model_adapter import BaseModelClient, build_key_pool, load_keys_from_env  # noqa: E402
from two_phase_helpers import (  # noqa: E402
    append_phase1_reminder,
    build_inner_text_client,
    extract_tagged_prompt,
    fallback_phase1_prompt,
    peer_artifact_target,
    pick_current_reference_image,
    resolve_base_url_from_cfg,
    strip_answer_format_from_messages,
)
from unified_types import Message, ModelResponse, SavedImage  # noqa: E402

from image_gen_client import OpenAICompatibleImageGenClient  # noqa: E402


_IMAGE_PROMPT_PATTERN = re.compile(
    r"<image_prompt>\s*(.*?)\s*</image_prompt>",
    re.IGNORECASE | re.DOTALL,
)

_FALLBACK_IMAGE_PROMPT = (
    "Imagine the next observation after the agent's intended action."
)

_PHASE1_REMINDER = (
    "[Phase 1 instruction]\n"
    "Do NOT output the final answer JSON in this turn. Output exactly one tag: "
    "<image_prompt>concise visual description of the annotated image that would "
    "help you decide the answer</image_prompt>. If no image would help, output "
    "<image_prompt>none</image_prompt>. The answer JSON is for Phase 2, after "
    "the imagined image is supplied."
)


def _build_image_gen_client(image_cfg: DictConfig) -> OpenAICompatibleImageGenClient:
    """Construct the image-gen client from `model.image_gen` config."""
    base_url = resolve_base_url_from_cfg(image_cfg, label="Image-gen")
    keys = load_keys_from_env(str(image_cfg.api_key_env))
    rpm = float(image_cfg.requests_per_minute)
    pool = build_key_pool(
        provider=f"image_gen:{image_cfg.upstream_model_name}",
        keys=keys,
        rpm=rpm,
        slots_per_key=1,
    )
    return OpenAICompatibleImageGenClient(
        image_cfg=image_cfg,
        upstream_model_name=str(image_cfg.upstream_model_name),
        base_url=base_url,
        pool=pool,
    )


@dataclass
class _PhaseDebug:
    text: str
    response_debug: Dict[str, Any]
    api_error: bool
    api_error_message: Optional[str]


class InterleavedModelClient(BaseModelClient):
    """Wraps a text VLM + an image-gen client into one BaseModelClient.

    Per call to generate(), we run two phases:
        Phase 1: send messages as-is to the text client; expect <image_prompt>...</image_prompt>.
        Phase 2: append assistant(phase1) + a user message holding the imagined image,
                 then call the text client again; the response is the canonical action.

    The runner does not need to know about interleaving — phase 2's text becomes the
    `raw_response_text` of the returned ModelResponse. The full phase-1 trace,
    extracted image prompt, and saved image path are stashed in `response_debug`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        provider_name: str,
        text_client: BaseModelClient,
        image_gen_client: OpenAICompatibleImageGenClient,
        output_dir: Path,
        enforce_image_call: bool = False,
        enforce_against_decline: bool = False,
        max_image_gen_calls: int = 0,
        max_image_gen_calls_fraction: float = 0.0,
        phase_retry_max_attempts: int = 10,
        phase_retry_initial_sleep_seconds: float = 15.0,
        phase_retry_max_sleep_seconds: float = 300.0,
    ) -> None:
        self.model_id = model_id
        self.provider_name = provider_name
        self.text_client = text_client
        self.image_gen_client = image_gen_client
        self.output_dir = Path(output_dir)
        self.enforce_image_call = bool(enforce_image_call)
        self.enforce_against_decline = bool(enforce_against_decline)
        # 0 (or negative) → unlimited. Counter resets per episode.
        self.max_image_gen_calls = max(0, int(max_image_gen_calls))
        self.max_image_gen_calls_fraction = max(0.0, float(max_image_gen_calls_fraction))
        # Retry budget per generate() call. Each attempt re-runs only the phase
        # that failed: phase-1 from scratch (also re-runs image-gen on success);
        # phase-2 reuses the cached phase-1 text + imagined image. After the cap
        # the call returns api_error=True and the runner aborts the episode
        # without committing a step (see rollout_runner causal-progression gate).
        self.phase_retry_max_attempts = max(0, int(phase_retry_max_attempts))
        self.phase_retry_initial_sleep_seconds = max(0.0, float(phase_retry_initial_sleep_seconds))
        self.phase_retry_max_sleep_seconds = max(
            self.phase_retry_initial_sleep_seconds,
            float(phase_retry_max_sleep_seconds),
        )
        self._counter_lock = threading.Lock()
        self._reset_index = 0
        self._step_index = 0
        self._image_gen_call_count = 0
        self._episode_image_gen_cap: Optional[int] = (
            self.max_image_gen_calls if self.max_image_gen_calls > 0 else None
        )

    @property
    def concurrency_hint(self) -> int:
        return self.text_client.concurrency_hint

    @property
    def effective_system_prompt(self) -> str:
        return self.text_client.effective_system_prompt

    def reset_episode(self) -> None:
        self.text_client.reset_episode()
        with self._counter_lock:
            self._reset_index += 1
            self._step_index = 0
            self._image_gen_call_count = 0
            self._episode_image_gen_cap = (
                self.max_image_gen_calls if self.max_image_gen_calls > 0 else None
            )

    def set_episode_step_budget(self, step_budget: Optional[int]) -> None:
        """Set a per-episode image-call cap from the environment step budget."""
        if self.max_image_gen_calls_fraction <= 0:
            return
        if step_budget is None or int(step_budget) <= 0:
            raise ValueError(
                "max_image_gen_calls_fraction requires a positive episode step budget"
            )
        fractional_cap = math.floor(int(step_budget) * self.max_image_gen_calls_fraction)
        absolute_cap = self.max_image_gen_calls if self.max_image_gen_calls > 0 else None
        with self._counter_lock:
            self._episode_image_gen_cap = (
                fractional_cap
                if absolute_cap is None
                else min(absolute_cap, fractional_cap)
            )

    def describe(self) -> Dict[str, Any]:
        inner = self.text_client.describe()
        return {
            "model_id": self.model_id,
            "model_name": str(inner.get("model_name", self.model_id)),
            "model_provider": self.provider_name,
            "model_base_url": str(inner.get("model_base_url", "")),
            "interleaved_text_model_id": str(inner.get("model_id", "")),
            "image_gen_model_name": self.image_gen_client.upstream_model_name,
            "image_gen_base_url": self.image_gen_client.base_url,
        }

    def _next_image_slot(self) -> Tuple[int, int]:
        with self._counter_lock:
            reset_idx = self._reset_index
            step_idx = self._step_index
            self._step_index += 1
        return reset_idx, step_idx

    def _imagined_target_for(self, reference: Optional[SavedImage]) -> Tuple[Path, str]:
        """Save peer to the source observation when possible, else a counter path."""
        target = peer_artifact_target(reference, output_dir=self.output_dir, suffix=".png")
        if target is not None:
            return target
        reset_idx, step_idx = self._next_image_slot()
        rel = Path("imagined") / f"ep_{reset_idx:04d}" / f"step_{step_idx:04d}.png"
        return (self.output_dir / rel), str(rel)

    def _phase1(
        self,
        *,
        messages: Sequence[Message],
        images: Sequence[SavedImage],
    ) -> _PhaseDebug:
        # Strip the env's `[Answer Format] {"answer": ...}` block and append
        # a phase-1 reminder so the model emits <image_prompt> instead of
        # short-circuiting to the answer JSON. Phase 2 reuses the un-stripped
        # original messages so its answer-format instruction is preserved.
        phase1_messages = append_phase1_reminder(
            strip_answer_format_from_messages(messages),
            _PHASE1_REMINDER,
        )
        response = self.text_client.generate(messages=phase1_messages, images=images)
        return _PhaseDebug(
            text=response.raw_response_text or "",
            response_debug=dict(response.response_debug or {}),
            api_error=bool(response.api_error),
            api_error_message=response.api_error_message,
        )

    @staticmethod
    def _phase1_reasoning_text(phase1_debug: Dict[str, Any]) -> Optional[str]:
        """Surface phase-1's `reasoning_content` for the top-level mirror; phase-2's stays canonical."""
        value = phase1_debug.get("reasoning_content")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _phase2(
        self,
        *,
        original_messages: Sequence[Message],
        phase1_text: str,
        original_images: Sequence[SavedImage],
        imagined_image: Optional[SavedImage],
        image_prompt: Optional[str],
        followup_blocks: List[Dict[str, Any]],
    ) -> _PhaseDebug:
        phase2_messages: List[Message] = list(original_messages)
        phase2_messages.append({"role": "assistant", "content": phase1_text})
        phase2_messages.append({"role": "user", "content": followup_blocks})

        all_images: List[SavedImage] = list(original_images)
        if imagined_image is not None:
            all_images.append(imagined_image)

        response = self.text_client.generate(messages=phase2_messages, images=all_images)
        return _PhaseDebug(
            text=response.raw_response_text or "",
            response_debug=dict(response.response_debug or {}),
            api_error=bool(response.api_error),
            api_error_message=response.api_error_message,
        )

    def _maybe_generate_imagined_image(
        self,
        *,
        image_prompt: Optional[str],
        images: Sequence[SavedImage],
        phase1_text: str = "",
        phase1_reasoning: Optional[str] = None,
    ) -> Tuple[Optional[SavedImage], Dict[str, Any], Optional[str]]:
        """Run image gen if phase 1 asked for it. Return (imagined, debug, skip_reason).

        `self.enforce_image_call=True` overrides `no_image_prompt_tag` — models
        that fail to emit the tag get a fallback prompt synthesized from phase-1
        text/reasoning. By default an explicit `<image_prompt>none</image_prompt>`
        decline is part of the protocol and is honored. Set
        `self.enforce_against_decline=True` (per-env knob, see agent_runner) to
        also override declines — used by reasoning envs where an imagined image
        is always beneficial.
        """
        # Hard per-episode cap (regardless of enforce_image_call). When hit,
        # skip image gen and let phase 2 proceed with no imagined image.
        with self._counter_lock:
            already_called = self._image_gen_call_count
            episode_cap = self._episode_image_gen_cap
        if episode_cap is not None:
            if already_called >= episode_cap:
                return None, {
                    "image_gen_call_count": already_called,
                    "max_image_gen_calls": episode_cap,
                    "max_image_gen_calls_fraction": self.max_image_gen_calls_fraction,
                }, "max_image_gen_calls_reached"

        skip_reason_for_missing: Optional[str] = None
        if image_prompt is None:
            skip_reason_for_missing = "no_image_prompt_tag"
        elif image_prompt.strip().lower() == "none":
            skip_reason_for_missing = "model_declined"

        fallback_used = False
        fallback_source: Optional[str] = None
        if skip_reason_for_missing is not None:
            overridable = (
                (skip_reason_for_missing == "no_image_prompt_tag" and self.enforce_image_call)
                or (
                    skip_reason_for_missing == "model_declined"
                    and self.enforce_image_call
                    and self.enforce_against_decline
                )
            )
            if not overridable:
                return None, {}, skip_reason_for_missing
            image_prompt, fallback_source = fallback_phase1_prompt(
                phase1_text,
                phase1_reasoning,
                default=_FALLBACK_IMAGE_PROMPT,
            )
            fallback_used = True

        # Reserve a slot before issuing the request so concurrent steps stay
        # under the cap. Counter is incremented even if the upstream call
        # fails — that matches "calls attempted" semantics.
        with self._counter_lock:
            self._image_gen_call_count += 1
            current_call_idx = self._image_gen_call_count

        reference = pick_current_reference_image(images)
        save_path, rel_path = self._imagined_target_for(reference)
        ref_id = f"imagined-{uuid.uuid4().hex[:8]}"
        # In `edits` mode, image-gen wants the reference bytes; in `generations`
        # mode it ignores them. The reference is also what drives the imagined
        # PNG's filename peering above, regardless of mode.
        reference_image_path = reference.abs_path if (reference and self.image_gen_client.mode == "edits") else None
        result = self.image_gen_client.generate_to_file(
            prompt=image_prompt,
            save_path=save_path,
            ref_id=ref_id,
            rel_path=rel_path,
            label="imagined",
            reference_image_path=reference_image_path,
        )
        debug = {
            "image_prompt": image_prompt,
            "imagined_image_rel_path": rel_path,
            "imagined_image_ref_id": ref_id,
            "image_gen_api_error": result.api_error,
            "image_gen_api_error_message": result.api_error_message,
            "image_gen_mode": self.image_gen_client.mode,
            "image_gen_reference_image": str(reference_image_path) if reference_image_path else None,
            "image_gen_request_size": result.request_size,
        }
        if fallback_used:
            debug["image_prompt_fallback_used"] = True
            debug["image_prompt_fallback_source"] = fallback_source
            debug["image_prompt_original_skip_reason"] = skip_reason_for_missing
        debug["image_gen_call_count"] = current_call_idx
        if episode_cap is not None:
            debug["max_image_gen_calls"] = episode_cap
            debug["max_image_gen_calls_fraction"] = self.max_image_gen_calls_fraction
        if result.api_error:
            return None, debug, f"image_gen_failed: {result.api_error_message}"
        return result.saved_image, debug, None

    @staticmethod
    def _phase2_followup_blocks(
        imagined_image: Optional[SavedImage],
        image_prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build the phase-2 user-message blocks (also recorded for debug rendering)."""
        if imagined_image is not None:
            return [
                {"type": "text", "text": "Imagined image:"},
                {"type": "image_ref", "ref_id": imagined_image.ref_id},
                {
                    "type": "text",
                    "text": (
                        "You are now in PHASE 2. Using both the original observation and "
                        "the imagined image above, output your final action in the JSON "
                        "format the task requires."
                    ),
                },
            ]
        note = "no imagined image was generated"
        if image_prompt and image_prompt.strip().lower() == "none":
            note = "you declined imagery"
        return [
            {
                "type": "text",
                "text": (
                    f"You are now in PHASE 2 ({note}). Output your final action in "
                    f"the JSON format the task requires."
                ),
            }
        ]

    def _retry_sleep_seconds(self, retry_index: int) -> float:
        """Exponential backoff with a hard cap. retry_index is 0-based (first retry = 0)."""
        scaled = self.phase_retry_initial_sleep_seconds * (2 ** retry_index)
        return min(scaled, self.phase_retry_max_sleep_seconds)

    def _success_response_debug(
        self,
        *,
        phase1: _PhaseDebug,
        phase1_reasoning: Optional[str],
        image_prompt: Optional[str],
        image_gen_debug: Dict[str, Any],
        skip_reason: Optional[str],
        phase2: _PhaseDebug,
        phase2_followup_blocks: List[Dict[str, Any]],
        phase1_attempts: int,
        phase2_attempts: int,
        retry_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response_debug: Dict[str, Any] = {
            **phase2.response_debug,
            "interleaved_phase": 2,
            "phase1_text": phase1.text,
            "phase1_response_debug": phase1.response_debug,
            "image_prompt": image_prompt,
            "image_gen_skip_reason": skip_reason,
            "phase2_followup_blocks": phase2_followup_blocks,
            "phase1_attempts": phase1_attempts,
            "phase2_attempts": phase2_attempts,
        }
        if phase1_reasoning is not None:
            response_debug["phase1_reasoning_content"] = phase1_reasoning
        if retry_log:
            response_debug["phase_retry_log"] = retry_log
        response_debug.update(image_gen_debug)
        return response_debug

    def _failed_response(
        self,
        *,
        last_phase: int,
        last_error: Optional[str],
        retry_log: List[Dict[str, Any]],
        phase1: Optional[_PhaseDebug],
        phase1_reasoning: Optional[str],
        image_prompt: Optional[str],
        image_gen_debug: Dict[str, Any],
        skip_reason: Optional[str],
        phase1_attempts: int,
        phase2_attempts: int,
    ) -> ModelResponse:
        # Mirror the success path's response_debug key set so downstream tooling
        # (artifact_writer.write_prompt_artifacts, summary aggregation) doesn't
        # branch on which phase failed.
        gen_debug = image_gen_debug or {}
        error_debug: Dict[str, Any] = {
            "interleaved_phase": last_phase,
            "phase1_text": phase1.text if phase1 is not None else "",
            "phase1_response_debug": phase1.response_debug if phase1 is not None else {},
            "image_prompt": image_prompt,
            "image_gen_skip_reason": skip_reason or ("phase1_api_error" if last_phase == 1 else None),
            "phase2_followup_blocks": [],
            "imagined_image_rel_path": gen_debug.get("imagined_image_rel_path"),
            "imagined_image_ref_id": gen_debug.get("imagined_image_ref_id"),
            "image_gen_api_error": bool(gen_debug.get("image_gen_api_error", False)),
            "image_gen_api_error_message": gen_debug.get("image_gen_api_error_message"),
            "image_gen_mode": self.image_gen_client.mode,
            "image_gen_reference_image": gen_debug.get("image_gen_reference_image"),
            "image_gen_call_count": int(gen_debug.get("image_gen_call_count", 0) or 0),
            "phase1_attempts": phase1_attempts,
            "phase2_attempts": phase2_attempts,
        }
        with self._counter_lock:
            episode_cap = self._episode_image_gen_cap
        if episode_cap is not None:
            error_debug["max_image_gen_calls"] = episode_cap
            error_debug["max_image_gen_calls_fraction"] = self.max_image_gen_calls_fraction
        if phase1_reasoning is not None:
            error_debug["phase1_reasoning_content"] = phase1_reasoning
        if retry_log:
            error_debug["phase_retry_log"] = retry_log
        message = (
            f"interleaved phase={last_phase} failed after "
            f"{self.phase_retry_max_attempts + 1} attempts; last_error="
            f"{last_error or '<none>'}"
        )
        return ModelResponse(
            raw_response_text="",
            api_error=True,
            api_error_message=message,
            response_debug=error_debug,
        )

    def generate(self, *, messages: Sequence[Message], images: Sequence[SavedImage]) -> ModelResponse:
        # Per-phase retry with phase-1 caching. On phase-2 api_error we keep the
        # phase-1 text + imagined image and only re-call phase-2; on phase-1
        # api_error we re-run phase-1 (which forces a fresh image-gen). After
        # the cap the call returns api_error=True so the runner can abort the
        # episode without committing a `None` action via env.step.
        phase1: Optional[_PhaseDebug] = None
        phase1_reasoning: Optional[str] = None
        image_prompt: Optional[str] = None
        imagined_image: Optional[SavedImage] = None
        image_gen_debug: Dict[str, Any] = {}
        skip_reason: Optional[str] = None
        phase2_followup_blocks: List[Dict[str, Any]] = []
        phase1_attempts = 0
        phase2_attempts = 0
        retry_log: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        last_phase = 1

        total_attempts = self.phase_retry_max_attempts + 1
        for attempt_idx in range(total_attempts):
            if phase1 is None:
                phase1_attempts += 1
                phase1_run = self._phase1(messages=messages, images=images)
                if phase1_run.api_error:
                    last_phase = 1
                    last_error = phase1_run.api_error_message
                    retry_log.append({
                        "attempt": attempt_idx + 1,
                        "phase": 1,
                        "error": last_error,
                    })
                    if attempt_idx + 1 < total_attempts:
                        sleep_s = self._retry_sleep_seconds(attempt_idx)
                        print(
                            f"[interleaved-retry] phase=1 attempt={attempt_idx+1}/"
                            f"{total_attempts} api_error={last_error}; "
                            f"sleep {sleep_s:.1f}s and retry phase 1",
                            flush=True,
                        )
                        time.sleep(sleep_s)
                        continue
                    return self._failed_response(
                        last_phase=1,
                        last_error=last_error,
                        retry_log=retry_log,
                        phase1=None,
                        phase1_reasoning=None,
                        image_prompt=None,
                        image_gen_debug={},
                        skip_reason="phase1_api_error",
                        phase1_attempts=phase1_attempts,
                        phase2_attempts=phase2_attempts,
                    )

                # Phase 1 succeeded — cache it and run image gen + followup blocks once.
                phase1 = phase1_run
                phase1_reasoning = self._phase1_reasoning_text(phase1.response_debug)
                image_prompt = extract_tagged_prompt(
                    phase1.text, phase1_reasoning, _IMAGE_PROMPT_PATTERN
                )
                imagined_image, image_gen_debug, skip_reason = self._maybe_generate_imagined_image(
                    image_prompt=image_prompt,
                    images=images,
                    phase1_text=phase1.text,
                    phase1_reasoning=phase1_reasoning,
                )
                phase2_followup_blocks = self._phase2_followup_blocks(imagined_image, image_prompt)

            phase2_attempts += 1
            phase2 = self._phase2(
                original_messages=messages,
                phase1_text=phase1.text,
                original_images=images,
                imagined_image=imagined_image,
                image_prompt=image_prompt,
                followup_blocks=phase2_followup_blocks,
            )
            if not phase2.api_error:
                response_debug = self._success_response_debug(
                    phase1=phase1,
                    phase1_reasoning=phase1_reasoning,
                    image_prompt=image_prompt,
                    image_gen_debug=image_gen_debug,
                    skip_reason=skip_reason,
                    phase2=phase2,
                    phase2_followup_blocks=phase2_followup_blocks,
                    phase1_attempts=phase1_attempts,
                    phase2_attempts=phase2_attempts,
                    retry_log=retry_log,
                )
                return ModelResponse(
                    raw_response_text=phase2.text,
                    api_error=False,
                    api_error_message=None,
                    response_debug=response_debug,
                )

            last_phase = 2
            last_error = phase2.api_error_message
            retry_log.append({
                "attempt": attempt_idx + 1,
                "phase": 2,
                "error": last_error,
            })
            if attempt_idx + 1 < total_attempts:
                sleep_s = self._retry_sleep_seconds(attempt_idx)
                print(
                    f"[interleaved-retry] phase=2 attempt={attempt_idx+1}/"
                    f"{total_attempts} api_error={last_error}; "
                    f"sleep {sleep_s:.1f}s and retry phase 2 (phase 1 + image-gen cached)",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            return self._failed_response(
                last_phase=2,
                last_error=last_error,
                retry_log=retry_log,
                phase1=phase1,
                phase1_reasoning=phase1_reasoning,
                image_prompt=image_prompt,
                image_gen_debug=image_gen_debug,
                skip_reason=skip_reason,
                phase1_attempts=phase1_attempts,
                phase2_attempts=phase2_attempts,
            )

        # Loop always returns from inside; this is just a defensive guard.
        return self._failed_response(
            last_phase=last_phase,
            last_error=last_error,
            retry_log=retry_log,
            phase1=phase1,
            phase1_reasoning=phase1_reasoning,
            image_prompt=image_prompt,
            image_gen_debug=image_gen_debug,
            skip_reason=skip_reason,
            phase1_attempts=phase1_attempts,
            phase2_attempts=phase2_attempts,
        )


def build_interleaved_client(
    *,
    cfg: DictConfig,
    system_prompt: str,
) -> Optional[BaseModelClient]:
    """Build the model client for the interleaved pipeline.

    Dispatches by `model.kind`:
      - 'local_policy'      → no client (runner uses local policies directly)
      - 'interleaved'       → InterleavedModelClient (phase1 + image_gen + phase2)
      - 'registry_remote' or 'openai_compatible' → fall back to planning's builder
        (useful for non-interleaved baselines run through the same harness).
    """
    model_cfg = cfg.model
    model_kind = str(model_cfg.kind)

    if model_kind == "local_policy":
        return None

    if model_kind == "interleaved":
        text_client = build_inner_text_client(
            text_cfg=model_cfg.text,
            system_prompt=system_prompt,
        )
        image_gen_client = _build_image_gen_client(model_cfg.image_gen)
        output_dir = resolve_repo_path(str(cfg.run.output_dir))
        enforce_image_call = bool(getattr(model_cfg.image_gen, "enforce_image_call", False))
        enforce_against_decline = bool(getattr(model_cfg.image_gen, "enforce_against_decline", False))
        max_image_gen_calls = int(getattr(model_cfg.image_gen, "max_image_gen_calls", 0) or 0)
        max_image_gen_calls_fraction = float(
            getattr(model_cfg.image_gen, "max_image_gen_calls_fraction", 0.0) or 0.0
        )
        phase_retry_max_attempts = int(getattr(model_cfg, "phase_retry_max_attempts", 10) or 0)
        phase_retry_initial_sleep = float(getattr(model_cfg, "phase_retry_initial_sleep_seconds", 15.0) or 0.0)
        phase_retry_max_sleep = float(getattr(model_cfg, "phase_retry_max_sleep_seconds", 300.0) or 0.0)
        return InterleavedModelClient(
            model_id=str(model_cfg.id),
            provider_name=str(getattr(model_cfg, "provider", "interleaved")),
            text_client=text_client,
            image_gen_client=image_gen_client,
            output_dir=output_dir,
            enforce_image_call=enforce_image_call,
            enforce_against_decline=enforce_against_decline,
            max_image_gen_calls=max_image_gen_calls,
            max_image_gen_calls_fraction=max_image_gen_calls_fraction,
            phase_retry_max_attempts=phase_retry_max_attempts,
            phase_retry_initial_sleep_seconds=phase_retry_initial_sleep,
            phase_retry_max_sleep_seconds=phase_retry_max_sleep,
        )

    # Non-interleaved fallback: reuse planning's builder verbatim.
    from model_adapter import build_model_client
    return build_model_client(model_cfg=model_cfg, system_prompt=system_prompt)
