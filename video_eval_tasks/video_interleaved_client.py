"""One cached imagined clip per episode, consumed one action at a time."""

from __future__ import annotations

import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))

from omegaconf import DictConfig  # noqa: E402

from model_adapter import BaseModelClient  # noqa: E402
from two_phase_helpers import (  # noqa: E402
    append_phase1_reminder,
    build_inner_text_client,
    extract_tagged_prompt,
    fallback_phase1_prompt,
    peer_artifact_target,
    pick_current_reference_image,
    strip_answer_format_from_messages,
)
from unified_types import Message, ModelResponse, SavedImage  # noqa: E402

from frame_extractor import extract_frames  # noqa: E402
from video_gen_client import VideoGenResult, build_video_gen_client  # noqa: E402


_VIDEO_PROMPT_PATTERN = re.compile(
    r"<video_prompt>\s*(.*?)\s*</video_prompt>", re.IGNORECASE | re.DOTALL
)
_PHASE1_REMINDER = (
    "[Phase 1 instruction]\n"
    "Output exactly one <video_prompt>...</video_prompt> tag and nothing else. "
    "Describe one continuous full-solution clip grounded in the attached initial "
    "state. Do not output action JSON yet."
)
_FALLBACK_VIDEO_PROMPT = (
    "Animate a complete solution from the attached initial board, keeping the "
    "camera fixed and all board geometry and labels unchanged."
)


class CachedVideoModelClient(BaseModelClient):
    """BaseModelClient adapter that reuses one generated clip for an episode.

    Phase 1 and Ark video generation run only on the first environment step.
    Phase 2 runs on every step with the live observation plus the same cached
    frames and returns the normal single-action JSON expected by RolloutRunner.
    """

    def __init__(
        self,
        *,
        model_id: str,
        text_client: BaseModelClient,
        video_gen_client: Any,
        output_dir: Path,
        num_extracted_frames: int,
    ) -> None:
        self.model_id = model_id
        self.text_client = text_client
        self.video_gen_client = video_gen_client
        self.output_dir = Path(output_dir)
        self.num_extracted_frames = max(1, int(num_extracted_frames))
        self._cycle_index = 0
        self._phase1_text: Optional[str] = None
        self._phase1_debug: Dict[str, Any] = {}
        self._phase1_reasoning: Optional[str] = None
        self._video_prompt: Optional[str] = None
        self._video_path: Optional[Path] = None
        self._frames: List[SavedImage] = []
        self._video_result: Optional[VideoGenResult] = None
        self._video_debug: Dict[str, Any] = {}

    @property
    def concurrency_hint(self) -> int:
        return self.text_client.concurrency_hint

    @property
    def effective_system_prompt(self) -> str:
        return self.text_client.effective_system_prompt

    def reset_episode(self) -> None:
        self.text_client.reset_episode()
        self._cycle_index = 0
        self._phase1_text = None
        self._phase1_debug = {}
        self._phase1_reasoning = None
        self._video_prompt = None
        self._video_path = None
        self._frames = []
        self._video_result = None
        self._video_debug = {}

    def describe(self) -> Dict[str, Any]:
        inner = self.text_client.describe()
        return {
            **inner,
            "model_id": self.model_id,
            "model_provider": "vlm_video",
            "text_model_id": inner.get("model_id"),
            "video_gen_model_name": self.video_gen_client.upstream_model_name,
            "video_gen_base_url": self.video_gen_client.base_url,
            "single_video_per_episode": True,
        }

    def _generate_video_once(
        self, *, messages: Sequence[Message], images: Sequence[SavedImage]
    ) -> Optional[ModelResponse]:
        phase1_messages = append_phase1_reminder(
            strip_answer_format_from_messages(messages), _PHASE1_REMINDER
        )
        phase1 = self.text_client.generate(messages=phase1_messages, images=images)
        self._phase1_text = phase1.raw_response_text or ""
        self._phase1_debug = dict(phase1.response_debug or {})
        self._phase1_reasoning = self._phase1_debug.get("reasoning_content")
        if phase1.api_error:
            return ModelResponse(
                raw_response_text="",
                api_error=True,
                api_error_message=phase1.api_error_message,
                response_debug={
                    "interleaved_phase": 1,
                    "phase1_text": self._phase1_text,
                    "phase1_response_debug": self._phase1_debug,
                },
            )

        prompt = extract_tagged_prompt(
            self._phase1_text,
            self._phase1_reasoning,
            _VIDEO_PROMPT_PATTERN,
        )
        fallback_used = prompt is None or prompt.strip().lower() == "none"
        fallback_source: Optional[str] = None
        if fallback_used:
            prompt, fallback_source = fallback_phase1_prompt(
                self._phase1_text,
                self._phase1_reasoning,
                default=_FALLBACK_VIDEO_PROMPT,
            )
        self._video_prompt = prompt
        reference = pick_current_reference_image(images)
        if reference is None:
            return self._video_error("No current observation image was available.")
        target = peer_artifact_target(reference, output_dir=self.output_dir, suffix=".mp4")
        if target is None:
            rel_path = Path("imagined") / f"episode_{uuid.uuid4().hex[:8]}.mp4"
            video_path = self.output_dir / rel_path
        else:
            video_path, rel_string = target
            rel_path = Path(rel_string)

        started = time.monotonic()
        result = self.video_gen_client.generate_to_file(
            prompt=prompt,
            save_path=video_path,
            reference_image_path=reference.abs_path,
        )
        result.latency_seconds = round(time.monotonic() - started, 3)
        self._video_result = result
        self._video_path = video_path if not result.api_error else None
        self._video_debug = {
            "video_prompt": prompt,
            "video_prompt_fallback_used": fallback_used,
            "video_prompt_fallback_source": fallback_source,
            "video_gen_mode": self.video_gen_client.mode,
            "video_gen_reference_image": str(reference.abs_path),
            "video_gen_rel_path": str(rel_path),
            "imagined_video_rel_path": str(rel_path),
            "video_gen_api_error": result.api_error,
            "video_gen_api_error_message": result.api_error_message,
            "video_gen_latency_seconds": result.latency_seconds,
        }
        if result.api_error:
            return self._video_error(result.api_error_message or "Video generation failed.")
        try:
            frame_paths = extract_frames(
                video_path,
                out_dir=video_path.with_suffix("").with_name(video_path.stem + "_frames"),
                num_frames=self.num_extracted_frames,
            )
        except Exception as exc:  # noqa: BLE001
            self._video_debug["frame_extract_error"] = f"{type(exc).__name__}: {exc}"
            return self._video_error(self._video_debug["frame_extract_error"])
        self._frames = [
            SavedImage(
                ref_id=f"video-frame-{uuid.uuid4().hex[:10]}",
                label="frame",
                abs_path=path,
                rel_path=str(path.relative_to(self.output_dir)),
            )
            for path in frame_paths
        ]
        self._video_debug["video_frame_rel_paths"] = [frame.rel_path for frame in self._frames]
        self._video_debug["video_frame_ref_ids"] = [frame.ref_id for frame in self._frames]
        return None

    def _video_error(self, message: str) -> ModelResponse:
        return ModelResponse(
            raw_response_text="",
            api_error=True,
            api_error_message=message,
            response_debug=self._response_debug([], cached=False),
        )

    def _response_debug(
        self, followup_blocks: List[Dict[str, Any]], *, cached: bool
    ) -> Dict[str, Any]:
        debug = {
            **self._video_debug,
            "interleaved_phase": 2,
            "phase1_text": self._phase1_text,
            "phase1_response_debug": self._phase1_debug,
            "phase1_reasoning_content": self._phase1_reasoning,
            "phase2_followup_blocks": followup_blocks,
            "vlm_video_cycle_index": self._cycle_index,
        }
        if cached:
            debug["video_gen_skip_reason"] = "cached_from_first_cycle"
            debug["vlm_video_cached_from_cycle"] = 0
        return debug

    def generate(
        self, *, messages: Sequence[Message], images: Sequence[SavedImage]
    ) -> ModelResponse:
        cached = self._phase1_text is not None
        if not cached:
            error = self._generate_video_once(messages=messages, images=images)
            if error is not None:
                return error

        followup: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Imagined trajectory ({len(self._frames)} frames):"}
        ]
        for index, frame in enumerate(self._frames, start=1):
            followup.extend(
                [
                    {"type": "text", "text": f"Frame {index}:"},
                    {"type": "image_ref", "ref_id": frame.ref_id},
                ]
            )
        followup.append(
            {
                "type": "text",
                "text": (
                    "You are now in PHASE 2. The live observation is authoritative; "
                    "use the cached trajectory only as a plan. Output exactly one "
                    "next action using the task's [Answer Format], and nothing else."
                ),
            }
        )
        phase2_messages = list(messages) + [
            {"role": "assistant", "content": self._phase1_text or ""},
            {"role": "user", "content": followup},
        ]
        phase2 = self.text_client.generate(
            messages=phase2_messages,
            images=[*images, *self._frames],
        )
        debug = {
            **dict(phase2.response_debug or {}),
            **self._response_debug(followup, cached=cached),
        }
        self._cycle_index += 1
        return ModelResponse(
            raw_response_text=phase2.raw_response_text or "",
            api_error=bool(phase2.api_error),
            api_error_message=phase2.api_error_message,
            response_debug=debug,
        )


def build_cached_video_client(
    *, cfg: DictConfig, system_prompt: str, output_dir: Path
) -> CachedVideoModelClient:
    return CachedVideoModelClient(
        model_id=str(cfg.model.id),
        text_client=build_inner_text_client(
            text_cfg=cfg.model.text, system_prompt=system_prompt
        ),
        video_gen_client=build_video_gen_client(cfg.model.video_gen),
        output_dir=output_dir,
        num_extracted_frames=int(cfg.run.num_extracted_frames),
    )
