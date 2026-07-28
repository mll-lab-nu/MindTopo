"""Reasoning runner for the interleaved harness.

Iterates `question.jsonl` rows for static-reasoning envs (continuity_2d_maze,
enclosure_sheep) and runs each row through the same `InterleavedModelClient`
the gym path uses. Per sample: one phase-1 reason → image gen → phase-2 final
answer cycle, then parse + score against the ground truth.

The runner mirrors planning's `RolloutRunner.run()` patterns where they make
sense (resume via `load_jsonl_ids`, incremental `summary.json`, prompt-artifact
generation via a synthesized one-step `TurnRecord`).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

# Reuse planning's modules (writer, types, repo-path helper).
_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))

from artifact_writer import ArtifactWriter  # noqa: E402
from model_adapter import BaseModelClient  # noqa: E402
from topobench_eval.answer_parser import ParseResult, parse_answer_result  # noqa: E402
from topobench_eval.answer_scoring import answers_equal  # noqa: E402
from unified_types import (  # noqa: E402
    EpisodeSpec,
    ModelResponse,
    PromptParts,
    SavedImage,
    TurnContext,
    TurnRecord,
)

from reasoning_loader import ReasoningSample, load_samples


# Coarse buckets for the per-run skip-reason breakdown. The runtime emits a
# more granular reason (e.g. "image_gen_failed: HTTP 429: ...") which collapses
# here into a single bucket so the summary stays readable.
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


class ReasoningRunner:
    """Single-shot QA runner for reasoning envs.

    Drives the same `InterleavedModelClient.generate(messages, images)` interface
    as the gym path; differences are: one sample = one model call (with internal
    phase1 + image_gen + phase2), no env / Playwright, scoring is exact-equality
    on a parsed answer.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        writer: ArtifactWriter,
        model_client: Optional[BaseModelClient],
    ) -> None:
        self.cfg = cfg
        self.writer = writer
        self.model_client = model_client
        # Populated by run(); summary surfaces drop counts so a partial-images
        # situation is visible in the artifact (rather than just a stderr line).
        self._load_result: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        manifest_offset = int(self.cfg.run.manifest_offset)
        manifest_limit = self.cfg.run.manifest_limit
        manifest_limit_int = int(manifest_limit) if manifest_limit not in (None, "null") else None

        load_result = load_samples(
            env_cfg=self.cfg.env,
            manifest_offset=manifest_offset,
            manifest_limit=manifest_limit_int,
        )
        self._load_result = load_result

        completed_ids = self.writer.load_jsonl_ids(filename="model_answer.jsonl")
        question_written_ids = self.writer.load_jsonl_ids(filename="question.jsonl")
        existing_summary = self.writer.load_existing_summary()
        sample_payloads: List[Dict[str, Any]] = list(existing_summary.get("samples") or [])
        # Drop stale payloads not in the current resume set
        sample_payloads = [p for p in sample_payloads if p.get("id") in completed_ids]

        copy_source_images = bool(self.cfg.run.write_step_images)

        for sample in load_result.samples:
            if sample.id in completed_ids:
                self._print_resume(sample)
                continue
            payload = self._run_sample(sample, copy_source_images=copy_source_images)
            sample_payloads.append(payload)
            completed_ids.add(sample.id)
            # Append question.jsonl row only the first time we see this id
            if sample.id not in question_written_ids:
                self.writer.append_jsonl_row(filename="question.jsonl", row=self._question_row(sample))
                question_written_ids.add(sample.id)
            self.writer.append_jsonl_row(filename="model_answer.jsonl", row=payload)
            self._write_summary(sample_payloads)

        summary = self._build_summary(sample_payloads)
        self.writer.write_summary(summary_payload=summary)
        return summary

    # ------------------------------------------------------------------
    # Per-sample execution
    # ------------------------------------------------------------------

    def _run_sample(
        self,
        sample: ReasoningSample,
        *,
        copy_source_images: bool,
    ) -> Dict[str, Any]:
        if self.model_client is not None:
            self.model_client.reset_episode()

        saved_images = self._materialize_source_images(sample, copy_source_images=copy_source_images)
        image_index = {img.ref_id: img for img in saved_images}

        prompt_parts = self._build_prompt(sample, saved_images)
        request_messages = [{"role": "user", "content": prompt_parts.model_blocks}]

        if self.model_client is None:
            # Local-policy bypass (oracle/random/greedy don't make sense here, but
            # keep the path open for future no-API reasoning baselines).
            response = ModelResponse(
                raw_response_text="",
                api_error=True,
                api_error_message="No model client wired for reasoning path.",
                response_debug={},
            )
        else:
            response = self.model_client.generate(messages=request_messages, images=saved_images)

        parsed = parse_answer_result(response.raw_response_text or "", sample.contract)
        correct = self._is_correct(parsed, sample)

        # Synthesize a single-step TurnRecord so the writer can emit prompt artifacts.
        turn = self._synthesize_turn(
            sample=sample,
            saved_images=saved_images,
            prompt_parts=prompt_parts,
            response=response,
            parsed_answer=parsed.answer,
            invalid_response=parsed.invalid_response,
            correct=correct,
        )
        if not response.api_error:
            artifact_debug_messages = list(request_messages)
            if self.model_client is not None:
                sys_prompt = self.model_client.effective_system_prompt
                if sys_prompt:
                    artifact_debug_messages.insert(0, {"role": "system", "content": sys_prompt})
            self.writer.write_prompt_artifacts(
                turn=turn,
                debug_messages=artifact_debug_messages,
                pure_messages=request_messages,
                image_index=image_index,
            )

        return self._build_answer_row(
            sample=sample,
            saved_images=saved_images,
            response=response,
            parsed=parsed,
            correct=correct,
        )

    def _materialize_source_images(
        self,
        sample: ReasoningSample,
        *,
        copy_source_images: bool,
    ) -> List[SavedImage]:
        """Copy each source PNG under <output_dir>/source/<id>/ so prompt artifacts
        and model_answer.jsonl carry self-contained relative paths.

        When `write_step_images=false`, point at the original abs path instead;
        the prompt artifact's [Image] block will then reference that absolute path.
        """
        out: List[SavedImage] = []
        for idx, (abs_path, relpath) in enumerate(zip(sample.image_paths, sample.image_relpaths)):
            ref_id = f"src-{sample.id}-{idx:02d}"
            if copy_source_images:
                png_bytes = abs_path.read_bytes()
                saved = self.writer.save_image(
                    path_segments=("source", sample.id),
                    filename=abs_path.name,
                    png_bytes=png_bytes,
                    ref_id=ref_id,
                    label="current",
                )
            else:
                # Use the original location — abs_path / abs_path-stringified rel_path.
                saved = SavedImage(
                    ref_id=ref_id,
                    label="current",
                    abs_path=abs_path,
                    rel_path=str(abs_path),
                )
            out.append(saved)
        return out

    def _build_prompt(
        self,
        sample: ReasoningSample,
        saved_images: List[SavedImage],
    ) -> PromptParts:
        # The question already contains its own [Image] / [Answer Format] blocks;
        # we feed it verbatim and append the image_refs the question references.
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": sample.question}]
        for img in saved_images:
            blocks.append({"type": "image_ref", "ref_id": img.ref_id})
        return PromptParts(model_blocks=blocks, debug_blocks=list(blocks), pure_blocks=list(blocks))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _is_correct(self, parsed: ParseResult, sample: ReasoningSample) -> bool:
        return parsed.semantic_valid and answers_equal(
            parsed.answer,
            sample.gt_answer,
            sample.contract,
        )

    # ------------------------------------------------------------------
    # Artifact synthesis
    # ------------------------------------------------------------------

    def _synthesize_turn(
        self,
        *,
        sample: ReasoningSample,
        saved_images: List[SavedImage],
        prompt_parts: PromptParts,
        response: ModelResponse,
        parsed_answer: str,
        invalid_response: bool,
        correct: bool,
    ) -> TurnRecord:
        spec = EpisodeSpec(
            episode_id=sample.id,
            level=str(sample.meta_info.get("difficulty", "reasoning")),
            seed=int(sample.meta_info.get("seed", 0) or 0),
            repeat_index=int(sample.meta_info.get("repeat_index", 0) or 0),
            category=list(sample.category),
            metadata=dict(sample.meta_info),
        )
        context = TurnContext(
            obs_rgb=None,
            planner_state=None,
            symbolic_observation=None,
            legal_actions=[],
            step_index=0,
            step_budget=1,
            termination_state=None,
            image_views=[],
            prompt_metadata={"task_name": str(self.cfg.env.name)},
        )
        return TurnRecord(
            spec=spec,
            step_index=0,
            context=context,
            saved_images=list(saved_images),
            prompt_parts=prompt_parts,
            parsed_action=parsed_answer,
            flattened_answer=parsed_answer,
            raw_response_text=response.raw_response_text or "",
            api_error=bool(response.api_error),
            api_error_message=response.api_error_message,
            invalid_response=invalid_response,
            illegal=False,
            reward=float(correct),
            terminated=True,
            truncated=False,
            reason="correct" if correct else "incorrect",
            response_debug=dict(response.response_debug or {}),
            next_info={"correct": correct},
            after_images=[],
        )

    # ------------------------------------------------------------------
    # Output rows
    # ------------------------------------------------------------------

    def _question_row(self, sample: ReasoningSample) -> Dict[str, Any]:
        return {
            "id": sample.id,
            "category": list(sample.category),
            "type": sample.type,
            "question": sample.question,
            "images": list(sample.image_relpaths),
            "gt_answer": sample.gt_answer,
            "meta_info": dict(sample.meta_info),
        }

    def _build_answer_row(
        self,
        *,
        sample: ReasoningSample,
        saved_images: List[SavedImage],
        response: ModelResponse,
        parsed: ParseResult,
        correct: bool,
    ) -> Dict[str, Any]:
        debug = dict(response.response_debug or {})
        imagined_relpath = debug.get("imagined_image_rel_path")
        meta_info = {
            **dict(sample.meta_info),
            "task_name": str(self.cfg.env.name),
            "model_id": str(self.cfg.model.id),
            "answer_mode": sample.answer_mode,
            "legal_spec": sample.legal_spec,
        }
        trajectory_step = {
            "step_index": 0,
            "current_images": [img.rel_path for img in saved_images],
            "imagined_images": [imagined_relpath] if imagined_relpath else [],
            "state": "success" if correct else "failed",
            "answer": parsed.answer,
            "gt": sample.gt_answer,
            "correct": bool(correct),
            "parse_method": parsed.method,
            "parse_error": parsed.error,
            "answer_format_hit": parsed.format_valid,
            "invalid_response": parsed.invalid_response,
            "api_error": bool(response.api_error),
            "raw_response_text": response.raw_response_text or "",
            "image_prompt": debug.get("image_prompt"),
            "phase1_text": debug.get("phase1_text"),
            "image_gen_skip_reason": debug.get("image_gen_skip_reason"),
        }
        if response.api_error:
            trajectory_step["api_error_message"] = response.api_error_message
        return {
            "id": sample.id,
            "category": list(sample.category),
            "type": "reasoning_qa",
            "meta_info": meta_info,
            "trajectory": [trajectory_step],
        }

    # ------------------------------------------------------------------
    # Summary aggregation
    # ------------------------------------------------------------------

    def _print_resume(self, sample: ReasoningSample) -> None:
        print(
            f"[resume-skip] env={self.cfg.env.name} sample_id={sample.id}"
        )

    def _write_summary(self, sample_payloads: List[Dict[str, Any]]) -> None:
        # Incremental writes omit the full `samples` list — `model_answer.jsonl`
        # already records every per-sample row. Including the list here makes
        # the per-commit serialization cost O(N²) on large runs.
        summary = self._build_summary(sample_payloads, include_samples=False)
        self.writer.write_summary_only(summary_payload=summary)

    def _build_summary(
        self,
        sample_payloads: List[Dict[str, Any]],
        *,
        include_samples: bool = True,
    ) -> Dict[str, Any]:
        total = len(sample_payloads)
        correct = 0
        invalid_responses = 0
        api_errors = 0
        image_gen_skipped = 0
        per_type_total: Dict[str, int] = defaultdict(int)
        per_type_correct: Dict[str, int] = defaultdict(int)
        skip_reason_counts: Dict[str, int] = defaultdict(int)
        # Episode-shaped projection so artifact_writer._render_summary_md emits a
        # By-Difficulty table for reasoning runs (one sample = one "episode").
        episode_rows: List[Dict[str, Any]] = []
        for payload in sample_payloads:
            steps = payload.get("trajectory") or []
            if not steps:
                continue
            step = steps[0]
            sample_type = next(iter(payload.get("category") or [payload.get("type", "reasoning_qa")]), "reasoning_qa")
            # Use the most-specific category leaf for breakdown
            categories = payload.get("category") or []
            if isinstance(categories, list) and categories:
                sample_type = str(categories[-1])
            per_type_total[sample_type] += 1
            is_correct = bool(step.get("correct"))
            if is_correct:
                correct += 1
                per_type_correct[sample_type] += 1
            is_invalid = bool(step.get("invalid_response"))
            if is_invalid:
                invalid_responses += 1
            is_api_err = bool(step.get("api_error"))
            if is_api_err:
                api_errors += 1
            skip_reason = step.get("image_gen_skip_reason")
            if skip_reason:
                image_gen_skipped += 1
                skip_reason_counts[_skip_reason_bucket(skip_reason)] += 1
            meta_info = payload.get("meta_info") or {}
            episode_rows.append({
                "level": str(meta_info.get("difficulty", "reasoning")),
                "success": is_correct,
                "total_steps": 1,
                "illegal_moves": 0,
                "invalid_responses": 1 if is_invalid else 0,
                "api_errors": 1 if is_api_err else 0,
            })

        per_type_accuracy = {
            t: round(per_type_correct[t] / per_type_total[t], 4) if per_type_total[t] else 0.0
            for t in per_type_total
        }
        model_info = self.model_client.describe() if self.model_client is not None else {}
        summary: Dict[str, Any] = {
            "env": str(self.cfg.env.name),
            "eval_kind": "reasoning",
            "model_id": str(model_info.get("model_id", self.cfg.model.id)),
            "model_name": str(model_info.get("model_name", getattr(self.cfg.model, "name", self.cfg.model.id))),
            "model_provider": str(model_info.get("model_provider", getattr(self.cfg.model, "provider", "local"))),
            "model_base_url": str(model_info.get("model_base_url", getattr(self.cfg.model, "base_url", ""))),
            "interleaved_text_model_id": str(model_info.get("interleaved_text_model_id", "")),
            "image_gen_model_name": str(model_info.get("image_gen_model_name", "")),
            "image_gen_base_url": str(model_info.get("image_gen_base_url", "")),
            # Canonical episode-shaped names so artifact_writer._render_summary_md
            # treats reasoning runs the same as gym rollouts.
            # "Success" here = correct answer (single-shot QA = one episode/sample).
            "total_episodes": total,
            "successes": correct,
            "success_rate": (correct / total) if total else 0.0,
            "per_type_accuracy": per_type_accuracy,
            "invalid_responses": invalid_responses,
            "api_errors": api_errors,
            "image_gen_skipped": image_gen_skipped,
            "image_gen_skip_reasons": dict(skip_reason_counts),
            "loader": (
                {
                    "rows_in_slice": self._load_result.rows_in_slice,
                    "samples_loaded": len(self._load_result.samples),
                    "dropped_missing_images": self._load_result.dropped_missing_images,
                    "dropped_malformed": self._load_result.dropped_malformed,
                }
                if self._load_result is not None
                else {}
            ),
            "config": {
                "env": OmegaConf.to_container(self.cfg.env, resolve=True),
                "model": OmegaConf.to_container(self.cfg.model, resolve=True),
                "run": OmegaConf.to_container(self.cfg.run, resolve=True),
            },
            "generated_at": time.time(),
        }
        if include_samples:
            summary["samples"] = list(sample_payloads)
            summary["episodes"] = episode_rows
        return summary
