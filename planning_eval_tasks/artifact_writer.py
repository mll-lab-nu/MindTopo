from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from evaluator_utils import resolve_repo_path
from planning_summary import write_model_planning_summary_for_run
from unified_types import EpisodeRunResult, SavedImage, TurnRecord


class ArtifactWriter:
    def __init__(self, *, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_dir = resolve_repo_path(str(cfg.run.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_prompt_artifacts_enabled = bool(cfg.run.write_prompt_artifacts)
        self.write_step_images_enabled = bool(cfg.run.write_step_images)

    def discard_step_artifacts(self, *, result: EpisodeRunResult) -> None:
        """Remove completed-episode image trees when retention is disabled.

        Images still exist for the duration of an episode because remote model
        adapters read them from disk. Cleanup happens only after the answer row
        has been constructed, and every target directory is verified to be a
        strict descendant of this run's output directory.
        """
        if self.write_step_images_enabled:
            return
        output_root = self.output_dir.resolve()
        episode_dirs: set[Path] = set()
        for turn in result.turns:
            for image in (*turn.saved_images, *turn.after_images):
                parent = image.abs_path.resolve().parent
                try:
                    parent.relative_to(output_root)
                except ValueError as exc:
                    raise ValueError(
                        f"refusing to prune artifact outside output dir: {parent}"
                    ) from exc
                if parent in {output_root, output_root / "images"}:
                    raise ValueError(f"refusing to prune broad artifact dir: {parent}")
                episode_dirs.add(parent)
        for episode_dir in episode_dirs:
            if episode_dir.exists():
                shutil.rmtree(episode_dir)

    def save_image(
        self,
        *,
        path_segments: Sequence[str],
        filename: str,
        png_bytes: bytes,
        ref_id: str,
        label: str,
    ) -> SavedImage:
        output_path = self.output_dir.joinpath(*path_segments, filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)
        return SavedImage(
            ref_id=ref_id,
            label=label,
            abs_path=output_path,
            rel_path=str(output_path.relative_to(self.output_dir)),
        )

    def _serialize_prompt_blocks(self, content: Any, *, image_index: Mapping[str, SavedImage]) -> List[str]:
        if isinstance(content, str):
            text = content.strip()
            return [text] if text else []
        if not isinstance(content, list):
            if content is None:
                return []
            return [json.dumps(content, ensure_ascii=False)]
        blocks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    blocks.append(text)
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    blocks.append(text)
            elif item_type == "image_ref":
                image = image_index.get(str(item.get("ref_id", "")))
                blocks.append("[Image]")
                blocks.append(image.rel_path if image is not None else "<missing-image>")
            else:
                serialized = json.dumps(item, ensure_ascii=False).strip()
                if serialized:
                    blocks.append(serialized)
        return blocks

    def write_prompt_artifacts(
        self,
        *,
        turn: TurnRecord,
        debug_messages: Sequence[Dict[str, Any]],
        pure_messages: Sequence[Dict[str, Any]],
        image_index: Mapping[str, SavedImage],
    ) -> Dict[str, str]:
        if not self.write_prompt_artifacts_enabled:
            return {}
        if turn.saved_images:
            base_dir = Path(turn.saved_images[0].rel_path).parent
        else:
            base_dir = Path("images") / turn.spec.episode_id
        prompt_path = self.output_dir / base_dir / f"step_{turn.step_index:04d}_prompt_debug.txt"
        pure_path = self.output_dir / base_dir / f"step_{turn.step_index:04d}_pure_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        debug_dict = turn.response_debug if isinstance(turn.response_debug, dict) else {}
        is_interleaved = "phase1_text" in debug_dict
        if is_interleaved:
            prompt_blocks = self._render_interleaved_debug(turn, debug_messages, debug_dict, image_index)
        else:
            prompt_blocks = self._render_single_phase_debug(turn, debug_messages, image_index)
        prompt_path.write_text("\n\n".join(prompt_blocks) + "\n", encoding="utf-8")

        pure_blocks: List[str] = []
        for message in pure_messages:
            pure_blocks.extend(self._serialize_prompt_blocks(message.get("content"), image_index=image_index))
        pure_path.write_text("\n\n".join(pure_blocks) + "\n", encoding="utf-8")
        return {
            "prompt_debug": str(prompt_path.relative_to(self.output_dir)),
            "pure_prompt": str(pure_path.relative_to(self.output_dir)),
        }

    def _render_single_phase_debug(
        self,
        turn: TurnRecord,
        debug_messages: Sequence[Dict[str, Any]],
        image_index: Mapping[str, SavedImage],
    ) -> List[str]:
        blocks: List[str] = ["[prompt_messages]"]
        for message in debug_messages:
            blocks.append(f"[{message.get('role', 'message')}]")
            blocks.extend(self._serialize_prompt_blocks(message.get("content"), image_index=image_index))
        blocks.extend(
            [
                "[model_response]",
                turn.raw_response_text or "<empty>",
            ]
        )
        reasoning = (turn.response_debug or {}).get("reasoning_content") if isinstance(turn.response_debug, dict) else None
        if isinstance(reasoning, str) and reasoning.strip():
            blocks.extend(["[reasoning_content]", reasoning.strip()])
        blocks.extend(
            [
                "[parsed_answer]",
                json.dumps(turn.flattened_answer, ensure_ascii=False) if turn.flattened_answer is not None else "<none>",
                "[api_error]",
                turn.api_error_message or ("<transport-error>" if turn.api_error else "<none>"),
                "[response_debug]",
                json.dumps(turn.response_debug, indent=2, ensure_ascii=False) if turn.response_debug else "<none>",
            ]
        )
        return blocks

    def _render_interleaved_debug(
        self,
        turn: TurnRecord,
        debug_messages: Sequence[Dict[str, Any]],
        debug: Mapping[str, Any],
        image_index: Mapping[str, SavedImage],
    ) -> List[str]:
        """Chronological per-step artifact for interleaved runs.

        Order: phase 1 prompt → phase 1 response (+ reasoning) → image gen
        request/result → phase 2 prompt → phase 2 response (+ reasoning) →
        parsed answer → api error → full response_debug JSON (kept as a
        backstop for fields not explicitly surfaced above).
        """
        blocks: List[str] = []

        blocks.append("[phase_1_prompt]")
        for message in debug_messages:
            blocks.append(f"[{message.get('role', 'message')}]")
            blocks.extend(self._serialize_prompt_blocks(message.get("content"), image_index=image_index))

        blocks.append("[phase_1_response]")
        blocks.append(str(debug.get("phase1_text") or "<empty>"))

        phase1_reasoning = debug.get("phase1_reasoning_content")
        if isinstance(phase1_reasoning, str) and phase1_reasoning.strip():
            blocks.append("[phase_1_reasoning_content]")
            blocks.append(phase1_reasoning)

        # Rendering branches off whether the cycle ran image-gen (interleaved
        # with one imagined PNG) or video-gen (vlm_video with mp4 + N frames).
        # vlm_video_client populates `video_prompt` / `imagined_video_rel_path`
        # / `video_frame_rel_paths`; interleaved_client populates `image_prompt`
        # / `imagined_image_rel_path`. Both share `phase1_text`, which is what
        # selects this renderer in the first place.
        is_video_cycle = (
            "video_prompt" in debug
            or "imagined_video_rel_path" in debug
            or "video_frame_rel_paths" in debug
        )
        section = "[video_gen]" if is_video_cycle else "[image_gen]"
        skip = debug.get("video_gen_skip_reason") if is_video_cycle else debug.get("image_gen_skip_reason")
        mode = (debug.get("video_gen_mode") if is_video_cycle else debug.get("image_gen_mode")) or "<unset>"
        prompt_key = "video_prompt" if is_video_cycle else "image_prompt"
        ref_key = "video_gen_reference_image" if is_video_cycle else "image_gen_reference_image"

        blocks.append(section)
        if skip:
            blocks.append(f"status=skipped ({skip})")
            ip = debug.get(prompt_key)
            if ip:
                blocks.append(f"{prompt_key}={ip}")
        else:
            blocks.append(f"status=ran  mode={mode}")
            blocks.append(f"{prompt_key}={debug.get(prompt_key) or '<none>'}")
            ref = debug.get(ref_key)
            if ref:
                blocks.append(f"reference_image={ref}")
            if is_video_cycle:
                imagined_video = debug.get("imagined_video_rel_path") or debug.get("video_gen_rel_path")
                if imagined_video:
                    blocks.append(f"imagined_video={imagined_video}")
                frame_paths = debug.get("video_frame_rel_paths") or []
                if frame_paths:
                    blocks.append(f"imagined_frames ({len(frame_paths)}):")
                    for idx, p in enumerate(frame_paths):
                        blocks.append(f"  {idx + 1}. {p}")
            else:
                imagined = debug.get("imagined_image_rel_path")
                if imagined:
                    blocks.append(f"imagined_image={imagined}")

        blocks.append("[phase_2_prompt]")
        followup_blocks = debug.get("phase2_followup_blocks") or []
        if followup_blocks:
            # Image refs in followup_blocks are minted inside the inner client
            # (InterleavedModelClient or VLMVideoModelClient) and never reach
            # the runner's image_index, so augment locally from response_debug
            # to avoid <missing-image> in the log.
            followup_image_index: Dict[str, SavedImage] = dict(image_index)
            # Interleaved (single imagined PNG).
            imagined_ref_id = debug.get("imagined_image_ref_id")
            imagined_rel = debug.get("imagined_image_rel_path")
            if imagined_ref_id and imagined_rel and imagined_ref_id not in followup_image_index:
                followup_image_index[str(imagined_ref_id)] = SavedImage(
                    ref_id=str(imagined_ref_id),
                    label="imagined",
                    abs_path=self.output_dir / str(imagined_rel),
                    rel_path=str(imagined_rel),
                )
            # vlm_video (N frames). The followup_blocks reference each frame by
            # ref_id; vlm_video_client emits `video_frame_ref_ids` in response_debug
            # zipped with `video_frame_rel_paths`.
            frame_ref_ids = debug.get("video_frame_ref_ids") or []
            frame_rel_paths = debug.get("video_frame_rel_paths") or []
            for ref_id, rel_path in zip(frame_ref_ids, frame_rel_paths):
                if ref_id and rel_path and ref_id not in followup_image_index:
                    followup_image_index[str(ref_id)] = SavedImage(
                        ref_id=str(ref_id),
                        label="frame",
                        abs_path=self.output_dir / str(rel_path),
                        rel_path=str(rel_path),
                    )
            blocks.append("[user (continuation)]")
            blocks.extend(self._serialize_prompt_blocks(followup_blocks, image_index=followup_image_index))
        else:
            blocks.append("<no followup recorded>")

        blocks.append("[phase_2_response]")
        blocks.append(turn.raw_response_text or "<empty>")

        phase2_reasoning = debug.get("reasoning_content")
        if isinstance(phase2_reasoning, str) and phase2_reasoning.strip():
            blocks.append("[phase_2_reasoning_content]")
            blocks.append(phase2_reasoning)

        blocks.extend(
            [
                "[parsed_answer]",
                json.dumps(turn.flattened_answer, ensure_ascii=False) if turn.flattened_answer is not None else "<none>",
                "[api_error]",
                turn.api_error_message or ("<transport-error>" if turn.api_error else "<none>"),
                "[response_debug]",
                json.dumps(turn.response_debug, indent=2, ensure_ascii=False),
            ]
        )
        return blocks

    def write_episode_summary_json(self, *, base_dir: Path, payload: Dict[str, Any]) -> Path:
        path = self.output_dir / base_dir / "episode_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def write_episode_demo_gif(self, *, result: EpisodeRunResult) -> Optional[Path]:
        frame_paths: List[Path] = []
        for turn in result.turns:
            for img in turn.saved_images:
                if img.label == "current":
                    frame_paths.append(img.abs_path)
                    break
            else:
                if turn.saved_images:
                    frame_paths.append(turn.saved_images[0].abs_path)
            for img in turn.after_images:
                if img.label == "after":
                    frame_paths.append(img.abs_path)
                    break
            else:
                if turn.after_images:
                    frame_paths.append(turn.after_images[0].abs_path)
        if not frame_paths:
            return None
        try:
            from PIL import Image as PILImage  # type: ignore[import]
        except ImportError:
            return None
        frames = []
        for p in frame_paths:
            try:
                frames.append(PILImage.open(p).convert("RGB"))
            except Exception:
                continue
        if not frames:
            return None
        gif_path = frame_paths[0].parent / "demo.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=False)
        return gif_path

    def write_summary(self, *, summary_payload: Dict[str, Any]) -> Path:
        """Write summary.json and snapshot the resolved config alongside."""
        output_json = self.write_summary_only(summary_payload=summary_payload)
        self.write_resolved_config(summary_payload=summary_payload)
        try:
            write_model_planning_summary_for_run(
                summary_path=output_json,
                summary_payload=summary_payload,
            )
        except Exception as exc:
            print(f"[artifact-writer] failed to write model planning summary: {exc}")
        return output_json

    def write_summary_only(self, *, summary_payload: Dict[str, Any]) -> Path:
        """Write summary.json without touching resolved_config.yaml.

        Use for per-episode incremental updates — the cfg never changes mid-run,
        so re-serializing it on every commit is pure waste (OmegaConf.to_yaml
        with resolve=True is non-trivial on deep configs).
        """
        output_json = self.summary_path()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically: a non-atomic write_text truncates then writes, so a
        # kill mid-write (this runs on every per-episode commit) leaves a torn or
        # 0-byte summary that crashes the next resume. Stage to a tmp file and
        # rename — rename is atomic, so resume always sees a complete file.
        tmp = output_json.with_suffix(output_json.suffix + ".tmp")
        tmp.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(output_json)
        return output_json

    def write_resolved_config(self, *, summary_payload: Optional[Dict[str, Any]] = None) -> Path:
        output_config = self.config_path()
        output_config.parent.mkdir(parents=True, exist_ok=True)
        output_config.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")
        # Only render the markdown summary on the final write — incremental
        # writes drop the `episodes` list and would produce an empty breakdown.
        if summary_payload is not None and isinstance(summary_payload.get("episodes"), list):
            summary_md = self._render_summary_md(summary_payload)
            summary_md_path = self.summary_path().parent / "summary.md"
            summary_md_path.parent.mkdir(parents=True, exist_ok=True)
            summary_md_path.write_text(summary_md, encoding="utf-8")
        return output_config

    @staticmethod
    def _render_summary_md(summary: Dict[str, Any]) -> str:
        def fmt_pct(n: float, d: float) -> str:
            return f"{(n / d * 100):.2f}%" if d else "—"

        def fmt_float(value: Any, digits: int = 2) -> str:
            if value is None:
                return "—"
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "—"

        env = str(summary.get("env", "?"))
        model_name = str(summary.get("model_name") or summary.get("model_id") or "?")
        model_id = str(summary.get("model_id") or "?")
        total = int(summary.get("total_episodes", 0) or 0)
        # success_rate=None signals "no verdict" (e.g. video replay modes) — drop
        # the success columns instead of printing 0% which would be misleading.
        raw_success_rate = summary.get("success_rate")
        has_success = raw_success_rate is not None
        successes = int(summary.get("successes", 0) or 0)
        success_rate = float(raw_success_rate or 0.0)
        skipped = summary.get("skipped_episodes") or []
        episodes = summary.get("episodes") or []

        # Two episode payload shapes coexist: planning/interleaved set
        # `level`/`success`/`total_steps`/... at the top, video runs nest
        # them under `meta_info` (model_answer.jsonl row shape). Read
        # either to keep the difficulty table working in both cases.
        def _field(payload: Dict[str, Any], *names: str) -> Any:
            meta = payload.get("meta_info") or {}
            for n in names:
                if payload.get(n) not in (None, ""):
                    return payload.get(n)
                if meta.get(n) not in (None, ""):
                    return meta.get(n)
            return None

        def _level(payload: Dict[str, Any]) -> str:
            # Some envs (e.g. separation_one_stroke) use spec.level as a seed
            # integer and stash the human-readable bucket under
            # meta_info.difficulty. Prefer top-level `level` (planning shape),
            # then `meta_info.difficulty` (video shape), and only fall back to
            # `meta_info.level` last so we never bucket by seed-as-level.
            meta = payload.get("meta_info") or {}
            for candidate in (payload.get("level"), meta.get("difficulty"), meta.get("level")):
                if candidate not in (None, ""):
                    return str(candidate)
            return "?"

        by_diff: Dict[str, Dict[str, float]] = {}
        for payload in episodes:
            if not isinstance(payload, dict):
                continue
            meta_info = payload.get("meta_info")
            level = "?"
            if isinstance(meta_info, dict) and meta_info.get("task_name") == "enclosure_chat_noir":
                policy = str(meta_info.get("cat_policy_name") or "").strip().lower()
                level = policy if policy in {"easy", "medium", "hard"} else level
            if isinstance(meta_info, dict) and meta_info.get("task_name") == "separation_one_stroke":
                explicit = meta_info.get("difficulty")
                if explicit not in (None, ""):
                    level = str(explicit)
                else:
                    board_size = int(meta_info.get("board_size") or 0)
                    board_size_difficulty = {4: "easy", 5: "medium", 6: "hard"}
                    level = board_size_difficulty.get(board_size, level)
            if level == "?":
                level = _level(payload)
            bucket = by_diff.setdefault(
                level,
                {"n": 0, "successes": 0, "steps": 0, "illegal": 0, "invalid": 0, "api_errors": 0},
            )
            bucket["n"] += 1
            if _field(payload, "success"):
                bucket["successes"] += 1
            bucket["steps"] += int(_field(payload, "total_steps") or 0)
            bucket["illegal"] += int(_field(payload, "illegal_moves") or 0)
            bucket["invalid"] += int(_field(payload, "invalid_responses") or 0)
            bucket["api_errors"] += int(_field(payload, "api_errors") or 0)

        preferred = [d for d in ("easy", "medium", "hard") if d in by_diff]
        other = sorted(d for d in by_diff if d not in preferred)
        diff_order = preferred + other

        # Each pipeline only sets the fields it can compute meaningfully:
        #   planning gym → success_rate, avg_steps_*, avg_illegal_moves, avg_*
        #   reasoning   → success_rate, invalid_responses, api_errors (raw counts)
        #   video vlm    → success_rate, avg_steps_all
        #   video replay → none of the above (just total + by-difficulty count)
        # The renderer skips lines/columns whose source field is absent so each
        # summary.md only reports metrics that actually apply.
        has_avg_steps_all = "avg_steps_all" in summary
        has_avg_steps_succ = "avg_steps_on_success" in summary
        has_avg_illegal = "avg_illegal_moves" in summary
        has_avg_invalid = "avg_invalid_responses" in summary
        has_invalid_count = "invalid_responses" in summary
        has_avg_api_err = "avg_api_errors" in summary
        has_api_err_count = "api_errors" in summary

        lines = [
            f"# {env} — {model_name}",
            "",
            f"- Model ID: `{model_id}`",
            f"- Total episodes: {total}",
        ]
        if has_success:
            lines += [
                f"- Successes: {successes}",
                f"- Success rate: **{(success_rate * 100):.2f}%**",
            ]
        if has_avg_steps_all:
            lines.append(f"- Avg steps (all): {fmt_float(summary.get('avg_steps_all'))}")
        if has_avg_steps_succ:
            lines.append(f"- Avg steps (on success): {fmt_float(summary.get('avg_steps_on_success'))}")
        if has_avg_illegal:
            lines.append(f"- Avg illegal moves: {fmt_float(summary.get('avg_illegal_moves'))}")
        if has_avg_invalid:
            lines.append(f"- Avg invalid responses: {fmt_float(summary.get('avg_invalid_responses'))}")
        elif has_invalid_count:
            lines.append(f"- Invalid responses: {int(summary.get('invalid_responses') or 0)}")
        if has_avg_api_err:
            lines.append(f"- Avg API errors: {fmt_float(summary.get('avg_api_errors'))}")
        elif has_api_err_count:
            lines.append(f"- API errors: {int(summary.get('api_errors') or 0)}")
        if "skipped_episodes" in summary:
            lines.append(f"- Skipped episodes: {len(skipped)}")

        if diff_order:
            lines += ["", "## By Difficulty", ""]
            header = ["Difficulty", "N"]
            if has_success:
                header += ["Successes", "Success rate"]
            if has_avg_steps_all:
                header.append("Avg steps")
            if has_avg_illegal:
                header.append("Avg illegal")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for d in diff_order:
                b = by_diff[d]
                n = int(b["n"])
                row = [d, str(n)]
                if has_success:
                    row += [str(int(b["successes"])), fmt_pct(b["successes"], n)]
                if has_avg_steps_all:
                    row.append(fmt_float(b["steps"] / n if n else 0))
                if has_avg_illegal:
                    row.append(fmt_float(b["illegal"] / n if n else 0))
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return "\n".join(lines)

    def summary_path(self) -> Path:
        return resolve_repo_path(str(self.cfg.run.output_json))

    def config_path(self) -> Path:
        return resolve_repo_path(str(self.cfg.run.output_config))

    def load_existing_summary(self) -> Dict[str, Any]:
        summary_path = self.summary_path()
        if not summary_path.exists():
            return {}
        text = summary_path.read_text(encoding="utf-8")
        if not text.strip():
            # Torn write from an unclean shutdown left an empty/0-byte file.
            # Treat as "no summary state" — the per-episode episode_summary.json
            # files are the source of truth and resume recovers from those.
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Existing summary at {summary_path} must be a JSON object.")
        return payload

    def jsonl_artifact_paths(self) -> Dict[str, str]:
        return {"model_answer_jsonl": "model_answer.jsonl"}

    def load_jsonl_ids(self, *, filename: str) -> set[str]:
        """Read ids from `<output_dir>/filename`. Tolerates a torn trailing line
        from an aborted append (kernel buffer not flushed before crash) so
        resume after an unclean shutdown doesn't itself crash on JSONDecodeError.
        """
        path = self.output_dir / filename
        if not path.exists():
            return set()
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                # Only forgive a malformed *last* line — that's the torn-write
                # case. A malformed row in the middle is real corruption and
                # should still surface.
                if index == len(lines) - 1:
                    print(
                        f"[artifact-writer] discarded torn trailing line in {path.name} "
                        f"(line {index + 1}, {len(stripped)} bytes); resume will redo that row."
                    )
                    continue
                raise
            row_id = row.get("id")
            if row_id is not None:
                ids.add(str(row_id))
        return ids

    def scrub_api_error_rows(self, *, filename: str) -> int:
        path = self.output_dir / filename
        if not path.exists():
            return 0
        kept: List[str] = []
        removed = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if self._row_has_api_error(row):
                    removed += 1
                    continue
                kept.append(line)
        if removed:
            if kept:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text("".join(kept), encoding="utf-8")
                tmp.replace(path)
            else:
                path.unlink(missing_ok=True)
        return removed

    @staticmethod
    def _row_has_api_error(row: Dict[str, Any]) -> bool:
        if bool(row.get("api_error", False)):
            return True
        trajectory = row.get("trajectory")
        if isinstance(trajectory, list):
            for step in trajectory:
                if not isinstance(step, dict):
                    continue
                # planning rollout shape (single `api_error` flag) and video
                # rollout shape (separate `vlm_api_error` / `video_gen_api_error`
                # flags per cycle) both need to gate the scrub.
                if any(bool(step.get(flag, False)) for flag in (
                    "api_error",
                    "vlm_api_error",
                    "video_gen_api_error",
                )):
                    return True
        return False

    def append_jsonl_row(self, *, filename: str, row: Dict[str, Any]) -> Path:
        """Append `row` durably. fsync ensures the row survives a kernel crash;
        the resume gate (`load_jsonl_ids`) reads from this file, so a non-durable
        append could resurface an episode/sample that's already been billed.
        """
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def build_answer_row(self, *, result: EpisodeRunResult, meta_info: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = []
        total_steps = len(result.turns)
        for turn_index, turn in enumerate(result.turns):
            current_images: List[str] = []
            if self.write_step_images_enabled:
                current_images = [image.rel_path for image in turn.saved_images if image.label == "current"]
                if not current_images:
                    current_images = [image.rel_path for image in turn.saved_images]
            row = {
                "step_index": int(turn.step_index),
                "current_images": current_images,
                "state": "in progress" if turn_index < total_steps - 1 else ("success" if result.success else "failed"),
                "answer": turn.flattened_answer,
                "invalid_response": bool(turn.invalid_response),
                "api_error": bool(turn.api_error),
                "illegal": bool(turn.illegal),
                "raw_response_text": turn.raw_response_text,
            }
            if turn.api_error and turn.api_error_message:
                row["api_error_message"] = turn.api_error_message
            if self.write_step_images_enabled and turn.after_images:
                row["post_action_images"] = [image.rel_path for image in turn.after_images]
            # Surface interleaved-only fields when the model client wrote them.
            # Absent for raw planning baselines (oracle/random/greedy/non-interleaved VLMs).
            debug = turn.response_debug or {}
            for key in (
                "image_prompt",
                "image_gen_skip_reason",
                "image_gen_mode",
                "phase1_text",
                "phase1_reasoning_content",
                "reasoning_content",
                "phase1_attempts",
                "phase2_attempts",
                "phase_retry_log",
                "video_prompt",
                "video_gen_skip_reason",
                "video_gen_mode",
                "video_gen_api_error",
                "video_gen_api_error_message",
                "frame_extract_error",
            ):
                if key in debug:
                    row[key] = debug[key]
            imagined_relpath = debug.get("imagined_image_rel_path")
            if imagined_relpath:
                row["imagined_images"] = [imagined_relpath]
            # Surface the generated clip so `video_metrics evaluate --run-dir`
            # can consume this run directly (io.py reads trajectory[].imagined_video).
            imagined_video = debug.get("imagined_video_rel_path") or debug.get("video_gen_rel_path")
            if imagined_video:
                row["imagined_video"] = imagined_video
            frame_rel_paths = debug.get("video_frame_rel_paths")
            if frame_rel_paths:
                row["imagined_frames"] = list(frame_rel_paths)
            trajectory.append(row)
        return {
            "id": result.spec.episode_id,
            "category": list(result.spec.category),
            "type": "episode_rollout",
            "meta_info": meta_info,
            "trajectory": trajectory,
        }
