"""The minimal video path generates one clip and reuses it per episode."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from conftest import ROOT
from unified_types import Message, ModelResponse, SavedImage

VIDEO_DIR = ROOT / "video_eval_tasks"
if str(VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_DIR))

import video_interleaved_client as vic  # noqa: E402
from video_gen_client import VideoGenResult  # noqa: E402


class _TextClient:
    effective_system_prompt = "video system"
    concurrency_hint = 1

    def __init__(self) -> None:
        self.calls = 0

    def reset_episode(self) -> None:
        pass

    def describe(self) -> Dict[str, Any]:
        return {"model_id": "text-stub"}

    def generate(
        self, *, messages: Sequence[Message], images: Sequence[SavedImage]
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            text = "<video_prompt>animate the complete solution</video_prompt>"
        else:
            text = '{"answer":"U"}'
        return ModelResponse(text, False, None, {})


class _VideoClient:
    upstream_model_name = "video-stub"
    base_url = "https://example.invalid"
    mode = "i2v"

    def __init__(self) -> None:
        self.calls = 0

    def generate_to_file(self, *, prompt: str, save_path: Path, reference_image_path: Path):
        self.calls += 1
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"video")
        return VideoGenResult(save_path, {"stub": True})


def test_cached_video_client_generates_once_per_episode(tmp_path: Path, monkeypatch):
    current = tmp_path / "images" / "episode" / "step_0000_current.png"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"png")
    saved = SavedImage("current", "current", current, "images/episode/step_0000_current.png")
    frame = tmp_path / "images" / "episode" / "step_0000_imagined_frames" / "frame_00.png"

    def fake_extract(video_path: Path, *, out_dir: Path, num_frames: int):
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
        return [frame]

    monkeypatch.setattr(vic, "extract_frames", fake_extract)
    text_client = _TextClient()
    video_client = _VideoClient()
    client = vic.CachedVideoModelClient(
        model_id="cached-video-stub",
        text_client=text_client,
        video_gen_client=video_client,
        output_dir=tmp_path,
        num_extracted_frames=1,
    )
    messages = [{"role": "user", "content": [{"type": "image_ref", "ref_id": "current"}]}]

    first = client.generate(messages=messages, images=[saved])
    second = client.generate(messages=messages, images=[saved])

    assert not first.api_error and not second.api_error
    assert video_client.calls == 1
    assert text_client.calls == 3
    assert second.response_debug["video_gen_skip_reason"] == "cached_from_first_cycle"
    assert first.response_debug["video_frame_rel_paths"] == second.response_debug["video_frame_rel_paths"]


def test_video_modules_compile_and_configs_are_minimal():
    for name in (
        "agent_runner.py",
        "frame_extractor.py",
        "video_gen_client.py",
        "video_gen_client_ark.py",
        "video_interleaved_client.py",
        "configs/system_prompts.py",
    ):
        py_compile.compile(str(VIDEO_DIR / name), doraise=True)

    envs = {path.stem for path in (VIDEO_DIR / "configs" / "env").glob("*.yaml")}
    assert envs == {"continuity_pipe", "knots_untangle", "separation_one_stroke"}
    model = yaml.safe_load(
        (
            VIDEO_DIR
            / "configs"
            / "model"
            / "seedance_2_0_mini_i2v_ark_vlm_e2e_gpt56luna.yaml"
        ).read_text()
    )
    assert model["kind"] == "vlm_video"
    assert model["video_gen"]["backend"] == "ark"
