"""Focused coverage for the minimal GPT-5.6 + GPT-Image-2 port."""

from __future__ import annotations

from io import BytesIO
import json
import sys

import yaml
from PIL import Image

from conftest import ROOT


def _png(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_gpt_5_6_luna_provider_and_configs_are_wired():
    providers = yaml.safe_load((ROOT / "topobench_eval" / "providers.yaml").read_text())
    assert providers["models"]["gpt_5_6_luna"]["upstream_model_name"] == "gpt-5.6-luna"

    planning = yaml.safe_load(
        (ROOT / "planning_eval_tasks" / "configs" / "model" / "gpt_5_6_luna.yaml").read_text()
    )
    imagined = yaml.safe_load(
        (
            ROOT
            / "interleaved_eval_tasks"
            / "configs"
            / "model"
            / "gpt_5_6_luna_imagined.yaml"
        ).read_text()
    )
    assert planning["id"] == "gpt_5_6_luna"
    assert imagined["text"]["text_model_id"] == "gpt_5_6_luna"
    assert imagined["image_gen"]["size"] == "match_input"
    assert imagined["image_gen"]["max_image_gen_calls_fraction"] == 0.5


def test_match_input_preserves_supported_reference_size():
    sys.path.insert(0, str(ROOT / "interleaved_eval_tasks"))
    from image_gen_client import OpenAICompatibleImageGenClient

    assert OpenAICompatibleImageGenClient._match_reference_size(_png(1440, 1080)) == "1440x1088"


def test_match_input_scales_small_reference_into_supported_pixel_range():
    sys.path.insert(0, str(ROOT / "interleaved_eval_tasks"))
    from image_gen_client import OpenAICompatibleImageGenClient

    width, height = map(
        int,
        OpenAICompatibleImageGenClient._match_reference_size(_png(320, 240)).split("x"),
    )
    assert width % 16 == height % 16 == 0
    assert width * height >= OpenAICompatibleImageGenClient._MIN_PIXELS


def test_balanced_105_slices_are_complete_and_resolve_to_source_rows():
    for env_name in ("continuity_pipe", "knots_untangle", "separation_one_stroke"):
        selected = json.loads(
            (ROOT / "data" / "slices" / "balanced_105" / f"{env_name}.json").read_text()
        )
        source_rows = [
            json.loads(line)
            for line in (
                ROOT / "environments" / env_name / "output" / "question.jsonl"
            ).read_text().splitlines()
        ]
        source_by_id = {row["id"]: row for row in source_rows}
        assert len(selected) == len(set(selected)) == 105
        assert set(selected) <= set(source_by_id)
        difficulties = [source_by_id[episode_id]["meta_info"]["difficulty"] for episode_id in selected]
        assert {name: difficulties.count(name) for name in set(difficulties)} == {
            "easy": 35,
            "medium": 35,
            "hard": 35,
        }
