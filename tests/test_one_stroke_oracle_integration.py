"""Regression tests for the unified One-Stroke oracle adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNING_DIR = REPO_ROOT / "planning_eval_tasks"
if str(PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DIR))

from env_adapters import build_adapter  # noqa: E402
from unified_types import EpisodeSpec, TurnContext, TurnInput  # noqa: E402


def test_episode_spec_carries_dataset_reference_solution(tmp_path):
    adapter = build_adapter("separation_one_stroke")
    setup = SimpleNamespace(
        board_size=6,
        level_number=1,
        difficulty="hard",
        solution_length=3,
        setup_id="hard-1",
        question_max_steps=4,
        question_row_id="episode-1",
        question_seed=401,
        question_repeat_index=0,
        reset_config={"levelJson": {"solution": ["U", "R", "D"]}},
        level_json={"solution": ["U", "R", "D"]},
    )

    class _StubBench:
        @staticmethod
        def load_benchmark_setups_from_question_jsonl(_path):
            return [setup]

    adapter._benchmark = _StubBench()
    adapter.source_question_jsonl_path = lambda _cfg: tmp_path / "unused.jsonl"
    cfg = OmegaConf.create(
        {
            "env": {},
            "run": {
                "output_dir": str(tmp_path),
                "include_legal_moves_in_prompt": True,
            },
        }
    )

    specs = adapter.enumerate_episodes(cfg)

    assert specs[0].metadata["reference_solution"] == ["U", "R", "D"]


def test_oracle_receives_reference_solution_and_legal_directions():
    adapter = build_adapter("separation_one_stroke")
    seen = {}

    class _StubBench:
        @staticmethod
        def build_oracle_direction(info, *, reference_solution=None):
            seen["info"] = info
            seen["reference_solution"] = reference_solution
            return "R"

    adapter._benchmark = _StubBench()
    spec = EpisodeSpec(
        episode_id="episode-1",
        level=1,
        seed=401,
        repeat_index=0,
        category=[],
        metadata={"reference_solution": ["R", "U"]},
    )
    context = TurnContext(
        obs_rgb=None,
        planner_state={"stroke": [{"x": 0, "y": 0}]},
        symbolic_observation=None,
        legal_actions=[{"direction": "R"}, {"direction": "U"}],
        step_index=0,
        step_budget=20,
        termination_state=None,
        image_views=[],
        prompt_metadata={},
    )

    result = adapter.choose_local_action(
        "oracle", TurnInput(spec=spec, context=context, history=[], saved_images=[])
    )

    assert seen["info"]["legalDirections"] == ["R", "U"]
    assert seen["reference_solution"] == ["R", "U"]
    assert result.parsed_action == "R"
