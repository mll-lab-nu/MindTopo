from pathlib import Path

import pytest

from video_eval_tasks.video_metrics.adapters import adapter_for
from video_eval_tasks.video_metrics.types import EpisodeSpec


class _Legacy:
    def __init__(self):
        self.called = None

    def evaluate_episode(self, *args):
        self.called = args
        return {"task": args[0]}


CASES = [
    (
        "knots_untangle",
        "_process_untangle_video",
        {"initial_state": {"gridSize": 5, "ropes": [{"color": 123}]}},
    ),
    (
        "separation_one_stroke",
        "_process_one_stroke_video",
        {"board_size": 4, "initial_state": {}},
    ),
    (
        "order_swap_2d_puzzle",
        "_process_swap2d_video",
        {"initial_state": {"initialGrid": [["_", "A"], ["B", "C"]]}},
    ),
    (
        "continuity_pipe",
        "_process_pipe_video",
        {"level": "grid_4x4", "initial_state": {"gridSize": 4}},
    ),
    (
        "enclosure_chat_noir",
        "_process_chat_noir_video",
        {"board_radius": 3, "initial_block_count": 10, "cat_policy_name": "easy", "initial_state": {}},
    ),
]


@pytest.mark.parametrize("task,method,metadata", CASES)
def test_all_five_task_adapters_use_the_common_contract(task, method, metadata, tmp_path: Path) -> None:
    spec = EpisodeSpec(
        episode_id="episode",
        source_episode_id="episode",
        task_name=task,
        video_path=tmp_path / "video.mp4",
        video_rel_path="video.mp4",
        difficulty="easy",
        seed=1,
        metadata=metadata,
        model_id="internvl",
        vlm_text_model_id="internvl",
        video_gen_model_name="wan",
    )
    legacy = _Legacy()
    adapter_for(task).evaluate(legacy, spec, tmp_path / "artifacts", "motion", True)
    assert legacy.called[0] == task
    assert legacy.called[1] == spec
