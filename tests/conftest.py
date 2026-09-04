"""Shared test setup.

The evaluation entry points inject the shared-package and per-chain source
directories onto ``sys.path`` at runtime (they are not pip-installed). The tests
reproduce the same bootstrap so imports resolve the way the runners do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Repo root first so `video_eval_tasks.video_metrics` (a real package) imports
# regardless of how pytest was invoked; then the path-injected runner dirs.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _sub in ("topobench_eval", "planning_eval_tasks"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


STATIC_TASKS = [
    "continuity_2d_maze",
    "continuity_3d_maze",
    "enclosure_sheep",
    "enclosure_hole_detection",
    "knots_static",
    "order_bead_string",
    "order_origami",
    "separation_objects",
]

PLANNING_TASKS = [
    "continuity_pipe",
    "enclosure_chat_noir",
    "knots_untangle",
    "order_swap_2d_puzzle",
    "separation_one_stroke",
]

INTERLEAVED_TASKS = [
    "continuity_2d_maze",
    "continuity_pipe",
    "enclosure_sheep",
    "knots_untangle",
    "separation_one_stroke",
]
