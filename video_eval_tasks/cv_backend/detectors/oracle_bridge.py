"""Reuse the env's oracle_solver (without reimplementing crossing computation), and add a variant that returns crossing_pairs.

The existing oracle API only returns the crossing **count**; the spike needs the
indexed crossing pairs for comparison, so this module reproduces its geometric
checks (`_segments_intersect` / `_segments_close`) and returns the list of pairs.
The logic is identical to the oracle; it merely also collects the indices.
"""
from __future__ import annotations

import sys
from typing import List, Optional, Set, Tuple

from . import config

# Add the env backend to the path so we can import the existing oracle directly
if str(config.ENV_BACKEND) not in sys.path:
    sys.path.insert(0, str(config.ENV_BACKEND))

from oracle_solver import (  # noqa: E402  (can only import after the path is injected)
    SolverState,
    hole_xy_from_id,
    state_crossings,
    state_logical_crossings,
    symbolic_to_state,
    _segments_close,
    VISUAL_OVERLAP_THRESHOLD,
)

__all__ = [
    "SolverState",
    "state_crossings",
    "state_logical_crossings",
    "symbolic_to_state",
    "visual_crossing_pairs",
]


def visual_crossing_pairs(state: SolverState, grid_size: int) -> Set[Tuple[int, int]]:
    """Crossing pairs (rope_i, rope_j) with thickness-aware contact, i<j. Same convention as `state_crossings`."""
    pairs: Set[Tuple[int, int]] = set()
    for i in range(len(state)):
        a1 = hole_xy_from_id(state[i][0], grid_size)
        a2 = hole_xy_from_id(state[i][1], grid_size)
        for j in range(i + 1, len(state)):
            b1 = hole_xy_from_id(state[j][0], grid_size)
            b2 = hole_xy_from_id(state[j][1], grid_size)
            if _segments_close(a1, a2, b1, b2, VISUAL_OVERLAP_THRESHOLD):
                pairs.add((i, j))
    return pairs
