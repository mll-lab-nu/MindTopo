"""Ground-truth / topology oracle for the ``continuity_pipe`` ("pipe") task.

Same philosophy as :mod:`oracle_bridge` and :mod:`swap_oracle` — we never re-derive
the puzzle logic, we reuse the env's own pure helpers.  ``environments/continuity_pipe/gym/env.py``
imports ``playwright`` at module top (only needed for the live browser env), so we
stub it before importing.  The functions we use (``rotate_mask``, ``connected_indices``,
``current_masks`` ...) are self-contained and deterministic, so a stubbed import is
safe and gives GT that is bit-identical to what generated the episodes.

Pipe bit convention (from the env): N=1, E=2, S=4, W=8.  A cell's ``current_mask`` is
its solved mask rotated clockwise by ``rotation`` 90-degree ticks.  The board is solved
when every active (non-empty) pipe cell is graph-connected to the source cell.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import config

PIPE_ENV_GYM = config.REPO_ROOT / "environments" / "continuity_pipe" / "gym"
PIPE_QUESTION_JSONL = config.REPO_ROOT / "environments" / "continuity_pipe" / "output" / "question.jsonl"

# --- import the env's pure helpers without pulling in playwright ------------------
# Every env in this repo names its module ``env``; load it under a unique module name so
# importing several oracles in one process does not collide on ``sys.modules["env"]``.
if "playwright" not in sys.modules:  # pragma: no cover - trivial shim
    _pw = types.ModuleType("playwright")
    _pw_async = types.ModuleType("playwright.async_api")
    _pw_async.async_playwright = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["playwright"] = _pw
    sys.modules["playwright.async_api"] = _pw_async


def _load_env_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipe_env = _load_env_module("continuity_pipe_env", PIPE_ENV_GYM / "env.py")

# Direction bits, matching the env DIRS order (N, E, S, W).  ``d(row/col)`` are in image
# axes (x = column grows right, y = row grows down); N points up (smaller y).
DIR_BITS: Tuple[Tuple[str, int, int, int], ...] = (
    ("N", 0, -1, 1),
    ("E", 1, 0, 2),
    ("S", 0, 1, 4),
    ("W", -1, 0, 8),
)
BIT_OF_DIR: Dict[str, int] = {name: bit for name, _dx, _dy, bit in DIR_BITS}

__all__ = [
    "DIR_BITS",
    "BIT_OF_DIR",
    "PipeGT",
    "load_pipe_gt",
    "pipe_shape",
    "rotate_mask",
    "current_masks",
    "connected_indices",
    "is_solved",
    "grid_size_for_difficulty",
]


def rotate_mask(mask: int, ticks: int) -> int:
    return int(pipe_env.rotate_mask(int(mask), int(ticks)))


def current_masks(solved_masks: Sequence[int], rotations: Sequence[int]) -> List[int]:
    # Reimplemented locally (not via pipe_env.current_masks) so it runs on Python 3.9,
    # where the env's ``zip(..., strict=True)`` is unavailable.  Bit-identical result.
    return [rotate_mask(int(m), int(rot)) for m, rot in zip(solved_masks, rotations)]


def connected_indices(grid_size: int, source_index: int, masks: Sequence[int]) -> List[int]:
    return list(
        pipe_env.connected_indices(
            grid_size=int(grid_size), source_index=int(source_index), masks=list(masks)
        )
    )


def grid_size_for_difficulty(difficulty: str) -> int:
    """easy -> 4x4, medium/hard -> 5x5 (env DIFFICULTY_SETUPS)."""
    return int(pipe_env.grid_size_for_difficulty(difficulty))


def pipe_shape(mask: int) -> str:
    """Rotation-invariant pipe type from a cell mask.

    endpoint (1 arm), straight (2 opposite arms), elbow (2 adjacent arms),
    tee (3 arms), cross (4 arms, illegal in this env), empty (0 arms).
    """
    m = int(mask) & 15
    bits = bin(m).count("1")
    if bits == 0:
        return "empty"
    if bits == 1:
        return "endpoint"
    if bits == 2:
        return "straight" if m in (5, 10) else "elbow"  # 5 = N|S, 10 = E|W
    if bits == 3:
        return "tee"
    return "cross"


def arms_of_mask(mask: int) -> List[str]:
    m = int(mask) & 15
    return [name for name, _dx, _dy, bit in DIR_BITS if m & bit]


@dataclass(frozen=True)
class PipeGT:
    """Deterministic ground truth for one pipe episode."""

    question_id: str
    level: str
    difficulty: str
    seed: int
    grid_size: int
    source_index: int
    solved_masks: Tuple[int, ...]
    initial_rotations: Tuple[int, ...]

    @property
    def total_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def active_indices(self) -> Set[int]:
        """Indices of non-empty pipe cells (invariant: pipes never appear/disappear)."""
        return {i for i, m in enumerate(self.solved_masks) if int(m) & 15}

    @property
    def shape_by_index(self) -> Dict[int, str]:
        """Rotation-invariant pipe type per active cell (invariant across the video)."""
        return {i: pipe_shape(self.solved_masks[i]) for i in self.active_indices}

    def xy(self, index: int) -> Tuple[int, int]:
        return int(index) % self.grid_size, int(index) // self.grid_size

    def index(self, gx: int, gy: int) -> int:
        return int(gy) * self.grid_size + int(gx)


def is_solved(grid_size: int, source_index: int, masks: Sequence[int], active_indices: Sequence[int]) -> bool:
    """Win predicate: every active pipe cell is connected to the source."""
    connected = set(connected_indices(grid_size, source_index, masks))
    active = set(int(i) for i in active_indices)
    return active.issubset(connected)


_GT_INDEX: Optional[Dict[str, Dict[str, object]]] = None


def _load_index() -> Dict[str, Dict[str, object]]:
    global _GT_INDEX
    if _GT_INDEX is None:
        index: Dict[str, Dict[str, object]] = {}
        with PIPE_QUESTION_JSONL.open() as handle:
            for line in handle:
                row = json.loads(line)
                index[str(row["id"])] = row
        _GT_INDEX = index
    return _GT_INDEX


def question_id(level: str, difficulty: str, seed: int) -> str:
    """Reconstruct the env question id, e.g. continuity_pipe_grid_4_solution_steps_easy_seed_1."""
    grid_dim = int(str(level).split("x")[0].replace("grid_", ""))
    return f"continuity_pipe_grid_{grid_dim}_solution_steps_{difficulty}_seed_{int(seed)}"


def load_pipe_gt(
    level: str,
    difficulty: str,
    seed: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> PipeGT:
    """Look up the exact GT board for (level, difficulty, seed) from the env question.jsonl."""
    qid = question_id(level, difficulty, seed)
    meta = dict(metadata or {})
    init = dict(meta.get("initial_state") or {})
    if not init:
        row = _load_index().get(qid)
        if row is None:
            raise KeyError(f"no continuity_pipe GT for {qid}")
        meta = dict(row["meta_info"])
        init = dict(meta["initial_state"])
    reset = dict(init["reset_config"])
    grid_size = int(reset.get("gridSize") or init.get("gridSize"))
    solved = [int(v) for v in reset["solvedMasks"]]
    rotations = [int(v) % 4 for v in reset["rotations"]]
    return PipeGT(
        question_id=qid,
        level=str(level),
        difficulty=str(meta.get("difficulty") or difficulty),
        seed=int(seed),
        grid_size=grid_size,
        source_index=int(reset.get("sourceIndex") or init.get("sourceIndex") or 0),
        solved_masks=tuple(solved),
        initial_rotations=tuple(rotations),
    )
