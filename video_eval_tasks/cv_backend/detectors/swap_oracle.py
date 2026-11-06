"""Ground-truth / legality oracle for the order_swap_2d_puzzle ("2D swap") task.

Same philosophy as :mod:`oracle_bridge` — we never re-derive the puzzle logic, we
reuse the env's own pure helpers.  ``environments/order_swap_2d_puzzle/gym/env.py``
imports ``playwright`` at module top (only needed for the live browser env), so we
stub it before importing.  The functions we use
(``sample_swap_2d_puzzle_instance``, ``shortest_action_sequence``,
``BLOCK_LIBRARY`` ...) are self-contained and deterministic, so a stubbed import is
safe and gives us GT that is bit-identical to what generated the episodes.
"""
from __future__ import annotations

import sys
import types
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config

SWAP_ENV_GYM = config.REPO_ROOT / "environments" / "order_swap_2d_puzzle" / "gym"

# --- import the env's pure helpers without pulling in playwright ------------------
if "playwright" not in sys.modules:  # pragma: no cover - trivial shim
    _pw = types.ModuleType("playwright")
    _pw_async = types.ModuleType("playwright.async_api")
    _pw_async.async_playwright = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["playwright"] = _pw
    sys.modules["playwright.async_api"] = _pw_async

def _load_env_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load swap environment module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# All gym environments use the filename ``env.py``. A unique module name keeps
# this oracle correct when pytest or a benchmark process has already imported
# a different task's ``env`` module.
swap_env = _load_env_module("order_swap_2d_puzzle_env", SWAP_ENV_GYM / "env.py")

BLANK_TOKEN: str = swap_env.BLANK_TOKEN
BLOCK_LIBRARY: Tuple[Dict[str, str], ...] = swap_env.BLOCK_LIBRARY
BLOCK_BY_ID: Dict[str, Dict[str, str]] = swap_env.BLOCK_BY_ID

__all__ = [
    "BLANK_TOKEN",
    "BLOCK_LIBRARY",
    "BLOCK_BY_ID",
    "SwapGT",
    "load_swap_gt",
    "is_legal_config",
    "is_single_swap",
    "block_multiset",
    "swap_distance",
]


@dataclass(frozen=True)
class SwapGT:
    """Deterministic ground truth for one episode (row-major arrangements)."""

    grid_rows: int
    grid_cols: int
    seed: int
    selected_block_ids: Tuple[str, ...]
    initial_arrangement: Tuple[str, ...]
    goal_arrangement: Tuple[str, ...]
    theoretical_min_steps: int

    @property
    def num_cells(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def difficulty(self) -> str:
        return swap_env.difficulty_label_for_grid(self.grid_rows, self.grid_cols)

    def grid(self, arrangement: Sequence[str]) -> List[List[str]]:
        return swap_env.arrangement_to_grid(list(arrangement), self.grid_rows, self.grid_cols)


def load_swap_gt(
    grid_rows: int,
    grid_cols: int,
    seed: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> SwapGT:
    """Reproduce the episode instance from (rows, cols, seed).

    The env samples deterministically from ``StableRng(seed)``, so this returns the
    exact initial/goal arrangements and block set used to render the episode.
    """
    state = dict((metadata or {}).get("initial_state") or {})
    reset = dict(state.get("reset_config") or {})
    if reset:
        initial = reset.get("initialArrangement")
        goal = reset.get("goalArrangement")
        selected = reset.get("selectedBlockIds")
        minimum = reset.get("theoreticalMinSteps")
        if initial and goal and selected and minimum is not None:
            return SwapGT(
                grid_rows=int(reset.get("gridRows") or grid_rows),
                grid_cols=int(reset.get("gridCols") or grid_cols),
                seed=int(reset.get("seed") if reset.get("seed") is not None else seed),
                selected_block_ids=tuple(str(value) for value in selected),
                initial_arrangement=tuple(str(value) for value in initial),
                goal_arrangement=tuple(str(value) for value in goal),
                theoretical_min_steps=int(minimum),
            )

    inst = swap_env.sample_swap_2d_puzzle_instance(grid_rows, grid_cols, seed=seed)
    return SwapGT(
        grid_rows=int(inst["grid_rows"]),
        grid_cols=int(inst["grid_cols"]),
        seed=int(seed),
        selected_block_ids=tuple(inst["selected_block_ids"]),
        initial_arrangement=tuple(inst["initial_arrangement"]),
        goal_arrangement=tuple(inst["goal_arrangement"]),
        theoretical_min_steps=int(inst["theoretical_min_steps"]),
    )


def block_multiset(arrangement: Sequence[Optional[str]]) -> Dict[str, int]:
    """Token -> count, with ``None`` (unreadable cell) mapped to the sentinel ``"?"``."""
    counts: Dict[str, int] = {}
    for tok in arrangement:
        key = "?" if tok is None else str(tok)
        counts[key] = counts.get(key, 0) + 1
    return counts


def swap_distance(
    initial_arrangement: Sequence[Optional[str]],
    goal_arrangement: Sequence[Optional[str]],
) -> Optional[int]:
    """Exact minimum number of legal blank swaps between two readable states."""
    if any(token is None for token in initial_arrangement) or any(token is None for token in goal_arrangement):
        return None
    try:
        return int(
            swap_env.shortest_swap_distance(
                [str(token) for token in initial_arrangement],
                [str(token) for token in goal_arrangement],
            )
        )
    except (ValueError, RuntimeError):
        return None


def is_legal_config(
    arrangement: Sequence[Optional[str]],
    selected_block_ids: Sequence[str],
    grid_rows: int,
    grid_cols: int,
) -> Tuple[bool, str]:
    """A legal board == the env's ``normalize_arrangement`` accepts it.

    That means: correct length, exactly one blank, and the non-blank tokens are the
    expected block set with no duplicates/missing/extras.  Returns (ok, reason).
    """
    if any(tok is None for tok in arrangement):
        n_missing = sum(1 for tok in arrangement if tok is None)
        return False, f"{n_missing} cell(s) unreadable"
    try:
        swap_env.normalize_arrangement(
            list(arrangement),
            selected_block_ids=list(selected_block_ids),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
    except ValueError as exc:
        return False, str(exc)
    return True, "legal permutation with exactly one blank"


def is_single_swap(
    prev: Sequence[Optional[str]],
    cur: Sequence[Optional[str]],
) -> Tuple[bool, str, Optional[Tuple[int, int]]]:
    """True iff ``cur`` is ``prev`` unchanged, or exactly one legal blank<->block swap.

    The 2D swap puzzle lets the blank swap with *any* tile (not just neighbours), so
    there is no adjacency constraint: a legal move changes exactly two cells, one of
    which held the blank and now holds the block that used to sit in the other cell,
    and vice-versa.  Returns (ok, reason, swapped_cell_pair_or_None).
    """
    if len(prev) != len(cur):
        return False, "cell count changed", None
    if any(p is None for p in prev) or any(c is None for c in cur):
        return False, "unreadable cell(s) — swap not verifiable", None

    diff = [i for i, (p, c) in enumerate(zip(prev, cur)) if p != c]
    if not diff:
        return True, "no change (idle frame)", None
    if len(diff) != 2:
        return False, f"{len(diff)} cells changed (a single swap changes exactly 2)", None

    i, j = diff
    blank = BLANK_TOKEN
    prev_has_blank = blank in (prev[i], prev[j])
    # A transposition: prev[i]==cur[j] and prev[j]==cur[i].
    is_transposition = prev[i] == cur[j] and prev[j] == cur[i]
    if not is_transposition:
        return False, "the two changed cells are not a clean swap of contents", (i, j)
    if not prev_has_blank:
        return False, "swap did not involve the blank cell", (i, j)
    return True, "exactly one legal blank<->block swap", (i, j)
