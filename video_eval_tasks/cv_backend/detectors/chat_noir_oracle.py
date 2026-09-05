"""Ground-truth / connectivity oracle for the ``enclosure_chat_noir`` ("chat noir") task.

Same philosophy as :mod:`swap_oracle` — reuse the env's own pure helpers instead of
re-deriving the hex board / escape logic.  ``environments/enclosure_chat_noir/gym/env.py``
imports ``playwright`` at module top, so we stub it before importing.  ``build_hex_board``,
``shortest_boundary_path`` and the neighbour adjacency are self-contained and deterministic.

Win predicate (cat trapped): from the cat cell there is no path over open (non-blocked)
cells to any boundary cell, i.e. ``shortest_boundary_path`` returns ``None``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

from . import config

CHAT_NOIR_ENV_GYM = config.REPO_ROOT / "environments" / "enclosure_chat_noir" / "gym"
CHAT_NOIR_QUESTION_JSONL = (
    config.REPO_ROOT / "environments" / "enclosure_chat_noir" / "output" / "question.jsonl"
)

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


chat_noir_env = _load_env_module("enclosure_chat_noir_env", CHAT_NOIR_ENV_GYM / "env.py")

ALLOWED_BOARD_RADII: Tuple[int, ...] = tuple(chat_noir_env.ALLOWED_BOARD_RADII)

__all__ = [
    "ALLOWED_BOARD_RADII",
    "ChatNoirGT",
    "build_hex_board",
    "board_cell_count",
    "load_chat_noir_gt",
    "is_trapped",
    "shortest_boundary_path",
    "choose_cat_move",
    "are_adjacent",
    "neighbor_indices",
]


def build_hex_board(radius: int) -> Dict[str, object]:
    return chat_noir_env.build_hex_board(int(radius))


def board_cell_count(radius: int) -> int:
    return len(build_hex_board(radius)["cells"])


def neighbor_indices(board: Dict[str, object], index: int) -> List[int]:
    return list(board["cells"][int(index)]["neighbor_indices"])


def are_adjacent(board: Dict[str, object], a: int, b: int) -> bool:
    return int(b) in board["cells"][int(a)]["neighbor_indices"]


def shortest_boundary_path(board: Dict[str, object], cat_index: int, blocked: Collection[int]):
    return chat_noir_env.shortest_boundary_path(board, int(cat_index), set(int(i) for i in blocked))


def choose_cat_move(
    board: Dict[str, object], cat_index: int, blocked: Collection[int], policy: str
):
    """Return the env policy decision for deterministic (medium/hard/static) policies."""
    return chat_noir_env.choose_cat_move(
        board,
        int(cat_index),
        set(int(i) for i in blocked),
        str(policy),
    )


def is_trapped(board: Dict[str, object], cat_index: int, blocked: Collection[int]) -> bool:
    """Cat is trapped iff no open path from the cat cell reaches the boundary."""
    return shortest_boundary_path(board, cat_index, blocked) is None


@dataclass(frozen=True)
class ChatNoirGT:
    """Deterministic ground truth for one chat noir episode."""

    question_id: str
    radius: int
    policy: str
    seed: int
    cat_index: int
    initial_block_count: int
    initial_blocked_indices: Tuple[int, ...]

    @property
    def board(self) -> Dict[str, object]:
        return build_hex_board(self.radius)

    @property
    def cell_count(self) -> int:
        return board_cell_count(self.radius)

    @property
    def boundary_indices(self) -> Set[int]:
        return set(self.board["boundary_indices"])


_GT_INDEX: Optional[Dict[str, Dict[str, object]]] = None


def _load_index() -> Dict[str, Dict[str, object]]:
    global _GT_INDEX
    if _GT_INDEX is None:
        index: Dict[str, Dict[str, object]] = {}
        with CHAT_NOIR_QUESTION_JSONL.open() as handle:
            for line in handle:
                row = json.loads(line)
                index[str(row["id"])] = row
        _GT_INDEX = index
    return _GT_INDEX


def question_id(radius: int, blocked: int, policy: str, seed: int) -> str:
    return f"enclosure_chat_noir_radius_{int(radius)}_blocked_{int(blocked)}_policy_{policy}_seed_{int(seed)}"


def load_chat_noir_gt(
    radius: int,
    blocked: int,
    policy: str,
    seed: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> ChatNoirGT:
    qid = question_id(radius, blocked, policy, seed)
    meta = dict(metadata or {})
    init = dict(meta.get("initial_state") or {})
    if not init:
        row = _load_index().get(qid)
        if row is None:
            raise KeyError(f"no enclosure_chat_noir GT for {qid}")
        meta = dict(row["meta_info"])
        init = dict(meta["initial_state"])
    blocked_indices = tuple(int(v) for v in init["blockedIndices"])
    return ChatNoirGT(
        question_id=qid,
        radius=int(meta["board_radius"]),
        policy=str(meta.get("cat_policy_name") or policy),
        seed=int(seed),
        cat_index=int(init["catIndex"]),
        initial_block_count=int(meta.get("initial_block_count") or len(blocked_indices)),
        initial_blocked_indices=blocked_indices,
    )
