"""Ground-truth loading and episode iteration.

GT source: environments/knots_untangle/output/question.jsonl (the authoritative
structured initial state).

Note: the run being evaluated used the now-deleted question_interleaved.jsonl,
whose seed numbering does not match question.jsonl (easy still lines up by seed,
medium/hard do not). Therefore:
  - easy: look up question.jsonl directly by (level, seed) (states have been
    verified to match exactly).
  - medium/hard: take the state parsed from the clean render and recover the
    authoritative GT by **state matching** against the same level/gridSize
    subset of question.jsonl (a successful match also validates the parse).
    State matching happens in the orchestrator (it needs the parser); this
    module only provides the lookup primitives.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from . import config
from .oracle_bridge import SolverState, symbolic_to_state

# Undirected endpoint pair of a single rope (the two hole_ids sorted); the full
# board state = the frozenset of these pairs
RopeKey = Tuple[int, int]
StateKey = FrozenSet[RopeKey]


def _state_key(state: SolverState) -> StateKey:
    return frozenset(tuple(sorted(pair)) for pair in state)


@dataclass
class QuestionEntry:
    level: str
    seed: int
    grid_size: int
    state: SolverState
    visual_crossings: Optional[int]
    logical_crossings: Optional[int]
    raw_initial_state: dict


class QuestionDB:
    """Index over question.jsonl: by (level, seed) and by (level, state_key)."""

    def __init__(self, path: Path = config.QUESTION_JSONL):
        self.by_seed: Dict[Tuple[str, int], QuestionEntry] = {}
        self.by_state: Dict[Tuple[str, StateKey], QuestionEntry] = {}
        self._grid_sizes: Dict[str, set] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mi = row["meta_info"]
            st = mi["initial_state"]
            gs = int(st["gridSize"])
            state = symbolic_to_state(st, gs)
            if state is None:
                continue
            entry = QuestionEntry(
                level=mi["level"],
                seed=mi["seed"],
                grid_size=gs,
                state=state,
                visual_crossings=st.get("visualCrossings"),
                logical_crossings=st.get("logicalCrossings"),
                raw_initial_state=st,
            )
            self.by_seed[(entry.level, entry.seed)] = entry
            self.by_state[(entry.level, _state_key(state))] = entry
            self._grid_sizes.setdefault(entry.level, set()).add(gs)

    def grid_size_for_level(self, level: str) -> Optional[int]:
        sizes = self._grid_sizes.get(level)
        return next(iter(sizes)) if sizes and len(sizes) == 1 else None

    def lookup_seed(self, level: str, seed: int) -> Optional[QuestionEntry]:
        return self.by_seed.get((level, seed))

    def match_state(self, level: str, state: SolverState) -> Optional[QuestionEntry]:
        """Look up the authoritative entry within the same level by state
        (independent of the seed numbering)."""
        return self.by_state.get((level, _state_key(state)))


# ── untangled.jsonl: the authoritative GT for this batch of videos ────────────────────────────────────
_ROPE_RE = re.compile(r"(\w+) rope from peg \((\d+),(\d+)\) to peg \((\d+),(\d+)\)")
_GRID_RE = re.compile(r"(\d+)x(\d+) peg grid")
_LEVEL_GS = {"easy": 5, "medium": 6, "hard": 6}


@dataclass
class UntangledEntry:
    level: str
    seed: int
    grid_size: int
    initial_crossings: Optional[int]      # true crossing count from physics simulation (GT, use directly)
    min_steps: Optional[int]              # theoretical_min_steps
    state: Optional[SolverState]          # full initial state (parsed from image_prompt; None if absent)
    color_state: Optional[Dict[int, frozenset]]   # color_hex -> frozenset(hole_id)


class UntangledGT:
    """Load the authoritative GT from logs/video_eval/untangled.jsonl, looked up by (level, seed)."""

    def __init__(self, path: Path = config.UNTANGLED_GT):
        self.by_seed: Dict[Tuple[str, int], UntangledEntry] = {}
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mi = row["meta_info"]
            level, seed = mi["level"], mi["seed"]
            traj = row.get("trajectory") or [{}]
            prompt = traj[0].get("image_prompt", "") if traj else ""
            gm = _GRID_RE.search(prompt)
            gs = int(gm.group(1)) if gm else _LEVEL_GS.get(level, 6)
            color_state, sym_ropes = {}, []
            for m in _ROPE_RE.finditer(prompt):
                name = m.group(1).lower()
                hexv = config.COLOR_HEX_BY_NAME.get(name)
                if hexv is None:
                    continue
                a = (int(m.group(2)), int(m.group(3)))
                b = (int(m.group(4)), int(m.group(5)))
                color_state[hexv] = frozenset((a[0] * gs + a[1], b[0] * gs + b[1]))
                sym_ropes.append({"id": len(sym_ropes), "startHole": {"row": a[0], "col": a[1]},
                                  "endHole": {"row": b[0], "col": b[1]}})
            state = symbolic_to_state({"ropes": sym_ropes}, gs) if sym_ropes else None
            self.by_seed[(level, seed)] = UntangledEntry(
                level=level, seed=seed, grid_size=gs,
                initial_crossings=mi.get("initial_crossings"),
                min_steps=mi.get("theoretical_min_steps"),
                state=state, color_state=color_state or None,
            )

    def get(self, level: str, seed: int) -> Optional[UntangledEntry]:
        return self.by_seed.get((level, seed))


@dataclass
class Episode:
    episode_id: str
    level: str
    seed: int
    ep_dir: Path                       # absolute path to images/<difficulty>/<seeddir>/
    rel_dir: str                       # images/difficulty_xxx/seed_nn
    current_pngs: List[Path]           # step_*_current.png (clean GT renders, sorted by step)
    mp4_path: Optional[Path]           # step_0000_imagined.mp4
    actions: List[dict]                # per-step parsed_action (used to advance the GT state)
    grid_size: Optional[int] = None    # inferred from level / filled in after GT parsing
    # GT (filled in after resolve)
    gt_state: Optional[SolverState] = None
    gt_source: str = "unresolved"
    gt_visual_crossings: Optional[int] = None
    gt_logical_crossings: Optional[int] = None


def iter_episodes(run_dir: Path) -> List[Episode]:
    """Build the episode list from the run's model_answer.jsonl (in model_answer line order)."""
    episodes: List[Episode] = []
    answer_path = run_dir / "model_answer.jsonl"
    for line in answer_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mi = row["meta_info"]
        traj = row.get("trajectory") or []
        if not traj:
            continue
        cyc0 = traj[0]
        cur_rel = cyc0["current_image"]            # images/difficulty_x/seed_nn/step_0000_current.png
        rel_dir = "/".join(cur_rel.split("/")[:3])
        ep_dir = run_dir / rel_dir
        current_pngs = sorted(ep_dir.glob("step_*_current.png"))
        mp4 = ep_dir / "step_0000_imagined.mp4"
        actions: List[dict] = []
        for cyc in traj:
            for sr in cyc.get("step_results", []):
                pa = sr.get("parsed_action") or sr.get("action", {}).get("answer")
                if pa:
                    actions.append(pa)
        episodes.append(
            Episode(
                episode_id=row.get("id", f"{mi['level']}_seed_{mi['seed']}"),
                level=mi["level"],
                seed=mi["seed"],
                ep_dir=ep_dir,
                rel_dir=rel_dir,
                current_pngs=current_pngs,
                mp4_path=mp4 if mp4.exists() else None,
                actions=actions,
            )
        )
    return episodes
