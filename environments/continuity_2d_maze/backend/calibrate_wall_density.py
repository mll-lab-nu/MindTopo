"""Calibrate wall_density per difficulty tier by sweeping f and measuring yes_ratio.

Pure-Python port of the TypeScript algorithm in `frontend/src/maze/{generator,points,connectivity}.ts`
following the v2 semantics from dev_trace_3:

    closed_count = round(f * M),  M = 2 * N * (N - 1)

Python's RNG differs from JS seedrandom, so individual seed→maze mappings will not match
the TS output bit-for-bit. But `yes_ratio` is a statistical aggregate over many seeds ×
many pairs; its population value depends only on the *algorithm semantics* (which are
identical by construction) and not on the RNG stream. Over 50 seeds × 6 pairs = 300 samples,
Python vs TS should agree to within ~0.05.

Usage:
    python backend/calibrate_wall_density.py
    python backend/calibrate_wall_density.py --difficulty 1,2,3 --f-min 0.40 --f-max 0.70 --f-step 0.02 --seeds 50 --point-count 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple


GRID_SIZE_BY_DIFFICULTY: Dict[int, int] = {1: 4, 2: 5, 3: 6}
LABEL_BY_DIFFICULTY: Dict[int, str] = {1: "easy", 2: "medium", 3: "hard"}


def generate_maze(size: int, seed: int, wall_density: float) -> List[List[Dict[str, bool]]]:
    """Port of generator.ts::generateMaze.

    Returns an NxN grid of cell-wall dicts (north/east/south/west -> bool, True=closed).
    """
    if size < 2:
        raise ValueError(f"size must be >= 2, got {size}")
    if not (0.0 <= wall_density <= 1.0):
        raise ValueError(f"wall_density must be in [0,1], got {wall_density}")

    rng = random.Random(seed)

    cells = [
        [{"north": True, "east": True, "south": True, "west": True} for _ in range(size)]
        for _ in range(size)
    ]
    visited = [[False] * size for _ in range(size)]
    # is_tree[y][x] = [east-is-tree-edge, south-is-tree-edge]
    is_tree = [[[False, False] for _ in range(size)] for _ in range(size)]
    tree_walls: List[Tuple[int, int, str]] = []  # (x, y, "east"|"south") canonicalized

    # DFS carve (recursive backtracker)
    DIRS = [
        (0, -1, "north", "south"),
        (1, 0, "east", "west"),
        (0, 1, "south", "north"),
        (-1, 0, "west", "east"),
    ]

    def canonicalize(x: int, y: int, self_side: str) -> Tuple[int, int, str]:
        if self_side == "east":
            return (x, y, "east")
        if self_side == "south":
            return (x, y, "south")
        if self_side == "west":
            return (x - 1, y, "east")
        return (x, y - 1, "south")

    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        x, y = stack[-1]
        frontier = []
        for dx, dy, self_side, other_side in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and not visited[ny][nx]:
                frontier.append((dx, dy, self_side, other_side))
        if not frontier:
            stack.pop()
            continue
        dx, dy, self_side, other_side = frontier[rng.randrange(len(frontier))]
        nx, ny = x + dx, y + dy
        cells[y][x][self_side] = False
        cells[ny][nx][other_side] = False
        cx, cy, cside = canonicalize(x, y, self_side)
        is_tree[cy][cx][0 if cside == "east" else 1] = True
        tree_walls.append((cx, cy, cside))
        visited[ny][nx] = True
        stack.append((nx, ny))

    M = 2 * size * (size - 1)
    K = round(wall_density * M)
    non_tree_count = (size - 1) * (size - 1)

    def set_wall(w: Tuple[int, int, str], closed: bool) -> None:
        x, y, side = w
        if side == "east":
            cells[y][x]["east"] = closed
            cells[y][x + 1]["west"] = closed
        else:
            cells[y][x]["south"] = closed
            cells[y + 1][x]["north"] = closed

    if K == non_tree_count:
        return cells

    wall_rng = random.Random(f"{seed}:walls")

    if K < non_tree_count:
        non_tree_walls: List[Tuple[int, int, str]] = []
        for y in range(size):
            for x in range(size):
                if x + 1 < size and not is_tree[y][x][0]:
                    non_tree_walls.append((x, y, "east"))
                if y + 1 < size and not is_tree[y][x][1]:
                    non_tree_walls.append((x, y, "south"))
        wall_rng.shuffle(non_tree_walls)
        to_open = non_tree_count - K
        for i in range(to_open):
            set_wall(non_tree_walls[i], False)
    else:
        pool = list(tree_walls)
        wall_rng.shuffle(pool)
        to_close = K - non_tree_count
        for i in range(to_close):
            set_wall(pool[i], True)

    return cells


def sample_points(size: int, seed: int, count: int) -> List[Tuple[int, int]]:
    """Port of points.ts::samplePoints — returns list of (x, y)."""
    total = size * size
    if count > total:
        raise ValueError(f"count {count} > total {total}")
    rng = random.Random(f"{seed}:points")
    indices = list(range(total))
    # Partial Fisher-Yates
    for i in range(count):
        j = i + rng.randrange(total - i)
        indices[i], indices[j] = indices[j], indices[i]
    return [(idx % size, idx // size) for idx in indices[:count]]


def is_connected(cells: List[List[Dict[str, bool]]], a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    size = len(cells)
    if a == b:
        return True
    visited = [[False] * size for _ in range(size)]
    queue = deque([a])
    visited[a[1]][a[0]] = True
    while queue:
        x, y = queue.popleft()
        if (x, y) == b:
            return True
        cell = cells[y][x]
        if not cell["north"] and y > 0 and not visited[y - 1][x]:
            visited[y - 1][x] = True
            queue.append((x, y - 1))
        if not cell["south"] and y < size - 1 and not visited[y + 1][x]:
            visited[y + 1][x] = True
            queue.append((x, y + 1))
        if not cell["west"] and x > 0 and not visited[y][x - 1]:
            visited[y][x - 1] = True
            queue.append((x - 1, y))
        if not cell["east"] and x < size - 1 and not visited[y][x + 1]:
            visited[y][x + 1] = True
            queue.append((x + 1, y))
    return False


def yes_ratio(size: int, wall_density: float, seeds: int, point_count: int, base_seed: int = 12345) -> Tuple[int, int, float]:
    yes = 0
    total = 0
    for s in range(seeds):
        seed = base_seed + s
        cells = generate_maze(size, seed, wall_density)
        points = sample_points(size, seed, point_count)
        for i in range(point_count):
            for j in range(i + 1, point_count):
                total += 1
                if is_connected(cells, points[i], points[j]):
                    yes += 1
    return yes, total, yes / total if total else 0.0


def parse_int_list(s: str) -> List[int]:
    return [int(part) for part in s.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep wall_density and report yes_ratio per difficulty.")
    parser.add_argument("--difficulty", default="1,2,3", help="Comma-separated difficulty keys.")
    parser.add_argument("--f-min", type=float, default=0.40)
    parser.add_argument("--f-max", type=float, default=0.70)
    parser.add_argument("--f-step", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, default=50, help="Scenes per (N, f) cell.")
    parser.add_argument("--point-count", type=int, default=4)
    parser.add_argument("--target", type=float, default=0.5, help="Ideal yes_ratio.")
    parser.add_argument("--tolerance-default", type=float, default=0.05, help="|yes_ratio - target| tolerance for N>=5.")
    parser.add_argument("--tolerance-n4", type=float, default=0.10, help="Looser tolerance for N=4 (discretisation).")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "calibrated_wall_density_4_5_6.json"))
    args = parser.parse_args()

    difficulties = [d for d in parse_int_list(args.difficulty) if d in GRID_SIZE_BY_DIFFICULTY]
    # Build f grid (avoid floating drift)
    steps = int(round((args.f_max - args.f_min) / args.f_step)) + 1
    f_values = [round(args.f_min + i * args.f_step, 4) for i in range(steps)]

    results: Dict[str, Dict] = {}

    for d in difficulties:
        N = GRID_SIZE_BY_DIFFICULTY[d]
        label = LABEL_BY_DIFFICULTY[d]
        print(f"\n=== difficulty {d} ({label}) — N={N}, M={2*N*(N-1)}, critical f_min={round((N-1)/(2*N),3)} ===")
        print(f"  {'f':>6}  {'yes':>5}  {'total':>5}  {'yes_ratio':>9}")
        table: List[Tuple[float, int, int, float]] = []
        for f in f_values:
            yes, total, ratio = yes_ratio(N, f, args.seeds, args.point_count)
            table.append((f, yes, total, ratio))
            print(f"  {f:>6.2f}  {yes:>5d}  {total:>5d}  {ratio:>9.3f}")

        tolerance = args.tolerance_n4 if N == 4 else args.tolerance_default
        in_band = [row for row in table if abs(row[3] - args.target) <= tolerance]
        if not in_band:
            print(f"  WARN: no f satisfies |yes_ratio - {args.target}| <= {tolerance}. Picking closest.")
            best = min(table, key=lambda r: abs(r[3] - args.target))
        else:
            best = min(in_band, key=lambda r: abs(r[3] - args.target))
        f_star, yes_star, total_star, ratio_star = best
        print(f"  --> pick f = {f_star:.2f}  (yes_ratio={ratio_star:.3f})")
        results[label] = {
            "difficulty": d,
            "grid_size": N,
            "wall_density": f_star,
            "yes_ratio": ratio_star,
            "total_pairs": total_star,
            "tolerance_used": tolerance,
            "sweep": [{"f": f, "yes": y, "total": t, "yes_ratio": r} for f, y, t, r in table],
        }

    # Summary
    print("\n=== Summary (copy into --wall-density-map) ===")
    parts = []
    for d in difficulties:
        label = LABEL_BY_DIFFICULTY[d]
        f = results[label]["wall_density"]
        parts.append(f"{d}:{f:.2f}")
        print(f"  {label}: grid_size={results[label]['grid_size']} wall_density={f:.2f} yes_ratio={results[label]['yes_ratio']:.3f}")
    print(f"  map: '{','.join(parts)}'")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "target": args.target,
        "seeds": args.seeds,
        "point_count": args.point_count,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote calibration artefact to {out_path}")


if __name__ == "__main__":
    main()
