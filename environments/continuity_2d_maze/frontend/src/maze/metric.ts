// Difficulty metric (iter D8a). Decouples benchmark difficulty from generation
// parameters (grid_size / wall_density / diagonal_ratio / partial_ratio) so
// the same difficulty tier can span diverse mazes.
//
// For reachability_set questions with target A and n-1 other points X:
//   if A connected to X:
//     detour(A, X) = max(0, d_bfs(A, X) - d_manhattan(A, X))
//   else:
//     iso(A, X) = min(isolation_perim(C_A), isolation_perim(C_X))
//   scene_score = sum(detour) + sum(iso)
//
// Implemented on the 4-triangle BFS graph (iter D3) so diagonals (dA) are
// uniformly treated as "blocked edges" alongside H/V Mode A walls. Partial
// walls (B/C/dB/dC) are not blocked → they don't contribute to perim and
// are freely traversable in d_bfs.

import {
  TRIANGLES,
  TRI_INDEX,
  Triangle,
  adjacentTriangles,
  intraCellBlocked,
  neighborCoord,
  oppositeTriangle,
  sideOf,
  wallBlocks,
} from './connectivity';
import { LabeledPoint, MazeGrid } from './types';

export interface PairBreakdown {
  other_name: string;
  connected: boolean;
  detour?: number;
  iso?: number;
}

export interface DifficultyBreakdown {
  scene_score: number;
  detour_total: number;
  iso_total: number;
  per_pair: PairBreakdown[];
}

// iter D15 (2026-04-24): tier is no longer score-driven. The frontend
// generator stores diagnostic numbers (scene_score / detour_total /
// iso_total / min_path / area_a / area_b) for the UI Stats panel only;
// the displayed `tier` comes from the config the caller passed in (UI
// tier button or CLI tier_config), not from a threshold lookup.
export type DifficultyTier = 'easy' | 'medium' | 'hard';

// Reachable 4-triangle area from a start point. Uses existing BFS, which
// respects blocking walls (Mode-A including bar-recolored), dA diagonals,
// and DOES NOT block on partial walls (length<1) or dB/dC partial diagonals
// — exactly the semantics required by Q2's area-based tier rules.
export function reachableTriangleArea(
  maze: MazeGrid,
  from: { x: number; y: number },
): number {
  const { visited } = bfsFull(maze, from);
  let count = 0;
  const { size } = maze;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (let t = 0; t < 4; t++) {
        if (visited[y][x][t]) count += 1;
      }
    }
  }
  return count;
}

// Compute (area_A, area_B, total) for two points on a maze. Used by Q2
// (iter D12) to classify scenes by component size balance & absolute
// magnitude rather than perimeter.
export function computeReachableAreas(
  maze: MazeGrid,
  a: { x: number; y: number },
  b: { x: number; y: number },
): { area_a: number; area_b: number; total: number } {
  return {
    area_a: reachableTriangleArea(maze, a),
    area_b: reachableTriangleArea(maze, b),
    total: maze.size * maze.size * 4,
  };
}

// BFS cell-hop distance from `a` to `b` on the 4-triangle graph. Returns
// Number.POSITIVE_INFINITY if b is unreachable. Used by question_gen.ts for
// Q2's avg-detour calculation over correct_removals.
export function bfsCellDistance(
  maze: MazeGrid,
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  if (a.x === b.x && a.y === b.y) return 0;
  const { dist } = bfsFull(maze, a);
  return dist[b.y][b.x];
}

// Isolation perimeter of the component containing each of two disconnected
// points. `perim_a` is the number of blocked edges along the boundary of
// C_a (the 4-triangle-BFS component reachable from a); `perim_b` the same
// for b. Used by Q2 (iter D9) to build `balanced_iso = min²/max` — a metric
// that rewards balanced components over lopsided ones (the latter are
// trivially solvable by opening the bar surrounding the tiny component).
export function computeIsolationPair(
  maze: MazeGrid,
  a: { x: number; y: number },
  b: { x: number; y: number },
): { perim_a: number; perim_b: number } {
  const { visited: visitedA } = bfsFull(maze, a);
  const { visited: visitedB } = bfsFull(maze, b);
  return {
    perim_a: isolationPerimeterFromVisited(maze, visitedA),
    perim_b: isolationPerimeterFromVisited(maze, visitedB),
  };
}

// 4-triangle BFS from a cell center, recording:
//   dist[y][x]      = min cell-hops to first reach any triangle of cell (x,y)
//   visited[y][x][i] = whether triangle i of cell (x,y) is reachable
//
// The start point lives at cell center, which is the shared vertex of all 4
// triangles, so all 4 triangles of the start cell begin as visited at hop 0.
// Triangles in other cells are visited at the cell-hop count they arrive at.
// Intra-cell transitions carry the SAME cell-hop count (we stayed in one
// cell); inter-cell transitions increment by 1.
function bfsFull(
  maze: MazeGrid,
  from: { x: number; y: number },
): { dist: number[][]; visited: boolean[][][] } {
  const { size, diagonals } = maze;
  const dist: number[][] = Array.from({ length: size }, () =>
    Array(size).fill(Number.POSITIVE_INFINITY),
  );
  const visited: boolean[][][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => [false, false, false, false]),
  );
  const queue: Array<{ x: number; y: number; t: Triangle; d: number }> = [];
  for (const t of TRIANGLES) {
    visited[from.y][from.x][TRI_INDEX[t]] = true;
    queue.push({ x: from.x, y: from.y, t, d: 0 });
  }
  dist[from.y][from.x] = 0;

  let head = 0;
  while (head < queue.length) {
    const { x, y, t, d } = queue[head++];
    const diag = diagonals[`${x},${y}`];

    // Intra-cell: same cell, same d
    for (const neighborT of adjacentTriangles(t)) {
      if (intraCellBlocked(diag, t, neighborT)) continue;
      if (visited[y][x][TRI_INDEX[neighborT]]) continue;
      visited[y][x][TRI_INDEX[neighborT]] = true;
      queue.push({ x, y, t: neighborT, d });
    }

    // Inter-cell: adjacent cell via this triangle's edge, d + 1
    const side = sideOf(t);
    if (wallBlocks(maze, x, y, side)) continue;
    const [nx, ny] = neighborCoord(x, y, side);
    if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
    const arrivalT = oppositeTriangle(t);
    if (visited[ny][nx][TRI_INDEX[arrivalT]]) continue;
    visited[ny][nx][TRI_INDEX[arrivalT]] = true;
    const newDist = d + 1;
    if (newDist < dist[ny][nx]) dist[ny][nx] = newDist;
    queue.push({ x: nx, y: ny, t: arrivalT, d: newDist });
  }
  return { dist, visited };
}

// Isolation perimeter of the component identified by `visited`. Counts each
// blocked edge in the 4-triangle graph where one endpoint is in the component
// (visited === true) and the other is not (visited === false OR out-of-grid
// — the out-of-grid case is ignored because we only count internal walls and
// dA diagonals, not the outer border).
//
// Each blocked boundary edge is visited exactly once because we iterate nodes
// inside the component and look outward.
function isolationPerimeterFromVisited(
  maze: MazeGrid,
  visited: boolean[][][],
): number {
  const { size, diagonals } = maze;
  let perim = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (const t of TRIANGLES) {
        if (!visited[y][x][TRI_INDEX[t]]) continue;
        const diag = diagonals[`${x},${y}`];

        // Intra-cell blocked edges where the other triangle is NOT in C.
        for (const neighborT of adjacentTriangles(t)) {
          if (!intraCellBlocked(diag, t, neighborT)) continue;
          if (!visited[y][x][TRI_INDEX[neighborT]]) perim += 1;
        }

        // Inter-cell blocked edges where the arrival triangle is NOT in C
        // (or the neighbor cell is out-of-grid → skip, outer border doesn't
        // count toward isolation identification difficulty).
        const side = sideOf(t);
        if (!wallBlocks(maze, x, y, side)) continue;
        const [nx, ny] = neighborCoord(x, y, side);
        if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
        const arrivalT = oppositeTriangle(t);
        if (!visited[ny][nx][TRI_INDEX[arrivalT]]) perim += 1;
      }
    }
  }
  return perim;
}

export function computeDifficultyMetric(
  maze: MazeGrid,
  target: LabeledPoint,
  others: LabeledPoint[],
): DifficultyBreakdown {
  const { dist: distFromTarget, visited: visitedFromTarget } = bfsFull(
    maze,
    target,
  );
  const perimTargetCached = isolationPerimeterFromVisited(
    maze,
    visitedFromTarget,
  );

  const perPair: PairBreakdown[] = [];
  let detourTotal = 0;
  let isoTotal = 0;

  for (const X of others) {
    const d_bfs = distFromTarget[X.y][X.x];
    const d_man = Math.abs(X.x - target.x) + Math.abs(X.y - target.y);
    if (Number.isFinite(d_bfs)) {
      const detour = Math.max(0, d_bfs - d_man);
      detourTotal += detour;
      perPair.push({ other_name: X.name, connected: true, detour });
    } else {
      const { visited: visitedFromX } = bfsFull(maze, X);
      const perimX = isolationPerimeterFromVisited(maze, visitedFromX);
      const iso = Math.min(perimTargetCached, perimX);
      isoTotal += iso;
      perPair.push({ other_name: X.name, connected: false, iso });
    }
  }

  return {
    scene_score: detourTotal + isoTotal,
    detour_total: detourTotal,
    iso_total: isoTotal,
    per_pair: perPair,
  };
}
