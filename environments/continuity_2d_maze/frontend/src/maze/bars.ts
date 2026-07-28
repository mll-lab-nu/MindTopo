import seedrandom from 'seedrandom';

import {
  CellCoord,
  Side,
  allPairs,
  isConnected,
  neighborCoord,
  wallBlocks,
} from './connectivity';
import {
  Bar,
  BarColor,
  BarEdgeKind,
  CellWalls,
  DiagonalKey,
  DiagonalSpec,
  LabeledPoint,
  MazeGrid,
  Pair,
  WallKey,
} from './types';

export const BAR_COLORS: readonly BarColor[] = [
  'purple',
  'red',
  'green',
  'blue',
  'yellow',
  'orange',
] as const;

// H/V WallKey format: '${x},${y},e' or '${x},${y},s' (3-part)
// Diagonal DiagonalKey format: '${x},${y}' (2-part; the key of maze.diagonals)
export function parseEdgeKey(key: WallKey): { x: number; y: number; side: 'east' | 'south' } {
  const parts = key.split(',');
  if (parts.length !== 3) throw new Error(`invalid WallKey: ${key}`);
  const x = Number(parts[0]);
  const y = Number(parts[1]);
  const tag = parts[2];
  if (!Number.isInteger(x) || !Number.isInteger(y)) throw new Error(`invalid WallKey: ${key}`);
  if (tag === 'e') return { x, y, side: 'east' };
  if (tag === 's') return { x, y, side: 'south' };
  throw new Error(`invalid WallKey: ${key}`);
}

function parseDiagKey(key: DiagonalKey): { x: number; y: number } {
  const parts = key.split(',');
  if (parts.length !== 2) throw new Error(`invalid DiagonalKey: ${key}`);
  const x = Number(parts[0]);
  const y = Number(parts[1]);
  if (!Number.isInteger(x) || !Number.isInteger(y)) throw new Error(`invalid DiagonalKey: ${key}`);
  return { x, y };
}

export type ParsedBarEdge =
  | { kind: 'hv'; x: number; y: number; side: 'east' | 'south' }
  | { kind: 'diag'; x: number; y: number };

export function parseBarEdge(bar: Bar): ParsedBarEdge {
  if (bar.kind === 'diag') {
    const { x, y } = parseDiagKey(bar.edge);
    return { kind: 'diag', x, y };
  }
  const { x, y, side } = parseEdgeKey(bar.edge);
  return { kind: 'hv', x, y, side };
}

function oppositeSide(side: Side): Side {
  if (side === 'east') return 'west';
  if (side === 'west') return 'east';
  if (side === 'north') return 'south';
  return 'north';
}

type BarCandidate =
  | { kind: 'hv'; edge: WallKey }
  | { kind: 'diag'; edge: DiagonalKey };

// Candidate bar pool = H/V Mode-A walls (wallBlocks true) PLUS every
// `dA`-mode diagonal in maze.diagonals. Partial walls (H/V length<1, or
// dB/dC diagonals) are excluded — removing them wouldn't change
// connectivity, so they would produce invalid ground-truth bars.
export function enumerateBarCandidates(maze: MazeGrid): BarCandidate[] {
  const candidates: BarCandidate[] = [];
  const { size, diagonals } = maze;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x + 1 < size && wallBlocks(maze, x, y, 'east')) {
        candidates.push({ kind: 'hv', edge: `${x},${y},e` });
      }
      if (y + 1 < size && wallBlocks(maze, x, y, 'south')) {
        candidates.push({ kind: 'hv', edge: `${x},${y},s` });
      }
    }
  }
  // Diagonals after H/V to keep the insertion order deterministic for a
  // given maze (shuffle inside placeRandomBars then picks the top N).
  for (const [key, spec] of Object.entries(diagonals)) {
    if ((spec as DiagonalSpec).mode !== 'dA') continue;
    candidates.push({ kind: 'diag', edge: key });
  }
  return candidates;
}

export function placeRandomBars(
  maze: MazeGrid,
  seed: number | string,
  count: number,
  colors: readonly BarColor[] = BAR_COLORS,
): Bar[] {
  if (count <= 0) return [];
  if (count > colors.length) {
    throw new Error(
      `placeRandomBars: count=${count} exceeds palette size=${colors.length}`,
    );
  }
  const candidates = enumerateBarCandidates(maze);
  if (candidates.length < count) {
    throw new Error(
      `placeRandomBars: only ${candidates.length} blocking walls (H/V + dA) available, need ${count}`,
    );
  }
  const rng = seedrandom(`${seed}:bars`);
  const shuffled = candidates.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const picked = shuffled.slice(0, count);
  return picked.map((cand, i) => ({
    id: `bar_${i}`,
    color: colors[i],
    kind: cand.kind as BarEdgeKind,
    edge: cand.edge,
  }));
}

// Shallow-clone maze with every bar's backing wall reopened. H/V bars clear
// cells[y][x][side] + mirror neighbor + drop shapes[edge]; diag bars drop
// diagonals[edge]. Input maze is never mutated. Each bucket is cloned at
// most once (lazy copy-on-first-touch) to keep the hot path fast for D2's
// bar iteration loop.
export function withBarsRemoved(maze: MazeGrid, removedBars: readonly Bar[]): MazeGrid {
  if (removedBars.length === 0) return maze;
  const { size } = maze;
  let cells: CellWalls[][] = maze.cells;
  let shapes = maze.shapes;
  let diagonals = maze.diagonals;
  let clonedCells = false;
  let clonedShapes = false;
  let clonedDiagonals = false;

  for (const bar of removedBars) {
    const parsed = parseBarEdge(bar);
    if (parsed.kind === 'hv') {
      if (!clonedCells) {
        cells = cells.map((row) => row.slice());
        clonedCells = true;
      }
      if (!clonedShapes) {
        shapes = { ...shapes };
        clonedShapes = true;
      }
      const { x, y, side } = parsed;
      cells[y][x] = { ...cells[y][x], [side]: false };
      const [nx, ny] = neighborCoord(x, y, side);
      if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
        const opp = oppositeSide(side);
        cells[ny][nx] = { ...cells[ny][nx], [opp]: false };
      }
      delete shapes[bar.edge];
    } else {
      if (!clonedDiagonals) {
        diagonals = { ...diagonals };
        clonedDiagonals = true;
      }
      delete diagonals[bar.edge];
    }
  }
  return { size, cells, shapes, diagonals };
}

export function isConnectedWithBars(
  maze: MazeGrid,
  removedBars: readonly Bar[],
  a: CellCoord,
  b: CellCoord,
): boolean {
  if (removedBars.length === 0) return isConnected(maze, a, b);
  return isConnected(withBarsRemoved(maze, removedBars), a, b);
}

export function allPairsWithBars(
  points: LabeledPoint[],
  maze: MazeGrid,
  removedBars: readonly Bar[],
): Pair[] {
  if (removedBars.length === 0) return allPairs(points, maze);
  return allPairs(points, withBarsRemoved(maze, removedBars));
}
