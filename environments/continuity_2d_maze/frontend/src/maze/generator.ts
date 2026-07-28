import seedrandom from 'seedrandom';
import {
  CellWalls,
  DiagonalKey,
  DiagonalOrientation,
  DiagonalSpec,
  MazeGrid,
  WallKey,
  WallSpec,
} from './types';

// Default knob values for the D4 unified pipeline.
export const DEFAULT_DIAGONAL_RATIO = 0;
export const DEFAULT_PARTIAL_RATIO = 0;
export const DEFAULT_PARTIAL_BC_SPLIT = 0.5;

// Fixed partial-wall lengths (iter D1). Relative to cell edge; cell diagonal
// equals Math.SQRT2 in these units.
const PARTIAL_LENGTH = 0.5;
const DIAG_PARTIAL_LENGTH = 0.5 * Math.SQRT2;

type Side = keyof CellWalls;
type InternalSide = 'east' | 'south';

interface Direction {
  dx: number;
  dy: number;
  self: Side;
  other: Side;
}

interface InternalWall {
  x: number;
  y: number;
  side: InternalSide;
}

const DIRS: readonly Direction[] = [
  { dx: 0, dy: -1, self: 'north', other: 'south' },
  { dx: 1, dy: 0, self: 'east', other: 'west' },
  { dx: 0, dy: 1, self: 'south', other: 'north' },
  { dx: -1, dy: 0, self: 'west', other: 'east' },
];

function canonicalize(x: number, y: number, self: Side): InternalWall {
  if (self === 'east') return { x, y, side: 'east' };
  if (self === 'south') return { x, y, side: 'south' };
  if (self === 'west') return { x: x - 1, y, side: 'east' };
  return { x, y: y - 1, side: 'south' };
}

function wallKeyOf(w: InternalWall): WallKey {
  return `${w.x},${w.y},${w.side === 'east' ? 'e' : 's'}`;
}

function setWall(cells: CellWalls[][], w: InternalWall, closed: boolean): void {
  if (w.side === 'east') {
    cells[w.y][w.x].east = closed;
    cells[w.y][w.x + 1].west = closed;
  } else {
    cells[w.y][w.x].south = closed;
    cells[w.y + 1][w.x].north = closed;
  }
}

function fisherYates<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

// The two cells adjacent to an internal wall.
//   east wall of (x, y) separates (x, y) and (x+1, y)
//   south wall of (x, y) separates (x, y) and (x, y+1)
function adjacentCells(w: InternalWall): [{ x: number; y: number }, { x: number; y: number }] {
  if (w.side === 'east') {
    return [{ x: w.x, y: w.y }, { x: w.x + 1, y: w.y }];
  }
  return [{ x: w.x, y: w.y }, { x: w.x, y: w.y + 1 }];
}

export function generateMaze(
  size: number,
  seed: number,
  wall_density: number = 0,
  diagonal_ratio: number = DEFAULT_DIAGONAL_RATIO,
  partial_ratio: number = DEFAULT_PARTIAL_RATIO,
  partial_bc_split: number = DEFAULT_PARTIAL_BC_SPLIT,
): MazeGrid {
  if (!Number.isInteger(size) || size < 2) {
    throw new Error(`grid_size must be an integer >= 2, got ${size}`);
  }
  if (!(wall_density >= 0 && wall_density <= 1)) {
    throw new Error(`wall_density must be in [0, 1], got ${wall_density}`);
  }
  if (!(diagonal_ratio >= 0 && diagonal_ratio <= 1)) {
    throw new Error(`diagonal_ratio must be in [0, 1], got ${diagonal_ratio}`);
  }
  if (!(partial_ratio >= 0 && partial_ratio <= 1)) {
    throw new Error(`partial_ratio must be in [0, 1], got ${partial_ratio}`);
  }
  if (!(partial_bc_split >= 0 && partial_bc_split <= 1)) {
    throw new Error(`partial_bc_split must be in [0, 1], got ${partial_bc_split}`);
  }

  // ------------------------------------------------------------------
  // Stage 1: DFS + wall_density adjust → K closed H/V walls (all Mode A).
  // Unchanged from iter B1 logic.
  // ------------------------------------------------------------------
  const rng = seedrandom(String(seed));
  const cells: CellWalls[][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => ({ north: true, east: true, south: true, west: true })),
  );
  const visited: boolean[][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => false),
  );

  const isTree: boolean[][][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => [false, false]),
  );
  const treeWalls: InternalWall[] = [];

  const stack: [number, number][] = [[0, 0]];
  visited[0][0] = true;

  while (stack.length > 0) {
    const [x, y] = stack[stack.length - 1];
    const frontier: Direction[] = [];
    for (const dir of DIRS) {
      const nx = x + dir.dx;
      const ny = y + dir.dy;
      if (nx >= 0 && nx < size && ny >= 0 && ny < size && !visited[ny][nx]) {
        frontier.push(dir);
      }
    }
    if (frontier.length === 0) {
      stack.pop();
      continue;
    }
    const pick = frontier[Math.floor(rng() * frontier.length)];
    const nx = x + pick.dx;
    const ny = y + pick.dy;
    cells[y][x][pick.self] = false;
    cells[ny][nx][pick.other] = false;
    const w = canonicalize(x, y, pick.self);
    isTree[w.y][w.x][w.side === 'east' ? 0 : 1] = true;
    treeWalls.push(w);
    visited[ny][nx] = true;
    stack.push([nx, ny]);
  }

  const M = 2 * size * (size - 1);
  const K = Math.round(wall_density * M);
  const nonTreeCount = (size - 1) * (size - 1);
  const wallRng = seedrandom(`${seed}:walls`);

  if (K < nonTreeCount) {
    const nonTreeWalls: InternalWall[] = [];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (x + 1 < size && !isTree[y][x][0]) nonTreeWalls.push({ x, y, side: 'east' });
        if (y + 1 < size && !isTree[y][x][1]) nonTreeWalls.push({ x, y, side: 'south' });
      }
    }
    fisherYates(nonTreeWalls, wallRng);
    const toOpen = nonTreeCount - K;
    for (let i = 0; i < toOpen; i++) {
      setWall(cells, nonTreeWalls[i], false);
    }
  } else if (K > nonTreeCount) {
    const pool = treeWalls.slice();
    fisherYates(pool, wallRng);
    const toClose = K - nonTreeCount;
    for (let i = 0; i < toClose; i++) {
      setWall(cells, pool[i], true);
    }
  }

  // Enumerate all currently closed H/V walls (K of them, all Mode A at this
  // stage). These are the walls subject to later conversion in stages 2/3.
  const closedWalls: InternalWall[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x + 1 < size && cells[y][x].east) closedWalls.push({ x, y, side: 'east' });
      if (y + 1 < size && cells[y][x].south) closedWalls.push({ x, y, side: 'south' });
    }
  }

  // ------------------------------------------------------------------
  // Stage 2: Convert diagonal_ratio * K walls from H/V to dA diagonal.
  // For each selected wall: remove it from its edge, pick one of the two
  // adjacent cells at random, and place a dA diagonal there with random
  // orientation. If both adjacent cells already carry a diagonal (from
  // earlier conversions), skip this wall (collision → actual count may be
  // slightly less than requested).
  // ------------------------------------------------------------------
  const diagonals: Record<DiagonalKey, DiagonalSpec> = {};
  const convertedSet = new Set<string>();
  const K_diag = Math.round(diagonal_ratio * closedWalls.length);

  if (K_diag > 0) {
    const convertRng = seedrandom(`${seed}:convert`);
    const shuffled = closedWalls.slice();
    fisherYates(shuffled, convertRng);
    let converted = 0;
    for (const w of shuffled) {
      if (converted >= K_diag) break;
      const [c1, c2] = adjacentCells(w);
      const key1 = `${c1.x},${c1.y}`;
      const key2 = `${c2.x},${c2.y}`;
      const pickFirst = convertRng() < 0.5;
      let targetKey = pickFirst ? key1 : key2;
      if (diagonals[targetKey] !== undefined) {
        targetKey = pickFirst ? key2 : key1;
      }
      if (diagonals[targetKey] !== undefined) continue;
      setWall(cells, w, false);
      const orientation: DiagonalOrientation = convertRng() < 0.5 ? 'slash' : 'backslash';
      diagonals[targetKey] = {
        orientation,
        mode: 'dA',
        length: Math.SQRT2,
        offset: 0,
      };
      convertedSet.add(wallKeyOf(w));
      converted++;
    }
  }

  // Remaining H/V walls (still closed as Mode A after Stage 2).
  const remainingHvWalls: InternalWall[] = [];
  for (const w of closedWalls) {
    if (!convertedSet.has(wallKeyOf(w))) remainingHvWalls.push(w);
  }

  // ------------------------------------------------------------------
  // Stage 3: Assign shapes. Mode A for every remaining H/V wall (default),
  // then pick round(partial_ratio * totalWalls) walls uniformly from the
  // union {remaining H/V, diagonals} to convert to partial (H/V A→B/C,
  // diagonal dA→dB/dC). The partial_bc_split applies symmetrically:
  //   random < split → centered variant (B for H/V, dB for diag)
  //   otherwise → endpoint variant (C, dC)
  // ------------------------------------------------------------------
  const shapes: Record<WallKey, WallSpec> = {};
  for (const w of remainingHvWalls) {
    const orientation = w.side === 'east' ? 'vertical' : 'horizontal';
    shapes[wallKeyOf(w)] = { orientation, length: 1, offset: 0 };
  }

  const totalWalls = remainingHvWalls.length + Object.keys(diagonals).length;
  const K_partial = Math.round(partial_ratio * totalWalls);

  if (K_partial > 0 && totalWalls > 0) {
    const partialRng = seedrandom(`${seed}:partial`);
    type HvTarget = { kind: 'hv'; wall: InternalWall };
    type DiagTarget = { kind: 'diag'; cellKey: DiagonalKey };
    type PartialTarget = HvTarget | DiagTarget;
    const allWalls: PartialTarget[] = [];
    for (const w of remainingHvWalls) allWalls.push({ kind: 'hv', wall: w });
    for (const key of Object.keys(diagonals)) allWalls.push({ kind: 'diag', cellKey: key });
    fisherYates(allWalls, partialRng);
    const toConvert = Math.min(K_partial, allWalls.length);

    for (let i = 0; i < toConvert; i++) {
      const target = allWalls[i];
      const pickCentered = partialRng() < partial_bc_split;
      if (target.kind === 'hv') {
        const orientation = target.wall.side === 'east' ? 'vertical' : 'horizontal';
        const key = wallKeyOf(target.wall);
        if (pickCentered) {
          shapes[key] = {
            orientation,
            length: PARTIAL_LENGTH,
            offset: (1 - PARTIAL_LENGTH) / 2,
          };
        } else {
          const endOffset = partialRng() < 0.5 ? 0 : 1 - PARTIAL_LENGTH;
          shapes[key] = { orientation, length: PARTIAL_LENGTH, offset: endOffset };
        }
      } else {
        const diag = diagonals[target.cellKey];
        if (pickCentered) {
          diagonals[target.cellKey] = {
            orientation: diag.orientation,
            mode: 'dB',
            length: DIAG_PARTIAL_LENGTH,
            offset: (Math.SQRT2 - DIAG_PARTIAL_LENGTH) / 2,
          };
        } else {
          const endOffset = partialRng() < 0.5 ? 0 : Math.SQRT2 - DIAG_PARTIAL_LENGTH;
          diagonals[target.cellKey] = {
            orientation: diag.orientation,
            mode: 'dC',
            length: DIAG_PARTIAL_LENGTH,
            offset: endOffset,
          };
        }
      }
    }
  }

  return { size, cells, shapes, diagonals };
}
