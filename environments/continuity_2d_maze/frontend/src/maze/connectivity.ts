import { DiagonalSpec, LabeledPoint, MazeGrid, Pair } from './types';

export interface CellCoord {
  x: number;
  y: number;
}

export type Side = 'north' | 'east' | 'south' | 'west';

// 4-triangle decomposition (iter D3). Every cell is conceptually split into
// four micro-triangles whose apex is the cell center:
//   T_N: top triangle, owns the north edge
//   T_E: right triangle, owns the east edge
//   T_S: bottom triangle, owns the south edge
//   T_W: left triangle, owns the west edge
// Each edge belongs to exactly one triangle, and each pair of adjacent
// triangles shares one half of a diagonal. Whether the shared diagonal is
// closed by a dA wall determines whether the pair is traversable.
export type Triangle = 'N' | 'E' | 'S' | 'W';

export const TRIANGLES: readonly Triangle[] = ['N', 'E', 'S', 'W'];
export const TRI_INDEX: Record<Triangle, number> = { N: 0, E: 1, S: 2, W: 3 };

// True if the wall on `side` of cell (x, y) blocks passage. A wall blocks
// when (a) the boolean flag is set AND (b) its shape entry is Mode A
// (length >= 1, full-edge). Partial walls (length < 1) are visible but do
// not impede BFS traversal. Missing shape entry (should never happen for a
// closed wall) is treated conservatively as blocking.
export function wallBlocks(maze: MazeGrid, x: number, y: number, side: Side): boolean {
  const cell = maze.cells[y][x];
  if (!cell[side]) return false;
  let key: string;
  if (side === 'east') key = `${x},${y},e`;
  else if (side === 'south') key = `${x},${y},s`;
  else if (side === 'west') key = `${x - 1},${y},e`;
  else key = `${x},${y - 1},s`;
  const spec = maze.shapes[key];
  if (spec === undefined) return true;
  return spec.length >= 1;
}

export function sideOf(t: Triangle): Side {
  if (t === 'N') return 'north';
  if (t === 'E') return 'east';
  if (t === 'S') return 'south';
  return 'west';
}

export function oppositeTriangle(t: Triangle): Triangle {
  if (t === 'N') return 'S';
  if (t === 'S') return 'N';
  if (t === 'E') return 'W';
  return 'E';
}

// The two triangles geometrically adjacent to t within its cell. Adjacent
// means they share half of a diagonal (slash or backslash). Non-adjacent
// pairs (N↔S, E↔W) share only the center point and are never traversable
// without passing through one of the other two triangles.
export function adjacentTriangles(t: Triangle): [Triangle, Triangle] {
  if (t === 'N') return ['E', 'W'];
  if (t === 'E') return ['N', 'S'];
  if (t === 'S') return ['W', 'E'];
  return ['S', 'N']; // W
}

// Which diagonal separates two adjacent triangles within a cell?
//   slash '/':     separates N↔E and S↔W
//   backslash '\': separates E↔S and W↔N
// Returns null for non-adjacent pairs.
function intraCellSeparator(from: Triangle, to: Triangle): 'slash' | 'backslash' | null {
  if ((from === 'N' && to === 'E') || (from === 'E' && to === 'N')) return 'slash';
  if ((from === 'S' && to === 'W') || (from === 'W' && to === 'S')) return 'slash';
  if ((from === 'E' && to === 'S') || (from === 'S' && to === 'E')) return 'backslash';
  if ((from === 'W' && to === 'N') || (from === 'N' && to === 'W')) return 'backslash';
  return null;
}

// Is movement between two intra-cell triangles blocked by a diagonal wall?
// Only a dA (full √2) diagonal of the matching orientation blocks. Partial
// dB/dC diagonals are visual-only and do not block.
export function intraCellBlocked(
  diag: DiagonalSpec | undefined,
  from: Triangle,
  to: Triangle,
): boolean {
  if (!diag || diag.mode !== 'dA') return false;
  const sep = intraCellSeparator(from, to);
  return sep === diag.orientation;
}

export function neighborCoord(x: number, y: number, side: Side): [number, number] {
  if (side === 'north') return [x, y - 1];
  if (side === 'south') return [x, y + 1];
  if (side === 'west') return [x - 1, y];
  return [x + 1, y];
}

export function isConnected(maze: MazeGrid, a: CellCoord, b: CellCoord): boolean {
  const { size, diagonals } = maze;
  if (a.x === b.x && a.y === b.y) return true;

  // visited[y][x][triIndex]: one slot per (cell, triangle). A labeled point
  // sits at the cell center which is the shared vertex of all 4 triangles,
  // so the start cell's 4 triangles are all visited at t=0. Symmetrically,
  // the target is considered reached if BFS visits ANY triangle of (b.x, b.y).
  const visited: boolean[][][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => [false, false, false, false]),
  );
  const queue: [number, number, Triangle][] = [];
  for (const t of TRIANGLES) {
    visited[a.y][a.x][TRI_INDEX[t]] = true;
    queue.push([a.x, a.y, t]);
  }

  let head = 0;
  while (head < queue.length) {
    const [x, y, t] = queue[head++];
    if (x === b.x && y === b.y) return true;
    const diag = diagonals[`${x},${y}`];

    // Intra-cell: move to one of the two geometrically adjacent triangles.
    for (const neighborT of adjacentTriangles(t)) {
      if (intraCellBlocked(diag, t, neighborT)) continue;
      if (visited[y][x][TRI_INDEX[neighborT]]) continue;
      visited[y][x][TRI_INDEX[neighborT]] = true;
      queue.push([x, y, neighborT]);
    }

    // Inter-cell: exit through this triangle's own outer edge. Triangle T
    // owns exactly one side; if that side is not a Mode A wall, we cross
    // into the neighbor cell and land in the opposite triangle there.
    const side = sideOf(t);
    if (wallBlocks(maze, x, y, side)) continue;
    const [nx, ny] = neighborCoord(x, y, side);
    if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
    const arrivalT = oppositeTriangle(t);
    if (visited[ny][nx][TRI_INDEX[arrivalT]]) continue;
    visited[ny][nx][TRI_INDEX[arrivalT]] = true;
    queue.push([nx, ny, arrivalT]);
  }
  return false;
}

export function allPairs(points: LabeledPoint[], maze: MazeGrid): Pair[] {
  const pairs: Pair[] = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      pairs.push({
        a_name: points[i].name,
        b_name: points[j].name,
        connected: isConnected(maze, points[i], points[j]),
      });
    }
  }
  return pairs;
}
