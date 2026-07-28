import { calculateRegions, edgeKey } from './game';
import type { Direction, LevelJson, Vertex } from './types';

export const BOARD_SIZES = [4, 5, 6] as const;
export type BoardSize = (typeof BOARD_SIZES)[number];

const COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'gray'] as const;
const DIRS: Direction[] = ['U', 'D', 'L', 'R'];
const DIR_TO_DELTA: Record<Direction, Vertex> = {
  U: { x: 0, y: 1 },
  D: { x: 0, y: -1 },
  L: { x: -1, y: 0 },
  R: { x: 1, y: 0 },
};

const MASK_64 = (1n << 64n) - 1n;
const SETUP_SEED_MIX = 0x9e3779b97f4a7c15n;
const BOARD_SEED_MIX = 0x94d049bb133111ebn;

class PythonRandom {
  private static readonly N = 624;
  private static readonly M = 397;
  private static readonly MATRIX_A = 0x9908b0df;
  private static readonly UPPER_MASK = 0x80000000;
  private static readonly LOWER_MASK = 0x7fffffff;

  private mt = new Uint32Array(PythonRandom.N);
  private index = PythonRandom.N;

  constructor(seed: bigint) {
    this.seed(seed);
  }

  private seed(seed: bigint): void {
    let value = seed < 0n ? -seed : seed;
    const key: number[] = [];
    if (value === 0n) {
      key.push(0);
    } else {
      while (value > 0n) {
        key.push(Number(value & 0xffffffffn));
        value >>= 32n;
      }
    }
    this.initByArray(key);
  }

  private initGenrand(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < PythonRandom.N; i++) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, previous) + i) >>> 0;
    }
    this.index = PythonRandom.N;
  }

  private initByArray(key: number[]): void {
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(PythonRandom.N, key.length); k > 0; k--) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1664525)) + key[j] + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= PythonRandom.N) {
        this.mt[0] = this.mt[PythonRandom.N - 1];
        i = 1;
      }
      if (j >= key.length) {
        j = 0;
      }
    }
    for (let k = PythonRandom.N - 1; k > 0; k--) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= PythonRandom.N) {
        this.mt[0] = this.mt[PythonRandom.N - 1];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  private uint32(): number {
    if (this.index >= PythonRandom.N) {
      for (let kk = 0; kk < PythonRandom.N - PythonRandom.M; kk++) {
        const y = (this.mt[kk] & PythonRandom.UPPER_MASK) | (this.mt[kk + 1] & PythonRandom.LOWER_MASK);
        this.mt[kk] = (this.mt[kk + PythonRandom.M] ^ (y >>> 1) ^ ((y & 1) ? PythonRandom.MATRIX_A : 0)) >>> 0;
      }
      for (let kk = PythonRandom.N - PythonRandom.M; kk < PythonRandom.N - 1; kk++) {
        const y = (this.mt[kk] & PythonRandom.UPPER_MASK) | (this.mt[kk + 1] & PythonRandom.LOWER_MASK);
        this.mt[kk] = (this.mt[kk + PythonRandom.M - PythonRandom.N] ^ (y >>> 1) ^ ((y & 1) ? PythonRandom.MATRIX_A : 0)) >>> 0;
      }
      const y = (this.mt[PythonRandom.N - 1] & PythonRandom.UPPER_MASK) | (this.mt[0] & PythonRandom.LOWER_MASK);
      this.mt[PythonRandom.N - 1] = (this.mt[PythonRandom.M - 1] ^ (y >>> 1) ^ ((y & 1) ? PythonRandom.MATRIX_A : 0)) >>> 0;
      this.index = 0;
    }

    let y = this.mt[this.index];
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random(): number {
    const high = this.uint32() >>> 5;
    const low = this.uint32() >>> 6;
    return (high * 67108864 + low) / 9007199254740992;
  }

  private getrandbits(k: number): bigint {
    if (k <= 0) {
      return 0n;
    }
    let remaining = Math.trunc(k);
    let shift = 0n;
    let result = 0n;
    while (remaining > 0) {
      let word = this.uint32();
      if (remaining < 32) {
        word >>>= 32 - remaining;
      }
      result |= BigInt(word >>> 0) << shift;
      shift += 32n;
      remaining -= 32;
    }
    return result;
  }

  private randbelow(n: number): number {
    if (n <= 0) {
      throw new Error('randbelow requires n > 0');
    }
    const bits = Math.floor(Math.log2(n)) + 1;
    let value = Number(this.getrandbits(bits));
    while (value >= n) {
      value = Number(this.getrandbits(bits));
    }
    return value;
  }

  randint(min: number, max: number): number {
    return Math.trunc(min) + this.randbelow(Math.trunc(max) - Math.trunc(min) + 1);
  }

  choice<T>(items: T[]): T {
    return items[this.randbelow(items.length)];
  }

  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.randbelow(i + 1);
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  }
}

function mixBoardSeed(seed: number, boardSize: BoardSize, setupIndex = 0): bigint {
  const boardIndex = BOARD_SIZES.indexOf(boardSize);
  let value = BigInt(Math.trunc(seed)) & MASK_64;
  value ^= (BigInt(setupIndex + 1) * SETUP_SEED_MIX) & MASK_64;
  value ^= (BigInt(boardIndex + 1) * BOARD_SEED_MIX) & MASK_64;
  return value & MASK_64;
}

export function normalizeBoardSize(value: number): BoardSize {
  if (value <= 4) return 4;
  if (value >= 6) return 6;
  return 5;
}

interface DifficultySpec {
  pathLengths: number[];
  colorCounts: number[];
  shortestSteps: [number, number];
  minTurns: number;
  minRegions: number;
  fillFraction: [number, number];
}

function specForBoardSize(boardSize: BoardSize): DifficultySpec {
  if (boardSize === 4) {
    return {
      pathLengths: [10, 12, 14, 16],
      colorCounts: [3],
      shortestSteps: [9, 14],
      minTurns: 5,
      minRegions: 3,
      fillFraction: [0.6, 1.0],
    };
  }
  if (boardSize === 5) {
    return {
      pathLengths: [14, 16, 18, 20, 22],
      colorCounts: [3, 4, 5],
      shortestSteps: [11, 18],
      minTurns: 7,
      minRegions: 4,
      fillFraction: [0.6, 1.0],
    };
  }
  return {
    pathLengths: [20, 22, 24, 26, 28],
    colorCounts: [5, 6, 7],
    shortestSteps: [19, 20],
    minTurns: 8,
    minRegions: 5,
    fillFraction: [0.65, 1.0],
  };
}

function manhattan(left: Vertex, right: Vertex): number {
  return Math.abs(left.x - right.x) + Math.abs(left.y - right.y);
}

function neighbors(vertex: Vertex, W: number, H: number): Array<{ dir: Direction; vertex: Vertex }> {
  const result: Array<{ dir: Direction; vertex: Vertex }> = [];
  for (const dir of DIRS) {
    const delta = DIR_TO_DELTA[dir];
    const next = { x: vertex.x + delta.x, y: vertex.y + delta.y };
    if (next.x >= 0 && next.x < W && next.y >= 0 && next.y < H) {
      result.push({ dir, vertex: next });
    }
  }
  return result;
}

function vertexKey(vertex: Vertex): string {
  return `${vertex.x},${vertex.y}`;
}

function randomSimplePath(
  W: number,
  H: number,
  targetLength: number,
  rng: PythonRandom,
): Direction[] | null {
  const start = { x: 0, y: 0 };
  const goal = { x: W - 1, y: H - 1 };
  const visited = new Set<string>([vertexKey(start)]);
  const path: Direction[] = [];
  let exploredNodes = 0;

  const dfs = (current: Vertex, remaining: number): boolean => {
    exploredNodes += 1;
    if (exploredNodes > 50000) {
      return false;
    }

    const distance = manhattan(current, goal);
    if (distance > remaining || (remaining - distance) % 2 !== 0) {
      return false;
    }
    if (remaining === 0) {
      return current.x === goal.x && current.y === goal.y;
    }

    const choices = rng.shuffle(neighbors(current, W, H))
      .map((choice) => ({
        choice,
        late: manhattan(choice.vertex, goal) > remaining - 1 ? 1 : 0,
        tie: rng.random(),
      }))
      .sort((left, right) => {
        if (left.late !== right.late) {
          return left.late - right.late;
        }
        return left.tie - right.tie;
      })
      .map((item) => item.choice);

    for (const choice of choices) {
      const nextKey = vertexKey(choice.vertex);
      if (visited.has(nextKey)) {
        continue;
      }
      if (choice.vertex.x === goal.x && choice.vertex.y === goal.y && remaining !== 1) {
        continue;
      }
      const nextDistance = manhattan(choice.vertex, goal);
      if (nextDistance > remaining - 1 || (remaining - 1 - nextDistance) % 2 !== 0) {
        continue;
      }
      visited.add(nextKey);
      path.push(choice.dir);
      if (dfs(choice.vertex, remaining - 1)) {
        return true;
      }
      path.pop();
      visited.delete(nextKey);
    }
    return false;
  };

  return dfs(start, targetLength) ? [...path] : null;
}

function pathToEdges(directions: Direction[]): Set<string> {
  const current = { x: 0, y: 0 };
  const edges = new Set<string>();
  for (const direction of directions) {
    const delta = DIR_TO_DELTA[direction];
    const next = { x: current.x + delta.x, y: current.y + delta.y };
    edges.add(edgeKey(current, next));
    current.x = next.x;
    current.y = next.y;
  }
  return edges;
}

function turnCount(directions: Direction[]): number {
  let turns = 0;
  for (let index = 1; index < directions.length; index++) {
    if (directions[index] !== directions[index - 1]) {
      turns += 1;
    }
  }
  return turns;
}

function groupedRegions(W: number, H: number, edges: Set<string>): Map<number, Array<{ i: number; j: number }>> {
  const regions = calculateRegions(W, H, edges);
  const groups = new Map<number, Array<{ i: number; j: number }>>();
  for (let j = 0; j < regions.length; j++) {
    for (let i = 0; i < regions[j].length; i++) {
      const regionId = regions[j][i];
      if (!groups.has(regionId)) {
        groups.set(regionId, []);
      }
      groups.get(regionId)!.push({ i, j });
    }
  }
  return groups;
}

function buildCells(
  W: number,
  H: number,
  regionsById: Map<number, Array<{ i: number; j: number }>>,
  colorCounts: number[],
  fillFraction: [number, number],
  rng: PythonRandom,
): (string | null)[][] | null {
  const regionIds = [...regionsById.keys()].filter((regionId) => regionsById.get(regionId)!.length > 0);
  const possibleColorCounts = colorCounts.filter((count) => count <= regionIds.length && count <= COLORS.length);
  if (possibleColorCounts.length === 0) {
    return null;
  }

  const colorCount = rng.choice(possibleColorCounts);
  rng.shuffle(regionIds);
  const selectedRegions = regionIds.slice(0, colorCount);
  const colors = rng.shuffle([...COLORS]).slice(0, colorCount);
  const cells: (string | null)[][] = Array.from({ length: H - 1 }, () => Array(W - 1).fill(null));
  const [minFraction, maxFraction] = fillFraction;

  for (let index = 0; index < selectedRegions.length; index++) {
    const regionCells = [...regionsById.get(selectedRegions[index])!];
    rng.shuffle(regionCells);
    let count: number;
    if (regionCells.length <= 2) {
      count = regionCells.length;
    } else {
      const minCount = Math.max(1, Math.ceil(regionCells.length * minFraction));
      const maxCount = Math.max(minCount, Math.min(regionCells.length, Math.ceil(regionCells.length * maxFraction)));
      count = rng.randint(minCount, maxCount);
    }
    for (const cell of regionCells.slice(0, count)) {
      cells[cell.j][cell.i] = colors[index];
    }
  }

  return cells;
}

function colorCount(cells: (string | null)[][]): number {
  const colors = new Set<string>();
  for (const row of cells) {
    for (const cell of row) {
      if (cell !== null) {
        colors.add(cell);
      }
    }
  }
  return colors.size;
}

function constraintsSatisfied(
  W: number,
  H: number,
  cells: (string | null)[][],
  edges: Set<string>,
): boolean {
  const regions = calculateRegions(W, H, edges);
  const colorRegions = new Map<string, Set<number>>();
  const regionColors = new Map<number, Set<string>>();

  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const color = cells[j][i];
      if (color === null) {
        continue;
      }
      const regionId = regions[j][i];
      if (!colorRegions.has(color)) {
        colorRegions.set(color, new Set());
      }
      if (!regionColors.has(regionId)) {
        regionColors.set(regionId, new Set());
      }
      colorRegions.get(color)!.add(regionId);
      regionColors.get(regionId)!.add(color);
    }
  }

  return (
    [...colorRegions.values()].every((regionIds) => regionIds.size === 1)
    && [...regionColors.values()].every((colors) => colors.size <= 1)
  );
}

interface PathRecord {
  path: Direction[];
  regions: Uint8Array;
}

const pathRecordCache = new Map<string, PathRecord[]>();

function enumeratedPathRecords(W: number, H: number, maxDepth: number): PathRecord[] {
  const cacheKey = `${W}x${H}:${maxDepth}`;
  const cached = pathRecordCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const edgeIds = new Map<string, number>();
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const src = { x, y };
      for (const dir of DIRS) {
        const delta = DIR_TO_DELTA[dir];
        const dst = { x: x + delta.x, y: y + delta.y };
        if (dst.x < 0 || dst.x >= W || dst.y < 0 || dst.y >= H) {
          continue;
        }
        const key = edgeKey(src, dst);
        if (!edgeIds.has(key)) {
          edgeIds.set(key, edgeIds.size);
        }
      }
    }
  }

  const cellW = W - 1;
  const cellH = H - 1;
  const cellCount = cellW * cellH;
  const cellNeighbors: Array<Array<{ next: number; separator: bigint }>> = Array.from(
    { length: cellCount },
    () => [],
  );

  for (let j = 0; j < cellH; j++) {
    for (let i = 0; i < cellW; i++) {
      const srcCell = j * cellW + i;
      if (i + 1 < cellW) {
        const key = edgeKey({ x: i + 1, y: j }, { x: i + 1, y: j + 1 });
        const separator = 1n << BigInt(edgeIds.get(key)!);
        const dstCell = j * cellW + i + 1;
        cellNeighbors[srcCell].push({ next: dstCell, separator });
        cellNeighbors[dstCell].push({ next: srcCell, separator });
      }
      if (j + 1 < cellH) {
        const key = edgeKey({ x: i, y: j + 1 }, { x: i + 1, y: j + 1 });
        const separator = 1n << BigInt(edgeIds.get(key)!);
        const dstCell = (j + 1) * cellW + i;
        cellNeighbors[srcCell].push({ next: dstCell, separator });
        cellNeighbors[dstCell].push({ next: srcCell, separator });
      }
    }
  }

  const vertexId = (vertex: Vertex): number => vertex.y * W + vertex.x;
  const goal = { x: W - 1, y: H - 1 };
  const path: Direction[] = [];
  const records: PathRecord[] = [];

  const regionLabelsFor = (edgeMask: bigint): Uint8Array => {
    const labels = new Uint8Array(cellCount);
    labels.fill(255);
    let regionId = 0;
    for (let startCell = 0; startCell < cellCount; startCell++) {
      if (labels[startCell] !== 255) {
        continue;
      }
      labels[startCell] = regionId;
      const queue = [startCell];
      while (queue.length > 0) {
        const cellId = queue.pop()!;
        for (const neighbor of cellNeighbors[cellId]) {
          if ((edgeMask & neighbor.separator) !== 0n) {
            continue;
          }
          if (labels[neighbor.next] !== 255) {
            continue;
          }
          labels[neighbor.next] = regionId;
          queue.push(neighbor.next);
        }
      }
      regionId += 1;
    }
    return labels;
  };

  const dfs = (current: Vertex, vertexMask: bigint, edgeMask: bigint): void => {
    const depth = path.length;
    if (manhattan(current, goal) > maxDepth - depth) {
      return;
    }
    if (current.x === goal.x && current.y === goal.y) {
      records.push({ path: [...path], regions: regionLabelsFor(edgeMask) });
      return;
    }
    if (depth >= maxDepth) {
      return;
    }

    const choices = neighbors(current, W, H).sort((left, right) => {
      const leftDistance = manhattan(left.vertex, goal);
      const rightDistance = manhattan(right.vertex, goal);
      if (leftDistance !== rightDistance) {
        return leftDistance - rightDistance;
      }
      return DIRS.indexOf(left.dir) - DIRS.indexOf(right.dir);
    });

    for (const choice of choices) {
      const nextBit = 1n << BigInt(vertexId(choice.vertex));
      if ((vertexMask & nextBit) !== 0n) {
        continue;
      }
      const edgeId = edgeIds.get(edgeKey(current, choice.vertex))!;
      path.push(choice.dir);
      dfs(choice.vertex, vertexMask | nextBit, edgeMask | (1n << BigInt(edgeId)));
      path.pop();
    }
  };

  dfs({ x: 0, y: 0 }, 1n, 0n);
  records.sort((left, right) => {
    if (left.path.length !== right.path.length) {
      return left.path.length - right.path.length;
    }
    return left.path.join('').localeCompare(right.path.join(''));
  });
  pathRecordCache.set(cacheKey, records);
  return records;
}

function shortestSolution(
  W: number,
  H: number,
  cells: (string | null)[][],
  maxDepth: number,
): Direction[] | null {
  const colorIds = new Map<string, number>();
  const coloredCells: Array<{ cellId: number; colorId: number }> = [];
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const color = cells[j][i];
      if (color === null) {
        continue;
      }
      if (!colorIds.has(color)) {
        colorIds.set(color, colorIds.size);
      }
      coloredCells.push({ cellId: j * (W - 1) + i, colorId: colorIds.get(color)! });
    }
  }

  for (const record of enumeratedPathRecords(W, H, maxDepth)) {
    const colorRegion = Array(colorIds.size).fill(-1);
    const regionColor = Array((W - 1) * (H - 1)).fill(-1);
    let valid = true;
    for (const coloredCell of coloredCells) {
      const regionId = record.regions[coloredCell.cellId];
      const previousRegion = colorRegion[coloredCell.colorId];
      if (previousRegion === -1) {
        colorRegion[coloredCell.colorId] = regionId;
      } else if (previousRegion !== regionId) {
        valid = false;
        break;
      }

      const previousColor = regionColor[regionId];
      if (previousColor === -1) {
        regionColor[regionId] = coloredCell.colorId;
      } else if (previousColor !== coloredCell.colorId) {
        valid = false;
        break;
      }
    }
    if (valid) {
      return [...record.path];
    }
  }
  return null;
}

function targetLengthsForSize(boardSize: BoardSize, spec: DifficultySpec): number[] {
  const vertexSize = boardSize + 1;
  const minDistance = boardSize * 2;
  return spec.pathLengths.filter(
    (length) => minDistance <= length && length < vertexSize * vertexSize && (length - minDistance) % 2 === 0,
  );
}

export function generateOneStrokeLevel(boardSize: number, seed: number): LevelJson {
  const size = normalizeBoardSize(boardSize);
  const vertexSize = size + 1;
  const rng = new PythonRandom(mixBoardSeed(seed, size));
  const spec = specForBoardSize(size);
  const targetLengths = targetLengthsForSize(size, spec);
  const maxAttempts = 1000;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    rng.choice([size]);
    const targetLength = rng.choice(targetLengths);
    const constructionPath = randomSimplePath(vertexSize, vertexSize, targetLength, rng);
    if (constructionPath === null) {
      continue;
    }
    if (turnCount(constructionPath) < spec.minTurns) {
      continue;
    }

    const edges = pathToEdges(constructionPath);
    const regionsById = groupedRegions(vertexSize, vertexSize, edges);
    if (regionsById.size < spec.minRegions) {
      continue;
    }
    const cells = buildCells(vertexSize, vertexSize, regionsById, spec.colorCounts, spec.fillFraction, rng);
    if (cells === null) {
      continue;
    }
    if (!spec.colorCounts.includes(colorCount(cells))) {
      continue;
    }
    if (!constraintsSatisfied(vertexSize, vertexSize, cells, edges)) {
      continue;
    }

    const solution = shortestSolution(vertexSize, vertexSize, cells, spec.shortestSteps[1]);
    if (solution === null) {
      continue;
    }
    if (solution.length < spec.shortestSteps[0] || solution.length > spec.shortestSteps[1]) {
      continue;
    }

    return {
      W: vertexSize,
      H: vertexSize,
      cells,
      solution,
    };
  }

  throw new Error(`Could not generate a solvable ${size}x${size} setup for seed ${seed}.`);
}
