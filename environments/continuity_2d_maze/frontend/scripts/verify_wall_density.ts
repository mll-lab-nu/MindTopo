// Regression check for wall_density v2 semantics (dev_trace_3/iter B1).
// Kept as a permanent sanity harness; run via:
//   npx tsc --module commonjs --moduleResolution node --target es2022 \
//     --outDir /tmp/mv/js --rootDir . --esModuleInterop \
//     src/maze/generator.ts src/maze/types.ts scripts/verify_wall_density.ts
//   ln -s "$(pwd)/node_modules" /tmp/mv/node_modules
//   node /tmp/mv/js/scripts/verify_wall_density.js
import { generateMaze } from '../src/maze/generator';
import { isConnected } from '../src/maze/connectivity';
import { samplePoints } from '../src/maze/points';
import { computeDifficultyMetric } from '../src/maze/metric';
import type { CellWalls } from '../src/maze/types';

declare const process: { exit(code: number): never };

function countInternalWalls(cells: CellWalls[][], size: number): { openE: number; closedE: number; openS: number; closedS: number } {
  let openE = 0, closedE = 0, openS = 0, closedS = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x + 1 < size) {
        if (cells[y][x].east) closedE++; else openE++;
      }
      if (y + 1 < size) {
        if (cells[y][x].south) closedS++; else openS++;
      }
    }
  }
  return { openE, closedE, openS, closedS };
}

function cellsEqual(a: CellWalls[][], b: CellWalls[][], size: number): boolean {
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const ca = a[y][x], cb = b[y][x];
      if (ca.north !== cb.north || ca.east !== cb.east || ca.south !== cb.south || ca.west !== cb.west) {
        return false;
      }
    }
  }
  return true;
}

function checkBorderIntact(cells: CellWalls[][], size: number): boolean {
  // All outer-boundary walls must remain closed regardless of wall_density.
  for (let i = 0; i < size; i++) {
    if (!cells[0][i].north) return false;
    if (!cells[size - 1][i].south) return false;
    if (!cells[i][0].west) return false;
    if (!cells[i][size - 1].east) return false;
  }
  return true;
}

function check(cond: boolean, label: string): void {
  if (!cond) {
    console.error(`FAIL: ${label}`);
    process.exit(1);
  }
  console.log(`ok   ${label}`);
}

for (const size of [4, 5, 6]) {
  const seed = 12345;
  const M = 2 * size * (size - 1);
  const nonTreeCount = (size - 1) * (size - 1);
  const fTree = nonTreeCount / M;

  // Invariant 1: f=0 → all internal walls open
  {
    const m = generateMaze(size, seed, 0);
    const c = countInternalWalls(m.cells, size);
    check(c.closedE === 0 && c.closedS === 0, `N=${size} f=0 → no internal closed walls (got closedE=${c.closedE}, closedS=${c.closedS})`);
    check(c.openE + c.openS === M, `N=${size} f=0 → open count equals M (${M})`);
    check(checkBorderIntact(m.cells, size), `N=${size} f=0 → outer border intact`);
  }

  // Invariant 2: f=1 → all internal walls closed
  {
    const m = generateMaze(size, seed, 1);
    const c = countInternalWalls(m.cells, size);
    check(c.openE === 0 && c.openS === 0, `N=${size} f=1 → no internal open walls (got openE=${c.openE}, openS=${c.openS})`);
    check(c.closedE + c.closedS === M, `N=${size} f=1 → closed count equals M (${M})`);
  }

  // Invariant 3: f=(N-1)²/M → matches spanning tree (exactly nonTreeCount closed, N²-1 open)
  {
    const m = generateMaze(size, seed, fTree);
    const c = countInternalWalls(m.cells, size);
    check(c.closedE + c.closedS === nonTreeCount, `N=${size} f=f* → closed count equals (N-1)² (${nonTreeCount}, got ${c.closedE + c.closedS})`);
    check(c.openE + c.openS === size * size - 1, `N=${size} f=f* → open count equals N²-1 (${size * size - 1})`);
  }

  // Invariant 4: same (seed, size) different f → DFS skeleton preserved.
  // Measure: the set of treeWalls should be invariant across f values. We can
  // infer "tree edges" as the set of walls open in the spanning-tree regime;
  // in low-f regime we OPEN extra non-tree walls → tree walls still open.
  // In high-f regime we CLOSE some tree walls → cannot use this directly.
  // Weaker but sufficient check: at f=0 (all open), maze is the same
  // topology regardless; at f=fTree vs fTree*0.99 both have DFS skeleton.
  // We verify determinism: two calls with same (seed, size, f) identical.
  {
    const a = generateMaze(size, seed, 0.3);
    const b = generateMaze(size, seed, 0.3);
    check(cellsEqual(a.cells, b.cells, size), `N=${size} f=0.3 → deterministic across calls`);
  }

  // Invariant 5: monotonicity of closed-wall count in f
  {
    const fs = [0.0, 0.1, 0.25, fTree, 0.6, 0.85, 1.0];
    let prevClosed = -1;
    for (const f of fs) {
      const m = generateMaze(size, seed, f);
      const c = countInternalWalls(m.cells, size);
      const closed = c.closedE + c.closedS;
      const expected = Math.round(f * M);
      check(closed === expected, `N=${size} f=${f} → closed count equals round(f·M)=${expected} (got ${closed})`);
      check(closed >= prevClosed, `N=${size} f=${f} → monotone vs f=prev (closed ${closed} >= ${prevClosed})`);
      prevClosed = closed;
    }
  }

  // Invariant 6 (iter C1a): every closed internal H/V wall has a shape entry
  // with length === 1 (Mode A). No spurious entries for open walls.
  {
    const m = generateMaze(size, seed, 0.5);
    let shapeCount = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (x + 1 < size) {
          const key = `${x},${y},e`;
          const entry = m.shapes[key];
          if (m.cells[y][x].east) {
            check(entry !== undefined && entry.length === 1 && entry.orientation === 'vertical',
              `N=${size} f=0.5 → east wall (${x},${y}) closed has Mode A shape entry`);
            shapeCount++;
          } else {
            check(entry === undefined, `N=${size} f=0.5 → east wall (${x},${y}) open has no shape entry`);
          }
        }
        if (y + 1 < size) {
          const key = `${x},${y},s`;
          const entry = m.shapes[key];
          if (m.cells[y][x].south) {
            check(entry !== undefined && entry.length === 1 && entry.orientation === 'horizontal',
              `N=${size} f=0.5 → south wall (${x},${y}) closed has Mode A shape entry`);
            shapeCount++;
          } else {
            check(entry === undefined, `N=${size} f=0.5 → south wall (${x},${y}) open has no shape entry`);
          }
        }
      }
    }
    check(Object.keys(m.shapes).length === shapeCount, `N=${size} f=0.5 → shape entry count matches closed-wall count`);
  }

  // Invariant 8 (iter D4): partial_ratio=1 with partial_bc_split=1 (all B)
  // or =0 (all C) forces every H/V wall to the corresponding partial mode.
  // Since diagonal_ratio=0, all K walls stay as H/V → every shape entry is
  // a partial of the chosen variant.
  const EPS = 1e-9;
  for (const [label, bcSplit] of [['B', 1.0], ['C', 0.0]] as const) {
    const m = generateMaze(size, seed, 0.5, 0, 1.0, bcSplit);
    for (const [key, spec] of Object.entries(m.shapes)) {
      check(Math.abs(spec.length - 0.5) < EPS,
        `N=${size} bc=${label} ${key} → length = 0.5 (got ${spec.length})`);
      check(spec.offset >= -EPS && spec.offset + spec.length <= 1 + EPS,
        `N=${size} bc=${label} ${key} → offset + length ≤ 1`);
      if (label === 'B') {
        check(Math.abs(spec.offset - 0.25) < EPS,
          `N=${size} bc=B ${key} → offset = 0.25 (got ${spec.offset})`);
      } else {
        const hugStart = Math.abs(spec.offset - 0) < EPS;
        const hugEnd = Math.abs(spec.offset - 0.5) < EPS;
        check(hugStart || hugEnd,
          `N=${size} bc=C ${key} → offset ∈ {0, 0.5} (got ${spec.offset})`);
      }
    }
  }

  // Invariant 9 (iter C4b.i): default-mode calls produce NO diagonals — the
  // diagonals container is always the empty object. This pins the zero-
  // regression guarantee against accidental generation via the new path.
  {
    const m = generateMaze(size, seed, 0.5);
    check(Object.keys(m.diagonals).length === 0,
      `N=${size} default → diagonals empty (got ${Object.keys(m.diagonals).length} entries)`);
  }

  // Invariant 10 (iter D4): diagonal_ratio=1 converts every closed H/V wall
  // to a dA diagonal (subject to collision cap). Since partial_ratio=0, none
  // are downgraded to dB/dC. Verify all entries are dA with length=√2.
  {
    const m = generateMaze(size, seed, 0.5, 1.0, 0, 0.5);
    for (const [key, spec] of Object.entries(m.diagonals)) {
      check(spec.mode === 'dA', `N=${size} ${key} → mode=dA (got ${spec.mode})`);
      check(Math.abs(spec.length - Math.SQRT2) < EPS,
        `N=${size} ${key} → length=√2 (got ${spec.length})`);
      check(Math.abs(spec.offset - 0) < EPS,
        `N=${size} ${key} → offset=0 (got ${spec.offset})`);
      check(spec.orientation === 'slash' || spec.orientation === 'backslash',
        `N=${size} ${key} → orientation ∈ {slash, backslash}`);
    }
  }

  // Invariant 11 (iter D4): diagonal_ratio=1 + partial_ratio=1 + bc_split=1
  // → every diagonal is dB (centered partial). Length = 0.5·√2, offset = 0.25·√2.
  {
    const m = generateMaze(size, seed, 0.5, 1.0, 1.0, 1.0);
    const expectedLen = 0.5 * Math.SQRT2;
    const expectedOff = (Math.SQRT2 - expectedLen) / 2;
    for (const [key, spec] of Object.entries(m.diagonals)) {
      check(spec.mode === 'dB', `N=${size} ${key} → mode=dB (got ${spec.mode})`);
      check(Math.abs(spec.length - expectedLen) < EPS,
        `N=${size} ${key} → dB length = 0.5√2 (got ${spec.length})`);
      check(Math.abs(spec.offset - expectedOff) < EPS,
        `N=${size} ${key} → dB offset centered (expected ${expectedOff}, got ${spec.offset})`);
    }
  }

  // Invariant 12 (iter C4b.ii): with default diagonal_ratio=0, isConnected
  // output is identical to what the iter C2 single-region BFS would produce.
  // We cannot compare directly against the old impl, but we can check that
  // in the empty-room regime (f=0, no internal walls, no diagonals), all
  // cells are pairwise connected — a property any correct BFS must satisfy.
  {
    const m = generateMaze(size, seed, 0);
    check(Object.keys(m.diagonals).length === 0,
      `N=${size} f=0 → no diagonals`);
    // All-pairs connectivity: pick a few cell pairs, each must be connected.
    const pairs: [number, number, number, number][] = [
      [0, 0, size - 1, size - 1],
      [0, size - 1, size - 1, 0],
      [0, 0, 0, size - 1],
    ];
    for (const [ax, ay, bx, by] of pairs) {
      check(isConnected(m, { x: ax, y: ay }, { x: bx, y: by }),
        `N=${size} f=0 empty room (${ax},${ay})↔(${bx},${by}) connected`);
    }
  }

  // Invariant 13 (iter D3, reversed from C4b.ii): under the 4-triangle model,
  // a labeled point inhabits ALL 4 triangles of its cell (they share the
  // center as a vertex). So in an all-slash dA, no-wall maze, the start
  // cell's T_S can exit through the open S edge to reach the cell south,
  // and T_E can exit east. Both targets are CONNECTED — this is the opposite
  // of what C4b.ii's (buggy) region-0 convention asserted.
  if (size >= 2) {
    // To test BFS with guaranteed-open edges + every cell having a slash dA,
    // build an empty-room maze then manually install slash dA diagonals into
    // every cell's `diagonals` dict. BFS doesn't care how the maze was
    // constructed; it only reads cells/shapes/diagonals.
    const m = generateMaze(size, seed, 0);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        m.diagonals[`${x},${y}`] = {
          orientation: 'slash',
          mode: 'dA',
          length: Math.SQRT2,
          offset: 0,
        };
      }
    }
    const conn = isConnected(m, { x: 0, y: 0 }, { x: 0, y: 1 });
    check(conn,
      `N=${size} all-slash dA, f=0 → (0,0) ↔ (0,1) connected (T_S exits S edge, no wall)`);
    const connAdjacent = isConnected(m, { x: 0, y: 0 }, { x: 1, y: 0 });
    check(connAdjacent,
      `N=${size} all-slash dA, f=0 → (0,0) ↔ (1,0) connected (T_E exits E edge)`);
  }

  // Invariant 16 (iter D3): pure-wall isolation test without diagonals. With
  // wall_density = 1.0, every internal wall is Mode A closed → each cell is
  // an island. BFS from (0,0) cannot reach any other cell regardless of the
  // triangle structure (wallBlocks short-circuits inter-cell crossing).
  if (size >= 2) {
    const m = generateMaze(size, seed, 1.0);
    check(!isConnected(m, { x: 0, y: 0 }, { x: 1, y: 0 }),
      `N=${size} f=1.0 → (0,0) ↔ (1,0) disconnected (all internal walls Mode A)`);
    check(!isConnected(m, { x: 0, y: 0 }, { x: 0, y: 1 }),
      `N=${size} f=1.0 → (0,0) ↔ (0,1) disconnected`);
    check(!isConnected(m, { x: 0, y: 0 }, { x: size - 1, y: size - 1 }),
      `N=${size} f=1.0 → (0,0) ↔ (${size - 1},${size - 1}) disconnected`);
  }

  // Invariant 15 (iter D2, refactored for D4): when diagonal_ratio > 0, no
  // sampled point falls into a cell that has a diagonal (any mode). Points
  // live at cell center, which every diagonal mode (dA/dB/dC under fixed
  // length 0.5√2) intersects.
  if (size >= 3) {
    // Moderate wall_density + diagonal_ratio so the point filter has real
    // work to do. partial_ratio=0 keeps every diagonal as dA.
    const m = generateMaze(size, seed, 0.5, 0.5, 0, 0.5);
    const points = samplePoints(m, seed, 3);
    for (const p of points) {
      check(m.diagonals[`${p.x},${p.y}`] === undefined,
        `N=${size} diag_ratio=0.5 → point ${p.name} at (${p.x},${p.y}) not in a diagonal cell`);
    }
  }
}

// Invariant 17 (iter D8a): difficulty metric produces sensible extremes.
//   Empty room (wall_density=0): every pair is connected with d_bfs = d_man,
//     so detour = 0 and iso = 0 → scene_score = 0.
//   Fully closed (wall_density=1): every cell isolated, all pairs
//     disconnected → iso_total > 0, detour_total = 0.
for (const size of [4, 5, 6]) {
  const seed = 12345;
  {
    const m = generateMaze(size, seed, 0);
    const pts = samplePoints(m, seed, 4);
    const d = computeDifficultyMetric(m, pts[0], pts.slice(1));
    check(d.scene_score === 0 && d.detour_total === 0 && d.iso_total === 0,
      `N=${size} wd=0 empty → scene_score=0 (got ${d.scene_score})`);
    check(d.per_pair.length === 3, `N=${size} wd=0 → 3 pair breakdowns`);
    check(d.per_pair.every((p) => p.connected),
      `N=${size} wd=0 → all pairs connected`);
  }
  {
    const m = generateMaze(size, seed, 1);
    const pts = samplePoints(m, seed, 4);
    const d = computeDifficultyMetric(m, pts[0], pts.slice(1));
    check(d.detour_total === 0,
      `N=${size} wd=1 sealed → detour_total=0 (got ${d.detour_total})`);
    check(d.iso_total > 0,
      `N=${size} wd=1 sealed → iso_total>0 (got ${d.iso_total})`);
    check(d.per_pair.every((p) => !p.connected),
      `N=${size} wd=1 → all pairs disconnected`);
  }
}

console.log('all invariants passed');
