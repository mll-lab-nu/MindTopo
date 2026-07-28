import seedrandom from 'seedrandom';
import { LabeledPoint, MazeGrid } from './types';

const MIN_DIST_RETRIES = 50;

export function samplePoints(
  maze: MazeGrid,
  seed: number,
  count: number,
  minPairwiseDistance: number = 0,
): LabeledPoint[] {
  if (count < 1) {
    return [];
  }
  const total = maze.size * maze.size;
  if (count > total) {
    throw new Error(`point_count (${count}) exceeds total cells (${total})`);
  }
  if (count > 26) {
    throw new Error(`point_count (${count}) exceeds 26 letter labels (A..Z)`);
  }

  // iter D2: exclude cells that contain any diagonal wall. Under iter D1's
  // fixed partial length (0.5·√2), every diagonal mode (dA/dB/dC) geometrically
  // reaches the cell center — so a point circle at center would visually
  // overlap the diagonal. We keep only diagonal-free cells.
  const validIndices: number[] = [];
  for (let y = 0; y < maze.size; y++) {
    for (let x = 0; x < maze.size; x++) {
      if (maze.diagonals[`${x},${y}`] === undefined) {
        validIndices.push(y * maze.size + x);
      }
    }
  }
  if (validIndices.length < count) {
    throw new Error(
      `Not enough diagonal-free cells for point sampling: got ${validIndices.length}, ` +
        `need ${count}. Lower diagonal_ratio, reduce point_count, or increase grid_size.`,
    );
  }

  // iter D16: when minPairwiseDistance > 0, retry the shuffle with a
  // different sub-seed until every pair of selected points has Manhattan
  // distance ≥ minPairwiseDistance. Falls through to the last attempt
  // after MIN_DIST_RETRIES tries (caller decides whether to retry the
  // whole scene).
  const tryShuffle = (subSeed: string): LabeledPoint[] => {
    const indices = validIndices.slice();
    const rng = seedrandom(subSeed);
    for (let i = 0; i < count; i++) {
      const j = i + Math.floor(rng() * (indices.length - i));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices.slice(0, count).map((idx, i) => ({
      name: String.fromCharCode(65 + i),
      x: idx % maze.size,
      y: Math.floor(idx / maze.size),
    }));
  };

  if (minPairwiseDistance <= 0) {
    return tryShuffle(`${seed}:points`);
  }
  for (let r = 0; r < MIN_DIST_RETRIES; r++) {
    const pts = tryShuffle(`${seed}:points:r${r}`);
    let ok = true;
    for (let i = 0; i < pts.length && ok; i++) {
      for (let j = i + 1; j < pts.length && ok; j++) {
        const dx = Math.abs(pts[i].x - pts[j].x);
        const dy = Math.abs(pts[i].y - pts[j].y);
        if (dx + dy < minPairwiseDistance) ok = false;
      }
    }
    if (ok) return pts;
  }
  // Throw so the tier-mode caller can re-roll the maze seed; this beats
  // silently returning a too-close set and getting wrong tier semantics.
  throw new Error(
    `samplePoints: could not satisfy minPairwiseDistance=${minPairwiseDistance} ` +
      `for ${count} points after ${MIN_DIST_RETRIES} retries`,
  );
}
