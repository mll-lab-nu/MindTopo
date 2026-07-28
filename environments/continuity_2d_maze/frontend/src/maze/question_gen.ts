// Q2 "bar_removal" sample generator (iter D2). Builds a complete
// SceneMetadata whose target_pair is guaranteed disconnected in the base
// maze, plus a set of colored bars where at least one bar — removed alone —
// restores connectivity. The ground-truth answer (`correct_removals`) is the
// full set of colors whose single removal connects the pair.
//
// Pure frontend TS; not wired into main.ts yet (that happens in iter D4).

import {
  allPairsWithBars,
  isConnectedWithBars,
  placeRandomBars,
  withBarsRemoved,
} from './bars';
import { isConnected } from './connectivity';
import { generateMaze } from './generator';
import {
  bfsCellDistance,
  computeReachableAreas,
  DifficultyTier,
} from './metric';
import { samplePoints } from './points';
import {
  Bar,
  BarColor,
  DifficultyMetric,
  LabeledPoint,
  MazeGrid,
  Pair,
  SceneMetadata,
} from './types';

export const Q2_DEFAULT_WALL_DENSITY = 0.5;
export const Q2_DEFAULT_BAR_COUNT = 4;
export const Q2_DEFAULT_K_MAZE = 5;
export const Q2_DEFAULT_K_BAR = 10;
export const Q2_DEFAULT_K_POINTS = 20;
export const Q2_DEFAULT_CANVAS_SIZE = 1024;

export interface BarRemovalSceneConfig {
  seed: number;
  grid_size: number;
  wall_density?: number;
  bar_count?: number;
  canvas_size?: number;
  diagonal_ratio?: number;
  partial_ratio?: number;
  partial_bc_split?: number;
  k_maze?: number;
  k_bar?: number;
  k_points?: number;
  // iter D15: tier label for diagnostic display only. No score is
  // computed; whatever the caller passes here lands in `meta.difficulty.tier`.
  difficulty_tier?: DifficultyTier;
  // iter D16: floor on |correct_removals|. Tier-mode sets this to 2 for
  // medium/hard so single-answer scenes (which feel trivial — just spot
  // the one bar around the small pocket) are rejected.
  min_correct_removals?: number;
  // iter D17: minimum Manhattan distance between A and B. Tier-mode
  // sets this to 3 (medium) / 5 (hard); 0 = unconstrained.
  min_pairwise_distance?: number;
  // iter D17 hard rule: minimum fraction (in 4-triangle units) for the
  // smaller of (area_A, area_B). 1/3 means BOTH points reach at least
  // 1/3 of the maze, so neither is trapped in a tiny pocket.
  min_area_fraction?: number;
}

// Repeatedly sample 2 points via samplePoints() (which already assigns names
// 'A' and 'B') and accept the first pair that is base-disconnected. The seed
// varies per attempt so we walk the sample space rather than retrying the
// same draw.
export function sampleTwoDisconnectedPoints(
  maze: MazeGrid,
  seed: number | string,
  maxTries: number = Q2_DEFAULT_K_POINTS,
  minDistance: number = 0,
): [LabeledPoint, LabeledPoint] {
  for (let i = 0; i < maxTries; i++) {
    const pts = samplePoints(maze, hashStringToInt(`${seed}:t${i}`), 2);
    if (pts.length < 2) continue;
    const [p1, p2] = pts;
    if (minDistance > 0) {
      const d = Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
      if (d < minDistance) continue;
    }
    if (!isConnected(maze, p1, p2)) return [p1, p2];
  }
  throw new Error(
    `sampleTwoDisconnectedPoints: no disconnected pair after ${maxTries} tries`,
  );
}

// samplePoints expects a numeric seed; convert a composite string seed into
// a deterministic 32-bit integer so we can vary attempts cheaply.
function hashStringToInt(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function generateBarRemovalScene(
  config: BarRemovalSceneConfig,
): SceneMetadata {
  const wall_density = config.wall_density ?? Q2_DEFAULT_WALL_DENSITY;
  const bar_count = config.bar_count ?? Q2_DEFAULT_BAR_COUNT;
  const canvas_size = config.canvas_size ?? Q2_DEFAULT_CANVAS_SIZE;
  const diagonal_ratio = config.diagonal_ratio ?? 0;
  const partial_ratio = config.partial_ratio ?? 0;
  const partial_bc_split = config.partial_bc_split ?? 0.5;
  const k_maze = config.k_maze ?? Q2_DEFAULT_K_MAZE;
  const k_bar = config.k_bar ?? Q2_DEFAULT_K_BAR;
  const k_points = config.k_points ?? Q2_DEFAULT_K_POINTS;
  const min_correct_removals = Math.max(1, config.min_correct_removals ?? 1);
  const min_pairwise_distance = Math.max(0, config.min_pairwise_distance ?? 0);
  const min_area_fraction = Math.max(0, config.min_area_fraction ?? 0);

  for (let ms = 0; ms < k_maze; ms++) {
    const mazeSeed = hashStringToInt(`${config.seed}:maze:${ms}`);
    const maze = generateMaze(
      config.grid_size,
      mazeSeed,
      wall_density,
      diagonal_ratio,
      partial_ratio,
      partial_bc_split,
    );

    let pair: [LabeledPoint, LabeledPoint];
    try {
      pair = sampleTwoDisconnectedPoints(
        maze,
        `${config.seed}:pts:${ms}`,
        k_points,
        min_pairwise_distance,
      );
    } catch {
      continue;
    }
    const [a, b] = pair;

    // iter D17 hard rule: reject mazes where one of A, B is trapped in a
    // small pocket (min reachable area < min_area_fraction · total). This
    // depends only on the base maze + a/b — no need to re-check per bar
    // attempt. Easy/medium pass min_area_fraction=0 ⇒ no-op.
    if (min_area_fraction > 0) {
      const { area_a, area_b, total } = computeReachableAreas(maze, a, b);
      const minArea = Math.min(area_a, area_b);
      if (total > 0 && minArea / total < min_area_fraction) {
        continue;
      }
    }

    for (let bs = 0; bs < k_bar; bs++) {
      let bars: Bar[];
      try {
        bars = placeRandomBars(maze, `${config.seed}:bars:${ms}:${bs}`, bar_count);
      } catch {
        // Pool smaller than bar_count — same maze → same pool; reroll maze.
        break;
      }

      const correct_removals: BarColor[] = [];
      const correctBars: Bar[] = [];
      for (const bar of bars) {
        if (isConnectedWithBars(maze, [bar], a, b)) {
          correct_removals.push(bar.color);
          correctBars.push(bar);
        }
      }
      if (correct_removals.length < min_correct_removals) continue;
      correct_removals.sort();

      // Q2 diagnostics (iter D15: no score / no threshold).
      //   area_a / area_b = #triangles reachable from each point by BFS
      //     through open corridors (Mode-A walls + dA diagonals block;
      //     partial walls + dB/dC do not).
      //   min_path = min over correct_removals r of d_bfs(A, B | remove r)
      // tier is whatever the caller passed; downstream UI/CLI just attach
      // it to the scene as a label.
      const { area_a, area_b } = computeReachableAreas(maze, a, b);
      const minArea = Math.min(area_a, area_b);

      const manhattan = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
      let minPath = Number.POSITIVE_INFINITY;
      let detourSum = 0;
      for (const bar of correctBars) {
        const opened = withBarsRemoved(maze, [bar]);
        const d_bfs = bfsCellDistance(opened, a, b);
        if (!Number.isFinite(d_bfs)) continue; // shouldn't happen — bar is a correct_removal
        if (d_bfs < minPath) minPath = d_bfs;
        detourSum += Math.max(0, d_bfs - manhattan);
      }
      const avg_detour = detourSum / correctBars.length;
      const difficulty: DifficultyMetric = {
        scene_score: 0,                  // unused (iter D15)
        detour_total: avg_detour,        // diagnostic only
        iso_total: minArea,              // diagnostic only (= min_area)
        min_path: minPath,
        area_a,
        area_b,
        per_pair: [
          { other_name: b.name, connected: false, iso: minArea },
        ],
        tier: config.difficulty_tier,
      };

      const pairs: Pair[] = [
        { a_name: a.name, b_name: b.name, connected: false },
      ];

      // Defensive: allPairsWithBars([], ...) with no removals should agree.
      // Used here purely as a post-condition check; cheap for 2 points.
      const crossCheck = allPairsWithBars([a, b], maze, []);
      if (crossCheck[0].connected) {
        throw new Error(
          `generateBarRemovalScene: invariant broken — post-check says pair is connected (seed=${config.seed})`,
        );
      }

      return {
        seed: config.seed,
        grid_size: config.grid_size,
        wall_density,
        point_count: 2,
        maze,
        points: [a, b],
        pairs,
        canvas_size,
        difficulty,
        question_type: 'bar_removal',
        bars,
        question: {
          target_pair: [a.name, b.name],
          correct_removals,
        },
      };
    }
    // Inner bar loop exhausted (or pool too small) → try next maze seed.
  }

  throw new Error(
    `generateBarRemovalScene: all ${k_maze}×${k_bar} retries failed (seed=${config.seed})`,
  );
}
