export interface MazeConfig {
  seed: number;
  grid_size: number;
  wall_density?: number;
  point_count?: number;
  // iter D4 unified pipeline knobs (all optional, default 0 except bc_split).
  // diagonal_ratio: of the K closed H/V walls produced by wall_density, this
  //   fraction are converted to dA diagonal walls (relocated into an
  //   adjacent cell).
  // partial_ratio: of the remaining K walls (both H/V A and diagonal dA),
  //   this fraction are converted to partial walls (H/V A→B/C; dA→dB/dC).
  // partial_bc_split: when converting to partial, fraction going to the
  //   "centered-segment" variant (B for H/V, dB for diagonal). Remainder goes
  //   to the "half-wall" variant (C, dC). Defaults to 0.5.
  diagonal_ratio?: number;
  partial_ratio?: number;
  partial_bc_split?: number;
  // Q2 bar_removal plumbing (iter D4). Q1 paths ignore these; main.ts dispatches
  // to runGenerateQ2 when question_type === 'bar_removal'.
  question_type?: QuestionType;
  bar_count?: number;
  // iter D16 extra rules:
  //   min_pairwise_distance: Q1 only — minimum Manhattan distance between any
  //     two sampled points (medium/hard set this to 3 to avoid trivially
  //     close pairs). Q2 rejects scenes that don't satisfy.
  //   min_correct_removals: Q2 only — reject scenes whose correct_removals
  //     count is below this floor (medium/hard set to 2 for ≥2 valid bars).
  min_pairwise_distance?: number;
  min_correct_removals?: number;
}

export interface CellWalls {
  north: boolean;
  east: boolean;
  south: boolean;
  west: boolean;
}

export type WallOrientation = 'horizontal' | 'vertical';

// Visual shape specification for an internal wall. Populated per closed wall.
// length ∈ (0, 1] (relative to cell edge); offset ∈ [0, 1 - length] is the
// start position along the edge. length === 1 means Mode A (full-edge, blocks
// passage). length < 1 means Mode B/C (partial, does not block passage).
export interface WallSpec {
  orientation: WallOrientation;
  length: number;
  offset: number;
}

// H/V wall shape mode produced by the D4 pipeline:
//   A: full-edge, blocks passage (iter C2 BFS treats as wall)
//   B: centered-segment partial (length=0.5, offset=0.25)
//   C: half-wall partial hugging one endpoint (length=0.5, offset=0 or 0.5)
export type WallMode = 'A' | 'B' | 'C';

// Canonical key for an internal wall.
//   `${x},${y},e` = east-side wall between (x, y) and (x+1, y)
//   `${x},${y},s` = south-side wall between (x, y) and (x, y+1)
export type WallKey = string;

// Diagonal wall orientation within a cell:
//   'slash'     = '/' diagonal, from bottom-left to top-right
//   'backslash' = '\' diagonal, from top-left to bottom-right
export type DiagonalOrientation = 'slash' | 'backslash';

// Diagonal wall shape mode. Connectivity-wise only 'dA' (full length √2) can
// block passage via triangle-splitting. 'dB' and 'dC' are partial diagonals
// that do not block.
//   dA: full corner-to-corner (length = √2, blocks conditionally)
//   dB: centered-segment partial (length = 0.5·√2, centered)
//   dC: half-wall partial hugging one corner (length = 0.5·√2)
export type DiagonalMode = 'dA' | 'dB' | 'dC';

export interface DiagonalSpec {
  orientation: DiagonalOrientation;
  mode: DiagonalMode;
  length: number;
  offset: number;
}

// Canonical key for a diagonal: `${x},${y}` (at most one diagonal per cell).
export type DiagonalKey = string;

export interface MazeGrid {
  size: number;
  cells: CellWalls[][];
  shapes: Record<WallKey, WallSpec>;
  diagonals: Record<DiagonalKey, DiagonalSpec>;
}

export interface LabeledPoint {
  name: string;
  x: number;
  y: number;
}

export interface Pair {
  a_name: string;
  b_name: string;
  connected: boolean;
}

// Difficulty metric produced by iter D8a. Decouples benchmark difficulty from
// generation parameters (grid_size/wall_density/diagonal_ratio/partial_ratio).
// Q2 additions:
//   iter D10: `min_path` — shortest A→B distance over all correct_removals
//   iter D12: `area_a` / `area_b` — reachable 4-triangle count from each
//     point; tier classification uses area_ratio and area_min_fraction
//     (both derived from these). `iso_total` for Q2 is repurposed to
//     min(area_a, area_b) for backward-compat with existing consumers.
// Q1 scenes leave all D10/D12 fields undefined.
export interface DifficultyMetric {
  scene_score: number;
  detour_total: number;
  iso_total: number;
  min_path?: number;
  area_a?: number;
  area_b?: number;
  per_pair: Array<{
    other_name: string;
    connected: boolean;
    detour?: number;
    iso?: number;
  }>;
  tier?: 'easy' | 'medium' | 'hard';
}

export type BarColor = 'purple' | 'red' | 'green' | 'blue' | 'yellow' | 'orange';

// Bar is a recolored reference to an EXISTING blocking wall on the maze.
// Two kinds (D7):
//   * 'hv'   — recolors an H/V Mode-A wall; `edge` is a WallKey of the
//              form '${x},${y},e' or '${x},${y},s'. Removing it clears
//              cells[y][x][side] (and the mirrored neighbor side) and
//              drops maze.shapes[edge].
//   * 'diag' — recolors a full dA diagonal inside cell (x, y); `edge` is
//              a DiagonalKey '${x},${y}' (2-part). Removing it deletes
//              maze.diagonals[edge] so intra-cell BFS no longer sees the
//              dA blocker.
// `shape` is reserved for a future "partial visual bar" iter; currently unused.
export type BarEdgeKind = 'hv' | 'diag';

export interface Bar {
  id: string;
  color: BarColor;
  kind: BarEdgeKind;
  edge: string;
  shape?: WallSpec;
}

export type QuestionType = 'connectivity' | 'bar_removal';

// Populated by iter D2 generator. Q2 premise: exactly 2 labeled points
// (target_pair) that are guaranteed disconnected in the base maze. The
// task: find every bar whose SINGLE removal restores connectivity.
//
// Answer format mirrors Q1's reachability list — the model emits the full
// set of qualifying bar identifiers (by color name). `correct_removals` is
// the ground-truth set; the D2 generator guarantees it is non-empty.
// Evaluation is set-equality (order-insensitive).
export interface BarRemovalQuestion {
  target_pair: [string, string];
  correct_removals: string[];
}

export interface SceneMetadata {
  seed: number;
  grid_size: number;
  wall_density: number;
  point_count: number;
  maze: MazeGrid;
  points: LabeledPoint[];
  pairs: Pair[];
  canvas_size: number;
  difficulty?: DifficultyMetric;
  question_type?: QuestionType;
  bars?: Bar[];
  question?: BarRemovalQuestion;
}
