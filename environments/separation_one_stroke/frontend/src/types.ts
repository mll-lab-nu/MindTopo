/**
 * TopoBench Type Definitions
 */

// Direction type for movement
export type Direction = 'U' | 'D' | 'L' | 'R';

// Vertex coordinate on the grid
export interface Vertex {
  x: number;
  y: number;
}

// Edge between two vertices (stored as sorted pair for uniqueness)
export interface Edge {
  v1: Vertex;
  v2: Vertex;
}

// Level definition JSON format
export interface LevelJson {
  W: number;           // Width (vertex grid)
  H: number;           // Height (vertex grid)
  cells: (string | null)[][];  // (H-1) x (W-1) color matrix, null = empty
  solution?: Direction[] | Vertex[];  // Optional solution
}

// Current game state
export interface GameState {
  W: number;
  H: number;
  cells: (string | null)[][];
  stroke: Vertex[];      // Sequence of vertices visited
  edges: Set<string>;    // Set of edge keys for quick lookup
  currentPos: Vertex;
  done: boolean;
  success: boolean;
  failureReason?: FailureReason;
}

// Failure reasons
export type FailureReason = 
  | 'OUT_OF_BOUNDS'
  | 'NOT_ADJACENT' 
  | 'REUSE_EDGE'
  | 'CREATES_CYCLE'
  | 'DONE_BUT_CONSTRAINT_FAIL';

// Step result
export interface StepResult {
  ok: boolean;
  reason?: FailureReason;
  done?: boolean;
  success?: boolean;
}

// Evaluation result
export interface EvaluateResult {
  done: boolean;
  success: boolean;
  reason?: string;
  regions?: number[][];  // Region IDs for each cell
  colorRegions?: Map<string, number>;  // Color -> region ID mapping
}

// API state response
export interface StateResponse {
  W: number;
  H: number;
  currentPos: Vertex;
  stroke: Vertex[];
  cells: (string | null)[][];
  stepCount: number;
  done: boolean;
  success: boolean;
  failureReason?: FailureReason;
}

// Color palette for rendering
export const COLORS: Record<string, number> = {
  'red': 0xe6194b,
  'green': 0x3cb44b,
  'blue': 0x4363d8,
  'yellow': 0xffe119,
  'purple': 0x911eb4,
  'cyan': 0x42d4f4,
  'gray': 0x9ca3af,
  'white': 0xffffff,
};

// Get hex color for a color name
export function getColorHex(colorName: string): number {
  return COLORS[colorName.toLowerCase()] ?? 0x888888;
}
