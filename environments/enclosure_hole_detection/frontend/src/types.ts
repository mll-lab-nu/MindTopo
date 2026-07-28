// Type definitions for the Hole Scene Generator

// Board shapes: basic geometric shapes for the main board
export type BoardShape = 
  | 'rect'           // Rectangle
  | 'circle'         // Circle
  | 'polygon'        // Regular polygon (5-8 sides)
  | 'rounded_rect'   // Rounded rectangle
  | 'ellipse'        // Ellipse
  | 'star'           // Star shape
  | 'cross'          // Cross/plus shape
  | 'l_shape'        // L-shaped
  | 'irregular';     // Irregular blob

// Hole shapes: various hole cutting shapes
export type HoleShape =
  | 'circle'         // Basic circle
  | 'ellipse'        // Ellipse
  | 'polygon'        // Regular polygon
  | 'star'           // Star shape
  | 'keyhole'        // Keyhole shape
  | 'slot'           // Elongated slot
  | 'cross'          // Cross/plus shape
  | 'blob'           // Irregular blob
  | 'crescent'       // Crescent/moon shape
  | 'heart'          // Heart shape
  | 'random';        // Random weird shape

export type HoleType = 'hole' | 'pit' | 'mixed' | 'mixed2';
export type ShapeVariant = 'vertical' | 'distorted';
export type CoverageType = 'full' | 'partial' | 'random';
export type MaterialType = 'phong' | 'standard' | 'wood' | 'rock' | 'metal';
export type MixedCoverMode = 'edge' | 'center_split';

export interface Size2D {
  width: number;
  height: number;
}

export interface Offset2D {
  x: number;
  y: number;
}

export interface CountRange {
  min: number;
  max: number;
}

export interface SizeRange {
  min: number;
  max: number;
}

export interface BoardConfig {
  shape: BoardShape;
  size: Size2D;
  thickness: number;
}

export interface HolesConfig {
  count: CountRange;
  minDistance: number;
  edgeBuffer: number;
  types: HoleType[];
  shapeVariant: ShapeVariant;
  sizeRange: SizeRange;
  mixedCoverMode?: MixedCoverMode;
  typePlan?: HoleType[];
  shapePool?: HoleShape[];
  mixed2ShapePool?: HoleShape[];
  mixed2Probability?: number;
}

export interface UnderBoardConfig {
  enable: boolean;
  offset: Offset2D;
  coverage: CoverageType;
}

export interface GeometryParams {
  board: BoardConfig;
  holes: HolesConfig;
  underBoard: UnderBoardConfig;
}

export interface CameraParams {
  pitch: number;  // 60-90 degrees, 90 = top-down
  yaw: number;    // 60-90 degrees
  distance: number;
  fov: number;
}

export interface AppearanceParams {
  boardColor: string;
  materialType: MaterialType;
  underBoardColor: string;
}

export interface LightingParams {
  ambientIntensity: number;
  dirLightPosition: [number, number, number];
}

export interface RenderParams {
  camera: CameraParams;
  appearance: AppearanceParams;
  lighting: LightingParams;
}

export interface SceneConfig {
  seed: number;
  geometry: GeometryParams;
  render: RenderParams;
}

// Hole placement data structure
export interface HolePlacement {
  x: number;
  y: number;
  radius: number;
  type: HoleType;
  depth: number;  // For pits, the depth; for holes, equals board thickness
  shape?: HoleShape;  // The shape of this specific hole
  rotation?: number;  // Rotation angle for the hole shape
  mixedCoverage?: number;  // For mixed/mixed2: coverage ratio used for the cover
}

// Generation result statistics
export interface GenerationStats {
  holesGenerated: number;
  vertexCount: number;
  renderTimeMs: number;
}

// Batch generation options
export interface BatchOptions {
  count: number;
  mode: 'static' | 'invariance';
  baseConfig: SceneConfig;
}
