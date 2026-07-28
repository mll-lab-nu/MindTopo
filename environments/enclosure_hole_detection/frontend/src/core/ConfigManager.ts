// Configuration management with defaults and validation

import { 
  SceneConfig, 
  GeometryParams, 
  RenderParams,
  BoardConfig,
  HolesConfig,
  UnderBoardConfig,
  CameraParams,
  AppearanceParams,
  LightingParams
} from '../types';
import { SeededRandom } from '../utils/RandomUtils';

// Default configuration values
const DEFAULT_BOARD: BoardConfig = {
  shape: 'rect',
  size: { width: 10, height: 10 },
  thickness: 0.5
};

const DEFAULT_HOLES: HolesConfig = {
  count: { min: 3, max: 5 },
  minDistance: 0.5,
  edgeBuffer: 1.0,
  types: ['hole', 'pit', 'mixed', 'mixed2'],
  shapeVariant: 'vertical',
  sizeRange: { min: 0.3, max: 1.2 }
};

const DEFAULT_UNDERBOARD: UnderBoardConfig = {
  enable: true,
  offset: { x: 0.2, y: 0.1 },
  coverage: 'partial'
};

const DEFAULT_GEOMETRY: GeometryParams = {
  board: DEFAULT_BOARD,
  holes: DEFAULT_HOLES,
  underBoard: DEFAULT_UNDERBOARD
};

const DEFAULT_CAMERA: CameraParams = {
  pitch: 85,
  yaw: 75,
  distance: 18,
  fov: 45
};

const DEFAULT_APPEARANCE: AppearanceParams = {
  boardColor: '#FF5733',
  materialType: 'standard',
  underBoardColor: '#78d6ff'
};

const DEFAULT_LIGHTING: LightingParams = {
  ambientIntensity: 0.5,
  dirLightPosition: [10, 20, 10]
};

const STANDARD_PBR_COLOR_PALETTE: Array<{
  boardColor: string;
  underBoardColor: string;
}> = [
    { boardColor: '#D6C2B3', underBoardColor: '#78d6ff' }, // warm clay / bright light blue
    { boardColor: '#D3CBB7', underBoardColor: '#ffe45c' }, // muted straw / bright light yellow
    { boardColor: '#C8D1C4', underBoardColor: '#ff8f8f' }, // soft sage / bright light red
    { boardColor: '#C2D3CE', underBoardColor: '#78d6ff' }, // muted teal / bright light blue
    { boardColor: '#C3CEDA', underBoardColor: '#ffe45c' }, // misty steel blue / bright light yellow
    { boardColor: '#BFC6D6', underBoardColor: '#ff8f8f' }, // periwinkle gray / bright light red
    { boardColor: '#CBBFD6', underBoardColor: '#78d6ff' }, // dusty lavender / bright light blue
    { boardColor: '#D4BFC8', underBoardColor: '#ffe45c' }, // dusty rose / bright light yellow
    { boardColor: '#D1B6AC', underBoardColor: '#ff8f8f' }, // muted terracotta / bright light red
    { boardColor: '#C8C9CF', underBoardColor: '#78d6ff' }, // soft neutral gray / bright light blue
    { boardColor: '#C7D3B6', underBoardColor: '#ffe45c' }, // muted olive / bright light yellow
    { boardColor: '#D2C6D3', underBoardColor: '#ff8f8f' }  // soft lilac gray / bright light red
];

const DEFAULT_RENDER: RenderParams = {
  camera: DEFAULT_CAMERA,
  appearance: DEFAULT_APPEARANCE,
  lighting: DEFAULT_LIGHTING
};

const DEFAULT_CONFIG: SceneConfig = {
  seed: 12345,
  geometry: DEFAULT_GEOMETRY,
  render: DEFAULT_RENDER
};

/**
 * Configuration Manager - Handles config merging, validation, and defaults
 */
export class ConfigManager {
  private config: SceneConfig;
  private random?: SeededRandom;

  constructor(initialConfig?: Partial<SceneConfig>) {
    this.config = this.mergeWithDefaults(initialConfig || {});
  }

  /**
   * Set the random generator for randomization
   */
  setRandom(random: SeededRandom): void {
    this.random = random;
  }

  /**
   * Get the current configuration
   */
  getConfig(): SceneConfig {
    return JSON.parse(JSON.stringify(this.config));
  }

  /**
   * Update configuration (partial update supported)
   */
  updateConfig(partial: Partial<SceneConfig>): SceneConfig {
    this.config = this.mergeWithDefaults({ ...this.config, ...partial });
    if (partial.geometry?.holes) {
      if (partial.geometry.holes.typePlan === undefined) {
        delete this.config.geometry.holes.typePlan;
      }
      if (partial.geometry.holes.shapePool === undefined) {
        delete this.config.geometry.holes.shapePool;
      }
      if (partial.geometry.holes.mixed2ShapePool === undefined) {
        delete this.config.geometry.holes.mixed2ShapePool;
      }
      if (partial.geometry.holes.mixed2Probability === undefined) {
        delete this.config.geometry.holes.mixed2Probability;
      }
    }
    return this.getConfig();
  }

  /**
   * Update geometry configuration
   */
  updateGeometry(partial: Partial<GeometryParams>): void {
    this.config.geometry = this.deepMerge(this.config.geometry, partial);
  }

  /**
   * Update render configuration
   */
  updateRender(partial: Partial<RenderParams>): void {
    this.config.render = this.deepMerge(this.config.render, partial);
  }

  /**
   * Set seed value
   */
  setSeed(seed: number): void {
    this.config.seed = seed;
    if (this.random) {
      this.random.reset(seed);
    }
  }

  /**
   * Get geometry parameters
   */
  getGeometry(): GeometryParams {
    return JSON.parse(JSON.stringify(this.config.geometry));
  }

  /**
   * Get render parameters
   */
  getRender(): RenderParams {
    return JSON.parse(JSON.stringify(this.config.render));
  }

  /**
   * Merge partial config with defaults
   */
  private mergeWithDefaults(partial: Partial<SceneConfig>): SceneConfig {
    return {
      seed: partial.seed ?? DEFAULT_CONFIG.seed,
      geometry: this.deepMerge(DEFAULT_GEOMETRY, partial.geometry || {}),
      render: this.deepMerge(DEFAULT_RENDER, partial.render || {})
    };
  }

  /**
   * Deep merge two objects
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private deepMerge<T>(target: T, source: Partial<T>): T {
    const result = { ...target } as any;
    
    for (const key in source) {
      const sourceVal = (source as any)[key];
      const targetVal = result[key];
      
      if (
        sourceVal !== undefined &&
        typeof sourceVal === 'object' &&
        !Array.isArray(sourceVal) &&
        sourceVal !== null &&
        typeof targetVal === 'object' &&
        !Array.isArray(targetVal) &&
        targetVal !== null
      ) {
        result[key] = this.deepMerge(targetVal, sourceVal);
      } else if (sourceVal !== undefined) {
        result[key] = sourceVal;
      }
    }
    
    return result as T;
  }

  /**
   * Randomize geometry parameters
   */
  randomizeGeometry(): void {
    if (!this.random) return;
    
    // Randomize board
    const shapes = ['rect', 'circle', 'polygon'] as const;
    this.config.geometry.board.shape = this.random.pick([...shapes]);
    this.config.geometry.board.size.width = this.random.range(6, 15);
    this.config.geometry.board.size.height = this.random.range(6, 15);
    this.config.geometry.board.thickness = this.random.range(0.3, 1.0);
    
    // Randomize holes
    const minCount = this.random.rangeInt(1, 4);
    this.config.geometry.holes.count.min = minCount;
    this.config.geometry.holes.count.max = minCount + this.random.rangeInt(1, 4);
    this.config.geometry.holes.minDistance = this.random.range(0.3, 1.0);
    this.config.geometry.holes.edgeBuffer = this.random.range(0.5, 1.5);
    
    // Include mixed2 (sampling logic controls its appearance probability)
    this.config.geometry.holes.types = ['hole', 'pit', 'mixed', 'mixed2'];
    
    // Randomize variant
    this.config.geometry.holes.shapeVariant = this.random.chance(0.3) ? 'distorted' : 'vertical';
    
    // Randomize under-board
    this.config.geometry.underBoard.enable = this.random.chance(0.8);
    const coverages = ['full', 'partial', 'random'] as const;
    this.config.geometry.underBoard.coverage = this.random.pick([...coverages]);

    // Clear any explicit hole planning overrides
    delete this.config.geometry.holes.typePlan;
    delete this.config.geometry.holes.shapePool;
    delete this.config.geometry.holes.mixed2ShapePool;
    delete this.config.geometry.holes.mixed2Probability;
  }

  /**
   * Randomize render parameters
   */
  randomizeRender(): void {
    if (!this.random) return;
    
    // Randomize camera
    this.config.render.camera.pitch = this.random.range(35, 85);
    this.config.render.camera.yaw = this.random.range(0, 360);
    this.config.render.camera.distance = this.random.range(12, 22);
    this.config.render.camera.fov = this.random.range(35, 60);
    
    const materials = ['phong', 'standard', 'wood', 'rock', 'metal'] as const;
    this.config.render.appearance.materialType = this.random.pick([...materials]);
    
    // Randomize colors only for standard PBR (fixed 5-color palette)
    if (this.config.render.appearance.materialType === 'standard') {
      const palette = this.random.pick([...STANDARD_PBR_COLOR_PALETTE]);
      this.config.render.appearance.boardColor = palette.boardColor;
      this.config.render.appearance.underBoardColor = palette.underBoardColor;
    }
    
    // Randomize lighting
    this.config.render.lighting.ambientIntensity = this.random.range(0.3, 0.7);
    const lightDist = this.random.range(10, 25);
    const lightAngle = this.random.range(0, Math.PI * 2);
    this.config.render.lighting.dirLightPosition = [
      Math.cos(lightAngle) * lightDist,
      Math.sin(lightAngle) * lightDist,
      this.random.range(15, 25)
    ];
  }

  /**
   * Randomize all parameters
   */
  randomizeAll(): void {
    this.randomizeGeometry();
    this.randomizeRender();
  }

  /**
   * HSL to Hex color conversion
   */
  private hslToHex(h: number, s: number, l: number): string {
    s /= 100;
    l /= 100;
    
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    
    let r = 0, g = 0, b = 0;
    
    if (h >= 0 && h < 60) { r = c; g = x; b = 0; }
    else if (h >= 60 && h < 120) { r = x; g = c; b = 0; }
    else if (h >= 120 && h < 180) { r = 0; g = c; b = x; }
    else if (h >= 180 && h < 240) { r = 0; g = x; b = c; }
    else if (h >= 240 && h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    
    const toHex = (n: number) => {
      const hex = Math.round((n + m) * 255).toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };
    
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }

  /**
   * Validate configuration
   */
  validate(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    // Validate board
    if (this.config.geometry.board.size.width <= 0) {
      errors.push('Board width must be positive');
    }
    if (this.config.geometry.board.size.height <= 0) {
      errors.push('Board height must be positive');
    }
    if (this.config.geometry.board.thickness <= 0) {
      errors.push('Board thickness must be positive');
    }
    
    // Validate holes
    if (this.config.geometry.holes.count.min < 0) {
      errors.push('Minimum hole count cannot be negative');
    }
    if (this.config.geometry.holes.count.max < this.config.geometry.holes.count.min) {
      errors.push('Maximum hole count must be >= minimum');
    }
    if (this.config.geometry.holes.minDistance < 0) {
      errors.push('Minimum distance cannot be negative');
    }
    if (this.config.geometry.holes.edgeBuffer < 0) {
      errors.push('Edge buffer cannot be negative');
    }
    
    // Validate camera
    if (this.config.render.camera.pitch < 25 || this.config.render.camera.pitch > 89) {
      errors.push('Camera pitch must be between 25 and 89');
    }
    if (this.config.render.camera.yaw < 0 || this.config.render.camera.yaw > 360) {
      errors.push('Camera yaw must be between 0 and 360');
    }
    if (this.config.render.camera.distance <= 0) {
      errors.push('Camera distance must be positive');
    }
    if (this.config.render.camera.fov <= 0 || this.config.render.camera.fov > 180) {
      errors.push('Camera FOV must be between 0 and 180');
    }
    
    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Export configuration as JSON string
   */
  toJSON(): string {
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Import configuration from JSON string
   */
  fromJSON(json: string): void {
    try {
      const parsed = JSON.parse(json);
      this.config = this.mergeWithDefaults(parsed);
    } catch {
      console.error('Failed to parse configuration JSON');
    }
  }

  /**
   * Reset to defaults
   */
  reset(): void {
    this.config = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
  }
}

/**
 * Create a ConfigManager instance
 */
export function createConfigManager(initialConfig?: Partial<SceneConfig>): ConfigManager {
  return new ConfigManager(initialConfig);
}

/**
 * Get default configuration
 */
export function getDefaultConfig(): SceneConfig {
  return JSON.parse(JSON.stringify(DEFAULT_CONFIG));
}
