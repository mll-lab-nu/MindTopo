// Main entry point for Hole Scene Generator

import { 
  SceneGenerator, 
  createSceneGenerator 
} from './core/SceneGenerator';
import { 
  LoopController, 
  createLoopController,
  BatchResult 
} from './core/LoopController';
import { getDefaultConfig } from './core/ConfigManager';
import { exportCanvasAsJPG, getSquareCanvasDataURL } from './utils/Exporter';
import { SceneConfig, HoleType, BoardShape } from './types';
import JSZip from 'jszip';

type DeepPartial<T> = T extends Array<infer _Item>
  ? T
  : T extends object
    ? { [Key in keyof T]?: DeepPartial<T[Key]> }
    : T;
type SceneConfigPatch = DeepPartial<SceneConfig>;
const BOARD_SHAPES: BoardShape[] = [
  'rect',
  'circle',
  'polygon',
  // 'rounded_rect',
  // 'ellipse',
  // 'star',
  // 'cross',
  // 'l_shape',
  // 'irregular'
];
const UNDER_BOARD_COLORS = [
  '#78d6ff', // bright light blue
  '#ffe45c', // bright light yellow
  '#ff8f8f'  // bright light red
];
const VISIBLE_HOLE_COUNT_RANGES: Record<number, [number, number]> = {
  1: [5, 10],
  2: [7, 13],
  3: [9, 17]
};

interface HoleSceneMetadata {
  seed: number;
  board_shape: string;
  board_width: number;
  board_height: number;
  board_thickness: number;
  total_placements: number;
  visible_feature_count: number;
  visible_through_hole_count: number;
  visible_pit_count: number;
  visible_mixed_count: number;
  visible_mixed2_count: number;
  visible_opening_count: number;
  hole_type_counts: Record<HoleType, number>;
  configured_hole_types: HoleType[];
  under_board_enabled: boolean;
  material_type: string;
  difficulty: number | null;
  stats: {
    holesGenerated: number;
    vertexCount: number;
    renderTimeMs: number;
  };
}

declare global {
  interface Window {
    topoBench?: {
      generate: (config?: SceneConfigPatch & { difficulty?: number }) => HoleSceneMetadata;
      screenshot: () => string;
    };
  }
}

// DOM Elements
const container = document.getElementById('canvas-container') as HTMLElement;
const generateBtn = document.getElementById('generateBtn') as HTMLButtonElement;
const randomizeBtn = document.getElementById('randomizeBtn') as HTMLButtonElement;
const exportBtn = document.getElementById('exportBtn') as HTMLButtonElement;
const batchBtn = document.getElementById('batchBtn') as HTMLButtonElement;

// Stats display
const statHoles = document.getElementById('stat-holes') as HTMLElement;
const statVertices = document.getElementById('stat-vertices') as HTMLElement;
const statTime = document.getElementById('stat-time') as HTMLElement;

// Batch progress
const batchProgress = document.getElementById('batch-progress') as HTMLElement;
const batchStatus = document.getElementById('batch-status') as HTMLElement;
const progressFill = document.getElementById('progress-fill') as HTMLElement;

// Input elements
const inputs = {
  difficulty: document.getElementById('difficulty') as HTMLSelectElement,
  seed: document.getElementById('seed') as HTMLInputElement,
  boardShape: document.getElementById('boardShape') as HTMLSelectElement,
  boardWidth: document.getElementById('boardWidth') as HTMLInputElement,
  boardHeight: document.getElementById('boardHeight') as HTMLInputElement,
  boardThickness: document.getElementById('boardThickness') as HTMLInputElement,
  holesMin: document.getElementById('holesMin') as HTMLInputElement,
  holesMax: document.getElementById('holesMax') as HTMLInputElement,
  minDistance: document.getElementById('minDistance') as HTMLInputElement,
  edgeBuffer: document.getElementById('edgeBuffer') as HTMLInputElement,
  holeTypes: document.getElementById('holeTypes') as HTMLSelectElement,
  shapeVariant: document.getElementById('shapeVariant') as HTMLSelectElement,
  underBoardEnable: document.getElementById('underBoardEnable') as HTMLInputElement,
  underBoardOffsetX: document.getElementById('underBoardOffsetX') as HTMLInputElement,
  underBoardOffsetY: document.getElementById('underBoardOffsetY') as HTMLInputElement,
  underBoardCoverage: document.getElementById('underBoardCoverage') as HTMLSelectElement,
  cameraPitch: document.getElementById('cameraPitch') as HTMLInputElement,
  cameraYaw: document.getElementById('cameraYaw') as HTMLInputElement,
  cameraDistance: document.getElementById('cameraDistance') as HTMLInputElement,
  cameraFov: document.getElementById('cameraFov') as HTMLInputElement,
  boardColor: document.getElementById('boardColor') as HTMLInputElement,
  underBoardColor: document.getElementById('underBoardColor') as HTMLInputElement,
  materialType: document.getElementById('materialType') as HTMLSelectElement,
  ambientIntensity: document.getElementById('ambientIntensity') as HTMLInputElement,
  lightX: document.getElementById('lightX') as HTMLInputElement,
  lightY: document.getElementById('lightY') as HTMLInputElement,
  lightZ: document.getElementById('lightZ') as HTMLInputElement,
  batchCount: document.getElementById('batchCount') as HTMLInputElement,
  batchMode: document.getElementById('batchMode') as HTMLSelectElement
};

// Initialize generator
let generator: SceneGenerator;
let loopController: LoopController;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function deepMerge<T>(target: T, source: DeepPartial<T>): T {
  if (!isPlainObject(target) || !isPlainObject(source)) {
    return source as T;
  }

  const merged: Record<string, unknown> = { ...target };
  for (const [key, value] of Object.entries(source)) {
    const current = merged[key];
    if (isPlainObject(current) && isPlainObject(value)) {
      merged[key] = deepMerge(current, value);
    } else if (value !== undefined) {
      merged[key] = value;
    }
  }
  return merged as T;
}

function boardShapeForSeed(seed: number): BoardShape {
  const normalizedSeed = Number.isFinite(seed) ? Math.abs(Math.trunc(seed)) : 0;
  return BOARD_SHAPES[normalizedSeed % BOARD_SHAPES.length];
}

function underBoardColorForSeed(seed: number): string {
  const normalizedSeed = Number.isFinite(seed) ? Math.abs(Math.trunc(seed)) : 0;
  return UNDER_BOARD_COLORS[normalizedSeed % UNDER_BOARD_COLORS.length];
}

function stableU32(seed: number, salt: number): number {
  let value = (Math.trunc(seed) ^ Math.trunc(salt)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

function underBoardEnabledForSeed(seed: number): boolean {
  return stableU32(seed, 701) % 2 === 0;
}

function deterministicInt(seed: number, salt: number, low: number, high: number): number {
  if (high < low) return low;
  return low + (stableU32(seed, salt) % (high - low + 1));
}

function deterministicShuffle<T>(values: T[], seed: number, salt: number): T[] {
  return values
    .map((value, index) => ({ value, index, key: stableU32(seed, salt + index) }))
    .sort((a, b) => a.key - b.key || a.index - b.index)
    .map((item) => item.value);
}

function visibleOpeningCountForPlan(typePlan: HoleType[]): number {
  return typePlan.reduce((sum, holeType) => {
    if (holeType === 'mixed2') return sum + 2;
    if (holeType === 'hole' || holeType === 'mixed') return sum + 1;
    return sum;
  }, 0);
}

function visibleHoleCountRangeForDifficulty(difficulty: number): [number, number] {
  return VISIBLE_HOLE_COUNT_RANGES[difficulty] ?? VISIBLE_HOLE_COUNT_RANGES[3];
}

function typePlanForDifficulty(seed: number, difficulty: number): HoleType[] {
  const [minVisible, maxVisible] = visibleHoleCountRangeForDifficulty(difficulty);
  if (difficulty === 1) {
    const visibleCount = deterministicInt(seed, 101, minVisible, maxVisible);
    const pitCount = deterministicInt(seed, 201, 1, 3);
    return deterministicShuffle(
      [
        ...Array.from({ length: visibleCount }, () => 'hole' as HoleType),
        ...Array.from({ length: pitCount }, () => 'pit' as HoleType)
      ],
      seed,
      301
    );
  }

  if (difficulty === 2) {
    const visibleCount = deterministicInt(seed, 102, minVisible, maxVisible);
    const pitCount = deterministicInt(seed, 202, 1, 3);
    const mixedCount = deterministicInt(seed, 402, 1, Math.min(3, visibleCount));
    const holeCount = visibleCount - mixedCount;
    return deterministicShuffle(
      [
        ...Array.from({ length: holeCount }, () => 'hole' as HoleType),
        ...Array.from({ length: mixedCount }, () => 'mixed' as HoleType),
        ...Array.from({ length: pitCount }, () => 'pit' as HoleType)
      ],
      seed,
      302
    );
  }

  const visibleCount = deterministicInt(seed, 103, minVisible, maxVisible);
  const plan: HoleType[] = ['hole', 'pit', 'mixed', 'mixed2'];
  let remaining = visibleCount - visibleOpeningCountForPlan(plan);
  let fillIndex = 0;
  while (remaining > 0) {
    const preferMixed2 = remaining >= 2 && stableU32(seed, 403 + fillIndex) % 3 === 0;
    if (preferMixed2) {
      plan.push('mixed2');
      remaining -= 2;
    } else {
      plan.push(stableU32(seed, 503 + fillIndex) % 2 === 0 ? 'hole' : 'mixed');
      remaining -= 1;
    }
    fillIndex += 1;
  }
  return deterministicShuffle(plan, seed, 603);
}

function difficultyToConfig(difficulty: number, seed: number): SceneConfigPatch {
  const bucket = Math.max(1, Math.min(3, Math.round(difficulty)));
  const boardShape = boardShapeForSeed(seed);
  const typePlan = typePlanForDifficulty(seed, bucket);
  const holeTypes = Array.from(new Set(typePlan));
  const commonGeometry = {
    board: { shape: boardShape, size: { width: 10, height: 10 }, thickness: 0.5 },
    holes: {
      count: { min: typePlan.length, max: typePlan.length },
      typePlan,
      sizeRange: { min: 0.35, max: 0.9 },
      minDistance: 0.45,
      edgeBuffer: 0.85,
      shapeVariant: 'vertical' as const
    },
    underBoard: {
      enable: underBoardEnabledForSeed(seed),
      offset: { x: 0.2, y: 0.1 },
      coverage: 'partial' as const
    }
  };
  const render = {
    appearance: {
      underBoardColor: underBoardColorForSeed(seed)
    }
  };
  if (bucket === 1) {
    return {
      geometry: {
        ...commonGeometry,
        holes: {
          ...commonGeometry.holes,
          types: holeTypes,
        }
      },
      render
    };
  }
  if (bucket === 2) {
    return {
      geometry: {
        ...commonGeometry,
        holes: {
          ...commonGeometry.holes,
          types: holeTypes,
        }
      },
      render
    };
  }
  return {
    geometry: {
      ...commonGeometry,
      holes: {
        ...commonGeometry.holes,
        types: holeTypes,
      }
    },
    render
  };
}

function selectedDifficulty(): number | null {
  const rawValue = inputs.difficulty?.value ?? '';
  if (!rawValue) {
    return null;
  }
  const parsed = Number.parseInt(rawValue, 10);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.max(1, Math.min(3, Math.round(parsed)));
}

function buildSceneMetadata(config: SceneConfig, difficulty: number | null): HoleSceneMetadata {
  const holeTypeCounts = generator.getEffectiveHoleTypeCounts();
  const visibleFeatureCount = Object.values(holeTypeCounts).reduce((sum, count) => sum + count, 0);
  const visibleOpeningCount =
    holeTypeCounts.hole + holeTypeCounts.mixed + holeTypeCounts.mixed2 * 2;
  const stats = generator.getStats();
  return {
    seed: config.seed,
    board_shape: config.geometry.board.shape,
    board_width: config.geometry.board.size.width,
    board_height: config.geometry.board.size.height,
    board_thickness: config.geometry.board.thickness,
    total_placements: generator.getHolePlacements().length,
    visible_feature_count: visibleFeatureCount,
    visible_through_hole_count: holeTypeCounts.hole,
    visible_pit_count: holeTypeCounts.pit,
    visible_mixed_count: holeTypeCounts.mixed,
    visible_mixed2_count: holeTypeCounts.mixed2,
    visible_opening_count: visibleOpeningCount,
    hole_type_counts: holeTypeCounts,
    configured_hole_types: [...config.geometry.holes.types],
    under_board_enabled: config.geometry.underBoard.enable,
    material_type: config.render.appearance.materialType,
    difficulty,
    stats,
  };
}

function generateTopoBenchScene(config?: SceneConfigPatch & { difficulty?: number }): HoleSceneMetadata {
  const baseConfig = generator.getConfig();
  const requestedDifficulty =
    typeof config?.difficulty === 'number' ? Math.max(1, Math.min(3, Math.round(config.difficulty))) : null;
  const requestedSeed = typeof config?.seed === 'number' ? config.seed : baseConfig.seed;
  const difficultyConfig = requestedDifficulty !== null ? difficultyToConfig(requestedDifficulty, requestedSeed) : {};
  const mergedConfig = deepMerge<SceneConfig>(
    deepMerge<SceneConfig>(baseConfig, difficultyConfig),
    config ? (({ difficulty: _ignored, ...rest }) => rest)(config) : {}
  );
  const stats = generator.generate(mergedConfig);
  updateStats(stats);
  syncCameraInputs();
  const finalConfig = generator.getConfig();
  return buildSceneMetadata(finalConfig, requestedDifficulty);
}

/**
 * Get configuration from UI inputs
 */
function getConfigFromUI(): SceneConfig {
  // Get selected hole types
  const selectedTypes: HoleType[] = [];
  for (const option of inputs.holeTypes.selectedOptions) {
    selectedTypes.push(option.value as HoleType);
  }
  
  return {
    seed: parseInt(inputs.seed.value) || 12345,
    geometry: {
      board: {
        shape: inputs.boardShape.value as BoardShape,
        size: {
          width: parseFloat(inputs.boardWidth.value) || 10,
          height: parseFloat(inputs.boardHeight.value) || 10
        },
        thickness: parseFloat(inputs.boardThickness.value) || 0.5
      },
      holes: {
        count: {
          min: parseInt(inputs.holesMin.value) || 5,
          max: parseInt(inputs.holesMax.value) || 13
        },
        minDistance: parseFloat(inputs.minDistance.value) || 0.45,
        edgeBuffer: parseFloat(inputs.edgeBuffer.value) || 0.85,
        types: selectedTypes.length > 0 ? selectedTypes : ['hole'],
        shapeVariant: inputs.shapeVariant.value as 'vertical' | 'distorted',
        sizeRange: { min: 0.35, max: 0.9 }
      },
      underBoard: {
        enable: inputs.underBoardEnable.checked,
        offset: {
          x: parseFloat(inputs.underBoardOffsetX.value) || 0.2,
          y: parseFloat(inputs.underBoardOffsetY.value) || 0.1
        },
        coverage: inputs.underBoardCoverage.value as 'full' | 'partial' | 'random'
      }
    },
    render: {
      camera: {
        pitch: parseFloat(inputs.cameraPitch.value) || 85,
        yaw: parseFloat(inputs.cameraYaw.value) || 75,
        distance: parseFloat(inputs.cameraDistance.value) || 18,
        fov: parseFloat(inputs.cameraFov.value) || 45
      },
      appearance: {
        boardColor: inputs.boardColor.value || '#FF5733',
        materialType: inputs.materialType.value as 'phong' | 'standard' | 'wood' | 'rock' | 'metal',
        underBoardColor: inputs.underBoardColor.value || '#78d6ff'
      },
      lighting: {
        ambientIntensity: parseFloat(inputs.ambientIntensity.value) || 0.5,
        dirLightPosition: [
          parseFloat(inputs.lightX.value) || 10,
          parseFloat(inputs.lightY.value) || 20,
          parseFloat(inputs.lightZ.value) || 10
        ]
      }
    }
  };
}

/**
 * Update UI from configuration
 */
function updateUIFromConfig(config: SceneConfig): void {
  inputs.seed.value = String(config.seed);
  inputs.boardShape.value = config.geometry.board.shape;
  inputs.boardWidth.value = String(config.geometry.board.size.width);
  inputs.boardHeight.value = String(config.geometry.board.size.height);
  inputs.boardThickness.value = String(config.geometry.board.thickness);
  inputs.holesMin.value = String(config.geometry.holes.count.min);
  inputs.holesMax.value = String(config.geometry.holes.count.max);
  inputs.minDistance.value = String(config.geometry.holes.minDistance);
  inputs.edgeBuffer.value = String(config.geometry.holes.edgeBuffer);
  inputs.shapeVariant.value = config.geometry.holes.shapeVariant;
  inputs.underBoardEnable.checked = config.geometry.underBoard.enable;
  inputs.underBoardOffsetX.value = String(config.geometry.underBoard.offset.x);
  inputs.underBoardOffsetY.value = String(config.geometry.underBoard.offset.y);
  inputs.underBoardCoverage.value = config.geometry.underBoard.coverage;
  inputs.cameraPitch.value = String(config.render.camera.pitch);
  inputs.cameraYaw.value = String(config.render.camera.yaw);
  inputs.cameraDistance.value = String(config.render.camera.distance);
  inputs.cameraFov.value = String(config.render.camera.fov);
  inputs.boardColor.value = config.render.appearance.boardColor;
  inputs.underBoardColor.value = config.render.appearance.underBoardColor;
  inputs.materialType.value = config.render.appearance.materialType;
  inputs.ambientIntensity.value = String(config.render.lighting.ambientIntensity);
  inputs.lightX.value = String(config.render.lighting.dirLightPosition[0]);
  inputs.lightY.value = String(config.render.lighting.dirLightPosition[1]);
  inputs.lightZ.value = String(config.render.lighting.dirLightPosition[2]);
  
  // Update hole types multi-select
  for (const option of inputs.holeTypes.options) {
    option.selected = config.geometry.holes.types.includes(option.value as HoleType);
  }
}

/**
 * Update stats display
 */
function updateStats(stats: { holesGenerated: number; vertexCount: number; renderTimeMs: number }): void {
  statHoles.textContent = String(stats.holesGenerated);
  statVertices.textContent = String(stats.vertexCount);
  statTime.textContent = String(stats.renderTimeMs);
}

/**
 * Keep camera inputs in sync with interactive camera controls.
 */
function syncCameraInputs(): void {
  const params = generator.getCameraRig().getParams();
  inputs.cameraPitch.value = String(Math.round(params.pitch * 10) / 10);
  inputs.cameraYaw.value = String(Math.round(params.yaw * 10) / 10);
  inputs.cameraDistance.value = String(Math.round(params.distance * 100) / 100);
  inputs.cameraFov.value = String(Math.round(params.fov * 10) / 10);
}

/**
 * Enable mouse drag orbit and wheel zoom directly on the canvas.
 */
function attachCameraInteraction(): void {
  const canvas = generator.getCanvas();
  let dragging = false;
  let pointerId: number | null = null;
  let lastX = 0;
  let lastY = 0;
  let pendingCamera: { pitch: number; yaw: number } | null = null;
  let pendingFrame = 0;

  const flushPreview = () => {
    pendingFrame = 0;
    if (!pendingCamera) {
      return;
    }
    generator.previewCamera(pendingCamera);
    pendingCamera = null;
  };

  const schedulePreview = () => {
    if (pendingFrame !== 0) {
      return;
    }
    pendingFrame = window.requestAnimationFrame(flushPreview);
  };

  const endDrag = () => {
    if (pendingFrame !== 0) {
      window.cancelAnimationFrame(pendingFrame);
      flushPreview();
    }
    generator.commitCamera();
    syncCameraInputs();
    dragging = false;
    pointerId = null;
    pendingCamera = null;
    canvas.classList.remove('dragging');
  };

  canvas.addEventListener('pointerdown', (event: PointerEvent) => {
    if (event.button !== 0) {
      return;
    }
    dragging = true;
    pointerId = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.classList.add('dragging');
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener('pointermove', (event: PointerEvent) => {
    if (!dragging || pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - lastX;
    const deltaY = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;

    const current = pendingCamera ?? generator.getCameraRig().getParams();
    pendingCamera = {
      pitch: current.pitch + deltaY * 0.16,
      yaw: current.yaw - deltaX * 0.32
    };
    schedulePreview();
  });

  canvas.addEventListener('pointerup', (event: PointerEvent) => {
    if (pointerId === event.pointerId) {
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      endDrag();
    }
  });

  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('lostpointercapture', endDrag);

  canvas.addEventListener(
    'wheel',
    (event: WheelEvent) => {
      event.preventDefault();
      const current = generator.getCameraRig().getParams();
      const factor = event.deltaY > 0 ? 1.08 : 0.92;
      generator.setCamera({ distance: current.distance * factor });
      syncCameraInputs();
    },
    { passive: false }
  );
}

/**
 * Generate scene with current config
 */
function generateScene(): void {
  const difficulty = selectedDifficulty();
  const uiConfig = getConfigFromUI();
  const config: SceneConfig = difficulty === null
    ? uiConfig
    : deepMerge<SceneConfig>(uiConfig, difficultyToConfig(difficulty, uiConfig.seed));
  if (difficulty !== null) {
    updateUIFromConfig(config);
    inputs.difficulty.value = String(difficulty);
  }
  const stats = generator.generate(config);
  updateStats(stats);
}

/**
 * Randomize all parameters
 */
function randomize(): void {
  inputs.difficulty.value = '';
  // Generate new seed
  inputs.seed.value = String(Math.floor(Math.random() * 1000000));
  
  // Randomize generator
  generator.randomizeAll();
  
  // Get updated config and sync UI
  const config = generator.getConfig();
  updateUIFromConfig(config);
  
  // Generate with new config
  const stats = generator.generate(config);
  updateStats(stats);
}

/**
 * Export current canvas as PNG
 */
function exportImage(): void {
  const canvas = generator.getCanvas();
  const config = generator.getConfig();
  const filename = `hole_scene_${config.seed}.jpg`;
  exportCanvasAsJPG(canvas, filename);
}

/**
 * Run batch generation
 */
async function runBatch(): Promise<void> {
  const count = parseInt(inputs.batchCount.value) || 10;
  const mode = inputs.batchMode.value as 'static' | 'invariance';
  
  // Show progress
  batchProgress.style.display = 'block';
  batchBtn.disabled = true;
  generateBtn.disabled = true;
  randomizeBtn.disabled = true;
  
  try {
    if (mode === 'static') {
      const { folderName, items } = await loopController.generateStaticDataset(
        count,
        (current: number, total: number, result: BatchResult) => {
          const percent = (current / total) * 100;
          progressFill.style.width = `${percent}%`;
          batchStatus.textContent = `Generating ${current}/${total}...`;
          if (result.stats) updateStats(result.stats);
        }
      );

      // Package zip: folderName/images/*.jpg + settings.jsonl
      const zip = new JSZip();
      const root = zip.folder(folderName);
      const imagesFolder = root?.folder('images');
      const jsonl = items.map(item => item.settingsLine).join('\n');
      root?.file('settings.jsonl', jsonl);
      items.forEach(item => {
        const base64 = item.dataUrl.split(',')[1];
        imagesFolder?.file(`${item.id}.jpg`, base64, { base64: true });
      });

      const blob = await zip.generateAsync({ type: 'blob' });
      const link = document.createElement('a');
      link.download = `${folderName}.zip`;
      link.href = URL.createObjectURL(blob);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 2000);

      batchStatus.textContent = `Complete! Generated ${items.length} images.`;
    } else {
      const { folderName, pairs, images } = await loopController.generateInvarianceDataset(
        count,
        (current: number, total: number, result?: BatchResult) => {
          const percent = (current / total) * 100;
          progressFill.style.width = `${percent}%`;
          batchStatus.textContent = `Generating pairs ${current}/${total}...`;
          if (result?.stats) updateStats(result.stats);
        }
      );

      // Package zip: folderName/images/*.jpg + settings.jsonl
      const zip = new JSZip();
      const root = zip.folder(folderName);
      const imagesFolder = root?.folder('images');
      const jsonl = pairs.map(pair => JSON.stringify(pair)).join('\n');
      root?.file('settings.jsonl', jsonl);
      images.forEach(item => {
        const base64 = item.dataUrl.split(',')[1];
        imagesFolder?.file(item.filename, base64, { base64: true });
      });

      const blob = await zip.generateAsync({ type: 'blob' });
      const link = document.createElement('a');
      link.download = `${folderName}.zip`;
      link.href = URL.createObjectURL(blob);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 2000);

      batchStatus.textContent = `Complete! Generated ${pairs.length} pairs (${images.length} images).`;
    }
  } catch (error) {
    console.error('Batch generation error:', error);
    batchStatus.textContent = 'Error during generation';
  } finally {
    batchBtn.disabled = false;
    generateBtn.disabled = false;
    randomizeBtn.disabled = false;
    
    setTimeout(() => {
      batchProgress.style.display = 'none';
      progressFill.style.width = '0%';
    }, 2000);
  }
}

/**
 * Initialize the application
 */
function init(): void {
  // Create generator with default config
  const defaultConfig = getDefaultConfig();
  generator = createSceneGenerator(container, defaultConfig);
  loopController = createLoopController(generator);
  
  // Initial generation
  const difficulty = selectedDifficulty();
  const initialConfig: SceneConfig = difficulty === null
    ? defaultConfig
    : deepMerge<SceneConfig>(defaultConfig, difficultyToConfig(difficulty, defaultConfig.seed));
  if (difficulty !== null) {
    updateUIFromConfig(initialConfig);
    inputs.difficulty.value = String(difficulty);
  }
  const stats = generator.generate(initialConfig);
  updateStats(stats);
  syncCameraInputs();
  attachCameraInteraction();
  
  // Event listeners
  generateBtn.addEventListener('click', generateScene);
  randomizeBtn.addEventListener('click', randomize);
  exportBtn.addEventListener('click', exportImage);
  batchBtn.addEventListener('click', runBatch);
  
  // Auto-generate on input change (debounced)
  let debounceTimer: ReturnType<typeof setTimeout>;
  const debounceGenerate = () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(generateScene, 300);
  };
  
  // Add change listeners to all inputs
  Object.values(inputs).forEach(input => {
    input.addEventListener('change', debounceGenerate);
    if (input instanceof HTMLInputElement && input.type === 'number') {
      input.addEventListener('input', debounceGenerate);
    }
  });

  window.topoBench = {
    generate: (config = {}) => generateTopoBenchScene(config),
    screenshot: () => {
      const currentConfig = generator.getConfig();
      generator.prepareExportFraming(
        currentConfig.geometry.board.size.width,
        currentConfig.geometry.board.size.height
      );
      return getSquareCanvasDataURL(generator.getCanvas(), 'png');
    }
  };
  
  console.log('Hole Scene Generator initialized');
}

// Start the application
init();
