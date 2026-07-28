// Batch generation and loop control

import { SceneGenerator } from './SceneGenerator';
import {
  SceneConfig,
  BatchOptions,
  GenerationStats,
  HolePlacement,
  HoleType,
  MaterialType,
  HoleShape,
  BoardShape
} from '../types';
import { BatchExporter, BatchExportItem, DEFAULT_JPEG_QUALITY, getSquareCanvasDataURL } from '../utils/Exporter';
import { SeededRandom } from '../utils/RandomUtils';

/**
 * Result of a single batch item generation
 */
export interface BatchResult {
  index: number;
  seed: number;
  stats: GenerationStats;
  dataUrl?: string;
}

/**
 * Progress callback for batch generation
 */
export type BatchProgressCallback = (
  current: number,
  total: number,
  result: BatchResult
) => void;

export interface DatasetItem {
  id: string;
  seed: number;
  dataUrl: string;
  settingsLine: string;
}

type DifficultyBucket = 'easy' | 'medium' | 'hard';
type SimilarityMode = 'high' | 'medium' | 'low';
type TypeStyle = 'simple' | 'mixed' | 'diverse';

interface HoleSizeStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

interface InvarianceImageMeta {
  filename: string;
  seed: number;
  holeCount: number;
  holeTypeCounts: Record<HoleType, number>;
  holeShapeCounts: Record<HoleShape, number>;
  holeSizeStats: HoleSizeStats;
  boardShape: BoardShape;
  underBoard: boolean;
  materialType: MaterialType;
  boardColor: string;
  geometry: SceneConfig['geometry'];
  render: SceneConfig['render'];
  stats: GenerationStats;
}

interface InvariancePairRecord {
  pairId: string;
  labelEquivalent: boolean;
  similarityScore: number;
  difficultyScore: number;
  difficultyBucket: DifficultyBucket;
  targetHolecountA: number;
  targetHolecountB: number;
  imageA: InvarianceImageMeta;
  imageB: InvarianceImageMeta;
}

interface InvarianceImageExportItem {
  filename: string;
  dataUrl: string;
}

interface PairFeatures {
  holecount: number;
  holeTypeCounts: Record<HoleType, number>;
  holeShapeCounts: Record<HoleShape, number>;
  holeSizeStats: HoleSizeStats;
  boardShape: BoardShape;
  underBoard: boolean;
  materialType: MaterialType;
  boardColor: string;
}

interface InvarianceImageResult {
  config: SceneConfig;
  meta: InvarianceImageMeta;
  features: PairFeatures;
  dataUrl: string;
  preview: BatchResult;
}

export type PairProgressCallback = (
  currentPair: number,
  totalPairs: number,
  preview?: BatchResult
) => void;

const HOLE_SHAPES: HoleShape[] = [
  'circle',
  'ellipse',
  'polygon',
  'star',
  'keyhole',
  'slot',
  'cross',
  'crescent',
  'heart'
];

const SIMPLE_HOLE_SHAPES: HoleShape[] = ['circle', 'ellipse', 'polygon'];
const MEDIUM_HOLE_SHAPES: HoleShape[] = ['circle', 'ellipse', 'polygon', 'slot', 'cross'];
const MIXED2_HOLE_SHAPES: HoleShape[] = ['circle', 'ellipse', 'polygon', 'keyhole', 'slot'];

const BOARD_SHAPES: BoardShape[] = ['rect', 'circle', 'polygon'];
const HOLE_TYPES: HoleType[] = ['hole', 'pit', 'mixed', 'mixed2'];

const COLOR_PALETTE: string[] = [
  '#D6C2B3',
  '#D3CBB7',
  '#C8D1C4',
  '#C2D3CE',
  '#C3CEDA',
  '#BFC6D6',
  '#CBBFD6',
  '#D4BFC8',
  '#D1B6AC',
  '#C8C9CF',
  '#C7D3B6',
  '#D2C6D3'
];

const UNDER_BOARD_COLORS: string[] = [
  '#78d6ff',
  '#ffe45c',
  '#ff8f8f'
];

/**
 * Loop Controller - Manages batch generation
 * Supports both Static Reasoning and Invariance modes
 */
export class LoopController {
  private generator: SceneGenerator;
  private exporter: BatchExporter;
  private isRunning: boolean = false;
  private shouldStop: boolean = false;

  constructor(generator: SceneGenerator) {
    this.generator = generator;
    this.exporter = new BatchExporter();
  }

  /**
   * Check if batch generation is running
   */
  getIsRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Stop the current batch generation
   */
  stop(): void {
    this.shouldStop = true;
  }

  /**
   * Generate a batch of scenes
   * @param options Batch generation options
   * @param onProgress Optional progress callback
   * @returns Array of batch results
   */
  async batchGenerate(
    options: BatchOptions,
    onProgress?: BatchProgressCallback
  ): Promise<BatchResult[]> {
    if (this.isRunning) {
      throw new Error('Batch generation already in progress');
    }

    this.isRunning = true;
    this.shouldStop = false;
    this.exporter.clear();

    const results: BatchResult[] = [];
    const { count, mode, baseConfig } = options;

    // Store original geometry seed for Invariance mode
    const geometrySeed = baseConfig.seed;

    try {
      for (let i = 0; i < count; i++) {
        if (this.shouldStop) {
          break;
        }

        let currentConfig: Partial<SceneConfig>;
        let currentSeed: number;

        if (mode === 'static') {
          // Static Reasoning: Randomize everything including seed
          currentSeed = Math.floor(Math.random() * 1000000);
          this.generator.updateConfig({ seed: currentSeed });
          this.generator.randomizeAll();
          currentConfig = this.generator.getConfig();
        } else {
          // Invariance: Keep geometry seed, randomize render params
          currentSeed = geometrySeed;
          this.generator.updateConfig({ seed: geometrySeed });
          this.generator.randomizeRender();
          currentConfig = this.generator.getConfig();
        }

        // Generate the scene
        const stats = this.generator.generate(currentConfig);

        // Capture the image
        this.generator.prepareExportFraming(
          currentConfig.geometry.board.size.width,
          currentConfig.geometry.board.size.height
        );
        const canvas = this.generator.getCanvas();
        const dataUrl = getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY);

        // Store in exporter
        this.exporter.add(canvas, currentSeed, i);

        const result: BatchResult = {
          index: i,
          seed: currentSeed,
          stats,
          dataUrl
        };

        results.push(result);

        // Call progress callback
        if (onProgress) {
          onProgress(i + 1, count, result);
        }

        // Small delay to allow UI updates
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } finally {
      this.isRunning = false;
    }

    return results;
  }

  /**
   * Generate a static reasoning dataset with balanced sampling
   */
  async generateStaticDataset(
    total: number,
    onProgress?: BatchProgressCallback
  ): Promise<{ folderName: string; items: DatasetItem[] }> {
    if (this.isRunning) {
      throw new Error('Batch generation already in progress');
    }
    this.isRunning = true;
    this.shouldStop = false;
    const items: DatasetItem[] = [];
    const folderName = this.createTimestampFolderName();

    // Uniform hole count sampling across 0-15
    const holeCounts = Array.from({ length: 16 }, (_, i) => i); // 0-15
    const holeTypeCombos: HoleType[][] = [
      ['hole', 'pit', 'mixed', 'mixed2']
    ];
    const underBoardOptions = [true, false];
    const materials: MaterialType[] = ['phong', 'standard', 'wood', 'rock', 'metal'];

    // Shuffle each list once to avoid strict cycling patterns
    const shuffle = <T>(arr: T[]): T[] => {
      const a = [...arr];
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
      }
      return a;
    };

    const holeCountsShuffled = shuffle(holeCounts);
    const holeTypesShuffled = shuffle(holeTypeCombos);
    const underBoardShuffled = shuffle(underBoardOptions);
    const materialsShuffled = shuffle(materials);

    const pickCycle = <T>(arr: T[], idx: number): T => arr[idx % arr.length];

    try {
      for (let i = 0; i < total; i++) {
        if (this.shouldStop) break;

        const seed = Math.floor(Math.random() * 1_000_000_000);
        let targetHoleCount = pickCycle(holeCountsShuffled, i);
        let targetHoleTypes = pickCycle(holeTypesShuffled, i);
        const targetUnderBoard = pickCycle(underBoardShuffled, i);
        const targetMaterial = pickCycle(materialsShuffled, i);
        if (targetHoleCount === 0) {
          if (Math.random() < 0.75) {
            targetHoleCount = 1 + Math.floor(Math.random() * 3);
            targetHoleTypes = ['pit'];
          } else {
            targetHoleTypes = [];
          }
        }

        // Set seed and randomize other params for variety
        this.generator.updateConfig({ seed });
        this.generator.randomizeAll();
        const cfg = this.generator.getConfig();

        // Override with balanced picks
        cfg.geometry.holes.count = { min: targetHoleCount, max: targetHoleCount };
        cfg.geometry.holes.types = targetHoleTypes;
        cfg.geometry.underBoard.enable = targetUnderBoard;
        cfg.render.appearance.materialType = targetMaterial;

        const stats = this.generator.generate(cfg);

        const holeTypeCounts = this.generator.getEffectiveHoleTypeCounts();
        const actualHoleCount = Object.values(holeTypeCounts).reduce(
          (sum, count) => sum + count,
          0
        );
        const holeCount =
          holeTypeCounts.hole + holeTypeCounts.mixed + holeTypeCounts.mixed2 * 2;

        this.generator.prepareExportFraming(
          cfg.geometry.board.size.width,
          cfg.geometry.board.size.height
        );
        const canvas = this.generator.getCanvas();
        const dataUrl = getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY);

        const id = (i + 1).toString().padStart(5, '0');
        const settingsObj = {
          id,
          seed,
          holeCount,
          actualHoleCount,
          targetHoleCount,
          holeTypes: targetHoleTypes,
          holeTypeCounts,
          underBoard: targetUnderBoard,
          materialType: targetMaterial,
          geometry: cfg.geometry,
          render: cfg.render,
          stats
        };

        items.push({
          id,
          seed,
          dataUrl,
          settingsLine: JSON.stringify(settingsObj)
        });

        if (onProgress) {
          onProgress(i + 1, total, {
            index: i,
            seed,
            stats,
            dataUrl
          });
        }

        await new Promise(resolve => setTimeout(resolve, 5));
      }
    } finally {
      this.isRunning = false;
    }

    return { folderName, items };
  }

  /**
   * Generate invariance dataset (paired images with equivalence label)
   */
  async generateInvarianceDataset(
    pairCount: number,
    onProgress?: PairProgressCallback
  ): Promise<{ folderName: string; pairs: InvariancePairRecord[]; images: InvarianceImageExportItem[] }> {
    if (this.isRunning) {
      throw new Error('Batch generation already in progress');
    }
    this.isRunning = true;
    this.shouldStop = false;

    const pairs: InvariancePairRecord[] = [];
    const images: InvarianceImageExportItem[] = [];
    const folderName = this.createTimestampFolderName();
    const rng = new SeededRandom(Date.now());

    const eqHolecounts = this.buildUniformSchedule(pairCount, 0, 8, rng);
    const neqHolecounts = this.buildUniformSchedule(pairCount, 0, 8, rng);
    const eqBuckets = this.buildBucketSchedule(pairCount, rng);
    const neqBuckets = this.buildBucketSchedule(pairCount, rng);

    const totalPairs = pairCount * 2;
    let pairIndex = 0;

    try {
      for (let i = 0; i < pairCount; i++) {
        if (this.shouldStop) break;

        const targetHolecountA = eqHolecounts[i];
        const targetHolecountB = targetHolecountA;
        const bucket = eqBuckets[i];
        const pairId = (pairIndex + 1).toString().padStart(4, '0');

        const result = await this.generateInvariancePair(
          pairId,
          true,
          targetHolecountA,
          targetHolecountB,
          bucket,
          rng
        );

        pairs.push(result.pair);
        images.push(...result.images);
        pairIndex += 1;

        if (onProgress) {
          onProgress(pairIndex, totalPairs, result.preview);
        }

        await new Promise(resolve => setTimeout(resolve, 5));
      }

      for (let i = 0; i < pairCount; i++) {
        if (this.shouldStop) break;

        const targetHolecountA = neqHolecounts[i];
        const bucket = neqBuckets[i];
        const targetHolecountB = this.pickNonEquivalentHolecount(
          targetHolecountA,
          bucket,
          rng
        );
        const pairId = (pairIndex + 1).toString().padStart(4, '0');

        const result = await this.generateInvariancePair(
          pairId,
          false,
          targetHolecountA,
          targetHolecountB,
          bucket,
          rng
        );

        pairs.push(result.pair);
        images.push(...result.images);
        pairIndex += 1;

        if (onProgress) {
          onProgress(pairIndex, totalPairs, result.preview);
        }

        await new Promise(resolve => setTimeout(resolve, 5));
      }
    } finally {
      this.isRunning = false;
    }

    return { folderName, pairs, images };
  }

  /**
   * Generate invariance dataset (same geometry, different views)
   */
  async generateInvarianceSet(
    baseConfig: SceneConfig,
    viewCount: number,
    onProgress?: BatchProgressCallback
  ): Promise<BatchResult[]> {
    return this.batchGenerate(
      {
        count: viewCount,
        mode: 'invariance',
        baseConfig
      },
      onProgress
    );
  }

  /**
   * Generate static reasoning dataset (random everything)
   */
  async generateStaticSet(
    baseConfig: SceneConfig,
    count: number,
    onProgress?: BatchProgressCallback
  ): Promise<BatchResult[]> {
    return this.batchGenerate(
      {
        count,
        mode: 'static',
        baseConfig
      },
      onProgress
    );
  }

  /**
   * Get the batch exporter for downloading results
   */
  getExporter(): BatchExporter {
    return this.exporter;
  }

  /**
   * Get all exported items
   */
  getExportedItems(): BatchExportItem[] {
    return this.exporter.getItems();
  }

  /**
   * Download all generated images
   */
  async downloadAll(delayMs: number = 100): Promise<void> {
    await this.exporter.downloadAll(delayMs);
  }

  /**
   * Clear exported items
   */
  clearExports(): void {
    this.exporter.clear();
  }

  /**
   * Generate a single scene and return result
   */
  generateSingle(config?: Partial<SceneConfig>): BatchResult {
    const currentConfig = config || this.generator.getConfig();
    const stats = this.generator.generate(currentConfig);
    
    this.generator.prepareExportFraming(
      currentConfig.geometry.board.size.width,
      currentConfig.geometry.board.size.height
    );
    const canvas = this.generator.getCanvas();
    const dataUrl = getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY);

    return {
      index: 0,
      seed: currentConfig.seed || 0,
      stats,
      dataUrl
    };
  }

  /**
   * Generate a sequence with gradual parameter changes
   * Useful for creating animations or parameter sweeps
   */
  async generateSequence(
    startConfig: SceneConfig,
    endConfig: SceneConfig,
    steps: number,
    onProgress?: BatchProgressCallback
  ): Promise<BatchResult[]> {
    if (this.isRunning) {
      throw new Error('Batch generation already in progress');
    }

    this.isRunning = true;
    this.shouldStop = false;
    this.exporter.clear();

    const results: BatchResult[] = [];

    try {
      for (let i = 0; i < steps; i++) {
        if (this.shouldStop) {
          break;
        }

        const t = steps > 1 ? i / (steps - 1) : 0;
        
        // Interpolate parameters
        const interpolatedConfig = this.interpolateConfig(startConfig, endConfig, t);

        // Generate the scene
        const stats = this.generator.generate(interpolatedConfig);

        // Capture the image
        this.generator.prepareExportFraming(
          interpolatedConfig.geometry.board.size.width,
          interpolatedConfig.geometry.board.size.height
        );
        const canvas = this.generator.getCanvas();
        const dataUrl = getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY);

        this.exporter.add(canvas, interpolatedConfig.seed, i);

        const result: BatchResult = {
          index: i,
          seed: interpolatedConfig.seed,
          stats,
          dataUrl
        };

        results.push(result);

        if (onProgress) {
          onProgress(i + 1, steps, result);
        }

        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } finally {
      this.isRunning = false;
    }

    return results;
  }

  /**
   * Interpolate between two configurations
   */
  private interpolateConfig(
    start: SceneConfig,
    end: SceneConfig,
    t: number
  ): SceneConfig {
    const lerp = (a: number, b: number) => a + (b - a) * t;

    return {
      seed: start.seed, // Keep seed constant
      geometry: start.geometry, // Keep geometry constant
      render: {
        camera: {
          pitch: lerp(start.render.camera.pitch, end.render.camera.pitch),
          yaw: lerp(start.render.camera.yaw, end.render.camera.yaw),
          distance: lerp(start.render.camera.distance, end.render.camera.distance),
          fov: lerp(start.render.camera.fov, end.render.camera.fov)
        },
        appearance: t < 0.5 ? start.render.appearance : end.render.appearance,
        lighting: {
          ambientIntensity: lerp(
            start.render.lighting.ambientIntensity,
            end.render.lighting.ambientIntensity
          ),
          dirLightPosition: [
            lerp(start.render.lighting.dirLightPosition[0], end.render.lighting.dirLightPosition[0]),
            lerp(start.render.lighting.dirLightPosition[1], end.render.lighting.dirLightPosition[1]),
            lerp(start.render.lighting.dirLightPosition[2], end.render.lighting.dirLightPosition[2])
          ]
        }
      }
    };
  }

  private async generateInvariancePair(
    pairId: string,
    labelEquivalent: boolean,
    targetHolecountA: number,
    targetHolecountB: number,
    bucket: DifficultyBucket,
    rng: SeededRandom
  ): Promise<{ pair: InvariancePairRecord; images: InvarianceImageExportItem[]; preview: BatchResult }> {
    const similarityMode = this.getSimilarityMode(labelEquivalent, bucket);
    const pairLabel = `pair${pairId}`;
    const filenameA = `${pairLabel}_1.jpg`;
    const filenameB = `${pairLabel}_2.jpg`;
    const maxAAttempts = 24;
    const maxBAttempts = 32;

    let imageA: InvarianceImageResult | null = null;
    for (let attempt = 0; attempt < maxAAttempts; attempt++) {
      const relaxLevel = Math.floor(attempt / 6);
      const configA = this.buildConfigForInvarianceImage(
        targetHolecountA,
        rng,
        {
          labelEquivalent,
          difficultyBucket: bucket,
          similarityMode,
          relaxLevel
        }
      );
      const candidateA = this.tryGenerateInvarianceImage(configA, filenameA);
      if (candidateA.features.holecount === targetHolecountA) {
        imageA = candidateA;
        break;
      }
    }

    if (!imageA) {
      throw new Error(`Failed to generate pair ${pairId}: unable to place target A holes`);
    }

    const bucketRelaxAfter = 12;

    for (let attempt = 0; attempt < maxBAttempts; attempt++) {
      const relaxLevel = Math.floor(attempt / 6);
      const configB = this.buildConfigForInvarianceImage(
        targetHolecountB,
        rng,
        {
          labelEquivalent,
          difficultyBucket: bucket,
          similarityMode,
          baseConfig: imageA.config,
          relaxLevel
        }
      );
      const imageB = this.tryGenerateInvarianceImage(configB, filenameB);
      if (imageB.features.holecount !== targetHolecountB) {
        continue;
      }

      const scores = this.computePairScores(imageA.features, imageB.features, labelEquivalent);
      const requireBucketMatch = attempt < bucketRelaxAfter;
      if (requireBucketMatch && !this.difficultyMatchesBucket(scores.difficulty, bucket)) {
        continue;
      }

      const labelMatches = labelEquivalent
        ? imageA.features.holecount === imageB.features.holecount
        : imageA.features.holecount !== imageB.features.holecount;
      if (!labelMatches) {
        continue;
      }

      const pair: InvariancePairRecord = {
        pairId: pairLabel,
        labelEquivalent,
        similarityScore: Number(scores.similarity.toFixed(4)),
        difficultyScore: Number(scores.difficulty.toFixed(4)),
        difficultyBucket: bucket,
        targetHolecountA,
        targetHolecountB,
        imageA: imageA.meta,
        imageB: imageB.meta
      };

      const images: InvarianceImageExportItem[] = [
        { filename: filenameA, dataUrl: imageA.dataUrl },
        { filename: filenameB, dataUrl: imageB.dataUrl }
      ];

      return { pair, images, preview: imageB.preview };
    }

    throw new Error(`Failed to generate pair ${pairId}: unable to satisfy B constraints`);
  }

  private tryGenerateInvarianceImage(
    config: SceneConfig,
    filename: string
  ): InvarianceImageResult {
    const stats = this.generator.generate(config);
    const placements = this.generator.getHolePlacements();
    const holeTypeCounts = this.countByType(placements);
    const holecount = holeTypeCounts.hole + holeTypeCounts.mixed + holeTypeCounts.mixed2 * 2;
    const holeShapeCounts = this.countHoleShapes(placements);
    const holeSizeStats = this.computeHoleSizeStats(placements);

    this.generator.prepareExportFraming(
      config.geometry.board.size.width,
      config.geometry.board.size.height
    );
    const canvas = this.generator.getCanvas();
    const dataUrl = getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY);

    const meta: InvarianceImageMeta = {
      filename,
      seed: config.seed,
      holeCount: holecount,
      holeTypeCounts,
      holeShapeCounts,
      holeSizeStats,
      boardShape: config.geometry.board.shape,
      underBoard: config.geometry.underBoard.enable,
      materialType: config.render.appearance.materialType,
      boardColor: config.render.appearance.boardColor,
      geometry: config.geometry,
      render: config.render,
      stats
    };

    const features: PairFeatures = {
      holecount,
      holeTypeCounts,
      holeShapeCounts,
      holeSizeStats,
      boardShape: meta.boardShape,
      underBoard: meta.underBoard,
      materialType: meta.materialType,
      boardColor: meta.boardColor
    };

    const preview: BatchResult = {
      index: 0,
      seed: config.seed,
      stats,
      dataUrl
    };

    return {
      config,
      meta,
      features,
      dataUrl,
      preview
    };
  }

  private buildConfigForInvarianceImage(
    targetHolecount: number,
    rng: SeededRandom,
    options: {
      labelEquivalent: boolean;
      difficultyBucket: DifficultyBucket;
      similarityMode: SimilarityMode;
      baseConfig?: SceneConfig;
      relaxLevel?: number;
    }
  ): SceneConfig {
    const baseConfig = options.baseConfig
      ? JSON.parse(JSON.stringify(options.baseConfig))
      : this.generator.getConfig();
    const similarityMode = options.similarityMode;
    const useBase = Boolean(options.baseConfig);
    const relaxLevel = options.relaxLevel ?? 0;
    const relaxScale = this.clamp(1 + relaxLevel * 0.12, 1, 1.6);
    const sizeScale = this.clamp(1 - relaxLevel * 0.12, 0.4, 1);

    const reuseShape = useBase && this.shouldReuse(similarityMode, rng, 0.6);
    const boardShape = reuseShape
      ? baseConfig.geometry.board.shape
      : this.pickBoardShape(
        rng,
        useBase ? baseConfig.geometry.board.shape : undefined,
        similarityMode
      );

    const reuseSize = useBase && this.shouldReuse(similarityMode, rng, 0.6);
    const baseBoardSize = reuseSize
      ? { ...baseConfig.geometry.board.size }
      : this.buildBoardSize(targetHolecount, rng, boardShape);
    const boardSize = {
      width: this.clamp(baseBoardSize.width * relaxScale, 6, 20),
      height: this.clamp(baseBoardSize.height * relaxScale, 6, 20)
    };

    const thickness = useBase && this.shouldReuse(similarityMode, rng, 0.6)
      ? baseConfig.geometry.board.thickness
      : rng.range(0.35, 0.7);

    const underBoardEnable = useBase && this.shouldReuse(similarityMode, rng, 0.6)
      ? baseConfig.geometry.underBoard.enable
      : rng.chance(0.5);

    const coverageOptions: Array<'full' | 'partial' | 'random'> = ['full', 'partial', 'random'];
    const underBoardCoverage = useBase && this.shouldReuse(similarityMode, rng, 0.5)
      ? baseConfig.geometry.underBoard.coverage
      : rng.pick(coverageOptions);

    const materialType = useBase
      ? (this.shouldReuse(similarityMode, rng, 0.6)
        ? baseConfig.render.appearance.materialType
        : this.pickMaterialType(rng, baseConfig.render.appearance.materialType, similarityMode))
      : rng.pick(['phong', 'standard', 'wood', 'rock', 'metal']);

    const boardColor = useBase
      ? (this.shouldReuse(similarityMode, rng, 0.6)
        ? baseConfig.render.appearance.boardColor
        : this.pickBoardColor(rng, baseConfig.render.appearance.boardColor, similarityMode))
      : rng.pick(COLOR_PALETTE);

    const underBoardColor = useBase && this.shouldReuse(similarityMode, rng, 0.5)
      ? baseConfig.render.appearance.underBoardColor
      : rng.pick(UNDER_BOARD_COLORS);

    const reuseSizeRange = useBase && this.shouldReuse(similarityMode, rng, 0.6);
    const baseSizeRange = this.buildSizeRange(
      rng,
      reuseSizeRange ? baseConfig.geometry.holes.sizeRange : undefined,
      similarityMode
    );
    const sizeRangeMin = this.clamp(baseSizeRange.min * sizeScale, 0.15, 1.0);
    const sizeRangeMax = this.clamp(
      baseSizeRange.max * sizeScale,
      Math.max(sizeRangeMin * 1.05, 0.35),
      1.4
    );
    const sizeRange = {
      min: sizeRangeMin,
      max: sizeRangeMax
    };

    let shapePool = useBase && similarityMode === 'high' &&
      baseConfig.geometry.holes.shapePool &&
      baseConfig.geometry.holes.shapePool.length > 0
      ? baseConfig.geometry.holes.shapePool
      : this.pickShapePool(options.difficultyBucket, similarityMode);
    if (relaxLevel >= 2) {
      shapePool = SIMPLE_HOLE_SHAPES;
    }

    const typeStyle = this.getTypeStyle(options.labelEquivalent, options.difficultyBucket);
    let typePlan: HoleType[] = [];

    if (useBase && similarityMode === 'high' &&
      baseConfig.geometry.holes.typePlan &&
      baseConfig.geometry.holes.typePlan.length > 0) {
      typePlan = this.adjustTypePlanForHolecount(
        baseConfig.geometry.holes.typePlan,
        targetHolecount,
        rng
      );
    } else {
      typePlan = this.buildTypePlanForTarget(targetHolecount, typeStyle, rng);
    }
    if (relaxLevel >= 2) {
      typePlan = this.buildRelaxedTypePlanForTarget(targetHolecount, rng);
    }

    const holeTypes = Array.from(new Set(typePlan.length > 0 ? typePlan : ['hole']));
    const spacingScale = this.clamp(1 - relaxLevel * 0.15, 0.35, 1);
    const minDistance = this.clamp(sizeRange.min * 0.9 * spacingScale, 0.2, 0.9);
    const edgeBuffer = this.clamp(sizeRange.max * 0.9 * spacingScale, 0.4, 1.4);
    const shapeVariant = rng.chance(0.3) ? 'distorted' : 'vertical';

    return {
      seed: rng.rangeInt(0, 1_000_000_000),
      geometry: {
        board: {
          shape: boardShape,
          size: boardSize,
          thickness
        },
        holes: {
          count: { min: typePlan.length, max: typePlan.length },
          minDistance,
          edgeBuffer,
          types: holeTypes,
          shapeVariant,
          sizeRange,
          typePlan,
          shapePool,
          mixed2ShapePool: MIXED2_HOLE_SHAPES,
          mixed2Probability: typePlan.includes('mixed2') ? 0.2 : 0
        },
        underBoard: {
          enable: underBoardEnable,
          offset: { x: 0.2, y: 0.1 },
          coverage: underBoardCoverage
        }
      },
      render: {
        camera: baseConfig.render.camera,
        appearance: {
          boardColor,
          materialType,
          underBoardColor
        },
        lighting: baseConfig.render.lighting
      }
    };
  }

  private buildUniformSchedule(
    count: number,
    min: number,
    max: number,
    rng: SeededRandom
  ): number[] {
    const bucketCount = max - min + 1;
    const base = Math.floor(count / bucketCount);
    const remainder = count % bucketCount;
    const schedule: number[] = [];

    for (let value = min; value <= max; value++) {
      for (let i = 0; i < base; i++) {
        schedule.push(value);
      }
    }

    if (remainder > 0) {
      const extras: number[] = [];
      for (let value = min; value <= max; value++) {
        extras.push(value);
      }
      const shuffled = rng.shuffle(extras);
      for (let i = 0; i < remainder; i++) {
        schedule.push(shuffled[i]);
      }
    }

    return rng.shuffle(schedule);
  }

  private buildBucketSchedule(
    count: number,
    rng: SeededRandom
  ): DifficultyBucket[] {
    const buckets: DifficultyBucket[] = [];
    const labels: DifficultyBucket[] = ['easy', 'medium', 'hard'];
    const base = Math.floor(count / labels.length);
    const remainder = count % labels.length;

    for (const label of labels) {
      for (let i = 0; i < base; i++) {
        buckets.push(label);
      }
    }

    if (remainder > 0) {
      const shuffled = rng.shuffle([...labels]);
      for (let i = 0; i < remainder; i++) {
        buckets.push(shuffled[i]);
      }
    }

    return rng.shuffle(buckets);
  }

  private pickNonEquivalentHolecount(
    target: number,
    bucket: DifficultyBucket,
    rng: SeededRandom
  ): number {
    const deltas = bucket === 'easy'
      ? [3, 4, 5]
      : bucket === 'medium'
        ? [2]
        : [1];
    const delta = rng.pick(deltas);
    const candidates: number[] = [];

    if (target - delta >= 0) candidates.push(target - delta);
    if (target + delta <= 8) candidates.push(target + delta);

    if (candidates.length === 0) {
      return this.clamp(target + delta, 0, 8);
    }

    return rng.pick(candidates);
  }

  private getSimilarityMode(
    labelEquivalent: boolean,
    bucket: DifficultyBucket
  ): SimilarityMode {
    if (labelEquivalent) {
      return bucket === 'easy' ? 'high' : bucket === 'medium' ? 'medium' : 'low';
    }
    return bucket === 'easy' ? 'low' : bucket === 'medium' ? 'medium' : 'high';
  }

  private getTypeStyle(
    labelEquivalent: boolean,
    bucket: DifficultyBucket
  ): TypeStyle {
    if (labelEquivalent) {
      return bucket === 'easy' ? 'simple' : bucket === 'medium' ? 'mixed' : 'diverse';
    }
    return bucket === 'hard' ? 'simple' : bucket === 'medium' ? 'mixed' : 'diverse';
  }

  private buildTypePlanForTarget(
    targetHolecount: number,
    style: TypeStyle,
    rng: SeededRandom
  ): HoleType[] {
    if (targetHolecount <= 0) {
      if (style === 'simple') return [];
      const pitCount = rng.rangeInt(0, 2);
      return rng.shuffle(Array.from({ length: pitCount }, () => 'pit'));
    }

    const allowPits = style !== 'simple';
    const allowMixed = style !== 'simple';
    const allowMixed2 = style === 'diverse';

    let mixed2Count = allowMixed2 ? rng.rangeInt(0, Math.floor(targetHolecount / 2)) : 0;
    if (allowMixed2 && mixed2Count === 0 && targetHolecount >= 2 && rng.chance(0.4)) {
      mixed2Count = 1;
    }

    const remaining = targetHolecount - mixed2Count * 2;
    const mixedCount = allowMixed ? rng.rangeInt(0, remaining) : 0;
    const holeCount = remaining - mixedCount;
    const pitCount = allowPits ? rng.rangeInt(0, 2) : 0;

    const plan: HoleType[] = [];
    for (let i = 0; i < holeCount; i++) plan.push('hole');
    for (let i = 0; i < mixedCount; i++) plan.push('mixed');
    for (let i = 0; i < mixed2Count; i++) plan.push('mixed2');
    for (let i = 0; i < pitCount; i++) plan.push('pit');

    return rng.shuffle(plan);
  }

  private buildRelaxedTypePlanForTarget(
    targetHolecount: number,
    rng: SeededRandom
  ): HoleType[] {
    if (targetHolecount <= 0) {
      return [];
    }

    const mixed2Count = Math.floor(targetHolecount / 2);
    const remainder = targetHolecount - mixed2Count * 2;
    const plan: HoleType[] = [];

    for (let i = 0; i < mixed2Count; i++) {
      plan.push('mixed2');
    }
    for (let i = 0; i < remainder; i++) {
      plan.push('hole');
    }

    return rng.shuffle(plan);
  }

  private adjustTypePlanForHolecount(
    plan: HoleType[],
    targetHolecount: number,
    rng: SeededRandom
  ): HoleType[] {
    const adjusted = [...plan];
    let current = this.holecountFromPlan(adjusted);
    let guard = 0;

    while (current !== targetHolecount && guard < 100) {
      if (current < targetHolecount) {
        const pitIndex = adjusted.findIndex(type => type === 'pit');
        if (pitIndex >= 0) {
          adjusted[pitIndex] = 'hole';
        } else {
          const holeIndex = adjusted.findIndex(type => type === 'hole' || type === 'mixed');
          if (holeIndex >= 0) {
            adjusted[holeIndex] = 'mixed2';
          } else {
            adjusted.push('hole');
          }
        }
      } else {
        const holeIndex = adjusted.findIndex(type => type === 'hole' || type === 'mixed');
        if (holeIndex >= 0) {
          adjusted[holeIndex] = 'pit';
        } else {
          const mixed2Index = adjusted.findIndex(type => type === 'mixed2');
          if (mixed2Index >= 0) {
            adjusted[mixed2Index] = 'hole';
          } else if (adjusted.length > 0) {
            adjusted.pop();
          }
        }
      }
      current = this.holecountFromPlan(adjusted);
      guard += 1;
    }

    return rng.shuffle(adjusted);
  }

  private holecountFromPlan(plan: HoleType[]): number {
    return plan.reduce((sum, type) => {
      if (type === 'pit') return sum;
      if (type === 'mixed2') return sum + 2;
      return sum + 1;
    }, 0);
  }

  private countHoleShapes(placements: HolePlacement[]): Record<HoleShape, number> {
    const counts = HOLE_SHAPES.reduce((acc, shape) => {
      acc[shape] = 0;
      return acc;
    }, {} as Record<HoleShape, number>);

    for (const placement of placements) {
      const shape = placement.shape ?? 'circle';
      if (counts[shape] !== undefined) {
        counts[shape] += 1;
      }
    }

    return counts;
  }

  private computeHoleSizeStats(placements: HolePlacement[]): HoleSizeStats {
    const sizes = placements
      .filter(p => p.type !== 'pit')
      .map(p => p.radius);

    if (sizes.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0 };
    }

    const sum = sizes.reduce((acc, value) => acc + value, 0);
    const mean = sum / sizes.length;
    const variance = sizes.reduce((acc, value) => acc + (value - mean) ** 2, 0) / sizes.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...sizes);
    const max = Math.max(...sizes);

    return { mean, std, min, max };
  }

  private computePairScores(
    a: PairFeatures,
    b: PairFeatures,
    labelEquivalent: boolean
  ): { similarity: number; difficulty: number } {
    const holecountDelta = Math.abs(a.holecount - b.holecount);
    const holecountSim = 1 - Math.min(1, holecountDelta / 8);
    const boardShapeSim = a.boardShape === b.boardShape ? 1 : 0;
    const underBoardSim = a.underBoard === b.underBoard ? 1 : 0;
    const materialSim = a.materialType === b.materialType ? 1 : 0;
    const holeTypeSim = this.distributionSimilarity(a.holeTypeCounts, b.holeTypeCounts, HOLE_TYPES);
    const holeShapeSim = this.distributionSimilarity(a.holeShapeCounts, b.holeShapeCounts, HOLE_SHAPES);
    const sizeSim = this.sizeSimilarity(a.holeSizeStats, b.holeSizeStats);
    const colorSim = this.computeColorSimilarity(
      a.boardColor,
      b.boardColor,
      a.materialType,
      b.materialType
    );

    const similarity = (
      0.15 * boardShapeSim +
      0.1 * underBoardSim +
      0.2 * holeTypeSim +
      0.15 * holeShapeSim +
      0.15 * sizeSim +
      0.1 * colorSim +
      0.05 * materialSim +
      0.1 * holecountSim
    );

    const difficulty = labelEquivalent
      ? 1 - similarity
      : similarity * (1 - Math.min(1, holecountDelta / 8));

    return {
      similarity: this.clamp(similarity, 0, 1),
      difficulty: this.clamp(difficulty, 0, 1)
    };
  }

  private difficultyMatchesBucket(
    value: number,
    bucket: DifficultyBucket
  ): boolean {
    const ranges: Record<DifficultyBucket, [number, number]> = {
      easy: [0, 0.45],
      medium: [0.3, 0.7],
      hard: [0.55, 1]
    };
    const [min, max] = ranges[bucket];
    return value >= min && value <= max;
  }

  private distributionSimilarity<T extends string>(
    countsA: Record<T, number>,
    countsB: Record<T, number>,
    keys: T[]
  ): number {
    const totalA = keys.reduce((sum, key) => sum + (countsA[key] || 0), 0);
    const totalB = keys.reduce((sum, key) => sum + (countsB[key] || 0), 0);

    if (totalA === 0 && totalB === 0) return 1;
    if (totalA === 0 || totalB === 0) return 0;

    const l1 = keys.reduce((sum, key) => {
      const a = (countsA[key] || 0) / totalA;
      const b = (countsB[key] || 0) / totalB;
      return sum + Math.abs(a - b);
    }, 0);

    return 1 - Math.min(1, l1 / 2);
  }

  private sizeSimilarity(a: HoleSizeStats, b: HoleSizeStats): number {
    if (a.max === 0 && b.max === 0) return 1;
    const range = Math.max(a.max, b.max, 0.6);
    const meanDelta = Math.abs(a.mean - b.mean);
    return 1 - Math.min(1, meanDelta / range);
  }

  private computeColorSimilarity(
    colorA: string,
    colorB: string,
    materialA: MaterialType,
    materialB: MaterialType
  ): number {
    const colorMaterials: MaterialType[] = ['standard', 'phong'];
    const isColorA = colorMaterials.includes(materialA);
    const isColorB = colorMaterials.includes(materialB);

    if (!isColorA || !isColorB) {
      return materialA === materialB ? 0.7 : 0.4;
    }

    const rgbA = this.parseHexColor(colorA);
    const rgbB = this.parseHexColor(colorB);
    const distance = Math.sqrt(
      (rgbA.r - rgbB.r) ** 2 +
      (rgbA.g - rgbB.g) ** 2 +
      (rgbA.b - rgbB.b) ** 2
    ) / Math.sqrt(3);

    return 1 - this.clamp(distance, 0, 1);
  }

  private parseHexColor(color: string): { r: number; g: number; b: number } {
    const hex = color.replace('#', '');
    const full = hex.length === 3
      ? hex.split('').map(ch => `${ch}${ch}`).join('')
      : hex.padEnd(6, '0').slice(0, 6);
    const r = parseInt(full.slice(0, 2), 16) / 255;
    const g = parseInt(full.slice(2, 4), 16) / 255;
    const b = parseInt(full.slice(4, 6), 16) / 255;
    return { r, g, b };
  }

  private buildSizeRange(
    rng: SeededRandom,
    baseRange?: { min: number; max: number },
    similarityMode?: SimilarityMode
  ): { min: number; max: number } {
    if (!baseRange || !similarityMode) {
      const min = rng.range(0.25, 0.55);
      const max = rng.range(Math.max(min + 0.2, 0.6), 1.2);
      return { min, max };
    }

    const factor = similarityMode === 'high'
      ? rng.range(0.9, 1.1)
      : similarityMode === 'medium'
        ? rng.range(0.75, 1.25)
        : rng.range(0.6, 1.4);
    const min = this.clamp(baseRange.min * factor, 0.2, 1.0);
    const max = this.clamp(baseRange.max * factor, Math.max(min + 0.1, 0.35), 1.4);
    return { min, max };
  }

  private buildBoardSize(
    targetHolecount: number,
    rng: SeededRandom,
    shape: BoardShape
  ): { width: number; height: number } {
    const base = this.clamp(8 + targetHolecount * 0.6, 6, 16);
    let width = rng.range(base, base + 3);
    let height = rng.range(base, base + 3);
    if (shape === 'circle') {
      const size = Math.min(width, height);
      width = size;
      height = size;
    }
    return { width, height };
  }

  private pickBoardShape(
    rng: SeededRandom,
    baseShape?: BoardShape,
    similarityMode?: SimilarityMode
  ): BoardShape {
    if (baseShape && similarityMode === 'low') {
      return this.pickDifferentFrom(BOARD_SHAPES, baseShape, rng);
    }
    if (baseShape && similarityMode === 'high') {
      return baseShape;
    }
    return rng.pick(BOARD_SHAPES);
  }

  private pickMaterialType(
    rng: SeededRandom,
    baseType: MaterialType,
    similarityMode: SimilarityMode
  ): MaterialType {
    const materials: MaterialType[] = ['phong', 'standard', 'wood', 'rock', 'metal'];
    if (similarityMode === 'high') return baseType;
    if (similarityMode === 'low') return this.pickDifferentFrom(materials, baseType, rng);
    return rng.chance(0.5) ? baseType : this.pickDifferentFrom(materials, baseType, rng);
  }

  private pickBoardColor(
    rng: SeededRandom,
    baseColor: string,
    similarityMode: SimilarityMode
  ): string {
    if (similarityMode === 'high') return baseColor;
    if (similarityMode === 'low') return this.pickDifferentFrom(COLOR_PALETTE, baseColor, rng);
    return rng.chance(0.5) ? baseColor : this.pickDifferentFrom(COLOR_PALETTE, baseColor, rng);
  }

  private pickShapePool(
    bucket: DifficultyBucket,
    similarityMode: SimilarityMode
  ): HoleShape[] {
    if (similarityMode === 'high') {
      return bucket === 'hard' ? MEDIUM_HOLE_SHAPES : SIMPLE_HOLE_SHAPES;
    }
    if (similarityMode === 'low') {
      return bucket === 'easy' ? MEDIUM_HOLE_SHAPES : HOLE_SHAPES;
    }
    return bucket === 'hard' ? HOLE_SHAPES : MEDIUM_HOLE_SHAPES;
  }

  private pickDifferentFrom<T>(
    values: T[],
    current: T,
    rng: SeededRandom
  ): T {
    const options = values.filter(value => value !== current);
    if (options.length === 0) return current;
    return rng.pick(options);
  }

  private shouldReuse(
    similarityMode: SimilarityMode,
    rng: SeededRandom,
    chance: number
  ): boolean {
    if (similarityMode === 'high') return true;
    if (similarityMode === 'low') return false;
    return rng.chance(chance);
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  private countHoleAndMixed(placements: HolePlacement[]): number {
    const counts = this.countByType(placements);
    return counts.hole + counts.mixed + counts.mixed2 * 2;
  }

  private countByType(placements: HolePlacement[]): Record<HoleType, number> {
    return placements.reduce(
      (acc, p) => {
        acc[p.type] = (acc[p.type] || 0) + 1;
        return acc;
      },
      { hole: 0, pit: 0, mixed: 0, mixed2: 0 } as Record<HoleType, number>
    );
  }

  private createTimestampFolderName(): string {
    const now = new Date();
    const pad = (n: number) => n.toString().padStart(2, '0');
    const name = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    return name;
  }
}

/**
 * Create a LoopController instance
 */
export function createLoopController(generator: SceneGenerator): LoopController {
  return new LoopController(generator);
}
