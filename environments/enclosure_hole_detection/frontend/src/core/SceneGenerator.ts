// Main Scene Generator - Coordinates all components

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { Brush } from 'three-bvh-csg';
import { 
  SceneConfig, 
  HolePlacement, 
  GenerationStats,
  HoleType
} from '../types';
import { SeededRandom, initGlobalRandom } from '../utils/RandomUtils';
import { countVertices } from '../utils/MathUtils';
import { getCSGProcessor } from '../geometry/CSGProcessor';
import { getBoardFactory } from '../geometry/BoardFactory';
import { HoleSampler, createHoleSampler } from '../geometry/HoleSampler';
import { HoleShaper, createHoleShaper } from '../geometry/HoleShaper';
import { CameraRig, createCameraRig } from '../components/CameraRig';
import { LightingRig, createLightingRig } from '../components/LightingRig';
import { MaterialLibrary, createMaterialLibrary } from '../components/MaterialLibrary';
import { ConfigManager, createConfigManager } from './ConfigManager';

const BACKGROUND_GRADIENTS: Array<[string, string]> = [
  ['#f3efe8', '#d6e4f1'],
  ['#eff4ee', '#d4e7da'],
  ['#f6ebe8', '#e0e9f5'],
  ['#f4f1e8', '#dbe5d8'],
  ['#f2edf6', '#d9e6f0'],
  ['#f6f0e2', '#d3e2e9']
];

const BOARD_GRADIENTS: Array<[string, string]> = [
  ['#f1e2d4', '#d6c3ae'],
  ['#e7f0e4', '#cdd9c6'],
  ['#efe6dd', '#d8c8bc'],
  ['#e6edf1', '#c9d3df'],
  ['#f0e9e1', '#d6cfc4']
];

/**
 * Scene Generator - Main controller class
 * Coordinates board generation, hole cutting, camera, lighting, and rendering
 */
export class SceneGenerator {
  private scene: THREE.Scene;
  private renderer: THREE.WebGLRenderer;
  private environmentMap: THREE.Texture | null = null;
  private cameraRig: CameraRig;
  private lightingRig: LightingRig;
  private materialLibrary: MaterialLibrary;
  private configManager: ConfigManager;
  private random: SeededRandom;
  private boardMaterial?: THREE.Material;
  private backgroundCanvas?: HTMLCanvasElement;
  private backgroundTexture?: THREE.CanvasTexture;
  private boardCanvas?: HTMLCanvasElement;
  private boardTexture?: THREE.CanvasTexture;
  
  private boardGroup: THREE.Group;
  private currentPlacements: HolePlacement[] = [];
  private holeShaper: HoleShaper | null = null;
  private mixedCoverApplied: Set<number> = new Set();
  private stats: GenerationStats = {
    holesGenerated: 0,
    vertexCount: 0,
    renderTimeMs: 0
  };

  constructor(container: HTMLElement, config?: Partial<SceneConfig>) {
    // Initialize configuration
    this.configManager = createConfigManager(config);
    const fullConfig = this.configManager.getConfig();
    
    // Initialize random generator
    this.random = initGlobalRandom(fullConfig.seed);
    this.configManager.setRandom(this.random);
    
    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xd0d5dd); // Placeholder, replaced per-scene
    
    // Initialize renderer
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      preserveDrawingBuffer: true // Required for image export
    });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    // Add a neutral environment map for PBR reflections (metal needs this)
    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    this.environmentMap = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = this.environmentMap;
    pmremGenerator.dispose();
    
    // Initialize camera
    const aspect = container.clientWidth / container.clientHeight;
    this.cameraRig = createCameraRig(aspect);
    this.cameraRig.setParams(fullConfig.render.camera);
    
    // Initialize lighting
    this.lightingRig = createLightingRig();
    this.lightingRig.setParams(fullConfig.render.lighting);
    this.scene.add(this.lightingRig.getGroup());
    
    // Initialize materials
    this.materialLibrary = createMaterialLibrary(this.random);
    this.materialLibrary.setOnTextureLoad(() => this.render());
    this.materialLibrary.preloadTextureSets();
    
    // Create board group
    this.boardGroup = new THREE.Group();
    this.boardGroup.name = 'BoardGroup';
    this.scene.add(this.boardGroup);
    
    // Add coordinate helpers (optional, for debugging)
    // this.addHelpers();
    
    // Handle resize
    window.addEventListener('resize', () => this.onResize(container));
  }

  /**
   * Generate the scene based on current configuration
   */
  generate(config?: Partial<SceneConfig>): GenerationStats {
    const startTime = performance.now();
    
    // Update config if provided
    if (config) {
      this.configManager.updateConfig(config);
    }
    
    const fullConfig = this.configManager.getConfig();
    
    // Reset random with seed
    this.random.reset(fullConfig.seed);
    this.mixedCoverApplied.clear();

    this.updateBackgroundGradient(fullConfig.seed);
    
    // Clear existing board
    this.clearBoard();
    
    // Generate new board with holes
    this.generateBoard(fullConfig);
    
    // Update camera and lighting
    this.cameraRig.setParams(fullConfig.render.camera);
    this.lightingRig.setParams(fullConfig.render.lighting);
    
    // Update shadow camera based on board size (but don't override user's camera distance)
    const { board } = fullConfig.geometry;
    this.lightingRig.updateShadowCamera(board.size.width, board.size.height);
    
    // Only fit camera if distance would make board too small, otherwise respect user's distance
    const minDistance = this.cameraRig.getMinDistanceForBoard(board.size.width, board.size.height);
    if (fullConfig.render.camera.distance < minDistance) {
      this.cameraRig.setParams({ distance: minDistance });
    }
    
    // Render
    this.render();
    
    // Calculate stats
    const endTime = performance.now();
    this.stats = {
      holesGenerated: this.currentPlacements.length,
      vertexCount: countVertices(this.boardGroup),
      renderTimeMs: Math.round(endTime - startTime)
    };
    
    return this.stats;
  }

  /**
   * Generate the board mesh with holes
   */
  private generateBoard(config: SceneConfig): void {
    const { geometry, render } = config;
    const { board, holes, underBoard } = geometry;
    
    // Create materials
    const { boardMaterial, underBoardMaterial } = this.materialLibrary.createFromParams(
      render.appearance
    );
    this.boardMaterial = boardMaterial;
    this.updateBoardGradient(boardMaterial, config.seed);
    
    // Get factories
    const boardFactory = getBoardFactory();
    const csgProcessor = getCSGProcessor();
    
    // Create base board brush
    let boardBrush = boardFactory.createBoardBrush(board, boardMaterial, this.random);
    
    // Sample hole positions
    const holeSampler: HoleSampler = createHoleSampler(this.random);
    this.currentPlacements = holeSampler.sampleHoles(board, holes);
    
    // Create hole cutters (store holeShaper for later use in addPitBottoms)
    this.holeShaper = createHoleShaper(this.random);
    const cutters: Brush[] = this.holeShaper.createTypedCutters(
      this.currentPlacements,
      holes.shapeVariant,
      board.thickness
    );
    
    // Perform CSG operations
    if (cutters.length > 0) {
      boardBrush = csgProcessor.subtractMultiple(boardBrush, cutters);
      
      // Dispose cutters
      csgProcessor.disposeBrushes(cutters);
    }
    
    // Convert to mesh
    const boardMesh = csgProcessor.brushToMesh(boardBrush, boardMaterial);
    boardMesh.castShadow = true;
    boardMesh.receiveShadow = true;
    boardMesh.name = 'MainBoard';
    this.boardGroup.add(boardMesh);
    
    // Add under-board if enabled
    if (underBoard.enable) {
      this.addUnderBoard(config, underBoardMaterial);
    }
    
    // Add mixed covers regardless of under-board so mixed is visible
    this.addMixedHoleCovers(config);
    this.addMixed2HoleCovers(config);
  }

  /**
   * Add under-board layer
   * Under-board is offset in X/Y based on coverage setting
   * It shows through holes but NOT through pits (pits have their own bottom)
   */
  private addUnderBoard(config: SceneConfig, material: THREE.Material): void {
    const { geometry } = config;
    const { board, underBoard } = geometry;

    const boardFactory = getBoardFactory();
    
    // Calculate dynamic offset based on coverage type
    let offsetX: number;
    let offsetY: number;
    const baseOffset = Math.max(board.size.width, board.size.height) * 0.15;
    
    switch (underBoard.coverage) {
      case 'full':
        // Minimal offset, mostly aligned
        offsetX = underBoard.offset.x * 0.5;
        offsetY = underBoard.offset.y * 0.5;
        break;
      case 'partial':
        // Moderate offset, partially visible
        offsetX = baseOffset * this.random.range(0.5, 1.5);
        offsetY = baseOffset * this.random.range(0.3, 1.0);
        break;
      case 'random':
        // Large random offset, very visible
        const angle = this.random.angle();
        const distance = baseOffset * this.random.range(1.0, 2.5);
        offsetX = Math.cos(angle) * distance;
        offsetY = Math.sin(angle) * distance;
        break;
      default:
        offsetX = underBoard.offset.x;
        offsetY = underBoard.offset.y;
    }
    
    // Z offset - ensure clear separation
    const offsetZ = board.thickness * 1.5;
    
    // Create a copy of board geometry for under-board
    const underBoardGeometry = boardFactory.createBoard(board, this.random);
    const underBoardMesh = new THREE.Mesh(underBoardGeometry, material);
    
    // Position under-board with dynamic offset
    underBoardMesh.position.set(offsetX, offsetY, -offsetZ);
    underBoardMesh.receiveShadow = true;
    underBoardMesh.castShadow = true;
    underBoardMesh.name = 'UnderBoard';
    this.boardGroup.add(underBoardMesh);
    
    // Add pit bottom fills (same color as main board, to cover under-board showing through)
    this.addPitBottoms(config);
  }

  /**
   * Add bottom fills for pit holes
   * These are thin discs at the pit bottom to ensure pit has visible bottom
   * Uses the SAME 2D shape as the pit cutter for consistency
   */
  private addPitBottoms(config: SceneConfig): void {
    const { geometry } = config;
    const { board } = geometry;
    
    // Get the main board material
    const boardMaterial = this.boardMaterial;
    if (!boardMaterial) {
      return;
    }
    
    // Find pit holes
    const pitHoles = this.currentPlacements.filter(p => p.type === 'pit');
    
    // Need holeShaper to get the actual shape
    if (!this.holeShaper) {
      console.warn('HoleShaper not available for pit bottoms');
      return;
    }
    
    for (const pit of pitHoles) {
      // Get the SAME shape as was used for the pit cutter
      const holeShape = this.holeShaper.getHoleShape(pit);
      
      // Scale down slightly to fit inside pit walls
      const scaledShape = this.scaleShape(holeShape, 0.97);
      
      // Create a thin extruded disc using the same shape
      const discGeometry = new THREE.ExtrudeGeometry(scaledShape, {
        depth: 0.03,
        bevelEnabled: false
      });
      discGeometry.translate(0, 0, -0.015);
      
      const discMesh = new THREE.Mesh(discGeometry, boardMaterial);
      // Position slightly above the theoretical pit bottom to avoid z-fighting
      // Pit cuts from top (boardThickness/2) down by pit.depth
      const pitBottomZ = board.thickness / 2 - pit.depth + 0.02;
      discMesh.position.set(pit.x, pit.y, pitBottomZ);
      discMesh.receiveShadow = true;
      discMesh.castShadow = true;
      discMesh.name = 'PitBottom';
      this.boardGroup.add(discMesh);
    }
  }

  /**
   * Scale a 2D shape uniformly around origin
   */
  private scaleShape(shape: THREE.Shape, scale: number): THREE.Shape {
    const points = shape.getPoints(32);
    const scaledShape = new THREE.Shape();
    
    for (let i = 0; i < points.length; i++) {
      const x = points[i].x * scale;
      const y = points[i].y * scale;
      
      if (i === 0) scaledShape.moveTo(x, y);
      else scaledShape.lineTo(x, y);
    }
    scaledShape.closePath();
    return scaledShape;
  }

  /**
   * Add partial covers for mixed holes using RECTANGULAR mask
   * Mixed hole = part is through-hole (open), part has bottom (like pit)
   * Coverage ratio is random between 0.25 and 0.5
   * 
   * ROBUST APPROACH:
   * - Get the actual hole shape's bounding box
   * - Create a rectangle that covers 'coverage' portion of the area
   * - Intersect rectangle with actual hole shape
   * - This ensures only ONE connected open region
  */
  private addMixedHoleCovers(config: SceneConfig): void {
    const { geometry } = config;
    const { board } = geometry;
    
    // Use main board material for the pit portion
    const boardMaterial = this.boardMaterial;
    if (!boardMaterial) {
      return;
    }
    
    // Need holeShaper to get the actual shape
    if (!this.holeShaper) {
      console.warn('HoleShaper not available for mixed hole covers');
      return;
    }
    
    for (let i = 0; i < this.currentPlacements.length; i++) {
      const hole = this.currentPlacements[i];
      if (hole.type !== 'mixed') continue;

      // Get the actual hole shape
      const holeShape = this.holeShaper.getHoleShape(hole);
      const holePoints = holeShape.getPoints(128);
      
      // Calculate bounding box of actual shape
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      for (const p of holePoints) {
        minX = Math.min(minX, p.x);
        maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
      }
      
      const boxHeight = maxY - minY;
      
      // Random coverage between 25% and 75%
      const coverage = hole.mixedCoverage ?? this.random.range(0.25, 0.5);
      
      const rectMinY = minY;
      const rectMaxY = minY + boxHeight * coverage;
      const rectMinX = minX;
      const rectMaxX = maxX;
      
      // Intersect the actual hole shape with rectangle
      const finalCoverShape = this.clipShapeWithRect(holeShape, rectMinX, rectMaxX, rectMinY, rectMaxY);
      if (finalCoverShape.getPoints(8).length === 0) {
        continue;
      }
      
      // The pit portion depth
      const pitDepth = board.thickness * 0.5;
      
      const coverGeometry = new THREE.ExtrudeGeometry(finalCoverShape, {
        depth: 0.05,
        bevelEnabled: false
      });
      
      coverGeometry.translate(0, 0, -0.025);
      
      const coverMesh = new THREE.Mesh(coverGeometry, boardMaterial);
      const pitBottomZ = board.thickness / 2 - pitDepth + 0.02;
      coverMesh.position.set(hole.x, hole.y, pitBottomZ);
      coverMesh.receiveShadow = true;
      coverMesh.castShadow = true;
      coverMesh.name = 'MixedHoleCover';
      this.boardGroup.add(coverMesh);
      this.mixedCoverApplied.add(i);
    }
  }

  /**
   * Add centered covers for mixed2 holes using a fixed coverage ratio
   */
  private addMixed2HoleCovers(config: SceneConfig): void {
    const { geometry } = config;
    const { board } = geometry;

    const boardMaterial = this.boardMaterial;
    if (!boardMaterial) {
      return;
    }

    if (!this.holeShaper) {
      console.warn('HoleShaper not available for mixed2 hole covers');
      return;
    }

    for (let i = 0; i < this.currentPlacements.length; i++) {
      const hole = this.currentPlacements[i];
      if (hole.type !== 'mixed2') continue;

      const holeShape = this.holeShaper.getHoleShape(hole);
      const holePoints = holeShape.getPoints(128);

      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      for (const p of holePoints) {
        minX = Math.min(minX, p.x);
        maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
      }

      const boxHeight = maxY - minY;
      const centerY = (minY + maxY) / 2;
      const coverage = hole.mixedCoverage ?? 0.25;
      const coverHeight = boxHeight * coverage;

      const rectMinY = centerY - coverHeight / 2;
      const rectMaxY = centerY + coverHeight / 2;
      const rectMinX = minX;
      const rectMaxX = maxX;

      const finalCoverShape = this.clipShapeWithRect(holeShape, rectMinX, rectMaxX, rectMinY, rectMaxY);
      if (finalCoverShape.getPoints(8).length === 0) {
        continue;
      }

      const pitDepth = board.thickness * 0.5;

      const coverGeometry = new THREE.ExtrudeGeometry(finalCoverShape, {
        depth: 0.05,
        bevelEnabled: false
      });
      coverGeometry.translate(0, 0, -0.025);

      const coverMesh = new THREE.Mesh(coverGeometry, boardMaterial);
      const pitBottomZ = board.thickness / 2 - pitDepth + 0.02;
      coverMesh.position.set(hole.x, hole.y, pitBottomZ);
      coverMesh.receiveShadow = true;
      coverMesh.castShadow = true;
      coverMesh.name = 'Mixed2HoleCover';
      this.boardGroup.add(coverMesh);
      this.mixedCoverApplied.add(i);
    }
  }

  /**
   * Intersect a shape with an axis-aligned rectangle using polygon clipping
   */
  private clipShapeWithRect(
    shape: THREE.Shape,
    minX: number,
    maxX: number,
    minY: number,
    maxY: number
  ): THREE.Shape {
    const points = shape.getPoints(128);
    const clipped = this.clipPolygonWithRect(points, minX, maxX, minY, maxY);
    const result = new THREE.Shape();
    
    const deduped = this.dedupePoints(clipped, 1e-4);
    if (deduped.length === 0) {
      return result;
    }
    
    result.moveTo(deduped[0].x, deduped[0].y);
    for (let i = 1; i < deduped.length; i++) {
      result.lineTo(deduped[i].x, deduped[i].y);
    }
    result.closePath();
    return result;
  }

  /**
   * Intersect a shape with a rotated rectangle (rotate shape to local frame, clip, rotate back)
   */
  private clipShapeWithRectRotated(
    shape: THREE.Shape,
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
    angle: number,
    cx: number,
    cy: number
  ): THREE.Shape {
    const points = shape.getPoints(128).map(p => {
      const dx = p.x - cx;
      const dy = p.y - cy;
      const cos = Math.cos(-angle);
      const sin = Math.sin(-angle);
      return new THREE.Vector2(
        dx * cos - dy * sin + cx,
        dx * sin + dy * cos + cy
      );
    });

    const clipped = this.clipPolygonWithRect(points, minX, maxX, minY, maxY);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const rotatedBack = clipped.map(p => {
      const dx = p.x - cx;
      const dy = p.y - cy;
      return new THREE.Vector2(
        dx * cos - dy * sin + cx,
        dx * sin + dy * cos + cy
      );
    });

    const result = new THREE.Shape();
    const deduped = this.dedupePoints(rotatedBack, 1e-4);
    if (deduped.length === 0) {
      return result;
    }

    result.moveTo(deduped[0].x, deduped[0].y);
    for (let i = 1; i < deduped.length; i++) {
      result.lineTo(deduped[i].x, deduped[i].y);
    }
    result.closePath();
    return result;
  }

  private clipPolygonWithRect(
    points: THREE.Vector2[],
    minX: number,
    maxX: number,
    minY: number,
    maxY: number
  ): THREE.Vector2[] {
    let output = points.map(p => p.clone());
    
    output = this.clipPolygon(
      output,
      p => p.x >= minX,
      (s, e) => this.intersectWithVertical(s, e, minX)
    );
    output = this.clipPolygon(
      output,
      p => p.x <= maxX,
      (s, e) => this.intersectWithVertical(s, e, maxX)
    );
    output = this.clipPolygon(
      output,
      p => p.y >= minY,
      (s, e) => this.intersectWithHorizontal(s, e, minY)
    );
    output = this.clipPolygon(
      output,
      p => p.y <= maxY,
      (s, e) => this.intersectWithHorizontal(s, e, maxY)
    );
    
    return output;
  }

  private clipPolygon(
    points: THREE.Vector2[],
    inside: (p: THREE.Vector2) => boolean,
    intersect: (s: THREE.Vector2, e: THREE.Vector2) => THREE.Vector2 | null
  ): THREE.Vector2[] {
    const output: THREE.Vector2[] = [];
    if (points.length === 0) return output;
    
    let prev = points[points.length - 1];
    let prevInside = inside(prev);
    
    for (const curr of points) {
      const currInside = inside(curr);
      
      if (currInside) {
        if (!prevInside) {
          const ip = intersect(prev, curr);
          if (ip) output.push(ip);
        }
        output.push(curr.clone());
      } else if (prevInside) {
        const ip = intersect(prev, curr);
        if (ip) output.push(ip);
      }
      
      prev = curr;
      prevInside = currInside;
    }
    
    return output;
  }

  private intersectWithVertical(
    s: THREE.Vector2,
    e: THREE.Vector2,
    x: number
  ): THREE.Vector2 | null {
    const dx = e.x - s.x;
    if (Math.abs(dx) < 1e-8) return null;
    const t = (x - s.x) / dx;
    if (t < 0 || t > 1) return null;
    return new THREE.Vector2(x, s.y + (e.y - s.y) * t);
  }

  private intersectWithHorizontal(
    s: THREE.Vector2,
    e: THREE.Vector2,
    y: number
  ): THREE.Vector2 | null {
    const dy = e.y - s.y;
    if (Math.abs(dy) < 1e-8) return null;
    const t = (y - s.y) / dy;
    if (t < 0 || t > 1) return null;
    return new THREE.Vector2(s.x + (e.x - s.x) * t, y);
  }

  private dedupePoints(points: THREE.Vector2[], eps: number): THREE.Vector2[] {
    const result: THREE.Vector2[] = [];
    
    for (const p of points) {
      const last = result[result.length - 1];
      if (!last || Math.abs(last.x - p.x) > eps || Math.abs(last.y - p.y) > eps) {
        result.push(p);
      }
    }
    
    if (result.length > 1) {
      const first = result[0];
      const last = result[result.length - 1];
      if (Math.abs(first.x - last.x) < eps && Math.abs(first.y - last.y) < eps) {
        result.pop();
      }
    }
    
    return result;
  }

  private updateBackgroundGradient(seed: number): void {
    const rng = new SeededRandom(`${seed}_bg`);
    const palette = rng.pick(BACKGROUND_GRADIENTS);
    const { canvas, texture } = this.updateGradientTexture(
      this.backgroundCanvas,
      this.backgroundTexture,
      palette,
      rng
    );

    this.backgroundCanvas = canvas;
    this.backgroundTexture = texture;
    this.scene.background = texture;
  }

  private updateBoardGradient(material: THREE.Material, seed: number): void {
    if (!(material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshPhongMaterial)) {
      return;
    }

    const rng = new SeededRandom(`${seed}_board`);
    const palette = rng.pick(BOARD_GRADIENTS);
    const { canvas, texture } = this.updateGradientTexture(
      this.boardCanvas,
      this.boardTexture,
      palette,
      rng
    );

    this.boardCanvas = canvas;
    this.boardTexture = texture;

    if (material.map) {
      material.emissiveMap = texture;
      material.emissive = new THREE.Color(0x222222);
      if (material instanceof THREE.MeshStandardMaterial) {
        material.emissiveIntensity = 0.35;
      }
    } else {
      material.map = texture;
      material.color = new THREE.Color(0xffffff);
    }

    material.needsUpdate = true;
  }

  private updateGradientTexture(
    canvas: HTMLCanvasElement | undefined,
    texture: THREE.CanvasTexture | undefined,
    colors: [string, string],
    rng: SeededRandom
  ): { canvas: HTMLCanvasElement; texture: THREE.CanvasTexture } {
    const size = 512;
    const targetCanvas = canvas ?? document.createElement('canvas');
    targetCanvas.width = size;
    targetCanvas.height = size;

    const ctx = targetCanvas.getContext('2d');
    if (ctx) {
      const angle = rng.range(0, Math.PI * 2);
      const x0 = (0.5 - 0.5 * Math.cos(angle)) * size;
      const y0 = (0.5 - 0.5 * Math.sin(angle)) * size;
      const x1 = (0.5 + 0.5 * Math.cos(angle)) * size;
      const y1 = (0.5 + 0.5 * Math.sin(angle)) * size;
      const gradient = ctx.createLinearGradient(x0, y0, x1, y1);

      gradient.addColorStop(0, colors[0]);
      gradient.addColorStop(1, colors[1]);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);
    }

    const targetTexture = texture ?? new THREE.CanvasTexture(targetCanvas);
    targetTexture.colorSpace = THREE.SRGBColorSpace;
    targetTexture.needsUpdate = true;
    return { canvas: targetCanvas, texture: targetTexture };
  }

  /**
   * Clear the board group
   */
  private clearBoard(): void {
    while (this.boardGroup.children.length > 0) {
      const child = this.boardGroup.children[0];
      this.boardGroup.remove(child);
      
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach(m => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    }
    
    this.currentPlacements = [];
  }

  /**
   * Render the scene
   */
  render(): void {
    this.renderer.render(this.scene, this.cameraRig.getCamera());
  }

  /**
   * Clamp camera distance for exports so framing isn't too far
   */
  prepareExportFraming(
    boardWidth: number,
    boardHeight: number,
    maxDistanceMultiplier: number = 1.2
  ): void {
    const minDistance = this.cameraRig.getMinDistanceForBoard(boardWidth, boardHeight);
    const maxDistance = minDistance * maxDistanceMultiplier;
    const currentDistance = this.cameraRig.getParams().distance;
    
    if (currentDistance > maxDistance) {
      this.cameraRig.setParams({ distance: maxDistance });
      this.render();
    }
  }

  /**
   * Handle container resize
   */
  private onResize(container: HTMLElement): void {
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    this.renderer.setSize(width, height);
    this.cameraRig.setAspect(width / height);
    this.render();
  }

  /**
   * Get the canvas element
   */
  getCanvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /**
   * Get current configuration
   */
  getConfig(): SceneConfig {
    return this.configManager.getConfig();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SceneConfig>): void {
    this.configManager.updateConfig(config);
  }

  /**
   * Get current generation stats
   */
  getStats(): GenerationStats {
    return { ...this.stats };
  }

  /**
   * Get current hole placements
   */
  getHolePlacements(): HolePlacement[] {
    return [...this.currentPlacements];
  }

  /**
   * Get effective hole type counts based on actual mixed cover placement
   */
  getEffectiveHoleTypeCounts(): Record<HoleType, number> {
    const counts: Record<HoleType, number> = { hole: 0, pit: 0, mixed: 0, mixed2: 0 };
    
    for (let i = 0; i < this.currentPlacements.length; i++) {
      const placement = this.currentPlacements[i];
      if ((placement.type === 'mixed' || placement.type === 'mixed2') && !this.mixedCoverApplied.has(i)) {
        counts.hole += 1;
      } else {
        counts[placement.type] += 1;
      }
    }
    
    return counts;
  }

  /**
   * Randomize only the seed (for Static Reasoning mode)
   */
  randomizeSeed(): number {
    const newSeed = Math.floor(Math.random() * 1000000);
    this.configManager.setSeed(newSeed);
    return newSeed;
  }

  /**
   * Randomize render parameters only (for Invariance mode)
   */
  randomizeRender(): void {
    this.configManager.randomizeRender();
  }

  /**
   * Randomize all parameters
   */
  randomizeAll(): void {
    this.configManager.randomizeAll();
  }

  /**
   * Preview camera parameters without committing them to configuration.
   */
  previewCamera(params: Partial<import('../types').CameraParams>): void {
    this.cameraRig.setParams(params);
    this.render();
  }

  /**
   * Persist the current camera pose into configuration.
   */
  commitCamera(): void {
    this.configManager.updateRender({ camera: this.cameraRig.getParams() });
  }

  /**
   * Set camera parameters
   */
  setCamera(params: Partial<import('../types').CameraParams>): void {
    this.previewCamera(params);
    this.commitCamera();
  }

  /**
   * Get camera rig for direct control
   */
  getCameraRig(): CameraRig {
    return this.cameraRig;
  }

  /**
   * Get lighting rig for direct control
   */
  getLightingRig(): LightingRig {
    return this.lightingRig;
  }

  /**
   * Add coordinate helpers (for debugging)
   */
  addHelpers(): void {
    const axesHelper = new THREE.AxesHelper(5);
    this.scene.add(axesHelper);
    
    const gridHelper = new THREE.GridHelper(20, 20);
    gridHelper.rotation.x = Math.PI / 2;
    this.scene.add(gridHelper);
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.clearBoard();
    this.lightingRig.dispose();
    this.materialLibrary.dispose();
    if (this.environmentMap) {
      this.environmentMap.dispose();
      this.environmentMap = null;
    }
    this.renderer.dispose();
  }
}

/**
 * Create a SceneGenerator instance
 */
export function createSceneGenerator(
  container: HTMLElement,
  config?: Partial<SceneConfig>
): SceneGenerator {
  return new SceneGenerator(container, config);
}
