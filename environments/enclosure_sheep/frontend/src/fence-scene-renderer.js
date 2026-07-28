/**
 * fence-scene-renderer.js
 * Core Three.js rendering: fence + grass ground + sheep + camera controls
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FenceGenerator } from './fence-generator.js';
import { SheepPlacer } from './sheep-placer.js';
import { RegionAnalyzer } from './region-analyzer.js';

const MIN_FENCE_DISTANCE = 1.45;
const MIN_SHEEP_CENTER_DISTANCE = 1.25;
const SCENE_BACKGROUND_COLOR = 0x1a2236;
const SCENE_GROUND_SHADE = 0x2a335a;
const DEFAULT_GROUND_SIZE = 80;
// Ground must extend well past the scene so tilted cameras (up to 45°) don't
// reveal the dark background at the far edges of the frame.
const GROUND_MIN_SIZE = 80;
const GROUND_MARGIN = 40;
const GROUND_REPEAT_SCALE = 6;
const SHEEP_FOOTPRINT_RADIUS = 1.05;
const FENCE_FOOTPRINT_PADDING = 0.75;
const CAMERA_OBJECT_HEIGHT = 2.25;
const CAMERA_FRAME_MARGIN = 0.03;

export class FenceSceneRenderer {

  constructor(container, options = {}) {
    this.container = container;
    this.width = options.width || 1024;
    this.height = options.height || 1024;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(SCENE_BACKGROUND_COLOR);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.appendChild(this.renderer.domElement);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.1, 200);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;

    // State
    this.fenceLayers = [];
    this.sheepData = [];
    this.metadata = null;
    this.groundMesh = null;
    this._currentGroundTexture = 'grass';

    this._setupLights();
    this._createGround('grass', { size: DEFAULT_GROUND_SIZE, center: { x: 0, z: 0 } });
    this._animate();
  }

  _setupLights() {
    // Ambient
    const ambient = new THREE.AmbientLight(0xffffff, 0.52);
    this.scene.add(ambient);
    this.ambientLight = ambient;

    // Hemisphere light for sky/ground color
    const hemi = new THREE.HemisphereLight(0xffffff, SCENE_GROUND_SHADE, 0.22);
    this.scene.add(hemi);
    this.hemiLight = hemi;

    // Directional sun
    const sun = new THREE.DirectionalLight(0xfff5e0, 1.2);
    sun.position.set(10, 15, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.left = -20;
    sun.shadow.camera.right = 20;
    sun.shadow.camera.top = 20;
    sun.shadow.camera.bottom = -20;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 50;
    sun.shadow.bias = -0.001;
    const sunTarget = new THREE.Object3D();
    sunTarget.position.set(0, 0, 0);
    this.scene.add(sunTarget);
    sun.target = sunTarget;
    this.scene.add(sun);
    this.sunLight = sun;
    this.sunTarget = sunTarget;

    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-8, 6, -10);
    this.scene.add(fill);
    this.fillLight = fill;
  }

  _removeGround() {
    if (!this.groundMesh) return;
    this.scene.remove(this.groundMesh);
    this.groundMesh.geometry?.dispose();
    this.groundMesh.material?.map?.dispose();
    this.groundMesh.material?.dispose();
    this.groundMesh = null;
  }

  _createGround(texture = 'grass', options = {}) {
    const {
      size = DEFAULT_GROUND_SIZE,
      center = { x: 0, z: 0 }
    } = options;
    const groundSize = Math.max(GROUND_MIN_SIZE, size);
    const groundGeom = new THREE.PlaneGeometry(groundSize, groundSize, 40, 40);

    // Create procedural grass texture
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Base grass color
    const colors = {
      grass: { base: '#5a8a3c', detail: '#4a7a2e', accent: '#6b9b4d' },
      dirt: { base: '#8B7355', detail: '#7a6345', accent: '#9b8365' },
      snow: { base: '#e8e8f0', detail: '#d0d0e0', accent: '#f0f0ff' }
    };
    const c = colors[texture] || colors.grass;

    ctx.fillStyle = c.base;
    ctx.fillRect(0, 0, 512, 512);

    // Add detail noise
    for (let i = 0; i < 5000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      ctx.fillStyle = Math.random() > 0.5 ? c.detail : c.accent;
      ctx.fillRect(x, y, 2 + Math.random() * 3, 1 + Math.random() * 2);
    }

    const groundTexture = new THREE.CanvasTexture(canvas);
    groundTexture.wrapS = groundTexture.wrapT = THREE.RepeatWrapping;
    const repeatCount = Math.max(2, groundSize / GROUND_REPEAT_SCALE);
    groundTexture.repeat.set(repeatCount, repeatCount);

    const groundMat = new THREE.MeshStandardMaterial({
      map: groundTexture,
      roughness: 0.9,
      metalness: 0
    });
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(center.x, 0, center.z);
    ground.receiveShadow = true;
    ground.userData = { type: 'ground' };

    // Add subtle height variation
    const positions = groundGeom.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      positions.setZ(i, (Math.sin(x * 0.5) * Math.cos(y * 0.5) * 0.1));
    }
    groundGeom.computeVertexNormals();

    this._removeGround();
    this.scene.add(ground);
    this.groundMesh = ground;
    this._currentGroundTexture = texture;
  }

  _updateGroundForBounds(texture, bounds) {
    this._createGround(texture, {
      size: Math.max(GROUND_MIN_SIZE, Math.max(bounds.width, bounds.depth) + GROUND_MARGIN * 2),
      center: { x: bounds.centerX, z: bounds.centerZ }
    });
  }

  _updateLightsForBounds(bounds) {
    if (this.sunTarget) {
      this.sunTarget.position.set(bounds.centerX, 0, bounds.centerZ);
    }
    if (this.sunLight) {
      this.sunLight.position.set(bounds.centerX + 10, 15, bounds.centerZ + 8);
      const shadowRadius = Math.max(18, bounds.radius + 6);
      this.sunLight.shadow.camera.left = -shadowRadius;
      this.sunLight.shadow.camera.right = shadowRadius;
      this.sunLight.shadow.camera.top = shadowRadius;
      this.sunLight.shadow.camera.bottom = -shadowRadius;
      this.sunLight.shadow.camera.far = Math.max(60, shadowRadius * 3.5);
      this.sunLight.shadow.camera.updateProjectionMatrix();
    }
    if (this.fillLight) {
      this.fillLight.position.set(bounds.centerX - 8, 6, bounds.centerZ - 10);
    }
  }

  /**
   * Compute the maximum safe placement distance from center for the given camera.
   * Sheep beyond this distance risk having labels clipped by the image edge.
   * Returns {maxX, maxZ} — half-extents of the safe placement rectangle.
   */
  static getViewableExtent(cameraAngle, sceneRadius) {
    const dist = sceneRadius * 2.2;
    const fov = 50;
    const halfFovRad = (fov / 2) * Math.PI / 180; // ~0.4363 rad
    const tanHalf = Math.tan(halfFovRad); // ~0.4663

    // Label sprites extend ~1.3 units above sheep; from angled views they
    // project further horizontally.  Reserve a generous margin (2.0 units)
    // so labels are never clipped.
    const labelMargin = 2.0;

    const tiltDeg = FenceSceneRenderer._cameraTiltDegrees(cameraAngle);
    const h = tiltDeg === 0 ? dist * 1.5 : dist * Math.cos(tiltDeg * Math.PI / 180);
    let extent = h * tanHalf - labelMargin;
    // Never let extent go below the scene radius itself (fences must fit)
    extent = Math.max(extent, sceneRadius * 0.85);
    return extent;
  }

  /**
   * Clamp sheep positions so they stay within the camera's visible area.
   * Mutates the sheep array in place.
   */
  static clampSheepToView(sheepPositions, viewExtent) {
    for (const s of sheepPositions) {
      s.x = Math.max(-viewExtent, Math.min(viewExtent, s.x));
      s.z = Math.max(-viewExtent, Math.min(viewExtent, s.z));
    }
  }

  /**
   * After clamping, re-validate sheep positions against fences and push away
   * any sheep that ended up too close (e.g. due to view clamping).
   * @param {Array} sheepPositions
   * @param {Array<Array>} fenceLayers - all fence polygons
   * @param {number} minFenceDist - minimum distance to any fence segment
   */
  static enforceMinFenceDist(sheepPositions, fenceLayers, minFenceDist = 1.8) {
    for (const s of sheepPositions) {
      const fixed = SheepPlacer._pushAwayFromFences(
        { x: s.x, z: s.z }, fenceLayers, minFenceDist
      );
      s.x = fixed.x;
      s.z = fixed.z;
    }
  }

  /**
   * Same as enforceMinFenceDist but for partition cells (all cell edges).
   */
  static enforceMinCellEdgeDist(sheepPositions, cells, minFenceDist = 1.8) {
    for (const s of sheepPositions) {
      if (s.cellIndex >= 0 && s.cellIndex < cells.length) {
        const fixed = SheepPlacer._pushAwayFromPolygonEdges(
          { x: s.x, z: s.z }, cells[s.cellIndex], minFenceDist
        );
        s.x = fixed.x;
        s.z = fixed.z;
      } else {
        // Outside sheep: push away from all cell edges
        for (const cell of cells) {
          const fixed = SheepPlacer._pushAwayFromPolygonEdges(
            { x: s.x, z: s.z }, cell, minFenceDist
          );
          s.x = fixed.x;
          s.z = fixed.z;
        }
      }
    }
  }

  static recalculatePartitionAssignments(sheepPositions, cells) {
    const cellCounts = {};
    for (const sheep of sheepPositions) {
      let newCellIndex = -1;
      for (let cellIndex = 0; cellIndex < cells.length; cellIndex++) {
        if (RegionAnalyzer.pointInPolygon({ x: sheep.x, z: sheep.z }, cells[cellIndex])) {
          newCellIndex = cellIndex;
          break;
        }
      }
      sheep.cellIndex = newCellIndex;
      sheep.isInside = newCellIndex >= 0;
      sheep.cellLabel = newCellIndex >= 0 ? String.fromCharCode(65 + newCellIndex) : 'outside';
      if (newCellIndex >= 0) {
        cellCounts[sheep.cellLabel] = (cellCounts[sheep.cellLabel] || 0) + 1;
      }
    }
    return cellCounts;
  }

  static getCameraSceneRadius(sheepPositions, baseRadius = 0) {
    let radius = baseRadius;
    for (const sheep of sheepPositions) {
      radius = Math.max(radius, Math.hypot(sheep.x, sheep.z) + 2.2);
    }
    return radius;
  }

  static _expandBounds(bounds, x, z, padding = 0) {
    bounds.minX = Math.min(bounds.minX, x - padding);
    bounds.maxX = Math.max(bounds.maxX, x + padding);
    bounds.minZ = Math.min(bounds.minZ, z - padding);
    bounds.maxZ = Math.max(bounds.maxZ, z + padding);
  }

  static _finalizeBounds(bounds) {
    if (!Number.isFinite(bounds.minX) || !Number.isFinite(bounds.minZ)) {
      return {
        minX: -DEFAULT_GROUND_SIZE / 2,
        maxX: DEFAULT_GROUND_SIZE / 2,
        minZ: -DEFAULT_GROUND_SIZE / 2,
        maxZ: DEFAULT_GROUND_SIZE / 2,
        centerX: 0,
        centerZ: 0,
        width: DEFAULT_GROUND_SIZE,
        depth: DEFAULT_GROUND_SIZE,
        radius: DEFAULT_GROUND_SIZE / 2
      };
    }

    const width = Math.max(1, bounds.maxX - bounds.minX);
    const depth = Math.max(1, bounds.maxZ - bounds.minZ);
    return {
      ...bounds,
      centerX: (bounds.minX + bounds.maxX) / 2,
      centerZ: (bounds.minZ + bounds.maxZ) / 2,
      width,
      depth,
      radius: Math.hypot(width / 2, depth / 2)
    };
  }

  static getSceneFootprintBounds({ sheepPositions = [], fenceLayers = null, partitionData = null } = {}) {
    const bounds = {
      minX: Infinity,
      maxX: -Infinity,
      minZ: Infinity,
      maxZ: -Infinity
    };

    for (const sheep of sheepPositions) {
      FenceSceneRenderer._expandBounds(bounds, sheep.x, sheep.z, SHEEP_FOOTPRINT_RADIUS);
    }

    const addPolygon = (polygon, padding = FENCE_FOOTPRINT_PADDING) => {
      for (const vertex of polygon || []) {
        FenceSceneRenderer._expandBounds(bounds, vertex.x, vertex.z, padding);
      }
    };

    if (fenceLayers) {
      for (const layer of fenceLayers) {
        addPolygon(layer);
      }
    }

    if (partitionData?.cells) {
      for (const cell of partitionData.cells) {
        addPolygon(cell);
      }
    }

    if (partitionData?.wallSegments) {
      for (const segment of partitionData.wallSegments) {
        FenceSceneRenderer._expandBounds(bounds, segment.start.x, segment.start.z, FENCE_FOOTPRINT_PADDING);
        FenceSceneRenderer._expandBounds(bounds, segment.end.x, segment.end.z, FENCE_FOOTPRINT_PADDING);
      }
    }

    return FenceSceneRenderer._finalizeBounds(bounds);
  }

  static _appendCameraFootprint(points, x, z, radius, height) {
    const offsets = radius > 0
      ? [
          [0, 0],
          [radius, 0],
          [-radius, 0],
          [0, radius],
          [0, -radius],
          [radius * 0.7, radius * 0.7],
          [radius * 0.7, -radius * 0.7],
          [-radius * 0.7, radius * 0.7],
          [-radius * 0.7, -radius * 0.7]
        ]
      : [[0, 0]];

    for (const [ox, oz] of offsets) {
      points.push(new THREE.Vector3(x + ox, 0, z + oz));
    }
    points.push(new THREE.Vector3(x, height, z));
  }

  static getSceneCameraPoints({ sheepPositions = [], fenceLayers = null, partitionData = null } = {}) {
    const points = [];

    const addPolygon = (polygon, padding = FENCE_FOOTPRINT_PADDING, height = 1.55) => {
      for (let i = 0; i < (polygon?.length || 0); i++) {
        const start = polygon[i];
        const end = polygon[(i + 1) % polygon.length];
        FenceSceneRenderer._appendCameraFootprint(points, start.x, start.z, padding, height);
        FenceSceneRenderer._appendCameraFootprint(
          points,
          (start.x + end.x) / 2,
          (start.z + end.z) / 2,
          padding * 0.8,
          height
        );
      }
    };

    for (const sheep of sheepPositions) {
      FenceSceneRenderer._appendCameraFootprint(
        points,
        sheep.x,
        sheep.z,
        SHEEP_FOOTPRINT_RADIUS,
        CAMERA_OBJECT_HEIGHT
      );
    }

    if (fenceLayers) {
      for (const layer of fenceLayers) {
        addPolygon(layer);
      }
    }

    if (partitionData?.cells) {
      for (const cell of partitionData.cells) {
        addPolygon(cell);
      }
    }

    if (partitionData?.wallSegments) {
      for (const segment of partitionData.wallSegments) {
        FenceSceneRenderer._appendCameraFootprint(
          points,
          segment.start.x,
          segment.start.z,
          FENCE_FOOTPRINT_PADDING,
          1.55
        );
        FenceSceneRenderer._appendCameraFootprint(
          points,
          segment.end.x,
          segment.end.z,
          FENCE_FOOTPRINT_PADDING,
          1.55
        );
        FenceSceneRenderer._appendCameraFootprint(
          points,
          (segment.start.x + segment.end.x) / 2,
          (segment.start.z + segment.end.z) / 2,
          FENCE_FOOTPRINT_PADDING * 0.8,
          1.55
        );
      }
    }

    return points;
  }

  static getCameraFitPoints(bounds, objectHeight = CAMERA_OBJECT_HEIGHT) {
    const points = [];
    const xs = [bounds.minX, bounds.centerX, bounds.maxX];
    const zs = [bounds.minZ, bounds.centerZ, bounds.maxZ];
    const ys = [0, objectHeight];

    for (const x of xs) {
      for (const z of zs) {
        for (const y of ys) {
          points.push(new THREE.Vector3(x, y, z));
        }
      }
    }

    return points;
  }

  static hasMinSheepSpacing(sheepPositions, minDistance = MIN_SHEEP_CENTER_DISTANCE) {
    const minDistanceSq = minDistance * minDistance;
    for (let i = 0; i < sheepPositions.length; i++) {
      for (let j = i + 1; j < sheepPositions.length; j++) {
        const dx = sheepPositions[i].x - sheepPositions[j].x;
        const dz = sheepPositions[i].z - sheepPositions[j].z;
        if (dx * dx + dz * dz < minDistanceSq) return false;
      }
    }
    return true;
  }

  static enforceMinSheepSpacing(
    sheepPositions,
    {
      fenceLayers = null,
      cells = null,
      minDistance = MIN_SHEEP_CENTER_DISTANCE,
      minFenceDist = MIN_FENCE_DISTANCE
    } = {}
  ) {
    if (sheepPositions.length < 2) return;

    const maxPasses = 36;
    const minDistanceSq = minDistance * minDistance;

    for (let pass = 0; pass < maxPasses; pass++) {
      let moved = false;

      if (cells) {
        FenceSceneRenderer.recalculatePartitionAssignments(sheepPositions, cells);
      }

      for (let i = 0; i < sheepPositions.length; i++) {
        for (let j = i + 1; j < sheepPositions.length; j++) {
          const first = sheepPositions[i];
          const second = sheepPositions[j];
          const dx = first.x - second.x;
          const dz = first.z - second.z;
          const distSq = dx * dx + dz * dz;
          if (distSq >= minDistanceSq) continue;

          moved = true;

          let nx;
          let nz;
          let dist;
          if (distSq < 1e-8) {
            const angle = (pass + 1) * 0.73 + (i + 1) * 1.37 + (j + 1) * 2.11;
            nx = Math.cos(angle);
            nz = Math.sin(angle);
            dist = 0;
          } else {
            dist = Math.sqrt(distSq);
            nx = dx / dist;
            nz = dz / dist;
          }

          const push = (minDistance - dist) * 0.5 + 0.02;
          first.x += nx * push;
          first.z += nz * push;
          second.x -= nx * push;
          second.z -= nz * push;
        }
      }

      if (cells) {
        FenceSceneRenderer.recalculatePartitionAssignments(sheepPositions, cells);
        FenceSceneRenderer.enforceMinCellEdgeDist(sheepPositions, cells, minFenceDist);
      } else if (fenceLayers) {
        FenceSceneRenderer.enforceMinFenceDist(sheepPositions, fenceLayers, minFenceDist);
      }

      if (!moved && FenceSceneRenderer.hasMinSheepSpacing(sheepPositions, minDistance)) {
        break;
      }
    }

    if (cells) {
      FenceSceneRenderer.recalculatePartitionAssignments(sheepPositions, cells);
    }
  }

  /**
   * Set camera angle preset
   * @param {string} angle - 'top_down' | 'tilt_15' | 'tilt_30' | 'tilt_45'
   * @param {Object|number} sceneBounds - tight content bounds or a legacy radius
   */
  static _cameraTiltDegrees(angle) {
    if (angle === 'top_down') return 0;
    const match = String(angle ?? '').match(/\d+/);
    const tilt = match ? Number(match[0]) : 15;
    return Math.min(45, Math.max(0, tilt));
  }

  static _cameraDirectionForAngle(angle) {
    const tiltDeg = FenceSceneRenderer._cameraTiltDegrees(angle);
    if (tiltDeg === 0) {
      return new THREE.Vector3(0, 1, 0.001).normalize();
    }

    const tiltRad = tiltDeg * Math.PI / 180;
    return new THREE.Vector3(0, Math.cos(tiltRad), Math.sin(tiltRad)).normalize();
  }

  _cameraFitsPoints(position, target, points, margin = CAMERA_FRAME_MARGIN) {
    this.camera.position.copy(position);
    this.camera.lookAt(target);
    this.camera.near = 0.1;
    this.camera.far = Math.max(200, position.distanceTo(target) * 6);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);

    const maxNdc = 1 - margin;
    for (const point of points) {
      const ndc = point.clone().project(this.camera);
      if (!Number.isFinite(ndc.x) || !Number.isFinite(ndc.y) || !Number.isFinite(ndc.z)) {
        return false;
      }
      if (ndc.z < -1 || ndc.z > 1) return false;
      if (Math.abs(ndc.x) > maxNdc || Math.abs(ndc.y) > maxNdc) return false;
    }
    return true;
  }

  setCameraAngle(angle = 'tilt_15', sceneBounds = 10) {
    const bounds = typeof sceneBounds === 'number'
      ? {
          minX: -sceneBounds,
          maxX: sceneBounds,
          minZ: -sceneBounds,
          maxZ: sceneBounds,
          centerX: 0,
          centerZ: 0,
          width: sceneBounds * 2,
          depth: sceneBounds * 2,
          radius: sceneBounds
        }
      : FenceSceneRenderer._finalizeBounds(sceneBounds.bounds || sceneBounds);

    // target.y at half the object height vertically centers the scene
    // (ground points and sheep-top points project symmetrically around the
    // frame midline, so neither edge gets a chunk of empty sky/ground first).
    const target = new THREE.Vector3(bounds.centerX, CAMERA_OBJECT_HEIGHT * 0.5, bounds.centerZ);
    const points = typeof sceneBounds === 'number'
      ? FenceSceneRenderer.getCameraFitPoints(bounds, CAMERA_OBJECT_HEIGHT)
      : (sceneBounds.points?.length ? sceneBounds.points : FenceSceneRenderer.getCameraFitPoints(bounds, CAMERA_OBJECT_HEIGHT));
    const direction = FenceSceneRenderer._cameraDirectionForAngle(angle);

    let low = Math.max(bounds.radius * 0.8, CAMERA_OBJECT_HEIGHT * 1.5, 4);
    let high = Math.max(bounds.radius * 2.4, CAMERA_OBJECT_HEIGHT * 4.0, 10);
    while (!this._cameraFitsPoints(target.clone().addScaledVector(direction, high), target, points) && high < 400) {
      low = high;
      high *= 1.2;
    }
    for (let i = 0; i < 28; i++) {
      const mid = (low + high) / 2;
      if (this._cameraFitsPoints(target.clone().addScaledVector(direction, mid), target, points)) {
        high = mid;
      } else {
        low = mid;
      }
    }

    const finalPosition = target.clone().addScaledVector(direction, high);
    this._cameraFitsPoints(finalPosition, target, points);
    this.controls.target.copy(target);
    this.controls.update();
  }

  /**
   * Generate and render a complete fence + sheep scene
   * @param {Object} params - scene parameters
   * @returns {Object} metadata with ground truth
   */
  generateScene(params = {}) {
    const {
      fenceShape = 'convex',
      fenceSegments = 8,
      fenceRadius = 5,
      nestingDepth = 1,
      numGaps = 0,
      gapSize = 0.5,
      numSheep = 5,
      sheepColors = ['WHITE', 'RED', 'BLUE'],
      insideRatio = 0.5,
      cameraAngle = 'tilt_15',
      groundTexture = 'grass',
      numGates = 0,
      // Partitioned mode params
      numPartitionCols = 2,
      numPartitionRows = 2,
      partitionSeparated = false,
      partitionLayout = 'grid',
      numSectors = 5,
      numPolygonSides = 6,
      numPolygonDivisions = 3,
      seed = null
    } = params;

    // Clear previous scene objects (keep lights and ground)
    this._clearScene();

    // ── Partitioned mode ──
    if (fenceShape === 'partitioned') {
      const outsideCount = Math.max(1, Math.round(numSheep * (1 - insideRatio)));
      const maxPlacementRetries = 1024;
      let partitionData = null;
      let sheepPositions = [];
      let cellCounts = {};

      for (let attempt = 0; attempt < maxPlacementRetries; attempt++) {
        const candidatePartitionData = FenceGenerator.generatePartitioned({
          layout: partitionLayout,
          numCols: numPartitionCols,
          numRows: numPartitionRows,
          numSectors,
          numPolygonSides,
          numPolygonDivisions,
          width: fenceRadius * 2.2,
          height: fenceRadius * 2.0,
          separated: partitionSeparated,
          gapSize: 1.5,
          irregularity: 0.15,
          numGates
        });

        // Some partition layouts produce narrow cells (e.g. nested_polygon
        // trapezoidal rings, polygon_star wedge tips) where the global 2.0m
        // fence buffer leaves no usable interior and every outer cell ends up
        // empty. Relax the buffer for those layouts.
        const layoutFenceDist = (partitionLayout === 'nested_polygon' || partitionLayout === 'polygon_star' || partitionLayout === 'hex_cross')
          ? 1.2
          : MIN_FENCE_DISTANCE;

        const candidatePositions = SheepPlacer.placeSheepInPartitions({
          cells: candidatePartitionData.cells,
          numSheep,
          sceneRadius: fenceRadius * 2.0,
          outsideCount,
          colors: sheepColors,
          minFenceDist: layoutFenceDist
        });

        FenceSceneRenderer.enforceMinCellEdgeDist(candidatePositions, candidatePartitionData.cells, layoutFenceDist);
        FenceSceneRenderer.enforceMinSheepSpacing(candidatePositions, {
          cells: candidatePartitionData.cells,
          minDistance: MIN_SHEEP_CENTER_DISTANCE,
          minFenceDist: layoutFenceDist
        });

        const candidateCellCounts = FenceSceneRenderer.recalculatePartitionAssignments(
          candidatePositions,
          candidatePartitionData.cells
        );
        const hasValidMaxCellDataset = SheepPlacer._isValidMaxCellDataset(Object.values(candidateCellCounts));
        const hasMinSpacing = FenceSceneRenderer.hasMinSheepSpacing(candidatePositions, MIN_SHEEP_CENTER_DISTANCE);
        if (candidatePositions.length === numSheep && hasValidMaxCellDataset && hasMinSpacing) {
          partitionData = candidatePartitionData;
          sheepPositions = candidatePositions;
          cellCounts = candidateCellCounts;
          break;
        }

        partitionData = candidatePartitionData;
        sheepPositions = candidatePositions;
        cellCounts = candidateCellCounts;
      }

      const sceneBounds = FenceSceneRenderer.getSceneFootprintBounds({
        sheepPositions,
        partitionData
      });
      const sceneCameraPoints = FenceSceneRenderer.getSceneCameraPoints({
        sheepPositions,
        partitionData
      });
      this._updateGroundForBounds(groundTexture, sceneBounds);
      this._updateLightsForBounds(sceneBounds);

      const { group } = FenceGenerator.buildPartitionedMesh(partitionData, { sheepPositions });
      group.userData = { type: 'fence' };
      this.scene.add(group);

      this.sheepData = sheepPositions;

      const sheepMeshes = [];
      for (const sheep of sheepPositions) {
        const sheepMesh = SheepPlacer.createSheepMesh(sheep.color, sheep.id);
        sheepMesh.position.set(sheep.x, 0, sheep.z);
        sheepMesh.userData = { type: 'sheep', data: sheep };
        this.scene.add(sheepMesh);
        sheepMeshes.push(sheepMesh);
      }

      this.setCameraAngle(cameraAngle, { bounds: sceneBounds, points: sceneCameraPoints });

      // Resolve label overlaps now that camera is positioned
      SheepPlacer.resolveLabelsOverlap(sheepMeshes, this.camera);

      if (
        sheepPositions.length !== numSheep ||
        !SheepPlacer._isValidMaxCellDataset(Object.values(cellCounts)) ||
        !FenceSceneRenderer.hasMinSheepSpacing(sheepPositions, MIN_SHEEP_CENTER_DISTANCE)
      ) {
        console.warn('[enclosure_sheep] partition placement fell back to a non-ideal result', {
          totalPlaced: sheepPositions.length,
          expectedTotal: numSheep,
          cellCounts
        });
      }

      // Compute gate edges: which cells are connected by each gate
      const gateEdges = [];
      if (partitionData.wallSegments) {
        for (const seg of partitionData.wallSegments) {
          if (!seg.isGate) continue;
          // Find the two cells that share this wall segment
          const midX = (seg.start.x + seg.end.x) / 2;
          const midZ = (seg.start.z + seg.end.z) / 2;
          const adjacentCells = [];
          for (let ci = 0; ci < partitionData.cells.length; ci++) {
            const cell = partitionData.cells[ci];
            // Check if the wall midpoint is on the boundary of this cell
            for (let vi = 0; vi < cell.length; vi++) {
              const v0 = cell[vi];
              const v1 = cell[(vi + 1) % cell.length];
              const emx = (v0.x + v1.x) / 2;
              const emz = (v0.z + v1.z) / 2;
              if (Math.hypot(emx - midX, emz - midZ) < 0.5) {
                adjacentCells.push(String.fromCharCode(65 + ci));
                break;
              }
            }
          }
          if (adjacentCells.length === 2) {
            gateEdges.push(adjacentCells);
          }
        }
      }

      const insideCount = sheepPositions.filter(s => s.isInside).length;
      const placedOutsideCount = sheepPositions.filter(s => !s.isInside).length;
      const cannotEscapeCount = insideCount;

      this.metadata = {
        params,
        partitionData: { cells: partitionData.cells, separated: partitionData.separated, numGates: partitionData.numGates || 0 },
        sheepPositions,
        groundTruth: {
          insideCount,
          outsideCount: placedOutsideCount,
          cannotEscapeCount,
          totalSheep: sheepPositions.length,
          numCells: partitionData.cells.length,
          cellCounts,
          isPartitioned: true,
          isSeparated: partitionData.separated,
          numGates: partitionData.numGates || 0,
          gateEdges,
          sheepDetails: sheepPositions.map(s => ({
            id: s.id,
            color: s.color,
            isInside: s.isInside,
            cannotEscape: s.isInside,
            cellIndex: s.cellIndex,
            cellLabel: s.cellLabel,
            position: { x: s.x.toFixed(3), z: s.z.toFixed(3) }
          }))
        }
      };

      this.render();
      return this.metadata;
    }

    // ── Standard nested fence mode ── (minimum 2 layers)
    const actualNesting = Math.max(2, nestingDepth);
    // Enforce minimum radius: need 3.5 units per layer + 3.5 inner minimum
    const minRadiusNeeded = 3.5 + (actualNesting - 1) * 3.5;
    const actualRadius = Math.max(fenceRadius, minRadiusNeeded);
    const layers = FenceGenerator.generateNested({
      depth: actualNesting,
      outerRadius: actualRadius,
      shape: fenceShape,
      segments: fenceSegments
    });
    this.fenceLayers = layers;

    // Generate gate indices
    const gateIndices = [];
    if (numGates > 0) {
      const step = Math.floor(layers[0].length / numGates);
      for (let g = 0; g < numGates; g++) {
        gateIndices.push((g * step) % layers[0].length);
      }
    }

    // Build fence meshes
    const allFenceMetadata = [];
    for (let i = 0; i < layers.length; i++) {
      const fenceResult = FenceGenerator.buildFenceMesh(layers[i], {
        numGaps: i === 0 ? numGaps : 0,
        gapSize,
        gateIndices: i === 0 ? gateIndices : [],
        postColor: new THREE.Color().setHSL(0.08, 0.6, 0.35 + i * 0.05),
        railColor: new THREE.Color().setHSL(0.08, 0.5, 0.4 + i * 0.05)
      });
      fenceResult.group.userData = { type: 'fence', layerIndex: i };
      this.scene.add(fenceResult.group);
      allFenceMetadata.push(fenceResult.metadata);
    }

    const maxPlacementRetries = 128;
    let sheepPositions = [];

    for (let attempt = 0; attempt < maxPlacementRetries; attempt++) {
      const candidatePositions = SheepPlacer.placeSheep({
        numSheep,
        fenceLayers: layers,
        sceneRadius: actualRadius * 1.8,
        insideRatio,
        colors: sheepColors,
        minFenceDist: MIN_FENCE_DISTANCE
      });

      FenceSceneRenderer.enforceMinFenceDist(candidatePositions, layers, MIN_FENCE_DISTANCE);
      FenceSceneRenderer.enforceMinSheepSpacing(candidatePositions, {
        fenceLayers: layers,
        minDistance: MIN_SHEEP_CENTER_DISTANCE,
        minFenceDist: MIN_FENCE_DISTANCE
      });

      // Recalculate topology after position adjustments — clamping/enforcement can
      // move sheep across fence boundaries, invalidating the original isInside label.
      for (const sheep of candidatePositions) {
        sheep.nestingDepth = RegionAnalyzer.nestingDepth({ x: sheep.x, z: sheep.z }, layers);
        sheep.isInside = sheep.nestingDepth > 0;
        sheep.regionLabel = RegionAnalyzer.regionLabel({ x: sheep.x, z: sheep.z }, layers);
      }

      sheepPositions = candidatePositions;
      const candidateCannotEscapeCount = candidatePositions.filter(
        s => s.isInside && !(s.nestingDepth === 1 && numGaps > 0)
      ).length;
      if (
        candidatePositions.length === numSheep &&
        FenceSceneRenderer.hasMinSheepSpacing(candidatePositions, MIN_SHEEP_CENTER_DISTANCE) &&
        candidateCannotEscapeCount > 0
      ) {
        break;
      }
    }

    this.sheepData = sheepPositions;

    const sceneBounds = FenceSceneRenderer.getSceneFootprintBounds({
      sheepPositions,
      fenceLayers: layers
    });
    const sceneCameraPoints = FenceSceneRenderer.getSceneCameraPoints({
      sheepPositions,
      fenceLayers: layers
    });
    this._updateGroundForBounds(groundTexture, sceneBounds);
    this._updateLightsForBounds(sceneBounds);

    // Create sheep meshes
    const sheepMeshes = [];
    for (const sheep of sheepPositions) {
      const sheepMesh = SheepPlacer.createSheepMesh(sheep.color, sheep.id);
      sheepMesh.position.set(sheep.x, 0, sheep.z);
      sheepMesh.userData = { type: 'sheep', data: sheep };
      this.scene.add(sheepMesh);
      sheepMeshes.push(sheepMesh);
    }

    // Set camera
    this.setCameraAngle(cameraAngle, { bounds: sceneBounds, points: sceneCameraPoints });

    // Resolve label overlaps now that camera is positioned
    SheepPlacer.resolveLabelsOverlap(sheepMeshes, this.camera);

    if (
      sheepPositions.length !== numSheep ||
      !FenceSceneRenderer.hasMinSheepSpacing(sheepPositions, MIN_SHEEP_CENTER_DISTANCE)
    ) {
      console.warn('[enclosure_sheep] fence placement fell back to a non-ideal result', {
        totalPlaced: sheepPositions.length,
        expectedTotal: numSheep
      });
    }

    // Compute metadata / ground truth
    const insideCount = sheepPositions.filter(s => s.isInside).length;
    const outsideCount = sheepPositions.filter(s => !s.isInside).length;
    const sheepDetails = sheepPositions.map(s => {
      // A sheep can escape only if it is inside AND in the outermost layer
      // (nestingDepth===1) AND that layer has gaps. Inner layers never have gaps.
      const canEscape = s.isInside && s.nestingDepth === 1 && numGaps > 0;
      return {
        id: s.id,
        color: s.color,
        isInside: s.isInside,
        nestingDepth: s.nestingDepth,
        regionLabel: s.regionLabel,
        canEscape,
        cannotEscape: s.isInside && !canEscape,
        position: { x: s.x.toFixed(3), z: s.z.toFixed(3) }
      };
    });
    const cannotEscapeCount = sheepDetails.filter(s => s.cannotEscape).length;

    // Extract outermost fence vertex coordinates and gap details for metadata
    const outerMeta = allFenceMetadata[0] || {};
    const outerVertices = (outerMeta.vertices || layers[0] || []).map((v, i) => ({
      id: i, x: parseFloat(v.x.toFixed(2)), z: parseFloat(v.z.toFixed(2))
    }));
    const outerGaps = (outerMeta.gaps || []).map(g => ({
      segmentIndex: g.segmentIndex,
      between: [g.segmentIndex, (g.segmentIndex + 1) % outerVertices.length]
    }));

    this.metadata = {
      params,
      fenceLayers: layers,
      fenceMetadata: allFenceMetadata,
      sheepPositions,
      groundTruth: {
        insideCount,
        outsideCount,
        cannotEscapeCount,
        totalSheep: sheepPositions.length,
        isFenceClosed: numGaps === 0,
        numGaps,
        numRegions: RegionAnalyzer.countRegions(layers),
        nestingDepth: actualNesting,
        sheepDetails,
        gateIndices,
        gateLabels: gateIndices.map((_, i) => `G${i + 1}`),
        fenceVertices: outerVertices,
        fenceGaps: outerGaps
      }
    };

    this.render();
    return this.metadata;
  }

  _clearScene() {
    const toRemove = [];
    this.scene.traverse(obj => {
      if (obj.userData && (obj.userData.type === 'fence' || obj.userData.type === 'sheep' ||
          obj.userData.type === 'decoration')) {
        toRemove.push(obj);
      }
    });
    for (const obj of toRemove) {
      this.scene.remove(obj);
      obj.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => FenceSceneRenderer._disposeMaterial(m));
          } else {
            FenceSceneRenderer._disposeMaterial(child.material);
          }
        }
      });
    }
    this.renderer.renderLists?.dispose?.();
  }

  static _disposeMaterial(material) {
    if (!material) return;
    for (const value of Object.values(material)) {
      if (value && value.isTexture) value.dispose();
    }
    material.dispose?.();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Resize the renderer output while keeping the same scene state.
   * @param {number} width
   * @param {number} height
   */
  setSize(width, height) {
    const nextWidth = Math.max(1, Math.round(width || this.width));
    const nextHeight = Math.max(1, Math.round(height || this.height));
    if (nextWidth === this.width && nextHeight === this.height) {
      return;
    }

    this.width = nextWidth;
    this.height = nextHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.render();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.render();
  }

  /**
   * Capture current view as data URL
   * @returns {string} PNG data URL
   */
  captureImage() {
    this.render();
    return this.renderer.domElement.toDataURL('image/png');
  }

  /**
   * Capture current view as Blob
   * @returns {Promise<Blob>}
   */
  captureBlob() {
    return new Promise(resolve => {
      this.render();
      this.renderer.domElement.toBlob(resolve, 'image/png');
    });
  }

  dispose() {
    this.controls.dispose();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
