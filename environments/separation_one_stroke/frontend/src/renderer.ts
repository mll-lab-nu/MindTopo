/**
 * TopoBench Three.js Renderer
 * 
 * Renders the game board in 2.5D perspective
 */

import * as THREE from 'three';
import type { Vertex, LevelJson } from './types';
import { getColorHex } from './types';

const CELL_SIZE = 1.0;
const BOARD_HEIGHT = 0.1;
const CELL_PATCH_SCALE = 0.6;
const STROKE_RADIUS = 0.08;
const PEG_RADIUS = 0.15;
const PEG_HEIGHT = 0.3;

export class Renderer {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private readonly resizeHandler: () => void;
  
  private boardGroup: THREE.Group;
  private strokeGroup: THREE.Group;
  private pegGroup: THREE.Group;
  
  private W: number = 4;
  private H: number = 4;
  
  // Materials
  private boardMaterial: THREE.MeshStandardMaterial;
  private gridMaterial: THREE.LineBasicMaterial;
  private strokeMaterial: THREE.MeshStandardMaterial;
  private startPegMaterial: THREE.MeshStandardMaterial;
  private endPegMaterial: THREE.MeshStandardMaterial;
  
  constructor(container: HTMLElement) {
    this.container = container;
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1117);
    
    // Create camera
    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    this.camera.position.set(0, 8, 6);
    this.camera.lookAt(0, 0, 0);
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      preserveDrawingBuffer: true // For snapshot
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    
    // Create groups
    this.boardGroup = new THREE.Group();
    this.strokeGroup = new THREE.Group();
    this.pegGroup = new THREE.Group();
    this.scene.add(this.boardGroup);
    this.scene.add(this.strokeGroup);
    this.scene.add(this.pegGroup);
    
    // Create materials
    this.boardMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x21262d,
      roughness: 0.8,
      metalness: 0.1,
    });
    
    this.gridMaterial = new THREE.LineBasicMaterial({ 
      color: 0x30363d,
      linewidth: 1,
    });
    
    this.strokeMaterial = new THREE.MeshStandardMaterial({
      color: 0xf0f6fc,
      roughness: 0.3,
      metalness: 0.5,
      emissive: 0xf0f6fc,
      emissiveIntensity: 0.1,
    });
    
    this.startPegMaterial = new THREE.MeshStandardMaterial({
      color: 0x3fb950,
      roughness: 0.3,
      metalness: 0.6,
      emissive: 0x3fb950,
      emissiveIntensity: 0.3,
    });
    
    this.endPegMaterial = new THREE.MeshStandardMaterial({
      color: 0xf85149,
      roughness: 0.3,
      metalness: 0.6,
      emissive: 0xf85149,
      emissiveIntensity: 0.3,
    });
    
    
    // Setup lighting
    this.setupLighting();

    // Handle resize
    this.resizeHandler = this.onResize.bind(this);
    window.addEventListener('resize', this.resizeHandler);

    // The constructor often runs before CSS layout completes (the container
    // has flex: 1, so clientWidth/Height read 0 here). Re-size on the next
    // animation frame, once layout has settled, so the GL drawing buffer is
    // not stuck at 0×0 (which presents as "context lost" + all-black snapshots).
    requestAnimationFrame(() => this.onResize());

    this.renderNow();

    // Continuous rAF loop so the GL drawing buffer is always composited and
    // ready for toDataURL. Without this, the very first snapshot after
    // initBoard reads an empty back buffer in headless Chromium because
    // antialias=true uses an MSAA buffer that only blits at end-of-frame.
    const tick = () => {
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  
  private setupLighting(): void {
    // Ambient light
    const ambient = new THREE.AmbientLight(0x404040, 0.5);
    this.scene.add(ambient);
    
    // Main directional light
    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
    mainLight.position.set(5, 10, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 50;
    mainLight.shadow.camera.left = -10;
    mainLight.shadow.camera.right = 10;
    mainLight.shadow.camera.top = 10;
    mainLight.shadow.camera.bottom = -10;
    this.scene.add(mainLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0x58a6ff, 0.3);
    fillLight.position.set(-5, 5, -5);
    this.scene.add(fillLight);
    
    // Rim light
    const rimLight = new THREE.DirectionalLight(0xa371f7, 0.2);
    rimLight.position.set(0, 3, -8);
    this.scene.add(rimLight);
  }
  
  private onResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderNow();
  }

  private renderNow(): void {
    this.renderer.render(this.scene, this.camera);
  }
  
  /**
   * Convert grid position to world position
   */
  private gridToWorld(x: number, y: number): THREE.Vector3 {
    const offsetX = -(this.W - 1) * CELL_SIZE / 2;
    const offsetZ = -(this.H - 1) * CELL_SIZE / 2;
    return new THREE.Vector3(
      offsetX + x * CELL_SIZE,
      BOARD_HEIGHT,
      offsetZ + (this.H - 1 - y) * CELL_SIZE // Flip Y for display
    );
  }
  
  /**
   * Initialize the board for a level
   */
  initBoard(level: LevelJson): void {
    this.W = level.W;
    this.H = level.H;
    
    // Clear existing objects
    this.clearGroup(this.boardGroup);
    this.clearGroup(this.strokeGroup);
    this.clearGroup(this.pegGroup);
    
    // Create board base
    this.createBoardBase();
    
    // Create grid lines
    this.createGrid();
    
    // Create cell patches
    this.createCellPatches(level.cells);
    
    // Create start/end pegs
    this.createPegs();
    
    // Update camera position based on board size
    this.updateCamera();
    this.renderNow();
  }
  
  private clearGroup(group: THREE.Group): void {
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
      }
    }
  }
  
  private createBoardBase(): void {
    const width = (this.W - 1) * CELL_SIZE + 0.4;
    const depth = (this.H - 1) * CELL_SIZE + 0.4;
    
    const geometry = new THREE.BoxGeometry(width, BOARD_HEIGHT, depth);
    const mesh = new THREE.Mesh(geometry, this.boardMaterial);
    mesh.position.set(0, BOARD_HEIGHT / 2 - 0.05, 0);
    mesh.receiveShadow = true;
    this.boardGroup.add(mesh);
    
    // Add border
    const borderGeometry = new THREE.BoxGeometry(width + 0.1, BOARD_HEIGHT + 0.05, depth + 0.1);
    const borderMaterial = new THREE.MeshStandardMaterial({
      color: 0x30363d,
      roughness: 0.7,
      metalness: 0.2,
    });
    const border = new THREE.Mesh(borderGeometry, borderMaterial);
    border.position.set(0, BOARD_HEIGHT / 2 - 0.08, 0);
    border.receiveShadow = true;
    this.boardGroup.add(border);
  }
  
  private createGrid(): void {
    const points: THREE.Vector3[] = [];
    
    // Vertical lines
    for (let x = 0; x < this.W; x++) {
      const start = this.gridToWorld(x, 0);
      const end = this.gridToWorld(x, this.H - 1);
      points.push(start, end);
    }
    
    // Horizontal lines
    for (let y = 0; y < this.H; y++) {
      const start = this.gridToWorld(0, y);
      const end = this.gridToWorld(this.W - 1, y);
      points.push(start, end);
    }
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const lines = new THREE.LineSegments(geometry, this.gridMaterial);
    lines.position.y = 0.01;
    this.boardGroup.add(lines);
  }
  
  private createCellPatches(cells: (string | null)[][]): void {
    for (let j = 0; j < this.H - 1; j++) {
      for (let i = 0; i < this.W - 1; i++) {
        const color = cells[j][i];
        if (color === null) continue;
        
        // Cell center in grid coordinates
        const centerX = i + 0.5;
        const centerY = j + 0.5;
        
        const worldPos = this.gridToWorld(centerX, centerY);
        
        // Create colored patch
        const size = CELL_SIZE * CELL_PATCH_SCALE;
        const geometry = new THREE.BoxGeometry(size, 0.08, size);
        const material = new THREE.MeshStandardMaterial({
          color: getColorHex(color),
          roughness: 0.4,
          metalness: 0.3,
          emissive: getColorHex(color),
          emissiveIntensity: 0.2,
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(worldPos);
        mesh.position.y = BOARD_HEIGHT + 0.04;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add glow ring
        const ringGeometry = new THREE.RingGeometry(size / 2 + 0.02, size / 2 + 0.06, 4);
        const ringMaterial = new THREE.MeshBasicMaterial({
          color: getColorHex(color),
          transparent: true,
          opacity: 0.3,
          side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = -Math.PI / 2;
        ring.rotation.z = Math.PI / 4;
        ring.position.copy(worldPos);
        ring.position.y = BOARD_HEIGHT + 0.01;
        
        this.boardGroup.add(mesh);
        this.boardGroup.add(ring);
      }
    }
  }
  
  private createPegs(): void {
    const pegGeometry = new THREE.CylinderGeometry(
      PEG_RADIUS, PEG_RADIUS * 1.2, PEG_HEIGHT, 16
    );
    const sphereGeometry = new THREE.SphereGeometry(PEG_RADIUS * 1.1, 16, 16);
    
    // Start peg at (0, 0)
    const startPos = this.gridToWorld(0, 0);
    const startPeg = new THREE.Mesh(pegGeometry, this.startPegMaterial);
    startPeg.position.copy(startPos);
    startPeg.position.y = BOARD_HEIGHT + PEG_HEIGHT / 2;
    startPeg.castShadow = true;
    
    const startSphere = new THREE.Mesh(sphereGeometry, this.startPegMaterial);
    startSphere.position.copy(startPos);
    startSphere.position.y = BOARD_HEIGHT + PEG_HEIGHT;
    startSphere.castShadow = true;
    
    this.pegGroup.add(startPeg);
    this.pegGroup.add(startSphere);
    
    // End peg at (W-1, H-1)
    const endPos = this.gridToWorld(this.W - 1, this.H - 1);
    const endPeg = new THREE.Mesh(pegGeometry, this.endPegMaterial);
    endPeg.position.copy(endPos);
    endPeg.position.y = BOARD_HEIGHT + PEG_HEIGHT / 2;
    endPeg.castShadow = true;
    
    const endSphere = new THREE.Mesh(sphereGeometry, this.endPegMaterial);
    endSphere.position.copy(endPos);
    endSphere.position.y = BOARD_HEIGHT + PEG_HEIGHT;
    endSphere.castShadow = true;
    
    this.pegGroup.add(endPeg);
    this.pegGroup.add(endSphere);
  }
  
  private updateCamera(): void {
    const maxDim = Math.max(this.W, this.H);
    const distance = maxDim * 1.5;
    this.camera.position.set(0, distance * 0.8, distance * 0.6);
    this.camera.lookAt(0, 0, 0);
  }
  
  /**
   * Update the stroke rendering
   */
  updateStroke(stroke: Vertex[], currentPos: Vertex): void {
    this.clearGroup(this.strokeGroup);
    
    if (stroke.length < 2) {
      // Just show current position peg
      this.addCurrentPositionPeg(currentPos);
      this.renderNow();
      return;
    }
    
    // Create tube segments for each edge
    for (let i = 0; i < stroke.length - 1; i++) {
      const v1 = stroke[i];
      const v2 = stroke[i + 1];
      
      const p1 = this.gridToWorld(v1.x, v1.y);
      const p2 = this.gridToWorld(v2.x, v2.y);
      
      // Create tube segment
      const path = new THREE.LineCurve3(p1, p2);
      const tubeGeometry = new THREE.TubeGeometry(path, 1, STROKE_RADIUS, 8, false);
      const tube = new THREE.Mesh(tubeGeometry, this.strokeMaterial);
      tube.position.y = 0.05;
      tube.castShadow = true;
      this.strokeGroup.add(tube);
      
      // Add joint sphere at v1
      if (i === 0 || true) {
        const jointGeometry = new THREE.SphereGeometry(STROKE_RADIUS * 1.1, 8, 8);
        const joint = new THREE.Mesh(jointGeometry, this.strokeMaterial);
        joint.position.copy(p1);
        joint.position.y += 0.05;
        joint.castShadow = true;
        this.strokeGroup.add(joint);
      }
    }
    
    // Add final joint at last vertex
    const lastV = stroke[stroke.length - 1];
    const lastP = this.gridToWorld(lastV.x, lastV.y);
    const jointGeometry = new THREE.SphereGeometry(STROKE_RADIUS * 1.1, 8, 8);
    const joint = new THREE.Mesh(jointGeometry, this.strokeMaterial);
    joint.position.copy(lastP);
    joint.position.y += 0.05;
    joint.castShadow = true;
    this.strokeGroup.add(joint);
    
    // Add current position peg
    this.addCurrentPositionPeg(currentPos);
    this.renderNow();
  }
  
  private addCurrentPositionPeg(pos: Vertex): void {
    const worldPos = this.gridToWorld(pos.x, pos.y);
    
    // Glowing ring around current position
    const ringGeometry = new THREE.RingGeometry(0.15, 0.22, 32);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0x58a6ff,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(worldPos);
    ring.position.y = BOARD_HEIGHT + 0.02;
    this.strokeGroup.add(ring);
    
    // Pulsing outer ring
    const outerRingGeometry = new THREE.RingGeometry(0.22, 0.28, 32);
    const outerRingMaterial = new THREE.MeshBasicMaterial({
      color: 0x58a6ff,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
    });
    const outerRing = new THREE.Mesh(outerRingGeometry, outerRingMaterial);
    outerRing.rotation.x = -Math.PI / 2;
    outerRing.position.copy(worldPos);
    outerRing.position.y = BOARD_HEIGHT + 0.015;
    this.strokeGroup.add(outerRing);
  }
  
  /**
   * Show success effect
   */
  showSuccess(): void {
    // Flash green on all stroke segments
    this.strokeMaterial.emissive.setHex(0x3fb950);
    this.strokeMaterial.emissiveIntensity = 0.5;
    this.renderNow();
    
    setTimeout(() => {
      this.strokeMaterial.emissive.setHex(0xf0f6fc);
      this.strokeMaterial.emissiveIntensity = 0.1;
      this.renderNow();
    }, 500);
  }
  
  /**
   * Show failure effect
   */
  showFailure(): void {
    // Flash red on all stroke segments
    this.strokeMaterial.emissive.setHex(0xf85149);
    this.strokeMaterial.emissiveIntensity = 0.5;
    this.renderNow();
    
    setTimeout(() => {
      this.strokeMaterial.emissive.setHex(0xf0f6fc);
      this.strokeMaterial.emissiveIntensity = 0.1;
      this.renderNow();
    }, 500);
  }
  
  /**
   * Take a snapshot of the current view
   */
  async snapshot(): Promise<string> {
    // antialias=true means Three.js renders to an MSAA buffer that only blits
    // to the drawing buffer at end-of-frame composition. Without this rAF wait
    // between render and toDataURL, the first capture after initBoard reads an
    // empty (all-black) drawing buffer.
    this.renderNow();
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    );
    this.renderNow();
    return this.renderer.domElement.toDataURL('image/png');
  }
  
  /**
   * Dispose of renderer resources
   */
  dispose(): void {
    this.renderer.dispose();
    window.removeEventListener('resize', this.resizeHandler);
  }
}
