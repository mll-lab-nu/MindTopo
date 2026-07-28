// Camera control with BEV (Bird's Eye View) support

import * as THREE from 'three';
import { CameraParams } from '../types';
import { degToRad, clamp } from '../utils/MathUtils';

const MIN_PITCH = 25;
const MAX_PITCH = 89;

function normalizeYaw(yaw: number): number {
  return ((yaw % 360) + 360) % 360;
}

/**
 * Camera Rig for controlling viewing angles
 * Supports BEV (Bird's Eye View) and various pitch/yaw configurations
 */
export class CameraRig {
  private camera: THREE.PerspectiveCamera;
  private target: THREE.Vector3;
  private params: CameraParams;

  constructor(aspect: number = 1) {
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 5000); // Extended far plane for large distances
    this.camera.up.set(0, 0, 1);
    this.target = new THREE.Vector3(0, 0, 0);
    this.params = {
      pitch: 85,
      yaw: 75,
      distance: 18,
      fov: 45
    };
    
    this.updateCamera();
  }

  /**
   * Get the underlying THREE.PerspectiveCamera
   */
  getCamera(): THREE.PerspectiveCamera {
    return this.camera;
  }

  /**
   * Update camera with new parameters
   */
  setParams(params: Partial<CameraParams>): void {
    const nextParams = { ...this.params, ...params };
    nextParams.pitch = clamp(nextParams.pitch, MIN_PITCH, MAX_PITCH);
    nextParams.yaw = normalizeYaw(nextParams.yaw);
    this.params = nextParams;
    this.updateCamera();
  }

  /**
   * Get current camera parameters
   */
  getParams(): CameraParams {
    return { ...this.params };
  }

  /**
   * Set the camera target (look-at point)
   */
  setTarget(target: THREE.Vector3): void {
    this.target.copy(target);
    this.camera.lookAt(this.target);
  }

  /**
   * Update camera aspect ratio
   */
  setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Apply parameters and update camera position
   */
  private updateCamera(): void {
    const { pitch, yaw, distance, fov } = this.params;
    
    // Clamp pitch to valid range
    const clampedPitch = clamp(pitch, MIN_PITCH, MAX_PITCH);
    const normalizedYaw = normalizeYaw(yaw);
    
    // Convert angles to radians
    const pitchRad = degToRad(clampedPitch);
    const yawRad = degToRad(normalizedYaw);
    
    // Calculate camera position using spherical coordinates
    // pitch: 0 = horizontal, 90 = directly above (BEV)
    const height = Math.sin(pitchRad) * distance;
    const horizontalDist = Math.cos(pitchRad) * distance;
    
    const x = Math.cos(yawRad) * horizontalDist;
    const y = Math.sin(yawRad) * horizontalDist;
    const z = height;
    
    this.camera.position.set(x, y, z);
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(this.target);
  }

  /**
   * Set camera to pure BEV (Bird's Eye View)
   */
  setBEV(distance?: number): void {
    this.setParams({
      pitch: MAX_PITCH,
      yaw: 75,
      distance: distance ?? this.params.distance
    });
  }

  /**
   * Set camera to a high-angle view within limits
   */
  setIsometric(distance?: number): void {
    this.setParams({
      pitch: 80,
      yaw: 75,
      distance: distance ?? this.params.distance
    });
  }

  /**
   * Orbit the camera around the target
   */
  orbit(deltaPitch: number, deltaYaw: number): void {
    this.setParams({
      pitch: clamp(this.params.pitch + deltaPitch, MIN_PITCH, MAX_PITCH),
      yaw: normalizeYaw(this.params.yaw + deltaYaw)
    });
  }

  /**
   * Zoom the camera
   */
  zoom(factor: number): void {
    const newDistance = clamp(this.params.distance * factor, 1, 2000);
    this.setParams({ distance: newDistance });
  }

  /**
   * Calculate minimum distance needed to see the entire board
   */
  getMinDistanceForBoard(boardWidth: number, boardHeight: number, padding: number = 1.2): number {
    const maxDim = Math.max(boardWidth, boardHeight) * padding;
    const fovRad = degToRad(this.params.fov);
    return (maxDim / 2) / Math.tan(fovRad / 2);
  }

  /**
   * Get world position of a point on the board plane (z=0)
   */
  screenToBoard(
    screenX: number,
    screenY: number,
    screenWidth: number,
    screenHeight: number
  ): THREE.Vector3 | null {
    const ndc = new THREE.Vector2(
      (screenX / screenWidth) * 2 - 1,
      -(screenY / screenHeight) * 2 + 1
    );
    
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    
    // Intersect with board plane (z=0)
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const intersection = new THREE.Vector3();
    
    if (raycaster.ray.intersectPlane(plane, intersection)) {
      return intersection;
    }
    
    return null;
  }

  /**
   * Calculate camera settings to fit a board in view
   */
  fitToBoard(boardWidth: number, boardHeight: number, padding: number = 1.2): void {
    const minDistance = this.getMinDistanceForBoard(boardWidth, boardHeight, padding);
    
    // Only update if current distance is less than needed
    if (this.params.distance < minDistance) {
      this.setParams({ distance: minDistance });
    }
  }

  /**
   * Randomize camera angles within constraints
   */
  randomize(
    random: { range: (min: number, max: number) => number },
    pitchRange: [number, number] = [35, 85],
    yawRange: [number, number] = [0, 360]
  ): void {
    this.setParams({
      pitch: random.range(pitchRange[0], pitchRange[1]),
      yaw: random.range(yawRange[0], yawRange[1])
    });
  }
}

/**
 * Create a CameraRig instance
 */
export function createCameraRig(aspect: number = 1): CameraRig {
  return new CameraRig(aspect);
}
