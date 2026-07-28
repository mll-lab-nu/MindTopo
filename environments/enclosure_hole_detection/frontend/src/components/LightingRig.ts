// Lighting setup for the scene

import * as THREE from 'three';
import { LightingParams } from '../types';

/**
 * Lighting Rig - Manages scene lighting
 */
export class LightingRig {
  private ambientLight: THREE.AmbientLight;
  private directionalLight: THREE.DirectionalLight;
  private fillLight: THREE.DirectionalLight;
  private group: THREE.Group;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'LightingRig';
    
    // Ambient light for overall illumination (brighter for better visibility)
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.group.add(this.ambientLight);
    
    // Main directional light (sun-like, stronger)
    this.directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    this.directionalLight.position.set(10, 20, 25);
    this.directionalLight.castShadow = true;
    
    // Shadow configuration
    this.directionalLight.shadow.mapSize.width = 2048;
    this.directionalLight.shadow.mapSize.height = 2048;
    this.directionalLight.shadow.camera.near = 0.5;
    this.directionalLight.shadow.camera.far = 200;
    this.directionalLight.shadow.camera.left = -30;
    this.directionalLight.shadow.camera.right = 30;
    this.directionalLight.shadow.camera.top = 30;
    this.directionalLight.shadow.camera.bottom = -30;
    this.directionalLight.shadow.bias = -0.0001;
    
    this.group.add(this.directionalLight);
    
    // Fill light (softer, from opposite direction)
    this.fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    this.fillLight.position.set(-10, -15, 15);
    this.group.add(this.fillLight);
  }

  /**
   * Get the lighting group to add to scene
   */
  getGroup(): THREE.Group {
    return this.group;
  }

  /**
   * Apply lighting parameters
   */
  setParams(params: LightingParams): void {
    this.ambientLight.intensity = params.ambientIntensity;
    this.directionalLight.position.set(
      params.dirLightPosition[0],
      params.dirLightPosition[1],
      params.dirLightPosition[2]
    );
  }

  /**
   * Set ambient light intensity
   */
  setAmbientIntensity(intensity: number): void {
    this.ambientLight.intensity = intensity;
  }

  /**
   * Set directional light position
   */
  setDirectionalPosition(x: number, y: number, z: number): void {
    this.directionalLight.position.set(x, y, z);
  }

  /**
   * Set directional light intensity
   */
  setDirectionalIntensity(intensity: number): void {
    this.directionalLight.intensity = intensity;
  }

  /**
   * Enable/disable shadows
   */
  setShadowsEnabled(enabled: boolean): void {
    this.directionalLight.castShadow = enabled;
  }

  /**
   * Update shadow camera to match board size
   */
  updateShadowCamera(boardWidth: number, boardHeight: number, padding: number = 1.5): void {
    const maxDim = Math.max(boardWidth, boardHeight) * padding;
    const halfDim = maxDim / 2;
    
    this.directionalLight.shadow.camera.left = -halfDim;
    this.directionalLight.shadow.camera.right = halfDim;
    this.directionalLight.shadow.camera.top = halfDim;
    this.directionalLight.shadow.camera.bottom = -halfDim;
    this.directionalLight.shadow.camera.updateProjectionMatrix();
  }

  /**
   * Set ambient light color
   */
  setAmbientColor(color: string | number): void {
    this.ambientLight.color.set(color);
  }

  /**
   * Set directional light color
   */
  setDirectionalColor(color: string | number): void {
    this.directionalLight.color.set(color);
  }

  /**
   * Create a dramatic lighting setup
   */
  setDramatic(): void {
    this.ambientLight.intensity = 0.2;
    this.directionalLight.intensity = 1.5;
    this.directionalLight.position.set(5, 30, 10);
    this.fillLight.intensity = 0.1;
  }

  /**
   * Create a soft lighting setup
   */
  setSoft(): void {
    this.ambientLight.intensity = 0.7;
    this.directionalLight.intensity = 0.6;
    this.directionalLight.position.set(10, 20, 20);
    this.fillLight.intensity = 0.4;
  }

  /**
   * Create a studio lighting setup
   */
  setStudio(): void {
    this.ambientLight.intensity = 0.4;
    this.directionalLight.intensity = 0.8;
    this.directionalLight.position.set(15, 15, 20);
    this.fillLight.intensity = 0.3;
    this.fillLight.position.set(-15, -10, 15);
  }

  /**
   * Randomize lighting parameters
   */
  randomize(random: { range: (min: number, max: number) => number }): void {
    this.ambientLight.intensity = random.range(0.3, 0.7);
    this.directionalLight.intensity = random.range(0.6, 1.2);
    
    const distance = random.range(15, 30);
    const angle = random.range(0, Math.PI * 2);
    const height = random.range(15, 25);
    
    this.directionalLight.position.set(
      Math.cos(angle) * distance,
      Math.sin(angle) * distance,
      height
    );
  }

  /**
   * Dispose of lights
   */
  dispose(): void {
    this.ambientLight.dispose();
    this.directionalLight.dispose();
    this.fillLight.dispose();
  }
}

/**
 * Create a LightingRig instance
 */
export function createLightingRig(): LightingRig {
  return new LightingRig();
}

