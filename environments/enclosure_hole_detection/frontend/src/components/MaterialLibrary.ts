// Material management and factory

import * as THREE from 'three';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { MaterialType, AppearanceParams } from '../types';
import { SeededRandom } from '../utils/RandomUtils';
import { simpleNoise2D } from '../utils/MathUtils';

const WOOD_0015_TEXTURES = {
  color: new URL('../../assets/textures/wood_0015/wood_0015_color_1k.jpg', import.meta.url).href,
  roughness: new URL('../../assets/textures/wood_0015/wood_0015_roughness_1k.jpg', import.meta.url).href,
  normal: new URL('../../assets/textures/wood_0015/wood_0015_normal_opengl_1k.png', import.meta.url).href,
  displacement: new URL('../../assets/textures/wood_0015/wood_0015_height_1k.png', import.meta.url).href,
  ao: new URL('../../assets/textures/wood_0015/wood_0015_ao_1k.jpg', import.meta.url).href
};

const ROCK_0005_TEXTURES = {
  color: new URL('../../assets/textures/rock_0005/rock_0005_color_1k.jpg', import.meta.url).href,
  roughness: new URL('../../assets/textures/rock_0005/rock_0005_roughness_1k.jpg', import.meta.url).href,
  normal: new URL('../../assets/textures/rock_0005/rock_0005_normal_opengl_1k.png', import.meta.url).href,
  displacement: new URL('../../assets/textures/rock_0005/rock_0005_height_1k.png', import.meta.url).href,
  ao: new URL('../../assets/textures/rock_0005/rock_0005_ao_1k.jpg', import.meta.url).href
};

const METAL_0058_TEXTURES = {
  color: new URL('../../assets/textures/metal_0058/metal_0058_color_4k.jpg', import.meta.url).href,
  roughness: new URL('../../assets/textures/metal_0058/metal_0058_roughness_4k.jpg', import.meta.url).href,
  normal: new URL('../../assets/textures/metal_0058/metal_0058_normal_opengl_4k.png', import.meta.url).href,
  displacement: new URL('../../assets/textures/metal_0058/metal_0058_height_4k.png', import.meta.url).href,
  ao: new URL('../../assets/textures/metal_0058/metal_0058_ao_4k.jpg', import.meta.url).href
};

const TEXTURE_REPEAT = {
  wood: [1, 1] as [number, number],
  rock: [5, 5] as [number, number],
  metal: [1, 1] as [number, number]
};

/**
 * Material Library - Creates and manages materials
 */
export class MaterialLibrary {
  private cache: Map<string, THREE.Material>;
  private random?: SeededRandom;
  private textureCache: Map<string, THREE.Texture>;
  private textureLoader: THREE.TextureLoader;
  private exrLoader: EXRLoader;
  private onTextureLoad?: () => void;

  constructor(random?: SeededRandom) {
    this.cache = new Map();
    this.random = random;
    this.textureCache = new Map();
    this.textureLoader = new THREE.TextureLoader();
    this.exrLoader = new EXRLoader();
  }

  /**
   * Set random generator for color variations
   */
  setRandom(random: SeededRandom): void {
    this.random = random;
  }

  setOnTextureLoad(callback?: () => void): void {
    this.onTextureLoad = callback;
  }

  /**
   * Create a material based on type
   */
  createMaterial(
    type: MaterialType,
    color: string | number,
    options: Partial<{
      roughness: number;
      metalness: number;
      shininess: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.Material {
    switch (type) {
      case 'phong':
        return this.createPhongMaterial(color, options);
      case 'standard':
        return this.createStandardMaterial(color, options);
      case 'wood':
        return this.createWoodMaterial(color, options);
      case 'rock':
        return this.createRockMaterial(color, options);
      case 'metal':
        return this.createMetalMaterial(color, options);
      default:
        return this.createStandardMaterial(color, options);
    }
  }

  /**
   * Create Phong material (classic shading with strong specular)
   */
  createPhongMaterial(
    color: string | number,
    options: Partial<{
      shininess: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.MeshPhongMaterial {
    return new THREE.MeshPhongMaterial({
      color,
      shininess: options.shininess ?? 80, // Higher shininess for visible specular
      specular: 0x444444, // Visible specular highlight
      emissive: options.emissive ?? 0x000000,
      opacity: options.opacity ?? 1.0,
      transparent: options.transparent ?? false,
      side: THREE.DoubleSide
    });
  }

  /**
   * Create Standard PBR material
   */
  createStandardMaterial(
    color: string | number,
    options: Partial<{
      roughness: number;
      metalness: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: options.roughness ?? 0.35, // Lower roughness for more visible reflections
      metalness: options.metalness ?? 0.2,  // Slight metalness for visual interest
      emissive: options.emissive ?? 0x000000,
      opacity: options.opacity ?? 1.0,
      transparent: options.transparent ?? false,
      side: THREE.DoubleSide,
      envMapIntensity: 0.5
    });
  }

  /**
   * Create textured wood material
   */
  createWoodMaterial(
    color: string | number,
    options: Partial<{
      roughness: number;
      metalness: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.MeshStandardMaterial {
    const textures = this.getWoodTextureSet();
    return new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.map,
      roughnessMap: textures.roughnessMap,
      normalMap: textures.normalMap,
      roughness: options.roughness ?? 0.9,
      metalness: options.metalness ?? 0.0,
      emissive: options.emissive ?? 0x000000,
      opacity: options.opacity ?? 1.0,
      transparent: options.transparent ?? false,
      side: THREE.DoubleSide
    });
  }

  /**
   * Create textured rock material
   */
  createRockMaterial(
    color: string | number,
    options: Partial<{
      roughness: number;
      metalness: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.MeshStandardMaterial {
    const textures = this.getRockTextureSet();
    return new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.map,
      roughnessMap: textures.roughnessMap,
      normalMap: textures.normalMap,
      roughness: options.roughness ?? 0.95,
      metalness: options.metalness ?? 0.0,
      emissive: options.emissive ?? 0x000000,
      opacity: options.opacity ?? 1.0,
      transparent: options.transparent ?? false,
      side: THREE.DoubleSide
    });
  }

  /**
   * Create textured metal material
   */
  createMetalMaterial(
    color: string | number,
    options: Partial<{
      roughness: number;
      metalness: number;
      emissive: string | number;
      opacity: number;
      transparent: boolean;
    }> = {}
  ): THREE.MeshStandardMaterial {
    const textures = this.getMetalTextureSet();
    return new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: textures.map,
      roughnessMap: textures.roughnessMap,
      normalMap: textures.normalMap,
      roughness: options.roughness ?? 0.75,
      metalness: options.metalness ?? 1.0,
      emissive: options.emissive ?? 0x000000,
      opacity: options.opacity ?? 1.0,
      transparent: options.transparent ?? false,
      side: THREE.DoubleSide
    });
  }

  /**
   * Create materials from appearance params
   */
  createFromParams(params: AppearanceParams): {
    boardMaterial: THREE.Material;
    underBoardMaterial: THREE.Material;
  } {
    const boardMaterial = this.createMaterial(
      params.materialType,
      params.boardColor
    );
    
    const underBoardMaterial = this.createStandardMaterial(
      params.underBoardColor,
      { roughness: 0.7, metalness: 0.0 }
    );
    
    return { boardMaterial, underBoardMaterial };
  }

  preloadTextureSets(): void {
    this.getWoodTextureSet();
    this.getRockTextureSet();
    this.getMetalTextureSet();
  }

  private getWoodTextureSet(): {
    map: THREE.Texture;
    roughnessMap?: THREE.Texture;
    normalMap?: THREE.Texture;
  } {
    const map = this.loadTexture(WOOD_0015_TEXTURES.color, {
      colorSpace: THREE.SRGBColorSpace,
      repeat: TEXTURE_REPEAT.wood
    });
    const roughnessMap = this.loadTexture(WOOD_0015_TEXTURES.roughness, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.wood
    });
    const normalMap = this.loadTexture(WOOD_0015_TEXTURES.normal, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.wood
    });
    
    return { map, roughnessMap, normalMap };
  }

  private getRockTextureSet(): {
    map: THREE.Texture;
    roughnessMap?: THREE.Texture;
    normalMap?: THREE.Texture;
  } {
    const map = this.loadTexture(ROCK_0005_TEXTURES.color, {
      colorSpace: THREE.SRGBColorSpace,
      repeat: TEXTURE_REPEAT.rock
    });
    const roughnessMap = this.loadTexture(ROCK_0005_TEXTURES.roughness, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.rock
    });
    const normalMap = this.loadTexture(ROCK_0005_TEXTURES.normal, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.rock
    });
    
    return { map, roughnessMap, normalMap };
  }

  private getMetalTextureSet(): {
    map: THREE.Texture;
    roughnessMap?: THREE.Texture;
    normalMap?: THREE.Texture;
  } {
    const map = this.loadTexture(METAL_0058_TEXTURES.color, {
      colorSpace: THREE.SRGBColorSpace,
      repeat: TEXTURE_REPEAT.metal
    });
    const roughnessMap = this.loadTexture(METAL_0058_TEXTURES.roughness, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.metal
    });
    const normalMap = this.loadTexture(METAL_0058_TEXTURES.normal, {
      colorSpace: THREE.NoColorSpace,
      repeat: TEXTURE_REPEAT.metal
    });
    
    return { map, roughnessMap, normalMap };
  }

  private loadTexture(
    url: string,
    options: {
      colorSpace: THREE.ColorSpace;
      repeat: [number, number];
      isExr?: boolean;
    }
  ): THREE.Texture {
    const cacheKey = `${url}|${options.colorSpace}|${options.repeat[0]}x${options.repeat[1]}`;
    const cached = this.textureCache.get(cacheKey);
    if (cached) return cached;
    
    const loader = options.isExr ? this.exrLoader : this.textureLoader;
    const texture = loader.load(
      url,
      () => {
        this.onTextureLoad?.();
      },
      undefined,
      (err) => {
        console.warn('Failed to load texture', url, err);
      }
    );
    
    this.configureTexture(texture, options.colorSpace, options.repeat);
    this.textureCache.set(cacheKey, texture);
    return texture;
  }

  private configureTexture(
    texture: THREE.Texture,
    colorSpace: THREE.ColorSpace,
    repeat: [number, number]
  ): void {
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(repeat[0], repeat[1]);
    texture.colorSpace = colorSpace;
    texture.flipY = true;
    texture.anisotropy = 8;
  }

  private createWoodTexture(color: string | number): THREE.Texture {
    const size = 256;
    const base = this.colorToRgb(color);
    const seed = Math.floor(this.rand() * 10000);
    const cx = size * (0.3 + this.rand() * 0.4);
    const cy = size * (0.3 + this.rand() * 0.4);
    const freq = 0.06 + this.rand() * 0.05;
    
    return this.createCanvasTexture(size, (ctx) => {
      const image = ctx.createImageData(size, size);
      const data = image.data;
      
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dx = x - cx;
          const dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const noise = simpleNoise2D(x * 0.05, y * 0.05, seed);
          const grain = simpleNoise2D(x * 0.2, y * 0.2, seed + 13);
          const ring = Math.sin(dist * freq + noise * 4);
          const v = 0.65 + ring * 0.18 + grain * 0.08;
          
          const r = this.clamp255(base.r * v);
          const g = this.clamp255(base.g * v);
          const b = this.clamp255(base.b * v);
          const idx = (y * size + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = 255;
        }
      }
      
      ctx.putImageData(image, 0, 0);
    }, 2, 2);
  }

  private createMetalTexture(color: string | number): THREE.Texture {
    const size = 256;
    const base = this.colorToRgb(color);
    const seed = Math.floor(this.rand() * 10000);
    const stripeFreq = 0.35 + this.rand() * 0.2;
    
    return this.createCanvasTexture(size, (ctx) => {
      const image = ctx.createImageData(size, size);
      const data = image.data;
      
      for (let y = 0; y < size; y++) {
        const rowNoise = simpleNoise2D(0, y * 0.1, seed);
        for (let x = 0; x < size; x++) {
          const noise = simpleNoise2D(x * 0.2, y * 0.05, seed + 7);
          const stripe = Math.sin((y + noise * 8) * stripeFreq);
          const v = 0.7 + stripe * 0.08 + (noise - 0.5) * 0.1 + (rowNoise - 0.5) * 0.08;
          
          const r = this.clamp255(base.r * v);
          const g = this.clamp255(base.g * v);
          const b = this.clamp255(base.b * v);
          const idx = (y * size + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = 255;
        }
      }
      
      ctx.putImageData(image, 0, 0);
    }, 3, 3);
  }

  private createCanvasTexture(
    size: number,
    draw: (ctx: CanvasRenderingContext2D) => void,
    repeatX: number,
    repeatY: number
  ): THREE.Texture {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      draw(ctx);
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(repeatX, repeatY);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    texture.needsUpdate = true;
    return texture;
  }

  private rand(): number {
    return this.random ? this.random.random() : Math.random();
  }

  private colorToRgb(color: string | number): { r: number; g: number; b: number } {
    const c = new THREE.Color(color as number | string);
    return {
      r: Math.round(c.r * 255),
      g: Math.round(c.g * 255),
      b: Math.round(c.b * 255)
    };
  }

  private clamp255(value: number): number {
    return Math.max(0, Math.min(255, Math.round(value)));
  }

  /**
   * Create a cached material (reusable)
   */
  getOrCreate(
    key: string,
    type: MaterialType,
    color: string | number,
    options: Record<string, unknown> = {}
  ): THREE.Material {
    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }
    
    const material = this.createMaterial(type, color, options);
    this.cache.set(key, material);
    return material;
  }

  /**
   * Generate a random color
   */
  randomColor(): string {
    if (!this.random) {
      return '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0');
    }
    
    // Generate a pleasing random color using HSL
    const h = this.random.range(0, 360);
    const s = this.random.range(50, 80);
    const l = this.random.range(40, 60);
    
    return this.hslToHex(h, s, l);
  }

  /**
   * Generate a complementary color
   */
  complementaryColor(baseColor: string): string {
    const rgb = this.hexToRgb(baseColor);
    if (!rgb) return '#333333';
    
    return this.rgbToHex(255 - rgb.r, 255 - rgb.g, 255 - rgb.b);
  }

  /**
   * Generate a darker version of a color
   */
  darkenColor(color: string, factor: number = 0.3): string {
    const rgb = this.hexToRgb(color);
    if (!rgb) return '#000000';
    
    return this.rgbToHex(
      Math.floor(rgb.r * (1 - factor)),
      Math.floor(rgb.g * (1 - factor)),
      Math.floor(rgb.b * (1 - factor))
    );
  }

  /**
   * Generate a lighter version of a color
   */
  lightenColor(color: string, factor: number = 0.3): string {
    const rgb = this.hexToRgb(color);
    if (!rgb) return '#ffffff';
    
    return this.rgbToHex(
      Math.min(255, Math.floor(rgb.r + (255 - rgb.r) * factor)),
      Math.min(255, Math.floor(rgb.g + (255 - rgb.g) * factor)),
      Math.min(255, Math.floor(rgb.b + (255 - rgb.b) * factor))
    );
  }

  /**
   * Convert HSL to hex color
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
    
    return this.rgbToHex(
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255)
    );
  }

  /**
   * Convert hex to RGB
   */
  private hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  }

  /**
   * Convert RGB to hex
   */
  private rgbToHex(r: number, g: number, b: number): string {
    return '#' + [r, g, b].map(x => {
      const hex = Math.max(0, Math.min(255, x)).toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    }).join('');
  }

  /**
   * Create preset material sets
   */
  getPreset(name: 'metal' | 'plastic' | 'wood' | 'matte'): {
    type: MaterialType;
    options: Record<string, unknown>;
  } {
    switch (name) {
      case 'metal':
        return { type: 'standard', options: { roughness: 0.0, metalness: 0.9 } };
      case 'plastic':
        return { type: 'standard', options: { roughness: 0.4, metalness: 0.0 } };
      case 'wood':
        return { type: 'standard', options: { roughness: 0.8, metalness: 0.0 } };
      case 'matte':
        return { type: 'standard', options: { roughness: 1.0, metalness: 0.0 } };
      default:
        return { type: 'standard', options: {} };
    }
  }

  /**
   * Dispose all cached materials
   */
  dispose(): void {
    this.cache.forEach(material => material.dispose());
    this.cache.clear();
    this.textureCache.forEach(texture => texture.dispose());
    this.textureCache.clear();
  }

  /**
   * Dispose a specific material
   */
  disposeMaterial(material: THREE.Material): void {
    material.dispose();
  }
}

/**
 * Create a MaterialLibrary instance
 */
export function createMaterialLibrary(random?: SeededRandom): MaterialLibrary {
  return new MaterialLibrary(random);
}
