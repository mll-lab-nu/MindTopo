// Hole shape generation - SIMPLE AND CONSISTENT
// All holes are 2D shapes extruded into prisms, only differing in depth

import * as THREE from 'three';
import { Brush } from 'three-bvh-csg';
import { HolePlacement, ShapeVariant, HoleShape } from '../types';
import { getCSGProcessor } from './CSGProcessor';
import { SeededRandom } from '../utils/RandomUtils';
import { regularPolygonPoints, pointsToShape } from '../utils/MathUtils';

// Available hole shapes (simplified, no complex multi-part shapes)
const ALL_HOLE_SHAPES: HoleShape[] = [
      'circle', 'ellipse', 'polygon', 'star', 'keyhole', 'slot', 'cross', 'crescent', 'heart'
];

/**
 * Generates hole cutting geometries
 * 
 * CORE PRINCIPLE: All holes are just 2D shapes extruded into prisms
 * - hole: full depth prism (cuts through board)
 * - pit: partial depth prism (doesn't cut through)
 * - mixed: full depth prism (with cover added separately in SceneGenerator)
 */
export class HoleShaper {
  private csg = getCSGProcessor();
  private random: SeededRandom;
  
  // Cache for shapes to ensure consistency
  private shapeCache: Map<string, THREE.Shape> = new Map();

  constructor(random: SeededRandom) {
    this.random = random;
  }

  /**
   * Generate a deterministic random value based on placement
   * This ensures the SAME shape is generated for the SAME placement
   */
  private getPlacementSeed(placement: HolePlacement): number {
    const x = placement.x;
    const y = placement.y;
    const r = placement.radius;
    // Use position and radius to create a deterministic seed
    return Math.abs(Math.sin(x * 12.9898 + y * 78.233 + r * 37.719) * 43758.5453) % 1;
  }

  /**
   * Get a deterministic random value in range based on placement
   */
  private placementRange(placement: HolePlacement, min: number, max: number, offset: number = 0): number {
    const seed = this.getPlacementSeed(placement);
    // Add offset to get different values for different properties
    const val = Math.abs(Math.sin(seed * 1000 + offset * 7.3)) % 1;
    return min + val * (max - min);
  }

  /**
   * Get a deterministic integer in range based on placement
   */
  private placementRangeInt(placement: HolePlacement, min: number, max: number, offset: number = 0): number {
    return Math.floor(this.placementRange(placement, min, max + 1, offset));
  }

  /**
   * Get cache key for a placement
   */
  private getCacheKey(placement: HolePlacement): string {
    return `${placement.x.toFixed(4)}_${placement.y.toFixed(4)}_${placement.radius.toFixed(4)}_${placement.shape || 'circle'}`;
  }

  /**
   * Get the 2D shape for a hole placement
   * This is the SINGLE SOURCE OF TRUTH for the hole's shape
   * Uses caching and deterministic randomness for consistency
   */
  getHoleShape(placement: HolePlacement): THREE.Shape {
    const cacheKey = this.getCacheKey(placement);
    
    // Return cached shape if exists
    if (this.shapeCache.has(cacheKey)) {
      return this.shapeCache.get(cacheKey)!.clone();
    }
    
    const holeShapeType = placement.shape || this.random.pick(ALL_HOLE_SHAPES);
    const rotation = placement.rotation || 0;
    const shape = this.createShapeGeometry(holeShapeType, placement.radius, rotation, placement);
    
    // Cache the shape
    this.shapeCache.set(cacheKey, shape);
    
    return shape.clone();
  }

  /**
   * Create a cutter brush for a HOLE (through-hole, full depth)
   */
  createHoleCutter(
    placement: HolePlacement,
    variant: ShapeVariant,
    boardThickness: number
  ): Brush {
    const height = boardThickness * 3; // Ensure it cuts through completely
    const position = new THREE.Vector3(placement.x, placement.y, 0);
    
    // Get the 2D shape
    const shape = this.getHoleShape(placement);
    
    // Create extruded prism cutter
    if (variant === 'distorted') {
      return this.createTaperedPrismCutter(shape, height, position, placement);
    }
    
    return this.createPrismCutter(shape, height, position);
  }

  /**
   * Create a cutter brush for a PIT (partial depth, has bottom)
   * Uses the SAME 2D shape as hole, just shorter height
   */
  createPitCutter(
    placement: HolePlacement,
    variant: ShapeVariant,
    boardThickness: number
  ): Brush {
    // Pit depth from placement
    const height = placement.depth + 0.1; // Slightly deeper for clean cut
    
    // Position from top of board going down
    const position = new THREE.Vector3(
      placement.x,
      placement.y,
      boardThickness / 2 - height / 2 + 0.05
    );
    
    // Get the SAME 2D shape as would be used for hole
    const shape = this.getHoleShape(placement);
    
    // Create extruded prism cutter (same shape, just different height/position)
    if (variant === 'distorted') {
      return this.createTaperedPrismCutter(shape, height, position, placement);
    }
    
    return this.createPrismCutter(shape, height, position);
  }

  /**
   * Create appropriate cutters based on hole types
   */
  createTypedCutters(
    placements: HolePlacement[],
    variant: ShapeVariant,
    boardThickness: number
  ): Brush[] {
    const cutters: Brush[] = [];
    
    for (const placement of placements) {
      if (placement.type === 'pit') {
        cutters.push(this.createPitCutter(placement, variant, boardThickness));
      } else {
        // 'hole', 'mixed', or 'mixed2' - full depth cut
        cutters.push(this.createHoleCutter(placement, variant, boardThickness));
      }
    }
    
    return cutters;
  }

  /**
   * Create a simple prism cutter from 2D shape
   */
  private createPrismCutter(
    shape: THREE.Shape,
    height: number,
    position: THREE.Vector3
  ): Brush {
    return this.csg.createExtrudedShapeCutter(shape, height, position);
  }

  /**
   * Create a tapered prism cutter (frustum-style)
   * Far face area is at least 50% of the near face area.
   */
  private createTaperedPrismCutter(
    shape: THREE.Shape,
    height: number,
    position: THREE.Vector3,
    placement: HolePlacement
  ): Brush {
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: height,
      bevelEnabled: false
    });
    geometry.translate(0, 0, -height / 2);
    
    const minAreaRatio = 0.5;
    const minScale = Math.sqrt(minAreaRatio);
    const maxScale = 0.95; // Keep far face smaller than near face
    const bottomScale = this.placementRange(placement, minScale, maxScale, 101);
    const topScale = 1.0;
    
    // Taper along the Z axis: near face larger, far face smaller
    const positions = geometry.getAttribute('position');
    let minZ = Infinity;
    let maxZ = -Infinity;
    
    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i);
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    
    const zRange = maxZ - minZ || 1;
    
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);
      const t = (z - minZ) / zRange;
      const scale = bottomScale + t * (topScale - bottomScale);
      
      positions.setX(i, x * scale);
      positions.setY(i, y * scale);
    }
    
    positions.needsUpdate = true;
    geometry.computeVertexNormals();
    
    const brush = this.csg.createBrush(geometry);
    brush.position.copy(position);
    brush.updateMatrixWorld();
    
    return brush;
  }

  /**
   * Create a 2D shape based on shape type
   * Uses placement for deterministic random values
   */
  private createShapeGeometry(
    shapeType: HoleShape,
    radius: number,
    rotation: number,
    placement: HolePlacement
  ): THREE.Shape {
    let shape: THREE.Shape;
    
    switch (shapeType) {
      case 'circle':
        shape = this.createCircleShape(radius);
        break;
      case 'ellipse':
        shape = this.createEllipseShape(radius, placement);
        break;
      case 'polygon':
        shape = this.createPolygonShape(radius, placement);
        break;
      case 'star':
        shape = this.createStarShape(radius, placement);
        break;
      case 'slot':
        shape = this.createSlotShape(radius, placement);
        break;
      case 'cross':
        shape = this.createCrossShape(radius, placement);
        break;
      case 'blob':
        shape = this.createBlobShape(radius, placement);
        break;
      case 'heart':
        shape = this.createHeartShape(radius);
        break;
      default:
        shape = this.createCircleShape(radius);
    }
    
    // Apply rotation if needed
    if (rotation !== 0) {
      shape = this.rotateShape(shape, rotation);
    }
    
    return shape;
  }

  // ==================== SIMPLE 2D SHAPE GENERATORS ====================
  // All use deterministic random values based on placement for consistency

  private createCircleShape(radius: number): THREE.Shape {
    const shape = new THREE.Shape();
    shape.absarc(0, 0, radius, 0, Math.PI * 2, false);
    return shape;
  }

  private createEllipseShape(radius: number, placement: HolePlacement): THREE.Shape {
    const shape = new THREE.Shape();
    const rx = radius * this.placementRange(placement, 0.6, 1.0, 1);
    const ry = radius * this.placementRange(placement, 1.0, 1.4, 2);
    shape.ellipse(0, 0, rx, ry, 0, Math.PI * 2, false, 0);
    return shape;
  }

  private createPolygonShape(radius: number, placement: HolePlacement): THREE.Shape {
    const sides = this.placementRangeInt(placement, 3, 8, 3);
    const points = regularPolygonPoints(sides, radius, 0);
    return pointsToShape(points);
  }

  private createStarShape(radius: number, placement: HolePlacement): THREE.Shape {
    const shape = new THREE.Shape();
    const numPoints = this.placementRangeInt(placement, 4, 7, 4);
    const innerRadius = radius * this.placementRange(placement, 0.35, 0.55, 5);
    const angleStep = Math.PI / numPoints;
    
    for (let i = 0; i < numPoints * 2; i++) {
      const r = i % 2 === 0 ? radius : innerRadius;
      const angle = i * angleStep - Math.PI / 2;
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r;
      
      if (i === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    }
    shape.closePath();
    return shape;
  }

  private createSlotShape(radius: number, placement: HolePlacement): THREE.Shape {
    const shape = new THREE.Shape();
    const length = radius * this.placementRange(placement, 1.5, 2.5, 6);
    const width = radius * this.placementRange(placement, 0.4, 0.6, 7);
    
    // Stadium/capsule shape
    shape.moveTo(-length / 2 + width, -width);
    shape.lineTo(length / 2 - width, -width);
    shape.absarc(length / 2 - width, 0, width, -Math.PI / 2, Math.PI / 2, false);
    shape.lineTo(-length / 2 + width, width);
    shape.absarc(-length / 2 + width, 0, width, Math.PI / 2, -Math.PI / 2, false);
    shape.closePath();
    return shape;
  }

  private createCrossShape(radius: number, placement: HolePlacement): THREE.Shape {
    const shape = new THREE.Shape();
    const armWidth = radius * this.placementRange(placement, 0.3, 0.45, 8);
    const armLength = radius * 0.95;
    const hw = armWidth / 2;
    
    shape.moveTo(-hw, -armLength);
    shape.lineTo(hw, -armLength);
    shape.lineTo(hw, -hw);
    shape.lineTo(armLength, -hw);
    shape.lineTo(armLength, hw);
    shape.lineTo(hw, hw);
    shape.lineTo(hw, armLength);
    shape.lineTo(-hw, armLength);
    shape.lineTo(-hw, hw);
    shape.lineTo(-armLength, hw);
    shape.lineTo(-armLength, -hw);
    shape.lineTo(-hw, -hw);
    shape.closePath();
    return shape;
  }

  private createBlobShape(radius: number, placement: HolePlacement): THREE.Shape {
    const shape = new THREE.Shape();
    const numPoints = this.placementRangeInt(placement, 6, 10, 9);
    
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      // Use different offset for each point to create varied blob
      const r = radius * (0.65 + this.placementRange(placement, 0, 0.5, 10 + i));
      points.push({
        x: Math.cos(angle) * r,
        y: Math.sin(angle) * r
      });
    }
    
    // Smooth curve through points
    shape.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length; i++) {
      const p1 = points[(i + 1) % points.length];
      const midX = (points[i].x + p1.x) / 2;
      const midY = (points[i].y + p1.y) / 2;
      shape.quadraticCurveTo(p1.x, p1.y, midX, midY);
    }
    shape.closePath();
    return shape;
  }

  private createHeartShape(radius: number): THREE.Shape {
    const shape = new THREE.Shape();
    const scale = radius * 0.7;
    
    shape.moveTo(0, -scale * 0.7);
    shape.bezierCurveTo(
      scale * 0.5, -scale * 1.2,
      scale * 1.3, -scale * 0.4,
      0, scale * 0.6
    );
    shape.bezierCurveTo(
      -scale * 1.3, -scale * 0.4,
      -scale * 0.5, -scale * 1.2,
      0, -scale * 0.7
    );
    return shape;
  }

  /**
   * Rotate a shape around origin
   */
  private rotateShape(shape: THREE.Shape, angle: number): THREE.Shape {
    if (angle === 0) return shape;
    
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const points = shape.getPoints(32);
    
    const rotatedShape = new THREE.Shape();
    for (let i = 0; i < points.length; i++) {
      const x = points[i].x * cos - points[i].y * sin;
      const y = points[i].x * sin + points[i].y * cos;
      
      if (i === 0) rotatedShape.moveTo(x, y);
      else rotatedShape.lineTo(x, y);
    }
    rotatedShape.closePath();
    return rotatedShape;
  }
}

/**
 * Create a HoleShaper instance
 */
export function createHoleShaper(random: SeededRandom): HoleShaper {
  return new HoleShaper(random);
}
