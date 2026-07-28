// Board geometry factory - supports various board shapes

import * as THREE from 'three';
import { Brush } from 'three-bvh-csg';
import { BoardConfig } from '../types';
import { getCSGProcessor } from './CSGProcessor';
import { regularPolygonPoints, pointsToShape } from '../utils/MathUtils';
import { SeededRandom } from '../utils/RandomUtils';

/**
 * Factory for creating board geometries
 * Supports: rect, circle, polygon, rounded_rect, ellipse, star, cross, l_shape, irregular
 */
export class BoardFactory {
  private csg = getCSGProcessor();

  /**
   * Create a board geometry based on configuration
   */
  createBoard(config: BoardConfig, random?: SeededRandom): THREE.BufferGeometry {
    const { width, height } = config.size;
    const { thickness } = config;
    const minDim = Math.min(width, height);
    
    switch (config.shape) {
      case 'rect':
        return this.createRectBoard(width, height, thickness);
        
      case 'circle':
        return this.createCircleBoard(minDim / 2, thickness);
        
      case 'polygon': {
        const sides = random ? random.rangeInt(5, 8) : 6;
        return this.createPolygonBoard(minDim / 2, thickness, sides);
      }
      
      case 'rounded_rect':
        return this.createRoundedRectBoard(width, height, thickness, minDim * 0.15);
        
      case 'ellipse':
        return this.createEllipseBoard(width / 2, height / 2, thickness);
        
      case 'star': {
        const points = random ? random.rangeInt(5, 8) : 5;
        return this.createStarBoard(minDim / 2, thickness, points, random);
      }
      
      case 'cross':
        return this.createCrossBoard(width, height, thickness, random);
        
      case 'l_shape':
        return this.createLShapeBoard(width, height, thickness, random);
        
      case 'irregular':
        return this.createIrregularBoard(minDim / 2, thickness, random);
        
      default:
        return this.createRectBoard(width, height, thickness);
    }
  }

  /**
   * Create a rectangular board
   */
  createRectBoard(width: number, height: number, thickness: number): THREE.BufferGeometry {
    const geometry = new THREE.BoxGeometry(width, height, thickness);
    return geometry;
  }

  /**
   * Create a circular board
   */
  createCircleBoard(radius: number, thickness: number, segments: number = 64): THREE.BufferGeometry {
    const geometry = new THREE.CylinderGeometry(radius, radius, thickness, segments);
    // Rotate to lay flat (Z-up)
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }

  /**
   * Create a regular polygon board
   */
  createPolygonBoard(
    radius: number,
    thickness: number,
    sides: number = 6,
    startAngle: number = 0
  ): THREE.BufferGeometry {
    const points = regularPolygonPoints(sides, radius, startAngle);
    const shape = pointsToShape(points);
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    
    // Center the geometry on Z-axis
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create a rounded rectangle board
   */
  createRoundedRectBoard(
    width: number,
    height: number,
    thickness: number,
    cornerRadius: number
  ): THREE.BufferGeometry {
    const shape = new THREE.Shape();
    const r = Math.min(cornerRadius, width / 4, height / 4);
    const w = width / 2;
    const h = height / 2;
    
    shape.moveTo(-w + r, -h);
    shape.lineTo(w - r, -h);
    shape.quadraticCurveTo(w, -h, w, -h + r);
    shape.lineTo(w, h - r);
    shape.quadraticCurveTo(w, h, w - r, h);
    shape.lineTo(-w + r, h);
    shape.quadraticCurveTo(-w, h, -w, h - r);
    shape.lineTo(-w, -h + r);
    shape.quadraticCurveTo(-w, -h, -w + r, -h);
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create an ellipse board
   */
  createEllipseBoard(
    radiusX: number,
    radiusY: number,
    thickness: number,
    segments: number = 64
  ): THREE.BufferGeometry {
    const shape = new THREE.Shape();
    shape.ellipse(0, 0, radiusX, radiusY, 0, Math.PI * 2, false, 0);
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false,
      curveSegments: segments
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create a star-shaped board
   */
  createStarBoard(
    outerRadius: number,
    thickness: number,
    points: number = 5,
    random?: SeededRandom
  ): THREE.BufferGeometry {
    const innerRadius = outerRadius * (random ? random.range(0.4, 0.6) : 0.5);
    const shape = new THREE.Shape();
    const angleStep = Math.PI / points;
    
    for (let i = 0; i < points * 2; i++) {
      const radius = i % 2 === 0 ? outerRadius : innerRadius;
      const angle = i * angleStep - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      if (i === 0) {
        shape.moveTo(x, y);
      } else {
        shape.lineTo(x, y);
      }
    }
    shape.closePath();
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create a cross/plus-shaped board
   */
  createCrossBoard(
    width: number,
    height: number,
    thickness: number,
    random?: SeededRandom
  ): THREE.BufferGeometry {
    const armWidth = (random ? random.range(0.25, 0.4) : 0.3) * Math.min(width, height);
    const shape = new THREE.Shape();
    const hw = width / 2;
    const hh = height / 2;
    const aw = armWidth / 2;
    
    // Draw cross shape
    shape.moveTo(-aw, -hh);
    shape.lineTo(aw, -hh);
    shape.lineTo(aw, -aw);
    shape.lineTo(hw, -aw);
    shape.lineTo(hw, aw);
    shape.lineTo(aw, aw);
    shape.lineTo(aw, hh);
    shape.lineTo(-aw, hh);
    shape.lineTo(-aw, aw);
    shape.lineTo(-hw, aw);
    shape.lineTo(-hw, -aw);
    shape.lineTo(-aw, -aw);
    shape.closePath();
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create an L-shaped board
   */
  createLShapeBoard(
    width: number,
    height: number,
    thickness: number,
    random?: SeededRandom
  ): THREE.BufferGeometry {
    const cutRatio = random ? random.range(0.4, 0.6) : 0.5;
    const shape = new THREE.Shape();
    const hw = width / 2;
    const hh = height / 2;
    
    // L-shape (bottom-left to top-right cut)
    shape.moveTo(-hw, -hh);
    shape.lineTo(hw, -hh);
    shape.lineTo(hw, -hh + height * (1 - cutRatio));
    shape.lineTo(-hw + width * cutRatio, -hh + height * (1 - cutRatio));
    shape.lineTo(-hw + width * cutRatio, hh);
    shape.lineTo(-hw, hh);
    shape.closePath();
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create an irregular blob-shaped board
   */
  createIrregularBoard(
    radius: number,
    thickness: number,
    random?: SeededRandom
  ): THREE.BufferGeometry {
    const points: { x: number; y: number }[] = [];
    const numPoints = random ? random.rangeInt(8, 16) : 12;
    
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      const r = radius * (0.7 + (random ? random.range(0, 0.6) : Math.random() * 0.6));
      points.push({
        x: Math.cos(angle) * r,
        y: Math.sin(angle) * r
      });
    }
    
    // Create smooth curve through points
    const shape = new THREE.Shape();
    shape.moveTo(points[0].x, points[0].y);
    
    for (let i = 0; i < points.length; i++) {
      const p0 = points[i];
      const p1 = points[(i + 1) % points.length];
      const p2 = points[(i + 2) % points.length];
      
      const cp1x = p1.x - (p2.x - p0.x) * 0.2;
      const cp1y = p1.y - (p2.y - p0.y) * 0.2;
      const cp2x = p1.x + (p2.x - p0.x) * 0.2;
      const cp2y = p1.y + (p2.y - p0.y) * 0.2;
      
      shape.quadraticCurveTo(cp1x, cp1y, (p0.x + p1.x) / 2, (p0.y + p1.y) / 2);
    }
    shape.closePath();
    
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: thickness,
      bevelEnabled: false,
      curveSegments: 32
    });
    geometry.translate(0, 0, -thickness / 2);
    
    return geometry;
  }

  /**
   * Create a board as a CSG Brush (for boolean operations)
   */
  createBoardBrush(config: BoardConfig, material?: THREE.Material, random?: SeededRandom): Brush {
    const geometry = this.createBoard(config, random);
    return this.csg.createBrush(geometry, material);
  }

  /**
   * Create an under-board (bottom layer for pits)
   */
  createUnderBoard(
    config: BoardConfig,
    offsetZ: number,
    material?: THREE.Material,
    random?: SeededRandom
  ): THREE.Mesh {
    const geometry = this.createBoard(config, random);
    const mesh = new THREE.Mesh(
      geometry,
      material || new THREE.MeshStandardMaterial({ color: 0x333333 })
    );
    
    // Position below the main board
    mesh.position.z = -offsetZ - config.thickness / 2;
    
    return mesh;
  }

  /**
   * Create a partial under-board (for mixed holes)
   * Returns a mesh that only covers specific areas
   */
  createPartialUnderBoard(
    config: BoardConfig,
    holePlacements: Array<{ x: number; y: number; radius: number; covered: boolean }>,
    offsetZ: number,
    material?: THREE.Material,
    random?: SeededRandom
  ): THREE.Mesh | null {
    // Create base under-board geometry
    const baseGeometry = this.createBoard(config, random);
    let baseBrush = this.csg.createBrush(baseGeometry, material);
    
    // Cut out areas that should NOT be covered (holes, not pits)
    const uncoveredHoles = holePlacements.filter(h => !h.covered);
    
    if (uncoveredHoles.length === 0) {
      // Full coverage
      const mesh = new THREE.Mesh(
        baseGeometry,
        material || new THREE.MeshStandardMaterial({ color: 0x333333 })
      );
      mesh.position.z = -offsetZ - config.thickness / 2;
      return mesh;
    }
    
    // Create cutters for uncovered areas
    for (const hole of uncoveredHoles) {
      const cutterBrush = this.csg.createCylinderCutter(
        hole.radius * 1.1, // Slightly larger to ensure full cutout
        config.thickness * 2,
        new THREE.Vector3(hole.x, hole.y, 0)
      );
      baseBrush = this.csg.subtract(baseBrush, cutterBrush);
    }
    
    const mesh = this.csg.brushToMesh(baseBrush, material);
    mesh.position.z = -offsetZ - config.thickness / 2;
    
    return mesh;
  }

  /**
   * Get the bounding dimensions for a board shape
   */
  getBoardBounds(config: BoardConfig): { width: number; height: number; radius?: number } {
    switch (config.shape) {
      case 'rect':
        return { width: config.size.width, height: config.size.height };
      case 'circle':
        const radius = Math.min(config.size.width, config.size.height) / 2;
        return { width: radius * 2, height: radius * 2, radius };
      case 'polygon':
        const polyRadius = Math.min(config.size.width, config.size.height) / 2;
        // Inscribed rectangle approximation
        return { width: polyRadius * 1.5, height: polyRadius * 1.5, radius: polyRadius };
      default:
        return { width: config.size.width, height: config.size.height };
    }
  }
}

// Singleton instance
let boardFactoryInstance: BoardFactory | null = null;

export function getBoardFactory(): BoardFactory {
  if (!boardFactoryInstance) {
    boardFactoryInstance = new BoardFactory();
  }
  return boardFactoryInstance;
}

