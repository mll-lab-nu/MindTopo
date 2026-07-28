// CSG Boolean Operations wrapper using three-bvh-csg

import * as THREE from 'three';
import { Brush, Evaluator, SUBTRACTION } from 'three-bvh-csg';

/**
 * CSG Processor - Handles boolean operations on geometries
 * Uses three-bvh-csg for high-performance CSG operations
 */
export class CSGProcessor {
  private evaluator: Evaluator;

  constructor() {
    this.evaluator = new Evaluator();
  }

  /**
   * Create a brush from a mesh for CSG operations
   */
  createBrush(geometry: THREE.BufferGeometry, material?: THREE.Material): Brush {
    const brush = new Brush(geometry, material);
    brush.updateMatrixWorld();
    return brush;
  }

  /**
   * Create a brush from mesh with transformation
   */
  createBrushWithTransform(
    geometry: THREE.BufferGeometry,
    position: THREE.Vector3,
    rotation?: THREE.Euler,
    scale?: THREE.Vector3,
    material?: THREE.Material
  ): Brush {
    const brush = new Brush(geometry.clone(), material);
    
    brush.position.copy(position);
    if (rotation) {
      brush.rotation.copy(rotation);
    }
    if (scale) {
      brush.scale.copy(scale);
    }
    
    brush.updateMatrixWorld();
    return brush;
  }

  /**
   * Subtract geometry B from geometry A
   * Result = A - B (A with B carved out)
   */
  subtract(brushA: Brush, brushB: Brush): Brush {
    const result = this.evaluator.evaluate(brushA, brushB, SUBTRACTION);
    return result;
  }

  /**
   * Subtract multiple brushes from a base brush
   */
  subtractMultiple(base: Brush, cutters: Brush[]): Brush {
    let result = base;
    
    for (const cutter of cutters) {
      result = this.evaluator.evaluate(result, cutter, SUBTRACTION);
    }
    
    return result;
  }

  /**
   * Create a cylinder brush for hole cutting
   */
  createCylinderCutter(
    radius: number,
    height: number,
    position: THREE.Vector3,
    segments: number = 32
  ): Brush {
    const geometry = new THREE.CylinderGeometry(radius, radius, height, segments);
    // Rotate to align with Z-axis (up)
    geometry.rotateX(Math.PI / 2);
    
    const brush = new Brush(geometry);
    brush.position.copy(position);
    brush.updateMatrixWorld();
    
    return brush;
  }

  /**
   * Create a tapered cylinder brush (for distorted holes)
   */
  createTaperedCylinderCutter(
    radiusTop: number,
    radiusBottom: number,
    height: number,
    position: THREE.Vector3,
    segments: number = 32
  ): Brush {
    const geometry = new THREE.CylinderGeometry(radiusTop, radiusBottom, height, segments);
    geometry.rotateX(Math.PI / 2);
    
    const brush = new Brush(geometry);
    brush.position.copy(position);
    brush.updateMatrixWorld();
    
    return brush;
  }

  /**
   * Create a prism brush from a shape (for polygon holes)
   */
  createExtrudedShapeCutter(
    shape: THREE.Shape,
    depth: number,
    position: THREE.Vector3
  ): Brush {
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: depth,
      bevelEnabled: false
    });
    
    // Center the extrusion
    geometry.translate(0, 0, -depth / 2);
    
    const brush = new Brush(geometry);
    brush.position.copy(position);
    brush.updateMatrixWorld();
    
    return brush;
  }

  /**
   * Apply vertex distortion to a geometry
   */
  distortGeometry(
    geometry: THREE.BufferGeometry,
    noiseAmount: number,
    noiseFn: (x: number, y: number) => number
  ): THREE.BufferGeometry {
    const positions = geometry.getAttribute('position');
    const newGeometry = geometry.clone();
    const newPositions = newGeometry.getAttribute('position');
    
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);
      
      // Apply noise to x and y based on position
      const noiseX = noiseFn(x * 10, y * 10) * noiseAmount;
      const noiseY = noiseFn(x * 10 + 100, y * 10 + 100) * noiseAmount;
      
      newPositions.setXYZ(i, x + noiseX, y + noiseY, z);
    }
    
    newPositions.needsUpdate = true;
    newGeometry.computeVertexNormals();
    
    return newGeometry;
  }

  /**
   * Convert a Brush result back to a regular Mesh
   */
  brushToMesh(brush: Brush, material?: THREE.Material): THREE.Mesh {
    const geometry = brush.geometry.clone();
    const mat = material || brush.material as THREE.Material || new THREE.MeshStandardMaterial();
    
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.position.copy(brush.position);
    mesh.rotation.copy(brush.rotation);
    mesh.scale.copy(brush.scale);
    
    return mesh;
  }

  /**
   * Dispose of brushes to free memory
   */
  disposeBrush(brush: Brush): void {
    brush.geometry.dispose();
      if (brush.material) {
        if (Array.isArray(brush.material)) {
          brush.material.forEach((m: THREE.Material) => m.dispose());
        } else {
        brush.material.dispose();
      }
    }
  }

  /**
   * Dispose of multiple brushes
   */
  disposeBrushes(brushes: Brush[]): void {
    brushes.forEach(brush => this.disposeBrush(brush));
  }
}

// Singleton instance
let csgProcessorInstance: CSGProcessor | null = null;

export function getCSGProcessor(): CSGProcessor {
  if (!csgProcessorInstance) {
    csgProcessorInstance = new CSGProcessor();
  }
  return csgProcessorInstance;
}

