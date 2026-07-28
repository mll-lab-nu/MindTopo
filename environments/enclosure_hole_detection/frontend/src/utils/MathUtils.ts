// Mathematical utility functions

import * as THREE from 'three';

/**
 * Convert degrees to radians
 */
export function degToRad(degrees: number): number {
  return degrees * (Math.PI / 180);
}

/**
 * Convert radians to degrees
 */
export function radToDeg(radians: number): number {
  return radians * (180 / Math.PI);
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Linear interpolation
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Calculate distance between two 2D points
 */
export function distance2D(
  x1: number, y1: number,
  x2: number, y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Check if a point is inside a rectangle
 */
export function pointInRect(
  px: number, py: number,
  rx: number, ry: number,
  rw: number, rh: number
): boolean {
  const halfW = rw / 2;
  const halfH = rh / 2;
  return px >= rx - halfW && px <= rx + halfW &&
         py >= ry - halfH && py <= ry + halfH;
}

/**
 * Check if a point is inside a circle
 */
export function pointInCircle(
  px: number, py: number,
  cx: number, cy: number,
  radius: number
): boolean {
  return distance2D(px, py, cx, cy) <= radius;
}

/**
 * Check if two circles overlap
 */
export function circlesOverlap(
  x1: number, y1: number, r1: number,
  x2: number, y2: number, r2: number,
  minGap: number = 0
): boolean {
  const dist = distance2D(x1, y1, x2, y2);
  return dist < (r1 + r2 + minGap);
}

/**
 * Check if a circle fits inside a rectangle with buffer
 */
export function circleFitsInRect(
  cx: number, cy: number, cr: number,
  rw: number, rh: number,
  buffer: number = 0
): boolean {
  const halfW = rw / 2;
  const halfH = rh / 2;
  const effectiveRadius = cr + buffer;
  
  return cx - effectiveRadius >= -halfW &&
         cx + effectiveRadius <= halfW &&
         cy - effectiveRadius >= -halfH &&
         cy + effectiveRadius <= halfH;
}

/**
 * Check if a circle fits inside a circular boundary with buffer
 */
export function circleFitsInCircle(
  cx: number, cy: number, cr: number,
  boundaryRadius: number,
  buffer: number = 0
): boolean {
  const dist = distance2D(cx, cy, 0, 0);
  return dist + cr + buffer <= boundaryRadius;
}

/**
 * Generate points for a regular polygon
 */
export function regularPolygonPoints(
  sides: number,
  radius: number,
  startAngle: number = 0
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];
  const angleStep = (Math.PI * 2) / sides;
  
  for (let i = 0; i < sides; i++) {
    const angle = startAngle + i * angleStep;
    points.push({
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius
    });
  }
  
  return points;
}

/**
 * Apply noise distortion to a set of points
 */
export function distortPoints(
  points: { x: number; y: number }[],
  noiseAmount: number,
  noiseFn: () => number
): { x: number; y: number }[] {
  return points.map(p => ({
    x: p.x + (noiseFn() - 0.5) * 2 * noiseAmount,
    y: p.y + (noiseFn() - 0.5) * 2 * noiseAmount
  }));
}

/**
 * Create a THREE.Shape from 2D points
 */
export function pointsToShape(points: { x: number; y: number }[]): THREE.Shape {
  const shape = new THREE.Shape();
  
  if (points.length === 0) return shape;
  
  shape.moveTo(points[0].x, points[0].y);
  
  for (let i = 1; i < points.length; i++) {
    shape.lineTo(points[i].x, points[i].y);
  }
  
  shape.closePath();
  return shape;
}

/**
 * Compute bounding box of geometry
 */
export function computeBounds(geometry: THREE.BufferGeometry): THREE.Box3 {
  geometry.computeBoundingBox();
  return geometry.boundingBox!.clone();
}

/**
 * Count total vertices in a mesh or group
 */
export function countVertices(object: THREE.Object3D): number {
  let count = 0;
  
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      const geometry = child.geometry as THREE.BufferGeometry;
      const position = geometry.getAttribute('position');
      if (position) {
        count += position.count;
      }
    }
  });
  
  return count;
}

/**
 * Simple 2D Perlin-like noise (simplified version)
 */
export function simpleNoise2D(x: number, y: number, seed: number = 0): number {
  // Simple hash function for pseudo-random values
  const hash = (n: number) => {
    let h = (n + seed) * 1234567;
    h = h ^ (h >> 15);
    h = h * 2246822519;
    h = h ^ (h >> 13);
    h = h * 3266489917;
    h = h ^ (h >> 16);
    return (h & 0x7fffffff) / 0x7fffffff;
  };
  
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  
  // Smooth interpolation
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  
  // Corner values
  const v00 = hash(ix + iy * 57);
  const v10 = hash(ix + 1 + iy * 57);
  const v01 = hash(ix + (iy + 1) * 57);
  const v11 = hash(ix + 1 + (iy + 1) * 57);
  
  // Bilinear interpolation
  const v0 = lerp(v00, v10, ux);
  const v1 = lerp(v01, v11, ux);
  
  return lerp(v0, v1, uy);
}

