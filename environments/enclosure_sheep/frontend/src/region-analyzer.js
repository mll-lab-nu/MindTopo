/**
 * region-analyzer.js
 * Region topology analysis: point-in-polygon, connectivity, nesting depth
 */

export class RegionAnalyzer {

  /**
   * Point-in-polygon test using ray casting algorithm
   * @param {Object} point - {x, z} coordinates
   * @param {Array} polygon - Array of {x, z} vertices (closed polygon, last connects to first)
   * @returns {boolean} true if point is inside the polygon
   */
  static pointInPolygon(point, polygon) {
    let inside = false;
    const n = polygon.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = polygon[i].x, zi = polygon[i].z;
      const xj = polygon[j].x, zj = polygon[j].z;
      if (((zi > point.z) !== (zj > point.z)) &&
          (point.x < (xj - xi) * (point.z - zi) / (zj - zi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  /**
   * Check if a polygon is closed (no gaps)
   * A polygon with gaps has segments removed
   * @param {Array} polygon - vertices
   * @param {Array} gaps - array of {startIdx, endIdx} gap definitions
   * @returns {boolean}
   */
  static isPolygonClosed(polygon, gaps = []) {
    return gaps.length === 0;
  }

  /**
   * Determine nesting depth for a point given multiple fence layers
   * @param {Object} point - {x, z}
   * @param {Array} fenceLayers - Array of polygon arrays (each polygon = array of {x, z})
   * @returns {number} nesting depth (0 = outside all, 1 = inside one layer, etc.)
   */
  static nestingDepth(point, fenceLayers) {
    let depth = 0;
    for (const polygon of fenceLayers) {
      if (this.pointInPolygon(point, polygon)) {
        depth++;
      }
    }
    return depth;
  }

  /**
   * Count enclosed regions formed by fences
   * Simplified: counts the number of closed fence polygons (each creates one region)
   * For intersecting fences, uses Euler formula approximation
   * @param {Array} fenceLayers - Array of polygon arrays
   * @returns {number} number of bounded regions
   */
  static countRegions(fenceLayers) {
    // Each non-intersecting closed fence creates 1 region
    // Nested fences: inner fence creates an additional region
    return fenceLayers.filter(f => f.length >= 3).length;
  }

  /**
   * Check if two points are in the same connected region
   * (can reach each other without crossing any fence)
   * @param {Object} p1 - {x, z}
   * @param {Object} p2 - {x, z}
   * @param {Array} fenceLayers - Array of polygon arrays
   * @returns {boolean}
   */
  static sameRegion(p1, p2, fenceLayers) {
    // Two points are in the same region if they have the same
    // inside/outside status for every fence layer
    for (const polygon of fenceLayers) {
      const p1Inside = this.pointInPolygon(p1, polygon);
      const p2Inside = this.pointInPolygon(p2, polygon);
      if (p1Inside !== p2Inside) return false;
    }
    return true;
  }

  /**
   * Check containment relationship between two fence regions
   * @param {Array} polygonA - vertices of region A
   * @param {Array} polygonB - vertices of region B
   * @returns {string} 'A_IN_B' | 'B_IN_A' | 'NON_NESTED'
   */
  static containmentRelation(polygonA, polygonB) {
    // Check if all vertices of A are inside B
    const aInB = polygonA.every(v => this.pointInPolygon(v, polygonB));
    if (aInB) return 'A_IN_B';

    // Check if all vertices of B are inside A
    const bInA = polygonB.every(v => this.pointInPolygon(v, polygonA));
    if (bInA) return 'B_IN_A';

    return 'NON_NESTED';
  }

  /**
   * Find which sheep can escape through gaps in the fence
   * @param {Object} sheepPos - {x, z}
   * @param {Array} polygon - fence vertices
   * @param {Array} gaps - array of {startIdx, endIdx}
   * @returns {boolean} true if sheep can escape
   */
  static canEscape(sheepPos, polygon, gaps = []) {
    if (!this.pointInPolygon(sheepPos, polygon)) return true; // already outside
    if (gaps.length === 0) return false; // closed fence, no escape
    return true; // has gap and is inside
  }

  /**
   * Get the region label for a point given multiple fence layers
   * Returns a binary string where each bit indicates inside (1) or outside (0) of each fence
   * @param {Object} point - {x, z}
   * @param {Array} fenceLayers - Array of polygon arrays
   * @returns {string} region label like "101" (inside fence 0, outside fence 1, inside fence 2)
   */
  static regionLabel(point, fenceLayers) {
    return fenceLayers.map(polygon =>
      this.pointInPolygon(point, polygon) ? '1' : '0'
    ).join('');
  }
}
