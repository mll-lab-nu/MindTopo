/**
 * fence-generator.js
 * Parametric fence generation: closed, with gaps, nested, network
 */

import * as THREE from 'three';

export class FenceGenerator {

  /**
   * Generate a polygon fence shape
   * @param {Object} options
   * @param {string} options.shape - 'convex' | 'concave' | 'irregular' | 'circle'
   * @param {number} options.segments - number of fence segments (4-20)
   * @param {number} options.radius - base radius
   * @param {Object} options.center - {x, z} center position
   * @param {number} options.irregularity - 0-1, how irregular the shape is
   * @returns {Array} Array of {x, z} vertices
   */
  static generatePolygon(options = {}) {
    const {
      shape = 'convex',
      segments = 8,
      radius = 5,
      center = { x: 0, z: 0 },
      irregularity = 0
    } = options;

    const vertices = [];
    const n = Math.max(3, segments);

    switch (shape) {
      case 'circle':
        for (let i = 0; i < n; i++) {
          const angle = (i / n) * Math.PI * 2;
          vertices.push({
            x: center.x + Math.cos(angle) * radius,
            z: center.z + Math.sin(angle) * radius
          });
        }
        break;

      case 'convex':
        for (let i = 0; i < n; i++) {
          const angle = (i / n) * Math.PI * 2;
          const r = radius * (1 - irregularity * 0.3 * Math.random());
          vertices.push({
            x: center.x + Math.cos(angle) * r,
            z: center.z + Math.sin(angle) * r
          });
        }
        break;

      case 'concave': {
        // L-shape or U-shape style concave polygon
        const concaveType = Math.random() > 0.5 ? 'L' : 'U';
        if (concaveType === 'L') {
          const r = radius;
          vertices.push(
            { x: center.x - r, z: center.z - r },
            { x: center.x + r, z: center.z - r },
            { x: center.x + r, z: center.z },
            { x: center.x, z: center.z },
            { x: center.x, z: center.z + r },
            { x: center.x - r, z: center.z + r }
          );
        } else {
          const r = radius;
          vertices.push(
            { x: center.x - r, z: center.z - r },
            { x: center.x + r, z: center.z - r },
            { x: center.x + r, z: center.z + r },
            { x: center.x + r * 0.3, z: center.z + r },
            { x: center.x + r * 0.3, z: center.z },
            { x: center.x - r * 0.3, z: center.z },
            { x: center.x - r * 0.3, z: center.z + r },
            { x: center.x - r, z: center.z + r }
          );
        }
        // Apply irregularity
        if (irregularity > 0) {
          for (const v of vertices) {
            v.x += (Math.random() - 0.5) * radius * irregularity * 0.3;
            v.z += (Math.random() - 0.5) * radius * irregularity * 0.3;
          }
        }
        break;
      }

      case 'irregular':
        for (let i = 0; i < n; i++) {
          const angle = (i / n) * Math.PI * 2 + (Math.random() - 0.5) * 0.3;
          const r = radius * (0.5 + Math.random() * 0.8);
          vertices.push({
            x: center.x + Math.cos(angle) * r,
            z: center.z + Math.sin(angle) * r
          });
        }
        break;

      default:
        return this.generatePolygon({ ...options, shape: 'convex' });
    }

    return vertices;
  }

  /**
   * Generate nested fence layers
   * @param {Object} options
   * @param {number} options.depth - nesting depth (1-4)
   * @param {number} options.outerRadius - radius of outermost fence
   * @param {Object} options.center - {x, z}
   * @param {string} options.shape - base shape
   * @returns {Array} Array of polygon arrays (outer to inner)
   */
  static generateNested(options = {}) {
    const {
      depth: requestedDepth = 2,
      outerRadius = 8,
      center = { x: 0, z: 0 },
      shape = 'convex',
      segments = 8
    } = options;

    // Innermost layer must hold at least one sheep with the 2.0m fence
    // buffer in placeSheep(), so apothem > 2.0 → radius > ~2.3 for a hex.
    // Use 4.0 to give roughly 1.5m of usable interior.
    const minInnerRadius = 4.0;
    // Cap depth so each layer band is ≥3.5m wide. The 2.0m fence buffer on
    // both sides of a band needs at least that to leave any usable strip;
    // tighter bands cause most sheep placements to fail.
    const feasibleDepth = Math.max(1, 1 + Math.floor((outerRadius - minInnerRadius) / 3.5));
    const depth = Math.min(requestedDepth, feasibleDepth);
    const layerClearance = depth > 1
      ? Math.max(3.0, (outerRadius - minInnerRadius) / (depth - 1))
      : 0;

    // Generate outermost layer using the requested shape
    const outerLayer = this.generatePolygon({
      shape, segments, radius: outerRadius, center, irregularity: 0
    });
    const layers = [outerLayer];

    // Generate inner layers as inward offsets of the previous layer
    for (let d = 1; d < depth; d++) {
      const prevLayer = layers[d - 1];

      // Strategy 1: polygon inward offset (follows outer shape naturally)
      let innerLayer = this._offsetPolygonInward(prevLayer, layerClearance);
      const area = this._polygonArea(innerLayer);
      const selfIntersects = this._hasSelfIntersection(innerLayer);

      if (area < 6.0 || selfIntersects) {
        // Strategy 2: convex fallback — find the best center inside prev layer
        // (area centroid can be near narrow channels; grid-sample for best inscribed circle)
        const best = this._bestInscribedCenter(prevLayer);
        const innerRadius = Math.max(minInnerRadius, best.radius - layerClearance);
        innerLayer = this.generatePolygon({
          shape: 'convex',
          segments: Math.max(6, segments),
          radius: innerRadius,
          center: best.center,
          irregularity: 0
        });
      }

      layers.push(innerLayer);
    }

    // Final validation: ensure clearance between all adjacent layers
    const minDist = Math.min(layerClearance * 0.6, 2.0);
    for (let d = 1; d < layers.length; d++) {
      this._enforceLayerClearance(layers[d], layers[d - 1], minDist);
    }

    return layers;
  }

  /** Helper: point-to-segment distance in XZ plane */
  static _pointToSegDist(p, a, b) {
    const dx = b.x - a.x, dz = b.z - a.z;
    const lenSq = dx * dx + dz * dz;
    if (lenSq < 0.0001) return Math.hypot(p.x - a.x, p.z - a.z);
    let t = ((p.x - a.x) * dx + (p.z - a.z) * dz) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const projX = a.x + t * dx, projZ = a.z + t * dz;
    return Math.hypot(p.x - projX, p.z - projZ);
  }

  /**
   * Compute polygon inward offset using edge normals and miter joints.
   * Each vertex is moved inward along the bisector of its two adjacent edge normals.
   */
  static _offsetPolygonInward(polygon, distance) {
    const n = polygon.length;
    if (n < 3) return polygon.map(v => ({ ...v }));

    // Signed area: positive → inward normal is (-dz, dx)
    let signedArea2 = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      signedArea2 += polygon[i].x * polygon[j].z - polygon[j].x * polygon[i].z;
    }
    const s = signedArea2 > 0 ? 1 : -1;

    const result = [];
    for (let i = 0; i < n; i++) {
      const prev = polygon[(i - 1 + n) % n];
      const curr = polygon[i];
      const next = polygon[(i + 1) % n];

      // Edge vectors
      const e1x = curr.x - prev.x, e1z = curr.z - prev.z;
      const e2x = next.x - curr.x, e2z = next.z - curr.z;
      const len1 = Math.hypot(e1x, e1z) || 0.001;
      const len2 = Math.hypot(e2x, e2z) || 0.001;

      // Inward unit normals for each edge
      const n1x = s * (-e1z / len1), n1z = s * (e1x / len1);
      const n2x = s * (-e2z / len2), n2z = s * (e2x / len2);

      // Bisector direction
      let bx = n1x + n2x, bz = n1z + n2z;
      const blen = Math.hypot(bx, bz);
      if (blen < 0.001) { bx = n1x; bz = n1z; }
      else { bx /= blen; bz /= blen; }

      // Miter length: distance / cos(half-angle between normals)
      const cosHalf = n1x * bx + n1z * bz;
      const miter = cosHalf > 0.15 ? distance / cosHalf : distance / 0.15;
      // Cap to avoid extreme spikes at sharp corners
      const capped = Math.min(miter, distance * 3);

      result.push({ x: curr.x + bx * capped, z: curr.z + bz * capped });
    }
    return result;
  }

  /** Absolute polygon area using shoelace formula */
  static _polygonArea(polygon) {
    let sum = 0;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      sum += polygon[i].x * polygon[j].z - polygon[j].x * polygon[i].z;
    }
    return Math.abs(sum) / 2;
  }

  /** Area centroid of polygon */
  static _polygonCentroid(polygon) {
    const n = polygon.length;
    let cx = 0, cz = 0, areaSum = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const cross = polygon[i].x * polygon[j].z - polygon[j].x * polygon[i].z;
      cx += (polygon[i].x + polygon[j].x) * cross;
      cz += (polygon[i].z + polygon[j].z) * cross;
      areaSum += cross;
    }
    const area6 = areaSum * 3;
    if (Math.abs(area6) < 0.001) {
      // Degenerate: return vertex average
      return {
        x: polygon.reduce((s, v) => s + v.x, 0) / n,
        z: polygon.reduce((s, v) => s + v.z, 0) / n
      };
    }
    return { x: cx / area6, z: cz / area6 };
  }

  /** Point-in-polygon test (ray casting) */
  static _pointInsidePolygon(point, polygon) {
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
   * Find the point inside the polygon with the largest inscribed radius.
   * Samples a grid of candidates and picks the best one.
   */
  static _bestInscribedCenter(polygon) {
    const xs = polygon.map(v => v.x), zs = polygon.map(v => v.z);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);

    let bestCenter = this._polygonCentroid(polygon);
    let bestRadius = this._pointInsidePolygon(bestCenter, polygon)
      ? this._maxInscribedRadius(polygon, bestCenter) : 0;

    const steps = 7;
    for (let i = 1; i < steps; i++) {
      for (let j = 1; j < steps; j++) {
        const candidate = {
          x: minX + (maxX - minX) * i / steps,
          z: minZ + (maxZ - minZ) * j / steps
        };
        if (!this._pointInsidePolygon(candidate, polygon)) continue;
        const r = this._maxInscribedRadius(polygon, candidate);
        if (r > bestRadius) {
          bestRadius = r;
          bestCenter = candidate;
        }
      }
    }
    return { center: bestCenter, radius: bestRadius };
  }

  /** Max inscribed radius: min distance from point to any polygon edge */
  static _maxInscribedRadius(polygon, point) {
    let minDist = Infinity;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const dist = this._pointToSegDist(point, polygon[i], polygon[(i + 1) % n]);
      if (dist < minDist) minDist = dist;
    }
    return minDist;
  }

  /** Check if polygon has self-intersections (non-adjacent edge crossings) */
  static _hasSelfIntersection(polygon) {
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const a1 = polygon[i], a2 = polygon[(i + 1) % n];
      for (let j = i + 2; j < n; j++) {
        if (i === 0 && j === n - 1) continue; // adjacent wrap-around
        const b1 = polygon[j], b2 = polygon[(j + 1) % n];
        if (this._edgesIntersect(a1, a2, b1, b2)) return true;
      }
    }
    return false;
  }

  /** Test if two line segments intersect (proper crossing only) */
  static _edgesIntersect(a1, a2, b1, b2) {
    const d1 = (b2.x - b1.x) * (a1.z - b1.z) - (b2.z - b1.z) * (a1.x - b1.x);
    const d2 = (b2.x - b1.x) * (a2.z - b1.z) - (b2.z - b1.z) * (a2.x - b1.x);
    const d3 = (a2.x - a1.x) * (b1.z - a1.z) - (a2.z - a1.z) * (b1.x - a1.x);
    const d4 = (a2.x - a1.x) * (b2.z - a1.z) - (a2.z - a1.z) * (b2.x - a1.x);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
           ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  }

  /**
   * Push inner polygon vertices to maintain clearance from outer polygon.
   * Multi-pass: also samples edge midpoints to catch segment-to-segment overlap.
   */
  static _enforceLayerClearance(innerLayer, outerLayer, minDist) {
    const maxPasses = 8;
    for (let pass = 0; pass < maxPasses; pass++) {
      let anyFixed = false;
      const centroid = this._polygonCentroid(innerLayer);
      const n = innerLayer.length;

      // Check vertices + edge midpoints
      const checkPoints = [];
      for (let i = 0; i < n; i++) {
        checkPoints.push({ idx: i, isMid: false });
        // Also sample midpoint of each edge
        checkPoints.push({ idx: i, isMid: true });
      }

      for (const cp of checkPoints) {
        let px, pz;
        if (cp.isMid) {
          const a = innerLayer[cp.idx], b = innerLayer[(cp.idx + 1) % n];
          px = (a.x + b.x) / 2; pz = (a.z + b.z) / 2;
        } else {
          px = innerLayer[cp.idx].x; pz = innerLayer[cp.idx].z;
        }

        // Find min distance to outer polygon
        let closestDist = Infinity;
        for (let si = 0; si < outerLayer.length; si++) {
          const dist = this._pointToSegDist(
            { x: px, z: pz }, outerLayer[si], outerLayer[(si + 1) % outerLayer.length]
          );
          if (dist < closestDist) closestDist = dist;
        }

        if (closestDist < minDist) {
          // Push nearby vertices toward centroid
          const shrink = (minDist - closestDist + 0.3);
          const indices = cp.isMid
            ? [cp.idx, (cp.idx + 1) % n]
            : [cp.idx];
          for (const vi of indices) {
            const v = innerLayer[vi];
            const dx = v.x - centroid.x, dz = v.z - centroid.z;
            const r = Math.hypot(dx, dz);
            if (r > 0.01) {
              const scale = Math.max(0.1, (r - shrink) / r);
              innerLayer[vi] = { x: centroid.x + dx * scale, z: centroid.z + dz * scale };
              anyFixed = true;
            }
          }
        }
      }
      if (!anyFixed) break;
    }
  }

  /**
   * Apply gaps to a fence polygon
   * @param {Array} vertices - polygon vertices
   * @param {number} numGaps - number of gaps
   * @param {number} gapSize - gap size as fraction of segment length (0-2)
   * @returns {Object} { segments: Array of {start, end, isGap}, gaps: Array of gap info }
   */
  static applyGaps(vertices, numGaps = 0, gapSize = 0.5) {
    const n = vertices.length;
    const segments = [];
    const gaps = [];

    if (numGaps === 0) {
      for (let i = 0; i < n; i++) {
        segments.push({
          start: vertices[i],
          end: vertices[(i + 1) % n],
          isGap: false,
          index: i
        });
      }
      return { segments, gaps };
    }

    // Choose gap positions evenly distributed
    const gapIndices = new Set();
    const step = Math.floor(n / numGaps);
    for (let g = 0; g < numGaps && g < n; g++) {
      gapIndices.add((g * step) % n);
    }

    for (let i = 0; i < n; i++) {
      const isGap = gapIndices.has(i);
      const start = vertices[i];
      const end = vertices[(i + 1) % n];

      if (isGap) {
        // Create gap by shortening the segment from both ends
        // Cap absolute gap width: sheep body is ~0.91 units wide, gap just barely fits one
        const dx = end.x - start.x;
        const dz = end.z - start.z;
        const len = Math.sqrt(dx * dx + dz * dz);
        const maxAbsGap = 1.0;
        const absGap = Math.min(len * Math.min(gapSize, 0.9), maxAbsGap);
        const gapFraction = absGap / len;
        const keepFraction = (1 - gapFraction) / 2;

        segments.push({
          start: start,
          end: { x: start.x + dx * keepFraction, z: start.z + dz * keepFraction },
          isGap: false,
          index: i
        });
        segments.push({
          start: { x: start.x + dx * keepFraction, z: start.z + dz * keepFraction },
          end: { x: end.x - dx * keepFraction, z: end.z - dz * keepFraction },
          isGap: true,
          index: i
        });
        segments.push({
          start: { x: end.x - dx * keepFraction, z: end.z - dz * keepFraction },
          end: end,
          isGap: false,
          index: i
        });
        gaps.push({
          segmentIndex: i,
          gapStart: { x: start.x + dx * keepFraction, z: start.z + dz * keepFraction },
          gapEnd: { x: end.x - dx * keepFraction, z: end.z - dz * keepFraction },
          width: len * gapFraction
        });
      } else {
        segments.push({ start, end, isGap: false, index: i });
      }
    }

    return { segments, gaps };
  }

  /**
   * Build 3D fence mesh from polygon vertices
   * @param {Array} vertices - polygon {x, z} vertices
   * @param {Object} options
   * @param {number} options.postHeight - height of fence posts
   * @param {number} options.postRadius - radius of fence posts
   * @param {number} options.railHeight - height of horizontal rails
   * @param {number} options.numRails - number of horizontal rails
   * @param {number} options.numGaps - number of gaps
   * @param {number} options.gapSize - gap size
   * @param {THREE.Color} options.postColor - color for posts
   * @param {THREE.Color} options.railColor - color for rails
   * @returns {Object} { group: THREE.Group, metadata: Object }
   */
  static buildFenceMesh(vertices, options = {}) {
    const {
      postHeight = 1.2,
      postRadius = 0.10,
      railHeight = 0.08,
      numRails = 3,
      numGaps = 0,
      gapSize = 0.5,
      postColor = new THREE.Color(0x8B6914),
      railColor = new THREE.Color(0xA0804D),
      gateIndices = [],
      gateColor = new THREE.Color(0x225588)
    } = options;

    const group = new THREE.Group();
    const { segments, gaps } = this.applyGaps(vertices, numGaps, gapSize);

    // Create fence posts at each vertex
    const postGeom = new THREE.CylinderGeometry(postRadius, postRadius * 1.2, postHeight, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: postColor, roughness: 0.8 });

    for (const v of vertices) {
      const post = new THREE.Mesh(postGeom, postMat);
      post.position.set(v.x, postHeight / 2, v.z);
      post.castShadow = true;
      post.receiveShadow = true;
      group.add(post);
    }

    // Create horizontal rails between posts (skip gaps)
    // Shorten rails by postRadius at each end to prevent corner clipping
    const railMat = new THREE.MeshStandardMaterial({ color: railColor, roughness: 0.7 });
    const boardColor = new THREE.Color(railColor).lerp(postColor, 0.3);
    const boardMat = new THREE.MeshStandardMaterial({ color: boardColor, roughness: 0.8 });

    const gateMat = new THREE.MeshStandardMaterial({ color: gateColor, roughness: 0.5, metalness: 0.3 });
    const gateBoardMat = new THREE.MeshStandardMaterial({ color: gateColor, roughness: 0.5, metalness: 0.2 });

    for (const seg of segments) {
      if (seg.isGap) continue;

      const dx = seg.end.x - seg.start.x;
      const dz = seg.end.z - seg.start.z;
      const len = Math.sqrt(dx * dx + dz * dz);
      if (len < 0.01) continue;

      // Shorten rail to avoid corner overlap
      const shrink = Math.min(postRadius, len * 0.15);
      const railLen = len - shrink * 2;
      if (railLen < 0.01) continue;

      const angle = Math.atan2(dx, dz);
      const isGate = gateIndices.includes(seg.index);
      const mat = isGate ? gateMat : railMat;

      const dirX = dx / len;
      const dirZ = dz / len;
      const midX = (seg.start.x + seg.end.x) / 2;
      const midZ = (seg.start.z + seg.end.z) / 2;

      for (let r = 0; r < numRails; r++) {
        const y = postHeight * (0.3 + 0.5 * r / (numRails - 1 || 1));
        const railGeom = new THREE.BoxGeometry(railHeight, railHeight, railLen);
        const rail = new THREE.Mesh(railGeom, mat);
        rail.position.set(midX, y, midZ);
        rail.rotation.y = angle;
        rail.castShadow = true;
        rail.receiveShadow = true;
        group.add(rail);
      }

      // Add vertical boards between posts for fence solidity
      const boardThick = 0.04;
      const boardDepth = 0.05;
      const boardHeight = postHeight * 0.75;
      const boardSpacing = 0.45;
      const numBoards = Math.max(0, Math.floor((railLen - boardThick) / boardSpacing));
      const bMat = isGate ? gateBoardMat : boardMat;

      for (let b = 0; b < numBoards; b++) {
        const t = (b + 0.5) / numBoards;
        const bx = seg.start.x + dirX * (shrink + t * railLen);
        const bz = seg.start.z + dirZ * (shrink + t * railLen);
        const boardGeom = new THREE.BoxGeometry(boardDepth, boardHeight, boardThick);
        const board = new THREE.Mesh(boardGeom, bMat);
        board.position.set(bx, boardHeight / 2, bz);
        board.rotation.y = angle;
        board.castShadow = true;
        board.receiveShadow = true;
        group.add(board);
      }
    }

    return {
      group,
      metadata: {
        vertices,
        segments,
        gaps,
        isClosed: numGaps === 0,
        gateIndices
      }
    };
  }

  /**
   * Generate a partitioned fence layout
   * @param {Object} options
   * @param {string} options.layout - 'grid' | 'radial' | 'scattered'
   * @param {number} options.numCols - columns (grid mode)
   * @param {number} options.numRows - rows (grid mode)
   * @param {number} options.numSectors - sectors (radial mode)
   * @param {number} options.numPens - number of pens (scattered mode)
   * @param {number} options.width - total width
   * @param {number} options.height - total height
   * @param {Object} options.center - {x, z}
   * @param {boolean} options.separated - if true, cells have gaps
   * @param {number} options.irregularity - 0-1
   * @returns {Object} { cells, wallSegments?, gridPoints?, separated }
   */
  static generatePartitioned(options = {}) {
    const {
      layout = 'grid',
      numCols = 2,
      numRows = 2,
      numSectors = 5,
      numPens = 4,
      numPolygonSides = 6,
      numPolygonDivisions = 3,
      width = 12,
      height = 10,
      center = { x: 0, z: 0 },
      separated = false,
      gapSize = 1.5,
      irregularity = 0.15,
      numGates = 0
    } = options;

    let result;
    if (layout === 'radial') {
      result = this._generateRadialPartition({ numSectors, radius: width / 2, center, irregularity });
    } else if (layout === 'polygon') {
      result = this._generatePolygonPartition({ numSides: numPolygonSides, numDivisions: numPolygonDivisions, radius: width / 2, center, irregularity });
    } else if (layout === 'hex_cross') {
      result = this._generateHexCrossPartition({ radius: width / 2, center, irregularity });
    } else if (layout === 'nested_polygon') {
      result = this._generateNestedPolygonPartition({ numSides: numPolygonSides, radius: width / 2, center, irregularity });
    } else if (layout === 'polygon_star') {
      result = this._generatePolygonStarPartition({ numSides: numPolygonSides, radius: width / 2, center, irregularity });
    } else if (layout === 'scattered' || separated) {
      result = this._generateScatteredPens({ numPens: numCols * numRows, width, height, center, irregularity });
    } else {
      result = this._generateGridPartition({ numCols, numRows, width, height, center, irregularity });
    }

    // Add gates to inner wall segments
    if (numGates > 0 && result.wallSegments) {
      const innerWalls = result.wallSegments.filter(s => !s.isOuter);
      if (innerWalls.length > 0) {
        const numActualGates = Math.min(numGates, innerWalls.length);
        const shuffled = [...innerWalls].sort(() => Math.random() - 0.5);
        const gateWalls = shuffled.slice(0, numActualGates);
        for (const wall of gateWalls) {
          wall.isGate = true;
        }
        result.numGates = numActualGates;
      } else {
        result.numGates = 0;
      }
    } else {
      result.numGates = 0;
    }

    return result;
  }

  /** Grid partition: rectangular cells sharing walls */
  static _generateGridPartition({ numCols, numRows, width, height, center, irregularity }) {
    const cells = [];
    const cellW = width / numCols;
    const cellH = height / numRows;

    const grid = [];
    for (let r = 0; r <= numRows; r++) {
      grid[r] = [];
      for (let c = 0; c <= numCols; c++) {
        let x = center.x - width / 2 + c * cellW;
        let z = center.z - height / 2 + r * cellH;
        if (irregularity > 0 && r > 0 && r < numRows && c > 0 && c < numCols) {
          x += (Math.random() - 0.5) * cellW * irregularity;
          z += (Math.random() - 0.5) * cellH * irregularity;
        }
        grid[r][c] = { x, z };
      }
    }

    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        cells.push([
          { x: grid[r][c].x, z: grid[r][c].z },
          { x: grid[r][c + 1].x, z: grid[r][c + 1].z },
          { x: grid[r + 1][c + 1].x, z: grid[r + 1][c + 1].z },
          { x: grid[r + 1][c].x, z: grid[r + 1][c].z }
        ]);
      }
    }

    const wallSegments = [];
    for (let r = 0; r <= numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        wallSegments.push({ start: grid[r][c], end: grid[r][c + 1], isOuter: r === 0 || r === numRows });
      }
    }
    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c <= numCols; c++) {
        wallSegments.push({ start: grid[r][c], end: grid[r + 1][c], isOuter: c === 0 || c === numCols });
      }
    }

    const gridPoints = [];
    for (let r = 0; r <= numRows; r++) {
      for (let c = 0; c <= numCols; c++) gridPoints.push(grid[r][c]);
    }

    return { cells, wallSegments, gridPoints, grid, separated: false };
  }

  /** Radial partition: pie-slice sectors from a polygon center */
  static _generateRadialPartition({ numSectors, radius, center, irregularity }) {
    const n = Math.max(3, numSectors);
    const outerVerts = [];
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * Math.PI * 2;
      const r = radius * (1 - irregularity * 0.2 * Math.random());
      outerVerts.push({
        x: center.x + Math.cos(angle) * r,
        z: center.z + Math.sin(angle) * r
      });
    }

    // Ring layouts create many narrow inner triangles, so only enable them
    // when the outer radius is large enough to keep those cells usable.
    const hasRing = n >= 5 && radius >= 8.5 && Math.random() > 0.65;
    const cells = [];
    const wallSegments = [];
    const gridPoints = [{ x: center.x, z: center.z }];

    if (hasRing) {
      // Two rings: inner triangular sectors + outer trapezoidal sectors
      const innerRadius = radius * (0.35 + Math.random() * 0.15);
      const innerVerts = [];
      for (let i = 0; i < n; i++) {
        const angle = (i / n) * Math.PI * 2;
        const r = innerRadius * (1 - irregularity * 0.15 * Math.random());
        innerVerts.push({
          x: center.x + Math.cos(angle) * r,
          z: center.z + Math.sin(angle) * r
        });
      }

      for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        // Inner cell: triangle from center
        cells.push([
          { x: center.x, z: center.z },
          { x: innerVerts[i].x, z: innerVerts[i].z },
          { x: innerVerts[next].x, z: innerVerts[next].z }
        ]);
        // Outer cell: trapezoid between rings
        cells.push([
          { x: innerVerts[i].x, z: innerVerts[i].z },
          { x: outerVerts[i].x, z: outerVerts[i].z },
          { x: outerVerts[next].x, z: outerVerts[next].z },
          { x: innerVerts[next].x, z: innerVerts[next].z }
        ]);
      }

      // Outer boundary walls
      for (let i = 0; i < n; i++) {
        wallSegments.push({ start: outerVerts[i], end: outerVerts[(i + 1) % n], isOuter: true });
      }
      // Inner ring walls
      for (let i = 0; i < n; i++) {
        wallSegments.push({ start: innerVerts[i], end: innerVerts[(i + 1) % n], isOuter: false });
      }
      // Radial walls: center to inner, inner to outer
      for (let i = 0; i < n; i++) {
        wallSegments.push({ start: { x: center.x, z: center.z }, end: innerVerts[i], isOuter: false });
        wallSegments.push({ start: innerVerts[i], end: outerVerts[i], isOuter: false });
      }

      gridPoints.push(...innerVerts, ...outerVerts);
    } else {
      // Simple pie slices: triangles from center
      for (let i = 0; i < n; i++) {
        const next = (i + 1) % n;
        cells.push([
          { x: center.x, z: center.z },
          { x: outerVerts[i].x, z: outerVerts[i].z },
          { x: outerVerts[next].x, z: outerVerts[next].z }
        ]);
      }

      // Outer boundary walls
      for (let i = 0; i < n; i++) {
        wallSegments.push({ start: outerVerts[i], end: outerVerts[(i + 1) % n], isOuter: true });
      }
      // Radial walls: center to each vertex
      for (let i = 0; i < n; i++) {
        wallSegments.push({ start: { x: center.x, z: center.z }, end: outerVerts[i], isOuter: false });
      }

      gridPoints.push(...outerVerts);
    }

    return { cells, wallSegments, gridPoints, separated: false };
  }

  /**
   * Hexagon with two long diagonals through the center, producing 4 cells:
   * two triangles ({ctr,v0,v1}, {ctr,v3,v4}) and two quads ({ctr,v1,v2,v3}, {ctr,v4,v5,v0}).
   */
  static _generateHexCrossPartition({ radius, center, irregularity }) {
    const n = 6;
    const rotation = Math.random() * Math.PI * 2;
    const verts = [];
    for (let i = 0; i < n; i++) {
      const angle = rotation + (i / n) * Math.PI * 2 - Math.PI / 2;
      const r = radius * (1 - irregularity * 0.15 * Math.random());
      verts.push({
        x: center.x + Math.cos(angle) * r,
        z: center.z + Math.sin(angle) * r
      });
    }
    const ctr = { x: center.x, z: center.z };

    const cells = [
      [ctr, verts[0], verts[1]],
      [ctr, verts[1], verts[2], verts[3]],
      [ctr, verts[3], verts[4]],
      [ctr, verts[4], verts[5], verts[0]]
    ];

    const wallSegments = [];
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: verts[i], end: verts[(i + 1) % n], isOuter: true });
    }
    // Two long diagonals, each split at the center post into two spokes
    for (const i of [0, 1, 3, 4]) {
      wallSegments.push({ start: ctr, end: verts[i], isOuter: false });
    }

    const gridPoints = [ctr, ...verts];
    return { cells, wallSegments, gridPoints, separated: false };
  }

  /**
   * Outer polygon with a concentric inner polygon, joined by corresponding-vertex spokes.
   * Produces numSides + 1 cells: one inner polygon plus numSides trapezoidal rings.
   * For numSides = 6 → 7 cells (hex inside hex with 6 spokes).
   */
  static _generateNestedPolygonPartition({ numSides, radius, center, irregularity }) {
    const n = Math.max(5, Math.min(8, numSides || 6));
    const rotation = Math.random() * Math.PI * 2;
    // Inner polygon must clear the per-cell fence buffer (~2 units) AND have
    // enough usable area for at least one sheep; 0.42 was too tight, leaving
    // the inner cell empty across most renders. 0.55..0.70 keeps outer
    // trapezoids viable while giving the inner cell real placement room.
    const innerScale = 0.55 + Math.random() * 0.15;
    const outerVerts = [];
    const innerVerts = [];
    for (let i = 0; i < n; i++) {
      const angle = rotation + (i / n) * Math.PI * 2 - Math.PI / 2;
      const rOuter = radius * (1 - irregularity * 0.12 * Math.random());
      const rInner = radius * innerScale * (1 - irregularity * 0.12 * Math.random());
      outerVerts.push({
        x: center.x + Math.cos(angle) * rOuter,
        z: center.z + Math.sin(angle) * rOuter
      });
      innerVerts.push({
        x: center.x + Math.cos(angle) * rInner,
        z: center.z + Math.sin(angle) * rInner
      });
    }

    const cells = [innerVerts.map(v => ({ x: v.x, z: v.z }))];
    for (let i = 0; i < n; i++) {
      const next = (i + 1) % n;
      cells.push([
        { x: outerVerts[i].x, z: outerVerts[i].z },
        { x: outerVerts[next].x, z: outerVerts[next].z },
        { x: innerVerts[next].x, z: innerVerts[next].z },
        { x: innerVerts[i].x, z: innerVerts[i].z }
      ]);
    }

    const wallSegments = [];
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: outerVerts[i], end: outerVerts[(i + 1) % n], isOuter: true });
    }
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: innerVerts[i], end: innerVerts[(i + 1) % n], isOuter: false });
    }
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: innerVerts[i], end: outerVerts[i], isOuter: false });
    }

    const gridPoints = [...outerVerts, ...innerVerts];
    return { cells, wallSegments, gridPoints, separated: false };
  }

  /**
   * Outer polygon with radial spokes from the center to every vertex, producing N triangular cells.
   * Differs from radial mode by always using a polygon outer boundary and no ring.
   */
  static _generatePolygonStarPartition({ numSides, radius, center, irregularity }) {
    const n = Math.max(4, Math.min(8, numSides || 6));
    const rotation = Math.random() * Math.PI * 2;
    const verts = [];
    for (let i = 0; i < n; i++) {
      const angle = rotation + (i / n) * Math.PI * 2 - Math.PI / 2;
      const r = radius * (1 - irregularity * 0.15 * Math.random());
      verts.push({
        x: center.x + Math.cos(angle) * r,
        z: center.z + Math.sin(angle) * r
      });
    }
    const ctr = { x: center.x, z: center.z };

    const cells = [];
    for (let i = 0; i < n; i++) {
      cells.push([ctr, verts[i], verts[(i + 1) % n]]);
    }

    const wallSegments = [];
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: verts[i], end: verts[(i + 1) % n], isOuter: true });
    }
    for (let i = 0; i < n; i++) {
      wallSegments.push({ start: ctr, end: verts[i], isOuter: false });
    }

    const gridPoints = [ctr, ...verts];
    return { cells, wallSegments, gridPoints, separated: false };
  }

  /** Polygon partition: polygon outer boundary with internal divisions via cut lines */
  static _generatePolygonPartition({ numSides, numDivisions, radius, center, irregularity }) {
    const n = Math.max(3, numSides);

    // Generate polygon outer boundary
    const outerVerts = [];
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
      const r = radius * (1 - irregularity * 0.2 * Math.random());
      outerVerts.push({
        x: center.x + Math.cos(angle) * r,
        z: center.z + Math.sin(angle) * r
      });
    }

    // Generate random cut lines through the polygon to create cells
    // Each cut is a line from one edge to another edge of the polygon
    const cuts = [];
    const numCuts = Math.max(1, numDivisions - 1);

    for (let c = 0; c < numCuts; c++) {
      // Pick two different edges to cut between
      let edgeA = Math.floor(Math.random() * n);
      let edgeB = (edgeA + Math.floor(1 + Math.random() * (n - 2))) % n;

      // Pick a random point along each edge
      const tA = 0.2 + Math.random() * 0.6;
      const tB = 0.2 + Math.random() * 0.6;

      const ptA = {
        x: outerVerts[edgeA].x + tA * (outerVerts[(edgeA + 1) % n].x - outerVerts[edgeA].x),
        z: outerVerts[edgeA].z + tA * (outerVerts[(edgeA + 1) % n].z - outerVerts[edgeA].z)
      };
      const ptB = {
        x: outerVerts[edgeB].x + tB * (outerVerts[(edgeB + 1) % n].x - outerVerts[edgeB].x),
        z: outerVerts[edgeB].z + tB * (outerVerts[(edgeB + 1) % n].z - outerVerts[edgeB].z)
      };

      cuts.push({ start: ptA, end: ptB, edgeA, edgeB });
    }

    // Build cells by splitting the polygon with cuts
    // Use a simpler approach: place cut points on the boundary, then create cells
    // by walking around the polygon boundary between cut endpoints
    const boundaryPoints = []; // { point, edgeIndex, t }
    for (let i = 0; i < n; i++) {
      boundaryPoints.push({ point: outerVerts[i], edgeIndex: i, t: 0, isVertex: true });
    }
    for (const cut of cuts) {
      const tA = 0.2 + Math.random() * 0.6;
      const tB = 0.2 + Math.random() * 0.6;
      boundaryPoints.push({ point: cut.start, edgeIndex: cut.edgeA, t: tA, isVertex: false });
      boundaryPoints.push({ point: cut.end, edgeIndex: cut.edgeB, t: tB, isVertex: false });
    }

    // Sort boundary points by edge index then by t
    boundaryPoints.sort((a, b) => a.edgeIndex - b.edgeIndex || a.t - b.t);

    // Build cells: each cut creates two cells by splitting
    // For simplicity, use a fan-based approach from polygon centroid with cut intersection points
    const cells = [];
    const wallSegments = [];
    const gridPoints = [];

    // Collect all cut endpoints on boundary in order
    const cutPoints = [];
    for (const cut of cuts) {
      cutPoints.push(cut.start, cut.end);
    }

    // Add all unique points to gridPoints
    const allPoints = [...outerVerts, ...cutPoints];
    for (const p of allPoints) {
      if (!gridPoints.some(g => Math.abs(g.x - p.x) < 0.01 && Math.abs(g.z - p.z) < 0.01)) {
        gridPoints.push(p);
      }
    }

    // Build ordered boundary with cut points inserted
    const orderedBoundary = [];
    for (let i = 0; i < n; i++) {
      orderedBoundary.push(outerVerts[i]);
      // Find cut points on this edge, sorted by parameter
      const edgeCutPts = [];
      for (const cut of cuts) {
        if (cut.edgeA === i) {
          const t = this._paramOnEdge(cut.start, outerVerts[i], outerVerts[(i + 1) % n]);
          edgeCutPts.push({ point: cut.start, t });
        }
        if (cut.edgeB === i) {
          const t = this._paramOnEdge(cut.end, outerVerts[i], outerVerts[(i + 1) % n]);
          edgeCutPts.push({ point: cut.end, t });
        }
      }
      edgeCutPts.sort((a, b) => a.t - b.t);
      for (const cp of edgeCutPts) {
        orderedBoundary.push(cp.point);
      }
    }

    // Now split: for each cut, trace cells on either side
    // Simple approach: use cut lines to divide the polygon
    // Walk the boundary between consecutive cut endpoints, adding the cut line to close each cell
    if (cuts.length === 0) {
      // No cuts: single cell = entire polygon
      cells.push([...outerVerts]);
    } else {
      // Find all cut endpoint indices in the ordered boundary
      const cutEndpointIndices = [];
      for (const cut of cuts) {
        const startIdx = orderedBoundary.findIndex(p => Math.abs(p.x - cut.start.x) < 0.01 && Math.abs(p.z - cut.start.z) < 0.01);
        const endIdx = orderedBoundary.findIndex(p => Math.abs(p.x - cut.end.x) < 0.01 && Math.abs(p.z - cut.end.z) < 0.01);
        cutEndpointIndices.push({ startIdx, endIdx, cut });
      }

      // For a single cut: creates 2 cells
      // For multiple cuts: more complex, but we handle sequentially
      // Start with the full polygon, then split by each cut
      let currentPolygons = [[...orderedBoundary]];

      for (const cut of cuts) {
        const newPolygons = [];
        for (const poly of currentPolygons) {
          const split = this._splitPolygonByCut(poly, cut.start, cut.end);
          if (split) {
            newPolygons.push(split.cellA, split.cellB);
          } else {
            newPolygons.push(poly);
          }
        }
        currentPolygons = newPolygons;
      }

      for (const poly of currentPolygons) {
        if (poly.length >= 3) {
          cells.push(poly);
        }
      }
    }

    // Outer boundary walls
    for (let i = 0; i < orderedBoundary.length; i++) {
      const next = (i + 1) % orderedBoundary.length;
      wallSegments.push({
        start: orderedBoundary[i],
        end: orderedBoundary[next],
        isOuter: true
      });
    }

    // Inner cut walls
    for (const cut of cuts) {
      wallSegments.push({
        start: cut.start,
        end: cut.end,
        isOuter: false
      });
    }

    return { cells, wallSegments, gridPoints, separated: false };
  }

  /** Helper: compute parameter t of point on edge (v0 -> v1) */
  static _paramOnEdge(point, v0, v1) {
    const dx = v1.x - v0.x;
    const dz = v1.z - v0.z;
    if (Math.abs(dx) > Math.abs(dz)) {
      return (point.x - v0.x) / dx;
    }
    return (point.z - v0.z) / dz;
  }

  /** Helper: split a polygon into two cells by a cut line */
  static _splitPolygonByCut(polygon, cutStart, cutEnd) {
    const eps = 0.05;
    // Find the two points in the polygon closest to cutStart and cutEnd
    let idxA = -1, idxB = -1;
    let bestDistA = Infinity, bestDistB = Infinity;

    for (let i = 0; i < polygon.length; i++) {
      const dA = Math.hypot(polygon[i].x - cutStart.x, polygon[i].z - cutStart.z);
      const dB = Math.hypot(polygon[i].x - cutEnd.x, polygon[i].z - cutEnd.z);
      if (dA < bestDistA) { bestDistA = dA; idxA = i; }
      if (dB < bestDistB) { bestDistB = dB; idxB = i; }
    }

    if (idxA === idxB || bestDistA > eps || bestDistB > eps) return null;

    // Ensure idxA < idxB
    if (idxA > idxB) { [idxA, idxB] = [idxB, idxA]; }

    const cellA = polygon.slice(idxA, idxB + 1);
    const cellB = [...polygon.slice(idxB), ...polygon.slice(0, idxA + 1)];

    if (cellA.length < 3 || cellB.length < 3) return null;

    return { cellA, cellB };
  }

  /** Scattered pens: independent polygon enclosures at random positions */
  static _generateScatteredPens({ numPens, width, height, center, irregularity }) {
    const cells = [];
    const placed = []; // { x, z, r, verts }
    const sideOptions = [5, 6, 7]; // higher-sided pens keep a larger usable interior

    for (let i = 0; i < numPens; i++) {
      const sides = sideOptions[Math.floor(Math.random() * sideOptions.length)];
      const availableSpan = Math.min(width, height) / Math.max(1, Math.sqrt(numPens));
      const targetRadius = Math.max(3.2, Math.min(5.0, availableSpan / 2.2));
      const penRadius = targetRadius * (0.92 + Math.random() * 0.16);
      // Account for irregularity expansion + fence board thickness + safety margin
      const maxRadius = penRadius * (1 + 0.3 * irregularity) + 0.2;

      // Find non-overlapping position using inflated radii
      let pos = null;
      for (let attempt = 0; attempt < 200; attempt++) {
        const candidate = {
          x: center.x + (Math.random() - 0.5) * (width - maxRadius * 2),
          z: center.z + (Math.random() - 0.5) * (height - maxRadius * 2)
        };
        let ok = true;
        for (const p of placed) {
          const dx = candidate.x - p.x, dz = candidate.z - p.z;
          if (Math.sqrt(dx * dx + dz * dz) < maxRadius + p.maxR + 0.8) { ok = false; break; }
        }
        if (ok) { pos = candidate; break; }
      }
      if (!pos) {
        // Fallback: place in a row with generous spacing
        pos = {
          x: center.x + (i - numPens / 2) * (maxRadius * 2.5),
          z: center.z
        };
      }
      placed.push({ x: pos.x, z: pos.z, r: penRadius, maxR: maxRadius });

      // Generate polygon pen
      const verts = [];
      const rotation = Math.random() * Math.PI * 2;
      for (let v = 0; v < sides; v++) {
        const angle = rotation + (v / sides) * Math.PI * 2;
        const r = penRadius * (0.85 + Math.random() * 0.3 * irregularity);
        verts.push({
          x: pos.x + Math.cos(angle) * r,
          z: pos.z + Math.sin(angle) * r
        });
      }

      // Verify no vertex-to-segment overlap with previously placed pens
      let clipping = false;
      for (const prev of cells) {
        for (const v of verts) {
          for (let si = 0; si < prev.length; si++) {
            const dist = this._pointToSegDist(v, prev[si], prev[(si + 1) % prev.length]);
            if (dist < 0.6) { clipping = true; break; }
          }
          if (clipping) break;
        }
        if (!clipping) {
          // Also check previous vertices against new pen edges
          for (const pv of prev) {
            for (let si = 0; si < verts.length; si++) {
              const dist = this._pointToSegDist(pv, verts[si], verts[(si + 1) % verts.length]);
              if (dist < 0.6) { clipping = true; break; }
            }
            if (clipping) break;
          }
        }
        if (clipping) break;
      }

      if (clipping) {
        // Shrink pen inward to avoid overlap
        const shrinkFactor = 0.75;
        for (const v of verts) {
          v.x = pos.x + (v.x - pos.x) * shrinkFactor;
          v.z = pos.z + (v.z - pos.z) * shrinkFactor;
        }
      }

      cells.push(verts);
    }

    return { cells, separated: true };
  }

  /**
   * Build 3D fence mesh from partition data
   * @param {Object} partitionData - from generatePartitioned()
   * @param {Object} options - post/rail options
   * @returns {Object} { group: THREE.Group }
   */
  static buildPartitionedMesh(partitionData, options = {}) {
    const {
      postHeight = 1.2,
      postRadius = 0.10,
      railHeight = 0.08,
      numRails = 3,
      postColor = new THREE.Color(0x8B6914),
      outerRailColor = new THREE.Color(0xA0804D),
      innerRailColor = new THREE.Color(0x7a6345),
      sheepPositions = []
    } = options;

    const group = new THREE.Group();

    if (partitionData.separated) {
      // Build each cell as an independent fence
      for (let i = 0; i < partitionData.cells.length; i++) {
        const hue = 0.06 + i * 0.02;
        const result = this.buildFenceMesh(partitionData.cells[i], {
          postHeight, postRadius, railHeight, numRails,
          postColor: new THREE.Color().setHSL(hue, 0.6, 0.35),
          railColor: new THREE.Color().setHSL(hue, 0.5, 0.4)
        });
        group.add(result.group);
      }
    } else {
      // Connected grid: posts at grid points, rails along wall segments
      const { wallSegments, gridPoints } = partitionData;

      const postGeom = new THREE.CylinderGeometry(postRadius, postRadius * 1.2, postHeight, 8);
      const postMat = new THREE.MeshStandardMaterial({ color: postColor, roughness: 0.8 });

      for (const pt of gridPoints) {
        const post = new THREE.Mesh(postGeom, postMat);
        post.position.set(pt.x, postHeight / 2, pt.z);
        post.castShadow = true;
        post.receiveShadow = true;
        group.add(post);
      }

      const outerMat = new THREE.MeshStandardMaterial({ color: outerRailColor, roughness: 0.7 });
      const innerMat = new THREE.MeshStandardMaterial({ color: innerRailColor, roughness: 0.7 });
      const gateColor = new THREE.Color(0x225588);
      const gateMat = new THREE.MeshStandardMaterial({ color: gateColor, roughness: 0.5, metalness: 0.3 });
      const outerBoardColor = new THREE.Color(outerRailColor).lerp(postColor, 0.3);
      const innerBoardColor = new THREE.Color(innerRailColor).lerp(postColor, 0.3);
      const outerBoardMat = new THREE.MeshStandardMaterial({ color: outerBoardColor, roughness: 0.8 });
      const innerBoardMat = new THREE.MeshStandardMaterial({ color: innerBoardColor, roughness: 0.8 });
      const gateBoardMat = new THREE.MeshStandardMaterial({ color: gateColor, roughness: 0.5, metalness: 0.2 });

      for (const seg of wallSegments) {
        const dx = seg.end.x - seg.start.x;
        const dz = seg.end.z - seg.start.z;
        const len = Math.sqrt(dx * dx + dz * dz);
        if (len < 0.01) continue;

        // Shorten rail to avoid corner overlap
        const shrink = Math.min(postRadius, len * 0.15);
        const railLen = len - shrink * 2;
        if (railLen < 0.01) continue;

        const angle = Math.atan2(dx, dz);
        const midX = (seg.start.x + seg.end.x) / 2;
        const midZ = (seg.start.z + seg.end.z) / 2;
        const mat = seg.isGate ? gateMat : (seg.isOuter ? outerMat : innerMat);

        for (let r = 0; r < numRails; r++) {
          const y = postHeight * (0.3 + 0.5 * r / (numRails - 1 || 1));
          const railGeom = new THREE.BoxGeometry(railHeight, railHeight, railLen);
          const rail = new THREE.Mesh(railGeom, mat);
          rail.position.set(midX, y, midZ);
          rail.rotation.y = angle;
          rail.castShadow = true;
          rail.receiveShadow = true;
          group.add(rail);
        }

        const dirX = dx / len;
        const dirZ = dz / len;
        const boardThick = 0.04;
        const boardDepth = 0.05;
        const boardHeight = postHeight * 0.75;
        const boardSpacing = 0.45;
        const numBoards = Math.max(0, Math.floor((railLen - boardThick) / boardSpacing));
        const bMat = seg.isGate ? gateBoardMat : (seg.isOuter ? outerBoardMat : innerBoardMat);

        for (let b = 0; b < numBoards; b++) {
          const t = (b + 0.5) / numBoards;
          const bx = seg.start.x + dirX * (shrink + t * railLen);
          const bz = seg.start.z + dirZ * (shrink + t * railLen);
          const boardGeom = new THREE.BoxGeometry(boardDepth, boardHeight, boardThick);
          const board = new THREE.Mesh(boardGeom, bMat);
          board.position.set(bx, boardHeight / 2, bz);
          board.rotation.y = angle;
          board.castShadow = true;
          board.receiveShadow = true;
          group.add(board);
        }

      }
    }

    // Add floating cell labels after sheep placement so markers stay away from sheep.
    for (let i = 0; i < partitionData.cells.length; i++) {
      const cell = partitionData.cells[i];
      const anchor = this._labelAnchorForCell(cell, sheepPositions);

      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(50, 50, 80, 0.7)';
      ctx.beginPath();
      ctx.arc(64, 64, 52, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 56px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String.fromCharCode(65 + i), 64, 66); // A, B, C, D...

      const texture = new THREE.CanvasTexture(canvas);
      const labelMat = new THREE.MeshStandardMaterial({
        map: texture,
        transparent: true,
        opacity: 0.85,
        depthTest: true,
        side: THREE.DoubleSide,
        roughness: 0.55,
        metalness: 0
      });
      const label = new THREE.Mesh(new THREE.CircleGeometry(0.48, 48), labelMat);
      label.rotation.x = -Math.PI / 2;
      label.position.set(anchor.x, 1.55, anchor.z);
      label.userData = { type: 'cellLabel', cellIndex: i };
      group.add(label);

      const stemMat = new THREE.MeshStandardMaterial({ color: 0x777b91, roughness: 0.7 });
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 1.35, 8), stemMat);
      stem.position.set(anchor.x, 0.7, anchor.z);
      stem.userData = { type: 'cellLabel', cellIndex: i };
      group.add(stem);
    }

    return { group };
  }

  static _labelAnchorForCell(cell, sheepPositions = []) {
    const centroid = this._polygonCentroid(cell);
    const xs = cell.map(v => v.x);
    const zs = cell.map(v => v.z);
    const candidates = [centroid];

    for (let ix = 1; ix <= 4; ix++) {
      for (let iz = 1; iz <= 4; iz++) {
        const candidate = {
          x: Math.min(...xs) + (Math.max(...xs) - Math.min(...xs)) * ix / 5,
          z: Math.min(...zs) + (Math.max(...zs) - Math.min(...zs)) * iz / 5
        };
        if (this._pointInsidePolygon(candidate, cell)) candidates.push(candidate);
      }
    }

    let best = centroid;
    let bestScore = -Infinity;
    for (const candidate of candidates) {
      const edgeDist = this._minDistToPolygonEdge(candidate, cell);
      const sheepDist = sheepPositions.reduce(
        (minDist, sheep) => Math.min(minDist, Math.hypot(candidate.x - sheep.x, candidate.z - sheep.z)),
        Infinity
      );
      const score = Math.min(edgeDist, sheepDist - 1.4);
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    return best;
  }

  static _minDistToPolygonEdge(pos, polygon) {
    let minDist = Infinity;
    for (let i = 0; i < polygon.length; i++) {
      minDist = Math.min(minDist, this._pointToSegDist(pos, polygon[i], polygon[(i + 1) % polygon.length]));
    }
    return minDist;
  }
}
