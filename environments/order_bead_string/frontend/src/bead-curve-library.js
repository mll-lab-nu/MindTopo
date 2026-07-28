/**
 * Bead Curve Library
 *
 * Parametric curve definitions for bead string paths,
 * plus arc-length equidistant sampling utilities.
 */

import * as THREE from 'three';

// ============= Seeded RNG =============
function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRng(seedStr) {
  const seedFn = xmur3(seedStr || 'seed');
  return mulberry32(seedFn());
}

// ============= Curve Classes =============

/** Straight line with optional slight sag */
export class StraightCurve extends THREE.Curve {
  constructor(length = 6, sagAmount = 0) {
    super();
    this.length = length;
    this.sagAmount = sagAmount;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const x = (t - 0.5) * this.length;
    const y = -this.sagAmount * Math.sin(t * Math.PI);
    return target.set(x, y, 0);
  }
}

/** Circular arc in the XY plane */
export class ArcCurve extends THREE.Curve {
  constructor(radius = 3, arcAngle = Math.PI * 0.6) {
    super();
    this.radius = radius;
    this.arcAngle = arcAngle;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const startAngle = Math.PI / 2 + this.arcAngle / 2;
    const angle = startAngle - t * this.arcAngle;
    const x = this.radius * Math.cos(angle);
    const y = this.radius * Math.sin(angle) - this.radius * Math.cos(this.arcAngle / 2);
    return target.set(x, y, 0);
  }
}

/** S-shaped curve in 3D */
export class SCurve extends THREE.Curve {
  constructor(amplitude = 1.5, length = 6, zAmplitude = 0) {
    super();
    this.amplitude = amplitude;
    this.length = length;
    this.zAmplitude = zAmplitude;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const x = (t - 0.5) * this.length;
    const y = this.amplitude * Math.sin(t * Math.PI * 2 - Math.PI);
    const z = this.zAmplitude * Math.sin(t * Math.PI);
    return target.set(x, y, z);
  }
}

/** Helix curve */
export class HelixCurve extends THREE.Curve {
  constructor(radius = 1.2, turns = 2, height = 5) {
    super();
    this.radius = radius;
    this.turns = turns;
    this.height = height;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const angle = t * Math.PI * 2 * this.turns;
    const x = this.radius * Math.cos(angle);
    const z = this.radius * Math.sin(angle);
    const y = (t - 0.5) * this.height;
    return target.set(x, y, z);
  }
}

/** Random spline using CatmullRomCurve3 with seeded control points */
export class RandomSplineCurve extends THREE.Curve {
  constructor(rng, numPoints = 6, spread = 2.5, length = 6) {
    super();
    const points = [];
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const x = (t - 0.5) * length;
      const y = (rng() - 0.5) * spread;
      const z = (rng() - 0.5) * spread;
      points.push(new THREE.Vector3(x, y, z));
    }
    this._inner = new THREE.CatmullRomCurve3(points, false, 'centripetal');
  }
  getPoint(t, target = new THREE.Vector3()) {
    return this._inner.getPoint(t, target);
  }
}

/** Ring (closed circle / ellipse) for cyclic bead strings */
export class RingCurve extends THREE.Curve {
  constructor(radius = 2, eccentricity = 0) {
    super();
    this.radius = radius;
    this.eccentricity = eccentricity;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const angle = t * Math.PI * 2;
    const rx = this.radius;
    const ry = this.radius * (1 - this.eccentricity * 0.5);
    return target.set(rx * Math.cos(angle), ry * Math.sin(angle), 0);
  }
}

/** Wavy ring — a closed loop with 3D wobble */
export class WavyRingCurve extends THREE.Curve {
  constructor(radius = 2, wobbleAmp = 0.5, wobbleFreq = 3) {
    super();
    this.radius = radius;
    this.wobbleAmp = wobbleAmp;
    this.wobbleFreq = wobbleFreq;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const angle = t * Math.PI * 2;
    const r = this.radius + this.wobbleAmp * Math.sin(this.wobbleFreq * angle);
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    const z = this.wobbleAmp * 0.6 * Math.cos(this.wobbleFreq * angle * 0.7);
    return target.set(x, y, z);
  }
}

/**
 * Tangled loop — a torus knot T(p,q), a closed self-crossing curve.
 *
 * The xy projection self-crosses several times but the z component gives
 * every crossing a clear over/under offset. Compared to the previous
 * Lissajous form, torus knots are cusp-free (no sharp lobe-tip turns) and
 * the z separation at xy crossings is bounded away from zero, reducing the
 * "two ropes threading one bead" illusion.
 */
export class TangledLoopCurve extends THREE.Curve {
  constructor(rng, complexity = 0.5) {
    super();
    // (p,q) coprime → non-trivial knot; xy crossings ≈ q·(p−1).
    const presets = [
      { p: 2, q: 3 },   // trefoil (3 crossings)
      { p: 2, q: 5 },   // cinquefoil (5 crossings)
      { p: 3, q: 5 },   // 10 crossings, most tangled
    ];
    const idx = Math.min(
      presets.length - 1,
      Math.max(0, Math.floor(complexity * presets.length)),
    );
    const { p, q } = presets[idx];
    this.p = p;
    this.q = q;
    this.R = 2.3;
    // Keep the knot visibly loose so bead order remains readable instead of
    // collapsing into a dense ball of crossings.
    this.r = 1.05;
    this.phi = rng() * Math.PI * 2;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const u = t * Math.PI * 2 + this.phi;
    const cq = Math.cos(this.q * u);
    const sq = Math.sin(this.q * u);
    const x = (this.R + this.r * cq) * Math.cos(this.p * u);
    const y = (this.R + this.r * cq) * Math.sin(this.p * u);
    const z = this.r * sq;
    return target.set(x, y, z);
  }
}

// ============= Curve-to-CatmullRom Conversion =============

/**
 * Convert a parametric curve to a CatmullRomCurve3 by sampling control points.
 * This ensures compatibility with Three.js TubeGeometry.
 * @param {THREE.Curve} parametricCurve
 * @param {number} numControlPoints
 * @param {boolean} closed
 * @returns {THREE.CatmullRomCurve3}
 */
function toCatmullRom(parametricCurve, numControlPoints = 60, closed = false) {
  const points = [];
  const n = closed ? numControlPoints : numControlPoints - 1;
  for (let i = 0; i <= n; i++) {
    const t = closed ? (i / numControlPoints) : (i / n);
    points.push(parametricCurve.getPoint(t));
  }
  // For closed curves, don't duplicate the last point (CatmullRom handles it)
  if (closed && points.length > 1) {
    points.pop();
  }
  return new THREE.CatmullRomCurve3(points, closed, 'centripetal');
}

// ============= Curve Factory =============

/**
 * Create a curve based on type name and complexity.
 * Returns a CatmullRomCurve3 for guaranteed TubeGeometry compatibility.
 * @param {string} curveType - 'straight' | 'arc' | 's_curve' | 'helix' | 'random_spline' | 'ring' | 'wavy_ring'
 * @param {number} complexity - 0..1
 * @param {Function} rng - seeded RNG
 * @returns {{ curve: THREE.CatmullRomCurve3, closed: boolean, parametricCurve: THREE.Curve }}
 */
export function createCurve(curveType, complexity = 0.3, rng = Math.random) {
  const c = Math.max(0, Math.min(1, complexity));
  let parametricCurve;
  let closed = false;
  let numCtrlPts = 60;

  switch (curveType) {
    case 'straight': {
      const sag = c * 1.2;
      parametricCurve = new StraightCurve(7.1, sag);
      numCtrlPts = 20;
      break;
    }
    case 'arc': {
      const arcAngle = Math.PI * (0.3 + c * 0.9);
      parametricCurve = new ArcCurve(3.6, arcAngle);
      numCtrlPts = 30;
      break;
    }
    case 's_curve': {
      const amp = 0.65 + c * 2.3;
      const zAmp = c * 1.8;
      parametricCurve = new SCurve(amp, 7.2, zAmp);
      numCtrlPts = 40;
      break;
    }
    case 'helix': {
      const turns = 0.9 + c * 2.35;
      const radius = 0.85 + c * 1.25;
      parametricCurve = new HelixCurve(radius, turns, 6.2);
      numCtrlPts = Math.max(40, Math.floor(turns * 20));
      break;
    }
    case 'random_spline': {
      const numPts = 5 + Math.floor(c * 5);
      const spread = 1.5 + c * 3.5;
      parametricCurve = new RandomSplineCurve(rng, numPts, spread, 7.6);
      numCtrlPts = 50;
      break;
    }
    case 'ring': {
      parametricCurve = new RingCurve(2.45, c * 0.4);
      closed = true;
      numCtrlPts = 40;
      break;
    }
    case 'wavy_ring': {
      const wobbleAmp = 0.2 + c * 0.8;
      const wobbleFreq = 2 + Math.floor(c * 4);
      parametricCurve = new WavyRingCurve(2.45, wobbleAmp, wobbleFreq);
      closed = true;
      numCtrlPts = 60;
      break;
    }
    case 'tangled_loop': {
      parametricCurve = new TangledLoopCurve(rng, c);
      closed = true;
      numCtrlPts = 200;
      break;
    }
    default:
      parametricCurve = new StraightCurve(6, 0);
      numCtrlPts = 20;
  }

  const curve = toCatmullRom(parametricCurve, numCtrlPts, closed);
  return { curve, closed, parametricCurve };
}

// ============= Arc-Length Sampling =============

/**
 * Sample N equidistant points along a curve using arc-length parameterization.
 * Also returns tangent vectors at each point.
 * @param {THREE.Curve} curve
 * @param {number} n - number of points
 * @param {boolean} closed - if true, distribute evenly around the full loop
 * @returns {{ positions: THREE.Vector3[], tangents: THREE.Vector3[] }}
 */
export function sampleEquidistantPoints(curve, n, closed = false, startOffsetT = 0) {
  // Dense sampling for arc-length computation
  const denseSamples = Math.max(500, n * 50);
  const dense = [];
  for (let i = 0; i <= denseSamples; i++) {
    const t = i / denseSamples;
    dense.push(curve.getPoint(t));
  }

  // Compute cumulative arc lengths
  const arcLengths = [0];
  for (let i = 1; i < dense.length; i++) {
    arcLengths.push(arcLengths[i - 1] + dense[i].distanceTo(dense[i - 1]));
  }
  const totalLength = arcLengths[arcLengths.length - 1];
  if (totalLength < 1e-9) {
    return {
      positions: Array.from({ length: n }, () => curve.getPoint(0).clone()),
      tangents: Array.from({ length: n }, () => new THREE.Vector3(1, 0, 0)),
    };
  }

  // For open curves: place beads from some margin to some margin
  // For closed curves: distribute evenly around the full loop, optionally
  // rotated by startOffsetT ∈ [0,1) so callers can shift where bead 0 lands.
  const positions = [];
  const tangents = [];
  const offset = closed ? (((startOffsetT % 1) + 1) % 1) * totalLength : 0;

  for (let i = 0; i < n; i++) {
    let targetArc;
    if (closed) {
      const raw = offset + (i / n) * totalLength;
      targetArc = raw - Math.floor(raw / totalLength) * totalLength;
    } else {
      // Add slight margin at ends so beads don't sit right on the tips
      const margin = totalLength * 0.04;
      const usableLength = totalLength - 2 * margin;
      if (n === 1) {
        targetArc = totalLength / 2;
      } else {
        targetArc = margin + (i / (n - 1)) * usableLength;
      }
    }

    // Binary search for the segment
    let lo = 0, hi = arcLengths.length - 2;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arcLengths[mid + 1] < targetArc) lo = mid + 1;
      else hi = mid;
    }
    const seg = lo;
    const s0 = arcLengths[seg];
    const s1 = arcLengths[seg + 1];
    const localT = (s1 - s0) > 1e-9 ? (targetArc - s0) / (s1 - s0) : 0;
    const pos = dense[seg].clone().lerp(dense[seg + 1], localT);
    positions.push(pos);

    // Approximate tangent
    const globalT = (seg + localT) / denseSamples;
    const dt = 0.001;
    const pA = curve.getPoint(Math.max(0, globalT - dt));
    const pB = curve.getPoint(Math.min(1, globalT + dt));
    const tangent = pB.clone().sub(pA).normalize();
    tangents.push(tangent);
  }

  return { positions, tangents };
}

export default {
  makeRng,
  createCurve,
  sampleEquidistantPoints,
  StraightCurve, ArcCurve, SCurve, HelixCurve,
  RandomSplineCurve, RingCurve, WavyRingCurve, TangledLoopCurve,
};
