/**
 * Unified Gallery & Dataset Generator
 * 
 * Merges gallery rendering and dataset generation logic.
 * Preserves high-quality rendering output.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as CurveExtras from 'three/addons/curves/CurveExtras.js';
import { computeGaussCodeBestProjection } from './gauss-code-generator.js';
import {
  computeSingleClosedLoopDifficulty,
  getBucketTopology,
  getBucketSaliency,
  getTrapType,
} from './difficulty-controller.js';
import { createRopeMesh, checkSelfIntersection, applyRainbowGradient } from './rope-renderer-unified.js?v=7';
import { processCenterline } from './centerline-pipeline.js?v=7';
import { KNOT_TYPE_REGISTRY } from './knot-type-registry.js?v=7';
import { buildGeometryForKnotType as buildDatasetGeometryForKnotType } from './invariance-renderer.js?v=13';

const Curves = CurveExtras.Curves || CurveExtras;
const FIXED_ROPE_RADIUS = 0.02; // fixed thin rope
const DISPLAY_ROPE_RADIUS = FIXED_ROPE_RADIUS;
const FIXED_ROPE_COLOR = '#ffffff';
const STANDARD_CAMERA_ANGLES = [
  { name: 'front', pos: [0, 2, 8] },
  { name: 'back', pos: [0, 2, -8] },
  { name: 'left', pos: [-8, 2, 0] },
  { name: 'right', pos: [8, 2, 0] },
  { name: 'top', pos: [0, 10, 0.1] },
  { name: 'iso_fr', pos: [5, 5, 7] },
  { name: 'iso_bk', pos: [-5, 5, -7] },
  { name: 'oblique', pos: [3, 8, 4] },
];

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

function makeRng(seedStr) {
  const seedFn = xmur3(seedStr || 'seed');
  return mulberry32(seedFn());
}

function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
function parsePositiveInt(value, fallback, { min = 1, max = 100000 } = {}) {
  const n = Math.floor(Number(value));
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}
function parseNumber(value, fallback, { min = -Infinity, max = Infinity } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}
function pick(rng, arr) {
  return arr[Math.min(arr.length - 1, Math.floor(rng() * arr.length))];
}
function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }

// ============= Curve Classes =============

class TorusKnotCurve extends THREE.Curve {
  constructor({ p = 2, q = 3, R = 1.0, r = 0.4 } = {}) {
    super();
    this.p = p; this.q = q; this.R = R; this.r = r;
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const phi = t * Math.PI * 2;
    const { p, q, R, r } = this;
    // Use q for winding around tube, p for winding around center
    const radial = R + r * Math.cos(q * phi);
    return optionalTarget.set(
      radial * Math.cos(p * phi),
      radial * Math.sin(p * phi),
      r * Math.sin(q * phi)
    );
  }
}

/**
 * Figure-eight knot (4_1) — proper closed parametric curve.
 * Replaces Three.js FigureEightPolynomialKnot which is an open polynomial curve.
 */
class FigureEightKnotCurve extends THREE.Curve {
  constructor({ R = 2.0, r = 0.5, zAmp = 0.8 } = {}) {
    super();
    this.R = R; this.r = r; this.zAmp = zAmp;
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const phi = t * Math.PI * 2;
    const radial = this.R + this.r * Math.cos(2 * phi);
    return optionalTarget.set(
      radial * Math.cos(3 * phi),
      radial * Math.sin(3 * phi),
      this.zAmp * Math.sin(4 * phi)
    );
  }
}

class CircleCurve extends THREE.Curve {
  constructor({ radius = 1.0, center = new THREE.Vector3(0, 0, 0), normal = new THREE.Vector3(0, 0, 1) } = {}) {
    super();
    this.radius = radius;
    this.center = center.clone();
    this.normal = normal.clone().normalize();
    const tmp = Math.abs(this.normal.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    this.u = new THREE.Vector3().crossVectors(this.normal, tmp).normalize();
    this.v = new THREE.Vector3().crossVectors(this.normal, this.u).normalize();
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const a = t * Math.PI * 2;
    return optionalTarget
      .copy(this.center)
      .addScaledVector(this.u, this.radius * Math.cos(a))
      .addScaledVector(this.v, this.radius * Math.sin(a));
  }
}

class TwistedRingCurve extends THREE.Curve {
  constructor({ R = 1.0, twist = 3, wobble = 0.22, height = 0.35 } = {}) {
    super();
    this.R = R; this.twist = twist; this.wobble = wobble; this.height = height;
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const a = t * Math.PI * 2;
    const k = this.twist;
    const rr = this.R * (1 + this.wobble * Math.sin(k * a));
    const z = this.height * Math.cos(k * a);
    return optionalTarget.set(rr * Math.cos(a), rr * Math.sin(a), z);
  }
}

// SpiralLoopCurve - CLOSED spiral loop with physical clearance (no self-intersection for TubeGeometry)
// Design goal: a single closed rope (unknot) that "looks like a spiral" but respects thickness:
// - adjacent turns separated by >= ~2*tubeRadius (radial gap)
// - layers separated by >= ~2*tubeRadius (pitch in Z)
// - connector routed OUTSIDE the spiral envelope (no cutting through)
function minNonNeighborDistanceVec3(points, neighborSkip = 6, { closed = true } = {}) {
  const n = points.length;
  if (n < neighborSkip * 2 + 2) return Infinity;
  let minD2 = Infinity;
  for (let i = 0; i < n; i++) {
    const a = points[i];
    for (let j = i + 1; j < n; j++) {
      const dj = j - i;
      if (closed) {
        const wrapDj = Math.min(dj, n - dj);
        if (wrapDj <= neighborSkip) continue;
      } else {
        if (dj <= neighborSkip) continue;
      }
      const b = points[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < minD2) minD2 = d2;
    }
  }
  return Math.sqrt(minD2);
}

// SpiralLoopCurve - PHYSICAL CLOSED LOOP (No Self-Intersection)
// Strategy: spiral outward/upward, then continue a helical arc that dives BELOW the spiral and comes back to start.
class SpiralLoopCurve extends THREE.Curve {
  constructor({
    turns = 3,
    pitch = 0.2,
    innerRadius = 0.7,
    radialGap = 0.3,
    tubeRadius = 0.24,
  } = {}) {
    super();
    this.turns = Math.max(1, turns);
    this.tubeRadius = Math.max(0.01, tubeRadius);

    // Enforce physical clearance: keep centerline gaps > rope diameter (more conservative)
    const minClearance = this.tubeRadius * 2.8;
    this.radialGap = Math.max(radialGap, minClearance);
    this.pitch = Math.max(pitch, minClearance);
    this.innerRadius = Math.max(innerRadius, this.tubeRadius * 3.5);

    // Spiral end state
    this.endR = this.innerRadius + this.turns * this.radialGap;
    this.endZ = this.turns * this.pitch;
    this.endAngle = this.turns * Math.PI * 2;

    // Return path: add one more turn while diving below z=0
    this.returnEndAngle = this.endAngle + Math.PI * 2; // one extra full turn
    this.returnMidZ = -(this.tubeRadius * 5.5); // dip well below first layer for clearance

    // Phase distribution (forward spiral / return helix)
    this.tForward = 0.65;
    this.tReturn = 0.35;
  }

  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const target = optionalTarget;
    const smooth = (x) => {
      const y = clamp(x, 0, 1);
      return y * y * (3 - 2 * y);
    };

    if (t <= this.tForward) {
      // === PHASE 1: Spiral Outward & Upward ===
      const localT = t / this.tForward;
      const angle = localT * this.turns * Math.PI * 2;
      const r = this.innerRadius + localT * this.turns * this.radialGap;
      const z = localT * this.turns * this.pitch;
      return target.set(r * Math.cos(angle), r * Math.sin(angle), z);
    }

    // === PHASE 2: Helical Arc Return (wrap underneath) ===
    const u = (t - this.tForward) / this.tReturn; // 0..1
    // Two-stage easing: first half descend and start shrinking radius, second half rise to z=0 and finish shrinking
    if (u < 0.5) {
      const k = smooth(u * 2); // 0..1
      const angle = this.endAngle + (this.returnEndAngle - this.endAngle) * 0.5 * k;
      const rStart = this.endR;
      const rMid = (this.endR + this.innerRadius) * 0.5;
      const r = rStart + (rMid - rStart) * k;
      const z = this.endZ + (this.returnMidZ - this.endZ) * k;
      return target.set(r * Math.cos(angle), r * Math.sin(angle), z);
    } else {
      const k = smooth((u - 0.5) * 2); // 0..1
      const angle = this.endAngle + (this.returnEndAngle - this.endAngle) * (0.5 + 0.5 * k);
      const rMid = (this.endR + this.innerRadius) * 0.5;
      const r = rMid + (this.innerRadius - rMid) * k;
      const z = this.returnMidZ + (0 - this.returnMidZ) * k; // rise back to z=0
      return target.set(r * Math.cos(angle), r * Math.sin(angle), z);
    }
  }
}

class PlanarWobbleCircleCurve extends THREE.Curve {
  constructor({ radius = 1.0, center = new THREE.Vector3(0,0,0), normal = new THREE.Vector3(0,0,1), waves = 3, amp = 0.06, phase = 0 } = {}) {
    super();
    this.radius = radius;
    this.center = center.clone();
    this.normal = normal.clone().normalize();
    this.waves = waves; this.amp = amp; this.phase = phase;
    const tmp = Math.abs(this.normal.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    this.u = new THREE.Vector3().crossVectors(this.normal, tmp).normalize();
    this.v = new THREE.Vector3().crossVectors(this.normal, this.u).normalize();
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const a = t * Math.PI * 2;
    const r = this.radius * (1 + this.amp * Math.sin(this.waves * a + this.phase));
    return optionalTarget
      .copy(this.center)
      .addScaledVector(this.u, r * Math.cos(a))
      .addScaledVector(this.v, r * Math.sin(a));
  }
}

// Kinky Unknot - looks complex but is topologically trivial
class KinkyUnknotCurve extends THREE.Curve {
  constructor({ k = 4, baseRadius = 1.0, kinkAmplitude = 0.25, seed = 12345 } = {}) {
    super();
    this.k = Math.max(2, Math.floor(k));
    this.baseRadius = baseRadius;
    this.kinkAmplitude = kinkAmplitude;
    this.rng = mulberry32(seed);
    // Pre-generate kink parameters
    this.kinks = [];
    for (let i = 0; i < this.k; i++) {
      this.kinks.push({
        phase: this.rng() * Math.PI * 2,
        sigma: 0.06 + this.rng() * 0.04,
        bulgePhase: this.rng() * Math.PI * 2,
      });
    }
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const angle = t * Math.PI * 2;
    const r = this.baseRadius;
    let x = r * Math.cos(angle);
    let y = r * Math.sin(angle);
    let z = 0;
    
    for (let i = 0; i < this.k; i++) {
      const kinkCenter = (i + 0.5) / this.k;
      const dist = Math.abs(((t - kinkCenter + 0.5) % 1) - 0.5);
      const envelope = Math.exp(-dist * dist / (this.kinks[i].sigma * this.kinks[i].sigma));
      const bp = this.kinks[i].bulgePhase;
      
      x += this.kinkAmplitude * envelope * Math.sin(angle * 3 + bp) * 0.5;
      y += this.kinkAmplitude * envelope * Math.cos(angle * 2 + bp) * 0.5;
      z += this.kinkAmplitude * 1.2 * envelope * Math.sin(angle * 4 + i + bp);
    }
    
    return optionalTarget.set(x, y, z);
  }
}

function buildLooseOpenKnotGeometry({
  rng,
  quality = 'mid',
  radius = 0.24,
  deformStrength = 0.3,
  slackness = 0.4,
  segments = null,
} = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const ropeColor = FIXED_ROPE_COLOR;
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const s = clamp(Number(slackness) || 0, 0, 1);
  const d = clamp(Number(deformStrength) || 0, 0, 1);

  // Base shape from trefoil-like centerline, then cut open and add loose tails.
  const baseCurve = new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.38 });
  const sampleN = 420;
  let ringPts = baseCurve.getPoints(sampleN);
  ringPts = ringPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (ringPts.length < 40) {
    const fallback = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.4, -0.2, 0.1),
      new THREE.Vector3(-0.8, 0.5, 0.3),
      new THREE.Vector3(0.2, 0.7, -0.2),
      new THREE.Vector3(0.9, 0.1, 0.4),
      new THREE.Vector3(0.5, -0.7, -0.1),
      new THREE.Vector3(-0.6, -0.8, 0.2),
      new THREE.Vector3(-1.3, -0.1, -0.2),
      new THREE.Vector3(-1.9, 0.4, 0.1),
    ], false, 'centripetal');
    const fallbackMesh = createRopeMesh(fallback, {
      radius: tubeRadius,
      color: ropeColor,
      closed: false,
      tubularSegments: Math.max(260, tubularSegments),
      radialSegments,
    });
    const fallbackGeom = fallbackMesh.geometry;
    if (fallbackMesh.material) fallbackMesh.material.dispose();
    fallbackGeom.center();
    fallbackGeom.computeBoundingSphere();
    const s0 = fallbackGeom.boundingSphere?.radius > 1e-6 ? (1.35 / fallbackGeom.boundingSphere.radius) : 1.0;
    fallbackGeom.scale(s0, s0, s0);
    fallbackGeom.center();
    return fallbackGeom;
  }

  const start = Math.floor((rng ? rng() : 0.15) * ringPts.length * 0.35);
  const span = Math.max(80, Math.floor(ringPts.length * 0.82));
  let corePts = [];
  for (let i = 0; i <= span; i++) {
    corePts.push(ringPts[(start + i) % ringPts.length].clone());
  }

  // Make it look looser and ring-like while keeping residual knot structure.
  const zScale = clamp(1.0 - 0.72 * s, 0.18, 1.0);
  const deformAmp = d * (0.05 + 0.09 * (1 - 0.5 * s));
  const tau = Math.PI * 2;
  const ph1 = (rng ? rng() : 0.31) * tau;
  const ph2 = (rng ? rng() : 0.47) * tau;
  const ph3 = (rng ? rng() : 0.63) * tau;
  for (let i = 0; i < corePts.length; i++) {
    const t = i / Math.max(1, corePts.length - 1);
    const w = Math.exp(-((t - 0.5) ** 2) / 0.07);
    const stretch = 1 + (0.03 + 0.18 * s) * w;
    corePts[i].multiplyScalar(stretch);
    corePts[i].z *= zScale;
    if (deformAmp > 1e-6) {
      const nx = 0.65 * Math.sin(tau * (1.6 + 0.8 * s) * t + ph1) + 0.35 * Math.cos(tau * 3.1 * t + ph2);
      const ny = 0.60 * Math.cos(tau * (1.2 + 0.6 * d) * t + ph2) + 0.40 * Math.sin(tau * 2.7 * t + ph1);
      const nz = 0.75 * Math.sin(tau * (1.0 + 0.7 * s) * t + ph3);
      corePts[i].x += deformAmp * nx;
      corePts[i].y += deformAmp * ny;
      corePts[i].z += deformAmp * 0.7 * nz;
    }
  }

  // Add open rope tails on both ends.
  const headDir = corePts[0].clone().sub(corePts[1]).normalize();
  const tailDir = corePts[corePts.length - 1].clone().sub(corePts[corePts.length - 2]).normalize();
  const tailLenA = (0.70 + 0.55 * s) + (rng ? rng() * 0.35 : 0.2);
  const tailLenB = (0.75 + 0.60 * s) + (rng ? rng() * 0.35 : 0.2);
  const liftZ = 1.0 - 0.6 * s;
  const liftA = new THREE.Vector3(-0.10, 0.18, 0.08 * liftZ);
  const liftB = new THREE.Vector3(0.12, -0.15, -0.10 * liftZ);

  const h0 = corePts[0].clone();
  const h1 = h0.clone().addScaledVector(headDir, tailLenA * 0.7).addScaledVector(liftA, 0.6);
  const h2 = h0.clone().addScaledVector(headDir, tailLenA * 1.5).add(liftA);
  const t0 = corePts[corePts.length - 1].clone();
  const t1 = t0.clone().addScaledVector(tailDir, tailLenB * 0.7).addScaledVector(liftB, 0.6);
  const t2 = t0.clone().addScaledVector(tailDir, tailLenB * 1.5).add(liftB);

  corePts = [h2, h1, ...corePts, t1, t2];

  // Open-curve smoothing and resampling to suppress local jaggedness.
  let openCurve = new THREE.CatmullRomCurve3(corePts, false, 'centripetal');
  let smoothPts = openCurve.getPoints(Math.max(260, tubularSegments));
  smoothPts = smoothPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (smoothPts.length >= 20) {
    const iters = Math.round(4 + 8 * s + 4 * d);
    const beta = 0.12 + 0.10 * s;
    for (let it = 0; it < iters; it++) {
      const next = smoothPts.map((p) => p.clone());
      for (let i = 1; i < smoothPts.length - 1; i++) {
        const avg = smoothPts[i - 1].clone().add(smoothPts[i]).add(smoothPts[i + 1]).divideScalar(3);
        next[i].lerp(avg, beta);
      }
      smoothPts = next;
    }
    openCurve = new THREE.CatmullRomCurve3(smoothPts, false, 'centripetal');
  }

  const mesh = createRopeMesh(openCurve, {
    radius: tubeRadius,
    color: ropeColor,
    closed: false,
    tubularSegments: Math.max(280, tubularSegments),
    radialSegments,
  });

  const geom = mesh.geometry;
  if (mesh.material) mesh.material.dispose();
  geom.center();
  geom.computeBoundingSphere();
  const sNorm = geom.boundingSphere?.radius > 1e-6 ? (1.35 / geom.boundingSphere.radius) : 1.0;
  geom.scale(sNorm, sNorm, sNorm);
  geom.center();
  return geom;
}

// ============= New Geometry Builders (loose, link groups, mixed) =============

/**
 * Generic loose knot builder - generates loose open variants from existing knot types.
 * Similar to buildLooseOpenKnotGeometry but supports different base curves.
 */
function buildGenericLooseKnotGeometry({
  rng,
  quality = 'mid',
  baseCurveFactory,  // () => THREE.Curve
  deformStrength = 0.3,
  slackness = 0.6,
  segments = null,
} = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const s = clamp(Number(slackness) || 0, 0, 1);
  const d = clamp(Number(deformStrength) || 0, 0, 1);

  const baseCurve = baseCurveFactory();
  const sampleN = 420;
  let ringPts = baseCurve.getPoints(sampleN);
  ringPts = ringPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (ringPts.length < 40) return null;

  const start = Math.floor((rng ? rng() : 0.15) * ringPts.length * 0.35);
  const span = Math.max(80, Math.floor(ringPts.length * 0.82));
  let corePts = [];
  for (let i = 0; i <= span; i++) {
    corePts.push(ringPts[(start + i) % ringPts.length].clone());
  }

  const zScale = clamp(1.0 - 0.72 * s, 0.18, 1.0);
  const deformAmp = d * (0.05 + 0.09 * (1 - 0.5 * s));
  const tau = Math.PI * 2;
  const ph1 = (rng ? rng() : 0.31) * tau;
  const ph2 = (rng ? rng() : 0.47) * tau;
  const ph3 = (rng ? rng() : 0.63) * tau;
  for (let i = 0; i < corePts.length; i++) {
    const t = i / Math.max(1, corePts.length - 1);
    const w = Math.exp(-((t - 0.5) ** 2) / 0.07);
    corePts[i].multiplyScalar(1 + (0.03 + 0.18 * s) * w);
    corePts[i].z *= zScale;
    if (deformAmp > 1e-6) {
      corePts[i].x += deformAmp * (0.65 * Math.sin(tau * (1.6 + 0.8 * s) * t + ph1) + 0.35 * Math.cos(tau * 3.1 * t + ph2));
      corePts[i].y += deformAmp * (0.60 * Math.cos(tau * (1.2 + 0.6 * d) * t + ph2) + 0.40 * Math.sin(tau * 2.7 * t + ph1));
      corePts[i].z += deformAmp * 0.7 * 0.75 * Math.sin(tau * (1.0 + 0.7 * s) * t + ph3);
    }
  }

  // Open rope tails
  const headDir = corePts[0].clone().sub(corePts[1]).normalize();
  const tailDir = corePts[corePts.length - 1].clone().sub(corePts[corePts.length - 2]).normalize();
  const tailLenA = (0.70 + 0.55 * s) + (rng ? rng() * 0.35 : 0.2);
  const tailLenB = (0.75 + 0.60 * s) + (rng ? rng() * 0.35 : 0.2);
  const liftZ = 1.0 - 0.6 * s;
  const liftA = new THREE.Vector3(-0.10, 0.18, 0.08 * liftZ);
  const liftB = new THREE.Vector3(0.12, -0.15, -0.10 * liftZ);
  const h0 = corePts[0].clone();
  const h1 = h0.clone().addScaledVector(headDir, tailLenA * 0.7).addScaledVector(liftA, 0.6);
  const h2 = h0.clone().addScaledVector(headDir, tailLenA * 1.5).add(liftA);
  const t0 = corePts[corePts.length - 1].clone();
  const t1 = t0.clone().addScaledVector(tailDir, tailLenB * 0.7).addScaledVector(liftB, 0.6);
  const t2 = t0.clone().addScaledVector(tailDir, tailLenB * 1.5).add(liftB);
  corePts = [h2, h1, ...corePts, t1, t2];

  let openCurve = new THREE.CatmullRomCurve3(corePts, false, 'centripetal');
  let smoothPts = openCurve.getPoints(Math.max(260, tubularSegments));
  smoothPts = smoothPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (smoothPts.length >= 20) {
    const iters = Math.round(4 + 8 * s + 4 * d);
    const beta = 0.12 + 0.10 * s;
    for (let it = 0; it < iters; it++) {
      const next = smoothPts.map((p) => p.clone());
      for (let i = 1; i < smoothPts.length - 1; i++) {
        const avg = smoothPts[i - 1].clone().add(smoothPts[i]).add(smoothPts[i + 1]).divideScalar(3);
        next[i].lerp(avg, beta);
      }
      smoothPts = next;
    }
    openCurve = new THREE.CatmullRomCurve3(smoothPts, false, 'centripetal');
  }

  const mesh = createRopeMesh(openCurve, {
    radius: tubeRadius, color: FIXED_ROPE_COLOR, closed: false,
    tubularSegments: Math.max(280, tubularSegments), radialSegments,
  });
  const geom = mesh.geometry;
  if (mesh.material) mesh.material.dispose();
  geom.center();
  geom.computeBoundingSphere();
  const sNorm = geom.boundingSphere?.radius > 1e-6 ? (1.35 / geom.boundingSphere.radius) : 1.0;
  geom.scale(sNorm, sNorm, sNorm);
  geom.center();
  return geom;
}


/**
 * Helper: build a single chain segment, returns geometry array
 */
function buildOneChainSegmentGallery({ rng, R, tubeRadius, numLinks, offset, tubularSegments, radialSegments, hueBase = 0 }) {
  const { centers, normals } = generateDiverseChainLayout({
    rng, numLinks, R, tubeRadius,
    effectiveStep: R * 1.3, linkOffsetY: -R * 1.3,
  });
  const geoms = [];
  for (let i = 0; i < numLinks; i++) {
    const center = centers[i].clone().add(offset);
    const normal = normals[i];
    const ringR = rng ? (R * (0.92 + rng() * 0.16)) : R;
    const amp = rng ? 0.035 + rng() * 0.06 : 0.05;
    const waves = rng ? 2 + Math.floor(rng() * 5) : 3;
    const phase = rng ? rng() * Math.PI * 2 : 0;
    const curve = new PlanarWobbleCircleCurve({ radius: ringR, center, normal, waves, amp, phase });
    const mesh = createRopeMesh(curve, { radius: tubeRadius, color: FIXED_ROPE_COLOR, closed: true, tubularSegments, radialSegments });
    const g = mesh.geometry;
    if (mesh.material) mesh.material.dispose();
    applyRainbowGradient(g, radialSegments, hueBase + (i / Math.max(1, numLinks)) * 0.3);
    geoms.push(g);
  }
  return geoms;
}

function buildFreeRingGallery({ rng, R, tubeRadius, center, tubularSegments, radialSegments, hue = 0.5 }) {
  const freeR = R * (0.7 + (rng?.() ?? 0.5) * 0.5);
  const freeNormal = new THREE.Vector3((rng?.() ?? 0.5) - 0.5, (rng?.() ?? 0.5) - 0.5, 0.5 + (rng?.() ?? 0.5) * 0.5).normalize();
  const curve = new PlanarWobbleCircleCurve({ radius: freeR, center, normal: freeNormal, waves: 2 + Math.floor((rng?.() ?? 0.5) * 3), amp: 0.04 + (rng?.() ?? 0.5) * 0.06 });
  const mesh = createRopeMesh(curve, { radius: tubeRadius, color: FIXED_ROPE_COLOR, closed: true, tubularSegments, radialSegments });
  const g = mesh.geometry;
  if (mesh.material) mesh.material.dispose();
  applyRainbowGradient(g, radialSegments, hue);
  return g;
}

/**
 * Double chain group - 2 independent chains, each internally interlinked
 */
function buildDoubleHopfGeometry({ rng, quality = 'mid', segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const R = 1.0;
  const numChains = 2;
  const chainSep = R * 5;
  const allGeoms = [];
  for (let c = 0; c < numChains; c++) {
    const numLinks = rng ? (2 + Math.floor(rng() * 3)) : 3;
    const offset = new THREE.Vector3((c - (numChains - 1) / 2) * chainSep, ((rng?.() ?? 0.5) - 0.5) * 1.5, ((rng?.() ?? 0.5) - 0.5) * 1.5);
    allGeoms.push(...buildOneChainSegmentGallery({ rng, R, tubeRadius, numLinks, offset, tubularSegments, radialSegments, hueBase: c * 0.45 }));
  }
  const merged = mergeBufferGeometries(allGeoms);
  allGeoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.8 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

/**
 * Link cluster - mixed arrangement of various link types
 */
function buildLinkClusterGeometry({ rng, quality = 'mid', segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const R = 0.9;
  const numChains = rng ? (2 + Math.floor(rng() * 3)) : 3;
  const allGeoms = [];
  for (let c = 0; c < numChains; c++) {
    const numLinks = rng ? (2 + Math.floor(rng() * 3)) : 3;
    const angle = (c / numChains) * Math.PI * 2 + (rng?.() ?? 0) * 0.5;
    const dist = R * 4 + (rng?.() ?? 0.5) * R * 2;
    const offset = new THREE.Vector3(Math.cos(angle) * dist, ((rng?.() ?? 0.5) - 0.5) * 2, Math.sin(angle) * dist);
    allGeoms.push(...buildOneChainSegmentGallery({ rng, R, tubeRadius, numLinks, offset, tubularSegments, radialSegments, hueBase: c / numChains }));
  }
  const merged = mergeBufferGeometries(allGeoms);
  allGeoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 2.0 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

/**
 * Hopf link + free ring - two interlinked rings + independent ring
 */
function buildHopfPlusFreeGeometry({ rng, quality = 'mid', segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const R = 1.0;
  const numFree = rng ? (1 + Math.floor(rng() * 2)) : 1;
  const chainLinks = rng ? (2 + Math.floor(rng() * 2)) : 3;

  const allGeoms = buildOneChainSegmentGallery({
    rng, R, tubeRadius, numLinks: chainLinks,
    offset: new THREE.Vector3(0, 0, 0),
    tubularSegments, radialSegments, hueBase: 0.0,
  });
  for (let i = 0; i < numFree; i++) {
    const offsetAngle = (rng?.() ?? 0.5) * Math.PI * 2;
    const dist = R * 4.5 + (rng?.() ?? 0.5) * R * 2;
    const center = new THREE.Vector3(Math.cos(offsetAngle) * dist, ((rng?.() ?? 0.5) - 0.5) * 2, Math.sin(offsetAngle) * dist);
    allGeoms.push(buildFreeRingGallery({ rng, R, tubeRadius, center, tubularSegments, radialSegments, hue: 0.55 + i * 0.2 }));
  }
  const merged = mergeBufferGeometries(allGeoms);
  allGeoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.8 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

/**
 * Chain + free rings - short chain with independent rings nearby
 */
function buildChainPlusFreeGeometry({ rng, quality = 'mid', segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const R = 0.9;
  const chainLinks = rng ? (3 + Math.floor(rng() * 3)) : 4;
  const numFree = rng ? (1 + Math.floor(rng() * 3)) : 2;

  // Build chain segment using proper interlocking layout
  const allGeoms = buildOneChainSegmentGallery({
    rng, R, tubeRadius, numLinks: chainLinks,
    offset: new THREE.Vector3(0, 0, 0),
    tubularSegments, radialSegments, hueBase: 0.0,
  });

  // Compute chain extent to place free rings outside
  const tmpMerged = mergeBufferGeometries(allGeoms.map(g => g.clone()));
  tmpMerged.computeBoundingSphere();
  const chainExtent = tmpMerged.boundingSphere?.radius || 2;
  tmpMerged.dispose();

  for (let i = 0; i < numFree; i++) {
    const offsetAngle = (i / numFree) * Math.PI * 2 + (rng?.() ?? 0) * 1.0;
    const dist = chainExtent + R * 2.5 + (rng?.() ?? 0.5) * R * 2;
    const center = new THREE.Vector3(
      Math.cos(offsetAngle) * dist,
      ((rng?.() ?? 0.5) - 0.5) * 2,
      Math.sin(offsetAngle) * dist
    );
    allGeoms.push(buildFreeRingGallery({ rng, R, tubeRadius, center, tubularSegments, radialSegments, hue: 0.5 + i * 0.18 }));
  }

  const merged = mergeBufferGeometries(allGeoms);
  allGeoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.8 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  merged._chainLen = chainLinks;
  merged._numFree = numFree;
  return merged;
}

// ============= Preset Definitions =============

const PRESETS = {
  // Basic Knots
  unknot: { name: 'Unknot', kind: 'curve', make: () => new CircleCurve({ radius: 1.0 }), difficulty: 'easy', crossings: 0 },
  trefoil: { name: 'Trefoil', kind: 'torus', p: 2, q: 3, R: 1.0, r: 0.4, difficulty: 'easy', crossings: 3 },
  figure8: { name: 'Figure-8', kind: 'curve', make: () => new FigureEightKnotCurve(), difficulty: 'easy', crossings: 4 },
  
  // Torus Knots - with slimmer r values
  torus_2_5: { name: 'T(2,5) Cinquefoil', kind: 'torus', p: 2, q: 5, R: 1.0, r: 0.38, difficulty: 'medium', crossings: 5 },
  torus_2_7: { name: 'T(2,7) Septafoil', kind: 'torus', p: 2, q: 7, R: 1.0, r: 0.35, difficulty: 'hard', crossings: 7 },
  torus_2_9: { name: 'T(2,9)', kind: 'torus', p: 2, q: 9, R: 1.0, r: 0.32, difficulty: 'hard', crossings: 9 },
  torus_3_4: { name: 'T(3,4)', kind: 'torus', p: 3, q: 4, R: 1.0, r: 0.35, difficulty: 'hard', crossings: 8 },
  torus_3_5: { name: 'T(3,5)', kind: 'torus', p: 3, q: 5, R: 1.0, r: 0.32, difficulty: 'hard', crossings: 10 },
  torusKnot_random: { name: 'Random Torus', kind: 'torusRandom', difficulty: 'mixed' },
  
  // Unknot Variants
  twisted_ring: { name: 'Twisted Ring', kind: 'curve', make: (rng) => new TwistedRingCurve({ R: 1.0, twist: 2 + Math.floor((rng?.() || 0.5) * 5), wobble: 0.18 + (rng?.() || 0.5) * 0.18, height: 0.3 + (rng?.() || 0.5) * 0.25 }), difficulty: 'easy', crossings: 0, isUnknot: true },
  spiral_disk: { name: 'Spiral Loop', kind: 'spiralLoop', difficulty: 'medium', crossings: 0, isUnknot: true },
  kinky_unknot: { name: 'Kinky Unknot', kind: 'kinky', difficulty: 'hard', crossings: 0, isUnknot: true, isDeceptive: true },
  ring_like_open_rope: { name: 'Ring-Like Open-Ended Rope', kind: 'dataset', difficulty: 'hard', crossings: 0, isOpen: true, isUnknot: true, isDeceptive: true },
  loose_open_knot: { name: 'Loose Open Knot', kind: 'openLooseKnot', difficulty: 'hard', crossings: 3, isOpen: true },
  ring_like_open_knot: { name: 'Ring-Like Open Knot', kind: 'dataset', difficulty: 'hard', crossings: 3, isOpen: true },
  
  // Loose Knot Variants
  loose_cinquefoil: { name: 'Loose Cinquefoil', kind: 'looseKnot', baseCurveKind: 'torus_2_5', difficulty: 'hard', crossings: 5 },

  // Occluded Knot
  occluded_knot: { name: 'Occluded Knot', kind: 'occludedKnot', difficulty: 'hard', crossings: 3 },

  // Links
  hopf_link: { name: 'Hopf Link', kind: 'hopfReal', difficulty: 'easy', isLink: true },
  unlinked_rings: { name: 'Unlinked Rings', kind: 'hopfUnlinked', difficulty: 'easy', isLink: true },
  chain: { name: 'Chain', kind: 'chain', difficulty: 'medium', isLink: true },
  borromean: { name: 'Borromean Rings', kind: 'borromean', difficulty: 'hard', isLink: true },

  // Link Groups (multiple links together)
  double_hopf: { name: 'Double Hopf', kind: 'doubleHopf', difficulty: 'hard', isLink: true },
  link_cluster: { name: 'Link Cluster', kind: 'linkCluster', difficulty: 'hard', isLink: true },

  // Mixed (some linked, some free)
  hopf_plus_free: { name: 'Hopf + Free Ring', kind: 'hopfPlusFree', difficulty: 'hard', isLink: true },
  chain_plus_free: { name: 'Chain + Free Rings', kind: 'chainPlusFree', difficulty: 'hard', isLink: true },

  // Benchmark
  benchmark_easy: { name: 'Benchmark Easy', kind: 'benchmark', level: 0 },
  benchmark_medium: { name: 'Benchmark Medium', kind: 'benchmark', level: 1 },
  benchmark_hard: { name: 'Benchmark Hard', kind: 'benchmark', level: 2 },
  benchmark_mix: { name: 'Benchmark Mix', kind: 'benchmarkMix' },
  
  // All
  all: { name: 'All Types', kind: 'all' },
};

// ============= Geometry Builders =============

function tubeQualityParams(q, overrides = null) {
  if (overrides && (Number.isFinite(overrides.tubularSegments) || Number.isFinite(overrides.radialSegments))) {
    const base = tubeQualityParams(q, null);
    return {
      tubularSegments: Number.isFinite(overrides.tubularSegments) ? Math.floor(overrides.tubularSegments) : base.tubularSegments,
      radialSegments: Number.isFinite(overrides.radialSegments) ? Math.floor(overrides.radialSegments) : base.radialSegments,
    };
  }
  if (q === 'high') return { tubularSegments: 280, radialSegments: 18 };
  if (q === 'mid') return { tubularSegments: 200, radialSegments: 14 };
  return { tubularSegments: 120, radialSegments: 10 };
}

function mergeBufferGeometries(geoms) {
  const valid = geoms.filter(Boolean);
  if (!valid.length) return null;
  const out = new THREE.BufferGeometry();
  const attrs = Object.keys(valid[0].attributes);
  for (const a of attrs) {
    const arrays = valid.map(g => g.attributes[a].array);
    const itemSize = valid[0].attributes[a].itemSize;
    const totalLen = arrays.reduce((s, arr) => s + arr.length, 0);
    const merged = new arrays[0].constructor(totalLen);
    let off = 0;
    for (const arr of arrays) { merged.set(arr, off); off += arr.length; }
    out.setAttribute(a, new THREE.BufferAttribute(merged, itemSize));
  }
  const hasIndex = valid.every(g => g.index?.array);
  if (hasIndex) {
    const indexArrays = valid.map(g => g.index.array);
    const total = indexArrays.reduce((s, arr) => s + arr.length, 0);
    const mergedIndex = new indexArrays[0].constructor(total);
    let vertexOffset = 0, off = 0;
    for (let gi = 0; gi < valid.length; gi++) {
      const g = valid[gi], idx = g.index.array;
      for (let j = 0; j < idx.length; j++) mergedIndex[off + j] = idx[j] + vertexOffset;
      off += idx.length;
      vertexOffset += g.attributes.position.count;
    }
    out.setIndex(new THREE.BufferAttribute(mergedIndex, 1));
  }
  out.computeVertexNormals();
  return out;
}

function estimateAndNormalizeTube({ makeCurve, closed = true, quality = 'mid', radius = 0.24, targetOuterRadius = 1.25, segments = null }) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const curve = makeCurve();

  const check = checkSelfIntersection(curve, 0.05);
  if (check.hasIntersection) {
    // eslint-disable-next-line no-console
    console.warn(`Self-intersection detected! minDist=${check.minDist}`);
    // optional: auto-scale curve or regenerate
  }

  // Estimate scale using centerline points (avoid creating TubeGeometry here).
  let pts = curve.getPoints(Math.max(64, Math.floor(tubularSegments)));
  pts = pts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (pts.length < 10) throw new Error('[pipeline] Too few valid points for estimateAndNormalizeTube');
  const sphere = new THREE.Sphere();
  new THREE.Box3().setFromPoints(pts).getBoundingSphere(sphere);
  const sNorm = sphere.radius > 1e-6 ? (targetOuterRadius / sphere.radius) : 1.0;

  // Keep world-space rope radius == `radius` after scaling.
  const baseRadius = radius / sNorm;

  const ropeColor = FIXED_ROPE_COLOR;
  const mesh = createRopeMesh(curve, { radius: baseRadius, color: ropeColor, closed, tubularSegments, radialSegments });
  const geom = mesh.geometry;
  if (mesh.material) mesh.material.dispose();

  geom.scale(sNorm, sNorm, sNorm);
  geom.computeVertexNormals();
  geom.center();
  geom.computeBoundingSphere();
  return geom;
}

// Build Hopf Link (two interlinked rings)
function buildRealHopfLinkGeometry({ rng, quality = 'mid', radius = 0.24, segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const ampA = rng ? 0.04 + rng() * 0.08 : 0.06;
  const ampB = rng ? 0.04 + rng() * 0.08 : 0.06;
  const wavesA = rng ? 2 + Math.floor(rng() * 4) : 3;
  const wavesB = rng ? 2 + Math.floor(rng() * 4) : 3;
  const phaseA = rng ? rng() * Math.PI * 2 : 0;
  const phaseB = rng ? rng() * Math.PI * 2 : 0;

  const R = 1.5;
  const tubeRadius = DISPLAY_ROPE_RADIUS;

  const curveA = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: wavesA, amp: ampA, phase: phaseA });
  const curveB = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, -2, 0), normal: new THREE.Vector3(1, 0, 0), waves: wavesB, amp: ampB, phase: phaseB });

  const ropeColor = FIXED_ROPE_COLOR;
  {
    const check = checkSelfIntersection(curveA, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }
  }
  {
    const check = checkSelfIntersection(curveB, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }
  }

  const aMesh = createRopeMesh(curveA, { radius: tubeRadius, color: ropeColor, closed: true, tubularSegments, radialSegments });
  const bMesh = createRopeMesh(curveB, { radius: tubeRadius, color: ropeColor, closed: true, tubularSegments, radialSegments });
  const a = aMesh.geometry;
  const b = bMesh.geometry;
  if (aMesh.material) aMesh.material.dispose();
  if (bMesh.material) bMesh.material.dispose();

  // Colors
  const colorsA = new Float32Array(a.attributes.position.count * 3);
  for (let i = 0; i < a.attributes.position.count; i++) { colorsA[i*3] = 1.0; colorsA[i*3+1] = 0.85; colorsA[i*3+2] = 0.85; }
  a.setAttribute('color', new THREE.BufferAttribute(colorsA, 3));

  const colorsB = new Float32Array(b.attributes.position.count * 3);
  for (let i = 0; i < b.attributes.position.count; i++) { colorsB[i*3] = 0.85; colorsB[i*3+1] = 0.9; colorsB[i*3+2] = 1.0; }
  b.setAttribute('color', new THREE.BufferAttribute(colorsB, 3));

  const merged = mergeBufferGeometries([a, b]);
  a.dispose(); b.dispose();
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.35 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  merged.computeBoundingSphere();
  return merged;
}

// Build Unlinked Rings
function buildUnlinkedRingsGeometry({ rng, quality = 'mid', radius = 0.24, segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  const ampA = rng ? 0.08 + rng() * 0.1 : 0.1;
  const ampB = rng ? 0.08 + rng() * 0.1 : 0.1;
  const R = 1.0;

  const curveA = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: 3, amp: ampA });
  const offsetX = rng ? 2.2 + rng() * 0.5 : 2.5;
  const curveB = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(offsetX, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: 3, amp: ampB });

  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const ropeColor = FIXED_ROPE_COLOR;
  {
    const check = checkSelfIntersection(curveA, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }
  }
  {
    const check = checkSelfIntersection(curveB, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }
  }

  const aMesh = createRopeMesh(curveA, { radius: tubeRadius, color: ropeColor, closed: true, tubularSegments, radialSegments });
  const bMesh = createRopeMesh(curveB, { radius: tubeRadius, color: ropeColor, closed: true, tubularSegments, radialSegments });
  const a = aMesh.geometry;
  const b = bMesh.geometry;
  if (aMesh.material) aMesh.material.dispose();
  if (bMesh.material) bMesh.material.dispose();

  const colorsA = new Float32Array(a.attributes.position.count * 3);
  for (let i = 0; i < a.attributes.position.count; i++) { colorsA[i*3] = 1.0; colorsA[i*3+1] = 0.95; colorsA[i*3+2] = 0.95; }
  a.setAttribute('color', new THREE.BufferAttribute(colorsA, 3));

  const colorsB = new Float32Array(b.attributes.position.count * 3);
  for (let i = 0; i < b.attributes.position.count; i++) { colorsB[i*3] = 0.9; colorsB[i*3+1] = 0.85; colorsB[i*3+2] = 0.85; }
  b.setAttribute('color', new THREE.BufferAttribute(colorsB, 3));

  const merged = mergeBufferGeometries([a, b]);
  a.dispose(); b.dispose();
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.35 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

// Build Chain
function randomIntInclusive(rng, a, b) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return lo + Math.floor((rng ? rng() : Math.random()) * (hi - lo + 1));
}
function randomChoice(rng, arr) {
  return arr[Math.min(arr.length - 1, Math.floor((rng ? rng() : Math.random()) * arr.length))];
}
function shuffleInPlace(arr, rng) {
  const r = rng || Math.random;
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
function randomSplitTotal(total, { minPart = 2, maxPart = 4, maxParts = 4 } = {}, rng) {
  const r = rng || Math.random;
  const T = Math.max(1, Math.floor(total));
  const minP = Math.max(1, Math.floor(minPart));
  const maxP = Math.max(minP, Math.floor(maxPart));
  const maxK = Math.max(1, Math.floor(maxParts));

  // Decide number of parts (at least 1; prefer 2+ when possible)
  const maxPossibleParts = Math.min(maxK, Math.max(1, Math.floor(T / minP)));
  const k = T >= minP * 2 ? randomIntInclusive(r, 2, maxPossibleParts) : 1;

  const parts = [];
  let remaining = T;
  for (let i = 0; i < k; i++) {
    const partsLeft = k - i - 1;
    const minRemainForRest = partsLeft * minP;
    const maxThis = Math.min(maxP, remaining - minRemainForRest);
    const minThis = Math.max(1, Math.min(minP, maxThis));
    const size = i === k - 1 ? remaining : (minThis + Math.floor(r() * (maxThis - minThis + 1)));
    parts.push(size);
    remaining -= size;
  }

  // If we were forced into 1 part, still return a single segment
  if (parts.reduce((s, x) => s + x, 0) !== T) return [T];
  return parts;
}

function generateDiverseChainLayout({
  rng,
  numLinks,
  R,
  tubeRadius,
  effectiveStep,
  linkOffsetY,
} = {}) {
  const r = rng || Math.random;
  const n = Math.max(2, Math.floor(numLinks || 4));

  // === 1) Random split into segments ===
  const parts = randomSplitTotal(n, { minPart: 2, maxPart: 4, maxParts: 4 }, r);
  const segments = [];
  let cursor = 0;
  for (const sz of parts) {
    const nodes = [];
    for (let i = 0; i < sz; i++) nodes.push(cursor++);
    segments.push({ nodes, size: sz });
  }

  // === 2) Segment internal pattern (graph) ===
  const allEdges = [];
  const addEdge = (a, b) => {
    if (a === b) return;
    const x = Math.min(a, b), y = Math.max(a, b);
    allEdges.push([x, y]);
  };

  for (const seg of segments) {
    const nodes = seg.nodes.slice();
    shuffleInPlace(nodes, r);
    const sz = nodes.length;
    const pattern = randomChoice(r, ['linear', 'linear', 'branch', 'loop_back']); // weight linear a bit
    seg.pattern = pattern;

    if (pattern === 'loop_back' && sz >= 3) {
      // cycle
      for (let i = 0; i < sz; i++) addEdge(nodes[i], nodes[(i + 1) % sz]);
    } else if (pattern === 'branch' && sz >= 4) {
      // small Y/tree inside segment
      const hub = nodes[0];
      addEdge(hub, nodes[1]);
      addEdge(hub, nodes[2]);
      // chain the rest off one branch to keep graph sane
      let prev = nodes[1];
      for (let i = 3; i < sz; i++) { addEdge(prev, nodes[i]); prev = nodes[i]; }
    } else {
      // linear
      for (let i = 0; i < sz - 1; i++) addEdge(nodes[i], nodes[i + 1]);
    }
  }

  // === 3) Connect segments with diverse topology ===
  if (segments.length > 1) {
    const topology = randomChoice(r, ['sequential', 'tree', 'random_graph', 'has_cycle']);
    const segCount = segments.length;

    const pickNodeFromSeg = (si) => {
      const nodes = segments[si].nodes;
      return nodes[Math.floor(r() * nodes.length)];
    };

    // Build a spanning tree over segments first (ensures connectivity)
    const parents = new Array(segCount).fill(-1);
    for (let i = 1; i < segCount; i++) {
      const p = topology === 'sequential' ? (i - 1) : Math.floor(r() * i);
      parents[i] = p;
      addEdge(pickNodeFromSeg(i), pickNodeFromSeg(p));
    }

    // Add extra inter-segment connections (1..3)
    const extra = randomIntInclusive(r, 1, Math.min(3, segCount));
    const addRandomInterEdge = () => {
      let a = Math.floor(r() * segCount);
      let b = Math.floor(r() * segCount);
      if (a === b) b = (b + 1) % segCount;
      addEdge(pickNodeFromSeg(a), pickNodeFromSeg(b));
    };

    if (topology === 'random_graph') {
      for (let i = 0; i < extra; i++) addRandomInterEdge();
    } else if (topology === 'has_cycle') {
      // ensure at least one cycle
      addRandomInterEdge();
      if (r() < 0.6) for (let i = 0; i < extra; i++) addRandomInterEdge();
    } else if (topology === 'tree') {
      // tree: maybe add 0-1 extra for mild redundancy
      if (r() < 0.35) addRandomInterEdge();
    }
  }

  // Dedup edges
  const dedup = new Map();
  for (const [a, b] of allEdges) dedup.set(`${a},${b}`, [a, b]);
  const edges = Array.from(dedup.values());

  // === 4) Embed graph into 3D: assign center + normal per ring ===
  const centers = Array.from({ length: n }, () => new THREE.Vector3());
  const normals = Array.from({ length: n }, () => new THREE.Vector3(0, 0, 1));
  const placed = new Array(n).fill(false);
  const adj = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) { adj[a].push(b); adj[b].push(a); }

  const basisAxes = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0, -1),
  ];
  const pickPerpAxis = (nrm) => {
    const nn = nrm.clone().normalize();
    const candidates = basisAxes
      .map(a => ({ a, d: Math.abs(a.dot(nn)) }))
      .sort((p, q) => p.d - q.d)
      .filter(p => p.d < 0.35);
    return (candidates.length ? randomChoice(r, candidates).a : randomChoice(r, basisAxes)).clone().normalize();
  };

  const linkSepBase = Math.max(0.9 * R, Math.abs(effectiveStep) || (1.2 * R));
  const jitter = 0.18 * R;
  const zJitter = 0.12 * R;
  const offsetMag = Math.max(0.0, Math.min(2.5 * R, Math.abs(linkOffsetY))) * 0.15;

  // Root
  centers[0].set(0, 0, 0);
  normals[0].copy(new THREE.Vector3(0, 0, 1));
  placed[0] = true;

  const queue = [0];
  while (queue.length) {
    const u = queue.shift();
    const nu = normals[u].clone().normalize();
    for (const v of adj[u]) {
      if (placed[v]) continue;

      const nv = pickPerpAxis(nu);
      let t = new THREE.Vector3().crossVectors(nu, nv);
      if (t.lengthSq() < 1e-6) t = pickPerpAxis(nu).cross(nu);
      t.normalize();

      // Main placement along t, with small extra offsets so it doesn't look like a straight chain
      const sep = linkSepBase * (0.85 + 0.35 * r());
      const c = centers[u].clone()
        .addScaledVector(t, sep)
        .addScaledVector(nu, (r() - 0.5) * zJitter)
        .addScaledVector(nv, (r() - 0.5) * zJitter);

      // Borrow UI "linkOffsetY" as a subtle extra offset along nu to preserve intuitive control
      c.addScaledVector(nu, (r() - 0.5) * offsetMag);

      centers[v].copy(c);
      normals[v].copy(nv);
      placed[v] = true;
      queue.push(v);
    }
  }

  // Any isolated nodes (shouldn't happen, but be safe)
  for (let i = 0; i < n; i++) {
    if (placed[i]) continue;
    centers[i].set((r() - 0.5) * 2.0, (r() - 0.5) * 2.0, (r() - 0.5) * 2.0);
    normals[i].copy(pickPerpAxis(new THREE.Vector3(0, 0, 1)));
  }

  // === 5) Repulsion pass to reduce ugly overlaps ===
  const minCenterDist = Math.max(0.85 * R, R + tubeRadius * 2.2);
  for (let iter = 0; iter < 8; iter++) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = centers[i].clone().sub(centers[j]);
        const dist = d.length();
        if (dist < 1e-6) {
          centers[i].x += (r() - 0.5) * 0.01;
          centers[j].y += (r() - 0.5) * 0.01;
          continue;
        }
        const target = minCenterDist * (0.95 + 0.25 * r());
        if (dist < target) {
          const push = (target - dist) / target;
          d.multiplyScalar((push * 0.55));
          centers[i].add(d);
          centers[j].sub(d);
        }
      }
    }
  }

  // Add final gentle jitter so different segments don't look grid-aligned
  for (let i = 0; i < n; i++) {
    centers[i].add(new THREE.Vector3((r() - 0.5) * jitter, (r() - 0.5) * jitter, (r() - 0.5) * jitter));
  }

  return { centers, normals };
}

function buildChainGeometry({ rng, quality = 'mid', radius = 0.24, segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
  
  const parseVal = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : fallback;
  };
  const parseIntVal = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = parseInt(el.value);
    return Number.isFinite(v) ? v : fallback;
  };

  const R = parseVal('chainR', 1.5);
  const numLinks = parseIntVal('chainNumLinks', rng ? 3 + Math.floor(rng() * 4) : 4);
  const linkOffsetY = parseVal('chainOffsetY', -2.0);
  const chainStep = parseVal('chainSpacing', 0) * R;
  const effectiveStep = Math.abs(chainStep) < 0.01 ? Math.abs(linkOffsetY) : chainStep;

  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const geoms = [];

  const { centers, normals } = generateDiverseChainLayout({
    rng,
    numLinks,
    R,
    tubeRadius,
    effectiveStep,
    linkOffsetY,
  });

  for (let i = 0; i < numLinks; i++) {
    const center = centers[i];
    const normal = normals[i];

    // Per-ring variation
    const ringR = rng ? (R * (0.92 + rng() * 0.16)) : R;
    const amp = rng ? 0.035 + rng() * 0.06 : 0.05;
    const waves = rng ? 2 + Math.floor(rng() * 5) : 3;
    const phase = rng ? rng() * Math.PI * 2 : 0;

    const curve = new PlanarWobbleCircleCurve({ radius: ringR, center, normal, waves, amp, phase });

    const check = checkSelfIntersection(curve, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }

    const ropeColor = FIXED_ROPE_COLOR;
    const mesh = createRopeMesh(curve, { radius: tubeRadius, color: ropeColor, closed: true, tubularSegments, radialSegments });
    const g = mesh.geometry;
    if (mesh.material) mesh.material.dispose();

    applyRainbowGradient(g, radialSegments, i / Math.max(1, numLinks));
    geoms.push(g);
  }

  const merged = mergeBufferGeometries(geoms);
  geoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.8 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

// Build Borromean Rings
function buildBorromeanRingsGeometry({ rng, quality = 'mid', radius = 0.24, segments = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);

  const parseVal = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : fallback;
  };

  const R = parseVal('borromeanR', 1.5);
  const ratio = parseVal('borromeanRatio', 1.6);
  const yellowY = parseVal('borromeanYellowY', 0.70);
  const blueY = parseVal('borromeanBlueY', 0);

  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const a = R * ratio, b = R;

  const ringColors = [
    new THREE.Color(0.95, 0.25, 0.25),
    new THREE.Color(0.95, 0.85, 0.15),
    new THREE.Color(0.25, 0.45, 0.95),
  ];

  class EllipseCurve3D extends THREE.Curve {
    constructor({ a, b, center, xAxis, yAxis }) {
      super();
      this.a = a; this.b = b;
      this.center = center;
      this.xAxis = xAxis.clone().normalize();
      this.yAxis = yAxis.clone().normalize();
    }
    getPoint(t, optionalTarget = new THREE.Vector3()) {
      const angle = t * Math.PI * 2;
      return optionalTarget.set(0, 0, 0)
        .addScaledVector(this.center, 1)
        .addScaledVector(this.xAxis, this.a * Math.cos(angle))
        .addScaledVector(this.yAxis, this.b * Math.sin(angle));
    }
  }

  const configs = [
    { center: new THREE.Vector3(0, 0, 0), xAxis: new THREE.Vector3(1, 0, 0), yAxis: new THREE.Vector3(0, 1, 0) },
    { center: new THREE.Vector3(0, yellowY, 0), xAxis: new THREE.Vector3(0, 1, 0), yAxis: new THREE.Vector3(0, 0, 1) },
    { center: new THREE.Vector3(0, blueY, 0), xAxis: new THREE.Vector3(1, 0, 0), yAxis: new THREE.Vector3(0, 0, 1) },
  ];

  // --- Step 1: Sample all three ring centerlines ---
  const N = 360;
  const ringPoints = [];
  for (let ri = 0; ri < 3; ri++) {
    const curve = new EllipseCurve3D({ a, b, ...configs[ri] });
    const pts = [];
    for (let k = 0; k < N; k++) pts.push(curve.getPoint(k / N));
    ringPoints.push(pts);
  }

  // --- Step 2: Find crossings between each pair and apply over/under bumps ---
  // Borromean topology: each pair is pairwise unlinked (alternating over/under),
  // but the triple is non-trivially linked because crossings interleave on each ring.
  const bumpMag = tubeRadius * 5;         // displacement per side at crossing peak
  const sigma = N * 0.05;                 // Gaussian width in index units (~18 pts)
  const crossingThreshold = tubeRadius * 20; // only bump crossings within this distance

  const circDist = (ia, ib) => { const d = Math.abs(ia - ib); return Math.min(d, N - d); };

  const findCrossings = (ptsA, ptsB) => {
    // For each point on A, find closest distance to B's polyline
    const cDist = new Float64Array(N);
    const cIdx  = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      let minD = Infinity, minJ = 0;
      for (let j = 0; j < N; j++) {
        const d = ptsA[i].distanceTo(ptsB[j]);
        if (d < minD) { minD = d; minJ = j; }
      }
      cDist[i] = minD;
      cIdx[i]  = minJ;
    }
    // Collect local minima as crossing candidates
    const results = [];
    for (let i = 0; i < N; i++) {
      const prev = (i - 1 + N) % N, next = (i + 1) % N;
      if (cDist[i] > cDist[prev] || cDist[i] > cDist[next]) continue;
      if (cDist[i] > crossingThreshold) continue;
      // Skip if too close to an already-found crossing on ring A
      if (results.some(r => circDist(i, r.idxA) < N * 0.1)) continue;
      const j = cIdx[i];
      // Crossing normal = cross product of the two tangent vectors
      const tA = new THREE.Vector3().subVectors(ptsA[(i + 1) % N], ptsA[(i - 1 + N) % N]).normalize();
      const tB = new THREE.Vector3().subVectors(ptsB[(j + 1) % N], ptsB[(j - 1 + N) % N]).normalize();
      const normal = new THREE.Vector3().crossVectors(tA, tB);
      if (normal.length() < 1e-6) continue; // tangents parallel, not a real crossing
      normal.normalize();
      results.push({ idxA: i, idxB: j, normal, dist: cDist[i] });
    }
    return results;
  };

  const pairs = [[0, 1], [0, 2], [1, 2]];
  for (const [ri, rj] of pairs) {
    const crossings = findCrossings(ringPoints[ri], ringPoints[rj]);
    crossings.forEach((cr, idx) => {
      // Alternate over/under at each crossing → linking number = 0 (pairwise unlinked)
      const sign = (idx % 2 === 0) ? 1 : -1;
      for (let k = 0; k < N; k++) {
        const dA = circDist(k, cr.idxA);
        const wA = Math.exp(-(dA * dA) / (2 * sigma * sigma));
        ringPoints[ri][k].addScaledVector(cr.normal, sign * bumpMag * wA);

        const dB = circDist(k, cr.idxB);
        const wB = Math.exp(-(dB * dB) / (2 * sigma * sigma));
        ringPoints[rj][k].addScaledVector(cr.normal, -sign * bumpMag * wB);
      }
    });
  }

  // --- Step 3: Build tube geometries from displaced centerlines ---
  const geoms = [];
  for (let i = 0; i < 3; i++) {
    const smoothCurve = new THREE.CatmullRomCurve3(ringPoints[i], true, 'centripetal');
    const g = new THREE.TubeGeometry(smoothCurve, tubularSegments, tubeRadius, radialSegments, true);
    applyRainbowGradient(g, radialSegments, i / 3);
    geoms.push(g);
  }

  const merged = mergeBufferGeometries(geoms);
  geoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.8 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

// ============= Main Build Function =============

// Apply random transform to geometry
function applyRandomTransform(geometry, rng) {
  const rx = rng() * Math.PI * 2;
  const ry = rng() * Math.PI * 2;
  const rz = rng() * Math.PI * 2;
  const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz));
  geometry.applyMatrix4(rotMatrix);
  if (geometry.attributes.normal) geometry.normalizeNormals();
  geometry.center();
}

// Apply random rotation only (no anisotropic scale) - better for "physical rope" look
function applyRandomRotation(geometry, rng) {
  const rx = rng() * Math.PI * 2;
  const ry = rng() * Math.PI * 2;
  const rz = rng() * Math.PI * 2;
  const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz));
  geometry.applyMatrix4(rotMatrix);
  geometry.computeVertexNormals();
  geometry.center();
}

function buildGeometryForPreset(presetId, {
  rng,
  quality = 'mid',
  radius = 0.24,
  applyDeform = true,
  deformStrength = 0.3,
  slackness = 0.4,
  segments = null,
} = {}) {
  const p = PRESETS[presetId];
  if (!p || p.kind === 'all') return null;

  // Dataset-visible presets use the same geometry builder as batch generation.
  // This keeps gallery previews and generated benchmark images on one rendering path.
  if (KNOT_TYPE_REGISTRY[presetId]) {
    return buildDatasetGeometryForKnotType(presetId, {
      rng,
      quality,
      radius: 0.07,
      deformStrength: applyDeform ? deformStrength : 0,
      slackness: applyDeform ? slackness : 0,
      numLinks: presetId === 'chain'
        ? parsePositiveInt(document.getElementById('chainNumLinks')?.value, 4, { min: 2, max: 8 })
        : undefined,
    });
  }

  // Fixed thin rope (removed UI thickness control)
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const buildProcessedTube = (rawCurve, { closed = true, targetOuterRadius = 1.35 } = {}) => {
    const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
    const curve = processCenterline(rawCurve, {
      rng: rng || null,
      deformStrength: applyDeform ? deformStrength : 0,
      slackness: applyDeform ? slackness : 0,
      tubeRadius,
      sampleN: Math.max(240, tubularSegments),
      closed,
      knotType: presetId,
    });
    const geom = new THREE.TubeGeometry(curve, tubularSegments, tubeRadius, radialSegments, closed);
    geom.center();
    geom.computeBoundingSphere();
    const sNorm = geom.boundingSphere?.radius > 1e-6 ? (targetOuterRadius / geom.boundingSphere.radius) : 1.0;
    geom.scale(sNorm, sNorm, sNorm);
    if (geom.attributes.normal) geom.normalizeNormals();
    return geom;
  };

  if (p.kind === 'hopfReal') {
    const geom = buildRealHopfLinkGeometry({ rng, quality, radius: tubeRadius, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'hopfUnlinked') {
    const geom = buildUnlinkedRingsGeometry({ rng, quality, radius: tubeRadius, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'chain') {
    const geom = buildChainGeometry({ rng, quality, radius: tubeRadius, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'borromean') {
    const geom = buildBorromeanRingsGeometry({ rng, quality, radius: tubeRadius, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  // --- New types: loose, link groups, mixed ---
  if (p.kind === 'looseKnot') {
    const baseCurveFactory = () => {
      if (p.baseCurveKind === 'torus_2_5') return new TorusKnotCurve({ p: 2, q: 5, R: 1.0, r: 0.38 });
      return new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.38 }); // fallback trefoil
    };
    return buildGenericLooseKnotGeometry({
      rng, quality, baseCurveFactory,
      deformStrength: applyDeform ? deformStrength : 0,
      slackness: applyDeform ? Math.max(slackness, 0.50) : 0,
      segments,
    });
  }
  if (p.kind === 'doubleHopf') {
    const geom = buildDoubleHopfGeometry({ rng, quality, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'linkCluster') {
    const geom = buildLinkClusterGeometry({ rng, quality, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'hopfPlusFree') {
    const geom = buildHopfPlusFreeGeometry({ rng, quality, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'chainPlusFree') {
    const geom = buildChainPlusFreeGeometry({ rng, quality, segments });
    if (applyDeform && rng) applyRandomTransform(geom, rng);
    return geom;
  }
  if (p.kind === 'occludedKnot') {
    // Occluded trefoil: heavily deformed trefoil that obscures crossing structure
    const rawCurve = new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.55 });
    return buildProcessedTube(rawCurve, { closed: true, targetOuterRadius: 1.35 });
  }

  if (p.kind === 'kinky') {
    const baseK = parseInt(document.getElementById('kinkCount')?.value) || 4;
    const baseAmp = parseFloat(document.getElementById('kinkAmplitude')?.value) || 0.25;
    return buildProcessedTube(new KinkyUnknotCurve({
      k: rng ? Math.max(2, baseK + Math.floor((rng() - 0.5) * 4)) : baseK,
      baseRadius: 1.0,
      kinkAmplitude: rng ? baseAmp * (0.7 + rng() * 0.6) : baseAmp,
      seed: rng ? Math.floor(rng() * 100000) : 12345,
    }), { closed: true, targetOuterRadius: 1.35 });
  }
  if (p.kind === 'openLooseKnot') {
    return buildLooseOpenKnotGeometry({
      rng,
      quality,
      radius: tubeRadius,
      deformStrength: applyDeform ? deformStrength : 0,
      slackness: applyDeform ? slackness : 0,
      segments,
    });
  }

  if (p.kind === 'torus') {
    const torusR = p.R || 1.0;
    const torusMinorR = p.r || Math.max(0.25, 0.45 - p.q * 0.02);
    const rawCurve = new TorusKnotCurve({ p: p.p, q: p.q, R: torusR, r: torusMinorR });
    return buildProcessedTube(rawCurve, { closed: true, targetOuterRadius: 1.35 });
  }

  if (p.kind === 'torusRandom') {
    let pp, qq;
    do {
      pp = 2 + Math.floor((rng?.() || Math.random()) * 4);
      qq = 3 + Math.floor((rng?.() || Math.random()) * 8);
    } while (gcd(pp, qq) !== 1 || pp >= qq);
    const torusMinorR = Math.max(0.25, 0.45 - qq * 0.015);
    const rawCurve = new TorusKnotCurve({ p: pp, q: qq, R: 1.0, r: torusMinorR });
    const geom = buildProcessedTube(rawCurve, { closed: true, targetOuterRadius: 1.35 });
    if (applyDeform && rng) applyRandomRotation(geom, rng);
    return geom;
  }

  if (p.kind === 'curveExtras') {
    const hasCurve = Curves && typeof Curves[p.extrasName] === 'function';
    const rawCurve = hasCurve ? new Curves[p.extrasName]() : p.fallback();
    const geom = buildProcessedTube(rawCurve, { closed: true, targetOuterRadius: 1.35 });
    if (applyDeform && rng) applyRandomRotation(geom, rng);
    return geom;
  }

  if (p.kind === 'curve') {
    const rawCurve = p.make(rng);
    const geom = buildProcessedTube(rawCurve, { closed: true, targetOuterRadius: 1.35 });
    if (applyDeform && rng) applyRandomRotation(geom, rng);
    return geom;
  }

  if (p.kind === 'spiralLoop') {
    // Read UI values if available, or use random values
    const parseUI = (id, fallback) => {
      const el = document.getElementById(id);
      if (el) {
        const v = parseFloat(el.value);
        return Number.isFinite(v) ? v : fallback;
      }
      return fallback;
    };
    const parseUIInt = (id, fallback) => {
      const el = document.getElementById(id);
      if (el) {
        const v = parseInt(el.value);
        return Number.isFinite(v) ? v : fallback;
      }
      return fallback;
    };
    
    const baseTurns = parseUIInt('spiralTurns', 3);
    const basePitch = parseUI('spiralPitch', 0.15);
    const baseGap = parseUI('spiralGap', 0.25);
    
    // Add some variation if rng available
    const turns = rng ? Math.max(1, baseTurns + Math.floor((rng() - 0.5) * 2)) : baseTurns;
    const pitch = rng ? basePitch * (0.7 + rng() * 0.6) : basePitch;
    const radialGap = rng ? baseGap * (0.7 + rng() * 0.6) : baseGap;

    // Physical rope: enforce minimum clearance based on tubeRadius
    const makeCurve = () => new SpiralLoopCurve({
      turns,
      pitch,
      radialGap,
      tubeRadius,
    });

    // For Spiral Loop, we do NOT scale or deform it after creation, as it ruins the physical gap.
    const { tubularSegments, radialSegments } = tubeQualityParams(quality, segments);
    const curve = makeCurve();
    const check = checkSelfIntersection(curve, 0.05);
    if (check.hasIntersection) {
      // eslint-disable-next-line no-console
      console.warn(`Self-intersection detected! minDist=${check.minDist}`);
      // optional: auto-scale curve or regenerate
    }

    const ropeColor = FIXED_ROPE_COLOR;
    const mesh = createRopeMesh(curve, { radius: DISPLAY_ROPE_RADIUS, color: ropeColor, closed: true, tubularSegments: Math.max(280, tubularSegments), radialSegments });
    const geom = mesh.geometry;
    if (mesh.material) mesh.material.dispose();
    geom.center();
    return geom;
  }

  if (p.kind === 'benchmark') {
    const level = p.level;
    const easyPresets = ['unknot', 'trefoil', 'figure8'];
    const mediumPresets = ['torus_2_5', 'twisted_ring', 'spiral_disk'];
    const hardPresets = ['torus_2_7', 'torus_2_9', 'kinky_unknot', 'torus_3_4', 'torus_3_5'];
    
    let pool;
    if (level === 0) pool = easyPresets;
    else if (level === 1) pool = mediumPresets;
    else pool = hardPresets;
    
    const chosenId = pick(rng || Math.random, pool);
    return buildGeometryForPreset(chosenId, { rng, quality, radius, applyDeform, deformStrength, slackness, segments });
  }

  if (p.kind === 'benchmarkMix') {
    const easyPct = parseNumber(document.getElementById('easyPct')?.value, 33, { min: 0, max: 100 });
    const mediumPct = parseNumber(document.getElementById('mediumPct')?.value, 34, { min: 0, max: 100 });
    const total = easyPct + mediumPct + 100 - easyPct - mediumPct;
    const r = (rng?.() || Math.random()) * total;
    
    let level;
    if (r < easyPct) level = 0;
    else if (r < easyPct + mediumPct) level = 1;
    else level = 2;
    
    return buildGeometryForPreset(level === 0 ? 'benchmark_easy' : (level === 1 ? 'benchmark_medium' : 'benchmark_hard'), { rng, quality, radius, applyDeform, deformStrength, slackness, segments });
  }

  // Fallback
  return estimateAndNormalizeTube({ makeCurve: () => new CircleCurve({ radius: 1.0 }), closed: true, quality, radius: DISPLAY_ROPE_RADIUS, targetOuterRadius: 1.35, segments });
}

// ============= Three.js Scene =============

const viewEl = document.getElementById('view');
const statusEl = document.getElementById('status');

let scene, camera, renderer, controls;
let root = new THREE.Group();
let current = { meshes: [], geometries: [] };
let regenSeq = 0;

function setStatus({ title, three, presetName, count }) {
  statusEl.innerHTML = `
    <div><b>Status</b>: ${title || '-'}</div>
    <div><b>Three.js</b>: ${three || '-'}</div>
    <div><b>Preset</b>: ${presetName || '-'}</div>
    <div><b>Instances</b>: ${typeof count === 'number' ? count : '-'}</div>
  `;
}

function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a2236);

  const w = viewEl.clientWidth;
  const h = viewEl.clientHeight;
  camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 500);
  camera.position.set(0, 3.2, 6.2);

  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewEl.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 0, 0);

  // Lighting tuned for clearer depth/occlusion cues.
  scene.add(new THREE.AmbientLight(0xffffff, 0.46));
  const hemi = new THREE.HemisphereLight(0xffffff, 0x2a335a, 0.18);
  scene.add(hemi);
  
  const dir = new THREE.DirectionalLight(0xffffff, 1.95);
  dir.position.set(10, 18, 10);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 2048;
  dir.shadow.mapSize.height = 2048;
  dir.shadow.camera.near = 0.1;
  dir.shadow.camera.far = 80;
  dir.shadow.camera.left = -18;
  dir.shadow.camera.right = 18;
  dir.shadow.camera.top = 18;
  dir.shadow.camera.bottom = -18;
  dir.shadow.bias = -0.0006;
  scene.add(dir);

  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-10, 5, -10);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xffffff, 0.42);
  rim.position.set(0, 6, -14);
  scene.add(rim);

  const shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(220, 220),
    new THREE.ShadowMaterial({ opacity: 0.2 }),
  );
  shadowPlane.rotation.x = -Math.PI * 0.5;
  shadowPlane.position.y = -1.22;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  const grid = new THREE.GridHelper(120, 60, 0x2a335a, 0x1a2040);
  grid.position.y = -1.2;
  if (Array.isArray(grid.material)) {
    grid.material.forEach((m) => {
      m.transparent = true;
      m.opacity = 0.35;
    });
  } else if (grid.material) {
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
  }
  scene.add(grid);

  scene.add(root);

  window.addEventListener('resize', () => {
    const ww = viewEl.clientWidth;
    const hh = viewEl.clientHeight;
    camera.aspect = ww / hh;
    camera.updateProjectionMatrix();
    renderer.setSize(ww, hh);
  });

  setStatus({ title: 'Initialized', three: `${THREE.REVISION}`, presetName: '-', count: 0 });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function disposeCurrent() {
  for (const m of current.meshes) {
    root.remove(m);
    if (m.material) m.material.dispose();
  }
  for (const g of current.geometries) g.dispose();
  current = { meshes: [], geometries: [] };
}

function randomBright(rng) {
  const h = rng();
  const s = 0.72 + 0.22 * rng();
  const l = 0.62 + 0.18 * rng();
  return new THREE.Color().setHSL(h, s, l);
}

function placeMatrix(m, x, y, z, rx, ry, rz, s) {
  const pos = new THREE.Vector3(x, y, z);
  const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz));
  const scl = new THREE.Vector3(s, s, s);
  m.compose(pos, quat, scl);
}

function computeSpacingFromGeometry({ geometry, layout, globalScale }) {
  geometry.computeBoundingSphere();
  const r = (geometry.boundingSphere?.radius || 1) * globalScale;
  const maxR = r * 1.25;
  const minGrid = layout === 'field' ? 10.0 : 7.0;
  const base = 2 * maxR * 1.55 + 2.0;
  const baseSpacing = Math.max(minGrid, base);
  const jitter = layout === 'jitter' ? baseSpacing * 0.22 : (layout === 'field' ? baseSpacing * 0.55 : 0.0);
  return { baseSpacing, jitter };
}

function layoutPosition({ i, count, cols, rng, baseSpacing, jitter }) {
  const colsEff = Math.max(1, Math.min(cols, count));
  const row = Math.floor(i / colsEff);
  const col = i % colsEff;
  const rowsEff = Math.ceil(count / colsEff);
  const x = (col - (colsEff - 1) * 0.5) * baseSpacing + (rng() - 0.5) * jitter;
  const z = (row - (rowsEff - 1) * 0.5) * baseSpacing + (rng() - 0.5) * jitter;
  return { x, z };
}

function buildInstancedMesh(geometry, { count, cols, rng, layout, globalScale, instanceOffset = 0, totalCount = 0, spacing, canonicalOrientation = false }) {
  if (!geometry.attributes.color) {
    // Apply rainbow gradient instead of flat white — matches batch renderer
    const radialSeg = geometry.parameters?.radialSegments || 18;
    applyRainbowGradient(geometry, radialSeg);
  }

  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.5,
    metalness: 0.04,
    vertexColors: true,
    emissive: new THREE.Color(0x050505),
    emissiveIntensity: 0.04,
  });
  
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);

  const m = new THREE.Matrix4();
  const tmpColor = new THREE.Color();
  const finalTotal = totalCount || count;
  const { baseSpacing, jitter } = spacing || computeSpacingFromGeometry({ geometry, layout, globalScale });

  for (let i = 0; i < count; i++) {
    const globalIdx = instanceOffset + i;
    const { x, z } = finalTotal === 1 ? { x: 0, z: 0 } : layoutPosition({ i: globalIdx, count: finalTotal, cols, rng, baseSpacing, jitter });
    
    const rx = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const ry = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const rz = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const s = globalScale * (canonicalOrientation ? 1.0 : (0.8 + rng() * 0.4));
    
    placeMatrix(m, x, 0, z, rx, ry, rz, s);
    mesh.setMatrixAt(i, m);
    // Use white instance color so rainbow vertex colors show through
    tmpColor.setRGB(1, 1, 1);
    mesh.setColorAt(i, tmpColor);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  return mesh;
}

function buildInstancedMeshesLOD(geometryHigh, geometryLow, {
  count,
  cols,
  rng,
  layout,
  globalScale,
  instanceOffset = 0,
  totalCount = 0,
  spacing,
  // LOD rule (world XZ distance from origin)
  nearRadiusFactor = 1.75, // threshold = nearRadiusFactor * baseSpacing
  canonicalOrientation = false,
} = {}) {
  // Ensure both geometries have vertex colors (white), so instanceColor works as expected.
  for (const g of [geometryHigh, geometryLow]) {
    if (g && !g.attributes.color) {
      const colors = new Float32Array(g.attributes.position.count * 3).fill(1.0);
      g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
  }

  const finalTotal = totalCount || count;
  const { baseSpacing, jitter } = spacing || computeSpacingFromGeometry({ geometry: geometryHigh || geometryLow, layout, globalScale });
  const nearR = Math.max(0, Number(nearRadiusFactor) || 1.75) * baseSpacing;

  const matricesHigh = [];
  const colorsHigh = [];
  const matricesLow = [];
  const colorsLow = [];

  const m = new THREE.Matrix4();
  const tmpColor = new THREE.Color();

  for (let i = 0; i < count; i++) {
    const globalIdx = instanceOffset + i;
    const { x, z } = finalTotal === 1 ? { x: 0, z: 0 } : layoutPosition({ i: globalIdx, count: finalTotal, cols, rng, baseSpacing, jitter });

    const rx = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const ry = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const rz = canonicalOrientation ? 0 : rng() * Math.PI * 2;
    const s = globalScale * (canonicalOrientation ? 1.0 : (0.8 + rng() * 0.4));

    placeMatrix(m, x, 0, z, rx, ry, rz, s);
    tmpColor.copy(randomBright(rng));

    const r = Math.sqrt(x * x + z * z);
    if (r <= nearR) {
      matricesHigh.push(m.clone());
      colorsHigh.push(tmpColor.clone());
    } else {
      matricesLow.push(m.clone());
      colorsLow.push(tmpColor.clone());
    }
  }

  const makeMaterial = () => new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.5,
    metalness: 0.04,
    vertexColors: true,
    emissive: new THREE.Color(0x050505),
    emissiveIntensity: 0.04,
  });

  const meshes = [];

  if (matricesHigh.length) {
    const mat = makeMaterial();
    const meshH = new THREE.InstancedMesh(geometryHigh, mat, matricesHigh.length);
    meshH.castShadow = true;
    meshH.receiveShadow = true;
    meshH.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    meshH.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(matricesHigh.length * 3), 3);
    for (let i = 0; i < matricesHigh.length; i++) {
      meshH.setMatrixAt(i, matricesHigh[i]);
      meshH.setColorAt(i, colorsHigh[i]);
    }
    meshH.instanceMatrix.needsUpdate = true;
    meshH.instanceColor.needsUpdate = true;
    meshes.push(meshH);
  }

  if (matricesLow.length) {
    const mat = makeMaterial();
    const meshL = new THREE.InstancedMesh(geometryLow, mat, matricesLow.length);
    meshL.castShadow = true;
    meshL.receiveShadow = true;
    meshL.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    meshL.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(matricesLow.length * 3), 3);
    for (let i = 0; i < matricesLow.length; i++) {
      meshL.setMatrixAt(i, matricesLow[i]);
      meshL.setColorAt(i, colorsLow[i]);
    }
    meshL.instanceMatrix.needsUpdate = true;
    meshL.instanceColor.needsUpdate = true;
    meshes.push(meshL);
  }

  return { meshes, baseSpacing, jitter };
}

function fitCameraToContent({ geometry, count, cols, layout, globalScale, spacing }) {
  const colsEff = Math.max(1, Math.min(cols, count));
  const rowsEff = Math.max(1, Math.ceil(count / colsEff));
  const { baseSpacing } = spacing || computeSpacingFromGeometry({ geometry, layout, globalScale });

  geometry.computeBoundingSphere();
  const rObj = (geometry.boundingSphere?.radius || 1) * globalScale * 1.05;
  const w = (colsEff - 1) * baseSpacing + 2 * rObj;
  const d = (rowsEff - 1) * baseSpacing + 2 * rObj;
  const rScene = 0.5 * Math.sqrt(w * w + d * d);

  const fov = (camera.fov * Math.PI) / 180;
  const dist = (rScene / Math.tan(fov / 2)) * 1.08;

  controls.target.set(0, 0, 0);
  camera.position.set(0, Math.max(1.8, rScene * 0.65), dist);
  camera.near = Math.max(0.01, dist / 200);
  camera.far = dist * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

function fitRingLikeOpenKnotCamera({ geometry, globalScale }) {
  geometry.computeBoundingSphere();
  const r = Math.max(1.0, (geometry.boundingSphere?.radius || 1.35) * globalScale);
  controls.target.set(0, 0, 0);
  camera.position.set(-r * 4.4, r * 0.8, 0);
  camera.near = 0.01;
  camera.far = Math.max(80, r * 40);
  camera.updateProjectionMatrix();
  controls.update();
}

async function regenerate() {
  const mySeq = ++regenSeq;
  const seedStr = document.getElementById('seed')?.value || 'knot-gallery-v1';
  const rng = makeRng(seedStr);

  const presetId = document.getElementById('preset')?.value || 'trefoil';
  const presetName = PRESETS[presetId]?.name || presetId;

  const count = parsePositiveInt(document.getElementById('count')?.value, 1, { min: 1, max: 500 });
  const cols = parsePositiveInt(document.getElementById('cols')?.value, 4, { min: 1, max: 50 });
  const quality = document.getElementById('quality')?.value || 'mid';
  const layout = document.getElementById('layout')?.value || 'grid';
  const radius = DISPLAY_ROPE_RADIUS;
  const globalScale = parseNumber(document.getElementById('scale')?.value, 1.0, { min: 0.3, max: 3.0 });
  const slackness = parseNumber(document.getElementById('slackness')?.value, 0.4, { min: 0, max: 1 });
  const deformStrength = parseNumber(document.getElementById('deformStrength')?.value, 0.3, { min: 0, max: 1 });

  disposeCurrent();

  try {
    if (presetId === 'all') {
      // Build all dataset-visible types only.
      const allKeys = Object.keys(KNOT_TYPE_REGISTRY);
      const perType = Math.max(1, Math.floor(count / allKeys.length));
      let total = 0;
      
      const firstGeom = buildGeometryForPreset(allKeys[0], { rng, quality, radius, deformStrength, slackness });
      const spacing = firstGeom ? computeSpacingFromGeometry({ geometry: firstGeom, layout: 'grid', globalScale }) : { baseSpacing: 9.0, jitter: 0.0 };
      if (firstGeom) firstGeom.dispose();

      for (const k of allKeys) {
        const n = Math.min(perType, count - total);
        if (n <= 0) break;
        
        const geometry = buildGeometryForPreset(k, { rng, quality, radius, deformStrength, slackness });
        if (!geometry) continue;
        
        const mesh = buildInstancedMesh(geometry, { count: n, cols: Math.min(cols, n), rng, layout: 'grid', globalScale, instanceOffset: total, totalCount: count, spacing, canonicalOrientation: KNOT_TYPE_REGISTRY[k]?.isOpenRope === true });
        root.add(mesh);
        current.meshes.push(mesh);
        current.geometries.push(geometry);
        total += n;
      }

      setStatus({ title: 'Generated (All Types)', three: `${THREE.REVISION}`, presetName: 'All Types', count: total });
      return;
    }

    // Single preset type
    const VARIATION_COUNT = count === 1 ? 1 : Math.min(12, Math.max(3, Math.floor(count / 2)));
    const perVar = Math.ceil(count / VARIATION_COUNT);
    let totalCreated = 0;

    // Spacing should reflect the visible (high) geometry.
    const spacingGeomRng = makeRng(`${seedStr}|${presetId}|spacing`);
    const firstGeom = buildGeometryForPreset(presetId, {
      rng: spacingGeomRng,
      quality,
      radius,
      applyDeform: true,
      deformStrength,
      slackness,
      segments: { tubularSegments: 300, radialSegments: 16 },
    });
    const spacing = computeSpacingFromGeometry({ geometry: firstGeom, layout, globalScale });
    firstGeom.dispose();

    for (let v = 0; v < VARIATION_COUNT; v++) {
      const numForThisVar = Math.min(perVar, count - totalCreated);
      if (numForThisVar <= 0) break;

      const segHigh = { tubularSegments: 300, radialSegments: 16 };
      const segLow = { tubularSegments: 100, radialSegments: 8 };

      // Use the SAME seeded RNG stream for both LOD geometries so shapes match.
      const geomSeed = `${seedStr}|${presetId}|v=${v}`;
      const geometryHigh = buildGeometryForPreset(presetId, {
        rng: makeRng(geomSeed),
        quality,
        radius,
        applyDeform: true,
        deformStrength,
        slackness,
        segments: segHigh,
      });
      const geometryLow = count > 1 ? buildGeometryForPreset(presetId, {
        rng: makeRng(geomSeed),
        quality,
        radius,
        applyDeform: true,
        deformStrength,
        slackness,
        segments: segLow,
      }) : null;
      if (!geometryHigh) continue;

      if (count > 1 && geometryLow) {
        const { meshes } = buildInstancedMeshesLOD(geometryHigh, geometryLow, { count: numForThisVar, cols, rng, layout, globalScale, instanceOffset: totalCreated, totalCount: count, spacing, canonicalOrientation: KNOT_TYPE_REGISTRY[presetId]?.isOpenRope === true });
        for (const m of meshes) {
          root.add(m);
          current.meshes.push(m);
        }
        current.geometries.push(geometryHigh, geometryLow);
      } else {
        const mesh = buildInstancedMesh(geometryHigh, { count: numForThisVar, cols, rng, layout, globalScale, instanceOffset: totalCreated, totalCount: count, spacing, canonicalOrientation: KNOT_TYPE_REGISTRY[presetId]?.isOpenRope === true });
        root.add(mesh);
        current.meshes.push(mesh);
        current.geometries.push(geometryHigh);
      }
      totalCreated += numForThisVar;

      if (v === 0) {
        if (presetId === 'ring_like_open_knot') {
          fitRingLikeOpenKnotCamera({ geometry: geometryHigh, globalScale });
        } else {
          fitCameraToContent({ geometry: geometryHigh, count, cols, layout, globalScale, spacing });
        }
      }
    }

    setStatus({ title: 'Generated', three: `${THREE.REVISION}`, presetName, count });
  } catch (e) {
    console.error(e);
    setStatus({ title: 'Generation failed', three: `${THREE.REVISION}`, presetName: '-', count: 0 });
  }
}

// ============= Export JSON =============

function exportDataset() {
  const seedStr = document.getElementById('seed')?.value || 'knot-gallery-v1';
  const presetId = document.getElementById('preset')?.value || 'trefoil';
  const count = parsePositiveInt(document.getElementById('count')?.value, 1, { min: 1, max: 500 });
  const deformStrength = clamp(parseNumber(document.getElementById('deformStrength')?.value, 0.3, { min: 0, max: 1 }), 0, 1);
  const slackness = clamp(parseNumber(document.getElementById('slackness')?.value, 0.4, { min: 0, max: 1 }), 0, 1);
  const tubeRadius = DISPLAY_ROPE_RADIUS;
  const cameraPosition = camera ? [camera.position.x, camera.position.y, camera.position.z] : [0, 3.2, 6.2];
  const dataset = {
    version: 1,
    createdAt: new Date().toISOString(),
    seed: seedStr,
    preset: presetId,
    presetInfo: PRESETS[presetId] || {},
    count,
    samples: [],
  };

  // Generate sample metadata
  const rng = makeRng(seedStr);
  for (let i = 0; i < count; i++) {
    const presetInfo = PRESETS[presetId] || {};
    const unified = computeSingleClosedLoopDifficulty({
      knotType: presetId,
      deformStrength,
      slackness,
      cameraPosition,
      tubeRadius,
    });

    dataset.samples.push({
      id: `sample_${String(i).padStart(5, '0')}`,
      preset: presetId,
      difficulty: unified?.difficulty || (PRESETS[presetId]?.difficulty || 'unknown'),
      difficulty_score: unified ? Number(unified.difficulty_score.toFixed(3)) : null,
      difficulty_factors: unified?.factors || null,
      crossings: PRESETS[presetId]?.crossings ?? null,
      isUnknot: PRESETS[presetId]?.isUnknot || false,
      isLink: PRESETS[presetId]?.isLink || false,
      isDeceptive: PRESETS[presetId]?.isDeceptive || false,
      slackness,
    });
  }

  const text = JSON.stringify(dataset, null, 2);
  const blob = new Blob([text], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `knot_dataset_${seedStr}_${presetId}_N${count}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function makeCaptureBaseName() {
  const presetId = document.getElementById('preset')?.value || 'trefoil';
  const seedStrRaw = document.getElementById('seed')?.value || 'default';
  const seedStr = String(seedStrRaw).replace(/[^a-zA-Z0-9_-]/g, '_');
  const slackness = clamp(parseNumber(document.getElementById('slackness')?.value, 0.4, { min: 0, max: 1 }), 0, 1);
  return { presetId, seedStr, slackness };
}

function captureCurrentPng() {
  if (!renderer || !scene || !camera) return;
  const { presetId, seedStr, slackness } = makeCaptureBaseName();
  renderer.render(scene, camera);
  const filename = `knot_${presetId}_s${seedStr}_slack${slackness.toFixed(2)}_current.png`;
  downloadDataUrl(renderer.domElement.toDataURL('image/png'), filename);
}

async function batchCapture() {
  if (!renderer || !scene || !camera) return;

  const { presetId, seedStr, slackness } = makeCaptureBaseName();
  const deformStrength = clamp(parseNumber(document.getElementById('deformStrength')?.value, 0.3, { min: 0, max: 1 }), 0, 1);
  const imageMeta = [];

  const oldPos = camera.position.clone();
  const oldTarget = controls ? controls.target.clone() : new THREE.Vector3(0, 0, 0);
  const oldFov = camera.fov;

  try {
    for (const angle of STANDARD_CAMERA_ANGLES) {
      camera.position.set(...angle.pos);
      if (controls) controls.target.set(0, 0, 0);
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      if (controls) controls.update();
      renderer.render(scene, camera);

      const filename = `knot_${presetId}_s${seedStr}_slack${slackness.toFixed(2)}_${angle.name}.png`;
      downloadDataUrl(renderer.domElement.toDataURL('image/png'), filename);
      imageMeta.push({ filename, cameraAngle: angle.name, cameraPos: angle.pos });
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
  } finally {
    camera.position.copy(oldPos);
    camera.fov = oldFov;
    if (controls) controls.target.copy(oldTarget);
    camera.lookAt(oldTarget);
    camera.updateProjectionMatrix();
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  const knotEntry = KNOT_TYPE_REGISTRY?.[presetId] || {};
  const metadata = {
    version: '1.1',
    knotType: presetId,
    topologicalId: knotEntry.topologicalId || null,
    isKnot: !(knotEntry.isUnknot || knotEntry.isLink),
    isUnknot: knotEntry.topologicalId === 'unknot' || false,
    isDeceptive: knotEntry.isDeceptive || false,
    isLoose: slackness > 0.55 && !(knotEntry.isUnknot || knotEntry.isLink),
    crossingNumber: knotEntry.crossingNumber ?? null,
    slackness,
    deformStrength,
    seed: seedStr,
    difficulty: getBucketTopology(presetId),
    bucket_topology: getBucketTopology(presetId),
    bucket_saliency: getBucketSaliency(slackness),
    trap_type: getTrapType(presetId, slackness),
    images: imageMeta,
    groundTruth: {
      task_knotted_or_not: (knotEntry.isUnknot || knotEntry.isLink) ? 'unknotted' : 'knotted',
      task_knot_family: knotEntry.family || 'unknown',
    },
  };

  const metaBlob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
  const metaUrl = URL.createObjectURL(metaBlob);
  downloadDataUrl(metaUrl, `knot_${presetId}_s${seedStr}_metadata.json`);
  URL.revokeObjectURL(metaUrl);
}

// ============= Event Listeners =============

document.getElementById('btnGenerate')?.addEventListener('click', regenerate);
document.getElementById('btnExport')?.addEventListener('click', exportDataset);
document.getElementById('btnCapture')?.addEventListener('click', captureCurrentPng);
document.getElementById('btnBatchCapture')?.addEventListener('click', () => {
  batchCapture().catch((err) => console.error('[batchCapture] failed:', err));
});

// Slider auto-regenerate
['deformStrength', 'slackness', 'chainR', 'chainNumLinks', 'chainOffsetY', 'chainSpacing', 'borromeanR', 'borromeanRatio', 'borromeanYellowY', 'borromeanBlueY', 'kinkCount', 'kinkAmplitude'].forEach(id => {
  document.getElementById(id)?.addEventListener('input', regenerate);
});

// ============= Initialize =============

function applyInitialUrlParams() {
  const params = new URLSearchParams(window.location.search || '');
  const ids = ['preset', 'seed', 'count', 'cols', 'quality', 'layout', 'scale', 'slackness', 'deformStrength'];
  for (const id of ids) {
    if (!params.has(id)) continue;
    const el = document.getElementById(id);
    if (el) el.value = params.get(id);
  }
}

initThree();
applyInitialUrlParams();
animate();
regenerate();
