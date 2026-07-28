 /**
 * Invariance Renderer
 * 
 * Renders single knot images and exports as PNG.
 * Reuses geometry building logic from unified-gallery.js.
 */

import * as THREE from 'three';
import * as CurveExtras from 'three/addons/curves/CurveExtras.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { processCenterline, applyDeformation, applySlackness } from './centerline-pipeline.js?v=7';
import { createRopeMesh, applyPhysicsConstraints, fixTubeSeam, applyRainbowGradient, applySolidColor } from './rope-renderer-unified.js?v=8';

const Curves = CurveExtras.Curves || CurveExtras;

// Cache PMREM environment map to avoid regeneration
let cachedEnvMap = null;
let cachedPmremGenerator = null;
let sharedRenderer = null;

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
  const seedFn = xmur3(String(seedStr || 'render-seed'));
  return mulberry32(seedFn());
}

function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ============= Curve Classes =============

class TorusKnotCurve extends THREE.Curve {
  constructor({ p = 2, q = 3, R = 1.0, r = 0.4 } = {}) {
    super();
    this.p = p; this.q = q; this.R = R; this.r = r;
  }
  getPoint(t, optionalTarget = new THREE.Vector3()) {
    const phi = t * Math.PI * 2;
    const { p, q, R, r } = this;
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

class KinkyUnknotCurve extends THREE.Curve {
  constructor({ k = 4, baseRadius = 1.0, kinkAmplitude = 0.25, seed = 12345 } = {}) {
    super();
    this.k = Math.max(2, Math.floor(k));
    this.baseRadius = baseRadius;
    this.kinkAmplitude = kinkAmplitude;
    this.rng = mulberry32(seed);
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

// ============= Geometry Building =============

function tubeQualityParams(quality) {
  if (quality === 'high') return { tubularSegments: 280, radialSegments: 18 };
  if (quality === 'mid') return { tubularSegments: 200, radialSegments: 14 };
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

function estimateAndNormalizeTube({ makeCurve, closed = true, quality = 'high', radius = 0.24, targetOuterRadius = 1.25 }) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const curve = makeCurve();
  const thin = new THREE.TubeGeometry(curve, tubularSegments, 0.01, radialSegments, closed);
  thin.computeBoundingSphere();
  const sNorm = thin.boundingSphere?.radius > 1e-6 ? (targetOuterRadius / thin.boundingSphere.radius) : 1.0;
  thin.dispose();
  const baseRadius = radius / sNorm;
  const geom = new THREE.TubeGeometry(curve, tubularSegments, baseRadius, radialSegments, closed);
  geom.scale(sNorm, sNorm, sNorm);
  geom.center();
  geom.computeBoundingSphere();
  // TubeGeometry has continuous normals; avoid recomputing which causes seam cracks
  if (geom.attributes.normal) geom.normalizeNormals();
  return geom;
}

function applyRandomTransform(geometry, rng) {
  const rx = rng() * Math.PI * 2;
  const ry = rng() * Math.PI * 2;
  const rz = rng() * Math.PI * 2;
  const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz));
  geometry.applyMatrix4(rotMatrix);
  
  // Preserve TubeGeometry stitched normals, only normalize
  if (geometry.attributes.normal) geometry.normalizeNormals();
  geometry.center();
}

// ============= Knot Type Builders =============

function buildHopfLinkGeometry({ rng, quality = 'high', radius = 0.12 } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.8;  // increased from 1.5 to prevent interpenetration with thicker tubes
  const tubeRadius = radius * 0.9;

  const curveA = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: 3, amp: 0.06 });
  const curveB = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, -2, 0), normal: new THREE.Vector3(1, 0, 0), waves: 3, amp: 0.06 });
  const aMesh = createRopeMesh(curveA, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
  const bMesh = createRopeMesh(curveB, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
  const a = aMesh.geometry;
  const b = bMesh.geometry;
  if (aMesh.material) aMesh.material.dispose();
  if (bMesh.material) bMesh.material.dispose();

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

function buildUnlinkedRingsGeometry({ rng, quality = 'high', radius = 0.12, nearMiss = false } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.0;
  const r = rng || Math.random;

  const curveA = new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: 3, amp: 0.1 });
  const curveB = nearMiss
    ? new PlanarWobbleCircleCurve({
        radius: R,
        center: new THREE.Vector3(1.28 + r() * 0.16, (r() - 0.5) * 0.18, 0.10 + r() * 0.18),
        normal: new THREE.Vector3(1, 0.16 + r() * 0.18, 0.08 + r() * 0.12).normalize(),
        waves: 4,
        amp: 0.14,
      })
    : new PlanarWobbleCircleCurve({ radius: R, center: new THREE.Vector3(2.5, 0, 0), normal: new THREE.Vector3(0, 0, 1), waves: 3, amp: 0.1 });

  const tubeRadius = radius * 0.8;
  const aMesh = createRopeMesh(curveA, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
  const bMesh = createRopeMesh(curveB, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
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

function chainRandomChoice(rng, arr) {
  return arr[Math.min(arr.length - 1, Math.floor((rng ? rng() : Math.random()) * arr.length))];
}

const READABLE_RING_CAMERA_DIRS = [
  new THREE.Vector3(3.5, 3.5, 5).normalize(),
  new THREE.Vector3(0, 1.5, 6).normalize(),
  new THREE.Vector3(2, 6, 3).normalize(),
];
const MIN_RING_VISIBILITY_SCORE = 0.18;
const MAX_LINK_COMPONENTS = 10;
const LINK_SCENE_TYPES = new Set([
  'hopf_link',
  'unlinked_rings',
  'chain',
  'borromean',
  'double_hopf',
  'link_cluster',
  'hopf_plus_free',
  'chain_plus_free',
]);

function ringVisibilityScore(normal) {
  const n = normal.clone().normalize();
  return Math.min(...READABLE_RING_CAMERA_DIRS.map((viewDir) => Math.abs(n.dot(viewDir))));
}

function makeReadableFreeRingNormal(rng) {
  const r = rng || Math.random;
  let best = null;
  for (let i = 0; i < 12; i++) {
    const candidate = new THREE.Vector3(
      r() - 0.5,
      r() - 0.5,
      0.45 + r() * 0.75
    ).normalize();
    const score = ringVisibilityScore(candidate);
    if (!best || score > best.score) best = { normal: candidate, score };
    if (score >= MIN_RING_VISIBILITY_SCORE) return candidate;
  }
  return best?.normal || new THREE.Vector3(0.25, 0.35, 0.9).normalize();
}

function chainGenerateDiverseLayout({
  rng,
  numLinks,
  R,
  tubeRadius,
  effectiveStep,
  linkOffsetY,
} = {}) {
  const r = rng || Math.random;
  const n = Math.max(2, Math.floor(numLinks || 4));

  const centers = Array.from({ length: n }, () => new THREE.Vector3());
  const normals = Array.from({ length: n }, () => new THREE.Vector3(0, 0, 1));
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
      .map(a => ({ a, d: Math.abs(a.dot(nn)), score: ringVisibilityScore(a) }))
      .sort((p, q) => p.d - q.d)
      .filter(p => p.d < 0.35);
    if (!candidates.length) return chainRandomChoice(r, basisAxes).clone().normalize();
    candidates.sort((p, q) => (q.score - p.score) || (p.d - q.d));
    const bestScore = candidates[0].score;
    const readable = candidates.filter(p => p.score >= bestScore - 0.03);
    return chainRandomChoice(r, readable).a.clone().normalize();
  };

  // Keep the visual topology aligned with metadata: a chain is a simple linear
  // path, not a branch/cycle.  Separation stays well below the centerline
  // intersection case for perpendicular equal-radius rings (about 2R), while
  // leaving enough tube clearance for the wobble deformation.
  const linkSepBase = Math.max(1.28 * R, Math.min(Math.abs(effectiveStep) || (1.38 * R), 1.46 * R));
  const jitter = 0.04 * R;
  const zJitter = 0.035 * R;
  const offsetMag = Math.max(0.0, Math.min(2.5 * R, Math.abs(linkOffsetY))) * 0.15;

  centers[0].set(0, 0, 0);
  normals[0].copy(new THREE.Vector3(0, 0, 1));

  let lastStep = new THREE.Vector3(1, 0, 0);
  for (let i = 1; i < n; i++) {
    const prevNormal = normals[i - 1].clone().normalize();
    const nextNormal = pickPerpAxis(prevNormal);
    let stepDir = new THREE.Vector3().crossVectors(prevNormal, nextNormal);
    if (stepDir.lengthSq() < 1e-6) {
      stepDir = pickPerpAxis(prevNormal).cross(prevNormal);
    }
    stepDir.normalize();
    if (i > 1 && stepDir.dot(lastStep) < 0) stepDir.negate();
    lastStep.copy(stepDir);

    const sep = linkSepBase * (0.97 + 0.08 * r());
    centers[i].copy(centers[i - 1])
      .addScaledVector(stepDir, sep)
      .addScaledVector(prevNormal, (r() - 0.5) * zJitter)
      .addScaledVector(nextNormal, (r() - 0.5) * zJitter)
      .addScaledVector(prevNormal, (r() - 0.5) * offsetMag * 0.25);
    normals[i].copy(nextNormal);
  }

  for (let i = 1; i < n; i++) {
    centers[i].add(new THREE.Vector3(
      (r() - 0.5) * jitter,
      (r() - 0.5) * jitter,
      (r() - 0.5) * jitter
    ));
  }

  return { centers, normals };
}

function buildChainGeometry({ rng, quality = 'high', radius = 0.10, numLinks = 4, solidColors = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.8;  // increased from 1.5 to prevent interpenetration
  const linkOffsetY = -2.4;  // increased separation accordingly
  const effectiveStep = Math.abs(linkOffsetY);
  const tubeRadius = radius * 0.9;

  const geoms = [];
  const { centers, normals } = chainGenerateDiverseLayout({
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
    const ringR = rng ? (R * (0.92 + rng() * 0.16)) : R;
    const amp = rng ? 0.035 + rng() * 0.06 : 0.05;
    const waves = rng ? 2 + Math.floor(rng() * 5) : 3;
    const phase = rng ? rng() * Math.PI * 2 : 0;

    const curve = new PlanarWobbleCircleCurve({ radius: ringR, center, normal, waves, amp, phase });
    const mesh = createRopeMesh(curve, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
    const g = mesh.geometry;
    if (mesh.material) mesh.material.dispose();
    if (solidColors && solidColors[i]) {
      applySolidColor(g, solidColors[i]);
    } else {
      applyRainbowGradient(g, radialSegments, i / Math.max(1, numLinks));
    }
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

function buildBorromeanRingsGeometry({ rng, quality = 'high', radius = 0.08, solidColors = null } = {}) {
  // =====================================================================
  // Borromean Rings - 3 ellipses on 3 orthogonal planes (XY, YZ, XZ)
  //
  // Construction matching gallery preview:
  //   1. Red ring on XY plane, yellow on YZ plane, blue on XZ plane
  //   2. Ellipse (semi-major a = R*ratio, semi-minor b = R) for clearer 3D structure
  //   3. Auto-detect crossings, apply Gaussian bump for over/under
  //   4. Cyclic sign ensures Borromean topology
  // =====================================================================
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.5;            // ellipse semi-minor radius
  const ratio = 1.6;        // axis ratio
  const yellowY = 0.70;     // yellow ring Y offset
  const blueY = 0.0;        // blue ring Y offset
  const tubeR = 0.055;      // tube radius

  const a = R * ratio, b = R;  // ellipse semi-axes

  const ringColors = [
    new THREE.Color(0.95, 0.25, 0.25),  // red
    new THREE.Color(0.95, 0.85, 0.15),  // yellow
    new THREE.Color(0.25, 0.45, 0.95),  // blue
  ];

  // 3D ellipse curve
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
      return (optionalTarget || new THREE.Vector3()).set(0, 0, 0)
        .addScaledVector(this.center, 1)
        .addScaledVector(this.xAxis, this.a * Math.cos(angle))
        .addScaledVector(this.yAxis, this.b * Math.sin(angle));
    }
  }

  // 3 ellipses on 3 orthogonal planes
  const configs = [
    { center: new THREE.Vector3(0, 0, 0),       xAxis: new THREE.Vector3(1, 0, 0), yAxis: new THREE.Vector3(0, 1, 0) },  // XY plane
    { center: new THREE.Vector3(0, yellowY, 0),  xAxis: new THREE.Vector3(0, 1, 0), yAxis: new THREE.Vector3(0, 0, 1) },  // YZ plane
    { center: new THREE.Vector3(0, blueY, 0),    xAxis: new THREE.Vector3(1, 0, 0), yAxis: new THREE.Vector3(0, 0, 1) },  // XZ plane
  ];

  // --- Step 1: Sample all 3 centerlines ---
  const N = 360;
  const ringPoints = [];
  for (let ri = 0; ri < 3; ri++) {
    const curve = new EllipseCurve3D({ a, b, ...configs[ri] });
    const pts = [];
    for (let k = 0; k < N; k++) pts.push(curve.getPoint(k / N));
    ringPoints.push(pts);
  }

  // --- Step 2: Detect crossings and apply over/under bump ---
  const bumpMag = tubeR * 5;
  const sigma = N * 0.05;
  const crossingThreshold = tubeR * 20;

  const circDist = (ia, ib) => { const d = Math.abs(ia - ib); return Math.min(d, N - d); };

  const findCrossings = (ptsA, ptsB) => {
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
    const results = [];
    for (let i = 0; i < N; i++) {
      const prev = (i - 1 + N) % N, next = (i + 1) % N;
      if (cDist[i] > cDist[prev] || cDist[i] > cDist[next]) continue;
      if (cDist[i] > crossingThreshold) continue;
      if (results.some(r => circDist(i, r.idxA) < N * 0.1)) continue;
      const j = cIdx[i];
      const tA = new THREE.Vector3().subVectors(ptsA[(i + 1) % N], ptsA[(i - 1 + N) % N]).normalize();
      const tB = new THREE.Vector3().subVectors(ptsB[(j + 1) % N], ptsB[(j - 1 + N) % N]).normalize();
      const normal = new THREE.Vector3().crossVectors(tA, tB);
      if (normal.length() < 1e-6) continue;
      normal.normalize();
      results.push({ idxA: i, idxB: j, normal, dist: cDist[i] });
    }
    return results;
  };

  const pairs = [[0, 1], [0, 2], [1, 2]];
  for (const [ri, rj] of pairs) {
    const crossings = findCrossings(ringPoints[ri], ringPoints[rj]);
    crossings.forEach((cr, idx) => {
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

  // --- Step 3: Build TubeGeometry from displaced centerlines ---
  const geoms = [];
  for (let i = 0; i < 3; i++) {
    const smoothCurve = new THREE.CatmullRomCurve3(ringPoints[i], true, 'centripetal');
    const segments = Math.max(tubularSegments, 256);
    const g = new THREE.TubeGeometry(smoothCurve, segments, tubeR, radialSegments, true);
    if (solidColors && solidColors[i]) {
      applySolidColor(g, solidColors[i]);
    } else {
      applyRainbowGradient(g, radialSegments, i / 3);
    }
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

// ============= New Type Builders (loose, link groups, mixed) =============

/**
 * Generic loose knot builder
 */
function buildGenericLooseKnotGeometry({ rng, quality = 'high', radius = 0.24, baseCurveFactory, deformStrength = 0.3, slackness = 0.6 } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const s = clamp(Number(slackness) || 0.6, 0, 1);
  const d = clamp(Number(deformStrength) || 0.3, 0, 1);

  const baseCurve = baseCurveFactory();
  const sampleN = 420;
  let ringPts = baseCurve.getPoints(sampleN);
  ringPts = ringPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (ringPts.length < 40) {
    return new THREE.TubeGeometry(new CircleCurve({ radius: 1.0 }), tubularSegments, 0.02, radialSegments, true);
  }

  const start = Math.floor((rng ? rng() : 0.15) * ringPts.length * 0.35);
  const span = Math.max(80, Math.floor(ringPts.length * 0.82));
  let corePts = [];
  for (let i = 0; i <= span; i++) corePts.push(ringPts[(start + i) % ringPts.length].clone());

  const zScale = clamp(1.0 - 0.72 * s, 0.18, 1.0);
  const deformAmp = d * (0.05 + 0.09 * (1 - 0.5 * s));
  const tau = Math.PI * 2;
  const ph1 = (rng ? rng() : 0.31) * tau, ph2 = (rng ? rng() : 0.47) * tau, ph3 = (rng ? rng() : 0.63) * tau;
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

  // Open tails
  const headDir = corePts[0].clone().sub(corePts[1]).normalize();
  const tailDir = corePts[corePts.length - 1].clone().sub(corePts[corePts.length - 2]).normalize();
  const tailLenA = (0.70 + 0.55 * s) + (rng ? rng() * 0.35 : 0.2);
  const tailLenB = (0.75 + 0.60 * s) + (rng ? rng() * 0.35 : 0.2);
  const liftZ = 1.0 - 0.6 * s;
  const h0 = corePts[0].clone(), t0 = corePts[corePts.length - 1].clone();
  corePts = [
    h0.clone().addScaledVector(headDir, tailLenA * 1.5).add(new THREE.Vector3(-0.10, 0.18, 0.08 * liftZ)),
    h0.clone().addScaledVector(headDir, tailLenA * 0.7).addScaledVector(new THREE.Vector3(-0.10, 0.18, 0.08 * liftZ), 0.6),
    ...corePts,
    t0.clone().addScaledVector(tailDir, tailLenB * 0.7).addScaledVector(new THREE.Vector3(0.12, -0.15, -0.10 * liftZ), 0.6),
    t0.clone().addScaledVector(tailDir, tailLenB * 1.5).add(new THREE.Vector3(0.12, -0.15, -0.10 * liftZ)),
  ];

  // Tube radius proportional to curve size; matches TUBE_RATIO 0.02 after the
  // geom.scale normalization (post-scale outer ≈ 1.35, post-scale tube ≈ 0.027).
  const _looseBox = new THREE.Box3();
  for (const p of corePts) _looseBox.expandByPoint(p);
  const _looseBs = _looseBox.getBoundingSphere(new THREE.Sphere()).radius || 1;
  const tubeRadius = _looseBs * 0.02;

  let openCurve = new THREE.CatmullRomCurve3(corePts, false, 'centripetal');
  let smoothPts = openCurve.getPoints(Math.max(260, tubularSegments));
  smoothPts = smoothPts.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
  if (smoothPts.length >= 20) {
    const iters = Math.round(4 + 8 * s + 4 * d);
    for (let it = 0; it < iters; it++) {
      const next = smoothPts.map((p) => p.clone());
      for (let i = 1; i < smoothPts.length - 1; i++) {
        next[i].lerp(smoothPts[i - 1].clone().add(smoothPts[i]).add(smoothPts[i + 1]).divideScalar(3), 0.15);
      }
      smoothPts = next;
    }
    openCurve = new THREE.CatmullRomCurve3(smoothPts, false, 'centripetal');
  }

  return buildOpenRopeWithSticks({
    curve: openCurve,
    tubeRadius,
    tubularSegments: Math.max(280, tubularSegments),
    radialSegments,
  });
}

function bezierSegment(p0, p1, p2, p3, label = '') {
  const curve = new THREE.CubicBezierCurve3(p0, p1, p2, p3);
  curve.userData = { label, p0, p1, p2, p3 };
  return curve;
}

// Vertical "infinite" gray post through each open rope endpoint. Stays
// world-vertical so the viewer sees the endpoint can only slide along Y —
// the rope cannot be untangled by freely moving its ends.
const STICK_COLOR = new THREE.Color(0.30, 0.30, 0.32);
const STICK_LENGTH = 100.0;      // in normalized space; effectively infinite at any reasonable camera distance
const STICK_RADIUS_RATIO = 2.6;  // multiple of rope tube radius
const OPEN_ROPE_WITH_STICKS = new Set([
  'ring_like_open_rope', 'loose_open_knot', 'ring_like_open_knot', 'loose_cinquefoil',
]);

function buildVerticalStickGeometry(origin, length, radius, radialSegments) {
  const segs = Math.max(8, radialSegments);
  const geom = new THREE.CylinderGeometry(radius, radius, length, segs, 1, false);
  geom.translate(origin.x, origin.y, origin.z);
  return geom;
}

function buildOpenRopeWithSticks({ curve, tubeRadius, tubularSegments, radialSegments, normalizeTo = 1.35 }) {
  // Build rope and normalize ALONE so it keeps its full visible size in frame.
  const ropeGeom = new THREE.TubeGeometry(curve, tubularSegments, tubeRadius, radialSegments, false);
  applyRainbowGradient(ropeGeom, radialSegments);
  ropeGeom.computeBoundingBox();
  const center = new THREE.Vector3();
  ropeGeom.boundingBox.getCenter(center);
  ropeGeom.translate(-center.x, -center.y, -center.z);
  ropeGeom.computeBoundingSphere();
  const sNorm = ropeGeom.boundingSphere?.radius > 1e-6 ? (normalizeTo / ropeGeom.boundingSphere.radius) : 1.0;
  ropeGeom.scale(sNorm, sNorm, sNorm);

  // Map endpoint positions through the same center+scale transform.
  const toNormalized = (p) => p.clone().sub(center).multiplyScalar(sNorm);
  const headPos = toNormalized(curve.getPoint(0));
  const tailPos = toNormalized(curve.getPoint(1));

  const stickRadius = tubeRadius * sNorm * STICK_RADIUS_RATIO;
  const headStick = buildVerticalStickGeometry(headPos, STICK_LENGTH, stickRadius, radialSegments);
  const tailStick = buildVerticalStickGeometry(tailPos, STICK_LENGTH, stickRadius, radialSegments);
  applySolidColor(headStick, STICK_COLOR);
  applySolidColor(tailStick, STICK_COLOR);

  const merged = mergeBufferGeometries([ropeGeom, headStick, tailStick]);
  ropeGeom.dispose(); headStick.dispose(); tailStick.dispose();
  if (merged.attributes.normal) merged.normalizeNormals();
  // Pin bounding sphere to rope size and override computeBoundingSphere so
  // downstream callers (fitCameraToContent, computeSpacingFromGeometry) don't
  // recompute from the 100-unit posts and pull the camera way back, shrinking
  // the rope to a dot.
  const pinnedSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), normalizeTo);
  merged.boundingSphere = pinnedSphere;
  merged.computeBoundingSphere = function () { this.boundingSphere = pinnedSphere; };
  return merged;
}

function buildRingLikeOpenRopeGeometry({ rng, quality = 'high', radius = 0.24, deformStrength = 0.20 } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const d = clamp(Number(deformStrength) || 0.20, 0, 1);
  const rand = rng || (() => 0.5);

  const v = (x, y, z) => new THREE.Vector3(x, y, z);

  const controlPoints = [
    v(-6.0, 2.0, 0.0),
    v(-4.0, 2.0, 1.5),
    v(-1.5, 2.0, 3.0),
    v(1.0, 3.5, 2.0),
    v(2.5, 3.0, 0.0),
    v(1.5, 2.0, -2.0),
    v(-1.0, 1.0, -2.0),
    v(-2.0, 0.5, 0.0),
    v(-0.5, -0.5, 2.0),
    v(1.5, 0.0, 3.0),
    v(3.5, 1.0, 2.0),
    v(5.0, 2.0, 0.0),
    v(7.0, 2.0, 0.0),
  ];

  for (let i = 1; i < controlPoints.length - 1; i++) {
    const p = controlPoints[i];
    const endpointFalloff = Math.sin(Math.PI * i / (controlPoints.length - 1));
    p.x += (rand() - 0.5) * 0.18 * d * endpointFalloff;
    p.y += (rand() - 0.5) * 0.14 * d * endpointFalloff;
    p.z += (rand() - 0.5) * 0.18 * d * endpointFalloff;
  }

  const openCurve = new THREE.CatmullRomCurve3(controlPoints, false, 'centripetal');

  const box = new THREE.Box3();
  for (const p of openCurve.getPoints(140)) box.expandByPoint(p);
  const bs = box.getBoundingSphere(new THREE.Sphere()).radius || 1;
  return buildOpenRopeWithSticks({
    curve: openCurve,
    tubeRadius: bs * 0.021,
    tubularSegments: Math.max(340, tubularSegments),
    radialSegments,
  });
}

function buildRingLikeOpenKnotGeometry({ quality = 'high', radius = 0.24, slackness = 0.85 } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const s = clamp(Number(slackness) || 0.85, 0, 1);
  const loopX = 1.35 + 0.05 * s;
  const loopY = 1.18 + 0.04 * s;
  const tubeRadius = Math.max(0.036, radius * 0.62);
  const v = (x, y, depth) => new THREE.Vector3(depth, y * loopY, x * loopX);

  // Ring-like open knot. The visible diagram lives in x/y; depth is mapped to
  // Three.js X. The canonical gallery camera looks from negative X, so smaller
  // depth values are closer to the viewer.
  const crossingXY = { x: -0.16, y: -0.78 };
  const purpleOver = v(crossingXY.x, crossingXY.y, -0.32);
  const yellowUnder = v(crossingXY.x, crossingXY.y, 0.32);

  const path = new THREE.CurvePath();
  path.add(bezierSegment(
    v(-1.70, 0.10, -0.24),
    v(-1.20, 0.10, -0.24),
    v(-0.32, 0.12, 0.24),
    v(0.36, 0.12, 0.34),
    'horizontal_free_end',
  ));
  path.add(bezierSegment(
    v(0.36, 0.12, 0.34),
    v(0.92, 0.14, 0.34),
    v(1.62, 0.04, 0.30),
    v(1.46, -0.58, 0.22),
    'yellow_turn_to_horizontal',
  ));
  path.add(bezierSegment(
    v(1.46, -0.58, 0.22),
    v(1.18, -1.50, 0.14),
    v(0.04, -1.22, 0.26),
    yellowUnder,
    'lower_right_loop',
  ));
  path.add(bezierSegment(
    yellowUnder,
    v(-1.42, 0.06, 0.24),
    v(-1.52, 1.20, 0.08),
    v(-0.58, 1.44, -0.06),
    'left_side_under_crossing',
  ));
  path.add(bezierSegment(
    v(-0.58, 1.44, -0.06),
    v(0.48, 1.56, -0.02),
    v(1.04, 1.08, 0.02),
    v(0.68, 0.30, 0.05),
    'upper_ring_loop',
  ));
  path.add(bezierSegment(
    v(0.68, 0.30, 0.05),
    v(0.36, -0.08, -0.08),
    v(0.08, -0.48, -0.24),
    purpleOver,
    'inner_rise_to_upper_loop',
  ));
  path.add(bezierSegment(
    purpleOver,
    v(-0.36, -1.10, -0.30),
    v(-1.06, -1.42, -0.02),
    v(-1.58, -1.12, 0.20),
    'purple_over_crossing',
  ));
  path.add(bezierSegment(
    v(-1.58, -1.12, 0.20),
    v(-2.28, -0.70, 0.14),
    v(-2.28, 0.62, 0.12),
    v(-2.28, 1.34, 0.12),
    'left_vertical_tail',
  ));

  return buildOpenRopeWithSticks({
    curve: path,
    tubeRadius,
    tubularSegments: Math.max(360, tubularSegments),
    radialSegments,
  });
}

/**
 * Cable Knot — multiple strands twisted together following a real knot centerline.
 * Each strand orbits the knot path, creating the visual impression of ropes
 * genuinely tied together in a knot.
 *
 * @param {Object}       opts
 * @param {Function}     opts.rng
 * @param {string}       opts.quality
 * @param {number}       opts.numStrands   - 2 or 3
 * @param {THREE.Curve}  opts.baseCurve    - the knot centerline
 * @param {number}       opts.twistRate    - full twists per loop (integer)
 * @param {number}       opts.deformStrength
 * @param {number}       opts.slackness
 */
function buildCableKnotGeometry({
  rng,
  quality = 'high',
  numStrands = 3,
  baseCurve,
  twistRate = 5,
  deformStrength = 0.18,
  slackness = 0.08,
} = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);

  // 1. Process the knot centerline (deform + slackness + arc-length resample)
  const sampleN = 400;
  const processedCurve = processCenterline(baseCurve, {
    rng, deformStrength, slackness,
    tubeRadius: 0.07, sampleN, closed: true, knotType: 'cable_knot',
  });

  // 2. Compute Frenet frames along the processed curve
  const frameCount = sampleN;
  const frames = processedCurve.computeFrenetFrames(frameCount, true);
  const centerPoints = processedCurve.getPoints(frameCount);
  // getPoints returns frameCount+1 points; last ≈ first for closed curves
  const nPts = frameCount; // use indices 0..frameCount-1

  // 3. Adaptive sizing based on curve extent
  const box = new THREE.Box3();
  for (let i = 0; i < nPts; i++) box.expandByPoint(centerPoints[i]);
  const sz = new THREE.Vector3();
  box.getSize(sz);
  const curveScale = Math.max(sz.x, sz.y, sz.z) * 0.5 || 1.0;

  const strandRadius = curveScale * (numStrands <= 2 ? 0.045 : 0.035);
  const offsetRadius = strandRadius * 1.8;

  const strandColors = [
    new THREE.Color(0.95, 0.70, 0.65),
    new THREE.Color(0.65, 0.88, 0.65),
    new THREE.Color(0.68, 0.73, 0.98),
    new THREE.Color(0.92, 0.82, 0.58),
  ];

  // 4. Build each strand by offsetting from centerline using Frenet frames
  const geoms = [];
  for (let s = 0; s < numStrands; s++) {
    const phase = (s * Math.PI * 2) / numStrands;
    const strandPoints = [];

    for (let i = 0; i < nPts; i++) {
      const t = i / nPts;
      const angle = phase + twistRate * t * Math.PI * 2;
      const p = centerPoints[i].clone();
      p.addScaledVector(frames.normals[i], Math.cos(angle) * offsetRadius);
      p.addScaledVector(frames.binormals[i], Math.sin(angle) * offsetRadius);
      strandPoints.push(p);
    }

    // Apply inter-strand physics to prevent overlap
    const constrainedPts = applyPhysicsConstraints(strandPoints, {
      minDistance: strandRadius * 2.5,
      repulsionStrength: 0.12,
      iterations: 10,
      neighborSkip: 8,
      closed: true,
      pinEnds: false,
    });

    const strandCurve = new THREE.CatmullRomCurve3(constrainedPts, true, 'centripetal');
    const g = new THREE.TubeGeometry(strandCurve, tubularSegments, strandRadius, radialSegments, true);
    fixTubeSeam(g, tubularSegments, radialSegments);

    // Color — rainbow gradient along each strand
    applyRainbowGradient(g, radialSegments, s * 0.33);
    geoms.push(g);
  }

  // 5. Merge and normalize
  const merged = mergeBufferGeometries(geoms);
  geoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 1.5 / (merged.boundingSphere?.radius || 1);
  merged.scale(scale, scale, scale);
  merged._numRopes = numStrands;
  return merged;
}


/**
 * Helper: build a single chain segment (multiple interlinked rings), returns geometry array
 */
function buildOneChainSegment({ rng, R, tubeRadius, numLinks, offset, tubularSegments, radialSegments, hueBase = 0, solidColors = null, ringIndexOffset = 0 }) {
  const { centers, normals } = chainGenerateDiverseLayout({
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
    const mesh = createRopeMesh(curve, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
    const g = mesh.geometry;
    if (mesh.material) mesh.material.dispose();
    const globalIdx = ringIndexOffset + i;
    if (solidColors && solidColors[globalIdx]) {
      applySolidColor(g, solidColors[globalIdx]);
    } else {
      applyRainbowGradient(g, radialSegments, hueBase + (i / Math.max(1, numLinks)) * 0.3);
    }
    geoms.push(g);
  }
  return geoms;
}

/**
 * Helper: build an independent free ring (not linked to anything)
 */
function buildFreeRing({ rng, R, tubeRadius, center, tubularSegments, radialSegments, hue = 0.5, solidColor = null }) {
  const freeR = R * (0.7 + (rng?.() ?? 0.5) * 0.5);
  const freeNormal = makeReadableFreeRingNormal(rng);
  const amp = 0.04 + (rng?.() ?? 0.5) * 0.06;
  const waves = 2 + Math.floor((rng?.() ?? 0.5) * 3);
  const curve = new PlanarWobbleCircleCurve({ radius: freeR, center, normal: freeNormal, waves, amp });
  const mesh = createRopeMesh(curve, { radius: tubeRadius, color: '#ffffff', closed: true, tubularSegments, radialSegments });
  const g = mesh.geometry;
  if (mesh.material) mesh.material.dispose();
  if (solidColor) {
    applySolidColor(g, solidColor);
  } else {
    applyRainbowGradient(g, radialSegments, hue);
  }
  return g;
}

/**
 * Double chain group (double_hopf) - 2 independent chains, each internally interlinked
 */
function buildDoubleHopfGeometry({ rng, quality = 'high', radius = 0.24, clusterSegments = null, solidColors = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.0;
  const tubeRadius = radius * 0.7;
  const numChains = 2;
  const chainSep = R * 5;
  const allGeoms = [];
  let ringIndexOffset = 0;

  for (let c = 0; c < numChains; c++) {
    const numLinks = (clusterSegments && clusterSegments[c] != null)
      ? clusterSegments[c]
      : (rng ? (2 + Math.floor(rng() * 3)) : 3);
    const offset = new THREE.Vector3(
      (c - (numChains - 1) / 2) * chainSep,
      ((rng?.() ?? 0.5) - 0.5) * 1.5,
      ((rng?.() ?? 0.5) - 0.5) * 1.5
    );
    allGeoms.push(...buildOneChainSegment({
      rng, R, tubeRadius, numLinks, offset,
      tubularSegments, radialSegments, hueBase: c * 0.45,
      solidColors, ringIndexOffset,
    }));
    ringIndexOffset += numLinks;
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
 * Link cluster (link_cluster) - 2-4 independent chains of varying sizes
 */
function buildLinkClusterGeometry({ rng, quality = 'high', radius = 0.24, clusterSegments = null, solidColors = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 0.9;
  const tubeRadius = radius * 0.65;
  const rawSegments = (clusterSegments && clusterSegments.length)
    ? clusterSegments.map(n => Math.max(2, Math.floor(Number(n) || 2)))
    : null;
  const numChains = rawSegments
    ? Math.min(rawSegments.length, Math.floor(MAX_LINK_COMPONENTS / 2))
    : (rng ? (2 + Math.floor(rng() * 3)) : 3);
  const allGeoms = [];
  let ringIndexOffset = 0;
  let usedLinks = 0;

  for (let c = 0; c < numChains; c++) {
    const remainingChains = numChains - c - 1;
    const maxThis = Math.min(4, MAX_LINK_COMPONENTS - usedLinks - remainingChains * 2);
    const requestedLinks = rawSegments
      ? rawSegments[c]
      : (rng ? (2 + Math.floor(rng() * 3)) : 3);
    const numLinks = Math.max(2, Math.min(requestedLinks, maxThis));
    usedLinks += numLinks;
    const angle = (c / numChains) * Math.PI * 2 + (rng?.() ?? 0) * 0.5;
    const dist = R * 4.1 + (rng?.() ?? 0.5) * R * 1.4;
    const offset = new THREE.Vector3(
      Math.cos(angle) * dist,
      ((rng?.() ?? 0.5) - 0.5) * 2,
      Math.sin(angle) * dist
    );
    allGeoms.push(...buildOneChainSegment({
      rng, R, tubeRadius, numLinks, offset,
      tubularSegments, radialSegments, hueBase: c / numChains,
      solidColors, ringIndexOffset,
    }));
    ringIndexOffset += numLinks;
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
 * Chain + free ring (hopf_plus_free) - 1 chain (2-3 interlinked rings) + 1-2 independent rings
 */
function buildHopfPlusFreeGeometry({ rng, quality = 'high', radius = 0.24, chainLinks: chainLinksParam = null, numFree: numFreeParam = null, solidColors = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 1.0;
  const tubeRadius = radius * 0.7;
  const chainLinks = chainLinksParam ?? (rng ? (2 + Math.floor(rng() * 2)) : 3);
  const numFree = numFreeParam ?? (rng ? (1 + Math.floor(rng() * 2)) : 1);

  const allGeoms = buildOneChainSegment({
    rng, R, tubeRadius, numLinks: chainLinks,
    offset: new THREE.Vector3(0, 0, 0),
    tubularSegments, radialSegments, hueBase: 0.0,
    solidColors, ringIndexOffset: 0,
  });

  for (let i = 0; i < numFree; i++) {
    const offsetAngle = (rng?.() ?? 0.5) * Math.PI * 2;
    const dist = R * 4.2 + (rng?.() ?? 0.5) * R * 1.4;
    const center = new THREE.Vector3(
      Math.cos(offsetAngle) * dist,
      ((rng?.() ?? 0.5) - 0.5) * 2,
      Math.sin(offsetAngle) * dist
    );
    const freeIdx = chainLinks + i;
    allGeoms.push(buildFreeRing({
      rng, R, tubeRadius, center, tubularSegments, radialSegments,
      hue: 0.55 + i * 0.2,
      solidColor: (solidColors && solidColors[freeIdx]) || null,
    }));
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
 * Large chain + multiple free rings (chain_plus_free) - 1 longer chain (3-5 rings) + 1-3 independent rings
 */
function buildChainPlusFreeGeometry({ rng, quality = 'high', radius = 0.24, chainLinks: chainLinksParam = null, numFree: numFreeParam = null, solidColors = null } = {}) {
  const { tubularSegments, radialSegments } = tubeQualityParams(quality);
  const R = 0.9;
  const tubeRadius = radius * 0.65;
  const chainLinks = chainLinksParam ?? (rng ? (3 + Math.floor(rng() * 3)) : 4);
  const numFree = numFreeParam ?? (rng ? (1 + Math.floor(rng() * 3)) : 2);

  const allGeoms = buildOneChainSegment({
    rng, R, tubeRadius, numLinks: chainLinks,
    offset: new THREE.Vector3(0, 0, 0),
    tubularSegments, radialSegments, hueBase: 0.0,
    solidColors, ringIndexOffset: 0,
  });

  // Compute chain extent to place free rings outside
  const tmpMerged = mergeBufferGeometries(allGeoms.map(g => g.clone()));
  tmpMerged.computeBoundingSphere();
  const chainExtent = tmpMerged.boundingSphere?.radius || 2;
  tmpMerged.dispose();

  for (let i = 0; i < numFree; i++) {
    const offsetAngle = (i / numFree) * Math.PI * 2 + (rng?.() ?? 0) * 1.0;
    const dist = chainExtent + R * 2.4 + (rng?.() ?? 0.5) * R * 1.2;
    const center = new THREE.Vector3(
      Math.cos(offsetAngle) * dist,
      ((rng?.() ?? 0.5) - 0.5) * 2,
      Math.sin(offsetAngle) * dist
    );
    const freeIdx = chainLinks + i;
    allGeoms.push(buildFreeRing({
      rng, R, tubeRadius, center, tubularSegments, radialSegments,
      hue: 0.5 + i * 0.18,
      solidColor: (solidColors && solidColors[freeIdx]) || null,
    }));
  }

  const merged = mergeBufferGeometries(allGeoms);
  allGeoms.forEach(g => g.dispose());
  merged.center();
  merged.computeBoundingSphere();
  const scale = 2.0 / (merged.boundingSphere.radius || 1);
  merged.scale(scale, scale, scale);
  return merged;
}

// ============= Main Build Function =============

/**
 * Build geometry for a given knot type.
 * @param {string} knotType - knot type key
 * @param {Object} options - build options
 * @returns {THREE.BufferGeometry}
 */
export function buildGeometryForKnotType(knotType, options = {}) {
  const {
    rng = null,
    quality = 'high',
    radius = 0.15,
    deformStrength = 0.3,
    slackness = 0,
    anisotropicScale = [1, 1, 1],
  } = options;
  
  const localRng = rng || makeRng(String(Date.now()));

  const TUBE_RATIO = 0.02;

  const buildProcessedTube = (rawCurve, { closed = true, targetOuterRadius = 1.35, sampleN = 300 } = {}) => {
    // 1. Sample curve, estimate raw size, compute adaptive tubeRadius
    const samplePts = rawCurve.getPoints ? rawCurve.getPoints(100) : [];
    let curveRadius = 1.0;
    if (samplePts.length > 2) {
      const box = new THREE.Box3();
      samplePts.forEach(p => box.expandByPoint(p));
      const size = new THREE.Vector3();
      box.getSize(size);
      curveRadius = Math.max(size.x, size.y, size.z) * 0.5 || 1.0;
    }
    // tubeRadius proportional to curve size; after normalization = targetOuterRadius * TUBE_RATIO
    const tubeRadius = curveRadius * TUBE_RATIO;

    const processedCurve = processCenterline(rawCurve, {
      rng: localRng,
      deformStrength,
      slackness,
      tubeRadius,
      sampleN,
      closed,
      knotType,
    });

    const { tubularSegments, radialSegments } = tubeQualityParams(quality);
    const geom = new THREE.TubeGeometry(processedCurve, tubularSegments, tubeRadius, radialSegments, closed);
    // Fix closure seam for closed tubes
    if (closed) {
      fixTubeSeam(geom, tubularSegments, radialSegments);
    }
    geom.center();
    geom.computeBoundingSphere();
    const sNorm = geom.boundingSphere?.radius > 1e-6 ? (targetOuterRadius / geom.boundingSphere.radius) : 1.0;
    geom.scale(sNorm, sNorm, sNorm);
    geom.computeBoundingSphere();
    if (geom.attributes.normal) geom.normalizeNormals();
    return geom;
  };
  
  let geometry = null;
  
  switch (knotType) {
    case 'unknot':
      // ====================================================
      // Unknot: topologically unknottable, but should have apparent visual crossings
      //
      // Strategy: use KinkyUnknotCurve (lite version)
      //   - Produces local kinks (visually appear as crossings)
      //   - But topologically always unknot (all kinks can be continuously deformed away)
      //   - Different kink counts control visual complexity
      //
      // 30%: low complexity (2 kinks, simple but still has crossings)
      // 40%: medium complexity (3-4 kinks, may look knotted)
      // 30%: high complexity (5-6 kinks, easily misjudged as knotted)
      // ====================================================
      {
        const v = localRng();
        const seed = Math.floor(localRng() * 100000);
        let kinkCount, kinkAmp;
        if (v < 0.3) {
          kinkCount = 2;
          kinkAmp = 0.15 + localRng() * 0.10;  // 0.15-0.25
        } else if (v < 0.7) {
          kinkCount = 3 + Math.floor(localRng() * 2); // 3-4
          kinkAmp = 0.20 + localRng() * 0.10;  // 0.20-0.30
        } else {
          kinkCount = 5 + Math.floor(localRng() * 2); // 5-6
          kinkAmp = 0.25 + localRng() * 0.10;  // 0.25-0.35
        }
        const unknotCurve = new KinkyUnknotCurve({
          k: kinkCount,
          baseRadius: 1.0,
          kinkAmplitude: kinkAmp,
          seed,
        });
        geometry = buildProcessedTube(unknotCurve, { closed: true, targetOuterRadius: 1.35, sampleN: 400 });
      }
      break;
      
    case 'twisted_ring':
      geometry = buildProcessedTube(
        new TwistedRingCurve({
          R: 1.0,
          twist: 2 + Math.floor(localRng() * 5),
          wobble: 0.18 + localRng() * 0.18,
          height: 0.3 + localRng() * 0.25,
        }),
        { closed: true, targetOuterRadius: 1.35, sampleN: 400 }
      );
      break;
      
    case 'spiral_disk':
      {
        geometry = buildProcessedTube(new SpiralLoopCurve({
          tubeRadius: TUBE_RATIO,
          turns: 2 + Math.floor(localRng() * 2),   // 2 or 3 whole turns (fractional turns break closure → creates knot)
          pitch: 0.12 + localRng() * 0.10,
          innerRadius: 0.55,
          radialGap: 0.22 + localRng() * 0.12,
        }), { closed: true, targetOuterRadius: 1.35, sampleN: 400 });
      }
      break;
      
    case 'kinky_unknot':
      geometry = buildProcessedTube(new KinkyUnknotCurve({
        k: 3 + Math.floor(localRng() * 4),
        baseRadius: 1.0,
        kinkAmplitude: 0.2 + localRng() * 0.15,
        seed: Math.floor(localRng() * 100000),
      }), { closed: true, targetOuterRadius: 1.35, sampleN: 400 });
      break;
      
    case 'occluded_knot':
      // ====================================================
      // Occluded knot: trefoil base + strong deformation + enhanced physics constraints to prevent clipping
      // Bypasses buildProcessedTube, handles physics constraints separately for stronger anti-clipping
      // ====================================================
      {
        const baseCurve = new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.55 });
        const occSampleN = 450;
        let occPts = baseCurve.getPoints(occSampleN);
        occPts = occPts.filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
        // Remove duplicate endpoint for closed curve
        if (occPts.length > 10) {
          const f = occPts[0], l = occPts[occPts.length - 1];
          if (f.distanceTo(l) < 1e-4) occPts.pop();
        }
        // Apply deformation
        if (deformStrength > 0 && localRng) {
          occPts = applyDeformation(occPts, { deformStrength: deformStrength * 1.2, rng: localRng, closed: true });
        }
        // Apply slackness
        if (slackness > 0) {
          occPts = applySlackness(occPts, { slackness, tubeRadius: 0.08, closed: true });
        }
        // Enhanced physics: more iterations, larger minDistance, stronger repulsion
        const occTubeR = 0.065;
        occPts = applyPhysicsConstraints(occPts, {
          minDistance: occTubeR * 3.5,
          repulsionStrength: 0.25,
          iterations: 40,
          neighborSkip: 10,
          closed: true,
          pinEnds: false,
        });
        // Second pass: finer constraint
        occPts = applyPhysicsConstraints(occPts, {
          minDistance: occTubeR * 2.8,
          repulsionStrength: 0.15,
          iterations: 20,
          neighborSkip: 8,
          closed: true,
          pinEnds: false,
        });
        const occCurve = new THREE.CatmullRomCurve3(occPts, true, 'centripetal');
        const { tubularSegments: occTS, radialSegments: occRS } = tubeQualityParams(quality);
        const occGeom = new THREE.TubeGeometry(occCurve, Math.max(300, occTS), occTubeR, occRS, true);
        occGeom.center();
        occGeom.computeBoundingSphere();
        const occScale = 1.35 / (occGeom.boundingSphere?.radius || 1);
        occGeom.scale(occScale, occScale, occScale);
        if (occGeom.attributes.normal) occGeom.normalizeNormals();
        geometry = occGeom;
      }
      break;

    case 'trefoil':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.55 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'loose_open_knot':
      // ====================================================
      // Loose open knot: take ~82% arc of trefoil, add open tails at both ends
      // Key: **bypasses processCenterline** - slackness would flatten crossings
      // ====================================================
      {
        const { tubularSegments: ts, radialSegments: rs } = tubeQualityParams(quality);
        const baseCurve = new TorusKnotCurve({ p: 2, q: 3, R: 1.0, r: 0.50 });
        const sampleN = 420;
        let ringPts = baseCurve.getPoints(sampleN);
        ringPts = ringPts.filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));

        // Cut ~82% arc from random position to create opening
        const start = Math.floor(localRng() * ringPts.length * 0.35);
        const span = Math.max(80, Math.floor(ringPts.length * 0.82));
        let corePts = [];
        for (let i = 0; i <= span; i++) {
          corePts.push(ringPts[(start + i) % ringPts.length].clone());
        }

        // Slightly compress Z + add deformation
        const s = clamp(slackness, 0, 1);
        const d = clamp(deformStrength, 0, 1);
        const zScale = clamp(1.0 - 0.72 * s, 0.18, 1.0);
        const deformAmp = d * (0.05 + 0.09 * (1 - 0.5 * s));
        const tau = Math.PI * 2;
        const ph1 = localRng() * tau, ph2 = localRng() * tau, ph3 = localRng() * tau;
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

        // Add open tails at both ends
        const headDir = corePts[0].clone().sub(corePts[1]).normalize();
        const tailDir = corePts[corePts.length - 1].clone().sub(corePts[corePts.length - 2]).normalize();
        const tailLenA = 0.70 + 0.55 * s + localRng() * 0.35;
        const tailLenB = 0.75 + 0.60 * s + localRng() * 0.35;
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

        // Smooth curve
        let openCurve = new THREE.CatmullRomCurve3(corePts, false, 'centripetal');
        let smoothPts = openCurve.getPoints(Math.max(260, ts));
        smoothPts = smoothPts.filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z));
        if (smoothPts.length >= 20) {
          const iters = Math.round(4 + 8 * s + 4 * d);
          const beta = 0.12 + 0.10 * s;
          for (let it = 0; it < iters; it++) {
            const next = smoothPts.map(p => p.clone());
            for (let i = 1; i < smoothPts.length - 1; i++) {
              const avg = smoothPts[i - 1].clone().add(smoothPts[i]).add(smoothPts[i + 1]).divideScalar(3);
              next[i].lerp(avg, beta);
            }
            smoothPts = next;
          }
          openCurve = new THREE.CatmullRomCurve3(smoothPts, false, 'centripetal');
        }

        // Estimate curve size, compute tube radius
        const estPts = openCurve.getPoints(100);
        const estBox = new THREE.Box3();
        estPts.forEach(p => estBox.expandByPoint(p));
        const estSize = new THREE.Vector3(); estBox.getSize(estSize);
        const curveR = Math.max(estSize.x, estSize.y, estSize.z) * 0.5 || 1.0;
        const looseTubeR = curveR * TUBE_RATIO;

        geometry = buildOpenRopeWithSticks({
          curve: openCurve,
          tubeRadius: looseTubeR,
          tubularSegments: Math.max(ts, 300),
          radialSegments: rs,
        });
      }
      break;

    case 'ring_like_open_knot':
      geometry = buildRingLikeOpenKnotGeometry({
        rng: localRng, quality, radius,
        deformStrength: Math.max(deformStrength, 0.10),
        slackness: Math.max(slackness, 0.92),
      });
      break;

    case 'ring_like_open_rope':
      geometry = buildRingLikeOpenRopeGeometry({
        rng: localRng, quality, radius,
        deformStrength: Math.max(deformStrength, 0.12),
      });
      break;
      
    case 'figure8':
      geometry = buildProcessedTube(new FigureEightKnotCurve(), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'torus_2_5':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 2, q: 5, R: 1.0, r: 0.50 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'torus_2_7':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 2, q: 7, R: 1.0, r: 0.45 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'torus_2_9':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 2, q: 9, R: 1.0, r: 0.42 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'torus_3_4':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 3, q: 4, R: 1.0, r: 0.45 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;

    case 'torus_3_5':
      geometry = buildProcessedTube(new TorusKnotCurve({ p: 3, q: 5, R: 1.0, r: 0.40 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
      break;
      
    case 'hopf_link':
      geometry = buildHopfLinkGeometry({ rng: localRng, quality, radius });
      break;

    case 'unlinked_rings':
      geometry = buildUnlinkedRingsGeometry({
        rng: localRng,
        quality,
        radius,
        nearMiss: !!options.nearMiss,
      });
      break;

    case 'chain':
      geometry = buildChainGeometry({
        rng: localRng, quality, radius,
        numLinks: options.numLinks || 4,
        solidColors: options.solidColors || null,
      });
      break;

    case 'borromean':
      geometry = buildBorromeanRingsGeometry({
        rng: localRng, quality, radius: 0.08,
        solidColors: options.solidColors || null,
      });
      break;

    // --- Loose Knots ---
    case 'loose_cinquefoil':
      geometry = buildGenericLooseKnotGeometry({
        rng: localRng, quality, radius,
        baseCurveFactory: () => new TorusKnotCurve({ p: 2, q: 5, R: 1.0, r: 0.38 }),
        deformStrength: deformStrength || 0.35, slackness: Math.max(slackness, 0.55),
      });
      break;

    // --- Link Groups ---
    case 'double_hopf':
      geometry = buildDoubleHopfGeometry({
        rng: localRng, quality, radius,
        clusterSegments: options.clusterSegments || null,
        solidColors: options.solidColors || null,
      });
      break;

    case 'link_cluster':
      geometry = buildLinkClusterGeometry({
        rng: localRng, quality, radius,
        clusterSegments: options.clusterSegments || null,
        solidColors: options.solidColors || null,
      });
      break;

    // --- Mixed ---
    case 'hopf_plus_free':
      geometry = buildHopfPlusFreeGeometry({
        rng: localRng, quality, radius,
        chainLinks: options.chainLinks ?? null,
        numFree: options.numFree ?? null,
        solidColors: options.solidColors || null,
      });
      break;

    case 'chain_plus_free':
      geometry = buildChainPlusFreeGeometry({
        rng: localRng, quality, radius,
        chainLinks: options.chainLinks ?? null,
        numFree: options.numFree ?? null,
        solidColors: options.solidColors || null,
      });
      break;

    default:
      // Fallback to circle
      geometry = buildProcessedTube(new CircleCurve({ radius: 1.0 }), { closed: true, targetOuterRadius: 1.35, sampleN: 300 });
  }
  
// No normal perturbation on tube walls, keep uniform tube radius
// No anisotropic scaling, keep circular cross-section
  
  // Random rotation for diversity. Skip for open-rope-with-sticks types so the
  // vertical posts stay world-vertical (rotation would tilt them).
  if (geometry && rng && !LINK_SCENE_TYPES.has(knotType) && !OPEN_ROPE_WITH_STICKS.has(knotType)) {
    applyRandomTransform(geometry, localRng);
  }
  
  return geometry;
}

// ============= Scene & Renderer =============

/**
 * Create an independent render scene.
 * Replicates knot_gallery.html high-quality rendering.
 */
export function createRenderScene(options = {}) {
  const {
    width = 2048,
    height = 2048,
    backgroundColor = '#1a2236',
    antialias = true,
  } = options;
  
  // Scene - use knot_gallery.html background color
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(backgroundColor);
  
  // Camera - adjusted position for better viewpoint
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500);
  camera.position.set(0, 2.5, 5.5);
  camera.lookAt(0, 0, 0);
  
  // Renderer - reused across calls so the WebGL context is not recreated per image
  // (browsers cap concurrent contexts and would otherwise crash the page after ~16-1700 frames).
  if (!sharedRenderer) {
    sharedRenderer = new THREE.WebGLRenderer({
      antialias,
      preserveDrawingBuffer: true,
      alpha: false,
    });
    sharedRenderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    sharedRenderer.outputColorSpace = THREE.SRGBColorSpace;
    sharedRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    sharedRenderer.toneMappingExposure = 1.2;
  }
  const renderer = sharedRenderer;
  renderer.setSize(width, height, false);
  
  // Generate PMREM environment map - key for realistic metallic materials!
  if (!cachedPmremGenerator) {
    cachedPmremGenerator = new THREE.PMREMGenerator(renderer);
    cachedPmremGenerator.compileEquirectangularShader();
  }
  
  if (!cachedEnvMap) {
    const roomEnv = new RoomEnvironment();
    cachedEnvMap = cachedPmremGenerator.fromScene(roomEnv).texture;
    roomEnv.dispose();
  }
  
  // Set scene environment map
  scene.environment = cachedEnvMap;
  
  return { scene, camera, renderer };
}

export function setupLighting(scene) {
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.46);
  scene.add(ambientLight);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x2a335a, 0.18);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.95);
  dir.position.set(10, 18, 10);
  scene.add(dir);

  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-10, 5, -10);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xffffff, 0.42);
  rim.position.set(0, 6, -14);
  scene.add(rim);
}

/**
 * Add vertex colors to geometry (if not present).
 * Key for knot_gallery.html rendering quality.
 */
function ensureVertexColors(geometry, baseColor = null) {
  if (!geometry.attributes.color) {
    // Default: apply rainbow gradient using high-quality radial segments
    const { radialSegments } = tubeQualityParams('high');
    applyRainbowGradient(geometry, radialSegments);
  }
  return geometry;
}

/**
 * Create high-quality material - replicates knot_gallery.html rendering.
 * Key: vertexColors: true enables vertex color output.
 * Combined with RoomEnvironment PMREM for realistic metallic reflections.
 */
function createHighQualityMaterial(color, options = {}) {
  const {
    metalness = 0.04,   // matched to gallery (unified-gallery.js line 2173)
    roughness = 0.5,    // matched to gallery (line 2174)
  } = options;

  return new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness,
    metalness,
    vertexColors: true,
    envMapIntensity: 1.0,
    emissive: new THREE.Color(0x050505),   // matched to gallery (line 2176)
    emissiveIntensity: 0.04,               // matched to gallery (line 2177)
  });
}

function projectedFrameSpan(camera, object) {
  object.updateWorldMatrix(true, false);
  const pos = object.geometry?.attributes?.position;
  const maxProjectedVertices = 6000;
  const step = pos ? Math.max(1, Math.floor(pos.count / maxProjectedVertices)) : 1;
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  const point = new THREE.Vector3();

  if (pos) {
    for (let i = 0; i < pos.count; i += step) {
      point.fromBufferAttribute(pos, i).applyMatrix4(object.matrixWorld).project(camera);
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
  } else {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return 0;
    const corners = [
      new THREE.Vector3(box.min.x, box.min.y, box.min.z),
      new THREE.Vector3(box.min.x, box.min.y, box.max.z),
      new THREE.Vector3(box.min.x, box.max.y, box.min.z),
      new THREE.Vector3(box.min.x, box.max.y, box.max.z),
      new THREE.Vector3(box.max.x, box.min.y, box.min.z),
      new THREE.Vector3(box.max.x, box.min.y, box.max.z),
      new THREE.Vector3(box.max.x, box.max.y, box.min.z),
      new THREE.Vector3(box.max.x, box.max.y, box.max.z),
    ];
    for (const corner of corners) {
      corner.project(camera);
      minX = Math.min(minX, corner.x);
      maxX = Math.max(maxX, corner.x);
      minY = Math.min(minY, corner.y);
      maxY = Math.max(maxY, corner.y);
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return 0;
  return Math.max(maxX - minX, maxY - minY);
}

function fitMeshToFrame(camera, mesh, { targetSpan = 1.55, maxScale = 2.1 } = {}) {
  camera.updateProjectionMatrix();
  const span = projectedFrameSpan(camera, mesh);
  if (!Number.isFinite(span) || span <= 1e-6 || span >= targetSpan) return;
  const factor = Math.min(maxScale, targetSpan / span);
  mesh.scale.multiplyScalar(factor);
  mesh.updateWorldMatrix(true, false);
}

/**
 * Render a single image.
 * @param {Object} imageParams - image parameters (from invariance-generator.js)
 * @returns {Promise<string>} Data URL (PNG)
 */
export async function renderSingleImage(imageParams, options = {}) {
  const {
    width = 2048,
    height = 2048,
  } = options;
  
  const rng = makeRng(String(imageParams.seed));
  
  // Create scene - knot_gallery.html style background
  const { scene, camera, renderer } = createRenderScene({
    width,
    height,
    backgroundColor: imageParams.backgroundColor || '#1a2236',
  });
  
  // Setup camera - better viewpoint
  if (imageParams.cameraPosition) {
    camera.position.set(...imageParams.cameraPosition);
  }
  if (imageParams.cameraTarget) {
    camera.lookAt(...imageParams.cameraTarget);
  }
  if (imageParams.cameraFov) {
    camera.fov = imageParams.cameraFov;
    camera.updateProjectionMatrix();
  }
  
  // Setup lighting — gallery-matched fixed values (ignores per-image overrides)
  setupLighting(scene);
  
  // Build geometry
  const geometry = buildGeometryForKnotType(imageParams.knotType, {
    rng,
    quality: 'high',
    radius: imageParams.tubeRadius ?? 0.07,
    deformStrength: imageParams.deformStrength || 0.3,
    slackness: imageParams.slackness || 0,
    anisotropicScale: imageParams.anisotropicScale || [1, 1, 1],
    numLinks: imageParams.numLinks,
    chainLinks: imageParams.chainLinks,
    numFree: imageParams.numFree,
    clusterSegments: imageParams.clusterSegments,
    nearMiss: imageParams.nearMiss,
    solidColors: imageParams.solidColors,
  });
  
  // Ensure geometry has vertex colors - key for knot_gallery.html rendering
  ensureVertexColors(geometry, imageParams.color || '#72e6ff');

  // Create high-quality material — gallery-matched defaults
  const material = createHighQualityMaterial(
    imageParams.color || '#72e6ff',
    {
      metalness: imageParams.metalness ?? 0.04,
      roughness: imageParams.roughness ?? 0.5,
    }
  );
  
  // Create mesh
  const mesh = new THREE.Mesh(geometry, material);

  // Center and uniform scale: ensure knot is large enough in frame without overflow
  geometry.computeBoundingSphere();
  const bsRadius = geometry.boundingSphere?.radius || 1;
  const TARGET_VISUAL_RADIUS = 1.8; // target bounding sphere radius
  if (bsRadius > 1e-6) {
    const uniformScale = TARGET_VISUAL_RADIUS / bsRadius;
    mesh.scale.setScalar(uniformScale);
  }
  fitMeshToFrame(camera, mesh);

  scene.add(mesh);
  
  // Render
  renderer.render(scene, camera);
  
  // Get data URL
  const dataUrl = renderer.domElement.toDataURL('image/png');
  
  // Cleanup (renderer is shared and intentionally NOT disposed)
  geometry.dispose();
  material.dispose();

  return dataUrl;
}

/**
 * Batch render a Pair
 * @param {Object} pair - PairRecord
 * @param {Object} options
 * @returns {Promise<{ imageA: string, imageB: string }>} Data URLs
 */
export async function renderPair(pair, options = {}) {
  const imageA = await renderSingleImage(pair.imageA, options);
  const imageB = await renderSingleImage(pair.imageB, options);
  return { imageA, imageB };
}

/**
 * Convert Data URL to Blob
 */
export function dataUrlToBlob(dataUrl) {
  const parts = dataUrl.split(',');
  const mime = parts[0].match(/:(.*?);/)[1];
  const bstr = atob(parts[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

/**
 * Trigger file download
 */
export function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/**
 * Dispose PMREM cache (call when rendering is no longer needed)
 */
export function disposeEnvironmentCache() {
  if (cachedEnvMap) {
    cachedEnvMap.dispose();
    cachedEnvMap = null;
  }
  if (cachedPmremGenerator) {
    cachedPmremGenerator.dispose();
    cachedPmremGenerator = null;
  }
  if (sharedRenderer) {
    sharedRenderer.dispose();
    sharedRenderer = null;
  }
}

// ============= Exports =============

export default {
  buildGeometryForKnotType,
  createRenderScene,
  setupLighting,
  renderSingleImage,
  renderPair,
  dataUrlToBlob,
  downloadDataUrl,
  disposeEnvironmentCache,
};
