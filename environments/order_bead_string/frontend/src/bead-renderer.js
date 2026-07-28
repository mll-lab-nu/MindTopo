/**
 * Bead Renderer
 *
 * Rendering primitives for rope (TubeGeometry) and beads (SphereGeometry).
 * Composes a complete bead string scene from curve + parameters.
 */

import * as THREE from 'three';
import { createCurve, sampleEquidistantPoints, makeRng } from './bead-curve-library.js';

// ============= Color Palette =============

const ROPE_COLOR = '#8B7355';
const BEAD_OPACITY = 0.85;

export const BEAD_PALETTE = {
  RED:    { name: 'RED',    hex: '#E74C3C' },
  BLUE:   { name: 'BLUE',   hex: '#3498DB' },
  GREEN:  { name: 'GREEN',  hex: '#2ECC71' },
  YELLOW: { name: 'YELLOW', hex: '#F1C40F' },
  ORANGE: { name: 'ORANGE', hex: '#E67E22' },
  PURPLE: { name: 'PURPLE', hex: '#9B59B6' },
  WHITE:  { name: 'WHITE',  hex: '#ECF0F1' },
  BROWN:  { name: 'BROWN',  hex: '#8D6E63' },
};

export const PALETTE_NAMES = Object.keys(BEAD_PALETTE);

// Similar-color groups for hard difficulty
const SIMILAR_GROUPS = [
  ['RED', 'ORANGE'],
  ['BLUE', 'PURPLE'],
  ['WHITE', 'YELLOW'],
  ['BROWN', 'ORANGE'],
];

/**
 * Select bead colors from the palette.
 * @param {number} numBeads
 * @param {Function} rng
 * @param {'distinct'|'mixed'|'similar'|'palindrome'|'periodic'} colorMode
 * @param {string[]} [fixedSequence] - if provided, use this exact sequence (for cyclic pair generation)
 * @returns {string[]} array of color names
 */
export function selectBeadColors(numBeads, rng, colorMode = 'distinct', fixedSequence = null) {
  // ── Fixed sequence override (used for cyclic pair generation) ──
  if (fixedSequence) {
    return [...fixedSequence];
  }

  const n = numBeads;

  // ── Palindrome mode: generate first half, then mirror ──
  if (colorMode === 'palindrome') {
    const halfLen = Math.ceil(n / 2);
    const shuffled = [...PALETTE_NAMES];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const half = [];
    for (let i = 0; i < halfLen; i++) {
      half.push(shuffled[i % shuffled.length]);
    }
    // Avoid adjacent duplicates in first half
    for (let i = 1; i < half.length; i++) {
      if (half[i] === half[i - 1]) {
        const alt = shuffled.find(c => c !== half[i]) || shuffled[0];
        half[i] = alt;
      }
    }
    const mirrored = [...half, ...half.slice(0, Math.floor(n / 2)).reverse()];
    return mirrored.slice(0, n);
  }

  // ── Periodic mode: generate a short unit, then repeat ──
  if (colorMode === 'periodic') {
    // Period length: 2-4
    const periodLen = 2 + Math.floor(rng() * 3); // 2, 3, or 4
    const shuffled = [...PALETTE_NAMES];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const unit = shuffled.slice(0, periodLen);
    // Ensure no adjacent duplicates in unit
    for (let i = 1; i < unit.length; i++) {
      if (unit[i] === unit[i - 1]) {
        const alt = shuffled.find(c => c !== unit[i] && !unit.slice(Math.max(0, i - 1), i).includes(c));
        if (alt) unit[i] = alt;
      }
    }
    const colors = [];
    for (let i = 0; i < n; i++) {
      colors.push(unit[i % periodLen]);
    }
    return colors;
  }

  if (colorMode === 'similar') {
    // Pick from a similar-color group + fill remaining
    const group = SIMILAR_GROUPS[Math.floor(rng() * SIMILAR_GROUPS.length)];
    const colors = [];
    for (let i = 0; i < n; i++) {
      colors.push(group[Math.floor(rng() * group.length)]);
    }
    // Ensure no two adjacent are the same (swap if needed)
    for (let i = 1; i < colors.length; i++) {
      if (colors[i] === colors[i - 1]) {
        const alt = group.find(c => c !== colors[i]) || group[0];
        colors[i] = alt;
      }
    }
    return colors;
  }

  if (colorMode === 'mixed') {
    // Use full palette, allow near-duplicates
    const colors = [];
    for (let i = 0; i < n; i++) {
      colors.push(PALETTE_NAMES[Math.floor(rng() * PALETTE_NAMES.length)]);
    }
    // Avoid immediate adjacent duplicates
    for (let i = 1; i < colors.length; i++) {
      if (colors[i] === colors[i - 1]) {
        const others = PALETTE_NAMES.filter(c => c !== colors[i]);
        colors[i] = others[Math.floor(rng() * others.length)];
      }
    }
    return colors;
  }

  // 'distinct' — all different colors (up to palette size)
  const shuffled = [...PALETTE_NAMES];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  if (n <= shuffled.length) {
    return shuffled.slice(0, n);
  }
  // More beads than palette: cycle through
  const colors = [];
  for (let i = 0; i < n; i++) {
    colors.push(shuffled[i % shuffled.length]);
  }
  return colors;
}

// ============= Rope Mesh =============

/**
 * Create the rope (string) mesh as a TubeGeometry.
 */
export function createRopeMesh(curve, {
  thickness = 0.08,
  closed = false,
  color = ROPE_COLOR,
  tubularSegments = 200,
  radialSegments = 12,
} = {}) {
  const geometry = new THREE.TubeGeometry(
    curve,
    tubularSegments,
    thickness / 2,
    radialSegments,
    closed
  );
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness: 0.82,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ============= Bead Mesh =============

/**
 * Create a single bead (sphere) at a given position.
 */
export function createBeadMesh(position, colorName, beadSize = 0.35) {
  const colorInfo = BEAD_PALETTE[colorName] || BEAD_PALETTE.RED;
  const geometry = new THREE.SphereGeometry(beadSize, 32, 24);
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colorInfo.hex),
    roughness: 0.28,
    metalness: 0.12,
    envMapIntensity: 0.8,
    transparent: true,
    opacity: BEAD_OPACITY,
    depthWrite: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.beadColor = colorName;
  return mesh;
}

// ============= Start Marker =============

/**
 * Create a small marker ring around the first bead to indicate the start.
 */
export function createStartMarker(position, beadSize = 0.35) {
  const geometry = new THREE.TorusGeometry(beadSize * 1.35, beadSize * 0.08, 12, 32);
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.5,
    emissive: 0xffffff,
    emissiveIntensity: 0.3,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(position);
  return mesh;
}

// ============= Direction Arrow =============

/**
 * Create a tangential arrow near the first bead to indicate counting direction.
 * The arrow is placed just before the first bead and points along the tangent
 * (toward the second bead), so the viewer knows which direction to read.
 *
 * @param {THREE.Vector3} position - first bead position
 * @param {THREE.Vector3} tangent  - tangent vector at the first bead (pointing toward bead 2)
 * @param {number} beadSize
 * @returns {THREE.Group} arrow group (shaft + head)
 */
export function createDirectionArrow(position, tangent, beadSize = 0.35) {
  const arrowGroup = new THREE.Group();
  const dir = tangent.clone().normalize();

  // Arrow dimensions relative to bead size
  const headLength = beadSize * 0.8;
  const headRadius = beadSize * 0.35;
  const shaftLength = beadSize * 1.2;
  const shaftRadius = beadSize * 0.1;

  const headMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.4,
    emissive: 0xffffff,
    emissiveIntensity: 0.25,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const shaftMat = new THREE.MeshStandardMaterial({
    color: 0xd8dde6,
    roughness: 0.38,
    metalness: 0.25,
    emissive: 0xd8dde6,
    emissiveIntensity: 0.12,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 12),
    shaftMat,
  );
  shaft.position.set(0, -shaftLength / 2 - headLength / 2, 0);
  arrowGroup.add(shaft);

  const head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLength, 16), headMat);
  arrowGroup.add(head);

  // Orient: default +Y should align with tangent direction
  arrowGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

  // Position: offset to the side of the first bead (perpendicular to tangent)
  // so the arrow never collides with the rope or neighbouring beads.
  const sideRef = Math.abs(dir.y) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
  const side = new THREE.Vector3().crossVectors(dir, sideRef).normalize();
  arrowGroup.position.copy(position).addScaledVector(side, beadSize * 2.5);

  arrowGroup.renderOrder = 999;

  return arrowGroup;
}

// ============= Compose Full Bead String =============

/**
 * Create a complete bead string (rope + beads + start marker).
 *
 * @param {Object} config
 * @param {string} config.curveType - curve type name
 * @param {number} config.curveComplexity - 0..1
 * @param {number} config.numBeads - 3..10
 * @param {string[]} [config.beadColors] - explicit color sequence (overrides auto selection)
 * @param {string} [config.colorMode] - 'distinct' | 'mixed' | 'similar'
 * @param {number} [config.beadSize] - 0.2..0.8
 * @param {number} [config.ropeThickness] - 0.04..0.15
 * @param {string} [config.seed] - for reproducibility
 * @param {boolean} [config.isRing] - force ring/cyclic mode
 * @param {boolean} [config.showStartMarker] - show ring around first bead
 * @returns {{ group: THREE.Group, metadata: Object }}
 */
export function createBeadString(config = {}) {
  const {
    curveType = 'arc',
    curveComplexity = 0.3,
    numBeads = 5,
    beadColors: explicitColors = null,
    colorMode = 'distinct',
    beadSize = 0.35,
    ropeThickness = 0.08,
    seed = 'bead-default',
    isRing = false,
    showStartMarker = true,
  } = config;

  const rng = makeRng(seed);

  // Determine actual curve type. Curve types that are inherently closed
  // (rings and tangled loops) keep their identity even when isRing is set.
  const NATURALLY_CLOSED = new Set(['ring', 'wavy_ring', 'tangled_loop']);
  let actualCurveType = curveType;
  if (isRing && !NATURALLY_CLOSED.has(curveType)) {
    actualCurveType = curveComplexity > 0.4 ? 'wavy_ring' : 'ring';
  }

  // Create curve (returns both CatmullRom for TubeGeometry and parametric for sampling)
  const { curve, closed, parametricCurve } = createCurve(actualCurveType, curveComplexity, rng);

  // Create rope mesh using CatmullRomCurve3 (guaranteed TubeGeometry compatibility)
  const rope = createRopeMesh(curve, {
    thickness: ropeThickness,
    closed,
    tubularSegments: 200,
  });

  // Select colors
  const colors = explicitColors || selectBeadColors(numBeads, rng, colorMode);

  // Place beads along the parametric curve for precise arc-length positioning.
  // For closed curves we try several starting offsets and pick the one that
  // maximises the minimum 3-D distance between any two beads. On
  // self-crossing curves (tangled_loop, wavy_ring) the default offset can
  // land two beads at the same xy crossing, which renders as visual overlap
  // ("穿模") and hides the start-bead behind a neighbour.
  let positions;
  let tangents;
  if (closed && numBeads >= 2) {
    // Exhaustively search 36 offsets; pick the global max-min-distance. We
    // don't early-exit at a "good enough" threshold because the camera
    // projection can still merge beads that are 3-D-separated by ~1×
    // diameter when they happen to share an xy crossing.
    const trials = 36;
    const minBeadGap = beadSize * 3.0;
    let best = null;
    let bestMinDist = -Infinity;
    for (let k = 0; k < trials; k++) {
      const offset = k / trials;
      const cand = sampleEquidistantPoints(parametricCurve, numBeads, true, offset);
      let candMin = Infinity;
      for (let i = 0; i < cand.positions.length; i++) {
        for (let j = i + 1; j < cand.positions.length; j++) {
          const d = cand.positions[i].distanceTo(cand.positions[j]);
          if (d < candMin) candMin = d;
        }
      }
      if (candMin > bestMinDist) {
        bestMinDist = candMin;
        best = cand;
      }
    }
    ({ positions, tangents } = best);
    if (bestMinDist < minBeadGap) {
      console.warn(
        `[bead-renderer] best min inter-bead distance ${bestMinDist.toFixed(3)} ` +
        `< target ${minBeadGap.toFixed(3)} after ${trials} offset trials ` +
        `(curve=${actualCurveType}, n=${numBeads}, complexity=${curveComplexity})`
      );
    }
  } else {
    ({ positions, tangents } = sampleEquidistantPoints(parametricCurve, numBeads, closed));
  }

  const group = new THREE.Group();
  group.add(rope);

  const beadMeshes = [];
  for (let i = 0; i < numBeads; i++) {
    const bead = createBeadMesh(positions[i], colors[i], beadSize);
    group.add(bead);
    beadMeshes.push(bead);
  }

  // Start marker on the first bead
  if (showStartMarker && positions.length > 0) {
    const marker = createStartMarker(positions[0], beadSize);
    // Orient marker to face the tangent direction
    if (tangents[0]) {
      const up = new THREE.Vector3(0, 1, 0);
      const right = new THREE.Vector3().crossVectors(tangents[0], up).normalize();
      if (right.length() < 0.01) right.set(1, 0, 0);
      marker.lookAt(marker.position.clone().add(tangents[0]));
    }
    group.add(marker);

    // Direction arrow pointing along the tangent at the first bead
    if (tangents[0]) {
      const arrow = createDirectionArrow(positions[0], tangents[0], beadSize);
      group.add(arrow);
    }
  }

  // Center the group
  const bbox = new THREE.Box3().setFromObject(group);
  const center = bbox.getCenter(new THREE.Vector3());
  group.position.sub(center);

  const metadata = {
    bead_sequence: colors,
    bead_colors_hex: colors.map(c => BEAD_PALETTE[c]?.hex || '#FFFFFF'),
    num_beads: numBeads,
    curve_type: actualCurveType,
    curve_complexity: curveComplexity,
    bead_size: beadSize,
    rope_thickness: ropeThickness,
    is_ring: closed,
    seed,
    bead_positions: positions.map(p => [
      +p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4),
    ]),
  };

  return { group, metadata, beadMeshes, curve, closed };
}

export default {
  BEAD_PALETTE, PALETTE_NAMES,
  selectBeadColors, createRopeMesh, createBeadMesh,
  createStartMarker, createDirectionArrow, createBeadString,
};
