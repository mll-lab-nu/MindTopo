// @ts-nocheck
import * as THREE from "three";

const DEFAULT_CONFIG = Object.freeze({
  difficulty: "medium",
  seed: 0,
  animate: true,
  illegalReward: -1.0,
  moveReward: 0.1,
  stepPenalty: -0.01,
  winReward: 1.0,
  physicsFramesPerStep: 60,
  holeSpacing: 2.2,
  ropeRadius: 0.22,
  ropeSegments: 16,
  subSteps: 16,
  gravity: -0.12,
  friction: 0.9,
  groundFriction: 0.9,
  bendingStiffness: 0.2,
  compliance: 0.0001,
  minSlack: 1.15,
  tightenSpeed: 0.008,
  visualSmoothing: 0.15,
  sleepThreshold: 2e-6,
  liftClearance: 1.4,
  liftAndPlaceFrames: 24,
});

const DIFFICULTY_PRESETS = Object.freeze({
  easy: {
    gridSize: 5,
    ropeCount: 4,
    targetCrossings: [2, 3],
    scrambleSteps: [2, 3],
    minInvolved: 2,
    maxVisualLogicalGap: 0,
    maxDegree: 3,
    minCrossingSeparation: 1.8,
  },
  medium: {
    gridSize: 6,
    ropeCount: 5,
    targetCrossings: [5, 6],
    scrambleSteps: [5, 6],
    minInvolved: 4,
    maxVisualLogicalGap: 1,
    maxDegree: 3,
    minCrossingSeparation: 1.15,
  },
  hard: {
    gridSize: 6,
    ropeCount: 6,
    targetCrossings: [5, 7],
    scrambleSteps: [7, 9],
    minInvolved: 4,
    maxVisualLogicalGap: 2,
    maxDegree: 4,
    minCrossingSeparation: 0.45,
    maxCrossingSpread: 2.4,
    minLargestComponent: 3,
    maxLargestComponent: 4,
    minMultiCrossRopes: 3,
    maxMultiCrossRopes: 4,
    minCycleRank: 1,
    maxCycleRank: 2,
    searchMultiplier: 40,
  },
});

const ROPE_COLORS = [
  0xe53935, 0x43a047, 0x1e88e5, 0xfdd835,
  0x8e24aa, 0xfb8c00, 0x00acc1, 0xc0ca33,
];

function mulberry32(seed) {
  let state = (seed | 0) >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rngInt(rng, min, max) {
  return Math.floor(rng() * (max - min + 1)) + min;
}

function toFiniteInt(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.trunc(n);
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function direction2D(x1, y1, x2, y2, x3, y3) {
  return (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1);
}

function segmentsIntersect2D(a, b) {
  const d1 = direction2D(b.x1, b.y1, b.x2, b.y2, a.x1, a.y1);
  const d2 = direction2D(b.x1, b.y1, b.x2, b.y2, a.x2, a.y2);
  const d3 = direction2D(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1);
  const d4 = direction2D(a.x1, a.y1, a.x2, a.y2, b.x2, b.y2);
  return (
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
  );
}

// Width-aware crossing: true if 2D closest-point distance < threshold
function segmentsClose2D(a, b, threshold) {
  const ux = a.x2 - a.x1, uy = a.y2 - a.y1;
  const vx = b.x2 - b.x1, vy = b.y2 - b.y1;
  const wx = a.x1 - b.x1, wy = a.y1 - b.y1;
  const aa = ux * ux + uy * uy;
  const bb = ux * vx + uy * vy;
  const cc = vx * vx + vy * vy;
  const dd = ux * wx + uy * wy;
  const ee = vx * wx + vy * wy;
  const D = aa * cc - bb * bb;
  let sc, tc;
  if (D < 1e-8) { sc = 0; tc = (bb > cc ? dd / bb : ee / cc); }
  else { sc = (bb * ee - cc * dd) / D; tc = (aa * ee - bb * dd) / D; }
  sc = Math.max(0, Math.min(1, sc));
  tc = Math.max(0, Math.min(1, tc));
  const dx = a.x1 + ux * sc - (b.x1 + vx * tc);
  const dy = a.y1 + uy * sc - (b.y1 + vy * tc);
  return dx * dx + dy * dy < threshold * threshold;
}

export class UntangleGame {
  constructor() {
    this.config = { ...DEFAULT_CONFIG };
    this.gridSize = 4;
    this.ropeCount = 4;
    this.holes = [];
    this.ropes = [];
    this.stepCount = 0;
    this.lastMove = null;
    this.actionMap = [];
    this.reset({});
  }

  reset(incoming = {}) {
    this.config = this._normalizeConfig(incoming);
    const preset = DIFFICULTY_PRESETS[this.config.difficulty] || DIFFICULTY_PRESETS.medium;
    this.gridSize = preset.gridSize;
    this.ropeCount = preset.ropeCount;
    this.stepCount = 0;
    this.lastMove = null;

    const rng = mulberry32(this.config.seed);
    this._buildHoles();
    this._buildRopesWithShuffling(rng, preset);
    this._buildActionMap();

    for (let i = 0; i < 120; i += 1) this.tickPhysics(1 / 60);

    const crossings = this.crossings();
    return this._buildStateResponse({
      reward: 0,
      done: crossings === 0,
      success: crossings === 0,
      info: { reset: true, illegal: false, reason: null, crossings },
    });
  }

  step(action) {
    const decoded = this.decodeAction(action);
    if (!decoded) {
      return this._illegalResult({ reason: "invalid_action", action });
    }

    const validation = this._validateMove(
      decoded.ropeIdx,
      decoded.targetHoleIdx,
      decoded.sourceHoleIdx ?? null,
    );
    if (!validation.legal) {
      return this._illegalResult({ reason: validation.reason, action });
    }

    const rope = this.ropes[decoded.ropeIdx];
    const targetHole = this.holes[decoded.targetHoleIdx];
    const startParticle = rope.particles[0];
    const endParticle = rope.particles[rope.particles.length - 1];

    let isStartEnd;
    if (Number.isInteger(decoded.sourceHoleIdx)) {
      isStartEnd = rope.startHole.id === decoded.sourceHoleIdx;
    } else {
      const dStart = startParticle.pos.distanceTo(targetHole.pos);
      const dEnd = endParticle.pos.distanceTo(targetHole.pos);
      isStartEnd = dStart <= dEnd;
    }
    const sourceHole = isStartEnd ? rope.startHole : rope.endHole;

    const crossingsBefore = this.crossings();

    this._executeLiftAndPlaceMove(rope, isStartEnd, sourceHole, targetHole);

    const crossingsAfter = this.crossings();
    const solved = crossingsAfter === 0;
    this.stepCount += 1;
    this.lastMove = {
      actionIndex: decoded.actionIndex,
      ropeIdx: decoded.ropeIdx,
      targetHoleIdx: decoded.targetHoleIdx,
      sourceHoleIdx: sourceHole.id,
      isStartEnd,
      crossingsBefore,
      crossingsAfter,
      stepCount: this.stepCount,
    };

    const crossingDelta = crossingsBefore - crossingsAfter;
    const reward = solved
      ? this.config.winReward
      : this.config.stepPenalty + this.config.moveReward * crossingDelta;

    return this._buildStateResponse({
      reward,
      done: solved,
      success: solved,
      info: {
        illegal: false,
        reason: null,
        crossings: crossingsAfter,
        crossingsBefore,
        crossingsAfter,
        crossingDelta,
        ropeIdx: decoded.ropeIdx,
        targetHoleIdx: decoded.targetHoleIdx,
        sourceHoleIdx: sourceHole.id,
        solved,
      },
    });
  }

  getObservation() {
    const crossings = this.crossings();
    const visualCrossings = this.visualCrossings();
    const logicalCrossings = this.logicalCrossings();
    return {
      gridSize: this.gridSize,
      ropeCount: this.ropeCount,
      ropes: this.ropes.map((r, idx) => ({
        id: idx,
        color: r.color,
        startHole: { row: r.startHole.row, col: r.startHole.col, id: r.startHole.id },
        endHole: { row: r.endHole.row, col: r.endHole.col, id: r.endHole.id },
      })),
      occupiedHoles: this.holes.filter((h) => h.occupied).map((h) => h.id),
      crossings,
      visualCrossings,
      logicalCrossings,
      generatorInfo: this.generatorInfo ? structuredClone(this.generatorInfo) : null,
    };
  }

  getDebugState() {
    const crossings = this.crossings();
    const visualCrossings = this.visualCrossings();
    const logicalCrossings = this.logicalCrossings();
    return {
      gridSize: this.gridSize,
      ropeCount: this.ropeCount,
      stepCount: this.stepCount,
      crossings,
      visualCrossings,
      logicalCrossings,
      ropes: this.ropes.map((r, idx) => ({
        id: idx,
        color: r.color,
        startHole: { row: r.startHole.row, col: r.startHole.col, id: r.startHole.id },
        endHole: { row: r.endHole.row, col: r.endHole.col, id: r.endHole.id },
      })),
      holes: this.holes.map((h) => ({
        id: h.id,
        row: h.row,
        col: h.col,
        occupied: h.occupied,
      })),
      actionMap: this.actionMap.map(([ropeIdx, targetHoleIdx], index) => ({
        index,
        ropeIdx,
        targetHoleIdx,
      })),
      lastMove: this.lastMove ? { ...this.lastMove } : null,
      generatorInfo: this.generatorInfo ? structuredClone(this.generatorInfo) : null,
    };
  }

  getState() {
    const crossings = this.crossings();
    return this._buildStateResponse({
      reward: 0,
      done: crossings === 0,
      success: crossings === 0,
      info: { illegal: false, reason: null, snapshot: true, crossings },
    });
  }

  renderConfig() {
    return {
      ...this.config,
      gridSize: this.gridSize,
      ropeCount: this.ropeCount,
      actionCount: this.actionMap.length,
    };
  }

  isLegalMove(ropeIdx, targetHoleIdx) {
    return this._validateMove(ropeIdx, targetHoleIdx).legal;
  }

  decodeAction(action) {
    const numHoles = this.holes.length;
    if (Number.isInteger(action)) {
      if (action >= 0 && action < this.actionMap.length) {
        const [ropeIdx, targetHoleIdx] = this.actionMap[action];
        return { ropeIdx, targetHoleIdx, actionIndex: action };
      }
      const ropeIdx = Math.floor(action / numHoles);
      const targetHoleIdx = action % numHoles;
      if (ropeIdx >= 0 && ropeIdx < this.ropeCount && targetHoleIdx >= 0 && targetHoleIdx < numHoles) {
        return { ropeIdx, targetHoleIdx, actionIndex: action };
      }
      return null;
    }
    if (action && typeof action === "object") {
      const srcRow = toFiniteInt(action.src_row ?? action.source_row, -1);
      const srcCol = toFiniteInt(action.src_col ?? action.source_col, -1);
      const tgtRow = toFiniteInt(action.tgt_row ?? action.target_row, -1);
      const tgtCol = toFiniteInt(action.tgt_col ?? action.target_col, -1);
      if (srcRow >= 0 && srcCol >= 0 && tgtRow >= 0 && tgtCol >= 0) {
        const sourceHoleIdx = this._holeIdFromRowCol(srcRow, srcCol);
        const targetHoleIdx = this._holeIdFromRowCol(tgtRow, tgtCol);
        if (sourceHoleIdx >= 0 && targetHoleIdx >= 0) {
          const ropeIdx = this._findRopeIdxByHoleId(sourceHoleIdx);
          if (ropeIdx >= 0) {
            return {
              ropeIdx,
              targetHoleIdx,
              sourceHoleIdx,
              actionIndex: ropeIdx * numHoles + targetHoleIdx,
            };
          }
        }
      }
      const ropeIdx = toFiniteInt(action.ropeIdx ?? action.rope, -1);
      const targetHoleIdx = toFiniteInt(action.targetHoleIdx ?? action.hole, -1);
      if (ropeIdx >= 0 && targetHoleIdx >= 0) {
        return { ropeIdx, targetHoleIdx, actionIndex: ropeIdx * numHoles + targetHoleIdx };
      }
    }
    return null;
  }

  // ---------------- Physics (XPBD, ported from prototype) ----------------

  tickPhysics(dt) {
    const cfg = this.config;
    const subDt = dt / cfg.subSteps;
    this._detectAllCollisions();
    for (const rope of this.ropes) this._applyTightening(rope);
    for (let step = 0; step < cfg.subSteps; step += 1) {
      for (const rope of this.ropes) {
        this._applyGravity(rope, subDt);
        this._solveDistance(rope, subDt);
        this._solveBending(rope);
        this._solveGround(rope);
      }
      for (let iter = 0; iter < 2; iter += 1) this._solveRopeRopeCollisions();
      for (const rope of this.ropes) this._integrate(rope, subDt);
    }
    for (const rope of this.ropes) {
      rope.collisionPressure = 0;
      rope.canTighten = true;
    }
    for (const rope of this.ropes) this._smoothVisuals(rope);
  }

  // Canonical crossings: peg-to-peg straight-line intersections.
  // The puzzle is topological — physics enforces realistic moves, but success
  // is defined by whether the symbolic peg-line connections cross.
  logicalCrossings() {
    const segs = this.ropes.map((r) => ({
      x1: r.startHole.pos.x, y1: r.startHole.pos.z,
      x2: r.endHole.pos.x, y2: r.endHole.pos.z,
    }));
    let count = 0;
    for (let i = 0; i < segs.length; i += 1) {
      for (let j = i + 1; j < segs.length; j += 1) {
        if (segmentsIntersect2D(segs[i], segs[j])) count += 1;
      }
    }
    return count;
  }

  // Canonical visual crossings: true top-down intersections of simulated rope paths.
  // Counts each rope pair at most once, regardless of how many particle segments cross.
  visualCrossings() {
    const ropeSegs = this.ropes.map((rope) => {
      const segs = [];
      for (let i = 0; i < rope.particles.length - 1; i += 1) {
        const p1 = rope.particles[i].pos;
        const p2 = rope.particles[i + 1].pos;
        segs.push({ x1: p1.x, y1: p1.z, x2: p2.x, y2: p2.z });
      }
      return segs;
    });
    let count = 0;
    for (let i = 0; i < ropeSegs.length; i += 1) {
      for (let j = i + 1; j < ropeSegs.length; j += 1) {
        let pairCrossing = false;
        for (const sA of ropeSegs[i]) {
          if (pairCrossing) break;
          for (const sB of ropeSegs[j]) {
            if (segmentsIntersect2D(sA, sB)) {
              pairCrossing = true;
              break;
            }
          }
        }
        if (pairCrossing) count += 1;
      }
    }
    return count;
  }

  crossings() {
    return this.visualCrossings();
  }

  // ---------------- Internals ----------------

  _buildHoles() {
    this.holes = [];
    const offset = ((this.gridSize - 1) * this.config.holeSpacing) / 2;
    for (let row = 0; row < this.gridSize; row += 1) {
      for (let col = 0; col < this.gridSize; col += 1) {
        const x = row * this.config.holeSpacing - offset;
        const z = col * this.config.holeSpacing - offset;
        this.holes.push({
          id: this.holes.length,
          row,
          col,
          pos: new THREE.Vector3(x, 0, z),
          occupied: false,
        });
      }
    }
  }

  _buildRopesWithShuffling(rng, preset) {
    const templates = this._buildSolvedTemplateBank();
    const searchMultiplier = Number.isFinite(preset.searchMultiplier) ? preset.searchMultiplier : 8;
    const attempts = Math.max(96, templates.length * searchMultiplier);
    const acceptedCandidates = [];
    const acceptedSignatures = new Set();
    let bestCandidate = null;

    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const template = templates[rngInt(rng, 0, templates.length - 1)];
      let endpoints = this._cloneEndpoints(template.endpoints);
      const history = [];
      const targetSteps = rngInt(rng, preset.scrambleSteps[0], preset.scrambleSteps[1]);
      let completedSteps = 0;

      for (let stepIndex = 0; stepIndex < targetSteps; stepIndex += 1) {
        const currentMetrics = this._endpointLayoutMetrics(endpoints);
        const candidateEntries = [];
        for (const move of this._enumerateEndpointMoves(endpoints)) {
          if (this._isImmediateUndoMove(move, history)) continue;
          const nextEndpoints = this._applyEndpointMove(endpoints, move);
          const nextMetrics = this._endpointLayoutMetrics(nextEndpoints);
          if (!this._isAcceptableIntermediateMetrics(nextMetrics, preset, stepIndex, targetSteps)) {
            continue;
          }
          const score = this._scrambleMoveScore(
            currentMetrics,
            nextMetrics,
            preset,
            stepIndex,
            targetSteps,
            move,
            history,
          );
          candidateEntries.push({ move, nextEndpoints, nextMetrics, score });
        }
        if (!candidateEntries.length) break;
        const chosen = this._chooseWeightedCandidate(candidateEntries, rng);
        endpoints = chosen.nextEndpoints;
        history.push(chosen.move);
        completedSteps += 1;
      }

      const finalMetrics = this._endpointLayoutMetrics(endpoints);
      const candidate = {
        templateName: template.name,
        endpoints,
        history,
        metrics: finalMetrics,
        signature: this._endpointSignature(endpoints),
        targetSteps,
        completedSteps,
      };
      if (this._isFinalCandidateBetter(candidate, bestCandidate, preset)) {
        bestCandidate = candidate;
      }
      if (!this._isAcceptableFinalCandidate(candidate, preset)) continue;
      if (acceptedSignatures.has(candidate.signature)) continue;
      acceptedSignatures.add(candidate.signature);
      acceptedCandidates.push(candidate);
    }

    const chosen = acceptedCandidates.length
      ? acceptedCandidates[rngInt(rng, 0, acceptedCandidates.length - 1)]
      : bestCandidate || {
      templateName: "fallback/parallel_rows",
      endpoints: this._defaultSolvedEndpoints(),
      history: [],
      metrics: this._endpointLayoutMetrics(this._defaultSolvedEndpoints()),
      signature: this._endpointSignature(this._defaultSolvedEndpoints()),
      targetSteps: 0,
      completedSteps: 0,
    };
    this._materializeEndpoints(chosen.endpoints);
    if (this.config.difficulty === "hard") {
      this._seedHardCentralTangle(chosen.metrics, rng);
    }
    this.generatorInfo = {
      templateName: chosen.templateName,
      signature: chosen.signature,
      scrambleSteps: chosen.history.length,
      targetSteps: chosen.targetSteps,
      acceptedCandidateCount: acceptedCandidates.length,
      metrics: chosen.metrics,
    };
  }

  _defaultSolvedEndpoints() {
    const endpoints = [];
    for (let i = 0; i < this.ropeCount; i += 1) {
      const holeA = this._holeAt(i, 0);
      const holeB = this._holeAt(i, this.gridSize - 1);
      endpoints.push([holeA.id, holeB.id]);
    }
    return endpoints;
  }

  _buildSolvedTemplateBank() {
    const combinations = this._lineIndexCombinations();
    const ropeOrders = this._ropeOrderVariants();
    const templates = [];
    const seen = new Set();
    const baseTemplates = [];

    for (const combo of combinations) {
      baseTemplates.push({
        name: `rows:${combo.join("-")}`,
        endpoints: this._buildParallelTemplateEndpoints("rows", combo),
      });
      baseTemplates.push({
        name: `cols:${combo.join("-")}`,
        endpoints: this._buildParallelTemplateEndpoints("cols", combo),
      });
    }

    for (const base of baseTemplates) {
      for (const order of ropeOrders) {
        const endpoints = order.map((index) => [...base.endpoints[index]]);
        const signature = this._endpointSignature(endpoints);
        if (seen.has(signature)) continue;
        seen.add(signature);
        templates.push({
          name: `${base.name}/${order.join("-")}`,
          endpoints,
        });
      }
    }

    if (!templates.length) {
      templates.push({
        name: "rows:fallback",
        endpoints: this._defaultSolvedEndpoints(),
      });
    }
    return templates;
  }

  _lineIndexCombinations() {
    const combinations = [];
    const build = (start, acc) => {
      if (acc.length === this.ropeCount) {
        combinations.push([...acc]);
        return;
      }
      const remaining = this.ropeCount - acc.length;
      for (let value = start; value <= this.gridSize - remaining; value += 1) {
        acc.push(value);
        build(value + 1, acc);
        acc.pop();
      }
    };
    build(0, []);
    return combinations;
  }

  _ropeOrderVariants() {
    const variants = [];
    const seen = new Set();
    const addVariant = (order) => {
      const signature = order.join("-");
      if (seen.has(signature)) return;
      seen.add(signature);
      variants.push(order);
    };
    const base = [...Array(this.ropeCount).keys()];
    addVariant(base);
    addVariant([...base].reverse());
    for (let shift = 1; shift < this.ropeCount; shift += 1) {
      addVariant(base.map((_, idx) => (idx + shift) % this.ropeCount));
    }
    const evensFirst = base.filter((idx) => idx % 2 === 0).concat(base.filter((idx) => idx % 2 === 1));
    addVariant(evensFirst);
    const endsIn = [];
    let lo = 0;
    let hi = this.ropeCount - 1;
    while (lo <= hi) {
      endsIn.push(lo);
      if (lo !== hi) endsIn.push(hi);
      lo += 1;
      hi -= 1;
    }
    addVariant(endsIn);
    return variants;
  }

  _buildParallelTemplateEndpoints(axis, lineIndices) {
    return lineIndices.map((lineIndex) => {
      if (axis === "cols") {
        return [this._holeAt(0, lineIndex).id, this._holeAt(this.gridSize - 1, lineIndex).id];
      }
      return [this._holeAt(lineIndex, 0).id, this._holeAt(lineIndex, this.gridSize - 1).id];
    });
  }

  _cloneEndpoints(endpoints) {
    return endpoints.map(([a, b]) => [a, b]);
  }

  _endpointSignature(endpoints) {
    return endpoints
      .map(([a, b], ropeIdx) => {
        const [left, right] = a < b ? [a, b] : [b, a];
        return `${ropeIdx}:${left}-${right}`;
      })
      .join("|");
  }

  _materializeEndpoints(endpoints) {
    this.ropes = [];
    for (const h of this.holes) h.occupied = false;
    for (let i = 0; i < this.ropeCount; i += 1) {
      const [startHoleId, endHoleId] = endpoints[i];
      this.ropes.push(
        this._createRope(startHoleId, endHoleId, ROPE_COLORS[i % ROPE_COLORS.length])
      );
    }
  }

  _seedHardCentralTangle(metrics, rng) {
    const dominantComponent = Array.isArray(metrics?.dominantComponent) ? metrics.dominantComponent : [];
    const candidatePool = dominantComponent.length >= 3
      ? [...dominantComponent]
      : [...Array(this.ropes.length).keys()];
    if (candidatePool.length < 3) return;

    const center = new THREE.Vector3(0, 0.85, 0);
    const clusterRadius = this.config.holeSpacing * 0.24;
    const shoulderRadius = this.config.holeSpacing * 0.52;
    const order = [...candidatePool];
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = rngInt(rng, 0, i);
      [order[i], order[j]] = [order[j], order[i]];
    }
    const clusterSize = Math.min(order.length, rng() < 0.6 ? 4 : 3);
    order.length = clusterSize;

    order.forEach((ropeIdx, orderIdx) => {
      const rope = this.ropes[ropeIdx];
      if (!rope) return;

      const start = rope.startHole.pos.clone();
      const end = rope.endHole.pos.clone();
      const dirStart = center.clone().sub(start).setY(0).normalize();
      const dirEnd = center.clone().sub(end).setY(0).normalize();
      const planar = end.clone().sub(start).setY(0);
      const tangent = planar.lengthSq() > 1e-6
        ? new THREE.Vector3(-planar.z, 0, planar.x).normalize()
        : new THREE.Vector3(1, 0, 0);
      const angle = (orderIdx / Math.max(1, order.length)) * Math.PI * 2;
      const swirl = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
      const sideSign = orderIdx % 2 === 0 ? 1 : -1;

      const startShoulder = start.clone().lerp(center, 0.42);
      startShoulder.y = 0.9;
      startShoulder.add(dirStart.clone().multiplyScalar(-0.3));

      const knotEntry = center.clone()
        .add(dirStart.clone().multiplyScalar(shoulderRadius))
        .add(tangent.clone().multiplyScalar(sideSign * clusterRadius * 0.55))
        .add(swirl.clone().multiplyScalar(clusterRadius * 0.35));
      knotEntry.y = 1.15 + 0.08 * sideSign;

      const knotCore = center.clone()
        .add(tangent.clone().multiplyScalar(sideSign * clusterRadius * 0.42))
        .add(swirl.clone().multiplyScalar(clusterRadius * 0.24));
      knotCore.y = 0.72 + 0.12 * sideSign;

      const knotExit = center.clone()
        .add(dirEnd.clone().multiplyScalar(shoulderRadius))
        .add(tangent.clone().multiplyScalar(-sideSign * clusterRadius * 0.4))
        .add(swirl.clone().multiplyScalar(-clusterRadius * 0.28));
      knotExit.y = 1.05 - 0.05 * sideSign;

      const endShoulder = end.clone().lerp(center, 0.4);
      endShoulder.y = 0.86;
      endShoulder.add(dirEnd.clone().multiplyScalar(-0.2));

      const curve = new THREE.CatmullRomCurve3(
        [start, startShoulder, knotEntry, knotCore, knotExit, endShoulder, end],
        false,
        "centripetal",
        0.6,
      );

      let curveLength = 0;
      const sampled = [];
      let prevPoint = null;
      for (let i = 0; i <= this.config.ropeSegments; i += 1) {
        const point = curve.getPoint(i / this.config.ropeSegments);
        sampled.push(point);
        if (prevPoint) curveLength += prevPoint.distanceTo(point);
        prevPoint = point;
      }
      rope.segmentLength = Math.max(
        rope.segmentLength,
        (curveLength * 1.02) / this.config.ropeSegments,
      );

      for (let i = 0; i < sampled.length; i += 1) {
        rope.particles[i].pos.copy(sampled[i]);
        rope.particles[i].oldPos.copy(sampled[i]);
        rope.visualPositions[i].copy(sampled[i]);
      }
      rope.particles[0].pos.copy(start);
      rope.particles[0].oldPos.copy(start);
      rope.particles[rope.particles.length - 1].pos.copy(end);
      rope.particles[rope.particles.length - 1].oldPos.copy(end);
      rope.visualPositions[0].copy(start);
      rope.visualPositions[rope.visualPositions.length - 1].copy(end);
    });
  }

  _enumerateEndpointMoves(endpoints) {
    const occupied = new Set(endpoints.flat());
    const moves = [];
    for (let ropeIdx = 0; ropeIdx < endpoints.length; ropeIdx += 1) {
      for (let endIdx = 0; endIdx < 2; endIdx += 1) {
        const sourceHole = endpoints[ropeIdx][endIdx];
        const otherEnd = endpoints[ropeIdx][1 - endIdx];
        for (const hole of this.holes) {
          if (hole.id === otherEnd) continue;
          if (occupied.has(hole.id)) continue;
          moves.push({
            ropeIdx,
            endIdx,
            sourceHole,
            targetHole: hole.id,
          });
        }
      }
    }
    return moves;
  }

  _applyEndpointMove(endpoints, move) {
    const nextEndpoints = this._cloneEndpoints(endpoints);
    nextEndpoints[move.ropeIdx][move.endIdx] = move.targetHole;
    return nextEndpoints;
  }

  _isImmediateUndoMove(move, history) {
    if (!history.length) return false;
    const prev = history[history.length - 1];
    return (
      prev.ropeIdx === move.ropeIdx &&
      prev.endIdx === move.endIdx &&
      prev.sourceHole === move.targetHole &&
      prev.targetHole === move.sourceHole
    );
  }

  _scrambleMoveScore(currentMetrics, nextMetrics, preset, stepIndex, targetSteps, move, history) {
    const progress = (stepIndex + 1) / Math.max(1, targetSteps);
    const targetMid = preset.targetCrossings[0] + (preset.targetCrossings[1] - preset.targetCrossings[0]) * progress;
    const currentDistance = Math.abs(currentMetrics.visual - targetMid);
    const nextDistance = Math.abs(nextMetrics.visual - targetMid);
    const readabilityPenalty = this._readabilityPenalty(nextMetrics, preset);
    const sameRopePenalty = history.length && history[history.length - 1].ropeIdx === move.ropeIdx ? 8 : 0;
    const sameEndpointPenalty = history.length && history[history.length - 1].ropeIdx === move.ropeIdx && history[history.length - 1].endIdx === move.endIdx ? 14 : 0;
    const compactClusterBonus = nextMetrics.visual > 0
      ? (nextMetrics.visual * 20) / Math.max(0.6, nextMetrics.crossingSpread)
      : 0;
    const tangleBonus = (
      nextMetrics.largestComponent * 22
      + nextMetrics.multiCrossRopes * 28
      + nextMetrics.cycleRank * 60
      + compactClusterBonus
    );
    return (
      (currentDistance - nextDistance) * 35
      + nextMetrics.visual * 24
      + nextMetrics.logical * 10
      + nextMetrics.involved * 8
      + tangleBonus
      + nextMetrics.minRopeLength * 8
      - readabilityPenalty
      - sameRopePenalty
      - sameEndpointPenalty
    );
  }

  _chooseWeightedCandidate(candidateEntries, rng) {
    const scores = candidateEntries.map((entry) => entry.score);
    const minScore = Math.min(...scores);
    const weights = candidateEntries.map((entry) => Math.max(1, entry.score - minScore + 1));
    const totalWeight = weights.reduce((sum, value) => sum + value, 0);
    let target = rng() * totalWeight;
    for (let index = 0; index < candidateEntries.length; index += 1) {
      target -= weights[index];
      if (target <= 0) return candidateEntries[index];
    }
    return candidateEntries[candidateEntries.length - 1];
  }

  _finalCandidateQuality(candidate, preset) {
    const metrics = candidate.metrics;
    const targetMid = (preset.targetCrossings[0] + preset.targetCrossings[1]) / 2;
    let quality = 0;
    const inRange = metrics.visual >= preset.targetCrossings[0] && metrics.visual <= preset.targetCrossings[1];
    const compactClusterBonus = metrics.visual > 0
      ? (metrics.visual * 36) / Math.max(0.6, metrics.crossingSpread)
      : 0;
    if (inRange) quality += 5000;
    quality -= Math.abs(metrics.visual - targetMid) * 180;
    quality += candidate.history.length * 120;
    quality += metrics.involved * 30;
    quality += metrics.largestComponent * 80;
    quality += metrics.multiCrossRopes * 110;
    quality += metrics.cycleRank * 240;
    quality += compactClusterBonus;
    quality += metrics.minRopeLength * 20;
    quality -= this._readabilityPenalty(metrics, preset);
    return quality;
  }

  _isAcceptableIntermediateMetrics(metrics, preset, stepIndex, targetSteps) {
    const progress = (stepIndex + 1) / Math.max(1, targetSteps);
    const softMaxCrossings = preset.targetCrossings[1] + (progress < 0.7 ? 2 : 1);
    const softMaxDegree = preset.maxDegree + (progress < 0.7 ? 1 : 0);
    const softGap = preset.maxVisualLogicalGap + 1;
    const softSeparation = preset.minCrossingSeparation * (progress < 0.5 ? 0.55 : 0.7);
    const maxCrossingSpread = preset.maxCrossingSpread;
    const minLargestComponent = preset.minLargestComponent || 0;
    const maxLargestComponent = preset.maxLargestComponent;
    const minMultiCrossRopes = preset.minMultiCrossRopes || 0;
    const maxMultiCrossRopes = preset.maxMultiCrossRopes;
    const minCycleRank = preset.minCycleRank || 0;
    const maxCycleRank = preset.maxCycleRank;
    const softLargestComponent = progress < 0.45
      ? 0
      : progress < 0.7
        ? Math.max(0, minLargestComponent - 3)
        : Math.max(0, minLargestComponent - 1);
    const softMultiCrossRopes = progress < 0.45
      ? 0
      : progress < 0.7
        ? Math.max(0, minMultiCrossRopes - 3)
        : Math.max(0, minMultiCrossRopes - 1);
    const softCycleRank = progress < 0.8 ? 0 : minCycleRank;
    const softLargestComponentMax = Number.isFinite(maxLargestComponent)
      ? maxLargestComponent + (progress < 0.7 ? 2 : 1)
      : Infinity;
    const softMultiCrossRopesMax = Number.isFinite(maxMultiCrossRopes)
      ? maxMultiCrossRopes + (progress < 0.7 ? 2 : 1)
      : Infinity;
    const softCycleRankMax = Number.isFinite(maxCycleRank)
      ? maxCycleRank + 1
      : Infinity;
    const softCrossingSpreadMax = Number.isFinite(maxCrossingSpread)
      ? maxCrossingSpread + (progress < 0.7 ? 1.4 : 0.8)
      : Infinity;
    return (
      metrics.visual <= softMaxCrossings &&
      metrics.maxDegree <= softMaxDegree &&
      metrics.visualLogicalGap <= softGap &&
      metrics.minCrossingSeparation >= softSeparation &&
      metrics.crossingSpread <= softCrossingSpreadMax &&
      metrics.largestComponent >= softLargestComponent &&
      metrics.largestComponent <= softLargestComponentMax &&
      metrics.multiCrossRopes >= softMultiCrossRopes &&
      metrics.multiCrossRopes <= softMultiCrossRopesMax &&
      metrics.cycleRank >= softCycleRank &&
      metrics.cycleRank <= softCycleRankMax
    );
  }

  _isAcceptableFinalCandidate(candidate, preset) {
    const metrics = candidate.metrics;
    const minLargestComponent = preset.minLargestComponent || 0;
    const maxLargestComponent = preset.maxLargestComponent;
    const minMultiCrossRopes = preset.minMultiCrossRopes || 0;
    const maxMultiCrossRopes = preset.maxMultiCrossRopes;
    const minCycleRank = preset.minCycleRank || 0;
    const maxCycleRank = preset.maxCycleRank;
    const maxCrossingSpread = preset.maxCrossingSpread;
    return (
      candidate.completedSteps >= candidate.targetSteps &&
      metrics.visual >= preset.targetCrossings[0] &&
      metrics.visual <= preset.targetCrossings[1] &&
      metrics.involved >= preset.minInvolved &&
      metrics.maxDegree <= preset.maxDegree &&
      metrics.visualLogicalGap <= preset.maxVisualLogicalGap &&
      metrics.minCrossingSeparation >= preset.minCrossingSeparation &&
      metrics.crossingSpread <= (Number.isFinite(maxCrossingSpread) ? maxCrossingSpread : Infinity) &&
      metrics.largestComponent >= minLargestComponent &&
      metrics.largestComponent <= (Number.isFinite(maxLargestComponent) ? maxLargestComponent : Infinity) &&
      metrics.multiCrossRopes >= minMultiCrossRopes &&
      metrics.multiCrossRopes <= (Number.isFinite(maxMultiCrossRopes) ? maxMultiCrossRopes : Infinity) &&
      metrics.cycleRank >= minCycleRank &&
      metrics.cycleRank <= (Number.isFinite(maxCycleRank) ? maxCycleRank : Infinity)
    );
  }

  _isFinalCandidateBetter(candidate, incumbent, preset) {
    if (!incumbent) return true;
    return this._finalCandidateQuality(candidate, preset) > this._finalCandidateQuality(incumbent, preset);
  }

  _readabilityPenalty(metrics, preset) {
    let penalty = 0;
    penalty += Math.max(0, metrics.maxDegree - preset.maxDegree) * 260;
    penalty += Math.max(0, metrics.visualLogicalGap - preset.maxVisualLogicalGap) * 320;
    if (metrics.minCrossingSeparation < preset.minCrossingSeparation) {
      penalty += (preset.minCrossingSeparation - metrics.minCrossingSeparation) * 260;
    }
    if (Number.isFinite(preset.maxCrossingSpread) && metrics.crossingSpread > preset.maxCrossingSpread) {
      penalty += (metrics.crossingSpread - preset.maxCrossingSpread) * 240;
    }
    if (Number.isFinite(preset.maxLargestComponent) && metrics.largestComponent > preset.maxLargestComponent) {
      penalty += (metrics.largestComponent - preset.maxLargestComponent) * 220;
    }
    if (Number.isFinite(preset.maxMultiCrossRopes) && metrics.multiCrossRopes > preset.maxMultiCrossRopes) {
      penalty += (metrics.multiCrossRopes - preset.maxMultiCrossRopes) * 180;
    }
    if (Number.isFinite(preset.maxCycleRank) && metrics.cycleRank > preset.maxCycleRank) {
      penalty += (metrics.cycleRank - preset.maxCycleRank) * 320;
    }
    return penalty;
  }

  _endpointLayoutMetrics(endpoints) {
    const threshold = Math.max(this.config.ropeRadius * 2, this.config.holeSpacing * 0.28);
    const segs = this._endpointSegments(endpoints);
    let visual = 0;
    let logical = 0;
    const involved = new Set();
    const degrees = new Array(segs.length).fill(0);
    const adjacency = Array.from({ length: segs.length }, () => new Set());
    const crossPoints = [];
    let minRopeLength = Infinity;

    for (let i = 0; i < segs.length; i += 1) {
      const dx = segs[i].x2 - segs[i].x1;
      const dy = segs[i].y2 - segs[i].y1;
      minRopeLength = Math.min(minRopeLength, Math.sqrt(dx * dx + dy * dy));
      for (let j = i + 1; j < segs.length; j += 1) {
        const visualPair = segmentsClose2D(segs[i], segs[j], threshold);
        if (visualPair) {
          visual += 1;
          involved.add(i);
          involved.add(j);
          degrees[i] += 1;
          degrees[j] += 1;
          adjacency[i].add(j);
          adjacency[j].add(i);
        }
        if (segmentsIntersect2D(segs[i], segs[j])) {
          logical += 1;
          const point = this._intersectionPoint2D(segs[i], segs[j]);
          if (point) crossPoints.push(point);
        }
      }
    }

    let minCrossingSeparation = this.config.holeSpacing * 4;
    let crossingSpread = 0;
    if (crossPoints.length >= 2) {
      minCrossingSeparation = Infinity;
      for (let i = 0; i < crossPoints.length; i += 1) {
        for (let j = i + 1; j < crossPoints.length; j += 1) {
          const dx = crossPoints[i].x - crossPoints[j].x;
          const dy = crossPoints[i].y - crossPoints[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          minCrossingSeparation = Math.min(minCrossingSeparation, distance);
          crossingSpread = Math.max(crossingSpread, distance);
        }
      }
    }

    let largestComponent = 0;
    let componentCount = 0;
    let cycleRank = 0;
    let dominantComponent = [];
    const visited = new Array(segs.length).fill(false);
    for (let i = 0; i < segs.length; i += 1) {
      if (visited[i] || !adjacency[i].size) continue;
      const stack = [i];
      const members = [];
      let componentSize = 0;
      let componentEdgeDegree = 0;
      visited[i] = true;
      while (stack.length) {
        const node = stack.pop();
        members.push(node);
        componentSize += 1;
        componentEdgeDegree += adjacency[node].size;
        for (const neighbor of adjacency[node]) {
          if (visited[neighbor]) continue;
          visited[neighbor] = true;
          stack.push(neighbor);
        }
      }
      const componentEdges = componentEdgeDegree / 2;
      componentCount += 1;
      if (componentSize > largestComponent) {
        largestComponent = componentSize;
        dominantComponent = members;
      }
      cycleRank += Math.max(0, componentEdges - componentSize + 1);
    }
    const multiCrossRopes = degrees.filter((degree) => degree >= 2).length;

    return {
      visual,
      logical,
      involved: involved.size,
      maxDegree: Math.max(0, ...degrees),
      componentCount,
      largestComponent,
      multiCrossRopes,
      cycleRank,
      dominantComponent,
      visualLogicalGap: Math.max(0, visual - logical),
      minCrossingSeparation,
      crossingSpread,
      minRopeLength: Number.isFinite(minRopeLength) ? minRopeLength : 0,
    };
  }

  _endpointSegments(endpoints) {
    const segs = endpoints.map(([a, b]) => {
      const holeA = this.holes[a];
      const holeB = this.holes[b];
      return {
        x1: holeA.pos.x, y1: holeA.pos.z,
        x2: holeB.pos.x, y2: holeB.pos.z,
      };
    });
    return segs;
  }

  _intersectionPoint2D(a, b) {
    const denom = ((a.x1 - a.x2) * (b.y1 - b.y2)) - ((a.y1 - a.y2) * (b.x1 - b.x2));
    if (Math.abs(denom) < 1e-8) return null;
    const detA = a.x1 * a.y2 - a.y1 * a.x2;
    const detB = b.x1 * b.y2 - b.y1 * b.x2;
    return {
      x: (detA * (b.x1 - b.x2) - (a.x1 - a.x2) * detB) / denom,
      y: (detA * (b.y1 - b.y2) - (a.y1 - a.y2) * detB) / denom,
    };
  }

  _holeAt(row, col) {
    return this.holes[row * this.gridSize + col];
  }

  _holeIdFromRowCol(row, col) {
    if (!Number.isInteger(row) || !Number.isInteger(col)) return -1;
    if (row < 0 || row >= this.gridSize || col < 0 || col >= this.gridSize) return -1;
    return row * this.gridSize + col;
  }

  _findRopeIdxByHoleId(holeId) {
    for (let ropeIdx = 0; ropeIdx < this.ropes.length; ropeIdx += 1) {
      const rope = this.ropes[ropeIdx];
      if (rope.startHole.id === holeId || rope.endHole.id === holeId) return ropeIdx;
    }
    return -1;
  }

  _lineCrossings() {
    const segs = this.ropes.map((r, idx) => ({
      x1: r.startHole.pos.x,
      y1: r.startHole.pos.z,
      x2: r.endHole.pos.x,
      y2: r.endHole.pos.z,
      ropeIdx: idx,
    }));
    let count = 0;
    for (let i = 0; i < segs.length; i += 1) {
      for (let j = i + 1; j < segs.length; j += 1) {
        if (segmentsIntersect2D(segs[i], segs[j])) count += 1;
      }
    }
    return count;
  }

  _createRope(startHoleId, endHoleId, color) {
    const startHole = this.holes[startHoleId];
    const endHole = this.holes[endHoleId];
    startHole.occupied = true;
    endHole.occupied = true;

    const rope = {
      particles: [],
      visualPositions: [],
      color,
      startHole,
      endHole,
      segmentLength: 0,
      collisionPressure: 0,
      canTighten: true,
      isDragging: false,
    };

    const dist = startHole.pos.distanceTo(endHole.pos);
    const slackScale = this.config.difficulty === "hard" ? 1.55 : 1.3;
    const slackAdd = this.config.difficulty === "hard" ? 3.0 : 1.5;
    const totalLen = Math.max(dist * slackScale, dist + slackAdd);
    rope.segmentLength = totalLen / this.config.ropeSegments;

    for (let i = 0; i <= this.config.ropeSegments; i += 1) {
      const t = i / this.config.ropeSegments;
      const p = new THREE.Vector3().lerpVectors(startHole.pos, endHole.pos, t);
      p.y = Math.sin(t * Math.PI) * 1.5 + 0.5;
      rope.particles.push({
        pos: p.clone(),
        oldPos: p.clone(),
        invMass: 1.0,
        radius: this.config.ropeRadius,
      });
      rope.visualPositions.push(p.clone());
    }
    rope.particles[0].invMass = 0;
    rope.particles[rope.particles.length - 1].invMass = 0;
    return rope;
  }

  _buildActionMap() {
    this.actionMap = [];
    for (let r = 0; r < this.ropeCount; r += 1) {
      for (let h = 0; h < this.holes.length; h += 1) {
        this.actionMap.push([r, h]);
      }
    }
  }

  _validateMove(ropeIdx, targetHoleIdx, sourceHoleIdx = null) {
    if (!Number.isInteger(ropeIdx) || !Number.isInteger(targetHoleIdx)) {
      return { legal: false, reason: "non_integer" };
    }
    if (ropeIdx < 0 || ropeIdx >= this.ropeCount) {
      return { legal: false, reason: "rope_out_of_range" };
    }
    if (targetHoleIdx < 0 || targetHoleIdx >= this.holes.length) {
      return { legal: false, reason: "hole_out_of_range" };
    }
    const rope = this.ropes[ropeIdx];
    const targetHole = this.holes[targetHoleIdx];
    if (Number.isInteger(sourceHoleIdx)) {
      if (rope.startHole.id !== sourceHoleIdx && rope.endHole.id !== sourceHoleIdx) {
        return { legal: false, reason: "source_not_endpoint" };
      }
    }
    if (rope.startHole.id === targetHoleIdx || rope.endHole.id === targetHoleIdx) {
      return { legal: false, reason: "target_is_current_hole" };
    }
    if (targetHole.occupied) {
      return { legal: false, reason: "target_occupied" };
    }
    return { legal: true, reason: null };
  }

  _normalizeConfig(incoming) {
    const merged = { ...DEFAULT_CONFIG, ...this.config, ...(incoming || {}) };
    let difficulty =
      typeof merged.difficulty === "string" ? merged.difficulty.toLowerCase() : "medium";
    if (!DIFFICULTY_PRESETS[difficulty]) difficulty = "medium";
    return {
      ...merged,
      difficulty,
      seed: toFiniteInt(merged.seed, 0),
      animate: Boolean(merged.animate),
      illegalReward: Number(merged.illegalReward),
      moveReward: Number(merged.moveReward),
      stepPenalty: Number(merged.stepPenalty),
      winReward: Number(merged.winReward),
      physicsFramesPerStep: Math.max(1, toFiniteInt(merged.physicsFramesPerStep, 60)),
      liftClearance: Math.max(
        this.config.ropeRadius * 3,
        toFiniteNumber(merged.liftClearance, DEFAULT_CONFIG.liftClearance),
      ),
      liftAndPlaceFrames: Math.max(3, toFiniteInt(merged.liftAndPlaceFrames, 24)),
    };
  }

  _buildStateResponse({ reward, done, success, info }) {
    return {
      observation: this.getObservation(),
      reward,
      done,
      success,
      step_count: this.stepCount,
      info: {
        ...(info || {}),
        state: this.getDebugState(),
      },
    };
  }

  _illegalResult(payload) {
    return this._buildStateResponse({
      reward: this.config.illegalReward,
      done: false,
      success: false,
      info: {
        illegal: true,
        ...payload,
        solved: this.crossings() === 0,
      },
    });
  }

  _executeLiftAndPlaceMove(rope, isStartEnd, sourceHole, targetHole) {
    const movingParticle = isStartEnd
      ? rope.particles[0]
      : rope.particles[rope.particles.length - 1];
    const sourcePos = this._holeSurfacePosition(sourceHole);
    const targetPos = this._holeSurfacePosition(targetHole);
    const liftY = this._liftPlaneY();
    const sourceLifted = sourcePos.clone().setY(liftY);
    const targetLifted = targetPos.clone().setY(liftY);

    sourceHole.occupied = false;
    targetHole.occupied = true;
    if (isStartEnd) rope.startHole = targetHole;
    else rope.endHole = targetHole;

    const totalFrames = Math.max(3, this.config.liftAndPlaceFrames);
    const liftFrames = Math.max(1, Math.floor(totalFrames * 0.25));
    const traverseFrames = Math.max(1, Math.floor(totalFrames * 0.5));
    const lowerFrames = Math.max(1, totalFrames - liftFrames - traverseFrames);

    const wasDragging = rope.isDragging;
    this.setDragState(rope, true);
    try {
      this._dragPinnedParticle(movingParticle, sourcePos, sourceLifted, liftFrames);
      this._dragPinnedParticle(movingParticle, sourceLifted, targetLifted, traverseFrames);
      this._dragPinnedParticle(movingParticle, targetLifted, targetPos, lowerFrames);
    } finally {
      this.setDragState(rope, wasDragging);
    }

    for (let i = 0; i < this.config.physicsFramesPerStep; i += 1) {
      this.tickPhysics(1 / 60);
    }
  }

  setDragState(rope, isDragging) {
    if (rope) rope.isDragging = Boolean(isDragging);
  }

  _dragPinnedParticle(particle, fromPos, toPos, frames) {
    const frameCount = Math.max(1, frames);
    for (let f = 1; f <= frameCount; f += 1) {
      const t = f / frameCount;
      particle.pos.lerpVectors(fromPos, toPos, t);
      particle.oldPos.copy(particle.pos);
      this.tickPhysics(1 / 60);
    }
  }

  _holeSurfacePosition(hole) {
    const pos = hole.pos.clone();
    pos.y = Math.max(pos.y, this.config.ropeRadius);
    return pos;
  }

  _liftPlaneY() {
    let maxY = this.config.ropeRadius;
    for (const rope of this.ropes) {
      for (const particle of rope.particles) {
        maxY = Math.max(maxY, particle.pos.y);
      }
    }
    return maxY + this.config.liftClearance;
  }

  // ---- physics subroutines ----

  _detectAllCollisions() {
    const segments = [];
    this.ropes.forEach((rope, ropeId) => {
      for (let i = 0; i < rope.particles.length - 1; i += 1) {
        segments.push({
          p1: rope.particles[i],
          p2: rope.particles[i + 1],
          ropeId,
          rope,
        });
      }
    });

    const radiusSum = this.config.ropeRadius * 2.0;
    const blockingThresh = radiusSum * 2.0;
    for (let i = 0; i < segments.length; i += 1) {
      for (let j = i + 1; j < segments.length; j += 1) {
        const a = segments[i];
        const b = segments[j];
        if (a.ropeId === b.ropeId) continue;

        const dist = this._segmentDistance(a.p1, a.p2, b.p1, b.p2);
        if (dist < blockingThresh) {
          const pressure = (blockingThresh - dist) / blockingThresh;
          a.rope.collisionPressure += pressure;
          b.rope.collisionPressure += pressure;
          if (dist < radiusSum * 1.3) {
            a.rope.canTighten = false;
            b.rope.canTighten = false;
          }
        }
      }
    }
  }

  _segmentDistance(p1, p2, p3, p4) {
    const ux = p2.pos.x - p1.pos.x;
    const uy = p2.pos.y - p1.pos.y;
    const uz = p2.pos.z - p1.pos.z;
    const vx = p4.pos.x - p3.pos.x;
    const vy = p4.pos.y - p3.pos.y;
    const vz = p4.pos.z - p3.pos.z;
    const wx = p1.pos.x - p3.pos.x;
    const wy = p1.pos.y - p3.pos.y;
    const wz = p1.pos.z - p3.pos.z;
    const a = ux * ux + uy * uy + uz * uz;
    const b = ux * vx + uy * vy + uz * vz;
    const c = vx * vx + vy * vy + vz * vz;
    const d = ux * wx + uy * wy + uz * wz;
    const e = vx * wx + vy * wy + vz * wz;
    const D = a * c - b * b;
    let sc;
    let tc;
    if (D < 1e-6) {
      sc = 0.0;
      tc = b > c ? d / b : e / c;
    } else {
      sc = (b * e - c * d) / D;
      tc = (a * e - b * d) / D;
    }
    sc = Math.max(0, Math.min(1, Number.isFinite(sc) ? sc : 0));
    tc = Math.max(0, Math.min(1, Number.isFinite(tc) ? tc : 0));

    const dx = p1.pos.x + ux * sc - (p3.pos.x + vx * tc);
    const dy = p1.pos.y + uy * sc - (p3.pos.y + vy * tc);
    const dz = p1.pos.z + uz * sc - (p3.pos.z + vz * tc);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  _applyTightening(rope) {
    if (rope.isDragging) return;

    const pStart = rope.particles[0].pos;
    const pEnd = rope.particles[rope.particles.length - 1].pos;
    const directDist = pStart.distanceTo(pEnd);
    const targetSegLen = (directDist * this.config.minSlack) / this.config.ropeSegments;

    let currentPhysicalLen = 0;
    for (let i = 0; i < rope.particles.length - 1; i += 1) {
      currentPhysicalLen += rope.particles[i].pos.distanceTo(rope.particles[i + 1].pos);
    }
    const configTotalLen = rope.segmentLength * this.config.ropeSegments;

    if (currentPhysicalLen < configTotalLen - 0.1) {
      let speed = this.config.tightenSpeed;
      if (rope.collisionPressure > 0.5) speed *= 0.1;
      else if (!rope.canTighten) speed = 0;

      if (rope.segmentLength > targetSegLen) {
        rope.segmentLength = Math.max(targetSegLen, rope.segmentLength - speed);
      }
    }

    if (configTotalLen < currentPhysicalLen - 1.0) {
      rope.segmentLength += this.config.tightenSpeed;
    }
  }

  _applyGravity(rope, dt) {
    const g = this.config.gravity;
    for (const p of rope.particles) {
      if (p.invMass > 0) p.pos.y += g * dt * dt;
    }
  }

  _solveDistance(rope, dt) {
    const alpha = this.config.compliance / (dt * dt);
    for (let i = 0; i < rope.particles.length - 1; i += 1) {
      const p1 = rope.particles[i];
      const p2 = rope.particles[i + 1];
      const dx = p2.pos.x - p1.pos.x;
      const dy = p2.pos.y - p1.pos.y;
      const dz = p2.pos.z - p1.pos.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 1e-4) continue;
      const w1 = p1.invMass;
      const w2 = p2.invMass;
      if (w1 + w2 === 0) continue;
      const diff = dist - rope.segmentLength;
      const lambda = diff / (w1 + w2 + alpha);
      const cx = (dx / dist) * lambda;
      const cy = (dy / dist) * lambda;
      const cz = (dz / dist) * lambda;
      if (w1 > 0) {
        p1.pos.x += cx * w1;
        p1.pos.y += cy * w1;
        p1.pos.z += cz * w1;
      }
      if (w2 > 0) {
        p2.pos.x -= cx * w2;
        p2.pos.y -= cy * w2;
        p2.pos.z -= cz * w2;
      }
    }
  }

  _solveBending(rope) {
    const stiffness = this.config.bendingStiffness;
    for (let i = 1; i < rope.particles.length - 1; i += 1) {
      const pPrev = rope.particles[i - 1];
      const pCurr = rope.particles[i];
      const pNext = rope.particles[i + 1];
      if (pCurr.invMass === 0) continue;
      const cx = (pPrev.pos.x + pNext.pos.x) * 0.5;
      const cy = (pPrev.pos.y + pNext.pos.y) * 0.5;
      const cz = (pPrev.pos.z + pNext.pos.z) * 0.5;
      const dx = cx - pCurr.pos.x;
      const dy = cy - pCurr.pos.y;
      const dz = cz - pCurr.pos.z;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-3) {
        pCurr.pos.x += dx * stiffness;
        pCurr.pos.y += dy * stiffness;
        pCurr.pos.z += dz * stiffness;
      }
    }
  }

  _solveGround(rope) {
    const r = this.config.ropeRadius;
    const gf = this.config.groundFriction;
    for (const p of rope.particles) {
      if (p.pos.y < r) {
        p.pos.y = r;
        const vX = p.pos.x - p.oldPos.x;
        const vZ = p.pos.z - p.oldPos.z;
        p.oldPos.x = p.pos.x - vX * (1 - gf);
        p.oldPos.z = p.pos.z - vZ * (1 - gf);
      }
    }
  }

  _solveRopeRopeCollisions() {
    const segs = [];
    this.ropes.forEach((rope, rIdx) => {
      for (let i = 0; i < rope.particles.length - 1; i += 1) {
        segs.push({
          p1: rope.particles[i],
          p2: rope.particles[i + 1],
          ropeId: rIdx,
          segIdx: i,
        });
      }
    });
    const radiusSum = this.config.ropeRadius * 2.0;
    for (let i = 0; i < segs.length; i += 1) {
      for (let j = i + 1; j < segs.length; j += 1) {
        const a = segs[i];
        const b = segs[j];
        if (a.ropeId === b.ropeId && Math.abs(a.segIdx - b.segIdx) <= 2) continue;
        this._resolveSegmentCollision(a.p1, a.p2, b.p1, b.p2, radiusSum);
      }
    }
  }

  _resolveSegmentCollision(p1, p2, p3, p4, distThresh) {
    const ux = p2.pos.x - p1.pos.x;
    const uy = p2.pos.y - p1.pos.y;
    const uz = p2.pos.z - p1.pos.z;
    const vx = p4.pos.x - p3.pos.x;
    const vy = p4.pos.y - p3.pos.y;
    const vz = p4.pos.z - p3.pos.z;
    const wx = p1.pos.x - p3.pos.x;
    const wy = p1.pos.y - p3.pos.y;
    const wz = p1.pos.z - p3.pos.z;
    const a = ux * ux + uy * uy + uz * uz;
    const b = ux * vx + uy * vy + uz * vz;
    const c = vx * vx + vy * vy + vz * vz;
    const d = ux * wx + uy * wy + uz * wz;
    const e = vx * wx + vy * wy + vz * wz;
    const D = a * c - b * b;
    let sc;
    let tc;
    if (D < 1e-6) {
      sc = 0.0;
      tc = b > c ? d / b : e / c;
    } else {
      sc = (b * e - c * d) / D;
      tc = (a * e - b * d) / D;
    }
    sc = Math.max(0, Math.min(1, sc));
    tc = Math.max(0, Math.min(1, tc));
    const closeAx = p1.pos.x + ux * sc;
    const closeAy = p1.pos.y + uy * sc;
    const closeAz = p1.pos.z + uz * sc;
    const closeBx = p3.pos.x + vx * tc;
    const closeBy = p3.pos.y + vy * tc;
    const closeBz = p3.pos.z + vz * tc;
    const diffX = closeAx - closeBx;
    const diffY = closeAy - closeBy;
    const diffZ = closeAz - closeBz;
    const len = Math.sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
    if (len < distThresh && len > 1e-6) {
      const overlap = distThresh - len;
      const nx = (diffX / len) * overlap * 0.5;
      const ny = (diffY / len) * overlap * 0.5;
      const nz = (diffZ / len) * overlap * 0.5;
      const w1 = 1 - sc;
      const w2 = sc;
      const w3 = 1 - tc;
      const w4 = tc;
      if (p1.invMass) {
        p1.pos.x += nx * w1;
        p1.pos.y += ny * w1;
        p1.pos.z += nz * w1;
      }
      if (p2.invMass) {
        p2.pos.x += nx * w2;
        p2.pos.y += ny * w2;
        p2.pos.z += nz * w2;
      }
      if (p3.invMass) {
        p3.pos.x -= nx * w3;
        p3.pos.y -= ny * w3;
        p3.pos.z -= nz * w3;
      }
      if (p4.invMass) {
        p4.pos.x -= nx * w4;
        p4.pos.y -= ny * w4;
        p4.pos.z -= nz * w4;
      }
    }
  }

  _integrate(rope, dt) {
    const maxVel = this.config.ropeRadius * 0.9;
    const maxVelSq = maxVel * maxVel;
    const friction = this.config.friction;
    const sleepThresh = this.config.sleepThreshold;
    for (const p of rope.particles) {
      if (p.invMass > 0) {
        let vx = (p.pos.x - p.oldPos.x) * friction;
        let vy = (p.pos.y - p.oldPos.y) * friction;
        let vz = (p.pos.z - p.oldPos.z) * friction;
        const lenSq = vx * vx + vy * vy + vz * vz;
        if (lenSq > maxVelSq) {
          const scale = maxVel / Math.sqrt(lenSq);
          vx *= scale;
          vy *= scale;
          vz *= scale;
        } else if (lenSq < sleepThresh) {
          vx = 0;
          vy = 0;
          vz = 0;
        }
        p.oldPos.x = p.pos.x;
        p.oldPos.y = p.pos.y;
        p.oldPos.z = p.pos.z;
        p.pos.x += vx;
        p.pos.y += vy;
        p.pos.z += vz;
      }
    }
  }

  _smoothVisuals(rope) {
    if (rope.isDragging) {
      for (let i = 0; i < rope.particles.length; i += 1) {
        rope.visualPositions[i].copy(rope.particles[i].pos);
      }
      return;
    }
    const alpha = this.config.visualSmoothing;
    for (let i = 0; i < rope.particles.length; i += 1) {
      rope.visualPositions[i].lerp(rope.particles[i].pos, alpha);
    }
  }
}
