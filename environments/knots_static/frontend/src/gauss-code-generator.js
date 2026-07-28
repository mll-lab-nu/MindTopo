/**
 * Gauss Code Generator
 * 
 * Generates knots from Gauss codes (signed crossing sequences)
 * New recommended token format:
 *   1o = crossing 1, over-pass
 *   1u = crossing 1, under-pass
 * Optional writhe/sign is allowed: +1o, -2u (sign is parsed but embedding may ignore it)
 *
 * Legacy format supported in UI parsing only:
 *   +1 -2 +3 -1 +2 -3
 * Legacy fallback rule: first occurrence => over, second => under
 */

/**
 * Known Gauss codes for common knots
 */
export const KNOWN_GAUSS_CODES = {
  trefoil: '1o 2o 3o 1u 2u 3u',
  figure8: '1o 2o 3o 4o 1u 2u 3u 4u',
  cinquefoil: '1o 2o 3o 4o 5o 1u 2u 3u 4u 5u',
  '5_2': '1o 2o 3o 4o 5o 1u 2u 3u 4u 5u',
  '6_1': '1o 2o 3o 4o 5o 6o 1u 2u 3u 4u 5u 6u',
  '7_1': '1o 2o 3o 4o 5o 6o 7o 1u 2u 3u 4u 5u 6u 7u',
};

/**
 * Legacy parser (kept for compatibility with old callers).
 * Prefer `parseGaussTokens()` for new logic.
 */
export function parseGaussCode(code) {
  const parts = code.trim().split(/\s+/);
  const crossings = [];
  
  for (const part of parts) {
    const match = part.match(/^([+-]?)(\d+)$/);
    if (!match) {
      throw new Error(`Invalid Gauss code format: "${part}"`);
    }
    
    const sign = match[1] === '-' ? -1 : 1;
    const num = parseInt(match[2], 10);
    crossings.push({ num, sign });
  }
  
  return crossings;
}

/**
 * Parse Gauss tokens.
 *
 * Supported tokens: (\+|\-)?(\d+)(o|u)?
 * Output tokens are normalized to:
 *   { id:number, over:boolean, sign:(+1|-1) }
 *
 * If token has no (o|u), legacy fallback is used:
 *   first occurrence => over, second => under
 *
 * Validation:
 * - each id must appear exactly twice
 * - for single-rope Gauss code, the two occurrences must be one over + one under
 */
export function parseGaussTokens(str, options = {}) {
  const { legacyFallback = true } = options;
  const parts = (str || '').trim().length ? (str || '').trim().split(/\s+/) : [];
  if (parts.length === 0) return [];

  const raw = [];
  for (const part of parts) {
    const match = part.match(/^([+-]?)(\d+)([ou])?$/i);
    if (!match) {
      throw new Error(`Invalid Gauss token: "${part}" (expected like 1o / 2u / +3o / -4u)`);
    }
    const sign = match[1] === '-' ? -1 : 1;
    const id = parseInt(match[2], 10);
    const ou = match[3] ? match[3].toLowerCase() : null;
    raw.push({ id, ou, sign });
  }

  const useCount = new Map();
  const out = raw.map((t) => {
    let over;
    if (t.ou === 'o') over = true;
    else if (t.ou === 'u') over = false;
    else {
      if (!legacyFallback) {
        throw new Error(`Token "${t.id}" missing o/u; please use explicit format like "${t.id}o" or "${t.id}u"`);
      }
      const c = useCount.get(t.id) || 0;
      over = c === 0;
      useCount.set(t.id, c + 1);
    }
    return { id: t.id, over, sign: t.sign };
  });

  // Validate occurrences and over/under pairing
  const occ = new Map();
  for (const tok of out) {
    if (!occ.has(tok.id)) occ.set(tok.id, []);
    occ.get(tok.id).push(tok);
  }
  for (const [id, arr] of occ.entries()) {
    if (arr.length !== 2) throw new Error(`Crossing id ${id} must appear exactly twice (got ${arr.length})`);
    const a = arr[0].over;
    const b = arr[1].over;
    if (a === b) {
      throw new Error(`Crossing id ${id} must have one over (o) and one under (u)`);
    }
  }

  return out;
}

function mulberry32(seed) {
  let a = (seed >>> 0) || 0x12345678;
  return function rand() {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * rollRandomGaussCode(nCross, seed) -> gaussTokens[]
 *
 * v1 strategy:
 * - create a length-2n sequence containing each id 1..n twice (then shuffle)
 * - for each id, randomly choose whether first occurrence is over or under
 * - second occurrence is the opposite
 *
 * Returns: [{id, over, sign}]
 */
export function rollRandomGaussCode(nCross, seed = 1) {
  const n = Math.max(0, Math.floor(nCross || 0));
  if (n < 1) return [];

  const rng = mulberry32(Number.isFinite(seed) ? seed : 1);

  const total = 2 * n;

  // Improve tighten-ability: avoid placing the 2nd occurrence too close to the 1st.
  // This makes the traversal more "interwoven" and less like a bunch of loose long loops.
  const minSep = Math.max(2, Math.floor(total * 0.2)); // ~20% of loop
  const maxSep = Math.max(minSep + 2, Math.floor(total * 0.75));

  const ids = [];
  for (let i = 1; i <= n; i++) ids.push(i);
  shuffleInPlace(ids, rng);

  // Start with first occurrences in the first half (in random order)
  const seq = new Array(total).fill(null);
  const firstPos = new Map(); // id -> index
  for (let i = 0; i < n; i++) {
    const id = ids[i];
    seq[i] = id;
    firstPos.set(id, i);
  }

  function forwardDist(a, b) {
    const d = (b - a) % total;
    return d < 0 ? d + total : d;
  }

  // Place second occurrences into remaining slots, respecting minSep/maxSep when possible
  const empty = [];
  for (let i = 0; i < total; i++) if (seq[i] === null) empty.push(i);

  for (const id of ids) {
    const p = firstPos.get(id);
    let placed = false;
    for (let tries = 0; tries < 64 && !placed; tries++) {
      if (empty.length === 0) break;
      const candIdx = Math.floor(rng() * empty.length);
      const q = empty[candIdx];
      const d = forwardDist(p, q);
      if (d >= minSep && d <= maxSep) {
        seq[q] = id;
        empty.splice(candIdx, 1);
        placed = true;
      }
    }
    if (!placed) {
      // fallback: just take any remaining slot
      if (empty.length === 0) break;
      const candIdx = Math.floor(rng() * empty.length);
      const q = empty[candIdx];
      seq[q] = id;
      empty.splice(candIdx, 1);
    }
  }

  // Assign over/under: either alternating pattern or random, but always opposite for the pair.
  const useAlternating = rng() < 0.5;
  const firstOver = new Map(); // id -> over(boolean) for first occurrence
  const seen = new Map();
  const tokens = [];

  for (let i = 0; i < seq.length; i++) {
    const id = seq[i];
    if (id == null) continue;
    const c = (seen.get(id) || 0);
    seen.set(id, c + 1);

    if (c === 0) {
      const over0 = useAlternating ? (i % 2 === 0) : (rng() < 0.5);
      firstOver.set(id, over0);
    }
    const over = c === 0 ? firstOver.get(id) : !firstOver.get(id);

    // v1: sign (writhe) is parsed but not required; keep +1 for stability
    const sign = 1;
    tokens.push({ id, over, sign });
  }

  return tokens;
}

function wrapAngleRad(a) {
  let x = a;
  while (x <= -Math.PI) x += Math.PI * 2;
  while (x > Math.PI) x -= Math.PI * 2;
  return x;
}

function almostSamePoint(a, b, eps = 1e-6) {
  const dx = (a.x || 0) - (b.x || 0);
  const dy = (a.y || 0) - (b.y || 0);
  const dz = (a.z || 0) - (b.z || 0);
  return (dx * dx + dy * dy + dz * dz) < (eps * eps);
}

function normalizeAngle0ToTau(a) {
  const tau = Math.PI * 2;
  let x = a % tau;
  if (x < 0) x += tau;
  return x;
}

// Return one or two intervals in [0, 2π) that represent the SHORT arc used by routeLane.
function arcIntervalsShort(thetaFrom, thetaTo) {
  const tau = Math.PI * 2;
  // Match routeLane: use shortest signed angle delta in [-π, π]
  const dTheta = wrapAngleRad(thetaTo - thetaFrom);
  const a0 = thetaFrom;
  const a1 = thetaFrom + dTheta;

  const s = Math.min(a0, a1);
  const e = Math.max(a0, a1);
  const sN = normalizeAngle0ToTau(s);
  const eN = normalizeAngle0ToTau(e);

  if (sN <= eN) return [[sN, eN]];
  // crosses 0, split
  return [[0, eN], [sN, tau]];
}

function intervalsOverlap(intsA, intsB, pad = 0.03) {
  for (const [a0, a1] of intsA) {
    for (const [b0, b1] of intsB) {
      if (Math.max(a0, b0) <= Math.min(a1, b1) + pad) return true;
    }
  }
  return false;
}

function pickLane(laneOcc, ints, pad = 0.03) {
  for (let lane = 0; lane < laneOcc.length; lane++) {
    let ok = true;
    for (const occ of laneOcc[lane]) {
      if (intervalsOverlap(occ, ints, pad)) { ok = false; break; }
    }
    if (ok) return lane;
  }
  laneOcc.push([]);
  return laneOcc.length - 1;
}

/**
 * embedGaussToPolyline3D(gaussTokens, options) -> { points3D, crossings }
 *
 * Implements:
 * - Crossing Gadget: each crossing has a fixed local template (ports L/R/D/U)
 * - Lane Routing: long connections route out to unique outer lanes (different radii) to avoid extra crossings
 *
 * Output crossing entries include stable polyline indices:
 *   { overPointIndex, underPointIndex } referencing `points3D`
 */
export function embedGaussToPolyline3D(gaussTokens, options = {}) {
  const {
    scale = 0.8,
    innerR = 0.55 * scale,
    portR = 0.085 * scale,
    laneGap = 0.16 * scale,
    zSep = 0.22 * 1.0,
    radialSteps = 2,
    arcStepsBase = 10,
    tailLength = 1.8 * scale,
    tailPoints = 22,
    normalize = true,
    targetMaxRadius = 1.25 * scale,
    lanePad = 0.04, // radians; prevents lanes from "almost touching" due to discretization
  } = options;

  const tokens = Array.isArray(gaussTokens) ? gaussTokens : [];
  if (tokens.length === 0) {
    return { points3D: [], crossings: [] };
  }
  if (tokens.length % 2 !== 0) {
    throw new Error('Gauss code must have even length (each crossing appears twice)');
  }

  // Validate token structure
  const occ = new Map();
  for (const tok of tokens) {
    if (!Number.isFinite(tok.id)) throw new Error('Invalid gauss token: missing id');
    if (!occ.has(tok.id)) occ.set(tok.id, []);
    occ.get(tok.id).push(tok);
  }
  for (const [id, arr] of occ.entries()) {
    if (arr.length !== 2) throw new Error(`Crossing id ${id} must appear exactly twice (got ${arr.length})`);
    if (arr[0].over === arr[1].over) throw new Error(`Crossing id ${id} must have one over and one under`);
  }

  const ids = Array.from(occ.keys()).sort((a, b) => a - b);

  // Step A: place crossing centers on inner circle
  const centers = new Map();
  for (let k = 0; k < ids.length; k++) {
    const id = ids[k];
    const ang = (k / ids.length) * Math.PI * 2;
    centers.set(id, { x: Math.cos(ang) * innerR, y: Math.sin(ang) * innerR, z: 0 });
  }

  // Base lane just outside the crossing gadget
  const laneBaseR = innerR + portR + laneGap;

  // Prepare crossing records
  const crossingRecords = new Map();
  for (const id of ids) {
    const c = centers.get(id);
    crossingRecords.set(id, {
      crossingNum: id,
      overLeadIdx: 0,
      underLeadIdx: 0,
      x: c.x,
      y: c.y,
      point: { x: c.x, y: c.y },
      overPointIndex: -1,
      underPointIndex: -1,
    });
  }

  function portPos(center, portName) {
    switch (portName) {
      case 'L': return { x: center.x - portR, y: center.y, z: 0 };
      case 'R': return { x: center.x + portR, y: center.y, z: 0 };
      case 'D': return { x: center.x, y: center.y - portR, z: 0 };
      case 'U': return { x: center.x, y: center.y + portR, z: 0 };
      default: return { x: center.x, y: center.y, z: 0 };
    }
  }

  const points = [];
  function push(p) {
    const pp = { x: p.x, y: p.y, z: p.z || 0 };
    const last = points.length ? points[points.length - 1] : null;
    if (!last || !almostSamePoint(last, pp)) points.push(pp);
  }

  function routeLane(from, to, laneIdx, zFrom, zTo) {
    const thetaFrom = Math.atan2(from.y, from.x);
    const thetaTo = Math.atan2(to.y, to.x);
    const dTheta = wrapAngleRad(thetaTo - thetaFrom);
    const rLane = laneBaseR + laneGap * laneIdx;

    const outP = { x: Math.cos(thetaFrom) * rLane, y: Math.sin(thetaFrom) * rLane, z: 0 };
    const inP = { x: Math.cos(thetaTo) * rLane, y: Math.sin(thetaTo) * rLane, z: 0 };

    // radial out (from -> outP), interpolate z to 0
    for (let i = 1; i <= Math.max(1, radialSteps); i++) {
      const t = i / Math.max(1, radialSteps);
      push({
        x: from.x + (outP.x - from.x) * t,
        y: from.y + (outP.y - from.y) * t,
        z: (zFrom || 0) + (0 - (zFrom || 0)) * t,
      });
    }

    // arc on lane (outP -> inP) at z=0
    const arcSteps = Math.max(3, Math.ceil((Math.abs(dTheta) / (Math.PI * 2)) * arcStepsBase * 2));
    for (let i = 1; i <= arcSteps; i++) {
      const t = i / arcSteps;
      const th = thetaFrom + dTheta * t;
      push({ x: Math.cos(th) * rLane, y: Math.sin(th) * rLane, z: 0 });
    }

    // radial in (inP -> to), interpolate z from 0 to zTo
    for (let i = 1; i <= Math.max(1, radialSteps); i++) {
      const t = i / Math.max(1, radialSteps);
      push({
        x: inP.x + (to.x - inP.x) * t,
        y: inP.y + (to.y - inP.y) * t,
        z: 0 + ((zTo || 0) - 0) * t,
      });
    }
  }

  // Step B: build path in token order
  const useCount = new Map();
  const laneOcc = []; // lane -> list of arc interval lists (each interval list is 1-2 intervals)

  // First token: start at its entry port (no incoming route)
  const firstTok = tokens[0];
  const firstCenter = centers.get(firstTok.id);
  if (!firstCenter) throw new Error(`Missing center for crossing id ${firstTok.id}`);
  const firstEntry = portPos(firstCenter, 'L'); // useCount==0 => L->R
  const firstZ = firstTok.over ? (zSep * 0.5) : (-zSep * 0.5);
  push({ ...firstEntry, z: firstZ });

  let prevExit = null;
  let prevZ = firstZ;
  let firstEntryPos = { ...firstEntry, z: firstZ };

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const c = centers.get(tok.id);
    if (!c) throw new Error(`Missing center for crossing id ${tok.id}`);

    const uc = useCount.get(tok.id) || 0;
    if (uc > 1) throw new Error(`Crossing id ${tok.id} appears more than twice`);

    const z = tok.over ? (zSep * 0.5) : (-zSep * 0.5);
    const entryPort = uc === 0 ? portPos(c, 'L') : portPos(c, 'D');
    const exitPort = uc === 0 ? portPos(c, 'R') : portPos(c, 'U');

    // route from previous exit to this entry
    if (prevExit) {
      const a0 = Math.atan2(prevExit.y, prevExit.x);
      const a1 = Math.atan2(entryPort.y, entryPort.x);
      const ints = arcIntervalsShort(a0, a1);
      const lane = pickLane(laneOcc, ints, lanePad);
      laneOcc[lane].push(ints);
      routeLane(prevExit, entryPort, lane, prevZ, z);
    } else {
      // For the first token, we already started at its entry; ensure alignment
      if (!almostSamePoint(points[points.length - 1], { ...entryPort, z })) {
        push({ ...entryPort, z });
      }
    }

    // crossing gadget: entry -> center -> exit (z separated by over/under)
    push({ ...entryPort, z });
    const centerIdx = points.length;
    push({ x: c.x, y: c.y, z });
    push({ ...exitPort, z });

    const rec = crossingRecords.get(tok.id);
    if (rec) {
      if (tok.over) rec.overPointIndex = centerIdx;
      else rec.underPointIndex = centerIdx;
    }

    useCount.set(tok.id, uc + 1);
    prevExit = { ...exitPort, z };
    prevZ = z;
  }

  // Close loop: connect last exit back to first entry
  if (prevExit) {
    const a0 = Math.atan2(prevExit.y, prevExit.x);
    const a1 = Math.atan2(firstEntryPos.y, firstEntryPos.x);
    const ints = arcIntervalsShort(a0, a1);
    const lane = pickLane(laneOcc, ints, lanePad);
    laneOcc[lane].push(ints);
    routeLane(prevExit, firstEntryPos, lane, prevZ, firstZ);
  }
  push({ ...firstEntryPos });

  // Normalize layout (tighten visual scale) BEFORE adding tails so knot size is consistent.
  const closed = points.slice();
  let cx = 0, cy = 0;
  for (const p of closed) { cx += p.x; cy += p.y; }
  cx /= Math.max(1, closed.length);
  cy /= Math.max(1, closed.length);

  let maxR = 0;
  for (const p of closed) {
    const x = p.x - cx;
    const y = p.y - cy;
    maxR = Math.max(maxR, Math.hypot(x, y));
  }
  const s = (normalize && maxR > 1e-6) ? (targetMaxRadius / maxR) : 1.0;

  for (const p of closed) {
    p.x = (p.x - cx) * s;
    p.y = (p.y - cy) * s;
    // keep z as-is; it's only local separation
  }
  for (const rec of crossingRecords.values()) {
    rec.x = (rec.x - cx) * s;
    rec.y = (rec.y - cy) * s;
    if (rec.point) {
      rec.point.x = (rec.point.x - cx) * s;
      rec.point.y = (rec.point.y - cy) * s;
    }
  }

  // Open with tails (adjust crossing indices by the inserted front tail size)
  const opened = openWithTailsFromClosedLoop(closed, { tailLength: tailLength * s, tailPoints });

  const indexShift = tailPoints;
  const crossings = [];
  for (const id of ids) {
    const rec = crossingRecords.get(id);
    if (!rec) continue;
    if (rec.overPointIndex < 0 || rec.underPointIndex < 0) {
      throw new Error(`Crossing id ${id} missing over/under point indices after embedding`);
    }
    crossings.push({
      ...rec,
      overPointIndex: rec.overPointIndex + indexShift,
      underPointIndex: rec.underPointIndex + indexShift,
    });
  }

  return { points3D: opened, crossings };
}

/**
 * Generate knot object (leads + crossings) from Gauss code string.
 * The output is designed to be fed directly into the existing PBD engine.
 */
export function generateKnotFromGaussCode(code, options = {}) {
  const {
    scale = 0.8,
    zSep = 0.22 * (options.zScale || 1.0),
  } = options;

  const tokens = parseGaussTokens(code, { legacyFallback: true });
  const numCrossings = tokens.length / 2;

  const { points3D, crossings } = embedGaussToPolyline3D(tokens, {
    ...options,
    scale,
    zSep,
  });

  return {
    leads: [{
      name: `gauss_${numCrossings}`,
      points: points3D,
      isClosed: false,
    }],
    crossings,
    equation: 'Gauss Code',
    knotName: `Gauss Code (${numCrossings} crossings)`,
    params: {
      type: 'gauss',
      code,
      numCrossings,
    },
  };
}

function openWithTailsFromClosedLoop(points, options = {}) {
  const { tailLength = 1.2, tailPoints = 16 } = options;
  if (points.length < 3) return points.slice();
  
  const out = points.slice();
  const first = out[0];
  const last = out[out.length - 1];
  const dx0 = first.x - last.x;
  const dy0 = first.y - last.y;
  const dz0 = (first.z || 0) - (last.z || 0);
  if (Math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) < 1e-6) {
    out.pop();
  }
  
  if (out.length < 3) return out;
  
  const a = out[0];
  const b = out[out.length - 1];
  
  let cx = 0, cy = 0;
  for (const p of out) {
    cx += p.x;
    cy += p.y;
  }
  cx /= out.length;
  cy /= out.length;
  
  function norm2(x, y) {
    const l = Math.sqrt(x * x + y * y);
    return l > 1e-8 ? { x: x / l, y: y / l } : null;
  }
  
  const chord = norm2(a.x - b.x, a.y - b.y) || { x: 1, y: 0 };
  const ra = norm2(a.x - cx, a.y - cy) || chord;
  const rb = norm2(b.x - cx, b.y - cy) || { x: -chord.x, y: -chord.y };
  
  const startDir = norm2(chord.x * 0.75 + ra.x * 0.25, chord.y * 0.75 + ra.y * 0.25) || chord;
  const endDir = norm2((-chord.x) * 0.75 + rb.x * 0.25, (-chord.y) * 0.75 + rb.y * 0.25) || { x: -chord.x, y: -chord.y };
  
  const startTail = [];
  for (let i = tailPoints; i >= 1; i--) {
    const t = (i / tailPoints) * tailLength;
    startTail.push({ x: a.x + startDir.x * t, y: a.y + startDir.y * t, z: a.z });
  }
  
  const endTail = [];
  for (let i = 1; i <= tailPoints; i++) {
    const t = (i / tailPoints) * tailLength;
    endTail.push({ x: b.x + endDir.x * t, y: b.y + endDir.y * t, z: b.z });
  }
  
  return [...startTail, ...out, ...endTail];
}

function resampleClosedPolyline(points, targetCount) {
  if (points.length < 2) return points.slice();
  
  const closed = points[0].x === points[points.length - 1].x &&
                 points[0].y === points[points.length - 1].y &&
                 points[0].z === points[points.length - 1].z
                 ? points.slice()
                 : [...points, { ...points[0] }];
  
  const arc = [0];
  for (let i = 1; i < closed.length; i++) {
    const dx = closed[i].x - closed[i - 1].x;
    const dy = closed[i].y - closed[i - 1].y;
    const dz = closed[i].z - closed[i - 1].z;
    arc.push(arc[i - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  
  const total = arc[arc.length - 1];
  if (total < 1e-6) return closed.slice(0, 2);
  
  const out = [];
  for (let i = 0; i < targetCount; i++) {
    const t = (i / (targetCount - 1)) * total;
    let seg = 0;
    while (seg < arc.length - 1 && arc[seg + 1] < t) seg++;
    
    const segLen = arc[seg + 1] - arc[seg];
    const lt = segLen > 1e-6 ? (t - arc[seg]) / segLen : 0;
    const p1 = closed[seg];
    const p2 = closed[seg + 1];
    
    out.push({
      x: p1.x + (p2.x - p1.x) * lt,
      y: p1.y + (p2.y - p1.y) * lt,
      z: p1.z + (p2.z - p1.z) * lt,
    });
  }
  return out;
}

// ============= Gauss Code Computation from 3D Centerline =============

/**
 * Compute 2D segment intersection.
 * Returns { t, s } if intersection exists (0 <= t <= 1 and 0 <= s <= 1), null otherwise.
 * t is the parameter on segment (p1, p2), s is the parameter on segment (p3, p4).
 */
function segmentIntersection2D(p1, p2, p3, p4, eps = 1e-9) {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;

  const cross = d1x * d2y - d1y * d2x;
  if (Math.abs(cross) < eps) return null; // parallel or collinear

  const dx = p3.x - p1.x;
  const dy = p3.y - p1.y;

  const t = (dx * d2y - dy * d2x) / cross;
  const s = (dx * d1y - dy * d1x) / cross;

  // Use small margin to avoid endpoint issues
  const margin = 0.001;
  if (t >= margin && t <= 1 - margin && s >= margin && s <= 1 - margin) {
    return { t, s };
  }
  return null;
}

/**
 * Interpolate Z value at parameter t on a segment.
 */
function interpZ(p1, p2, t) {
  const z1 = p1.z !== undefined ? p1.z : (p1[2] || 0);
  const z2 = p2.z !== undefined ? p2.z : (p2[2] || 0);
  return z1 + (z2 - z1) * t;
}

/**
 * Normalize point to {x, y, z} format.
 */
function normalizePoint(p) {
  if (Array.isArray(p)) {
    return { x: p[0] || 0, y: p[1] || 0, z: p[2] || 0 };
  }
  return { x: p.x || 0, y: p.y || 0, z: p.z || 0 };
}

/**
 * Compute Gauss code from a 3D centerline (closed loop).
 *
 * Algorithm:
 * 1. Project points to 2D (XY plane by default, or use projection matrix)
 * 2. Find all segment-segment intersections (crossings)
 * 3. Determine over/under at each crossing by comparing Z values
 * 4. Walk the curve and record crossings in order
 *
 * @param {Array} points - Array of 3D points ([x,y,z] or {x,y,z})
 * @param {Object} options - Configuration options
 * @returns {Object} { gaussCode: string, gaussTokens: array, crossings: array, numCrossings: number }
 */
export function computeGaussCodeFromCenterline(points, options = {}) {
  const {
    closed = true,
    neighborSkip = 3, // Skip checking adjacent segments to avoid false positives
    projectionAxis = 'z', // Project along this axis (remove it from 2D view)
    minSegmentDistance = 2, // Minimum segment index distance for valid crossing
  } = options;

  if (!points || points.length < 4) {
    return { gaussCode: '', gaussTokens: [], crossings: [], numCrossings: 0 };
  }

  // Normalize all points
  const pts = points.map(normalizePoint);
  const n = pts.length;

  // Create 2D projection based on projectionAxis
  const project2D = (p) => {
    switch (projectionAxis) {
      case 'x': return { x: p.y, y: p.z, z: p.x };
      case 'y': return { x: p.x, y: p.z, z: p.y };
      case 'z':
      default: return { x: p.x, y: p.y, z: p.z };
    }
  };

  const pts2D = pts.map(project2D);

  // Find all crossings
  const crossings = [];
  let crossingId = 1;

  // For a closed loop, we have n segments: (0,1), (1,2), ..., (n-2, n-1), (n-1, 0)
  // For open, we have n-1 segments
  const segCount = closed ? n : n - 1;

  function getSegment(i) {
    const i1 = i % n;
    const i2 = (i + 1) % n;
    return { i1, i2, p1: pts2D[i1], p2: pts2D[i2] };
  }

  // Check all pairs of non-adjacent segments
  for (let i = 0; i < segCount; i++) {
    for (let j = i + 1; j < segCount; j++) {
      // Skip adjacent or near-adjacent segments
      const dist = Math.min(Math.abs(j - i), closed ? Math.abs(n - (j - i)) : Infinity);
      if (dist < Math.max(neighborSkip, minSegmentDistance)) continue;

      const segA = getSegment(i);
      const segB = getSegment(j);

      const inter = segmentIntersection2D(segA.p1, segA.p2, segB.p1, segB.p2);
      if (inter) {
        // Get Z values at intersection point
        const zA = interpZ(pts[segA.i1], pts[segA.i2], inter.t);
        const zB = interpZ(pts[segB.i1], pts[segB.i2], inter.s);

        // Crossing position (for reference)
        const crossX = segA.p1.x + (segA.p2.x - segA.p1.x) * inter.t;
        const crossY = segA.p1.y + (segA.p2.y - segA.p1.y) * inter.t;

        crossings.push({
          id: crossingId++,
          segA: i,
          segB: j,
          tA: inter.t,
          tB: inter.s,
          zA,
          zB,
          aIsOver: zA > zB,
          position: { x: crossX, y: crossY },
          // Parameter position along the curve (for ordering)
          paramA: i + inter.t,
          paramB: j + inter.s,
        });
      }
    }
  }

  if (crossings.length === 0) {
    return { gaussCode: '', gaussTokens: [], crossings: [], numCrossings: 0 };
  }

  // Build traversal sequence: walk the curve and record crossings in order
  // Each crossing appears twice (once from segA, once from segB)
  const encounters = [];
  for (const c of crossings) {
    encounters.push({ param: c.paramA, crossing: c, isA: true });
    encounters.push({ param: c.paramB, crossing: c, isA: false });
  }

  // Sort by parameter position
  encounters.sort((a, b) => a.param - b.param);

  // Generate Gauss tokens
  const gaussTokens = [];
  for (const enc of encounters) {
    const c = enc.crossing;
    const over = enc.isA ? c.aIsOver : !c.aIsOver;
    gaussTokens.push({
      id: c.id,
      over,
      sign: 1, // Writhe sign computation is complex; default to +1
    });
  }

  // Generate Gauss code string
  const gaussCode = gaussTokens.map(t => `${t.id}${t.over ? 'o' : 'u'}`).join(' ');

  return {
    gaussCode,
    gaussTokens,
    crossings: crossings.map(c => ({
      id: c.id,
      position: c.position,
      aIsOver: c.aIsOver,
    })),
    numCrossings: crossings.length,
  };
}

/**
 * Try multiple projection directions and pick the one with the most crossings.
 * This helps avoid degenerate projections where crossings are hidden.
 */
export function computeGaussCodeBestProjection(points, options = {}) {
  const axes = ['z', 'x', 'y'];
  let best = { gaussCode: '', gaussTokens: [], crossings: [], numCrossings: 0 };

  for (const axis of axes) {
    const result = computeGaussCodeFromCenterline(points, { ...options, projectionAxis: axis });
    if (result.numCrossings > best.numCrossings) {
      best = { ...result, projectionAxis: axis };
    }
  }

  return best;
}
