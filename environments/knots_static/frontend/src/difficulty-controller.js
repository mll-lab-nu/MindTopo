/**
 * Unified Difficulty Controller
 *
 * Provides consistent difficulty scoring across:
 * 1) Single closed-loop knot images
 * 2) Pair images (invariance dataset)
 *
 * Output convention:
 * - difficulty_score: number in [0,1]
 * - difficulty: 'easy' | 'medium' | 'hard'
 * - factors: object of normalized sub-scores (generally [0,1])
 */

import { KNOT_TYPE_REGISTRY, getCrossingNumber, isDeceptiveKnot, isConfusingPair } from './knot-type-registry.js';

// ============= Helpers =============

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function scoreToLevel(score) {
  const s = clamp(score, 0, 1);
  // 旧阈值: easy < 0.30, medium < 0.55, hard >= 0.55
  // 新阈值: 收紧 easy 区间，扩大 hard 区间，目标 easy ≤ 35%, hard ≥ 25%
  return s < 0.25 ? 'easy' : (s < 0.45 ? 'medium' : 'hard');
}

// ============= 难度分桶（正交维度）=============

/**
 * Factor A: 拓扑复杂度（基于交叉数）
 */
export function getBucketTopology(knotType) {
  const c = getCrossingNumber(knotType) ?? 0;
  if (c <= 3) return 'low';
  if (c <= 6) return 'mid';
  return 'high';
}

/**
 * Factor B: 视觉显著度（基于 slackness）
 * slackness 越高 -> crossing 越不明显 -> 越难
 */
export function getBucketSaliency(slackness) {
  const s = clamp(Number(slackness) || 0, 0, 1);
  if (s <= 0.25) return 'tight';
  if (s <= 0.60) return 'medium';
  return 'loose';
}

/**
 * Factor C: 认知陷阱类型
 */
export const TRAP_TYPES = {
  NONE: null,
  LOOSE_KNOT: 'loose_knot',
  DECEPTIVE_UNKNOT: 'deceptive_unknot',
  VIEW_COLLAPSE: 'view_collapse',
  OCCLUDED: 'occluded',
  MIXED_CONNECTIVITY: 'mixed_connectivity',
};

export function getTrapType(knotType, slackness) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  if (!entry) return TRAP_TYPES.NONE;

  // Occluded knots
  if (entry.hasOcclusion) return TRAP_TYPES.OCCLUDED;

  // Mixed connectivity (some linked, some free)
  if (entry.family === 'mixed') return TRAP_TYPES.MIXED_CONNECTIVITY;

  // Loose knots (including loose variants)
  if (entry.isLooseVariant) return TRAP_TYPES.LOOSE_KNOT;
  if (!entry.isLink && !entry.isUnknot && (Number(slackness) || 0) > 0.55) {
    return TRAP_TYPES.LOOSE_KNOT;
  }
  if (entry.isDeceptive) {
    return TRAP_TYPES.DECEPTIVE_UNKNOT;
  }
  return TRAP_TYPES.NONE;
}

/**
 * Compute view angle difficulty from camera position.
 * Top-down (high |y|) or front/back (high |z|) -> easier
 * Oblique angles -> harder
 * @param {number[]} cameraPos [x,y,z]
 * @returns {number} in [0,1]
 */
export function computeViewDifficultyScore(cameraPos) {
  const [x = 0, y = 0, z = 0] = cameraPos || [];
  const r = Math.sqrt(x * x + y * y + z * z) || 1;
  const ny = y / r;
  const nz = z / r;
  const alignScore = Math.max(Math.abs(ny), Math.abs(nz)); // 1 means aligned to easy axis
  return clamp(1 - alignScore, 0, 1);
}

/**
 * Compute angle difference between two camera positions (0-1).
 * @param {number[]} posA
 * @param {number[]} posB
 */
export function computeViewAngleDiff(posA, posB) {
  const a = posA || [0, 0, 1];
  const b = posB || [0, 0, 1];
  const normA = Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) || 1;
  const normB = Math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2) || 1;
  const dot = (
    (a[0] / normA) * (b[0] / normB) +
    (a[1] / normA) * (b[1] / normB) +
    (a[2] / normA) * (b[2] / normB)
  );
  return Math.acos(clamp(dot, -1, 1)) / Math.PI;
}

// ============= Single Closed-Loop =============

export function computeSingleClosedLoopDifficulty(params) {
  const {
    knotType,
    deformStrength = 0.3,
    cameraPosition = [0, 5, 10],
    tubeRadius = 0.08,
  } = params || {};

  const weights = {
    crossing: 0.25,
    deform: 0.20,
    view: 0.20,
    occlusion: 0.15,
    deceptive: 0.20,
  };

  const crossingNumber = getCrossingNumber(knotType) || 0;
  const crossingScore = clamp(crossingNumber / 10, 0, 1);
  const deformScore = clamp(deformStrength, 0, 1);
  const viewScore = computeViewDifficultyScore(cameraPosition);

  // proxy for "occlusion/visual ambiguity": thicker rope + more crossings => more self-occlusion
  const occlusionScore = clamp((tubeRadius / 0.1) * 0.5 + (crossingNumber / 10) * 0.5, 0, 1);
  const deceptiveScore = isDeceptiveKnot(knotType) ? 1.0 : 0.0;

  const score =
    weights.crossing * crossingScore +
    weights.deform * deformScore +
    weights.view * viewScore +
    weights.occlusion * occlusionScore +
    weights.deceptive * deceptiveScore;

  return {
    difficulty_score: clamp(score, 0, 1),
    difficulty: scoreToLevel(score),
    factors: {
      crossing: crossingScore,
      deform: deformScore,
      view: viewScore,
      occlusion: occlusionScore,
      deceptive: deceptiveScore,
    },
  };
}

// ============= Pair (Invariance) =============

export function computePairDifficulty(imageA, imageB, isEquivalent, topoA, topoB) {
  if (isEquivalent) return computePositivePairDifficulty(imageA, imageB);
  return computeNegativePairDifficulty(imageA, imageB, topoA, topoB);
}

function computePositivePairDifficulty(paramsA, paramsB) {
  const weights = { viewGap: 0.35, shapeGap: 0.40, maxSingle: 0.25 };

  const viewGap = computeViewAngleDiff(paramsA?.cameraPosition, paramsB?.cameraPosition);
  const deformA = clamp(paramsA?.deformStrength ?? 0.3, 0, 1);
  const deformB = clamp(paramsB?.deformStrength ?? 0.3, 0, 1);
  const deformDiff = Math.abs(deformA - deformB);
  const maxDeform = Math.max(deformA, deformB);
  const shapeGap = deformDiff * 0.6 + maxDeform * 0.4;

  const singleA = computeSingleClosedLoopDifficulty(paramsA || {}).difficulty_score;
  const singleB = computeSingleClosedLoopDifficulty(paramsB || {}).difficulty_score;
  const maxSingle = Math.max(singleA, singleB);

  const score = weights.viewGap * viewGap + weights.shapeGap * shapeGap + weights.maxSingle * maxSingle;
  return {
    difficulty_score: clamp(score, 0, 1),
    difficulty: scoreToLevel(score),
    factors: { viewGap, shapeGap, maxSingle },
  };
}

function computeNegativePairDifficulty(paramsA, paramsB, topoA, topoB) {
  const weights = { visualSimilarity: 0.40, crossingProximity: 0.25, confusing: 0.20, deceptive: 0.15 };

  const viewSim = 1 - computeViewAngleDiff(paramsA?.cameraPosition, paramsB?.cameraPosition);
  const deformA = clamp(paramsA?.deformStrength ?? 0.3, 0, 1);
  const deformB = clamp(paramsB?.deformStrength ?? 0.3, 0, 1);
  const deformSim = 1 - Math.abs(deformA - deformB);
  const visualSimilarity = clamp(viewSim * 0.5 + deformSim * 0.5, 0, 1);

  const crossA = getCrossingNumber(paramsA?.knotType) || 0;
  const crossB = getCrossingNumber(paramsB?.knotType) || 0;
  const crossingProximity = 1 - clamp(Math.abs(crossA - crossB) / 6, 0, 1);

  const confusing = isConfusingPair(paramsA?.knotType, paramsB?.knotType);
  const confusingScore = confusing.isConfusing ? 1.0 : 0.0;

  const hasDeceptive = isDeceptiveKnot(paramsA?.knotType) || isDeceptiveKnot(paramsB?.knotType);
  const deceptiveScore = hasDeceptive ? 1.0 : 0.0;

  const score =
    weights.visualSimilarity * visualSimilarity +
    weights.crossingProximity * crossingProximity +
    weights.confusing * confusingScore +
    weights.deceptive * deceptiveScore;

  return {
    difficulty_score: clamp(score, 0, 1),
    difficulty: scoreToLevel(score),
    factors: {
      visualSimilarity,
      crossingProximity,
      confusing: confusingScore,
      confusingReason: confusing.reason || null,
      deceptive: deceptiveScore,
      topoA: topoA ?? null,
      topoB: topoB ?? null,
    },
  };
}
