/**
 * Bead String Difficulty Controller
 *
 * Computes difficulty scores based on:
 * - num_beads, curve_complexity, occlusion, camera_angle, color_similarity
 */

import { BEAD_PALETTE } from './bead-renderer.js';

const SIMILAR_PAIRS = new Set([
  'RED-ORANGE', 'ORANGE-RED',
  'BLUE-PURPLE', 'PURPLE-BLUE',
  'WHITE-YELLOW', 'YELLOW-WHITE',
  'ORANGE-YELLOW', 'YELLOW-ORANGE',
  'BROWN-ORANGE', 'ORANGE-BROWN',
  'BROWN-RED', 'RED-BROWN',
]);

/**
 * Count visually similar adjacent color pairs in a bead sequence.
 */
function countSimilarPairs(beadColors) {
  let count = 0;
  for (let i = 1; i < beadColors.length; i++) {
    const key = `${beadColors[i - 1]}-${beadColors[i]}`;
    if (SIMILAR_PAIRS.has(key)) count++;
  }
  return count;
}

/**
 * Compute color similarity score (0..1).
 */
function colorSimilarityScore(beadColors) {
  if (beadColors.length <= 1) return 0;
  const maxPossiblePairs = beadColors.length - 1;
  const similarCount = countSimilarPairs(beadColors);
  return similarCount / maxPossiblePairs;
}

/**
 * Camera angle difficulty score.
 */
function cameraAngleScore(cameraAngle) {
  const scores = {
    front: 0.1,
    top: 0.15,
    iso_fr: 0.4,
    oblique: 0.7,
  };
  return scores[cameraAngle] || 0.3;
}

/**
 * Occlusion level score.
 */
function occlusionScore(occlusionLevel) {
  const scores = {
    none: 0,
    partial: 0.5,
    heavy: 1.0,
  };
  return scores[occlusionLevel] || 0;
}

/**
 * Compute overall difficulty for a bead string configuration.
 *
 * @param {Object} params
 * @param {number} params.numBeads - 3..10
 * @param {number} params.curveComplexity - 0..1
 * @param {string} params.occlusionLevel - 'none' | 'partial' | 'heavy'
 * @param {string} params.cameraAngle - 'front' | 'top' | 'iso_fr' | 'oblique'
 * @param {string[]} params.beadColors - array of color names
 * @returns {{ score: number, level: string, factors: Object }}
 */
export function computeBeadStringDifficulty(params) {
  const {
    numBeads = 5,
    curveComplexity = 0.3,
    occlusionLevel = 'none',
    cameraAngle = 'iso_fr',
    beadColors = [],
  } = params;

  const factors = {
    numBeads: Math.min(1, (numBeads - 3) / 7),     // 3→0, 10→1
    curveComplexity: curveComplexity,                 // already 0..1
    occlusion: occlusionScore(occlusionLevel),
    camera: cameraAngleScore(cameraAngle),
    colorSimilarity: colorSimilarityScore(beadColors),
  };

  // Weighted sum
  const weights = {
    numBeads: 0.25,
    curveComplexity: 0.20,
    occlusion: 0.25,
    camera: 0.15,
    colorSimilarity: 0.15,
  };

  let score = 0;
  for (const [key, weight] of Object.entries(weights)) {
    score += (factors[key] || 0) * weight;
  }
  score = Math.max(0, Math.min(1, score));

  const level = score < 0.30 ? 'easy' : score < 0.55 ? 'medium' : 'hard';

  return { score: +score.toFixed(4), level, factors };
}

export function difficultyPillClass(level) {
  return `pill-${level}`;
}

export default { computeBeadStringDifficulty, difficultyPillClass };
