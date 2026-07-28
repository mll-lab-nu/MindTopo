/**
 * Invariance Generator
 * 
 * 生成用于测试拓扑不变性的 Pair 数据集。
 * 
 * 核心功能：
 * 1. 生成 Positive Pairs（拓扑等价）
 * 2. 生成 Negative Pairs（拓扑不等价）
 * 3. 计算 difficulty_score 和 similarity_score
 * 4. 支持难度分层采样
 */

import {
  KNOT_TYPE_REGISTRY,
  TOPOLOGICAL_CLASSES,
  CONFUSING_PAIRS,
  KNOT_TYPES_BY_DIFFICULTY,
  KNOT_ONLY_TOPOLOGICAL_IDS,
  getTopologicalId,
  areTopologicallyEquivalent,
  getGeneratorsByTopologicalId,
  isDeceptiveKnot,
  isLink,
  getCrossingNumber,
  isConfusingPair,
} from './knot-type-registry.js';
import { KNOWN_GAUSS_CODES } from './gauss-code-generator.js';
import {
  computePairDifficulty,
  computeViewAngleDiff as computeViewAngleDiffUnified,
  getBucketTopology,
  getBucketSaliency,
  getTrapType,
} from './difficulty-controller.js';

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
  const seedFn = xmur3(String(seedStr || 'invariance-seed'));
  return mulberry32(seedFn());
}

// ============= 辅助函数 =============

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// 针对特定 knot 类型收紧参数，避免尖刺/穿模或视觉彻底退化
function clampParamsByKnotType(knotType, params) {
  const slacknessCaps = {
    unknot: 0.3,
    twisted_ring: 0.3,
    spiral_disk: 0.2,
    figure8: 0.35,
    trefoil: 0.70,
    torus_2_5: 0.75,
    kinky_unknot: 0.35,
    ring_like_open_rope: 0.95,
    loose_open_knot: 0.60,
    ring_like_open_knot: 1.00,
  };
  const slacknessFloors = {
  };
  const deformCaps = {
    figure8: 0.20,
    spiral_disk: 0.28,
    twisted_ring: 0.40,
  };

  return {
    ...params,
    slackness: Math.max(
      slacknessFloors[knotType] ?? 0,
      Math.min(params.slackness, slacknessCaps[knotType] ?? 0.80),
    ),
    deformStrength: Math.min(params.deformStrength, deformCaps[knotType] ?? 1.0),
  };
}

function pick(rng, arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(rng() * arr.length)];
}

function pickExcluding(rng, arr, exclude) {
  const filtered = arr.filter(item => item !== exclude);
  if (filtered.length === 0) return pick(rng, arr);
  return pick(rng, filtered);
}

function shuffleArray(rng, arr) {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function randRange(rng, min, max) {
  return min + (max - min) * rng();
}

function randInt(rng, min, max) {
  return Math.floor(randRange(rng, min, max + 1));
}

// ============= 颜色与材质（固定） =============

// 颜色/材质：随机糖果色 + 固定材质，避免干扰拓扑判断
const PASTEL_COLORS = [
  '#ffd5e5', '#d5f4ff', '#fff4d5', '#e5d5ff',
  '#d5ffe5', '#ffe5d5', '#f4d5ff', '#d5fff4',
  '#e8e8f5', '#f3e2ff'
];
const FIXED_BACKGROUND = '#1a2236';     // 深灰背景
const FIXED_METALNESS = 0.10;           // 塑胶/橡胶质感
const FIXED_ROUGHNESS = 0.40;
const FIXED_TUBE_RADIUS = 0.07;         // 统一粗细，进一步变细以减少穿模
const FIXED_LIGHT_INTENSITY = 1.2;
const FIXED_AMBIENT_INTENSITY = 0.9;

// ============= 相机工具 =============

/**
 * 生成随机相机位置（球坐标）
 */
function randomCameraPosition(rng, options = {}) {
  const {
    minRadius = 10.0,         // 拉远相机距离，避免出框
    maxRadius = 14.0,
    minPhi = 0.2,             // 避免正上方
    maxPhi = Math.PI - 0.2,   // 避免正下方
  } = options;
  
  const radius = randRange(rng, minRadius, maxRadius);
  const theta = rng() * Math.PI * 2;  // 水平角度
  const phi = randRange(rng, minPhi, maxPhi);  // 垂直角度
  
  const x = radius * Math.sin(phi) * Math.cos(theta);
  const y = radius * Math.cos(phi);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  
  return [x, y, z];
}

/**
 * 计算两个相机位置的视角差异（0-1）
 */
function computeViewAngleDiff(posA, posB) {
  // Keep signature for local callers; delegate to unified controller.
  return computeViewAngleDiffUnified(posA, posB);
}

// ============= Image Params 生成 =============

/**
 * @typedef {Object} ImageParams
 * @property {string} knotType - 绳结类型 key
 * @property {number} seed - RNG seed
 * @property {number} deformStrength - 变形强度 (0-1)
 * @property {number} slackness - 松紧度 (0-1)
 * @property {string} bucket_topology - 拓扑分桶 (low|mid|high)
 * @property {string} bucket_saliency - 显著度分桶 (tight|medium|loose)
 * @property {string|null} trap_type - 陷阱类型
 * @property {number[]} cameraPosition - 相机位置 [x, y, z]
 * @property {number[]} cameraTarget - 相机目标 [x, y, z]
 * @property {number} cameraFov - 相机视场角
 * @property {string} color - 颜色 (hex) 固定
 * @property {number} metalness - 金属度 (0-1) 固定
 * @property {number} roughness - 粗糙度 (0-1) 固定
 * @property {number} tubeRadius - 管道半径 固定
 * @property {number} lightIntensity - 光照强度 固定
 * @property {number} ambientIntensity - 环境光强度 固定
 * @property {string} backgroundColor - 背景色 (hex) 固定
 * @property {string} gaussCode - 预期的 Gauss Code（若已知，否则空字符串）
 */
// 根据 knotType/topoId 选择已知 Gauss Code；未知则返回空字符串
function resolveGaussCode(knotType) {
  if (!knotType) return '';
  const id = getTopologicalId(knotType);
  // 明确映射
  const map = {
    trefoil: KNOWN_GAUSS_CODES.trefoil,
    figure8: KNOWN_GAUSS_CODES.figure8,
    torus_2_5: KNOWN_GAUSS_CODES.cinquefoil, // cinquefoil 5_1
    cinquefoil: KNOWN_GAUSS_CODES.cinquefoil,
    unknot: '0',
  };
  // 直接按 key 命中
  if (map[knotType]) return map[knotType];
  // 按 topologicalId 尝试
  const topoMap = {
    [TOPOLOGICAL_CLASSES.TREFOIL]: KNOWN_GAUSS_CODES.trefoil,
    [TOPOLOGICAL_CLASSES.FIGURE8]: KNOWN_GAUSS_CODES.figure8,
    [TOPOLOGICAL_CLASSES.CINQUEFOIL_5_1]: KNOWN_GAUSS_CODES.cinquefoil,
    [TOPOLOGICAL_CLASSES.UNKNOT]: '0',
  };
  if (topoMap[id]) return topoMap[id];
  return '';
}

/**
 * 生成随机的 ImageParams
 * @param {Function} rng - 随机数生成器
 * @param {string} knotType - 绳结类型 key
 * @param {Object} constraints - 约束条件
 * @returns {ImageParams}
 */
export function generateRandomImageParams(rng, knotType, constraints = {}) {
  const {
    deformRange = [0.1, 0.5],   // 上限收紧，减少过度扰动导致的穿模
    slacknessRange = [0.0, 0.80],
    fovRange = [35, 55],
  } = constraints;
  
  const seed = Math.floor(rng() * 1000000);
  const clampedGeomParams = clampParamsByKnotType(knotType, {
    deformStrength: randRange(rng, deformRange[0], deformRange[1]),
    slackness: randRange(rng, slacknessRange[0], slacknessRange[1]),
  });
  const slackness = clampedGeomParams.slackness;
    
    return {
    knotType,
    seed,
    
    // Geometry Deformation（核心随机因素）
    deformStrength: clampedGeomParams.deformStrength,
    slackness,
    bucket_topology: getBucketTopology(knotType),
    bucket_saliency: getBucketSaliency(slackness),
    trap_type: getTrapType(knotType, slackness),
    
    // Camera（保持完全随机视角）
    cameraPosition: randomCameraPosition(rng),
    cameraTarget: [0, 0, 0],
    cameraFov: randRange(rng, fovRange[0], fovRange[1]),
    
    // Material & Lighting（固定，去除干扰）
    color: pick(rng, PASTEL_COLORS),
    metalness: FIXED_METALNESS,
    roughness: FIXED_ROUGHNESS,
    tubeRadius: FIXED_TUBE_RADIUS,
    lightIntensity: FIXED_LIGHT_INTENSITY,
    ambientIntensity: FIXED_AMBIENT_INTENSITY,
      
    // Background（固定）
    backgroundColor: FIXED_BACKGROUND,

    // Gauss Code（若已知则填充）
    gaussCode: resolveGaussCode(knotType),
    };
  }

  /**
 * 生成"相似"的 ImageParams（用于 Easy Pairs）
 * 基于参考参数，只做微小调整
 */
export function generateSimilarImageParams(rng, knotType, referenceParams, similarity = 0.8) {
  const blend = (a, b, t) => a + (b - a) * t;
  const diff = 1 - similarity;
  
  // 基于参考参数，但保持固定材质与粗细
  const newParams = generateRandomImageParams(rng, knotType);
  const blendedDeform = blend(referenceParams.deformStrength, newParams.deformStrength, diff);
  const blendedSlackness = blend(referenceParams.slackness ?? 0, newParams.slackness ?? 0, diff);
  const clampedGeomParams = clampParamsByKnotType(knotType, {
    deformStrength: blendedDeform,
    slackness: blendedSlackness,
  });
    
    return {
    ...newParams,
    knotType,
    
    // 混合：保持大部分相似
    deformStrength: clampedGeomParams.deformStrength,
    slackness: clampedGeomParams.slackness,
    bucket_topology: getBucketTopology(knotType),
    bucket_saliency: getBucketSaliency(clampedGeomParams.slackness),
    trap_type: getTrapType(knotType, clampedGeomParams.slackness),
    
    // 相机位置：只做小幅调整
    cameraPosition: referenceParams.cameraPosition.map((v, i) => 
      v + (newParams.cameraPosition[i] - v) * diff * 0.5
    ),
    cameraFov: blend(referenceParams.cameraFov, newParams.cameraFov, diff * 0.3),
    
    // 材质/颜色/粗细全部固定
    color: pick(rng, PASTEL_COLORS),
    metalness: FIXED_METALNESS,
    roughness: FIXED_ROUGHNESS,
    tubeRadius: FIXED_TUBE_RADIUS,
    backgroundColor: FIXED_BACKGROUND,

    // Gauss Code（若已知则填充）
    gaussCode: resolveGaussCode(knotType),
  };
}

// 确保正负样本的形变差异至少达到阈值
function ensureDeformGap(paramsA, paramsB, rng, minGap = 0.18) {
  let attempts = 0;
  while (Math.abs(paramsA.deformStrength - paramsB.deformStrength) < minGap && attempts < 5) {
    // 随机重新抽样 B 的形变
    paramsB.deformStrength = randRange(rng, 0.1, 0.8);
    attempts++;
  }
}

// ============= Pair 生成 =============

/**
 * @typedef {Object} PairRecord
 * @property {string} pairId
 * @property {ImageParams} imageA
 * @property {ImageParams} imageB
 * @property {boolean} label_equivalent
 * @property {string} topologicalIdA
 * @property {string} topologicalIdB
 * @property {number} difficulty_score
 * @property {number} similarity_score
 * @property {Object} difficulty_factors
 * @property {string} imagePathA
 * @property {string} imagePathB
 */

/**
 * 生成 Positive Pair（拓扑等价）
 * @param {Function} rng 
 * @param {'easy'|'medium'|'hard'} targetDifficulty 
 * @param {Object} options
 * @returns {PairRecord}
 */
export function generatePositivePair(rng, targetDifficulty = 'medium', options = {}) {
  const {
    allowedTypes = null,  // 如果指定，只从这些类型中选择
    includeDeceptive = true,
    imageConstraintsA = null,
    imageConstraintsB = null,
  } = options;
  
  // 1. 选择一个拓扑类
  let availableTopoIds = KNOT_ONLY_TOPOLOGICAL_IDS;
  
  // 2. 获取该拓扑类下的所有 generator
  const topologicalId = pick(rng, availableTopoIds);
  let generators = getGeneratorsByTopologicalId(topologicalId);
  
  // 如果有类型限制，过滤
  if (allowedTypes) {
    generators = generators.filter(g => allowedTypes.includes(g.key));
  }
  
  // 如果不包含欺骗性类型，过滤
  if (!includeDeceptive) {
    generators = generators.filter(g => !g.entry.isDeceptive);
  }
  
  if (generators.length === 0) {
    // Fallback：使用 unknot
    generators = getGeneratorsByTopologicalId(TOPOLOGICAL_CLASSES.UNKNOT);
  }
  
  // 3. 根据难度决定如何选择两个 generator
  let knotTypeA, knotTypeB;
  
  if (targetDifficulty === 'easy') {
    // Easy：使用同一个 generator，参数相似
    const gen = pick(rng, generators);
    knotTypeA = gen.key;
    knotTypeB = gen.key;
  } else if (targetDifficulty === 'hard') {
    // Hard：尽量使用不同的 generator（如果可用）
    if (generators.length >= 2) {
      const shuffled = shuffleArray(rng, generators);
      knotTypeA = shuffled[0].key;
      knotTypeB = shuffled[1].key;
    } else {
      // 只有一个 generator，增加变形差异
      knotTypeA = generators[0].key;
      knotTypeB = generators[0].key;
    }
  } else {
    // Medium：随机选择
    knotTypeA = pick(rng, generators).key;
    knotTypeB = pick(rng, generators).key;
  }
  
  // 4. 生成 ImageParams
  let imageA, imageB;
  const mergeConstraints = (base, extra) => ({ ...(base || {}), ...(extra || {}) });
  
  if (targetDifficulty === 'easy') {
    // Easy：参数相似
    imageA = generateRandomImageParams(rng, knotTypeA, imageConstraintsA || {});
    imageB = generateSimilarImageParams(rng, knotTypeB, imageA, 0.75);
    if (imageConstraintsB) {
      imageB = generateRandomImageParams(rng, knotTypeB, imageConstraintsB);
    }
  } else if (targetDifficulty === 'hard') {
    // Hard：参数差异大
    imageA = generateRandomImageParams(rng, knotTypeA, {
      ...mergeConstraints({ deformRange: [0.1, 0.4] }, imageConstraintsA),
    });
    imageB = generateRandomImageParams(rng, knotTypeB, {
      ...mergeConstraints({ deformRange: [0.5, 0.8] }, imageConstraintsB),
    });
  } else {
    // Medium：随机
    imageA = generateRandomImageParams(rng, knotTypeA, imageConstraintsA || {});
    imageB = generateRandomImageParams(rng, knotTypeB, imageConstraintsB || {});
  }
  // 强制形变差异和不同 seed
  ensureDeformGap(imageA, imageB, rng, 0.18);
  if (imageA.seed === imageB.seed) imageB.seed += 1;
  
  // 5. 计算评分
  const difficulty = computePairDifficulty(imageA, imageB, true, topologicalId, topologicalId);
  const similarity = computeVisualSimilarity(imageA, imageB);
  
  return {
    pairId: '',  // 稍后填充
    imageA,
    imageB,
    label_equivalent: true,
    topologicalIdA: topologicalId,
    topologicalIdB: topologicalId,
    difficulty: difficulty.difficulty,
    difficulty_score: difficulty.difficulty_score,
    similarity_score: similarity,
    difficulty_factors: difficulty.factors,
    imagePathA: '',
    imagePathB: '',
  };
}

/**
 * 生成 Negative Pair（拓扑不等价）
 * @param {Function} rng 
 * @param {'easy'|'medium'|'hard'} targetDifficulty 
 * @param {Object} options
 * @returns {PairRecord}
 */
export function generateNegativePair(rng, targetDifficulty = 'medium', options = {}) {
  const {
    allowedTypes = null,
    includeDeceptive = true,
    imageConstraintsA = null,
    imageConstraintsB = null,
  } = options;
  
  let knotTypeA, knotTypeB;
  let topologicalIdA, topologicalIdB;
  
  if (targetDifficulty === 'hard' && includeDeceptive) {
    // Hard：优先使用预定义的混淆组合
    const confusingPair = pick(rng, CONFUSING_PAIRS);
    if (confusingPair && rng() < 0.7) {
      knotTypeA = confusingPair.a;
      knotTypeB = confusingPair.b;
      topologicalIdA = getTopologicalId(knotTypeA);
      topologicalIdB = getTopologicalId(knotTypeB);
    }
  }
  
  // 如果没有选择混淆组合，随机选择两个不同的拓扑类
  if (!knotTypeA || !knotTypeB) {
    let availableTopoIds = [...KNOT_ONLY_TOPOLOGICAL_IDS];
    
    if (targetDifficulty === 'easy') {
      // Easy：选择交叉数差异大的
      // 策略：一个 unknot，一个高交叉数
      topologicalIdA = TOPOLOGICAL_CLASSES.UNKNOT;
      topologicalIdB = pick(rng, [
        TOPOLOGICAL_CLASSES.TORUS_2_7,
        TOPOLOGICAL_CLASSES.TORUS_2_9,
        TOPOLOGICAL_CLASSES.TORUS_3_5,
      ]);
    } else if (targetDifficulty === 'medium') {
      // Medium：随机选择两个不同的
      topologicalIdA = pick(rng, availableTopoIds);
      topologicalIdB = pickExcluding(rng, availableTopoIds, topologicalIdA);
  } else {
      // Hard：选择交叉数接近的
      const closeTopoIds = [
        [TOPOLOGICAL_CLASSES.TREFOIL, TOPOLOGICAL_CLASSES.FIGURE_EIGHT],  // 3 vs 4
        [TOPOLOGICAL_CLASSES.CINQUEFOIL, TOPOLOGICAL_CLASSES.TORUS_2_7],  // 5 vs 7
        [TOPOLOGICAL_CLASSES.UNKNOT, TOPOLOGICAL_CLASSES.TREFOIL],        // 欺骗性 unknot vs trefoil
      ];
      const selectedPair = pick(rng, closeTopoIds);
      topologicalIdA = selectedPair[0];
      topologicalIdB = selectedPair[1];
    }
    
    // 从拓扑类中选择具体的 generator
    const gensA = getGeneratorsByTopologicalId(topologicalIdA, { excludeDeceptive: !includeDeceptive });
    const gensB = getGeneratorsByTopologicalId(topologicalIdB, { excludeDeceptive: !includeDeceptive });
    
    knotTypeA = (gensA.length > 0 ? pick(rng, gensA) : { key: 'unknot' }).key;
    knotTypeB = (gensB.length > 0 ? pick(rng, gensB) : { key: 'trefoil' }).key;
    
    topologicalIdA = getTopologicalId(knotTypeA);
    topologicalIdB = getTopologicalId(knotTypeB);
    }
    
  // 生成 ImageParams
  let imageA, imageB;
  const mergeConstraints = (base, extra) => ({ ...(base || {}), ...(extra || {}) });
  
  if (targetDifficulty === 'easy') {
    // Easy：视觉差异大
    imageA = generateRandomImageParams(rng, knotTypeA, imageConstraintsA || {});
    imageB = generateRandomImageParams(rng, knotTypeB, imageConstraintsB || {});
  } else if (targetDifficulty === 'hard') {
    // Hard：视觉尽量相似
    imageA = generateRandomImageParams(rng, knotTypeA, imageConstraintsA || {});
    imageB = generateSimilarImageParams(rng, knotTypeB, imageA, 0.6);
    if (imageConstraintsB) {
      imageB = generateRandomImageParams(rng, knotTypeB, imageConstraintsB);
    }
  } else {
    // Medium：随机
    imageA = generateRandomImageParams(rng, knotTypeA, mergeConstraints({}, imageConstraintsA));
    imageB = generateRandomImageParams(rng, knotTypeB, mergeConstraints({}, imageConstraintsB));
  }
    
  // 计算评分
  const difficulty = computePairDifficulty(imageA, imageB, false, topologicalIdA, topologicalIdB);
  const similarity = computeVisualSimilarity(imageA, imageB);

  return { 
    pairId: '',
    imageA,
    imageB,
    label_equivalent: false,
    topologicalIdA,
    topologicalIdB,
    difficulty: difficulty.difficulty,
    difficulty_score: difficulty.difficulty_score,
    similarity_score: similarity,
    difficulty_factors: difficulty.factors,
    imagePathA: '',
    imagePathB: '',
  };
}

// ============= Similarity (kept) =============

/**
 * 计算视觉相似度（0-1，越高越相似）
 * 仅关注形变 + 视角，不考虑颜色/材质/光照
 */
function computeVisualSimilarity(paramsA, paramsB) {
  const viewSim = 1 - computeViewAngleDiff(paramsA.cameraPosition, paramsB.cameraPosition);
  const deformSim = 1 - Math.abs(paramsA.deformStrength - paramsB.deformStrength);
  return clamp(0.55 * viewSim + 0.45 * deformSim, 0, 1);
}

// ============= 批量生成 =============

/**
 * @typedef {Object} DatasetConfig
 * @property {number} numPairs - 总 pair 数
 * @property {number} positiveRatio - positive pairs 占比 (0-1)
 * @property {string} seed - 随机种子
 * @property {Object} difficultyDistribution - 难度分布 { easy, medium, hard }
 * @property {boolean} includeDeceptive - 是否包含欺骗性类型
 * @property {string[]} allowedTypes - 允许的绳结类型（null 表示全部）
 */

/**
 * 根据难度分布采样难度级别
 */
function sampleDifficulty(rng, distribution) {
  const { easy = 0.33, medium = 0.34, hard = 0.33 } = distribution;
  const total = easy + medium + hard;
  const r = rng() * total;
  
  if (r < easy) return 'easy';
  if (r < easy + medium) return 'medium';
  return 'hard';
  }

const TOPOLOGY_BUCKETS = ['low', 'mid', 'high'];
const SALIENCY_BUCKETS = ['tight', 'medium', 'loose'];

function slacknessRangeForBucket(bucket) {
  if (bucket === 'tight') return [0.0, 0.25];
  if (bucket === 'medium') return [0.26, 0.60];
  return [0.61, 0.80];
}

function getKnotTypesByTopologyBucket(bucket, { allowedTypes = null, includeDeceptive = true } = {}) {
  return Object.entries(KNOT_TYPE_REGISTRY)
    .filter(([key, entry]) => {
      if (!entry || entry.isLink) return false;
      if (allowedTypes && !allowedTypes.includes(key)) return false;
      if (!includeDeceptive && entry.isDeceptive) return false;
      return getBucketTopology(key) === bucket;
    })
    .map(([key]) => key);
}

  /**
 * 生成完整的 Invariance 数据集
 * @param {DatasetConfig} config
 * @returns {Object} { pairs: PairRecord[], statistics: Object }
 */
export function generateInvarianceDataset(config) {
  const {
    numPairs = 20,
    positiveRatio = 0.5,
    seed = 'invariance-v1',
    difficultyDistribution = { easy: 0.33, medium: 0.34, hard: 0.33 },
    includeDeceptive = true,
    allowedTypes = null,
  } = config;
  
  const rng = makeRng(seed);
  
  const numPositive = Math.round(numPairs * positiveRatio);
  const numNegative = numPairs - numPositive;
  const TARGET_PER_BUCKET = Math.floor(numPairs / (TOPOLOGY_BUCKETS.length * SALIENCY_BUCKETS.length));
  const bucketCounts = {};
  TOPOLOGY_BUCKETS.forEach((t) => SALIENCY_BUCKETS.forEach((s) => {
    bucketCounts[`${t}_${s}`] = 0;
  }));

  function pickUndersampledBucket() {
    const entries = Object.entries(bucketCounts).sort((a, b) => a[1] - b[1]);
    const underTarget = entries.find(([, count]) => count < TARGET_PER_BUCKET);
    return (underTarget || entries[0])[0];
  }

  function generatePairForBucket(labelEquivalent, difficulty, targetBucketKey) {
    const [topologyBucket, saliencyBucket] = String(targetBucketKey).split('_');
    const bucketAllowedTypes = getKnotTypesByTopologyBucket(topologyBucket, { allowedTypes, includeDeceptive });
    const imageConstraintsA = { slacknessRange: slacknessRangeForBucket(saliencyBucket) };

    const pairOptions = {
      allowedTypes: bucketAllowedTypes.length > 0 ? bucketAllowedTypes : allowedTypes,
      includeDeceptive,
      imageConstraintsA,
    };

    let pair = null;
    for (let attempt = 0; attempt < 10; attempt++) {
      pair = labelEquivalent
        ? generatePositivePair(rng, difficulty, pairOptions)
        : generateNegativePair(rng, difficulty, pairOptions);

      const t = pair?.imageA?.bucket_topology || getBucketTopology(pair?.imageA?.knotType);
      const s = pair?.imageA?.bucket_saliency || getBucketSaliency(pair?.imageA?.slackness ?? 0);
      if (t === topologyBucket && s === saliencyBucket) {
        return pair;
      }
    }

    return pair;
  }
  
  const pairs = [];
  const statistics = {
    total: numPairs,
    positive: numPositive,
    negative: numNegative,
    byDifficulty: { easy: 0, medium: 0, hard: 0 },
    byKnotType: {},
    byTopologicalId: {},
    byBucket: bucketCounts,
  };
        
  // 生成 Positive Pairs
  for (let i = 0; i < numPositive; i++) {
    const difficulty = sampleDifficulty(rng, difficultyDistribution);
    const targetBucket = pickUndersampledBucket();
    const pair = generatePairForBucket(true, difficulty, targetBucket);
    pairs.push(pair);
    
    // 统计
    statistics.byDifficulty[difficulty]++;
    statistics.byKnotType[pair.imageA.knotType] = (statistics.byKnotType[pair.imageA.knotType] || 0) + 1;
    statistics.byKnotType[pair.imageB.knotType] = (statistics.byKnotType[pair.imageB.knotType] || 0) + 1;
    statistics.byTopologicalId[pair.topologicalIdA] = (statistics.byTopologicalId[pair.topologicalIdA] || 0) + 1;
    const bk = `${pair.imageA.bucket_topology || getBucketTopology(pair.imageA.knotType)}_${pair.imageA.bucket_saliency || getBucketSaliency(pair.imageA.slackness ?? 0)}`;
    if (bucketCounts[bk] !== undefined) bucketCounts[bk] += 1;
  }
  
  // 生成 Negative Pairs
  for (let i = 0; i < numNegative; i++) {
    const difficulty = sampleDifficulty(rng, difficultyDistribution);
    const targetBucket = pickUndersampledBucket();
    const pair = generatePairForBucket(false, difficulty, targetBucket);
    pairs.push(pair);
    
    // 统计
    statistics.byDifficulty[difficulty]++;
    statistics.byKnotType[pair.imageA.knotType] = (statistics.byKnotType[pair.imageA.knotType] || 0) + 1;
    statistics.byKnotType[pair.imageB.knotType] = (statistics.byKnotType[pair.imageB.knotType] || 0) + 1;
    statistics.byTopologicalId[pair.topologicalIdA] = (statistics.byTopologicalId[pair.topologicalIdA] || 0) + 1;
    statistics.byTopologicalId[pair.topologicalIdB] = (statistics.byTopologicalId[pair.topologicalIdB] || 0) + 1;
    const bk = `${pair.imageA.bucket_topology || getBucketTopology(pair.imageA.knotType)}_${pair.imageA.bucket_saliency || getBucketSaliency(pair.imageA.slackness ?? 0)}`;
    if (bucketCounts[bk] !== undefined) bucketCounts[bk] += 1;
  }
  
  // 打乱顺序
  const shuffledPairs = shuffleArray(rng, pairs);
    
  // 分配 pairId 和文件路径
  shuffledPairs.forEach((pair, index) => {
    const pairId = `pair${String(index + 1).padStart(4, '0')}`;
    pair.pairId = pairId;
    pair.imagePathA = `${pairId}_1.png`;
    pair.imagePathB = `${pairId}_2.png`;
    });

  console.group('=== Dataset Bucket Distribution ===');
  console.table(
    TOPOLOGY_BUCKETS.flatMap((t) =>
      SALIENCY_BUCKETS.map((s) => ({
        topology: t,
        saliency: s,
        count: bucketCounts[`${t}_${s}`],
        trap_loose_knot: pairs.filter((p) =>
          p.imageA.bucket_topology === t &&
          p.imageA.bucket_saliency === s &&
          p.imageA.trap_type === 'loose_knot'
        ).length,
      }))
    )
  );
  console.groupEnd();
    
  return {
    pairs: shuffledPairs,
    statistics,
    config: {
      numPairs,
      positiveRatio,
      seed,
      difficultyDistribution,
      includeDeceptive,
      allowedTypes,
      generatedAt: new Date().toISOString(),
      },
    };
}

/**
 * 将数据集转换为 JSONL 格式
 * @param {PairRecord[]} pairs 
 * @returns {string} JSONL 字符串
 */
export function pairsToJsonl(pairs) {
  return pairs.map(pair => JSON.stringify(pair)).join('\n');
}

/**
 * 导出数据集元信息（用于生成 metadata.json）
 */
export function generateDatasetMetadata(dataset) {
  return {
    version: '1.0.0',
    name: 'Knot Invariance Dataset',
    description: '用于测试模型拓扑不变性理解能力的图像对数据集',
    ...dataset.config,
    statistics: dataset.statistics,
    schema: {
      pairId: 'string - 唯一标识符',
      imageA: 'ImageParams - 图片 A 的生成参数',
      imageB: 'ImageParams - 图片 B 的生成参数',
      label_equivalent: 'boolean - 是否拓扑等价（核心标签）',
      topologicalIdA: 'string - 图片 A 的拓扑类 ID',
      topologicalIdB: 'string - 图片 B 的拓扑类 ID',
      difficulty: "string: 'easy'|'medium'|'hard' - unified difficulty level",
      difficulty_score: 'number [0-1] - 任务难度',
      similarity_score: 'number [0-1] - 视觉相似度',
      difficulty_factors: 'object - 难度计算的详细因素',
      imagePathA: 'string - 图片 A 的文件名',
      imagePathB: 'string - 图片 B 的文件名',
    },
  };
}

// ============= 单图数据集生成（Single-Image Tasks T01-T09） =============

/**
 * 数据集规格 — 每种 knot 类型生成多少样本、多少视角
 */
const SINGLE_IMAGE_SPEC = {
  // 每个 knot 类型 × 参数变体数 × 相机角度数 = 总图数
  anglesPerSample: 3,       // 每个参数变体渲染3个视角（比8角精简）
  preferredAngles: [        // 优选视角子集
    { name: 'iso_fr',  pos: [5, 5, 7] },
    { name: 'front',   pos: [0, 2, 8] },
    { name: 'oblique', pos: [3, 8, 4] },
  ],
};

/**
 * 为单个 knot 类型生成一批 ImageParams
 * @param {Function} rng
 * @param {string} knotType
 * @param {number} numVariants - 参数变体数
 * @returns {Object[]} - 包含 metadata 的样本数组
 */
function generateSingleSamplesForType(rng, knotType, numVariants) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  if (!entry) return [];

  const samples = [];

  for (let v = 0; v < numVariants; v++) {
    // 均匀覆盖 slackness 空间
    const slacknessTarget = v / Math.max(numVariants - 1, 1);  // 0→1 均匀
    const params = generateRandomImageParams(rng, knotType, {
      slacknessRange: [
        Math.max(0, slacknessTarget - 0.15),
        Math.min(0.80, slacknessTarget + 0.15),
      ],
    });

    // 为每个变体选择多个视角
    const angles = SINGLE_IMAGE_SPEC.preferredAngles;
    const images = angles.map(angle => ({
      filename: '', // 渲染时填充
      cameraAngle: angle.name,
      cameraPos: angle.pos,
    }));

    const sample = {
      version: '2.0',
      knotType,
      topologicalId: entry.topologicalId,
      isKnot: entry.isKnot ?? false,
      isUnknot: entry.isUnknot ?? false,
      isDeceptive: entry.isDeceptive ?? false,
      isLink: entry.isLink ?? false,
      numComponents: entry.numComponents ?? 1,
      crossingNumber: entry.crossingNumber,
      family: entry.family,
      ...(entry.isOpenRope ? { isOpenRope: true } : {}),

      // 生成参数
      seed: params.seed,
      slackness: params.slackness,
      deformStrength: params.deformStrength,

      // Difficulty
      bucket_topology: params.bucket_topology,
      bucket_saliency: params.bucket_saliency,
      trap_type: params.trap_type,
      difficulty: '', // 渲染后计算

      // 渲染参数
      color: params.color,
      metalness: params.metalness,
      roughness: params.roughness,
      tubeRadius: params.tubeRadius,
      backgroundColor: params.backgroundColor,
      lightIntensity: params.lightIntensity,
      ambientIntensity: params.ambientIntensity,
      gaussCode: params.gaussCode,

      // 图片列表
      images,

      // Ground truth（与 vlm_benchmark.py 对齐）
      groundTruth: {
        T01_knotted: entry.isKnot ? 'KNOTTED' : 'UNKNOTTED',
        T04_can_untie: entry.isKnot ? 'NO' : 'YES',
        T06_family: _FAMILY_MAP_JS[knotType] || 'OTHER',
        T08_trefoil: knotType === 'trefoil' ? 'TREFOIL' : 'NOT_TREFOIL',
        T09_trap: (entry.isDeceptive && !entry.isKnot) ? 'LOOSE_ILLUSION' : 'ACTUAL_KNOT',
      },
    };

    samples.push(sample);
  }

  return samples;
}

// Family map (JS side, matches Python _FAMILY_MAP)
const _FAMILY_MAP_JS = {
  unknot: 'UNKNOT', twisted_ring: 'UNKNOT', kinky_unknot: 'UNKNOT',
  ring_like_open_rope: 'UNKNOT',
  spiral_disk: 'UNKNOT',
  loose_open_knot: 'TORUS', ring_like_open_knot: 'TORUS',
  trefoil: 'TORUS', torus_2_5: 'TORUS', torus_2_7: 'TORUS',
  torus_2_9: 'TORUS', torus_3_4: 'TORUS', torus_3_5: 'TORUS',
  figure8: 'TWIST',
};

/**
 * 生成完整的单图数据集（用于 T01-T09 + T10-T12）
 *
 * @param {Object} config
 * @param {number} config.samplesPerType - 每种 knot 类型的参数变体数
 * @param {string} config.seed
 * @param {boolean} config.includeLinks - 是否包含链环类型
 * @returns {Object} { samples: Object[], statistics: Object }
 */
export function generateSingleImageDataset(config = {}) {
  const {
    samplesPerType = 5,
    seed = 'single-v1',
    includeLinks = true,
  } = config;

  const rng = makeRng(seed);
  const allSamples = [];
  const stats = {
    byType: {},
    byDifficulty: { easy: 0, medium: 0, hard: 0 },
    byFamily: {},
    totalImages: 0,
  };

  // 按照 knot 类型生成
  const typeList = Object.entries(KNOT_TYPE_REGISTRY)
    .filter(([_, entry]) => includeLinks || !entry.isLink)
    .map(([key]) => key);

  for (const knotType of typeList) {
    const samples = generateSingleSamplesForType(rng, knotType, samplesPerType);

    for (const sample of samples) {
      // 分配 ID
      const idx = allSamples.length;
      sample.id = `single_${String(idx).padStart(4, '0')}`;

      // 计算 difficulty
      const crossing = sample.crossingNumber || 0;
      const slackness = sample.slackness || 0;
      const trap = sample.trap_type;
      const score = 0.35 * Math.min(crossing / 10, 1) +
                    0.45 * slackness +
                    0.20 * (trap ? 0.3 : 0);
      sample.difficulty = score < 0.30 ? 'easy' : score < 0.60 ? 'medium' : 'hard';
      sample.difficulty_score = Math.round(score * 1000) / 1000;

      allSamples.push(sample);

      // Stats
      stats.byType[knotType] = (stats.byType[knotType] || 0) + 1;
      stats.byDifficulty[sample.difficulty]++;
      const family = _FAMILY_MAP_JS[knotType] || 'OTHER';
      stats.byFamily[family] = (stats.byFamily[family] || 0) + 1;
      stats.totalImages += sample.images.length;
    }
  }

  // Shuffle
  const shuffled = shuffleArray(rng, allSamples);

  console.log(`=== Single Image Dataset ===`);
  console.log(`Types: ${typeList.length}, Samples: ${shuffled.length}, Images: ${stats.totalImages}`);
  console.log(`Difficulty: easy=${stats.byDifficulty.easy}, medium=${stats.byDifficulty.medium}, hard=${stats.byDifficulty.hard}`);
  console.log(`Family:`, stats.byFamily);

  return {
    samples: shuffled,
    statistics: stats,
    config: { samplesPerType, seed, includeLinks, generatedAt: new Date().toISOString() },
  };
}

// ============= 导出 =============

export default {
  makeRng,
  generateRandomImageParams,
  generateSimilarImageParams,
  generatePositivePair,
  generateNegativePair,
  generateInvarianceDataset,
  generateSingleImageDataset,
  pairsToJsonl,
  generateDatasetMetadata,
};
