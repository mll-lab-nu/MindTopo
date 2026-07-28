/**
 * Knot Type Registry
 * 
 * Centralized topological invariants for all knot types, used for equivalence checks.
 *
 * Key concepts:
 * - topologicalId: unique identifier for a topological equivalence class (invariant)
 * - All generators under the same topologicalId produce topologically equivalent knots
 * - Different topologicalIds are guaranteed non-equivalent
 */

// ============= Topological Equivalence Class Definitions =============

/**
 * Topological equivalence class enum.
 * Each class represents a unique topological structure.
 */
export const TOPOLOGICAL_CLASSES = {
  // === Unknot (trivial knot) ===
  UNKNOT: 'unknot',                    // 0_1, can be unknotted
  
  // === Prime Knots ===
  TREFOIL: 'trefoil_3_1',              // 3_1, trefoil
  FIGURE_EIGHT: 'figure8_4_1',         // 4_1, figure-eight
  CINQUEFOIL: 'cinquefoil_5_1',        // 5_1, T(2,5)
  THREE_TWIST: 'three_twist_5_2',      // 5_2
  STEVEDORE: 'stevedore_6_1',          // 6_1
  TORUS_2_7: 'torus_7_1',              // 7_1, T(2,7)
  TORUS_2_9: 'torus_9_1',              // 9_1, T(2,9)
  TORUS_3_4: 'torus_3_4',              // T(3,4), 8 crossings
  TORUS_3_5: 'torus_3_5',              // T(3,5), 10 crossings
  
  // === Links ===
  HOPF_LINK: 'hopf_link',              // two interlinked rings
  UNLINKED_2: 'unlinked_2',            // two unlinked rings
  CHAIN_N: 'chain',                    // chain (n rings)
  BORROMEAN: 'borromean_rings',        // three mutually interlinked rings

  // === Link Groups ===
  DOUBLE_HOPF: 'double_hopf_group',    // two Hopf link groups
  LINK_CLUSTER: 'link_cluster',        // mixed link cluster

  // === Mixed (linked + free) ===
  HOPF_PLUS_FREE: 'hopf_with_free',    // Hopf link + free ring
  CHAIN_PLUS_FREE: 'chain_with_free',  // chain + free ring

};

// ============= Knot Type Registry =============

/**
 * Complete knot type registry.
 *
 * Each entry contains:
 * - topologicalId: topological equivalence class ID (key for equivalence checks)
 * - crossingNumber: minimal crossing number (topological invariant)
 * - family: knot family (for UI grouping)
 * - generator: generator key (maps to PRESETS in unified-gallery.js)
 * - difficulty: visual recognition difficulty
 * - isDeceptive: whether visually complex but topologically simple
 * - isLink: whether it is a link (multi-component)
 * - aliases: alias list
 * - description: description
 */
export const KNOT_TYPE_REGISTRY = {
  // ========================================
  // === Topologically equivalent to Unknot ===
  // ========================================
  
  unknot: {
    topologicalId: TOPOLOGICAL_CLASSES.UNKNOT,
    crossingNumber: 0,
    family: 'unknot',
    generator: 'unknot',
    difficulty: 'easy',
    isKnot: false,
    isUnknot: true,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    aliases: ['circle', '0_1', 'trivial'],
    description: 'Trivial knot (simple circle)',
  },
  
  twisted_ring: {
    topologicalId: TOPOLOGICAL_CLASSES.UNKNOT,  // topologically still unknot!
    crossingNumber: 0,
    family: 'unknot_variant',
    generator: 'twisted_ring',
    difficulty: 'medium',
    isKnot: false,
    isUnknot: true,
    isDeceptive: true,  // visually has "fake crossings"
    isLink: false,
    numComponents: 1,
    aliases: ['wavy_ring', 'wobble_ring'],
    description: 'Twisted ring (visually wavy but topologically trivial)',
    visualComplexity: 'medium',
  },

  spiral_disk: {
    topologicalId: TOPOLOGICAL_CLASSES.UNKNOT,
    crossingNumber: 0,
    family: 'unknot_variant',
    generator: 'spiral_disk',
    difficulty: 'medium',
    isKnot: false,
    isUnknot: true,
    isDeceptive: true,
    isLink: false,
    numComponents: 1,
    aliases: ['spiral_loop', 'coil'],
    description: 'Spiral loop (closed spiral, topologically trivial)',
    visualComplexity: 'medium',
  },

  kinky_unknot: {
    topologicalId: TOPOLOGICAL_CLASSES.UNKNOT,
    crossingNumber: 0,
    family: 'unknot_variant',
    generator: 'kinky_unknot',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: true,
    isDeceptive: true,  // highly deceptive!
    isLink: false,
    numComponents: 1,
    aliases: ['messy_unknot', 'fake_knot'],
    description: 'Kinky unknot (visually complex but topologically trivial, for testing models)',
    visualComplexity: 'high',
    isHardNegative: true,  // marked as "hard negative"
  },

  ring_like_open_rope: {
    topologicalId: TOPOLOGICAL_CLASSES.UNKNOT,
    crossingNumber: 0,
    family: 'unknot_variant',
    generator: 'ring_like_open_rope',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: true,
    isDeceptive: true,
    isLink: false,
    numComponents: 1,
    isOpenRope: true,
    aliases: ['ring_like_open_ended', 'open_ring_like_rope', 'unknotted_open_loop'],
    description: 'Open-ended unknotted rope arranged like a loose ring with overlapping sides',
    visualComplexity: 'medium',
    trapType: 'ring_like_open_rope',
    isHardNegative: true,
  },
  
  loose_open_knot: {
    topologicalId: TOPOLOGICAL_CLASSES.TREFOIL,  // topologically trefoil (just loosely presented)
    crossingNumber: 3,
    family: 'torus_knot',
    generator: 'loose_open_knot',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,  // not deceptive - it is a real knot, just loose enough that VLMs misjudge
    isLink: false,
    numComponents: 1,
    isOpenRope: true,
    aliases: ['loose_trefoil', 'slack_knot'],
    description: 'Loose knot (real trefoil but very loose, VLMs easily misjudge as unknotted)',
    visualComplexity: 'low',  // visually appears simple
    isLooseVariant: true,     // marked as loose variant
  },

  ring_like_open_knot: {
    topologicalId: TOPOLOGICAL_CLASSES.TREFOIL,
    crossingNumber: 3,
    family: 'torus_knot',
    generator: 'ring_like_open_knot',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: true,
    isLink: false,
    numComponents: 1,
    isOpenRope: true,
    aliases: ['loose_ring_knot', 'ring_like_trefoil', 'open_knot_like_loop'],
    description: 'Open-ended real trefoil presented as a very loose ring-like loop',
    visualComplexity: 'low',
    trapType: 'ring_like_knot',
    isLooseVariant: true,
  },

  // ========================================
  // === Loose Knots (loose variants, high occlusion) ===
  // ========================================


  loose_cinquefoil: {
    topologicalId: TOPOLOGICAL_CLASSES.CINQUEFOIL,
    crossingNumber: 5,
    family: 'torus_knot',
    generator: 'loose_cinquefoil',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    isOpenRope: true,
    aliases: ['loose_5_1', 'slack_cinquefoil'],
    description: 'Loose cinquefoil (real T(2,5) but very loose)',
    visualComplexity: 'low',
    isLooseVariant: true,
  },


  // ========================================
  // === Occluded Knot (tests occlusion scenarios) ===
  // ========================================

  occluded_knot: {
    topologicalId: TOPOLOGICAL_CLASSES.TREFOIL,  // topologically trefoil
    crossingNumber: 3,
    family: 'torus_knot',
    generator: 'occluded_knot',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    aliases: ['hidden_crossing_knot'],
    description: 'Occluded knot (trefoil with high deformation+slack causing crossing occlusion, tests VLM recognition under occlusion)',
    visualComplexity: 'high',
    trapType: 'occluded',
    hasOcclusion: true,
  },

  // ========================================
  // === True Knots (non-trivial) ===
  // ========================================

  trefoil: {
    topologicalId: TOPOLOGICAL_CLASSES.TREFOIL,
    crossingNumber: 3,
    family: 'torus_knot',
    generator: 'trefoil',
    difficulty: 'easy',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 2, q: 3 },
    aliases: ['3_1', 'T(2,3)', 'overhand'],
    description: 'Trefoil (simplest non-trivial knot)',
  },

  figure8: {
    topologicalId: TOPOLOGICAL_CLASSES.FIGURE_EIGHT,
    crossingNumber: 4,
    family: 'twist_knot',
    generator: 'figure8',
    difficulty: 'easy',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    aliases: ['4_1', 'figure_eight', 'flemish'],
    description: 'Figure-eight (simplest twist knot)',
  },

  torus_2_5: {
    topologicalId: TOPOLOGICAL_CLASSES.CINQUEFOIL,
    crossingNumber: 5,
    family: 'torus_knot',
    generator: 'torus_2_5',
    difficulty: 'medium',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 2, q: 5 },
    aliases: ['5_1', 'T(2,5)', 'cinquefoil', 'solomon_seal'],
    description: 'T(2,5) cinquefoil',
  },

  torus_2_7: {
    topologicalId: TOPOLOGICAL_CLASSES.TORUS_2_7,
    crossingNumber: 7,
    family: 'torus_knot',
    generator: 'torus_2_7',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 2, q: 7 },
    aliases: ['7_1', 'T(2,7)', 'septafoil'],
    description: 'T(2,7) septafoil',
  },

  torus_2_9: {
    topologicalId: TOPOLOGICAL_CLASSES.TORUS_2_9,
    crossingNumber: 9,
    family: 'torus_knot',
    generator: 'torus_2_9',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 2, q: 9 },
    aliases: ['9_1', 'T(2,9)'],
    description: 'T(2,9) nonafoil',
  },

  torus_3_4: {
    topologicalId: TOPOLOGICAL_CLASSES.TORUS_3_4,
    crossingNumber: 8,
    family: 'torus_knot',
    generator: 'torus_3_4',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 3, q: 4 },
    aliases: ['T(3,4)', '8_19'],
    description: 'T(3,4) torus knot',
  },

  torus_3_5: {
    topologicalId: TOPOLOGICAL_CLASSES.TORUS_3_5,
    crossingNumber: 10,
    family: 'torus_knot',
    generator: 'torus_3_5',
    difficulty: 'hard',
    isKnot: true,
    isUnknot: false,
    isDeceptive: false,
    isLink: false,
    numComponents: 1,
    torusParams: { p: 3, q: 5 },
    aliases: ['T(3,5)', '10_124'],
    description: 'T(3,5) torus knot',
  },
  
  // ========================================
  // === Links (multi-component) ===
  // ========================================
  
  hopf_link: {
    topologicalId: TOPOLOGICAL_CLASSES.HOPF_LINK,
    crossingNumber: 2,
    family: 'link',
    generator: 'hopf_link',
    difficulty: 'easy',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: 2,
    aliases: ['2^2_1', 'hopf'],
    description: 'Hopf link (two interlinked rings)',
  },

  unlinked_rings: {
    topologicalId: TOPOLOGICAL_CLASSES.UNLINKED_2,
    crossingNumber: 0,
    family: 'link',
    generator: 'unlinked_rings',
    difficulty: 'easy',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: 2,
    aliases: ['unlink_2', 'separate_rings'],
    description: 'Unlinked rings (two separate rings)',
  },

  chain: {
    topologicalId: TOPOLOGICAL_CLASSES.CHAIN_N,
    crossingNumber: null,  // depends on ring count
    family: 'link',
    generator: 'chain',
    difficulty: 'medium',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: null,  // variable, default 3
    aliases: ['chain_link'],
    description: 'Chain (multiple interlinked rings)',
  },

  borromean: {
    topologicalId: TOPOLOGICAL_CLASSES.BORROMEAN,
    crossingNumber: 6,
    family: 'link',
    generator: 'borromean',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: 3,
    aliases: ['borromean_rings', '6^3_2'],
    description: 'Borromean rings (three mutually linked, any two are unlinked)',
  },

  // === Link Groups (multiple links together) ===
  // ========================================

  double_hopf: {
    topologicalId: TOPOLOGICAL_CLASSES.DOUBLE_HOPF,
    crossingNumber: 4,
    family: 'link_group',
    generator: 'double_hopf',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: null,  // generated as two linked groups, usually 5-6 rings
    aliases: ['two_hopf_links'],
    description: 'Double Hopf-style link group (two separate linked chains, tests grouped-link recognition)',
  },

  link_cluster: {
    topologicalId: TOPOLOGICAL_CLASSES.LINK_CLUSTER,
    crossingNumber: null,
    family: 'link_group',
    generator: 'link_cluster',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: null,  // 4-6 rings
    aliases: ['multi_link_group'],
    description: 'Link cluster (mixed link arrangement, tests overall recognition)',
  },

  // ========================================
  // === Mixed (some linked, some free) ===
  // ========================================

  hopf_plus_free: {
    topologicalId: TOPOLOGICAL_CLASSES.HOPF_PLUS_FREE,
    crossingNumber: 2,
    family: 'mixed',
    generator: 'hopf_plus_free',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: 3,  // 2 linked + 1 free
    numLinkedComponents: 2,
    numFreeComponents: 1,
    aliases: ['hopf_and_free_ring'],
    description: 'Hopf link + free ring (two linked rings + one independent ring, tests linking recognition)',
  },

  chain_plus_free: {
    topologicalId: TOPOLOGICAL_CLASSES.CHAIN_PLUS_FREE,
    crossingNumber: null,
    family: 'mixed',
    generator: 'chain_plus_free',
    difficulty: 'hard',
    isKnot: false,
    isUnknot: false,
    isDeceptive: false,
    isLink: true,
    numComponents: null,  // chain count + 1-2 free rings
    aliases: ['chain_and_free_rings'],
    description: 'Chain + free rings (chain with independent rings nearby, tests connectivity recognition)',
  },

};

// ============= Utility Functions =============

/**
 * Get the topological ID for a knot type.
 * @param {string} knotType - knot type key
 * @returns {string|null} topological ID
 */
export function getTopologicalId(knotType) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  return entry ? entry.topologicalId : null;
}

/**
 * Check if two knot types are topologically equivalent.
 * @param {string} typeA - first knot type key
 * @param {string} typeB - second knot type key
 * @returns {boolean} whether equivalent
 */
export function areTopologicallyEquivalent(typeA, typeB) {
  const idA = getTopologicalId(typeA);
  const idB = getTopologicalId(typeB);
  if (!idA || !idB) {
    console.warn(`Unknown knot type: ${!idA ? typeA : typeB}`);
    return false;
  }
  return idA === idB;
}

/**
 * Get all generators belonging to a topological class.
 * @param {string} topologicalId - topological ID
 * @returns {string[]} generator key list
 */
export function getGeneratorsForTopologicalClass(topologicalId) {
  const generators = [];
  for (const [key, entry] of Object.entries(KNOT_TYPE_REGISTRY)) {
    if (entry.topologicalId === topologicalId) {
      generators.push(key);
    }
  }
  return generators;
}

/**
 * Get all available topological class IDs.
 * @param {Object} options - filter options
 * @param {boolean} options.includeLinks - whether to include links
 * @param {boolean} options.includeDeceptive - whether to include deceptive types
 * @returns {string[]} topological ID list
 */
export function getAllTopologicalIds(options = {}) {
  const { includeLinks = false, includeDeceptive = true } = options;
  const ids = new Set();
  
  for (const entry of Object.values(KNOT_TYPE_REGISTRY)) {
    if (!includeLinks && entry.isLink) continue;
    if (!includeDeceptive && entry.isDeceptive) continue;
    ids.add(entry.topologicalId);
  }
  
  return Array.from(ids);
}

/**
 * Get all generators for a topological class (for random selection).
 * @param {string} topologicalId
 * @param {Object} options
 * @returns {Object[]} { key, entry }
 */
export function getGeneratorsByTopologicalId(topologicalId, options = {}) {
  const { excludeDeceptive = false } = options;
  const result = [];
  
  for (const [key, entry] of Object.entries(KNOT_TYPE_REGISTRY)) {
    if (entry.topologicalId !== topologicalId) continue;
    if (excludeDeceptive && entry.isDeceptive) continue;
    result.push({ key, entry });
  }
  
  return result;
}

/**
 * Check if a type is a deceptive unknot.
 * @param {string} knotType
 * @returns {boolean}
 */
export function isDeceptiveKnot(knotType) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  return entry ? (entry.isDeceptive === true) : false;
}

/**
 * Check if a type is a link.
 * @param {string} knotType
 * @returns {boolean}
 */
export function isLink(knotType) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  return entry ? (entry.isLink === true) : false;
}

/**
 * Get the minimal crossing number for a knot type.
 * @param {string} knotType
 * @returns {number|null}
 */
export function getCrossingNumber(knotType) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  return entry ? entry.crossingNumber : null;
}

/**
 * Get the difficulty level for a knot type.
 * @param {string} knotType
 * @returns {'easy'|'medium'|'hard'|null}
 */
export function getDifficulty(knotType) {
  const entry = KNOT_TYPE_REGISTRY[knotType];
  return entry ? entry.difficulty : null;
}

// ============= Predefined Confusing Pairs (Hard Negatives) =============

/**
 * Predefined "confusing pairs".
 * Visually similar but topologically non-equivalent, used for Hard Negative Pairs.
 */
export const CONFUSING_PAIRS = [
  // 1. Kinky Unknot vs real Knot (visually similar but topologically different)
  { a: 'kinky_unknot', b: 'trefoil', reason: 'kinky unknot looks complex but is trivial' },
  { a: 'kinky_unknot', b: 'figure8', reason: 'kinky unknot vs figure-8' },
  { a: 'kinky_unknot', b: 'torus_2_5', reason: 'kinky unknot vs cinquefoil' },
  
  // 1b. Occluded knot confusion
  { a: 'occluded_knot', b: 'unknot', reason: 'occluded trefoil may look untangled' },
  { a: 'occluded_knot', b: 'twisted_ring', reason: 'occluded real knot vs deceptive unknot' },

  // 2. Twisted Ring vs Trefoil (harder to distinguish after deformation)
  { a: 'twisted_ring', b: 'trefoil', reason: 'deformed unknot variant vs trefoil' },
  { a: 'spiral_disk', b: 'trefoil', reason: 'spiral unknot vs trefoil' },

  // 2b. Loose knot vs Unknot (loose real knot vs loose fake knot)
  { a: 'loose_open_knot', b: 'unknot', reason: 'loose real knot looks like simple loop' },
  { a: 'loose_open_knot', b: 'twisted_ring', reason: 'loose real knot vs twisted unknot' },
  { a: 'ring_like_open_knot', b: 'ring_like_open_rope', reason: 'ring-like open knot vs ring-like open rope with no knot' },
  { a: 'ring_like_open_knot', b: 'unknot', reason: 'ring-like real open knot vs closed trivial loop' },
  { a: 'ring_like_open_rope', b: 'trefoil', reason: 'open unknotted rope vs true knot' },
  
  // 3. Adjacent crossing number Torus Knots
  { a: 'trefoil', b: 'torus_2_5', reason: '3 vs 5 crossings, same family' },
  { a: 'torus_2_5', b: 'torus_2_7', reason: '5 vs 7 crossings, same family' },
  { a: 'torus_2_7', b: 'torus_2_9', reason: '7 vs 9 crossings, same family' },
  
  // 4. Different families but close crossing numbers
  { a: 'trefoil', b: 'figure8', reason: '3 vs 4 crossings, different families' },
  
  // 5. Links confusion
  { a: 'hopf_link', b: 'unlinked_rings', reason: 'linked vs unlinked (subtle)' },

  // 6. Loose knots confusion
  { a: 'loose_cinquefoil', b: 'unknot', reason: 'loose cinquefoil looks untangled' },

  // 7. Mixed confusion (linked vs unlinked)
  { a: 'hopf_plus_free', b: 'unlinked_rings', reason: 'some linked some free vs all unlinked' },
  { a: 'hopf_plus_free', b: 'chain', reason: 'partial linking vs full chain' },
  { a: 'double_hopf', b: 'chain', reason: 'two separate hopf links vs connected chain' },

];

/**
 * Check if two types form a predefined confusing pair.
 * @param {string} typeA
 * @param {string} typeB
 * @returns {{ isConfusing: boolean, reason: string|null }}
 */
export function isConfusingPair(typeA, typeB) {
  for (const pair of CONFUSING_PAIRS) {
    if ((pair.a === typeA && pair.b === typeB) ||
        (pair.a === typeB && pair.b === typeA)) {
      return { isConfusing: true, reason: pair.reason };
    }
  }
  return { isConfusing: false, reason: null };
}

// ============= Grouped by Difficulty =============

/**
 * Knot types grouped by difficulty.
 * Used for stratified sampling by difficulty.
 */
export const KNOT_TYPES_BY_DIFFICULTY = {
  easy: [
    'unknot', 'trefoil', 'figure8', 'hopf_link', 'unlinked_rings',
  ],
  medium: [
    'twisted_ring', 'spiral_disk', 'torus_2_5', 'chain',
  ],
  hard: [
    'kinky_unknot', 'ring_like_open_rope', 'loose_open_knot', 'ring_like_open_knot', 'occluded_knot', 'torus_2_7', 'torus_2_9', 'torus_3_4', 'torus_3_5', 'borromean',
    'loose_cinquefoil',
    'double_hopf', 'link_cluster',
    'hopf_plus_free', 'chain_plus_free',
  ],
};

/**
 * Grouped by topological class (excluding Links).
 */
export const KNOT_ONLY_TYPES = Object.entries(KNOT_TYPE_REGISTRY)
  .filter(([_, entry]) => !entry.isLink)
  .map(([key]) => key);

/**
 * Topological class IDs for knots only (excluding Links).
 */
export const KNOT_ONLY_TOPOLOGICAL_IDS = getAllTopologicalIds({ includeLinks: false });

// ============= Exports =============

export default {
  TOPOLOGICAL_CLASSES,
  KNOT_TYPE_REGISTRY,
  CONFUSING_PAIRS,
  KNOT_TYPES_BY_DIFFICULTY,
  KNOT_ONLY_TYPES,
  KNOT_ONLY_TOPOLOGICAL_IDS,
  
  // Functions
  getTopologicalId,
  areTopologicallyEquivalent,
  getGeneratorsForTopologicalClass,
  getAllTopologicalIds,
  getGeneratorsByTopologicalId,
  isDeceptiveKnot,
  isLink,
  getCrossingNumber,
  getDifficulty,
  isConfusingPair,
};
