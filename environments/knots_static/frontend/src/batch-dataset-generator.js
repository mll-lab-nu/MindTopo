/**
 * batch-dataset-generator.js
 *
 * One-click batch generation of the full benchmark dataset.
 * Runs in-browser, calls invariance-renderer.js for rendering.
 *
 * Generation spec:
 * - all active single-rope entries from KNOT_TYPE_REGISTRY
 * - all active multi-component link, link-group, and mixed entries from KNOT_TYPE_REGISTRY
 * - optional pair samples for legacy equivalence tasks
 */

import { KNOT_TYPE_REGISTRY } from './knot-type-registry.js?v=6';
import { renderSingleImage, dataUrlToBlob, downloadDataUrl } from './invariance-renderer.js?v=8';
import { generatePositivePair, generateNegativePair, makeRng } from './invariance-generator.js?v=7';
import { getBucketTopology, getBucketSaliency, getTrapType } from './difficulty-controller.js?v=6';

// ============= Generation Spec =============

const CAMERA_ANGLES = [
  { name: 'iso_fr',  pos: [3.5, 3.5, 5] },   // ~30% closer, crossings clearer
  { name: 'front',   pos: [0, 1.5, 6] },
  { name: 'oblique', pos: [2, 6, 3] },
  { name: 'top_tilt', pos: [1, 7, 1.5] },     // near-top view, hard but still traceable
  { name: 'low_side', pos: [4.5, 1.8, 2.8] }, // side-oblique view; avoids fully edge-on ring collapse
];

// Slackness sample points per type (biased toward medium/hard, fewer easy)
// Old: uniform 0.05-0.78; New: 2 tight + 4 medium + 4 loose, higher difficulty
const SLACKNESS_LEVELS = [
  { label: 'tight_1',     value: 0.08 },
  { label: 'tight_2',     value: 0.18 },
  { label: 'medium_1',    value: 0.28 },
  { label: 'medium_2',    value: 0.35 },
  { label: 'medium_3',    value: 0.42 },
  { label: 'medium_4',    value: 0.50 },
  { label: 'loose_1',     value: 0.58 },
  { label: 'loose_2',     value: 0.65 },
  { label: 'very_loose_1', value: 0.72 },
  { label: 'very_loose_2', value: 0.80 },
];

const SLACKNESS_CAPS = {
  unknot: 0.40, twisted_ring: 0.40, spiral_disk: 0.30, occluded_knot: 0.55,
  figure8: 0.35, trefoil: 0.75, torus_2_5: 0.80,
  torus_2_7: 0.60, torus_2_9: 0.50,  // relaxed to allow more hard samples
  torus_3_4: 0.60, torus_3_5: 0.50,
  kinky_unknot: 0.35,
  ring_like_open_rope: 0.95,
  loose_open_knot: 0.60,
  ring_like_open_knot: 1.00,
  // New loose variants
  loose_cinquefoil: 0.95,
};

// Slackness floor for certain types
const SLACKNESS_FLOORS = {
  loose_open_knot: 0.50,  // min 0.5 to appear "loose"
  ring_like_open_knot: 0.88,
  kinky_unknot: 0.25,     // too low and kinky features vanish, degrades to plain ring
  ring_like_open_rope: 0.68,
  occluded_knot: 0.25,    // needs slack to produce occlusion effects
  // New loose variants - must be loose
  loose_cinquefoil: 0.50,
};

const DEFORM_RANGE = { min: 0.10, max: 0.45 };
const FIXED_LINK_TUBE_RADIUS = 0.07;

// Some types need higher deform strength for interesting visual effects
const DEFORM_FLOORS = {
  unknot: 0.25,         // unknot needs enough deformation to not look like a perfect circle
  twisted_ring: 0.20,   // twisted_ring needs visible twisting
  kinky_unknot: 0.30,   // kinky_unknot needs high deform to produce complex kinks
  ring_like_open_rope: 0.12,
  occluded_knot: 0.20,  // needs deformation so crossings are occluded
};

const RENDER_OPTIONS = {
  width: 1024,
  height: 1024,
};

// ============= T04 (link_property) per-ring solid-color setup =============
// Deterministic ring-index → color: ring 0 = red, ring 1 = blue, etc.
// Keep T04 to the original six high-contrast colors. Dense link scenes may
// still contain more components, but they do not get T04 colored variants.
const T04_PALETTE_NAMES = ['red', 'blue', 'green', 'brown', 'white', 'purple'];
const T04_PALETTE_HEX = {
  red:    '#E60026',
  blue:   '#3498DB',
  green:  '#2ECC71',
  brown:  '#8D6E63',
  white:  '#ECF0F1',
  purple: '#5B21B6',
};
const MAX_LINK_COMPONENTS = 10;

// Knot types where the "remove a ring, what becomes free" question is
// non-trivial AND the linkage graph is deterministic from the structural
// params. Excludes hopf_link/unlinked_rings (only 2 rings — trivial after
// removal) and multi-cluster scenes whose per-ring relation is too ambiguous
// from a single static view.
const T04_ELIGIBLE_KNOT_TYPES = new Set([
  'chain',
  'borromean',
  'hopf_plus_free',
  'chain_plus_free',
]);

const LINK_VARIANT_COUNTS = {
  hopf_link: 4,
  unlinked_rings: 12,
  chain: 10,
  borromean: 18,
  double_hopf: 36,
  link_cluster: 42,
  hopf_plus_free: 28,
  chain_plus_free: 48,
};

const HARD_LINK_TYPES = new Set([
  'chain',
  'borromean',
  'double_hopf',
  'link_cluster',
  'hopf_plus_free',
  'chain_plus_free',
]);

function linkSlacknessForVariant(knotType, variantIndex, variantCount) {
  const t = variantCount <= 1 ? 0 : variantIndex / (variantCount - 1);
  if (knotType === 'unlinked_rings') {
    return variantIndex < 8
      ? clamp(0.10 + variantIndex * 0.045, 0.10, 0.45)
      : clamp(0.48 + ((variantIndex - 8) % 8) * 0.055, 0.48, 0.86);
  }
  if (HARD_LINK_TYPES.has(knotType)) {
    return clamp(0.24 + t * 0.64, 0.24, 0.88);
  }
  return clamp(0.12 + t * 0.56, 0.10, 0.80);
}

function linkDeformStrengthForVariant(knotType, rng, variantIndex) {
  const r = rng || Math.random;
  if (HARD_LINK_TYPES.has(knotType) || (knotType === 'unlinked_rings' && variantIndex >= 8)) {
    return clamp(0.26 + r() * 0.22, 0.20, 0.55);
  }
  return clamp(0.15 + r() * 0.2, 0.10, 0.40);
}

function linkVisualTrap(knotType, variantIndex, slackness) {
  if (knotType === 'unlinked_rings' && variantIndex >= 8) return 'near_miss_unlink';
  if (knotType === 'hopf_plus_free' || knotType === 'chain_plus_free') return 'linked_plus_free';
  if (knotType === 'double_hopf' || knotType === 'link_cluster') return 'multiple_link_groups';
  if (knotType === 'chain' && slackness >= 0.55) return 'long_slack_chain';
  return null;
}

/**
 * Compute structural params + ring linkage graph for a link scene.
 * Uses a dedicated rng so that batch generator and renderer agree on
 * chainLinks / numFree / clusterSegments (renderer respects these as overrides).
 *
 * Returns:
 *   {
 *     numComponents,
 *     numLinks?, chainLinks?, numFree?, clusterSegments?,
 *     linkageEdges:   [[i,j], ...]   // pairwise Hopf-style links
 *     brunnianGroups: [[i,j,k], ...] // group is locked iff every ring in it remains (Borromean)
 *   }
 */
function computeLinkStructure(knotType, paramRng, variantIndex) {
  const r = () => paramRng();
  switch (knotType) {
    case 'chain': {
      const numLinks = 3 + (variantIndex % 4); // 3-6, matches existing chainNumLinks formula
      const edges = [];
      for (let i = 0; i < numLinks - 1; i++) edges.push([i, i + 1]);
      return { numComponents: numLinks, numLinks, linkageEdges: edges, brunnianGroups: [] };
    }
    case 'borromean': {
      return { numComponents: 3, linkageEdges: [], brunnianGroups: [[0, 1, 2]] };
    }
    case 'hopf_link': {
      return { numComponents: 2, linkageEdges: [[0, 1]], brunnianGroups: [] };
    }
    case 'unlinked_rings': {
      return { numComponents: 2, linkageEdges: [], brunnianGroups: [] };
    }
    case 'double_hopf': {
      const patterns = [
        [4, 4], [5, 4], [4, 5], [5, 5],
        [3, 5], [5, 3], [4, 4], [3, 6],
      ];
      const segs = patterns[variantIndex % patterns.length];
      const edges = [];
      let off = 0;
      for (const sz of segs) {
        for (let i = 0; i < sz - 1; i++) edges.push([off + i, off + i + 1]);
        off += sz;
      }
      return { numComponents: off, clusterSegments: segs, linkageEdges: edges, brunnianGroups: [] };
    }
    case 'link_cluster': {
      const patterns = [
        [3, 3, 2], [4, 3, 2], [3, 3, 3], [4, 4, 2],
        [2, 3, 3, 2], [5, 3], [3, 4, 3], [2, 2, 3, 3],
      ];
      const segs = patterns[variantIndex % patterns.length];
      const edges = [];
      let off = 0;
      for (const sz of segs) {
        for (let i = 0; i < sz - 1; i++) edges.push([off + i, off + i + 1]);
        off += sz;
      }
      return { numComponents: off, clusterSegments: segs, linkageEdges: edges, brunnianGroups: [] };
    }
    case 'hopf_plus_free': {
      const chainLinks = 2 + Math.floor(r() * 2); // 2-3
      const numFree = 1 + Math.floor(r() * 2);    // 1-2
      const edges = [];
      for (let i = 0; i < chainLinks - 1; i++) edges.push([i, i + 1]);
      return { numComponents: chainLinks + numFree, chainLinks, numFree, linkageEdges: edges, brunnianGroups: [] };
    }
    case 'chain_plus_free': {
      const t04ReadablePatterns = [
        [4, 1], [4, 2], [5, 1], [3, 2],
        [4, 2], [5, 1], [3, 3], [4, 1],
      ];
      const densePatterns = [
        [6, 2], [7, 1], [7, 2], [8, 1],
        [6, 3], [8, 2], [7, 3], [6, 4],
      ];
      const patterns = variantIndex < 8 ? t04ReadablePatterns : densePatterns;
      const [chainLinks, numFree] = patterns[variantIndex % patterns.length];
      const edges = [];
      for (let i = 0; i < chainLinks - 1; i++) edges.push([i, i + 1]);
      return { numComponents: chainLinks + numFree, chainLinks, numFree, linkageEdges: edges, brunnianGroups: [] };
    }
    default:
      return { numComponents: 2, linkageEdges: [], brunnianGroups: [] };
  }
}

function countLinkedComponents(struct) {
  const linked = new Set();
  for (const pair of struct.linkageEdges || []) {
    if (!Array.isArray(pair) || pair.length !== 2) continue;
    linked.add(Number(pair[0]));
    linked.add(Number(pair[1]));
  }
  for (const group of struct.brunnianGroups || []) {
    if (!Array.isArray(group)) continue;
    for (const idx of group) linked.add(Number(idx));
  }
  linked.delete(NaN);
  return linked.size;
}

// Family map (same as vlm_benchmark.py)
const FAMILY_MAP = {
  unknot: 'UNKNOT', twisted_ring: 'UNKNOT', kinky_unknot: 'UNKNOT',
  ring_like_open_rope: 'UNKNOT',
  spiral_disk: 'UNKNOT',
  trefoil: 'TORUS', occluded_knot: 'TORUS', loose_open_knot: 'TORUS',
  ring_like_open_knot: 'TORUS',
  torus_2_5: 'TORUS', torus_2_7: 'TORUS',
  torus_2_9: 'TORUS', torus_3_4: 'TORUS', torus_3_5: 'TORUS',
  figure8: 'TWIST',
  // New loose variants
  loose_cinquefoil: 'TORUS',
  // Links
  hopf_link: 'LINK', unlinked_rings: 'LINK', chain: 'LINK', borromean: 'LINK',
  // Link groups
  double_hopf: 'LINK_GROUP', link_cluster: 'LINK_GROUP',
  // Mixed
  hopf_plus_free: 'MIXED', chain_plus_free: 'MIXED',
};

// ============= Utilities =============

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Seeded RNG
function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); h ^= h >>> 16; return h >>> 0; };
}
function mulberry32(seed) {
  return () => { let t = (seed += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function localRng(seedStr) { return mulberry32(xmur3(seedStr)()); }

function sectionForSingleMetadata(meta) {
  const family = meta?.family || '';
  if (family === 'LINK_GROUP') return 'link_groups';
  if (family === 'MIXED') return 'mixed';
  if (meta?.isLink) return 'links';
  return 'singles';
}

// ============= Main Generation Function =============

/**
 * Generate the full dataset and package for download (ZIP or individual).
 * @param {Object} config
 * @param {Function} onProgress - (current, total, message) => void
 * @param {Function|null} onSample - async callback for streamed sample payloads
 * @returns {Promise<Object>} { samples: [], metadata: [] }
 */
export async function generateFullDataset(config = {}, onProgress = () => {}, onSample = null) {
  const {
    seed = 'benchmark-v3',
    variantsPerType = 10,          // slackness variants per type
    anglesPerVariant = 5,          // camera angles per variant
    numPairs = 100,                // total pairs
    renderWidth = 1024,
    renderHeight = 1024,
    retainSamples = true,
    skipSampleIds = [],            // resume: sample ids already on disk; iterate but don't render or emit
    shardIndex = 0,                // parallel generation: this worker's zero-based shard
    shardCount = 1,                // parallel generation: total shards
  } = config;
  const skipSet = new Set(skipSampleIds);
  const normalizedShardCount = Math.max(1, Math.floor(Number(shardCount) || 1));
  const normalizedShardIndex = clamp(Math.floor(Number(shardIndex) || 0), 0, normalizedShardCount - 1);
  const ownsSample = (generationIndex) =>
    normalizedShardCount === 1 || generationIndex % normalizedShardCount === normalizedShardIndex;

  const rng = localRng(seed);
  const allFiles = [];       // { filename, blob, metadata }
  const allMetadata = [];
  let renderedFileCount = 0;
  let sampleOrdinal = 0;

  const emitSample = async (payload) => {
    if (typeof onSample === 'function') {
      await onSample(payload);
    }
  };

  // -- Step 1: Single-rope knots/unknots --
  const knotTypes = Object.entries(KNOT_TYPE_REGISTRY)
    .filter(([_, e]) => !e.isLink)
    .map(([key]) => key);

  const linkTypes = Object.entries(KNOT_TYPE_REGISTRY)
    .filter(([_, e]) => e.isLink)
    .map(([key]) => key);

  const totalSingle = knotTypes.length * variantsPerType * anglesPerVariant;
  const LINK_VARIANTS = LINK_VARIANT_COUNTS;
  const linkVariantsDefault = 12;
  const totalLink = linkTypes.reduce((sum, lt) =>
    sum + (LINK_VARIANTS[lt] ?? linkVariantsDefault), 0) * anglesPerVariant;
  const totalPair = numPairs * 2;
  const grandTotal = totalSingle + totalLink + totalPair;
  let current = 0;

  onProgress(0, grandTotal, 'Generating single-loop knot data...');

  for (const knotType of knotTypes) {
    const entry = KNOT_TYPE_REGISTRY[knotType];
    const maxSlack = SLACKNESS_CAPS[knotType] ?? 0.80;
    const levels = SLACKNESS_LEVELS.filter(l => l.value <= maxSlack + 0.05);
    // Ensure enough variants; fill with intermediate values if short
    while (levels.length < variantsPerType) {
      const mid = (levels[levels.length - 1].value + levels[0].value) / 2;
      levels.push({ label: `custom_${levels.length}`, value: clamp(mid + rng() * 0.1, 0, maxSlack) });
    }

    for (let v = 0; v < variantsPerType; v++) {
      const minSlack = SLACKNESS_FLOORS[knotType] ?? 0;
      const slackness = clamp(levels[v % levels.length].value, minSlack, maxSlack);
      const deformMin = Math.max(DEFORM_RANGE.min, DEFORM_FLOORS[knotType] ?? 0);
      const deformStrength = clamp(deformMin + rng() * (DEFORM_RANGE.max - deformMin), 0, 1);
      const variantSeed = Math.floor(rng() * 999999);
      const sampleId = `${knotType}_s${slackness.toFixed(2)}_d${deformStrength.toFixed(2)}_v${v}`;
      const generationIndex = sampleOrdinal++;

      if (!ownsSample(generationIndex) || skipSet.has(sampleId)) {
        current += Math.min(anglesPerVariant, CAMERA_ANGLES.length);
        continue;
      }

      const images = [];
      const imagePayloads = [];

      for (let a = 0; a < Math.min(anglesPerVariant, CAMERA_ANGLES.length); a++) {
        const angle = CAMERA_ANGLES[a];
        const filename = `${sampleId}_${angle.name}.png`;

        const imageParams = {
          knotType,
          seed: variantSeed,
          deformStrength,
          slackness,
          cameraPosition: angle.pos,
          cameraTarget: [0, 0, 0],
          cameraFov: 45,
          useRainbow: true,    // rainbow gradient like gallery
          metalness: 0.04,     // matched to gallery
          roughness: 0.5,      // matched to gallery
          backgroundColor: '#1a2236',
        };

        try {
          const dataUrl = await renderSingleImage(imageParams, {
            width: renderWidth,
            height: renderHeight,
          });
          const blob = dataUrlToBlob(dataUrl);
          renderedFileCount++;
          imagePayloads.push({ filename, dataUrl });
          if (retainSamples) {
            allFiles.push({ filename, blob });
          }
          images.push({ filename, cameraAngle: angle.name, cameraPos: angle.pos });
        } catch (err) {
          console.warn(`Render failed: ${filename}`, err);
          images.push({ filename, cameraAngle: angle.name, cameraPos: angle.pos, error: err.message });
        }

        current++;
        onProgress(current, grandTotal, `[${current}/${grandTotal}] ${filename}`);

        // Yield to browser to avoid freezing UI
        if (current % 3 === 0) await sleep(10);
      }

      // Build metadata
      const crossing = entry.crossingNumber ?? 0;
      const trap = getTrapType(knotType, slackness);
      const score = 0.35 * Math.min(crossing / 10, 1) + 0.45 * slackness + 0.20 * (trap ? 0.3 : 0);

      const meta = {
        version: '3.0',
        id: sampleId,
        generationIndex,
        knotType,
        topologicalId: entry.topologicalId,
        isKnot: entry.isKnot ?? false,
        isUnknot: entry.isUnknot ?? false,
        isDeceptive: entry.isDeceptive ?? false,
        isLink: false,
        numComponents: 1,
        crossingNumber: crossing,
        family: FAMILY_MAP[knotType] || 'OTHER',
        ...(entry.isOpenRope ? { isOpenRope: true } : {}),
        seed: variantSeed,
        slackness,
        deformStrength,
        bucket_topology: getBucketTopology(knotType),
        bucket_saliency: getBucketSaliency(slackness),
        trap_type: trap,
        difficulty_score: Math.round(score * 1000) / 1000,
        difficulty: score < 0.25 ? 'easy' : score < 0.45 ? 'medium' : 'hard',
        images,
      };
      allMetadata.push(meta);
      await emitSample({
        kind: 'single',
        section: sectionForSingleMetadata(meta),
        metadata: meta,
        files: imagePayloads,
      });
    }
  }

  // -- Step 2: Multi-component link scenes --
  onProgress(current, grandTotal, 'Generating multi-component link scene data...');

  for (const knotType of linkTypes) {
    const entry = KNOT_TYPE_REGISTRY[knotType];
    const thisLinkVariants = LINK_VARIANTS[knotType] ?? linkVariantsDefault;

    for (let v = 0; v < thisLinkVariants; v++) {
      const slackness = linkSlacknessForVariant(knotType, v, thisLinkVariants);
      const deformStrength = linkDeformStrengthForVariant(knotType, rng, v);
      const visualTrap = linkVisualTrap(knotType, v, slackness);
      const variantSeed = Math.floor(rng() * 999999);
      const sampleId = `${knotType}_v${v}`;
      const generationIndex = sampleOrdinal++;
      if (!ownsSample(generationIndex) || skipSet.has(sampleId)) {
        current += Math.min(anglesPerVariant, CAMERA_ANGLES.length);
        continue;
      }

      // Pre-compute structural params + linkage graph deterministically.
      // Uses a dedicated rng (separate from the renderer's internal rng) so
      // batch generator and renderer agree on chainLinks/numFree/clusterSegments
      // regardless of any other rng calls inside the renderer.
      const structRng = localRng(`${variantSeed}-struct`);
      const struct = computeLinkStructure(knotType, structRng, v);
      const numComp = struct.numComponents;
      const linkedComponentCount = countLinkedComponents(struct);

      // T04 colored variant: only for ≥3-ring scenes in T04_ELIGIBLE set
      // and only when palette is large enough.
      const t04Colored = T04_ELIGIBLE_KNOT_TYPES.has(knotType)
        && numComp >= 3
        && numComp <= T04_PALETTE_NAMES.length;
      const ringColors = t04Colored
        ? T04_PALETTE_NAMES.slice(0, numComp)
        : null;
      const ringColorsHex = t04Colored
        ? ringColors.map(name => T04_PALETTE_HEX[name])
        : null;

      const images = [];
      const imagePayloads = [];

      // Structural params passed to the renderer for both gradient and colored variants
      const structuralImageParams = {
        ...(struct.numLinks != null ? { numLinks: struct.numLinks } : {}),
        ...(struct.chainLinks != null ? { chainLinks: struct.chainLinks } : {}),
        ...(struct.numFree != null ? { numFree: struct.numFree } : {}),
        ...(struct.clusterSegments ? { clusterSegments: struct.clusterSegments } : {}),
        ...(visualTrap === 'near_miss_unlink' ? { nearMiss: true } : {}),
      };

      for (let a = 0; a < Math.min(anglesPerVariant, CAMERA_ANGLES.length); a++) {
        const angle = CAMERA_ANGLES[a];
        const filename = `${sampleId}_${angle.name}.png`;

        const imageParams = {
          knotType,
          seed: variantSeed,
          deformStrength,
          slackness,
          cameraPosition: angle.pos,
          cameraTarget: [0, 0, 0],
          cameraFov: 45,
          useRainbow: true,
          metalness: 0.04, roughness: 0.5, tubeRadius: FIXED_LINK_TUBE_RADIUS,
          backgroundColor: '#1a2236',
          ...structuralImageParams,
        };

        try {
          const dataUrl = await renderSingleImage(imageParams, { width: renderWidth, height: renderHeight });
          renderedFileCount++;
          imagePayloads.push({ filename, dataUrl });
          if (retainSamples) {
            allFiles.push({ filename, blob: dataUrlToBlob(dataUrl) });
          }
          images.push({ filename, cameraAngle: angle.name, cameraPos: angle.pos });
        } catch (err) {
          console.warn(`Render failed: ${filename}`, err);
          images.push({ filename, cameraAngle: angle.name, cameraPos: angle.pos, error: err.message });
        }

        // Render T04 colored variant alongside the gradient version, if applicable.
        // Same scene, same camera, same seed — only the per-ring colors differ.
        if (t04Colored) {
          const coloredFilename = `${sampleId}_${angle.name}_colored.png`;
          const coloredParams = { ...imageParams, solidColors: ringColorsHex };
          try {
            const coloredDataUrl = await renderSingleImage(coloredParams, { width: renderWidth, height: renderHeight });
            renderedFileCount++;
            imagePayloads.push({ filename: coloredFilename, dataUrl: coloredDataUrl });
            if (retainSamples) {
              allFiles.push({ filename: coloredFilename, blob: dataUrlToBlob(coloredDataUrl) });
            }
            images.push({ filename: coloredFilename, cameraAngle: angle.name, cameraPos: angle.pos, colorMode: 'solid_per_ring' });
          } catch (err) {
            console.warn(`Colored render failed: ${coloredFilename}`, err);
          }
        }

        current++;
        onProgress(current, grandTotal, `[${current}/${grandTotal}] ${filename}`);
        if (current % 3 === 0) await sleep(10);
      }

      const family = FAMILY_MAP[knotType] || 'link';
      const meta = {
        version: '3.1',
        id: sampleId,
        generationIndex,
        knotType,
        topologicalId: entry.topologicalId,
        isKnot: false, isUnknot: false,
        isDeceptive: false,
        isLink: true,
        numComponents: numComp,
        crossingNumber: entry.crossingNumber,
        family,
        // Extra metadata for new types
        numLinkedComponents: linkedComponentCount,
        ...(entry.numFreeComponents != null ? { numFreeComponents: entry.numFreeComponents } : {}),
        ...(entry.isOpenRope ? { isOpenRope: true } : {}),
        // Structural params (used by T04 GT computation)
        ...(struct.numLinks != null ? { numLinks: struct.numLinks } : {}),
        ...(struct.chainLinks != null ? { chainLinks: struct.chainLinks } : {}),
        ...(struct.numFree != null ? { numFree: struct.numFree } : {}),
        ...(struct.clusterSegments ? { clusterSegments: struct.clusterSegments } : {}),
        // T04 linkage graph + per-ring color assignment
        linkageEdges: struct.linkageEdges,
        brunnianGroups: struct.brunnianGroups,
        ...(t04Colored ? { t04Colored: true, ringColors } : { t04Colored: false }),
        seed: variantSeed, slackness, deformStrength,
        bucket_topology: getBucketTopology(knotType),
        bucket_saliency: getBucketSaliency(slackness),
        trap_type: visualTrap,
        ...(visualTrap ? { visualTrap, isHardNegative: true } : {}),
        // Link difficulty score: based on crossing number + slackness + component count
        ...(() => {
          const crossing = entry.crossingNumber ?? 0;
          const compScore = clamp((numComp - 2) / 3, 0, 1);  // 2→0, 5→1
          const crossScore = clamp(crossing / 6, 0, 1);
          const slackScore = clamp(slackness, 0, 1);
          const trapScore = visualTrap ? 1 : 0;
          let score = 0.25 * crossScore + 0.30 * slackScore + 0.25 * compScore + 0.20 * trapScore;
          if (visualTrap === 'near_miss_unlink') score = Math.max(score, 0.48);
          if (visualTrap && HARD_LINK_TYPES.has(knotType)) score = Math.max(score, 0.46);
          return {
            difficulty_score: Math.round(score * 1000) / 1000,
            difficulty: score < 0.25 ? 'easy' : score < 0.45 ? 'medium' : 'hard',
          };
        })(),
        images,
      };
      allMetadata.push(meta);
      await emitSample({
        kind: 'single',
        section: sectionForSingleMetadata(meta),
        metadata: meta,
        files: imagePayloads,
      });
    }
  }

  // -- Step 3: Pair data --
  onProgress(current, grandTotal, 'Generating pair data...');
  const pairRng = makeRng(seed + '-pairs');
  const numPos = Math.floor(numPairs / 2);
  const numNeg = numPairs - numPos;
  const pairMetadata = [];

  for (let i = 0; i < numPos + numNeg; i++) {
    const isPositive = i < numPos;
    const difficulty = ['easy', 'medium', 'hard'][i % 3];
    const pair = isPositive
      ? generatePositivePair(pairRng, difficulty)
      : generateNegativePair(pairRng, difficulty);

    const pairId = `pair${String(i + 1).padStart(4, '0')}`;
    const generationIndex = sampleOrdinal++;
    if (!ownsSample(generationIndex) || skipSet.has(pairId)) {
      current += 2;
      continue;
    }
    const pairFiles = [];

    for (const [side, imgParams] of [['A', pair.imageA], ['B', pair.imageB]]) {
      const filename = `${pairId}_${side}.png`;
      try {
        const dataUrl = await renderSingleImage(imgParams, { width: renderWidth, height: renderHeight });
        renderedFileCount++;
        pairFiles.push({ filename, dataUrl });
        if (retainSamples) {
          allFiles.push({ filename, blob: dataUrlToBlob(dataUrl) });
        }
      } catch (err) {
        console.warn(`Pair render failed: ${filename}`, err);
      }

      current++;
      onProgress(current, grandTotal, `[${current}/${grandTotal}] ${filename}`);
      if (current % 2 === 0) await sleep(10);
    }

    pairMetadata.push({
      version: '3.0',
      id: pairId,
      generationIndex,
      label_equivalent: pair.label_equivalent,
      topologicalIdA: pair.topologicalIdA,
      topologicalIdB: pair.topologicalIdB,
      difficulty: pair.difficulty,
      difficulty_score: pair.difficulty_score,
      image1: `${pairId}_A.png`,
      image2: `${pairId}_B.png`,
      imageA: pair.imageA,
      imageB: pair.imageB,
    });
    await emitSample({
      kind: 'pair',
      section: 'pairs',
      metadata: pairMetadata[pairMetadata.length - 1],
      files: pairFiles,
    });
  }

  onProgress(grandTotal, grandTotal, 'Generation complete! Preparing download...');

  return {
    files: allFiles,
    singleMetadata: allMetadata,
    pairMetadata,
    stats: {
      totalImages: renderedFileCount,
      singleKnots: knotTypes.length * variantsPerType,
      links: linkTypes.reduce((sum, lt) => sum + (LINK_VARIANTS[lt] ?? linkVariantsDefault), 0),
      pairs: numPairs,
      grandTotalImages: grandTotal,
      expectedSingleSampleCount: knotTypes.length * variantsPerType
        + linkTypes.reduce((sum, lt) => sum + (LINK_VARIANTS[lt] ?? linkVariantsDefault), 0),
      expectedPairSampleCount: numPairs,
    },
  };
}

export function buildDatasetMetadata(dataset) {
  const singleMetadata = dataset?.singleMetadata || [];
  const pairMetadata = dataset?.pairMetadata || [];
  return {
    version: '4.0',
    task: 'knots_static',
    generated_at: new Date().toISOString(),
    samples: [...singleMetadata, ...pairMetadata],
    stats: dataset?.stats || {},
  };
}

/**
 * Package the dataset as a single ZIP file for download.
 * Directory structure:
 *   dataset/
 *     images/
 *       *.png
 *     dataset_metadata.json
 */
export async function downloadDatasetAsZip(dataset, onProgress = () => {}) {
  const { files } = dataset;

  // Dynamically load JSZip (CDN)
  onProgress(0, 1, 'Loading JSZip library...');
  if (!window.JSZip) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
      script.onload = resolve;
      script.onerror = () => reject(new Error('Failed to load JSZip'));
      document.head.appendChild(script);
    });
  }

  const zip = new window.JSZip();
  const total = files.length + 1;
  let done = 0;

  // 1. Global metadata
  const fullMetadata = buildDatasetMetadata(dataset);
  zip.file('dataset/dataset_metadata.json', JSON.stringify(fullMetadata, null, 2));
  done++;
  onProgress(done, total, 'dataset_metadata.json');

  // 2. Single-loop metadata + images
  const fileMap = new Map(); // filename → blob
  for (const f of files) {
    fileMap.set(f.filename, f.blob);
  }

  for (const [name, blob] of fileMap.entries()) {
    zip.file(`dataset/images/${name}`, blob);
    done++;
    onProgress(done, total, name);
  }

  // 3. Generate ZIP
  onProgress(done, total, 'Compressing ZIP file (may take ~30s)...');
  const zipBlob = await zip.generateAsync(
    { type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } },
    (meta) => {
      onProgress(done, total, `Compressing ${Math.round(meta.percent)}%`);
    }
  );

  // 4. Trigger download
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `knot_benchmark_dataset_${new Date().toISOString().slice(0,10)}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  return { zipSize: zipBlob.size };
}

export default { generateFullDataset, buildDatasetMetadata, downloadDatasetAsZip };
