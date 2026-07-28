import * as THREE from 'three';
import { allPairs } from './maze/connectivity';
import {
  DEFAULT_DIAGONAL_RATIO,
  DEFAULT_PARTIAL_BC_SPLIT,
  DEFAULT_PARTIAL_RATIO,
  generateMaze,
} from './maze/generator';
import {
  DifficultyTier,
  computeDifficultyMetric,
} from './maze/metric';
import { samplePoints } from './maze/points';
import {
  BarRemovalSceneConfig,
  generateBarRemovalScene,
} from './maze/question_gen';
import {
  Q2_NEUTRAL_POINT_COLORS,
  clearGroup,
  drawBars,
  drawHighlight,
  drawLabels,
  drawMaze,
  worldBounds,
} from './maze/renderer';
import { MazeConfig, SceneMetadata } from './maze/types';
import {
  BaseMazeConfig,
  Q2UIConfig,
  readConfigFromDOM,
  writeConfigToDOM,
  writeMetaDOM,
  writeStatsDOM,
} from './ui/controls';
import { downloadCanvasPng } from './ui/export';
import { clearPairSelectionDOM, renderPairListDOM } from './ui/pairs';
import { DIFFICULTY_PRESETS, DifficultyKey } from './ui/presets';
import { clearPreview, renderPreview } from './ui/preview';
import { TabKey, getActiveTab, initTabUI, onTabChange } from './ui/tabs';

const CANVAS_SIZE = 1024;

export function spanningTreeDensity(grid_size: number): number {
  return (grid_size - 1) / (2 * grid_size);
}

declare global {
  interface Window {
    topoBench?: {
      generate: (config?: Partial<MazeConfig>) => SceneMetadata;
      screenshot: () => string;
    };
  }
}

const container = document.getElementById('canvas-container') as HTMLElement;
if (!container) {
  throw new Error('Missing #canvas-container');
}

const { half } = worldBounds();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.OrthographicCamera(-half, half, half, -half, 0.1, 10);
camera.position.set(0, 0, 2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(CANVAS_SIZE, CANVAS_SIZE, false);
container.appendChild(renderer.domElement);

const mazeGroup = new THREE.Group();
scene.add(mazeGroup);

const highlightGroup = new THREE.Group();
scene.add(highlightGroup);

// ======================================================================
// Q1 reachability path
// ======================================================================
function runGenerate(
  config?: Partial<MazeConfig> & { difficulty_tier?: DifficultyTier },
): SceneMetadata {
  const seed = Number.isFinite(config?.seed as number) ? (config!.seed as number) : 12345;
  const grid_size = Number.isFinite(config?.grid_size as number)
    ? (config!.grid_size as number)
    : 4;
  const wall_density = Number.isFinite(config?.wall_density as number)
    ? (config!.wall_density as number)
    : spanningTreeDensity(grid_size);
  const point_count = Number.isFinite(config?.point_count as number)
    ? (config!.point_count as number)
    : 4;

  const maze = generateMaze(
    grid_size,
    seed,
    wall_density,
    config?.diagonal_ratio ?? DEFAULT_DIAGONAL_RATIO,
    config?.partial_ratio ?? DEFAULT_PARTIAL_RATIO,
    config?.partial_bc_split ?? DEFAULT_PARTIAL_BC_SPLIT,
  );
  const points = samplePoints(
    maze,
    seed,
    point_count,
    config?.min_pairwise_distance ?? 0,
  );
  const pairs = allPairs(points, maze);
  // iter D15: scene_score / detour_total / iso_total are now diagnostic
  // only; tier comes from the caller's config.difficulty_tier label.
  const difficulty =
    points.length >= 2
      ? (() => {
          const breakdown = computeDifficultyMetric(maze, points[0], points.slice(1));
          return { ...breakdown, tier: config?.difficulty_tier };
        })()
      : undefined;

  clearGroup(mazeGroup);
  drawMaze(mazeGroup, maze);
  drawLabels(mazeGroup, maze, points);
  renderer.render(scene, camera);

  return {
    seed,
    grid_size,
    wall_density,
    point_count,
    maze,
    points,
    pairs,
    canvas_size: CANVAS_SIZE,
    difficulty,
  };
}

// ======================================================================
// Q2 bar_removal path
// ======================================================================
function runGenerateQ2(
  config?: Partial<BarRemovalSceneConfig> & { bar_count?: number },
): SceneMetadata {
  const seed = Number.isFinite(config?.seed as number) ? (config!.seed as number) : 12345;
  const grid_size = Number.isFinite(config?.grid_size as number)
    ? (config!.grid_size as number)
    : 6;
  const meta = generateBarRemovalScene({
    ...(config ?? {}),
    seed,
    grid_size,
  });

  clearGroup(mazeGroup);
  drawMaze(mazeGroup, meta.maze);
  if (meta.bars) drawBars(mazeGroup, meta.maze, meta.bars);
  drawLabels(mazeGroup, meta.maze, meta.points, Q2_NEUTRAL_POINT_COLORS);
  renderer.render(scene, camera);
  return meta;
}

function isBarRemovalRequest(config?: Partial<MazeConfig>): boolean {
  return config?.question_type === 'bar_removal';
}

// ======================================================================
// Global Playwright entry. Dispatch by config.question_type; falls back to
// the active tab when no question_type is specified (manual browser use).
// ======================================================================
window.topoBench = {
  generate: (config) => {
    const tab = getActiveTab();
    if (isBarRemovalRequest(config) || tab === 'q2') {
      return runGenerateQ2(config as Partial<BarRemovalSceneConfig> | undefined);
    }
    return runGenerate(config);
  },
  screenshot: () => {
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png');
  },
};

// ======================================================================
// UI-driven flow
// ======================================================================
let lastMetadata: SceneMetadata | null = null;

function sceneIdPrefix(tab: TabKey, seed: number): string {
  const family = tab === 'q2' ? 'bar_removal' : 'reachability_set';
  return `continuity_2d_maze_${seed}_${family}_*`;
}

// Commit a pre-computed scene to the active tab's UI without regenerating.
// Used by both the Generate buttons (which regenerate from scratch) and the
// tier rejection-sampling buttons (which have already built a scene they
// want to commit verbatim; going through readConfigFromDOM → regenerate
// would silently lose precision because DOM stores numbers via toFixed(2),
// and the re-rendered scene could land in a different tier).
function commitMetaToUI(tab: TabKey, meta: SceneMetadata, elapsed: number): void {
  lastMetadata = meta;
  writeStatsDOM(tab, meta);
  writeMetaDOM(tab, sceneIdPrefix(tab, meta.seed), elapsed);
  renderPairListDOM(tab, meta.pairs, onPairSelect);
  clearGroup(highlightGroup);
  renderer.render(scene, camera);
  renderPreview(tab, meta);
}

function renderActiveFromDOM(): void {
  const tab = getActiveTab();
  const cfg = readConfigFromDOM(tab);
  const t0 = performance.now();
  let meta: SceneMetadata;
  if (tab === 'q1') {
    meta = runGenerate(cfg as BaseMazeConfig);
  } else {
    const q2cfg = cfg as Q2UIConfig;
    meta = runGenerateQ2({
      seed: q2cfg.seed,
      grid_size: q2cfg.grid_size,
      wall_density: q2cfg.wall_density,
      diagonal_ratio: q2cfg.diagonal_ratio,
      partial_ratio: q2cfg.partial_ratio,
      partial_bc_split: q2cfg.partial_bc_split,
      bar_count: q2cfg.bar_count,
    });
  }
  const elapsed = performance.now() - t0;
  commitMetaToUI(tab, meta, elapsed);
}

function onPairSelect(pairIndex: number | null): void {
  if (!lastMetadata) return;
  clearGroup(highlightGroup);
  if (pairIndex !== null) {
    const pair = lastMetadata.pairs[pairIndex];
    if (pair) {
      const aIdx = lastMetadata.points.findIndex((p) => p.name === pair.a_name);
      const bIdx = lastMetadata.points.findIndex((p) => p.name === pair.b_name);
      if (aIdx >= 0 && bIdx >= 0) {
        drawHighlight(highlightGroup, lastMetadata.maze, lastMetadata.points, [aIdx, bIdx]);
      }
    }
  }
  renderer.render(scene, camera);
  renderPreview(getActiveTab(), lastMetadata);
}

// ======================================================================
// Tier-mode (iter D15): each tier has a FIXED config. Click "Generate
// hard" → load hard's TIER_CONFIG into the form and run generate. No
// rejection sampling, no score thresholds. Mirror of backend
// generate_samples.py::Q1_TIER_CONFIGS / Q2_TIER_CONFIGS.
// ======================================================================
type Tier = 'easy' | 'medium' | 'hard';

interface TierConfig {
  grid_size: number;
  wd_lo: number;
  wd_hi: number;
  diagonal_ratio: number;
  partial_ratio: number;
}

const Q1_TIER_CONFIGS: Record<Tier, TierConfig> = {
  easy:   { grid_size: 4, wd_lo: 0.35, wd_hi: 0.45, diagonal_ratio: 0.0,  partial_ratio: 0.0 },
  medium: { grid_size: 5, wd_lo: 0.40, wd_hi: 0.50, diagonal_ratio: 0.0,  partial_ratio: 0.30 },
  hard:   { grid_size: 6, wd_lo: 0.45, wd_hi: 0.55, diagonal_ratio: 0.25, partial_ratio: 0.30 },
};
const Q1_POINT_COUNT_LO = 3;
const Q1_POINT_COUNT_HI = 5;

const Q2_TIER_CONFIGS: Record<Tier, TierConfig> = {
  easy:   { grid_size: 4, wd_lo: 0.35, wd_hi: 0.45, diagonal_ratio: 0.0,  partial_ratio: 0.0 },
  medium: { grid_size: 5, wd_lo: 0.40, wd_hi: 0.50, diagonal_ratio: 0.0,  partial_ratio: 0.30 },
  hard:   { grid_size: 6, wd_lo: 0.45, wd_hi: 0.55, diagonal_ratio: 0.25, partial_ratio: 0.30 },
};

// iter D16/D17 extra rules. Per-tier minimums applied by the tier-mode
// loops (UI tier buttons + CLI tier_config). easy stays unconstrained;
// medium/hard force visually non-trivial scenes.
const Q1_MIN_PAIRWISE_DIST_BY_TIER: Record<Tier, number> = {
  easy: 0,
  medium: 3,
  hard: 3,
};
const Q2_MIN_PAIRWISE_DIST_BY_TIER: Record<Tier, number> = {
  easy: 0,
  medium: 3,
  hard: 5,
};
const Q2_MIN_CORRECT_REMOVALS_BY_TIER: Record<Tier, number> = {
  easy: 1,
  medium: 2,
  hard: 2,
};
const Q2_MIN_AREA_FRACTION_BY_TIER: Record<Tier, number> = {
  easy: 0,
  medium: 1 / 4,
  hard: 1 / 3,
};

const MAX_TIER_ATTEMPTS = 30;

// Tiny shared LCG. MUST match backend generate_samples.py::_lcg_step exactly
// so a given (seed, salt) yields the same jitter on UI tier buttons and
// CLI runs.
function lcgStep(seed: number, salt = 0): number {
  let s = ((seed >>> 0) ^ (salt >>> 0)) >>> 0;
  s = (Math.imul(s, 1103515245) + 12345) >>> 0;
  return s;
}
function jitterUniform(seed: number, lo: number, hi: number, salt = 0): number {
  const s = lcgStep(seed, salt);
  return lo + (s % 10000) / 10000 * (hi - lo);
}
function jitterIntInclusive(seed: number, lo: number, hi: number, salt = 1): number {
  const s = lcgStep(seed, salt);
  return lo + (s % (hi - lo + 1));
}

function tryGenerateForTierQ1(targetTier: Tier): void {
  const base = readConfigFromDOM('q1') as BaseMazeConfig;
  const tcfg = Q1_TIER_CONFIGS[targetTier];
  for (let attempt = 0; attempt < MAX_TIER_ATTEMPTS; attempt++) {
    const seed = base.seed + attempt + 1;
    const wd = jitterUniform(seed, tcfg.wd_lo, tcfg.wd_hi, 1);
    const pc = jitterIntInclusive(seed, Q1_POINT_COUNT_LO, Q1_POINT_COUNT_HI, 2);
    const cfg: BaseMazeConfig & {
      difficulty_tier: Tier;
      min_pairwise_distance: number;
    } = {
      seed,
      grid_size: tcfg.grid_size,
      wall_density: wd,
      point_count: pc,
      diagonal_ratio: tcfg.diagonal_ratio,
      partial_ratio: tcfg.partial_ratio,
      partial_bc_split: 0.5,
      difficulty_tier: targetTier,
      min_pairwise_distance: Q1_MIN_PAIRWISE_DIST_BY_TIER[targetTier],
    };
    try {
      writeConfigToDOM('q1', cfg);
      const t0 = performance.now();
      const meta = runGenerate(cfg);
      commitMetaToUI('q1', meta, performance.now() - t0);
      return;
    } catch {
      continue;
    }
  }
  console.warn(`Q1: could not generate a ${targetTier} maze in ${MAX_TIER_ATTEMPTS} attempts.`);
  alert(`Q1: could not generate a ${targetTier} maze in ${MAX_TIER_ATTEMPTS} attempts.`);
}

function tryGenerateForTierQ2(targetTier: Tier): void {
  const base = readConfigFromDOM('q2') as Q2UIConfig;
  const tcfg = Q2_TIER_CONFIGS[targetTier];
  for (let attempt = 0; attempt < MAX_TIER_ATTEMPTS; attempt++) {
    const trial_seed = base.seed + attempt + 1;
    const wd = jitterUniform(trial_seed, tcfg.wd_lo, tcfg.wd_hi, 1);
    try {
      const meta = generateBarRemovalScene({
        seed: trial_seed,
        grid_size: tcfg.grid_size,
        wall_density: wd,
        bar_count: base.bar_count,
        diagonal_ratio: tcfg.diagonal_ratio,
        partial_ratio: tcfg.partial_ratio,
        partial_bc_split: 0.5,
        difficulty_tier: targetTier,
        min_correct_removals: Q2_MIN_CORRECT_REMOVALS_BY_TIER[targetTier],
        min_pairwise_distance: Q2_MIN_PAIRWISE_DIST_BY_TIER[targetTier],
        min_area_fraction: Q2_MIN_AREA_FRACTION_BY_TIER[targetTier],
      });
      writeConfigToDOM('q2', {
        seed: trial_seed,
        grid_size: tcfg.grid_size,
        wall_density: wd,
        diagonal_ratio: tcfg.diagonal_ratio,
        partial_ratio: tcfg.partial_ratio,
      });
      clearGroup(mazeGroup);
      drawMaze(mazeGroup, meta.maze);
      if (meta.bars) drawBars(mazeGroup, meta.maze, meta.bars);
      drawLabels(mazeGroup, meta.maze, meta.points, Q2_NEUTRAL_POINT_COLORS);
      commitMetaToUI('q2', meta, 0);
      return;
    } catch {
      continue;
    }
  }
  console.warn(`Q2: could not generate a ${targetTier} maze in ${MAX_TIER_ATTEMPTS} attempts.`);
  alert(`Q2: could not generate a ${targetTier} maze in ${MAX_TIER_ATTEMPTS} attempts.`);
}

// ======================================================================
// Load Sample (paste a sample.json element / its meta_info / metadata) →
// switch tab + fill form + render. Diag/partial ratios aren't stored in
// sample metadata; we infer them from the tier configs above.
// ======================================================================
function setLoadStatus(msg: string, color = '#888'): void {
  const el = document.getElementById('load-sample-status');
  if (el) {
    el.textContent = msg;
    (el as HTMLElement).style.color = color;
  }
}

function loadSampleFromJSON(rawJson: string): void {
  // Tolerate common copy-paste mistakes: trailing comma after the closing
  // brace (when grabbing one element out of the `samples` array), and
  // surrounding whitespace.
  const cleaned = rawJson.trim().replace(/,\s*$/, '');
  let parsed: any;
  try {
    parsed = JSON.parse(cleaned);
  } catch (e) {
    setLoadStatus(`JSON parse error: ${(e as Error).message}`, '#c33');
    return;
  }
  // If the user pasted the whole samples.json payload, descend into the
  // first sample.
  if (parsed && Array.isArray(parsed.samples) && parsed.samples.length > 0) {
    parsed = parsed.samples[0];
  }
  // Accept either a full sample, or the meta_info / metadata block alone.
  const meta = parsed.metadata ?? parsed.meta_info ?? parsed;
  const qtype: string =
    parsed.question_type ?? meta.question_type ?? parsed.type ??
    (meta.target_pair !== undefined ? 'bar_removal' : 'reachability_set');
  const tier: string | undefined =
    parsed.difficulty ?? meta.difficulty ?? meta.difficulty_tier;
  const seed = Number(meta.seed);
  const grid_size = Number(meta.grid_size);
  const wall_density = Number(meta.wall_density);
  if (!Number.isFinite(seed) || !Number.isFinite(grid_size) || !Number.isFinite(wall_density)) {
    setLoadStatus('Missing seed/grid_size/wall_density in pasted JSON.', '#c33');
    return;
  }
  // Recover diag/partial from per-tier configs (sample metadata doesn't
  // store them since they are tier-fixed). Default to 0/0 if no tier.
  const tierKey = (tier === 'easy' || tier === 'medium' || tier === 'hard') ? tier as Tier : null;
  const tcfg = tierKey
    ? (qtype === 'bar_removal' ? Q2_TIER_CONFIGS[tierKey] : Q1_TIER_CONFIGS[tierKey])
    : null;
  const diagonal_ratio = tcfg?.diagonal_ratio ?? 0;
  const partial_ratio = tcfg?.partial_ratio ?? 0;

  const targetTab: TabKey = qtype === 'bar_removal' ? 'q2' : 'q1';
  // Switch tab via the tab button click (re-uses existing init wiring).
  const tabBtn = document.getElementById(`tab-${targetTab}`) as HTMLElement | null;
  if (tabBtn) tabBtn.click();

  if (targetTab === 'q1') {
    const point_count = Number(meta.point_count) || 4;
    const cfg = {
      seed,
      grid_size,
      wall_density,
      point_count,
      diagonal_ratio,
      partial_ratio,
      partial_bc_split: 0.5,
      difficulty_tier: tierKey ?? undefined,
    };
    writeConfigToDOM('q1', cfg);
    const t0 = performance.now();
    const m = runGenerate(cfg);
    commitMetaToUI('q1', m, performance.now() - t0);
  } else {
    const bar_count = Array.isArray(meta.bars) ? meta.bars.length : Number(meta.bar_count) || 4;
    try {
      const m = generateBarRemovalScene({
        seed,
        grid_size,
        wall_density,
        bar_count,
        diagonal_ratio,
        partial_ratio,
        partial_bc_split: 0.5,
        difficulty_tier: tierKey ?? undefined,
      });
      writeConfigToDOM('q2', { seed, grid_size, wall_density, diagonal_ratio, partial_ratio });
      clearGroup(mazeGroup);
      drawMaze(mazeGroup, m.maze);
      if (m.bars) drawBars(mazeGroup, m.maze, m.bars);
      drawLabels(mazeGroup, m.maze, m.points, Q2_NEUTRAL_POINT_COLORS);
      commitMetaToUI('q2', m, 0);
    } catch (e) {
      setLoadStatus(`Q2 generator failed: ${(e as Error).message}`, '#c33');
      return;
    }
  }
  setLoadStatus(
    `Loaded ${qtype} ${tier ?? '?'} seed=${seed} grid=${grid_size} wd=${wall_density.toFixed(3)}.`,
    '#2a7',
  );
}

// ======================================================================
// Init
// ======================================================================
initTabUI();
const initialTab = getActiveTab();
clearPreview();
renderActiveFromDOM();

onTabChange(() => {
  renderActiveFromDOM();
});

if (document.getElementById('controls')) {
  // Load Sample binding (shared across tabs).
  document.getElementById('load-sample-btn')?.addEventListener('click', () => {
    const ta = document.getElementById('load-sample-json') as HTMLTextAreaElement | null;
    if (!ta || !ta.value.trim()) {
      setLoadStatus('Paste a sample JSON first.', '#c33');
      return;
    }
    loadSampleFromJSON(ta.value);
  });

  // Q1 bindings
  document
    .getElementById('q1-btn-generate')
    ?.addEventListener('click', () => renderActiveFromDOM());
  document.getElementById('q1-btn-randomize')?.addEventListener('click', () => {
    const cfg = readConfigFromDOM('q1') as BaseMazeConfig;
    writeConfigToDOM('q1', { seed: cfg.seed + 1 });
    renderActiveFromDOM();
  });
  (['easy', 'medium', 'hard'] as const).forEach((tier) => {
    document
      .getElementById(`q1-btn-gen-${tier}`)
      ?.addEventListener('click', () => tryGenerateForTierQ1(tier));
  });
  (['easy', 'medium', 'hard'] as DifficultyKey[]).forEach((key) => {
    document.getElementById(`q1-btn-preset-${key}`)?.addEventListener('click', () => {
      const p = DIFFICULTY_PRESETS[key];
      writeConfigToDOM('q1', { grid_size: p.grid_size, wall_density: p.wall_density });
      renderActiveFromDOM();
    });
  });
  document.getElementById('q1-btn-export')?.addEventListener('click', () => {
    renderer.render(scene, camera);
    downloadCanvasPng(renderer.domElement, readConfigFromDOM('q1') as BaseMazeConfig);
  });
  document.getElementById('q1-btn-clear-highlight')?.addEventListener('click', () => {
    clearPairSelectionDOM('q1');
    onPairSelect(null);
  });

  // Q2 bindings
  document
    .getElementById('q2-btn-generate')
    ?.addEventListener('click', () => renderActiveFromDOM());
  document.getElementById('q2-btn-randomize')?.addEventListener('click', () => {
    const cfg = readConfigFromDOM('q2') as Q2UIConfig;
    writeConfigToDOM('q2', { seed: cfg.seed + 1 });
    renderActiveFromDOM();
  });
  (['easy', 'medium', 'hard'] as const).forEach((tier) => {
    document
      .getElementById(`q2-btn-gen-${tier}`)
      ?.addEventListener('click', () => tryGenerateForTierQ2(tier));
  });
  document.getElementById('q2-btn-export')?.addEventListener('click', () => {
    renderer.render(scene, camera);
    const cfg = readConfigFromDOM('q2') as Q2UIConfig;
    // Map Q2 config into the shape downloadCanvasPng expects; Q2 has
    // point_count=2 and no extra fields it cares about.
    downloadCanvasPng(renderer.domElement, {
      seed: cfg.seed,
      grid_size: cfg.grid_size,
      wall_density: cfg.wall_density,
      point_count: cfg.point_count,
      diagonal_ratio: cfg.diagonal_ratio,
      partial_ratio: cfg.partial_ratio,
      partial_bc_split: cfg.partial_bc_split,
    });
  });
  document.getElementById('q2-btn-clear-highlight')?.addEventListener('click', () => {
    clearPairSelectionDOM('q2');
    onPairSelect(null);
  });
}

void initialTab; // reserved for future use
