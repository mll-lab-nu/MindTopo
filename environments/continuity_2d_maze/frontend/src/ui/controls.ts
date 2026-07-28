import { MazeConfig, SceneMetadata } from '../maze/types';
import { TabKey } from './tabs';

// Per-tab DOM input state. Each tab owns a disjoint id namespace
// (`q1-seed` vs `q2-seed` etc) so Q1 and Q2 can coexist in the same page
// without interfering; tab switching just hides/shows the corresponding
// panel and reads from the active tab when the user hits Generate.

export type BaseMazeConfig = Required<
  Omit<MazeConfig, 'question_type' | 'bar_count' | 'min_pairwise_distance' | 'min_correct_removals'>
>;

export interface Q2UIConfig extends BaseMazeConfig {
  bar_count: number;
}

export const Q1_DEFAULTS: BaseMazeConfig = {
  seed: 12345,
  grid_size: 4,
  wall_density: 0.45,
  point_count: 4,
  diagonal_ratio: 0.15,
  partial_ratio: 0.2,
  partial_bc_split: 0.5,
};

export const Q2_DEFAULTS: Q2UIConfig = {
  seed: 12345,
  grid_size: 6,
  wall_density: 0.45,
  point_count: 2,
  diagonal_ratio: 0.15,
  partial_ratio: 0.2,
  partial_bc_split: 0.5,
  bar_count: 4,
};

function getInput(id: string): HTMLInputElement | null {
  const el = document.getElementById(id);
  return el instanceof HTMLInputElement ? el : null;
}

function readInt(id: string, fallback: number): number {
  const el = getInput(id);
  if (!el) return fallback;
  const n = parseInt(el.value, 10);
  return Number.isFinite(n) ? n : fallback;
}

function readFloat(id: string, fallback: number): number {
  const el = getInput(id);
  if (!el) return fallback;
  const n = parseFloat(el.value);
  return Number.isFinite(n) ? n : fallback;
}

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function readConfigFromDOM(tab: TabKey): BaseMazeConfig | Q2UIConfig {
  if (tab === 'q1') {
    const grid_size = readInt('q1-grid_size', Q1_DEFAULTS.grid_size);
    // Dynamic fallback matches D8a: blank wall_density → spanning-tree density.
    const wall_density_fallback = (grid_size - 1) / (2 * grid_size);
    return {
      seed: readInt('q1-seed', Q1_DEFAULTS.seed),
      grid_size,
      wall_density: readFloat('q1-wall_density', wall_density_fallback),
      point_count: readInt('q1-point_count', Q1_DEFAULTS.point_count),
      diagonal_ratio: readFloat('q1-diagonal_ratio', Q1_DEFAULTS.diagonal_ratio),
      partial_ratio: readFloat('q1-partial_ratio', Q1_DEFAULTS.partial_ratio),
      partial_bc_split: readFloat('q1-partial_bc_split', Q1_DEFAULTS.partial_bc_split),
    };
  }
  // Q2 — point_count is fixed at 2 downstream; we still return a concrete
  // number so the config shape stays uniform.
  return {
    seed: readInt('q2-seed', Q2_DEFAULTS.seed),
    grid_size: readInt('q2-grid_size', Q2_DEFAULTS.grid_size),
    wall_density: readFloat('q2-wall_density', Q2_DEFAULTS.wall_density),
    point_count: 2,
    diagonal_ratio: readFloat('q2-diagonal_ratio', Q2_DEFAULTS.diagonal_ratio),
    partial_ratio: readFloat('q2-partial_ratio', Q2_DEFAULTS.partial_ratio),
    partial_bc_split: readFloat('q2-partial_bc_split', Q2_DEFAULTS.partial_bc_split),
    bar_count: readInt('q2-bar_count', Q2_DEFAULTS.bar_count),
  };
}

export function writeConfigToDOM(tab: TabKey, config: Partial<Q2UIConfig>): void {
  const prefix = tab === 'q1' ? 'q1' : 'q2';
  const writeNum = (suffix: string, value: number | undefined, toFixed?: number) => {
    if (value === undefined) return;
    const el = getInput(`${prefix}-${suffix}`);
    if (!el) return;
    el.value = toFixed !== undefined ? value.toFixed(toFixed) : String(value);
  };
  writeNum('seed', config.seed);
  writeNum('grid_size', config.grid_size);
  writeNum('wall_density', config.wall_density, 2);
  writeNum('diagonal_ratio', config.diagonal_ratio, 2);
  writeNum('partial_ratio', config.partial_ratio, 2);
  writeNum('partial_bc_split', config.partial_bc_split, 2);
  if (tab === 'q1') writeNum('point_count', config.point_count);
  if (tab === 'q2') writeNum('bar_count', config.bar_count);
}

export function writeStatsDOM(tab: TabKey, metadata: SceneMetadata): void {
  if (tab === 'q1') {
    const total = metadata.pairs.length;
    const yes = metadata.pairs.filter((p) => p.connected).length;
    setText('q1-stat-total', String(total));
    setText('q1-stat-yes', String(yes));
    setText('q1-stat-no', String(total - yes));
    setText('q1-stat-ratio', total ? (yes / total).toFixed(2) : '-');
    setText('q1-stat-tier', metadata.difficulty?.tier ?? '-');
    return;
  }
  // Q2 (iter D15: tier from config, diagnostic-only numbers below)
  if (metadata.difficulty) {
    setText('q2-stat-tier', metadata.difficulty.tier ?? '-');
    const mp = metadata.difficulty.min_path;
    setText('q2-stat-min-path', mp !== undefined ? mp.toFixed(0) : '-');
    const aa = metadata.difficulty.area_a;
    const ab = metadata.difficulty.area_b;
    if (aa !== undefined && ab !== undefined) {
      const minA = Math.min(aa, ab);
      setText('q2-stat-min-area', minA.toFixed(0));
      setText('q2-stat-areas', `${aa} / ${ab}`);
    } else {
      setText('q2-stat-min-area', '-');
      setText('q2-stat-areas', '-');
    }
  } else {
    setText('q2-stat-tier', '-');
    setText('q2-stat-min-path', '-');
    setText('q2-stat-min-area', '-');
    setText('q2-stat-areas', '-');
  }
  const q = metadata.question;
  if (q) {
    setText('q2-stat-target-pair', `${q.target_pair[0]}—${q.target_pair[1]}`);
    setText(
      'q2-stat-correct-removals',
      q.correct_removals.length ? `[${q.correct_removals.join(', ')}]` : '[]',
    );
  } else {
    setText('q2-stat-target-pair', '-');
    setText('q2-stat-correct-removals', '-');
  }
  setText('q2-stat-bar-count', String((metadata.bars ?? []).length));
}

export function writeMetaDOM(tab: TabKey, sceneIdPrefix: string, generatedMs: number | null): void {
  const prefix = tab === 'q1' ? 'q1' : 'q2';
  setText(`${prefix}-scene-id`, sceneIdPrefix);
  setText(`${prefix}-scene-ms`, generatedMs === null ? '-' : `${generatedMs.toFixed(1)} ms`);
}
