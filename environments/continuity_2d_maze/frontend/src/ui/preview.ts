import {
  BAR_REMOVAL_TEMPLATE,
  REACHABILITY_SET_TEMPLATE,
  buildBarRemovalQuestion,
  buildReachabilitySetQuestion,
  renderFullPrompt,
} from '../maze/prompts';
import { SceneMetadata } from '../maze/types';
import { TabKey } from './tabs';

const IMAGE_STUB = 'images/<scene>/topdown.png';
const Q1_TARGET = 'A';

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function renderPreview(tab: TabKey, metadata: SceneMetadata): void {
  if (tab === 'q1') {
    renderQ1(metadata);
  } else {
    renderQ2(metadata);
  }
}

function renderQ1(metadata: SceneMetadata): void {
  if (metadata.pairs.length === 0) {
    setText('prev-question-text', '(no pairs)');
    setText('prev-gt-answer', '-');
    setText('prev-metadata-json', '-');
    setText('prev-full-prompt', '-');
    return;
  }
  // reachability_set view: question asks which OTHER points are connected
  // to the fixed target A; answer is the lex-sorted set of such names.
  const connected = metadata.pairs
    .filter(
      (p) =>
        p.connected && (p.a_name === Q1_TARGET || p.b_name === Q1_TARGET),
    )
    .map((p) => (p.a_name === Q1_TARGET ? p.b_name : p.a_name));
  const connectedSet = Array.from(new Set(connected)).sort();
  const pointNames = metadata.points.map((p) => p.name);
  const question = buildReachabilitySetQuestion(Q1_TARGET, pointNames);
  const answer = JSON.stringify(connectedSet);
  setText('prev-question-text', question);
  setText('prev-gt-answer', answer);

  const metaSummary = {
    question_type: 'reachability_set',
    seed: metadata.seed,
    grid_size: metadata.grid_size,
    wall_density: metadata.wall_density,
    point_count: metadata.point_count,
    difficulty_tier: metadata.difficulty?.tier ?? null,
    target: Q1_TARGET,
    connected_set: connectedSet,
    points: metadata.points.map((p) => ({ name: p.name, x: p.x, y: p.y })),
  };
  setText('prev-metadata-json', JSON.stringify(metaSummary, null, 2));
  setText(
    'prev-full-prompt',
    renderFullPrompt(REACHABILITY_SET_TEMPLATE, question, [IMAGE_STUB]),
  );
}

function renderQ2(metadata: SceneMetadata): void {
  const bars = metadata.bars ?? [];
  const q = metadata.question;
  if (!q) {
    setText('prev-question-text', '(no bar_removal question)');
    setText('prev-gt-answer', '-');
    setText('prev-metadata-json', '-');
    setText('prev-full-prompt', '-');
    return;
  }
  const barColors = bars.map((b) => b.color);
  const question = buildBarRemovalQuestion(q.target_pair, barColors);
  const answer = JSON.stringify([...q.correct_removals].sort());
  setText('prev-question-text', question);
  setText('prev-gt-answer', answer);

  const aa = metadata.difficulty?.area_a;
  const ab = metadata.difficulty?.area_b;
  const mp = metadata.difficulty?.min_path;
  const metaSummary = {
    question_type: 'bar_removal',
    seed: metadata.seed,
    grid_size: metadata.grid_size,
    wall_density: metadata.wall_density,
    point_count: metadata.point_count,
    difficulty_tier: metadata.difficulty?.tier ?? null,
    // iter D15: diagnostic numbers, not used for tier classification.
    diagnostics: {
      min_path: mp ?? null,
      min_area:
        aa !== undefined && ab !== undefined ? Math.min(aa, ab) : null,
      area_a: aa ?? null,
      area_b: ab ?? null,
    },
    target_pair: q.target_pair,
    correct_removals: q.correct_removals,
    bars: bars.map((b) => ({ id: b.id, color: b.color, edge: b.edge })),
  };
  setText('prev-metadata-json', JSON.stringify(metaSummary, null, 2));
  setText(
    'prev-full-prompt',
    renderFullPrompt(BAR_REMOVAL_TEMPLATE, question, [IMAGE_STUB]),
  );
}

export function clearPreview(): void {
  setText('prev-question-text', '-');
  setText('prev-gt-answer', '-');
  setText('prev-metadata-json', '-');
  setText('prev-full-prompt', '-');
}
