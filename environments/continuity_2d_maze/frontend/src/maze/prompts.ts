// Frontend mirror of backend/prompts.py. Used by the UI preview panel so the
// user can see exactly what backend will emit without running the CLI.
//
// MUST STAY IN SYNC with backend/prompts.py (REACHABILITY_SET_TEMPLATE,
// BAR_REMOVAL_TEMPLATE, build_reachability_set_question, build_bar_removal_question).
// Run `python backend/check_prompts_sync.py` after editing either side.
//
// NOTE on sync mechanics: check_prompts_sync.py extracts only SINGLE-QUOTED
// string literals from this file via regex; variable concatenation (e.g.
// `'a' + SOME_CONST + 'b'`) is NOT recognized — it sees only the `'a'` and
// `'b'` parts. Therefore the wall-definition blurb is INLINED below in
// both templates rather than factored out into a shared const, and all
// strings here are single-quoted (apostrophes escaped as `\'`).

export const IMAGE_PLACEHOLDER = '{images}';

export const REACHABILITY_SET_TEMPLATE =
  '[Task]\n' +
  'You are looking at a top-down view of a 2D maze. Several cells ' +
  'are marked with labeled circles (A, B, C, ...). Your job is to ' +
  'decide, for the target point, which of the OTHER labeled points ' +
  'you can reach from it through the open maze corridors. A point Y ' +
  'is reachable from X exactly when you can travel from X to Y ' +
  'without crossing any wall. You can only step from one cell to a ' +
  'neighboring cell across an open edge — you cannot cut diagonally ' +
  'through a wall corner.\n' +
  'Wall types: Walls in the maze come in three visual styles, all ' +
  'of which block movement equally: (a) full edge walls — solid lines ' +
  'along an entire cell edge; (b) partial edge walls — solid line segments ' +
  'shorter than a full edge; (c) diagonal walls — solid lines cutting ' +
  'across a cell along its diagonal. Any solid line, regardless of length ' +
  'or orientation, is a wall and cannot be crossed. Each labeled circle ' +
  'marks one cell.\n' +
  'The visual evidence for this question is provided below.\n' +
  '[Image 1]\n' +
  'Attached image.\n' +
  '\n' +
  '[Rules]\n' +
  '1. Use only the images and text provided in this prompt.\n' +
  '2. If answer options are provided, choose only from the provided options.\n' +
  '3. Do not output explanation beyond the required final answer.\n' +
  '\n' +
  '[Question]\n' +
  '{question}\n' +
  '\n' +
  '[Answer Format]\n' +
  'Output exactly one JSON object using this schema: {"answer": {ans}} and nothing else.\n' +
  'Replace {ans} with the actual answer: a JSON array of point ' +
  'names, picked from the other labeled points shown in the image and ' +
  'NOT including the target itself, listed in alphabetical order. Use ' +
  '[] if no other point is reachable.\n';

export const BAR_REMOVAL_TEMPLATE =
  '[Task]\n' +
  'You are looking at a top-down view of a 2D maze. Two cells are ' +
  'marked with labeled circles A and B; in the current maze they are ' +
  'blocked from each other. Some of the maze\'s internal walls have been ' +
  'recolored as colored bars (purple / red / green / blue / yellow / ' +
  'orange). Suppose you are allowed to remove EXACTLY ONE bar — which ' +
  'colors of bar, when removed alone, would let A and B reach each ' +
  'other? Evaluate each bar independently: imagine removing only that ' +
  'one bar, leave every other bar in place, and check whether a path ' +
  'from A to B opens up. List EVERY color that works.\n' +
  'Wall types: Walls in the maze come in three visual styles, all ' +
  'of which block movement equally: (a) full edge walls — solid lines ' +
  'along an entire cell edge; (b) partial edge walls — solid line segments ' +
  'shorter than a full edge; (c) diagonal walls — solid lines cutting ' +
  'across a cell along its diagonal. Any solid line, regardless of length ' +
  'or orientation, is a wall and cannot be crossed. Each labeled circle ' +
  'marks one cell. A bar is a colored (non-black) wall; black walls are ' +
  'NOT bars.\n' +
  'Scoring: your answer is correct ONLY if you list every color whose ' +
  'single-bar removal reconnects A and B — no missing colors and no ' +
  'extras. A partial list, an extra color, or an empty answer when at ' +
  'least one qualifying bar exists, all count as wrong.\n' +
  'The visual evidence for this question is provided below.\n' +
  '[Image 1]\n' +
  'Attached image.\n' +
  '\n' +
  '[Rules]\n' +
  '1. Use only the images and text provided in this prompt.\n' +
  '2. If answer options are provided, choose only from the provided options.\n' +
  '3. Do not output explanation beyond the required final answer.\n' +
  '\n' +
  '[Question]\n' +
  '{question}\n' +
  '\n' +
  '[Answer Format]\n' +
  'Output exactly one JSON object using this schema: {"answer": {ans}} and nothing else.\n' +
  'Replace {ans} with the actual answer: a JSON array of ' +
  'color names from the bars shown in the image, in alphabetical order. ' +
  'Use [] only if NO single-bar removal connects them.\n';

export function buildReachabilitySetQuestion(
  target: string,
  pointNames: readonly string[],
): string {
  const others = pointNames.filter((n) => n !== target).slice().sort();
  const othersStr = others.length === 0 ? '(none)' : others.join(', ');
  return (
    `Which of the other labeled points can you reach from point ` +
    `${target}? The other labeled points in this maze are: ${othersStr}.`
  );
}

export function buildBarRemovalQuestion(
  targetPair: readonly [string, string],
  barColors: readonly string[],
): string {
  const [a, b] = targetPair;
  const uniqueSorted = Array.from(new Set(barColors)).sort();
  const colorsStr = uniqueSorted.join(', ');
  return (
    `Identify EVERY color of bar that, if removed alone, reconnects ` +
    `point ${a} and point ${b} (which are currently blocked from each ` +
    `other). List all qualifying colors — missing any one of them ` +
    `counts as wrong. The bars in this maze are colored: ${colorsStr}.`
  );
}

function formatImageBlock(images: readonly string[]): string {
  if (images.length === 0) return '[No images provided]';
  const lines: string[] = [];
  for (const path of images) {
    lines.push('[Image]');
    lines.push(path);
  }
  return lines.join('\n');
}

export function renderFullPrompt(
  template: string,
  question: string,
  images: readonly string[] = [],
): string {
  const body = template.replace('{question}', question);
  return body.replace(IMAGE_PLACEHOLDER, formatImageBlock(images));
}
