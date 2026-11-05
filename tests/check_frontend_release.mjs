// Offline replay of the release's explicit rules and saved reference solutions.
import fs from 'node:fs';
import assert from 'node:assert/strict';

function moduleValues(path, names) {
  const source = fs.readFileSync(path, 'utf8').replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
  return new Function(source + '\nreturn {' + names.join(',') + '}')();
}
function hardRows(task) {
  return fs.readFileSync(`data/core/${task}/question.jsonl`, 'utf8').trim().split('\n')
    .map(JSON.parse).filter(row => row.meta_info.benchmark_level === 'Hard');
}

const {OrderSwap2DPuzzleGame: Swap} = moduleValues(
  'environments/order_swap_2d_puzzle/frontend/src/swap_2d_puzzle.ts', ['OrderSwap2DPuzzleGame']);
for (const row of hardRows('order_swap_2d_puzzle')) {
  const game = new Swap(null), cfg = row.meta_info.initial_state.reset_config;
  game.reset({...cfg, animate: false});
  assert.equal(game.config.gridRows, 5);
  assert.equal(game.getDebugState().legalActions.length, 8);
  for (const action of cfg.referenceActions) assert.equal((await game.step(action)).info.illegal, false);
  assert.equal(game.getState().success, true, row.id);
}
console.log('PASS: all 150 Swap Hard reference solutions and constrained legal moves');

const {ContinuityPipeGame: Pipe} = moduleValues(
  'environments/continuity_pipe/frontend/src/pipe.js', ['ContinuityPipeGame']);
for (const row of hardRows('continuity_pipe')) {
  const game = new Pipe(null), cfg = row.meta_info.initial_state.reset_config;
  const zero = cfg.rotations.map(() => 0);
  game.reset({...cfg, rotations: zero});
  assert.equal(game.getState().success, true, row.id);
  const wrong = zero.slice(); wrong[cfg.doubleBendPairings.findIndex(value => value !== 0)] = 1;
  game.reset({...cfg, rotations: wrong});
  assert.equal(game.getState().success, false, row.id);
  const overpass = zero.slice();
  overpass[cfg.solvedMasks.findIndex((mask, index) => mask === 15 && !cfg.doubleBendPairings[index])] = 1;
  game.reset({...cfg, rotations: overpass});
  assert.equal(game.getState().success, true, row.id);
}
console.log('PASS: all 150 Pipe Hard solutions, wrong-pair counterexamples and invariant overpasses');

const cat = moduleValues('environments/enclosure_chat_noir/frontend/src/chat_noir.ts',
  ['EnclosureChatNoirGame', 'oneTurnRiskSet', 'hasOneTurnCaptureRisk']);
for (const row of hardRows('enclosure_chat_noir')) {
  const game = new cat.EnclosureChatNoirGame(null), meta = row.meta_info;
  game.reset(meta.initial_state.reset_config);
  assert.equal(game.config.catPolicy.name, 'expert');
  const sequence = Object.values(meta.solvability_filter.winning_action_sequences)[0];
  for (const action of sequence) if ((await game.step(action)).done) break;
  assert.equal(game.getState().success, true, row.id);
}
// Independently compare the fast articulation test with the existing exhaustive
// one-block enclosure test on open cells of three saved boards.
for (const row of hardRows('enclosure_chat_noir').slice(0, 3)) {
  const game = new cat.EnclosureChatNoirGame(null);
  game.reset(row.meta_info.initial_state.reset_config);
  const risk = cat.oneTurnRiskSet(game.board, game.blockedSet, new Map());
  for (const cell of game.board.cells) {
    if (!game.blockedSet.has(cell.index)) assert.equal(risk.has(cell.index),
      cat.hasOneTurnCaptureRisk(game.board, cell.index, game.blockedSet));
  }
}
console.log('PASS: all 150 Chat Noir expert reference wins and independent capture-risk checks');
