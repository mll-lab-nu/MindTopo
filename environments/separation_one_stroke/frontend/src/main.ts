/**
 * TopoBench - One Stroke Partition Game
 * 
 * Main entry point
 */

import { Game } from './game';
import { Renderer } from './renderer';
import { BOARD_SIZES, generateOneStrokeLevel, normalizeBoardSize, type BoardSize } from './generator';
import { getLevel, getLevelCount } from './levels';
import type { Direction, LevelJson, StateResponse, EvaluateResult, StepResult } from './types';

// Global game instances
let game: Game;
let renderer: Renderer;
let currentLevelIndex = 0;
let currentBoardSize: BoardSize = 4;
let currentSeed = 1;
let currentMode: 'generated' | 'builtin' | 'custom' = 'generated';

/**
 * Initialize the game
 */
function init(): void {
  const container = document.getElementById('game-container');
  if (!container) {
    console.error('Game container not found');
    return;
  }

  // Wait for CSS layout to compute non-zero container dimensions before
  // constructing the Three.js WebGLRenderer. The container is `flex: 1`, so
  // its clientWidth/clientHeight read 0 if we run before layout — that
  // initializes the WebGL drawing buffer at 0×0, after which the GL context
  // is unrecoverable and every snapshot reads back as fully transparent
  // black. requestAnimationFrame fires post-layout (and post-paint in most
  // browsers), so by the time the callback runs the container is sized.
  const startWhenSized = () => {
    if (!container.clientWidth || !container.clientHeight) {
      requestAnimationFrame(startWhenSized);
      return;
    }
    game = new Game();
    renderer = new Renderer(container);

    setupGeneratorControls();
    setupKeyboardControls();
    setupButtonControls();

    loadGeneratedSetup(currentBoardSize, currentSeed);

    exposeAPI();
  };
  requestAnimationFrame(startWhenSized);
}

/**
 * Load a level by index
 */
function loadLevelByIndex(index: number): void {
  const level = getLevel(index);
  if (!level) {
    console.error(`Level ${index} not found`);
    return;
  }
  
  currentLevelIndex = index;
  currentMode = 'builtin';
  applyLevel(level);
  updateSetupUI();
}

/**
 * Load a level from JSON
 */
function loadLevel(levelJson: LevelJson): void {
  currentMode = 'custom';
  applyLevel(levelJson);
  updateSetupUI();
}

function difficultyForBoardSize(boardSize: number): 'easy' | 'medium' | 'hard' {
  const size = normalizeBoardSize(boardSize);
  if (size === 4) return 'easy';
  if (size === 5) return 'medium';
  return 'hard';
}

function applyLevel(levelJson: LevelJson): void {
  game.loadLevel(levelJson);
  renderer.initBoard(levelJson);
  
  const state = game.getState();
  renderer.updateStroke(state.stroke, state.currentPos);
  
  updateStatusUI();
  updateLegendUI(levelJson);
  hideMessage();
}

function loadGeneratedSetup(boardSize: number, seed: number): void {
  currentBoardSize = normalizeBoardSize(boardSize);
  currentSeed = Math.max(1, Math.trunc(seed));
  currentMode = 'generated';
  applyLevel(generateOneStrokeLevel(currentBoardSize, currentSeed));
  updateSetupUI();
}

function loadQuestionRow(row: unknown): void {
  const payload = row as any;
  const resetConfig = payload?.meta_info?.initial_state?.reset_config ?? payload?.reset_config;
  const levelJson = resetConfig?.levelJson
    ?? payload?.meta_info?.initial_state?.level_json
    ?? payload?.meta_info?.level_json;
  if (levelJson) {
    if (resetConfig?.boardSize !== undefined) {
      currentBoardSize = normalizeBoardSize(Number(resetConfig.boardSize));
    } else if (typeof levelJson.W === 'number') {
      currentBoardSize = normalizeBoardSize(Number(levelJson.W) - 1);
    }
    if (resetConfig?.seed !== undefined) {
      currentSeed = Math.max(1, Math.trunc(Number(resetConfig.seed)));
    }
    currentMode = 'generated';
    applyLevel(levelJson);
    updateSetupUI();
    return;
  }
  if (resetConfig?.boardSize !== undefined || resetConfig?.seed !== undefined) {
    loadGeneratedSetup(resetConfig.boardSize ?? currentBoardSize, resetConfig.seed ?? currentSeed);
    return;
  }
  throw new Error('Question row does not contain a loadable One Stroke setup.');
}

/**
 * Reset current level
 */
function reset(): void {
  game.reset();
  
  const state = game.getState();
  renderer.updateStroke(state.stroke, state.currentPos);
  
  updateStatusUI();
  hideMessage();
}

/**
 * Execute a step in the given direction
 */
function stepDir(dir: Direction): StepResult {
  const result = game.stepDir(dir);
  
  if (result.ok) {
    const state = game.getState();
    renderer.updateStroke(state.stroke, state.currentPos);
    updateStatusUI();
    
    if (result.done) {
      if (result.success) {
        showMessage('Success!', 'You solved the puzzle!', 'success');
        renderer.showSuccess();
      } else {
        const evaluation = game.evaluate();
        showMessage('Not Quite...', evaluation.reason || 'Color constraints not satisfied', 'error');
        renderer.showFailure();
      }
    }
  } else {
    // Flash the key to indicate invalid move
    flashInvalidMove(dir);
  }
  
  return result;
}

/**
 * Execute a step to the given position
 */
function stepTo(x: number, y: number): StepResult {
  const result = game.stepTo(x, y);
  
  if (result.ok) {
    const state = game.getState();
    renderer.updateStroke(state.stroke, state.currentPos);
    updateStatusUI();
    
    if (result.done) {
      if (result.success) {
        showMessage('Success!', 'You solved the puzzle!', 'success');
        renderer.showSuccess();
      } else {
        const evaluation = game.evaluate();
        showMessage('Not Quite...', evaluation.reason || 'Color constraints not satisfied', 'error');
        renderer.showFailure();
      }
    }
  }
  
  return result;
}

/**
 * Undo the last step
 */
function undo(): boolean {
  const result = game.undo();
  
  if (result) {
    const state = game.getState();
    renderer.updateStroke(state.stroke, state.currentPos);
    updateStatusUI();
    hideMessage();
  }
  
  return result;
}

/**
 * Get current game state
 */
function getState(): StateResponse {
  return game.getState();
}

/**
 * Evaluate current state
 */
function evaluate(): EvaluateResult {
  return game.evaluate();
}

/**
 * Take a snapshot of the current view
 */
function snapshot(): Promise<string> {
  return renderer.snapshot();
}

// ============================================================================
// UI Functions
// ============================================================================

function setupGeneratorControls(): void {
  const boardSizeSelect = document.getElementById('board-size') as HTMLSelectElement | null;
  const seedInput = document.getElementById('seed-input') as HTMLInputElement | null;
  const generateBtn = document.getElementById('btn-generate');
  const nextSeedBtn = document.getElementById('btn-next-seed');

  if (boardSizeSelect) {
    boardSizeSelect.innerHTML = '';
    for (const size of BOARD_SIZES) {
      const option = document.createElement('option');
      option.value = String(size);
      option.textContent = `${size}x${size} (${difficultyForBoardSize(size)})`;
      boardSizeSelect.appendChild(option);
    }
    boardSizeSelect.addEventListener('change', loadGeneratedSetupFromUI);
  }

  if (seedInput) {
    seedInput.addEventListener('change', loadGeneratedSetupFromUI);
    seedInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        loadGeneratedSetupFromUI();
      }
    });
  }

  if (generateBtn) {
    generateBtn.addEventListener('click', loadGeneratedSetupFromUI);
  }

  if (nextSeedBtn) {
    nextSeedBtn.addEventListener('click', () => {
      const nextSeed = currentSeed + 1;
      if (seedInput) {
        seedInput.value = String(nextSeed);
      }
      loadGeneratedSetup(currentBoardSize, nextSeed);
    });
  }
}

function loadGeneratedSetupFromUI(): void {
  const boardSizeSelect = document.getElementById('board-size') as HTMLSelectElement | null;
  const seedInput = document.getElementById('seed-input') as HTMLInputElement | null;
  const boardSize = normalizeBoardSize(Number(boardSizeSelect?.value ?? currentBoardSize));
  const seed = Number(seedInput?.value ?? currentSeed);
  loadGeneratedSetup(boardSize, Number.isFinite(seed) ? seed : currentSeed);
}

function updateSetupUI(): void {
  const boardSizeSelect = document.getElementById('board-size') as HTMLSelectElement | null;
  const seedInput = document.getElementById('seed-input') as HTMLInputElement | null;
  const boardEl = document.getElementById('status-board');
  const seedEl = document.getElementById('status-seed');
  const difficultyEl = document.getElementById('status-difficulty');

  if (boardSizeSelect) {
    boardSizeSelect.value = String(currentBoardSize);
  }
  if (seedInput) {
    seedInput.value = String(currentSeed);
  }
  if (boardEl) {
    boardEl.textContent = currentMode === 'generated' ? `${currentBoardSize}x${currentBoardSize}` : `Level ${currentLevelIndex + 1}`;
  }
  if (seedEl) {
    seedEl.textContent = currentMode === 'generated' ? String(currentSeed) : '-';
  }
  if (difficultyEl) {
    difficultyEl.textContent = currentMode === 'generated' ? difficultyForBoardSize(currentBoardSize) : '-';
  }
}

function setupKeyboardControls(): void {
  document.addEventListener('keydown', (e) => {
    // Ignore if typing in input
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    
    switch (e.key) {
      case 'ArrowUp':
      case 'w':
      case 'W':
        e.preventDefault();
        stepDir('U');
        break;
      case 'ArrowDown':
      case 's':
      case 'S':
        e.preventDefault();
        stepDir('D');
        break;
      case 'ArrowLeft':
      case 'a':
      case 'A':
        e.preventDefault();
        stepDir('L');
        break;
      case 'ArrowRight':
      case 'd':
      case 'D':
        e.preventDefault();
        stepDir('R');
        break;
      case 'r':
      case 'R':
        e.preventDefault();
        reset();
        break;
      case 'z':
      case 'Z':
        e.preventDefault();
        undo();
        break;
    }
  });
}

function setupButtonControls(): void {
  // Reset button
  const resetBtn = document.getElementById('btn-reset');
  if (resetBtn) {
    resetBtn.addEventListener('click', reset);
  }
  
  // Undo button
  const undoBtn = document.getElementById('btn-undo');
  if (undoBtn) {
    undoBtn.addEventListener('click', undo);
  }
  
  // Direction keys
  const dirKeys = document.querySelectorAll('.key[data-dir]');
  dirKeys.forEach((key) => {
    key.addEventListener('click', () => {
      const dir = (key as HTMLElement).dataset.dir as Direction;
      if (dir) {
        stepDir(dir);
        key.classList.add('active');
        setTimeout(() => key.classList.remove('active'), 100);
      }
    });
  });
}

function updateStatusUI(): void {
  const state = game.getState();
  
  const posEl = document.getElementById('status-pos');
  if (posEl) {
    posEl.textContent = `(${state.currentPos.x}, ${state.currentPos.y})`;
  }
  
  const stepsEl = document.getElementById('status-steps');
  if (stepsEl) {
    stepsEl.textContent = String(state.stepCount);
  }
  
  const stateEl = document.getElementById('status-state');
  if (stateEl) {
    if (state.done) {
      if (state.success) {
        stateEl.textContent = 'Solved!';
        stateEl.className = 'status-value status-success';
      } else {
        stateEl.textContent = 'Failed';
        stateEl.className = 'status-value status-error';
      }
    } else {
      stateEl.textContent = 'Drawing';
      stateEl.className = 'status-value status-pending';
    }
  }
}

function updateLegendUI(level: LevelJson): void {
  const legendEl = document.getElementById('legend-content');
  if (!legendEl) return;
  
  // Collect unique colors
  const colors = new Set<string>();
  for (const row of level.cells) {
    for (const cell of row) {
      if (cell) colors.add(cell);
    }
  }
  
  // Build legend HTML
  let html = '';
  colors.forEach((color) => {
    const colorHex = getColorStyle(color);
    html += `
      <div class="status-row">
        <span class="status-label" style="display: flex; align-items: center; gap: 8px;">
          <span style="width: 12px; height: 12px; background: ${colorHex}; border-radius: 2px;"></span>
          ${color}
        </span>
      </div>
    `;
  });
  
  if (html === '') {
    html = '<div class="status-row"><span class="status-label">No colors</span></div>';
  }
  
  legendEl.innerHTML = html;
}

function getColorStyle(colorName: string): string {
  const colors: Record<string, string> = {
    'red': '#e6194b',
    'green': '#3cb44b',
    'blue': '#4363d8',
    'yellow': '#ffe119',
    'purple': '#911eb4',
    'cyan': '#42d4f4',
    'gray': '#9ca3af',
    'white': '#ffffff',
  };
  return colors[colorName.toLowerCase()] ?? '#888888';
}

function showMessage(title: string, detail: string, type: 'success' | 'error'): void {
  const overlay = document.getElementById('message-overlay');
  const titleEl = document.getElementById('message-title');
  const detailEl = document.getElementById('message-detail');
  
  if (overlay && titleEl && detailEl) {
    titleEl.textContent = title;
    detailEl.textContent = detail;
    overlay.className = `visible ${type}`;
    
    // Auto-hide after delay
    setTimeout(() => {
      overlay.classList.remove('visible');
    }, 3000);
  }
}

function hideMessage(): void {
  const overlay = document.getElementById('message-overlay');
  if (overlay) {
    overlay.classList.remove('visible');
  }
}

function flashInvalidMove(dir: Direction): void {
  const keyMap: Record<Direction, string> = {
    'U': 'key-up',
    'D': 'key-down',
    'L': 'key-left',
    'R': 'key-right',
  };
  
  const keyEl = document.getElementById(keyMap[dir]);
  if (keyEl) {
    keyEl.style.background = '#f85149';
    setTimeout(() => {
      keyEl.style.background = '';
    }, 200);
  }
}

// ============================================================================
// API Exposure for Playwright/Gym
// ============================================================================

interface TopoBenchAPI {
  loadLevel: (levelJson: LevelJson) => void;
  loadGeneratedSetup: (boardSize: number, seed: number) => void;
  loadQuestionRow: (row: unknown) => void;
  reset: () => void;
  stepDir: (dir: Direction) => StepResult;
  stepTo: (x: number, y: number) => StepResult;
  getState: () => StateResponse;
  evaluate: () => EvaluateResult;
  snapshot: () => Promise<string>;
  undo: () => boolean;
  getAvailableMoves: () => Direction[];
  getLevelCount: () => number;
  loadLevelByIndex: (index: number) => void;
  getGeneratedConfig: () => { boardSize: number; seed: number; mode: string };
}

function exposeAPI(): void {
  const api: TopoBenchAPI = {
    loadLevel,
    loadGeneratedSetup,
    loadQuestionRow,
    reset,
    stepDir,
    stepTo,
    getState,
    evaluate,
    snapshot,
    undo,
    getAvailableMoves: () => game.getAvailableMoves(),
    getLevelCount,
    loadLevelByIndex,
    getGeneratedConfig: () => ({ boardSize: currentBoardSize, seed: currentSeed, mode: currentMode }),
  };
  
  // Expose to window
  (window as unknown as { topoBench: TopoBenchAPI }).topoBench = api;
  
  console.log('TopoBench API exposed to window.topoBench');
  console.log('Available methods: loadLevel, loadGeneratedSetup, reset, stepDir, stepTo, getState, evaluate, snapshot, undo, getAvailableMoves, getLevelCount, loadLevelByIndex, getGeneratedConfig');
}

// ============================================================================
// Initialize on DOM ready
// ============================================================================

document.addEventListener('DOMContentLoaded', init);
