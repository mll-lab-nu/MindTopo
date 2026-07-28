import { Swap2DPuzzleRenderer } from "./render.ts";

const BLANK_TOKEN = "_";
const MAX_GRID_ROWS = 4;
const MAX_GRID_COLS = 4;
const DIFFICULTY_GRID_SHAPES = Object.freeze({
  easy: Object.freeze([
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3]
  ]),
  medium: Object.freeze([
    [3, 4],
    [4, 3]
  ]),
  hard: Object.freeze([[4, 4]])
});

const BLOCK_LIBRARY = Object.freeze([
  { id: "G", label: "G", name: "Green", color: "#55a612" },
  { id: "R", label: "R", name: "Red", color: "#fa4045" },
  { id: "P", label: "P", name: "Purple", color: "#7f75dc" },
  { id: "B", label: "B", name: "Blue", color: "#178adb" },
  { id: "Y", label: "Y", name: "Yellow", color: "#fda31d" },
  { id: "O", label: "O", name: "Orange", color: "#f05a2a" },
  { id: "T", label: "T", name: "Teal", color: "#1fa6a2" },
  { id: "M", label: "M", name: "Magenta", color: "#c64191" },
  { id: "C", label: "C", name: "Cyan", color: "#00a6d6" },
  { id: "L", label: "L", name: "Lime", color: "#8cc63e" },
  { id: "N", label: "N", name: "Navy", color: "#2c4f8f" },
  { id: "S", label: "S", name: "Silver", color: "#9aa7b2" },
  { id: "V", label: "V", name: "Violet", color: "#9a4bd7" },
  { id: "I", label: "I", name: "Indigo", color: "#4b5bdc" },
  { id: "A", label: "A", name: "Amber", color: "#d79a00" }
]);

const BLOCK_BY_ID = new Map(BLOCK_LIBRARY.map((spec) => [spec.id, spec]));

const DEFAULT_CONFIG = Object.freeze({
  gridRows: 3,
  gridCols: 3,
  difficulty: null,
  targetDifficulty: null,
  seed: 1,
  selectedBlockIds: null,
  initialArrangement: null,
  goalArrangement: null,
  animate: true,
  illegalReward: -1.0,
  moveReward: -0.01,
  winReward: 1.0,
  budgetMultiplier: 1.2,
  stepBudget: null,
  theoreticalMinSteps: null,
  minTheoreticalSteps: 1
});

function clampInt(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function normalizeSeed(value, fallback) {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.trunc(parsed);
}

function normalizeDifficulty(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim().toLowerCase();
  if (!text || text === "custom" || text === "grid" || text === "manual" || text === "none" || text.startsWith("grid_")) {
    return null;
  }
  const aliases = new Map([
    ["easy", "easy"],
    ["e", "easy"],
    ["medium", "medium"],
    ["med", "medium"],
    ["m", "medium"],
    ["hard", "hard"],
    ["h", "hard"]
  ]);
  if (!aliases.has(text)) {
    throw new Error("Difficulty must be one of: easy, medium, hard.");
  }
  return aliases.get(text);
}

class StableRng {
  constructor(seed) {
    this.state = normalizeSeed(seed, 1) >>> 0;
  }

  nextUint32() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  }

  nextInt(stop) {
    if (!Number.isFinite(stop) || stop <= 0) {
      throw new Error("StableRng.nextInt requires a positive stop value.");
    }
    return this.nextUint32() % Math.trunc(stop);
  }
}

function normalizeToken(value) {
  if (value === null || value === undefined) {
    return BLANK_TOKEN;
  }
  const text = String(value).trim().toUpperCase();
  if (!text || text === BLANK_TOKEN) {
    return BLANK_TOKEN;
  }
  return text;
}

function normalizeShape(gridRows, gridCols) {
  return {
    gridRows: clampInt(gridRows, 2, MAX_GRID_ROWS, DEFAULT_CONFIG.gridRows),
    gridCols: clampInt(gridCols, 2, MAX_GRID_COLS, DEFAULT_CONFIG.gridCols)
  };
}

function gridShapeForDifficulty(difficulty, seed) {
  const normalized = normalizeDifficulty(difficulty);
  if (!normalized) {
    throw new Error("Difficulty must be one of: easy, medium, hard.");
  }
  const shapes = DIFFICULTY_GRID_SHAPES[normalized];
  if (shapes.length === 1) {
    return {
      gridRows: shapes[0][0],
      gridCols: shapes[0][1]
    };
  }
  const rng = new StableRng(seed);
  const [gridRows, gridCols] = shapes[rng.nextInt(shapes.length)];
  return { gridRows, gridCols };
}

function difficultyLabelForGrid(gridRows, gridCols) {
  for (const [difficulty, shapes] of Object.entries(DIFFICULTY_GRID_SHAPES)) {
    if (shapes.some(([rows, cols]) => rows === gridRows && cols === gridCols)) {
      return difficulty;
    }
  }
  return `grid_${gridRows}x${gridCols}`;
}

function flattenArrangement(rawArrangement) {
  if (!Array.isArray(rawArrangement)) {
    return null;
  }
  if (rawArrangement.length > 0 && rawArrangement.every((row) => Array.isArray(row))) {
    return rawArrangement.flat();
  }
  return rawArrangement.slice();
}

function arrangementToGrid(arrangement, gridRows, gridCols) {
  const rows = [];
  for (let row = 0; row < gridRows; row += 1) {
    rows.push(arrangement.slice(row * gridCols, (row + 1) * gridCols));
  }
  return rows;
}

function cellIndexToRowCol(cellIndex, gridCols) {
  return {
    row: Math.floor(cellIndex / gridCols),
    col: cellIndex % gridCols
  };
}

function rowColToCellIndex(row, col, gridCols) {
  return row * gridCols + col;
}

function shuffleCopy(items, rng = null) {
  const next = items.slice();
  for (let index = next.length - 1; index > 0; index -= 1) {
    const swapIndex = rng ? rng.nextInt(index + 1) : Math.floor(Math.random() * (index + 1));
    [next[index], next[swapIndex]] = [next[swapIndex], next[index]];
  }
  return next;
}

function serializeArrangement(arrangement) {
  return arrangement.join("|");
}

function chooseBlockIds(numBlocks, requestedBlockIds) {
  if (Array.isArray(requestedBlockIds) && requestedBlockIds.length > 0) {
    const unique = [];
    const seen = new Set();
    for (const rawId of requestedBlockIds) {
      const token = normalizeToken(rawId);
      if (token === BLANK_TOKEN || seen.has(token) || !BLOCK_BY_ID.has(token)) {
        continue;
      }
      unique.push(token);
      seen.add(token);
    }
    if (unique.length >= numBlocks) {
      return unique.slice(0, numBlocks);
    }
  }

  if (numBlocks > BLOCK_LIBRARY.length) {
    throw new Error(`2D swap puzzle supports at most ${BLOCK_LIBRARY.length} colored blocks.`);
  }
  return BLOCK_LIBRARY.slice(0, numBlocks).map((spec) => spec.id);
}

function normalizeArrangement(rawArrangement, selectedBlockIds, gridRows, gridCols) {
  const flattened = flattenArrangement(rawArrangement);
  if (!flattened) {
    return null;
  }

  const cellCount = gridRows * gridCols;
  if (flattened.length !== cellCount) {
    return null;
  }

  const tokens = flattened.map((value) => normalizeToken(value));
  const blankCount = tokens.filter((token) => token === BLANK_TOKEN).length;
  if (blankCount !== 1) {
    return null;
  }

  const expected = new Map(selectedBlockIds.map((blockId) => [blockId, 1]));
  for (const token of tokens) {
    if (token === BLANK_TOKEN) {
      continue;
    }
    if (!expected.has(token)) {
      return null;
    }
    expected.set(token, expected.get(token) - 1);
  }

  for (const remaining of expected.values()) {
    if (remaining !== 0) {
      return null;
    }
  }
  return tokens;
}

function shortestSwapDistance(initialArrangement, goalArrangement) {
  return shortestActionSequence(initialArrangement, goalArrangement).length;
}

function shortestActionSequence(initialArrangement, goalArrangement) {
  const startKey = serializeArrangement(initialArrangement);
  const goalKey = serializeArrangement(goalArrangement);
  if (startKey === goalKey) {
    return [];
  }
  if (initialArrangement.length !== goalArrangement.length) {
    throw new Error("Initial and goal arrangements must have the same length.");
  }
  const startTokens = new Set(initialArrangement);
  const goalTokens = new Set(goalArrangement);
  if (
    startTokens.size !== initialArrangement.length ||
    goalTokens.size !== goalArrangement.length ||
    startTokens.size !== goalTokens.size ||
    !Array.from(startTokens).every((token) => goalTokens.has(token)) ||
    initialArrangement.filter((token) => token === BLANK_TOKEN).length !== 1 ||
    goalArrangement.filter((token) => token === BLANK_TOKEN).length !== 1
  ) {
    throw new Error("Initial and goal arrangements must contain the same unique tokens and one blank.");
  }

  const current = initialArrangement.slice();
  const target = goalArrangement.slice();
  const actions = [];
  const guardLimit = current.length * 3;

  while (serializeArrangement(current) !== goalKey) {
    if (actions.length > guardLimit) {
      throw new Error("Unable to reconstruct a shortest 2D swap puzzle action sequence.");
    }
    const blankIndex = current.indexOf(BLANK_TOKEN);
    const desiredAtBlank = target[blankIndex];
    let actionIndex = -1;
    if (desiredAtBlank !== BLANK_TOKEN) {
      actionIndex = current.indexOf(desiredAtBlank);
    } else {
      actionIndex = current.findIndex((token, index) => token !== target[index] && token !== BLANK_TOKEN);
    }
    [current[blankIndex], current[actionIndex]] = [current[actionIndex], current[blankIndex]];
    actions.push(actionIndex);
  }

  return actions;
}

function samplePuzzle(selectedBlockIds, minTheoreticalSteps, seed) {
  const rng = seed === null || seed === undefined ? null : new StableRng(seed);
  const goalArrangement = shuffleCopy(selectedBlockIds, rng);
  goalArrangement.push(BLANK_TOKEN);
  const cellTokens = goalArrangement.slice();

  for (let attempt = 0; attempt < 512; attempt += 1) {
    const initialArrangement = shuffleCopy(cellTokens, rng);
    const theoreticalMinSteps = shortestSwapDistance(initialArrangement, goalArrangement);
    if (theoreticalMinSteps >= minTheoreticalSteps && theoreticalMinSteps > 0) {
      return {
        initialArrangement,
        goalArrangement,
        theoreticalMinSteps
      };
    }
  }

  throw new Error("Unable to sample a 2D swap puzzle with the requested minimum difficulty.");
}

export class OrderSwap2DPuzzleGame {
  constructor(rendererRoots) {
    this.renderer = rendererRoots ? new Swap2DPuzzleRenderer(rendererRoots) : null;
    this.config = { ...DEFAULT_CONFIG };
    this.currentArrangement = [];
    this.initialArrangement = [];
    this.goalArrangement = [];
    this.history = [];
    this.stepCount = 0;
    this.lastMove = null;
    this.reset({});
  }

  reset(config = {}) {
    this.config = this._normalizeConfig(config);
    this.currentArrangement = this.config.initialArrangement.slice();
    this.initialArrangement = this.config.initialArrangement.slice();
    this.goalArrangement = this.config.goalArrangement.slice();
    this.history = [];
    this.stepCount = 0;
    this.lastMove = null;

    if (this.renderer) {
      this.renderer.setAnimationEnabled(this.config.animate);
      this.renderer.setup({
        ...this.config,
        blockById: BLOCK_BY_ID
      });
      this.renderer.syncState(this.getDebugState());
    }

    const solved = this._isSolved();
    return this._buildStateResponse({
      reward: 0,
      done: solved,
      success: solved,
      info: {
        reset: true,
        illegal: false,
        reason: null
      }
    });
  }

  getObservation() {
    const blankCellIndex = this.currentArrangement.indexOf(BLANK_TOKEN);
    const goalBlankCellIndex = this.goalArrangement.indexOf(BLANK_TOKEN);
    const blank = cellIndexToRowCol(blankCellIndex, this.config.gridCols);
    const goalBlank = cellIndexToRowCol(goalBlankCellIndex, this.config.gridCols);
    return {
      currentArrangement: this.currentArrangement.slice(),
      goalArrangement: this.goalArrangement.slice(),
      difficulty: this.config.difficulty,
      currentGrid: arrangementToGrid(this.currentArrangement, this.config.gridRows, this.config.gridCols),
      goalGrid: arrangementToGrid(this.goalArrangement, this.config.gridRows, this.config.gridCols),
      blankCellIndex,
      blankRow: blank.row,
      blankCol: blank.col,
      goalBlankCellIndex,
      goalBlankRow: goalBlank.row,
      goalBlankCol: goalBlank.col
    };
  }

  getDebugState() {
    const blankCellIndex = this.currentArrangement.indexOf(BLANK_TOKEN);
    const goalBlankCellIndex = this.goalArrangement.indexOf(BLANK_TOKEN);
    const blank = cellIndexToRowCol(blankCellIndex, this.config.gridCols);
    const goalBlank = cellIndexToRowCol(goalBlankCellIndex, this.config.gridCols);
    return {
      gridRows: this.config.gridRows,
      gridCols: this.config.gridCols,
      numBlocks: this.config.selectedBlockIds.length,
      cellCount: this.currentArrangement.length,
      difficulty: this.config.difficulty,
      targetDifficulty: this.config.targetDifficulty,
      seed: this.config.seed,
      selectedBlockIds: this.config.selectedBlockIds.slice(),
      blockPalette: this.config.selectedBlockIds.map((blockId) => ({ ...BLOCK_BY_ID.get(blockId) })),
      initialArrangement: this.initialArrangement.slice(),
      initialGrid: arrangementToGrid(this.initialArrangement, this.config.gridRows, this.config.gridCols),
      currentArrangement: this.currentArrangement.slice(),
      currentGrid: arrangementToGrid(this.currentArrangement, this.config.gridRows, this.config.gridCols),
      goalArrangement: this.goalArrangement.slice(),
      goalGrid: arrangementToGrid(this.goalArrangement, this.config.gridRows, this.config.gridCols),
      blankCellIndex,
      blankRow: blank.row,
      blankCol: blank.col,
      goalBlankCellIndex,
      goalBlankRow: goalBlank.row,
      goalBlankCol: goalBlank.col,
      legalActions: this._legalActions(),
      theoreticalMinSteps: this.config.theoreticalMinSteps,
      stepBudget: this.config.stepBudget,
      stepCount: this.stepCount,
      historyLength: this.history.length,
      canUndo: this.history.length > 0,
      status: this._status(),
      lastMove: this.lastMove ? { ...this.lastMove } : null
    };
  }

  getState() {
    const solved = this._isSolved();
    return this._buildStateResponse({
      reward: 0,
      done: solved,
      success: solved,
      info: {
        illegal: false,
        reason: null,
        snapshot: true
      }
    });
  }

  renderConfig() {
    return {
      ...this.config,
      cellCount: this.currentArrangement.length,
      actionCount: this.currentArrangement.length,
      actionMap: this.currentArrangement.map((_value, cellIndex) => ({
        cellIndex,
        cell_index: cellIndex,
        ...cellIndexToRowCol(cellIndex, this.config.gridCols)
      })),
      blockPalette: this.config.selectedBlockIds.map((blockId) => ({ ...BLOCK_BY_ID.get(blockId) }))
    };
  }

  isLegalMove(action) {
    return this._validateMove(this.decodeAction(action)).legal;
  }

  decodeAction(action) {
    if (Number.isInteger(action)) {
      return action;
    }

    if (action && typeof action === "object") {
      const cellCandidates = [action.cell_index, action.cellIndex, action.index, action.slot_index, action.slotIndex, action.action];
      for (const candidate of cellCandidates) {
        if (Number.isInteger(candidate)) {
          return candidate;
        }
        const parsed = Number.parseInt(String(candidate ?? ""), 10);
        if (Number.isFinite(parsed)) {
          return parsed;
        }
      }

      const row = Number.parseInt(String(action.row ?? action.r ?? ""), 10);
      const col = Number.parseInt(String(action.col ?? action.column ?? action.c ?? ""), 10);
      if (Number.isFinite(row) && Number.isFinite(col)) {
        return rowColToCellIndex(row, col, this.config.gridCols);
      }
    }

    return null;
  }

  async step(action) {
    const cellIndex = this.decodeAction(action);
    if (!Number.isInteger(cellIndex)) {
      return this._illegalResult({
        reason: "invalid_action",
        action
      });
    }

    const validation = this._validateMove(cellIndex);
    if (!validation.legal) {
      return this._illegalResult({
        reason: validation.reason,
        action,
        cellIndex,
        ...cellIndexToRowCol(Math.max(0, cellIndex), this.config.gridCols)
      });
    }

    const blankIndexBefore = this.currentArrangement.indexOf(BLANK_TOKEN);
    const movedBlockId = this.currentArrangement[cellIndex];
    this.history.push({
      currentArrangement: this.currentArrangement.slice(),
      stepCount: this.stepCount,
      lastMove: this.lastMove ? { ...this.lastMove } : null
    });
    [this.currentArrangement[blankIndexBefore], this.currentArrangement[cellIndex]] = [
      this.currentArrangement[cellIndex],
      this.currentArrangement[blankIndexBefore]
    ];
    this.stepCount += 1;
    this.lastMove = {
      cellIndex,
      ...cellIndexToRowCol(cellIndex, this.config.gridCols),
      blankIndexBefore,
      blankIndexAfter: cellIndex,
      movedBlockId,
      movedBlockLabel: BLOCK_BY_ID.get(movedBlockId)?.label ?? movedBlockId,
      stepCount: this.stepCount
    };

    if (this.renderer) {
      await this.renderer.animateSwap({
        fromCell: cellIndex,
        toCell: blankIndexBefore,
        blockId: movedBlockId
      });
      this.renderer.syncState(this.getDebugState());
    }

    const solved = this._isSolved();
    return this._buildStateResponse({
      reward: solved ? this.config.winReward : this.config.moveReward,
      done: solved,
      success: solved,
      info: {
        illegal: false,
        reason: null,
        cellIndex,
        ...cellIndexToRowCol(cellIndex, this.config.gridCols),
        blankIndexBefore,
        blankIndexAfter: cellIndex,
        movedBlockId,
        solved
      }
    });
  }

  undo() {
    const snapshot = this.history.pop();
    if (!snapshot) {
      const solved = this._isSolved();
      return this._buildStateResponse({
        reward: 0,
        done: solved,
        success: solved,
        info: {
          undo: false,
          illegal: false,
          reason: "no_history"
        }
      });
    }

    this.currentArrangement = snapshot.currentArrangement.slice();
    this.stepCount = snapshot.stepCount;
    this.lastMove = snapshot.lastMove ? { ...snapshot.lastMove } : null;

    if (this.renderer) {
      this.renderer.syncState(this.getDebugState());
    }

    const solved = this._isSolved();
    return this._buildStateResponse({
      reward: 0,
      done: solved,
      success: solved,
      info: {
        undo: true,
        illegal: false,
        reason: null
      }
    });
  }

  _buildStateResponse({ reward, done, success, info }) {
    return {
      observation: this.getObservation(),
      reward,
      done,
      success,
      step_count: this.stepCount,
      info: {
        ...(info || {}),
        state: this.getDebugState()
      }
    };
  }

  _illegalResult(payload) {
    return this._buildStateResponse({
      reward: this.config.illegalReward,
      done: false,
      success: false,
      info: {
        illegal: true,
        blankCellIndex: this.currentArrangement.indexOf(BLANK_TOKEN),
        legalActions: this._legalActions(),
        ...payload
      }
    });
  }

  _validateMove(cellIndex) {
    if (!Number.isInteger(cellIndex)) {
      return { legal: false, reason: "non_integer_action" };
    }
    if (cellIndex < 0 || cellIndex >= this.currentArrangement.length) {
      return { legal: false, reason: "out_of_range" };
    }
    if (this.currentArrangement[cellIndex] === BLANK_TOKEN) {
      return { legal: false, reason: "selected_blank" };
    }
    return { legal: true, reason: null };
  }

  _normalizeConfig(incoming) {
    const explicitConfig = incoming || {};
    const merged = { ...DEFAULT_CONFIG, ...this.config, ...explicitConfig };
    const seed = normalizeSeed(merged.seed, DEFAULT_CONFIG.seed);
    const hasExplicitGrid = (
      Object.prototype.hasOwnProperty.call(explicitConfig, "gridRows") ||
      Object.prototype.hasOwnProperty.call(explicitConfig, "rows") ||
      Object.prototype.hasOwnProperty.call(explicitConfig, "gridCols") ||
      Object.prototype.hasOwnProperty.call(explicitConfig, "cols")
    );
    let requestedDifficulty = null;
    if (Object.prototype.hasOwnProperty.call(explicitConfig, "targetDifficulty")) {
      requestedDifficulty = normalizeDifficulty(explicitConfig.targetDifficulty);
    } else if (Object.prototype.hasOwnProperty.call(explicitConfig, "difficulty") && !hasExplicitGrid) {
      requestedDifficulty = normalizeDifficulty(explicitConfig.difficulty);
    }
    const shape = requestedDifficulty
      ? gridShapeForDifficulty(requestedDifficulty, seed)
      : normalizeShape(
        explicitConfig.gridRows ?? explicitConfig.rows ?? merged.gridRows ?? merged.rows,
        explicitConfig.gridCols ?? explicitConfig.cols ?? merged.gridCols ?? merged.cols
      );
    const gridRows = shape.gridRows;
    const gridCols = shape.gridCols;
    const numBlocks = gridRows * gridCols - 1;
    const selectedBlockIds = chooseBlockIds(numBlocks, merged.selectedBlockIds);
    const hasExplicitArrangements = (
      Object.prototype.hasOwnProperty.call(explicitConfig, "initialArrangement") &&
      Object.prototype.hasOwnProperty.call(explicitConfig, "goalArrangement")
    );

    let initialArrangement = hasExplicitArrangements
      ? normalizeArrangement(explicitConfig.initialArrangement, selectedBlockIds, gridRows, gridCols)
      : null;
    let goalArrangement = hasExplicitArrangements
      ? normalizeArrangement(explicitConfig.goalArrangement, selectedBlockIds, gridRows, gridCols)
      : null;
    let theoreticalMinSteps = null;

    if (!initialArrangement || !goalArrangement) {
      const sampled = samplePuzzle(
        selectedBlockIds,
        Math.max(1, clampInt(merged.minTheoreticalSteps, 1, 100, DEFAULT_CONFIG.minTheoreticalSteps)),
        seed
      );
      initialArrangement = sampled.initialArrangement;
      goalArrangement = sampled.goalArrangement;
      theoreticalMinSteps = sampled.theoreticalMinSteps;
    }

    theoreticalMinSteps = shortestSwapDistance(initialArrangement, goalArrangement);

    const budgetMultiplier = Math.max(1.0, Number(merged.budgetMultiplier) || DEFAULT_CONFIG.budgetMultiplier);
    const stepBudget = clampInt(
      Object.prototype.hasOwnProperty.call(explicitConfig, "stepBudget")
        ? explicitConfig.stepBudget
        : undefined,
      1,
      1000,
      Math.max(1, Math.ceil(theoreticalMinSteps * budgetMultiplier))
    );

    return {
      gridRows,
      gridCols,
      difficulty: requestedDifficulty || difficultyLabelForGrid(gridRows, gridCols),
      targetDifficulty: requestedDifficulty,
      seed,
      selectedBlockIds,
      initialArrangement,
      goalArrangement,
      animate: Boolean(merged.animate),
      illegalReward: Number(merged.illegalReward),
      moveReward: Number(merged.moveReward),
      winReward: Number(merged.winReward),
      budgetMultiplier,
      stepBudget,
      theoreticalMinSteps
    };
  }

  _legalActions() {
    const legal = [];
    for (let cellIndex = 0; cellIndex < this.currentArrangement.length; cellIndex += 1) {
      if (this.currentArrangement[cellIndex] !== BLANK_TOKEN) {
        legal.push({
          cell_index: cellIndex,
          cellIndex,
          ...cellIndexToRowCol(cellIndex, this.config.gridCols)
        });
      }
    }
    return legal;
  }

  _isSolved() {
    return serializeArrangement(this.currentArrangement) === serializeArrangement(this.goalArrangement);
  }

  _status() {
    if (this._isSolved()) {
      return "success";
    }
    if (this.stepCount >= this.config.stepBudget) {
      return "step_limit";
    }
    return "in_progress";
  }
}
