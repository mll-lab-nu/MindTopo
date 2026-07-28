import { OrderSwap2DPuzzleGame } from "./swap_2d_puzzle.ts";

const currentRoot = document.getElementById("current-board");
const goalRoot = document.getElementById("goal-board");
if (!currentRoot || !goalRoot) {
  throw new Error("Missing 2D swap puzzle board roots");
}

const stateElement = document.getElementById("state-json");
const difficultyInput = document.getElementById("difficulty-input");
const rowInput = document.getElementById("row-input");
const colInput = document.getElementById("col-input");
const seedInput = document.getElementById("seed-input");
const actionRowInput = document.getElementById("action-row-input");
const actionColInput = document.getElementById("action-col-input");
const resetButton = document.getElementById("reset-btn");
const backButton = document.getElementById("back-btn");
const stepButton = document.getElementById("step-btn");
const currentStateLabel = document.getElementById("current-state-label");
const stepStatusLabel = document.getElementById("step-status-label");
const statusPill = document.getElementById("status-pill");

const game = new OrderSwap2DPuzzleGame({ currentRoot, goalRoot });
let clickStepPending = false;

function parseIntInput(element, fallback) {
  if (!element || element.tagName !== "INPUT") {
    return fallback;
  }
  const parsed = Number.parseInt(element.value, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return parsed;
}

function normalizeDifficultyValue(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim().toLowerCase();
  return ["easy", "medium", "hard"].includes(text) ? text : null;
}

function parseDifficultyInput(element) {
  if (!element || element.tagName !== "SELECT") {
    return null;
  }
  return normalizeDifficultyValue(element.value);
}

function statusText(status) {
  if (status === "success") {
    return "Success";
  }
  if (status === "step_limit") {
    return "Step Limit";
  }
  return "In Progress";
}

function updatePanel(lastStep = null) {
  const debugState = game.getDebugState();
  const status = debugState.status ?? "in_progress";
  if (currentStateLabel) {
    currentStateLabel.textContent = `State ${debugState.stepCount ?? 0}`;
  }
  if (stepStatusLabel) {
    stepStatusLabel.textContent = "Status";
  }
  if (statusPill) {
    statusPill.textContent = statusText(status);
    statusPill.dataset.status = status;
  }
  if (backButton instanceof HTMLButtonElement) {
    backButton.disabled = !debugState.canUndo;
  }

  if (!stateElement) {
    return;
  }

  const payload = {
    state_response: game.getState(),
    debug_state: debugState,
    render_config: game.renderConfig(),
    last_step: lastStep
  };
  stateElement.textContent = JSON.stringify(payload, null, 2);
}

function resetConfigFromQuestionRow(row) {
  const resetConfig = row?.meta_info?.initial_state?.reset_config ?? row?.reset_config;
  if (!resetConfig || typeof resetConfig !== "object") {
    throw new Error("Question row does not contain meta_info.initial_state.reset_config.");
  }
  return resetConfig;
}

function buildManualResetConfig() {
  const targetDifficulty = parseDifficultyInput(difficultyInput);
  const config = {
    gridRows: parseIntInput(rowInput, 3),
    gridCols: parseIntInput(colInput, 3),
    seed: parseIntInput(seedInput, 1)
  };
  if (targetDifficulty) {
    config.targetDifficulty = targetDifficulty;
  }
  return config;
}

function syncInputsFromResetConfig(resetConfig) {
  if (difficultyInput && difficultyInput.tagName === "SELECT") {
    const difficulty = normalizeDifficultyValue(resetConfig.targetDifficulty ?? resetConfig.difficulty);
    difficultyInput.value = difficulty ?? "";
  }
  if (rowInput && rowInput.tagName === "INPUT" && resetConfig.gridRows !== undefined) {
    rowInput.value = String(resetConfig.gridRows);
  }
  if (colInput && colInput.tagName === "INPUT" && resetConfig.gridCols !== undefined) {
    colInput.value = String(resetConfig.gridCols);
  }
  if (seedInput && seedInput.tagName === "INPUT" && resetConfig.seed !== undefined) {
    seedInput.value = String(resetConfig.seed);
  }
}

const api = {
  reset(config = {}) {
    syncInputsFromResetConfig(config);
    const result = game.reset(config);
    syncInputsFromResetConfig(game.renderConfig());
    updatePanel({ type: "reset", result });
    return result;
  },
  async step(action) {
    const result = await game.step(action);
    updatePanel({ type: "step", result });
    return result;
  },
  undo() {
    const result = game.undo();
    updatePanel({ type: "undo", result });
    return result;
  },
  getState() {
    return game.getState();
  },
  getDebugState() {
    return game.getDebugState();
  },
  isLegalMove(action) {
    return game.isLegalMove(action);
  },
  renderConfig() {
    return game.renderConfig();
  },
  loadQuestionRow(row) {
    const resetConfig = resetConfigFromQuestionRow(row);
    syncInputsFromResetConfig(resetConfig);
    return api.reset(resetConfig);
  }
};

window.topoBench = api;

if (resetButton) {
  resetButton.addEventListener("click", () => {
    api.reset(buildManualResetConfig());
  });
}

if (backButton instanceof HTMLButtonElement) {
  backButton.addEventListener("click", () => {
    api.undo();
  });
}

if (stepButton) {
  stepButton.addEventListener("click", async () => {
    const row = parseIntInput(actionRowInput, 0);
    const col = parseIntInput(actionColInput, 0);
    const result = await api.step({ row, col });
    updatePanel({ type: "step", row, col, result });
  });
}

currentRoot.addEventListener("click", async (event) => {
  if (clickStepPending) {
    return;
  }
  const clickedCell = event.target instanceof Element
    ? event.target.closest("[data-cell-index]")
    : null;
  if (!(clickedCell instanceof HTMLElement) || !currentRoot.contains(clickedCell)) {
    return;
  }

  const cellIndex = Number.parseInt(clickedCell.dataset.cellIndex ?? "", 10);
  if (!Number.isFinite(cellIndex)) {
    return;
  }
  const debugState = game.getDebugState();
  if (debugState.currentArrangement?.[cellIndex] === "_") {
    return;
  }

  const row = Math.floor(cellIndex / debugState.gridCols);
  const col = cellIndex % debugState.gridCols;
  if (actionRowInput && actionRowInput.tagName === "INPUT") {
    actionRowInput.value = String(row);
  }
  if (actionColInput && actionColInput.tagName === "INPUT") {
    actionColInput.value = String(col);
  }

  clickStepPending = true;
  try {
    await api.step(cellIndex);
  } finally {
    clickStepPending = false;
  }
});

api.reset(buildManualResetConfig());
