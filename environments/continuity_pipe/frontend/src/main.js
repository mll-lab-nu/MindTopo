import { ContinuityPipeGame } from "./pipe.js";

const boardRoot = document.getElementById("board-root");
if (!boardRoot) {
  throw new Error("Missing Continuity Pipe board root");
}

const stateElement = document.getElementById("state-json");
const sizeInput = document.getElementById("size-input");
const seedInput = document.getElementById("seed-input");
const difficultyInput = document.getElementById("difficulty-input");
const resetButton = document.getElementById("reset-btn");
const undoButton = document.getElementById("undo-btn");
const statusBadge = document.getElementById("status-badge");
const sourceBadge = document.getElementById("source-badge");

const game = new ContinuityPipeGame(boardRoot, updatePanel);

function parseIntInput(element, fallback) {
  if (!element || element.tagName !== "INPUT") {
    return fallback;
  }
  const parsed = Number.parseInt(element.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseDifficultyInput(element, fallback) {
  if (!(element instanceof HTMLSelectElement)) {
    return fallback;
  }
  return ["easy", "medium", "hard"].includes(element.value) ? element.value : fallback;
}

function readResetConfig() {
  return {
    gridSize: parseIntInput(sizeInput, 3),
    seed: parseIntInput(seedInput, 1),
    difficulty: parseDifficultyInput(difficultyInput, "easy")
  };
}

function syncInputsFromResetConfig(resetConfig) {
  if (sizeInput && sizeInput.tagName === "INPUT" && resetConfig.gridSize !== undefined) {
    sizeInput.value = String(resetConfig.gridSize);
  }
  if (seedInput && seedInput.tagName === "INPUT" && resetConfig.seed !== undefined) {
    seedInput.value = String(resetConfig.seed);
  }
  if (difficultyInput instanceof HTMLSelectElement && resetConfig.difficulty !== undefined) {
    difficultyInput.value = String(resetConfig.difficulty);
  }
}

function resetConfigFromQuestionRow(row) {
  const resetConfig = row?.meta_info?.initial_state?.reset_config ?? row?.reset_config;
  if (!resetConfig || typeof resetConfig !== "object") {
    throw new Error("Question row does not contain meta_info.initial_state.reset_config.");
  }
  return resetConfig;
}

function updatePanel(lastStep = null) {
  const state = game.getDebugState();
  if (stateElement) {
    const lastStepSummary = lastStep?.result
      ? {
          type: lastStep.type,
          x: lastStep.result.info?.x ?? null,
          y: lastStep.result.info?.y ?? null,
          reason: lastStep.result.info?.reason ?? null,
          done: Boolean(lastStep.result.done),
          success: Boolean(lastStep.result.success),
          reward: lastStep.result.reward
        }
      : lastStep;
    const visibleState = {
      gridSize: state.gridSize,
      seed: state.seed,
      source: state.source,
      connected: `${state.connectedCount}/${state.totalPipes}`,
      difficulty: state.difficulty,
      junctionCount: state.junctionCount,
      solutionTicks: state.solutionTicks,
      solutionSteps: state.solutionSteps ?? state.solutionTicks,
      stepCount: state.stepCount,
      maxSteps: state.maxSteps,
      canUndo: Boolean(state.canUndo),
      terminal: state.terminal,
      lastMove: state.lastTurn
        ? {
            x: state.lastTurn.x,
            y: state.lastTurn.y,
            beforeRotation: state.lastTurn.beforeRotation,
            afterRotation: state.lastTurn.afterRotation
          }
        : null
    };
    stateElement.textContent = JSON.stringify(
      {
        state: visibleState,
        last_step: lastStepSummary
      },
      null,
      2
    );
  }

  if (statusBadge) {
    const connected = `${state.connectedCount}/${state.totalPipes}`;
    if (state.terminal?.reason === "all_pipes_connected") {
      statusBadge.textContent = `State: solved (${connected})`;
      statusBadge.style.background = "rgba(25, 217, 85, 0.16)";
      statusBadge.style.color = "#0f8b39";
    } else if (state.terminal?.reason === "step_budget_exhausted") {
      statusBadge.textContent = `State: failed (${connected})`;
      statusBadge.style.background = "rgba(255, 82, 113, 0.16)";
      statusBadge.style.color = "#b32846";
    } else {
      statusBadge.textContent = `State: active (${connected})`;
      statusBadge.style.background = "rgba(10, 162, 255, 0.14)";
      statusBadge.style.color = "#12689f";
    }
  }

  if (sourceBadge) {
    sourceBadge.textContent = `Source: x=${state.source.x}, y=${state.source.y}`;
  }

  if (undoButton) {
    undoButton.disabled = !state.canUndo;
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
  step(action) {
    const result = game.step(action);
    updatePanel({ type: "step", action, result });
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
    api.reset(readResetConfig());
  });
}

if (undoButton) {
  undoButton.addEventListener("click", () => {
    api.undo();
  });
}

api.reset(readResetConfig());
