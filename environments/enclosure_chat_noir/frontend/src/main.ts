import { EnclosureChatNoirGame } from "./chat_noir.ts";

const boardRoot = document.getElementById("board-root");
if (!boardRoot) {
  throw new Error("Missing Chat Noir board root");
}

const stateElement = document.getElementById("state-json");
const radiusInput = document.getElementById("radius-input");
const blockedInput = document.getElementById("blocked-input");
const minStartsInput = document.getElementById("min-starts-input");
const seedInput = document.getElementById("seed-input");
const policyInput = document.getElementById("policy-input");
const actionInput = document.getElementById("action-input");
const resetButton = document.getElementById("reset-btn");
const stepButton = document.getElementById("step-btn");
const policyBadge = document.getElementById("policy-badge");
const statusBadge = document.getElementById("status-badge");

const game = new EnclosureChatNoirGame(boardRoot);

function parseIntInput(element, fallback) {
  if (!element || element.tagName !== "INPUT") {
    return fallback;
  }
  const parsed = Number.parseInt(element.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseSetupInput(element, fallback) {
  if (!element || element.tagName !== "INPUT") {
    return fallback;
  }
  const value = String(element.value || "").trim();
  if (!value || value.toLowerCase() === "auto") {
    return "auto";
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readResetConfig() {
  const seed = parseIntInput(seedInput, 1);
  return {
    boardRadius: parseSetupInput(radiusInput, "auto"),
    initialBlockedCount: parseSetupInput(blockedInput, "auto"),
    minWinningFirstActions: parseSetupInput(minStartsInput, "auto"),
    catPolicy: policyInput && policyInput.tagName === "SELECT" ? policyInput.value : "easy",
    seed,
    rngSeed: seed
  };
}

function resetConfigFromQuestionRow(row) {
  const resetConfig = row?.meta_info?.initial_state?.reset_config ?? row?.reset_config;
  if (!resetConfig || typeof resetConfig !== "object") {
    throw new Error("Question row does not contain meta_info.initial_state.reset_config.");
  }
  return resetConfig;
}

function updatePanel(lastStep = null) {
  if (stateElement) {
    stateElement.textContent = JSON.stringify(
      {
        state_response: game.getState(),
        debug_state: game.getDebugState(),
        render_config: game.renderConfig(),
        last_step: lastStep
      },
      null,
      2
    );
  }

  const state = game.getDebugState();
  if (policyBadge) {
    policyBadge.textContent = `Difficulty: ${state.catPolicy.name}`;
  }
  if (statusBadge) {
    const terminalReason = state.terminal?.reason;
    if (terminalReason === "cat_trapped") {
      statusBadge.textContent = "State: captured";
      statusBadge.style.background = "rgba(104, 166, 91, 0.18)";
      statusBadge.style.color = "#386d2c";
    } else if (terminalReason === "cat_escaped") {
      statusBadge.textContent = "State: escaped";
      statusBadge.style.background = "rgba(229, 92, 112, 0.18)";
      statusBadge.style.color = "#9b2435";
    } else {
      statusBadge.textContent = "State: active";
      statusBadge.style.background = "rgba(234, 127, 38, 0.12)";
      statusBadge.style.color = "#934d16";
    }
  }
}

const api = {
  reset(config = {}) {
    const result = game.reset(config);
    updatePanel({ type: "reset", result });
    return result;
  },
  async step(action) {
    const result = await game.step(action);
    updatePanel({ type: "step", action, result });
    return result;
  },
  getState() {
    return game.getState();
  },
  getDebugState() {
    return game.getDebugState();
  },
  isLegalMove(cellIndex) {
    return game.isLegalMove(cellIndex);
  },
  renderConfig() {
    return game.renderConfig();
  },
  loadQuestionRow(row) {
    return api.reset(resetConfigFromQuestionRow(row));
  }
};

window.topoBench = api;

if (resetButton) {
  resetButton.addEventListener("click", () => {
    api.reset(readResetConfig());
  });
}

if (stepButton) {
  stepButton.addEventListener("click", async () => {
    const actionIndex = parseIntInput(actionInput, 0);
    await api.step(actionIndex);
  });
}

api.reset(readResetConfig());
