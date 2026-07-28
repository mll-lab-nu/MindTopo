import { UntangleGame } from "./untangle.ts";
import { UntangleRenderer } from "./render.ts";

const container = document.getElementById("scene-container");
if (!container) {
  throw new Error("Missing #scene-container");
}

const stateElement = document.getElementById("state-json");
const difficultyInput = document.getElementById("difficulty-input");
const seedInput = document.getElementById("seed-input");
const actionInput = document.getElementById("action-input");
const resetButton = document.getElementById("reset-btn");
const stepButton = document.getElementById("step-btn");

const game = new UntangleGame();
const renderer = new UntangleRenderer(container);

function parseIntInput(element, fallback) {
  if (!element || element.tagName !== "INPUT") return fallback;
  const parsed = Number.parseInt(element.value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed;
}

function parseDifficulty() {
  if (!difficultyInput) return "medium";
  const value = difficultyInput.value;
  return typeof value === "string" ? value : "medium";
}

function updatePanel(lastStep = null) {
  if (!stateElement) return;
  const payload = {
    state_response: game.getState(),
    debug_state: game.getDebugState(),
    render_config: game.renderConfig(),
    last_step: lastStep,
  };
  stateElement.textContent = JSON.stringify(payload, null, 2);
}

const api = {
  reset(config = {}) {
    const result = game.reset(config);
    renderer.setAnimationEnabled(game.config.animate);
    renderer.setup(game);
    updatePanel({ type: "reset", result });
    return result;
  },
  step(action) {
    const result = game.step(action);
    renderer.syncState();
    updatePanel({ type: "step", result });
    return result;
  },
  getState() {
    return game.getState();
  },
  getDebugState() {
    return game.getDebugState();
  },
  isLegalMove(ropeIdx, targetHoleIdx) {
    return game.isLegalMove(ropeIdx, targetHoleIdx);
  },
  renderConfig() {
    return game.renderConfig();
  },
};

window.topoBench = api;

renderer.setup(game);

let dragInfo = null;

function endpointParticle(rope, isStartEnd) {
  return isStartEnd ? rope.particles[0] : rope.particles[rope.particles.length - 1];
}

function endpointVisualIndex(rope, isStartEnd) {
  return isStartEnd ? 0 : rope.visualPositions.length - 1;
}

function sourceHoleForEndpoint(rope, isStartEnd) {
  return isStartEnd ? rope.startHole : rope.endHole;
}

function resetDraggedEndpoint() {
  if (!dragInfo) return;
  const { rope, isStartEnd, sourceHole } = dragInfo;
  const particle = endpointParticle(rope, isStartEnd);
  const sourcePos = game._holeSurfacePosition(sourceHole);
  particle.pos.copy(sourcePos);
  particle.oldPos.copy(sourcePos);
  rope.visualPositions[endpointVisualIndex(rope, isStartEnd)].copy(sourcePos);
}

function moveDraggedEndpoint(point) {
  if (!dragInfo || !point) return;
  const { rope, isStartEnd } = dragInfo;
  const particle = endpointParticle(rope, isStartEnd);
  particle.pos.copy(point);
  particle.oldPos.copy(point);
  rope.visualPositions[endpointVisualIndex(rope, isStartEnd)].copy(point);
  renderer.syncState();
}

function nearestDropHole(point, sourceHole) {
  const snapDistance = Number.isFinite(game.config.snapDistance)
    ? game.config.snapDistance
    : game.config.holeSpacing * 0.48;
  let best = sourceHole;
  let bestDistance = snapDistance;
  for (const hole of game.holes) {
    if (hole.occupied && hole.id !== sourceHole.id) continue;
    const dx = hole.pos.x - point.x;
    const dz = hole.pos.z - point.z;
    const distance = Math.sqrt(dx * dx + dz * dz);
    if (distance < bestDistance) {
      best = hole;
      bestDistance = distance;
    }
  }
  return best;
}

const canvas = renderer.renderer.domElement;
canvas.addEventListener("pointerdown", (event) => {
  const endpoint = renderer.pickEndpoint(event.clientX, event.clientY);
  if (!endpoint) return;

  const rope = game.ropes[endpoint.ropeIdx];
  if (!rope) return;

  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  dragInfo = {
    pointerId: event.pointerId,
    rope,
    ropeIdx: endpoint.ropeIdx,
    isStartEnd: endpoint.isStartEnd,
    sourceHole: sourceHoleForEndpoint(rope, endpoint.isStartEnd),
    liftY: game._liftPlaneY(),
  };
  dragInfo.sourceHole.occupied = false;
  game.setDragState(rope, true);
  canvas.style.cursor = "grabbing";

  const point = renderer.projectPointerToPlane(event.clientX, event.clientY, dragInfo.liftY);
  if (point) moveDraggedEndpoint(point);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragInfo) {
    const endpoint = renderer.pickEndpoint(event.clientX, event.clientY);
    canvas.style.cursor = endpoint ? "grab" : "default";
    return;
  }
  if (event.pointerId !== dragInfo.pointerId) return;

  event.preventDefault();
  const point = renderer.projectPointerToPlane(event.clientX, event.clientY, dragInfo.liftY);
  if (point) moveDraggedEndpoint(point);
});

function finishDrag(event) {
  if (!dragInfo || event.pointerId !== dragInfo.pointerId) return;

  event.preventDefault();
  const point = renderer.projectPointerToPlane(event.clientX, event.clientY, dragInfo.liftY);
  const targetHole = point ? nearestDropHole(point, dragInfo.sourceHole) : dragInfo.sourceHole;
  const sourceHole = dragInfo.sourceHole;
  const rope = dragInfo.rope;

  resetDraggedEndpoint();
  sourceHole.occupied = true;
  game.setDragState(rope, false);
  if (!canvas.hasPointerCapture || canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  canvas.style.cursor = "default";

  if (targetHole.id !== sourceHole.id) {
    api.step({
      src_row: sourceHole.row,
      src_col: sourceHole.col,
      tgt_row: targetHole.row,
      tgt_col: targetHole.col,
    });
  } else {
    renderer.syncState();
    updatePanel({ type: "drag_cancel" });
  }
  dragInfo = null;
}

canvas.addEventListener("pointerup", finishDrag);
canvas.addEventListener("pointercancel", finishDrag);

if (resetButton) {
  resetButton.addEventListener("click", () => {
    api.reset({
      difficulty: parseDifficulty(),
      seed: parseIntInput(seedInput, 0),
    });
  });
}

if (stepButton) {
  stepButton.addEventListener("click", () => {
    const actionIndex = parseIntInput(actionInput, 0);
    api.step(actionIndex);
  });
}

api.reset({
  difficulty: parseDifficulty(),
  seed: parseIntInput(seedInput, 0),
});
