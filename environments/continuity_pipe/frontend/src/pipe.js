const DIRS = Object.freeze([
  { name: "N", dx: 0, dy: -1, bit: 1 },
  { name: "E", dx: 1, dy: 0, bit: 2 },
  { name: "S", dx: 0, dy: 1, bit: 4 },
  { name: "W", dx: -1, dy: 0, bit: 8 }
]);

const OPPOSITE_BITS = Object.freeze({
  1: 4,
  2: 8,
  4: 1,
  8: 2
});

const DEFAULT_CONFIG = Object.freeze({
  gridSize: 3,
  seed: 1,
  sourceIndex: 0,
  activePipeCount: null,
  difficulty: null,
  solvedMasks: null,
  rotations: null,
  maxSteps: null,
  illegalReward: -1.0,
  moveReward: -0.01,
  successReward: 1.0,
  timeoutPenalty: -1.0
});

const MIN_GRID_SIZE = 3;
const MAX_GRID_SIZE = 6;
const MIN_BENCHMARK_SOLUTION_TICKS = 13;
const EASY_MAX_SOLUTION_TICKS = 17;
const MEDIUM_MAX_SOLUTION_TICKS = 23;
const DIFFICULTY_SETUPS = Object.freeze({
  easy: Object.freeze({
    gridSize: [4, 4],
    activePipeCount: [10, 13],
    junctionCount: [2, 3],
    solutionTicks: [13, 17]
  }),
  medium: Object.freeze({
    gridSize: [5, 5],
    activePipeCount: [13, 17],
    junctionCount: [3, 5],
    solutionTicks: [17, 23]
  }),
  hard: Object.freeze({
    gridSize: [5, 5],
    activePipeCount: [17, 23],
    junctionCount: [5, 7],
    solutionTicks: [23, 27]
  })
});

function clampInt(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function normalizeSeed(value, fallback = DEFAULT_CONFIG.seed) {
  const parsed = Number(value);
  if (Number.isFinite(parsed)) {
    return Math.trunc(parsed);
  }
  return fallback;
}

function pythonSeedKey(seed) {
  let value = Math.abs(normalizeSeed(seed));
  if (value === 0) {
    return [0];
  }
  const key = [];
  while (value > 0) {
    key.push(value >>> 0);
    value = Math.floor(value / 4294967296);
  }
  return key;
}

class PythonRandom {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.mti = 625;
    this.seed(seed);
  }

  seed(seed) {
    this._initByArray(pythonSeedKey(seed));
  }

  _initGenrand(seed) {
    this.mt[0] = seed >>> 0;
    for (this.mti = 1; this.mti < 624; this.mti += 1) {
      const previous = this.mt[this.mti - 1] ^ (this.mt[this.mti - 1] >>> 30);
      this.mt[this.mti] = (Math.imul(1812433253, previous) + this.mti) >>> 0;
    }
  }

  _initByArray(initKey) {
    this._initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(624, initKey.length);
    for (; k > 0; k -= 1) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1664525)) + initKey[j] + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
      if (j >= initKey.length) {
        j = 0;
      }
    }
    for (k = 623; k > 0; k -= 1) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  _genrandUInt32() {
    const mag01 = [0, 0x9908b0df];
    let y = 0;
    if (this.mti >= 624) {
      let kk = 0;
      for (; kk < 227; kk += 1) {
        y = (this.mt[kk] & 0x80000000) | (this.mt[kk + 1] & 0x7fffffff);
        this.mt[kk] = (this.mt[kk + 397] ^ (y >>> 1) ^ mag01[y & 1]) >>> 0;
      }
      for (; kk < 623; kk += 1) {
        y = (this.mt[kk] & 0x80000000) | (this.mt[kk + 1] & 0x7fffffff);
        this.mt[kk] = (this.mt[kk - 227] ^ (y >>> 1) ^ mag01[y & 1]) >>> 0;
      }
      y = (this.mt[623] & 0x80000000) | (this.mt[0] & 0x7fffffff);
      this.mt[623] = (this.mt[396] ^ (y >>> 1) ^ mag01[y & 1]) >>> 0;
      this.mti = 0;
    }

    y = this.mt[this.mti];
    this.mti += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random() {
    const a = this._genrandUInt32() >>> 5;
    const b = this._genrandUInt32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  getRandBits(bits) {
    const k = Math.trunc(Number(bits));
    if (k <= 0) {
      return 0;
    }
    if (k <= 32) {
      return this._genrandUInt32() >>> (32 - k);
    }
    let value = 0;
    let remaining = k;
    let shift = 0;
    while (remaining > 0) {
      const take = Math.min(remaining, 32);
      value += (this._genrandUInt32() >>> (32 - take)) * 2 ** shift;
      shift += take;
      remaining -= take;
    }
    return value;
  }

  randBelow(stop) {
    const n = Math.trunc(Number(stop));
    if (n <= 0) {
      throw new Error(`Invalid randBelow bound: ${stop}`);
    }
    const bits = Math.floor(Math.log2(n)) + 1;
    let value = this.getRandBits(bits);
    while (value >= n) {
      value = this.getRandBits(bits);
    }
    return value;
  }

  randIntInclusive(minimum, maximum) {
    const low = Math.trunc(Number(minimum));
    const high = Math.trunc(Number(maximum));
    if (high < low) {
      throw new Error(`Invalid randint range: ${minimum}-${maximum}`);
    }
    return low + this.randBelow(high - low + 1);
  }
}

function seededRng(seed) {
  return new PythonRandom(seed);
}

function randomInt(rng, max) {
  return rng.randBelow(max);
}

function maxStepsForSolutionTicks(solutionTicks) {
  const ticks = Math.trunc(Number(solutionTicks) || 0);
  return Math.max(1, Math.floor((ticks * 13 + 9) / 10));
}

function normalizeDifficulty(value) {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return ["easy", "medium", "hard"].includes(normalized) ? normalized : null;
}

function solutionTickBoundsForDifficulty(difficulty) {
  const normalized = normalizeDifficulty(difficulty);
  if (normalized) {
    const [min, max] = DIFFICULTY_SETUPS[normalized].solutionTicks;
    return { min, max };
  }
  return null;
}

function gridSizeForDifficulty(difficulty) {
  const normalized = normalizeDifficulty(difficulty);
  if (!normalized) {
    return DEFAULT_CONFIG.gridSize;
  }
  const [min, max] = DIFFICULTY_SETUPS[normalized].gridSize;
  return min === max ? min : DEFAULT_CONFIG.gridSize;
}

function normalizeOptionalInt(value, fallback = null) {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
}

function difficultyLabelForSolutionTicks(solutionTicks) {
  const ticks = Number(solutionTicks) | 0;
  if (ticks < MIN_BENCHMARK_SOLUTION_TICKS) {
    return "warmup";
  }
  if (ticks <= EASY_MAX_SOLUTION_TICKS) {
    return "easy";
  }
  if (ticks <= MEDIUM_MAX_SOLUTION_TICKS) {
    return "medium";
  }
  return "hard";
}

function rotateMask(mask, ticks) {
  let normalized = Number(mask) | 0;
  const steps = ((Number(ticks) | 0) % 4 + 4) % 4;
  for (let step = 0; step < steps; step += 1) {
    let next = 0;
    if (normalized & 1) next |= 2;
    if (normalized & 2) next |= 4;
    if (normalized & 4) next |= 8;
    if (normalized & 8) next |= 1;
    normalized = next;
  }
  return normalized;
}

function targetTicksForMaskRotation(mask, rotation) {
  const solvedMask = (Number(mask) | 0) & 15;
  if (solvedMask === 0) {
    return 0;
  }
  const currentMask = rotateMask(solvedMask, rotation);
  for (let ticks = 0; ticks < 4; ticks += 1) {
    if (rotateMask(currentMask, ticks) === solvedMask) {
      return ticks;
    }
  }
  return 0;
}

function actionToCoord(action, gridSize) {
  const actionIndex = Number(action);
  if (!Number.isInteger(actionIndex)) {
    return null;
  }
  if (actionIndex < 0 || actionIndex >= gridSize * gridSize) {
    return null;
  }
  return {
    x: actionIndex % gridSize,
    y: Math.floor(actionIndex / gridSize),
    actionIndex
  };
}

function coordToAction(x, y, gridSize) {
  const parsedX = Number(x);
  const parsedY = Number(y);
  if (!Number.isInteger(parsedX) || !Number.isInteger(parsedY)) {
    return null;
  }
  if (parsedX < 0 || parsedX >= gridSize || parsedY < 0 || parsedY >= gridSize) {
    return null;
  }
  return parsedY * gridSize + parsedX;
}

function normalizeAction(action, gridSize) {
  if (Number.isInteger(action)) {
    return actionToCoord(action, gridSize);
  }
  if (action && typeof action === "object") {
    if (Number.isInteger(action.action_index) || Number.isInteger(action.actionIndex)) {
      return actionToCoord(action.action_index ?? action.actionIndex, gridSize);
    }
    if (Number.isInteger(action.cell_index) || Number.isInteger(action.cellIndex)) {
      return actionToCoord(action.cell_index ?? action.cellIndex, gridSize);
    }
    if (Number.isInteger(action.x) && Number.isInteger(action.y)) {
      const actionIndex = coordToAction(action.x, action.y, gridSize);
      return actionIndex === null ? null : { x: action.x, y: action.y, actionIndex };
    }
  }
  return null;
}

function connectedIndicesFromMasks(gridSize, sourceIndex, masks) {
  const connected = new Set([sourceIndex]);
  const queue = [sourceIndex];
  for (let head = 0; head < queue.length; head += 1) {
    const currentIndex = queue[head];
    const x = currentIndex % gridSize;
    const y = Math.floor(currentIndex / gridSize);
    const currentMask = masks[currentIndex];
    for (const dir of DIRS) {
      if ((currentMask & dir.bit) === 0) {
        continue;
      }
      const nextIndex = neighborIndex(x, y, gridSize, dir);
      if (nextIndex === null || connected.has(nextIndex)) {
        continue;
      }
      if ((masks[nextIndex] & OPPOSITE_BITS[dir.bit]) === 0) {
        continue;
      }
      connected.add(nextIndex);
      queue.push(nextIndex);
    }
  }
  return Array.from(connected).sort((a, b) => a - b);
}

function defaultActivePipeCount(gridSize, rng) {
  const total = gridSize * gridSize;
  const minActive = Math.min(total - 1, Math.max(3, Math.ceil(total / 2)));
  const maxActive = Math.max(minActive, total - 1);
  return rng.randIntInclusive(minActive, maxActive);
}

function activePipeIndicesFromMasks(masks) {
  return masks
    .map((mask, index) => ((Number(mask) | 0) !== 0 ? index : null))
    .filter((index) => index !== null);
}

function junctionCountFromMasks(masks) {
  return masks.filter((mask) => {
    let value = (Number(mask) | 0) & 15;
    let bits = 0;
    while (value > 0) {
      bits += value & 1;
      value >>= 1;
    }
    return bits === 3;
  }).length;
}

function hasFourWayJunction(masks) {
  return masks.some((mask) => {
    let value = (Number(mask) | 0) & 15;
    let bits = 0;
    while (value > 0) {
      bits += value & 1;
      value >>= 1;
    }
    return bits >= 4;
  });
}

function sampleCountRange(rng, exact, minimum, maximum, fallbackMinimum, fallbackMaximum) {
  if (exact !== null && exact !== undefined && exact !== "") {
    return Math.trunc(Number(exact));
  }
  const low = normalizeOptionalInt(minimum, fallbackMinimum);
  const high = normalizeOptionalInt(maximum, fallbackMaximum);
  if (high < low) {
    throw new Error(`Invalid count range: ${low}-${high}`);
  }
  return rng.randIntInclusive(low, high);
}

function generateSolvedMasks(gridSize, sourceIndex, rng, activePipeCount = null) {
  const total = gridSize * gridSize;
  const fallbackActiveCount =
    activePipeCount === null || activePipeCount === undefined ? defaultActivePipeCount(gridSize, rng) : activePipeCount;
  const targetActiveCount = clampInt(
    activePipeCount ?? fallbackActiveCount,
    2,
    total,
    Math.min(total, Math.max(2, fallbackActiveCount))
  );
  const masks = new Array(total).fill(0);
  const active = new Set([sourceIndex]);

  while (active.size < targetActiveCount) {
    const candidates = [];
    for (const currentIndex of Array.from(active).sort((a, b) => a - b)) {
      const x = currentIndex % gridSize;
      const y = Math.floor(currentIndex / gridSize);
      for (const dir of DIRS) {
        const nextIndex = neighborIndex(x, y, gridSize, dir);
        if (nextIndex !== null && !active.has(nextIndex)) {
          candidates.push({
            currentIndex,
            nextIndex,
            bit: dir.bit,
            oppositeBit: OPPOSITE_BITS[dir.bit]
          });
        }
      }
    }
    if (candidates.length === 0) {
      break;
    }
    const candidate = candidates[randomInt(rng, candidates.length)];
    masks[candidate.currentIndex] |= candidate.bit;
    masks[candidate.nextIndex] |= candidate.oppositeBit;
    active.add(candidate.nextIndex);
  }

  if (active.size !== targetActiveCount) {
    throw new Error("Failed to generate a connected sparse pipe board.");
  }
  return masks;
}

function currentMasksFromArrays(solvedMasks, rotations) {
  return solvedMasks.map((mask, index) => rotateMask(mask, rotations[index]));
}

function solutionTicks(solvedMasks, rotations) {
  return rotations.reduce((total, rotation, index) => {
    if (((Number(solvedMasks[index]) | 0) & 15) === 0) {
      return total;
    }
    return total + targetTicksForMaskRotation(solvedMasks[index], rotation);
  }, 0);
}

function rotationWeight(targetTicks, difficulty) {
  void targetTicks;
  void difficulty;
  return 1;
}

function weightedChoice(rng, choices, weights) {
  let total = 0;
  const cumulativeWeights = weights.map((weight) => {
    total += weight;
    return total;
  });
  const marker = rng.random() * total;
  for (let index = 0; index < choices.length; index += 1) {
    if (marker < cumulativeWeights[index]) {
      return choices[index];
    }
  }
  return choices[choices.length - 1];
}

function sampleRotations(solvedMasks, rng, difficulty) {
  return solvedMasks.map((mask) => {
    if (((Number(mask) | 0) & 15) === 0) {
      return 0;
    }
    const choices = [0, 1, 2, 3];
    const weights = choices.map((rotation) => rotationWeight(targetTicksForMaskRotation(mask, rotation), difficulty));
    return weightedChoice(rng, choices, weights);
  });
}

function samplePipeConfig(
  gridSize,
  seed,
  explicitSourceIndex = null,
  targetDifficulty = null,
  explicitActivePipeCount = null,
  minSolutionTicks = MIN_BENCHMARK_SOLUTION_TICKS,
  maxSolutionTicks = null,
  minActivePipeCount = null,
  maxActivePipeCount = null,
  minJunctionCount = null,
  maxJunctionCount = null
) {
  const rng = seededRng(seed);
  const total = gridSize * gridSize;
  const normalizedDifficulty = normalizeDifficulty(targetDifficulty);
  const difficultyBounds = normalizedDifficulty ? solutionTickBoundsForDifficulty(normalizedDifficulty) : null;
  const difficultySetup = normalizedDifficulty ? DIFFICULTY_SETUPS[normalizedDifficulty] : null;
  if (difficultySetup) {
    const [gridMin, gridMax] = difficultySetup.gridSize;
    if (gridSize < gridMin || gridSize > gridMax) {
      throw new Error(`Difficulty ${normalizedDifficulty} requires grid size ${gridMin}-${gridMax}; got ${gridSize}.`);
    }
  }
  const minTicks = difficultyBounds
    ? difficultyBounds.min
    : normalizeOptionalInt(minSolutionTicks, MIN_BENCHMARK_SOLUTION_TICKS);
  const maxTicks = difficultyBounds ? difficultyBounds.max : normalizeOptionalInt(maxSolutionTicks, null);
  const fallbackActiveMin = Math.min(total, Math.max(2, Math.ceil(total / 2)));
  const fallbackActiveMax = Math.max(fallbackActiveMin, total - 1);
  const activeMin = minActivePipeCount ?? difficultySetup?.activePipeCount[0] ?? null;
  const activeMax = maxActivePipeCount ?? difficultySetup?.activePipeCount[1] ?? null;
  const junctionMin = minJunctionCount ?? difficultySetup?.junctionCount[0] ?? null;
  const junctionMax = maxJunctionCount ?? difficultySetup?.junctionCount[1] ?? null;

  for (let attempt = 0; attempt < 20000; attempt += 1) {
    const sourceIndex = Number.isInteger(explicitSourceIndex)
      ? clampInt(explicitSourceIndex, 0, total - 1, 0)
      : rng.randBelow(total);
    const sampledActivePipeCount = sampleCountRange(
      rng,
      explicitActivePipeCount,
      activeMin,
      activeMax,
      fallbackActiveMin,
      fallbackActiveMax
    );
    const solvedMasks = generateSolvedMasks(gridSize, sourceIndex, rng, sampledActivePipeCount);
    if (hasFourWayJunction(solvedMasks)) {
      continue;
    }
    const totalPipes = activePipeIndicesFromMasks(solvedMasks).length;
    const junctionCount = junctionCountFromMasks(solvedMasks);
    if (junctionMin !== null && junctionCount < Number(junctionMin)) {
      continue;
    }
    if (junctionMax !== null && junctionCount > Number(junctionMax)) {
      continue;
    }

    for (let rotationAttempt = 0; rotationAttempt < 40; rotationAttempt += 1) {
      const rotations = sampleRotations(solvedMasks, rng, normalizedDifficulty);
      const ticks = solutionTicks(solvedMasks, rotations);
      if (ticks < minTicks) {
        continue;
      }
      if (maxTicks !== null && ticks > maxTicks) {
        continue;
      }
      const masks = currentMasksFromArrays(solvedMasks, rotations);
      const connected = connectedIndicesFromMasks(gridSize, sourceIndex, masks);
      if (connected.length < totalPipes && ticks > 0) {
        return {
          sourceIndex,
          solvedMasks,
          rotations,
          solutionTicks: ticks,
          difficulty: normalizedDifficulty ?? difficultyLabelForSolutionTicks(ticks),
          junctionCount
        };
      }
    }
  }
  throw new Error("Unable to sample a scrambled continuity pipe board.");
}

function normalizeMaskArray(values, gridSize) {
  const total = gridSize * gridSize;
  if (!Array.isArray(values)) {
    return buildSnakeSolvedMasks(gridSize);
  }
  const masks = Array.isArray(values) ? values.map((value) => Number(value) | 0) : [];
  if (masks.length !== total) {
    throw new Error(`Expected ${total} pipe masks, got ${masks.length}.`);
  }
  return masks.map((value) => value & 15);
}

function normalizeRotationArray(values, gridSize) {
  const total = gridSize * gridSize;
  if (!Array.isArray(values)) {
    return Array.from({ length: total }, (_value, index) => (index % 3) + 1);
  }
  const rotations = Array.isArray(values) ? values.map((value) => Number(value) | 0) : [];
  if (rotations.length !== total) {
    throw new Error(`Expected ${total} rotations, got ${rotations.length}.`);
  }
  return rotations.map((value) => ((value % 4) + 4) % 4);
}

function buildSnakeSolvedMasks(gridSize) {
  const masks = new Array(gridSize * gridSize).fill(0);
  const path = [];
  for (let y = 0; y < gridSize; y += 1) {
    if (y % 2 === 0) {
      for (let x = 0; x < gridSize; x += 1) {
        path.push({ x, y });
      }
    } else {
      for (let x = gridSize - 1; x >= 0; x -= 1) {
        path.push({ x, y });
      }
    }
  }
  for (let index = 1; index < path.length; index += 1) {
    const prev = path[index - 1];
    const curr = path[index];
    const prevIndex = prev.y * gridSize + prev.x;
    const currIndex = curr.y * gridSize + curr.x;
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    if (dx === 1) {
      masks[prevIndex] |= 2;
      masks[currIndex] |= 8;
    } else if (dx === -1) {
      masks[prevIndex] |= 8;
      masks[currIndex] |= 2;
    } else if (dy === 1) {
      masks[prevIndex] |= 4;
      masks[currIndex] |= 1;
    } else if (dy === -1) {
      masks[prevIndex] |= 1;
      masks[currIndex] |= 4;
    }
  }
  return masks;
}

function buildCells(gridSize, solvedMasks, rotations, sourceIndex) {
  const cells = [];
  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const index = y * gridSize + x;
      const solvedMask = solvedMasks[index];
      const hasPipe = ((Number(solvedMask) | 0) & 15) !== 0 || index === sourceIndex;
      cells.push({
        index,
        x,
        y,
        hasPipe,
        solvedMask,
        rotation: hasPipe ? rotations[index] : 0,
        currentMask: hasPipe ? rotateMask(solvedMask, rotations[index]) : 0,
        isSource: index === sourceIndex
      });
    }
  }
  return cells;
}

function neighborIndex(x, y, gridSize, dir) {
  const nx = x + dir.dx;
  const ny = y + dir.dy;
  if (nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize) {
    return null;
  }
  return ny * gridSize + nx;
}

function computeConnectedIndices(cells, gridSize, sourceIndex) {
  const connected = new Set([sourceIndex]);
  const queue = [sourceIndex];
  for (let head = 0; head < queue.length; head += 1) {
    const current = cells[queue[head]];
    for (const dir of DIRS) {
      if ((current.currentMask & dir.bit) === 0) {
        continue;
      }
      const nextIndex = neighborIndex(current.x, current.y, gridSize, dir);
      if (nextIndex === null || connected.has(nextIndex)) {
        continue;
      }
      const next = cells[nextIndex];
      if ((next.currentMask & OPPOSITE_BITS[dir.bit]) === 0) {
        continue;
      }
      connected.add(nextIndex);
      queue.push(nextIndex);
    }
  }
  return Array.from(connected).sort((a, b) => a - b);
}

function currentMasks(cells) {
  return cells.map((cell) => cell.currentMask);
}

function targetRotationTicks(cells) {
  return cells.map((cell) => (cell.hasPipe ? targetTicksForMaskRotation(cell.solvedMask, cell.rotation) : 0));
}

function activePipeCells(cells) {
  return cells.filter((cell) => cell.hasPipe);
}

function buildActionCatalog(gridSize, cells = null) {
  const actions = [];
  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const actionIndex = y * gridSize + x;
      if (cells && !cells[actionIndex]?.hasPipe) {
        continue;
      }
      actions.push({
        action_index: actionIndex,
        x,
        y,
        label: `x=${x}, y=${y}`
      });
    }
  }
  return actions;
}

class PipeRenderer {
  constructor(container, onCellClick) {
    this.container = container;
    this.onCellClick = onCellClick;
    this.root = document.createElement("div");
    this.root.className = "pipe-board";
    this.container.replaceChildren(this.root);
  }

  syncState(state) {
    const gridSize = state.gridSize;
    const connected = new Set(state.connectedIndices || []);
    this.root.style.setProperty("--grid-size", String(gridSize));
    this.root.replaceChildren();

    const corner = document.createElement("div");
    corner.className = "coord-corner";
    corner.textContent = "y/x";
    this.root.appendChild(corner);

    for (let x = 0; x < gridSize; x += 1) {
      const label = document.createElement("div");
      label.className = "coord-label coord-x";
      label.textContent = String(x);
      this.root.appendChild(label);
    }

    for (let y = 0; y < gridSize; y += 1) {
      const yLabel = document.createElement("div");
      yLabel.className = "coord-label coord-y";
      yLabel.textContent = String(y);
      this.root.appendChild(yLabel);

      for (let x = 0; x < gridSize; x += 1) {
        const index = y * gridSize + x;
        const cell = state.cells[index];
        const hasPipe = Boolean(cell.hasPipe);
        const connectedToSource = connected.has(index);
        const cellElement = document.createElement("button");
        cellElement.type = "button";
        cellElement.className = hasPipe ? "pipe-cell" : "pipe-cell empty";
        cellElement.dataset.actionIndex = String(index);
        cellElement.dataset.x = String(x);
        cellElement.dataset.y = String(y);
        cellElement.title = hasPipe ? `x=${x}, y=${y}` : `x=${x}, y=${y} (empty)`;
        cellElement.disabled = !hasPipe;
        if (hasPipe) {
          cellElement.addEventListener("click", () => this.onCellClick(index));
        }

        if (!hasPipe) {
          this.root.appendChild(cellElement);
          continue;
        }

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 100 100");
        svg.setAttribute("class", connectedToSource ? "pipe-svg connected" : "pipe-svg disconnected");
        const strokeColor = connectedToSource ? "#19d955" : "#0aa2ff";
        const glowColor = connectedToSource ? "rgba(25,217,85,0.5)" : "rgba(10,162,255,0.42)";

        for (const dir of DIRS) {
          if ((cell.currentMask & dir.bit) === 0) {
            continue;
          }
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", "50");
          line.setAttribute("y1", "50");
          line.setAttribute("x2", String(50 + dir.dx * 40));
          line.setAttribute("y2", String(50 + dir.dy * 40));
          line.setAttribute("stroke", strokeColor);
          line.setAttribute("stroke-width", "15");
          line.setAttribute("stroke-linecap", "round");
          line.style.filter = `drop-shadow(0 0 5px ${glowColor})`;
          svg.appendChild(line);
        }

        const center = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        center.setAttribute("cx", "50");
        center.setAttribute("cy", "50");
        center.setAttribute("r", cell.isSource ? "18" : "10");
        center.setAttribute("fill", cell.isSource ? "#19d955" : strokeColor);
        center.style.filter = `drop-shadow(0 0 7px ${glowColor})`;
        svg.appendChild(center);
        cellElement.appendChild(svg);
        this.root.appendChild(cellElement);
      }
    }
  }
}

export class ContinuityPipeGame {
  constructor(container, onStateChange = null) {
    this.onStateChange = typeof onStateChange === "function" ? onStateChange : null;
    this.renderer = container
      ? new PipeRenderer(container, (actionIndex) => {
          const result = this.step(actionIndex);
          if (this.onStateChange) {
            this.onStateChange({ type: "cell-click", action: actionIndex, result });
          }
        })
      : null;
    this.reset({});
  }

  _normalizeConfig(config = {}) {
    const targetDifficulty = normalizeDifficulty(config.targetDifficulty ?? config.difficulty ?? DEFAULT_CONFIG.difficulty);
    const hasSolvedMasks = Array.isArray(config.solvedMasks);
    const hasRotations = Array.isArray(config.rotations);
    const difficultyGridSize = gridSizeForDifficulty(targetDifficulty);
    const requestedGridSize = targetDifficulty && !hasSolvedMasks ? difficultyGridSize : config.gridSize;
    const gridSize = clampInt(requestedGridSize, MIN_GRID_SIZE, MAX_GRID_SIZE, difficultyGridSize);
    const total = gridSize * gridSize;
    const seed = normalizeSeed(config.seed, DEFAULT_CONFIG.seed);
    if (hasSolvedMasks !== hasRotations) {
      throw new Error("solvedMasks and rotations must be provided together.");
    }
    const sampled = hasSolvedMasks
      ? null
      : samplePipeConfig(
          gridSize,
          seed,
          config.sourceIndex,
          targetDifficulty,
          config.activePipeCount ?? DEFAULT_CONFIG.activePipeCount,
          config.minSolutionTicks ?? config.solutionStepsMin,
          config.maxSolutionTicks ?? config.solutionStepsMax,
          config.minActivePipeCount,
          config.maxActivePipeCount,
          config.minJunctionCount,
          config.maxJunctionCount
        );
    const solvedMasks = hasSolvedMasks ? normalizeMaskArray(config.solvedMasks, gridSize) : sampled.solvedMasks;
    const rotations = hasRotations ? normalizeRotationArray(config.rotations, gridSize) : sampled.rotations;
    const sourceIndex = hasSolvedMasks
      ? clampInt(config.sourceIndex, 0, total - 1, DEFAULT_CONFIG.sourceIndex)
      : sampled.sourceIndex;
    const ticks = solutionTicks(solvedMasks, rotations);
    return {
      gridSize,
      seed,
      sourceIndex,
      solvedMasks,
      rotations,
      solutionTicks: ticks,
      difficulty: hasSolvedMasks ? (targetDifficulty ?? difficultyLabelForSolutionTicks(ticks)) : sampled.difficulty,
      junctionCount: hasSolvedMasks ? junctionCountFromMasks(solvedMasks) : sampled.junctionCount,
      maxSteps: Math.max(1, clampInt(config.maxSteps, 1, 1000, maxStepsForSolutionTicks(ticks))),
      illegalReward: Number(config.illegalReward ?? DEFAULT_CONFIG.illegalReward),
      moveReward: Number(config.moveReward ?? DEFAULT_CONFIG.moveReward),
      successReward: Number(config.successReward ?? DEFAULT_CONFIG.successReward),
      timeoutPenalty: Number(config.timeoutPenalty ?? DEFAULT_CONFIG.timeoutPenalty)
    };
  }

  reset(config = {}) {
    this.config = this._normalizeConfig(config);
    this.cells = buildCells(
      this.config.gridSize,
      this.config.solvedMasks,
      this.config.rotations,
      this.config.sourceIndex
    );
    this.stepCount = 0;
    this.lastTurn = null;
    this.moveHistory = [];
    this.terminal = this._deriveTerminalState();
    this._syncRenderer();
    return this._buildStateResponse({
      reward: 0,
      done: Boolean(this.terminal),
      success: this.terminal?.success ?? false,
      info: {
        reset: true,
        illegal: false,
        reason: this.terminal?.reason ?? null
      }
    });
  }

  decodeAction(action) {
    return normalizeAction(action, this.config.gridSize);
  }

  isLegalMove(action) {
    return this._validateAction(this.decodeAction(action)).legal;
  }

  getObservation() {
    const connected = computeConnectedIndices(this.cells, this.config.gridSize, this.config.sourceIndex);
    const activeCells = activePipeCells(this.cells);
    return {
      gridSize: this.config.gridSize,
      seed: this.config.seed,
      sourceIndex: this.config.sourceIndex,
      source: actionToCoord(this.config.sourceIndex, this.config.gridSize),
      connectedCount: connected.length,
      totalPipes: activeCells.length,
      activePipeIndices: activeCells.map((cell) => cell.index),
      legalActionIndices: this.terminal ? [] : activeCells.map((cell) => cell.index),
      actionCatalog: buildActionCatalog(this.config.gridSize, this.cells),
      solutionTicks: this.config.solutionTicks,
      solutionSteps: this.config.solutionTicks,
      difficulty: this.config.difficulty,
      junctionCount: this.config.junctionCount,
      targetRotationTicks: targetRotationTicks(this.cells)
    };
  }

  getDebugState() {
    const connectedIndices = computeConnectedIndices(this.cells, this.config.gridSize, this.config.sourceIndex);
    const activeCells = activePipeCells(this.cells);
    return {
      gridSize: this.config.gridSize,
      seed: this.config.seed,
      sourceIndex: this.config.sourceIndex,
      source: actionToCoord(this.config.sourceIndex, this.config.gridSize),
      cells: this.cells.map((cell) => ({
        index: cell.index,
        x: cell.x,
        y: cell.y,
        hasPipe: cell.hasPipe,
        solvedMask: cell.solvedMask,
        rotation: cell.rotation,
        currentMask: cell.currentMask,
        targetRotationTicks: cell.hasPipe ? targetTicksForMaskRotation(cell.solvedMask, cell.rotation) : 0,
        isSource: cell.isSource
      })),
      currentMasks: currentMasks(this.cells),
      solvedMasks: this.config.solvedMasks.slice(),
      rotations: this.cells.map((cell) => cell.rotation),
      connectedIndices,
      connectedCount: connectedIndices.length,
      totalPipes: activeCells.length,
      activePipeIndices: activeCells.map((cell) => cell.index),
      legalActionIndices: this.terminal ? [] : activeCells.map((cell) => cell.index),
      actionCatalog: buildActionCatalog(this.config.gridSize, this.cells),
      stepCount: this.stepCount,
      maxSteps: this.config.maxSteps,
      canUndo: this.moveHistory.length > 0,
      undoDepth: this.moveHistory.length,
      solutionTicks: this.config.solutionTicks,
      solutionSteps: this.config.solutionTicks,
      difficulty: this.config.difficulty,
      junctionCount: this.config.junctionCount,
      lastTurn: this.lastTurn ? { ...this.lastTurn } : null,
      terminal: this.terminal ? { ...this.terminal } : null,
      targetRotationTicks: targetRotationTicks(this.cells)
    };
  }

  getState() {
    return this._buildStateResponse({
      reward: 0,
      done: Boolean(this.terminal),
      success: this.terminal?.success ?? false,
      info: {
        illegal: false,
        reason: this.terminal?.reason ?? null,
        snapshot: true
      }
    });
  }

  step(action) {
    if (this.terminal) {
      return this._illegalResult({ reason: "already_done", action });
    }
    const decoded = this.decodeAction(action);
    const validation = this._validateAction(decoded);
    if (!validation.legal) {
      return this._illegalResult({ reason: validation.reason, action });
    }

    const cell = this.cells[decoded.actionIndex];
    const beforeRotation = cell.rotation;
    cell.rotation = (cell.rotation + 1) % 4;
    cell.currentMask = rotateMask(cell.solvedMask, cell.rotation);
    this.stepCount += 1;

    this.terminal = this._deriveTerminalState();
    let reward = this.config.moveReward;
    let reason = "pipe_rotated";
    let success = false;
    let done = false;
    if (this.terminal) {
      done = true;
      success = Boolean(this.terminal.success);
      reason = this.terminal.reason;
      reward = success ? this.config.successReward : this.config.timeoutPenalty;
    }

    this.lastTurn = {
      actionIndex: decoded.actionIndex,
      x: decoded.x,
      y: decoded.y,
      beforeRotation,
      afterRotation: cell.rotation
    };
    this.moveHistory.push({ ...this.lastTurn });
    this._syncRenderer();

    return this._buildStateResponse({
      reward,
      done,
      success,
      info: {
        illegal: false,
        reason,
        actionIndex: decoded.actionIndex,
        x: decoded.x,
        y: decoded.y
      }
    });
  }

  undo() {
    if (!this.moveHistory.length) {
      return this._illegalResult({ reason: "nothing_to_undo" });
    }

    const move = this.moveHistory.pop();
    const cell = this.cells[move.actionIndex];
    cell.rotation = move.beforeRotation;
    cell.currentMask = rotateMask(cell.solvedMask, cell.rotation);
    this.stepCount = Math.max(0, this.stepCount - 1);
    this.lastTurn = this.moveHistory.length ? { ...this.moveHistory[this.moveHistory.length - 1] } : null;
    this.terminal = this._deriveTerminalState();
    this._syncRenderer();

    return this._buildStateResponse({
      reward: 0,
      done: Boolean(this.terminal),
      success: this.terminal?.success ?? false,
      info: {
        illegal: false,
        reason: "undo",
        actionIndex: move.actionIndex,
        x: move.x,
        y: move.y,
        beforeRotation: move.afterRotation,
        afterRotation: move.beforeRotation
      }
    });
  }

  renderConfig() {
    return {
      gridSize: this.config.gridSize,
      seed: this.config.seed,
      sourceIndex: this.config.sourceIndex,
      actionCount: activePipeCells(this.cells).length,
      solutionTicks: this.config.solutionTicks,
      solutionSteps: this.config.solutionTicks,
      difficulty: this.config.difficulty,
      junctionCount: this.config.junctionCount,
      maxSteps: this.config.maxSteps
    };
  }

  _deriveTerminalState() {
    const connectedCount = computeConnectedIndices(this.cells, this.config.gridSize, this.config.sourceIndex).length;
    if (connectedCount === activePipeCells(this.cells).length) {
      return { success: true, reason: "all_pipes_connected" };
    }
    if (this.stepCount >= this.config.maxSteps) {
      return { success: false, reason: "step_budget_exhausted" };
    }
    return null;
  }

  _validateAction(decoded) {
    if (!decoded) {
      return { legal: false, reason: "invalid_action" };
    }
    if (decoded.actionIndex < 0 || decoded.actionIndex >= this.cells.length) {
      return { legal: false, reason: "out_of_range" };
    }
    if (!this.cells[decoded.actionIndex].hasPipe) {
      return { legal: false, reason: "empty_cell" };
    }
    return { legal: true, reason: null };
  }

  _illegalResult(extraInfo) {
    return this._buildStateResponse({
      reward: this.config.illegalReward,
      done: Boolean(this.terminal),
      success: this.terminal?.success ?? false,
      info: {
        illegal: true,
        ...extraInfo
      }
    });
  }

  _buildStateResponse({ reward, done, success, info }) {
    return {
      observation: this.getObservation(),
      reward,
      done: Boolean(done),
      success: Boolean(success),
      step_count: this.stepCount,
      info: {
        state: this.getDebugState(),
        ...info
      }
    };
  }

  _syncRenderer() {
    if (this.renderer) {
      this.renderer.syncState(this.getDebugState());
    }
  }
}
