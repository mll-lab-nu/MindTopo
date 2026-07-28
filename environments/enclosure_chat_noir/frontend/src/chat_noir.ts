import { ChatNoirRenderer } from "./render.ts";

const HEX_DIRECTIONS = Object.freeze([
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1]
]);

const MIN_BOARD_RADIUS = 3;
const MAX_BOARD_RADIUS = 4;
const ALLOWED_BOARD_RADII = Object.freeze([3, 4]);
const SAMPLE_ATTEMPTS = 512;
const SOLVER_ACTION_BRANCH_LIMIT = 6;
const SOLVER_MAX_NODES = 1000;
const SOLVER_MAX_DEPTH_BY_RADIUS = Object.freeze({
  3: 7,
  4: 6
});
const AUTO_SETUP_VALUE = "auto";
const DEFAULT_MIN_WINNING_FIRST_ACTIONS = 5;
const DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC = AUTO_SETUP_VALUE;
const DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS = 3;
const MIN_WINNING_FIRST_ACTIONS_BY_POLICY = Object.freeze({
  easy: 5,
  medium: 5,
  hard: 5
});
const INITIAL_BLOCK_COUNT_RANGES = Object.freeze({
  easy: Object.freeze({ 3: [9, 12], 4: [11, 14] }),
  medium: Object.freeze({ 3: [12, 14], 4: [14, 17] }),
  hard: Object.freeze({ 3: [14, 16], 4: [17, 21] })
});

const CAT_POLICIES = Object.freeze({
  static: {
    name: "static",
    level: 0,
    description: "Stationary cat. It never moves after the player blocks a cell."
  },
  easy: {
    name: "easy",
    level: 1,
    description: "Mixed walker. On each move it uses shortest-path greedy behavior with 30% probability and otherwise chooses uniformly from the currently legal adjacent open cells."
  },
  medium: {
    name: "medium",
    level: 2,
    description: "Greedy shortest-path runner. It chooses an adjacent open cell that minimizes current distance to the boundary, with simple lowest-index tie-breaks."
  },
  hard: {
    name: "hard",
    level: 3,
    description: "Connectivity-aware greedy runner. It avoids immediate one-move traps when possible, then prefers shorter boundary distance while preserving more escape connectivity."
  }
});

const DEFAULT_CONFIG = Object.freeze({
  boardRadius: AUTO_SETUP_VALUE,
  initialBlockedCount: AUTO_SETUP_VALUE,
  initialBlockedIndices: null,
  catIndex: null,
  catPolicy: "easy",
  rngSeed: null,
  minWinningFirstActions: DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
  minAdjacentWinningFirstActions: DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS,
  animate: true,
  illegalReward: -1.0,
  moveReward: -0.02,
  trapReward: 1.0,
  escapePenalty: -1.0
});

function clampInt(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function coordKey(q, r) {
  return `${q},${r}`;
}

class PythonRandom {
  constructor(seed) {
    this.n = 624;
    this.m = 397;
    this.matrixA = 0x9908b0df;
    this.upperMask = 0x80000000;
    this.lowerMask = 0x7fffffff;
    this.mt = new Array(this.n).fill(0);
    this.index = this.n + 1;
    this.seed(seed);
  }

  seed(seed) {
    this._initByArray(this._seedKey(seed));
  }

  _seedKey(seed) {
    const numericSeed = Number(seed);
    let value = Number.isFinite(numericSeed) ? Math.trunc(numericSeed) : 0;
    if (value < 0) {
      value = Math.abs(value);
    }
    let bigintSeed = BigInt(value);
    const key = [];
    do {
      key.push(Number(bigintSeed & 0xffffffffn) >>> 0);
      bigintSeed >>= 32n;
    } while (bigintSeed > 0n);
    return key;
  }

  _initGenrand(seed) {
    this.mt[0] = seed >>> 0;
    for (this.index = 1; this.index < this.n; this.index += 1) {
      const previous = this.mt[this.index - 1];
      this.mt[this.index] = (Math.imul(1812433253, previous ^ (previous >>> 30)) + this.index) >>> 0;
    }
  }

  _initByArray(initKey) {
    this._initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(this.n, initKey.length);
    for (; k > 0; k -= 1) {
      const previous = this.mt[i - 1];
      this.mt[i] = (
        (this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1664525)) +
        initKey[j] +
        j
      ) >>> 0;
      i += 1;
      j += 1;
      if (i >= this.n) {
        this.mt[0] = this.mt[this.n - 1];
        i = 1;
      }
      if (j >= initKey.length) {
        j = 0;
      }
    }
    for (k = this.n - 1; k > 0; k -= 1) {
      const previous = this.mt[i - 1];
      this.mt[i] = (
        (this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1566083941)) -
        i
      ) >>> 0;
      i += 1;
      if (i >= this.n) {
        this.mt[0] = this.mt[this.n - 1];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  getrandbits(bitCount) {
    const bits = Math.trunc(bitCount);
    if (bits <= 0) {
      return 0;
    }
    if (bits > 32) {
      let value = 0n;
      let remaining = bits;
      while (remaining > 0) {
        const take = Math.min(remaining, 32);
        value = (value << BigInt(take)) | BigInt(this.getrandbits(take));
        remaining -= take;
      }
      return Number(value);
    }
    return this._genrandInt32() >>> (32 - bits);
  }

  randbelow(limit) {
    const n = Math.trunc(limit);
    if (n <= 0) {
      throw new Error("randbelow limit must be positive.");
    }
    const bits = Math.floor(Math.log2(n)) + 1;
    let value = this.getrandbits(bits);
    while (value >= n) {
      value = this.getrandbits(bits);
    }
    return value;
  }

  randrange(start, stop = null) {
    let lower = 0;
    let upper = Math.trunc(start);
    if (stop !== null) {
      lower = Math.trunc(start);
      upper = Math.trunc(stop);
    }
    if (upper <= lower) {
      throw new Error("empty range for randrange().");
    }
    return lower + this.randbelow(upper - lower);
  }

  sample(population, count) {
    const n = population.length;
    const k = Math.trunc(count);
    if (k < 0 || k > n) {
      throw new Error("Sample larger than population or is negative.");
    }
    const result = new Array(k);
    let setsize = 21;
    if (k > 5) {
      setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    }

    if (n <= setsize) {
      const pool = population.slice();
      for (let i = 0; i < k; i += 1) {
        const j = this.randbelow(n - i);
        result[i] = pool[j];
        pool[j] = pool[n - i - 1];
      }
    } else {
      const selected = new Set();
      for (let i = 0; i < k; i += 1) {
        let j = this.randbelow(n);
        while (selected.has(j)) {
          j = this.randbelow(n);
        }
        selected.add(j);
        result[i] = population[j];
      }
    }
    return result;
  }

  _genrandInt32() {
    const mag01 = [0x0, this.matrixA];
    if (this.index >= this.n) {
      let kk = 0;
      for (; kk < this.n - this.m; kk += 1) {
        const y = (this.mt[kk] & this.upperMask) | (this.mt[kk + 1] & this.lowerMask);
        this.mt[kk] = (this.mt[kk + this.m] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      }
      for (; kk < this.n - 1; kk += 1) {
        const y = (this.mt[kk] & this.upperMask) | (this.mt[kk + 1] & this.lowerMask);
        this.mt[kk] = (this.mt[kk + (this.m - this.n)] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      }
      const y = (this.mt[this.n - 1] & this.upperMask) | (this.mt[0] & this.lowerMask);
      this.mt[this.n - 1] = (this.mt[this.m - 1] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      this.index = 0;
    }

    let y = this.mt[this.index];
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

function toSortedNumberArray(values) {
  return Array.from(new Set(values.map((value) => Number(value)))).sort((a, b) => a - b);
}

function appendUnique(values, value) {
  if (!values.includes(value)) {
    values.push(value);
  }
}

function buildBoard(boardRadius) {
  const cells = [];
  const indexByCoord = new Map();
  let index = 0;
  const scale = 1.22;

  for (let r = -boardRadius; r <= boardRadius; r += 1) {
    const minQ = Math.max(-boardRadius, -r - boardRadius);
    const maxQ = Math.min(boardRadius, -r + boardRadius);
    for (let q = minQ; q <= maxQ; q += 1) {
      const s = -q - r;
      const x = Math.sqrt(3) * (q + r / 2) * scale;
      const y = -1.5 * r * scale;
      const isBoundary = Math.max(Math.abs(q), Math.abs(r), Math.abs(s)) === boardRadius;
      const cell = {
        index,
        q,
        r,
        s,
        x,
        y,
        radius: 0.74,
        isBoundary,
        neighborIndices: []
      };
      cells.push(cell);
      indexByCoord.set(coordKey(q, r), index);
      index += 1;
    }
  }

  for (const cell of cells) {
    const neighbors = [];
    for (const [dq, dr] of HEX_DIRECTIONS) {
      const neighborIndex = indexByCoord.get(coordKey(cell.q + dq, cell.r + dr));
      if (neighborIndex !== undefined) {
        neighbors.push(neighborIndex);
      }
    }
    cell.neighborIndices = neighbors;
  }

  const boundaryIndices = cells.filter((cell) => cell.isBoundary).map((cell) => cell.index);
  const centerCell = cells.find((cell) => cell.q === 0 && cell.r === 0) || cells[0];
  return {
    radius: boardRadius,
    cells,
    boundaryIndices,
    centerIndex: centerCell.index
  };
}

function shortestBoundaryPath(board, startIndex, blockedSet) {
  if (blockedSet.has(startIndex)) {
    return null;
  }
  const startCell = board.cells[startIndex];
  if (startCell.isBoundary) {
    return [startIndex];
  }

  const queue = [startIndex];
  const parents = new Map([[startIndex, null]]);

  for (let head = 0; head < queue.length; head += 1) {
    const current = queue[head];
    const cell = board.cells[current];
    for (const neighborIndex of cell.neighborIndices) {
      if (blockedSet.has(neighborIndex) || parents.has(neighborIndex)) {
        continue;
      }
      parents.set(neighborIndex, current);
      if (board.cells[neighborIndex].isBoundary) {
        const path = [neighborIndex];
        let cursor = current;
        while (cursor !== null) {
          path.push(cursor);
          cursor = parents.get(cursor) ?? null;
        }
        path.reverse();
        return path;
      }
      queue.push(neighborIndex);
    }
  }

  return null;
}

function shortestBoundaryDistance(board, startIndex, blockedSet) {
  const path = shortestBoundaryPath(board, startIndex, blockedSet);
  return path ? path.length - 1 : null;
}

function hexDistance(board, leftIndex, rightIndex) {
  const left = board.cells[leftIndex];
  const right = board.cells[rightIndex];
  return Math.max(
    Math.abs(left.q - right.q),
    Math.abs(left.r - right.r),
    Math.abs(left.s - right.s)
  );
}

function legalCatMoves(board, catIndex, blockedSet) {
  return board.cells[catIndex].neighborIndices.filter((neighborIndex) => !blockedSet.has(neighborIndex));
}

function legalPlayerActions(board, catIndex, blockedSet) {
  return board.cells
    .map((cell) => cell.index)
    .filter((index) => index !== catIndex && !blockedSet.has(index));
}

function countOpenNeighborDegree(board, cellIndex, blockedSet) {
  return board.cells[cellIndex].neighborIndices.filter((neighborIndex) => !blockedSet.has(neighborIndex)).length;
}

function totalBlockedDistance(board, moveIndex, blockedSet) {
  let total = 0;
  for (const blockedIndex of blockedSet) {
    total += hexDistance(board, moveIndex, blockedIndex);
  }
  return total;
}

function reachableRegionStats(board, startIndex, blockedSet) {
  if (blockedSet.has(startIndex)) {
    return {
      distance: null,
      reachableOpenCount: 0,
      boundaryCount: 0,
      shortestPathCount: 0
    };
  }

  const queue = [startIndex];
  const distances = new Map([[startIndex, 0]]);
  const pathCounts = new Map([[startIndex, 1]]);

  for (let head = 0; head < queue.length; head += 1) {
    const current = queue[head];
    const currentDistance = distances.get(current) ?? 0;
    const currentPathCount = pathCounts.get(current) ?? 0;
    for (const neighborIndex of board.cells[current].neighborIndices) {
      if (blockedSet.has(neighborIndex)) {
        continue;
      }
      const nextDistance = currentDistance + 1;
      if (!distances.has(neighborIndex)) {
        distances.set(neighborIndex, nextDistance);
        pathCounts.set(neighborIndex, currentPathCount);
        queue.push(neighborIndex);
      } else if (distances.get(neighborIndex) === nextDistance) {
        pathCounts.set(neighborIndex, (pathCounts.get(neighborIndex) ?? 0) + currentPathCount);
      }
    }
  }

  const reachableBoundaries = [];
  for (const [cellIndex, distance] of distances.entries()) {
    if (board.cells[cellIndex].isBoundary) {
      reachableBoundaries.push({ cellIndex, distance });
    }
  }

  if (!reachableBoundaries.length) {
    return {
      distance: null,
      reachableOpenCount: distances.size,
      boundaryCount: 0,
      shortestPathCount: 0
    };
  }

  const shortestDistance = Math.min(...reachableBoundaries.map((entry) => entry.distance));
  const shortestPathCount = reachableBoundaries
    .filter((entry) => entry.distance === shortestDistance)
    .reduce((total, entry) => total + (pathCounts.get(entry.cellIndex) ?? 0), 0);

  return {
    distance: shortestDistance,
    reachableOpenCount: distances.size,
    boundaryCount: reachableBoundaries.length,
    shortestPathCount
  };
}

function hasOneTurnCaptureRisk(board, catIndex, blockedSet) {
  if (board.cells[catIndex].isBoundary) {
    return false;
  }
  if (shortestBoundaryPath(board, catIndex, blockedSet) === null) {
    return true;
  }
  for (const actionIndex of legalPlayerActions(board, catIndex, blockedSet)) {
    const blockedAfter = new Set(blockedSet);
    blockedAfter.add(actionIndex);
    if (shortestBoundaryPath(board, catIndex, blockedAfter) === null) {
      return true;
    }
  }
  return false;
}

function compareTuples(left, right) {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    if (leftValue < rightValue) {
      return -1;
    }
    if (leftValue > rightValue) {
      return 1;
    }
  }
  return 0;
}

function normalizeSeed(rawSeed) {
  if (Number.isInteger(rawSeed)) {
    const normalized = rawSeed >>> 0;
    return normalized === 0 ? 1 : normalized;
  }
  const fallback = Date.now() >>> 0;
  return fallback === 0 ? 1 : fallback;
}

function createSeededRandom(seed) {
  let state = normalizeSeed(seed);
  return {
    seed: state,
    next() {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    },
    getState() {
      return state >>> 0;
    }
  };
}

function chooseCatMove(board, catIndex, blockedSet, catPolicy, randomFn = Math.random) {
  const legalMoves = legalCatMoves(board, catIndex, blockedSet);
  if (!legalMoves.length) {
    return {
      chosenMove: null,
      legalMoves,
      evaluatedMoves: []
    };
  }

  if (catPolicy.name === "static") {
    return {
      chosenMove: catIndex,
      legalMoves,
      evaluatedMoves: []
    };
  }

  if (catPolicy.name === "easy") {
    const modeDraw = randomFn();
    const greedyEntries = legalMoves.map((moveIndex) => {
      const distance = shortestBoundaryDistance(board, moveIndex, blockedSet);
      return {
        moveIndex,
        distance,
        scoreTuple: [distance === null ? Number.POSITIVE_INFINITY : distance, moveIndex]
      };
    });
    greedyEntries.sort((left, right) => compareTuples(left.scoreTuple, right.scoreTuple));
    const greedyMove = greedyEntries[0]?.moveIndex ?? legalMoves[0] ?? null;
    const mode = modeDraw < 0.3 ? "greedy" : "random";
    const chosenMove = mode === "greedy"
      ? greedyMove
      : legalMoves[Math.floor(randomFn() * legalMoves.length)] ?? legalMoves[0] ?? null;
    const greedyByMove = new Map(greedyEntries.map((entry) => [entry.moveIndex, entry]));
    return {
      chosenMove,
      legalMoves,
      evaluatedMoves: legalMoves.map((moveIndex) => {
        const greedyEntry = greedyByMove.get(moveIndex);
        return {
          moveIndex,
          distance: greedyEntry?.distance ?? shortestBoundaryDistance(board, moveIndex, blockedSet),
          degree: countOpenNeighborDegree(board, moveIndex, blockedSet),
          blockedDistance: totalBlockedDistance(board, moveIndex, blockedSet),
          mode,
          modeDraw,
          greedyMove,
          scoreTuple: [moveIndex === chosenMove ? 0 : 1, moveIndex]
        };
      })
    };
  }

  const evaluatedMoves = legalMoves.map((moveIndex) => {
    const regionStats = reachableRegionStats(board, moveIndex, blockedSet);
    const distance = regionStats.distance;
    const degree = countOpenNeighborDegree(board, moveIndex, blockedSet);
    const blockedDistance = totalBlockedDistance(board, moveIndex, blockedSet);

    let oneTurnCaptureRisk = null;
    let scoreTuple = [distance === null ? Number.POSITIVE_INFINITY : distance, moveIndex];
    if (catPolicy.name === "hard") {
      oneTurnCaptureRisk = hasOneTurnCaptureRisk(board, moveIndex, blockedSet);
      scoreTuple = [
        distance === 0 ? 0 : 1,
        oneTurnCaptureRisk ? 1 : 0,
        distance === null ? Number.POSITIVE_INFINITY : distance,
        -regionStats.boundaryCount,
        -regionStats.shortestPathCount,
        -regionStats.reachableOpenCount,
        -degree,
        moveIndex
      ];
    }

    return {
      moveIndex,
      distance,
      degree,
      blockedDistance,
      reachableOpenCount: regionStats.reachableOpenCount,
      boundaryCount: regionStats.boundaryCount,
      shortestPathCount: regionStats.shortestPathCount,
      oneTurnCaptureRisk,
      scoreTuple
    };
  });

  evaluatedMoves.sort((left, right) => compareTuples(left.scoreTuple, right.scoreTuple));
  return {
    chosenMove: evaluatedMoves[0]?.moveIndex ?? null,
    legalMoves,
    evaluatedMoves
  };
}

function computeDifficulty(boardRadius, initialBlockedCount, catPolicyLevel) {
  return {
    board_size: boardRadius,
    initial_block_count: initialBlockedCount,
    cat_intelligence: catPolicyLevel,
    overall_score: Number(catPolicyLevel.toFixed(2))
  };
}

function normalizeBlockedIndices(board, rawIndices, catIndex) {
  const valid = [];
  for (const rawValue of rawIndices || []) {
    const index = Number(rawValue);
    if (!Number.isInteger(index) || index < 0 || index >= board.cells.length || index === catIndex) {
      continue;
    }
    valid.push(index);
  }
  return toSortedNumberArray(valid);
}

function normalizeCatPolicy(rawPolicy) {
  const key = String(rawPolicy || "").trim().toLowerCase();
  return CAT_POLICIES[key] || CAT_POLICIES[DEFAULT_CONFIG.catPolicy];
}

function isAutoSetupValue(value) {
  return String(value ?? "").trim().toLowerCase() === "" ||
    [AUTO_SETUP_VALUE, "random"].includes(String(value ?? "").trim().toLowerCase());
}

function validateBoardRadius(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || !ALLOWED_BOARD_RADII.includes(parsed)) {
    throw new Error("boardRadius must be 3 or 4.");
  }
  return parsed;
}

function initialBlockedCountRange(catPolicyName, boardRadius) {
  let policyName = normalizeCatPolicy(catPolicyName).name;
  if (policyName === "static") {
    policyName = "easy";
  }
  const byPolicy = INITIAL_BLOCK_COUNT_RANGES[policyName] || INITIAL_BLOCK_COUNT_RANGES[DEFAULT_CONFIG.catPolicy];
  return byPolicy[validateBoardRadius(boardRadius)];
}

function minWinningFirstActionsForPolicy(catPolicyName) {
  let policyName = normalizeCatPolicy(catPolicyName).name;
  if (policyName === "static") {
    policyName = "easy";
  }
  return MIN_WINNING_FIRST_ACTIONS_BY_POLICY[policyName] ?? DEFAULT_MIN_WINNING_FIRST_ACTIONS;
}

function resolveMinWinningFirstActionsSpec(value, catPolicyName) {
  if (value === null || value === undefined || isAutoSetupValue(value)) {
    return minWinningFirstActionsForPolicy(catPolicyName);
  }
  return Math.max(1, Math.trunc(Number(value)));
}

function resolveBoardRadiusSpec(value, setupRng) {
  if (isAutoSetupValue(value)) {
    return ALLOWED_BOARD_RADII[setupRng.randrange(ALLOWED_BOARD_RADII.length)];
  }
  return validateBoardRadius(value);
}

function resolveInitialBlockedCountSpec(value, boardRadius, catPolicyName, setupRng, boardCellCount) {
  if (isAutoSetupValue(value)) {
    const [minimum, maximum] = initialBlockedCountRange(catPolicyName, boardRadius);
    return setupRng.randrange(minimum, maximum + 1);
  }
  const parsed = Number(value);
  const fallback = initialBlockedCountRange(catPolicyName, boardRadius)[0];
  const count = Number.isInteger(parsed) ? parsed : fallback;
  return Math.max(0, Math.min(boardCellCount - 1, Math.trunc(count)));
}

function solverCatResponses(board, catIndex, blockedSet, catPolicyName) {
  const policyName = normalizeCatPolicy(catPolicyName).name;
  if (policyName === "static") {
    return [catIndex];
  }
  const legalMoves = legalCatMoves(board, catIndex, blockedSet);
  if (!legalMoves.length) {
    return [];
  }
  const projectedPolicy = policyName === "easy" ? "medium" : policyName;
  const decision = chooseCatMove(board, catIndex, blockedSet, normalizeCatPolicy(projectedPolicy));
  return decision.chosenMove === null ? [] : [decision.chosenMove];
}

function rankedSolverActions(board, catIndex, blockedSet, catPolicyName, limit) {
  const legalActions = legalPlayerActions(board, catIndex, blockedSet);
  if (!legalActions.length) {
    return [];
  }

  const currentPath = shortestBoundaryPath(board, catIndex, blockedSet) || [];
  const priorityCells = new Set(currentPath.slice(1));
  for (const cellIndex of currentPath.slice(1)) {
    for (const neighborIndex of board.cells[cellIndex].neighborIndices) {
      priorityCells.add(neighborIndex);
    }
  }
  const catMoveCells = new Set(legalCatMoves(board, catIndex, blockedSet));
  const legalActionSet = new Set(legalActions);
  const orderedCandidates = [];
  for (const cellIndex of Array.from(catMoveCells).sort((a, b) => a - b)) {
    if (legalActionSet.has(cellIndex)) {
      appendUnique(orderedCandidates, cellIndex);
    }
  }
  for (const cellIndex of currentPath.slice(1)) {
    if (legalActionSet.has(cellIndex)) {
      appendUnique(orderedCandidates, cellIndex);
    }
  }
  for (const cellIndex of Array.from(priorityCells).sort((a, b) => a - b)) {
    if (legalActionSet.has(cellIndex)) {
      appendUnique(orderedCandidates, cellIndex);
    }
  }
  for (const cellIndex of legalActions.slice().sort((left, right) => (
    hexDistance(board, catIndex, left) - hexDistance(board, catIndex, right) || left - right
  ))) {
    appendUnique(orderedCandidates, cellIndex);
  }

  const actionsToScore = orderedCandidates.slice(0, Math.min(orderedCandidates.length, Math.max(limit * 2, 12)));
  const scoredActions = actionsToScore.map((actionIndex) => {
    const blockedAfter = new Set(blockedSet);
    blockedAfter.add(actionIndex);
    const pathBeforeCatMove = shortestBoundaryPath(board, catIndex, blockedAfter);
    const catMovesAfter = legalCatMoves(board, catIndex, blockedAfter);
    const immediateTrap = pathBeforeCatMove === null || !catMovesAfter.length;
    const responses = solverCatResponses(board, catIndex, blockedAfter, catPolicyName);
    const responseDistances = responses
      .filter((responseIndex) => !board.cells[responseIndex].isBoundary)
      .map((responseIndex) => shortestBoundaryDistance(board, responseIndex, blockedAfter));
    const bestResponseDistance = immediateTrap || !responseDistances.length
      ? Number.POSITIVE_INFINITY
      : Math.min(...responseDistances.map((distance) => distance === null ? Number.POSITIVE_INFINITY : distance));
    return {
      actionIndex,
      scoreTuple: [
        immediateTrap ? 0 : 1,
        catMoveCells.has(actionIndex) ? 0 : 1,
        priorityCells.has(actionIndex) ? 0 : 1,
        -bestResponseDistance,
        hexDistance(board, catIndex, actionIndex),
        actionIndex
      ]
    };
  });
  scoredActions.sort((left, right) => compareTuples(left.scoreTuple, right.scoreTuple));
  return scoredActions.slice(0, Math.max(1, Math.trunc(limit))).map((entry) => entry.actionIndex);
}

function solveChatNoirInstance(
  board,
  catIndex,
  blockedIndices,
  catPolicyName,
  maxSteps,
  winningFirstActionLimit = null,
  adjacentWinningFirstActionLimit = null
) {
  const searchDepth = Math.min(
    Math.max(0, Math.trunc(maxSteps)),
    SOLVER_MAX_DEPTH_BY_RADIUS[board.radius] ?? 7
  );
  let nodeCount = 0;
  let hitNodeLimit = false;
  let firstAction = null;
  const winningFirstActions = [];
  const adjacentWinningFirstActions = [];
  const searchLimit = winningFirstActionLimit === null
    ? null
    : Math.max(1, Math.trunc(Number(winningFirstActionLimit)));
  const adjacentSearchLimit = adjacentWinningFirstActionLimit === null
    ? null
    : Math.max(0, Math.trunc(Number(adjacentWinningFirstActionLimit)));
  const memo = new Map();

  function canTrap(currentCatIndex, blockedTuple, remainingSteps) {
    nodeCount += 1;
    if (nodeCount > SOLVER_MAX_NODES) {
      hitNodeLimit = true;
      return false;
    }
    const blocked = new Set(blockedTuple);
    if (shortestBoundaryPath(board, currentCatIndex, blocked) === null) {
      return true;
    }
    if (board.cells[currentCatIndex].isBoundary || remainingSteps <= 0) {
      return false;
    }

    const key = `${currentCatIndex}|${blockedTuple.join(",")}|${remainingSteps}`;
    if (memo.has(key)) {
      return memo.get(key);
    }

    for (const actionIndex of rankedSolverActions(
      board,
      currentCatIndex,
      blocked,
      catPolicyName,
      SOLVER_ACTION_BRANCH_LIMIT
    )) {
      if (actionCanTrap(currentCatIndex, blocked, actionIndex, remainingSteps)) {
        memo.set(key, true);
        return true;
      }
    }

    memo.set(key, false);
    return false;
  }

  function actionCanTrap(currentCatIndex, blocked, actionIndex, remainingSteps) {
    const blockedAfter = new Set(blocked);
    blockedAfter.add(actionIndex);
    if (shortestBoundaryPath(board, currentCatIndex, blockedAfter) === null) {
      return true;
    }

    const responses = solverCatResponses(board, currentCatIndex, blockedAfter, catPolicyName);
    if (!responses.length) {
      return true;
    }

    const nextBlockedTuple = toSortedNumberArray(Array.from(blockedAfter));
    for (const nextCatIndex of responses) {
      if (board.cells[nextCatIndex].isBoundary || !canTrap(nextCatIndex, nextBlockedTuple, remainingSteps - 1)) {
        return false;
      }
    }
    return true;
  }

  const rootBlockedTuple = toSortedNumberArray(blockedIndices);
  const rootBlocked = new Set(rootBlockedTuple);
  const adjacentActionSet = new Set(legalCatMoves(board, catIndex, rootBlocked));
  let solvable = false;
  if (shortestBoundaryPath(board, catIndex, rootBlocked) === null) {
    solvable = true;
  } else if (!board.cells[catIndex].isBoundary && searchDepth > 0) {
    const rootActions = rankedSolverActions(
      board,
      catIndex,
      rootBlocked,
      catPolicyName,
      Math.max(1, legalPlayerActions(board, catIndex, rootBlocked).length)
    );
    for (const actionIndex of rootActions) {
      if (actionCanTrap(catIndex, rootBlocked, actionIndex, searchDepth)) {
        winningFirstActions.push(actionIndex);
        if (adjacentActionSet.has(actionIndex)) {
          adjacentWinningFirstActions.push(actionIndex);
        }
        if (firstAction === null) {
          firstAction = actionIndex;
        }
        const hasEnoughTotal = searchLimit === null || winningFirstActions.length >= searchLimit;
        const hasEnoughAdjacent = adjacentSearchLimit === null ||
          adjacentWinningFirstActions.length >= adjacentSearchLimit;
        if (hasEnoughTotal && hasEnoughAdjacent) {
          break;
        }
      }
    }
    solvable = winningFirstActions.length > 0;
  }

  return {
    solvable,
    maxSteps: Math.trunc(maxSteps),
    searchDepth,
    nodesSearched: nodeCount,
    hitNodeLimit,
    branchLimit: SOLVER_ACTION_BRANCH_LIMIT,
    firstAction,
    winningFirstActions,
    winningFirstActionCount: winningFirstActions.length,
    winningFirstActionSearchLimit: searchLimit,
    adjacentWinningFirstActions,
    adjacentWinningFirstActionCount: adjacentWinningFirstActions.length,
    adjacentWinningFirstActionSearchLimit: adjacentSearchLimit
  };
}

function sampleBlockedIndices(
  board,
  catIndex,
  initialBlockedCount,
  setupRng,
  catPolicyName,
  minWinningFirstActions = DEFAULT_MIN_WINNING_FIRST_ACTIONS_SPEC,
  minAdjacentWinningFirstActions = DEFAULT_MIN_ADJACENT_WINNING_FIRST_ACTIONS
) {
  const candidates = board.cells.map((cell) => cell.index).filter((index) => index !== catIndex);
  const cappedCount = Math.max(0, Math.min(candidates.length, Math.trunc(initialBlockedCount)));
  const requiredWinningFirstActions = resolveMinWinningFirstActionsSpec(minWinningFirstActions, catPolicyName);
  const requiredAdjacentWinningFirstActions = Math.max(0, Math.trunc(Number(minAdjacentWinningFirstActions)));
  for (let attempt = 0; attempt < SAMPLE_ATTEMPTS; attempt += 1) {
    const blockedIndices = toSortedNumberArray(setupRng.sample(candidates, cappedCount));
    const blockedSet = new Set(blockedIndices);
    if (!legalCatMoves(board, catIndex, blockedSet).length) {
      continue;
    }
    const escapeDistance = shortestBoundaryDistance(board, catIndex, blockedSet);
    const maxSteps = Math.max(0, board.cells.length - blockedIndices.length - 1);
    const solvability = solveChatNoirInstance(
      board,
      catIndex,
      blockedIndices,
      catPolicyName,
      maxSteps,
      requiredWinningFirstActions,
      requiredAdjacentWinningFirstActions
    );
    if (
      escapeDistance !== null &&
      solvability.solvable &&
      solvability.winningFirstActionCount >= requiredWinningFirstActions &&
      solvability.adjacentWinningFirstActionCount >= requiredAdjacentWinningFirstActions
    ) {
      return blockedIndices;
    }
  }
  throw new Error("Unable to sample a valid Chat Noir board with the requested setup and solution count.");
}

export class EnclosureChatNoirGame {
  constructor(rendererRoot) {
    this.renderer = rendererRoot ? new ChatNoirRenderer(rendererRoot) : null;
    this.config = null;
    this.board = null;
    this.catIndex = null;
    this.blockedIndices = [];
    this.blockedSet = new Set();
    this.stepCount = 0;
    this.lastTurn = null;
    this.terminal = null;
    this.reset({});
  }

  _normalizeConfig(config = {}) {
    const catPolicy = normalizeCatPolicy(config.catPolicy ?? DEFAULT_CONFIG.catPolicy);
    const rngSeed = normalizeSeed(config.rngSeed ?? config.seed);
    const setupSeed = Number.isInteger(config.seed) ? config.seed : rngSeed;
    const setupRng = new PythonRandom(setupSeed);
    const minWinningFirstActions = resolveMinWinningFirstActionsSpec(
      config.minWinningFirstActions ?? DEFAULT_CONFIG.minWinningFirstActions,
      catPolicy.name
    );
    const minAdjacentWinningFirstActions = Math.max(
      0,
      Math.trunc(Number(
        config.minAdjacentWinningFirstActions ?? DEFAULT_CONFIG.minAdjacentWinningFirstActions
      ))
    );

    const rawBlockedIndices = config.initialBlockedIndices ?? config.blockedIndices;
    const boardRadiusSpec = config.boardRadius ?? DEFAULT_CONFIG.boardRadius;
    if (Array.isArray(rawBlockedIndices) && isAutoSetupValue(boardRadiusSpec)) {
      throw new Error("Explicit initialBlockedIndices require a concrete boardRadius.");
    }
    const boardRadius = resolveBoardRadiusSpec(boardRadiusSpec, setupRng);
    const board = buildBoard(boardRadius);
    const catIndex = clampInt(
      config.catIndex,
      0,
      board.cells.length - 1,
      board.centerIndex
    );
    if (board.cells[catIndex].isBoundary) {
      throw new Error("catIndex must be an interior cell, not a boundary cell.");
    }

    let blockedIndices = null;
    if (Array.isArray(rawBlockedIndices)) {
      blockedIndices = normalizeBlockedIndices(board, rawBlockedIndices, catIndex);
    } else {
      const initialBlockedCount = resolveInitialBlockedCountSpec(
        config.initialBlockedCount,
        boardRadius,
        catPolicy.name,
        setupRng,
        board.cells.length
      );
      blockedIndices = sampleBlockedIndices(
        board,
        catIndex,
        initialBlockedCount,
        setupRng,
        catPolicy.name,
        minWinningFirstActions,
        minAdjacentWinningFirstActions
      );
    }

    const blockedSet = new Set(blockedIndices);
    const initialEscapePath = shortestBoundaryPath(board, catIndex, blockedSet);
    if (!initialEscapePath) {
      throw new Error("Initial Chat Noir state must leave the cat at least one path to the boundary.");
    }

    return {
      board,
      boardRadius,
      catPolicy,
      catIndex,
      rngSeed,
      minWinningFirstActions,
      minAdjacentWinningFirstActions,
      initialBlockedIndices: blockedIndices,
      animate: Boolean(config.animate ?? DEFAULT_CONFIG.animate),
      illegalReward: Number(config.illegalReward ?? DEFAULT_CONFIG.illegalReward),
      moveReward: Number(config.moveReward ?? DEFAULT_CONFIG.moveReward),
      trapReward: Number(config.trapReward ?? DEFAULT_CONFIG.trapReward),
      escapePenalty: Number(config.escapePenalty ?? DEFAULT_CONFIG.escapePenalty),
      difficulty: computeDifficulty(boardRadius, blockedIndices.length, catPolicy.level),
      initialEscapePath
    };
  }

  reset(config = {}) {
    this.config = this._normalizeConfig(config);
    this.board = this.config.board;
    this.catIndex = this.config.catIndex;
    this.blockedIndices = this.config.initialBlockedIndices.slice();
    this.blockedSet = new Set(this.blockedIndices);
    this.rng = createSeededRandom(this.config.rngSeed);
    this.stepCount = 0;
    this.lastTurn = null;
    this.terminal = this._deriveTerminalState();

    if (this.renderer) {
      this.renderer.setup(this.board.cells);
      this.renderer.syncState(this.getDebugState());
    }

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
    if (Number.isInteger(action)) {
      return action;
    }
    if (action && typeof action === "object") {
      const candidates = [action.cell_index, action.cellIndex, action.index, action.action];
      for (const candidate of candidates) {
        if (Number.isInteger(candidate)) {
          return candidate;
        }
      }
      const fallback = Number.parseInt(String(candidates[0] ?? ""), 10);
      if (Number.isFinite(fallback)) {
        return fallback;
      }
    }
    return null;
  }

  isLegalMove(cellIndex) {
    return this._validateAction(cellIndex).legal;
  }

  getObservation() {
    return {
      boardRadius: this.config.boardRadius,
      boardCellCount: this.board.cells.length,
      catIndex: this.catIndex,
      blockedIndices: this.blockedIndices.slice(),
      legalActionIndices: legalPlayerActions(this.board, this.catIndex, this.blockedSet),
      boundaryIndices: this.board.boundaryIndices.slice(),
      catPolicy: {
        name: this.config.catPolicy.name,
        level: this.config.catPolicy.level,
        description: this.config.catPolicy.description
      },
      rngSeed: this.config.rngSeed,
      minWinningFirstActions: this.config.minWinningFirstActions,
      minAdjacentWinningFirstActions: this.config.minAdjacentWinningFirstActions,
      escapeDistance: shortestBoundaryDistance(this.board, this.catIndex, this.blockedSet)
    };
  }

  getDebugState() {
    const escapePath = shortestBoundaryPath(this.board, this.catIndex, this.blockedSet);
    return {
      boardRadius: this.config.boardRadius,
      boardCells: this.board.cells.map((cell) => ({
        index: cell.index,
        q: cell.q,
        r: cell.r,
        s: cell.s,
        x: cell.x,
        y: cell.y,
        isBoundary: cell.isBoundary,
        neighborIndices: cell.neighborIndices.slice()
      })),
      boardCellCount: this.board.cells.length,
      boundaryIndices: this.board.boundaryIndices.slice(),
      catIndex: this.catIndex,
      blockedIndices: this.blockedIndices.slice(),
      legalActionIndices: legalPlayerActions(this.board, this.catIndex, this.blockedSet),
      catLegalMoves: legalCatMoves(this.board, this.catIndex, this.blockedSet),
      catPolicy: {
        name: this.config.catPolicy.name,
        level: this.config.catPolicy.level,
        description: this.config.catPolicy.description
      },
      rngSeed: this.config.rngSeed,
      minWinningFirstActions: this.config.minWinningFirstActions,
      minAdjacentWinningFirstActions: this.config.minAdjacentWinningFirstActions,
      rngState: this.rng?.getState?.() ?? null,
      escapePath,
      currentEscapeDistance: escapePath ? escapePath.length - 1 : null,
      initialBlockedCount: this.config.initialBlockedIndices.length,
      initialEscapeDistance: this.config.initialEscapePath.length - 1,
      difficulty: { ...this.config.difficulty },
      stepCount: this.stepCount,
      lastTurn: this.lastTurn ? { ...this.lastTurn } : null,
      terminal: this.terminal ? { ...this.terminal } : null
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

  renderConfig() {
    return {
      boardRadius: this.config.boardRadius,
      catPolicy: {
        name: this.config.catPolicy.name,
        level: this.config.catPolicy.level,
        description: this.config.catPolicy.description
      },
      actionCount: this.board.cells.length
    };
  }

  async step(action) {
    if (this.terminal) {
      return this._illegalResult({
        reason: "already_done",
        action
      });
    }

    const cellIndex = this.decodeAction(action);
    if (!Number.isInteger(cellIndex)) {
      return this._illegalResult({
        reason: "invalid_action",
        action
      });
    }

    const validation = this._validateAction(cellIndex);
    if (!validation.legal) {
      return this._illegalResult({
        reason: validation.reason,
        action,
        cellIndex
      });
    }

    this.stepCount += 1;
    this.blockedSet.add(cellIndex);
    this.blockedIndices = toSortedNumberArray(Array.from(this.blockedSet));

    const catEscapePathBeforeMove = shortestBoundaryPath(this.board, this.catIndex, this.blockedSet);
    const catDecision = chooseCatMove(
      this.board,
      this.catIndex,
      this.blockedSet,
      this.config.catPolicy,
      () => this.rng.next()
    );
    const catIndexBefore = this.catIndex;
    let reward = this.config.moveReward;
    let reason = "move_applied";
    let success = false;
    let done = false;

    if (!catEscapePathBeforeMove) {
      reward = this.config.trapReward;
      reason = "cat_trapped";
      success = true;
      done = true;
      this.terminal = { success: true, reason };
    } else if (catDecision.chosenMove === null) {
      reward = this.config.trapReward;
      reason = "cat_no_legal_move";
      success = true;
      done = true;
      this.terminal = { success: true, reason: "cat_trapped" };
    } else {
      this.catIndex = catDecision.chosenMove;
      if (this.board.cells[this.catIndex].isBoundary) {
        reward = this.config.escapePenalty;
        reason = "cat_escaped";
        success = false;
        done = true;
        this.terminal = { success: false, reason };
      } else {
        const escapePathAfterMove = shortestBoundaryPath(this.board, this.catIndex, this.blockedSet);
        if (!escapePathAfterMove) {
          reward = this.config.trapReward;
          reason = "cat_trapped";
          success = true;
          done = true;
          this.terminal = { success: true, reason };
        }
      }
    }

    const escapePathAfter = shortestBoundaryPath(this.board, this.catIndex, this.blockedSet);
    this.lastTurn = {
      blockedCellIndex: cellIndex,
      catIndexBefore,
      catIndexAfter: this.catIndex,
      catMoveIndex: catDecision.chosenMove,
      catLegalMovesBefore: catDecision.legalMoves.slice(),
      catMoveEvaluations: catDecision.evaluatedMoves.map((entry) => ({
        moveIndex: entry.moveIndex,
        distance: entry.distance,
        degree: entry.degree,
        blockedDistance: entry.blockedDistance,
        reachableOpenCount: entry.reachableOpenCount,
        boundaryCount: entry.boundaryCount,
        shortestPathCount: entry.shortestPathCount,
        oneTurnCaptureRisk: entry.oneTurnCaptureRisk,
        mode: entry.mode,
        modeDraw: entry.modeDraw,
        greedyMove: entry.greedyMove,
        scoreTuple: entry.scoreTuple.slice()
      })),
      escapePathBeforeMove: catEscapePathBeforeMove ? catEscapePathBeforeMove.slice() : null,
      escapePathAfterMove: escapePathAfter ? escapePathAfter.slice() : null
    };

    if (this.renderer) {
      this.renderer.syncState(this.getDebugState());
    }

    return this._buildStateResponse({
      reward,
      done,
      success,
      info: {
        illegal: false,
        reason,
        blockedCellIndex: cellIndex,
        catMoveIndex: catDecision.chosenMove,
        catEscaped: reason === "cat_escaped",
        playerCaptured: reason === "cat_trapped" || reason === "cat_no_legal_move"
      }
    });
  }

  _deriveTerminalState() {
    if (this.board.cells[this.catIndex].isBoundary) {
      return { success: false, reason: "cat_escaped" };
    }
    const escapePath = shortestBoundaryPath(this.board, this.catIndex, this.blockedSet);
    if (!escapePath) {
      return { success: true, reason: "cat_trapped" };
    }
    return null;
  }

  _validateAction(cellIndex) {
    if (!Number.isInteger(cellIndex)) {
      return { legal: false, reason: "invalid_action" };
    }
    if (cellIndex < 0 || cellIndex >= this.board.cells.length) {
      return { legal: false, reason: "out_of_range" };
    }
    if (cellIndex === this.catIndex) {
      return { legal: false, reason: "cat_cell" };
    }
    if (this.blockedSet.has(cellIndex)) {
      return { legal: false, reason: "already_blocked" };
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
    const payload = {
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
    return payload;
  }
}
