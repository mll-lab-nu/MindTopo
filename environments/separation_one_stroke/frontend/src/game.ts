/**
 * TopoBench Game Logic
 * 
 * Core game state management and rules validation
 */

import type { 
  Direction, 
  Vertex, 
  LevelJson, 
  GameState, 
  StepResult, 
  EvaluateResult,
  StateResponse
} from './types';

/**
 * Create a unique key for an edge (order-independent)
 */
function edgeKey(v1: Vertex, v2: Vertex): string {
  // Sort vertices to ensure consistent key regardless of direction
  if (v1.x < v2.x || (v1.x === v2.x && v1.y < v2.y)) {
    return `${v1.x},${v1.y}-${v2.x},${v2.y}`;
  }
  return `${v2.x},${v2.y}-${v1.x},${v1.y}`;
}

/**
 * Check if two vertices are adjacent (manhattan distance = 1)
 */
function isAdjacent(v1: Vertex, v2: Vertex): boolean {
  const dx = Math.abs(v1.x - v2.x);
  const dy = Math.abs(v1.y - v2.y);
  return (dx === 1 && dy === 0) || (dx === 0 && dy === 1);
}

/**
 * Convert direction to delta
 */
function dirToDelta(dir: Direction): Vertex {
  switch (dir) {
    case 'U': return { x: 0, y: 1 };
    case 'D': return { x: 0, y: -1 };
    case 'L': return { x: -1, y: 0 };
    case 'R': return { x: 1, y: 0 };
  }
}

/**
 * Check if adding an edge would create a cycle using Union-Find
 * The stroke must remain a tree (forest) at all times
 */
function wouldCreateCycle(
  existingEdges: Set<string>,
  newEdge: { v1: Vertex, v2: Vertex }
): boolean {
  // Build adjacency list from existing edges
  const adj = new Map<string, string[]>();
  
  const vertexKey = (v: Vertex) => `${v.x},${v.y}`;
  
  // Parse existing edges and build adjacency
  for (const key of existingEdges) {
    const [v1Str, v2Str] = key.split('-');
    const [x1, y1] = v1Str.split(',').map(Number);
    const [x2, y2] = v2Str.split(',').map(Number);
    
    const k1 = `${x1},${y1}`;
    const k2 = `${x2},${y2}`;
    
    if (!adj.has(k1)) adj.set(k1, []);
    if (!adj.has(k2)) adj.set(k2, []);
    adj.get(k1)!.push(k2);
    adj.get(k2)!.push(k1);
  }
  
  const startKey = vertexKey(newEdge.v1);
  const endKey = vertexKey(newEdge.v2);
  
  // If neither vertex is in the graph yet, no cycle possible
  if (!adj.has(startKey) && !adj.has(endKey)) {
    return false;
  }
  
  // If only one is in the graph, no cycle possible
  if (!adj.has(startKey) || !adj.has(endKey)) {
    return false;
  }
  
  // BFS/DFS to check if v1 and v2 are already connected
  const visited = new Set<string>();
  const queue = [startKey];
  visited.add(startKey);
  
  while (queue.length > 0) {
    const current = queue.shift()!;
    
    if (current === endKey) {
      return true; // Already connected, adding edge would create cycle
    }
    
    const neighbors = adj.get(current) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  
  return false;
}

/**
 * Calculate regions using flood-fill on cell grid
 * Two cells are connected if they share an edge not crossed by the stroke
 */
function calculateRegions(
  W: number, 
  H: number, 
  edges: Set<string>
): number[][] {
  const cellW = W - 1;
  const cellH = H - 1;
  const regions: number[][] = Array(cellH).fill(null).map(() => Array(cellW).fill(-1));
  
  let regionId = 0;
  
  // Check if two adjacent cells are separated by the stroke
  const isSeparated = (cell1: {i: number, j: number}, cell2: {i: number, j: number}): boolean => {
    // Cells are adjacent horizontally (cell2 is to the right of cell1)
    if (cell2.i === cell1.i + 1 && cell2.j === cell1.j) {
      // The separating edge is vertical: from (cell1.i+1, cell1.j) to (cell1.i+1, cell1.j+1)
      const v1: Vertex = { x: cell1.i + 1, y: cell1.j };
      const v2: Vertex = { x: cell1.i + 1, y: cell1.j + 1 };
      return edges.has(edgeKey(v1, v2));
    }
    // Cells are adjacent vertically (cell2 is above cell1)
    if (cell2.i === cell1.i && cell2.j === cell1.j + 1) {
      // The separating edge is horizontal: from (cell1.i, cell1.j+1) to (cell1.i+1, cell1.j+1)
      const v1: Vertex = { x: cell1.i, y: cell1.j + 1 };
      const v2: Vertex = { x: cell1.i + 1, y: cell1.j + 1 };
      return edges.has(edgeKey(v1, v2));
    }
    return true; // Not adjacent, treat as separated
  };
  
  // Flood fill
  for (let j = 0; j < cellH; j++) {
    for (let i = 0; i < cellW; i++) {
      if (regions[j][i] === -1) {
        // Start new region
        const queue: Array<{i: number, j: number}> = [{ i, j }];
        regions[j][i] = regionId;
        
        while (queue.length > 0) {
          const cell = queue.shift()!;
          
          // Check 4 neighbors
          const neighbors = [
            { i: cell.i - 1, j: cell.j },
            { i: cell.i + 1, j: cell.j },
            { i: cell.i, j: cell.j - 1 },
            { i: cell.i, j: cell.j + 1 },
          ];
          
          for (const neighbor of neighbors) {
            // Check bounds
            if (neighbor.i < 0 || neighbor.i >= cellW || 
                neighbor.j < 0 || neighbor.j >= cellH) {
              continue;
            }
            
            // Already assigned
            if (regions[neighbor.j][neighbor.i] !== -1) {
              continue;
            }
            
            // Check if separated by stroke
            // Need to check in correct order based on which is "first"
            let sep = false;
            if (neighbor.i === cell.i + 1) {
              sep = isSeparated(cell, neighbor);
            } else if (neighbor.i === cell.i - 1) {
              sep = isSeparated(neighbor, cell);
            } else if (neighbor.j === cell.j + 1) {
              sep = isSeparated(cell, neighbor);
            } else if (neighbor.j === cell.j - 1) {
              sep = isSeparated(neighbor, cell);
            }
            
            if (!sep) {
              regions[neighbor.j][neighbor.i] = regionId;
              queue.push(neighbor);
            }
          }
        }
        
        regionId++;
      }
    }
  }
  
  return regions;
}

/**
 * Main Game class
 */
export class Game {
  private state: GameState;
  private levelJson: LevelJson | null = null;
  
  constructor() {
    // Initialize with empty state
    this.state = this.createEmptyState(4, 4);
  }
  
  private createEmptyState(W: number, H: number): GameState {
    const cells = Array(H - 1).fill(null).map(() => Array(W - 1).fill(null));
    return {
      W,
      H,
      cells,
      stroke: [{ x: 0, y: 0 }],
      edges: new Set<string>(),
      currentPos: { x: 0, y: 0 },
      done: false,
      success: false,
    };
  }
  
  /**
   * Load a level from JSON
   */
  loadLevel(levelJson: LevelJson): void {
    this.levelJson = levelJson;
    const { W, H, cells } = levelJson;
    
    // Validate dimensions
    if (cells.length !== H - 1) {
      throw new Error(`Invalid cells height: expected ${H - 1}, got ${cells.length}`);
    }
    for (let j = 0; j < cells.length; j++) {
      if (cells[j].length !== W - 1) {
        throw new Error(`Invalid cells width at row ${j}: expected ${W - 1}, got ${cells[j].length}`);
      }
    }
    
    this.state = {
      W,
      H,
      cells: cells.map(row => [...row]),
      stroke: [{ x: 0, y: 0 }],
      edges: new Set<string>(),
      currentPos: { x: 0, y: 0 },
      done: false,
      success: false,
    };
  }
  
  /**
   * Reset current level
   */
  reset(): void {
    if (this.levelJson) {
      this.loadLevel(this.levelJson);
    } else {
      this.state = this.createEmptyState(this.state.W, this.state.H);
    }
  }
  
  /**
   * Undo the last step
   */
  undo(): boolean {
    if (this.state.stroke.length <= 1) {
      return false;
    }
    
    // Remove last vertex
    const lastVertex = this.state.stroke.pop()!;
    const prevVertex = this.state.stroke[this.state.stroke.length - 1];
    
    // Remove the edge
    const key = edgeKey(prevVertex, lastVertex);
    this.state.edges.delete(key);
    
    // Update current position
    this.state.currentPos = { ...prevVertex };
    
    // Reset done/success state
    this.state.done = false;
    this.state.success = false;
    delete this.state.failureReason;
    
    return true;
  }
  
  /**
   * Execute a step in the given direction
   */
  stepDir(dir: Direction): StepResult {
    if (this.state.done) {
      return { ok: false, reason: 'DONE_BUT_CONSTRAINT_FAIL' };
    }
    
    const delta = dirToDelta(dir);
    const newPos: Vertex = {
      x: this.state.currentPos.x + delta.x,
      y: this.state.currentPos.y + delta.y,
    };
    
    return this.stepTo(newPos.x, newPos.y);
  }
  
  /**
   * Execute a step to the given position
   */
  stepTo(x: number, y: number): StepResult {
    if (this.state.done) {
      return { ok: false, reason: 'DONE_BUT_CONSTRAINT_FAIL' };
    }
    
    const newPos: Vertex = { x, y };
    
    // Check bounds
    if (x < 0 || x >= this.state.W || y < 0 || y >= this.state.H) {
      return { ok: false, reason: 'OUT_OF_BOUNDS' };
    }
    
    // Check adjacency
    if (!isAdjacent(this.state.currentPos, newPos)) {
      return { ok: false, reason: 'NOT_ADJACENT' };
    }
    
    // Check if this is a backtrack (walking back along the last edge)
    const key = edgeKey(this.state.currentPos, newPos);
    if (this.state.stroke.length >= 2) {
      const prevPos = this.state.stroke[this.state.stroke.length - 2];
      if (prevPos.x === newPos.x && prevPos.y === newPos.y) {
        // This is a backtrack - equivalent to undo
        this.state.stroke.pop();
        this.state.edges.delete(key);
        this.state.currentPos = newPos;
        return { ok: true };
      }
    }
    
    // Check edge reuse (not allowed for non-backtrack moves)
    if (this.state.edges.has(key)) {
      return { ok: false, reason: 'REUSE_EDGE' };
    }
    
    // Check cycle creation
    if (wouldCreateCycle(this.state.edges, { v1: this.state.currentPos, v2: newPos })) {
      return { ok: false, reason: 'CREATES_CYCLE' };
    }
    
    // Valid move - update state
    this.state.edges.add(key);
    this.state.stroke.push(newPos);
    this.state.currentPos = newPos;
    
    // Check if reached end
    if (x === this.state.W - 1 && y === this.state.H - 1) {
      const evaluation = this.evaluate();
      this.state.done = true;
      this.state.success = evaluation.success;
      if (!evaluation.success) {
        this.state.failureReason = 'DONE_BUT_CONSTRAINT_FAIL';
      }
      return { 
        ok: true, 
        done: true, 
        success: evaluation.success 
      };
    }
    
    return { ok: true };
  }
  
  /**
   * Get current game state
   */
  getState(): StateResponse {
    return {
      W: this.state.W,
      H: this.state.H,
      currentPos: { ...this.state.currentPos },
      stroke: this.state.stroke.map(v => ({ ...v })),
      cells: this.state.cells.map(row => [...row]),
      stepCount: this.state.stroke.length - 1,
      done: this.state.done,
      success: this.state.success,
      failureReason: this.state.failureReason,
    };
  }
  
  /**
   * Get the raw edges set (for rendering)
   */
  getEdges(): Set<string> {
    return this.state.edges;
  }
  
  /**
   * Evaluate current state
   */
  evaluate(): EvaluateResult {
    const regions = calculateRegions(this.state.W, this.state.H, this.state.edges);
    
    // Build color -> region mapping and check constraints
    const colorRegions = new Map<string, Set<number>>();
    const regionColors = new Map<number, Set<string>>();
    
    for (let j = 0; j < this.state.H - 1; j++) {
      for (let i = 0; i < this.state.W - 1; i++) {
        const color = this.state.cells[j][i];
        if (color === null) continue;
        
        const rid = regions[j][i];
        
        // Track which regions each color appears in
        if (!colorRegions.has(color)) {
          colorRegions.set(color, new Set());
        }
        colorRegions.get(color)!.add(rid);
        
        // Track which colors appear in each region
        if (!regionColors.has(rid)) {
          regionColors.set(rid, new Set());
        }
        regionColors.get(rid)!.add(color);
      }
    }
    
    // Check constraint A: Same color must be in same region
    for (const [color, rids] of colorRegions.entries()) {
      if (rids.size > 1) {
        return {
          done: this.state.done,
          success: false,
          reason: `Color "${color}" is split across ${rids.size} regions`,
          regions,
        };
      }
    }
    
    // Check constraint B: Different colors must be in different regions
    for (const [rid, colors] of regionColors.entries()) {
      if (colors.size > 1) {
        const colorList = Array.from(colors).join(', ');
        return {
          done: this.state.done,
          success: false,
          reason: `Region ${rid} contains multiple colors: ${colorList}`,
          regions,
        };
      }
    }
    
    // Convert to simple color -> region map
    const colorRegionMap = new Map<string, number>();
    for (const [color, rids] of colorRegions.entries()) {
      colorRegionMap.set(color, Array.from(rids)[0]);
    }
    
    return {
      done: this.state.done,
      success: true,
      reason: 'All constraints satisfied',
      regions,
      colorRegions: colorRegionMap,
    };
  }
  
  /**
   * Get available moves from current position
   */
  getAvailableMoves(): Direction[] {
    if (this.state.done) return [];
    
    const moves: Direction[] = [];
    const dirs: Direction[] = ['U', 'D', 'L', 'R'];
    
    // Get previous position for backtrack check
    const prevPos = this.state.stroke.length >= 2 
      ? this.state.stroke[this.state.stroke.length - 2] 
      : null;
    
    for (const dir of dirs) {
      const delta = dirToDelta(dir);
      const newPos: Vertex = {
        x: this.state.currentPos.x + delta.x,
        y: this.state.currentPos.y + delta.y,
      };
      
      // Check bounds
      if (newPos.x < 0 || newPos.x >= this.state.W || 
          newPos.y < 0 || newPos.y >= this.state.H) {
        continue;
      }
      
      // Allow backtrack (moving to previous position)
      if (prevPos && newPos.x === prevPos.x && newPos.y === prevPos.y) {
        moves.push(dir);
        continue;
      }
      
      // Check edge reuse
      const key = edgeKey(this.state.currentPos, newPos);
      if (this.state.edges.has(key)) {
        continue;
      }
      
      // Check cycle
      if (wouldCreateCycle(this.state.edges, { v1: this.state.currentPos, v2: newPos })) {
        continue;
      }
      
      moves.push(dir);
    }
    
    return moves;
  }
}

// Export utility functions for testing
export { edgeKey, calculateRegions };

