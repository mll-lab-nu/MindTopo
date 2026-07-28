// Hole position sampling with collision avoidance

import { HolePlacement, BoardConfig, HolesConfig, HoleType, HoleShape } from '../types';
import { SeededRandom } from '../utils/RandomUtils';
import { 
  circleFitsInRect, 
  circleFitsInCircle, 
  circlesOverlap 
} from '../utils/MathUtils';

// All available hole shapes
const ALL_HOLE_SHAPES: HoleShape[] = [
  'circle', 'ellipse', 'polygon', 'star', 'keyhole', 
  'slot', 'cross', 'crescent', 'heart'
];

const MIXED2_SHAPES: HoleShape[] = ['circle', 'ellipse', 'polygon', 'keyhole', 'slot'];
const MIXED2_PROBABILITY = 0.25;

/**
 * Samples hole positions using rejection sampling
 * Ensures holes don't overlap and stay within board boundaries
 */
export class HoleSampler {
  private random: SeededRandom;
  private maxAttempts: number;

  constructor(random: SeededRandom, maxAttempts: number = 100) {
    this.random = random;
    this.maxAttempts = maxAttempts;
  }

  /**
   * Generate hole placements for a board
   */
  sampleHoles(
    boardConfig: BoardConfig,
    holesConfig: HolesConfig
  ): HolePlacement[] {
    const placements: HolePlacement[] = [];
    
    // Determine number of holes to generate
    const holeCount = holesConfig.typePlan && holesConfig.typePlan.length > 0
      ? holesConfig.typePlan.length
      : this.random.rangeInt(
        holesConfig.count.min,
        holesConfig.count.max
      );
    
    // Get available hole types
    const availableTypesRaw = holesConfig.types.length > 0 
      ? Array.from(new Set(holesConfig.types))
      : ['hole', 'pit', 'mixed', 'mixed2'] as HoleType[];
    const planTypes = holesConfig.typePlan && holesConfig.typePlan.length > 0
      ? Array.from(new Set([...availableTypesRaw, ...holesConfig.typePlan]))
      : availableTypesRaw;
    const allowMixed2 = planTypes.includes('mixed2');
    const baseTypes = planTypes.filter(type => type !== 'mixed2');
    const mixed2Probability = holesConfig.mixed2Probability ?? MIXED2_PROBABILITY;
    
    // Build a type plan that guarantees each available type appears when possible
    const typePlan = holesConfig.typePlan && holesConfig.typePlan.length > 0
      ? this.normalizeTypePlan(
        holesConfig.typePlan,
        holeCount,
        baseTypes,
        allowMixed2,
        mixed2Probability
      )
      : this.buildTypePlan(baseTypes, holeCount, allowMixed2, mixed2Probability);
    const missingTypes: HoleType[] = [];
    
    // First pass: strict placement
    for (let i = 0; i < holeCount; i++) {
      const desiredType = typePlan[i];
      const result = this.tryPlaceHole(
        boardConfig,
        holesConfig,
        placements,
        baseTypes,
        allowMixed2,
        desiredType
      );
      
      if (result.success && result.placement) {
        placements.push(result.placement);
      } else if (desiredType) {
        missingTypes.push(desiredType);
      }
    }

    // If we didn't reach target, try a relaxed pass (looser spacing, smaller radius)
    if (placements.length < holeCount) {
      const remaining = holeCount - placements.length;
      const relaxedMinDistance = holesConfig.minDistance * 0.6;
      const relaxedEdgeBuffer = holesConfig.edgeBuffer * 0.6;
      const relaxedSizeRange = {
        min: holesConfig.sizeRange.min * 0.8,
        max: holesConfig.sizeRange.max * 0.8
      };

      const relaxedTypePlan = this.buildRelaxedTypePlan(
        baseTypes,
        missingTypes,
        remaining,
        allowMixed2,
        mixed2Probability
      );

      const relaxedConfig: HolesConfig = {
        ...holesConfig,
        minDistance: relaxedMinDistance,
        edgeBuffer: relaxedEdgeBuffer,
        sizeRange: relaxedSizeRange
      };

      for (let i = 0; i < remaining; i++) {
        const desiredType = relaxedTypePlan[i];
        const result = this.tryPlaceHole(
          boardConfig,
          relaxedConfig,
          placements,
          baseTypes,
          allowMixed2,
          desiredType
        );

        if (result.success && result.placement) {
          placements.push(result.placement);
        } else if (desiredType) {
          missingTypes.push(desiredType);
        }
      }
    }
    
    // Final safety: if some types are still missing and we have space, try one last pass
    if (placements.length < holeCount) {
      const remaining = holeCount - placements.length;
      const finalTypes = this.buildRelaxedTypePlan(
        baseTypes,
        missingTypes,
        remaining,
        allowMixed2,
        mixed2Probability
      );

      for (let i = 0; i < remaining; i++) {
        const desiredType = finalTypes[i];
        const result = this.tryPlaceHole(
          boardConfig,
          holesConfig,
          placements,
          baseTypes,
          allowMixed2,
          desiredType
        );

        if (result.success && result.placement) {
          placements.push(result.placement);
        } else if (desiredType) {
          missingTypes.push(desiredType);
        }
      }
    }

    // Aggressive final attempt: further relax spacing/size to approach target count
    if (placements.length < holeCount) {
      const remaining = holeCount - placements.length;
      const aggressiveTypes = this.buildRelaxedTypePlan(
        baseTypes,
        missingTypes,
        remaining,
        allowMixed2,
        mixed2Probability
      );

      const minSize = holesConfig.sizeRange.min * 0.7;
      const maxSize = Math.max(minSize, holesConfig.sizeRange.max * 0.85);
      const aggressiveConfig: HolesConfig = {
        ...holesConfig,
        minDistance: holesConfig.minDistance * 0.45,
        edgeBuffer: holesConfig.edgeBuffer * 0.45,
        sizeRange: {
          min: minSize,
          max: maxSize
        }
      };

      for (let i = 0; i < remaining; i++) {
        const desiredType = aggressiveTypes[i];
        const placement = this.sampleSingleHole(
          boardConfig,
          aggressiveConfig,
          placements,
          baseTypes,
          allowMixed2,
          desiredType,
          3,
          0.9,
          aggressiveConfig.minDistance,
          aggressiveConfig.edgeBuffer
        );

        if (placement) {
          placements.push(placement);
        }
      }
    }
    
    return placements;
  }

  /**
   * Attempt to place a single hole and report success
   */
  private tryPlaceHole(
    boardConfig: BoardConfig,
    holesConfig: HolesConfig,
    existingPlacements: HolePlacement[],
    baseTypes: HoleType[],
    allowMixed2: boolean,
    desiredType?: HoleType
  ): { placement?: HolePlacement; success: boolean } {
    const placement = this.sampleSingleHole(
      boardConfig,
      holesConfig,
      existingPlacements,
      baseTypes,
      allowMixed2,
      desiredType
    );

    if (placement) {
      return { placement, success: true };
    }

    const fallbackMinDistance = holesConfig.minDistance * 0.8;
    const fallbackEdgeBuffer = holesConfig.edgeBuffer * 0.8;
    const fallbackPlacement = this.sampleSingleHole(
      boardConfig,
      holesConfig,
      existingPlacements,
      baseTypes,
      allowMixed2,
      desiredType,
      2,
      0.85,
      fallbackMinDistance,
      fallbackEdgeBuffer
    );

    return {
      placement: fallbackPlacement ?? undefined,
      success: Boolean(fallbackPlacement)
    };
  }

  /**
   * Sample a single hole position
   */
  private sampleSingleHole(
    boardConfig: BoardConfig,
    holesConfig: HolesConfig,
    existingPlacements: HolePlacement[],
    baseTypes: HoleType[],
    allowMixed2: boolean,
    forcedType?: HoleType,
    attemptMultiplier: number = 1,
    radiusScale: number = 1,
    minDistanceOverride?: number,
    edgeBufferOverride?: number
  ): HolePlacement | null {
    const { minDistance, edgeBuffer, sizeRange } = holesConfig;
    
    const attempts = Math.max(1, Math.floor(this.maxAttempts * attemptMultiplier));
    
    for (let attempt = 0; attempt < attempts; attempt++) {
      // Generate random radius (optionally scaled down when relaxing)
      const radius = this.random.range(sizeRange.min, sizeRange.max) * radiusScale;
      
      // Generate random position
      const position = this.generateRandomPosition(
        boardConfig,
        radius,
        edgeBufferOverride ?? edgeBuffer
      );
      
      if (!position) continue;
      
      const { x, y } = position;
      
      // Check for overlaps with existing holes
      // Use larger safety margin for irregular shapes (1.5x radius as bounding circle)
      const safetyFactor = 1.5; // Account for non-circular shapes that may extend beyond radius
      const hasOverlap = existingPlacements.some(existing => 
        circlesOverlap(
          x, y, radius * safetyFactor,
          existing.x, existing.y, existing.radius * safetyFactor,
          minDistanceOverride ?? minDistance
        )
      );
      
      if (!hasOverlap) {
        const mixed2Probability = holesConfig.mixed2Probability ?? MIXED2_PROBABILITY;
        const type = forcedType ?? this.pickTypeForFill(baseTypes, allowMixed2, mixed2Probability);
        const defaultShapePool = holesConfig.shapePool && holesConfig.shapePool.length > 0
          ? holesConfig.shapePool
          : ALL_HOLE_SHAPES;
        const mixed2ShapePool = holesConfig.mixed2ShapePool && holesConfig.mixed2ShapePool.length > 0
          ? holesConfig.mixed2ShapePool
          : MIXED2_SHAPES;
        const shapePool = type === 'mixed2' ? mixed2ShapePool : defaultShapePool;
        const shape = this.random.pick(shapePool);

        // Calculate depth based on type
        const depth = this.calculateDepth(type, boardConfig.thickness);
        
        // Assign random rotation
        const rotation = this.random.angle();
        
        // For mixed: random coverage (25-50%), mixed2: fixed coverage (25%)
        const mixedCoverage = type === 'mixed'
          ? this.random.range(0.25, 0.5)
          : type === 'mixed2'
            ? 0.25
            : undefined;
        
        return { x, y, radius, type, depth, shape, rotation, mixedCoverage };
      }
    }
    
    // Failed to place after max attempts
    return null;
  }

  /**
   * Build a hole type plan that ensures each available type appears when possible
   */
  private normalizeTypePlan(
    typePlan: HoleType[],
    holeCount: number,
    baseTypes: HoleType[],
    allowMixed2: boolean,
    mixed2Probability: number
  ): HoleType[] {
    if (holeCount <= 0) return [];
    const plan = typePlan.filter(Boolean).slice(0, holeCount);

    while (plan.length < holeCount) {
      plan.push(this.pickTypeForFill(baseTypes, allowMixed2, mixed2Probability));
    }

    return this.random.shuffle(plan);
  }

  /**
   * Build a hole type plan that ensures each available type appears when possible
   */
  private buildTypePlan(
    baseTypes: HoleType[],
    holeCount: number,
    allowMixed2: boolean,
    mixed2Probability: number
  ): HoleType[] {
    if (baseTypes.length === 0 || holeCount <= 0) {
      return allowMixed2 ? Array.from({ length: holeCount }, () => 'mixed2') : [];
    }

    const assignments: HoleType[] = [];
    const uniqueTypes = Array.from(new Set(baseTypes));

    const guaranteed = holeCount >= uniqueTypes.length
      ? uniqueTypes
      : uniqueTypes.slice(0, holeCount);
    assignments.push(...guaranteed);

    while (assignments.length < holeCount) {
      assignments.push(this.pickTypeForFill(baseTypes, allowMixed2, mixed2Probability));
    }

    return this.random.shuffle(assignments);
  }

  /**
   * Build a relaxed plan prioritizing any types we failed to place earlier
   */
  private buildRelaxedTypePlan(
    baseTypes: HoleType[],
    missingTypes: HoleType[],
    remaining: number,
    allowMixed2: boolean,
    mixed2Probability: number
  ): HoleType[] {
    if (remaining <= 0) return [];
    if (baseTypes.length === 0) {
      return allowMixed2 ? Array.from({ length: remaining }, () => 'mixed2') : [];
    }

    const assignments: HoleType[] = [];
    const uniqueMissing = Array.from(new Set(missingTypes));

    for (const type of uniqueMissing) {
      if (assignments.length >= remaining) break;
      assignments.push(type);
    }

    while (assignments.length < remaining) {
      assignments.push(this.pickTypeForFill(baseTypes, allowMixed2, mixed2Probability));
    }

    return this.random.shuffle(assignments);
  }

  private pickTypeForFill(
    baseTypes: HoleType[],
    allowMixed2: boolean,
    mixed2Probability: number
  ): HoleType {
    if (allowMixed2 && this.random.chance(mixed2Probability)) {
      return 'mixed2';
    }
    if (baseTypes.length === 0) {
      return allowMixed2 ? 'mixed2' : 'hole';
    }
    return this.random.pick(baseTypes);
  }

  /**
   * Generate a random position within the board bounds
   * Uses safety factor to account for irregular hole shapes
   */
  private generateRandomPosition(
    boardConfig: BoardConfig,
    holeRadius: number,
    edgeBuffer: number
  ): { x: number; y: number } | null {
    const { shape, size } = boardConfig;
    
    // Safety factor for irregular shapes (they may extend beyond the nominal radius)
    const safetyFactor = 1.5;
    const effectiveRadius = holeRadius * safetyFactor;
    
    for (let attempt = 0; attempt < this.maxAttempts; attempt++) {
      let x: number, y: number;
      
      if (shape === 'circle') {
        // For circular boards, sample within a circle
        const boardRadius = Math.min(size.width, size.height) / 2;
        const point = this.random.pointInCircle();
        const maxRadius = boardRadius - effectiveRadius - edgeBuffer;
        
        if (maxRadius <= 0) return null;
        
        x = point.x * maxRadius;
        y = point.y * maxRadius;
        
        // Verify it fits with safety margin
        if (circleFitsInCircle(x, y, effectiveRadius, boardRadius, edgeBuffer)) {
          return { x, y };
        }
      } else {
        // For rectangular/polygon boards, sample within rectangle
        const halfWidth = size.width / 2 - effectiveRadius - edgeBuffer;
        const halfHeight = size.height / 2 - effectiveRadius - edgeBuffer;
        
        if (halfWidth <= 0 || halfHeight <= 0) return null;
        
        x = this.random.range(-halfWidth, halfWidth);
        y = this.random.range(-halfHeight, halfHeight);
        
        // For polygon and irregular shapes, do additional check
        if (shape === 'polygon' || shape === 'irregular' || shape === 'star' || 
            shape === 'cross' || shape === 'l_shape') {
          const polyRadius = Math.min(size.width, size.height) / 2;
          if (!circleFitsInCircle(x, y, effectiveRadius, polyRadius * 0.7, edgeBuffer)) {
            continue;
          }
        }
        
        if (circleFitsInRect(x, y, effectiveRadius, size.width, size.height, edgeBuffer)) {
          return { x, y };
        }
      }
    }
    
    return null;
  }

  /**
   * Calculate hole depth based on type
   */
  private calculateDepth(type: HoleType, boardThickness: number): number {
    switch (type) {
      case 'hole':
        // Through-hole: extends beyond board
        return boardThickness * 2;
      case 'pit':
        // Pit: partial depth (30-70% of thickness)
        return boardThickness * this.random.range(0.3, 0.7);
      case 'mixed':
        // Mixed: full depth, but under-board will be partial
        return boardThickness * 2;
      case 'mixed2':
        // Mixed2: full depth, but cover is centered and fixed
        return boardThickness * 2;
      default:
        return boardThickness * 2;
    }
  }

  /**
   * Validate a set of placements
   */
  validatePlacements(
    placements: HolePlacement[],
    boardConfig: BoardConfig,
    minDistance: number,
    edgeBuffer: number
  ): boolean {
    const { shape, size } = boardConfig;
    
    for (let i = 0; i < placements.length; i++) {
      const hole = placements[i];
      
      // Check boundary
      if (shape === 'circle') {
        const boardRadius = Math.min(size.width, size.height) / 2;
        if (!circleFitsInCircle(hole.x, hole.y, hole.radius, boardRadius, edgeBuffer)) {
          return false;
        }
      } else {
        if (!circleFitsInRect(hole.x, hole.y, hole.radius, size.width, size.height, edgeBuffer)) {
          return false;
        }
      }
      
      // Check overlaps
      for (let j = i + 1; j < placements.length; j++) {
        const other = placements[j];
        if (circlesOverlap(
          hole.x, hole.y, hole.radius,
          other.x, other.y, other.radius,
          minDistance
        )) {
          return false;
        }
      }
    }
    
    return true;
  }
}

/**
 * Create a HoleSampler instance
 */
export function createHoleSampler(random: SeededRandom): HoleSampler {
  return new HoleSampler(random);
}
