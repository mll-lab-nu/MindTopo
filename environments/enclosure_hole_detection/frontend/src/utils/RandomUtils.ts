// Seeded random number generator utilities
import seedrandom from 'seedrandom';

export class SeededRandom {
  private rng: seedrandom.PRNG;
  private seed: number | string;

  constructor(seed: number | string = Date.now()) {
    this.seed = seed;
    this.rng = seedrandom(String(seed));
  }

  /**
   * Reset the RNG with a new seed
   */
  reset(seed?: number | string): void {
    if (seed !== undefined) {
      this.seed = seed;
    }
    this.rng = seedrandom(String(this.seed));
  }

  /**
   * Get the current seed
   */
  getSeed(): number | string {
    return this.seed;
  }

  /**
   * Generate a random number between 0 and 1
   */
  random(): number {
    return this.rng();
  }

  /**
   * Generate a random number between min and max (inclusive)
   */
  range(min: number, max: number): number {
    return min + this.rng() * (max - min);
  }

  /**
   * Generate a random integer between min and max (inclusive)
   */
  rangeInt(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }

  /**
   * Generate a random angle in radians (0 to 2π)
   */
  angle(): number {
    return this.rng() * Math.PI * 2;
  }

  /**
   * Pick a random element from an array
   */
  pick<T>(array: T[]): T {
    return array[Math.floor(this.rng() * array.length)];
  }

  /**
   * Shuffle an array in place (Fisher-Yates)
   */
  shuffle<T>(array: T[]): T[] {
    const result = [...array];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }

  /**
   * Generate random boolean with given probability of true
   */
  chance(probability: number = 0.5): boolean {
    return this.rng() < probability;
  }

  /**
   * Generate Gaussian (normal) distributed random number
   * Uses Box-Muller transform
   */
  gaussian(mean: number = 0, stdDev: number = 1): number {
    const u1 = this.rng();
    const u2 = this.rng();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev + mean;
  }

  /**
   * Generate a random point on a unit circle
   */
  pointOnCircle(): { x: number; y: number } {
    const angle = this.angle();
    return {
      x: Math.cos(angle),
      y: Math.sin(angle)
    };
  }

  /**
   * Generate a random point inside a unit circle
   */
  pointInCircle(): { x: number; y: number } {
    // Rejection sampling for uniform distribution
    let x, y;
    do {
      x = this.range(-1, 1);
      y = this.range(-1, 1);
    } while (x * x + y * y > 1);
    return { x, y };
  }
}

// Global instance for convenience
let globalRandom: SeededRandom | null = null;

export function initGlobalRandom(seed: number | string): SeededRandom {
  globalRandom = new SeededRandom(seed);
  return globalRandom;
}

export function getGlobalRandom(): SeededRandom {
  if (!globalRandom) {
    globalRandom = new SeededRandom();
  }
  return globalRandom;
}

