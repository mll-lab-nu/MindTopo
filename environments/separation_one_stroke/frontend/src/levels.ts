/**
 * TopoBench built-in levels.
 *
 * Difficulty increases primarily by:
 * - cell-grid size: 2 through 5
 * - number of colors: 2 or 3
 * - number of colored cells per color
 *
 * Within each size band, the puzzle layouts intentionally vary so the
 * solution path shape is not just one template with denser color fills.
 *
 * Each level defines:
 * - W, H: vertex grid dimensions
 * - cells: (H-1) x (W-1) color matrix (null = empty)
 * - solution: one known valid direction sequence
 */

import type { LevelJson } from './types';

export const LEVELS: LevelJson[] = [
  {
    W: 3,
    H: 3,
    cells: [
      ['red', null],
      [null, 'blue'],
    ],
    solution: ['U', 'R', 'U', 'R'],
  },
  {
    W: 3,
    H: 3,
    cells: [
      ['red', null],
      ['red', 'blue'],
    ],
    solution: ['R', 'U', 'U', 'R'],
  },
  {
    W: 3,
    H: 3,
    cells: [
      [null, 'blue'],
      ['red', 'blue'],
    ],
    solution: ['U', 'R', 'U', 'R'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      ['red', null, 'red'],
      [null, null, null],
      [null, 'blue', 'blue'],
    ],
    solution: ['U', 'R', 'R', 'R', 'U', 'U'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      [null, null, 'red'],
      ['blue', 'blue', null],
      ['blue', null, 'red'],
    ],
    solution: ['R', 'U', 'R', 'U', 'U', 'R'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      ['red', 'blue', 'blue'],
      ['red', 'blue', null],
      ['red', 'red', 'blue'],
    ],
    solution: ['R', 'U', 'U', 'R', 'U', 'R'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      ['red', 'green', null],
      [null, 'green', null],
      ['blue', null, 'blue'],
    ],
    solution: ['R', 'U', 'L', 'U', 'R', 'R', 'R', 'U'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      ['blue', null, 'red'],
      [null, 'red', null],
      ['green', 'green', 'red'],
    ],
    solution: ['R', 'U', 'L', 'U', 'R', 'R', 'U', 'R'],
  },
  {
    W: 4,
    H: 4,
    cells: [
      ['green', null, 'red'],
      ['green', null, 'green'],
      [null, 'green', 'blue'],
    ],
    solution: ['R', 'R', 'U', 'R', 'U', 'L', 'U', 'R'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['blue', 'blue', null, 'blue'],
      [null, null, null, null],
      ['red', 'red', null, null],
      [null, null, null, 'blue'],
    ],
    solution: ['U', 'U', 'R', 'R', 'U', 'U', 'R', 'R'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['red', null, 'blue', null],
      [null, 'red', 'blue', 'blue'],
      [null, 'red', null, 'blue'],
      ['red', null, 'red', null],
    ],
    solution: ['R', 'U', 'R', 'U', 'R', 'U', 'R', 'U'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['red', 'red', 'red', 'red'],
      ['blue', 'blue', 'blue', 'blue'],
      ['blue', null, 'blue', null],
      ['blue', null, 'blue', null],
    ],
    solution: ['U', 'R', 'R', 'R', 'R', 'U', 'U', 'U'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['red', 'green', null, 'green'],
      [null, 'green', null, 'green'],
      ['blue', null, 'green', null],
      ['blue', null, null, null],
    ],
    solution: ['R', 'U', 'L', 'U', 'R', 'U', 'U', 'R', 'R', 'R'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['blue', null, 'red', null],
      ['red', null, 'red', null],
      ['green', 'red', null, 'red'],
      ['green', 'green', null, 'red'],
    ],
    solution: ['R', 'U', 'L', 'U', 'R', 'U', 'R', 'U', 'R', 'R'],
  },
  {
    W: 5,
    H: 5,
    cells: [
      ['green', 'blue', 'blue', 'blue'],
      ['blue', null, null, null],
      ['red', 'red', null, 'blue'],
      ['red', 'red', 'blue', 'blue'],
    ],
    solution: ['R', 'U', 'L', 'U', 'R', 'R', 'U', 'U', 'R', 'R'],
  },
  {
    W: 6,
    H: 6,
    cells: [
      ['red', null, 'red', null, 'red'],
      [null, 'red', null, 'red', null],
      ['red', null, 'red', null, 'red'],
      [null, null, null, null, null],
      [null, 'blue', 'blue', 'blue', null],
    ],
    solution: ['U', 'U', 'U', 'U', 'R', 'R', 'R', 'R', 'R', 'U'],
  },
  {
    W: 6,
    H: 6,
    cells: [
      [null, 'blue', null, 'blue', null],
      ['red', 'blue', null, 'blue', null],
      ['red', 'blue', null, 'blue', null],
      ['red', 'blue', null, 'blue', null],
      [null, 'red', 'blue', null, 'blue'],
    ],
    solution: ['U', 'R', 'U', 'U', 'U', 'R', 'U', 'R', 'R', 'R'],
  },
  {
    W: 6,
    H: 6,
    cells: [
      [null, null, null, null, 'blue'],
      [null, 'red', 'red', 'blue', 'blue'],
      ['red', 'red', 'red', 'red', 'blue'],
      ['red', 'red', 'red', 'red', 'blue'],
      [null, 'red', 'red', null, null],
    ],
    solution: ['R', 'R', 'R', 'U', 'U', 'R', 'U', 'U', 'R', 'U'],
  },
  {
    W: 6,
    H: 6,
    cells: [
      ['blue', 'blue', 'red', 'red', 'red'],
      ['blue', 'blue', 'blue', 'blue', 'red'],
      ['blue', 'blue', 'blue', null, 'red'],
      ['blue', 'blue', 'blue', null, 'red'],
      ['blue', null, 'blue', null, 'red'],
    ],
    solution: ['R', 'R', 'U', 'R', 'R', 'U', 'U', 'U', 'R', 'U'],
  },
  {
    W: 6,
    H: 6,
    cells: [
      [null, 'blue', 'blue', 'blue', 'blue'],
      ['red', null, 'red', null, 'blue'],
      ['red', null, 'red', 'red', 'red'],
      ['red', 'red', 'red', 'red', 'red'],
      ['red', 'red', 'red', 'red', 'red'],
    ],
    solution: ['R', 'U', 'R', 'R', 'R', 'U', 'R', 'U', 'U', 'U'],
  },
];

/**
 * Get a level by index (1-based for user display, 0-based internally)
 */
export function getLevel(index: number): LevelJson | null {
  if (index < 0 || index >= LEVELS.length) {
    return null;
  }
  return LEVELS[index];
}

/**
 * Get total number of levels
 */
export function getLevelCount(): number {
  return LEVELS.length;
}
