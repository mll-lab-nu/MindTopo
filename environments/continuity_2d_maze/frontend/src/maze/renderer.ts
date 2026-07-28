import * as THREE from 'three';
import { parseBarEdge } from './bars';
import { Bar, BarColor, DiagonalSpec, LabeledPoint, MazeGrid, WallSpec } from './types';

// Rendering default for edges that have no shape entry (outer border walls
// and defensive fallback for any closed-but-unshaped internal wall).
const FULL_SPEC: Pick<WallSpec, 'length' | 'offset'> = { length: 1, offset: 0 };

const WORLD_HALF = 1;
const PLOT_HALF = 0.95;
const WALL_COLOR = 0x000000;
const WALL_THICKNESS_FACTOR = 0.12;
// Circle = small filled dot at the cell center (~0.20 cell diameter).
// Letter sprite is rendered in BLACK and placed to the upper-right of the
// dot so the label sits ON the white maze background, not inside the dot.
const CIRCLE_RADIUS_FACTOR = 0.10;
const LETTER_SIZE_FACTOR = 0.28;     // sprite world size relative to cellWorldSize
const LETTER_GAP_FACTOR = 0.02;      // gap between dot edge and sprite center
const LETTER_CANVAS_PX = 256;
const LETTER_FILL = '#000000';

const WALL_Z = 0.01;
const BAR_Z = 0.015;    // above walls, below point circles
const POINT_Z = 0.02;
const LETTER_Z = 0.03;

const POINT_COLORS: number[] = [
  0x2563eb, // A blue
  0xdc2626, // B red
  0x059669, // C green
  0x7c3aed, // D purple
  0xd97706, // E amber
  0x0891b2, // F cyan
  0xdb2777, // G pink
  0x65a30d, // H lime
];

// Q2 uses neutral greys for points so that bar colors (which include blue
// and red) don't collide with point colors. Passed to drawLabels via
// colorOverride parameter.
export const Q2_NEUTRAL_POINT_COLORS: readonly number[] = [0x1f2937, 0x1f2937];

// RGB for each BarColor. Rendered at z=BAR_Z so the bar plane visually covers
// the black wall plane (z=WALL_Z) beneath it at the same edge.
const BAR_COLOR_HEX: Record<BarColor, number> = {
  purple: 0x9333ea,
  red: 0xdc2626,
  green: 0x16a34a,
  blue: 0x2563eb,
  yellow: 0xeab308,
  orange: 0xf97316,
};

const wallMaterial = new THREE.MeshBasicMaterial({ color: WALL_COLOR });

function cellDims(size: number): { cellWorldSize: number; thickness: number } {
  const cellWorldSize = (PLOT_HALF * 2) / size;
  const thickness = Math.max(cellWorldSize * WALL_THICKNESS_FACTOR, 0.006);
  return { cellWorldSize, thickness };
}

function cellCenter(maze: MazeGrid, x: number, y: number): [number, number] {
  const { cellWorldSize } = cellDims(maze.size);
  const wx = -PLOT_HALF + (x + 0.5) * cellWorldSize;
  const wy = PLOT_HALF - (y + 0.5) * cellWorldSize;
  return [wx, wy];
}

// For a wall with relative length L and offset O along its edge, the rendered
// segment length and the center shift from the edge midpoint are:
//   wallLen = L * s  (+ t for L >= 1 to preserve corner overlap)
//   shift   = (O + L/2 - 0.5) * s   (applied along the edge axis)
// Horizontal walls shift along x; vertical walls shift along the y axis where
// offset=0 hugs the top end of the edge, so shiftY uses (0.5 - O - L/2) * s.
function addHorizontalWall(
  group: THREE.Group,
  edgeMidX: number,
  edgeMidY: number,
  s: number,
  t: number,
  spec: Pick<WallSpec, 'length' | 'offset'>,
  material: THREE.Material,
  z: number,
): void {
  const L = spec.length;
  const O = spec.offset;
  const wallLen = L * s + (L >= 1 ? t : 0);
  const shiftX = (O + L / 2 - 0.5) * s;
  const geo = new THREE.PlaneGeometry(wallLen, t);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(edgeMidX + shiftX, edgeMidY, z);
  group.add(mesh);
}

function addVerticalWall(
  group: THREE.Group,
  edgeMidX: number,
  edgeMidY: number,
  s: number,
  t: number,
  spec: Pick<WallSpec, 'length' | 'offset'>,
  material: THREE.Material,
  z: number,
): void {
  const L = spec.length;
  const O = spec.offset;
  const wallLen = L * s + (L >= 1 ? t : 0);
  const shiftY = (0.5 - O - L / 2) * s;
  const geo = new THREE.PlaneGeometry(t, wallLen);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(edgeMidX, edgeMidY + shiftY, z);
  group.add(mesh);
}

export function clearGroup(group: THREE.Group): void {
  for (const child of [...group.children]) {
    group.remove(child);
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const mat = child.material as THREE.Material;
      if (mat !== wallMaterial) {
        mat.dispose();
      }
    } else if (child instanceof THREE.Sprite) {
      const mat = child.material;
      mat.map?.dispose();
      mat.dispose();
    }
  }
}

export function drawMaze(group: THREE.Group, maze: MazeGrid): void {
  const size = maze.size;
  const { cellWorldSize: s, thickness: t } = cellDims(size);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [cx, cy] = cellCenter(maze, x, y);
      const cell = maze.cells[y][x];
      // Outer borders (always full-edge, no shape lookup).
      if (y === 0 && cell.north) {
        addHorizontalWall(group, cx, cy + s / 2, s, t, FULL_SPEC, wallMaterial, WALL_Z);
      }
      if (x === 0 && cell.west) {
        addVerticalWall(group, cx - s / 2, cy, s, t, FULL_SPEC, wallMaterial, WALL_Z);
      }
      // South edge of this cell: internal if not on bottom border.
      if (cell.south) {
        const spec = y + 1 < size ? maze.shapes[`${x},${y},s`] : undefined;
        addHorizontalWall(group, cx, cy - s / 2, s, t, spec ?? FULL_SPEC, wallMaterial, WALL_Z);
      }
      // East edge of this cell: internal if not on right border.
      if (cell.east) {
        const spec = x + 1 < size ? maze.shapes[`${x},${y},e`] : undefined;
        addVerticalWall(group, cx + s / 2, cy, s, t, spec ?? FULL_SPEC, wallMaterial, WALL_Z);
      }
    }
  }

  // Diagonal walls (iter C4c). DiagonalSpec.length/offset are in cell-edge
  // units where the full diagonal = Math.SQRT2. A segment of length L and
  // offset O starts distance O from the "start corner" along the diagonal
  // axis; its midpoint is at distance (O + L/2). Project that distance onto
  // x and y using the unit direction vector, scaled by world s / √2.
  //   slash '/':     direction (+1, +1)/√2, start corner = bottom-left
  //   backslash '\': direction (+1, −1)/√2, start corner = top-left
  for (const [key, spec] of Object.entries(maze.diagonals)) {
    const [xs, ys] = key.split(',');
    const x = parseInt(xs, 10);
    const y = parseInt(ys, 10);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const [cx, cy] = cellCenter(maze, x, y);
    addDiagonalWall(group, cx, cy, s, t, spec, wallMaterial, WALL_Z);
  }
}

function addDiagonalWall(
  group: THREE.Group,
  cx: number,
  cy: number,
  s: number,
  t: number,
  spec: DiagonalSpec,
  material: THREE.Material,
  z: number,
): void {
  const L = spec.length;
  const O = spec.offset;
  // Mirror H/V Mode-A behavior: full diagonals (dA, length = √2 ≈ 1.414)
  // get a +t length extension so adjacent-cell diagonals meeting at a
  // shared corner overlap by t instead of leaving a triangular gap from
  // the rectangle's square endpoints. Partial diagonals (dB/dC, length
  // < 1) keep their nominal length — they don't terminate at corners.
  const worldLen = L * s + (L >= 1 ? t : 0);
  const axisShift = (O + L / 2) * s / Math.SQRT2;
  let midX: number;
  let midY: number;
  let rotZ: number;
  if (spec.orientation === 'slash') {
    midX = cx - s / 2 + axisShift;
    midY = cy - s / 2 + axisShift;
    rotZ = Math.PI / 4;
  } else {
    midX = cx - s / 2 + axisShift;
    midY = cy + s / 2 - axisShift;
    rotZ = -Math.PI / 4;
  }
  const geo = new THREE.PlaneGeometry(worldLen, t);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(midX, midY, z);
  mesh.rotation.z = rotZ;
  group.add(mesh);
}

// Paint each bar as a colored segment on top of its underlying blocking wall.
// Bars reference existing Mode-A walls (H/V) or full dA diagonals
// (placeRandomBars invariant), so the visual segment matches the underlying
// black wall geometry exactly — only the color/z change. `bar.shape` reserved
// for future partial-bar visual distractors.
export function drawBars(group: THREE.Group, maze: MazeGrid, bars: readonly Bar[]): void {
  const { cellWorldSize: s, thickness: t } = cellDims(maze.size);
  for (const bar of bars) {
    const parsed = parseBarEdge(bar);
    const material = new THREE.MeshBasicMaterial({ color: BAR_COLOR_HEX[bar.color] });
    const [cx, cy] = cellCenter(maze, parsed.x, parsed.y);
    if (parsed.kind === 'hv') {
      if (parsed.side === 'east') {
        addVerticalWall(group, cx + s / 2, cy, s, t, FULL_SPEC, material, BAR_Z);
      } else {
        addHorizontalWall(group, cx, cy - s / 2, s, t, FULL_SPEC, material, BAR_Z);
      }
    } else {
      // diag bar — reuse the underlying dA spec for orientation/length so
      // the colored line tracks the actual wall geometry (defensive: if the
      // diagonal entry was already removed for some reason, skip silently).
      const spec = maze.diagonals[bar.edge];
      if (!spec || spec.mode !== 'dA') continue;
      addDiagonalWall(group, cx, cy, s, t, spec, material, BAR_Z);
    }
  }
}

export function drawLabels(
  group: THREE.Group,
  maze: MazeGrid,
  points: LabeledPoint[],
  colorOverride?: readonly number[],
): void {
  const { cellWorldSize, thickness } = cellDims(maze.size);
  const circleRadius = cellWorldSize * CIRCLE_RADIUS_FACTOR;
  const safeDist = cellWorldSize / 2 - thickness / 2;
  if (circleRadius + thickness / 2 >= cellWorldSize / 2) {
    throw new Error(
      `Point circle overlaps walls: radius=${circleRadius.toFixed(4)} >= safe=${safeDist.toFixed(4)}. ` +
        `Check WALL_THICKNESS_FACTOR / CIRCLE_RADIUS_FACTOR.`,
    );
  }
  const palette = colorOverride ?? POINT_COLORS;

  const spriteSize = cellWorldSize * LETTER_SIZE_FACTOR;
  const labelGap = cellWorldSize * LETTER_GAP_FACTOR;
  // Place the sprite center on the upper-right diagonal of the dot so the
  // letter sits clearly adjacent to (and outside of) the dot. World y is up,
  // so +y means up on screen.
  const labelOffset = circleRadius + spriteSize / 2 + labelGap;

  points.forEach((p, i) => {
    const [wx, wy] = cellCenter(maze, p.x, p.y);
    const color = palette[i % palette.length];

    const circleGeo = new THREE.CircleGeometry(circleRadius, 48);
    const circleMat = new THREE.MeshBasicMaterial({ color });
    const circle = new THREE.Mesh(circleGeo, circleMat);
    circle.position.set(wx, wy, POINT_Z);
    group.add(circle);

    const sprite = makeLetterSprite(p.name);
    sprite.scale.set(spriteSize, spriteSize, 1);
    sprite.position.set(wx + labelOffset, wy + labelOffset, LETTER_Z);
    group.add(sprite);
  });
}

function makeLetterSprite(letter: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = LETTER_CANVAS_PX;
  canvas.height = LETTER_CANVAS_PX;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, LETTER_CANVAS_PX, LETTER_CANVAS_PX);
  ctx.fillStyle = LETTER_FILL;
  ctx.font = `bold ${Math.floor(LETTER_CANVAS_PX * 0.75)}px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(letter, LETTER_CANVAS_PX / 2, LETTER_CANVAS_PX / 2 + LETTER_CANVAS_PX * 0.03);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  return new THREE.Sprite(material);
}

export function worldBounds(): { half: number; plotHalf: number } {
  return { half: WORLD_HALF, plotHalf: PLOT_HALF };
}

const HIGHLIGHT_COLOR = 0xffcc00;
// Ring sits just outside the small (radius=0.10 cell) dot so the highlight
// is visible without obscuring the letter to its upper-right.
const HIGHLIGHT_OUTER_FACTOR = 0.20;
const HIGHLIGHT_INNER_FACTOR = 0.13;

export function drawHighlight(
  group: THREE.Group,
  maze: MazeGrid,
  points: LabeledPoint[],
  selectedIndices: readonly number[],
): void {
  clearGroup(group);
  const { cellWorldSize } = cellDims(maze.size);
  const ringOuter = cellWorldSize * HIGHLIGHT_OUTER_FACTOR;
  const ringInner = cellWorldSize * HIGHLIGHT_INNER_FACTOR;
  for (const idx of selectedIndices) {
    const p = points[idx];
    if (!p) continue;
    const [wx, wy] = cellCenter(maze, p.x, p.y);
    const geo = new THREE.RingGeometry(ringInner, ringOuter, 48);
    const mat = new THREE.MeshBasicMaterial({ color: HIGHLIGHT_COLOR });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(wx, wy, 0.04);
    group.add(mesh);
  }
}
