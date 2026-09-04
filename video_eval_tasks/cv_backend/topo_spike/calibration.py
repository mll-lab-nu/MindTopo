"""Board calibration: detect empty holes -> fit a regular lattice -> cell<->pixel mapping.

The rope topology is never computed from pixels; calibration exists only to quantize
detected rope endpoints onto (row, col) holes. Empty holes have a stable HSV signature
in the rendering (measured V approx 141, S approx 76), which is used to detect them as
anchors. RANSAC then fits gs equidistant nodes along each axis (robust to occupied holes
and a few false detections).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import CV


@dataclass
class Lattice:
    """Affine mapping from row/col to pixels (axis-aligned, equidistant).

    Rendering axis convention (measured): image horizontal x maps to grid ROW, image
    vertical y maps to grid COL (in the board annotations, "row" is on the top edge and
    "col" is on the left edge).
    """

    grid_size: int
    origin_x: float
    step_x: float                # step along x (row axis)
    origin_y: float
    step_y: float                # step along y (col axis)
    n_holes_detected: int
    residual_px: float           # median residual from detected anchors to fitted nodes; calibration quality metric

    def cell_to_px(self, row: int, col: int) -> Tuple[float, float]:
        return self.origin_x + row * self.step_x, self.origin_y + col * self.step_y

    def px_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        row = int(round((x - self.origin_x) / self.step_x))
        col = int(round((y - self.origin_y) / self.step_y))
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size


@dataclass
class PointLattice:
    """Lattice that uses measured visible hole centers when available."""

    grid_size: int
    origin_x: float
    step_x: float
    origin_y: float
    step_y: float
    n_holes_detected: int
    residual_px: float
    measured_nodes: Dict[Tuple[int, int], Tuple[float, float]]
    homography: Optional[np.ndarray] = None

    def cell_to_px(self, row: int, col: int) -> Tuple[float, float]:
        measured = self.measured_nodes.get((row, col))
        if measured is not None:
            return measured
        if self.homography is not None:
            pt = np.array([[[float(row), float(col)]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(pt, self.homography)
            return float(dst[0, 0, 0]), float(dst[0, 0, 1])
        return self.origin_x + row * self.step_x, self.origin_y + col * self.step_y

    def px_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        best = None
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                nx, ny = self.cell_to_px(r, c)
                d = float(np.hypot(x - nx, y - ny))
                if best is None or d < best[0]:
                    best = (d, r, c)
        assert best is not None
        return best[1], best[2]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size


def detect_empty_holes(img_bgr: np.ndarray, grid_size: int) -> np.ndarray:
    """Return empty hole centers as Nx2 (x, y). Based on HSV signature plus circular connected-component filtering."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    _, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (val > CV.hole_v_min)
        & (val < CV.hole_v_max)
        & (sat > CV.hole_s_min)
        & (sat < CV.hole_s_max)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    h, w = img_bgr.shape[:2]
    pitch = min(h, w) / (grid_size + 1.0)
    rad_lo, rad_hi = CV.hole_rad_lo_frac * pitch, CV.hole_rad_hi_frac * pitch
    pts = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        rad = (bw + bh) / 4.0
        aspect = bw / max(bh, 1)
        fill = area / max(bw * bh, 1)
        if not (rad_lo <= rad <= rad_hi):
            continue
        if aspect < 1.0 / CV.hole_max_aspect or aspect > CV.hole_max_aspect:
            continue
        if fill < CV.hole_min_circularity:
            continue
        pts.append(cent[i])
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def _step(lattice: Lattice) -> float:
    return (lattice.step_x + lattice.step_y) / 2.0


def _fit_cell_homography(
    measured: Dict[Tuple[int, int], Tuple[float, float]],
    step: float,
) -> Optional[np.ndarray]:
    if len(measured) < 4:
        return None
    items = list(measured.items())
    src = np.array([[float(r), float(c)] for (r, c), _ in items], dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array([[float(x), float(y)] for _, (x, y) in items], dtype=np.float64).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, max(0.3 * step, 4.0))
    return homography


def with_measured_holes(
    img_bgr: np.ndarray,
    grid_size: int,
    lattice: Lattice,
) -> PointLattice:
    """Use detected hole centers for visible cells and homography for hidden cells."""
    holes = detect_empty_holes(img_bgr, grid_size)
    if len(holes) == 0:
        return PointLattice(
            lattice.grid_size,
            lattice.origin_x,
            lattice.step_x,
            lattice.origin_y,
            lattice.step_y,
            lattice.n_holes_detected,
            lattice.residual_px,
            {},
        )

    step = _step(lattice)
    by_cell: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for x, y in holes:
        best = None
        for r in range(grid_size):
            for c in range(grid_size):
                nx, ny = lattice.cell_to_px(r, c)
                d = float(np.hypot(x - nx, y - ny))
                if best is None or d < best[0]:
                    best = (d, r, c)
        assert best is not None
        d, r, c = best
        if d > 0.45 * step:
            continue
        old = by_cell.get((r, c))
        if old is None or d < old[0]:
            by_cell[(r, c)] = (d, float(x), float(y))

    measured = {cell: (x, y) for cell, (_, x, y) in by_cell.items()}
    residual = float(np.median([d for d, _, _ in by_cell.values()])) if by_cell else lattice.residual_px
    return PointLattice(
        lattice.grid_size,
        lattice.origin_x,
        lattice.step_x,
        lattice.origin_y,
        lattice.step_y,
        len(measured),
        residual,
        measured,
        _fit_cell_homography(measured, step),
    )


def _fit_origin(values: np.ndarray, grid_size: int, step: float) -> float:
    """With step fixed, do a 1D search over origin so the most anchors land on the gs equidistant nodes.

    The search range allows the first node to sit several cells ahead of the smallest
    observed point, because in generated frames an entire edge can be hidden by rope
    endpoints, so looking only at empty holes often misses row/col 0.
    """
    vals = np.sort(values)
    span = (grid_size - 1) * step
    lo = vals.min() - span - step * 0.25
    hi = vals.max() + step * 0.25
    best: Optional[Tuple[Tuple[int, float], float]] = None
    for origin in np.linspace(lo, hi, 24):
        grid = origin + np.arange(grid_size) * step
        dist = np.abs(vals[:, None] - grid[None, :]).min(axis=1)
        inliers = int((dist < step * CV.lattice_inlier_frac).sum())
        outside = int(((vals < grid[0] - step * CV.lattice_inlier_frac) |
                       (vals > grid[-1] + step * CV.lattice_inlier_frac)).sum())
        key = (-inliers, outside, float(np.median(dist)))
        if best is None or key < best[0]:
            best = (key, float(origin))
    assert best is not None
    return best[1]


def _fit_origin_tight(values: np.ndarray, grid_size: int, step: float) -> float:
    """Tighter legacy origin fit for clean high-resolution screenshots."""
    vals = np.sort(values)
    span = (grid_size - 1) * step
    lo = vals.min() - step * 0.5
    hi = max(vals.max() - span + step * 0.5, lo + step)
    best: Optional[Tuple[Tuple[int, float], float]] = None
    for origin in np.linspace(lo, hi, 120):
        grid = origin + np.arange(grid_size) * step
        dist = np.abs(vals[:, None] - grid[None, :]).min(axis=1)
        inliers = int((dist < step * CV.lattice_inlier_frac).sum())
        key = (-inliers, float(np.median(dist)))
        if best is None or key < best[0]:
            best = (key, float(origin))
    assert best is not None
    return best[1]


def _estimate_square_step(pts: np.ndarray, pitch: float) -> float:
    """Old square-grid estimate: nearest-neighbor distance in 2D."""
    if len(pts) < 2:
        return pitch
    nn = []
    for i in range(len(pts)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        nn.append(d.min())
    step = float(np.median(nn))
    lo, hi = CV.lattice_step_lo_frac * pitch, CV.lattice_step_hi_frac * pitch
    return min(max(step, lo), hi)


def _fit_axis(values: np.ndarray, grid_size: int, pitch: float) -> Tuple[float, float]:
    """Fit one lattice axis independently.

    Generated videos often stretch the board differently along x/y.  A single
    square step underfits those frames and can shift endpoint labels by a whole
    hole, so we search each axis independently.
    """
    vals = np.asarray(values, dtype=float)
    lo, hi = CV.lattice_step_lo_frac * pitch, CV.lattice_step_hi_frac * pitch
    best: Optional[Tuple[Tuple[int, int, float, float], Tuple[float, float]]] = None
    for step in np.linspace(lo, hi, 24):
        origin = _fit_origin(vals, grid_size, float(step))
        grid = origin + np.arange(grid_size) * step
        dist = np.abs(vals[:, None] - grid[None, :]).min(axis=1)
        inliers = int((dist < step * CV.lattice_inlier_frac).sum())
        outside = int(((vals < grid[0] - step * CV.lattice_inlier_frac) |
                       (vals > grid[-1] + step * CV.lattice_inlier_frac)).sum())
        key = (-inliers, outside, float(np.median(dist)), abs(float(step) - pitch))
        if best is None or key < best[0]:
            best = (key, (float(origin), float(step)))
    assert best is not None
    return best[1]


def _residual(anchors: np.ndarray, grid_size: int, ox: float, sx: float, oy: float, sy: float) -> float:
    residuals = []
    for x, y in anchors:
        row = round((x - ox) / sx)
        col = round((y - oy) / sy)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            residuals.append(abs(x - (ox + row * sx)) + abs(y - (oy + col * sy)))
    return float(np.median(residuals)) if residuals else float("inf")


def fit_lattice(anchors: np.ndarray, grid_size: int, img_shape: Tuple[int, int]) -> Optional[Lattice]:
    """Fit a **square** lattice from anchors (empty holes plus occupied peg positions).

    Using occupied pegs as anchors as well removes the +/-1 cell ambiguity in the origin
    when "all holes along an entire edge are occupied" (empty holes alone cannot locate an
    occupied edge column/row).
    """
    if len(anchors) < grid_size:
        return None
    h, w = img_shape[:2]
    pitch = min(h, w) / (grid_size + 1.0)
    if min(h, w) >= 900:
        step = _estimate_square_step(anchors, pitch)
        ox = _fit_origin_tight(anchors[:, 0], grid_size, step)
        oy = _fit_origin_tight(anchors[:, 1], grid_size, step)
        residual = _residual(anchors, grid_size, ox, step, oy, step)
        return Lattice(grid_size, ox, step, oy, step, len(anchors), residual)

    ax, step_x = _fit_axis(anchors[:, 0], grid_size, pitch)
    ay, step_y = _fit_axis(anchors[:, 1], grid_size, pitch)
    aniso_res = _residual(anchors, grid_size, ax, step_x, ay, step_y)

    square_step = _estimate_square_step(anchors, pitch)
    sx = _fit_origin(anchors[:, 0], grid_size, square_step)
    sy = _fit_origin(anchors[:, 1], grid_size, square_step)
    square_res = _residual(anchors, grid_size, sx, square_step, sy, square_step)

    ratio = abs(step_x - step_y) / max((step_x + step_y) / 2.0, 1.0)
    if ratio > 0.10 and aniso_res <= square_res * 1.35:
        return Lattice(grid_size, ax, step_x, ay, step_y, len(anchors), aniso_res)
    return Lattice(grid_size, sx, square_step, sy, square_step, len(anchors), square_res)


def calibrate(img_bgr: np.ndarray, grid_size: int) -> Optional[Lattice]:
    """Calibrate using empty holes only (for debugging/visualization; the parsing path uses fit_lattice with peg anchors)."""
    pts = detect_empty_holes(img_bgr, grid_size)
    return fit_lattice(pts, grid_size, img_bgr.shape[:2])
