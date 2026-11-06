"""``continuity_pipe`` ("pipe") frame parser + overlay.

Design mirrors the swap2d/untangle spikes: we never read topology from raw pixels beyond
what is needed to recover the discrete board state.  The board is a static ``grid_size`` x
``grid_size`` lattice of dark cells on a white card; every non-empty cell draws a coloured
pipe (green = connected to source, blue = not) as a set of arms radiating from a filled
centre dot, plus the centre dot itself.

Per clip we estimate one :class:`PipeGeometry` (the board never moves) by collecting the
distance-transform peak of each coloured pipe component across all frames — that peak sits
on the centre dot / arm junction, i.e. the true cell centre — and clustering the peaks into
``grid_size`` column and row centres.  Each frame is then parsed against that fixed lattice:
for every node we test the central disk for colour (active vs empty), probe the four cardinal
spokes for arms (``current_mask``), and classify the fill colour as green/blue.  The result is
a discrete ``{index: (mask, colour)}`` board that the metrics layer scores against the env
oracle GT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pipe_oracle import DIR_BITS, pipe_shape

# --- CV thresholds (the one hardcoded-knob spot, like config.py for untangle) --------
GREEN_HUE_LO, GREEN_HUE_HI = 35, 88     # OpenCV hue [0,180); green pipe ~68
BLUE_HUE_LO, BLUE_HUE_HI = 89, 128      # blue pipe ~102
PIPE_SAT_MIN, PIPE_VAL_MIN = 90, 90     # a pipe pixel is saturated+bright
COMPONENT_MIN_AREA_FRAC = 0.0004        # min coloured component area / frame area
CENTER_DISK_FRAC = 0.18                 # radius (as frac of pitch) of the active-test disk
CENTER_ACTIVE_FRAC = 0.10               # colour fraction in the disk to call a node active
ARM_R_LO, ARM_R_HI = 0.22, 0.38         # arm-probe band (frac of pitch) from the centre
ARM_PROBE_SAMPLES = 7
ARM_HIT_FRAC = 0.4                       # colour fraction in a probe window to count a hit
ARM_PRESENT_FRAC = 0.55                  # frac of band samples that must hit for an arm
COLOR_SAMPLE_FRAC = 0.42                 # radius (frac of pitch) of the colour-vote disk


@dataclass(frozen=True)
class PipeGeometry:
    """Clip-level board lattice shared by every sampled frame."""

    grid_size: int
    col_centers: Tuple[float, ...]
    row_centers: Tuple[float, ...]
    pitch: float
    board_bbox_xyxy: Tuple[int, int, int, int]
    reference_frame_ids: Tuple[int, ...] = ()

    def center_xy(self, gx: int, gy: int) -> Tuple[float, float]:
        return float(self.col_centers[gx]), float(self.row_centers[gy])


@dataclass
class PipeCell:
    index: int
    gx: int
    gy: int
    center_xy: Tuple[float, float]
    active: bool
    mask: int                     # current_mask bits (N=1,E=2,S=4,W=8)
    arms: List[str]
    shape: str                    # endpoint/straight/elbow/tee/cross/empty
    color: Optional[str]          # "green" | "blue" | None
    confidence: float


@dataclass
class ParsedPipeFrame:
    grid_size: int
    cells: List[PipeCell]
    active_indices: List[int]
    masks: Dict[int, int]         # index -> current_mask for active cells
    colors: Dict[int, str]        # index -> "green"/"blue" for active cells
    parser_status: str            # "ok" | "grid_not_found"
    warnings: List[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.parser_status == "ok"


# ---------------------------------------------------------------------------
# Colour masks
# ---------------------------------------------------------------------------


def _color_masks(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat_val = (s > PIPE_SAT_MIN) & (v > PIPE_VAL_MIN)
    green = (h >= GREEN_HUE_LO) & (h <= GREEN_HUE_HI) & sat_val
    blue = (h >= BLUE_HUE_LO) & (h <= BLUE_HUE_HI) & sat_val
    color = green | blue
    return color, green, blue


def _pipe_centers(frame_bgr: np.ndarray) -> List[Tuple[float, float]]:
    """Distance-transform peak of each coloured pipe component ~ its cell centre."""
    h, w = frame_bgr.shape[:2]
    color, _green, _blue = _color_masks(frame_bgr)
    mask = cv2.morphologyEx(color.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    min_area = COMPONENT_MIN_AREA_FRAC * h * w
    centers: List[Tuple[float, float]] = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _minv, _maxv, _minl, maxl = cv2.minMaxLoc(dt)
        centers.append((float(maxl[0]), float(maxl[1])))
    return centers


def _kmeans_1d(values: Sequence[float], k: int, n_iter: int = 60) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == 0 or k <= 0:
        return []
    if len(vals) <= k:
        return sorted(float(v) for v in vals)
    lo, hi = float(vals.min()), float(vals.max())
    centers = np.linspace(lo, hi, k)
    for _ in range(n_iter):
        labels = np.abs(vals[:, None] - centers[None, :]).argmin(axis=1)
        new = centers.copy()
        for i in range(k):
            grp = vals[labels == i]
            if len(grp):
                new[i] = float(grp.mean())
        if np.allclose(new, centers):
            break
        centers = new
    return sorted(float(c) for c in centers)


def estimate_pipe_geometry(
    frames_bgr: Sequence[np.ndarray],
    grid_size: int,
    max_reference_frames: int = 40,
) -> Optional[PipeGeometry]:
    """Aggregate coloured-pipe centre dots across frames and cluster into a lattice.

    The board is static, so pooling centres across many frames (each of which sees most
    active cells) yields stable column/row centres even when a single frame is corrupted.
    """
    all_x: List[float] = []
    all_y: List[float] = []
    used: List[int] = []
    for frame_id, frame in enumerate(frames_bgr[:max_reference_frames]):
        centers = _pipe_centers(frame)
        if len(centers) >= max(2, grid_size):
            used.append(frame_id)
            all_x.extend(c[0] for c in centers)
            all_y.extend(c[1] for c in centers)
    if len(all_x) < grid_size or len(all_y) < grid_size:
        return None
    col_centers = _kmeans_1d(all_x, grid_size)
    row_centers = _kmeans_1d(all_y, grid_size)
    if len(col_centers) != grid_size or len(row_centers) != grid_size:
        return None
    pitch_x = float(np.median(np.diff(col_centers))) if grid_size > 1 else 0.0
    pitch_y = float(np.median(np.diff(row_centers))) if grid_size > 1 else 0.0
    pitch = float(np.mean([p for p in (pitch_x, pitch_y) if p > 0]) or max(pitch_x, pitch_y))
    if pitch <= 0:
        return None
    x0 = int(round(col_centers[0] - 0.5 * pitch))
    x1 = int(round(col_centers[-1] + 0.5 * pitch))
    y0 = int(round(row_centers[0] - 0.5 * pitch))
    y1 = int(round(row_centers[-1] + 0.5 * pitch))
    return PipeGeometry(
        grid_size=grid_size,
        col_centers=tuple(col_centers),
        row_centers=tuple(row_centers),
        pitch=pitch,
        board_bbox_xyxy=(x0, y0, x1, y1),
        reference_frame_ids=tuple(used),
    )


# ---------------------------------------------------------------------------
# Per-frame parse
# ---------------------------------------------------------------------------


def _disk_fraction(mask: np.ndarray, cx: int, cy: int, radius: int) -> float:
    h, w = mask.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def _detect_arms(color: np.ndarray, cx: float, cy: float, pitch: float) -> List[str]:
    h, w = color.shape[:2]
    arms: List[str] = []
    for name, dx, dy, _bit in DIR_BITS:
        hits = 0
        total = 0
        for t in np.linspace(ARM_R_LO, ARM_R_HI, ARM_PROBE_SAMPLES):
            sx = int(round(cx + dx * t * pitch))
            sy = int(round(cy + dy * t * pitch))
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            total += 1
            win = color[max(0, sy - 3):sy + 4, max(0, sx - 3):sx + 4]
            if win.size and float(win.mean()) > ARM_HIT_FRAC:
                hits += 1
        if total and hits / total >= ARM_PRESENT_FRAC:
            arms.append(name)
    return arms


def parse_pipe_frame(
    frame_bgr: np.ndarray,
    grid_size: int,
    geometry: Optional[PipeGeometry],
) -> ParsedPipeFrame:
    warnings: List[str] = []
    if geometry is None:
        return ParsedPipeFrame(grid_size, [], [], {}, {}, "grid_not_found", ["no board geometry"])

    color_bool, green_bool, blue_bool = _color_masks(frame_bgr)
    color = color_bool.astype(np.uint8)
    green = green_bool.astype(np.uint8)
    blue = blue_bool.astype(np.uint8)
    pitch = geometry.pitch
    disk_r = max(2, int(round(CENTER_DISK_FRAC * pitch)))
    color_r = max(3, int(round(COLOR_SAMPLE_FRAC * pitch)))

    cells: List[PipeCell] = []
    masks: Dict[int, int] = {}
    colors: Dict[int, str] = {}
    for gy in range(grid_size):
        for gx in range(grid_size):
            cx, cy = geometry.center_xy(gx, gy)
            icx, icy = int(round(cx)), int(round(cy))
            index = gy * grid_size + gx
            active_frac = _disk_fraction(color, icx, icy, disk_r)
            if active_frac < CENTER_ACTIVE_FRAC:
                cells.append(PipeCell(index, gx, gy, (cx, cy), False, 0, [], "empty", None, 0.85))
                continue
            arms = _detect_arms(color, cx, cy, pitch)
            mask = 0
            for name, _dx, _dy, bit in DIR_BITS:
                if name in arms:
                    mask |= bit
            gw = int(green[max(0, icy - color_r):icy + color_r, max(0, icx - color_r):icx + color_r].sum())
            bw = int(blue[max(0, icy - color_r):icy + color_r, max(0, icx - color_r):icx + color_r].sum())
            color_name = "green" if gw >= bw else "blue"
            color_total = gw + bw
            color_conf = (max(gw, bw) / color_total) if color_total else 0.5
            # Arm confidence: a valid pipe has 1..3 arms; 0 arms on an active cell is
            # a parse failure, 4 arms (cross) never occurs in this env.
            n_arms = len(arms)
            arm_conf = 1.0 if 1 <= n_arms <= 3 else 0.25
            conf = float(np.clip(0.5 * arm_conf + 0.5 * color_conf, 0.0, 1.0))
            masks[index] = mask
            colors[index] = color_name
            cells.append(
                PipeCell(index, gx, gy, (cx, cy), True, mask, arms, pipe_shape(mask), color_name, conf)
            )

    active = sorted(masks.keys())
    status = "ok" if active else "grid_not_found"
    if not active:
        warnings.append("no active pipe cells detected")
    return ParsedPipeFrame(grid_size, cells, active, masks, colors, status, warnings)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

_GREEN_BGR = (85, 217, 25)
_BLUE_BGR = (255, 162, 10)


def _banner(img: np.ndarray, label: str, top: int = 0) -> None:
    cv2.rectangle(img, (0, top), (img.shape[1], top + 24), (25, 25, 25), -1)
    cv2.putText(img, label, (6, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_cell(img: np.ndarray, cell: PipeCell, pitch: float, *, source_index: Optional[int] = None,
               highlight: Optional[Tuple[int, int, int]] = None) -> None:
    cx, cy = int(round(cell.center_xy[0])), int(round(cell.center_xy[1]))
    half = int(round(0.5 * pitch))
    box_color = (70, 70, 70)
    if cell.active:
        box_color = _GREEN_BGR if cell.color == "green" else _BLUE_BGR
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), box_color, 1)
    if cell.active:
        for name, dx, dy, bit in DIR_BITS:
            if cell.mask & bit:
                ex = int(round(cx + dx * 0.38 * pitch))
                ey = int(round(cy + dy * 0.38 * pitch))
                cv2.line(img, (cx, cy), (ex, ey), box_color, 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5 if source_index != cell.index else 8, box_color, -1)
        tag = f"{cell.index}:{cell.shape[:3]}"
        cv2.putText(img, tag, (cx - half + 2, cy - half + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if highlight is not None:
        cv2.rectangle(img, (cx - half - 3, cy - half - 3), (cx + half + 3, cy + half + 3), highlight, 3)


def draw_pipe_overlay(
    frame_bgr: np.ndarray,
    parsed: ParsedPipeFrame,
    geometry: Optional[PipeGeometry],
    label: str,
    *,
    source_index: Optional[int] = None,
    highlight_indices: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    img = frame_bgr.copy()
    pitch = geometry.pitch if geometry is not None else 60.0
    highlight_indices = highlight_indices or {}
    for cell in parsed.cells:
        _draw_cell(img, cell, pitch, source_index=source_index, highlight=highlight_indices.get(cell.index))
    _banner(img, label)
    return img


def draw_pipe_transition_overlay(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    prev: ParsedPipeFrame,
    cur: ParsedPipeFrame,
    geometry: Optional[PipeGeometry],
    label: str,
    *,
    source_index: Optional[int] = None,
    changed_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    changed = {int(i): (0, 215, 255) for i in (changed_indices or [])}
    left = draw_pipe_overlay(prev_bgr, prev, geometry, "prev", source_index=source_index)
    right = draw_pipe_overlay(cur_bgr, cur, geometry, "cur", source_index=source_index,
                              highlight_indices=changed)
    if left.shape[0] != right.shape[0]:
        hgt = min(left.shape[0], right.shape[0])
        left, right = left[:hgt], right[:hgt]
    combined = np.hstack([left, right])
    _banner(combined, label)
    return combined
