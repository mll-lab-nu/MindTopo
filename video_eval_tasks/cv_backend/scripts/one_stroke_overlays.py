"""One Stroke — overlay generator for visual inspection.

Reads one clean/reference frame plus 3 generated keyframes per episode:
  clean / gen00 / genMid / genEnd.

The clean frame is resolved in this order:
  1. original input screenshot from a run artifact root
  2. freshly rendered screenshot from the One Stroke environment
  3. MP4 first frame as a proxy fallback

The source is recorded in episodes.json as clean_source.
Detects:
  - Colored grid cells
  - White stroke path
  - Start (green blob) and end (red dome) markers

Key design: start/end markers are identified by σ-trimming the blob-center
cluster.  Blobs outside the core grid region are classified as markers rather
than cells, which correctly handles all grid sizes (3×3 → 5×5).

Legacy output, if this standalone script is run:
  overlays_ablation_measured/one_stroke_keyframes_legacy/<size>_seed<NNN>_<frame>.png

Current reports use video_cv_pipeline.py instead, under
one_stroke_video_cv/static/ and one_stroke_video_cv/dynamic/.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = REPO_ROOT / "logs" / "video_eval" / "separation_one_stroke"
OUT_DIR = (
    REPO_ROOT
    / "logs/cv_backend/reports/overlays_ablation_measured/one_stroke_keyframes_legacy"
)
FRAME_TYPES = ("clean", "gen00", "genMid", "genEnd")
SOURCE_JSONL = REPO_ROOT / "logs/video_eval/one_stroke.jsonl"
ONE_STROKE_GYM_DIR = REPO_ROOT / "environments/separation_one_stroke/gym"

# If the original run directory is available, point ONE_STROKE_CLEAN_ROOT at it.
# Expected layout:
#   <root>/images/size_4x4/seed_001/step_0000_current.png
# The MP4 batch names are one size label smaller (3x3,4x4,5x5) than the
# video_eval JSONL/source-image layout (4x4,5x5,6x6), so both are tried.
CLEAN_ROOTS = [
    Path(p).expanduser()
    for p in [
        os.environ.get("ONE_STROKE_CLEAN_ROOT", ""),
        str(REPO_ROOT),
        str(REPO_ROOT / "logs/video_eval"),
    ]
    if p
]
ALLOW_GEN00_CLEAN_PROXY = os.environ.get("ONE_STROKE_ALLOW_GEN00_PROXY", "1") != "0"

_SOURCE_META_BY_SEED: Optional[Dict[int, Dict[str, Any]]] = None
_GENERATED_LEVEL_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}
_CLEAN_RENDER_ENV: Any = None
_CLEAN_RENDER_VIEWPORT: Optional[Tuple[int, int]] = None

cv2.setNumThreads(1)

# ── HSV color definitions ──────────────────────────────────────────────────

# (name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) — red hue wraps around 180
CELL_COLORS = [
    ("red",     0,  14, 120, 255, 80, 255),
    ("blue",   88, 135, 120, 255, 80, 255),
    ("green",  38,  88, 120, 255, 80, 255),
    ("yellow", 20,  38, 100, 255, 70, 255),
    ("purple", 132, 158, 80, 255, 50, 255),
    ("cyan",   82, 100, 80, 255, 70, 255),
    ("gray",    0, 180,   0,  55, 70, 210),
]

STROKE_S_MAX = 45
STROKE_V_MIN = 190

START_HALO_EXCLUDE_RADIUS = 76
END_HALO_EXCLUDE_RADIUS = 58
CELL_FRAGMENT_MERGE_GAP = 8

# ── helpers ────────────────────────────────────────────────────────────────

def _hsv_mask(hsv: np.ndarray, h_lo: int, h_hi: int, s_lo: int, v_lo: int,
              wrap_red: bool = False, *, s_hi: int = 255, v_hi: int = 255) -> np.ndarray:
    lo = np.array([h_lo, s_lo, v_lo], np.uint8)
    hi = np.array([h_hi, s_hi, v_hi], np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    if wrap_red:
        lo2 = np.array([160, s_lo, v_lo], np.uint8)
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo2, np.array([180, s_hi, v_hi], np.uint8)))
    return m


def _morph(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _cell_morph(mask: np.ndarray) -> np.ndarray:
    """Clean colored tile masks without merging neighboring grid cells."""
    return _morph(mask, 5)


# ── data types ─────────────────────────────────────────────────────────────

@dataclass
class Blob:
    color: str
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    area: int


@dataclass
class ParseResult:
    cells: List[Blob]
    stroke_mask: np.ndarray
    stroke_area: int
    stroke_connected: bool
    start_pos: Optional[Tuple[float, float]]
    end_pos: Optional[Tuple[float, float]]
    colors_found: List[str]
    status: str   # "ok" | "degraded" | "failed"
    # Metrics extras
    stroke_on_cell_px: int = 0                                    # stroke pixels overlapping any colored cell
    stroke_tip_a: Optional[Tuple[int, int]] = None                # geometric endpoint of stroke path
    stroke_tip_b: Optional[Tuple[int, int]] = None
    stroke_near_start: Optional[bool] = None                      # stroke pixel within 90px of start_pos
    stroke_near_end: Optional[bool] = None                        # stroke pixel within 90px of end_pos
    stroke_start_tip: Optional[Tuple[int, int]] = None            # tip nearest to start_pos (for direction)
    stroke_end_tip: Optional[Tuple[int, int]] = None              # tip nearest to end_pos   (for direction)


def _bbox_gap_and_overlap(a: Blob, b: Blob) -> Tuple[int, int, int, int]:
    ax1, ay1, aw, ah = a.bbox
    bx1, by1, bw, bh = b.bbox
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))
    x_overlap = min(ax2, bx2) - max(ax1, bx1)
    y_overlap = min(ay2, by2) - max(ay1, by1)
    return x_gap, y_gap, x_overlap, y_overlap


def _merge_blob_pair(a: Blob, b: Blob) -> Blob:
    area = int(a.area + b.area)
    if area <= 0:
        center = ((a.center[0] + b.center[0]) / 2.0, (a.center[1] + b.center[1]) / 2.0)
    else:
        center = (
            (a.center[0] * a.area + b.center[0] * b.area) / area,
            (a.center[1] * a.area + b.center[1] * b.area) / area,
        )
    x1 = min(a.bbox[0], b.bbox[0])
    y1 = min(a.bbox[1], b.bbox[1])
    x2 = max(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2])
    y2 = max(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3])
    return Blob(a.color, center, (x1, y1, x2 - x1, y2 - y1), area)


def _merge_split_cell_fragments(cells: List[Blob]) -> List[Blob]:
    """Merge same-color pieces of one tile split by a stroke occlusion.

    The merge is deliberately bbox-local: fragments must overlap or nearly touch
    in both axes.  That recovers diagonally cut tiles without using a large mask
    close that can glue adjacent grid cells together.
    """
    merged = list(cells)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(merged):
            hit = None
            for j in range(i + 1, len(merged)):
                b = merged[j]
                if a.color != b.color:
                    continue
                x_gap, y_gap, x_overlap, y_overlap = _bbox_gap_and_overlap(a, b)
                if (
                    x_gap <= CELL_FRAGMENT_MERGE_GAP
                    and y_gap <= CELL_FRAGMENT_MERGE_GAP
                    and (x_overlap > 0 or y_overlap > 0)
                ):
                    hit = j
                    break
            if hit is not None:
                merged[i] = _merge_blob_pair(a, merged[hit])
                merged.pop(hit)
                changed = True
                break
    return merged


# ── detection ───────────────────────────────────────────────────────────────

def _stroke_exclusion_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Dilated mask of the white/cyan stroke, used to exclude stroke pixels from cell detection.

    The game renders the stroke as bright cyan (H≈90-100, S≈0-150, V>180), which
    overlaps with the blue cell HSV range.  We detect with a loose saturation ceiling
    (S<150) and dilate so that the stroke edges don't leak into blue cell blobs.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # bright (V>180) AND low-to-medium saturation (S<150) → stroke / white area
    mask = cv2.inRange(hsv,
                       np.array([0,   0,   180], np.uint8),
                       np.array([180, 150, 255], np.uint8))
    # ignore title bar
    mask[: int(img_bgr.shape[0] * 0.12), :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, k)


def _marker_halo_exclusion_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask marker halos before cell connected-components are computed.

    The start marker has a blue/cyan orbital ring below it.  If that ring is
    allowed into the blue-cell HSV mask, morphology can connect it to the
    nearest blue tile and inflate the tile bbox.  Excluding the marker halos
    before connected-components keeps marker decoration separate from board
    cells.
    """
    return cv2.bitwise_or(
        _start_marker_halo_exclusion_mask(img_bgr),
        _end_marker_halo_exclusion_mask(img_bgr),
    )


def _start_marker_halo_exclusion_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask the start marker's blue/cyan halo without touching board cells."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    full = np.full((h, w), 255, np.uint8)
    start_pos = _detect_marker_pos(img_bgr, 35, 92, full, wrap_red=False, min_area=100)
    if start_pos is not None:
        cv2.circle(
            mask,
            (int(round(start_pos[0])), int(round(start_pos[1]))),
            START_HALO_EXCLUDE_RADIUS,
            255,
            -1,
        )

    return mask


def _end_marker_halo_exclusion_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask the end dome's red halo for marker-localization helpers."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    region_e = _region_mask(img_bgr, 0.60, 0.0, 1.0, 0.35)
    end_pos = _detect_marker_pos(img_bgr, 0, 14, region_e, wrap_red=True, prefer_rightmost=True)
    if end_pos is not None:
        cv2.circle(
            mask,
            (int(round(end_pos[0])), int(round(end_pos[1]))),
            END_HALO_EXCLUDE_RADIUS,
            255,
            -1,
        )

    return mask


def _detect_blobs(img_bgr: np.ndarray, min_area_frac: float = 0.0015) -> List[Blob]:
    """Detect all colored cell regions, excluding the stroke area first."""
    h, w = img_bgr.shape[:2]
    min_area = int(h * w * min_area_frac)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Mask out stroke pixels so cyan/white stroke isn't classified as blue.
    # Red cells are intentionally not marker-excluded here: the end dome can sit
    # close to the top-right red tile, and a broad circular exclusion deletes too
    # much of that real cell.  The small dome component is filtered after marker
    # localization instead.
    stroke_exclude = _stroke_exclusion_mask(img_bgr)
    start_marker_exclude = cv2.bitwise_or(stroke_exclude, _start_marker_halo_exclusion_mask(img_bgr))
    blobs: List[Blob] = []
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in CELL_COLORS:
        raw = _hsv_mask(
            hsv, h_lo, h_hi, s_lo, v_lo, wrap_red=(name == "red"), s_hi=s_hi, v_hi=v_hi
        )
        if name != "red":
            raw = cv2.bitwise_and(raw, cv2.bitwise_not(start_marker_exclude))
        mask = _cell_morph(raw)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            blobs.append(Blob(
                color=name,
                center=(float(centroids[i, 0]), float(centroids[i, 1])),
                bbox=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
                area=area,
            ))
    return blobs


def _detect_marker_pos(
    img_bgr: np.ndarray,
    h_lo: int, h_hi: int,
    region_mask: np.ndarray,
    wrap_red: bool = False,
    min_area: int = 300,
    prefer_rightmost: bool = False,
) -> Optional[Tuple[float, float]]:
    """Find a same-color blob within a fixed image region.

    Uses looser HSV thresholds (s_lo=50, v_lo=50) to catch the 3D-rendered
    markers (dome, blob) which are darker/less saturated than the flat tiles.

    prefer_rightmost=True picks the blob with the highest centroid x — used for
    the end dome, which is always placed outside the grid to the right, making it
    the rightmost red object in the top-right search region regardless of whether
    a tile fragment (smaller area) or a tile overlap (larger area) also falls in.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = _hsv_mask(hsv, h_lo, h_hi, 50, 50, wrap_red=wrap_red)
    mask = cv2.bitwise_and(mask, region_mask)
    mask = _morph(mask, 3)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
            if best is None:
                best = (area, cx, cy)
            elif prefer_rightmost and cx > best[1]:
                best = (area, cx, cy)
            elif not prefer_rightmost and area > best[0]:
                best = (area, cx, cy)
    return (best[1], best[2]) if best else None


def _region_mask(img_bgr: np.ndarray, x0f: float, y0f: float,
                 x1f: float, y1f: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(y0f * h): int(y1f * h), int(x0f * w): int(x1f * w)] = 255
    return m


def _identify_markers(
    blobs: List[Blob], img_bgr: np.ndarray,
) -> Tuple[List[Blob], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Two-pass marker detection.

    Pass 1 — per-color isolation (works well for 3×3 and 4×4):
      • Single green blob → start marker
      • Multiple greens → most isolated is start marker
      • Most isolated red blob → end marker (requires ≥ 2 red blobs)

    Pass 2 — region fallback (for 5×5 where markers are smaller and darker):
      If pass 1 misses a marker, look in fixed image corner regions with
      looser HSV thresholds (s_lo=50, v_lo=50, min_area=300px).
      Start region: bottom-left 25 % × 30 %  (green)
      End region:   top 30 % × right 35 %    (red dome)
    """
    green = [b for b in blobs if b.color == "green"]
    red   = [b for b in blobs if b.color == "red"]

    start_blob = end_blob = None

    # ── START marker (green blob) ──────────────────────────────────────────
    # Strategy: single green → start; multiple → most isolated.
    # Fallback: loose detection anywhere (green tiles are absent in this game).
    if len(green) == 1:
        start_blob = green[0]
    elif len(green) >= 2:
        centers = np.array([[b.center[0], b.center[1]] for b in green])
        dists = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
        start_blob = green[int(np.argmax(dists))]

    if start_blob is None:
        h, w = img_bgr.shape[:2]
        full = np.full((h, w), 255, np.uint8)
        pos = _detect_marker_pos(img_bgr, 35, 92, full, wrap_red=False, min_area=100)
        if pos is not None:
            start_blob = Blob("green", pos, (0, 0, 0, 0), 0)

    # ── END marker (red dome) ──────────────────────────────────────────────
    # ALWAYS prefer region-based (top-right corner, loose thresholds) because
    # per-color isolation picks the most-isolated red TILE, not the 3D dome.
    # The dome never passes _detect_blobs min_area (it's a small 3D object),
    # so we always use a synthetic Blob — never reuse a nearby grid tile.
    region_e = _region_mask(img_bgr, 0.60, 0.0, 1.0, 0.35)
    pos = _detect_marker_pos(img_bgr, 0, 14, region_e, wrap_red=True, prefer_rightmost=True)
    if pos is not None:
        end_blob = Blob("red", pos, (0, 0, 0, 0), 0)
    elif len(red) >= 2:  # fallback if dome not in top-right region
        centers = np.array([[b.center[0], b.center[1]] for b in red])
        dists = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
        end_blob = red[int(np.argmax(dists))]

    excluded = {id(b) for b in (start_blob, end_blob) if b is not None and b.area > 0}
    cells = [b for b in blobs if id(b) not in excluded]

    # Exclude the start dome's orbital ring: a small colored blob that sits
    # <70px from the dome center and is too small to be a real grid cell.
    # The ring can exceed min_area in mid-game frames (e.g., stroke pixels
    # at the dome base inflate it), so we must filter it here.
    if start_blob is not None:
        sp = np.array(start_blob.center)
        cells = [b for b in cells if not (
            np.linalg.norm(np.array(b.center) - sp) < 85 and b.area < 3500
        )]

    # Exclude the red end dome itself without carving away nearby red tiles.
    # The dome is small and centered at the detected end marker; a top-right
    # tile fragment is farther away and materially larger.
    if end_blob is not None:
        ep = np.array(end_blob.center)
        cells = [b for b in cells if not (
            b.color == "red"
            and np.linalg.norm(np.array(b.center) - ep) < 55
            and b.area < 2500
        )]

    # Some generated paths render a blue circular handle/halo at the lower-left
    # stroke corner even when the green start marker is elsewhere.  It sits well
    # outside the physical board, but its HSV/area look cell-like.
    h, w = img_bgr.shape[:2]
    cells = [b for b in cells if not (
        b.color == "blue"
        and b.center[0] < 0.22 * w
        and b.center[1] > 0.62 * h
        and b.area < 4000
    )]

    cells = _merge_split_cell_fragments(cells)

    return cells, (start_blob.center if start_blob else None), (end_blob.center if end_blob else None)


def _all_cell_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Union of all colored cell regions (for stroke-overlap computation)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(img_bgr.shape[:2], np.uint8)
    marker_exclude = cv2.bitwise_or(
        _marker_halo_exclusion_mask(img_bgr), _stroke_exclusion_mask(img_bgr)
    )
    for name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in CELL_COLORS:
        m = _morph(_hsv_mask(
            hsv, h_lo, h_hi, s_lo, v_lo, wrap_red=(name == "red"), s_hi=s_hi, v_hi=v_hi
        ))
        m = cv2.bitwise_and(m, cv2.bitwise_not(marker_exclude))
        combined = cv2.bitwise_or(combined, m)
    return combined


def _stroke_near(stroke_mask: np.ndarray,
                 pos: Optional[Tuple[float, float]],
                 threshold: float = 90.0) -> Optional[bool]:
    """True if any stroke pixel is within `threshold` pixels of `pos`."""
    if pos is None or (stroke_mask > 0).sum() == 0:
        return None
    h, w = stroke_mask.shape
    cx, cy = int(pos[0]), int(pos[1])
    r = int(threshold)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = stroke_mask[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return False
    dists = np.sqrt((xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2)
    return bool(dists.min() < threshold)


def _stroke_endpoints(stroke_mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]],
                                                         Optional[Tuple[int, int]]]:
    """Geometric start and end of the stroke path (O(n) farthest-pair approx)."""
    ys, xs = np.where(stroke_mask > 0)
    if len(xs) < 20:
        return None, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    p1 = pts[int(np.argmax(dists))]
    dists2 = np.linalg.norm(pts - p1, axis=1)
    p2 = pts[int(np.argmax(dists2))]
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def detect_stroke(img_bgr: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0,   0,           STROKE_V_MIN], np.uint8),
                       np.array([180, STROKE_S_MAX, 255         ], np.uint8))
    # Ignore top 12 % (title text "State N")
    mask[: int(h * 0.12), :] = 0
    # Remove thin text: open with a kernel that only passes thick strokes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # Shading / compression artifacts on the rendered 3D arrow mesh can drop a
    # thin band of pixels below STROKE_V_MIN and split an otherwise-continuous
    # stroke into several components (most visibly at the arrowhead tip).
    # Bridge small gaps before measuring connectivity so a handful of dropped
    # pixels doesn't read as a genuinely broken stroke.
    bridge_ksize = max(5, int(round(0.012 * min(h, w))))
    bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_ksize, bridge_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_k)

    area = int((mask > 0).sum())
    min_stroke = int(h * w * 0.002)
    if area < min_stroke:
        return mask, area, False

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask, area, False
    largest = max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))
    return mask, area, largest / area > 0.65


def parse_frame(img_bgr: np.ndarray) -> ParseResult:
    # 1. Detect all colored blobs
    blobs = _detect_blobs(img_bgr)

    # 2. Per-color isolation + region fallback: identify start/end markers
    cells, start_pos, end_pos = _identify_markers(blobs, img_bgr)

    # 3. Stroke
    stroke_mask, stroke_area, stroke_connected = detect_stroke(img_bgr)

    # 4. Stroke-on-cell overlap (for StrokeOnEdge / CellCut metrics)
    cell_mask = _all_cell_mask(img_bgr)
    stroke_on_cell_px = int((cv2.bitwise_and(stroke_mask, cell_mask) > 0).sum())

    # 5. Geometric stroke endpoints (for display)
    tip_a, tip_b = _stroke_endpoints(stroke_mask)

    # 6. Stroke proximity to markers (for StrokeConnects metric)
    stroke_near_s = _stroke_near(stroke_mask, start_pos)
    stroke_near_e = _stroke_near(stroke_mask, end_pos)

    # 7. Stroke direction: assign geometric tips to start/end by proximity
    stroke_start_tip = stroke_end_tip = None
    if tip_a and tip_b and start_pos and end_pos:
        ta, tb = np.array(tip_a), np.array(tip_b)
        sp = np.array(start_pos)
        if np.linalg.norm(ta - sp) <= np.linalg.norm(tb - sp):
            stroke_start_tip, stroke_end_tip = tip_a, tip_b
        else:
            stroke_start_tip, stroke_end_tip = tip_b, tip_a

    colors_found = sorted(set(b.color for b in cells))
    if not cells:
        status = "failed"
    elif start_pos is None or end_pos is None:
        status = "degraded"
    else:
        status = "ok"

    return ParseResult(
        cells=cells,
        stroke_mask=stroke_mask,
        stroke_area=stroke_area,
        stroke_connected=stroke_connected,
        start_pos=start_pos,
        end_pos=end_pos,
        colors_found=colors_found,
        status=status,
        stroke_on_cell_px=stroke_on_cell_px,
        stroke_tip_a=tip_a,
        stroke_tip_b=tip_b,
        stroke_near_start=stroke_near_s,
        stroke_near_end=stroke_near_e,
        stroke_start_tip=stroke_start_tip,
        stroke_end_tip=stroke_end_tip,
    )


# ── overlay ─────────────────────────────────────────────────────────────────

_CELL_BGR = {
    "red": (30, 30, 220), "blue": (220, 120, 30), "green": (30, 200, 30),
    "yellow": (0, 220, 220), "purple": (180, 30, 145), "gray": (170, 170, 170),
    "cyan": (220, 200, 30),
}
_STATUS_BGR = {"ok": (0, 220, 0), "degraded": (0, 200, 220), "failed": (0, 0, 220)}


def draw_overlay(img_bgr: np.ndarray, parsed: ParseResult, title: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # White stroke → cyan highlight
    if parsed.stroke_area > 0:
        layer = out.copy()
        layer[parsed.stroke_mask > 0] = (200, 220, 0)
        cv2.addWeighted(layer, 0.45, out, 0.55, 0, out)
        cnts, _ = cv2.findContours(parsed.stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1)

    # Cell bounding boxes
    for cell in parsed.cells:
        x, y, bw, bh = cell.bbox
        bgr = _CELL_BGR.get(cell.color, (180, 180, 180))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), bgr, 2)
        cx, cy = int(cell.center[0]), int(cell.center[1])
        cv2.putText(out, cell.color[0].upper(), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    # Stroke direction tips.  Do not draw a full raw direction line here:
    # video_cv_pipeline overlays reserve full yellow lines for snapped legal
    # board edges, and a raw diagonal arrow can otherwise look like a fitted path.
    if parsed.stroke_start_tip and parsed.stroke_end_tip:
        cv2.circle(out, parsed.stroke_start_tip, 6, (0, 200, 80), -1)   # green = S side
        cv2.circle(out, parsed.stroke_end_tip,   6, (0, 80, 200), -1)   # orange = E side
    else:
        for tip in (parsed.stroke_tip_a, parsed.stroke_tip_b):
            if tip is not None:
                cv2.circle(out, tip, 6, (255, 200, 0), -1)   # gold = undirected

    # Start marker
    if parsed.start_pos is not None:
        cx, cy = int(parsed.start_pos[0]), int(parsed.start_pos[1])
        cv2.circle(out, (cx, cy), 20, (0, 240, 0), 2)
        cv2.putText(out, "S", (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 0), 1, cv2.LINE_AA)

    # End marker
    if parsed.end_pos is not None:
        cx, cy = int(parsed.end_pos[0]), int(parsed.end_pos[1])
        cv2.circle(out, (cx, cy), 20, (0, 0, 240), 2)
        cv2.putText(out, "E", (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 240), 1, cv2.LINE_AA)

    # Title bar
    s_bgr = _STATUS_BGR.get(parsed.status, (128, 128, 128))
    conn  = "conn" if parsed.stroke_connected else ("frag" if parsed.stroke_area > 0 else "none")
    if parsed.stroke_start_tip and parsed.stroke_end_tip:
        dir_str = "S->E"
    elif parsed.stroke_area > 0:
        dir_str = "?"
    else:
        dir_str = "-"
    info  = (f"{title}  [{parsed.status}]  "
             f"cells={len(parsed.cells)} colors={','.join(parsed.colors_found) or '-'}  "
             f"stroke={conn}({parsed.stroke_area}px)  "
             f"S={'ok' if parsed.start_pos else 'miss'}  "
             f"E={'ok' if parsed.end_pos else 'miss'}  "
             f"Dir={dir_str}")
    cv2.rectangle(out, (0, 0), (w, 22), (25, 25, 25), -1)
    cv2.putText(out, info, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, s_bgr, 1, cv2.LINE_AA)
    return out


# ── main ───────────────────────────────────────────────────────────────────

def _parse_name(fname: str) -> Tuple[Optional[str], Optional[int]]:
    m = re.search(r'size_(\d+x\d+)_seed_(\d+)', fname)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _increment_size_label(size: str) -> Optional[str]:
    m = re.fullmatch(r"(\d+)x(\d+)", size)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return f"{w + 1}x{h + 1}"


def _board_size_from_label(size: str) -> Optional[int]:
    """Map MP4 size labels to the source board_size used by generation."""
    m = re.fullmatch(r"(\d+)x\d+", size)
    if not m:
        return None
    # The MP4 batch labels are one smaller than the source images/jsonl labels.
    return int(m.group(1)) + 1


def _extract_level_json(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = row.get("meta_info") or {}
    candidates = [
        row,
        meta,
        meta.get("initial_state") or {},
        meta.get("reset_config") or {},
        meta.get("frontend_config") or {},
    ]
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for key in ("level_json", "levelJson"):
            level_json = obj.get(key)
            if isinstance(level_json, dict):
                return level_json
    return None


def _load_source_meta_by_seed() -> Dict[int, Dict[str, Any]]:
    global _SOURCE_META_BY_SEED
    if _SOURCE_META_BY_SEED is not None:
        return _SOURCE_META_BY_SEED

    out: Dict[int, Dict[str, Any]] = {}
    if not SOURCE_JSONL.exists():
        _SOURCE_META_BY_SEED = out
        return out

    with SOURCE_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta_info") or {}
            seed = meta.get("seed") or row.get("seed")
            if seed is None:
                continue
            trajectory = row.get("trajectory") or []
            current_images: List[str] = []
            if trajectory and isinstance(trajectory[0], dict):
                current_images = list(trajectory[0].get("current_images") or [])
            out[int(seed)] = {
                "board_size": meta.get("board_size"),
                "level": meta.get("level"),
                "current_images": current_images,
                "level_json": _extract_level_json(row),
            }

    _SOURCE_META_BY_SEED = out
    return out


def _generate_level_json(size: str, seed: int) -> Tuple[Optional[Dict[str, Any]], str]:
    meta = _load_source_meta_by_seed().get(seed, {})
    board_size = meta.get("board_size") or _board_size_from_label(size)
    if board_size is None:
        return None, "missing_board_size"
    board_size = int(board_size)
    cache_key = (board_size, int(seed))
    if cache_key not in _GENERATED_LEVEL_CACHE:
        if str(ONE_STROKE_GYM_DIR) not in sys.path:
            sys.path.insert(0, str(ONE_STROKE_GYM_DIR))
        from generator import generate_level_for_board_size

        spec = generate_level_for_board_size(
            board_size=board_size,
            seed=int(seed),
            setup_index=0,
            max_attempts=1000,
            solver_timeout_seconds=2.0,
        )
        _GENERATED_LEVEL_CACHE[cache_key] = dict(spec.level_json)
    return _GENERATED_LEVEL_CACHE[cache_key], f"generated_level_json(board_size={board_size},seed={seed})"


def _clean_level_json(size: str, seed: int) -> Tuple[Optional[Dict[str, Any]], str]:
    meta = _load_source_meta_by_seed().get(seed, {})
    level_json = meta.get("level_json")
    if isinstance(level_json, dict):
        return level_json, f"level_json_from:{SOURCE_JSONL}"
    return _generate_level_json(size, seed)


def _get_clean_render_env(width: int, height: int):
    global _CLEAN_RENDER_ENV, _CLEAN_RENDER_VIEWPORT
    viewport = (int(width), int(height))
    if _CLEAN_RENDER_ENV is not None and _CLEAN_RENDER_VIEWPORT == viewport:
        return _CLEAN_RENDER_ENV
    if _CLEAN_RENDER_ENV is not None:
        _CLEAN_RENDER_ENV.close()
        _CLEAN_RENDER_ENV = None

    if str(ONE_STROKE_GYM_DIR) not in sys.path:
        sys.path.insert(0, str(ONE_STROKE_GYM_DIR))
    from env import SeparationOneStrokeEnv

    _CLEAN_RENDER_ENV = SeparationOneStrokeEnv(
        headless=True,
        viewport_width=int(width),
        viewport_height=int(height),
    )
    _CLEAN_RENDER_VIEWPORT = viewport
    return _CLEAN_RENDER_ENV


def _close_clean_render_env() -> None:
    global _CLEAN_RENDER_ENV, _CLEAN_RENDER_VIEWPORT
    if _CLEAN_RENDER_ENV is not None:
        _CLEAN_RENDER_ENV.close()
        _CLEAN_RENDER_ENV = None
        _CLEAN_RENDER_VIEWPORT = None


def _render_clean_image(
    size: str,
    seed: int,
    target_shape: Optional[Tuple[int, int, int]],
) -> Tuple[Optional[np.ndarray], str]:
    if target_shape is None:
        return None, "missing_target_shape"

    try:
        level_json, level_source = _clean_level_json(size, seed)
        if level_json is None:
            return None, level_source
        height, width = int(target_shape[0]), int(target_shape[1])
        env = _get_clean_render_env(width, height)
        obs, _ = env.reset(options={"frontend_config": {"levelJson": level_json}})
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return img, f"rendered_clean:{level_source}"
    except Exception as exc:
        print(
            f"warning: clean render failed for {size} seed{seed:03d}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None, f"render_failed:{type(exc).__name__}"


def _find_clean_image(size: str, seed: int) -> Optional[Path]:
    size_candidates = [s for s in (_increment_size_label(size), size) if s]
    rel_candidates = [
        Path("images") / f"size_{sz}" / f"seed_{seed:03d}" / "step_0000_current.png"
        for sz in size_candidates
    ]
    rel_candidates += [
        Path(p)
        for p in _load_source_meta_by_seed().get(seed, {}).get("current_images", [])
    ]
    for root in CLEAN_ROOTS:
        for rel in rel_candidates:
            candidate = root / rel
            if candidate.exists():
                return candidate
    return None


def _ep_status(statuses: List[str]) -> str:
    if not statuses or any(s == "failed"   for s in statuses): return "failed"
    if all(s == "ok" for s in statuses):                        return "ok"
    return "degraded"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4s = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(mp4s)} videos → {OUT_DIR}")

    frame_counts   = {"ok": 0, "degraded": 0, "failed": 0}
    episode_counts = {"ok": 0, "degraded": 0, "failed": 0}
    by_size: Dict[str, Dict[str, int]] = {}
    all_episodes = []
    processed = 0

    for mp4 in mp4s:
        size, seed = _parse_name(mp4.name)
        if size is None:
            continue

        cap = cv2.VideoCapture(str(mp4))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = {"gen00": 0, "genMid": max(n // 2, 0), "genEnd": max(n - 1, 0)}
        raw: Dict[str, Optional[np.ndarray]] = {"clean": None}
        for tag, idx in indices.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            raw[tag] = frame if ok else None
        cap.release()

        ep_statuses = []
        clean_source = "missing"
        clean_path = _find_clean_image(size, seed)
        if clean_path is not None:
            raw["clean"] = cv2.imread(str(clean_path))
            clean_source = str(clean_path)
        elif raw.get("gen00") is not None:
            rendered_clean, render_source = _render_clean_image(size, seed, raw["gen00"].shape)
            if rendered_clean is not None:
                raw["clean"] = rendered_clean
                clean_source = render_source
            elif ALLOW_GEN00_CLEAN_PROXY:
                raw["clean"] = raw["gen00"].copy()
                clean_source = f"gen00_proxy_after_{render_source}"

        ep_rec: Dict = {
            "size": size,
            "seed": seed,
            "clean_source": clean_source,
            "frames": {},
        }

        for ft in FRAME_TYPES:
            img = raw.get(ft)
            if img is None:
                ep_statuses.append("failed")
                continue

            parsed = parse_frame(img)
            frame_counts[parsed.status] = frame_counts.get(parsed.status, 0) + 1
            ep_statuses.append(parsed.status)
            ep_rec["frames"][ft] = {
                "status":             parsed.status,
                "n_cells":            len(parsed.cells),
                "colors":             parsed.colors_found,
                "stroke_area":        parsed.stroke_area,
                "stroke_connected":   parsed.stroke_connected,
                "start_found":        parsed.start_pos is not None,
                "end_found":          parsed.end_pos   is not None,
                # Metrics data
                "start_pos":          list(parsed.start_pos) if parsed.start_pos else None,
                "end_pos":            list(parsed.end_pos)   if parsed.end_pos   else None,
                "stroke_on_cell_px":  parsed.stroke_on_cell_px,
                "stroke_tip_a":       list(parsed.stroke_tip_a) if parsed.stroke_tip_a else None,
                "stroke_tip_b":       list(parsed.stroke_tip_b) if parsed.stroke_tip_b else None,
                "stroke_near_start":  parsed.stroke_near_start,
                "stroke_near_end":    parsed.stroke_near_end,
                "stroke_start_tip":   list(parsed.stroke_start_tip) if parsed.stroke_start_tip else None,
                "stroke_end_tip":     list(parsed.stroke_end_tip)   if parsed.stroke_end_tip   else None,
                "cell_centroids":     [(b.color, round(b.center[0], 1), round(b.center[1], 1))
                                       for b in parsed.cells],
                "cell_bboxes":        [(b.color, b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3])
                                       for b in parsed.cells],
            }

            overlay = draw_overlay(img, parsed, f"{size} seed{seed:03d} {ft}")
            cv2.imwrite(str(OUT_DIR / f"{size}_seed{seed:03d}_{ft}.png"), overlay)

        ep_st = _ep_status(ep_statuses)
        episode_counts[ep_st] = episode_counts.get(ep_st, 0) + 1
        by_size.setdefault(size, {"ok": 0, "degraded": 0, "failed": 0})[ep_st] += 1
        ep_rec["status"] = ep_st
        all_episodes.append(ep_rec)
        processed += 1
        print(f"  {size} seed{seed:03d}: {ep_st} ({', '.join(ep_statuses)})", flush=True)

    summary = {"n_episodes": processed, "episode_counts": episode_counts,
               "frame_counts": frame_counts, "by_size": by_size}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "episodes.json").write_text(json.dumps(all_episodes, indent=2))

    print(f"\nFrame counts:   {frame_counts}")
    print(f"Episode counts: {episode_counts}")
    print(f"By size:        {by_size}")
    print(f"Wrote {processed * len(FRAME_TYPES)} overlays → {OUT_DIR}")
    _close_clean_render_env()


if __name__ == "__main__":
    main()
