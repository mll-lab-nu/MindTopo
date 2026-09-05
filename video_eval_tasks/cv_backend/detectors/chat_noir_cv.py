"""``enclosure_chat_noir`` ("chat noir") frame parser + overlay.

Design mirrors the swap2d spike: never read topology from raw pixels beyond the discrete
board state.  The board is a static hexagonal ring of circular cells drawn on a cream card
(open = light tan disk, blocked = near-black disk, cat = orange disk with an orange halo and
a "CAT" tag).  Cell identity comes from the known rigid hex geometry rather than the printed
numbers.  The numbers are checked separately for four-checkpoint visual consistency because
the video model frequently hallucinates, removes, or duplicates them.

Per clip we estimate one :class:`ChatNoirGeometry` (the board never moves): HoughCircles
locates all cell disks in several frames (fill-independent, so it finds open, blocked, and
cat cells alike), then we fit the analytic hex lattice (``build_hex_board`` coordinates) to
the pooled circle centres with a scale+translation ICP.  Each frame is then classified against
the fixed cell centres: orange blob -> cat cell, dark disk -> blocked, otherwise open.  The
result is a discrete ``(cat_index, blocked_set)`` state scored against the env oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

# Hex geometry constant from the env (build_hex_board scale).
HEX_SCALE = 1.22

# --- CV thresholds --------------------------------------------------------------------
CAT_HUE_LO, CAT_HUE_HI = 12, 32          # cat orange ~ hue 19 (OpenCV)
CAT_SAT_MIN, CAT_VAL_MIN = 110, 140
BLOCKED_VAL_MAX = 85                      # near-black disk fill
BLOCKED_FRAC_MIN = 0.5                    # dark fraction in the sample disk to call blocked
CAT_BLOB_AREA_FRAC = 0.15                # min orange-blob area / cell-disk area to be a cat
SAMPLE_DISK_FRAC = 0.30                  # sample-disk radius as fraction of pitch
HOUGH_PARAM2 = 28
HOUGH_MIN_R_FRAC = 0.28                  # circle radius search window (frac of approx pitch)
HOUGH_MAX_R_FRAC = 0.62


def hex_lattice(radius: int, scale: float = HEX_SCALE) -> np.ndarray:
    """Analytic cell centres (index-ordered) in env space, mapped to image axes (+y down)."""
    out: List[Tuple[float, float]] = []
    for r in range(-radius, radius + 1):
        min_q = max(-radius, -r - radius)
        max_q = min(radius, -r + radius)
        for q in range(min_q, max_q + 1):
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale  # env uses -1.5*r (y up); image y grows down, so negate -> +1.5*r
            out.append((x, y))
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class ChatNoirGeometry:
    """Clip-level hex lattice shared by every sampled frame."""

    radius: int
    centers: Tuple[Tuple[float, float], ...]   # per cell index
    scale: float
    pitch: float                               # centre-to-centre spacing (= sqrt(3)*scale)
    residual: float
    reference_frame_ids: Tuple[int, ...] = ()

    def center_xy(self, index: int) -> Tuple[float, float]:
        return self.centers[index]


@dataclass
class ChatNoirCell:
    index: int
    center_xy: Tuple[float, float]
    state: Optional[str]      # "open" | "blocked" | "cat" | None (unreadable)
    dark_frac: float
    orange_frac: float
    confidence: float


@dataclass
class ParsedChatNoirFrame:
    radius: int
    cells: List[ChatNoirCell]
    cat_index: Optional[int]
    cat_center_xy: Optional[Tuple[float, float]]
    cat_count: int            # number of distinct orange blobs (should be exactly 1)
    blocked: Set[int]
    parser_status: str        # "ok" | "cat_not_found" | "grid_not_found"
    warnings: List[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.parser_status == "ok"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _hough_circles(frame_bgr: np.ndarray, radius: int) -> Optional[np.ndarray]:
    gray = cv2.medianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 5)
    approx_pitch = frame_bgr.shape[1] * 0.47 / (2 * radius + 1)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(0.7 * approx_pitch),
        param1=80,
        param2=HOUGH_PARAM2,
        minRadius=int(HOUGH_MIN_R_FRAC * approx_pitch),
        maxRadius=int(HOUGH_MAX_R_FRAC * approx_pitch),
    )
    if circles is None:
        return None
    return circles[0][:, :2].astype(float)


def _fit_lattice(circle_centers: np.ndarray, radius: int) -> Tuple[np.ndarray, float, float]:
    """Fit scale+translation mapping the analytic lattice onto detected circle centres (ICP)."""
    lattice = hex_lattice(radius)
    centroid = circle_centers.mean(axis=0)
    denom = float((lattice ** 2).sum())
    scale = math.sqrt(float(((circle_centers - centroid) ** 2).sum()) / denom) if denom else 1.0
    translation = centroid.copy()
    for _ in range(8):
        pred = lattice * scale + translation
        idx = [int(np.argmin(((circle_centers - p) ** 2).sum(axis=1))) for p in pred]
        matched = circle_centers[idx]
        matched_centroid = matched.mean(axis=0)
        num = float((lattice * matched).sum())
        scale = num / denom if denom else scale
        translation = matched_centroid - scale * lattice.mean(axis=0)
    pred = lattice * scale + translation
    residual = float(np.mean([math.sqrt(((pred - c) ** 2).sum(axis=1).min()) for c in circle_centers]))
    return pred, float(scale), residual


def estimate_chat_noir_geometry(
    frames_bgr: Sequence[np.ndarray],
    radius: int,
    max_reference_frames: int = 12,
) -> Optional[ChatNoirGeometry]:
    """Pool HoughCircle centres from several frames and fit the rigid hex lattice."""
    expected = len(hex_lattice(radius))
    candidates: List[Tuple[float, np.ndarray, float, int]] = []
    step = max(1, len(frames_bgr) // max_reference_frames)
    for frame_id in range(0, len(frames_bgr), step):
        circles = _hough_circles(frames_bgr[frame_id], radius)
        if circles is None or len(circles) < max(3, expected // 2):
            continue
        pred, scale, residual = _fit_lattice(circles, radius)
        candidates.append((residual, pred, scale, frame_id))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    selected = candidates[: min(len(candidates), max_reference_frames)]
    preds = np.stack([c[1] for c in selected], axis=0)  # (k, n_cells, 2)
    centers = preds.mean(axis=0)
    scale = float(np.median([c[2] for c in selected]))
    residual = float(np.median([c[0] for c in selected]))
    refs = tuple(int(c[3]) for c in selected)
    return ChatNoirGeometry(
        radius=radius,
        centers=tuple((float(x), float(y)) for x, y in centers),
        scale=scale,
        pitch=float(math.sqrt(3) * scale),
        residual=residual,
        reference_frame_ids=refs,
    )


# ---------------------------------------------------------------------------
# Per-frame parse
# ---------------------------------------------------------------------------


def parse_chat_noir_frame(
    frame_bgr: np.ndarray,
    radius: int,
    geometry: Optional[ChatNoirGeometry],
) -> ParsedChatNoirFrame:
    warnings: List[str] = []
    n_cells = len(hex_lattice(radius))
    if geometry is None:
        return ParsedChatNoirFrame(radius, [], None, None, 0, set(), "grid_not_found", ["no board geometry"])

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pitch = geometry.pitch
    disk_r = max(3, int(round(SAMPLE_DISK_FRAC * pitch)))
    cell_area = math.pi * (0.5 * pitch) ** 2

    # Cat: largest orange blob; assign it to the nearest cell centre.
    orange = ((h >= CAT_HUE_LO) & (h <= CAT_HUE_HI) & (s > CAT_SAT_MIN) & (v > CAT_VAL_MIN)).astype(np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_blob, _labels, stats, centroids = cv2.connectedComponentsWithStats(orange, 8)
    min_cat_area = CAT_BLOB_AREA_FRAC * cell_area
    cat_blobs = [i for i in range(1, n_blob) if int(stats[i, cv2.CC_STAT_AREA]) >= min_cat_area]
    centers_arr = np.asarray(geometry.centers, dtype=float)
    cat_index: Optional[int] = None
    cat_center_xy: Optional[Tuple[float, float]] = None
    if cat_blobs:
        biggest = max(cat_blobs, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
        cat_xy = np.array([centroids[biggest, 0], centroids[biggest, 1]])
        cat_center_xy = (float(cat_xy[0]), float(cat_xy[1]))
        cat_index = int(np.argmin(((centers_arr - cat_xy) ** 2).sum(axis=1)))
    cat_count = len(cat_blobs)

    cells: List[ChatNoirCell] = []
    blocked: Set[int] = set()
    dark = (v < BLOCKED_VAL_MAX)
    orange_bool = orange.astype(bool)
    hh, ww = frame_bgr.shape[:2]
    for index in range(n_cells):
        cx, cy = geometry.centers[index]
        icx, icy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, icx - disk_r), min(ww, icx + disk_r + 1)
        y0, y1 = max(0, icy - disk_r), min(hh, icy + disk_r + 1)
        if x1 <= x0 or y1 <= y0:
            cells.append(ChatNoirCell(index, (cx, cy), None, 0.0, 0.0, 0.2))
            warnings.append(f"cell {index} centre off-frame")
            continue
        dark_frac = float(dark[y0:y1, x0:x1].mean())
        orange_frac = float(orange_bool[y0:y1, x0:x1].mean())
        if index == cat_index:
            state = "cat"
            conf = float(np.clip(orange_frac + 0.3, 0.0, 1.0))
        elif dark_frac >= BLOCKED_FRAC_MIN:
            state = "blocked"
            blocked.add(index)
            conf = float(np.clip(dark_frac, 0.0, 1.0))
        else:
            state = "open"
            conf = float(np.clip(1.0 - dark_frac, 0.0, 1.0))
        cells.append(ChatNoirCell(index, (cx, cy), state, dark_frac, orange_frac, conf))

    if cat_index is None:
        status = "cat_not_found"
        warnings.append(f"{cat_count} orange cat blob(s) detected")
    else:
        status = "ok"
    return ParsedChatNoirFrame(radius, cells, cat_index, cat_center_xy, cat_count, blocked, status, warnings)


def _number_edge_patch(
    frame_bgr: np.ndarray,
    center_xy: Tuple[float, float],
    pitch: float,
) -> np.ndarray:
    """Return a fill-colour-invariant edge signature of a cell's printed number."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy = (int(round(v)) for v in center_xy)
    rx = max(4, int(round(0.27 * pitch)))
    ry = max(4, int(round(0.20 * pitch)))
    h, w = gray.shape
    x0, x1 = max(0, cx - rx), min(w, cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(h, cy + ry + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((32, 48), dtype=np.uint8)
    edges = cv2.Canny(gray[y0:y1, x0:x1], 60, 150)
    return cv2.resize(edges, (48, 32), interpolation=cv2.INTER_NEAREST)


def _number_patch_similarity(reference: np.ndarray, current: np.ndarray) -> float:
    """Symmetric tolerant edge overlap in [0, 1]."""
    ref = reference > 0
    cur = current > 0
    # A blank centre must not match another blank centre: the metric requires
    # the printed number to be present, including in the reference frame.
    if float(ref.mean()) < 0.04 or float(cur.mean()) < 0.04:
        return 0.0
    kernel = np.ones((5, 5), np.uint8)
    ref_dilated = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    cur_dilated = cv2.dilate(cur.astype(np.uint8), kernel) > 0
    ref_covered = float((ref & cur_dilated).sum()) / float(ref.sum())
    cur_covered = float((cur & ref_dilated).sum()) / float(cur.sum())
    return 0.5 * (ref_covered + cur_covered)


def compare_number_set(
    reference_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    geometry: Optional[ChatNoirGeometry],
    *,
    similarity_threshold: float = 0.50,
) -> Tuple[List[int], List[int], Dict[int, float]]:
    """Compare every fixed cell's printed-number signature with frame zero.

    Cell identity comes from the rigid lattice.  Comparing centre glyph edges rather
    than raw colour makes the check insensitive to an open cell becoming blocked.
    """
    if geometry is None:
        return [], [], {}
    matched: List[int] = []
    mismatched: List[int] = []
    scores: Dict[int, float] = {}
    for index, center_xy in enumerate(geometry.centers):
        reference = _number_edge_patch(reference_bgr, center_xy, geometry.pitch)
        current = _number_edge_patch(frame_bgr, center_xy, geometry.pitch)
        score = _number_patch_similarity(reference, current)
        scores[index] = score
        (matched if score >= similarity_threshold else mismatched).append(index)
    return matched, mismatched, scores


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

_STATE_BGR = {
    # Keep ordinary open-cell guides visually neutral.  Saturated colours are
    # reserved for semantic highlights below (cat movement / newly blocked), so
    # a base-state outline cannot be mistaken for a detected transition.
    "open": (175, 175, 175),
    "blocked": (60, 60, 60),
    "cat": (50, 175, 244),
    None: (0, 0, 255),
}


def _banner(img: np.ndarray, label: str, top: int = 0) -> None:
    cv2.rectangle(img, (0, top), (img.shape[1], top + 24), (25, 25, 25), -1)
    cv2.putText(img, label, (6, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_chat_noir_overlay(
    frame_bgr: np.ndarray,
    parsed: ParsedChatNoirFrame,
    geometry: Optional[ChatNoirGeometry],
    label: str,
    *,
    highlight_indices: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    img = frame_bgr.copy()
    highlight_indices = highlight_indices or {}
    r = int(round(0.42 * geometry.pitch)) if geometry is not None else 20
    for cell in parsed.cells:
        cx, cy = int(round(cell.center_xy[0])), int(round(cell.center_xy[1]))
        color = _STATE_BGR.get(cell.state, (0, 0, 255))
        thickness = 3 if cell.state in ("cat", "blocked") else 2
        cv2.circle(img, (cx, cy), r, color, thickness)
        cv2.putText(img, str(cell.index), (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA)
        hl = highlight_indices.get(cell.index)
        if hl is not None:
            cv2.circle(img, (cx, cy), r + 4, hl, 3)
    _banner(img, label)
    return img


def draw_chat_noir_transition_overlay(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    prev: ParsedChatNoirFrame,
    cur: ParsedChatNoirFrame,
    geometry: Optional[ChatNoirGeometry],
    label: str,
    *,
    cat_move: Optional[Tuple[Optional[int], Optional[int]]] = None,
    new_blocked: Optional[Sequence[int]] = None,
) -> np.ndarray:
    prev_hl: Dict[int, Tuple[int, int, int]] = {}
    cur_hl: Dict[int, Tuple[int, int, int]] = {}
    if cat_move is not None:
        if cat_move[0] is not None:
            prev_hl[int(cat_move[0])] = (50, 175, 244)
        if cat_move[1] is not None:
            cur_hl[int(cat_move[1])] = (50, 175, 244)
    for idx in new_blocked or []:
        cur_hl[int(idx)] = (0, 215, 255)
    left = draw_chat_noir_overlay(prev_bgr, prev, geometry, "prev", highlight_indices=prev_hl)
    right = draw_chat_noir_overlay(cur_bgr, cur, geometry, "cur", highlight_indices=cur_hl)
    if left.shape[0] != right.shape[0]:
        hgt = min(left.shape[0], right.shape[0])
        left, right = left[:hgt], right[:hgt]
    combined = np.hstack([left, right])
    _banner(combined, label)
    legend = "gray=open  dark=blocked  orange=cat  yellow=new blocked"
    cv2.rectangle(
        combined,
        (0, combined.shape[0] - 22),
        (min(combined.shape[1], 470), combined.shape[0]),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        combined,
        legend,
        (6, combined.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return combined
