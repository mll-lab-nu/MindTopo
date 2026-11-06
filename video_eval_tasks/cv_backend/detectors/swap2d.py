"""order_swap_2d_puzzle ("2D swap") frame parser + overlay.

Design mirrors the untangle spike: topology/identity is **never** read from raw
pixels beyond what is needed to recover the discrete board state.  We localise the
regular rows x cols tile lattice from the coloured tiles, then classify each cell by
its dominant fill *hue* against the episode's known block palette (hue attribution is
robust to the video model's lighting/shading drift, far more so than absolute Lab
distance).  The blank cell is the low-saturation / near-white cell.

The result is a row-major ``arrangement`` of block tokens (``"_"`` for blank, ``None``
for unreadable), which the metrics layer compares against the env oracle GT.  Video
callers should estimate one :class:`SwapGridGeometry` for the whole clip and pass it
to every frame parse.  Re-fitting the lattice on a mid-slide frame is unsafe because
touching tiles merge into one connected component and drag the inferred row/column
centres away from the real board.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import swap_oracle

BLANK_TOKEN = swap_oracle.BLANK_TOKEN

# ---------------------------------------------------------------------------
# Palette: block id -> HSV signature (OpenCV convention: H in [0,180)).
# ---------------------------------------------------------------------------


def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _bgr_to_hsv(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    px = np.uint8([[list(bgr)]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


@dataclass(frozen=True)
class BlockColor:
    block_id: str
    name: str
    bgr: Tuple[int, int, int]
    hue: float  # OpenCV hue [0,180)
    sat: float
    val: float
    lab: Tuple[float, float, float]


def _build_palette() -> Dict[str, BlockColor]:
    palette: Dict[str, BlockColor] = {}
    for spec in swap_oracle.BLOCK_LIBRARY:
        bgr = _hex_to_bgr(spec["color"])
        h, s, v = _bgr_to_hsv(bgr)
        lab = _bgr_to_lab(np.array(bgr))
        palette[spec["id"]] = BlockColor(
            spec["id"], spec["name"], bgr, h, s, v, (float(lab[0]), float(lab[1]), float(lab[2]))
        )
    return palette


PALETTE: Dict[str, BlockColor] = _build_palette()

# CV thresholds (the one hardcoded-knob spot, like config.py for untangle).
NEAR_WHITE_S_MAX = 40      # near-white pixel: S <= this ...
NEAR_WHITE_V_MIN = 210     # ... and V >= this
NEAR_BLACK_V_MAX = 55      # near-black (badge / index text) pixel
BLANK_WHITE_FRAC_MIN = 0.55  # cell is blank if near-white fraction >= this
MIN_TILE_AREA_FRAC = 0.004  # min coloured component area as fraction of frame
TILE_SAMPLE_FRAC = 0.62    # central patch fraction of a cell used for classification
LAB_CONF_FULL = 12.0       # Lab distance to matched colour for confidence 1.0
LAB_CONF_ZERO = 60.0       # Lab distance for confidence 0.0
TILE_CENTER_LAB_MAX = 48.0  # reject pixels too far from every expected tile colour


@dataclass
class Cell:
    row: int
    col: int
    cell_index: int
    center_xy: Tuple[float, float]
    bbox_xyxy: Tuple[int, int, int, int]
    token: Optional[str]          # block id, "_" for blank, None if unreadable
    is_blank: bool
    color_id: Optional[str]       # nearest palette id (even for blanks: None)
    hue: Optional[float]
    white_frac: float             # fraction of near-white pixels in the sample patch
    confidence: float
    hue_error: Optional[float]


@dataclass(frozen=True)
class SwapGridGeometry:
    """Clip-level board geometry shared by every sampled frame."""

    col_centers: Tuple[float, ...]
    row_centers: Tuple[float, ...]
    tile_width: float
    tile_height: float
    board_bbox_xyxy: Tuple[int, int, int, int]
    reference_frame_ids: Tuple[int, ...] = ()


@dataclass
class ParsedSwapFrame:
    rows: int
    cols: int
    cells: List[Cell]
    arrangement: List[Optional[str]]  # row-major tokens
    n_blanks: int
    board_bbox_xyxy: Optional[Tuple[int, int, int, int]]
    col_centers: List[float]
    row_centers: List[float]
    parser_status: str  # "ok" | "partial" | "invalid_state" | "grid_not_found"
    warnings: List[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.parser_status == "ok" and all(t is not None for t in self.arrangement)


def detect_tile_centers(
    frame_bgr: np.ndarray,
    expected_ids: Sequence[str],
    geometry: Optional[SwapGridGeometry] = None,
) -> Dict[str, Tuple[float, float]]:
    """Locate each coloured letter tile independently of the fixed cell patches.

    Stable-state parsing deliberately samples the lattice cells.  Motion metrics need
    the complementary observation: the actual centre of a tile while it travels
    between two cells.  Assigning fill pixels to the nearest expected palette colour
    keeps touching tiles separate and ignores the white glyph cut-outs.
    """
    ids = [token for token in expected_ids if token in PALETTE]
    if not ids:
        return {}

    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h)
    if geometry is not None:
        bx0, by0, bx1, by1 = geometry.board_bbox_xyxy
        margin_x = int(round(0.2 * geometry.tile_width))
        margin_y = int(round(0.2 * geometry.tile_height))
        x0, x1 = max(0, bx0 - margin_x), min(w, bx1 + margin_x)
        y0, y1 = max(0, by0 - margin_y), min(h, by1 + margin_y)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return {}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    fill = ~(((sat <= NEAR_WHITE_S_MAX) & (val >= NEAR_WHITE_V_MIN)) | (val <= NEAR_BLACK_V_MAX))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    palette_lab = np.asarray([PALETTE[token].lab for token in ids], dtype=np.float32)
    distances = np.linalg.norm(lab[:, :, None, :] - palette_lab[None, None, :, :], axis=3)
    nearest = distances.argmin(axis=2)
    best = distances.min(axis=2)

    min_area = max(25, int(0.08 * (geometry.tile_width * geometry.tile_height))) if geometry else 100
    centers: Dict[str, Tuple[float, float]] = {}
    kernel = np.ones((3, 3), np.uint8)
    for palette_index, token in enumerate(ids):
        mask = (fill & (nearest == palette_index) & (best <= TILE_CENTER_LAB_MAX)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            continue
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if int(stats[component, cv2.CC_STAT_AREA]) < min_area:
            continue
        centers[token] = (
            float(centroids[component, 0] + x0),
            float(centroids[component, 1] + y0),
        )
    return centers


def _binary_signature(patch_bgr: np.ndarray, *, threshold: int = 185) -> str:
    """Compact 16x16 signature for white glyph pixels in a known text region."""
    if patch_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    bits = (small >= threshold).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:064x}"


def text_region_signatures(
    frame_bgr: np.ndarray,
    geometry: Optional[SwapGridGeometry],
    rows: int,
    cols: int,
) -> Dict[int, Dict[str, str]]:
    """Return visual signatures for each cell's centre letter and index number.

    The generated board uses a fixed-size dark index badge near the cell's top-left
    corner and a large white letter at its centre.  These signatures intentionally
    measure rendered text rather than infer it from the colour token.
    """
    if geometry is None:
        return {}
    h, w = frame_bgr.shape[:2]
    signatures: Dict[int, Dict[str, str]] = {}
    half_letter = max(16, int(round(0.20 * min(geometry.tile_width, geometry.tile_height))))
    for row in range(rows):
        for col in range(cols):
            cx, cy = geometry.col_centers[col], geometry.row_centers[row]
            cell_x0 = int(round(cx - 0.5 * geometry.tile_width))
            cell_y0 = int(round(cy - 0.5 * geometry.tile_height))

            lx0, lx1 = max(0, int(cx - half_letter)), min(w, int(cx + half_letter))
            ly0, ly1 = max(0, int(cy - half_letter)), min(h, int(cy + half_letter))
            letter_patch = frame_bgr[ly0:ly1, lx0:lx1]

            # Badge geometry is CSS-pixel sized and remains approximately constant
            # across grid shapes in these 1280x720 videos.
            nx0, nx1 = max(0, cell_x0 + 14), min(w, cell_x0 + 45)
            ny0, ny1 = max(0, cell_y0 + 7), min(h, cell_y0 + 38)
            number_patch = frame_bgr[ny0:ny1, nx0:nx1]
            signatures[row * cols + col] = {
                "letter": _binary_signature(letter_patch),
                "number": _binary_signature(number_patch),
            }
    return signatures


# ---------------------------------------------------------------------------
# Grid localisation
# ---------------------------------------------------------------------------


def _cluster_1d(values: Sequence[float], k: int, n_iter: int = 30) -> List[float]:
    """Tiny 1D k-means; returns sorted cluster centers (mirrors one_stroke helper)."""
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


def _tile_fill_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """A tile-fill pixel is anything that is neither the near-white background/blank
    nor near-black text.  This catches saturated tiles *and* the low-saturation gray
    Silver tile, while excluding the white page, blank interiors, and dark badges."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    near_white = (s <= NEAR_WHITE_S_MAX) & (v >= NEAR_WHITE_V_MIN)
    near_black = v <= NEAR_BLACK_V_MAX
    return (~near_white & ~near_black).astype(np.uint8) * 255


def _coloured_components(frame_bgr: np.ndarray) -> List[Dict[str, object]]:
    h, w = frame_bgr.shape[:2]
    mask = _tile_fill_mask(frame_bgr)
    # No MORPH_CLOSE: the inter-tile gaps are only a few px, so closing would fuse
    # the whole board into one blob.  OPEN alone denoises while keeping tiles apart
    # (the white centre letter is an interior hole, so it doesn't split a tile).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = MIN_TILE_AREA_FRAC * h * w
    comps: List[Dict[str, object]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH]); ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if cw < 8 or ch < 8:
            continue
        comps.append(
            {
                "bbox": (x, y, x + cw, y + ch),
                "wh": (cw, ch),
                "area": area,
                "centroid": (float(centroids[i, 0]), float(centroids[i, 1])),
            }
        )
    return comps


def _geometry_candidate(
    frame_bgr: np.ndarray,
    rows: int,
    cols: int,
    frame_id: int,
) -> Optional[Tuple[float, SwapGridGeometry]]:
    """Fit and score a possible stable-frame lattice.

    A settled board has exactly ``rows * cols - 1`` similarly-sized coloured
    components, one per non-blank cell.  Motion frames tend to have merged components,
    stretched boxes, duplicate cell assignments, or large lattice residuals.  The
    score deliberately uses geometry only; token classification happens later.
    """
    comps = _coloured_components(frame_bgr)
    expected = rows * cols - 1
    if len(comps) < max(2, expected - 1) or len(comps) > rows * cols + 1:
        return None

    xs = [float(c["centroid"][0]) for c in comps]
    ys = [float(c["centroid"][1]) for c in comps]
    col_centers = _cluster_1d(xs, cols)
    row_centers = _cluster_1d(ys, rows)
    if len(col_centers) != cols or len(row_centers) != rows:
        return None

    widths = np.asarray([float(c["wh"][0]) for c in comps], dtype=float)
    heights = np.asarray([float(c["wh"][1]) for c in comps], dtype=float)
    med_w, med_h = float(np.median(widths)), float(np.median(heights))
    if med_w <= 0 or med_h <= 0:
        return None

    assignments = []
    residuals = []
    for comp in comps:
        x, y = (float(v) for v in comp["centroid"])
        c = int(np.argmin(np.abs(np.asarray(col_centers) - x)))
        r = int(np.argmin(np.abs(np.asarray(row_centers) - y)))
        assignments.append((r, c))
        residuals.append(abs(x - col_centers[c]) / med_w + abs(y - row_centers[r]) / med_h)
    if len(set(assignments)) != len(assignments):
        return None

    size_spread = float(np.median(np.abs(widths - med_w)) / med_w)
    size_spread += float(np.median(np.abs(heights - med_h)) / med_h)
    count_penalty = 2.0 * abs(len(comps) - expected)
    residual = float(np.mean(residuals)) if residuals else 1.0
    score = count_penalty + size_spread + residual

    x0 = int(round(col_centers[0] - 0.5 * med_w))
    x1 = int(round(col_centers[-1] + 0.5 * med_w))
    y0 = int(round(row_centers[0] - 0.5 * med_h))
    y1 = int(round(row_centers[-1] + 0.5 * med_h))
    geometry = SwapGridGeometry(
        tuple(col_centers), tuple(row_centers), med_w, med_h,
        (x0, y0, x1, y1), (int(frame_id),),
    )
    return score, geometry


def estimate_swap_grid(
    frames_bgr: Sequence[np.ndarray],
    rows: int,
    cols: int,
    max_reference_frames: int = 9,
) -> Optional[SwapGridGeometry]:
    """Robustly estimate one board lattice for an entire video.

    Candidate frames are ranked by component count, size consistency, and lattice
    residual.  Taking the coordinate-wise median of the best candidates prevents one
    visually corrupted frame from determining every later cell box.
    """
    candidates: List[Tuple[float, SwapGridGeometry]] = []
    for frame_id, frame in enumerate(frames_bgr):
        candidate = _geometry_candidate(frame, rows, cols, frame_id)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:max(1, min(int(max_reference_frames), len(candidates)))]
    geometries = [geometry for _, geometry in selected]
    col_centers = tuple(
        float(np.median([g.col_centers[c] for g in geometries])) for c in range(cols)
    )
    row_centers = tuple(
        float(np.median([g.row_centers[r] for g in geometries])) for r in range(rows)
    )
    med_w = float(np.median([g.tile_width for g in geometries]))
    med_h = float(np.median([g.tile_height for g in geometries]))
    x0 = int(round(col_centers[0] - 0.5 * med_w))
    x1 = int(round(col_centers[-1] + 0.5 * med_w))
    y0 = int(round(row_centers[0] - 0.5 * med_h))
    y1 = int(round(row_centers[-1] + 0.5 * med_h))
    refs = tuple(int(g.reference_frame_ids[0]) for g in geometries)
    return SwapGridGeometry(col_centers, row_centers, med_w, med_h, (x0, y0, x1, y1), refs)


def _sample_cell_patch(frame_bgr: np.ndarray, cx: float, cy: float, half_w: float, half_h: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = max(0, int(cx - half_w)); x1 = min(w, int(cx + half_w))
    y0 = max(0, int(cy - half_h)); y1 = min(h, int(cy + half_h))
    if x1 <= x0 or y1 <= y0:
        return frame_bgr[0:1, 0:1]
    return frame_bgr[y0:y1, x0:x1]


def _classify_patch(patch_bgr: np.ndarray, expected_ids: Sequence[str]) -> Cell:
    """Classify a cell patch: blank vs which block.

    Blank == the interior is mostly near-white (a real tile fill, even the gray
    Silver, is not near-white).  Otherwise take the mean fill colour (excluding the
    white centre letter and dark badge) and match it to the nearest episode block in
    Lab space, which stays stable for both saturated and desaturated colours.
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    near_white = (s <= NEAR_WHITE_S_MAX) & (v >= NEAR_WHITE_V_MIN)
    near_black = v <= NEAR_BLACK_V_MAX
    white_frac = float(near_white.mean()) if near_white.size else 1.0

    if white_frac >= BLANK_WHITE_FRAC_MIN:
        return Cell(0, 0, 0, (0, 0), (0, 0, 0, 0), BLANK_TOKEN, True, None, None, white_frac, 0.9, None)

    fill = ~near_white & ~near_black
    if int(fill.sum()) < 25:
        fill = ~near_white  # fall back if the badge/letter dominate a tiny cell
    fill_px = patch_bgr[fill]
    if fill_px.size == 0:
        return Cell(0, 0, 0, (0, 0), (0, 0, 0, 0), None, False, None, None, white_frac, 0.0, None)
    mean_bgr = fill_px.reshape(-1, 3).mean(axis=0)
    lab = _bgr_to_lab(mean_bgr)
    hue = float(cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])

    cand = [bid for bid in expected_ids if bid in PALETTE] or list(PALETTE.keys())
    dists = sorted((float(np.linalg.norm(lab - np.array(PALETTE[bid].lab))), bid) for bid in cand)
    best_err, best_id = dists[0]
    # confidence blends absolute closeness with separation from the runner-up.
    close_conf = (LAB_CONF_ZERO - best_err) / (LAB_CONF_ZERO - LAB_CONF_FULL)
    if len(dists) > 1:
        margin = dists[1][0] - best_err
        sep_conf = margin / (margin + best_err + 1e-6)
        conf = float(np.clip(0.5 * close_conf + 0.5 * (0.4 + 0.6 * sep_conf), 0.0, 1.0))
    else:
        conf = float(np.clip(close_conf, 0.0, 1.0))
    return Cell(0, 0, 0, (0, 0), (0, 0, 0, 0), best_id, False, best_id, hue, white_frac, conf, best_err)


def parse_swap_frame(
    frame_bgr: np.ndarray,
    rows: int,
    cols: int,
    expected_ids: Sequence[str],
    geometry: Optional[SwapGridGeometry] = None,
) -> ParsedSwapFrame:
    warnings: List[str] = []
    comps = _coloured_components(frame_bgr)

    if geometry is None and (len(comps) < max(1, rows * cols - 1) - 1 or len(comps) < 2):
        # Not enough coloured tiles to fit the lattice reliably.
        return ParsedSwapFrame(
            rows, cols, [], [None] * (rows * cols), 0, None, [], [],
            "grid_not_found", warnings + [f"only {len(comps)} coloured tiles detected"],
        )

    if geometry is None:
        xs = [c["centroid"][0] for c in comps]
        ys = [c["centroid"][1] for c in comps]
        col_centers = _cluster_1d(xs, cols)
        row_centers = _cluster_1d(ys, rows)
        if len(col_centers) < cols or len(row_centers) < rows:
            return ParsedSwapFrame(
                rows, cols, [], [None] * (rows * cols), 0, None, col_centers, row_centers,
                "grid_not_found", warnings + ["could not resolve full lattice"],
            )
        med_w = float(np.median([c["wh"][0] for c in comps]))
        med_h = float(np.median([c["wh"][1] for c in comps]))
        board_bbox = (
            int(min(c["bbox"][0] for c in comps)), int(min(c["bbox"][1] for c in comps)),
            int(max(c["bbox"][2] for c in comps)), int(max(c["bbox"][3] for c in comps)),
        )
    else:
        col_centers = list(geometry.col_centers)
        row_centers = list(geometry.row_centers)
        med_w, med_h = geometry.tile_width, geometry.tile_height
        board_bbox = geometry.board_bbox_xyxy
    half_w = 0.5 * TILE_SAMPLE_FRAC * med_w
    half_h = 0.5 * TILE_SAMPLE_FRAC * med_h

    cells: List[Cell] = []
    arrangement: List[Optional[str]] = []
    for r in range(rows):
        for c in range(cols):
            cx, cy = col_centers[c], row_centers[r]
            patch = _sample_cell_patch(frame_bgr, cx, cy, half_w, half_h)
            cell = _classify_patch(patch, expected_ids)
            cell.row, cell.col, cell.cell_index = r, c, r * cols + c
            cell.center_xy = (float(cx), float(cy))
            cell.bbox_xyxy = (
                int(cx - 0.5 * med_w), int(cy - 0.5 * med_h),
                int(cx + 0.5 * med_w), int(cy + 0.5 * med_h),
            )
            cells.append(cell)
            arrangement.append(cell.token)

    n_blanks = sum(1 for cell in cells if cell.is_blank)
    status = "ok"
    low_conf = [cell for cell in cells if not cell.is_blank and cell.confidence < 0.4]
    if low_conf:
        status = "partial"
        warnings.append(f"{len(low_conf)} low-confidence cell(s)")
    legal, legal_reason = swap_oracle.is_legal_config(arrangement, expected_ids, rows, cols)
    if not legal:
        status = "invalid_state"
        warnings.append(f"not a settled legal board: {legal_reason}")
    return ParsedSwapFrame(
        rows, cols, cells, arrangement, n_blanks, board_bbox,
        col_centers, row_centers, status, warnings,
    )


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

_BLANK_DRAW = (210, 210, 210)


def draw_swap_overlay(frame_bgr: np.ndarray, parsed: ParsedSwapFrame, label: str) -> np.ndarray:
    img = frame_bgr.copy()
    for cell in parsed.cells:
        x0, y0, x1, y1 = cell.bbox_xyxy
        if cell.is_blank:
            color = _BLANK_DRAW
            text = "_"
        else:
            color = PALETTE[cell.color_id].bgr if cell.color_id in PALETTE else (0, 0, 255)
            text = f"{cell.token}"
        cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 30), 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        tag = f"{text} {cell.confidence:.2f}" if not cell.is_blank else "_ blank"
        cv2.rectangle(img, (x0, y0), (x0 + 92, y0 + 20), (20, 20, 20), -1)
        cv2.putText(img, tag, (x0 + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    _banner(img, label)
    return img


def _banner(img: np.ndarray, label: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (25, 25, 25), -1)
    cv2.putText(img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_swap_transition_overlay(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    prev: ParsedSwapFrame,
    cur: ParsedSwapFrame,
    label: str,
    swapped: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    left = draw_swap_overlay(prev_bgr, prev, "prev")
    right = draw_swap_overlay(cur_bgr, cur, "cur")
    if swapped is not None:
        for parsed, img in ((prev, left), (cur, right)):
            for idx in swapped:
                if 0 <= idx < len(parsed.cells):
                    x0, y0, x1, y1 = parsed.cells[idx].bbox_xyxy
                    cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 215, 255), 3)
    if left.shape[0] != right.shape[0]:
        hgt = min(left.shape[0], right.shape[0])
        left, right = left[:hgt], right[:hgt]
    combined = np.hstack([left, right])
    _banner(combined, label)
    return combined
