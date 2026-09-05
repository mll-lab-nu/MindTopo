"""Color-segmentation baseline parser: frame -> ParsedFrame (discrete state + L1 checklist).

Each rope has a unique solid color, so color segmentation directly yields the
instance masks. Topology is not computed from pixels: we only resolve which hole
each rope's endpoint cap lands on, then hand it off to the oracle to count
crossings. Skeleton endpoints + PCA extremes are used for the initial coarse
lattice calibration / fallback; final endpoint semantics come preferentially
from the large circular caps on the pegs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import config
from .calibration import Lattice, detect_empty_holes, fit_lattice, with_measured_holes
from .config import CV
from .oracle_bridge import SolverState, state_crossings, state_logical_crossings

# Palette hues (OpenCV 0-180); each rope has a unique color -> segmenting by hue is the most robust.
_PALETTE_HEX = list(config.COLOR_NAME_BY_HEX.keys())


def _palette_hue(hexv: int) -> int:
    r, g, b = config.hex_to_rgb(hexv)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    return int(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])


_PALETTE_HUE = {h: _palette_hue(h) for h in _PALETTE_HEX}


def _hue_dist(a: np.ndarray, b: int) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - int(b))
    return np.minimum(d, 180 - d)


@dataclass
class RopeParse:
    color_hex: int
    color_name: str
    endpoints_rc: List[Optional[Tuple[int, int]]]   # length 2; None = missing/ambiguous
    n_endpoint_px: int                               # number of detected extreme endpoints (should be 2)
    mask_area: int
    ambiguous: bool = False
    n_endpoint_candidates: int = 0
    endpoint_candidates_px: List[Tuple[float, float]] = field(default_factory=list)
    selected_endpoints_px: List[Tuple[float, float]] = field(default_factory=list)
    n_unconnected_endpoint_candidates: int = 0

    @property
    def resolved(self) -> bool:
        return len(self.endpoints_rc) == 2 and all(e is not None for e in self.endpoints_rc)


@dataclass
class CrossingParse:
    x: float
    y: float
    color_a: int
    color_b: int


@dataclass
class ParsedFrame:
    status: str                                      # ok | degraded | failed
    lattice: Optional[Lattice]
    ropes: List[RopeParse]
    checklist: Dict[str, bool]
    solver_state: Optional[SolverState]              # only when all expected ropes are resolved
    crossings_visual: Optional[int]
    crossings_logical: Optional[int]
    crossing_points_px: List[CrossingParse] = field(default_factory=list)


def rope_pixel_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    return ((sat >= CV.rope_sat_min) & (val >= CV.rope_val_min)).astype(np.uint8)


def assign_colors(
    img_bgr: np.ndarray, fg: np.ndarray, expected_hex: List[int]
) -> Dict[int, np.ndarray]:
    """Classify foreground pixels by nearest palette hue; return {hex: binary mask}."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out: Dict[int, np.ndarray] = {h: np.zeros(img_bgr.shape[:2], np.uint8) for h in expected_hex}
    ys, xs = np.where(fg > 0)
    if len(ys) == 0:
        return out
    hue = hsv[ys, xs, 0]
    exp_hue = [_PALETTE_HUE[h] for h in expected_hex]
    d = np.stack([_hue_dist(hue, e) for e in exp_hue], axis=1)  # N x K
    nearest = d.argmin(axis=1)
    nd = d[np.arange(len(hue)), nearest]
    valid = nd <= CV.rope_hue_tol
    for k, h in enumerate(expected_hex):
        sel = valid & (nearest == k)
        out[h][ys[sel], xs[sel]] = 255
    return out


def _skeleton_endpoints(mask: np.ndarray) -> np.ndarray:
    """Skeleton endpoint pixels (degree 1); return an (x,y) array. Robust to curvature and occlusion breaks."""
    sk = skeletonize(mask > 0).astype(np.uint8)
    if sk.sum() == 0:
        return np.zeros((0, 2))
    # 3x3 convolution (including center): a skeleton pixel with 1 neighbor sums to 2 -> endpoint
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8))
    pts = np.argwhere((sk == 1) & (nb == 2))          # (row,col)=(y,x)
    return pts[:, ::-1].astype(float)                 # -> (x,y)


def _farthest_pair(pts: np.ndarray) -> List[Tuple[float, float]]:
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.hypot(*(pts[i] - pts[j])))
            if best is None or d > best[0]:
                best = (d, tuple(pts[i]), tuple(pts[j]))
    return [best[1], best[2]] if best else []


def _dedupe_points(pts: List[Tuple[float, float]], min_dist: float = 6.0) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in pts:
        if all(float(np.hypot(p[0] - q[0], p[1] - q[1])) >= min_dist for q in out):
            out.append(p)
    return out


def _pca_extremes(mask: np.ndarray) -> List[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = (pts - mean) @ axis
    return [tuple(pts[proj.argmin()]), tuple(pts[proj.argmax()])]


def _endpoint_candidates_from_mask(mask: np.ndarray) -> List[Tuple[float, float]]:
    """All plausible rope-end candidates, not just the final selected pair."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return []
    pts: List[Tuple[float, float]] = []
    eps = _skeleton_endpoints(mask)
    pts.extend(tuple(map(float, p)) for p in eps)
    if len(eps) >= 2:
        pts.extend(_farthest_pair(eps))
    else:
        pts.extend(_pca_extremes(mask))
    return _dedupe_points(pts)


def _legacy_endpoints_from_mask(mask: np.ndarray) -> List[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return []
    eps = _skeleton_endpoints(mask)
    if len(eps) >= 2:
        return _farthest_pair(eps)
    return _pca_extremes(mask)


def _endpoints_from_mask(mask: np.ndarray) -> List[Tuple[float, float]]:
    """The two ends of a single rope. **Does not assume a straight line**: take the
    two farthest-apart skeleton endpoints (across fragments); when the skeleton
    degenerates (closed loop / very short), fall back to the PCA principal-axis
    extremes. Returns [(x,y),(x,y)]."""
    candidates = _endpoint_candidates_from_mask(mask)
    if len(candidates) >= 2:
        return _farthest_pair(np.array(candidates, dtype=float))
    return candidates


def _coverage(mask: np.ndarray, sel: np.ndarray) -> float:
    if not bool(sel.any()):
        return 0.0
    return float((mask[sel] > 0).mean())


def _window_with_rr(
    mask: np.ndarray,
    x: float,
    y: float,
    radius: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    h, w = mask.shape[:2]
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(w, int(np.ceil(x + radius)) + 1)
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(h, int(np.ceil(y + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    roi = mask[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    return roi, rr


def _ring_exit_count(mask: np.ndarray, rr: np.ndarray, step: float) -> int:
    """Count rope-body exits outside a candidate endpoint cap.

    The old detector counted connected pieces on a thin ring right at the cap
    edge.  Thick rounded caps often intersect that ring in two arcs even when
    only one rope segment leaves the cap, so true endpoints were rejected as
    "pass-through" body points.  Count in an outer annulus instead: endpoints
    have one continuing body direction, while a rope body passing through a
    lattice node has two.
    """
    annulus = (
        (rr >= CV.endpoint_cap_exit_inner_radius_frac * step)
        & (rr <= CV.endpoint_cap_exit_outer_radius_frac * step)
    )
    annulus_mask = ((mask > 0) & annulus).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(annulus_mask, 8)
    min_area = max(3, int(CV.endpoint_cap_min_exit_area_frac * step * step))

    # Components in the same outgoing direction can be split by highlights,
    # shadows, or antialiasing.  Count angular clusters, not raw fragments.
    cy, cx = np.unravel_index(int(rr.argmin()), rr.shape)
    directions: List[Tuple[float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        dx = float(cent[i][0] - cx)
        dy = float(cent[i][1] - cy)
        if float(np.hypot(dx, dy)) <= 1e-6:
            continue
        angle = float(np.arctan2(dy, dx))
        directions.append((angle, float(area)))

    clusters: List[Tuple[float, float]] = []  # weighted unit-vector sums
    merge_cos = float(np.cos(np.deg2rad(CV.endpoint_cap_exit_merge_angle_deg)))
    for angle, weight in sorted(directions, key=lambda t: -t[1]):
        ux, uy = float(np.cos(angle)), float(np.sin(angle))
        best_idx = None
        best_dot = -1.0
        for idx, (sx, sy) in enumerate(clusters):
            norm = float(np.hypot(sx, sy))
            if norm <= 1e-6:
                continue
            dot = (ux * sx + uy * sy) / norm
            if dot > best_dot:
                best_idx = idx
                best_dot = dot
        if best_idx is not None and best_dot >= merge_cos:
            sx, sy = clusters[best_idx]
            clusters[best_idx] = (sx + ux * weight, sy + uy * weight)
        else:
            clusters.append((ux * weight, uy * weight))
    return len(clusters)


def _combined_other_mask(color_masks: Dict[int, np.ndarray], color_hex: int) -> np.ndarray:
    out: Optional[np.ndarray] = None
    for hexv, mask in color_masks.items():
        if hexv == color_hex:
            continue
        out = mask.copy() if out is None else np.maximum(out, mask)
    if out is None:
        first = next(iter(color_masks.values()))
        out = np.zeros_like(first)
    return out


def _has_competing_color_near_candidate(
    other_mask: np.ndarray,
    x: float,
    y: float,
    step: float,
) -> bool:
    if int((other_mask > 0).sum()) == 0:
        return False
    band = max(3.0, CV.endpoint_cap_ring_band_frac * step)
    max_radius = CV.endpoint_cap_ring_radius_frac * step + band
    window = _window_with_rr(other_mask, x, y, max_radius)
    if window is None:
        return False
    roi, rr = window
    disk = rr <= CV.endpoint_cap_disk_radius_frac * step
    ring = (
        (rr >= CV.endpoint_cap_ring_radius_frac * step - band)
        & (rr <= CV.endpoint_cap_ring_radius_frac * step + band)
    )
    return (
        _coverage(roi, disk) > CV.endpoint_candidate_other_disk_max_coverage
        or _coverage(roi, ring) > CV.endpoint_candidate_other_ring_max_coverage
    )


def _filter_endpoint_candidates_against_other_colors(
    candidates: List[Tuple[float, float]],
    other_mask: np.ndarray,
    step: float,
) -> List[Tuple[float, float]]:
    deduped = _dedupe_points(candidates, min_dist=max(6.0, 0.18 * step))
    filtered = [
        p for p in deduped
        if not _has_competing_color_near_candidate(other_mask, p[0], p[1], step)
    ]
    if len(filtered) >= 2 or len(deduped) < 2:
        return filtered
    return _farthest_pair(np.array(deduped, dtype=float))


def _merge_cap_and_free_endpoint_candidates(
    cap_candidates: List[Tuple[float, float]],
    free_candidates: List[Tuple[float, float]],
    step: float,
) -> List[Tuple[float, float]]:
    """Prefer peg caps, but keep a visible free endpoint when only one cap exists.

    During video generation an endpoint can be dragged between holes.  The cap
    detector only sees endpoints sitting on lattice nodes; if it finds exactly
    one cap and we replace all skeleton/PCA candidates with that single cap, the
    moving free end disappears from dynamic tracking.  Merge in far-away mask
    endpoint candidates so transient free endpoints remain visible.
    """
    if len(cap_candidates) >= 2:
        return cap_candidates
    merged = list(cap_candidates)
    min_sep = max(8.0, 0.35 * step)
    for candidate in free_candidates:
        if all(float(np.hypot(candidate[0] - cap[0], candidate[1] - cap[1])) >= min_sep for cap in cap_candidates):
            merged.append(candidate)
    return _dedupe_points(merged, min_dist=max(6.0, 0.18 * step))


def _skeleton_window_points(
    sk: np.ndarray, x: float, y: float, radius: float
) -> Optional[np.ndarray]:
    """Skeleton pixels (as (x,y) points) within `radius` of (x,y)."""
    h, w = sk.shape[:2]
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(w, int(np.ceil(x + radius)) + 1)
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(h, int(np.ceil(y + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.where(sk[y0:y1, x0:x1] > 0)
    if len(xs) == 0:
        return None
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    rr = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    pts = pts[rr <= radius]
    return pts if len(pts) else None


def _axis_from_points(pts: Optional[np.ndarray], min_linearity: float) -> Optional[np.ndarray]:
    """Unit principal axis of a point cloud, or None if it is too round (not line-like)."""
    if pts is None or len(pts) < 4:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    if float(eigvals.max() / eigvals.min()) < min_linearity:
        return None
    axis = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return None
    return axis / norm


def _points_straddle_line(
    pts: np.ndarray, x: float, y: float, direction: np.ndarray, margin: float
) -> bool:
    """True if pts lie on both sides of the line through (x,y) along `direction`."""
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    s = (pts - np.array([x, y], dtype=np.float32)) @ normal
    return bool(s.max() > margin and s.min() < -margin)


def _angle_between_axes_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(a, b)))
    dot = max(0.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _nearest_endpoint_distance(
    x: float,
    y: float,
    endpoint_candidates: List[Tuple[float, float]],
) -> Optional[float]:
    if not endpoint_candidates:
        return None
    return min(float(np.hypot(x - ex, y - ey)) for ex, ey in endpoint_candidates)


def _detect_crossing_points_from_masks(
    color_masks: Dict[int, np.ndarray],
    endpoint_candidates_by_hex: Dict[int, List[Tuple[float, float]]],
    lattice: Optional[Lattice],
) -> List[CrossingParse]:
    """Infer visual crossing points from rope skeletons.

    This is a debug overlay detector, not the oracle crossing count.  Each rope
    is segmented to a unique color, so we skeletonize each color mask to its 1px
    centerline and look for places where two centerlines meet transversally.
    Working on skeletons (not fat masks) keeps each crossing localized, so a
    dense knot is split into one marker per intersection instead of collapsing to
    a single mask-overlap centroid.

    At a real over/under crossing the under-rope's mask (hence skeleton) is broken
    by occlusion, so we (a) dilate skeletons by half a rope width to bridge that
    gap, and (b) accept a crossing where *either* rope's local skeleton straddles
    the other's line — the occluded rope may only show one arm inside the window.
    """
    if not color_masks:
        return []
    first = next(iter(color_masks.values()))
    h, w = first.shape[:2]
    if lattice is not None:
        step = (lattice.step_x + lattice.step_y) / 2.0
    else:
        step = min(h, w) / 7.0

    min_rope_area = int(CV.rope_min_area_frac * h * w)
    dilate_radius = max(2, int(round(CV.crossing_skel_dilate_frac * step)))
    split_radius = max(2, min(dilate_radius, int(round(CV.crossing_split_dilate_frac * step))))
    window_radius = max(8.0, CV.crossing_window_radius_frac * step)
    min_overlap_area = max(4, int(CV.crossing_min_overlap_area_frac * step * step))
    # Floor the merge/dedupe radii too: if a degenerate lattice yields a tiny
    # step, an un-floored merge radius collapses and a single crossing would emit
    # several near-coincident markers (over-detection) exactly when the fit is
    # already shaky.
    endpoint_radius = max(8.0, CV.crossing_endpoint_exclusion_frac * step)
    pair_merge_radius = max(8.0, CV.crossing_pair_merge_frac * step)
    dedupe_radius = max(4.0, CV.crossing_dedupe_radius_frac * step)
    straddle_margin = dilate_radius * 0.6
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilate_radius + 1, 2 * dilate_radius + 1),
    )
    split_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * split_radius + 1, 2 * split_radius + 1),
    )

    skeletons: Dict[int, np.ndarray] = {}
    for hexv, mask in color_masks.items():
        clean = _clean_mask(mask, min_rope_area)
        if int((clean > 0).sum()) < min_rope_area:
            continue
        skeletons[hexv] = skeletonize(clean > 0).astype(np.uint8)

    hexes = list(skeletons)
    hits: List[CrossingParse] = []
    for i, ha in enumerate(hexes):
        dil_a = cv2.dilate(skeletons[ha], kernel)
        split_a = cv2.dilate(skeletons[ha], split_kernel)
        for hb in hexes[i + 1:]:
            near_bridge = cv2.bitwise_and(dil_a, cv2.dilate(skeletons[hb], kernel))
            near_split = cv2.bitwise_and(split_a, cv2.dilate(skeletons[hb], split_kernel))
            split_n, _, split_stats, _ = cv2.connectedComponentsWithStats(near_split, 8)
            # The bridge radius recovers occluded under-ropes, but for shallow
            # contacts it can glue two nearby crossings into one component.
            # Prefer the smaller split radius when it yields usable components.
            has_split_components = any(
                int(split_stats[label_id, cv2.CC_STAT_AREA]) >= min_overlap_area
                for label_id in range(1, split_n)
            )
            near = near_split if has_split_components else near_bridge
            n, _, stats, cent = cv2.connectedComponentsWithStats(near, 8)
            pair_hits: List[Tuple[float, float, int]] = []
            for label_id in range(1, n):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < min_overlap_area:
                    continue
                x, y = float(cent[label_id][0]), float(cent[label_id][1])
                pts_a = _skeleton_window_points(skeletons[ha], x, y, window_radius)
                pts_b = _skeleton_window_points(skeletons[hb], x, y, window_radius)
                axis_a = _axis_from_points(pts_a, CV.crossing_min_linearity)
                axis_b = _axis_from_points(pts_b, CV.crossing_min_linearity)
                if axis_a is None or axis_b is None:
                    continue
                if _angle_between_axes_deg(axis_a, axis_b) < CV.crossing_min_angle_deg:
                    continue
                straddles_a = _points_straddle_line(pts_a, x, y, axis_b, straddle_margin)
                straddles_b = _points_straddle_line(pts_b, x, y, axis_a, straddle_margin)
                if not (straddles_a or straddles_b):
                    continue
                dist_a = _nearest_endpoint_distance(x, y, endpoint_candidates_by_hex.get(ha, []))
                dist_b = _nearest_endpoint_distance(x, y, endpoint_candidates_by_hex.get(hb, []))
                near_a = dist_a is not None and dist_a <= endpoint_radius
                near_b = dist_b is not None and dist_b <= endpoint_radius
                if near_a or near_b:
                    # Endpoint caps are fat and can create local skeleton overlap.
                    # Keep a near-endpoint hit only when a rope that is not itself
                    # at an endpoint crosses through the other rope's local axis.
                    non_endpoint_straddles = (
                        (not near_a and straddles_a)
                        or (not near_b and straddles_b)
                    )
                    if not non_endpoint_straddles:
                        continue
                pair_hits.append((x, y, area))
            # One crossing can split into several overlap components; merge this
            # pair's lobes, keeping the largest as the representative point.
            for x, y, _ in sorted(pair_hits, key=lambda t: -t[2]):
                if any(
                    hit.color_a == ha and hit.color_b == hb
                    and float(np.hypot(x - hit.x, y - hit.y)) <= pair_merge_radius
                    for hit in hits
                ):
                    continue
                hits.append(CrossingParse(x, y, ha, hb))

    # Dedupe near-identical hits arising from different color pairs.
    deduped: List[CrossingParse] = []
    for hit in hits:
        if any(float(np.hypot(hit.x - d.x, hit.y - d.y)) <= dedupe_radius for d in deduped):
            continue
        deduped.append(hit)
    return deduped


def _endpoint_cap_candidates_from_lattice(
    mask: np.ndarray,
    lattice: Lattice,
) -> Tuple[List[Tuple[float, float]], int]:
    """Detect peg endpoint caps directly from the rendered circles.

    Skeleton endpoints are good for rope paths, but they miss the visual fact
    that endpoint caps are the large colored circles at lattice nodes.  Around
    each node, a cap should fill the node center and have at most one colored
    branch leaving the cap.  A rope body passing through a node has two body
    directions beyond the cap, so it is not counted as a cap.
    """
    if int((mask > 0).sum()) == 0:
        return [], 0
    h, w = mask.shape[:2]
    step = (lattice.step_x + lattice.step_y) / 2.0
    min_core_coverage = (
        CV.endpoint_cap_lowres_min_core_coverage
        if min(h, w) < CV.endpoint_cap_lowres_max_side
        else CV.endpoint_cap_min_core_coverage
    )
    band = max(3.0, CV.endpoint_cap_ring_band_frac * step)
    max_radius = max(
        CV.endpoint_cap_ring_radius_frac * step + band,
        CV.endpoint_cap_exit_outer_radius_frac * step,
    )
    candidates: List[Tuple[float, float]] = []
    unconnected = 0
    for r in range(lattice.grid_size):
        for c in range(lattice.grid_size):
            x, y = lattice.cell_to_px(r, c)
            window = _window_with_rr(mask, x, y, max_radius)
            if window is None:
                continue
            roi, rr = window
            core = rr <= CV.endpoint_cap_core_radius_frac * step
            disk = rr <= CV.endpoint_cap_disk_radius_frac * step
            if _coverage(roi, core) < min_core_coverage:
                continue
            if _coverage(roi, disk) < CV.endpoint_cap_min_disk_coverage:
                continue
            exits = _ring_exit_count(roi, rr, step)
            if exits > CV.endpoint_cap_max_exits:
                continue
            candidates.append((float(x), float(y)))
            if exits == 0:
                unconnected += 1
    return candidates, unconnected


def _snap_endpoint_pair(
    lattice: Lattice,
    points: List[Tuple[float, float]],
) -> Tuple[List[Optional[Tuple[int, int]]], bool]:
    """Snap all endpoint candidates, then choose the most plausible two cells.

    Keeping all candidates avoids the old failure mode where a spurious middle
    protrusion became one of the two selected endpoints before the lattice knew
    where the holes were.
    """
    by_cell: Dict[Tuple[int, int], Tuple[float, bool]] = {}
    for x, y in points:
        cell, amb = _snap(lattice, x, y)
        if cell is None or not lattice.in_bounds(*cell):
            continue
        nx, ny = lattice.cell_to_px(*cell)
        dist = float(np.hypot(x - nx, y - ny))
        old = by_cell.get(cell)
        if old is None or dist < old[0]:
            by_cell[cell] = (dist, amb)
    cells = list(by_cell)
    if len(cells) >= 2:
        best = None
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                grid_dist = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                pixel_cost = by_cell[a][0] + by_cell[b][0]
                key = (-grid_dist, pixel_cost)
                if best is None or key < best[0]:
                    best = (key, a, b)
        assert best is not None
        a, b = best[1], best[2]
        ambiguous = by_cell[a][1] or by_cell[b][1]
        return [a, b], ambiguous

    snapped: List[Optional[Tuple[int, int]]] = []
    ambiguous = False
    for x, y in points[:2]:
        cell, amb = _snap(lattice, x, y)
        ambiguous = ambiguous or amb or (cell is not None and not lattice.in_bounds(*cell))
        snapped.append(cell if (cell and lattice.in_bounds(*cell)) else None)
    while len(snapped) < 2:
        snapped.append(None)
    return snapped, ambiguous


def _snap(lattice: Lattice, x: float, y: float) -> Tuple[Optional[Tuple[int, int]], bool]:
    """Endpoint pixel -> (row,col); out of range = None, ambiguous = (node, True)."""
    gs = lattice.grid_size
    nodes = [
        (r, c, *lattice.cell_to_px(r, c)) for r in range(gs) for c in range(gs)
    ]
    dists = sorted((np.hypot(x - nx, y - ny), r, c) for r, c, nx, ny in nodes)
    step = (lattice.step_x + lattice.step_y) / 2.0
    d0, r0, c0 = dists[0]
    if d0 > CV.snap_radius_frac * step:
        return None, False
    ambiguous = len(dists) > 1 and dists[1][0] < d0 * CV.snap_ambiguity_ratio
    return (r0, c0), ambiguous


def _clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def parse_frame(
    img_bgr: np.ndarray, grid_size: int, expected_hex: List[int],
    detect_crossings: bool = True,
    lattice_hint: Optional[Lattice] = None,
) -> ParsedFrame:
    """Main entry point of the color-segmentation parser. expected_hex = the rope colors expected for this episode.

    detect_crossings=False skips the per-color skeletonize crossing-point pass
    (the dominant cost) for callers that only need ropes/lattice and discard
    crossing_points_px.
    """
    h, w = img_bgr.shape[:2]
    high_res_clean_like = min(h, w) >= 900
    min_area = int(CV.rope_min_area_frac * h * w)
    checklist = {
        "hole_grid_detected": False,
        "expected_colors_present": False,
        "two_endpoints_per_rope": False,
        "endpoints_near_valid_holes": False,
        "endpoints_unambiguous_valid_holes": False,
        "no_extra_endpoint_candidates": False,
        "endpoint_caps_connected": False,
        "no_extra_missing_rope": False,
    }

    # 1) Color segmentation -> each rope's mask and its two endpoint extremes (peg positions), independent of the lattice
    fg = rope_pixel_mask(img_bgr)
    color_masks = assign_colors(img_bgr, fg, expected_hex)
    rope_masks: Dict[int, np.ndarray] = {}
    rope_ends: Dict[int, List[Tuple[float, float]]] = {}
    rope_candidates: Dict[int, List[Tuple[float, float]]] = {}
    rope_unconnected_candidates: Dict[int, int] = {hexv: 0 for hexv in expected_hex}
    peg_anchors: List[Tuple[float, float]] = []
    for hexv in expected_hex:
        mask = _clean_mask(color_masks[hexv], min_area)
        rope_masks[hexv] = mask
        if int((mask > 0).sum()) >= min_area:
            candidates = (
                _legacy_endpoints_from_mask(mask)
                if high_res_clean_like else
                _endpoint_candidates_from_mask(mask)
            )
        else:
            candidates = []
        rope_candidates[hexv] = candidates
        ends = _farthest_pair(np.array(candidates, dtype=float)) if len(candidates) >= 2 else candidates
        rope_ends[hexv] = ends
        if len(candidates) > max(8, grid_size * 2):
            peg_anchors.extend(ends)
        else:
            peg_anchors.extend(candidates)

    # 2) Calibration: fit the lattice from empty holes + peg anchors together.
    # Generated videos keep the board fixed, so callers may pass a clip-level
    # lattice hint after the first successful frame.
    if lattice_hint is None:
        holes = detect_empty_holes(img_bgr, grid_size)
        anchors = np.array(list(holes) + peg_anchors, dtype=float) if (len(holes) or peg_anchors) else np.zeros((0, 2))
        lattice = fit_lattice(anchors, grid_size, img_bgr.shape[:2])
        if lattice is None:
            return ParsedFrame("failed", None, [], checklist, None, None, None)
        lattice = with_measured_holes(img_bgr, grid_size, lattice)
    else:
        lattice = lattice_hint
    checklist["hole_grid_detected"] = True

    # In the image, endpoints are the large circular caps on the pegs. First use
    # skeleton candidates to help coarse-calibrate the lattice, then conversely
    # scan for caps at each lattice node, to avoid treating a mid-rope-body point
    # as an endpoint, while also detecting isolated caps with no rope segment attached.
    step = (lattice.step_x + lattice.step_y) / 2.0
    for hexv in expected_hex:
        if int((rope_masks[hexv] > 0).sum()) < min_area:
            continue
        other_mask = _combined_other_mask(color_masks, hexv)
        rope_candidates[hexv] = _filter_endpoint_candidates_against_other_colors(
            rope_candidates[hexv],
            other_mask,
            step,
        )
        rope_ends[hexv] = (
            _farthest_pair(np.array(rope_candidates[hexv], dtype=float))
            if len(rope_candidates[hexv]) >= 2 else
            rope_candidates[hexv]
        )
        cap_candidates, unconnected = _endpoint_cap_candidates_from_lattice(color_masks[hexv], lattice)
        cap_candidates = _filter_endpoint_candidates_against_other_colors(
            cap_candidates,
            other_mask,
            step,
        )
        if cap_candidates:
            rope_candidates[hexv] = _merge_cap_and_free_endpoint_candidates(
                cap_candidates,
                rope_candidates[hexv],
                step,
            )
            rope_ends[hexv] = (
                _farthest_pair(np.array(rope_candidates[hexv], dtype=float))
                if len(rope_candidates[hexv]) >= 2 else
                rope_candidates[hexv]
            )
            rope_unconnected_candidates[hexv] = unconnected

    # 3) Snap each rope's two ends to a hole
    ropes: List[RopeParse] = []
    for hexv in expected_hex:
        name = config.COLOR_NAME_BY_HEX[hexv]
        area = int((rope_masks[hexv] > 0).sum())
        candidates = rope_candidates[hexv]
        if area < min_area or not candidates:
            ropes.append(RopeParse(
                hexv, name, [None, None], 0, area,
                n_endpoint_candidates=0,
                n_unconnected_endpoint_candidates=rope_unconnected_candidates.get(hexv, 0),
            ))
            continue
        if high_res_clean_like:
            snapped: List[Optional[Tuple[int, int]]] = []
            ambiguous = False
            for (x, y) in rope_ends[hexv][:2]:
                cell, amb = _snap(lattice, x, y)
                ambiguous = ambiguous or amb or (cell is not None and not lattice.in_bounds(*cell))
                snapped.append(cell if (cell and lattice.in_bounds(*cell)) else None)
            while len(snapped) < 2:
                snapped.append(None)
        else:
            snapped, ambiguous = _snap_endpoint_pair(lattice, candidates)
        selected_count = sum(1 for cell in snapped if cell is not None)
        ropes.append(RopeParse(
            hexv, name, snapped, selected_count, area, ambiguous,
            n_endpoint_candidates=len(candidates),
            endpoint_candidates_px=candidates,
            selected_endpoints_px=list(rope_ends[hexv][:2]),
            n_unconnected_endpoint_candidates=rope_unconnected_candidates.get(hexv, 0),
        ))

    # Build state only from "present" ropes (mask large enough); colors absent from
    # the candidate set (e.g., medium has no cyan) should not block state construction.
    present = [r for r in ropes if r.mask_area >= min_area]
    crossing_points = (
        _detect_crossing_points_from_masks(
            color_masks,
            {r.color_hex: list(r.endpoint_candidates_px) for r in ropes},
            lattice,
        )
        if detect_crossings else []
    )
    checklist["expected_colors_present"] = bool(present)
    checklist["two_endpoints_per_rope"] = bool(present) and all(r.n_endpoint_px == 2 for r in present)
    checklist["endpoints_near_valid_holes"] = bool(present) and all(r.resolved for r in present)
    checklist["endpoints_unambiguous_valid_holes"] = (
        bool(present) and all(r.resolved and not r.ambiguous for r in present)
    )
    checklist["no_extra_endpoint_candidates"] = (
        bool(present) and all(r.n_endpoint_candidates <= 2 for r in present)
    )
    checklist["endpoint_caps_connected"] = (
        bool(present) and all(r.n_unconnected_endpoint_candidates == 0 for r in present)
    )
    checklist["no_extra_missing_rope"] = len(present) == len(expected_hex)

    all_resolved = bool(present) and all(r.resolved for r in present)
    solver_state: Optional[SolverState] = None
    cv_, cl_ = None, None
    if all_resolved:
        # Convert to the oracle's SolverState: each rope as a sorted (hole_id_a, hole_id_b)
        state = tuple(
            tuple(sorted((a[0] * grid_size + a[1], b[0] * grid_size + b[1])))
            for r in present
            for a, b in [(r.endpoints_rc[0], r.endpoints_rc[1])]
        )
        solver_state = state
        cv_ = state_crossings(state, grid_size)
        cl_ = state_logical_crossings(state, grid_size)

    clean_endpoint_evidence = (
        bool(present)
        and all(r.n_endpoint_candidates == 2 for r in present)
        and all(r.n_unconnected_endpoint_candidates == 0 for r in present)
    )
    if all_resolved and not any(r.ambiguous for r in present) and clean_endpoint_evidence:
        status = "ok"
    elif checklist["hole_grid_detected"] and present:
        status = "degraded"
    else:
        status = "failed"
    return ParsedFrame(status, lattice, ropes, checklist, solver_state, cv_, cl_, crossing_points)
