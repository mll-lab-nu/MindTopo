"""Centralized configuration: paths, color palette, CV thresholds.

All magic numbers live here (per the plan: no hardcoded constants scattered
anywhere). Default threshold values come from measurements on actual clean /
generated frames (see reports).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# ── In-repo paths ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_BACKEND = REPO_ROOT / "environments" / "knots_untangle" / "backend"
QUESTION_JSONL = REPO_ROOT / "environments" / "knots_untangle" / "output" / "question.jsonl"
# Authoritative GT rollout for this batch of videos (includes initial_crossings /
# theoretical_min_steps / the full initial state in image_prompt) — covers all 60
# easy/medium/hard items of the run.
UNTANGLED_GT = REPO_ROOT / "logs" / "video_eval" / "untangled.jsonl"

# Optional default run dir for the standalone scripts (the public evaluator
# always receives an explicit --run-dir and never reads this).
DEFAULT_RUN_DIR = Path(
    os.environ.get("TOPO_CV_RUN_DIR", str(REPO_ROOT / "logs" / "video_eval"))
)

# Writable artifact root for the spike (logs/ is already .gitignore'd)
SCRATCH_ROOT = REPO_ROOT / "logs" / "cv_backend"
FRAMES_DIR = SCRATCH_ROOT / "frames"
OUT_DIR = SCRATCH_ROOT / "out"
REPORTS_DIR = SCRATCH_ROOT / "reports"
OVERLAYS_DIR = REPORTS_DIR / "overlays"


# ── Color palette (matches the env) ──────────────────────────────────────
# Source: environments/knots_untangle/backend/internvl3_benchmark.py COLOR_NAME_BY_HEX
COLOR_NAME_BY_HEX: Dict[int, str] = {
    0xE53935: "red",
    0x43A047: "green",
    0x1E88E5: "blue",
    0xFDD835: "yellow",
    0x8E24AA: "purple",
    0xFB8C00: "orange",
    0x00ACC1: "cyan",
    0xC0CA33: "lime",
}


COLOR_HEX_BY_NAME: Dict[str, int] = {name: hexv for hexv, name in COLOR_NAME_BY_HEX.items()}


def hex_to_rgb(value: int) -> Tuple[int, int, int]:
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


# ── CV thresholds ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CVConfig:
    # HSV signature of an empty hole (measured on clean renders: V≈141, S≈76)
    hole_v_min: int = 116
    hole_v_max: int = 162
    hole_s_min: int = 50
    hole_s_max: int = 110
    # Accept window for hole radius relative to pitch (pitch = min(H,W)/(gs+1))
    hole_rad_lo_frac: float = 0.07
    hole_rad_hi_frac: float = 0.5
    hole_min_circularity: float = 0.55  # lower bound on fill ratio (area / bbox)
    hole_max_aspect: float = 1.9

    # Lattice RANSAC fit
    lattice_step_lo_frac: float = 0.72   # lower bound of step search relative to pitch
    lattice_step_hi_frac: float = 1.22
    lattice_inlier_frac: float = 0.20    # inlier test: distance to nearest node < step*this value

    # Rope pixels (high saturation)
    rope_sat_min: int = 90
    rope_val_min: int = 60
    # Assign color by hue (robust to render lighting/shading, far better than absolute
    # Lab distance). Unit is OpenCV hue (0-180); beyond the tolerance, assign no color.
    rope_hue_tol: int = 14
    rope_min_area_frac: float = 0.0008   # relative to full-frame area, denoising lower bound

    # endpoint -> hole snapping
    # When the rope is nearly straight and connects two holes, the endpoint extremum may
    # stick out by nearly half a cell due to the peg end cap / rounded corner, so we relax
    # to ~0.85*step; only beyond that is it judged missing. Ambiguity is decided by the
    # second-nearest/nearest ratio.
    snap_radius_frac: float = 0.85       # relative to step; beyond this, endpoint = missing
    snap_ambiguity_ratio: float = 1.15   # second-nearest/nearest < this ratio -> ambiguous

    # Endpoint cap detector.  Generated frames often draw the peg endpoint as a
    # fat colored cap.  A true endpoint cap covers the node center and has zero
    # or one colored branch continuing outside the cap; a rope body merely
    # passing through a node usually has two exits and should not count.
    endpoint_cap_core_radius_frac: float = 0.10
    endpoint_cap_disk_radius_frac: float = 0.20
    endpoint_cap_ring_radius_frac: float = 0.24
    endpoint_cap_ring_band_frac: float = 0.025
    endpoint_cap_exit_inner_radius_frac: float = 0.30
    endpoint_cap_exit_outer_radius_frac: float = 0.52
    endpoint_cap_min_core_coverage: float = 0.55
    endpoint_cap_lowres_min_core_coverage: float = 0.40
    endpoint_cap_lowres_max_side: int = 900
    endpoint_cap_min_disk_coverage: float = 0.15
    endpoint_cap_min_exit_area_frac: float = 0.003
    endpoint_cap_exit_merge_angle_deg: float = 35.0
    endpoint_cap_max_exits: int = 1
    endpoint_candidate_other_disk_max_coverage: float = 0.12
    endpoint_candidate_other_ring_max_coverage: float = 0.25

    # Pixel-level crossing markers for debug overlays.  Skeleton-based: each rope
    # is skeletonized to its 1px centerline and a crossing is where two
    # centerlines meet transversally.  Skeletons (not fat masks) keep each
    # crossing localized, so a dense knot is split into one marker per
    # intersection instead of collapsing to a single mask-overlap centroid.
    # Independent of the oracle crossing count.
    crossing_skel_dilate_frac: float = 0.12   # half rope-width: bridges the occlusion gap of the under-rope
    crossing_split_dilate_frac: float = 0.09  # smaller radius to split nearby shallow contacts into separate markers
    crossing_window_radius_frac: float = 0.34  # window for local skeleton-tangent PCA + transversal test
    crossing_min_overlap_area_frac: float = 0.0008
    crossing_min_linearity: float = 1.5
    crossing_min_angle_deg: float = 3.0
    crossing_endpoint_exclusion_frac: float = 0.42
    crossing_pair_merge_frac: float = 0.55     # merge same-pair lobes (one crossing split into overlap components)
    crossing_dedupe_radius_frac: float = 0.16  # merge near-identical hits across different color pairs

    # L3 temporal (dense frames)
    motion_thresh: float = 0.06          # inter-frame foreground 1-IoU below this = settled (static)
    settle_min_frames: int = 2           # minimum number of frames in a settled segment
    keystate_sample: int = 3             # how many frames to sample per settled segment for parse voting


CV = CVConfig()


def ensure_dirs() -> None:
    for d in (FRAMES_DIR, OUT_DIR, REPORTS_DIR, OVERLAYS_DIR):
        d.mkdir(parents=True, exist_ok=True)
