"""Debug visualization: draw lattice nodes, parsed endpoints, and status onto the frame, then save as PNG."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .parsing import ParsedFrame


def draw_overlay(img_bgr: np.ndarray, parsed: ParsedFrame, title: str = "") -> np.ndarray:
    vis = img_bgr.copy()
    lat = parsed.lattice
    if lat is not None:
        for r in range(lat.grid_size):
            for c in range(lat.grid_size):
                x, y = lat.cell_to_px(r, c)
                cv2.circle(vis, (int(x), int(y)), 3, (180, 180, 180), -1)
    for rope in parsed.ropes:
        rgb = config.hex_to_rgb(rope.color_hex)
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for x, y in getattr(rope, "endpoint_candidates_px", []):
            center = (int(round(x)), int(round(y)))
            cv2.circle(vis, center, 11, (0, 0, 0), 3)
            cv2.circle(vis, center, 11, bgr, 2)
            cv2.drawMarker(
                vis,
                center,
                bgr,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=11,
                thickness=1,
            )
        if (rope.n_endpoint_candidates > 2 or rope.n_unconnected_endpoint_candidates > 0) and rope.endpoint_candidates_px:
            x, y = rope.endpoint_candidates_px[0]
            label = f"cand={rope.n_endpoint_candidates}"
            if rope.n_unconnected_endpoint_candidates:
                label += f" iso={rope.n_unconnected_endpoint_candidates}"
            cv2.putText(vis, label, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(vis, label, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1)
    crossing_points = getattr(parsed, "crossing_points_px", [])
    for crossing in crossing_points:
        center = (int(round(crossing.x)), int(round(crossing.y)))
        cv2.drawMarker(
            vis,
            center,
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=5,
        )
        cv2.drawMarker(
            vis,
            center,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
    label = f"[{parsed.status}] {title}"
    if parsed.crossings_visual is not None:
        label += f" xings={parsed.crossings_visual}"
    label += f" xpts={len(crossing_points)}"
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def save_overlay(img_bgr: np.ndarray, parsed: ParsedFrame, out_path: Path, title: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), draw_overlay(img_bgr, parsed, title))
