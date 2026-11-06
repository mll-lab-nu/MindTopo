"""Compatibility mappings for pre-v1 detector/report field names."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


METRIC_ALIASES = {
    "crossing_or_solution_state": "final_solution_valid",
    "goal_reached": "final_solution_valid",
    "solution_valid": "final_solution_valid",
    "grid_move_valid": "grid_move_consistency",
    "alpha_numeric_consistency": "alpha_number_consistency",
}


def normalize_legacy_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy whose metric names follow the ``video_metrics/v1`` schema."""
    normalized = deepcopy(report)
    for breakdown_name in (
        "outcome_metric_breakdown",
        "outcome_confidence_breakdown",
        "process_metric_breakdown",
        "metric_breakdown",
        "metric_confidence_breakdown",
    ):
        breakdown = normalized.get(breakdown_name)
        if not isinstance(breakdown, dict):
            continue
        for old_name, new_name in METRIC_ALIASES.items():
            if old_name in breakdown and new_name not in breakdown:
                breakdown[new_name] = breakdown.pop(old_name)
    return normalized
