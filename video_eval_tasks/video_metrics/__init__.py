"""Public video-fidelity evaluator for TopoBench video runs."""

from .evaluator import evaluate_run

SCHEMA_VERSION = "video_metrics/v1"

__all__ = ["SCHEMA_VERSION", "evaluate_run"]
