from video_eval_tasks.video_metrics.compat import normalize_legacy_report


def test_legacy_metric_aliases_are_normalized_without_mutating_input() -> None:
    legacy = {"outcome_metric_breakdown": {"goal_reached": 1.0}}
    normalized = normalize_legacy_report(legacy)
    assert normalized["outcome_metric_breakdown"] == {"final_solution_valid": 1.0}
    assert legacy["outcome_metric_breakdown"] == {"goal_reached": 1.0}
