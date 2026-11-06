# Internal CV detector backend

This directory is an internal compatibility backend for
`video_eval_tasks.video_metrics`. It is not a second public CLI.

Production callers enter through `backend.py`, which dispatches the five supported
tasks to the detector implementation in `scripts/video_cv_pipeline.py`. The remaining
`detectors/` modules contain only the CV parsers, frame decoder, calibration helpers,
and environment-oracle bridges needed by that pipeline. `one_stroke_overlays.py` is
retained as the One Stroke frame parser.

Run the public evaluator instead of importing detector-private functions:

```bash
.venv/bin/python -m video_eval_tasks.video_metrics evaluate \
  --run-dir logs/video_eval/<task>/<run> --overlays failures
```

The backend expects a complete TopoBench checkout. Oracle metadata is loaded by the
public run loader from the configured source question JSONL, with the tracked
`environments/<task>/output/question.jsonl` files as fallbacks.
