from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .aggregate import aggregate_runs
from .evaluator import evaluate_run
from .manifest import import_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m video_eval_tasks.video_metrics")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate a standard video run")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--overlays", choices=("none", "failures", "all"), default="failures")
    evaluate.add_argument("--sampling-mode", choices=("sample_41", "all_frames"), default="sample_41")
    aggregate = commands.add_parser("aggregate", help="build a cross-run benchmark report")
    aggregate.add_argument("--runs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    convert = commands.add_parser("import-manifest", help="convert an explicit loose-video manifest into a standard run")
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--output-run-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        report = evaluate_run(args.run_dir, args.output_dir, args.overlays, args.sampling_mode)
    elif args.command == "aggregate":
        report = aggregate_runs(args.runs, args.output_dir)
    else:
        output = import_manifest(args.manifest, args.output_run_dir)
        report = {"output_run_dir": str(output), "model_answer": "model_answer.jsonl"}
    print(json.dumps(report, indent=2))
    return 0
