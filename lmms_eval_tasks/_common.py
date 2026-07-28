"""Shared LMMS hooks for TopoBench static reasoning tasks.

This file is deliberately an adapter: contracts, parsing, normalization, and
comparison are owned by ``topobench_eval`` so every evaluation entry point
uses the same semantics.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError
try:
    from loguru import logger as eval_logger
except ImportError:  # Lightweight parser-only/test installations.
    import logging

    eval_logger = logging.getLogger(__name__)


TOPOBENCH_EVAL_ROOT = Path(__file__).resolve().parent.parent / "topobench_eval"
if str(TOPOBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPOBENCH_EVAL_ROOT))

from topobench_eval.answer_parser import parse_answer_result  # noqa: E402
from topobench_eval.answer_scoring import answers_equal  # noqa: E402
from topobench_eval.contract_resolver import resolve_answer_contract  # noqa: E402


def make_hooks(env_data_root: Path | str) -> dict[str, Callable[..., Any]]:
    """Return all LMMS hooks bound to one environment output directory."""
    root = Path(env_data_root).resolve()

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else (root / path).resolve()

    def process_docs(dataset: Any) -> Any:
        """Drop rows whose declared images are missing or corrupt."""
        def valid(doc: dict[str, Any]) -> bool:
            images = doc.get("images")
            if not isinstance(images, list) or not images:
                return False
            for raw_path in images:
                path = resolve_path(str(raw_path))
                if not path.exists():
                    return False
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError, ValueError):
                    return False
            return True

        return dataset.filter(valid)

    def doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for raw_path in doc.get("images") or []:
            try:
                images.append(Image.open(resolve_path(str(raw_path))).convert("RGB"))
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                eval_logger.warning(
                    f"[topobench] image load fail: {raw_path} ({exc}) - "
                    "process_docs should have filtered this row"
                )
        return images

    def doc_to_text(
        doc: dict[str, Any],
        lmms_eval_specific_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Return one non-conflicting answer schema.

        Generated TopoBench questions already include an ``[Answer Format]``
        section.  Legacy rows without one receive the shared contract's schema.
        """
        question = str(doc.get("question") or "").strip()
        if "[Answer Format]" in question:
            return question

        kwargs = lmms_eval_specific_kwargs or {}
        parts = [question]
        post_prompt = str(kwargs.get("post_prompt") or "").strip()
        if post_prompt:
            parts.append(post_prompt)
        parts.append(resolve_answer_contract(doc).prompt_instruction())
        return "\n".join(part for part in parts if part)

    def doc_to_target(doc: dict[str, Any], *_: Any) -> str:
        return str(doc["gt_answer"])

    def process_results(doc: dict[str, Any], results: Any) -> dict[str, dict[str, Any]]:
        raw = results[0] if isinstance(results, (list, tuple)) else results
        raw_text = "" if raw is None else str(raw)
        ground_truth = doc["gt_answer"]
        contract = resolve_answer_contract(doc)
        parsed = parse_answer_result(raw_text, contract)
        correct = parsed.semantic_valid and answers_equal(
            parsed.answer,
            ground_truth,
            contract,
        )
        instance = {
            "id": doc.get("id"),
            "type": doc.get("type", ""),
            "category": doc.get("category", ""),
            "correct": bool(correct),
            "invalid_response": parsed.invalid_response,
            "answer_format_hit": parsed.format_valid,
            "parse_method": parsed.method,
            "parse_error": parsed.error,
            "raw": raw_text,
            "predicted": parsed.answer,
            "gt": str(ground_truth),
            "legal_spec": contract.legal_spec,
        }
        return {
            "topobench_exact_match": instance,
            "json_parse_rate": instance,
            "answer_format_hit_rate": instance,
        }

    def aggregate_results(results: list[dict[str, Any]]) -> float:
        """Return overall accuracy as a percentage and log per-type values."""
        if not results:
            return 0.0
        by_type: dict[str, list[float]] = defaultdict(list)
        for result in results:
            by_type[str(result.get("type", ""))].append(float(result["correct"]))
        per_type = {
            name: round(sum(values) / len(values) * 100, 2)
            for name, values in by_type.items()
        }
        eval_logger.info(f"[topobench] per-type accuracy: {per_type}")
        return round(
            sum(value for values in by_type.values() for value in values)
            / sum(len(values) for values in by_type.values())
            * 100,
            2,
        )

    def aggregate_json_parse_rate(results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        parsed = sum(
            1
            for result in results
            if result.get("parse_method") == "json" and not result.get("invalid_response")
        )
        return round(parsed / len(results) * 100, 2)

    def aggregate_answer_format_hit_rate(results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        hits = sum(bool(result.get("answer_format_hit")) for result in results)
        return round(hits / len(results) * 100, 2)

    return {
        "process_docs": process_docs,
        "doc_to_visual": doc_to_visual,
        "doc_to_text": doc_to_text,
        "doc_to_target": doc_to_target,
        "process_results": process_results,
        "aggregate_results": aggregate_results,
        "aggregate_json_parse_rate": aggregate_json_parse_rate,
        "aggregate_answer_format_hit_rate": aggregate_answer_format_hit_rate,
    }
