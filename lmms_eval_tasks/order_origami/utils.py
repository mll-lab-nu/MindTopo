from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import make_hooks  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError  # noqa: E402
try:
    from loguru import logger as eval_logger  # noqa: E402
except ImportError:  # Lightweight parser/test installations.
    import logging

    eval_logger = logging.getLogger(__name__)

ENV_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "environments/order_origami/output"
)

_BASE_HOOKS = make_hooks(ENV_DATA_ROOT)
globals().update(_BASE_HOOKS)

_NVIDIA_NIM_IMAGE_LIMIT = 8
_PACK_ENV = "TOPOBENCH_ORIGAMI_PACK_IMAGE_PAIRS"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_PACK_NOTE = (
    "Note: Some consecutive fold steps may be packed into one side-by-side "
    "image. Each panel is labeled Step N; follow those step labels in order."
)


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ENV_DATA_ROOT / pp).resolve()


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in _TRUTHY:
        return True
    if value in _FALSEY:
        return False
    return None


def _is_nvidia_nim_context() -> bool:
    override = _env_flag(_PACK_ENV)
    if override is not None:
        return override

    provider = os.environ.get("TOPOBENCH_OPENAI_PROVIDER", "").strip().lower()
    if provider == "nvidia_nim":
        return True

    base_url = os.environ.get("OPENAI_API_BASE", "").strip().lower()
    return "integrate.api.nvidia.com" in base_url


def _should_pack_image_pairs(doc: dict[str, Any]) -> bool:
    images = doc.get("images") or []
    return (
        isinstance(images, list)
        and len(images) > _NVIDIA_NIM_IMAGE_LIMIT
        and _is_nvidia_nim_context()
    )


def _step_label(path: str, fallback_index: int) -> str:
    match = re.search(r"step_(\d+)", Path(path).name)
    if match:
        return f"Step {int(match.group(1))}"
    return f"Step {fallback_index}"


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:
        return draw.textsize(text, font=font)


def _compose_labeled_row(panels: list[tuple[Image.Image, str]]) -> Image.Image:
    label_height = 36
    gutter = 8
    border = 2
    width = sum(image.width for image, _ in panels) + gutter * max(0, len(panels) - 1)
    height = label_height + max(image.height for image, _ in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _label_font()

    x = 0
    for image, label in panels:
        text_width, text_height = _text_size(draw, label, font)
        text_x = x + max(0, (image.width - text_width) // 2)
        text_y = max(0, (label_height - text_height) // 2)
        draw.text((text_x, text_y), label, fill=(20, 20, 20), font=font)
        canvas.paste(image, (x, label_height))
        draw.rectangle(
            [x, label_height, x + image.width - 1, label_height + image.height - 1],
            outline=(40, 40, 40),
            width=border,
        )
        x += image.width + gutter

    return canvas


def _pack_labeled_image_pairs(images: list[Image.Image], labels: list[str]) -> list[Image.Image]:
    packed = []
    for idx in range(0, len(images), 2):
        packed.append(
            _compose_labeled_row(list(zip(images[idx : idx + 2], labels[idx : idx + 2])))
        )
    return packed


def doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    if not _should_pack_image_pairs(doc):
        return _BASE_HOOKS["doc_to_visual"](doc)

    images: list[Image.Image] = []
    labels: list[str] = []
    for idx, path in enumerate(doc.get("images") or []):
        try:
            images.append(Image.open(_resolve(path)).convert("RGB"))
            labels.append(_step_label(str(path), idx))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            eval_logger.warning(
                f"[topobench] image load fail: {path} ({e}) - process_docs should have filtered this"
            )

    packed = _pack_labeled_image_pairs(images, labels)
    if len(packed) > _NVIDIA_NIM_IMAGE_LIMIT:
        eval_logger.warning(
            f"[topobench] order_origami packed {len(images)} images into {len(packed)} "
            f"images, still above NVIDIA NIM limit {_NVIDIA_NIM_IMAGE_LIMIT}"
        )
    return packed


def doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    text = _BASE_HOOKS["doc_to_text"](doc, lmms_eval_specific_kwargs)
    if not _should_pack_image_pairs(doc) or _PACK_NOTE in text:
        return text

    marker = "\nReturn JSON only:"
    idx = text.rfind(marker)
    if idx >= 0:
        return f"{text[:idx]}\n{_PACK_NOTE}{text[idx:]}"
    return f"{text}\n{_PACK_NOTE}"
