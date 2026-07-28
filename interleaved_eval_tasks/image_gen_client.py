from __future__ import annotations

import base64
import math
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Make planning_eval_tasks importable for KeyPool / unified_types reuse.
_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))

import requests  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from PIL import Image  # noqa: E402

from key_pool import KeyPool, PoolExhausted, _DAILY_KEYWORDS  # noqa: E402
from unified_types import SavedImage  # noqa: E402


@dataclass
class ImageGenResult:
    saved_image: SavedImage
    upstream_response: Dict[str, Any]
    api_error: bool = False
    api_error_message: Optional[str] = None
    request_size: Optional[str] = None


class ImageGenError(RuntimeError):
    pass


class OpenAICompatibleImageGenClient:
    """Calls an OpenAI-compatible image endpoint to produce a single PNG.

    Two modes (selected via `image_cfg.mode`):
      - "generations" → JSON POST to /v1/images/generations (text-only).
      - "edits"       → multipart POST to /v1/images/edits with one reference
                        image + the same prompt. Used to ground the imagined
                        image in the current observation (gpt-image-1 conditions
                        on both inputs). Falls back to "generations" semantics
                        when no reference image is supplied at call time.

    Designed to mirror OpenAICompatibleModelClient: same KeyPool routing,
    same backoff, same error classification. Only the request shape differs.
    """

    _SIZE_MULTIPLE = 16
    _MIN_PIXELS = 655_360
    _MAX_PIXELS = 8_294_400
    _MAX_EDGE_EXCLUSIVE = 3_840
    _MAX_ASPECT_RATIO = 3.0

    def __init__(
        self,
        *,
        image_cfg: DictConfig,
        upstream_model_name: str,
        base_url: str,
        pool: KeyPool,
    ) -> None:
        self.upstream_model_name = upstream_model_name
        self.base_url = str(base_url).rstrip("/")
        self.pool = pool
        self.mode = str(getattr(image_cfg, "mode", "generations") or "generations").strip().lower()
        if self.mode not in ("generations", "edits"):
            raise ValueError(
                f"image_gen.mode must be 'generations' or 'edits', got {self.mode!r}."
            )
        self.size = str(image_cfg.size)
        # Empty string in the YAML means "omit the param entirely" — gpt-image-1
        # rejects an explicit response_format and always returns b64_json by default.
        self.response_format = str(image_cfg.response_format or "").strip()
        self.api_retries = int(image_cfg.api_retries)
        self.retry_sleep_seconds = float(image_cfg.retry_sleep_seconds)
        self.request_timeout_seconds = float(image_cfg.request_timeout_seconds)
        self.extra_headers = dict(image_cfg.extra_headers or {})
        self.extra_payload = dict(image_cfg.extra_payload or {})
        # Wall-clock cap on the entire generate_to_file() call (key acquire +
        # all retries + all rotations). Without this, a key pool stuck on
        # daily-rate-limit can stall a step for 10+ minutes per attempt; over
        # 600 episodes × 10 steps that turns a 6h sweep into 50h+. 0 disables.
        self.step_deadline_seconds = float(getattr(image_cfg, "step_deadline_seconds", 180.0) or 0.0)
        self._session = requests.Session()

    @classmethod
    def _match_reference_size(cls, png_bytes: bytes) -> str:
        """Round a reference PNG to a valid GPT-Image-2 edit resolution."""
        try:
            with Image.open(BytesIO(png_bytes)) as image:
                source_width, source_height = image.size
        except Exception as exc:
            raise ImageGenError(f"Cannot inspect reference image dimensions: {exc}") from exc
        if source_width <= 0 or source_height <= 0:
            raise ImageGenError(f"Invalid reference dimensions: {source_width}x{source_height}")
        if max(source_width, source_height) / min(source_width, source_height) > cls._MAX_ASPECT_RATIO:
            raise ImageGenError(
                f"Reference {source_width}x{source_height} exceeds the supported 3:1 aspect ratio"
            )

        multiple = cls._SIZE_MULTIPLE

        def nearest(value: float) -> int:
            return max(
                multiple,
                int(math.floor((value + multiple / 2) / multiple)) * multiple,
            )

        def align_up(value: float) -> int:
            return max(multiple, math.ceil(value / multiple) * multiple)

        def align_down(value: float) -> int:
            return max(multiple, math.floor(value / multiple) * multiple)

        width, height = nearest(source_width), nearest(source_height)
        pixels = width * height
        if pixels < cls._MIN_PIXELS:
            scale = math.sqrt(cls._MIN_PIXELS / pixels)
            width, height = align_up(width * scale), align_up(height * scale)
        pixels = width * height
        if pixels > cls._MAX_PIXELS or max(width, height) >= cls._MAX_EDGE_EXCLUSIVE:
            scale = min(
                math.sqrt(cls._MAX_PIXELS / pixels),
                (cls._MAX_EDGE_EXCLUSIVE - multiple) / max(width, height),
            )
            width, height = align_down(width * scale), align_down(height * scale)
        ratio = max(width, height) / min(width, height)
        pixels = width * height
        if (
            ratio > cls._MAX_ASPECT_RATIO
            or pixels < cls._MIN_PIXELS
            or pixels > cls._MAX_PIXELS
            or max(width, height) >= cls._MAX_EDGE_EXCLUSIVE
        ):
            raise ImageGenError(
                "Unable to derive a supported reference-matched size from "
                f"{source_width}x{source_height}; derived {width}x{height}"
            )
        return f"{width}x{height}"

    def _request_size(self, reference_image: Optional[Tuple[str, bytes]]) -> str:
        if self.size.strip().lower() not in {"match_input", "match_reference"}:
            return self.size
        return (
            self._match_reference_size(reference_image[1])
            if reference_image is not None
            else "1024x1024"
        )

    def _headers(self, key: str, *, multipart: bool = False) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {key}"}
        if not multipart:
            headers["Content-Type"] = "application/json"
        # If the user explicitly sets Content-Type via extra_headers we honor it
        # (e.g. some self-hosted proxies want "application/octet-stream"); for
        # multipart we let `requests` compute the boundary unless overridden.
        headers.update({str(k): str(v) for k, v in self.extra_headers.items()})
        return headers

    def _key_label(self, key: str) -> str:
        labeler = getattr(self.pool, "key_label", None)
        if callable(labeler):
            return labeler(key)
        suffix = key[-6:] if key else "<empty>"
        return f"key ? (...{suffix})"

    def _classify(self, response: requests.Response) -> str:
        code = response.status_code
        if code == 401:
            return "auth_error"
        if code in (402, 403):
            return "quota_exhausted"
        if code == 429:
            body = response.text.lower()
            if any(kw in body for kw in _DAILY_KEYWORDS):
                return "daily_exhausted"
            return "rate_limited"
        if code >= 500:
            return "server_error"
        return "client_error"

    def _format_error(self, response: requests.Response) -> str:
        text = response.text.strip()
        return f"HTTP {response.status_code}: {text}" if text else f"HTTP {response.status_code}"

    def _decode_image_payload(self, payload: Dict[str, Any]) -> bytes:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise ImageGenError(f"Unexpected image-gen payload: missing 'data' list. payload={payload!r}")
        first = data[0]
        if not isinstance(first, dict):
            raise ImageGenError(f"Unexpected image-gen payload: first item not an object. payload={payload!r}")
        b64 = first.get("b64_json")
        if isinstance(b64, str) and b64:
            try:
                return base64.b64decode(b64)
            except (ValueError, TypeError) as exc:
                raise ImageGenError(f"Failed to base64-decode image: {exc}") from exc
        url = first.get("url")
        if isinstance(url, str) and url:
            response = self._session.get(url, timeout=self.request_timeout_seconds)
            if response.status_code >= 400:
                raise ImageGenError(f"Failed to fetch image URL {url!r}: HTTP {response.status_code}")
            return response.content
        raise ImageGenError(f"Image-gen payload has neither b64_json nor url. payload={payload!r}")

    def _post(
        self,
        *,
        key: str,
        prompt: str,
        reference_image: Optional[Tuple[str, bytes]],
        request_size: str,
    ) -> requests.Response:
        """Build and send the request. /generations vs /edits is decided here.

        `reference_image`, when provided, is `(filename, png_bytes)` — read once
        by `generate_to_file` outside the retry loop so we don't re-read the
        same file across rotation/retry attempts.
        """
        use_edits = self.mode == "edits" and reference_image is not None
        if use_edits:
            endpoint = f"{self.base_url}/images/edits"
            data: Dict[str, str] = {
                "model": self.upstream_model_name,
                "prompt": prompt,
                "n": "1",
                "size": request_size,
            }
            if self.response_format:
                data["response_format"] = self.response_format
            for k, v in self.extra_payload.items():
                data[str(k)] = str(v)
            files = {"image": (reference_image[0], reference_image[1], "image/png")}
            return self._session.post(
                endpoint,
                headers=self._headers(key, multipart=True),
                data=data,
                files=files,
                timeout=self.request_timeout_seconds,
            )
        # JSON /generations (text-only) — also the fallback when edits mode
        # is requested but no reference image is supplied.
        endpoint = f"{self.base_url}/images/generations"
        request_payload: Dict[str, Any] = {
            "model": self.upstream_model_name,
            "prompt": prompt,
            "n": 1,
            "size": request_size,
        }
        if self.response_format:
            request_payload["response_format"] = self.response_format
        request_payload.update(self.extra_payload)
        return self._session.post(
            endpoint,
            headers=self._headers(key),
            json=request_payload,
            timeout=self.request_timeout_seconds,
        )

    def generate_to_file(
        self,
        *,
        prompt: str,
        save_path: Path,
        ref_id: str,
        rel_path: str,
        label: str,
        reference_image_path: Optional[Path] = None,
    ) -> ImageGenResult:
        last_error: Optional[str] = None
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Read the reference once so retries don't re-stat/re-read the file.
        reference_image: Optional[Tuple[str, bytes]] = None
        if reference_image_path is not None and self.mode == "edits":
            reference_image = (reference_image_path.name, reference_image_path.read_bytes())
        try:
            request_size = self._request_size(reference_image)
        except ImageGenError as exc:
            return ImageGenResult(
                saved_image=SavedImage(
                    ref_id=ref_id,
                    label=label,
                    abs_path=save_path,
                    rel_path=rel_path,
                ),
                upstream_response={},
                api_error=True,
                api_error_message=str(exc),
            )

        loop_start = time.monotonic()
        deadline = loop_start + self.step_deadline_seconds if self.step_deadline_seconds > 0 else None

        def _remaining() -> Optional[float]:
            if deadline is None:
                return None
            return max(0.0, deadline - time.monotonic())

        def _saved_image() -> SavedImage:
            return SavedImage(ref_id=ref_id, label=label, abs_path=save_path, rel_path=rel_path)

        def _deadline_result() -> ImageGenResult:
            return ImageGenResult(
                saved_image=_saved_image(),
                upstream_response={},
                api_error=True,
                api_error_message=(
                    f"step_deadline_exceeded after {self.step_deadline_seconds:.1f}s "
                    f"(last_error={last_error or 'none'})"
                ),
            )

        for _ in range(self.pool.size + 1):
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                return _deadline_result()
            acquire_timeout = min(300.0, remaining) if remaining is not None else 300.0
            try:
                key = self.pool.acquire(timeout=acquire_timeout)
            except PoolExhausted as exc:
                return ImageGenResult(
                    saved_image=_saved_image(),
                    upstream_response={},
                    api_error=True,
                    api_error_message=f"PoolExhausted: {exc}",
                )

            try:
                key_label = self._key_label(key)
                for inner in range(self.api_retries + 1):
                    if _remaining() == 0:
                        return _deadline_result()
                    try:
                        response = self._post(
                            key=key,
                            prompt=prompt,
                            reference_image=reference_image,
                            request_size=request_size,
                        )
                    except requests.RequestException as exc:
                        last_error = f"{key_label}: {type(exc).__name__}: {exc}"
                        if inner < self.api_retries:
                            backoff = min(self.retry_sleep_seconds * (2 ** inner), 300.0)
                            remaining = _remaining()
                            if remaining is not None:
                                if remaining <= 0:
                                    return _deadline_result()
                                backoff = min(backoff, remaining)
                            time.sleep(backoff)
                            continue
                        self.pool.mark_transient_error(key)
                        print(f"[api-retry] image-gen network error on {key_label}; rotating key: {exc}", flush=True)
                        break

                    if response.status_code < 400:
                        self.pool.record_success(key)
                        try:
                            payload = response.json()
                        except ValueError:
                            return ImageGenResult(
                                saved_image=_saved_image(),
                                upstream_response={},
                                api_error=True,
                                api_error_message=f"{key_label}: Non-JSON success response.",
                            )
                        try:
                            png_bytes = self._decode_image_payload(payload)
                        except ImageGenError as exc:
                            return ImageGenResult(
                                saved_image=_saved_image(),
                                upstream_response=payload if isinstance(payload, dict) else {},
                                api_error=True,
                                api_error_message=f"{key_label}: {exc}",
                            )
                        save_path.write_bytes(png_bytes)
                        return ImageGenResult(
                            saved_image=_saved_image(),
                            upstream_response=payload if isinstance(payload, dict) else {},
                            api_error=False,
                            api_error_message=None,
                            request_size=request_size,
                        )

                    error_class = self._classify(response)
                    response_error = self._format_error(response)
                    last_error = f"{key_label}: {response_error}"

                    if error_class == "server_error":
                        self.pool.mark_transient_error(key)
                        if inner < self.api_retries:
                            backoff = min(self.retry_sleep_seconds * (2 ** inner), 300.0)
                            remaining = _remaining()
                            if remaining is not None:
                                if remaining <= 0:
                                    return _deadline_result()
                                backoff = min(backoff, remaining)
                            time.sleep(backoff)
                            continue
                        print(
                            f"[api-retry] image-gen server error on {key_label}: {response_error}",
                            flush=True,
                        )
                        return ImageGenResult(
                            saved_image=_saved_image(),
                            upstream_response={},
                            api_error=True,
                            api_error_message=last_error,
                        )

                    if error_class in ("auth_error", "quota_exhausted", "daily_exhausted"):
                        self.pool.mark_quota_exhausted(key, reason=error_class)
                    elif error_class == "rate_limited":
                        self.pool.mark_rate_limited(key, cooldown_seconds=30.0)
                    else:
                        return ImageGenResult(
                            saved_image=_saved_image(),
                            upstream_response={},
                            api_error=True,
                            api_error_message=last_error,
                        )
                    print(
                        f"[api-retry] image-gen {error_class} on {key_label}; rotating key: {response_error}",
                        flush=True,
                    )
                    break
            finally:
                self.pool.release(key)

        return ImageGenResult(
            saved_image=_saved_image(),
            upstream_response={},
            api_error=True,
            api_error_message=last_error or "Exhausted all key rotation attempts.",
        )
