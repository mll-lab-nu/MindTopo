"""Volcengine Ark async video-generation client (Seedance i2v)."""

from __future__ import annotations

import base64
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PLANNING_DIR = _THIS_DIR.parent / "planning_eval_tasks"
if str(_PLANNING_DIR) not in sys.path:
    sys.path.append(str(_PLANNING_DIR))

import requests  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from key_pool import KeyPool, PoolExhausted, _DAILY_KEYWORDS  # noqa: E402
from model_adapter import build_key_pool, load_keys_from_env, resolve_slots_per_key  # noqa: E402
from two_phase_helpers import resolve_base_url_from_cfg  # noqa: E402
from video_gen_client import VideoGenResult  # noqa: E402


_TERMINAL = {"succeeded", "failed", "expired", "cancelled", "canceled"}


def _to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _find_video_url(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("video_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
        for value in payload.values():
            found = _find_video_url(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_video_url(value)
            if found:
                return found
    return None


class ArkVideoGenClient:
    def __init__(
        self,
        *,
        video_cfg: DictConfig,
        upstream_model_name: str,
        base_url: str,
        pool: KeyPool,
    ) -> None:
        self.upstream_model_name = upstream_model_name
        self.base_url = base_url.rstrip("/")
        self.pool = pool
        self.mode = str(getattr(video_cfg, "mode", "i2v") or "i2v").lower()
        if self.mode not in {"i2v", "t2v"}:
            raise ValueError(f"Ark mode must be i2v or t2v, got {self.mode!r}")
        self.api_retries = int(video_cfg.api_retries)
        self.retry_sleep_seconds = float(video_cfg.retry_sleep_seconds)
        self.request_timeout_seconds = float(video_cfg.request_timeout_seconds)
        self.poll_interval_seconds = float(
            getattr(video_cfg, "queue_poll_interval_seconds", 5.0) or 5.0
        )
        self.extra_headers = dict(video_cfg.extra_headers or {})
        self.extra_payload = dict(video_cfg.extra_payload or {})
        self.fps = int(getattr(video_cfg, "fps", 24) or 24)
        self.num_frames = int(getattr(video_cfg, "num_frames", 0) or 0)
        self._session = requests.Session()

    @property
    def _create_url(self) -> str:
        return f"{self.base_url}/contents/generations/tasks"

    def _task_url(self, task_id: str) -> str:
        return f"{self._create_url}/{task_id}"

    def _headers(self, key: str) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update({str(k): str(v) for k, v in self.extra_headers.items()})
        return headers

    def _key_label(self, key: str) -> str:
        labeler = getattr(self.pool, "key_label", None)
        return labeler(key) if callable(labeler) else f"key (...{key[-6:]})"

    @staticmethod
    def _classify(response: requests.Response) -> str:
        if response.status_code == 401:
            return "auth_error"
        if response.status_code in {402, 403}:
            return "quota_exhausted"
        if response.status_code == 429:
            return (
                "daily_exhausted"
                if any(word in response.text.lower() for word in _DAILY_KEYWORDS)
                else "rate_limited"
            )
        if response.status_code >= 500:
            return "server_error"
        return "client_error"

    def _body(self, *, prompt: str, reference_image_path: Optional[Path]) -> Dict[str, Any]:
        content = [{"type": "text", "text": prompt}]
        if reference_image_path is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _to_data_uri(reference_image_path)},
                    "role": "reference_image",
                }
            )
        return {
            "model": self.upstream_model_name,
            "content": content,
            **self.extra_payload,
        }

    def _poll(self, *, key: str, task_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        deadline = time.monotonic() + self.request_timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_seconds)
            try:
                response = self._session.get(
                    self._task_url(task_id),
                    headers=self._headers(key),
                    timeout=min(self.request_timeout_seconds, 120.0),
                )
            except requests.RequestException:
                continue
            if response.status_code == 429 or response.status_code >= 500:
                continue
            if response.status_code >= 400:
                return "unknown", {}, f"poll HTTP {response.status_code}: {response.text.strip()}"
            try:
                payload = response.json()
            except ValueError:
                return "unknown", {}, "poll returned non-JSON response"
            status = str(payload.get("status", "")).lower()
            if status in _TERMINAL:
                return status, payload, None
        return "unknown", {}, f"poll timeout after {self.request_timeout_seconds:.0f}s"

    def generate_to_file(
        self,
        *,
        prompt: str,
        save_path: Path,
        reference_image_path: Optional[Path] = None,
    ) -> VideoGenResult:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "i2v" and reference_image_path is None:
            return VideoGenResult(
                save_path, {}, True, "Ark i2v requires a reference image."
            )
        body = self._body(prompt=prompt, reference_image_path=reference_image_path)
        last_error: Optional[str] = None
        for _ in range(self.pool.size + 1):
            try:
                key = self.pool.acquire(timeout=300.0)
            except PoolExhausted as exc:
                return VideoGenResult(save_path, {}, True, f"PoolExhausted: {exc}")
            try:
                label = self._key_label(key)
                task_id: Optional[str] = None
                rotate = False
                for attempt in range(self.api_retries + 1):
                    try:
                        response = self._session.post(
                            self._create_url,
                            headers=self._headers(key),
                            json=body,
                            timeout=min(self.request_timeout_seconds, 120.0),
                        )
                    except requests.RequestException as exc:
                        last_error = f"{label}: {type(exc).__name__}: {exc}"
                        if attempt < self.api_retries:
                            time.sleep(min(self.retry_sleep_seconds * 2**attempt, 60.0))
                            continue
                        self.pool.mark_transient_error(key)
                        rotate = True
                        break
                    if response.status_code < 400:
                        self.pool.record_success(key)
                        try:
                            created = response.json()
                        except ValueError:
                            return VideoGenResult(save_path, {}, True, f"{label}: non-JSON create response")
                        task_id = created.get("id") or created.get("task_id")
                        if not task_id:
                            return VideoGenResult(save_path, created, True, f"{label}: no task id")
                        break
                    kind = self._classify(response)
                    last_error = f"{label}: HTTP {response.status_code}: {response.text.strip()}"
                    if kind == "server_error" and attempt < self.api_retries:
                        time.sleep(min(self.retry_sleep_seconds * 2**attempt, 60.0))
                        continue
                    if kind in {"auth_error", "quota_exhausted", "daily_exhausted"}:
                        self.pool.mark_quota_exhausted(key, reason=kind)
                        rotate = True
                        break
                    if kind == "rate_limited":
                        self.pool.mark_rate_limited(key, cooldown_seconds=30.0)
                        rotate = True
                        break
                    return VideoGenResult(save_path, {}, True, last_error)
                if task_id is None:
                    if rotate:
                        continue
                    return VideoGenResult(save_path, {}, True, last_error or "Ark create failed")

                status, task, poll_error = self._poll(key=key, task_id=task_id)
                if poll_error or status != "succeeded":
                    message = poll_error or f"task ended with status={status!r}"
                    return VideoGenResult(save_path, task, True, f"{label}: {message}")
                video_url = _find_video_url(task.get("content", task))
                if not video_url:
                    return VideoGenResult(save_path, task, True, f"{label}: no video URL")
                try:
                    download = self._session.get(
                        video_url, timeout=max(self.request_timeout_seconds, 120.0)
                    )
                    download.raise_for_status()
                    save_path.write_bytes(download.content)
                except requests.RequestException as exc:
                    return VideoGenResult(
                        save_path, task, True, f"{label}: download {type(exc).__name__}: {exc}"
                    )
                duration = self.extra_payload.get("duration")
                return VideoGenResult(
                    video_path=save_path,
                    upstream_response={
                        "provider": "volcengine_ark",
                        "model": self.upstream_model_name,
                        "task_id": task_id,
                        "bytes": len(download.content),
                        "usage": task.get("usage"),
                    },
                    fps=self.fps,
                    num_frames=self.num_frames or None,
                    duration_seconds=float(duration) if duration is not None else None,
                )
            finally:
                self.pool.release(key)
        return VideoGenResult(save_path, {}, True, last_error or "Ark exhausted key rotation")


def build_ark_video_gen_client(video_cfg: DictConfig) -> ArkVideoGenClient:
    base_url = resolve_base_url_from_cfg(video_cfg, label="Video-gen (Ark)")
    keys = load_keys_from_env(str(video_cfg.api_key_env))
    pool = build_key_pool(
        provider=f"video_gen:{video_cfg.upstream_model_name}",
        keys=keys,
        rpm=float(getattr(video_cfg, "requests_per_minute", 0.0) or 0.0),
        slots_per_key=resolve_slots_per_key(video_cfg),
    )
    return ArkVideoGenClient(
        video_cfg=video_cfg,
        upstream_model_name=str(video_cfg.upstream_model_name),
        base_url=base_url,
        pool=pool,
    )
