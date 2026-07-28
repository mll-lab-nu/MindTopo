from __future__ import annotations

import base64
import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import requests
from omegaconf import DictConfig, OmegaConf


@functools.lru_cache(maxsize=64)
def _png_to_data_url_cached(path: str, mtime_ns: int) -> str:
    """Read+base64-encode a PNG once per (path, mtime_ns).

    Long-history rollouts re-send the same image N times; without this, every
    request re-reads and re-encodes from disk. mtime_ns in the key invalidates
    the cache when the file changes on disk.
    """
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")

from evaluator_utils import require_remote_api_key
from key_pool import KeyPool, PoolExhausted, SharedKeyPool, _DAILY_KEYWORDS
from topobench_eval.provider_registry import resolve_model_registry_entry
from unified_types import Message, ModelResponse, SavedImage


_RATE_LIMIT_ERROR_MARKERS = (
    "-20048",
    "请求过于频繁",
    "请求过快",
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "too frequent",
)
_INTERNVL_TRANSIENT_400_MARKERS = (
    "-20017",
    "稍后再试",
    "-10006",
    "数据解码错误",
    "-10004",
    "服务请求错误",
)
_SPEED_LIMIT_ONLY_PROVIDERS = frozenset({"nvidia_nim", "gemma_api"})
_SPEED_LIMIT_INITIAL_BACKOFF_SECONDS = 16.0
_SPEED_LIMIT_MAX_BACKOFF_SECONDS = 300.0
_OPENAI_REASONING_MODEL_PREFIXES = (
    "gpt-5",
    "gpt5",
    "o1",
    "o3",
    "o4",
)


class _RequestWallTimeout(requests.Timeout):
    pass


def _uses_openai_reasoning_chat_params(*, provider_name: str, model_name: str) -> bool:
    if str(provider_name).lower() != "openai":
        return False
    normalized = str(model_name or "").lower()
    return normalized.startswith(_OPENAI_REASONING_MODEL_PREFIXES)


class BaseModelClient:
    @property
    def concurrency_hint(self) -> int:
        return 1

    @property
    def effective_system_prompt(self) -> str:
        return str(getattr(self, "system_prompt", "") or "")

    def reset_episode(self) -> None:
        return

    def generate(self, *, messages: Sequence[Message], images: Sequence[SavedImage]) -> ModelResponse:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {}

    def _key_label(self, key: str) -> str:
        labeler = getattr(getattr(self, "pool", None), "key_label", None)
        if callable(labeler):
            return labeler(key)
        suffix = key[-6:] if key else "<empty>"
        return f"key ? (...{suffix})"


def _collect_text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if value is None:
        return []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: list[str] = []
        for key in ("text", "content", "value", "reasoning_content", "output_text", "refusal"):
            if key in value:
                fragments.extend(_collect_text_fragments(value.get(key)))
        if fragments:
            return fragments
    return []


def normalize_message_content(content: Any) -> str:
    fragments = _collect_text_fragments(content)
    if fragments:
        return "\n".join(fragments).strip()
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        return ""
    return str(content).strip()


def _strip_thought_content(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _strip_thought_content(item)
            for item in value
            if not (isinstance(item, dict) and item.get("thought") is True)
        ]
    if isinstance(value, dict):
        return {key: _strip_thought_content(item) for key, item in value.items()}
    return value


def _collect_thought_fragments(value: Any) -> list[str]:
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_thought_fragments(item))
        return fragments
    if isinstance(value, dict):
        if value.get("thought") is True:
            return _collect_text_fragments(value)
        fragments: list[str] = []
        for child in value.values():
            fragments.extend(_collect_thought_fragments(child))
        return fragments
    return []


def extract_response_text_responses(payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[str]]:
    """Parser for OpenAI Responses-API payloads (`POST /responses`).

    Walks `output[]` for `message` items (output_text) and `reasoning` items
    (summary blocks). Surfaces concatenated reasoning summaries as
    `reasoning_summary` in `debug_info` so downstream prompt artifacts can
    render them for analysis.
    """
    debug_info: Dict[str, Any] = {"response_payload": payload, "response_content_source": None, "finish_reason": None}
    if not isinstance(payload, dict):
        return "", debug_info, "Unexpected response payload type."
    output = payload.get("output")
    output_text_parts: list[str] = []
    reasoning_summary_parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                            text = block.get("text", "")
                            if isinstance(text, str) and text:
                                output_text_parts.append(text)
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    for s in summary:
                        if isinstance(s, dict) and s.get("type") == "summary_text":
                            text = s.get("text", "")
                            if isinstance(text, str) and text:
                                reasoning_summary_parts.append(text)
    output_text = "".join(output_text_parts).strip()
    if not output_text:
        top = payload.get("output_text")
        if isinstance(top, str):
            output_text = top.strip()
    if reasoning_summary_parts:
        joined_summary = "\n\n".join(reasoning_summary_parts).strip()
        debug_info["reasoning_summary"] = joined_summary
        # Mirror under `reasoning_content` so the existing artifact_writer
        # block (`[reasoning_content]` / `[phase_*_reasoning_content]`)
        # surfaces Responses-API summaries without a writer change.
        debug_info.setdefault("reasoning_content", joined_summary)
    debug_info["response_content_source"] = "responses.output[message]" if output_text else None
    debug_info["finish_reason"] = payload.get("status")
    if not output_text and not reasoning_summary_parts:
        return "", debug_info, "Responses payload had no message or reasoning content."
    return output_text, debug_info, None


def extract_response_text(payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[str]]:
    debug_info: Dict[str, Any] = {"response_payload": payload, "response_content_source": None, "finish_reason": None}
    if not isinstance(payload, dict):
        return "", debug_info, "Unexpected response payload type."
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", debug_info, "Unexpected response payload: missing choices."
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", debug_info, "Unexpected response payload: first choice is not an object."
    message = first_choice.get("message") or {}
    delta = first_choice.get("delta") or {}
    # Surface the model's separate reasoning field as a first-class debug
    # entry so the prompt artifact can render it under its own header. Field
    # name varies by provider:
    #   - InternVL 3.5 / GLM 4.5V / Qwen-VL DashScope: `reasoning_content`
    #   - OpenRouter (Nemotron, DeepSeek-R1, OpenAI o-series via OR): `reasoning`
    reasoning_value = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or delta.get("reasoning_content")
        or delta.get("reasoning")
    )
    if isinstance(reasoning_value, str):
        reasoning_text = reasoning_value.strip()
    else:
        reasoning_text = "\n".join(_collect_text_fragments(reasoning_value)).strip()
    thought_text = "\n".join(_collect_thought_fragments(payload)).strip()
    if thought_text:
        reasoning_text = "\n\n".join(part for part in (reasoning_text, thought_text) if part).strip()
    if reasoning_text:
        debug_info["reasoning_content"] = reasoning_text
    candidates = [
        ("message.content", _strip_thought_content(message.get("content"))),
        ("choice.text", first_choice.get("text")),
        ("choice.content", _strip_thought_content(first_choice.get("content"))),
        ("delta.content", _strip_thought_content(delta.get("content"))),
    ]
    for source_name, candidate in candidates:
        normalized_text = normalize_message_content(candidate)
        if normalized_text:
            debug_info["response_content_source"] = source_name
            debug_info["finish_reason"] = first_choice.get("finish_reason")
            return normalized_text, debug_info, None
    if reasoning_text:
        debug_info["response_content_source"] = "reasoning_content"
        debug_info["finish_reason"] = first_choice.get("finish_reason")
        return reasoning_text, debug_info, None
    return "", debug_info, None


class OpenAICompatibleModelClient(BaseModelClient):
    def __init__(
        self,
        *,
        model_cfg: DictConfig,
        system_prompt: str,
        model_id: str,
        model_name: str,
        provider_name: str,
        base_url: str,
        pool: KeyPool,
    ) -> None:
        self.model_id = model_id
        self.base_url = str(base_url).rstrip("/")
        self.pool = pool
        self.model_name = model_name
        self.provider_name = provider_name
        temperature = getattr(model_cfg, "temperature", None)
        self.temperature = float(1.0 if temperature is None else temperature)
        # Optional. Missing or 0 → omit the cap and let the provider use its
        # own default (model picks up to its natural max_output_tokens).
        self.max_tokens = int(getattr(model_cfg, "max_tokens", 0) or 0)
        self.api_retries = int(model_cfg.api_retries)
        configured_retry_sleep = float(model_cfg.retry_sleep_seconds)
        if self.provider_name in _SPEED_LIMIT_ONLY_PROVIDERS and configured_retry_sleep <= 2.0:
            configured_retry_sleep = _SPEED_LIMIT_INITIAL_BACKOFF_SECONDS
        self.retry_sleep_seconds = configured_retry_sleep
        self.request_timeout_seconds = float(model_cfg.request_timeout_seconds)
        service_tier = str(getattr(model_cfg, "service_tier", "") or "").strip()
        self.service_tier: Optional[str] = (
            service_tier if service_tier and self.provider_name == "openai" else None
        )
        self.extra_headers = dict(model_cfg.extra_headers or {})
        extra_body_cfg = getattr(model_cfg, "extra_body", None)
        if isinstance(extra_body_cfg, DictConfig):
            extra_body_cfg = OmegaConf.to_container(extra_body_cfg, resolve=True)
        self.extra_body = dict(extra_body_cfg or {})
        self.system_prompt = system_prompt.strip()
        self._session = requests.Session()
        self._rate_limit_backoff_attempts = 0
        # OpenAI gpt-5.x and o-series reject `max_tokens` and require
        # `max_completion_tokens`. Detect by upstream model name prefix; other
        # OpenAI-compatible providers (Gemini OAI shim, OpenRouter, self-hosted)
        # all still accept `max_tokens`.
        lower_name = self.model_name.strip().lower()
        self._max_tokens_field = (
            "max_completion_tokens"
            if lower_name.startswith(("gpt-5", "o1", "o3", "o4"))
            else "max_tokens"
        )
        # Wall-clock cap on the entire generate() call (key acquire + retries +
        # rotations). Prevents indefinite hangs when all keys are cooling on
        # 429/daily-rate-limit. 0 disables.
        self.step_deadline_seconds = float(getattr(model_cfg, "step_deadline_seconds", 180.0) or 0.0)
        if 0 < self.step_deadline_seconds <= self.request_timeout_seconds + self.retry_sleep_seconds:
            # A single slow request then eats the whole step budget and
            # api_retries never fire (each episode dies on the first timeout).
            print(
                f"[model-adapter] warning: step_deadline_seconds="
                f"{self.step_deadline_seconds:.0f} barely exceeds "
                f"request_timeout_seconds={self.request_timeout_seconds:.0f}; "
                f"api_retries={self.api_retries} cannot fire after one slow "
                f"request. Raise step_deadline_seconds to at least "
                f"{(self.api_retries + 1) * self.request_timeout_seconds:.0f}.",
                flush=True,
            )
        # Opt-in to OpenAI Responses API (`POST /responses`) when the config
        # provides a `reasoning:` block. Required to surface `reasoning.summary`
        # for analysis — chat.completions cannot return summaries.
        reasoning_cfg = getattr(model_cfg, "reasoning", None)
        self._reasoning_effort: Optional[str] = None
        self._reasoning_summary: Optional[str] = None
        if reasoning_cfg is not None:
            effort = str(getattr(reasoning_cfg, "effort", "") or "").strip()
            summary = str(getattr(reasoning_cfg, "summary", "") or "").strip()
            self._reasoning_effort = effort or None
            self._reasoning_summary = summary or None
        self._use_responses_api = (
            self.provider_name == "openai"
            and (self._reasoning_effort is not None or self._reasoning_summary is not None)
        )
        verbosity = str(getattr(model_cfg, "verbosity", "") or "").strip()
        self._text_verbosity: Optional[str] = verbosity or None

    @property
    def concurrency_hint(self) -> int:
        return self.pool.concurrency_hint

    def _post_with_wall_timeout(
        self,
        *,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> requests.Response:
        if timeout_seconds <= 0:
            return self._session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.request_timeout_seconds,
            )

        connect_timeout = max(1.0, min(self.request_timeout_seconds, timeout_seconds))
        read_timeout = max(1.0, min(30.0, timeout_seconds))
        deadline = time.monotonic() + timeout_seconds

        with self._session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(connect_timeout, read_timeout),
            stream=True,
        ) as response:
            chunks: list[bytes] = []
            for chunk in response.iter_content(chunk_size=1):
                if time.monotonic() >= deadline:
                    raise _RequestWallTimeout(f"HTTP request exceeded wall timeout {timeout_seconds:.1f}s")
                if chunk:
                    chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
            return response

    def describe(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_provider": self.provider_name,
            "model_base_url": self.base_url,
        }

    def _build_headers(self, key: str) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update({str(k): str(v) for k, v in self.extra_headers.items()})
        return headers

    def _safe_response_payload(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    def _format_error_response(self, response: requests.Response) -> str:
        text = response.text.strip()
        return f"HTTP {response.status_code}: {text}" if text else f"HTTP {response.status_code}"

    def _image_to_data_url(self, image_path: Path) -> str:
        return _png_to_data_url_cached(str(image_path), image_path.stat().st_mtime_ns)

    def _serialize_content(self, content: Any, *, image_index: Mapping[str, SavedImage]) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        blocks: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                blocks.append({"type": "text", "text": str(item)})
                continue
            if item.get("type") == "text":
                blocks.append({"type": "text", "text": str(item.get("text", ""))})
            elif item.get("type") == "image_ref":
                image = image_index.get(str(item.get("ref_id", "")))
                if image is None:
                    blocks.append({"type": "text", "text": "[missing image]"})
                else:
                    blocks.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image.abs_path)}})
            else:
                blocks.append(item)
        return blocks

    def _classify_error(self, response: requests.Response) -> str:
        """Classify HTTP error for key-pool routing.

        Returns one of:
          'auth_error'       — HTTP 401 (invalid key)
          'quota_exhausted'  — HTTP 402 or 403 (billing/quota)
          'daily_exhausted'  — HTTP 429 with daily-limit body keyword → retire for session
          'rate_limited'     — HTTP 429 without daily keyword → timed cooldown
          'server_error'     — HTTP 5xx → retry same key, never retire
          'client_error'     — other 4xx → return error, no pool action
        """
        code = response.status_code
        if code == 401:
            return "auth_error"
        body = response.text.lower()
        if self.provider_name in _SPEED_LIMIT_ONLY_PROVIDERS:
            if code == 429:
                return "rate_limited"
            if any(marker in body for marker in _RATE_LIMIT_ERROR_MARKERS) or "quota" in body:
                return "rate_limited"
        if self.provider_name == "internvl" and code == 400:
            # InternVL sometimes returns transient provider failures as 400
            # invalid_request_error (negative gateway codes with a vague Chinese
            # message). Retry these on the same key instead of aborting the episode.
            if any(marker in body for marker in _INTERNVL_TRANSIENT_400_MARKERS):
                return "server_error"
        if self.provider_name == "openai" and code == 400:
            # Some OpenAI-compatible gateways surface upstream transient failures
            # as HTTP 400 bad_response_status_code/openai_error. The request
            # shape is accepted by neighboring episodes, so retry instead of
            # permanently losing the episode as a worker exception.
            if "bad_response_status_code" in body or "openai_error" in body:
                return "server_error"
        if self.provider_name == "nvidia_nim" and code == 400:
            # NVIDIA NIM returns a transient 400 while a function deployment is
            # cold/scaling/unhealthy: "DEGRADED function cannot be invoked". This
            # is a server-side condition, not a client or key error (every key
            # hits the same function id), so retry with backoff instead of
            # aborting the episode.
            if "degraded" in body or "cannot be invoked" in body:
                return "server_error"
        if code in (402, 403):
            return "quota_exhausted"
        if code == 429:
            if any(kw in body for kw in _DAILY_KEYWORDS):
                return "daily_exhausted"
            return "rate_limited"
        if any(marker in body for marker in _RATE_LIMIT_ERROR_MARKERS):
            return "rate_limited"
        if code >= 500:
            return "server_error"
        return "client_error"

    def _rate_limit_cooldown_seconds(self) -> float:
        if self.provider_name not in _SPEED_LIMIT_ONLY_PROVIDERS:
            return 30.0
        cooldown = min(
            _SPEED_LIMIT_INITIAL_BACKOFF_SECONDS * (2 ** self._rate_limit_backoff_attempts),
            _SPEED_LIMIT_MAX_BACKOFF_SECONDS,
        )
        self._rate_limit_backoff_attempts += 1
        return cooldown

    def _build_payload_messages(
        self, messages: Sequence[Message], image_index: Mapping[str, SavedImage]
    ) -> list[dict[str, Any]]:
        payload_messages = []
        if self.system_prompt:
            payload_messages.append({"role": "system", "content": self.system_prompt})
        for message in messages:
            payload_messages.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": self._serialize_content(message.get("content"), image_index=image_index),
                }
            )
        return payload_messages

    def _serialize_content_responses(
        self,
        content: Any,
        *,
        image_index: Mapping[str, SavedImage],
        role: str,
    ) -> list[dict[str, Any]]:
        """Responses-API content blocks. Role determines text type:
        assistant prior turns must use `output_text`; user/system use
        `input_text` + `input_image`.
        """
        text_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            return [{"type": text_type, "text": content}]
        if not isinstance(content, list):
            return [{"type": text_type, "text": str(content)}]
        blocks: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                blocks.append({"type": text_type, "text": str(item)})
                continue
            if item.get("type") == "text":
                blocks.append({"type": text_type, "text": str(item.get("text", ""))})
            elif item.get("type") == "image_ref":
                image = image_index.get(str(item.get("ref_id", "")))
                if image is None:
                    blocks.append({"type": text_type, "text": "[missing image]"})
                else:
                    blocks.append(
                        {
                            "type": "input_image",
                            "image_url": self._image_to_data_url(image.abs_path),
                        }
                    )
            else:
                blocks.append(item)
        return blocks

    def _build_responses_payload(
        self, *, messages: Sequence[Message], images: Sequence[SavedImage]
    ) -> Dict[str, Any]:
        image_index = {image.ref_id: image for image in images}
        input_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            input_messages.append(
                {
                    "role": role,
                    "content": self._serialize_content_responses(
                        message.get("content"), image_index=image_index, role=role
                    ),
                }
            )
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_messages,
        }
        # `max_tokens: 0` in cfg → omit `max_output_tokens` so the model uses
        # its own default cap.
        if self.max_tokens > 0:
            payload["max_output_tokens"] = self.max_tokens
        if self._text_verbosity:
            payload["text"] = {"verbosity": self._text_verbosity}
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        if self.system_prompt:
            payload["instructions"] = self.system_prompt
        reasoning: Dict[str, str] = {}
        if self._reasoning_effort:
            reasoning["effort"] = self._reasoning_effort
        if self._reasoning_summary:
            reasoning["summary"] = self._reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning
        return payload

    def _build_payload(self, *, messages: Sequence[Message], images: Sequence[SavedImage]) -> Dict[str, Any]:
        image_index = {image.ref_id: image for image in images}
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_payload_messages(messages, image_index),
        }
        if _uses_openai_reasoning_chat_params(
            provider_name=self.provider_name,
            model_name=self.model_name,
        ):
            if self.max_tokens > 0:
                payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["temperature"] = self.temperature
            if self.max_tokens > 0:
                payload["max_tokens"] = self.max_tokens
        if self.extra_body:
            payload["extra_body"] = self.extra_body
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        return payload

    def generate(self, *, messages: Sequence[Message], images: Sequence[SavedImage]) -> ModelResponse:
        if self._use_responses_api:
            payload = self._build_responses_payload(messages=messages, images=images)
            endpoint = f"{self.base_url}/responses"
            extractor = extract_response_text_responses
        else:
            payload = self._build_payload(messages=messages, images=images)
            endpoint = f"{self.base_url}/chat/completions"
            extractor = extract_response_text
        last_error: Optional[str] = None

        deadline = (
            time.monotonic() + self.step_deadline_seconds
            if self.step_deadline_seconds > 0
            else None
        )

        def _remaining() -> Optional[float]:
            if deadline is None:
                return None
            return max(0.0, deadline - time.monotonic())

        def _deadline_response() -> ModelResponse:
            return ModelResponse(
                "", True,
                f"step_deadline_exceeded after {self.step_deadline_seconds:.1f}s "
                f"(last_error={last_error or 'none'})",
                {},
            )

        # Outer loop: rotate keys on hard errors (auth/quota/daily/rate-limit).
        # pool.size + 1 ensures each key gets at most one rotation attempt.
        for _ in range(self.pool.size + 1):
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                return _deadline_response()
            acquire_timeout = min(300.0, remaining) if remaining is not None else 300.0
            try:
                key = self.pool.acquire(timeout=acquire_timeout)
            except PoolExhausted as exc:
                return ModelResponse("", True, f"PoolExhausted: {exc}", {})
            key_label = self._key_label(key)

            # Inner loop: retry 5xx on the SAME key with exponential backoff.
            # 5xx = provider infra issue, NOT a key problem — never retire on 5xx.
            try:
                for inner in range(self.api_retries + 1):
                    if _remaining() == 0:
                        return _deadline_response()
                    try:
                        remaining = _remaining()
                        wall_timeout = self.request_timeout_seconds
                        if remaining is not None and remaining > 0:
                            wall_timeout = min(wall_timeout, remaining)
                        response = self._post_with_wall_timeout(
                            endpoint=endpoint,
                            headers=self._build_headers(key),
                            payload=payload,
                            timeout_seconds=wall_timeout,
                        )
                    except requests.RequestException as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if inner < self.api_retries:
                            backoff = min(self.retry_sleep_seconds * (2 ** inner), 300.0)
                            remaining = _remaining()
                            if remaining is not None:
                                if remaining <= 0:
                                    return _deadline_response()
                                backoff = min(backoff, remaining)
                            time.sleep(backoff)
                            continue
                        self.pool.mark_transient_error(key)
                        self.pool.mark_rate_limited(key, cooldown_seconds=30.0)
                        print(f"[api-retry] network error on {key_label}; rotating key: {last_error}", flush=True)
                        break  # network retries exhausted → try next key

                    if response.status_code < 400:
                        self.pool.record_success(key)
                        self._rate_limit_backoff_attempts = 0
                        try:
                            response_payload = response.json()
                        except ValueError:
                            last_error = "Non-JSON success response."
                            print(f"[api-retry] {key_label}: {last_error}", flush=True)
                            break
                        raw_text, response_debug, extraction_error = extractor(response_payload)
                        if extraction_error is not None:
                            last_error = extraction_error
                            print(f"[api-retry] {key_label}: {last_error}", flush=True)
                            break
                        return ModelResponse(raw_text, False, None, response_debug)

                    error_class = self._classify_error(response)
                    last_error = self._format_error_response(response)

                    if error_class == "server_error":
                        # Retry same key, no retirement regardless of failure count.
                        self.pool.mark_transient_error(key)
                        if inner < self.api_retries:
                            backoff = min(self.retry_sleep_seconds * (2 ** inner), 300.0)
                            remaining = _remaining()
                            if remaining is not None:
                                if remaining <= 0:
                                    return _deadline_response()
                                backoff = min(backoff, remaining)
                            time.sleep(backoff)
                            continue
                        self.pool.mark_rate_limited(key, cooldown_seconds=30.0)
                        print(f"[api-retry] server error on {key_label}; rotating key: {last_error}", flush=True)
                        break

                    # Hard errors → act on pool, break inner loop to rotate key.
                    cooldown_note = ""
                    if error_class in ("auth_error", "quota_exhausted", "daily_exhausted"):
                        self.pool.mark_quota_exhausted(key, reason=error_class)
                    elif error_class == "rate_limited":
                        cooldown_seconds = self._rate_limit_cooldown_seconds()
                        self.pool.mark_rate_limited(key, cooldown_seconds=cooldown_seconds)
                        cooldown_note = f" cooldown={cooldown_seconds:.1f}s"
                    elif (
                        self.service_tier
                        and "service_tier" in payload
                        and "service_tier" in last_error.lower()
                    ):
                        payload = dict(payload)
                        payload.pop("service_tier", None)
                        print(
                            f"[api-retry] {key_label}: service_tier rejected; "
                            f"retrying on the default tier: {last_error}",
                            flush=True,
                        )
                        if inner < self.api_retries:
                            continue
                        break
                    else:
                        raise RuntimeError(
                            f"Non-retryable API client error from {self.provider_name} on {key_label}: {last_error}"
                        )
                    print(f"[api-retry] {error_class} on {key_label};{cooldown_note} rotating key: {last_error}", flush=True)
                    break  # exit inner loop → outer loop acquires a new key
            finally:
                self.pool.release(key)
        return ModelResponse("", True, last_error or "API request failed after rotating all keys.", {})


def load_keys_from_env(env_var: str) -> list[str]:
    """Load API keys from environment. Tries plural form first, falls back to singular.

    Splits on commas and strips whitespace. Backward-compatible: a single key
    in the singular env var works unchanged.
    """
    plural_var = env_var if env_var.endswith("S") else env_var + "S"
    raw = os.environ.get(plural_var, "").strip() or os.environ.get(env_var, "").strip()
    if not raw:
        raise ValueError(
            f"No API keys found. Set ${plural_var} (comma-separated) or ${env_var} in .env."
        )
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError(f"${plural_var} / ${env_var} is set but contains no non-empty keys.")
    return keys


def resolve_slots_per_key(model_cfg: DictConfig) -> int:
    """Workers per key: explicit model_cfg.workers_per_key wins, else env var, else 1."""
    val = getattr(model_cfg, "workers_per_key", None)
    if val is not None:
        return max(1, int(val))
    env_val = os.environ.get("TOPOBENCH_WORKERS_PER_KEY", "").strip()
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return 1


def build_key_pool(*, provider: str, keys: list[str], rpm: float, slots_per_key: int = 1) -> KeyPool:
    min_interval = 60.0 / rpm if rpm > 0 else 0.0
    if os.environ.get("TOPOBENCH_DISABLE_SHARED_KEY_POOL", "").strip() in {"1", "true", "TRUE", "yes"}:
        return KeyPool(
            provider=provider,
            keys=keys,
            min_interval_seconds=min_interval,
            slots_per_key=slots_per_key,
        )
    return SharedKeyPool(
        provider=provider,
        keys=keys,
        min_interval_seconds=min_interval,
        slots_per_key=slots_per_key,
    )


def effective_requests_per_minute(*, provider: str, configured_rpm: float) -> float:
    if configured_rpm > 0:
        return configured_rpm
    if provider in _SPEED_LIMIT_ONLY_PROVIDERS:
        return 60.0 / _SPEED_LIMIT_INITIAL_BACKOFF_SECONDS
    return 0.0


def build_model_client(*, model_cfg: DictConfig, system_prompt: str) -> Optional[BaseModelClient]:
    model_kind = str(model_cfg.kind)

    if model_kind == "local_policy":
        return None

    if model_kind == "openai_compatible":
        api_key = require_remote_api_key(model_cfg)
        provider = str(model_cfg.provider)
        rpm = effective_requests_per_minute(
            provider=provider,
            configured_rpm=float(model_cfg.requests_per_minute),
        )
        pool = build_key_pool(
            provider=provider,
            keys=[api_key],
            rpm=rpm,
            slots_per_key=resolve_slots_per_key(model_cfg),
        )
        return OpenAICompatibleModelClient(
            model_cfg=model_cfg,
            system_prompt=system_prompt,
            model_id=str(model_cfg.id),
            model_name=str(model_cfg.name),
            provider_name=provider,
            base_url=str(model_cfg.base_url),
            pool=pool,
        )

    if model_kind == "registry_remote":
        resolved = resolve_model_registry_entry(str(model_cfg.id))
        keys = load_keys_from_env(resolved.api_key_env)
        rpm = effective_requests_per_minute(
            provider=resolved.provider,
            configured_rpm=float(model_cfg.requests_per_minute),
        )
        pool = build_key_pool(
            provider=resolved.provider,
            keys=keys,
            rpm=rpm,
            slots_per_key=resolve_slots_per_key(model_cfg),
        )

        if resolved.adapter in ("openai", "gemini_api"):
            # gemini_api uses the OpenAI-compatible endpoint, no special SDK needed
            return OpenAICompatibleModelClient(
                model_cfg=model_cfg,
                system_prompt=system_prompt,
                model_id=resolved.model_id,
                model_name=resolved.upstream_model_name,
                provider_name=resolved.provider,
                base_url=resolved.base_url or "https://api.openai.com/v1",
                pool=pool,
            )

        raise ValueError(
            f"Unsupported adapter {resolved.adapter!r} for model {resolved.model_id!r}. "
            f"Supported: 'openai', 'gemini_api'."
        )

    raise ValueError(f"Unsupported model kind: {model_cfg.kind!r}")
