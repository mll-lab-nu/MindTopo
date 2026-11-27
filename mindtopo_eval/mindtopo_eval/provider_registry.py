from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDERS_PATH = PACKAGE_ROOT / "providers.yaml"


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    upstream_model_name: str
    adapter: str
    provider: str
    api_key_env: str
    base_url: str | None = None


def _normalize_new_schema(model_id: str, payload: Any) -> ModelRegistryEntry:
    if not isinstance(payload, dict):
        raise ValueError(f"providers.yaml entry for {model_id!r} must be an object.")
    upstream_model_name = str(payload.get("upstream_model_name", "")).strip()
    adapter = str(payload.get("adapter", "")).strip()
    provider = str(payload.get("provider", "")).strip()
    api_key_env = str(payload.get("api_key_env", "")).strip()
    if not upstream_model_name or not adapter or not provider or not api_key_env:
        raise ValueError(
            f"providers.yaml entry for {model_id!r} must define upstream_model_name, adapter, provider, and api_key_env."
        )
    base_url_raw = payload.get("base_url")
    base_url_env = str(payload.get("base_url_env", "")).strip() or None

    # Env var override takes precedence over the static YAML value.
    # Safe: this runs after load_dotenv() in every entry-point (run.py, agent_runner.py).
    resolved_base_url: str | None = None
    if base_url_env:
        env_val = os.environ.get(base_url_env, "").strip()
        if env_val:
            resolved_base_url = env_val
    if resolved_base_url is None and base_url_raw:
        resolved_base_url = str(base_url_raw).strip() or None

    return ModelRegistryEntry(
        model_id=model_id,
        upstream_model_name=upstream_model_name,
        adapter=adapter,
        provider=provider,
        api_key_env=api_key_env,
        base_url=resolved_base_url,
    )


def _normalize_legacy_schema(model_id: str, provider_name: Any, providers: Dict[str, Any]) -> ModelRegistryEntry:
    if not isinstance(provider_name, str):
        raise ValueError(f"Legacy providers.yaml model mapping for {model_id!r} must be a string provider name.")
    provider_payload = providers.get(provider_name)
    if not isinstance(provider_payload, dict):
        raise ValueError(f"Legacy providers.yaml references unknown provider {provider_name!r} for model {model_id!r}.")
    adapter = str(provider_payload.get("adapter", "")).strip()
    api_key_env = str(provider_payload.get("key_env", "")).strip()
    if not adapter or not api_key_env:
        raise ValueError(f"Legacy provider {provider_name!r} for model {model_id!r} is missing adapter or key_env.")
    base_url = provider_payload.get("base_url")
    return ModelRegistryEntry(
        model_id=model_id,
        upstream_model_name=model_id,
        adapter=adapter,
        provider=provider_name,
        api_key_env=api_key_env,
        base_url=str(base_url).strip() if base_url else None,
    )


def load_model_registry(providers_path: Path | None = None) -> Dict[str, ModelRegistryEntry]:
    resolved_path = providers_path or DEFAULT_PROVIDERS_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"providers.yaml not found at {resolved_path}")
    raw_payload = yaml.safe_load(resolved_path.read_text()) or {}
    if not isinstance(raw_payload, dict):
        raise ValueError(f"providers.yaml at {resolved_path} must define a top-level mapping.")
    raw_models = raw_payload.get("models") or {}
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(f"providers.yaml at {resolved_path} must define a non-empty models mapping.")
    legacy_providers = raw_payload.get("providers") or {}
    registry: Dict[str, ModelRegistryEntry] = {}
    for model_id, payload in raw_models.items():
        normalized_model_id = str(model_id)
        if isinstance(payload, dict):
            registry[normalized_model_id] = _normalize_new_schema(normalized_model_id, payload)
        else:
            registry[normalized_model_id] = _normalize_legacy_schema(normalized_model_id, payload, legacy_providers)
    return registry


def list_model_ids(providers_path: Path | None = None) -> list[str]:
    return sorted(load_model_registry(providers_path).keys())


def resolve_model_registry_entry(model_id: str, providers_path: Path | None = None) -> ModelRegistryEntry:
    registry = load_model_registry(providers_path)
    if model_id not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown model id {model_id!r}. Known models: {known}")
    return registry[model_id]
