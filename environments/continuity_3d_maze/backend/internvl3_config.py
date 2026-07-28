from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "internvl3_local_config.json"


def _normalize_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def load_local_config(config_path: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    if config_path:
        raw_path = Path(config_path).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (Path.cwd() / raw_path).resolve()
    else:
        path = DEFAULT_LOCAL_CONFIG_PATH

    if not path.exists():
        return path, {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"InternVL3 local config must be a JSON object: {path}")
    return path, payload


def resolve_config_value(
    cli_value: Optional[str],
    *,
    env_keys: Tuple[str, ...],
    config: Dict[str, Any],
    config_key: str,
    default: str,
) -> str:
    resolved = _normalize_string(cli_value)
    if resolved:
        return resolved

    for env_key in env_keys:
        resolved = _normalize_string(os.getenv(env_key))
        if resolved:
            return resolved

    resolved = _normalize_string(config.get(config_key))
    if resolved:
        return resolved

    return default
