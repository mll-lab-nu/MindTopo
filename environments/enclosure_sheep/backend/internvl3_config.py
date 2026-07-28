from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "internvl3_local_config.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOTENV_CANDIDATES = (
    Path.cwd() / ".env",
    REPO_ROOT / ".env",
    REPO_ROOT / "lmms_eval_tasks" / ".env",
)


def _normalize_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _strip_wrapped_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def load_dotenv_candidates() -> None:
    for path in DEFAULT_DOTENV_CANDIDATES:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = _strip_wrapped_quotes(value)


def load_local_config(config_path: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    load_dotenv_candidates()
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
