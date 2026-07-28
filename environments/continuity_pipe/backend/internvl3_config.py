from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

DEFAULT_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "internvl3_local_config.json"


def load_local_config(config_path: Optional[str] = None) -> Tuple[Optional[Path], Dict[str, Any]]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_LOCAL_CONFIG_PATH
    if not path.exists():
        return path, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def resolve_config_value(
    cli_value: Optional[str],
    *,
    env_keys: Iterable[str],
    config: Dict[str, Any],
    config_key: str,
    default: str = "",
) -> str:
    if cli_value:
        return str(cli_value)
    import os

    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value
    value = config.get(config_key)
    if value:
        return str(value)
    return default
