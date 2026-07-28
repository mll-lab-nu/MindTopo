from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPBENCH_EVAL_ROOT = REPO_ROOT / "topobench_eval"

if str(TOPBENCH_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOPBENCH_EVAL_ROOT))


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_remote_api_key(model_cfg: DictConfig) -> str:
    api_key = str(model_cfg.api_key or "").strip()
    if api_key:
        return api_key
    raise ValueError(
        f"Model {model_cfg.id!r} requires an API key. "
        f"Set the env var referenced by planning_eval_tasks/configs/model/{model_cfg.id}.yaml."
    )


def ensure_common_tool_paths() -> None:
    """Make the standard non-system bin dirs visible to subprocesses."""
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(":") if current_path else []
    for candidate in ("/opt/homebrew/bin", "/usr/local/bin"):
        if candidate not in parts and Path(candidate).exists():
            parts.insert(0, candidate)
    os.environ["PATH"] = ":".join(parts)


def ensure_runtime_settings() -> None:
    """Best-effort runtime tuning needed by the gym envs' Vite dev servers.

    - Bumps the soft FD limit so chokidar's recursive file watcher doesn't
      hit EMFILE on large dependency trees.
    - Forces chokidar to use polling instead of inotify, sidestepping
      kernel-level inotify caps that don't move with ulimit.

    Both are no-ops if the environment is already configured.
    """
    os.environ.setdefault("CHOKIDAR_USEPOLLING", "1")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(1048576, hard) if hard != resource.RLIM_INFINITY else 1048576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, ValueError, OSError):
        pass
