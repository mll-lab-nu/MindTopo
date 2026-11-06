"""Fail fast when a frontend cannot honour the rules saved with a Core instance."""
from __future__ import annotations


def require_frontend_rules(spec, info):
    """Fail if a stale frontend build ignores a rule saved with this instance."""
    config = (spec.reset_kwargs.get("options") or {}).get("frontend_config") or {}
    state = info.get("state") or {}
    ignored = []
    if config.get("moveRule") and state.get("moveRule") != config["moveRule"]:
        ignored.append("moveRule")
    if config.get("pipeRule") and "channelConnections" not in state and state.get("pipeRule") != config["pipeRule"]:
        ignored.append("pipeRule")
    if config.get("catPolicy") == "expert" and (state.get("catPolicy") or {}).get("name") != "expert":
        ignored.append("catPolicy")
    if ignored:
        raise RuntimeError(f"Frontend build ignores {', '.join(ignored)} for {spec.episode_id}; "
                           "rebuild it with bin/setup_frontends.sh")
