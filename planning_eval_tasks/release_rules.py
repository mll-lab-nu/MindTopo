"""Expose the explicit saved Core variant rules in each turn's prompt."""
from __future__ import annotations


def require_frontend_rules(spec, info):
    """Fail if a stale frontend build ignores a rule saved with this instance."""
    config = (spec.reset_kwargs.get("options") or {}).get("frontend_config") or {}
    state = info.get("state") or {}
    ignored = []
    if config.get("moveRule") and state.get("moveRule") != config["moveRule"]:
        ignored.append("moveRule")
    if config.get("pipeRule") and "channelConnections" not in state:
        ignored.append("pipeRule")
    if config.get("catPolicy") == "expert" and (state.get("catPolicy") or {}).get("name") != "expert":
        ignored.append("catPolicy")
    if ignored:
        raise RuntimeError(f"Frontend build ignores {', '.join(ignored)} for {spec.episode_id}; "
                           "rebuild it with bin/setup_frontends.sh")


def apply_release_rules(parts, spec):
    config = (spec.reset_kwargs.get("options") or {}).get("frontend_config") or {}
    notes = []
    restricted_swap = config.get("moveRule") == "row_column_v1"
    if restricted_swap:
        notes.append("On this board, the selected non-empty cell must share a row or column with the blank. "
                     "It need not be adjacent. Only the chosen block and blank exchange positions; intervening blocks do not move.")
    if config.get("pipeRule") == "split_channels_v2":
        notes.append("Four-port pieces contain two separate channels. A raised overpass connects north-south and east-west "
                     "without joining them and is unchanged by rotation. A double bend pairs north-east with south-west, "
                     "or north-west with south-east; one clockwise quarter-turn switches the pairing. "
                     "Each channel turns green independently, and both channels must connect to the source.")
    if config.get("catPolicy") == "expert":
        notes.append("The cat uses a deterministic expert policy. It takes an immediate escape if available, "
                     "then avoids moves allowing forced capture within one, two or three complete player turns, "
                     "then prefers shorter escape paths and more shortest escape routes.")
    if not notes:
        return parts
    seen = set()
    for blocks in (parts.model_blocks, parts.debug_blocks, parts.pure_blocks):
        if blocks is None or id(blocks) in seen:
            continue
        seen.add(id(blocks))
        if restricted_swap:
            for block in blocks:
                if block.get("type") == "text":
                    block["text"] = block["text"].replace(
                        "A legal action is any non-empty cell in the current grid.",
                        "A legal action selects a non-empty cell in the same row or column as the blank.")
        blocks.insert(0, {"type": "text", "text": "\n".join(notes)})
    return parts
