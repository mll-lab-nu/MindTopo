from __future__ import annotations

import argparse

from env import KnotsUntangleEnv


def assert_state_response(payload: dict) -> None:
    required = {"observation", "reward", "done", "success", "step_count", "info"}
    missing = required.difference(payload)
    assert not missing, f"Missing protocol fields: {sorted(missing)}"
    assert isinstance(payload["done"], bool)
    assert isinstance(payload["success"], bool)
    assert isinstance(payload["step_count"], int)
    assert isinstance(payload["info"], dict)
    assert "state" in payload["info"]


def first_explicit_legal_action(symbolic: dict, grid_size: int) -> dict:
    occupied = set()
    endpoints = []
    for rope in symbolic.get("ropes", []):
        for key in ("startHole", "endHole"):
            hole = rope.get(key) or {}
            row = int(hole["row"])
            col = int(hole["col"])
            occupied.add((row, col))
            endpoints.append((row, col))

    empty = [
        (row, col)
        for row in range(grid_size)
        for col in range(grid_size)
        if (row, col) not in occupied
    ]
    assert endpoints and empty, "expected at least one endpoint and one empty hole"
    src_row, src_col = endpoints[0]
    tgt_row, tgt_col = empty[0]
    return {
        "src_row": src_row,
        "src_col": src_col,
        "tgt_row": tgt_row,
        "tgt_col": tgt_col,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol conformance smoke test for Knots Untangle."
    )
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Level difficulty to exercise.",
    )
    args = parser.parse_args()

    env = KnotsUntangleEnv(
        difficulty=args.difficulty,
        seed=7,
        headless=args.headless,
        animate=False,
        max_steps=16,
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.ndim == 3 and obs.shape[-1] == 3, "observation must be RGB array"
        assert "state" in info, "info must carry debug state"

        symbolic = info.get("symbolic_observation") or {}
        assert symbolic.get("ropeCount") == env.rope_count
        assert symbolic.get("gridSize") == env.grid_size
        assert len(symbolic.get("ropes", [])) == env.rope_count

        reset_payload = env.get_frontend_state()
        assert_state_response(reset_payload)

        explicit_action = first_explicit_legal_action(symbolic, env.grid_size)
        explicit_payload = env.evaluate_frontend_step(explicit_action)
        assert_state_response(explicit_payload)
        assert explicit_payload["step_count"] == 1
        assert explicit_payload["info"].get("illegal") is False

        obs, info = env.reset(seed=7)
        assert obs.ndim == 3 and obs.shape[-1] == 3

        # Probe actions until we find a legal one to exercise step().
        legal_action = None
        for action in range(env.action_space.n):
            step_payload = env.evaluate_frontend_step(action)
            assert_state_response(step_payload)
            if not step_payload["info"].get("illegal", True):
                legal_action = action
                break
        assert legal_action is not None, "expected at least one legal action"
        assert step_payload["step_count"] >= 1

        debug_state = env.get_state()
        assert "ropes" in debug_state
        assert "actionMap" in debug_state
        assert debug_state["gridSize"] == env.grid_size
        assert debug_state["ropeCount"] == env.rope_count
        assert len(debug_state["actionMap"]) == env.action_space.n

        print("protocol_ok=True")
    finally:
        env.close()


if __name__ == "__main__":
    main()
