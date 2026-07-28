from __future__ import annotations

import argparse
import random
from typing import List, Sequence

from env import KnotsUntangleEnv


def pretty_action(env: KnotsUntangleEnv, action_idx: int) -> str:
    rope_idx, hole_idx = env.decode_action(action_idx)
    row = hole_idx // env.grid_size
    col = hole_idx % env.grid_size
    return f"{action_idx} (rope={rope_idx} -> hole=({row},{col}))"


def legal_action_indices(env: KnotsUntangleEnv, state: dict) -> List[int]:
    holes = {h["id"]: h for h in state.get("holes", [])}
    rope_endpoints = {}
    for rope in state.get("ropes", []):
        rope_endpoints[rope["id"]] = {rope["startHole"]["id"], rope["endHole"]["id"]}
    legal: List[int] = []
    for idx, (rope_idx, hole_idx) in enumerate(env.action_map):
        hole = holes.get(hole_idx)
        if hole is None:
            continue
        if hole["occupied"]:
            continue
        if hole_idx in rope_endpoints.get(rope_idx, set()):
            continue
        legal.append(idx)
    return legal


def run_action_sequence(
    env: KnotsUntangleEnv, actions: Sequence[int], label: str
) -> None:
    obs, info = env.reset()
    assert obs.ndim == 3 and obs.shape[-1] == 3
    crossings = info["state"].get("crossings", "?")
    print(
        f"\n[{label}] reset rgb_shape={obs.shape} ropes={env.rope_count} "
        f"grid={env.grid_size}x{env.grid_size} crossings={crossings}"
    )
    for step_idx, action in enumerate(actions):
        obs, reward, terminated, truncated, step_info = env.step(action)
        print(
            f"[{label}] step={step_idx:02d} action={pretty_action(env, action)} "
            f"reward={reward:+.3f} illegal={step_info.get('illegal', False)} "
            f"crossings={step_info.get('crossings')} done={terminated} trunc={truncated}"
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        if terminated or truncated:
            print(f"[{label}] episode_end terminated={terminated} truncated={truncated}")
            break


def run_mixed_policy(env: KnotsUntangleEnv, steps: int, seed: int) -> None:
    rng = random.Random(seed)
    obs, info = env.reset()
    assert obs.ndim == 3 and obs.shape[-1] == 3
    illegal_count = 0
    print(
        f"\n[mixed-policy] reset ropes={env.rope_count} "
        f"grid={env.grid_size}x{env.grid_size} "
        f"crossings={info['state'].get('crossings')}"
    )

    last_step_idx = -1
    for step_idx in range(steps):
        last_step_idx = step_idx
        state = env.get_state()
        legal = legal_action_indices(env, state)
        use_legal = bool(legal) and rng.random() < 0.7
        if use_legal:
            action = rng.choice(legal)
            policy = "legal-biased"
        else:
            action = env.action_space.sample()
            policy = "random"

        obs, reward, terminated, truncated, step_info = env.step(action)
        illegal = step_info.get("illegal", False)
        illegal_count += int(illegal)
        print(
            f"[mixed-policy] step={step_idx:02d} policy={policy} "
            f"action={pretty_action(env, action)} reward={reward:+.3f} "
            f"illegal={illegal} crossings={step_info.get('crossings')} "
            f"done={terminated} trunc={truncated}"
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        if terminated or truncated:
            break

    denom = max(last_step_idx + 1, 1)
    print(f"[mixed-policy] illegal_ratio={illegal_count}/{denom}")


def run_difficulty_sweep(headless: bool, animate: bool, seed: int) -> None:
    for difficulty in ("easy", "medium", "hard"):
        env = KnotsUntangleEnv(
            difficulty=difficulty,
            seed=seed,
            animate=animate,
            headless=headless,
            max_steps=20,
        )
        try:
            print(
                f"\n=== difficulty={difficulty} grid={env.grid_size}x{env.grid_size} "
                f"ropes={env.rope_count} actions={env.action_space.n} ==="
            )
            run_mixed_policy(env, steps=10, seed=seed)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Knots Untangle environment smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for mixed policy.")
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Primary difficulty for the illegal-stress + mixed-policy runs.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the easy/medium/hard sweep at the end of the run.",
    )
    args = parser.parse_args()

    env = KnotsUntangleEnv(
        difficulty=args.difficulty,
        seed=args.seed,
        animate=args.animate,
        headless=args.headless,
        max_steps=40,
    )
    try:
        # Stress the illegal path: action 0 is very likely occupied or same-rope.
        illegal_stress = [0, 1, 2, 3, 4]
        run_action_sequence(env, illegal_stress, "illegal-stress")
        run_mixed_policy(env, steps=20, seed=args.seed)
    finally:
        env.close()

    if not args.skip_sweep:
        run_difficulty_sweep(headless=args.headless, animate=args.animate, seed=args.seed)


if __name__ == "__main__":
    main()
