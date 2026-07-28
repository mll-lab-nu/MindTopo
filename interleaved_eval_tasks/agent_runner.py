from __future__ import annotations

import os
import sys
from pathlib import Path

# Set up the import path before importing planning modules. The interleaved
# pipeline does not duplicate planning_eval_tasks code; it appends planning's
# directory to sys.path so env_adapters / rollout_runner / artifact_writer /
# unified_types / key_pool / model_adapter resolve directly. Local interleaved
# modules (configs.*, interleaved_client, image_gen_client) sit at sys.path[0]
# (script dir) and shadow planning's wherever they share a name.
ROOT = Path(__file__).resolve().parent.parent
PLANNING_DIR = ROOT / "planning_eval_tasks"
if str(PLANNING_DIR) not in sys.path:
    sys.path.append(str(PLANNING_DIR))

import hydra  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from artifact_writer import ArtifactWriter  # noqa: E402
from evaluator_utils import ensure_common_tool_paths, ensure_runtime_settings  # noqa: E402

from configs.system_prompts import get_system_prompt  # noqa: E402  (interleaved's)
from interleaved_client import build_interleaved_client  # noqa: E402


def _compose_system_prompt(cfg: DictConfig) -> str:
    """Resolve the system prompt for this run.

    Env-specific prompts (e.g. `knots_untangle_interleaved`) carry their own
    phase-1/phase-2 framing and disambiguation rule, so concatenating the
    generic `interleaved_default` protocol on top would inject two competing
    rule sets. Prefer the env-specific prompt when it is set; fall back to
    the generic protocol when the env's `system_prompt_key` is "none" / empty.
    """
    env_specific = get_system_prompt(str(cfg.env.system_prompt_key)).strip()
    if env_specific:
        return env_specific
    return get_system_prompt(str(cfg.run.protocol_prompt_key)).strip()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(ROOT / ".env", override=False)
    ensure_common_tool_paths()
    ensure_runtime_settings()

    # `_worker.py:worker_init` dispatches on `cfg.model.kind` and builds
    # interleaved clients in spawned workers, so parallelism is now safe for
    # `kind: interleaved`. Reasoning path is single-process only — clamp it
    # because ReasoningRunner does not (yet) shard work across workers.
    eval_kind = str(getattr(cfg.env, "eval_kind", "planning"))
    if eval_kind == "reasoning":
        OmegaConf.set_struct(cfg, False)
        cfg.run.parallel_sessions = 1

    # Guard against accidentally adding an env yaml without wiring an
    # interleaved prompt. This minimal snapshot keeps only wired env configs.
    system_prompt_key = str(getattr(cfg.env, "system_prompt_key", "")).strip()
    if system_prompt_key in ("", "none"):
        raise ValueError(
            f"env={cfg.env.name!r} carries system_prompt_key={system_prompt_key!r}. "
            f"Interleaved env configs must set an env-specific system prompt."
        )

    # Per-env "always image-gen" knob: lets reasoning envs (where an imagined
    # image is always beneficial) force the harness to override both
    # `no_image_prompt_tag` and `model_declined`, and clamp to one image-gen
    # call per sample. Sheep / 2D maze opt in via `require_image_gen: true` in
    # their env yaml. Mutates `cfg.model.image_gen` in place because the model
    # client is built from cfg directly below.
    if bool(getattr(cfg.env, "require_image_gen", False)) and "image_gen" in cfg.model:
        OmegaConf.set_struct(cfg, False)
        cfg.model.image_gen.enforce_image_call = True
        cfg.model.image_gen.enforce_against_decline = True
        cfg.model.image_gen.max_image_gen_calls = 1

    system_prompt = _compose_system_prompt(cfg)
    model_client = build_interleaved_client(cfg=cfg, system_prompt=system_prompt)
    writer = ArtifactWriter(cfg=cfg)

    if eval_kind == "planning":
        # Existing gym-rollout path, unchanged.
        from env_adapters import build_adapter  # noqa: E402  (planning import)
        from rollout_runner import RolloutRunner  # noqa: E402

        adapter = build_adapter(str(cfg.env.name))
        runner = RolloutRunner(cfg=cfg, adapter=adapter, writer=writer, model_client=model_client)
        summary = runner.run()
    elif eval_kind == "reasoning":
        # New static-QA path. Lazy import keeps planning startup costs unchanged.
        from reasoning_runner import ReasoningRunner  # noqa: E402

        runner = ReasoningRunner(cfg=cfg, writer=writer, model_client=model_client)
        summary = runner.run()
    else:
        raise ValueError(
            f"Unknown env.eval_kind: {eval_kind!r}. Expected 'planning' or 'reasoning'."
        )

    progress = f"successes={summary['successes']}/{summary['total_episodes']}"

    print(
        f"Wrote summary to {cfg.run.output_json}\n"
        f"env={summary['env']} model={cfg.model.id} {progress}"
    )


if __name__ == "__main__":
    main()
