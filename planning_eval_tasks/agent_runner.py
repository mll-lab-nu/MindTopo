from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from artifact_writer import ArtifactWriter
from configs.system_prompts import get_system_prompt
from env_adapters import build_adapter
from evaluator_utils import ensure_common_tool_paths, ensure_runtime_settings
from model_adapter import build_model_client
from rollout_runner import RolloutRunner

ROOT = Path(__file__).resolve().parent.parent

# Spawn is mandatory: child workers run Playwright (via env adapters), and
# Playwright's chromium subprocess handles are not safe across fork(). Setting
# this once at import time is required *before* any Pool / Process is created
# elsewhere in the process tree.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set by an outer caller; that's fine as long as it's spawn.
    pass

# Default to 2 workers per API key (matches the lmms-eval wrapper). A model yaml
# can pin a different value via `workers_per_key:`; users can also override with
# the env var directly.
os.environ.setdefault("TOPOBENCH_WORKERS_PER_KEY", "2")

# Playwright screenshots wait on document.fonts.ready by default. Under high
# parallel planning runs that can hang even for canvas-heavy local frontends.
os.environ.setdefault("PW_TEST_SCREENSHOT_NO_FONTS_READY", "1")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(ROOT / ".env", override=False)
    ensure_common_tool_paths()
    ensure_runtime_settings()
    adapter = build_adapter(str(cfg.env.name))
    model_client = build_model_client(model_cfg=cfg.model, system_prompt=get_system_prompt(str(cfg.env.system_prompt_key)))
    runner = RolloutRunner(
        cfg=cfg,
        adapter=adapter,
        writer=ArtifactWriter(cfg=cfg),
        model_client=model_client,
    )
    summary = runner.run()
    print(
        f"Wrote summary to {cfg.run.output_json}\n"
        f"env={summary['env']} model={cfg.model.id} successes={summary['successes']}/{summary['total_episodes']}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
