from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANNING_DIR = ROOT / "planning_eval_tasks"
if str(PLANNING_DIR) not in sys.path:
    sys.path.append(str(PLANNING_DIR))

import hydra  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from artifact_writer import ArtifactWriter  # noqa: E402
from env_adapters import build_adapter  # noqa: E402
from evaluator_utils import ensure_common_tool_paths, ensure_runtime_settings  # noqa: E402
from rollout_runner import RolloutRunner  # noqa: E402

from configs.system_prompts import get_system_prompt  # noqa: E402
from video_interleaved_client import build_cached_video_client  # noqa: E402


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(ROOT / ".env", override=False)
    ensure_common_tool_paths()
    ensure_runtime_settings()
    from topobench_eval.datasets import configure_run
    configure_run(cfg, probing=True)
    if bool(getattr(cfg.run, "check_only", False)):
        import json
        from topobench_eval.datasets import REPO_ROOT, read_rows
        path = Path(str(cfg.env.source_question_jsonl))
        path = path if path.is_absolute() else REPO_ROOT / path
        rows = read_rows(path)
        manifest = getattr(cfg.env, "episode_ids_path", None)
        if manifest:
            ids = json.loads((REPO_ROOT / str(manifest)).read_text())
            if not set(ids) <= {row["id"] for row in rows}:
                raise ValueError("Probing manifest contains missing instance IDs")
            count = len(ids)
        else:
            count = len(rows)
        print(json.dumps({"task": str(cfg.env.name), "instances": count, **dict(cfg.data)}))
        return
    if str(cfg.model.kind) != "vlm_video":
        raise ValueError("The minimal video pipeline requires model.kind=vlm_video")
    if int(cfg.run.parallel_sessions) != 1:
        OmegaConf.set_struct(cfg, False)
        cfg.run.parallel_sessions = 1
        print("[video-e2e] forcing run.parallel_sessions=1")

    writer = ArtifactWriter(cfg=cfg)
    client = build_cached_video_client(
        cfg=cfg,
        system_prompt=get_system_prompt(str(cfg.env.system_prompt_key)),
        output_dir=writer.output_dir,
    )
    runner = RolloutRunner(
        cfg=cfg,
        adapter=build_adapter(str(cfg.env.name)),
        writer=writer,
        model_client=client,
    )
    summary = runner.run()
    print(
        f"Wrote summary to {cfg.run.output_json}\n"
        f"env={summary['env']} model={cfg.model.id} "
        f"successes={summary['successes']}/{summary['total_episodes']}"
    )


if __name__ == "__main__":
    main()
