"""Spawn-process worker entry points for RolloutRunner._run_parallel.

This module is a *leaf* module. It must NOT import `agent_runner` (or anything
that imports `agent_runner`), because spawn-method child processes re-import
worker modules at startup, and a transitive import of an `@hydra.main`-decorated
function would re-trigger Hydra in every child.

Contract:
    `worker_init(cfg_dict, system_prompt)` is called once per child process.
    `worker_run(spec_dict)` is called per episode; returns a serialisable dict
    that the main process consumes (no shared-file writes happen in the child;
    only per-episode artifact writes, which are race-safe by virtue of using
    distinct subdirectories per episode_id).
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

# Imported lazily inside worker_init / worker_run so that spawn-time module load
# stays cheap. (None of these import agent_runner.)
from artifact_writer import ArtifactWriter
from env_adapters import build_adapter
from model_adapter import build_model_client
from rollout_runner import RolloutRunner
from unified_types import EpisodeSpec


_runner: Optional[RolloutRunner] = None
_pid: Optional[int] = None
# Per-worker env cache: chromium + http server are expensive to spin up, so we
# keep one env alive across episodes and reuse it whenever the adapter's
# env_cache_key matches. Mirrors the serial path in RolloutRunner._run_serial.
_cached_env: Any = None
_cached_env_key: Any = None


def _wlog(msg: str) -> None:
    print(f"[w{_pid}] {msg}", file=sys.stderr, flush=True)


def _close_cached_env() -> None:
    global _cached_env, _cached_env_key
    env = _cached_env
    _cached_env = None
    _cached_env_key = None
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def _get_or_make_env(runner: RolloutRunner, spec: EpisodeSpec) -> Any:
    global _cached_env, _cached_env_key
    desired_key = runner._env_cache_key(spec)  # noqa: SLF001 — same package
    if _cached_env is not None and desired_key == _cached_env_key:
        return _cached_env
    _close_cached_env()
    env = runner._make_env(spec)  # noqa: SLF001
    _cached_env = env
    _cached_env_key = desired_key
    return env


atexit.register(_close_cached_env)


def worker_init(cfg_dict: Dict[str, Any], system_prompt: str) -> None:
    """Build per-process adapter / model_client / writer / runner singletons.

    The cfg arrives as a plain dict so that no Hydra interpolation re-evaluates
    in the child (e.g. `${now:...}` would otherwise produce a different
    timestamp per worker, breaking output paths).

    Dispatches on `cfg.model.kind`: `interleaved` uses the phase1 + image-gen
    + phase2 client, while normal planning models use the standard client.
    """
    global _runner, _pid
    _pid = os.getpid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Keep screenshot capture from stalling on document.fonts.ready in spawned
    # Playwright workers.
    os.environ.setdefault("PW_TEST_SCREENSHOT_NO_FONTS_READY", "1")
    cfg: DictConfig = OmegaConf.create(cfg_dict)  # type: ignore[assignment]

    try:
        adapter = build_adapter(str(cfg.env.name))
        model_kind = str(getattr(cfg.model, "kind", "")).strip() or "default"
        if model_kind == "interleaved":
            interleaved_dir = Path(__file__).resolve().parent.parent / "interleaved_eval_tasks"
            if str(interleaved_dir) not in sys.path:
                sys.path.append(str(interleaved_dir))
            from interleaved_client import build_interleaved_client  # noqa: E402
            model_client = build_interleaved_client(cfg=cfg, system_prompt=system_prompt)
        elif model_kind in {"default", "registry_remote", "local_policy"}:
            model_client = build_model_client(model_cfg=cfg.model, system_prompt=system_prompt)
        else:
            raise ValueError(f"Unsupported model kind in minimal repo: {model_kind!r}")
        writer = ArtifactWriter(cfg=cfg)
        _runner = RolloutRunner(cfg=cfg, adapter=adapter, writer=writer, model_client=model_client)
        _wlog(f"worker init done env={cfg.env.name} model={cfg.model.id} kind={model_kind}")
    except Exception:
        _wlog("worker init FAILED:")
        traceback.print_exc()
        raise


def _restore_spec(spec_dict: Dict[str, Any]) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=str(spec_dict["episode_id"]),
        level=spec_dict.get("level"),
        seed=spec_dict.get("seed"),
        repeat_index=int(spec_dict.get("repeat_index", 0)),
        category=list(spec_dict.get("category") or []),
        metadata=dict(spec_dict.get("metadata") or {}),
        reset_kwargs=dict(spec_dict.get("reset_kwargs") or {}),
    )


def worker_run(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run one episode end-to-end and return a serialised payload bundle.

    Returned dict keys:
        kind: "ok" | "skip" | "error"
        spec: original spec_dict (echoed for log fidelity in main)
        payload, answer_row: present iff kind=="ok"
        summary_log: short success/steps string for the [episode] log line
        skip_reason: present iff kind=="skip"
        error: present iff kind=="error"  (worker-level exception string)
    """
    global _runner, _pid
    if _runner is None:
        return {"kind": "error", "spec": spec_dict, "error": "worker_init not called"}

    runner = _runner
    spec = _restore_spec(spec_dict)

    try:
        env = _get_or_make_env(runner, spec)
        result = runner._run_episode(env=env, spec=spec)  # noqa: SLF001 — same package
    except Exception as exc:
        # Drop the cached env on error: it may be in a bad state (dead browser,
        # half-reset frontend), so the next episode should build a fresh one.
        _close_cached_env()
        traceback.print_exc()
        return {"kind": "error", "spec": spec_dict, "error": repr(exc)}

    if result.skip_reason is not None:
        return {"kind": "skip", "spec": spec_dict, "skip_reason": result.skip_reason}

    payload = runner._episode_summary_payload(result)  # noqa: SLF001
    answer_row = runner._build_episode_jsonl_row(result)  # noqa: SLF001

    # Per-episode artifact writes (race-safe: each episode owns its subdir).
    if runner.writer.write_step_images_enabled and result.turns and result.turns[0].saved_images:
        ep_base = Path(result.turns[0].saved_images[0].rel_path).parent
        try:
            runner.writer.write_episode_summary_json(base_dir=ep_base, payload=payload)
            runner.writer.write_episode_demo_gif(result=result)
        except Exception as exc:
            _wlog(f"per-episode artifact write failed for {spec.episode_id}: {exc!r}")

    runner.writer.discard_step_artifacts(result=result)

    summary_log = (
        f"success={result.success} steps={result.total_steps} "
        f"illegal={result.illegal_moves} invalid={result.invalid_responses} "
        f"api_errors={result.api_errors}"
    )
    return {
        "kind": "ok",
        "spec": spec_dict,
        "payload": payload,
        "answer_row": answer_row,
        "summary_log": summary_log,
    }


def spec_to_dict(spec: EpisodeSpec) -> Dict[str, Any]:
    """Pickle-friendly conversion called by the main process before submitting."""
    return asdict(spec)
