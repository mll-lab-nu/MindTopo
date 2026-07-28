#!/usr/bin/env python3
"""topobench-eval launcher with multi-key API pool.

Usage:
    python lmms_eval_tasks/run.py [wrapper args] <model-id> <task1,task2,...> [extra lmms-eval args...]

Architecture:
    Parent process holds a `SharedKeyPool` (slots_per_key=workers-per-key) so
    concurrent invocations on the same provider coordinate via filesystem
    leases. For each (task, shard) pair the parent spawns one `lmms-eval eval`
    subprocess pinned to one API key. Subprocess slicing uses --offset/--limit
    against the post-`process_docs` filtered dataset; resume relies on
    lmms-eval's own --use_cache. Worker death detection mirrors the
    planning-eval contract: log [shard-error], release the key with a
    cooldown/retire, do NOT auto-respawn — re-run the launcher and lmms-eval
    cache resumes the shard.

Exit codes:
    0  all shards completed successfully
    1  at least one shard failed (others may still have succeeded; rerun to resume)
    2  bad CLI / missing config (no work attempted)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import re
import shutil
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, NoReturn, Optional, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    print(
        "[topobench-eval] error: missing dependency 'python-dotenv'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOPBENCH_EVAL_ROOT = ROOT / "topobench_eval"
PLANNING_EVAL_ROOT = ROOT / "planning_eval_tasks"
for _path in (TOPBENCH_EVAL_ROOT, PLANNING_EVAL_ROOT, HERE):
    _sp = str(_path)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

try:
    from topobench_eval.provider_registry import (  # type: ignore
        list_model_ids,
        resolve_model_registry_entry,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    print(
        "[topobench-eval] error: missing dependency 'PyYAML'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from key_pool import KeyPool, PoolExhausted, SharedKeyPool  # type: ignore  # noqa: E402

import _dataset_count  # type: ignore  # noqa: E402
import _meta_export  # type: ignore  # noqa: E402
from _retryable_response import (  # type: ignore  # noqa: E402
    RETRYABLE_FAILURE_SQL_LIKE_PATTERNS,
    is_retryable_failure_text,
)

OPENAI_COMPAT_BACKOFF_PROVIDERS = frozenset({"google", "nvidia_nim", "gemma_api"})


def _die(msg: str) -> NoReturn:
    print(f"[topobench-eval] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _is_placeholder_key(value: str) -> bool:
    return value.lower().startswith(("sk-xxx", "xxx", "placeholder"))


def _api_key_env_candidates(key_env: str) -> List[str]:
    if key_env.endswith("S"):
        return [key_env, key_env[:-1]]
    return [f"{key_env}S", key_env]


def _resolve_api_keys(key_env: str) -> List[str]:
    env_candidates = _api_key_env_candidates(key_env)
    raw = ""
    found_env = env_candidates[0]
    for candidate in env_candidates:
        raw = os.environ.get(candidate, "").strip()
        if raw:
            found_env = candidate
            break
    if not raw:
        joined = " or ".join(f"${name}" for name in env_candidates)
        _die(f"missing API key in .env — set {joined} before running")

    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if not keys or all(_is_placeholder_key(k) for k in keys):
        _die(f"missing or placeholder ${found_env} in .env — fill it in before running")
    return [k for k in keys if not _is_placeholder_key(k)]


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_preflight_timeout() -> float:
    raw = os.environ.get("TOPOBENCH_PROVIDER_PREFLIGHT_TIMEOUT", "").strip()
    if not raw:
        return 10.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 10.0


def _preflight_google_api(api_keys: List[str]) -> None:
    """Fail fast when the terminal cannot reach Google Gemini endpoints.

    Without this, lmms-eval records one failed row per sample and the wrapper
    keeps retrying shards, which burns time without making progress.
    """
    if _env_truthy("TOPOBENCH_SKIP_PROVIDER_PREFLIGHT"):
        return
    key = api_keys[0]
    timeout = _provider_preflight_timeout()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        headers={"User-Agent": "topobench-eval/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(2048)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(512).decode("utf-8", errors="replace")
        except Exception:
            pass
        body = _redact_secret(body, key).replace("\n", " ")
        _die(
            "Google Gemini preflight reached generativelanguage.googleapis.com "
            f"but returned HTTP {exc.code}. Check the Google API key/project. "
            f"Response preview: {body[:240]}"
        )
    except TimeoutError:
        _die(
            "Google Gemini preflight timed out. This terminal cannot reach "
            "generativelanguage.googleapis.com. Set HTTP_PROXY/HTTPS_PROXY in "
            ".env or your shell so Python subprocesses use your VPN/proxy, then rerun. "
            "Set TOPOBENCH_SKIP_PROVIDER_PREFLIGHT=1 only if you intentionally want to bypass this check."
        )
    except urllib.error.URLError as exc:
        reason = _redact_secret(str(exc.reason), key)
        _die(
            "Google Gemini preflight failed before evaluation started. This is a network/proxy issue, "
            f"not a benchmark issue. Reason: {reason}. Set HTTP_PROXY/HTTPS_PROXY in .env or your shell."
        )


def _has_flag(extra: List[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra)


def _safe_part(value: str) -> str:
    return value.replace("/", "_").replace(",", "_")


def _shard_dir(model_id: str, task: str, shard_index: int) -> Path:
    return HERE / "logs" / f"{_safe_part(model_id)}__{_safe_part(task)}" / f"shard_{shard_index:02d}"


def _shared_cache_dir(model_id: str, tasks_csv: str) -> Path:
    return HERE / "cache" / f"{_safe_part(model_id)}__{_safe_part(tasks_csv)}"


def _looks_rate_limited(raw: str) -> bool:
    s = raw.lower()
    return (
        "429" in s
        or "rate limit" in s
        or "ratelimit" in s
        or "too many requests" in s
        or "too frequent" in s
        or "-20048" in raw
        or "请求过于频繁" in raw
        or "请求过快" in raw
    )


def _looks_daily_quota(raw: str) -> bool:
    s = raw.lower()
    return (
        "daily" in s
        or "per day" in s
        or "requests per day" in s
        or "rpd" in s
    )


def _is_modelscope_daily_quota_exhausted(stderr_tail: str) -> bool:
    """Detect ModelScope's daily-quota exhaustion across both observed shapes:

    - 401 with ``Authentication failed ... ModelScope token`` (older / per-key)
    - 429 with ``exceeded today's quota ... try again tomorrow``

    Both signals mean the key is fine but the per-day quota is gone — retire
    until next day's reset rather than cooling for 120s in a tight loop.
    """
    s = (stderr_tail or "").lower()
    if "401" in s and "modelscope token" in s:
        return True
    if "exceeded today's quota" in s or "try again tomorrow" in s:
        return True
    return False


def _classify_error(stderr_tail: str, *, provider: Optional[str] = None) -> str:
    """Classify a failed shard's stderr tail into auth / quota / rate / other.

    Patterns cover both English/OpenAI-style errors and provider-specific
    Chinese messages. Some providers (e.g. InternVL) return rate-limits as
    HTTP 400 with a custom code (`-20048`) rather than the standard 429 — match
    those explicitly so the KeyPool gives the key a real cooldown.
    """
    raw = stderr_tail or ""
    s = raw.lower()
    if provider == "modelscope" and _is_modelscope_daily_quota_exhausted(raw):
        return "quota"
    if "401" in s or "invalid api key" in s or "unauthorized" in s or "authenticationerror" in s:
        return "auth"
    if provider == "nvidia_nim" and (
        _looks_rate_limited(raw) or "quota" in s or "too many requests" in s
    ):
        # Hosted NVIDIA NIM uses 429 for speed limits. Treat ambiguous quota
        # wording as temporary pressure so the key sleeps instead of retiring.
        return "rate"
    if provider == "google":
        if _looks_daily_quota(raw):
            return "quota"
        if _looks_rate_limited(raw) or "quota" in s:
            return "rate"
    if "insufficient_quota" in s or "billing" in s or "quota" in s:
        return "quota"
    if _looks_rate_limited(raw):
        return "rate"
    return "other"


def _build_model_args(
    *,
    upstream_model_name: str,
    num_cpus: int,
    adaptive_concurrency: bool,
    adaptive_max_concurrency: Optional[int],
) -> str:
    parts = [
        f"model_version={upstream_model_name}",
        f"num_concurrent={num_cpus}",
    ]
    # Always emit adaptive_concurrency explicitly. If we only emit it on True,
    # lmms-eval falls back to its own default (which is true) when the user
    # passes --no-adaptive-concurrency, silently re-enabling adaptive.
    parts.append(f"adaptive_concurrency={'true' if adaptive_concurrency else 'false'}")
    if adaptive_concurrency and adaptive_max_concurrency is not None:
        parts.append(f"adaptive_max_concurrency={adaptive_max_concurrency}")
    parts.append("max_retries=3")
    return ",".join(parts)


def _build_subprocess_cmd(
    *,
    task: str,
    offset: int,
    limit: int,
    output_dir: Path,
    cache_dir: Path,
    model_args: str,
    extra: List[str],
) -> List[str]:
    cmd: List[str] = [
        _lmms_eval_executable(), "eval",
        "--model", "openai",
        "--model_args", model_args,
        "--tasks", task,
        "--include_path", str(HERE),
        "--batch_size", "1",
        "--log_samples",
        "--offset", str(offset),
        "--limit", str(limit),
        "--output_path", str(output_dir),
    ]
    if not _has_flag(extra, "--force_simple"):
        cmd.append("--force_simple")
    if not _has_flag(extra, "--use_cache"):
        cmd.extend(["--use_cache", str(cache_dir)])
    if not _has_flag(extra, "--cache_requests"):
        cmd.extend(["--cache_requests", "true"])
    cmd.extend(extra)
    return cmd


def _lmms_eval_executable() -> str:
    sibling = Path(sys.executable).resolve().parent / "lmms-eval"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("lmms-eval")
    return found or "lmms-eval"


def _split_shards(size: int, num_workers: int) -> List[Tuple[int, int]]:
    """Return list of (offset, limit) covering [0, size). Empty workers are dropped."""
    if size <= 0 or num_workers <= 0:
        return []
    chunk = math.ceil(size / num_workers)
    out: List[Tuple[int, int]] = []
    for i in range(num_workers):
        offset = i * chunk
        if offset >= size:
            break
        limit = min(chunk, size - offset)
        out.append((offset, limit))
    return out


def _split_by_shard_size(size: int, shard_size: int) -> List[Tuple[int, int]]:
    """Return fixed-size ranges covering [0, size)."""
    if size <= 0 or shard_size <= 0:
        return []
    return [
        (offset, min(shard_size, size - offset))
        for offset in range(0, size, shard_size)
    ]


_LIVE_PROCS_LOCK = threading.Lock()
_LIVE_PROCS: List[subprocess.Popen] = []


def _track_proc(proc: subprocess.Popen) -> None:
    with _LIVE_PROCS_LOCK:
        _LIVE_PROCS.append(proc)


def _untrack_proc(proc: subprocess.Popen) -> None:
    with _LIVE_PROCS_LOCK:
        try:
            _LIVE_PROCS.remove(proc)
        except ValueError:
            pass


def _kill_live_procs() -> None:
    with _LIVE_PROCS_LOCK:
        procs = list(_LIVE_PROCS)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


_RATE_LIMIT_LINE_RE = re.compile(
    r"-20048|请求过于频繁|请求过快|"
    r"\b429\b|\brate[\s_-]*limit|too\s+many\s+requests|too\s+frequent",
    re.IGNORECASE,
)
_MODELSCOPE_DAILY_QUOTA_LINE_RE = re.compile(
    r"(\b401\b.*ModelScope token)|(exceeded today's quota)|(try again tomorrow)",
    re.IGNORECASE,
)
_MODELSCOPE_DAILY_QUOTA_THRESHOLD = 3

# Watchdog defaults: if lmms-eval logs more than N rate-limit lines within
# WINDOW seconds inside one shard, the shard is silently failing every sample
# and we need to intervene — kill the subprocess, cooldown the key, mark the
# shard failed.
_RATE_LIMIT_THRESHOLD = 8
_RATE_LIMIT_WINDOW_SEC = 30.0


def _latest_samples_file(output_dir: Path) -> Optional[Path]:
    samples_files = list(output_dir.rglob("*_samples_*.jsonl"))
    if not samples_files:
        return None
    return max(samples_files, key=lambda path: path.stat().st_mtime)


def _count_api_failed_sample_rows(output_dir: Path) -> Tuple[int, Optional[Path]]:
    samples_file = _latest_samples_file(output_dir)
    if samples_file is None:
        return 0, None
    failed = 0
    try:
        with samples_file.open("r", encoding="utf-8") as f:
            for line in f:
                if is_retryable_failure_text(line):
                    failed += 1
    except OSError:
        return 0, samples_file
    return failed, samples_file


def _rewrite_jsonl_without_failed_rows(path: Path) -> int:
    try:
        original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return 0
    kept = [line for line in original if not is_retryable_failure_text(line)]
    removed = len(original) - len(kept)
    if removed <= 0:
        return 0
    if kept:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        tmp.replace(path)
    else:
        path.unlink(missing_ok=True)
    return removed


def _delete_failed_cache_db_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path, timeout=30.0) as db:
            where = " OR ".join(
                "LOWER(response) LIKE ?"
                for _ in RETRYABLE_FAILURE_SQL_LIKE_PATTERNS
            )
            cur = db.execute(
                f"DELETE FROM responses WHERE {where}",
                tuple(RETRYABLE_FAILURE_SQL_LIKE_PATTERNS),
            )
            db.commit()
            return int(cur.rowcount if cur.rowcount is not None else 0)
    except sqlite3.Error:
        return 0


def _scrub_api_failed_artifacts(*, cache_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> int:
    """Remove lmms-eval API-failure sentinel rows so resume treats them as misses."""
    removed = 0
    roots = [root for root in (cache_dir, output_dir) if root is not None and root.exists()]
    for root in roots:
        for jsonl_path in root.rglob("*.jsonl"):
            removed += _rewrite_jsonl_without_failed_rows(jsonl_path)
        for db_path in root.rglob("*.db"):
            removed += _delete_failed_cache_db_rows(db_path)
    if removed:
        print(
            f"[api-failure-scrub] removed {removed} failed cache/log rows",
            file=sys.stderr,
            flush=True,
        )
    return removed


def _run_shard(
    *,
    pool: KeyPool,
    provider: str,
    base_url: Optional[str],
    request_min_interval_seconds: float,
    cmd_template: dict,
    shard_label: str,
    shard_attempt: int,
    api_failed_row_retry_limit: int,
) -> Tuple[str, int, str, bool]:
    """Acquire a key, run lmms-eval for one shard, classify failures, release the key.

    The shard is streamed (line-by-line) so a stderr watchdog can detect that
    the subprocess is in a fail-storm — lmms-eval logs every per-sample API
    failure but does NOT exit non-zero on rate-limit errors, so without
    real-time monitoring the wrapper would mark a fully-failed shard as
    successful.
    """
    _scrub_api_failed_artifacts(output_dir=cmd_template["output_dir"])
    key = pool.acquire(timeout=600.0)
    key_label = pool.key_label(key)
    proc: Optional[subprocess.Popen] = None
    try:
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = key
        env["TOPOBENCH_LMMS_INCLUDE_DEFAULT_TASKS"] = "0"
        env["TOPOBENCH_OPENAI_PROVIDER"] = provider
        if base_url:
            env["OPENAI_API_BASE"] = base_url
        if request_min_interval_seconds > 0:
            env.setdefault(
                "TOPOBENCH_OPENAI_REQUEST_MIN_INTERVAL_SECONDS",
                f"{request_min_interval_seconds:g}",
            )
        if provider in OPENAI_COMPAT_BACKOFF_PROVIDERS:
            env.setdefault("TOPOBENCH_OPENAI_RATE_LIMIT_RETRIES", "3")
            env.setdefault("TOPOBENCH_OPENAI_RATE_LIMIT_INITIAL_SLEEP", "16")
            env.setdefault("TOPOBENCH_OPENAI_RATE_LIMIT_MAX_SLEEP", "300")
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(HERE)
            if not existing_pythonpath
            else os.pathsep.join([str(HERE), existing_pythonpath])
        )
        cmd = _build_subprocess_cmd(**cmd_template)
        masked = " ".join(shlex.quote(c) for c in cmd).replace(key, "***")
        print(f"[shard-start] {shard_label} {key_label} {masked}", file=sys.stderr, flush=True)
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(ROOT),
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered so the watchdog sees each log line promptly
        )
        _track_proc(proc)

        # Watchdog: stream stderr line-by-line, count rate-limit hits in a
        # sliding window. Pass through to parent stderr unchanged so user still
        # sees lmms-eval's own logs.
        stderr_tail: Deque[str] = deque(maxlen=200)
        rate_limit_ts: Deque[float] = deque()
        modelscope_quota_ts: Deque[float] = deque()
        killed_by_watchdog = False
        watchdog_reason = ""
        watchdog_action = "rate"  # "rate" -> 120s cooldown; "quota" -> daily retire

        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line)
            print(line, end="", file=sys.stderr, flush=True)
            now = time.monotonic()
            cutoff = now - _RATE_LIMIT_WINDOW_SEC
            if provider == "modelscope" and _MODELSCOPE_DAILY_QUOTA_LINE_RE.search(line):
                modelscope_quota_ts.append(now)
                while modelscope_quota_ts and modelscope_quota_ts[0] < cutoff:
                    modelscope_quota_ts.popleft()
                if len(modelscope_quota_ts) >= _MODELSCOPE_DAILY_QUOTA_THRESHOLD:
                    killed_by_watchdog = True
                    watchdog_action = "quota"
                    watchdog_reason = (
                        f"{len(modelscope_quota_ts)} ModelScope daily-quota lines in last "
                        f"{_RATE_LIMIT_WINDOW_SEC:.0f}s (key exhausted until tomorrow)"
                    )
                    print(
                        f"[shard-watchdog] {shard_label} {key_label} {watchdog_reason}; killing subprocess",
                        file=sys.stderr,
                        flush=True,
                    )
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
            if not _RATE_LIMIT_LINE_RE.search(line):
                continue
            rate_limit_ts.append(now)
            while rate_limit_ts and rate_limit_ts[0] < cutoff:
                rate_limit_ts.popleft()
            if len(rate_limit_ts) >= _RATE_LIMIT_THRESHOLD:
                killed_by_watchdog = True
                watchdog_action = "rate"
                watchdog_reason = (
                    f"{len(rate_limit_ts)} rate-limit lines in last "
                    f"{_RATE_LIMIT_WINDOW_SEC:.0f}s"
                )
                print(
                    f"[shard-watchdog] {shard_label} {key_label} {watchdog_reason}; killing subprocess",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    proc.terminate()
                except Exception:
                    pass
                break

        # Reap (drain remaining stderr only if we killed).
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

        return_code = proc.returncode if proc.returncode is not None else -1
        tail_text = "".join(stderr_tail)[-2000:]

        if killed_by_watchdog:
            if watchdog_action == "quota":
                pool.mark_quota_exhausted(key, "quota_exhausted")
                action_msg = f"{key_label} retired (modelscope daily quota)"
            else:
                pool.mark_rate_limited(key, cooldown_seconds=120.0)
                action_msg = f"{key_label} cooled down 120s"
            print(
                f"[shard-error] {shard_label} killed-by-watchdog ({watchdog_reason}); {action_msg}",
                file=sys.stderr,
                flush=True,
            )
            _scrub_api_failed_artifacts(output_dir=cmd_template["output_dir"])
            return shard_label, -1, tail_text, True

        if return_code == 0:
            failed_rows, samples_file = _count_api_failed_sample_rows(cmd_template["output_dir"])
            if samples_file is None:
                print(
                    f"[shard-error] {shard_label} {key_label} no sample file produced; shard will be retried",
                    file=sys.stderr,
                    flush=True,
                )
                return shard_label, -1, tail_text, True
            if failed_rows:
                retries_used = max(shard_attempt - 1, 0)
                if (
                    api_failed_row_retry_limit >= 0
                    and retries_used >= api_failed_row_retry_limit
                ):
                    print(
                        f"[shard-api-final] {shard_label} {key_label} "
                        f"api-failed-rows={failed_rows} latest_samples={samples_file} "
                        f"retries_used={retries_used} retry_limit={api_failed_row_retry_limit}; "
                        "keeping rows as api_error",
                        file=sys.stderr,
                        flush=True,
                    )
                    return shard_label, 0, "", False
                if provider == "modelscope" and _is_modelscope_daily_quota_exhausted(tail_text):
                    pool.mark_quota_exhausted(key, "quota_exhausted")
                    action = "retired (modelscope daily quota)"
                else:
                    pool.mark_rate_limited(key, cooldown_seconds=60.0)
                    action = "cooled down 60s"
                _scrub_api_failed_artifacts(
                    cache_dir=cmd_template["cache_dir"],
                    output_dir=cmd_template["output_dir"],
                )
                print(
                    f"[shard-error] {shard_label} {key_label} api-failed-rows={failed_rows} "
                    f"latest_samples={samples_file} action={action}; shard will be retried",
                    file=sys.stderr,
                    flush=True,
                )
                return shard_label, -1, tail_text, True
            return shard_label, 0, "", False

        cls = _classify_error(tail_text, provider=provider)
        retryable = True
        if cls == "auth":
            pool.mark_quota_exhausted(key, "auth_error")
        elif cls == "quota":
            pool.mark_quota_exhausted(key, "quota_exhausted")
        elif cls == "rate":
            pool.mark_rate_limited(key, cooldown_seconds=60.0)
        else:
            # Unknown failure — short cooldown so peers still get a turn at this key.
            pool.mark_rate_limited(key, cooldown_seconds=30.0)
            retryable = False
        print(
            f"[shard-error] {shard_label} {key_label} exit={return_code} class={cls}",
            file=sys.stderr,
            flush=True,
        )
        if retryable:
            _scrub_api_failed_artifacts(
                cache_dir=cmd_template["cache_dir"],
                output_dir=cmd_template["output_dir"],
            )
        return shard_label, return_code, tail_text, retryable
    finally:
        if proc is not None:
            _untrack_proc(proc)
        pool.release(key)


def _parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--num-cpus", type=int, default=16)
    parser.add_argument("--workers-per-key", type=int, default=2)
    parser.add_argument("--max-shard-workers", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--key-rpm", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, dest="sample_limit")
    parser.add_argument("--max-api-failed-row-retries", type=int, default=3)
    parser.add_argument(
        "--adaptive-concurrency",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="adaptive_concurrency",
    )
    parser.add_argument("--adaptive-max-concurrency", type=int, default=32)
    parser.add_argument(
        "--export-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="export_meta",
    )
    parser.add_argument("-h", "--help", action="store_true", dest="show_help")
    return parser.parse_known_args(argv)


_USAGE = (
    "usage: topobench-eval [--dry-run] [--num-cpus N] [--workers-per-key N] "
    "[--max-shard-workers N] [--shard-size N] [--key-rpm R] [--limit N] "
    "[--max-api-failed-row-retries N] "
    "[--[no-]adaptive-concurrency] [--adaptive-max-concurrency N] [--[no-]export-meta] "
    "<model-id> <task1[,task2,...]> [extra lmms-eval args...]"
)


def main() -> None:
    args, remaining = _parse_args(sys.argv[1:])
    if args.show_help:
        print(_USAGE, file=sys.stderr)
        return
    if len(remaining) < 2:
        _die(_USAGE)

    model_id, tasks_csv, *extra = remaining

    load_dotenv(ROOT / ".env")

    try:
        resolved = resolve_model_registry_entry(model_id)
    except KeyError:
        known = "\n  ".join(list_model_ids())
        _die(f"unknown model-id {model_id!r}. Known models:\n  {known}")

    if resolved.adapter != "openai":
        _die(
            f"adapter {resolved.adapter!r} is not supported by the API pool wrapper "
            "(only adapter=openai). Use planning-eval for gemini_api flows."
        )

    if args.num_cpus < 1:
        _die("--num-cpus must be >= 1")
    if args.workers_per_key < 1:
        _die("--workers-per-key must be >= 1")
    if args.max_shard_workers < 1:
        _die("--max-shard-workers must be >= 1")
    if args.shard_size < 0:
        _die("--shard-size must be >= 0")
    if args.sample_limit is not None and args.sample_limit < 1:
        _die("--limit must be >= 1")
    if args.max_api_failed_row_retries < -1:
        _die("--max-api-failed-row-retries must be >= -1")

    api_keys = _resolve_api_keys(resolved.api_key_env)
    if resolved.provider == "google":
        _preflight_google_api(api_keys)
    min_interval = 60.0 / args.key_rpm if args.key_rpm > 0 else 0.0
    request_min_interval = min_interval
    if resolved.provider in OPENAI_COMPAT_BACKOFF_PROVIDERS and request_min_interval <= 0:
        request_min_interval = 16.0
    # Default to SharedKeyPool so concurrent `topobench-eval` invocations on the
    # same provider coordinate through on-disk lease state — one wrapper holding
    # `slots_per_key` slots prevents another wrapper from oversubscribing the
    # same key. Set TOPOBENCH_DISABLE_SHARED_KEY_POOL=1 to fall back to the
    # in-process pool (e.g. for tests, or when state files are stale).
    use_shared = os.environ.get("TOPOBENCH_DISABLE_SHARED_KEY_POOL", "").strip() not in {"1", "true", "TRUE", "yes"}
    pool: KeyPool
    if use_shared:
        pool = SharedKeyPool(
            provider=resolved.provider,
            keys=api_keys,
            min_interval_seconds=min_interval,
            slots_per_key=args.workers_per_key,
        )
    else:
        pool = KeyPool(
            provider=resolved.provider,
            keys=api_keys,
            min_interval_seconds=min_interval,
            slots_per_key=args.workers_per_key,
        )
    total_key_slots = len(api_keys) * args.workers_per_key
    shard_workers = min(total_key_slots, args.max_shard_workers)

    tasks = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    if not tasks:
        _die("at least one task name required")

    print(
        f"[topobench-eval] provider={resolved.provider} keys={len(api_keys)} "
        f"workers_per_key={args.workers_per_key} key_slots={total_key_slots} "
        f"shard_workers={shard_workers} max_shard_workers={args.max_shard_workers} "
        f"shard_size={args.shard_size or 'auto'} "
        f"tasks={tasks} num_cpus={args.num_cpus} "
        f"pool={'shared' if use_shared else 'in-process'} "
        f"request_min_interval={request_min_interval:g}s "
        f"api_failed_row_retries="
        f"{'infinite' if args.max_api_failed_row_retries < 0 else args.max_api_failed_row_retries}",
        file=sys.stderr,
    )

    cache_dir = _shared_cache_dir(model_id, tasks_csv)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_args = _build_model_args(
        upstream_model_name=resolved.upstream_model_name,
        num_cpus=args.num_cpus,
        adaptive_concurrency=bool(args.adaptive_concurrency),
        adaptive_max_concurrency=args.adaptive_max_concurrency,
    )

    # Plan shards across all tasks.
    shards: List[dict] = []
    for task in tasks:
        try:
            size = _dataset_count.get_or_compute(task)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _die(f"failed to count dataset for task={task!r}: {exc!r}")
        if args.sample_limit is not None:
            size = min(size, args.sample_limit)
        ranges = (
            _split_by_shard_size(size, args.shard_size)
            if args.shard_size > 0
            else _split_shards(size, shard_workers)
        )
        if not ranges:
            print(f"[topobench-eval] task={task} has 0 samples; skipping", file=sys.stderr)
            continue
        for i, (offset, limit) in enumerate(ranges):
            output_dir = _shard_dir(model_id, task, i)
            output_dir.mkdir(parents=True, exist_ok=True)
            shards.append({
                "task": task,
                "offset": offset,
                "limit": limit,
                "output_dir": output_dir,
                "cache_dir": cache_dir,
                "model_args": model_args,
                "extra": extra,
                "label": f"{task}.s{i:02d}[{offset}:{offset+limit}]",
            })
        print(
            f"[topobench-eval] task={task} size={size} shards={len(ranges)} "
            f"chunk={ranges[0][1] if ranges else 0}",
            file=sys.stderr,
        )

    if not shards:
        _die("no shards to run")

    if args.dry_run:
        for shard in shards:
            cmd = _build_subprocess_cmd(
                task=shard["task"],
                offset=shard["offset"],
                limit=shard["limit"],
                output_dir=shard["output_dir"],
                cache_dir=shard["cache_dir"],
                model_args=shard["model_args"],
                extra=shard["extra"],
            )
            print(f"[dry-run] {shard['label']}: {' '.join(shlex.quote(c) for c in cmd)}")
        return

    # Remove stale failure sentinels from older runs before any subprocess has a
    # chance to read them as cache hits.
    _scrub_api_failed_artifacts(cache_dir=cache_dir)

    # Run shards in parallel; KeyPool gates key use, while shard_workers caps
    # local subprocess/file-descriptor pressure for large key pools.
    # API-class row failures are retried, then left as final api_error rows when
    # the configured retry limit is reached.
    failed = 0
    succeeded = 0
    retries = 0
    futures: Dict[concurrent.futures.Future, dict] = {}
    pending: Deque[dict] = deque(shards)

    def _submit_shard(executor: concurrent.futures.ThreadPoolExecutor, shard: dict) -> None:
        shard["attempt"] = int(shard.get("attempt", 0)) + 1
        cmd_template = {
            "task": shard["task"],
            "offset": shard["offset"],
            "limit": shard["limit"],
            "output_dir": shard["output_dir"],
            "cache_dir": shard["cache_dir"],
            "model_args": shard["model_args"],
            "extra": shard["extra"],
        }
        label = f"{shard['label']} attempt={shard['attempt']}"
        fut = executor.submit(
            _run_shard,
            pool=pool,
            provider=resolved.provider,
            base_url=resolved.base_url,
            request_min_interval_seconds=request_min_interval,
            cmd_template=cmd_template,
            shard_label=label,
            shard_attempt=shard["attempt"],
            api_failed_row_retry_limit=args.max_api_failed_row_retries,
        )
        futures[fut] = shard

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=shard_workers) as executor:
            while pending or futures:
                while pending and len(futures) < shard_workers:
                    _submit_shard(executor, pending.popleft())

                if not futures:
                    continue

                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    shard = futures.pop(fut)
                    try:
                        label, exit_code, _, retryable = fut.result()
                    except PoolExhausted as exc:
                        retries += 1
                        print(
                            f"[shard-retry] {shard['label']} pool-exhausted: {exc}; "
                            "waiting 60s before retry",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(60.0)
                        pending.append(shard)
                        continue
                    except Exception as exc:
                        print(f"[shard-error] {shard['label']} worker-exception: {exc!r}", file=sys.stderr)
                        failed += 1
                        continue
                    if exit_code == 0:
                        succeeded += 1
                        print(f"[shard-done] {label}", file=sys.stderr, flush=True)
                    elif retryable:
                        retries += 1
                        print(
                            f"[shard-retry] {label} will be retried "
                            f"(completed={succeeded}/{len(shards)}, retries={retries})",
                            file=sys.stderr,
                            flush=True,
                        )
                        pending.append(shard)
                    else:
                        failed += 1
                        print(f"[shard-failed] {label} non-api failure; not retrying", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("[topobench-eval] interrupted; terminating shard subprocesses...", file=sys.stderr)
        _kill_live_procs()
        for fut in futures:
            fut.cancel()
        raise

    print(
        f"[topobench-eval] shards: succeeded={succeeded} failed={failed} "
        f"retries={retries} total={len(shards)}",
        file=sys.stderr,
    )

    if args.export_meta:
        try:
            _meta_export.export_all(model_id, tasks)
        except Exception as exc:
            print(f"[meta-export] error: {exc!r}", file=sys.stderr)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
