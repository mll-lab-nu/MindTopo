from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional


_DAILY_KEYWORDS = frozenset(["daily", "per day", "day limit", "rpd", "requests per day"])
_TEMPORARY_RETIRE_REASONS = frozenset({"daily_exhausted", "quota_exhausted"})

# Auth failures (HTTP 401) are frequently transient for self-hosted endpoints:
# a server redeploy, key rotation, or a not-yet-ready config can 401 a key that
# becomes valid again minutes later. Retiring such a key permanently means a
# recovered endpoint is never retried — the pool just reports PoolExhausted
# forever, even once the API is healthy again. Instead, retire auth failures for
# a bounded window so the key is automatically re-probed and brought back. Set
# TOPOBENCH_AUTH_RETIRE_SECONDS to override the window (0 restores the old
# permanent-retirement behavior).
_AUTH_RETIRE_REASONS = frozenset({"auth_error"})
_DEFAULT_AUTH_RETIRE_SECONDS = 600.0


@dataclass
class KeyState:
    key: str
    cooldown_until: float = 0.0  # monotonic; key unavailable until this
    last_used_at: float = 0.0    # monotonic; proactive throttle stamp
    retired: bool = False
    retired_until: Optional[float] = None  # epoch seconds; None means session-permanent
    failure_count: int = 0       # 5xx count, informational only — never triggers retirement
    slots_in_use: int = 0        # current concurrent holders (≤ pool.slots_per_key)


class PoolExhausted(Exception):
    """Raised when acquire() cannot return a key because all keys are retired,
    or when the timeout elapses before any key becomes available."""


class KeyPool:
    """Thread-safe round-robin API key pool with proactive per-key throttling.

    One instance per (provider, process). Retired keys are never persisted —
    they are re-tried fresh on the next session.

    Args:
        provider: Human-readable provider name for log messages.
        keys: List of API key strings. Must be non-empty.
        min_interval_seconds: Minimum time between successive uses of the *same*
            key (derived from per-key RPM: 60.0 / rpm). 0.0 means unlimited.
        slots_per_key: How many concurrent holders a single key tolerates. Default
            1 preserves the original 1:1 thread-to-key contract; pass 2 to let two
            workers share each key (used by lmms-eval API pool and the multi-process
            planning runner).
    """

    def __init__(
        self,
        provider: str,
        keys: List[str],
        min_interval_seconds: float = 0.0,
        *,
        slots_per_key: int = 1,
    ) -> None:
        if not keys:
            raise ValueError(f"KeyPool for provider {provider!r} requires at least one key.")
        if slots_per_key < 1:
            raise ValueError(f"slots_per_key must be >= 1, got {slots_per_key}")
        self._provider = provider
        self._min_interval = min_interval_seconds
        self._slots_per_key = int(slots_per_key)
        self._states: List[KeyState] = [KeyState(key=k) for k in keys]
        self._cursor: int = 0
        self._cv = threading.Condition(threading.Lock())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def acquire(self, timeout: float = 300.0) -> str:
        """Return the next available key (round-robin with proactive throttle).

        Blocks until a key is ready or `timeout` seconds elapse.
        Updates `last_used_at` atomically before returning so concurrent
        callers stagger by `min_interval_seconds`.

        Raises:
            PoolExhausted: All keys retired, or timeout elapsed.
        """
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                now_wall = time.time()
                active = [s for s in self._states if not self._is_retired(s, now_wall)]
                if not active:
                    raise PoolExhausted(
                        f"All keys for provider {self._provider!r} have been retired."
                    )

                now = time.monotonic()
                n = len(self._states)
                earliest_ready = float("inf")

                for offset in range(n):
                    idx = (self._cursor + offset) % n
                    s = self._states[idx]
                    if self._is_retired(s, now_wall) or s.slots_in_use >= self._slots_per_key:
                        continue
                    avail = max(s.cooldown_until, s.last_used_at + self._min_interval)
                    if avail <= now:
                        s.last_used_at = now
                        s.slots_in_use += 1
                        self._cursor = (idx + 1) % n
                        return s.key
                    earliest_ready = min(earliest_ready, avail)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PoolExhausted(
                        f"Timed out waiting for an available key from provider {self._provider!r}."
                    )
                # Sleep until the earliest key becomes ready (or remaining budget).
                self._cv.wait(timeout=min(earliest_ready - time.monotonic(), remaining))

    def release(self, key: str) -> None:
        """Release a previously acquired key, freeing one slot.

        The slot counter is what gates `slots_per_key` admission in `acquire()`.
        Callers MUST pair every successful `acquire()` with exactly one
        `release()` (use try/finally). SharedKeyPool overrides this to also drop
        the cross-process lease handle.
        """
        with self._cv:
            s = self._find(key)
            if s is not None and s.slots_in_use > 0:
                s.slots_in_use -= 1
            self._cv.notify_all()

    def mark_quota_exhausted(self, key: str, reason: str) -> None:
        """Retire a key for this session, with daily/quota reasons expiring next reset."""
        with self._cv:
            s = self._find(key)
            if s is not None and not s.retired:
                s.retired = True
                s.retired_until = self._retired_until_for_reason(reason)
                now_wall = time.time()
                remaining = sum(1 for state in self._states if not self._is_retired(state, now_wall))
                until_msg = (
                    f" until {datetime.fromtimestamp(s.retired_until).isoformat(timespec='seconds')}"
                    if s.retired_until is not None
                    else ""
                )
                print(
                    f"[key-pool] {self._provider}: {self._key_label_unlocked(key)} retired "
                    f"({reason}{until_msg}); {remaining}/{self.size} keys remaining."
                )
            self._cv.notify_all()

    def mark_rate_limited(self, key: str, cooldown_seconds: float) -> None:
        """Apply a timed cooldown (minute-limit 429). Key resumes after cooldown."""
        with self._cv:
            s = self._find(key)
            if s is not None:
                s.cooldown_until = time.monotonic() + cooldown_seconds
            self._cv.notify_all()

    def mark_transient_error(self, key: str) -> None:
        """Increment 5xx failure counter. Never retires the key."""
        with self._cv:
            s = self._find(key)
            if s is not None:
                s.failure_count += 1

    def record_success(self, key: str) -> None:
        """Reset failure_count after a successful call."""
        with self._cv:
            s = self._find(key)
            if s is not None:
                s.failure_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total key count (including retired)."""
        return len(self._states)

    @property
    def active_count(self) -> int:
        """Number of non-retired keys."""
        with self._cv:
            now_wall = time.time()
            return sum(1 for s in self._states if not self._is_retired(s, now_wall))

    @property
    def concurrency_hint(self) -> int:
        """Suggested ThreadPoolExecutor max_workers = key count × slots_per_key."""
        return len(self._states) * self._slots_per_key

    @property
    def slots_per_key(self) -> int:
        return self._slots_per_key

    def key_index(self, key: str) -> Optional[int]:
        """Return the 1-based position of `key` in the configured key list."""
        with self._cv:
            return self._key_index_unlocked(key)

    def key_label(self, key: str) -> str:
        """Return a redacted, user-facing key label for logs and errors."""
        with self._cv:
            return self._key_label_unlocked(key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _key_index_unlocked(self, key: str) -> Optional[int]:
        for idx, s in enumerate(self._states, start=1):
            if s.key == key:
                return idx
        return None

    def _key_label_unlocked(self, key: str) -> str:
        suffix = key[-6:] if key else "<empty>"
        idx = self._key_index_unlocked(key)
        if idx is None:
            return f"key ?/{self.size} (...{suffix})"
        return f"key #{idx}/{self.size} (...{suffix})"

    def _find(self, key: str) -> Optional[KeyState]:
        for s in self._states:
            if s.key == key:
                return s
        return None

    def _is_retired(self, state: KeyState, now_wall: Optional[float] = None) -> bool:
        if not state.retired:
            return False
        if state.retired_until is None:
            return True
        now = time.time() if now_wall is None else now_wall
        if now < state.retired_until:
            return True
        state.retired = False
        state.retired_until = None
        state.failure_count = 0
        return False

    def _retired_until_for_reason(self, reason: str, *, retired_at: Optional[float] = None) -> Optional[float]:
        reason_text = str(reason)
        if reason_text in _AUTH_RETIRE_REASONS:
            seconds = _DEFAULT_AUTH_RETIRE_SECONDS
            env_auth = os.environ.get("TOPOBENCH_AUTH_RETIRE_SECONDS", "").strip()
            if env_auth:
                try:
                    parsed = float(env_auth)
                except ValueError:
                    parsed = -1.0
                if parsed == 0:
                    return None  # opt back into permanent retirement for auth failures
                if parsed > 0:
                    seconds = parsed
            base = retired_at if retired_at else time.time()
            return base + seconds
        if reason_text not in _TEMPORARY_RETIRE_REASONS:
            return None
        env_seconds = os.environ.get("TOPOBENCH_KEY_RETIRE_SECONDS", "").strip()
        if env_seconds:
            try:
                seconds = float(env_seconds)
            except ValueError:
                seconds = 0.0
            if seconds > 0:
                return time.time() + seconds
            if seconds == 0:
                return None

        base = datetime.fromtimestamp(retired_at) if retired_at else datetime.now()
        tomorrow = (base + timedelta(days=1)).date()
        reset_at = datetime.combine(tomorrow, datetime.min.time()) + timedelta(minutes=5)
        return reset_at.timestamp()


class SharedKeyPool(KeyPool):
    """Cross-process API key pool using filesystem leases.

    This pool keeps the old in-process round-robin behavior, but adds:
    - an exclusive lease file per key, so two Python processes cannot use the
      same key at the same time;
    - shared per-key state for cooldown, retirement, and min-interval throttling.

    The shared directory defaults to the OS temp directory. Set
    TOPOBENCH_KEY_POOL_DIR to point multiple repos/runs at an explicit location,
    or delete that directory to clear shared state.
    """

    def __init__(
        self,
        provider: str,
        keys: List[str],
        min_interval_seconds: float = 0.0,
        *,
        slots_per_key: int = 1,
        state_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            keys=keys,
            min_interval_seconds=min_interval_seconds,
            slots_per_key=slots_per_key,
        )
        root = (
            Path(state_dir)
            if state_dir is not None
            else Path(os.environ.get("TOPOBENCH_KEY_POOL_DIR", Path(tempfile.gettempdir()) / "topobench_key_pool"))
        )
        # slots_per_key is part of the shared directory so two pools that disagree
        # on holder count never share lease files.
        self._shared_dir = root / f"{self._safe_name(provider)}_s{self._slots_per_key}"
        self._shared_dir.mkdir(parents=True, exist_ok=True)
        # Per key we may hold up to `slots_per_key` leases simultaneously (e.g. when
        # two threads inside the same process both `acquire()` the same key). The
        # value is a list of (handle, slot_index); release() pops one entry.
        self._held_leases: dict[str, list[tuple[Any, int]]] = {}

    def acquire(self, timeout: float = 300.0) -> str:
        """Return a key with an exclusive cross-process lease (one of N slots)."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                now_wall = time.time()
                active = [s for s in self._states if not self._is_retired(s, now_wall)]
                if not active:
                    raise PoolExhausted(
                        f"All keys for provider {self._provider!r} have been retired."
                    )

                n = len(self._states)
                earliest_ready_wall = float("inf")
                saw_busy_lease = False

                for offset in range(n):
                    idx = (self._cursor + offset) % n
                    s = self._states[idx]
                    if self._is_retired(s, now_wall) or s.slots_in_use >= self._slots_per_key:
                        continue

                    lease = self._try_acquire_lease(s.key)
                    if lease is None:
                        saw_busy_lease = True
                        continue

                    keep_lease = False
                    try:
                        state_result = self._update_shared_state(
                            s.key,
                            lambda state: self._claim_if_ready(state, now_wall),
                        )
                        claimed, available_at = state_result
                        if claimed:
                            s.last_used_at = time.monotonic()
                            s.slots_in_use += 1
                            self._cursor = (idx + 1) % n
                            self._held_leases.setdefault(s.key, []).append(lease)
                            keep_lease = True
                            return s.key
                        if available_at is None:
                            s.retired = True
                            s.retired_until = None
                            continue
                        if available_at < 0:
                            s.retired = True
                            s.retired_until = -available_at
                            earliest_ready_wall = min(earliest_ready_wall, s.retired_until)
                            continue
                        earliest_ready_wall = min(earliest_ready_wall, available_at)
                    finally:
                        if not keep_lease:
                            self._release_lease_handle(lease)

                if all(self._is_retired(s, time.time()) for s in self._states):
                    raise PoolExhausted(
                        f"All keys for provider {self._provider!r} have been retired."
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PoolExhausted(
                        f"Timed out waiting for an available shared key from provider {self._provider!r}."
                    )

                if earliest_ready_wall != float("inf"):
                    sleep_for = min(max(0.05, earliest_ready_wall - time.time()), remaining)
                elif saw_busy_lease:
                    sleep_for = min(0.5, remaining)
                else:
                    sleep_for = min(0.1, remaining)
                self._cv.wait(timeout=sleep_for)

    def release(self, key: str) -> None:
        """Release one cross-process lease for key, freeing one slot."""
        with self._cv:
            held = self._held_leases.get(key)
            if held:
                lease = held.pop()
                if not held:
                    self._held_leases.pop(key, None)
                self._release_lease_handle(lease)
            s = self._find(key)
            if s is not None and s.slots_in_use > 0:
                s.slots_in_use -= 1
            self._cv.notify_all()

    def mark_quota_exhausted(self, key: str, reason: str) -> None:
        super().mark_quota_exhausted(key, reason)
        self._update_shared_state(
            key,
            lambda state: self._set_shared_retired(state, reason),
        )

    def mark_rate_limited(self, key: str, cooldown_seconds: float) -> None:
        super().mark_rate_limited(key, cooldown_seconds)
        cooldown_until = time.time() + max(0.0, float(cooldown_seconds))
        self._update_shared_state(
            key,
            lambda state: self._set_shared_cooldown(state, cooldown_until),
        )

    def _claim_if_ready(self, state: dict[str, Any], now_wall: float) -> tuple[bool, float | None]:
        if bool(state.get("retired", False)):
            retired_until = state.get("retired_until")
            if retired_until is None:
                reason = str(state.get("retired_reason") or "")
                migrated_until = self._retired_until_for_reason(
                    reason,
                    retired_at=float(state.get("retired_at", 0.0) or 0.0) or None,
                )
                if migrated_until is None:
                    return False, None
                state["retired_until"] = migrated_until
                retired_until = migrated_until
            retired_until_float = float(retired_until or 0.0)
            if now_wall < retired_until_float:
                return False, -retired_until_float
            state["retired"] = False
            state["retired_reason"] = None
            state["retired_at"] = 0.0
            state["retired_until"] = None
        cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        last_used_at = float(state.get("last_used_at", 0.0) or 0.0)
        available_at = max(cooldown_until, last_used_at + self._min_interval)
        if available_at <= now_wall:
            state["last_used_at"] = now_wall
            return True, available_at
        return False, available_at

    def _set_shared_retired(self, state: dict[str, Any], reason: str) -> None:
        state["retired"] = True
        state["retired_reason"] = str(reason)
        state["retired_at"] = time.time()
        state["retired_until"] = self._retired_until_for_reason(reason)

    def _set_shared_cooldown(self, state: dict[str, Any], cooldown_until: float) -> None:
        state["cooldown_until"] = max(float(state.get("cooldown_until", 0.0) or 0.0), cooldown_until)

    def _update_shared_state(self, key: str, updater: Callable[[dict[str, Any]], Any]) -> Any:
        paths = self._paths_for_key(key)
        paths["state"].parent.mkdir(parents=True, exist_ok=True)
        with paths["state_lock"].open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state(paths["state"])
                result = updater(state)
                self._write_state(paths["state"], key, state)
                return result
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _try_acquire_lease(self, key: str) -> Any | None:
        """Try each of the N slot lease files; return the first that locks, or None."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        self._shared_dir.mkdir(parents=True, exist_ok=True)
        for slot in range(self._slots_per_key):
            lease_path = self._shared_dir / f"{digest}.lease.{slot}.lock"
            lease = lease_path.open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            return lease
        return None

    def _release_lease_handle(self, lease: Any) -> None:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()

    def _read_state(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_state(self, path: Path, key: str, state: dict[str, Any]) -> None:
        payload = {
            "key_suffix": key[-6:],
            "retired": bool(state.get("retired", False)),
            "retired_reason": state.get("retired_reason"),
            "retired_at": float(state.get("retired_at", 0.0) or 0.0),
            "retired_until": state.get("retired_until"),
            "cooldown_until": float(state.get("cooldown_until", 0.0) or 0.0),
            "last_used_at": float(state.get("last_used_at", 0.0) or 0.0),
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True))
        tmp_path.replace(path)

    def _paths_for_key(self, key: str) -> dict[str, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return {
            "state": self._shared_dir / f"{digest}.json",
            "state_lock": self._shared_dir / f"{digest}.state.lock",
        }

    def _safe_name(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value) or "provider"
