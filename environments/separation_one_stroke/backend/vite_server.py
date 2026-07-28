from __future__ import annotations

import contextlib
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


def _resolve_js_runtime() -> Tuple[str, str]:
    """Return (executable_path, runtime_name) for `<runtime> run dev`.

    Prefers `bun` (drop-in for `npm run` on the dev command, faster, no
    Node.js dependency), falls back to `npm` if bun isn't on PATH.
    """
    bun = shutil.which("bun")
    if bun:
        return bun, "bun"
    npm = shutil.which("npm")
    if npm:
        return npm, "npm"
    raise FileNotFoundError(
        "Neither `bun` nor `npm` found on PATH. Install bun (https://bun.sh) "
        "or Node.js+npm to run the Vite dev server."
    )


def choose_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class ViteFrontendServer:
    def __init__(
        self,
        *,
        frontend_dir: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        self.frontend_dir = Path(frontend_dir).resolve()
        self.host = host
        self.port = int(port) if int(port) > 0 else choose_free_port(host)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.process: Optional[subprocess.Popen[str]] = None
        self.base_url = f"http://{self.host}:{self.port}"
        self._log_lines: Deque[str] = deque(maxlen=200)
        self._log_thread: Optional[threading.Thread] = None

    def _consume_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self._log_lines.append(line.rstrip())

    def _log_indicates_ready(self) -> bool:
        return any(
            "ready in" in line.casefold() or "local:" in line.casefold()
            for line in self._log_lines
        )

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_seconds
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "Vite dev server exited early.\n"
                    + "\n".join(self._log_lines)
                )
            if self._log_indicates_ready():
                try:
                    with socket.create_connection((self.host, self.port), timeout=5.0):
                        return
                except OSError:
                    time.sleep(0.2)
                    continue
            try:
                with urlopen(self.base_url, timeout=5.0) as response:
                    if 200 <= int(response.status) < 500:
                        return
            except (TimeoutError, URLError):
                time.sleep(0.2)
        raise TimeoutError(
            f"Timed out waiting for Vite dev server at {self.base_url}.\n"
            + "\n".join(self._log_lines)
        )

    def start(self) -> str:
        if not self.frontend_dir.exists():
            raise FileNotFoundError(f"Frontend directory not found: {self.frontend_dir}")
        if not (self.frontend_dir / "package.json").exists():
            raise FileNotFoundError(f"Missing package.json in frontend directory: {self.frontend_dir}")
        if not (self.frontend_dir / "node_modules").exists():
            raise FileNotFoundError(
                f"Missing node_modules in {self.frontend_dir}. Run `bun install` (or `npm install`) in the frontend directory first."
            )
        if self.process is not None:
            return self.base_url

        runtime_path, _ = _resolve_js_runtime()
        command = [
            runtime_path,
            "run",
            "dev",
            "--",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--strictPort",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(self.frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._consume_logs, daemon=True)
        self._log_thread.start()
        self._wait_until_ready()
        return self.base_url

    def close(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        self.process = None
        if self._log_thread is not None:
            self._log_thread.join(timeout=1.0)
            self._log_thread = None

    def __enter__(self) -> "ViteFrontendServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def log_tail(self) -> str:
        return "\n".join(self._log_lines)
