"""Optional TensorBoard process owned by a workflow invocation."""

from __future__ import annotations

import shlex
import shutil
import socket
import subprocess
import time
import urllib.request
import warnings
import webbrowser
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def tensorboard_session(
    options: Mapping[str, Any], log_dir: Path, *, dry_run: bool = False
) -> Iterator[None]:
    """Print or serve logs; visualization failures never abort training."""
    mode = options.get("mode", "print")
    if mode not in {"off", "print", "serve"}:
        raise ValueError("TensorBoard mode must be off, print, or serve.")
    port = int(options.get("port", 6006))
    if not 1 <= port <= 65535:
        raise ValueError("TensorBoard port must be between 1 and 65535.")
    command = [
        "tensorboard",
        "--logdir",
        str(log_dir.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = None
    stream = None
    if mode != "off":
        print(f"TensorBoard: {shlex.join(command)}", flush=True)
    try:
        if mode == "serve" and not dry_run:
            try:
                if shutil.which("tensorboard") is None:
                    raise RuntimeError("Install with uv sync --extra monitoring")
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
                log_dir.mkdir(parents=True, exist_ok=True)
                stream = (log_dir / "tensorboard-server.log").open(
                    "a", encoding="utf-8"
                )
                process = subprocess.Popen(command, stdout=stream, stderr=stream)
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 10
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"TensorBoard exited; see {stream.name}")
                    try:
                        with urllib.request.urlopen(url, timeout=0.3):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "TensorBoard startup timed out"
                            ) from None
                        time.sleep(0.1)
                print(f"TensorBoard URL: {url}", flush=True)
                if options.get("open_browser", False):
                    webbrowser.open(url)
            except Exception as error:
                warnings.warn(f"TensorBoard unavailable: {error}", stacklevel=2)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if stream is not None:
            stream.close()
