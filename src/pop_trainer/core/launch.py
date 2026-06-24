"""Launch + connect helpers for the live Unity build (stdlib only — the shared spawn seam).

The two side-effecting steps every live consumer shares: build the windowed-launch ``Popen``
arg-list for the build, and open a TCP socket to its listener with a bounded retry/backoff.
Both the demo (one local watch) and the collection runner (one build per worker) need them, so
they live in ``core`` — the dependency-free root — and stay STDLIB-ONLY (``subprocess`` /
``socket`` / ``time`` / ``pathlib``). No internal imports (no ``env`` / ``data`` / ``agents`` /
``models`` / ``rl`` / ``pretraining`` / ``tank_twin``), so there is no import cycle and ``core``
stays the leaf of the graph.

These are the LIVE-path helpers (a real subprocess, a real socket); they are not unit-tested
(the caller's tests inject fakes). :func:`build_launch_cmd` is a PURE arg-list builder, so it is
trivially testable on its own.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

__all__ = [
    "DEFAULT_SCREEN_WIDTH",
    "DEFAULT_SCREEN_HEIGHT",
    "DEFAULT_CONNECT_ATTEMPTS",
    "DEFAULT_CONNECT_BACKOFF_SEC",
    "DEFAULT_SOCK_TIMEOUT_SEC",
    "build_launch_cmd",
    "connect",
]

# Windowed-launch screen size (the build is launched as a watchable window, never fullscreen,
# never batchmode). 1280x720 is 16:9 and comfortably visible.
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 720

# Socket connect retry budget. The build needs a moment to boot and start its TCP listener, so
# the first connects are refused; retry with backoff until the budget is spent.
DEFAULT_CONNECT_ATTEMPTS = 30
DEFAULT_CONNECT_BACKOFF_SEC = 1.0
DEFAULT_SOCK_TIMEOUT_SEC = 30.0


def build_launch_cmd(
    exe: str | Path,
    port: int,
    config: str | Path,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
) -> list[str]:
    """The ``Popen`` ARG-LIST that launches the build WINDOWED (never fullscreen / batchmode).

    ``args[1]`` is the TCP port (the build's ``DriverController`` parses it positionally);
    ``--config`` points at the config JSON (which must enable ``obs_pixels`` for the pixel-frame
    channel the env reads). ``-screen-fullscreen 0`` + an explicit ``screen_width`` x
    ``screen_height`` keep the build a watchable window. ``-batchmode`` is deliberately ABSENT
    (it would hide the window). Pure: returns the arg-list, spawns nothing.
    """
    return [
        str(exe),
        str(port),
        "--config",
        str(config),
        "-screen-fullscreen",
        "0",
        "-screen-width",
        str(screen_width),
        "-screen-height",
        str(screen_height),
    ]


def connect(
    port: int,
    *,
    attempts: int = DEFAULT_CONNECT_ATTEMPTS,
    backoff_sec: float = DEFAULT_CONNECT_BACKOFF_SEC,
    timeout_sec: float = DEFAULT_SOCK_TIMEOUT_SEC,
) -> socket.socket:
    """Connect to the running build at ``127.0.0.1:port`` with a bounded retry/backoff.

    The build needs a moment to boot and start its TCP listener, so connection is refused at
    first; retry up to ``attempts`` times, sleeping ``backoff_sec`` between tries. ``SO_REUSEADDR``
    avoids a lingering-``TIME_WAIT`` bind clash on a rapid relaunch. ``timeout_sec`` is set as the
    socket's read/write timeout so a wedged build surfaces as a ``ConnectionError`` in the env
    rather than hanging forever. Returns the connected ``socket.socket``; raises ``ConnectionError``
    if no connection is made within the budget.
    """
    last_err: OSError | None = None
    for _ in range(attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError as exc:
            last_err = exc
            sock.close()
            time.sleep(backoff_sec)
            continue
        sock.settimeout(timeout_sec)
        return sock
    raise ConnectionError(
        f"could not connect to the build at 127.0.0.1:{port} after {attempts} attempts"
    ) from last_err
