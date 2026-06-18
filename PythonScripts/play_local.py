"""Minimal Python "local-play host" for human play (M0, CTO decision 2026-06-17).

Keeps Python in the loop to clock the Unity simulator while humans control their
tanks locally inside Unity (keyboard / gamepad). This host does NOT pull in the RL
stack: no ``tank_env`` / gymnasium / SB3 / torch. It speaks the EXISTING unframed
TCP-JSON protocol that ``tank_env.py`` uses, sending zero-stub actions for both
human slots purely to keep the protocol well-formed and drive the simulation clock.

Protocol (as-built; traced from ``tank_env.py`` and ``DriverController.cs``):
  - Unity is the TCP *listener*; it blocks on ``AcceptTcpClient`` until we connect.
    The socket round-trip is the game's clock (``Time.timeScale`` is held at 0
    between exchanges), so the loop below is what actually advances the match.
  - Handshake:  send {"restart": true} -> recv {"restarting": true}
                send {"start":   true} -> recv {"starting":   true}
                recv {"state": [...52 floats...]}        (first state)
  - Per step:   send {"1": [0,0,0,0,0], "2": [0,0,0,0,0]} -> recv {"state": [...]}
                (the stubs occupy both player slots; humans are read locally by
                 Unity and the stubs are discarded for any non-AI slot.)
  - End-of-match: Unity tags the final state with "done" and/or "winner".
  - Teardown:   send {"restart": true} -> recv {"restarting": true}
                send {"end":     true} -> recv {"ending":     true}

Cross-platform: ``spawn``-safe (no problematic module-level work; guarded main),
``subprocess`` arg-lists (never shell strings), ``pathlib`` paths, no ``os.system``.
M0 only requires Windows play, but this is written to run on Linux too.

Deliberately NOT done here (out of M0 scope):
  - No message framing. The unframed one-JSON-object-per-``recv`` contract that
    ``tank_env`` relies on is preserved; framing is a Phase-4 both-sides task.
  - No recorder is built. The step loop is structured so a future ``record=True``
    path can log the host-side state stream via the ``on_state`` hook WITHOUT any
    protocol change (per the imitation-learning recording design intent).
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# Default ports kept off tank_env's 50000 family to avoid clashing with training.
DEFAULT_GAME_PORT = 50801
DEFAULT_CONNECT_ATTEMPTS = 60
DEFAULT_SOCK_TIMEOUT = 30.0
# Bound the step loop so a stuck match (or a stuck human-less test) cannot run
# forever; the build's own game_maxTime should end matches well before this.
DEFAULT_MAX_STEPS = 100_000

# A human-controlled slot's action is read locally by Unity and the value we send
# is discarded; the zero stub merely keeps the JSON well-formed and drives the clock.
STUB_ACTION = [0, 0, 0, 0, 0]


def _send(sock: socket.socket, message: dict) -> None:
    """Serialize STRICT JSON (no trailing commas / leading-dot floats) and send."""
    data = json.dumps(message)
    sock.sendall(data.encode("utf-8"))


def _receive(sock: socket.socket) -> dict:
    """Read one JSON object per ``recv`` (the unframed M0 contract; no framing)."""
    received = sock.recv(1024)
    if not received:
        raise ConnectionError("Unity closed the connection (empty recv)")
    return json.loads(received.decode("utf-8"))


def launch_build(game_path: Path, game_port: int, config_path: Optional[Path]) -> subprocess.Popen:
    """Launch the standalone build via a subprocess arg-list (never a shell string).

    Unity parses ``args[1]`` as the connection port, so the port MUST be the first
    positional arg. ``--config <path>`` is an optional named arg that the build's
    ``DriverController.ResolveConfigPath`` honors; without it the build falls back
    to its StreamingAssets ``config.json``.
    """
    cmd: list[str] = [str(game_path), str(game_port)]
    if config_path is not None:
        cmd += ["--config", str(config_path)]
    # No stdout/stderr capture here keeps this dependency-light; the build also
    # writes a Unity Player.log. Detach nothing: we own this process and reap it.
    return subprocess.Popen(cmd)


def connect_to_unity(
    game_ip: str,
    game_port: int,
    my_port: Optional[int],
    attempts: int,
    sock_timeout: float,
    verbose: bool,
) -> socket.socket:
    """Connect as the TCP *client* (Unity is the listener), mirroring tank_env.

    Binds our end to ``my_port`` (defaults to game_port + 1, matching tank_env), then
    retries ``connect`` while Unity boots and reaches ``AcceptTcpClient``. The socket
    timeout is applied only AFTER connect so the connect retries are not cut short.
    """
    if my_port is None:
        my_port = game_port + 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", my_port))

        connected = False
        for _ in range(attempts):
            try:
                sock.connect((game_ip, game_port))
                connected = True
                break
            except (ConnectionRefusedError, OSError):
                # Unity hasn't reached AcceptTcpClient yet; back off and retry.
                if verbose:
                    print(f"[play_local] waiting for Unity on {game_ip}:{game_port} ...", flush=True)
                time.sleep(1)
        if not connected:
            raise ConnectionError(
                f"Could not connect to Unity on {game_ip}:{game_port} after {attempts} attempts"
            )
    except BaseException:
        sock.close()
        raise

    sock.settimeout(sock_timeout)
    if verbose:
        print(f"[play_local] connected to {game_ip}:{game_port} (my_port={my_port})", flush=True)
    return sock


def handshake(sock: socket.socket, verbose: bool) -> dict:
    """Run the reset handshake exactly as tank_env.reset does; return first state."""
    _send(sock, {"restart": True})
    _receive(sock)  # expect {"restarting": true}
    _send(sock, {"start": True})
    received = _receive(sock)
    if "starting" not in received:
        raise RuntimeError(f"Expected 'starting' ack, got: {received!r}")
    state = _receive(sock)  # first {"state": [...]}
    if verbose:
        print("[play_local] handshake complete; match started", flush=True)
    return state


def _is_done(state: dict) -> bool:
    """Match tank_env.step's done condition: a 'done' tag or a 'winner' field."""
    return ("done" in state) or ("winner" in state)


def run_match(
    sock: socket.socket,
    max_steps: int,
    verbose: bool,
    on_state: Optional[Callable[[int, dict], None]] = None,
) -> Optional[int]:
    """Clock one match to completion with zero-stub actions for both slots.

    ``on_state(step_index, state)`` is the host-side recording hook: a future
    ``record=True`` path logs the state stream here WITHOUT any protocol change.
    It receives every state INCLUDING the initial and the terminal one. We do not
    build the recorder; we only expose the seam.

    Returns the winner id (Unity's convention) if the final state carries one,
    else ``None`` (e.g. a tie / max-steps cutoff).
    """
    # The first state already arrived during the handshake; feed it to the hook.
    state = handshake(sock, verbose)
    if on_state is not None:
        on_state(0, state)

    winner: Optional[int] = None
    for step_index in range(1, max_steps + 1):
        if _is_done(state):
            break
        actions = {"1": list(STUB_ACTION), "2": list(STUB_ACTION)}
        _send(sock, actions)
        state = _receive(sock)
        if on_state is not None:
            on_state(step_index, state)
        if "winner" in state:
            winner = int(state["winner"])
    else:
        if verbose:
            print(f"[play_local] reached max_steps={max_steps} without a done flag", flush=True)

    if verbose:
        print(f"[play_local] match ended (winner={winner})", flush=True)
    return winner


def shutdown(sock: socket.socket, verbose: bool) -> None:
    """Clean protocol teardown, mirroring tank_env.close: restart-ack then end-ack."""
    _send(sock, {"restart": True})
    _receive(sock)  # expect {"restarting": true}
    _send(sock, {"end": True})
    received = _receive(sock)
    if "ending" not in received and verbose:
        print(f"[play_local] warning: expected 'ending' ack, got: {received!r}", flush=True)


def _reap(proc: subprocess.Popen, verbose: bool) -> None:
    """Guarantee no orphaned Unity process on ANY exit path (normal / error / Ctrl-C).

    Tries a graceful wait first (the build quits itself once it receives {"end":true}
    and its listener stops), then escalates to terminate() and finally kill().
    """
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        if verbose:
            print("[play_local] build still running after end; terminating", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        if verbose:
            print("[play_local] terminate timed out; killing", flush=True)
    proc.kill()
    proc.wait()


def play_local(args: argparse.Namespace) -> int:
    game_path = Path(args.game_path).resolve()
    if not game_path.is_file():
        print(f"[play_local] build not found: {game_path}", file=sys.stderr)
        return 2

    config_path: Optional[Path] = None
    if args.config is not None:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            print(f"[play_local] config not found: {config_path}", file=sys.stderr)
            return 2

    proc = launch_build(game_path, args.port, config_path)
    sock: Optional[socket.socket] = None
    try:
        sock = connect_to_unity(
            game_ip=args.ip,
            game_port=args.port,
            my_port=args.my_port,
            attempts=args.connect_attempts,
            sock_timeout=args.sock_timeout,
            verbose=args.verbose,
        )
        # record=True would pass a real on_state sink here; the seam is wired,
        # the recorder is intentionally not built for M0.
        run_match(sock, max_steps=args.max_steps, verbose=args.verbose, on_state=None)
        shutdown(sock, args.verbose)
        return 0
    except KeyboardInterrupt:
        # Best-effort clean protocol teardown on Ctrl-C, then _reap guarantees no orphan.
        print("[play_local] interrupted; shutting down Unity", file=sys.stderr)
        if sock is not None:
            try:
                shutdown(sock, args.verbose)
            except (OSError, ValueError, RuntimeError):
                pass
        return 130
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        _reap(proc, args.verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play_local.py",
        description=(
            "Local-play host: launch the Unity build and clock a human-vs-human match "
            "over the existing TCP-JSON protocol (no RL stack)."
        ),
    )
    parser.add_argument(
        "--game-path",
        default=str(Path(__file__).resolve().parents[1] / "Builds" / "Windows" / "TankTwinStickShooter.exe"),
        help="Path to the standalone build executable (default: repo Builds/Windows build).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_GAME_PORT, help="TCP port Unity listens on (passed as args[1] to the build).")
    parser.add_argument("--my-port", type=int, default=None, help="Local TCP port to bind (default: port + 1, matching tank_env).")
    parser.add_argument("--ip", default="127.0.0.1", help="IP Unity listens on (default 127.0.0.1).")
    parser.add_argument("--config", default=None, help="Optional config.json to forward as '--config' (default: build's StreamingAssets config).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Safety cap on socket-clocked steps per match.")
    parser.add_argument("--connect-attempts", type=int, default=DEFAULT_CONNECT_ATTEMPTS, help="Connect retries while the build boots.")
    parser.add_argument("--sock-timeout", type=float, default=DEFAULT_SOCK_TIMEOUT, help="Socket recv/send timeout in seconds (after connect).")
    parser.add_argument("--verbose", action="store_true", help="Print handshake / lifecycle progress.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Make SIGTERM (e.g. orchestration shutdown) raise KeyboardInterrupt so the
    # same clean-teardown path runs as for Ctrl-C. Guarded: only on platforms/threads
    # where SIGTERM is settable.
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    except (ValueError, OSError, AttributeError):
        pass
    return play_local(args)


if __name__ == "__main__":
    sys.exit(main())
