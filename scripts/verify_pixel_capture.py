"""Stage-1 LIVE VERIFY for the pixel-capture frame channel (the gate evidence).

Launches the freshly-rebuilt standalone with a pixels-ON config, runs the existing
handshake, drives a few steps with DIVERSE actions, and on each step reads the raw 52-float
``state`` JSON (``Connection.receive``) and THEN the length-prefixed pixel frame
(``Connection.receive_frame``). It saves a few (frame PNG, paired state JSON) pairs under
``docs/pixel-capture-samples/`` for the CTO's eyeball check, and prints per-frame
diagnostics (W/H/C, tag-OK, shape/dtype, min/max pixel values, and the paired state[0:6]).

This is DELIBERATELY standalone: it opens its OWN socket and launches the build directly via
a subprocess arg-list ``[exe, str(port), "--config", <abs config>]`` (reusing
``env._build_game_cmd`` for the arg-list ONLY) — it does NOT route through ``TankEnv``'s
training path, so the frozen RL seam in env.py is untouched. Teardown mirrors
``TankEnv.close`` (best-effort end handshake, then terminate + wait) so no orphaned
``TankTwinStickShooter.exe`` is left.

Cross-platform rules: subprocess arg-list (never a shell string); stdlib ``socket`` /
``pathlib`` / ``json``; no ``fork``. Run from the repo with ``uv run python
scripts/verify_pixel_capture.py`` (optionally ``--config <path>`` / ``--steps N``).
"""

import argparse
import contextlib
import json
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make the src-layout package importable when run as a plain script (uv run handles the
# install, but this keeps it robust if invoked directly from the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tank_twin.env import _build_game_cmd  # noqa: E402  (after sys.path shim)
from tank_twin.protocol import Connection  # noqa: E402

DEFAULT_EXE = _REPO_ROOT / "build" / "TankTwinStickShooter.exe"
# Prefer the config the BUILD itself ships (its arena travels with it); fall back to the
# repo's StreamingAssets copy if present. Either is forwarded as an absolute --config path.
BUILD_PIXELS_CONFIG = (
    _REPO_ROOT / "build" / "TankTwinStickShooter_Data" / "StreamingAssets" / "config_pixels.json"
)
REPO_PIXELS_CONFIG = _REPO_ROOT / "Assets" / "StreamingAssets" / "config_pixels.json"
SAMPLES_DIR = _REPO_ROOT / "docs" / "pixel-capture-samples"


def connect_to_build(exe_path, config_path, *, game_port=50000, my_port=50001, attempts=60):
    """Launch the build (arg-list) and open the TCP socket, mirroring env._connect_to_unity.

    Standalone: opens its own socket and does the bind/connect retry loop. Returns
    ``(Connection, Popen, log_handle)``. The ``--config`` arg is forwarded (absolute) so the
    build runs the pixels-ON topology. NOT routed through TankEnv.
    """
    log_path = _REPO_ROOT / "gamelog_pixelverify.txt"
    log = open(log_path, "w")  # noqa: SIM115  (held for the child's lifetime; closed by caller)
    cmd = _build_game_cmd(exe_path, game_port, config_path)
    print(f"launching: {cmd}")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bound = False
    last_err = None
    for _ in range(attempts):
        try:
            sock.bind(("", my_port))
            bound = True
            break
        except OSError as exc:
            last_err = exc
            time.sleep(1)
    if not bound:
        for _ in range(attempts):
            try:
                sock.bind(("", random.randint(33000, 60000)))
                bound = True
                break
            except OSError as exc:
                last_err = exc
                time.sleep(1)
    if not bound:
        raise last_err if last_err else OSError("could not bind socket")

    connected = False
    for _ in range(attempts):
        try:
            sock.connect(("127.0.0.1", game_port))
            connected = True
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    if not connected:
        raise ConnectionError(f"could not connect to the build at 127.0.0.1:{game_port}")

    sock.settimeout(15.0)
    return Connection(sock), proc, log


def handshake(conn):
    """The reset handshake, pixels-ON variant: restart -> ack, start -> 'starting' ack.

    The restart / start acks are JSON-only control writes (no frame), so they are read with
    the plain ``receive``. The FIRST step's ``state`` (the build's first ``SendAndReceiveData``
    write) comes WITH a trailing frame on the pixels path, so it is read by the caller via
    ``receive_state_and_frame`` exactly like every subsequent step (do NOT read it here with
    the unframed ``receive`` — that would mis-split the coalesced state + frame).
    """
    conn.send({"restart": True})
    conn.receive()
    conn.send({"start": True})
    ack = conn.receive()
    if "starting" not in ack:
        raise RuntimeError(f"unexpected start ack: {ack!r}")


def diverse_action(i, rng):
    """A small repertoire of DISTINCT actions so the tanks actually move/fire.

    Action layout is the 5-float [move_x, move_y, aim_x, aim_y, fire] the env sends. We
    cycle two fixed motions + one random draw so successive frames differ visibly.
    """
    fixed = [
        [1.0, 0.0, 1.0, 0.0, 1.0],  # drive +x, aim +x, FIRE
        [0.0, 1.0, 0.0, 1.0, 1.0],  # drive +y, aim +y, FIRE
        [-1.0, -1.0, -1.0, 0.0, 0.0],  # drive -x/-y, aim -x, hold fire
    ]
    if i % 3 == 2:
        return rng.uniform(-1.0, 1.0, 5).tolist()
    return fixed[i % 3]


def teardown(conn, proc, log):
    """Mirror TankEnv.close: best-effort end handshake, close socket, terminate + wait."""
    try:
        conn.send({"restart": True})
        conn.receive()
        conn.send({"end": True})
        conn.receive()
    except (ConnectionError, OSError, KeyError, ValueError):
        pass
    with contextlib.suppress(OSError):
        conn.transport.close()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log.close()


DEFAULT_CONFIG = (
    _REPO_ROOT / "build" / "TankTwinStickShooter_Data" / "StreamingAssets" / "config.json"
)


def verify_no_pixel_path(exe_path, config_path, *, port, steps=3):
    """Launch the build with a NO-pixels config and confirm the wire is byte-identical.

    With ``obs_pixels`` absent/false the build's frame write is guarded off (FrameCapture.cs +
    DriverController.cs only write a frame when obsPixels==true), so NOTHING new follows the
    state JSON. This drives the plain handshake + a few steps reading state ONLY with the
    UNCHANGED ``receive`` (recv(1024)). Each ``receive`` must strict-decode to a state dict and
    the per-step buffer must stay EMPTY (no trailing frame bytes) — the proof there are no
    extra bytes on the no-pixel wire. NO frame read is attempted.
    """
    print("\n=== no-pixel path byte-identical check ===")
    print(f"config: {config_path}")
    conn, proc, log = connect_to_build(exe_path, config_path, game_port=port, my_port=port + 1)
    try:
        conn.send({"restart": True})
        conn.receive()
        conn.send({"start": True})
        ack = conn.receive()
        assert "starting" in ack, ack
        first = conn.receive()  # plain unframed read; no frame follows on the no-pixel wire
        assert "state" in first and conn._buffer == b"", (first.keys(), conn._buffer[:16])
        print(f"  handshake OK; first state len={len(first['state'])}, buffer empty (no frame)")
        rng = np.random.default_rng(1)
        for i in range(steps):
            conn.send({1: rng.uniform(-1, 1, 5).tolist(), 2: rng.uniform(-1, 1, 5).tolist()})
            received = conn.receive()  # the EXISTING path; must read exactly the state JSON
            assert "state" in received and conn._buffer == b"", received.keys()
            print(f"  step {i}: state len={len(received['state'])}, buffer empty -> no extra bytes")
        print("  PASS: no-pixel wire is byte-identical (receive() read exactly the state JSON).")
    finally:
        teardown(conn, proc, log)
        print("  teardown complete (no-pixel build terminated).")


def main():
    parser = argparse.ArgumentParser(description="Live verify the pixel-capture frame channel.")
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument(
        "--config", default=None, help="pixels-ON config (default: build's shipped copy)"
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--save", type=int, default=4, help="how many (frame,state) PNG pairs to save"
    )
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="run ONLY the no-pixel byte-identical check (default config, no frame read)",
    )
    args = parser.parse_args()

    exe_path_early = Path(args.exe).resolve()
    if args.no_pixels:
        if not exe_path_early.exists():
            raise SystemExit(f"build exe not found: {exe_path_early}")
        verify_no_pixel_path(exe_path_early, DEFAULT_CONFIG.resolve(), port=args.port)
        return

    config_path = args.config
    if config_path is None:
        config_path = BUILD_PIXELS_CONFIG if BUILD_PIXELS_CONFIG.exists() else REPO_PIXELS_CONFIG
    config_path = Path(config_path).resolve()
    exe_path = Path(args.exe).resolve()
    if not exe_path.exists():
        raise SystemExit(f"build exe not found: {exe_path}")
    if not config_path.exists():
        raise SystemExit(f"pixels config not found: {config_path}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"exe:    {exe_path}")
    print(f"config: {config_path}")
    conn, proc, log = connect_to_build(
        exe_path, config_path, game_port=args.port, my_port=args.port + 1
    )

    saved = 0
    save_stride = max(1, args.steps // max(1, args.save))
    try:
        handshake(conn)
        # The build wrote the FIRST state+frame immediately after the start ack (it is now
        # blocked reading our first action). Read that pair, THEN the loop sends an action and
        # reads the NEXT pair — mirroring TankEnv.reset (read first state) + step.
        received, frame = conn.receive_state_and_frame()
        print(f"handshake OK; first state len={len(received['state'])} (frame {frame.shape})")

        for i in range(args.steps):
            state = received["state"]
            h, w, c = frame.shape
            done = "done" in received
            winner = received.get("winner")
            tag_ok = True  # receive_frame asserts the 0x46 tag internally; reaching here == OK
            print(
                f"step {i:2d}: frame W={w} H={h} C={c} tag_ok={tag_ok} shape={frame.shape} "
                f"dtype={frame.dtype} min={int(frame.min())} max={int(frame.max())} "
                f"state[0:6]={[round(float(x), 3) for x in state[0:6]]} "
                f"winner={winner} done={done}"
            )

            if saved < args.save and (i % save_stride == 0 or done):
                png_path = SAMPLES_DIR / f"frame_{saved:02d}_step{i:02d}.png"
                json_path = SAMPLES_DIR / f"frame_{saved:02d}_step{i:02d}.state.json"
                Image.fromarray(frame, mode="RGB").save(png_path)
                json_path.write_text(
                    json.dumps(
                        {"step": i, "state": state, "winner": winner, "done": done}, indent=2
                    ),
                    encoding="utf-8",
                )
                print(f"          saved {png_path.name} + {json_path.name}")
                saved += 1

            if done:
                if saved >= args.save:
                    break
                # Re-handshake + read the new episode's first state+frame.
                handshake(conn)
                received, frame = conn.receive_state_and_frame()
                continue

            # Drive the next step with a diverse action and read the next state+frame pair.
            action = diverse_action(i, rng)
            opp = diverse_action(i + 1, rng)
            conn.send({1: list(action), 2: list(opp)})
            received, frame = conn.receive_state_and_frame()

        print(f"\nDONE: saved {saved} frame/state pairs under {SAMPLES_DIR}")
    finally:
        teardown(conn, proc, log)
        print("teardown complete (build terminated, socket closed)")


if __name__ == "__main__":
    main()
