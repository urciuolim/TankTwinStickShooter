"""Drive the real Unity build to PROVE timeScale=5 is computationally identical to 1x.

This is the integration half of the M1 validity gate (Workstream C, part 2). It
launches ``build/TankTwinStickShooter.exe`` against the two byte-identical configs
``config_determinism_t1.json`` / ``config_determinism_t5.json`` (differ only in
``timeScale``), drives an IDENTICAL canned action sequence through both, captures the
inbound 52-float state every step, and hands the recorded trajectories to the pure
:mod:`tank_twin.determinism_check` for the verdict.

It is NOT a unit test (it needs a live Unity build) — run it as a CLI:

    python -m tank_twin.determinism_harness --game-path build/TankTwinStickShooter.exe

WHY THIS IS NON-TRIVIAL — THE SIDE-FLIP. ``DriverController.Reset()`` does
``flip = Random.value >= .5f`` once per launch, point-reflecting the whole arena
(player1 spawns at x=-7 or x=+7). It is per-launch RNG, NOT pinnable via config, and
lives on the RL seam (we do NOT touch C#). We neutralize it IN THE HARNESS by
ALIGNMENT, not mirror-math: mirror-math is unsafe because the actions are in world
coordinates, so an x-asymmetric action under a mirrored start does NOT produce a
mirrored trajectory. Instead we detect the flip from the FIRST state (the sign of
player1's x = state index 0) and RELAUNCH a run until its first-state flip matches a
chosen target. Run A picks the target; every subsequent run (the t1 baseline twin AND
the t5 run) is relaunched until aligned. With a fair coin that is ~2 launches each;
we cap retries and error clearly if it never aligns.

Then: identical configs (except timeScale) + identical starting flip + identical
actions ⇒ ANY trajectory divergence is attributable PURELY to timeScale. The diff is
the verdict; a dropped physics step under the 0->5x burst manifests as divergence.

The harness MEASURES; it does NOT hard-fail on a chosen epsilon — per the board
decision the CTO sets the pass bar AFTER seeing the numbers. We write the numbers to
``runs/determinism/result.json`` (strict JSON) and print a clean summary.

SELF-EVIDENCE so the artifact STANDS ALONE (a reviewer need not read C#):

* NON-DEGENERACY. A frozen-tank build would ALSO report max-diff 0.0, so a bare 0.0
  is ambiguous. We record, over the t1 reference trajectory, each tank's positional
  range (max-min of x/y) and max step-to-step delta and a ``non_degenerate`` bool
  (motion clears a world-unit floor). 0.0 then means "identical DESPITE real motion."
  A degenerate result prints a LOUD warning.
* WALL-CLOCK. We time each run's drive loop and record t1 / t1_again / t5 seconds and
  ``t1_over_t5_ratio`` (≈5.0), CONFIRMING ``timeScale=5`` actually took effect rather
  than being silently ignored. Durations are recorded measurements, NOT determinism
  inputs — they never enter the trajectory diff.

Process hygiene (the Windows orphan / TIME_WAIT trap we hit before): a FRESH port per
launch, and every launched build is terminated + reaped before the next launch. No
fork/forkserver. Strict JSON via :mod:`tank_twin.protocol`.
"""

import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from tank_twin.protocol import Connection

# Repo root: src/tank_twin/determinism_harness.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STREAMING = _REPO_ROOT / "Assets" / "StreamingAssets"
CONFIG_T1 = _STREAMING / "config_determinism_t1.json"
CONFIG_T5 = _STREAMING / "config_determinism_t5.json"
RESULT_PATH = _REPO_ROOT / "runs" / "determinism" / "result.json"

# State index 0 is player1's x (GameController.UpdateState s[0]); its SIGN is the
# per-launch side-flip we align on.
P1_X_INDEX = 0

# Cap on relaunch attempts to hit a target flip (fair coin ⇒ ~2 expected).
DEFAULT_ALIGN_RETRIES = 12


def _canned_actions(n_steps):
    """A FIXED, deterministic action sequence (identical for every run, no RNG).

    Each action is the 5-float vector the env's action_space uses
    (``Box(-1, 1, shape=(5,))``); the same vector is sent on BOTH player slots
    ``{1: a, 2: a}`` every step. The sequence cycles through a small repertoire that
    stresses movement + rotation + firing + collisions so any dropped physics step
    under the 5x burst shows up as trajectory divergence:

    * forward (drive +y),
    * turn / rotate aim,
    * strafe (drive +x),
    * shoot (fire while moving),
    * reverse (drive -y) to provoke wall/edge collisions.

    Convention of the 5 floats mirrors the 2021 action vector: roughly
    ``[move_x, move_y, aim_x, aim_y, fire]``. We do not depend on the exact channel
    meaning — only that it is FIXED and identical across runs; the determinism claim
    is about the SIMULATION being a pure function of (config, start, actions).
    """
    repertoire = [
        [0.0, 1.0, 0.0, 1.0, 0.0],  # forward, aim up, no fire
        [0.0, 1.0, 1.0, 0.0, 0.0],  # forward, rotate aim to +x
        [1.0, 0.0, 1.0, 0.0, 1.0],  # strafe +x, fire
        [-1.0, 0.0, -1.0, 0.0, 1.0],  # strafe -x, fire (opposite)
        [0.0, -1.0, 0.0, -1.0, 0.0],  # reverse, aim down (provoke collisions)
        [1.0, 1.0, 1.0, 1.0, 1.0],  # diagonal + fire (saturate all channels)
    ]
    return [list(repertoire[i % len(repertoire)]) for i in range(n_steps)]


def _connect(game_ip, game_port, my_port, attempts, sock_timeout):
    """Bind a client socket and retry-connect while the build boots to AcceptTcpClient.

    Mirrors the env / play_local connect loop. SO_REUSEADDR is set so a fresh launch
    is not blocked by a lingering TIME_WAIT on a recently used port.
    """
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
                time.sleep(1)
        if not connected:
            raise ConnectionError(
                f"could not connect to Unity on {game_ip}:{game_port} after {attempts} attempts"
            )
    except BaseException:
        sock.close()
        raise
    sock.settimeout(sock_timeout)
    return sock


def _reap(proc):
    """Terminate + reap a launched build so it never orphans (graceful -> kill)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait()


def _run_one(
    game_path,
    config_path,
    n_steps,
    *,
    game_port,
    game_ip="127.0.0.1",
    connect_attempts=60,
    sock_timeout=15.0,
    log_dir=None,
):
    """Launch the build once, drive the canned actions, return (trajectory, winners, flip, elapsed).

    Performs the byte-identical handshake (``restart`` ack, ``start`` -> ``starting``
    ack, first state), records the first state's flip (sign of P1 x), then drives
    :func:`_canned_actions` on BOTH slots, capturing every inbound 52-float state.
    Stops early if the game reports a winner / done. Always terminates + reaps the
    build and closes the socket before returning (no orphan, port freed for the next
    launch).

    Returns ``(trajectory, winners, flip, elapsed)`` where ``trajectory`` is
    ``list[list[float]]`` (one 52-float state per recorded step, INCLUDING the first),
    ``winners`` is the per-step winner signal (``None`` where no winner key arrived;
    aligned with ``trajectory`` indices for the action steps, with a leading ``None``
    for the first state), ``flip`` is ``True``/``False`` for the detected side-flip, and
    ``elapsed`` is the wall-clock seconds spent DRIVING the canned actions (from just
    after the first state to the last recorded step), measured with
    :func:`time.perf_counter`. The drive loop is the only window where ``timeScale``
    governs how fast the simulation advances, so this is the duration whose t1/t5 ratio
    confirms ``timeScale=5`` actually took effect — it is a RECORDED measurement, not a
    determinism INPUT, so it never enters the trajectory diff.
    """
    my_port = game_port + 1
    cmd = [str(game_path), str(game_port), "--config", str(config_path)]

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"build_{game_port}.log"
        log_file = open(log_path, "w")  # noqa: SIM115 (lives for the subprocess's stdio)

    proc = subprocess.Popen(
        cmd,
        stdout=log_file if log_file else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
    )
    sock = None
    try:
        sock = _connect(game_ip, game_port, my_port, connect_attempts, sock_timeout)
        conn = Connection(sock)

        # Handshake (byte-identical to env.reset / play_local).
        conn.send({"restart": True})
        conn.receive()  # restart ack
        conn.send({"start": True})
        ack = conn.receive()
        if "starting" not in ack:
            raise RuntimeError(f"unexpected start ack: {ack!r}")

        first = conn.receive()
        first_state = [float(v) for v in first["state"]]
        flip = first_state[P1_X_INDEX] < 0.0

        trajectory = [first_state]
        winners = [int(first["winner"]) if "winner" in first else None]
        game_over = ("winner" in first) or ("done" in first)

        actions = _canned_actions(n_steps)
        # Time ONLY the drive loop: the window where timeScale governs sim advance.
        drive_start = time.perf_counter()
        for action in actions:
            # Stop driving once the game has ended (winner reported / done flag).
            if game_over:
                break
            conn.send({1: list(action), 2: list(action)})
            received = conn.receive()
            if "state" not in received:
                # A bare control / terminal message with no state — stop recording.
                if "winner" in received:
                    winners.append(int(received["winner"]))
                break
            trajectory.append([float(v) for v in received["state"]])
            winners.append(int(received["winner"]) if "winner" in received else None)
            if "winner" in received or "done" in received:
                break
        elapsed = time.perf_counter() - drive_start

        # Best-effort clean teardown so the build quits itself (then _reap guarantees it).
        try:
            conn.send({"restart": True})
            conn.receive()
            conn.send({"end": True})
            conn.receive()
        except (ConnectionError, OSError, KeyError, ValueError):
            pass

        return trajectory, winners, flip, elapsed
    finally:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        _reap(proc)
        if log_file is not None:
            log_file.close()


def _run_aligned(
    game_path,
    config_path,
    n_steps,
    *,
    base_port,
    launch_index,
    target_flip=None,
    align_retries=DEFAULT_ALIGN_RETRIES,
    log_dir=None,
    label="",
):
    """Run the build, relaunching (fresh port each time) until the flip matches target.

    When ``target_flip is None`` the FIRST successful run sets the target (this is run
    A — it picks the side everyone aligns to). Otherwise the run is relaunched up to
    ``align_retries`` times until its first-state flip equals ``target_flip``; if it
    never aligns it raises ``RuntimeError`` (clear failure, not a silent skew).

    A FRESH port (``base_port + launch_index*1000 + attempt``) is used per launch so a
    lingering TIME_WAIT never collides with the next attempt.

    Returns ``(trajectory, winners, flip, target, elapsed)`` — ``elapsed`` is the
    drive-loop wall-clock of the ALIGNED run (the one whose trajectory we keep), so the
    recorded duration matches the recorded trajectory.
    """
    attempts = 1 if target_flip is None else align_retries
    last = None
    for attempt in range(attempts):
        port = base_port + launch_index * 1000 + attempt
        traj, winners, flip, elapsed = _run_one(
            game_path,
            config_path,
            n_steps,
            game_port=port,
            log_dir=log_dir,
        )
        last = (traj, winners, flip)
        if target_flip is None:
            print(
                f"  [{label}] launch port={port} flip={flip} (set as target) drive={elapsed:.3f}s",
                flush=True,
            )
            return traj, winners, flip, flip, elapsed
        print(
            f"  [{label}] launch port={port} flip={flip} "
            f"(target={target_flip}, attempt {attempt + 1}/{attempts}) drive={elapsed:.3f}s",
            flush=True,
        )
        if flip == target_flip:
            return traj, winners, flip, target_flip, elapsed
    # Exhausted retries without aligning.
    _traj, _winners, flip = last
    raise RuntimeError(
        f"[{label}] could not align flip to target={target_flip} in {attempts} launches "
        f"(last flip={flip}); the side-flip coin never landed on the target — re-run."
    )


def run_harness(
    game_path,
    *,
    n_steps=120,
    base_port=51000,
    align_retries=DEFAULT_ALIGN_RETRIES,
    log_dir=None,
):
    """Run the three trajectories (t1 target, t1 twin, t5) and diff them.

    Returns the assembled result dict (also written to ``runs/determinism/result.json``).
    The pure diff is imported lazily so the module imports without numpy on a box that
    only wants the canned-action / launch helpers.
    """
    from tank_twin.determinism_check import diff_trajectories, motion_summary

    game_path = Path(game_path).resolve()
    if not game_path.is_file():
        raise FileNotFoundError(f"build not found: {game_path}")
    for cfg in (CONFIG_T1, CONFIG_T5):
        if not cfg.is_file():
            raise FileNotFoundError(f"determinism config missing: {cfg}")

    print(f"[determinism] game={game_path}", flush=True)
    print(f"[determinism] steps={n_steps} base_port={base_port}", flush=True)

    # Run A (t1) picks the target flip.
    print("[determinism] run t1 (baseline, sets target flip):", flush=True)
    t1_traj, t1_winners, t1_flip, target, t1_elapsed = _run_aligned(
        game_path,
        CONFIG_T1,
        n_steps,
        base_port=base_port,
        launch_index=0,
        target_flip=None,
        align_retries=align_retries,
        log_dir=log_dir,
        label="t1",
    )

    # Run A' (t1 twin) aligned to target -> the BASELINE floor (same timeScale).
    print("[determinism] run t1-again (baseline twin, aligned):", flush=True)
    t1b_traj, t1b_winners, _t1b_flip, _, t1b_elapsed = _run_aligned(
        game_path,
        CONFIG_T1,
        n_steps,
        base_port=base_port,
        launch_index=1,
        target_flip=target,
        align_retries=align_retries,
        log_dir=log_dir,
        label="t1b",
    )

    # Run B (t5) aligned to target -> the TEST (timeScale 5 vs 1).
    print("[determinism] run t5 (test, aligned):", flush=True)
    t5_traj, t5_winners, _t5_flip, _, t5_elapsed = _run_aligned(
        game_path,
        CONFIG_T5,
        n_steps,
        base_port=base_port,
        launch_index=2,
        target_flip=target,
        align_retries=align_retries,
        log_dir=log_dir,
        label="t5",
    )

    baseline = diff_trajectories(t1_traj, t1b_traj, t1_winners, t1b_winners)
    test = diff_trajectories(t1_traj, t5_traj, t1_winners, t5_winners)

    # Non-degeneracy evidence over the t1 REFERENCE trajectory: proves the 0.0 diff
    # means "identical DESPITE real motion," not "identical because nothing moved."
    motion = motion_summary(t1_traj)

    # Speed evidence: t1/t5 drive-loop wall-clock ratio. timeScale=5 should make the
    # t5 drive ~5x faster than t1; a ratio near 1.0 would mean timeScale was silently
    # ignored even though the trajectories matched. Guard the divide.
    ratio = (t1_elapsed / t5_elapsed) if t5_elapsed > 0 else None

    result = {
        "config": {
            "game_path": str(game_path),
            "config_t1": str(CONFIG_T1),
            "config_t5": str(CONFIG_T5),
            "n_steps_requested": n_steps,
            "base_port": base_port,
            "target_flip": bool(target),
        },
        "trajectory_lengths": {
            "t1": len(t1_traj),
            "t1_again": len(t1b_traj),
            "t5": len(t5_traj),
        },
        "motion_t1_reference": motion.to_dict(),
        "wall_clock": {
            "t1_drive_seconds": float(t1_elapsed),
            "t1_again_drive_seconds": float(t1b_elapsed),
            "t5_drive_seconds": float(t5_elapsed),
            "t1_over_t5_ratio": (None if ratio is None else float(ratio)),
            "expected_ratio_approx": float(_expected_speed_ratio()),
            "note": (
                "drive-loop wall-clock per run; t1_over_t5_ratio ~= timeScale_t5/timeScale_t1 "
                "confirms timeScale=5 took effect (not silently ignored). Recorded only; "
                "durations are NOT inputs to the trajectory diff."
            ),
        },
        "baseline_t1_vs_t1": baseline.to_dict(),
        "test_t1_vs_t5": test.to_dict(),
    }
    return result


def _expected_speed_ratio():
    """Expected t1/t5 drive-time ratio = timeScale_t5 / timeScale_t1 (≈5.0).

    Read from the two configs so the recorded expectation tracks the actual configs
    rather than a hard-coded 5.0. Strict JSON parse via stdlib (the configs are our
    own determinism fixtures). Falls back to 5.0 if a timeScale key is missing.
    """
    try:
        with open(CONFIG_T1, encoding="utf-8") as fh:
            ts1 = float(json.load(fh).get("timeScale", 1))
        with open(CONFIG_T5, encoding="utf-8") as fh:
            ts5 = float(json.load(fh).get("timeScale", 5))
        return ts5 / ts1 if ts1 else 5.0
    except (OSError, ValueError, json.JSONDecodeError):
        return 5.0


def _print_summary(result):
    """Print a clean human summary of the measured numbers (no pass/fail verdict)."""
    base = result["baseline_t1_vs_t1"]
    test = result["test_t1_vs_t5"]
    lens = result["trajectory_lengths"]
    motion = result["motion_t1_reference"]
    clock = result["wall_clock"]
    print("", flush=True)
    print("=" * 68, flush=True)
    print("DETERMINISM MEASUREMENT (timeScale 1 vs 5) — no pass bar applied", flush=True)
    print("=" * 68, flush=True)
    print(
        f"  trajectory lengths : t1={lens['t1']} t1_again={lens['t1_again']} t5={lens['t5']}",
        flush=True,
    )
    print(
        "  -- NON-DEGENERACY (t1 reference; did the tanks actually MOVE?) --",
        flush=True,
    )
    print(
        f"     P1 pos range x/y  : {motion['p1_x_range']:.3f} / {motion['p1_y_range']:.3f}",
        flush=True,
    )
    print(
        f"     P2 pos range x/y  : {motion['p2_x_range']:.3f} / {motion['p2_y_range']:.3f}",
        flush=True,
    )
    print(
        f"     max step delta    : P1={motion['p1_max_step_delta']:.3f} "
        f"P2={motion['p2_max_step_delta']:.3f}",
        flush=True,
    )
    print(
        f"     non_degenerate    : {motion['non_degenerate']} "
        f"(floor={motion['motion_floor']} world units)",
        flush=True,
    )
    if not motion["non_degenerate"]:
        print("", flush=True)
        print("  " + "!" * 64, flush=True)
        print("  !! WARNING: trajectory is DEGENERATE (tanks barely moved).", flush=True)
        print("  !! A max-diff of 0.0 over a frozen scene is MEANINGLESS — it does", flush=True)
        print("  !! NOT prove determinism. Investigate before trusting this result.", flush=True)
        print("  " + "!" * 64, flush=True)
        print("", flush=True)
    print("  -- WALL-CLOCK (drive-loop seconds; did timeScale=5 take effect?) --", flush=True)
    ratio = clock["t1_over_t5_ratio"]
    ratio_str = "n/a" if ratio is None else f"{ratio:.2f}x"
    print(
        f"     drive seconds     : t1={clock['t1_drive_seconds']:.3f} "
        f"t1_again={clock['t1_again_drive_seconds']:.3f} t5={clock['t5_drive_seconds']:.3f}",
        flush=True,
    )
    print(
        f"     t1/t5 ratio       : {ratio_str} (expected ~{clock['expected_ratio_approx']:.1f}x)",
        flush=True,
    )
    print("  -- BASELINE (t1 vs t1, same timeScale; the noise floor) --", flush=True)
    print(f"     bitwise identical : {base['bitwise_identical']}", flush=True)
    print(f"     max abs diff      : {base['max_abs_diff']}", flush=True)
    print(f"     first divergence  : {base['first_divergence_step']}", flush=True)
    print(f"     invariants match  : {base['invariants']['all_match']}", flush=True)
    print("  -- TEST (t1 vs t5; THE QUESTION) --", flush=True)
    print(f"     bitwise identical : {test['bitwise_identical']}", flush=True)
    print(f"     max abs diff      : {test['max_abs_diff']}", flush=True)
    print(f"     first divergence  : {test['first_divergence_step']}", flush=True)
    print(f"     per-group max     : {test['per_group_max_diff']}", flush=True)
    inv = test["invariants"]
    print(
        f"     invariants match  : {inv['all_match']} "
        f"(len={inv['same_length']} winner={inv['same_winner']} reward={inv['same_total_reward']})",
        flush=True,
    )
    print(f"     winners           : t1={inv['winner_a']} t5={inv['winner_b']}", flush=True)
    print("=" * 68, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tank_twin.determinism_harness",
        description=(
            "Measure whether Unity at timeScale=5 is computationally identical to 1x by "
            "diffing canned-action trajectories. Measures only; the CTO sets the pass bar."
        ),
    )
    parser.add_argument(
        "--game-path",
        default=str(_REPO_ROOT / "build" / "TankTwinStickShooter.exe"),
        help="Path to the standalone build (default: repo build/TankTwinStickShooter.exe).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Canned action steps to drive per run (default 120).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=51000,
        help="Base TCP port; each launch uses a fresh derived port (default 51000).",
    )
    parser.add_argument(
        "--align-retries",
        type=int,
        default=DEFAULT_ALIGN_RETRIES,
        help="Max relaunches to align the side-flip to the target (default 12).",
    )
    parser.add_argument(
        "--result-path",
        default=str(RESULT_PATH),
        help="Where to write the strict-JSON result (default runs/determinism/result.json).",
    )
    args = parser.parse_args(argv)

    log_dir = _REPO_ROOT / "runs" / "determinism" / "logs"
    try:
        result = run_harness(
            args.game_path,
            n_steps=args.steps,
            base_port=args.base_port,
            align_retries=args.align_retries,
            log_dir=log_dir,
        )
    except (FileNotFoundError, RuntimeError, ConnectionError) as exc:
        print(f"[determinism] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[determinism] wrote {result_path}", flush=True)

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
