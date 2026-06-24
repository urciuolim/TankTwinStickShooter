"""Runnable demo: drive two visibly-different agents through the live Tank game.

``python -m pop_trainer.demo`` launches the real Unity build WINDOWED, opens a TCP socket to
it, wraps that socket in :class:`pop_trainer.core.protocol.Connection`, constructs a
:class:`pop_trainer.env.tank_env.TankEnv` over it, and runs ONE episode with two different
policies so a human can watch the difference. BOTH policies are driven by this module: player1 by
``agent1.act(view1)`` (``view1`` is player1's own UNFLIPPED 52-float state) and player2 by
``agent2.act(view2)`` (``view2`` is player2's FLIPPED first-person view, computed here via
:func:`core.state.split_state_for_opponent`); both actions are passed to ``env.step(a1, a2)`` —
the env is a bare pure transport that owns neither player.

This is an APPLICATION / entry point. It imports ``pop_trainer.env``, ``pop_trainer.agents``,
and ``pop_trainer.core`` ONLY — never ``models`` / ``data`` / ``rl`` / ``pretraining`` and
nothing from ``tank_twin``. The episode-driving logic is the pure :func:`run_demo_episode`
(an already-built env + agent in, a :class:`DemoResult` out); the subprocess launch and socket
connect live in :func:`main`, which the tests do NOT exercise.

PIXELS MUST BE ON. ``TankEnv.reset``/``step`` always read a length-prefixed pixel frame after
each state JSON, but the shipped ``Assets/StreamingAssets/config.json`` has no ``obs_pixels``
key, so the build defaults ``obsPixels=false`` and sends NO frame — the env would block waiting
for bytes that never arrive. The demo therefore launches the build with
``Assets/StreamingAssets/demo_config.json``, which sets ``"obs_pixels": true`` plus
``obs_pixels_width``/``obs_pixels_height`` (640x360, the DriverController defaults and 16:9),
and constructs the env with ``frame_shape=(360, 640, 3)`` to MATCH. That config lives inside
StreamingAssets next to the ``Arenas/`` directory so its relative ``arena_path``
(``Arenas/custom1.json``) resolves against the config directory — exactly how
``DriverController.ResolveArenaPath`` resolves it. It also drops ``timeScale`` from the shipped
20 to a human-watchable 2.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pop_trainer import agents
from pop_trainer.core import agent as core_agent
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig
from pop_trainer.core.launch import build_launch_cmd, connect
from pop_trainer.core.protocol import Connection, WallLayout
from pop_trainer.env.tank_env import TankEnv

# --- launch / wire defaults -------------------------------------------------------------

# The repo root is three parents up from this file (src/pop_trainer/demo.py -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = _REPO_ROOT / "build" / "TankTwinStickShooter.exe"
# The demo config enables obs_pixels and lives in StreamingAssets so its relative arena_path
# resolves against the config directory (DriverController.ResolveArenaPath).
DEFAULT_CONFIG = _REPO_ROOT / "Assets" / "StreamingAssets" / "demo_config.json"
DEFAULT_PORT = 50000

# Rendered pixel-frame dimensions (W x H) — MUST match demo_config.json's obs_pixels_*.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# TankEnv frame_shape is (H, W, 3).
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)

DEFAULT_MAX_STEPS = 600
DEFAULT_SEED = 0


# --- agent selectors --------------------------------------------------------------------
# Selector name -> a zero-arg-friendly factory taking an optional seed. The default pairing is
# player1 = aggressive-coverage (sweeps the arena, aim-sweeps + fires) vs player2 =
# opponent-shadower (the same coverage movement but the aim layer tracks player1 and fires):
# two visibly different policies. ``random`` is the map-agnostic baseline.
AGENT_SELECTORS: dict[str, Callable[[int | None], core_agent.Agent]] = {
    "aggressive-coverage": lambda seed: agents.CoverageAgent.aggressive(seed=seed),
    "wall-hugger": lambda seed: agents.CoverageAgent.wall_hugger(seed=seed),
    "opponent-shadower": lambda seed: agents.CoverageAgent.opponent_shadower(seed=seed),
    "random": lambda seed: agents.RandomAgent(seed=seed),
}

DEFAULT_PLAYER1 = "aggressive-coverage"
DEFAULT_PLAYER2 = "opponent-shadower"


def make_agent(selector: str, *, seed: int | None = None) -> core_agent.Agent:
    """Build the agent named by ``selector`` (see :data:`AGENT_SELECTORS`).

    Raises ``ValueError`` on an unknown selector, listing the valid names.
    """
    factory = AGENT_SELECTORS.get(selector)
    if factory is None:
        valid = ", ".join(sorted(AGENT_SELECTORS))
        raise ValueError(f"unknown agent selector {selector!r}; choose one of: {valid}")
    return factory(seed)


# --- pure episode loop (unit-tested against a fake transport) ----------------------------


@dataclass
class DemoResult:
    """The outcome of one demo episode (everything the trace / a test needs).

    * ``steps`` — number of ``env.step`` calls made.
    * ``terminated`` / ``truncated`` — the gymnasium boundary that ended the episode.
    * ``winner`` — the reported winner int (``0`` == player1, ``1`` == player2, ``-1`` ==
      draw) on a decided terminal, else ``None``.
    * ``outcome`` — the env's free-form result tag (``"win"`` / ``"loss"`` / ``"draw"``) from
      player1's perspective on a decided terminal, else ``None``.
    * ``total_reward`` — the summed player1 step reward.
    * ``map`` — the tracked :class:`pop_trainer.core.protocol.WallLayout` (``info["map"]``), or
      ``None`` when no arena/walls were configured.
    """

    steps: int
    terminated: bool
    truncated: bool
    winner: int | None
    outcome: str | None
    total_reward: float
    map: WallLayout | None


def run_demo_episode(
    env: TankEnv,
    agent1: core_agent.Agent,
    agent2: core_agent.Agent,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> DemoResult:
    """Drive ONE episode of an already-built bare ``env`` with ``agent1`` / ``agent2``.

    ``env`` is a bare pure-transport :class:`TankEnv` that owns neither player; this loop drives
    BOTH agents. It resets ``agent1`` / ``agent2`` if they expose ``reset``, calls ``env.reset()``,
    then steps: each step it computes player1's action from the CURRENT 52-float state
    (``info["state"]`` — player1's own unflipped view) via ``agent1.act`` and player2's action from
    its FLIPPED first-person view (``split_state_for_opponent`` of that state) via ``agent2.act``,
    then feeds both to ``env.step(a1, a2)`` until ``terminated or truncated`` or ``max_steps`` is
    reached. The transport lives inside ``env``, so this is unit-testable against a fake-backed
    ``TankEnv`` with no subprocess and no live socket.
    """
    _maybe_reset(agent1)
    _maybe_reset(agent2)

    obs, info = env.reset()
    del obs  # the observation is the pixel frame; this loop drives off info["state"].
    state_vec = info["state"]
    tracked_map = info.get("map")

    # Hand the static layout to a map-aware agent (the OPTIONAL ``set_map`` hook); both agents are
    # driver-side. A map-agnostic agent does not expose ``set_map``.
    if tracked_map is not None:
        for agent in (agent1, agent2):
            set_map = getattr(agent, "set_map", None)
            if callable(set_map):
                set_map(tracked_map)

    steps = 0
    total_reward = 0.0
    terminated = truncated = False
    winner: int | None = None
    outcome: str | None = None

    while steps < max_steps:
        action = agent1.act(state_vec)
        a2 = agent2.act(S.split_state_for_opponent(np.asarray(state_vec)))
        _obs, reward, terminated, truncated, info = env.step(action, a2)
        steps += 1
        total_reward += float(reward)
        if "map" in info and info["map"] is not None:
            tracked_map = info["map"]
        if "state" in info:
            state_vec = info["state"]
        if terminated or truncated:
            winner = info.get("winner")
            outcome = info.get("outcome")
            break

    return DemoResult(
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        winner=winner,
        outcome=outcome,
        total_reward=total_reward,
        map=tracked_map,
    )


def _maybe_reset(agent: object) -> None:
    """Call ``agent.reset()`` if the (optional) method exists — ``core.Agent`` requires only
    ``act``; seeded / stateful agents implement ``reset`` for a deterministic replay."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


# --- live launch + socket connect (NOT unit-tested; the shared seam lives in core.launch) -


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.demo",
        description="Watch two different agents play the Tank Twin Stick Shooter through TankEnv.",
    )
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="path to the Unity build")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port for the build")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="config JSON for the build (MUST enable obs_pixels at 640x360)",
    )
    parser.add_argument(
        "--player1",
        default=DEFAULT_PLAYER1,
        choices=sorted(AGENT_SELECTORS),
        help="agent selector for player1 (driven by this module)",
    )
    parser.add_argument(
        "--player2",
        default=DEFAULT_PLAYER2,
        choices=sorted(AGENT_SELECTORS),
        help="agent selector for player2 (driven by this module)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def _print_trace(result: DemoResult, *, player1: str, player2: str) -> None:
    """Print a short human-readable summary of the episode."""
    print(f"player1={player1}  vs  player2={player2}")
    print(f"steps:        {result.steps}")
    print(f"terminated:   {result.terminated}")
    print(f"truncated:    {result.truncated}")
    print(f"winner:       {result.winner}")
    print(f"outcome (p1): {result.outcome}")
    print(f"total reward: {result.total_reward:.4f}")
    if result.map is None:
        print("map:          (no walls tracked)")
    else:
        m = result.map
        print(f"map:          {m.map_id} ({len(m.occupied)} wall cells)")


def main(argv: list[str] | None = None) -> int:
    """Launch the build, connect, run one episode, print the trace, tear down cleanly."""
    args = _parse_args(argv)

    if not args.exe.exists():
        print(f"error: build not found at {args.exe}", file=sys.stderr)
        return 2
    if not args.config.exists():
        print(f"error: config not found at {args.config}", file=sys.stderr)
        return 2

    agent1 = make_agent(args.player1, seed=args.seed)
    agent2 = make_agent(args.player2, seed=args.seed)

    cmd = build_launch_cmd(args.exe, args.port, args.config)
    print("launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)  # noqa: S603 (arg-list, trusted local build path)

    env: TankEnv | None = None
    try:
        connection = Connection(connect(args.port))
        env = TankEnv(
            connection=connection,
            frame_shape=FRAME_SHAPE,
            env_config=EnvConfig(max_steps=args.max_steps),
            seed=args.seed,
        )
        result = run_demo_episode(env, agent1, agent2, max_steps=args.max_steps)
        _print_trace(result, player1=args.player1, player2=args.player2)
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        _terminate(proc)
    return 0


def _terminate(proc: subprocess.Popen) -> None:
    """Reap the build subprocess, escalating to ``kill`` if it does not exit promptly."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
