"""Collection entry point: concrete factories + a CLI that drives :mod:`collect.collect_parallel`.

This is the APPLICATION layer over :mod:`pop_trainer.data.collect`. ``collect`` provides the
PURE step loop + the spawn-safe orchestration but leaves the env/agent construction to injected
factories; THIS module supplies the concrete factories that launch the live Unity build, connect
a socket, and build the bare gymnasium env + the driver-side player1/player2 agents. It is a
data-layer module, so it may import ``core`` (the launch + wire + config seams), ``env``
(``TankEnv``), and ``agents`` (the coverage / random presets) — exactly the
``core <- {env, agents} <- data`` direction. No ``models`` / ``pretraining`` / ``rl`` and nothing
from ``tank_twin``.

SPAWN SAFETY. :func:`env_factory` / :func:`player1_factory` / :func:`player2_factory` are
MODULE-LEVEL plain functions (not closures / lambdas) so ``multiprocessing`` with the ``spawn``
start method can re-import them by qualified name in a fresh interpreter. Everything they need
crosses the process boundary as plain data inside ``spec.extra`` (paths, ports, selector names,
seeds) — never a live socket / env / agent. Each worker computes its OWN port
(``base_port + worker_id``), launches its OWN build, and opens its OWN socket; nothing is
inherited from the parent.

LAUNCH-PROC REAPING. :func:`env_factory` stashes the live ``subprocess.Popen`` of the launched
build on the env (``env._launch_proc``) and WRAPS ``env.close`` so that closing the env (which
``collect.run_worker`` does in its ``finally``) first runs the env's own end-handshake/transport
release and THEN terminates the build process (terminate -> wait(10) -> kill on timeout). The
``TankEnv`` seam itself is untouched — this is caller-side wrapping only.

PIXELS. Collection MUST receive pixel frames (the env reads a length-prefixed frame after every
state; a build without ``obs_pixels`` would leave the env blocking on bytes that never arrive).
The default ``--map`` resolves to ``Assets/StreamingAssets/demo_config.json`` — the SAME
obs_pixels-enabled config the demo launches with (640x360, arena ``Arenas/custom1.json`` resolved
against the config dir) — so the captured ``(frame, state)`` rows are byte-for-byte the demo's /
RL's observation pipeline, with no drift. The env is built with the matching ``frame_shape``.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from pop_trainer import agents
from pop_trainer.core import agent as core_agent
from pop_trainer.core import launch
from pop_trainer.core.config import EnvConfig
from pop_trainer.core.protocol import Connection
from pop_trainer.data.collect import CollectionSpec, collect_parallel
from pop_trainer.env.tank_env import TankEnv

__all__ = [
    "AGENT_SELECTORS",
    "DEFAULT_PLAYER1",
    "DEFAULT_PLAYER2",
    "MAX_WORKERS",
    "DEFAULT_MAP",
    "MAP_CONFIGS",
    "FRAME_SHAPE",
    "make_agent",
    "build_env_from_connection",
    "env_factory",
    "player1_factory",
    "player2_factory",
    "build_specs",
    "main",
]

# The repo root is three parents up from this file (src/pop_trainer/data/collect_runner.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXE = _REPO_ROOT / "build" / "TankTwinStickShooter.exe"

# Rendered pixel-frame dimensions (W x H) — MUST match the launched config's obs_pixels_*.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# TankEnv frame_shape is (H, W, 3).
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)

# Map name -> the obs_pixels-enabled build config it launches with. Phase A ships a single map.
# Reusing the demo config keeps collection's frames byte-identical to the demo's / RL's pipeline
# (640x360 pixels on, arena Arenas/custom1.json resolved against the config dir).
_STREAMING_ASSETS = _REPO_ROOT / "Assets" / "StreamingAssets"
MAP_CONFIGS: dict[str, Path] = {
    "custom1": _STREAMING_ASSETS / "demo_config.json",
}
DEFAULT_MAP = "custom1"
# Map name -> the integer map_id tagged onto every captured sample for that map.
_MAP_IDS: dict[str, int] = {
    "custom1": 0,
}

DEFAULT_MAX_STEPS = 800
DEFAULT_BASE_PORT = 50000
DEFAULT_WORKERS = 2
# Hard cap on parallel workers (one live build + socket per worker). Above this the ports / build
# instances on one box stop being reasonable; requests over it are clamped.
MAX_WORKERS = 8
# Distinct-seed spacing so two workers never share a seed (and per-episode seeds inside a worker,
# which are seed + episode_id, never collide across workers either).
_SEED_STRIDE = 10_000


# --- agent selectors (the coverage/random presets; mirrors demo, built from agents directly) ---
# Built straight from ``pop_trainer.agents`` — NOT imported from ``demo`` (demo is an env-app;
# data reaching into it would be a cross-app boundary smell). The surface is intentionally the
# same coverage family + random baseline the demo exposes.
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


# --- the env-build core (pure given a connection; unit-testable with a fake transport) -------


def build_env_from_connection(
    connection: Connection,
    *,
    seed: int | None,
    max_steps: int,
    frame_shape: tuple[int, int, int] = FRAME_SHAPE,
) -> TankEnv:
    """Build a bare pure-transport :class:`TankEnv` over an ALREADY-CONNECTED ``connection``.

    The pure heart of :func:`env_factory`: it constructs the bare env around the injected
    ``connection`` (no socket / subprocess here). The env owns neither player — both agents are
    driver-side, built by :func:`player1_factory` / :func:`player2_factory`. Factored out so the
    env-build wiring is testable against an in-process fake transport with no live build.
    ``frame_shape`` fixes the observation space and MUST match the build's rendered frame.
    """
    return TankEnv(
        connection=connection,
        frame_shape=frame_shape,
        env_config=EnvConfig(max_steps=max_steps),
        seed=seed,
    )


# --- module-level, picklable, spawn-safe factories -------------------------------------------


def env_factory(spec: CollectionSpec) -> TankEnv:
    """Launch the build, connect, and build the bare env for one worker.

    MODULE-LEVEL + plain (spawn can import it by qualified name). Reads its launch params from
    ``spec.extra`` (``exe`` / ``config`` / ``base_port`` / ``frame_shape``), computes
    ``port = base_port + spec.worker_id``, launches the build via
    :func:`core.launch.build_launch_cmd` + ``subprocess.Popen``, connects via
    :func:`core.launch.connect`, wraps the socket in :class:`core.protocol.Connection`, and builds
    the bare env via :func:`build_env_from_connection`. Both agents are driver-side (built by
    :func:`player1_factory` / :func:`player2_factory`), so the env carries neither. The live
    ``Popen`` is stashed and the env's ``close`` is wrapped so :func:`collect.run_worker`'s
    ``finally`` reaps the build (see :func:`_attach_launch_proc`). The chosen port is recorded back
    into ``spec.extra["port"]`` so it is inspectable. Raises ``KeyError`` if a required ``extra``
    key is missing.
    """
    extra = spec.extra
    exe = extra["exe"]
    config = extra["config"]
    base_port = int(extra["base_port"])
    frame_shape = tuple(extra.get("frame_shape", FRAME_SHAPE))

    port = base_port + spec.worker_id
    extra["port"] = port  # record the chosen port for inspection / debugging.

    cmd = launch.build_launch_cmd(exe, port, config)
    proc = subprocess.Popen(cmd)  # noqa: S603 (arg-list, trusted local build path)
    try:
        sock = launch.connect(port)
        connection = Connection(sock)
        env = build_env_from_connection(
            connection,
            seed=spec.seed,
            max_steps=spec.max_steps,
            frame_shape=frame_shape,
        )
    except BaseException:
        # The build is already running; if env construction never completes, reap it so a failed
        # worker does not leak a process.
        _terminate(proc)
        raise
    _attach_launch_proc(env, proc)
    return env


def player1_factory(spec: CollectionSpec) -> core_agent.Agent:
    """Build the player1 agent from the selector name + seed carried in ``spec.extra``.

    MODULE-LEVEL + plain (spawn-importable). Reads ``spec.extra["player1"]`` and seeds the agent
    with ``spec.seed`` so a worker's player1 stream is reproducible.
    """
    return make_agent(spec.extra["player1"], seed=spec.seed)


def player2_factory(spec: CollectionSpec) -> core_agent.Agent:
    """Build the player2 agent from the selector name + seed carried in ``spec.extra``.

    MODULE-LEVEL + plain (spawn-importable), symmetric with :func:`player1_factory`: player2 is now
    a driver-side agent (the env owns neither player). Reads ``spec.extra["player2"]`` and seeds
    the agent with ``spec.seed`` so a worker's player2 stream is reproducible.
    """
    return make_agent(spec.extra["player2"], seed=spec.seed)


def _attach_launch_proc(env: TankEnv, proc: subprocess.Popen) -> None:
    """Stash ``proc`` on ``env`` and wrap ``env.close`` so closing the env also reaps the build.

    ``TankEnv.close`` releases only the transport (the socket / fake) — it does NOT know about the
    build subprocess. So we record the proc on ``env._launch_proc`` (inspectable) and replace
    ``env.close`` with a wrapper that runs the original close (end-handshake + transport release)
    and THEN terminates the build. The wrapper is idempotent (a second close is a no-op once the
    proc is reaped). The ``TankEnv`` class is untouched — this is per-instance caller-side wrapping.
    """
    env._launch_proc = proc
    original_close = env.close

    def close_and_reap() -> None:
        with contextlib.suppress(Exception):
            original_close()
        _terminate(proc)

    env.close = close_and_reap  # type: ignore[method-assign]


def _terminate(proc: subprocess.Popen) -> None:
    """Reap the build subprocess, escalating to ``kill`` if it does not exit promptly.

    Idempotent: a no-op once the process has already exited (``poll`` returns non-``None``).
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


# --- spec building (pure: CLI args -> a list of CollectionSpec) -------------------------------


def build_specs(
    *,
    player1: str,
    player2: str,
    map_name: str,
    config: Path,
    exe: Path,
    episodes: int,
    max_steps: int,
    out_dir: str | Path,
    workers: int,
    base_port: int,
    seed: int,
    frame_shape: tuple[int, int, int] = FRAME_SHAPE,
) -> list[CollectionSpec]:
    """Turn collection params into a list of N :class:`CollectionSpec` (one per worker).

    Pure: builds plain spec data, NO live launch. ``workers`` is CLAMPED to ``[1, MAX_WORKERS]``.
    Each worker gets a distinct ``worker_id`` (0..N-1), its own ``out_dir`` (per-worker
    subdirectory so shard files never clash), the single Phase-A map repeated ``episodes`` times in
    ``map_ids`` (one episode per entry — the map's integer id), the shared ``max_steps``, a distinct
    ``seed`` (``base_seed + worker_id * stride``), and an ``extra`` carrying the launch params +
    selector names. The module-level :func:`env_factory` / :func:`player1_factory` /
    :func:`player2_factory` are wired onto every spec. Validates the selector names and the map name
    up front (``ValueError``).
    """
    # Validate the selectors + map eagerly so a bad CLI fails before any launch.
    make_agent(player1, seed=seed)
    make_agent(player2, seed=seed)
    if map_name not in _MAP_IDS:
        valid = ", ".join(sorted(_MAP_IDS))
        raise ValueError(f"unknown map {map_name!r}; choose one of: {valid}")
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    n_workers = max(1, min(int(workers), MAX_WORKERS))
    map_id = _MAP_IDS[map_name]
    out_root = Path(out_dir)

    specs: list[CollectionSpec] = []
    for worker_id in range(n_workers):
        worker_out = out_root / f"worker_{worker_id}"
        spec = CollectionSpec(
            worker_id=worker_id,
            out_dir=str(worker_out),
            map_ids=[map_id] * episodes,  # one episode per entry; single Phase-A map.
            max_steps=max_steps,
            seed=seed + worker_id * _SEED_STRIDE,
            env_factory=env_factory,
            player1_factory=player1_factory,
            player2_factory=player2_factory,
            extra={
                "exe": str(exe),
                "config": str(config),
                "base_port": int(base_port),
                "frame_shape": tuple(frame_shape),
                "player1": player1,
                "player2": player2,
                "map": map_name,
            },
        )
        specs.append(spec)
    return specs


# --- CLI -------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.data.collect_runner",
        description="Collect (frame, state, action) samples by driving live Unity builds.",
    )
    parser.add_argument(
        "--player1",
        default=DEFAULT_PLAYER1,
        choices=sorted(AGENT_SELECTORS),
        help="agent selector for player1 (driven by the collection loop)",
    )
    parser.add_argument(
        "--player2",
        default=DEFAULT_PLAYER2,
        choices=sorted(AGENT_SELECTORS),
        help="agent selector for player2 (driven by the collection loop)",
    )
    parser.add_argument(
        "--map",
        default=DEFAULT_MAP,
        choices=sorted(MAP_CONFIGS),
        help="map to collect on (resolves to an obs_pixels-enabled build config)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes PER worker")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="per-episode step cap (the live coverage finding)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="root output directory (each worker writes shards to a worker_<id> subdir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel workers (one build per worker; clamped to [1, {MAX_WORKERS}])",
    )
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="path to the Unity build")
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_BASE_PORT,
        help="base TCP port; worker w uses base_port + w",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="base seed; worker w uses base + w*stride"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the specs from the CLI and drive :func:`collect.collect_parallel`; print a summary.

    Resolves ``--map`` to its obs_pixels-enabled config, validates the build / config exist, builds
    the per-worker specs, runs collection in parallel (spawn), and prints a short per-worker result
    line. Returns ``0`` on success, ``2`` on a missing build / config.
    """
    args = _parse_args(argv)
    config = MAP_CONFIGS[args.map]

    if not args.exe.exists():
        print(f"error: build not found at {args.exe}", file=sys.stderr)
        return 2
    if not config.exists():
        print(f"error: config not found at {config}", file=sys.stderr)
        return 2

    requested = args.workers
    specs = build_specs(
        player1=args.player1,
        player2=args.player2,
        map_name=args.map,
        config=config,
        exe=args.exe,
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        workers=args.workers,
        base_port=args.base_port,
        seed=args.seed,
    )
    if requested > len(specs):
        print(f"note: clamped --workers {requested} to {len(specs)} (MAX_WORKERS={MAX_WORKERS})")

    print(
        f"collecting: player1={args.player1} vs player2={args.player2} on map={args.map} "
        f"({len(specs)} worker(s), {args.episodes} episode(s) each, max_steps={args.max_steps})"
    )
    results = collect_parallel(specs)
    for result in results:
        # The port a worker used is base_port + worker_id (env_factory records it into the
        # worker's own spec.extra, but under spawn that mutation stays in the child; recompute it
        # here so the parent's summary is correct without relying on the child's mutation).
        port = args.base_port + result["worker_id"]
        print(
            f"worker {result['worker_id']}: port={port} "
            f"shards={result['num_shards']} files={result['shards']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
