"""Collection entry point: concrete factories + a CLI that drives :mod:`collect.collect_parallel`.

This is the APPLICATION layer over :mod:`pop_trainer.data.collect`. ``collect`` provides the
PURE step loop + the spawn-safe orchestration but leaves the env/agent construction to injected
factories; THIS module supplies the concrete factories that launch the live Unity build, connect
a socket, and build the bare gymnasium env + the driver-side agent pool. It is a data-layer
module, so it may import ``core`` (the launch + wire + config + map-rotation seams), ``env``
(``TankEnv``), and ``agents`` (the coverage / random presets) — exactly the
``core <- {env, agents} <- data`` direction. No ``models`` / ``pretraining`` / ``rl`` and nothing
from ``tank_twin``.

ROUND-ROBIN (map x pairing). A run is a deterministic round-robin over the grid of the ``--maps``
rotation set crossed with the ``--pairings`` set. :func:`round_robin_plan` (PURE, unit-tested with
no live build) computes each WORKER's slice of the grid: worker ``w`` episode ``i`` picks grid cell
``(w + i * n_workers) % G`` (``G`` = maps x pairings), which spreads the grid across workers so the
union covers it evenly and no two workers run identical episodes at the same step. The build is
launched ONCE per worker on a boot config and rotates between arenas via ``switch_arena`` between
episodes (the long-lived-build model; 8 long-lived builds is the end state).

TAG-FROM-ECHO (F5). The on-disk ``map_ids`` stay int32 (the group-split key in ``readers.py``); the
int is a STABLE index into the rotation set's ``maps`` list, resolved from the ECHOED arena path
Unity actually loaded (``info["map"].map_id``) — not the runner's intent — via ``map_index`` (see
:func:`collect.resolve_map_tag`). The int -> arena-path mapping is persisted to a ``maps.json``
sidecar (:func:`write_maps_sidecar`) next to the shards so the ints are reversible.

SPAWN SAFETY. :func:`env_factory` / :func:`agent_pool_factory` are MODULE-LEVEL plain functions
(not closures / lambdas) so ``multiprocessing`` with the ``spawn`` start method can re-import them
by qualified name in a fresh interpreter. Everything they need crosses the process boundary as
plain data inside ``spec`` (the ``episode_plan`` of :class:`collect.EpisodePlan`, the ``map_index``,
and ``extra`` with paths / ports / selector names / seeds) — never a live socket / env / agent.
Each worker computes its OWN port (``base_port + worker_id``), launches its OWN build, and opens
its OWN socket; nothing is inherited from the parent.

LAUNCH-PROC REAPING. :func:`env_factory` stashes the live ``subprocess.Popen`` of the launched
build on the env (``env._launch_proc``) and WRAPS ``env.close`` so that closing the env (which
``collect.run_worker`` does in its ``finally``) first runs the env's own end-handshake/transport
release and THEN terminates the build process (terminate -> wait(10) -> kill on timeout). The
``TankEnv`` seam itself is untouched — this is caller-side wrapping only.

PIXELS. Collection MUST receive pixel frames (the env reads a length-prefixed frame after every
state; a build without ``obs_pixels`` would leave the env blocking on bytes that never arrive). The
no-rotation default ``--map`` resolves to ``unity/Assets/StreamingAssets/demo_config.json`` — the
SAME obs_pixels-enabled config the demo launches with (640x360, arena ``Arenas/custom1.json``
resolved against the config dir) — so the captured ``(frame, state)`` rows are byte-for-byte the
demo's / RL's observation pipeline, with no drift. For a rotation run the boot config is the FIRST
rotation map config (it must likewise enable ``obs_pixels``). The env is built with the matching
``frame_shape`` either way.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from pop_trainer.agents import AGENT_SELECTORS, make_agent
from pop_trainer.core import agent as core_agent
from pop_trainer.core import launch
from pop_trainer.core import maps as core_maps
from pop_trainer.core.config import EnvConfig
from pop_trainer.core.protocol import Connection
from pop_trainer.data import collect, schema
from pop_trainer.data.collect import CollectionSpec, EpisodePlan, collect_parallel
from pop_trainer.env.tank_env import TankEnv

__all__ = [
    "AGENT_SELECTORS",
    "DEFAULT_PLAYER1",
    "DEFAULT_PLAYER2",
    "DEFAULT_PAIRINGS",
    "MAX_WORKERS",
    "DEFAULT_MAP",
    "MAP_CONFIGS",
    "MAPS_SIDECAR_NAME",
    "FRAME_SHAPE",
    "frame_nbytes",
    "make_agent",
    "arena_path_for_config",
    "build_rotation",
    "build_map_index",
    "round_robin_plan",
    "parse_pairing",
    "write_maps_sidecar",
    "read_maps_sidecar",
    "build_env_from_connection",
    "agent_pool_factory",
    "env_factory",
    "build_specs",
    "CollectionSummary",
    "summarize_collection",
    "main",
]

# The repo root is three parents up from this file (src/pop_trainer/data/collect_runner.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
# OS-aware default to the launchable build binary (.exe / .app inner / bare); --exe overrides it.
DEFAULT_EXE = launch.default_build_path(_REPO_ROOT / "unity")

# Rendered pixel-frame dimensions (W x H) — MUST match the launched config's obs_pixels_*.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# TankEnv frame_shape is (H, W, 3).
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)


def frame_nbytes(frame_shape: Sequence[int]) -> int:
    """The uncompressed byte size of ONE uint8 frame of ``frame_shape`` (the in-RAM cost/sample).

    uint8 is 1 byte/element, so this is just the element product. It is the per-sample buffer cost
    the memory guard + the byte-budget shard-size reason about (see :mod:`collect`).
    """
    return int(np.prod(frame_shape))


# Map name -> the obs_pixels-enabled build config it launches with (the single-map / no-rotation
# default). Reusing the demo config keeps collection's frames byte-identical to the demo's / RL's
# pipeline (640x360 pixels on, arena Arenas/custom1.json resolved against the config dir).
_STREAMING_ASSETS = _REPO_ROOT / "unity" / "Assets" / "StreamingAssets"
MAP_CONFIGS: dict[str, Path] = {
    "custom1": _STREAMING_ASSETS / "demo_config.json",
}
DEFAULT_MAP = "custom1"
# The single-map arena path Unity echoes for the default --map (its config's arena_path).
_DEFAULT_MAP_ARENA = "Arenas/custom1.json"

# The maps-list sidecar: the int -> arena-path mapping written next to the shards so the on-disk
# int32 ``map_ids`` are decodable back to the arena Unity loaded (the F5 persistence requirement).
MAPS_SIDECAR_NAME = "maps.json"

DEFAULT_MAX_STEPS = 800
DEFAULT_BASE_PORT = 50000
DEFAULT_WORKERS = 2
# Hard cap on parallel workers (one live build + socket per worker). Above this the ports / build
# instances on one box stop being reasonable; requests over it are clamped.
MAX_WORKERS = 8
# Distinct-seed spacing so two workers never share a seed (and per-episode seeds inside a worker,
# which are seed + episode_id, never collide across workers either).
_SEED_STRIDE = 10_000


# --- agent selectors ---
# The string -> agent registry (``AGENT_SELECTORS`` + ``make_agent``) is owned by
# ``pop_trainer.agents`` (the single source of truth) and imported above; collection re-exports
# the imported names in ``__all__`` so downstream callers are unaffected by the move. Only the
# collection-POLICY defaults below (the players / pairing mix) live here.

DEFAULT_PLAYER1 = "aggressive-coverage"
DEFAULT_PLAYER2 = "opponent-shadower"

# The default pairing MIX for a rotating run: the coverage family vs each other (the three round-
# robin orderings) plus the coverage family vs the random baseline. A "pairing" is a
# (player1-selector, player2-selector) tuple. This covers the coverage policies against one another
# AND against random, which is the sensible-coverage default the contract asks for.
DEFAULT_PAIRINGS: list[tuple[str, str]] = [
    ("aggressive-coverage", "opponent-shadower"),
    ("opponent-shadower", "wall-hugger"),
    ("wall-hugger", "aggressive-coverage"),
    ("aggressive-coverage", "random"),
    ("opponent-shadower", "random"),
    ("wall-hugger", "random"),
]


# --- map rotation: config path -> (build config, switch_arena path) + the int index ----------
#
# A rotation entry is a map-CONFIG path (e.g. exp-configs/maps/center_block.json). Two distinct
# things come off it:
#   * the BUILD CONFIG to launch a worker on (the config path itself — it carries obs_pixels + the
#     boot arena);
#   * the SWITCH_ARENA path the env hands to ``env.reset(options={"switch_arena": ...})`` between
#     episodes, which Unity resolves and ECHOES back as ``WallLayout.map_id``. That echoed string
#     is the F5 tag source, so the int index is keyed on the SAME arena-path string.
# The arena path is the config's ``arena_path`` field. Unity echoes that verbatim resolved string,
# so keying the index on ``arena_path`` makes the echo decode to the right int.


def arena_path_for_config(config_path: str | Path) -> str:
    """Read a map-config JSON and return its ``arena_path`` (the switch_arena / echo string).

    The ``arena_path`` field is the arena the build resolves AND the verbatim string Unity echoes
    back as ``WallLayout.map_id`` after a switch. Strict ``json`` (Python-parsed). Raises
    ``ValueError`` if the field is missing or not a string.
    """
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    arena = config.get("arena_path")
    if not isinstance(arena, str):
        raise ValueError(f"map config {config_path} has no string 'arena_path' (got {arena!r})")
    return arena


def build_rotation(map_values: list[str] | None) -> list[str]:
    """Resolve the ``--maps`` flag value into the ROTATION SET of arena-path strings.

    Delegates to :func:`core.maps.resolve_map_rotation` (the shared rotation contract: ``None`` ->
    single-map, ``[]`` / sentinel -> all 10 shipped exp-configs/maps sorted, a dir -> its *.json
    sorted, a list -> given order) and then maps each resolved map-config path to its
    ``arena_path`` (:func:`arena_path_for_config`). The returned list is the ordered rotation set;
    its INDEX in this list is the on-disk int for that arena (the dataset ``maps`` list). A ``None``
    flag value (single-map mode) yields ``[]`` — the caller uses the no-rotation path.
    """
    configs = core_maps.resolve_map_rotation(map_values)
    if configs is None:
        return []
    return [arena_path_for_config(p) for p in configs]


def build_map_index(rotation: Sequence[str]) -> dict[str, int]:
    """Build the deterministic ``arena-path -> int`` index over the rotation set.

    The int is the entry's position in ``rotation`` (the dataset ``maps`` list order), so it is a
    stable, reversible key. This is the index :func:`collect.resolve_map_tag` looks the ECHOED
    arena path up in (the F5 tag-from-echo). Raises ``ValueError`` on a duplicate arena path (an
    ambiguous index).
    """
    index: dict[str, int] = {}
    for i, arena in enumerate(rotation):
        if arena in index:
            raise ValueError(f"duplicate arena path in rotation: {arena!r}")
        index[arena] = i
    return index


def parse_pairing(value: str) -> tuple[str, str]:
    """Parse a ``--pairing`` value ``"p1:p2"`` into a validated ``(player1, player2)`` pair.

    Splits on the single ``:`` and validates BOTH selectors against :data:`AGENT_SELECTORS`
    eagerly (:func:`make_agent` raises on an unknown name). Raises ``ValueError`` on a malformed
    value (not exactly one ``:``).
    """
    parts = value.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"pairing must be 'player1:player2', got {value!r}")
    p1, p2 = parts[0], parts[1]
    make_agent(p1)  # validates / raises ValueError on an unknown selector
    make_agent(p2)
    return p1, p2


def round_robin_plan(
    rotation: Sequence[str],
    pairings: Sequence[tuple[str, str]],
    *,
    episodes: int,
    worker_id: int,
    n_workers: int,
    map_index: dict[str, int],
) -> list[EpisodePlan]:
    """The PURE per-worker round-robin over the (rotation x pairing) grid (no live build).

    The grid is ``rotation x pairings`` enumerated in a FIXED order (map-major: grid cell ``g``
    is ``(map = rotation[g // P], pairing = pairings[g % P])`` for ``P = len(pairings)``), giving
    ``G = M * P`` cells. Worker ``w`` runs ``episodes`` episodes; its episode ``i`` picks grid cell
    ``(w + i * n_workers) % G``.

    WORKER SPREAD: across all ``n_workers`` the chosen global indices ``w + i * n_workers`` form a
    BIJECTION onto ``[0, n_workers * episodes)``, so the union of every worker's episodes is exactly
    ``{ k % G : k in [0, n_workers * episodes) }`` — each grid cell appears ``floor / ceil`` of
    ``n_workers * episodes / G`` times (balanced), and at a fixed ``i`` two distinct workers
    (``w < n_workers <= ...``) always pick DIFFERENT cells, so no two workers do identical episodes
    while the union still covers the whole grid evenly.

    Each :class:`EpisodePlan` carries the cell's switch_arena (the arena-path string), its pairing's
    two selectors, and the cell's intended int (``map_index[arena]``) for the documented
    tag-from-echo fallback. ``rotation`` empty (single-map mode) is handled by the caller, not here
    (this asserts a non-empty grid). Returns the worker's ordered ``episode_plan``.
    """
    grid_size = len(rotation) * len(pairings)
    if grid_size == 0:
        raise ValueError("round_robin_plan needs a non-empty rotation x pairing grid")
    n_workers = max(1, n_workers)
    n_pairings = len(pairings)
    plan: list[EpisodePlan] = []
    for i in range(episodes):
        cell = (worker_id + i * n_workers) % grid_size
        arena = rotation[cell // n_pairings]
        p1, p2 = pairings[cell % n_pairings]
        plan.append(
            EpisodePlan(
                switch_arena=arena,
                player1=p1,
                player2=p2,
                intended_map_id=map_index[arena],
            )
        )
    return plan


def write_maps_sidecar(out_dir: str | Path, rotation: Sequence[str]) -> Path:
    """Write the maps-list sidecar (the int -> arena-path mapping) next to the shards.

    The on-disk ``map_ids`` are int32 indices into the dataset ``maps`` list; this sidecar IS that
    list, so the ints are reversible to the arena Unity loaded. Strict JSON (Python-parsed)::

        {"schema_version": 1, "maps": ["Arenas/center_block.json", ...]}

    where ``maps[i]`` is the arena path for int ``i``. Pure (stdlib ``json`` + ``pathlib``); writes
    ``maps.json`` (:data:`MAPS_SIDECAR_NAME`) into ``out_dir`` and returns its path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MAPS_SIDECAR_NAME
    payload = {"schema_version": 1, "maps": list(rotation)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read_maps_sidecar(path: str | Path) -> list[str]:
    """Read a maps-list sidecar and return the ``maps`` list (int ``i`` -> arena path ``maps[i]``).

    The inverse of :func:`write_maps_sidecar`. Strict ``json``. Raises ``ValueError`` if the
    ``maps`` key is missing or is not a list of strings.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    maps_list = payload.get("maps")
    if not isinstance(maps_list, list) or not all(isinstance(m, str) for m in maps_list):
        raise ValueError(f"maps sidecar {path} has no 'maps' list of strings")
    return maps_list


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
    driver-side, built per episode from the pool :func:`agent_pool_factory` builds. Factored out so
    the env-build wiring is testable against an in-process fake transport with no live build.
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
    :func:`agent_pool_factory`), so the env carries neither. The build is launched ONCE per worker
    on the boot config and rotates between arenas via ``switch_arena`` (the long-lived-build model).
    The live ``Popen`` is stashed and the env's ``close`` is wrapped so :func:`collect.run_worker`'s
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


def agent_pool_factory(spec: CollectionSpec) -> dict[str, core_agent.Agent]:
    """Build the worker's ``selector -> agent`` pool covering every selector its plan pairs.

    MODULE-LEVEL + plain (spawn-importable). The worker's ``episode_plan`` is a round-robin over
    pairings, so a selector recurs across episodes; building ONE agent per selector (seeded with
    ``spec.seed``) keeps that selector's RNG stream CONTINUOUS across the episodes it plays and
    avoids rebuilding it every episode. The pool's selectors are taken from ``spec.extra["pool"]``
    (the union of all selectors the plan names). The driver picks each episode's pairing's two
    agents out of this pool by name.
    """
    return {name: make_agent(name, seed=spec.seed) for name in spec.extra["pool"]}


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
    map_values: list[str] | None,
    pairings: Sequence[tuple[str, str]],
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
    shard_size: int | None = None,
) -> list[CollectionSpec]:
    """Turn collection params into a list of N :class:`CollectionSpec` (one per worker).

    Pure: builds plain spec data + reads map-config JSON for arena paths, NO live launch.

    Map resolution: ``map_values`` is the ``--maps`` flag value resolved by
    :func:`core.maps.resolve_map_rotation`. ``None`` (flag absent) -> SINGLE-MAP mode (today's
    behavior): no ``switch_arena``, every episode tagged int ``0``, paired with the SINGLE default
    pairing (``pairings[0]``). A non-``None`` value -> ROTATION mode: the rotation set of arena
    paths, a deterministic round-robin over (rotation x ``pairings``) per worker
    (:func:`round_robin_plan`), and each episode switches to its arena + is tagged from the echo.

    BOOT CONFIG (the obs_pixels gotcha): the build is launched ONCE per worker on the ``--map``
    config (``config``) in BOTH modes, then rotates via ``switch_arena``. ``obs_pixels`` is a
    LAUNCH-time config flag (the env blocks on a frame that never arrives without it), and the
    shipped ``exp-configs/maps`` rotation configs do NOT enable it — so they are used ONLY to
    extract each arena's ``arena_path`` (the switch targets), never as the boot config. The
    ``--map`` config (the demo config) enables obs_pixels, so the build boots correctly and the
    runtime ``switch_arena`` changes only the arena, not the pixel channel.

    ``workers`` is CLAMPED to ``[1, MAX_WORKERS]``. Each worker gets a distinct ``worker_id``
    (0..N-1), its own ``out_dir`` subdir (so shard files never clash), its slice of the round-robin
    as ``episode_plan``, the shared ``max_steps`` + ``map_index``, a distinct ``seed``
    (``base + worker_id * stride``), the ``shard_size`` in-RAM buffer bound (``None`` -> the worker
    auto-derives it from the byte budget), and an ``extra`` carrying the launch params + the agent
    pool (the union of selectors its plan names). The module-level :func:`env_factory` /
    :func:`agent_pool_factory` are wired onto every spec. Validates the selectors + episodes up
    front (``ValueError``). The maps-list sidecar (int -> arena path) is written by
    :func:`main` from each spec's ``extra["maps"]`.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if not pairings:
        raise ValueError("at least one pairing is required")
    # Validate every selector named by any pairing eagerly so a bad CLI fails before launch.
    for p1, p2 in pairings:
        make_agent(p1, seed=seed)
        make_agent(p2, seed=seed)
    if map_name not in MAP_CONFIGS:
        valid = ", ".join(sorted(MAP_CONFIGS))
        raise ValueError(f"unknown map {map_name!r}; choose one of: {valid}")

    rotation = build_rotation(map_values)  # [] in single-map mode
    # The build always boots on the obs_pixels-enabled --map config; switch_arena rotates the arena.
    boot_config = config

    if rotation:
        # Rotation mode: the int index is the position in the rotation set; the round-robin over
        # the grid is sliced per worker. The echo decodes through map_index.
        map_index = build_map_index(rotation)
        sidecar_maps = rotation
    else:
        # Single-map mode: one arena, int 0, no switch. The sole pairing is pairings[0].
        single_arena = arena_path_for_config(boot_config)
        map_index = {}  # no echo lookup -> the intended int 0 is used (tag_source "intended")
        sidecar_maps = [single_arena]
        pairings = [pairings[0]]

    n_workers = max(1, min(int(workers), MAX_WORKERS))
    out_root = Path(out_dir)
    pool = sorted({s for pairing in pairings for s in pairing})

    specs: list[CollectionSpec] = []
    for worker_id in range(n_workers):
        worker_out = out_root / f"worker_{worker_id}"
        if rotation:
            episode_plan = round_robin_plan(
                rotation,
                pairings,
                episodes=episodes,
                worker_id=worker_id,
                n_workers=n_workers,
                map_index=map_index,
            )
        else:
            # Single-map: every episode is the sole pairing on the no-switch arena, tagged int 0.
            p1, p2 = pairings[0]
            episode_plan = [
                EpisodePlan(switch_arena=None, player1=p1, player2=p2, intended_map_id=0)
                for _ in range(episodes)
            ]
        spec = CollectionSpec(
            worker_id=worker_id,
            out_dir=str(worker_out),
            episode_plan=episode_plan,
            max_steps=max_steps,
            seed=seed + worker_id * _SEED_STRIDE,
            map_index=dict(map_index),
            shard_size=shard_size,
            env_factory=env_factory,
            agent_pool_factory=agent_pool_factory,
            extra={
                "exe": str(exe),
                "config": str(boot_config),
                "base_port": int(base_port),
                "frame_shape": tuple(frame_shape),
                "pool": pool,
                "map": map_name,
                "maps": list(sidecar_maps),
            },
        )
        specs.append(spec)
    return specs


# --- end-of-run summary (PURE counting + format-only assembly) -------------------------------


@dataclass(frozen=True)
class CollectionSummary:
    """A scannable end-of-run rollup of what collection wrote (counts only, no filenames).

    * ``total_shards`` / ``total_samples`` — run-wide totals.
    * ``worker_samples`` — ``[(worker_id, samples)]`` in worker order (a worker that wrote nothing
      is ``(id, 0)``).
    * ``map_samples`` — ``[(map_id, arena_path, samples)]`` aligned to the ``maps`` list, ONE row
      per arena including arenas with zero samples (shown as 0).
    """

    total_shards: int
    total_samples: int
    worker_samples: list[tuple[int, int]]
    map_samples: list[tuple[int, str, int]]


def summarize_collection(
    worker_map_ids: Sequence[tuple[int, np.ndarray]],
    maps: Sequence[str],
    *,
    total_shards: int,
) -> CollectionSummary:
    """PURE counting (no disk I/O): per-worker + per-map sample counts from per-sample ``map_ids``.

    ``worker_map_ids`` is ``[(worker_id, map_ids)]`` where ``map_ids`` is that worker's per-sample
    int group-key array (the :attr:`readers.DatasetIndex.map_ids` slice). ``maps`` is the maps-list
    sidecar (``maps[i]`` is the arena path for int ``i``). ``total_shards`` is known from the worker
    results. Returns a :class:`CollectionSummary` whose ``map_samples`` has ONE row per entry in
    ``maps`` (arenas with no samples reported as 0), so the per-map table is complete and aligned to
    the maps list. Map ids outside ``range(len(maps))`` are counted into the totals but omitted from
    the per-map table (a defensive guard; the on-disk ints are always valid indices).
    """
    per_map: dict[int, int] = {}
    worker_samples: list[tuple[int, int]] = []
    total_samples = 0
    for worker_id, map_ids in worker_map_ids:
        ids = np.asarray(map_ids)
        n = int(ids.shape[0])
        worker_samples.append((worker_id, n))
        total_samples += n
        if n:
            values, counts = np.unique(ids, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                per_map[int(value)] = per_map.get(int(value), 0) + int(count)

    map_samples = [(i, arena, per_map.get(i, 0)) for i, arena in enumerate(maps)]
    return CollectionSummary(
        total_shards=int(total_shards),
        total_samples=total_samples,
        worker_samples=worker_samples,
        map_samples=map_samples,
    )


def _read_worker_map_ids(out_dir: str | Path) -> np.ndarray:
    """Per-sample ``map_ids`` written into a worker's out dir (empty array if it wrote no shards).

    Globs the worker dir's ``shard_*.npz`` (the runner's ``shard_w{id}_*`` prefix matches) in
    SORTED filename order and reads ONLY each shard's ``map_ids`` member. Accessing a single
    member of a lazy ``NpzFile`` decompresses ONLY that member, so the big ``frames`` array is
    NEVER inflated — the summary needs only the tiny group-key array. A worker that wrote no
    shards (empty glob) returns an empty int32 array so the summary never crashes on it.
    """
    parts: list[np.ndarray] = []
    for path in sorted(Path(out_dir).glob("shard_*.npz")):
        with np.load(path) as npz:
            parts.append(np.asarray(npz[schema.ARRAY_MAP_IDS], dtype=np.int32))
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(parts).astype(np.int32, copy=False)


def _format_summary(
    summary: CollectionSummary,
    out_dir: str | Path,
    worker_shards: dict[int, int],
) -> list[str]:
    """Render a :class:`CollectionSummary` into the concise multi-line printout (format only).

    ``worker_shards`` maps ``worker_id -> num_shards`` (from the worker results) so each terse
    per-worker line reports both its shard and sample counts. NO filenames are printed.
    """
    lines = [
        f"done: {summary.total_shards} shards, {summary.total_samples:,} samples, "
        f"{len(summary.worker_samples)} workers -> {out_dir}"
    ]
    lines.extend(
        f"worker {worker_id}: {worker_shards.get(worker_id, 0)} shards, {samples:,} samples"
        for worker_id, samples in summary.worker_samples
    )
    for map_id, arena, samples in summary.map_samples:
        lines.append(f"map {map_id}  {arena}  {samples:,}")
    return lines


# --- CLI -------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.data.collect_runner",
        description="Collect (frame, state, action) samples by driving live Unity builds.",
    )
    parser.add_argument(
        "--pairing",
        dest="pairings",
        action="append",
        metavar="P1:P2",
        help=(
            "a (player1:player2) selector pairing; repeat for several. Omitted -> the default "
            "coverage-family mix (the coverage policies vs each other + vs random)."
        ),
    )
    parser.add_argument(
        "--maps",
        "--map-rotation",
        dest="maps",
        nargs="*",
        default=None,
        metavar="MAP_CONFIG",
        help=(
            "Rotate over a set of map configs (round-robin map x pairing, via switch_arena). "
            "No value -> the shipped 10 exp-configs/maps; a directory -> its *.json configs "
            "(sorted); a list of paths -> that order. Absent -> no rotation (single --map)."
        ),
    )
    parser.add_argument(
        "--map",
        default=DEFAULT_MAP,
        choices=sorted(MAP_CONFIGS),
        help="single-map (no-rotation) config; the boot + only arena when --maps is absent",
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
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help=(
            "in-RAM BUFFER bound (samples per worker before a flush), NOT a file-size knob. "
            "Omitted -> auto: derived from the byte budget so the per-worker buffer stays bounded "
            "regardless of frame resolution (the OOM-safe default)."
        ),
    )
    parser.add_argument(
        "--allow-oversized",
        action="store_true",
        help=(
            "skip the pre-flight memory abort (the estimate is still printed). Use only when you "
            "know the box has the RAM the guard's conservative estimate does not account for."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the specs from the CLI and drive :func:`collect.collect_parallel`; print a summary.

    Resolves the boot config + the pairing set, validates the build / config exist, builds the
    per-worker round-robin specs, runs the PRE-FLIGHT MEMORY GUARD (always printing the peak-buffer
    estimate at startup; aborting before launch when the estimate blows the budget unless
    ``--allow-oversized``), writes the maps-list sidecar (int -> arena path) per worker out dir AND
    at the run root, runs collection in parallel (spawn), and prints a concise end-of-run summary
    (an aggregate line, a terse shard+sample count per worker, and a per-map sample-count table —
    no filename dumps; counts derived by :func:`summarize_collection` over each worker's map_ids).
    Returns ``0`` on success, ``2`` on a missing build / config OR a pre-flight memory abort.
    """
    args = _parse_args(argv)
    config = MAP_CONFIGS[args.map]

    if not args.exe.exists():
        print(f"error: build not found at {args.exe}", file=sys.stderr)
        return 2
    if not config.exists():
        print(f"error: config not found at {config}", file=sys.stderr)
        return 2

    # Resolve the pairing set: each --pairing "p1:p2" parsed + validated, else the default mix.
    pairings = [parse_pairing(p) for p in args.pairings] if args.pairings else DEFAULT_PAIRINGS

    requested = args.workers
    specs = build_specs(
        map_values=args.maps,
        pairings=pairings,
        map_name=args.map,
        config=config,
        exe=args.exe,
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        workers=args.workers,
        base_port=args.base_port,
        seed=args.seed,
        shard_size=args.shard_size,
    )
    if requested > len(specs):
        print(f"note: clamped --workers {requested} to {len(specs)} (MAX_WORKERS={MAX_WORKERS})")

    # PRE-FLIGHT MEMORY GUARD (psutil only here, the CLI glue). The per-worker in-RAM buffer — not
    # the on-disk shard — is the OOM surface. Resolve the EFFECTIVE shard_size (auto -> the byte
    # budget for this frame, so the printed estimate matches what the workers will actually buffer),
    # estimate the peak across all workers, ALWAYS print the estimate, then abort BEFORE launching
    # any build if it blows the budget (unless --allow-oversized).
    fb = frame_nbytes(FRAME_SHAPE)
    effective_shard_size = (
        args.shard_size if args.shard_size is not None else collect.default_shard_size(fb)
    )
    available = psutil.virtual_memory().available
    try:
        estimate_line = collect.check_memory_budget(
            frame_nbytes=fb,
            shard_size=effective_shard_size,
            workers=len(specs),
            available_bytes=available,
            allow_oversized=args.allow_oversized,
        )
    except MemoryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(estimate_line)

    # Persist the maps-list sidecar (int -> arena path) so on-disk map_ids are decodable. The
    # rotation set is identical across workers; write it at the run root AND into each worker dir
    # (a worker dir is a self-describing dataset slice).
    sidecar_maps = specs[0].extra["maps"]
    write_maps_sidecar(args.out_dir, sidecar_maps)
    for spec in specs:
        write_maps_sidecar(spec.out_dir, sidecar_maps)

    rotating = args.maps is not None
    mode = f"rotation over {len(sidecar_maps)} arenas" if rotating else f"single map={args.map}"
    print(
        f"collecting: {mode}, {len(pairings)} pairing(s) "
        f"({len(specs)} worker(s), {args.episodes} episode(s) each, max_steps={args.max_steps})"
    )
    results = collect_parallel(specs)

    # Concise end-of-run summary: read back each worker's per-sample map_ids to derive sample +
    # per-map counts, then count PURELY via summarize_collection. _read_worker_map_ids inflates ONLY
    # the tiny map_ids member of each shard (never the big frames array), so this stays cheap even
    # over hundreds of shards. The worker result dict carries only shards/num_shards (empty workers
    # read as 0 samples).
    out_dir_by_worker = {spec.worker_id: spec.out_dir for spec in specs}
    worker_shards = {r["worker_id"]: r["num_shards"] for r in results}
    total_shards = sum(r["num_shards"] for r in results)
    worker_map_ids = [
        (r["worker_id"], _read_worker_map_ids(out_dir_by_worker[r["worker_id"]])) for r in results
    ]
    summary = summarize_collection(worker_map_ids, sidecar_maps, total_shards=total_shards)
    for line in _format_summary(summary, args.out_dir, worker_shards):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
