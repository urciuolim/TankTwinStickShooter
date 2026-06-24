"""The collection driver: drive the game through ``TankEnv`` and capture samples.

Collection routes through :class:`pop_trainer.env.tank_env.TankEnv` — the SAME gymnasium
observation pipeline RL trains on — so the pretraining ``(frame, state)`` rows are byte-for-byte
the observations the policy will later see, with no drift between pretraining inputs and RL
observations. Collection pairs a ``player1`` agent + a ``player2`` agent (both
:mod:`pop_trainer.agents` policies). ``player2`` is injected into the ENV (the env owns it):
each step the env computes player2's flipped first-person view and calls ``player2.act``. The
driver here drives ``player1`` directly: it hands ``player1.act`` the current 52-float state —
which for player1 is the UNFLIPPED state (player1 already occupies the ``PLAYER_1`` slot), the
mirror of how the env hands ``player2`` its FLIPPED state. The two halves of the self-play pair
therefore each see their OWN first-person view.

BOTH players' actions are captured: ``env.step`` surfaces ``info["p1_action"]`` (the action
actually sent for player1, equal to what ``player1.act`` returned) and ``info["p2_action"]``
(player2's actual action, drawn inside the env). The recorded ``(2, 5)`` action array stores
BOTH — there is no zeroed-player2 slot.

LAYERING for testability:

* :func:`run_episode` is the PURE step loop over an INJECTED ``env`` (anything with
  ``reset()`` / ``step(action)`` returning the gymnasium tuples) built with its ``player2``.
  It reads the first ``(obs, info)`` from ``env.reset``, then each step computes the player1
  action from the current state, calls ``env.step``, and records the CURRENT ``(frame, state)``
  paired with BOTH actions that advanced t -> t+1 (taken from ``info``) — stopping on
  ``terminated or truncated`` or the step cap. The transport lives inside the env, so this loop
  is unit-testable against a real ``TankEnv`` built over an in-process fake connection with NO
  live socket.
* :func:`collect_to_shards` wraps a sequence of episodes on one env into
  :mod:`pop_trainer.data.shards` shards on disk (pure given a fake-backed env).
* :class:`CollectionSpec` / :func:`run_worker` / :func:`collect_parallel` are the LIVE,
  parallel-safe orchestration (multiprocessing with the SPAWN start method — never fork; this
  is a YOU MUST in CLAUDE.md). The env (with its player2) and the player1 agent are built INSIDE
  the worker via injected factories, so the live path is isolated and NOT unit-tested. The
  worker target is module-level + takes plain data (picklable for spawn).

An episode ends when the env reports ``terminated or truncated`` (its own boundary) or the step
cap is reached; the final pair for that step is recorded with the zero ``(2, 5)`` action (no
action is applied after it). ``step_idx`` resets per episode and ``episode_id`` increases
monotonically per worker.

stdlib + numpy + :mod:`pop_trainer.core.state` + :mod:`pop_trainer.env.tank_env` (``TankEnv``)
+ :mod:`pop_trainer.agents` (``validate_action``) + :mod:`pop_trainer.data.{schema,shards}`. No
models, no pretraining, no rl, no tank_twin.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pop_trainer import agents
from pop_trainer.core import agent as core_agent
from pop_trainer.core import state as state_schema
from pop_trainer.data import schema, shards
from pop_trainer.env.tank_env import TankEnv

__all__ = [
    "Sample",
    "EpisodeResult",
    "run_episode",
    "collect_to_shards",
    "CollectionSpec",
    "run_worker",
    "collect_parallel",
]


@dataclass
class Sample:
    """One captured step: the time-aligned ``(frame, state, action)`` plus provenance.

    * ``frame`` — ``(H, W, 3)`` uint8 RGB, the env observation for ``state``.
    * ``state`` — the paired 52-float wire state (``info["state"]``).
    * ``action`` — ``(2, 5)`` float: ``[player1, player2]`` x ``[mx, my, ax, ay, fire]``. Both
      rows are the actions APPLIED via ``env.step`` to advance the sim to the next frame: row 0
      is ``info["p1_action"]`` (player1's applied action) and row 1 is ``info["p2_action"]``
      (player2's actual action, drawn inside the env). On the LAST recorded step of an episode
      (the boundary step) no further action is applied, so the whole ``(2, 5)`` is the zero
      action.
    * ``map_id`` / ``episode_id`` / ``step_idx`` — provenance / the split group key.
    """

    frame: np.ndarray
    state: np.ndarray
    action: np.ndarray
    map_id: int
    episode_id: int
    step_idx: int


@dataclass
class EpisodeResult:
    """The samples captured in one episode, plus whether it ended on the env's boundary."""

    samples: list[Sample] = field(default_factory=list)
    ended_done: bool = False

    def __len__(self) -> int:
        return len(self.samples)


def _maybe_reset(agent: object, seed: int | None) -> None:
    """Call ``agent.reset(seed=seed)`` if the agent exposes ``reset`` (it is OPTIONAL).

    ``core.Agent`` requires only ``act``; ``reset`` belongs to the static-only ``StatefulAgent``
    surface. Stateful / seeded agents (e.g. ``RandomAgent``, ``ScriptedCycleAgent``) implement it
    so an episode replays deterministically; pure agents do not, and are left untouched.
    """
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed=seed)


def _zero_action() -> np.ndarray:
    """The ``(2, 5)`` zero action recorded on a boundary step (no action applied after it)."""
    return np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)


def run_episode(
    env: TankEnv,
    *,
    player1: core_agent.Agent,
    map_id: int,
    episode_id: int,
    max_steps: int,
    seed: int | None = None,
) -> EpisodeResult:
    """PURE step loop: drive ONE ``TankEnv`` for an episode and capture time-aligned samples.

    ``env`` MUST already be built with the desired ``player2`` agent (the env owns it); this
    loop drives ``player1`` only. It resets ``player1`` (and the env's ``player2``, if either
    exposes ``reset``) and calls ``env.reset(seed=seed)`` to start a fresh round, so the whole
    trajectory is a deterministic function of (player1, player2, seed). Then each step:

    1. records nothing yet — it computes player1's action ``a1 = player1.act(vec)`` from the
       CURRENT 52-float state (player1's own UNFLIPPED view), validates it, and calls
       ``env.step(a1)``,
    2. records the CURRENT ``(frame, state)`` paired with the BOTH actions that advanced this
       step, read back from ``info`` (row 0 = ``info["p1_action"]`` == ``a1``, row 1 =
       ``info["p2_action"]`` == player2's actual action),
    3. on the env's boundary (``terminated or truncated``) or the step cap, records the final
       ``(frame, state)`` with the zero ``(2, 5)`` action (no action is applied after it).

    The transport lives inside ``env``, so this is unit-testable against a real ``TankEnv`` built
    over an in-process fake connection. Returns an :class:`EpisodeResult`.
    """
    result = EpisodeResult()
    _maybe_reset(player1, seed)
    _maybe_reset(getattr(env, "player2", None), seed)
    obs, info = env.reset(seed=seed)
    frame = obs
    vec = info["state"]
    step_idx = 0
    done = False
    while True:
        at_cap = step_idx >= max_steps - 1
        if done or at_cap:
            # Final recorded step: no further action is applied; record the zero action.
            result.samples.append(
                _make_sample(frame, vec, _zero_action(), map_id, episode_id, step_idx)
            )
            result.ended_done = done
            return result

        # player1 acts on its OWN unflipped view (the current 52-float state). The env fills
        # player2 internally and surfaces BOTH applied actions in ``info`` after the step.
        a1 = agents.validate_action(player1.act(vec))
        next_obs, _reward, terminated, truncated, info = env.step(a1)

        action = np.array([info["p1_action"], info["p2_action"]], dtype=np.float32)
        result.samples.append(_make_sample(frame, vec, action, map_id, episode_id, step_idx))

        frame = next_obs
        vec = info["state"]
        done = bool(terminated or truncated)
        step_idx += 1


def _make_sample(frame, state_vec, action, map_id, episode_id, step_idx) -> Sample:
    state_arr = np.asarray(state_vec, dtype=np.float32)
    state_schema.validate(state_arr)
    return Sample(
        frame=np.ascontiguousarray(frame, dtype=np.uint8),
        state=state_arr,
        action=np.ascontiguousarray(action, dtype=np.float32),
        map_id=int(map_id),
        episode_id=int(episode_id),
        step_idx=int(step_idx),
    )


def samples_to_shard(samples: Sequence[Sample], *, with_actions: bool = True) -> shards.Shard:
    """Stack a list of :class:`Sample` into a :class:`pop_trainer.data.shards.Shard`.

    Pure: builds the parallel arrays the shard schema requires. ``with_actions=False`` drops the
    action array (a (frame, state)-only dataset). Raises ``ValueError`` on an empty list.
    """
    if not samples:
        raise ValueError("cannot build a shard from zero samples")
    frames = np.stack([s.frame for s in samples])
    states = np.stack([s.state for s in samples])
    map_ids = np.array([s.map_id for s in samples], dtype=np.int32)
    episode_ids = np.array([s.episode_id for s in samples], dtype=np.int32)
    step_idxs = np.array([s.step_idx for s in samples], dtype=np.int32)
    actions = np.stack([s.action for s in samples]) if with_actions else None
    return shards.Shard(
        frames=frames,
        states=states,
        map_ids=map_ids,
        episode_ids=episode_ids,
        step_idxs=step_idxs,
        actions=actions,
    )


def collect_to_shards(
    env: TankEnv,
    *,
    out_dir: str | Path,
    map_ids: Sequence[int],
    player1: core_agent.Agent,
    max_steps: int,
    seed: int | None = None,
    shard_prefix: str = "shard_w0",
    shard_size: int = 10_000,
    with_actions: bool = True,
) -> list[Path]:
    """Run one episode per entry of ``map_ids`` on ``env`` and write shards to ``out_dir``.

    ``env`` must already carry its ``player2``; this drives ``player1`` against it. Each episode
    calls ``env.reset`` (via :func:`run_episode`) to start a fresh round; ``map_ids`` tags each
    episode's samples with its map. The per-episode seed is ``seed + episode_id`` (when a base
    ``seed`` is given) so each episode resets ``player1`` (and the env's ``player2``) to a
    distinct yet reproducible state. Samples accumulate and flush to a shard every ``shard_size``,
    plus a final flush. Returns the list of written shard paths. Pure given a fake-backed ``env``
    (the live socket path lives in :func:`run_worker`).
    """
    out_dir = Path(out_dir)
    buffer: list[Sample] = []
    written: list[Path] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        shard = samples_to_shard(buffer, with_actions=with_actions)
        path = out_dir / f"{shard_prefix}_{shard_index:04d}.npz"
        written.append(shards.write_shard(path, shard))
        shard_index += 1
        buffer = []

    for episode_id, map_id in enumerate(map_ids):
        episode_seed = None if seed is None else seed + episode_id
        ep = run_episode(
            env,
            player1=player1,
            map_id=map_id,
            episode_id=episode_id,
            max_steps=max_steps,
            seed=episode_seed,
        )
        buffer.extend(ep.samples)
        while len(buffer) >= shard_size:
            chunk, buffer = buffer[:shard_size], buffer[shard_size:]
            shard = samples_to_shard(chunk, with_actions=with_actions)
            path = out_dir / f"{shard_prefix}_{shard_index:04d}.npz"
            written.append(shards.write_shard(path, shard))
            shard_index += 1
    flush()
    return written


# ===========================================================================================
# LIVE parallel orchestration (spawn-based; NOT unit-tested — the contract forbids live tests)
# ===========================================================================================
@dataclass
class CollectionSpec:
    """Plain, picklable data one worker needs to collect (spawn-safe: no live handles).

    Holds ONLY plain data + import-able factory names. The env (and its socket) is constructed
    INSIDE the worker via ``env_factory`` (a picklable callable, e.g. a module-level function);
    ``env_factory`` is also responsible for injecting the desired ``player2`` agent into the env
    (the env owns player2). The ``player1`` agent is built inside the worker from
    ``player1_factory``. So NO socket / env / agent crosses the spawn boundary.
    """

    worker_id: int
    out_dir: str
    map_ids: list[int]
    max_steps: int
    seed: int | None = None
    shard_size: int = 10_000
    with_actions: bool = True
    # Picklable factories, called INSIDE the worker process:
    #   env_factory(spec) -> a TankEnv built WITH its player2 (socket opened in-worker).
    #   player1_factory(spec) -> the player1 agent the driver advances.
    env_factory: Callable[[CollectionSpec], object] | None = None
    player1_factory: Callable[[CollectionSpec], core_agent.Agent] | None = None
    extra: dict = field(default_factory=dict)


def run_worker(spec: CollectionSpec) -> dict:
    """Module-level worker entry point (picklable for spawn). Builds the env IN-WORKER.

    Builds the ``TankEnv`` (with its player2) via ``spec.env_factory`` (its socket opened HERE,
    never inherited) and the ``player1`` agent via ``spec.player1_factory``, then delegates to
    :func:`collect_to_shards`. Closes the env in a ``finally`` so the worker that opened the
    socket also releases it. Returns a small result dict (worker id, shard file names, count).
    Live path — exercised by the orchestrator, not the unit tests.
    """
    if spec.env_factory is None:
        raise ValueError("run_worker needs an env_factory")
    if spec.player1_factory is None:
        raise ValueError("run_worker needs a player1_factory")
    env = spec.env_factory(spec)
    player1 = spec.player1_factory(spec)
    try:
        written = collect_to_shards(
            env,
            out_dir=spec.out_dir,
            map_ids=spec.map_ids,
            player1=player1,
            max_steps=spec.max_steps,
            seed=spec.seed,
            shard_prefix=f"shard_w{spec.worker_id}",
            shard_size=spec.shard_size,
            with_actions=spec.with_actions,
        )
    finally:
        # The worker built the env, so the worker closes it (releases the socket).
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "worker_id": spec.worker_id,
        "shards": [p.name for p in written],
        "num_shards": len(written),
    }


def collect_parallel(specs: Sequence[CollectionSpec]) -> list[dict]:
    """Run worker specs in parallel using the SPAWN start method (never fork; Windows-safe).

    Uses ``multiprocessing.get_context("spawn")`` explicitly so behavior is identical on Windows
    and POSIX and no parent file descriptors / sockets are inherited. A single spec runs
    in-process (simpler, still spawn-safe). Each ``CollectionSpec`` builds its own env (and socket)
    inside its worker; nothing live crosses the process boundary. Returns the per-worker result
    dicts, sorted by worker id.
    """
    specs = list(specs)
    if not specs:
        return []
    if len(specs) == 1:
        results = [run_worker(specs[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(specs)) as pool:
            results = pool.map(run_worker, specs)
    return sorted(results, key=lambda r: r["worker_id"])
