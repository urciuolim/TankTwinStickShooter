"""The collection driver: drive the game through ``TankEnv`` and capture samples.

Collection routes through :class:`pop_trainer.env.tank_env.TankEnv` — the SAME gymnasium
observation pipeline RL trains on — so the pretraining ``(frame, state)`` rows are byte-for-byte
the observations the policy will later see, with no drift between pretraining inputs and RL
observations. Each step feeds the deterministic agent policy (:mod:`pop_trainer.data.policies`)
the current 52-float state, applies its action via ``env.step``, and records the time-aligned
``(frame, state, action)`` sample into shards.

The env draws the OPPONENT action internally from its seeded ``np_random`` and does not surface
it through the public gym contract. The recorded trajectory is therefore a deterministic function
of (agent policy, seed): the agent action is ``p1_policy(state)``; the opponent action stream is
the env's seeded draw (reproducible from the per-episode seed) and is not observable, so the
sample's opponent slot is recorded as the zero action.

LAYERING for testability:

* :func:`run_episode` is the PURE step loop over an INJECTED ``env`` (anything with
  ``reset()`` / ``step(action)`` returning the gymnasium tuples). It reads the first
  ``(obs, info)`` from ``env.reset``, then each step computes the agent action from the current
  state, calls ``env.step``, and records the CURRENT ``(frame, state)`` paired with the action
  APPLIED to produce the next frame — stopping on ``terminated or truncated`` or the step cap.
  The transport lives inside the env, so this loop is unit-testable against a real ``TankEnv``
  built over an in-process fake connection with NO live socket.
* :func:`collect_to_shards` wraps a sequence of episodes on one env into
  :mod:`pop_trainer.data.shards` shards on disk (pure given a fake-backed env).
* :class:`CollectionSpec` / :func:`run_worker` / :func:`collect_parallel` are the LIVE,
  parallel-safe orchestration (multiprocessing with the SPAWN start method — never fork; this
  is a YOU MUST in CLAUDE.md). The env (and its socket) is built INSIDE the worker via an
  injected ``env_factory``, so the live path is isolated and NOT unit-tested. The worker target
  is module-level + takes plain data (picklable for spawn).

An episode ends when the env reports ``terminated or truncated`` (its own boundary) or the step
cap is reached; the final pair for that step is still recorded with the zero action. ``step_idx``
resets per episode and ``episode_id`` increases monotonically per worker.

stdlib + numpy + :mod:`pop_trainer.core.state` + :mod:`pop_trainer.env.tank_env` (``TankEnv``)
+ :mod:`pop_trainer.data.{policies,schema,shards}`. No models, no pretraining, no rl, no
tank_twin.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pop_trainer.core import state as state_schema
from pop_trainer.data import policies as policy_mod
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

# A policy is a pure callable. The loop dispatches on the policy's ARITY (decided once via
# ``inspect.signature``), so both shapes plug in without the caller declaring which:
#   policy(state_vector) -> action          (1-arg: idle_policy, ConstantPolicy, the aim policies)
#   policy(state_vector, step) -> action    (2-arg, scripted-by-step: ScriptedCyclePolicy)
PolicyFn = Callable[..., Sequence[float]]


def _policy_takes_step(policy: PolicyFn) -> bool:
    """Whether ``policy`` accepts a second positional ``step`` arg (i.e. is a 2-arg policy).

    Decided ONCE per policy via :func:`inspect.signature` by counting the leading parameters that
    can be passed POSITIONALLY (``POSITIONAL_ONLY`` / ``POSITIONAL_OR_KEYWORD``) up to a
    ``*args``. A keyword-only ``player`` (as on :func:`aim_at_opponent_policy`) is NOT positional,
    so that policy is correctly classified as 1-arg. ``*args`` callables are treated as 2-arg
    (they can take the step). A ``TypeError`` raised inside a correctly-dispatched policy body
    propagates as the real bug it is rather than being masked by an arity retry.

    Defensive: if a callable is not introspectable (e.g. a builtin), default to 1-arg — the safer
    shape that never passes an unexpected ``step``. The policies here are all introspectable.
    """
    try:
        sig = inspect.signature(policy)
    except (ValueError, TypeError):
        return False
    positional = 0
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif param.kind is param.VAR_POSITIONAL:
            return True  # *args can absorb the step
    return positional >= 2


@dataclass
class Sample:
    """One captured step: the time-aligned ``(frame, state, action)`` plus provenance.

    * ``frame`` — ``(H, W, 3)`` uint8 RGB, the env observation for ``state``.
    * ``state`` — the paired 52-float wire state (``info["state"]``).
    * ``action`` — ``(2, 5)`` float: ``[agent, opponent]`` x ``[mx, my, ax, ay, fire]``. The
      agent (P1) slot is the action APPLIED via ``env.step`` to advance the sim (the action the
      inverse-render objective pairs with the resulting next frame). The opponent (P2) slot is
      the zero action: the env draws the opponent internally and does not surface it. On the LAST
      recorded step of an episode (the boundary step) no further action is applied, so the whole
      ``(2, 5)`` is the zero action.
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


def _apply_policy(policy: PolicyFn, state_vec, step: int, *, takes_step: bool) -> list[float]:
    """Call ``policy`` with the call shape its (precomputed) arity dictates, validate the action.

    ``takes_step`` is the arity decided ONCE per policy by :func:`_policy_takes_step` (see
    :func:`run_episode`, which computes it before the step loop). A 2-arg policy is called
    ``policy(state, step)``; a 1-arg policy ``policy(state)``. A ``TypeError`` raised inside a
    correctly-dispatched policy body propagates as the real bug it is.
    """
    action = policy(state_vec, step) if takes_step else policy(state_vec)
    return policy_mod.validate_action(action)


def _zero_action() -> np.ndarray:
    """The ``(2, 5)`` zero action recorded on a boundary step (no action applied after it)."""
    return np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)


def run_episode(
    env: TankEnv,
    *,
    p1_policy: PolicyFn,
    map_id: int,
    episode_id: int,
    max_steps: int,
    seed: int | None = None,
) -> EpisodeResult:
    """PURE step loop: drive ONE ``TankEnv`` for an episode and capture time-aligned samples.

    Calls ``env.reset(seed=seed)`` to start a fresh round (the seed makes the env's opponent draw
    reproducible, so the trajectory is a deterministic function of ``p1_policy`` + ``seed``), then
    each step:

    1. records the CURRENT ``(frame, state)``,
    2. if the env has reported its boundary (``terminated or truncated``) or the cap is reached,
       stops — recording the zero action for that final step (no action is applied after it),
    3. otherwise computes the agent action ``a1`` from ``p1_policy`` on the current state, records
       it (P1 slot; the opponent slot is zero) with the current sample, calls
       ``env.step(a1)``, takes the next ``(frame, state)`` from the returned obs / ``info``, and
       loops.

    The transport lives inside ``env``, so this is unit-testable against a real ``TankEnv`` built
    over an in-process fake connection. Returns an :class:`EpisodeResult`.
    """
    result = EpisodeResult()
    obs, info = env.reset(seed=seed)
    frame = obs
    vec = info["state"]
    step_idx = 0
    done = False
    # Decide the policy's arity ONCE, here, by introspecting the actual callable wired in for
    # THIS episode — so the classification cannot leak across episodes/policies. The flag is then
    # reused for every step.
    p1_takes_step = _policy_takes_step(p1_policy)
    while True:
        at_cap = step_idx >= max_steps - 1
        if done or at_cap:
            # Final recorded step: no further action is applied; record the zero action.
            result.samples.append(
                _make_sample(frame, vec, _zero_action(), map_id, episode_id, step_idx)
            )
            result.ended_done = done
            return result

        a1 = _apply_policy(p1_policy, vec, step_idx, takes_step=p1_takes_step)
        action = np.array([a1, [0.0] * schema.ACTION_LEN], dtype=np.float32)
        result.samples.append(_make_sample(frame, vec, action, map_id, episode_id, step_idx))

        obs, _reward, terminated, truncated, info = env.step(a1)
        frame = obs
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
    p1_policy: PolicyFn,
    max_steps: int,
    seed: int | None = None,
    shard_prefix: str = "shard_w0",
    shard_size: int = 10_000,
    with_actions: bool = True,
) -> list[Path]:
    """Run one episode per entry of ``map_ids`` on ``env`` and write shards to ``out_dir``.

    Each episode calls ``env.reset`` (via :func:`run_episode`) to start a fresh round; ``map_ids``
    tags each episode's samples with its map. The per-episode seed is ``seed + episode_id`` (when
    a base ``seed`` is given) so each episode's opponent stream is distinct yet reproducible.
    Samples accumulate and flush to a shard every ``shard_size``, plus a final flush. Returns the
    list of written shard paths. Pure given a fake-backed ``env`` (the live socket path lives in
    :func:`run_worker`).
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
            p1_policy=p1_policy,
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
    INSIDE the worker via ``env_factory`` (a picklable callable, e.g. a module-level function),
    so NO socket / env / ``Connection`` crosses the spawn boundary. The policy is likewise built
    inside the worker from ``policy_factory``.
    """

    worker_id: int
    out_dir: str
    map_ids: list[int]
    max_steps: int
    seed: int | None = None
    shard_size: int = 10_000
    with_actions: bool = True
    # Picklable factories, called INSIDE the worker process:
    #   env_factory(spec) -> a TankEnv (socket opened in-worker).
    #   policy_factory(spec) -> the agent (P1) policy.
    env_factory: Callable[[CollectionSpec], object] | None = None
    policy_factory: Callable[[CollectionSpec], PolicyFn] | None = None
    extra: dict = field(default_factory=dict)


def run_worker(spec: CollectionSpec) -> dict:
    """Module-level worker entry point (picklable for spawn). Builds the env IN-WORKER.

    Builds the ``TankEnv`` via ``spec.env_factory`` (its socket opened HERE, never inherited) and
    the agent policy via ``spec.policy_factory``, then delegates to :func:`collect_to_shards`.
    Closes the env in a ``finally`` so the worker that opened the socket also releases it. Returns
    a small result dict (worker id, shard file names, count). Live path — exercised by the
    orchestrator, not the unit tests.
    """
    if spec.env_factory is None:
        raise ValueError("run_worker needs an env_factory")
    if spec.policy_factory is None:
        raise ValueError("run_worker needs a policy_factory")
    env = spec.env_factory(spec)
    p1_policy = spec.policy_factory(spec)
    try:
        written = collect_to_shards(
            env,
            out_dir=spec.out_dir,
            map_ids=spec.map_ids,
            p1_policy=p1_policy,
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
