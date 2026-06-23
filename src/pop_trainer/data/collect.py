"""The collection driver: drive the game via ``core.protocol.Connection`` and capture samples.

Drives the Unity simulator DIRECTLY over the wire seam (``core.protocol.Connection`` + the
length-prefixed pixel-frame channel) with DETERMINISTIC policies (:mod:`pop_trainer.data.
policies`), capturing one time-aligned ``(frame, state, action)`` sample per step into shards.
It does NOT route through ``env``'s gym wrapper (collection needs the wire driver, not the RL
reward machinery) and never touches the frozen RL seam beyond the public ``Connection`` API.

LAYERING for testability:

* :func:`run_episode` is the PURE step loop. It takes an ALREADY-CONNECTED, ALREADY-HANDSHAKEN
  ``Connection`` (or any object with ``send`` / ``receive_state_and_frame``), the two policies,
  and a step cap; it sends actions, reads ``(state, frame)`` pairs, records samples, and stops
  on ``done`` or the cap. The transport is INJECTED, so this loop is unit-testable against an
  in-process fake connection with NO live socket. It captures the action APPLIED to produce the
  NEXT observation alongside the CURRENT observation, matching the inverse-render contract.
* :func:`collect_to_shards` wraps a sequence of episodes from a handshaken connection into
  :mod:`pop_trainer.data.shards` shards on disk (pure given a fake connection + a fake
  handshake callable).
* :class:`CollectionSpec` / :func:`run_worker` / :func:`collect_parallel` are the LIVE,
  parallel-safe orchestration (multiprocessing with the SPAWN start method — never fork; this
  is a YOU MUST in CLAUDE.md). The socket-opening transport factory is INJECTED into the worker
  spec, so the live path is isolated and NOT unit-tested (the contract says: do not unit-test
  live collection). The worker target is module-level + takes plain data (picklable for spawn).

A "done" inbound state (the dict carries the ``"done"`` key the build sets at round end) ends
an episode; the final pair for that step is still recorded. ``step_idx`` resets per episode and
``episode_id`` increases monotonically per worker.

stdlib + numpy + :mod:`pop_trainer.core.{protocol,state,config}` + :mod:`pop_trainer.data.
{policies,schema,shards}` only. No env, no models, no tank_twin.
"""

from __future__ import annotations

import contextlib
import inspect
import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pop_trainer.core import state as state_schema
from pop_trainer.data import policies as policy_mod
from pop_trainer.data import schema, shards

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
    (they can take the step). This replaces the old TypeError catch-and-retry, so a ``TypeError``
    raised INSIDE a correctly-arity'd policy body propagates instead of being silently retried.

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

    * ``frame`` — ``(H, W, 3)`` uint8 RGB, the observation Unity rendered for ``state``.
    * ``state`` — the paired 52-float wire state.
    * ``action`` — ``(2, 5)`` float: the [p1, p2] action APPLIED at this step to advance the
      sim (the action the inverse-render objective pairs with the resulting next frame). On the
      LAST recorded step of an episode (the ``done`` step) no further action is applied, so it is
      the zero action.
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
    """The samples captured in one episode, plus whether it ended on a build ``done``."""

    samples: list[Sample] = field(default_factory=list)
    ended_done: bool = False

    def __len__(self) -> int:
        return len(self.samples)


def _apply_policy(policy: PolicyFn, state_vec, step: int, *, takes_step: bool) -> list[float]:
    """Call ``policy`` with the call shape its (precomputed) arity dictates, validate the action.

    ``takes_step`` is the arity decided ONCE per policy by :func:`_policy_takes_step` (see
    :func:`run_episode`, which computes it per policy before the step loop). A 2-arg policy is
    called ``policy(state, step)``; a 1-arg policy ``policy(state)``. No catch-and-retry: a
    ``TypeError`` raised inside a correctly-dispatched policy body propagates as the real bug it
    is, rather than being masked by a 1-arg retry.
    """
    action = policy(state_vec, step) if takes_step else policy(state_vec)
    return policy_mod.validate_action(action)


def run_episode(
    conn,
    *,
    first_state: dict,
    first_frame: np.ndarray,
    p1_policy: PolicyFn,
    p2_policy: PolicyFn,
    map_id: int,
    episode_id: int,
    max_steps: int,
) -> EpisodeResult:
    """PURE step loop: drive ONE handshaken connection and capture time-aligned samples.

    Given the episode's FIRST ``(state, frame)`` (already read by the handshake) plus the two
    deterministic policies and a step cap, this:

    1. records the CURRENT ``(frame, state)``,
    2. if the state is ``done`` (or the cap is reached), stops — recording the zero action for
       that final step (no action is applied after ``done``),
    3. otherwise computes ``a1`` / ``a2`` from the policies, records THAT action with the current
       sample (the action that produces the next frame), sends ``{1: a1, 2: a2}``, reads the next
       ``(state, frame)`` via ``conn.receive_state_and_frame()``, and loops.

    The transport is whatever ``conn`` wraps, so this is unit-testable against an in-process fake
    connection. Returns an :class:`EpisodeResult`.
    """
    result = EpisodeResult()
    state_dict = first_state
    frame = first_frame
    step_idx = 0
    # Decide each policy's arity ONCE, here, by introspecting the actual callable wired in for
    # THIS episode — so the cache cannot leak across episodes/policies (a different policy object
    # gets a fresh classification). The two flags are then reused for every step.
    p1_takes_step = _policy_takes_step(p1_policy)
    p2_takes_step = _policy_takes_step(p2_policy)
    while True:
        vec = state_dict["state"]
        done = "done" in state_dict
        at_cap = step_idx >= max_steps - 1
        if done or at_cap:
            # Final recorded step: no further action is applied; record the zero action.
            action = np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)
            result.samples.append(_make_sample(frame, vec, action, map_id, episode_id, step_idx))
            result.ended_done = done
            return result

        a1 = _apply_policy(p1_policy, vec, step_idx, takes_step=p1_takes_step)
        a2 = _apply_policy(p2_policy, vec, step_idx, takes_step=p2_takes_step)
        action = np.array([a1, a2], dtype=np.float32)
        result.samples.append(_make_sample(frame, vec, action, map_id, episode_id, step_idx))

        conn.send({1: a1, 2: a2})
        state_dict, frame = conn.receive_state_and_frame()
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


# A handshake callable opens/prepares an episode on a given map and returns its first
# (state_dict, frame). The LIVE implementation is supplied by the worker; tests pass a fake.
HandshakeFn = Callable[[object, int], tuple[dict, np.ndarray]]


def collect_to_shards(
    conn,
    *,
    out_dir: str | Path,
    handshake: HandshakeFn,
    map_ids: Sequence[int],
    p1_policy: PolicyFn,
    p2_policy: PolicyFn,
    max_steps: int,
    shard_prefix: str = "shard_w0",
    shard_size: int = 10_000,
    with_actions: bool = True,
) -> list[Path]:
    """Run one episode per entry of ``map_ids`` on ``conn`` and write shards to ``out_dir``.

    ``handshake(conn, map_id)`` prepares the episode (switch arena, restart, start, read first
    state+frame); :func:`run_episode` then drives it. Samples accumulate and flush to a shard
    every ``shard_size``, plus a final flush. Returns the list of written shard paths. Pure given
    a fake ``conn`` + fake ``handshake`` (the live socket path lives in :func:`run_worker`).
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
        first_state, first_frame = handshake(conn, map_id)
        ep = run_episode(
            conn,
            first_state=first_state,
            first_frame=first_frame,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            map_id=map_id,
            episode_id=episode_id,
            max_steps=max_steps,
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

    Holds ONLY plain data + import-able factory names. The transport (socket) and the live
    handshake are constructed INSIDE the worker via ``transport_factory`` / ``handshake_factory``
    (picklable callables, e.g. module-level functions), so NO socket or ``Connection`` crosses
    the spawn boundary. Policies are likewise built inside the worker from ``policy_factory``.
    """

    worker_id: int
    out_dir: str
    map_ids: list[int]
    max_steps: int
    shard_size: int = 10_000
    with_actions: bool = True
    # Picklable factories, called INSIDE the worker process:
    #   transport_factory(spec) -> a Connection (socket opened in-worker).
    #   handshake_factory(spec) -> HandshakeFn.
    #   policy_factory(spec) -> (p1_policy, p2_policy).
    transport_factory: Callable[[CollectionSpec], object] | None = None
    handshake_factory: Callable[[CollectionSpec], HandshakeFn] | None = None
    policy_factory: Callable[[CollectionSpec], tuple[PolicyFn, PolicyFn]] | None = None
    extra: dict = field(default_factory=dict)


def run_worker(spec: CollectionSpec) -> dict:
    """Module-level worker entry point (picklable for spawn). Opens the socket IN-WORKER.

    Builds the Connection via ``spec.transport_factory`` (socket opened HERE, never inherited),
    the handshake via ``spec.handshake_factory``, and the policies via ``spec.policy_factory``,
    then delegates to :func:`collect_to_shards`. Returns a small result dict (worker id, sample
    count, shard file names). Live path — exercised by the orchestrator, not the unit tests.
    """
    if spec.transport_factory is None or spec.handshake_factory is None:
        raise ValueError("run_worker needs transport_factory and handshake_factory")
    if spec.policy_factory is None:
        raise ValueError("run_worker needs a policy_factory")
    conn = spec.transport_factory(spec)
    handshake = spec.handshake_factory(spec)
    p1_policy, p2_policy = spec.policy_factory(spec)
    try:
        written = collect_to_shards(
            conn,
            out_dir=spec.out_dir,
            handshake=handshake,
            map_ids=spec.map_ids,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            max_steps=spec.max_steps,
            shard_prefix=f"shard_w{spec.worker_id}",
            shard_size=spec.shard_size,
            with_actions=spec.with_actions,
        )
    finally:
        # Best-effort socket close: the worker opened it, so the worker closes it.
        transport = getattr(conn, "transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            with contextlib.suppress(OSError):
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
    in-process (simpler, still spawn-safe). Each ``CollectionSpec`` opens its own socket inside
    its worker; nothing live crosses the process boundary. Returns the per-worker result dicts,
    sorted by worker id.
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
