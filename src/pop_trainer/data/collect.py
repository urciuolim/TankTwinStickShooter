"""The collection driver: drive the game through ``TankEnv`` and capture samples.

Collection routes through :class:`pop_trainer.env.tank_env.TankEnv` — the SAME gymnasium
observation pipeline RL trains on — so the pretraining ``(frame, state)`` rows are byte-for-byte
the observations the policy will later see, with no drift between pretraining inputs and RL
observations. Collection pairs a ``player1`` agent + a ``player2`` agent (both
:mod:`pop_trainer.agents` policies), BOTH driven HERE (the env is a bare pure transport that owns
neither player). The driver hands ``player1.act`` the current 52-float state — which for player1
is the UNFLIPPED state (player1 already occupies the ``PLAYER_1`` slot) — and hands ``player2.act``
its FLIPPED first-person view (computed here via :func:`core.state.split_state_for_opponent`),
then passes BOTH actions to ``env.step(a1, a2)``. The two halves of the self-play pair therefore
each see their OWN first-person view.

BOTH players' actions are captured: ``env.step`` surfaces ``info["p1_action"]`` (the action
actually sent for player1, equal to what ``player1.act`` returned) and ``info["p2_action"]``
(the ``a2`` the driver passed). The recorded ``(2, 5)`` action array stores BOTH — there is no
zeroed-player2 slot.

LAYERING for testability:

* :func:`run_episode` is the PURE step loop over an INJECTED ``env`` (a bare ``TankEnv`` with a
  ``reset()`` / ``step(action, opponent_action)`` returning the gymnasium tuples) plus the
  ``player1`` + ``player2`` agents. It reads the first ``(obs, info)`` from ``env.reset``, then
  each step computes both actions from the current state, calls ``env.step(a1, a2)``, and records
  the CURRENT ``(frame, state)`` paired with BOTH actions that advanced t -> t+1 (taken from
  ``info``) — stopping on ``terminated or truncated`` or the step cap. The transport lives inside
  the env, so this loop is unit-testable against a real ``TankEnv`` built over an in-process fake
  connection with NO live socket.
* :func:`collect_to_shards` drives an ``episode_plan`` (a deterministic round-robin of
  (map x pairing) :class:`EpisodePlan` episodes) on ONE long-lived env — each episode switches
  to its arena and pairs its two selectors from a ``selector -> agent`` pool — into
  :mod:`pop_trainer.data.shards` shards on disk (pure given a fake-backed env).
* :class:`CollectionSpec` / :func:`run_worker` / :func:`collect_parallel` are the LIVE,
  parallel-safe orchestration (multiprocessing with the SPAWN start method — never fork; this
  is a YOU MUST in CLAUDE.md). The bare env and the agent pool are built INSIDE the worker via
  injected factories, so the live path is isolated and NOT unit-tested. The worker target is
  module-level + takes plain data (the ``episode_plan`` + ``map_index`` are plain; picklable for
  spawn).

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
import queue as queue_mod
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pop_trainer import agents
from pop_trainer.core import agent as core_agent
from pop_trainer.core import state as state_schema
from pop_trainer.data import schema, shards
from pop_trainer.env.tank_env import TankEnv

__all__ = [
    "SHARD_BYTES_BUDGET",
    "SPIKE_FACTOR",
    "MARGIN",
    "default_shard_size",
    "estimate_peak_buffer_bytes",
    "check_memory_budget",
    "Sample",
    "EpisodeResult",
    "EpisodePlan",
    "run_episode",
    "collect_to_shards",
    "CollectionSpec",
    "run_worker",
    "collect_parallel",
]


# ===========================================================================================
# MEMORY SAFETY — the per-worker in-RAM buffer is the OOM surface, not the on-disk shard.
# ===========================================================================================
# Each buffered sample holds an UNCOMPRESSED (H, W, 3) uint8 frame. ``collect_to_shards``
# buffers up to ``shard_size`` samples per worker before flushing, so a worker's live buffer
# peaks at ``frame.nbytes * shard_size`` bytes — independent of how tiny the shards deflate to
# on disk (flat frames compress ~140x, so disk size badly under-reports RAM cost). A literal
# ``shard_size`` (e.g. 10_000) at a 0.69 MB frame is a ~6.9 GB-per-worker buffer, which OOMs a
# multi-worker box. We bound the BUFFER by a BYTE budget instead of a sample count, so the cap
# auto-adapts to the frame resolution: smaller frames -> more samples per shard, larger frames
# -> fewer, but the per-worker buffer stays ~budget bytes regardless.
SHARD_BYTES_BUDGET = 384 * 1024 * 1024  # ~384 MB per-worker in-RAM buffer ceiling.

# The ``savez_compressed`` flush copies/compresses the whole buffer, so peak RAM transiently
# exceeds the buffer itself. SPIKE_FACTOR folds that compress transient into the pre-flight
# estimate so the guard reasons about the true peak, not just the steady-state buffer.
SPIKE_FACTOR = 2

# The pre-flight estimate EXCLUDES the ~1 GB/worker live Unity instances + the OS, so the guard
# only lets the estimate consume a conservative fraction of available RAM, leaving headroom for
# everything it does not model.
MARGIN = 0.6


def default_shard_size(frame_nbytes: int) -> int:
    """The frame-size-aware default shard size from the byte budget (>= 1).

    Returns ``SHARD_BYTES_BUDGET // frame_nbytes`` (floored, clamped to at least 1) so a
    worker's in-RAM buffer (``frame_nbytes * shard_size``) is bounded to ~``SHARD_BYTES_BUDGET``
    regardless of frame resolution. For the 360x640x3 (= 691200 B) pixel frame this yields ~582
    samples per shard — far under the count-based default that caused the OOM, while keeping the
    on-disk schema byte-compatible (just MORE, SMALLER shards). Raises ``ValueError`` on a
    non-positive ``frame_nbytes``.
    """
    if frame_nbytes <= 0:
        raise ValueError(f"frame_nbytes must be positive, got {frame_nbytes}")
    return max(1, SHARD_BYTES_BUDGET // int(frame_nbytes))


def estimate_peak_buffer_bytes(frame_nbytes: int, shard_size: int, workers: int) -> int:
    """Estimate the PEAK in-RAM bytes a collection run buffers across all workers.

    ``frame_nbytes * shard_size`` is one worker's steady-state buffer ceiling; ``* workers``
    sums the concurrent per-worker buffers; ``* SPIKE_FACTOR`` folds in the ``savez_compressed``
    flush transient. Pure + unit-testable (no allocation). This is the quantity the pre-flight
    guard compares against available RAM.
    """
    return int(frame_nbytes) * int(shard_size) * int(workers) * SPIKE_FACTOR


def check_memory_budget(
    *,
    frame_nbytes: int,
    shard_size: int,
    workers: int,
    available_bytes: int,
    allow_oversized: bool = False,
) -> str:
    """Pre-flight memory guard. Returns the human-readable estimate line; may raise ``MemoryError``.

    Computes the peak buffer estimate (:func:`estimate_peak_buffer_bytes`) and compares it to a
    conservative fraction (:data:`MARGIN`) of ``available_bytes`` (INJECTED by the caller — this
    pure function never touches psutil). When the estimate exceeds the threshold and
    ``allow_oversized`` is ``False`` it raises ``MemoryError`` with an ACTIONABLE message (the
    estimate, the available RAM, the budget threshold, and the literal guidance to reduce
    ``--workers`` or ``--shard-size`` or pass ``--allow-oversized``). When within budget — or when
    overridden — it returns the estimate line so the caller can ALWAYS print it at startup.
    """
    estimate = estimate_peak_buffer_bytes(frame_nbytes, shard_size, workers)
    threshold = int(available_bytes * MARGIN)
    gib = 1024**3
    line = (
        f"memory estimate: peak buffer ~{estimate / gib:.2f} GB "
        f"({workers} worker(s) x {shard_size} samples x {frame_nbytes / 1024**2:.2f} MB/frame "
        f"x {SPIKE_FACTOR}x compress spike); "
        f"available {available_bytes / gib:.2f} GB, "
        f"budget {MARGIN:.0%} = {threshold / gib:.2f} GB"
    )
    if estimate > threshold and not allow_oversized:
        raise MemoryError(
            f"{line}. ABORT: estimated peak buffer ~{estimate / gib:.2f} GB exceeds the "
            f"{MARGIN:.0%} budget ({threshold / gib:.2f} GB of {available_bytes / gib:.2f} GB "
            f"available). reduce --workers or --shard-size (or pass --allow-oversized to override)."
        )
    return line


@dataclass
class Sample:
    """One captured step: the time-aligned ``(frame, state, action)`` plus provenance.

    * ``frame`` — ``(H, W, 3)`` uint8 RGB, the env observation for ``state``.
    * ``state`` — the paired 52-float wire state (``info["state"]``).
    * ``action`` — ``(2, 5)`` float: ``[player1, player2]`` x ``[mx, my, ax, ay, fire]``. Both
      rows are the actions APPLIED via ``env.step`` to advance the sim to the next frame: row 0
      is ``info["p1_action"]`` (player1's applied action) and row 1 is ``info["p2_action"]``
      (player2's applied action, the ``a2`` the driver passed). On the LAST recorded step of an
      episode (the boundary step) no further action is applied, so the whole ``(2, 5)`` is the
      zero action.
    * ``map_id`` — the int index (into the dataset ``maps`` list) of the arena Unity ACTUALLY
      loaded this episode (resolved from the ECHOED walls layout; see :func:`resolve_map_tag`).
      It is the split GROUP KEY — a whole map never crosses a train/val boundary.
    * ``episode_id`` / ``step_idx`` — provenance.
    """

    frame: np.ndarray
    state: np.ndarray
    action: np.ndarray
    map_id: int
    episode_id: int
    step_idx: int


@dataclass
class EpisodeResult:
    """The samples captured in one episode, plus how it ended and HOW the map was tagged.

    * ``ended_done`` — whether the episode ended on the env's own boundary (vs the step cap).
    * ``tag_source`` — how each sample's ``map_id`` was resolved (the F5 tag-from-echo audit):

      * ``"echo"`` — the int came from the ECHOED ``info["map"].map_id`` looked up in the
        rotation index (the normal, authoritative case: Unity confirmed which arena it loaded);
      * ``"fallback_no_echo"`` — no echo was available (``info["map"]`` was ``None``, e.g. a
        walls-absent arena) so the INTENDED ``map_id`` was used;
      * ``"fallback_unknown_echo"`` — an echo arrived but its path was NOT in the rotation index
        (a desync between what the runner targeted and what Unity loaded) so the INTENDED
        ``map_id`` was used. A run seeing this should be treated as suspect, not silently trusted.

      With no index supplied at all (``map_index=None``, the single-map / no-rotation path) the
      tag is always the intended ``map_id`` and ``tag_source`` is ``"intended"``.
    * ``echoed_map_id`` — the raw echoed arena path (``info["map"].map_id``) when one arrived,
      else ``None``; recorded for debugging a desync.
    """

    samples: list[Sample] = field(default_factory=list)
    ended_done: bool = False
    tag_source: str = "intended"
    echoed_map_id: str | None = None

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class EpisodePlan:
    """One scheduled episode: switch to ``switch_arena`` and pair ``player1`` vs ``player2``.

    A round-robin (map x pairing) schedule is a list of these — each describes exactly one
    episode :func:`collect_to_shards` runs in order. It is PLAIN data (selector names + an
    arena-path string + the intended int), so a whole plan crosses the spawn boundary inside
    ``CollectionSpec.extra`` with no live agent / env.

    * ``switch_arena`` — the arena-path string handed to ``env.reset(options={"switch_arena":
      ...})``; ``None`` means a no-switch reset (today's single-map behavior, byte-identical).
    * ``player1`` / ``player2`` — agent SELECTOR NAMES (looked up in an agent pool the worker
      builds once); the pairing for this episode.
    * ``intended_map_id`` — the int this episode WOULD be tagged with if the echo is missing /
      unknown (the documented fallback); the echo, when present + known, wins over it.
    """

    switch_arena: str | None
    player1: str
    player2: str
    intended_map_id: int


def resolve_map_tag(
    layout: object,
    *,
    intended_map_id: int,
    map_index: dict[str, int] | None,
) -> tuple[int, str, str | None]:
    """Resolve an episode's int ``map_id`` from the ECHOED walls layout (the F5 tag-from-echo).

    The F5 rule: the on-disk ``map_id`` int MUST reflect the arena Unity ACTUALLY loaded, not the
    runner's intent. ``layout`` is ``info["map"]`` (a :class:`core.protocol.WallLayout` or
    ``None``); its ``.map_id`` is the echoed arena path. With a ``map_index`` (a ``path -> int``
    map over the rotation set) the echo wins:

    * echo present AND its path is in the index -> ``(index[echo], "echo", echo)``;
    * echo present but its path is NOT in the index -> the intended int with
      ``"fallback_unknown_echo"`` (a desync — flagged, never silently mis-tagged);
    * no echo (``layout is None`` or it has no ``.map_id``) -> the intended int with
      ``"fallback_no_echo"``.

    With ``map_index is None`` (single-map / no rotation) the tag is always the intended int and
    the source is ``"intended"``. Returns ``(map_id, tag_source, echoed_map_id)``.
    """
    echoed = getattr(layout, "map_id", None) if layout is not None else None
    if map_index is None:
        return intended_map_id, "intended", echoed
    if echoed is None:
        return intended_map_id, "fallback_no_echo", None
    if echoed in map_index:
        return map_index[echoed], "echo", echoed
    return intended_map_id, "fallback_unknown_echo", echoed


def _maybe_reset(agent: object, seed: int | None) -> None:
    """Call ``agent.reset(seed=seed)`` if the agent exposes ``reset`` (it is OPTIONAL).

    ``core.Agent`` requires only ``act``; ``reset`` belongs to the static-only ``StatefulAgent``
    surface. Stateful / seeded agents (e.g. ``RandomAgent``, ``CoverageAgent``) implement it so
    an episode replays deterministically; pure agents do not, and are left untouched.
    """
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed=seed)


def _maybe_set_map(agent: object, layout) -> None:
    """Call ``agent.set_map(layout)`` if the agent exposes the OPTIONAL hook and ``layout`` exists.

    ``core.Agent`` requires only ``act``; ``set_map`` belongs to the static-only ``StatefulAgent``
    surface. Map-aware agents (the coverage family) implement it to (re)build their grid; map-
    agnostic agents do not, and are left untouched. A ``None`` layout (no arena) is a no-op.
    """
    if layout is None:
        return
    set_map = getattr(agent, "set_map", None)
    if callable(set_map):
        set_map(layout)


def _zero_action() -> np.ndarray:
    """The ``(2, 5)`` zero action recorded on a boundary step (no action applied after it)."""
    return np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)


def run_episode(
    env: TankEnv,
    *,
    player1: core_agent.Agent,
    player2: core_agent.Agent,
    map_id: int,
    episode_id: int,
    max_steps: int,
    seed: int | None = None,
    switch_arena: str | None = None,
    map_index: dict[str, int] | None = None,
) -> EpisodeResult:
    """PURE step loop: drive ONE bare ``TankEnv`` for an episode and capture time-aligned samples.

    ``env`` is a bare pure-transport :class:`TankEnv`; this loop drives BOTH ``player1`` and
    ``player2``. It resets both agents (if either exposes ``reset``) and resets the env to start a
    fresh round, so the whole trajectory is a deterministic function of (player1, player2, seed).

    Arena switching (additive, reset-time only): when ``switch_arena`` is a path string the reset
    is ``env.reset(seed=seed, options={"switch_arena": switch_arena})`` so Unity rotates to that
    arena before the first state. When ``switch_arena is None`` the reset is the byte-identical
    no-switch ``env.reset(seed=seed)`` (today's behavior exactly). This calls ONLY the Layer-1
    seam; no new wire message is introduced here.

    Tag-from-echo (F5): each sample's int ``map_id`` is resolved via :func:`resolve_map_tag` from
    the ECHOED ``info["map"].map_id`` (what Unity actually loaded) looked up in ``map_index`` (a
    ``path -> int`` map over the rotation set) — NOT the runner's intended target. The echo wins;
    a missing echo or an echo whose path is not in the index falls back to ``map_id`` and the
    fallback is recorded on the result (``tag_source`` / ``echoed_map_id``) so a desync is
    detectable. With ``map_index is None`` the tag is always the intended ``map_id``. The arena is
    static within an episode, so the tag is resolved ONCE after reset and applied to every sample.

    Then each step:

    1. records nothing yet — it computes player1's action ``a1 = player1.act(vec)`` from the
       CURRENT 52-float state (player1's own UNFLIPPED view), validates it, computes player2's
       action ``a2 = player2.act(split_state_for_opponent(vec))`` from player2's FLIPPED first-
       person view, and calls ``env.step(a1, a2)``,
    2. records the CURRENT ``(frame, state)`` paired with the BOTH actions that advanced this
       step, read back from ``info`` (row 0 = ``info["p1_action"]`` == ``a1``, row 1 =
       ``info["p2_action"]`` == player2's applied action),
    3. on the env's boundary (``terminated or truncated``) or the step cap, records the final
       ``(frame, state)`` with the zero ``(2, 5)`` action (no action is applied after it).

    The transport lives inside ``env``, so this is unit-testable against a real ``TankEnv`` built
    over an in-process fake connection. Returns an :class:`EpisodeResult`.
    """
    result = EpisodeResult()
    _maybe_reset(player1, seed)
    _maybe_reset(player2, seed)
    if switch_arena is None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset(seed=seed, options={"switch_arena": switch_arena})
    # Hand the static layout to a map-aware agent (the OPTIONAL ``set_map`` hook); both agents are
    # driver-side now. A map-agnostic agent does not expose ``set_map``.
    _maybe_set_map(player1, info.get("map"))
    _maybe_set_map(player2, info.get("map"))
    # F5 tag-from-echo: the arena is static for the episode, so resolve the int tag ONCE from the
    # echoed layout and stamp every sample with it.
    tagged_map_id, result.tag_source, result.echoed_map_id = resolve_map_tag(
        info.get("map"), intended_map_id=map_id, map_index=map_index
    )
    frame = obs
    vec = info["state"]
    step_idx = 0
    done = False
    while True:
        at_cap = step_idx >= max_steps - 1
        if done or at_cap:
            # Final recorded step: no further action is applied; record the zero action.
            result.samples.append(
                _make_sample(frame, vec, _zero_action(), tagged_map_id, episode_id, step_idx)
            )
            result.ended_done = done
            return result

        # player1 acts on its OWN unflipped view; player2 acts on its FLIPPED first-person view
        # (the perspective flip lives here in the driver). The env transports both and surfaces
        # BOTH applied actions in ``info`` after the step.
        a1 = agents.validate_action(player1.act(vec))
        a2 = player2.act(state_schema.split_state_for_opponent(np.asarray(vec)))
        next_obs, _reward, terminated, truncated, info = env.step(a1, a2)

        action = np.array([info["p1_action"], info["p2_action"]], dtype=np.float32)
        result.samples.append(_make_sample(frame, vec, action, tagged_map_id, episode_id, step_idx))

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
    episode_plan: Sequence[EpisodePlan],
    agent_pool: dict[str, core_agent.Agent],
    max_steps: int,
    seed: int | None = None,
    map_index: dict[str, int] | None = None,
    shard_prefix: str = "shard_w0",
    shard_size: int | None = None,
    with_actions: bool = True,
    on_episode_done: Callable[[int], None] | None = None,
) -> list[Path]:
    """Run the ``episode_plan`` (a round-robin of (map x pairing) episodes) on ONE long-lived env.

    ``env`` is a bare pure-transport env launched ONCE on a boot map; each :class:`EpisodePlan`
    rotates it to its ``switch_arena`` and pairs its ``player1`` / ``player2`` selectors. The two
    agents are taken from ``agent_pool`` (a ``selector -> agent`` dict the worker builds ONCE up
    front — so no live agent crosses the spawn boundary and a selector's RNG stream is continuous
    across the episodes it plays). Each episode runs through :func:`run_episode`, which switches the
    arena at reset and tags samples from the ECHOED layout via ``map_index`` (the F5 rule). The
    per-episode seed is ``seed + episode_id`` (when a base ``seed`` is given) so each episode resets
    BOTH agents to a distinct yet reproducible state. Returns the list of written shard paths. Pure
    given a fake-backed ``env`` (the live socket path lives in :func:`run_worker`).

    SHARD SIZE = the in-RAM BUFFER bound (the OOM surface), NOT a file-size knob. Samples
    accumulate and flush to a shard every ``shard_size`` rows, plus a final flush. When
    ``shard_size is None`` (the default) it is derived ONCE from the FIRST captured sample's
    ``frame.nbytes`` via :func:`default_shard_size`, so the per-worker buffer auto-adapts to the
    actual frame resolution and stays bounded to ~:data:`SHARD_BYTES_BUDGET` bytes (the OOM fix). A
    concrete int overrides that (the CLI's ``--shard-size``). The on-disk schema is unchanged either
    way — a smaller bound just yields MORE, SMALLER shards with byte-identical contents.

    ``on_episode_done`` is an OPTIONAL pure observability side-channel: when given, it is called
    once per COMPLETED episode with that episode's sample count (``len(ep)``) — the parent uses it
    to advance a progress bar (in-process) or push a "done" message onto a multiprocessing queue
    (the spawn workers). ``None`` (the default) is a no-op and is the byte-identical, today's exact
    behavior. The callback fires AFTER the episode's samples are buffered but is purely a notify; it
    cannot perturb the shards (it returns nothing and is handed only an int count).

    Raises ``KeyError`` if a plan names a selector absent from ``agent_pool`` (an eager wiring bug,
    not a silent skip).
    """
    out_dir = Path(out_dir)
    buffer: list[Sample] = []
    written: list[Path] = []
    shard_index = 0
    # ``None`` -> derive the buffer bound from the first sample's frame size (set on first flush
    # check, once a sample exists). A concrete int is used as-is.
    effective_shard_size = shard_size

    def flush() -> None:
        nonlocal shard_index, buffer
        if not buffer:
            return
        shard = samples_to_shard(buffer, with_actions=with_actions)
        path = out_dir / f"{shard_prefix}_{shard_index:04d}.npz"
        written.append(shards.write_shard(path, shard))
        shard_index += 1
        buffer = []

    for episode_id, plan in enumerate(episode_plan):
        episode_seed = None if seed is None else seed + episode_id
        ep = run_episode(
            env,
            player1=agent_pool[plan.player1],
            player2=agent_pool[plan.player2],
            map_id=plan.intended_map_id,
            episode_id=episode_id,
            max_steps=max_steps,
            seed=episode_seed,
            switch_arena=plan.switch_arena,
            map_index=map_index,
        )
        buffer.extend(ep.samples)
        if on_episode_done is not None:
            on_episode_done(len(ep))
        if effective_shard_size is None and buffer:
            # Derive the byte-budget buffer bound from the actual frame size, ONCE. After this the
            # per-worker buffer never exceeds ~SHARD_BYTES_BUDGET regardless of frame resolution.
            effective_shard_size = default_shard_size(int(buffer[0].frame.nbytes))
        while effective_shard_size is not None and len(buffer) >= effective_shard_size:
            chunk, buffer = buffer[:effective_shard_size], buffer[effective_shard_size:]
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

    Holds ONLY plain data + import-able factory names. The bare env (and its socket) is constructed
    INSIDE the worker via ``env_factory``; the agent POOL (a ``selector -> agent`` dict covering
    every selector this worker's plan pairs) is built inside the worker via ``agent_pool_factory``.
    So NO socket / env / agent crosses the spawn boundary — only the plain ``episode_plan`` (a list
    of :class:`EpisodePlan`), the ``map_index`` (``arena-path -> int``), and ``extra`` do.

    The ``episode_plan`` is this worker's slice of the deterministic round-robin (map x pairing)
    schedule; ``map_index`` decodes the echoed arena path to the on-disk int (the F5 tag-from-echo
    source) and is persisted to a sidecar so the int is reversible to the arena path.
    """

    worker_id: int
    out_dir: str
    episode_plan: list[EpisodePlan]
    max_steps: int
    seed: int | None = None
    map_index: dict[str, int] = field(default_factory=dict)
    # The in-RAM BUFFER bound (the OOM surface), NOT a file-size knob. ``None`` (the default) means
    # AUTO: :func:`collect_to_shards` derives the byte-budget bound from the first frame's size so
    # the per-worker buffer stays ~SHARD_BYTES_BUDGET. ``None`` pickles, so the spec stays plain /
    # spawn-safe. A concrete int (from ``--shard-size``) overrides the auto bound.
    shard_size: int | None = None
    with_actions: bool = True
    # Picklable factories, called INSIDE the worker process:
    #   env_factory(spec) -> a bare TankEnv (socket opened in-worker).
    #   agent_pool_factory(spec) -> a {selector -> agent} dict covering the plan's pairings.
    env_factory: Callable[[CollectionSpec], object] | None = None
    agent_pool_factory: Callable[[CollectionSpec], dict[str, core_agent.Agent]] | None = None
    extra: dict = field(default_factory=dict)


# The progress queue is passed to a spawn worker NOT through the (plain, picklable)
# ``CollectionSpec`` but through the Pool's ``initializer`` — it lands in this module-level global
# in the child interpreter, and :func:`run_worker` reads it to build its per-episode
# ``on_episode_done``. A global is the spawn-clean way to hand a live ``Manager().Queue()`` proxy to
# every Pool task without pickling it as a task argument (proxies are picklable through ``initargs``
# but NOT cleanly as a normal ``apply_async`` arg). The in-process paths never set it (stays None).
_WORKER_PROGRESS_QUEUE: object | None = None


def _init_worker(queue: object) -> None:
    """Pool ``initializer``: stash the shared progress queue in this child's module global.

    Runs ONCE per spawned worker process (in the fresh child interpreter), before any task. The
    ``queue`` is a ``Manager().Queue()`` proxy created in the parent from the SAME spawn context and
    passed via ``initargs`` — the spawn-safe channel for a live handle (it never rides on the plain
    ``CollectionSpec``). :func:`run_worker` reads it to emit per-episode "done" messages.
    """
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = queue


def run_worker(spec: CollectionSpec) -> dict:
    """Module-level worker entry point (picklable for spawn). Builds the env IN-WORKER.

    Builds the bare ``TankEnv`` via ``spec.env_factory`` (its socket opened HERE, never inherited)
    and the agent pool via ``spec.agent_pool_factory`` (built ONCE so each selector's RNG stream is
    continuous across the episodes it plays), then delegates the worker's ``episode_plan`` to
    :func:`collect_to_shards`. Closes the env in a ``finally`` so the worker that opened the socket
    also releases it. Returns a small result dict (worker id, shard file names, count). Live path —
    exercised by the orchestrator, not the unit tests.

    PROGRESS: if the Pool set this module's ``_WORKER_PROGRESS_QUEUE`` (via :func:`_init_worker`),
    each completed episode pushes a plain ``(worker_id, n_samples)`` message onto it so the PARENT's
    single bar advances. With no queue set (an in-process call outside :func:`collect_parallel`) no
    per-episode reporting happens here. The message is plain data — the queue carries no shard
    bytes, only a count.
    """
    if spec.env_factory is None:
        raise ValueError("run_worker needs an env_factory")
    if spec.agent_pool_factory is None:
        raise ValueError("run_worker needs an agent_pool_factory")
    queue = _WORKER_PROGRESS_QUEUE
    on_episode_done = None
    if queue is not None:
        worker_id = spec.worker_id

        def on_episode_done(n_samples: int) -> None:
            queue.put((worker_id, n_samples))

    env = spec.env_factory(spec)
    agent_pool = spec.agent_pool_factory(spec)
    try:
        written = collect_to_shards(
            env,
            out_dir=spec.out_dir,
            episode_plan=spec.episode_plan,
            agent_pool=agent_pool,
            max_steps=spec.max_steps,
            seed=spec.seed,
            map_index=spec.map_index or None,
            shard_prefix=f"shard_w{spec.worker_id}",
            shard_size=spec.shard_size,
            with_actions=spec.with_actions,
            on_episode_done=on_episode_done,
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


class _Progress:
    """The ONE aggregate collection bar + its running sample total (the only tqdm config site).

    Wraps a single :class:`tqdm.tqdm` so the TTY-disable decision and the per-episode advance live
    in one testable place. ``advance(n_samples)`` ticks the bar by ONE episode and folds
    ``n_samples`` into a running total shown in the postfix. The bar:

    * counts EPISODES (``unit="ep"``) — the bounded, accurate unit (step counts are unbounded);
    * is DISABLED on a non-TTY (``disable=not sys.stderr.isatty()``) so a redirected / captured run
      emits NOTHING (no progress noise in logs / pipes);
    * writes to ``sys.stderr`` and keeps tqdm's default in-place ``\r`` overwrite (one line, no
      competing ``print``).
    """

    def __init__(self, total: int):
        self.total_samples = 0
        self.bar = tqdm(
            total=total,
            disable=not sys.stderr.isatty(),
            file=sys.stderr,
            unit="ep",
            desc="collecting",
        )

    def advance(self, n_samples: int) -> None:
        self.total_samples += int(n_samples)
        self.bar.update(1)
        self.bar.set_postfix(samples=self.total_samples, refresh=False)

    def close(self) -> None:
        self.bar.close()


def collect_parallel(specs: Sequence[CollectionSpec]) -> list[dict]:
    """Run worker specs in parallel using the SPAWN start method (never fork; Windows-safe).

    Uses ``multiprocessing.get_context("spawn")`` explicitly so behavior is identical on Windows
    and POSIX and no parent file descriptors / sockets are inherited. A single spec runs in-process
    (simpler, still spawn-safe). Each ``CollectionSpec`` builds its own env (and socket) inside its
    worker; nothing live crosses the process boundary. Returns the per-worker result dicts, sorted
    by worker id.

    PROGRESS (the ONE aggregate bar; per-episode granularity): the parent owns a single
    :class:`_Progress` bar whose ``total`` is ``sum(len(spec.episode_plan))`` — the episode count
    known upfront across ALL workers. It is the only bar (never one-per-worker) and is a pure
    observability side-channel: it cannot alter a single shard byte (workers report only an int
    count). How the bar is advanced differs by path:

    * SINGLE-SPEC (in-process): :func:`run_worker` runs in this process, so the queue stays ``None``
      and the worker's ``on_episode_done`` advances the bar DIRECTLY. A tiny shim object (one with
      ``put``) is set into the module global so the SAME ``run_worker`` code path drives the bar in
      process with no second process; it is restored to ``None`` after.
    * MULTI-SPEC (spawn): a ``Manager().Queue()`` (from the SAME spawn context) is handed to every
      child via the Pool ``initializer`` (NOT pickled into the spec; a Manager queue is the spawn-
      safe, initializer-passable proxy — a raw ``ctx.Queue()`` cannot be passed as a normal task
      arg). Workers are submitted with ``apply_async`` (NOT blocking ``map``) and the parent DRAINS
      the queue, advancing the bar per "done" message, until every ``AsyncResult`` is ``.ready()``
      AND the queue is empty. A short ``get`` timeout bounds the poll so the loop neither deadlocks
      nor busy-spins. The bar is CLOSED before returning so the caller's summary prints cleanly.
    """
    specs = list(specs)
    if not specs:
        return []
    progress = _Progress(total=sum(len(spec.episode_plan) for spec in specs))
    try:
        if len(specs) == 1:
            results = [_run_single_spec(specs[0], progress)]
        else:
            results = _run_multi_spec(specs, progress)
    finally:
        progress.close()
    return sorted(results, key=lambda r: r["worker_id"])


class _DirectBarQueue:
    """In-process shim with a ``put`` matching the spawn queue's, draining straight into the bar.

    Lets the single-spec path reuse the SAME :func:`run_worker` reporting code (which writes
    ``(worker_id, n_samples)`` to whatever sits in ``_WORKER_PROGRESS_QUEUE``) while advancing the
    parent's bar directly — no real multiprocessing queue, no second process.
    """

    def __init__(self, progress: _Progress):
        self._progress = progress

    def put(self, message: tuple[int, int]) -> None:
        _worker_id, n_samples = message
        self._progress.advance(n_samples)


def _run_single_spec(spec: CollectionSpec, progress: _Progress) -> dict:
    """Run the single spec in-process, advancing ``progress`` directly per completed episode.

    Installs a :class:`_DirectBarQueue` into the module global so ``run_worker``'s per-episode
    reporting drives the parent's bar, then restores the global to ``None`` so no live handle
    lingers.
    """
    global _WORKER_PROGRESS_QUEUE
    previous = _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = _DirectBarQueue(progress)
    try:
        return run_worker(spec)
    finally:
        _WORKER_PROGRESS_QUEUE = previous


# The drain poll timeout: short enough that the loop reacts promptly when the last worker finishes,
# long enough that an idle parent is not busy-spinning on an empty queue.
_DRAIN_POLL_TIMEOUT = 0.1


def _run_multi_spec(specs: list[CollectionSpec], progress: _Progress) -> list[dict]:
    """Submit every worker async on a spawn Pool and DRAIN the progress queue into ``progress``.

    The queue is a ``Manager().Queue()`` from the spawn context, handed to children via the Pool
    ``initializer`` (:func:`_init_worker`) — never pickled into a spec. Workers are submitted with
    ``apply_async`` (so the parent drains WHILE they run, unlike blocking ``map``). The drain loop
    polls the queue with a short timeout, advancing the bar one tick per per-episode message, and
    exits only once every ``AsyncResult`` is ``.ready()`` AND the queue is fully drained — so no
    message is lost and the loop neither deadlocks nor busy-spins.
    """
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    try:
        with ctx.Pool(processes=len(specs), initializer=_init_worker, initargs=(queue,)) as pool:
            async_results = [pool.apply_async(run_worker, (spec,)) for spec in specs]
            while True:
                drained = _drain_progress_queue(queue, progress, block=True)
                if not drained and all(r.ready() for r in async_results):
                    # A final non-blocking sweep catches any message enqueued between the last
                    # drain's timeout and the ready() check, then we stop.
                    _drain_progress_queue(queue, progress, block=False)
                    break
            # ``get`` re-raises any worker exception; collect the results in submission order.
            results = [r.get() for r in async_results]
    finally:
        manager.shutdown()
    return results


def _drain_progress_queue(queue: object, progress: _Progress, *, block: bool) -> bool:
    """Pull every currently-available message off ``queue``, advancing ``progress`` per message.

    With ``block=True`` the first pull waits up to :data:`_DRAIN_POLL_TIMEOUT` for a message (so an
    idle parent does not busy-spin); subsequent pulls are non-blocking until the queue is empty.
    With ``block=False`` every pull is non-blocking. Returns whether ANY message was drained.
    """
    drained = False
    first = True
    while True:
        try:
            if first and block:
                message = queue.get(timeout=_DRAIN_POLL_TIMEOUT)
            else:
                message = queue.get_nowait()
        except queue_mod.Empty:
            return drained
        first = False
        _worker_id, n_samples = message
        progress.advance(n_samples)
        drained = True
