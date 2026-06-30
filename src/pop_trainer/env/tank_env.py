"""``TankEnv`` — the Gymnasium environment over the Unity simulator socket.

This is the SWAP POINT of the trainer: everything downstream (models / rl) speaks the
gymnasium API, so a different simulator could implement the same interface unchanged. The
environment wraps the Unity sim through :mod:`pop_trainer.core.protocol`'s strict-JSON wire
client and pixel-frame channel; it imports ``core`` only (plus gymnasium + numpy), never
torch / sb3 / tank_twin / any sibling component.

Observation model (the load-bearing decision): the OBSERVATION is the REAL rendered pixel
frame Unity sends on the wire (an ``(H, W, 3)`` uint8 array read via
:meth:`pop_trainer.core.protocol.Connection.receive_state_and_frame`), NOT a Python-drawn
grid. The 52-float wire state is the supervised-decode OBJECTIVE, not the observation, so it
is surfaced in the gymnasium ``info`` dict (``info["state"]``) for any consumer that wants
it — it is NOT part of ``observation_space``.

Pure transport (the symmetric self-play model): the env owns NEITHER player. It transports the
step message ``{1: a1, 2: a2}`` where BOTH actions come from the CALLER. ``a1`` is the player1
action passed to ``step``; ``a2`` is the ``opponent_action`` the caller supplies (a no-op zero
action when omitted). BOTH players' actions are captured on the env and surfaced in ``info``
(``info["p1_action"]`` / ``info["p2_action"]``) and stored as ``self.last_p1_action`` /
``self.last_p2_action``. The DRIVER (the data-collection / play loop) owns each agent, computes
player2's perspective-flipped first-person view via
:func:`core.state.split_state_for_opponent`, and passes the resulting ``a2`` to ``step``.

Seam-unchanged rationale: the SOURCE of ``a2`` moved to the caller, but the WIRE SHAPE is
identical. ``a1`` is still ``np.asarray(action, dtype=np.float32).tolist()`` (the unchanged
player1 path); ``a2`` is still coerced through ``_coerce_action`` (the float64 wire-shape +
reproducibility contract); the keys are still the integer 1 / 2. When ``opponent_action`` is
omitted the env sends a well-formed no-op ``2: [0.0, 0.0, 0.0, 0.0, 0.0]``. ``player2``-as-
learner / true 2-agent obs+reward is a FUTURE mode — the seam is left open, not built.

Wire contract (byte-identical to the frozen RL seam; see ``core.protocol``):

* handshake: ``{"restart": True}`` -> ack, ``{"start": True}`` -> a ``"starting"`` ack, then
  an OPTIONAL one-time ``{"type": "walls", ...}`` map-layout message (sent only when an arena
  with a "Walls" block is configured), then the first ``{"state": [...52...]}`` + its trailing
  pixel frame.
* step: ``{1: a1_list, 2: a2_list}`` (INTEGER keys; ``json.dumps`` coerces them to ``"1"`` /
  ``"2"``), then the next ``state`` + frame. ``a1`` is the player1 action passed to ``step``;
  ``a2`` is the caller's ``opponent_action`` (a zero no-op action when omitted).
* inbound state may carry ``"winner"`` (int: ``0`` == player1 won, ``1`` == player2, ``-1`` ==
  draw) and/or ``"done"``.

Map-layout tracking: Unity is the source of truth for the static wall geometry. When the
handshake delivers the optional walls message the env parses it (``core.protocol``) and stores
it on ``self.current_map`` (a :class:`pop_trainer.core.protocol.WallLayout`), surfaced in the
reset ``info`` as ``info["map"]`` (and in step ``info`` as ``info["map"]``). It stays ``None``
when no arena is configured. The env NEVER re-parses arena JSON.

Map rotation (additive, reset-time only): ``reset(options={"switch_arena": <arena_path>})``
requests a map change in Unity's ``!ingame`` window — AFTER the restart ack and BEFORE the
``{"start": True}`` send. The env sends the switch via
:meth:`pop_trainer.core.protocol.Connection.switch_arena`, reads the ``{"arena_switched": true}``
ack, then routes the OPTIONAL walls message Unity emits after the ack (present only when the new
arena has a "Walls" block) with the same ``"type"``-tag routing used after the start ack. A reset
WITHOUT this option is byte-identical to the no-switch handshake; the switch is purely additive
and never touches the per-step wire. The walls layout last received (after the switch ack or the
start ack) is stored on ``self.current_map`` so ``info["map"]`` reflects the switched arena.

Round boundary — Python NEVER sends ``restart`` mid-round (the wedge fix):

The episode boundary is driven OFF UNITY'S OWN CLOCK, not off a Python-side step cap. Unity
ends the round itself (a winner, OR its round-timer expiring) and emits ``done``/``winner`` on
that step, then sits in a "round-over, waiting-for-restart" state still SERVICING the socket.
That ``done`` is what makes the step ``terminated``; SB3 then calls :meth:`reset` which sends
``{"restart": True}`` — now landing in Unity's clean waiting state, NEVER mid-round. So a
``restart`` is only ever sent from :meth:`reset` (and :meth:`close`), and only when Unity has
already finished the round. ``max_steps`` is NOT the normal episode boundary; it is a GENEROUS
SAFETY CAP set ABOVE Unity's round length (Unity's ``game_maxTime`` / ``ai_actionFreq`` bounds
the round to ~300 RL steps; :data:`DEFAULT_MAX_STEPS` is 600 so the cap never preempts Unity).

Safety-cap corner case: if ``max_steps`` ever fires (a stuck round that Unity never decided),
:meth:`step` truncates the episode but does NOT itself send any ``restart`` — it just returns
``truncated``. The subsequent :meth:`reset` is what sends the ``restart``; even though Unity is
technically still mid-round there, the redesigned Unity driver handles a ``restart`` from the
mid-round read WITHOUT the synchronous-scene-reload collapse (it acks, abandons the round, and
defers the next-round load), so the handshake stays clean. The env never emits a RAW mid-round
restart on its own.

Episode boundaries (gymnasium 5-tuple; see :mod:`pop_trainer.env.rewards`):

* ``terminated`` — Unity decided the round (a winner or a bare ``done``). The NORMAL boundary.
* ``truncated`` — the ``max_steps`` SAFETY cap reached on an undecided step, OR a LOST
  CONNECTION. The safety truncation never sends a restart by itself (see above).
* a dropped connection (``core.protocol`` raises ``ConnectionError``) is translated to a
  ``truncated`` step with reward ``0.0`` and ``info["lost_connection"] = True`` — the game
  is counted as ending with no winner, NOT a crash.

Testability: the transport is INJECTED, exactly like ``core.protocol.Connection``. Pass a
``connection`` (a ready ``Connection`` over an in-process fake transport) or a
``connection_factory`` (a zero-arg callable returning one, re-invoked on reconnect). Tests
drive a fake socket with canned ``state`` + frame bytes — NO live Unity build / socket
required. The real production path (a TCP socket to a launched Unity build) is out of scope
for this component and is supplied by the caller via ``connection_factory``.

Lazy training-instance lifecycle: with a bare ``connection`` the env is "running" from
construction (``self.conn`` is that connection). With a ``connection_factory`` the env
constructs WITHOUT launching anything -- ``self.conn`` is ``None`` ("not running") and the
factory is invoked LAZILY on the first ``reset`` (guarded again in ``step``) via
:meth:`_ensure_connected`. :attr:`is_running` (``self.conn is not None``) is queryable at any
time. This lets a caller construct many envs cheaply and pay the Unity-launch cost only when a
training instance is actually driven.

Release / re-launch: :meth:`release` HARD-KILLS the live Unity child (via the injected reap
hook, see below) and frees the connection WITHOUT the graceful end handshake -- the point is to
reclaim a stalled instance that would not answer the handshake. The env OBJECT stays alive and
fully usable: the next ``reset`` re-launches lazily via the factory. ``release`` is idempotent.

Reap hook (the boundary-safe shared kill primitive): the env never imports a subprocess reap.
The caller injects ``reap`` -- a ``Callable[[object], None]`` that kills a ``Popen``-like
handle -- and the env reads the live process handle off
``getattr(self.conn, "_launch_proc", None)`` (the attribute the caller's ``connection_factory``
stashes on each produced ``Connection``). :meth:`release` and the kill-old-first
:meth:`_reconnect` are the ONLY two callers of the hook. With ``reap=None`` (the pure-test
default) there is no real process behind the stub, so both paths skip the proc-kill and only
close the transport.

Logfile safety: the env NEVER writes, truncates, or re-points any Unity ``-logFile`` (that arg
lives entirely in the caller's launch command). ``release`` / ``_reconnect`` only kill the
process and close the socket -- the prior instance's C# log is left intact for post-mortem.

Survivor / self-play perspective: player2's first-person view is available via
:meth:`player2_frame` (R/B channel swap, :func:`core.state.flip_frame_perspective`) and
:meth:`player2_state` (52-float half swap, :func:`core.state.split_state_for_opponent`) so the
league / self-play path can compose them without re-deriving the transforms.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.core.logging_setup import LAYER_ENV
from pop_trainer.core.protocol import (
    Connection,
    WallLayout,
    is_walls_message,
    parse_walls_message,
    state_message_is_valid,
)
from pop_trainer.env.rewards import shaped_step_reward, time_penalty_per_step

# The action vector is 5 floats in [-1, 1]: [move_x, move_y, aim_x, aim_y, fire].
ACTION_DIM = 5
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default rendered-frame shape (H, W, 3). The wire frame's true dimensions are not known
# until the first frame arrives; observation_space must be fixed up front, so the caller
# declares the expected shape (it must match what Unity renders). The default is a small
# placeholder for tests / introspection — production passes the build's real frame shape.
DEFAULT_FRAME_SHAPE = (36, 60, 3)

# Generous SAFETY cap on env steps per episode — NOT the normal round boundary. Unity ends
# the round on its OWN clock (a winner, or its round-timer): with game_maxTime=60,
# ai_fixedDeltaTime=0.02, ai_actionFreq=10 the round is ceil(60/0.02)/10 = ~300 RL steps. This
# cap is set ABOVE that (600) so it never preempts Unity's ``done`` in the normal case; it only
# trips on a round Unity somehow never decided, and even then the env does NOT send a raw
# mid-round restart (the next reset's restart advances Unity through the waiting-state handshake).
# Used as the env's fallback when no ``env_config`` is supplied; a caller that passes an
# ``EnvConfig`` should likewise keep ``max_steps`` strictly above Unity's round length.
DEFAULT_MAX_STEPS = 600


def _ms_since(t0: float) -> float:
    """Elapsed milliseconds since the monotonic timestamp ``t0`` (observability durations only)."""
    return (time.monotonic() - t0) * 1000.0


def _coerce_action(action) -> list[float]:
    """Clip ``action`` into ``[ACTION_LOW, ACTION_HIGH]`` and force length ``ACTION_DIM``.

    The wire-shape coercion for the caller's ``opponent_action`` (env is ``core``-only and does
    not use the ``agents`` validator): coerces to a 1-D ``float64`` array, fits it to
    ``ACTION_DIM`` (pads with zeros / truncates), clips to the action range, and returns a plain
    ``list[float]`` ready for the wire. ``float64`` is deliberate — it is the reproducibility
    contract for the ``a2`` path; no narrowing cast is introduced.
    """
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    fitted = np.zeros(ACTION_DIM, dtype=np.float64)
    n = min(arr.shape[0], ACTION_DIM)
    fitted[:n] = arr[:n]
    return np.clip(fitted, ACTION_LOW, ACTION_HIGH).tolist()


class TankEnv(gymnasium.Env):
    """Single-agent, pixel-observation gymnasium env over the Unity socket.

    Construct with an injected connection so the env is testable with no Unity:

    * ``connection`` — a ready :class:`pop_trainer.core.protocol.Connection` (over a real
      socket OR an in-process fake transport). Used as-is, "running" from construction; on a
      lost connection there is no reconnect (the next step would re-raise) unless a
      ``connection_factory`` is also given.
    * ``connection_factory`` — a zero-arg callable returning a fresh ``Connection``. NOT called
      on construction (the env starts "not running", ``self.conn is None``); invoked LAZILY on
      the first ``reset`` / ``step`` to launch+connect, and AGAIN to reconnect after a dropped
      connection (the kill-old-first reconnect path) and after :meth:`release`.

    Exactly one of ``connection`` / ``connection_factory`` must be provided.

    Args:
        connection: a ready ``Connection`` (the simplest test seam).
        connection_factory: a zero-arg callable returning a ``Connection`` (enables lazy launch
            + reconnect after a dropped connection + re-launch after :meth:`release`).
        reap: OPTIONAL ``Callable[[object], None]`` that kills a ``Popen``-like build process.
            Injected by the caller so the env can hard-kill the live Unity child WITHOUT
            importing a subprocess reap (boundary: ``env`` imports ``core`` only). The env reads
            the handle off ``getattr(self.conn, "_launch_proc", None)`` and calls ``reap(proc)``
            in :meth:`release` and the kill-old-first :meth:`_reconnect`. ``None`` (the default /
            pure-test path) skips the proc-kill (no real process behind a stub) and only closes
            the transport.
        env_config: an :class:`pop_trainer.core.config.EnvConfig` (only ``max_steps`` is used
            by the env loop; the socket-address fields belong to the caller's factory).
            ``max_steps`` is the GENEROUS SAFETY cap, NOT the round boundary (Unity's clock is)
            — keep it strictly above Unity's round length. Defaults to
            ``EnvConfig(max_steps=DEFAULT_MAX_STEPS)`` (600).
        reward_config: a :class:`pop_trainer.core.config.RewardConfig` (win / loss / time
            budgets). Defaults to ``RewardConfig()``.
        frame_shape: the ``(H, W, 3)`` shape of the rendered pixel frame, fixing
            ``observation_space``. Must match what Unity renders. Defaults to
            :data:`DEFAULT_FRAME_SHAPE`.
        survivor: survivor-mode terminal flip (threaded into the reward).
        seed: optional default seed (also overridable per ``reset``).
        logger: OPTIONAL observability logger (default ``None`` = no logging, behavior-identical
            to today, allocation-free on the hot path). When present the env logs its
            reset/step/episode milestones at the env layer. The SAME logger should be threaded into
            the :class:`~pop_trainer.core.protocol.Connection` (by the caller's
            ``connection_factory``) so protocol + env records for one socket share the
            ``env-<role>-<port>.log`` file.
        role: the connection's role tag (``"train"`` / ``"eval"``) surfaced in the env's log
            records; purely observational, defaults to ``"train"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        connection: Connection | None = None,
        connection_factory: Callable[[], Connection] | None = None,
        reap: Callable[[object], None] | None = None,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        survivor: bool = False,
        seed: int | None = None,
        logger: logging.Logger | None = None,
        role: str = "train",
    ):
        super().__init__()

        if (connection is None) == (connection_factory is None):
            raise ValueError(
                "provide exactly one of `connection` or `connection_factory` "
                "(connection_factory enables reconnect after a dropped connection)"
            )
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError(f"frame_shape must be (H, W, 3), got {frame_shape!r}")

        self._connection_factory = connection_factory
        self._reap = reap
        # Lazy lifecycle: a bare `connection` is "running" from construction; a factory is NOT
        # invoked here (conn stays None / "not running") and is called lazily on the first
        # reset/step via _ensure_connected. So constructing with a factory launches no Unity.
        self.conn: Connection | None = connection

        # No env_config -> default the SAFETY cap to DEFAULT_MAX_STEPS (above Unity's ~300-step
        # round), not EnvConfig's bare 300; an explicitly supplied env_config is honored as-is.
        self.env_config = (
            env_config if env_config is not None else EnvConfig(max_steps=DEFAULT_MAX_STEPS)
        )
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.max_steps = self.env_config.max_steps
        self.survivor = survivor
        self._default_seed = seed

        # OPTIONAL observability. None = no logging (the default behavior-identical path).
        self.logger = logger
        self.role = role

        self.frame_shape = tuple(frame_shape)
        # Observation = the real rendered pixel frame; state is in info, not the obs space.
        self.observation_space = spaces.Box(low=0, high=255, shape=self.frame_shape, dtype=np.uint8)
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

        # Per-episode state, set on reset.
        self._frame = np.zeros(self.frame_shape, dtype=np.uint8)
        self._raw_state: list[float] | None = None
        self.step_counter = 0
        self.last_winner = -1
        # Monotonic episode counter (observability only; surfaced in the reset/episode log records).
        self._episode = 0
        # The static wall layout of the current map, parsed from the optional one-time walls
        # message in the handshake. None until a walls message arrives (no arena configured).
        self.current_map: WallLayout | None = None
        # The actual length-5 actions sent on the wire on the most recent successful step.
        self.last_p1_action: list[float] | None = None
        self.last_p2_action: list[float] | None = None

    # --- observability helper (no-op when self.logger is None) ---------------------------

    def _log(self, level: int, event: str, **detail) -> None:
        """Emit one env-layer observability record at ``level`` if a logger is attached; else no-op.

        The ``self.logger is None`` early return keeps the default (no-logger) path allocation-free.
        ``layer`` is tagged ``env`` so env records are distinguishable from the protocol records the
        shared :class:`~pop_trainer.core.protocol.Connection` writes to the same file.
        """
        if self.logger is None:
            return
        self.logger.log(level, event, extra={"detail": {"layer": LAYER_ENV, **detail}})

    # --- lifecycle state -----------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` when a live ``Connection`` is held, ``False`` after construction-with-factory
        (before the first ``reset``) or after :meth:`release` / :meth:`close`."""
        return self.conn is not None

    # --- gymnasium API -------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Run the restart/start/first-state handshake; return ``(obs, info)``.

        Seeds ``self.np_random`` via ``super().reset``. LAZILY launches+connects the training
        instance (:meth:`_ensure_connected`) if the env is "not running" — the first ``reset``
        after construction-with-factory (or after :meth:`release`) is what actually starts Unity.
        Then performs the handshake (``{"restart": True}`` -> ack, ``{"start": True}`` ->
        ``"starting"`` ack) and reads the first ``state`` + pixel frame. The observation is the
        frame; the 52-float state is in ``info["state"]``.

        On a dropped connection during the handshake, reconnects via the
        ``connection_factory`` (kill-old-first, if one was given) and retries ONCE; a second
        failure raises ``ConnectionError`` (reset cannot return a valid first observation
        without the game).
        """
        if seed is None:
            seed = self._default_seed
        super().reset(seed=seed)

        switch_target = options.get("switch_arena") if options else None

        self._ensure_connected()
        self._episode += 1
        self._log(
            logging.INFO,
            "reset_begin",
            episode=self._episode,
            switch_arena=switch_target,
        )
        try:
            obs, info = self._handshake_and_first_state(switch_target)
        except ConnectionError:
            # A dropped connection during reset: reconnect (if possible) and retry once.
            self._log(logging.WARNING, "reset_lost_connection", episode=self._episode)
            self._reconnect()
            obs, info = self._handshake_and_first_state(switch_target)
        map_layout = info.get("map")
        self._log(
            logging.INFO,
            "reset_end",
            episode=self._episode,
            map=getattr(map_layout, "map_id", None),
        )
        return obs, info

    def _handshake_and_first_state(self, switch_target=None):
        """Send the restart/(switch)/start handshake and read the first ``state`` + frame.

        With ``switch_target`` ``None`` (a plain reset) this is the byte-identical no-switch
        handshake: ``{"restart": True}`` ack, ``{"start": True}`` -> ``"starting"`` ack, then the
        OPTIONAL one-time walls message (present only when an arena with a "Walls" block is
        configured) followed by the first ``state`` + frame.

        With ``switch_target`` set (a map-rotation reset) the env requests the map change in
        Unity's ``!ingame`` window — AFTER the restart ack and BEFORE the ``{"start": True}`` send.
        The Unity wire ordering is DETERMINISTIC per case (see ``DriverController.cs``):

        * the switch branch writes ``{"arena_switched": true}`` then (only for a walls arena) a
          walls message — BEFORE the start send;
        * the start branch then writes ``{"starting": true}`` then (only for a walls arena) a walls
          message, then the state + frame.

        So the wire for a switch-to-walls reset is: ``arena_switched``, ``walls`` (switch),
        ``starting``, ``walls`` (start), ``state``, ``frame`` — with the start ack INTERLEAVED
        between the two walls messages. The switch branch's walls is written BEFORE the start ack
        but is read AFTER the start send (the read order does not have to match the per-branch
        write boundary — TCP is a stream). Each inbound object is routed by its ``"type"`` tag via
        :meth:`_drain_optional_walls`: a walls message is parsed + stored (last wins), the first
        non-walls object is the next handshake message. An absent walls message never consumes a
        state, so this is correct for every switch/no-switch x walls-present/absent combination.
        """
        t0 = time.monotonic()
        self.conn.send({"restart": True})
        self._receive_restart_ack()
        self._log(logging.INFO, "handshake_restart_ack", elapsed_ms=_ms_since(t0))

        # !ingame window (after the restart ack, before the start send): optional map change.
        # switch_arena reads ONLY the {"arena_switched": true} ack. The switch branch's optional
        # walls message (present only for a walls arena) is left on the wire and drained below,
        # AFTER the start send, ahead of the start ack — a walls-absent switch wrote nothing there,
        # so nothing is mis-consumed.
        if switch_target is not None:
            t_switch = time.monotonic()
            self.conn.switch_arena(switch_target)
            self._log(
                logging.INFO,
                "handshake_switch_arena_ack",
                elapsed_ms=_ms_since(t_switch),
                switch_arena=switch_target,
            )

        t_start = time.monotonic()
        self.conn.send({"start": True})

        # Drain the switch branch's optional walls (if any) then the start ack it stops on.
        ack = self._drain_optional_walls()
        if "starting" not in ack:
            raise RuntimeError(f"unexpected start ack from game: {ack!r}")
        self._log(logging.INFO, "handshake_start_ack", elapsed_ms=_ms_since(t_start))

        # Drain the start branch's optional walls (if any) then the first state it stops on; pair
        # the state with its trailing pixel frame.
        t_state = time.monotonic()
        received = self._drain_optional_walls()
        frame = self.conn.receive_frame()
        self._log(logging.INFO, "handshake_first_state", elapsed_ms=_ms_since(t_state))

        self._raw_state = list(received["state"])
        self._frame = np.asarray(frame, dtype=np.uint8)
        self.step_counter = 0
        self.last_winner = -1
        return self._frame, {"state": self._raw_state, "map": self.current_map}

    def _drain_optional_walls(self):
        """Read JSON objects, storing any leading walls message(s), and return the first non-walls.

        Reads one object at a time and routes by its ``"type"`` tag: a walls message is parsed +
        stored on ``self.current_map`` (the LAST one wins) and the loop continues; the FIRST
        non-walls object is RETURNED (the next handshake message — a ``starting`` ack or the first
        state — never mis-read as walls). An absent walls message therefore never consumes a state.
        """
        while True:
            message = self.conn.receive()
            if not is_walls_message(message):
                return message
            self.current_map = parse_walls_message(message)

    def _receive_restart_ack(self):
        """Read the restart ack, DRAINING any leading stray mid-round ``state`` (+ its frame) first.

        In the NORMAL flow the env only sends ``restart`` after a ``done``-terminated step, so Unity
        is already in the waiting state and the very next inbound object is the restart ack. In the
        SAFETY-CAP corner case (the ``max_steps`` cap truncated an undecided round, then ``reset``
        sends ``restart`` while Unity is technically still mid-round) Unity's per-step send/read
        order means it emits ONE more ``state`` (+ pixel frame) BEFORE it reads the restart and
        acks. That stray state+frame would otherwise be mis-read as the ack and desync the wire. So
        we skip any leading state message — consuming its trailing frame too — and return on the
        first non-state object (the restart ack). This keeps the round-over -> restart handshake
        clean even in the corner case, with NO raw mid-round restart ever emitted by the env.
        """
        while True:
            message = self.conn.receive()
            if not state_message_is_valid(message):
                return message
            # A stray mid-round state carries a trailing pixel frame on the wire; consume it so the
            # next receive lands on the ack, not the binary frame bytes.
            self.conn.receive_frame()

    def step(self, action, opponent_action=None):
        """Send ``{1: a1, 2: a2}``, read the next ``state`` + frame, return the gymnasium
        5-tuple ``(obs, reward, terminated, truncated, info)``.

        Pure transport: BOTH actions come from the CALLER. ``a1`` is the player1 action passed
        in, sent exactly as ``np.asarray(action, dtype=np.float32).tolist()`` (the unchanged
        player1 wire path). ``a2`` is the caller's ``opponent_action`` coerced to the ``[-1, 1]``
        length-5 wire shape via :func:`_coerce_action`; when ``opponent_action`` is ``None`` the
        env sends a no-op zero action ``[0.0, 0.0, 0.0, 0.0, 0.0]``. The DRIVER owns player2 and
        is responsible for computing its perspective-flipped view (see
        :meth:`player2_state` / :func:`pop_trainer.core.state.split_state_for_opponent`) before
        passing ``a2`` here.

        Both wire actions are captured: ``self.last_p1_action`` / ``self.last_p2_action`` hold
        the lists sent this step, and they are surfaced in ``info["p1_action"]`` /
        ``info["p2_action"]`` for the data layer.

        The reward / boundary is computed by
        :func:`pop_trainer.env.rewards.shaped_step_reward`: a per-step time penalty accrues
        every step and the win/loss terminal is ADDED on the decided step.

        ``terminated`` is a decided game (Unity's ``winner`` / ``done`` — the NORMAL boundary);
        ``truncated`` is the ``max_steps`` SAFETY cap reached OR a dropped connection. The
        safety-cap truncation does NOT itself send any ``restart`` — it only returns
        ``truncated``; the next :meth:`reset` is the sole sender of ``restart`` (the env never
        emits a raw mid-round restart). On a dropped connection the step truncates with reward
        ``0.0`` and ``info["lost_connection"] = True`` (the game is counted as ending with no
        winner); ``last_p1_action`` / ``last_p2_action`` keep their last good values and the
        action keys are omitted from that step's ``info``. The env reconnects (if a
        ``connection_factory`` was given) so the next ``reset`` can start a fresh episode.

        Guards lazy lifecycle: if the env is "not running" (stepped before any ``reset``, or
        after :meth:`release`) :meth:`_ensure_connected` launches+connects via the factory first.
        """
        self._ensure_connected()
        a1 = np.asarray(action, dtype=np.float32).tolist()

        # a2's source is the caller: the opponent action (a no-op zero action when omitted),
        # coerced to the [-1, 1] length-5 wire shape. The driver owns player2.
        if opponent_action is None:
            opponent_action = np.zeros(ACTION_DIM)
        a2 = _coerce_action(opponent_action)

        message = {1: a1, 2: a2}

        time_penalty = time_penalty_per_step(self.reward_config.time_total, self.max_steps)

        try:
            self.conn.send(message)
            received, frame = self.conn.receive_state_and_frame()
        except ConnectionError:
            reward, terminated, truncated = shaped_step_reward(
                lost_connection=True,
                survivor=self.survivor,
                time_penalty=time_penalty,
                win_reward=self.reward_config.win_reward,
                loss_reward=self.reward_config.loss_reward,
            )
            self._log(
                logging.INFO,
                "step_lost_connection",
                episode=self._episode,
                step=self.step_counter,
            )
            self._reconnect()
            return self._frame, reward, terminated, truncated, {"lost_connection": True}

        self._raw_state = list(received["state"])
        self._frame = np.asarray(frame, dtype=np.uint8)
        self.step_counter += 1
        self.last_p1_action = a1
        self.last_p2_action = a2
        self._log(logging.DEBUG, "step", episode=self._episode, step=self.step_counter)

        winner = int(received["winner"]) if "winner" in received else None
        done = bool("done" in received)
        # SAFETY cap only (above Unity's round length): a truncation here returns truncated but
        # sends NO restart. The next reset's restart is what advances Unity. Unity's own done is
        # the normal boundary and is checked above via winner/done.
        max_steps_reached = self.step_counter >= self.max_steps

        reward, terminated, truncated = shaped_step_reward(
            winner=winner,
            done=done,
            survivor=self.survivor,
            time_penalty=time_penalty,
            win_reward=self.reward_config.win_reward,
            loss_reward=self.reward_config.loss_reward,
            max_steps_reached=max_steps_reached,
        )

        info: dict = {
            "state": self._raw_state,
            "p1_action": a1,
            "p2_action": a2,
            "map": self.current_map,
        }
        if winner is not None:
            info["winner"] = winner
            self.last_winner = winner

        # Unambiguous game-result tag on a DECIDED terminal (terminated only; a truncation is
        # not a decided game). Pure free-form gymnasium info — it does not touch the wire.
        if terminated:
            if winner == S.PLAYER_1:
                info["outcome"] = "win"
            elif winner is not None and winner != -1:
                info["outcome"] = "loss"
            else:
                info["outcome"] = "draw"

        if terminated or truncated:
            self._log(
                logging.INFO,
                "episode_end",
                episode=self._episode,
                step=self.step_counter,
                terminated=terminated,
                truncated=truncated,
                winner=winner,
                outcome=info.get("outcome"),
            )

        return self._frame, reward, terminated, truncated, info

    def render(self):
        """Not implemented (``metadata['render_modes'] == []``)."""
        raise NotImplementedError()

    def close(self):
        """Best-effort end handshake then release the transport (gymnasium lifecycle).

        Sends ``{"restart": True}`` / ``{"end": True}`` so Unity can exit cleanly (errors
        swallowed — a dropped connection or a fake test transport without these replies must
        not block teardown), closes the underlying transport if it exposes ``close()``, and
        nulls the connection. IDEMPOTENT: a second ``close()`` is a no-op.
        """
        if self.conn is not None:
            try:
                self.conn.send({"restart": True})
                self.conn.receive()
                self.conn.send({"end": True})
                self.conn.receive()
            except (ConnectionError, OSError, KeyError, ValueError):
                pass
            transport = getattr(self.conn, "transport", None)
            close = getattr(transport, "close", None)
            if callable(close):
                with contextlib.suppress(OSError):
                    close()
            self.conn = None

    def release(self):
        """Hard-kill the live Unity child and free the connection; the env OBJECT stays alive.

        Reaps the live build process via the injected reap hook (the proc handle is read off
        ``getattr(self.conn, "_launch_proc", None)``; skipped when ``reap`` is ``None`` or no
        proc is stashed — the pure-test path has no real process), then closes the transport
        (best-effort, errors suppressed) and sets ``self.conn = None`` so :attr:`is_running` is
        ``False``.

        UNLIKE :meth:`close`, ``release`` does NOT do the graceful end handshake: a stalled
        instance being hard-killed will not answer it — reclaiming such an instance is the whole
        point. It never touches the prior instance's ``-logFile`` (the env does not own it).

        IDEMPOTENT: a no-op when already "not running" (the reap hook is not called again). The
        env remains fully usable — a subsequent ``reset`` lazily re-launches via the factory.
        """
        if self.conn is None:
            return
        self._log(logging.INFO, "release", episode=self._episode)
        self._free_connection(reap_proc=True)

    # --- self-play perspective helpers ---------------------------------------------------

    def player2_frame(self):
        """player2's first-person (R<->B swapped) view of the current pixel frame.

        Composes :func:`pop_trainer.core.state.flip_frame_perspective` on the latest frame so
        the self-play path can score / act from player2's perspective.
        """
        return S.flip_frame_perspective(self._frame)

    def player2_state(self):
        """player2's first-person 52-float state (the two 26-float halves swapped).

        Composes :func:`pop_trainer.core.state.split_state_for_opponent` on the latest raw
        state, presenting player2's block in the player1 slot of the returned vector. Returns
        ``None`` before the first ``reset`` (no state yet).
        """
        if self._raw_state is None:
            return None
        return S.split_state_for_opponent(np.asarray(self._raw_state))

    # --- internal ------------------------------------------------------------------------

    def _ensure_connected(self):
        """Lazily launch+connect via the factory when the env is "not running".

        Invoked at the top of ``reset`` and as a guard in ``step``. A no-op when already running
        (``self.conn is not None``) OR when there is no factory (a bare ``connection`` that has
        not been released — its first ``reset`` proceeds straight to the handshake). When the env
        is "not running" AND a factory was given, this calls the factory to produce a fresh
        ``Connection`` (the LIVE path launches+connects Unity here, lazily) and assigns
        ``self.conn`` so :attr:`is_running` becomes ``True``.
        """
        if self.conn is None and self._connection_factory is not None:
            self.conn = self._connection_factory()

    def _free_connection(self, *, reap_proc: bool):
        """Kill the live build (optional) and release the current connection; set ``conn = None``.

        The ONE shared kill primitive behind :meth:`release` and :meth:`_reconnect`. When
        ``reap_proc`` is set and a reap hook was injected, it reaps the build process read off
        ``getattr(self.conn, "_launch_proc", None)`` (skipped when ``reap`` is ``None`` or no
        proc is stashed). Then it closes the transport best-effort (errors suppressed) and nulls
        ``self.conn``. Does NOT send the end handshake (the process may be a stalled instance, or
        already dead) and NEVER touches the instance's log file.
        """
        conn = self.conn
        if conn is None:
            return
        if reap_proc and self._reap is not None:
            proc = getattr(conn, "_launch_proc", None)
            if proc is not None:
                self._reap(proc)
        transport = getattr(conn, "transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            with contextlib.suppress(OSError):
                close()
        self.conn = None

    def _reconnect(self):
        """Kill the OLD instance FIRST, then re-launch a fresh connection from the factory.

        The reconnect-on-``ConnectionError`` path. The ORDER is load-bearing: a dropped socket
        may leave the OLD Unity instance alive-but-stalled, still bound to its port; relaunching
        the new build before reaping the old one would collide on bind. So this:

        1. closes the old transport (best-effort) and REAPS the old build process via the reap
           hook FIRST (freeing the port; via :meth:`_free_connection` — skipped when ``reap`` is
           ``None`` / no proc, the pure-test path), then
        2. invokes the factory to launch+connect the NEW build, assigning ``self.conn``.

        A no-op when the env was constructed with a bare ``connection`` (no factory) — the env is
        left "not running" and the next wire op re-raises, which is the documented behaviour for
        that construction mode. Never touches the old instance's ``-logFile``.
        """
        if self._connection_factory is None:
            return
        # 1. Kill the old (possibly stalled) instance to free its port BEFORE relaunching.
        self._free_connection(reap_proc=True)
        # 2. Launch+connect the new build.
        self.conn = self._connection_factory()
