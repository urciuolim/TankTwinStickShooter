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

Player2 injection (the league / self-play seam): ``player2`` is an injected
:class:`pop_trainer.core.agent.Agent` (any object exposing ``act(obs) -> action``). The env
NEVER imports ``agents`` — when no ``player2`` is supplied it falls back to the trivial
built-in :class:`_RandomPlayer2` defined here, which draws a uniform action from the env's own
seeded ``np_random`` generator. On every step the env computes player2's OWN first-person view
(the perspective-flipped 52-float state via :func:`core.state.split_state_for_opponent`) and
calls ``player2.act(view)`` to obtain player2's action; BOTH players' actions are captured on
the env and surfaced in ``info`` (``info["p1_action"]`` / ``info["p2_action"]``) and stored as
``self.last_p1_action`` / ``self.last_p2_action``.

Seam-unchanged rationale: the default ``_RandomPlayer2`` draws from the SAME ``self.np_random``
generator the env already seeds, with the SAME ``rng.uniform(ACTION_LOW, ACTION_HIGH,
ACTION_DIM)`` call. The default-player2 step therefore produces a byte-identical ``"2"`` action
stream for a given seed; only the SOURCE of ``a2`` is now an injected agent, never the wire
shape. ``player2``-as-learner / true 2-agent obs+reward is a FUTURE mode — the seam is left
open, not built.

Wire contract (byte-identical to the frozen RL seam; see ``core.protocol``):

* handshake: ``{"restart": True}`` -> ack, ``{"start": True}`` -> a ``"starting"`` ack, then
  an OPTIONAL one-time ``{"type": "walls", ...}`` map-layout message (sent only when an arena
  with a "Walls" block is configured), then the first ``{"state": [...52...]}`` + its trailing
  pixel frame.
* step: ``{1: a1_list, 2: a2_list}`` (INTEGER keys; ``json.dumps`` coerces them to ``"1"`` /
  ``"2"``), then the next ``state`` + frame. ``a1`` is the player1 action passed to ``step``;
  ``a2`` is ``player2``'s action.
* inbound state may carry ``"winner"`` (int: ``0`` == player1 won, ``1`` == player2, ``-1`` ==
  draw) and/or ``"done"``.

Map-layout tracking: Unity is the source of truth for the static wall geometry. When the
handshake delivers the optional walls message the env parses it (``core.protocol``) and stores
it on ``self.current_map`` (a :class:`pop_trainer.core.protocol.WallLayout`), surfaced in the
reset ``info`` as ``info["map"]`` (and in step ``info`` as ``info["map"]``). It stays ``None``
when no arena is configured. The env NEVER re-parses arena JSON.

Episode boundaries (gymnasium 5-tuple; see :mod:`pop_trainer.env.rewards`):

* ``terminated`` — the game decided the round (a winner or a bare ``done``).
* ``truncated`` — ``max_steps`` reached on an undecided step, OR a LOST CONNECTION.
* a dropped connection (``core.protocol`` raises ``ConnectionError``) is translated to a
  ``truncated`` step with reward ``0.0`` and ``info["lost_connection"] = True`` — the game
  is counted as ending with no winner, NOT a crash.

Testability: the transport is INJECTED, exactly like ``core.protocol.Connection``. Pass a
``connection`` (a ready ``Connection`` over an in-process fake transport) or a
``connection_factory`` (a zero-arg callable returning one, re-invoked on reconnect). Tests
drive a fake socket with canned ``state`` + frame bytes — NO live Unity build / socket
required. The real production path (a TCP socket to a launched Unity build) is out of scope
for this component and is supplied by the caller via ``connection_factory``.

Survivor / self-play perspective: player2's first-person view is available via
:meth:`player2_frame` (R/B channel swap, :func:`core.state.flip_frame_perspective`) and
:meth:`player2_state` (52-float half swap, :func:`core.state.split_state_for_opponent`) so the
league / self-play path can compose them without re-deriving the transforms.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from pop_trainer.core import agent as core_agent
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.core.protocol import (
    Connection,
    WallLayout,
    is_walls_message,
    parse_walls_message,
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


class _RandomPlayer2:
    """The trivial built-in player2: a uniform random action, ignoring its observation.

    Implements the :class:`pop_trainer.core.agent.Agent` protocol (``act(obs) -> action``).
    The env falls back to this when no ``player2`` agent is injected, which keeps ``env`` free
    of any ``agents`` import. It is handed a zero-arg ``rng_source`` returning the env's LIVE
    ``np_random`` generator, and re-reads it on every ``act`` — gymnasium's seeded ``reset``
    REPLACES that generator object, so reading it lazily is what makes the draw track the most
    recent seed. The result is a ``rng.uniform(ACTION_LOW, ACTION_HIGH, ACTION_DIM)`` draw off
    the env's single seeded source.
    """

    def __init__(self, rng_source: Callable[[], np.random.Generator]):
        self._rng_source = rng_source

    def act(self, obs):
        """Return a uniform action in ``[ACTION_LOW, ACTION_HIGH]`` of length ``ACTION_DIM``.

        ``obs`` is ignored: this player2 makes no use of its view of the world. The draw comes
        from the env's current ``np_random`` generator (re-read via the injected source).
        """
        return self._rng_source().uniform(ACTION_LOW, ACTION_HIGH, ACTION_DIM)


def _coerce_action(action) -> list[float]:
    """Clip ``action`` into ``[ACTION_LOW, ACTION_HIGH]`` and force length ``ACTION_DIM``.

    A minimal local guard for any injected player2 action (env is ``core``-only and does not
    use the ``agents`` validator): coerces to a 1-D ``float64`` array, fits it to ``ACTION_DIM``
    (pads with zeros / truncates), clips to the action range, and returns a plain
    ``list[float]`` ready for the wire ``.tolist()`` shape. ``float64`` is deliberate — the
    built-in random player2 draws ``float64`` and this preserves its exact wire bytes (the
    frozen-seam reproducibility contract); no narrowing cast is introduced.
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
      socket OR an in-process fake transport). Used as-is; on a lost connection there is no
      reconnect (the next step would re-raise) unless a ``connection_factory`` is also given.
    * ``connection_factory`` — a zero-arg callable returning a fresh ``Connection``. Called
      on construction (if no ``connection`` was passed) and AGAIN to reconnect after a
      dropped connection (the reconnect-on-``ConnectionError`` path).

    Exactly one of ``connection`` / ``connection_factory`` must be provided.

    Args:
        connection: a ready ``Connection`` (the simplest test seam).
        connection_factory: a zero-arg callable returning a ``Connection`` (enables
            reconnect after a dropped connection).
        env_config: an :class:`pop_trainer.core.config.EnvConfig` (only ``max_steps`` is used
            by the env loop; the socket-address fields belong to the caller's factory).
            Defaults to ``EnvConfig()``.
        reward_config: a :class:`pop_trainer.core.config.RewardConfig` (win / loss / time
            budgets). Defaults to ``RewardConfig()``.
        frame_shape: the ``(H, W, 3)`` shape of the rendered pixel frame, fixing
            ``observation_space``. Must match what Unity renders. Defaults to
            :data:`DEFAULT_FRAME_SHAPE`.
        player2: the injected :class:`pop_trainer.core.agent.Agent` driving player2. When
            ``None`` (the default) the env uses the built-in :class:`_RandomPlayer2` seeded
            off ``self.np_random``, preserving the byte-identical default action stream.
        survivor: survivor-mode terminal flip (threaded into the reward).
        seed: optional default seed (also overridable per ``reset``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        connection: Connection | None = None,
        connection_factory: Callable[[], Connection] | None = None,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        player2: core_agent.Agent | None = None,
        survivor: bool = False,
        seed: int | None = None,
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
        self.conn: Connection | None = (
            connection if connection is not None else connection_factory()
        )

        self.env_config = env_config if env_config is not None else EnvConfig()
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.max_steps = self.env_config.max_steps
        self.survivor = survivor
        self._default_seed = seed

        self.frame_shape = tuple(frame_shape)
        # Observation = the real rendered pixel frame; state is in info, not the obs space.
        self.observation_space = spaces.Box(low=0, high=255, shape=self.frame_shape, dtype=np.uint8)
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

        # The injected player2 agent. When none is given, the built-in random player2 draws off
        # the env's own seeded generator. It reads ``self.np_random`` lazily through a source
        # closure (gymnasium's seeded ``reset`` REPLACES that generator object), so its action
        # stream is reproducible under a seeded ``reset`` and byte-identical to a direct
        # ``self.np_random.uniform`` draw.
        self.player2 = player2 if player2 is not None else _RandomPlayer2(lambda: self.np_random)

        # Per-episode state, set on reset.
        self._frame = np.zeros(self.frame_shape, dtype=np.uint8)
        self._raw_state: list[float] | None = None
        self.step_counter = 0
        self.last_winner = -1
        # The static wall layout of the current map, parsed from the optional one-time walls
        # message in the handshake. None until a walls message arrives (no arena configured).
        self.current_map: WallLayout | None = None
        # The actual length-5 actions sent on the wire on the most recent successful step.
        self.last_p1_action: list[float] | None = None
        self.last_p2_action: list[float] | None = None

    # --- gymnasium API -------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Run the restart/start/first-state handshake; return ``(obs, info)``.

        Seeds ``self.np_random`` via ``super().reset`` (so the built-in player2 action draw is
        reproducible). Performs the handshake (``{"restart": True}`` -> ack,
        ``{"start": True}`` -> ``"starting"`` ack), then reads the first ``state`` + pixel
        frame. The observation is the frame; the 52-float state is in ``info["state"]``.

        On a dropped connection during the handshake, reconnects via the
        ``connection_factory`` (if one was given) and retries ONCE; a second failure raises
        ``ConnectionError`` (reset cannot return a valid first observation without the game).
        """
        if seed is None:
            seed = self._default_seed
        super().reset(seed=seed)

        try:
            obs, info = self._handshake_and_first_state()
        except ConnectionError:
            # A dropped connection during reset: reconnect (if possible) and retry once.
            self._reconnect()
            obs, info = self._handshake_and_first_state()
        return obs, info

    def _handshake_and_first_state(self):
        """Send the restart/start handshake and read the first ``state`` + frame.

        After the ``"starting"`` ack, the game MAY send a one-time ``{"type": "walls", ...}``
        map-layout message (only when an arena with a "Walls" block is configured) as its own
        discrete message BEFORE the first state. The env reads one JSON object and routes by
        its ``"type"`` tag: a walls message is parsed + stored on ``self.current_map``, then the
        first ``state`` + frame is read; otherwise the object already IS the first state and only
        its trailing pixel frame is read (so an absent walls message never consumes a state).
        """
        self.conn.send({"restart": True})
        self.conn.receive()  # restart ack
        self.conn.send({"start": True})
        ack = self.conn.receive()  # start ack
        if "starting" not in ack:
            raise RuntimeError(f"unexpected start ack from game: {ack!r}")

        # One JSON object: either the optional walls message or the first state itself.
        message = self.conn.receive()
        if is_walls_message(message):
            self.current_map = parse_walls_message(message)
            received, frame = self.conn.receive_state_and_frame()
        else:
            # The object already held is the first state; pair it with its trailing frame.
            received = message
            frame = self.conn.receive_frame()

        self._raw_state = list(received["state"])
        self._frame = np.asarray(frame, dtype=np.uint8)
        self.step_counter = 0
        self.last_winner = -1
        return self._frame, {"state": self._raw_state, "map": self.current_map}

    def step(self, action):
        """Send ``{1: a1, 2: a2}``, read the next ``state`` + frame, return the gymnasium
        5-tuple ``(obs, reward, terminated, truncated, info)``.

        ``a1`` is the player1 action passed in, sent exactly as ``action.tolist()`` (the
        unchanged player1 wire path). ``a2`` is ``player2``'s action: the env builds player2's
        OWN first-person view of the
        CURRENT state — the perspective-flipped 52-float vector via
        :func:`pop_trainer.core.state.split_state_for_opponent` of ``self._raw_state`` — and
        calls ``self.player2.act(view)``. A state-reading player2 reads player1 of its own view
        (the self-play convention); the built-in random player2 ignores the view. player2's
        returned action is coerced to the same ``[-1, 1]`` length-5 wire shape.

        Both wire actions are captured: ``self.last_p1_action`` / ``self.last_p2_action`` hold
        the lists sent this step, and they are surfaced in ``info["p1_action"]`` /
        ``info["p2_action"]`` for the data layer.

        The reward / boundary is computed by
        :func:`pop_trainer.env.rewards.shaped_step_reward`: a per-step time penalty accrues
        every step and the win/loss terminal is ADDED on the decided step.

        ``terminated`` is a decided game (winner / ``done``); ``truncated`` is ``max_steps``
        reached OR a dropped connection. On a dropped connection the step truncates with
        reward ``0.0`` and ``info["lost_connection"] = True`` (the game is counted as ending
        with no winner); ``last_p1_action`` / ``last_p2_action`` keep their last good values and
        the action keys are omitted from that step's ``info``. The env reconnects (if a
        ``connection_factory`` was given) so the next ``reset`` can start a fresh episode.
        """
        a1 = np.asarray(action, dtype=np.float32).tolist()

        # player2 acts on ITS first-person view: the perspective-flipped current state, with
        # player2's own block presented in the player1 slot (the self-play convention).
        player2_view = self.player2_state()
        a2 = _coerce_action(self.player2.act(player2_view))

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
            self._reconnect()
            return self._frame, reward, terminated, truncated, {"lost_connection": True}

        self._raw_state = list(received["state"])
        self._frame = np.asarray(frame, dtype=np.uint8)
        self.step_counter += 1
        self.last_p1_action = a1
        self.last_p2_action = a2

        winner = int(received["winner"]) if "winner" in received else None
        done = bool("done" in received)
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

    def _reconnect(self):
        """Replace ``self.conn`` with a fresh connection from the factory, if one was given.

        The reconnect-on-``ConnectionError`` path. A no-op when the env was constructed with a
        bare ``connection`` (no factory) — the next wire op would then re-raise, which is the
        documented behaviour for that construction mode.
        """
        if self._connection_factory is not None:
            self.conn = self._connection_factory()
