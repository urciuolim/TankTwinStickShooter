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

Wire contract (byte-identical to the frozen RL seam; see ``core.protocol``):

* handshake: ``{"restart": True}`` -> ack, ``{"start": True}`` -> a ``"starting"`` ack,
  then the first ``{"state": [...52...]}`` + its trailing pixel frame.
* step: ``{1: action_list, 2: opp_action_list}`` (INTEGER keys; ``json.dumps`` coerces them
  to ``"1"`` / ``"2"``), then the next ``state`` + frame.
* inbound state may carry ``"winner"`` (int: ``0`` == P1/agent won, ``1`` == opponent,
  ``-1`` == draw) and/or ``"done"``.

Episode boundaries (gymnasium 5-tuple; see :mod:`pop_trainer.env.rewards`):

* ``terminated`` — the game decided the round (a winner or a bare ``done``).
* ``truncated`` — ``max_steps`` reached on an undecided step, OR a LOST CONNECTION.
* a dropped connection (``core.protocol`` raises ``ConnectionError``) is translated to a
  ``truncated`` step with reward ``0.0`` and ``info["lost_connection"] = True`` — the game
  is counted as ending with no winner (the 2021 ``TankEnv`` behaviour), NOT a crash.

Testability: the transport is INJECTED, exactly like ``core.protocol.Connection``. Pass a
``connection`` (a ready ``Connection`` over an in-process fake transport) or a
``connection_factory`` (a zero-arg callable returning one, re-invoked on reconnect). Tests
drive a fake socket with canned ``state`` + frame bytes — NO live Unity build / socket
required. The real production path (a TCP socket to a launched Unity build) is out of scope
for this component and is supplied by the caller via ``connection_factory``.

Survivor / self-play perspective: the opponent's first-person view is available via
:meth:`opp_frame` (R/B channel swap, :func:`core.state.flip_frame_perspective`) and
:meth:`opp_state` (52-float half swap, :func:`core.state.split_state_for_opponent`) so the
league / self-play opponent path can compose them without re-deriving the transforms.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.core.protocol import Connection
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


class TankEnv(gymnasium.Env):
    """Single-agent, pixel-observation gymnasium env over the Unity socket.

    Construct with an injected connection so the env is testable with no Unity:

    * ``connection`` — a ready :class:`pop_trainer.core.protocol.Connection` (over a real
      socket OR an in-process fake transport). Used as-is; on a lost connection there is no
      reconnect (the next step would re-raise) unless a ``connection_factory`` is also given.
    * ``connection_factory`` — a zero-arg callable returning a fresh ``Connection``. Called
      on construction (if no ``connection`` was passed) and AGAIN to reconnect after a
      dropped connection, mirroring the 2021 reconnect-on-``ConnectionError`` path.

    Exactly one of ``connection`` / ``connection_factory`` must be provided.

    Args:
        connection: a ready ``Connection`` (the simplest test seam).
        connection_factory: a zero-arg callable returning a ``Connection`` (enables
            reconnect after a dropped connection).
        env_config: an :class:`pop_trainer.core.config.EnvConfig` (only ``max_steps`` is used
            by the env loop; the socket-address fields belong to the caller's factory).
            Defaults to ``EnvConfig()``.
        reward_config: a :class:`pop_trainer.core.config.RewardConfig` (win / loss / time
            budgets; the action-cost component was dropped per the CTO ruling). Defaults to
            ``RewardConfig()``.
        frame_shape: the ``(H, W, 3)`` shape of the rendered pixel frame, fixing
            ``observation_space``. Must match what Unity renders. Defaults to
            :data:`DEFAULT_FRAME_SHAPE`.
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

        # Per-episode state, set on reset.
        self._frame = np.zeros(self.frame_shape, dtype=np.uint8)
        self._raw_state: list[float] | None = None
        self.step_counter = 0
        self.last_winner = -1

    # --- gymnasium API -------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Run the restart/start/first-state handshake; return ``(obs, info)``.

        Seeds ``self.np_random`` via ``super().reset`` (so the opponent action draw is
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
        """Send the restart/start handshake and read the first ``state`` + frame."""
        self.conn.send({"restart": True})
        self.conn.receive()  # restart ack
        self.conn.send({"start": True})
        ack = self.conn.receive()  # start ack
        if "starting" not in ack:
            raise RuntimeError(f"unexpected start ack from game: {ack!r}")

        received, frame = self.conn.receive_state_and_frame()  # first state + frame
        self._raw_state = list(received["state"])
        self._frame = np.asarray(frame, dtype=np.uint8)
        self.step_counter = 0
        self.last_winner = -1
        return self._frame, {"state": self._raw_state}

    def step(self, action):
        """Send ``{1: action, 2: opp_action}``, read the next ``state`` + frame, return the
        gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.

        The opponent action is drawn from ``self.np_random`` (so a seeded ``reset`` makes the
        episode reproducible). The reward / boundary is computed by
        :func:`pop_trainer.env.rewards.shaped_step_reward`: a per-step time penalty accrues
        every step and the win/loss terminal is ADDED on the decided step (no action cost).

        ``terminated`` is a decided game (winner / ``done``); ``truncated`` is ``max_steps``
        reached OR a dropped connection. On a dropped connection the step truncates with
        reward ``0.0`` and ``info["lost_connection"] = True`` (the game is counted as ending
        with no winner), and the env reconnects (if a ``connection_factory`` was given) so the
        next ``reset`` can start a fresh episode.
        """
        action = np.asarray(action, dtype=np.float32)
        opp_action = self.np_random.uniform(ACTION_LOW, ACTION_HIGH, ACTION_DIM)
        message = {1: action.tolist(), 2: opp_action.tolist()}

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

        info: dict = {"state": self._raw_state}
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

    def opp_frame(self):
        """The opponent's first-person (R<->B swapped) view of the current pixel frame.

        Composes :func:`pop_trainer.core.state.flip_frame_perspective` on the latest frame so
        the self-play opponent path can score / act from the opponent's perspective.
        """
        return S.flip_frame_perspective(self._frame)

    def opp_state(self):
        """The opponent's first-person 52-float state (the two 26-float halves swapped).

        Composes :func:`pop_trainer.core.state.split_state_for_opponent` on the latest raw
        state. Returns ``None`` before the first ``reset`` (no state yet).
        """
        if self._raw_state is None:
            return None
        return S.split_state_for_opponent(np.asarray(self._raw_state))

    # --- internal ------------------------------------------------------------------------

    def _reconnect(self):
        """Replace ``self.conn`` with a fresh connection from the factory, if one was given.

        Mirrors the 2021 reconnect-on-``ConnectionError`` path. A no-op when the env was
        constructed with a bare ``connection`` (no factory) — the next wire op would then
        re-raise, which is the documented behaviour for that construction mode.
        """
        if self._connection_factory is not None:
            self.conn = self._connection_factory()
