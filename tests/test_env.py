"""Gymnasium contract test for tank_twin.env.TankEnv (M1 Wave 2, task 2.2).

Drives the env with a FAKE in-process transport that replays canned STRICT-JSON
frames — NO Unity, NO subprocess, NO GPU, NO torch/sb3. Construct with
``game_path=None`` + an injected ``transport`` so nothing spawns.

What this pins:

* the gymnasium contract via ``gymnasium.utils.env_checker.check_env`` AND
  ``stable_baselines3.common.env_checker.check_env`` (when sb3 is importable — it is
  in this venv, but the env module itself stays sb3/torch-free);
* ``reset`` returns ``(obs, info)`` with obs IN the declared (36,60,3) uint8 space;
* ``step`` returns the 5-tuple with bool ``terminated`` / ``truncated``;
* DETERMINISM: ``reset(seed=0)`` twice yields identical obs AND identical
  random-opponent action draws (the opponent is seeded via the env's ``np_random``);
* the BYTE-IDENTICAL wire contract: the handshake shapes, the integer-keyed step
  message, and the inbound ``state`` / ``winner`` / ``done`` keys.

NORMAL (non-integration) test: runs in CI with no Unity. The live-Unity end-to-end
version is a later integration-marked test (not this task).
"""

import json

import numpy as np
import pytest

from tank_twin.env import TankEnv

# Canonical raw 52-float states. Field map per player (26 floats): [pos_x, pos_y, vec_x,
# vec_y, aim_x, aim_y, <5 bullets x (pos_x, pos_y, vec_x, vec_y)>]; P1 = 0..25, P2 = 26..51.
# Tank coords sit inside the default arena's world bounds so draw_state maps them on-grid;
# every bullet's pos_x is set to a far-negative sentinel (OFF_BOARD) so draw_state skips it
# (matching the legacy "no active bullet" case) instead of stacking 5 markers on one pixel.
OFF_BOARD = -999.0


def _state(p1_pos, p2_pos):
    s = [0.0] * 52
    s[0], s[1] = p1_pos  # P1 pos_x, pos_y
    s[26], s[27] = p2_pos  # P2 pos_x, pos_y
    for i in range(6, 26, 4):  # P1 bullets off-board
        s[i] = OFF_BOARD
    for i in range(32, 52, 4):  # P2 bullets off-board
        s[i] = OFF_BOARD
    return s


FIRST_STATE = _state((-5.0, 0.0), (4.0, 0.0))
STEP_STATE = _state((-3.0, 1.0), (2.0, -1.0))  # a distinct per-step state so obs can differ


def _frame(obj):
    """Serialize an outbound (game -> env) reply EXACTLY as Unity would: strict JSON bytes."""
    return bytes(json.dumps(obj), encoding="utf-8")


class ScriptedUnity:
    """A FAKE Unity transport: socket-like (``sendall`` / ``recv``), strict-JSON, stateful.

    Instead of a fixed reply queue (which would run dry under ``check_env``'s many
    reset/step calls), it answers like the real Unity side: it inspects the LAST message
    the env sent and returns the protocol-correct reply. This keeps the wire contract
    honest while supporting an arbitrary number of resets/steps.

    Reply rules (mirror ``PythonScripts/tank_env.py`` handshake + step):

    * ``{"restart": True}``           -> ``{"restarting": true}``
    * ``{"start": True}``             -> ``{"starting": true}``  then the NEXT recv is the
      first state ``{"state": [...52...]}`` (the env reads ack then first state).
    * ``{"end": True}``               -> ``{"ending": true}``
    * a step ``{"1": [...], "2": [...]}`` -> ``{"state": [...52...]}`` for the first
      ``win_after`` steps of an episode, then a terminal ``{"state":[...], "winner": 0,
      "done": true}``.

    Records every inbound message in ``received`` so the test can assert the byte-level
    wire shapes (integer keys -> "1"/"2", control-message shapes).
    """

    def __init__(self, win_after=3, winner=0):
        self.win_after = win_after
        self.winner = winner
        self.received = []  # decoded dicts the env sent us
        self._pending = []  # queued raw replies to hand back on recv()
        self._episode_step = 0
        self._expect_first_state = False

    # -- socket surface -----------------------------------------------------

    def sendall(self, data):
        assert isinstance(data, bytes | bytearray)
        msg = json.loads(data.decode("utf-8"))
        self.received.append(msg)
        self._respond(msg)

    def recv(self, bufsize):
        assert bufsize == 1024  # unframed recv(1024) preserved
        # If a start ack was just queued, the env's NEXT recv after reading it is the
        # first state; we queue both so they pop in order.
        return self._pending.pop(0)

    def close(self):
        pass

    # -- reply logic --------------------------------------------------------

    def _respond(self, msg):
        if msg.get("restart") is True:
            self._pending.append(_frame({"restarting": True}))
        elif msg.get("start") is True:
            self._episode_step = 0
            self._pending.append(_frame({"starting": True}))
            self._pending.append(_frame({"state": list(FIRST_STATE)}))
        elif msg.get("end") is True:
            self._pending.append(_frame({"ending": True}))
        elif "1" in msg and "2" in msg:
            self._episode_step += 1
            if self._episode_step >= self.win_after:
                self._pending.append(
                    _frame({"state": list(STEP_STATE), "winner": self.winner, "done": True})
                )
            else:
                self._pending.append(_frame({"state": list(STEP_STATE)}))
        else:
            raise AssertionError(f"unexpected message from env: {msg!r}")


def make_env(**kwargs):
    """Construct a TankEnv with an injected scripted transport and NO game_path."""
    transport = kwargs.pop("transport", None) or ScriptedUnity(**kwargs)
    return TankEnv(game_path=None, transport=transport)


# --- spaces / shape -------------------------------------------------------


def test_observation_space_is_36x60x3_uint8():
    env = make_env()
    assert env.observation_space.shape == (36, 60, 3)
    assert env.observation_space.dtype == np.uint8
    assert env.observation_space.low.min() == 0
    assert env.observation_space.high.max() == 255


def test_action_space_is_box_5_float32():
    env = make_env()
    assert env.action_space.shape == (5,)
    assert env.action_space.dtype == np.float32
    assert float(env.action_space.low.min()) == -1.0
    assert float(env.action_space.high.max()) == 1.0


# --- reset / step contract ------------------------------------------------


def test_reset_returns_obs_info_in_space_uint8():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.dtype == np.uint8
    assert obs.shape == (36, 60, 3)
    assert env.observation_space.contains(obs)


def test_step_returns_five_tuple_with_bool_flags():
    env = make_env(win_after=3)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, int | float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_terminates_on_winner_with_plus_one_reward():
    env = make_env(win_after=2, winner=0)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))  # step 1: ongoing
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert reward == 1  # winner == P1 (0) -> +1
    assert info["winner"] == 0


def test_info_outcome_win_on_player1_winner():
    # winner == P1 (0): terminal step reports info["outcome"] == "win" (the M1 metric seam).
    env = make_env(win_after=2, winner=0)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))  # ongoing
    _obs, _reward, terminated, _truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is True
    assert info["outcome"] == "win"


def test_info_outcome_loss_on_opponent_winner():
    # winner == 1 (opponent): info["outcome"] == "loss", independent of reward sign.
    env = make_env(win_after=2, winner=1)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))  # ongoing
    _obs, _reward, terminated, _truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is True
    assert info["outcome"] == "loss"


def test_info_outcome_draw_on_explicit_draw_winner():
    # winner == -1 (explicit draw): info["outcome"] == "draw".
    env = make_env(win_after=2, winner=-1)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))  # ongoing
    _obs, _reward, terminated, _truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is True
    assert info["outcome"] == "draw"


def test_no_outcome_key_on_non_terminal_step():
    # An ongoing step is not a decided game: no "outcome" key in info.
    env = make_env(win_after=5)
    env.reset(seed=0)
    _obs, _reward, terminated, _truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is False
    assert "outcome" not in info


def test_no_outcome_key_on_truncation():
    # A max_steps truncation is NOT a decided game -> no "outcome" key (won't inflate winrate).
    class NeverWins(ScriptedUnity):
        def _respond(self, msg):
            if "1" in msg and "2" in msg:
                self._pending.append(_frame({"state": list(STEP_STATE)}))
            else:
                super()._respond(msg)

    env = TankEnv(game_path=None, transport=NeverWins(), max_steps=2)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))
    _obs, _reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert "outcome" not in info


def test_truncates_on_max_steps_without_winner():
    # A scripted transport that NEVER reports a winner; env max_steps cap truncates.
    class NeverWins(ScriptedUnity):
        def _respond(self, msg):
            if "1" in msg and "2" in msg:
                self._pending.append(_frame({"state": list(STEP_STATE)}))
            else:
                super()._respond(msg)

    env = TankEnv(game_path=None, transport=NeverWins(), max_steps=2)
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))  # step_counter 1
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated is False
    assert truncated is True  # hit max_steps == 2


# --- determinism ----------------------------------------------------------


def test_reset_seed0_twice_identical_obs():
    env_a = make_env()
    env_b = make_env()
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    assert np.array_equal(obs_a, obs_b)


def test_opponent_action_draw_is_seeded_and_reproducible():
    # The random opponent must draw from the env's np_random (seeded by reset), NOT a
    # bare np.random. Two seed-0 envs must send IDENTICAL opponent actions on the wire.
    env_a = make_env()
    env_b = make_env()
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    action = np.zeros(5, dtype=np.float32)
    env_a.step(action)
    env_b.step(action)
    # The most recent inbound message to each scripted transport is the step message;
    # its "2" entry is the opponent action draw.
    opp_a = env_a.conn.transport.received[-1]["2"]
    opp_b = env_b.conn.transport.received[-1]["2"]
    assert opp_a == opp_b
    # And it is genuinely a draw (not all zeros / not the agent's action).
    assert any(v != 0.0 for v in opp_a)


def test_different_seeds_give_different_opponent_draws():
    env_a = make_env()
    env_b = make_env()
    env_a.reset(seed=0)
    env_b.reset(seed=1)
    action = np.zeros(5, dtype=np.float32)
    env_a.step(action)
    env_b.step(action)
    opp_a = env_a.conn.transport.received[-1]["2"]
    opp_b = env_b.conn.transport.received[-1]["2"]
    assert opp_a != opp_b


# --- byte-identical wire contract -----------------------------------------


def test_reset_handshake_wire_shapes_are_byte_identical():
    env = make_env()
    env.reset(seed=0)
    sent = env.conn.transport.received
    # Legacy reset(): {"restart":True} then {"start":True}, in that order.
    assert sent[0] == {"restart": True}
    assert sent[1] == {"start": True}


def test_step_message_has_integer_keys_as_strings_on_wire():
    env = make_env()
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))
    step_msg = env.conn.transport.received[-1]
    # json coerces the integer keys 1/2 to the strings "1"/"2" — what Unity reads.
    assert set(step_msg.keys()) == {"1", "2"}
    assert len(step_msg["1"]) == 5
    assert len(step_msg["2"]) == 5


def test_action_is_forwarded_verbatim_on_key_1():
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.5, -0.5, 0.25, -0.25, 1.0], dtype=np.float32)
    env.step(action)
    step_msg = env.conn.transport.received[-1]
    assert step_msg["1"] == pytest.approx(action.tolist())


# --- the env_checker contracts (gymnasium + sb3) --------------------------


def test_passes_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    env = make_env()
    # skip_render_check: render is intentionally unimplemented (render_modes == []).
    check_env(env, skip_render_check=True)


def test_passes_stable_baselines3_env_checker():
    # sb3 is importable in this venv; the env module itself stays sb3/torch-free.
    from stable_baselines3.common.env_checker import check_env

    env = make_env()
    check_env(env, warn=True, skip_render_check=True)


# --- close() lifecycle (env-teardown leak fix) ----------------------------


class ClosableUnity(ScriptedUnity):
    """A scripted transport that ALSO answers the close() end handshake and counts close().

    Replies ``{"restarting": true}`` / ``{"ending": true}`` to the close handshake (so
    ``close`` does not raise on a real-looking transport) and records how many times its
    ``close`` was invoked — letting the test assert the connection was released AND that a
    second ``close`` is a no-op (idempotent), with NO real Unity / subprocess / socket.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_close_closes_the_connection_transport():
    # close() must release the injected transport (the real path frees the bound socket).
    transport = ClosableUnity()
    env = TankEnv(game_path=None, transport=transport)
    env.reset(seed=0)
    env.close()
    assert transport.close_count == 1
    # The Connection handle is dropped so the env no longer holds the transport.
    assert env.conn is None


def test_close_is_idempotent():
    # Calling close() twice must NOT raise and must NOT re-close the transport.
    transport = ClosableUnity()
    env = TankEnv(game_path=None, transport=transport)
    env.reset(seed=0)
    env.close()
    env.close()  # second call: a safe no-op
    assert transport.close_count == 1  # not re-closed
    assert env.conn is None


def test_close_safe_with_no_connection():
    # Introspection-only construction (no transport, no game_path): close() is a no-op.
    env = TankEnv(game_path=None)
    assert env.conn is None
    env.close()  # must not raise even though there is nothing to tear down
    env.close()  # idempotent


def test_close_swallows_transport_errors_during_end_handshake():
    # A transport whose close handshake errors (dropped connection) must not break close():
    # the error is swallowed and the transport is still released.
    class BrokenHandshake(ClosableUnity):
        def sendall(self, data):
            raise ConnectionError("connection dropped before teardown")

    transport = BrokenHandshake()
    env = TankEnv(game_path=None, transport=transport)
    # No reset(): a connection that drops before any handshake still tears down cleanly.
    env.close()
    assert transport.close_count == 1
    assert env.conn is None
