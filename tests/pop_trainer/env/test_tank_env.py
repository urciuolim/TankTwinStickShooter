"""Contract tests for ``pop_trainer.env.tank_env.TankEnv`` with an injected fake transport.

No live Unity / socket: a scripted in-process transport feeds canned handshake acks, state
JSON, and length-prefixed pixel frames through ``core.protocol.Connection`` (exactly the way
the protocol tests drive it). We verify the gymnasium 5-tuple, that the observation is the
pixel frame and the 52-float state is in ``info``, the terminated/truncated boundaries, and
the documented lost-connection semantics.
"""

import numpy as np
import pytest

from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.env.tank_env import ACTION_DIM, DEFAULT_FRAME_SHAPE, TankEnv

FRAME_H, FRAME_W = DEFAULT_FRAME_SHAPE[0], DEFAULT_FRAME_SHAPE[1]


# --- wire-byte builders -----------------------------------------------------------------


def _frame_bytes(w, h, fill):
    """Build a length-prefixed RGB24 frame whose every pixel is ``fill`` (an (r,g,b))."""
    payload = bytes(fill) * (w * h)
    header = (
        bytes([P.FRAME_TAG])
        + len(payload).to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([P.FRAME_CHANNELS])
    )
    return header + payload


def _state_and_frame_bytes(state_list, *, winner=None, done=False, fill=(10, 20, 30)):
    """A state JSON message glued to its trailing pixel frame (one step on the wire)."""
    msg = {"state": list(state_list)}
    if winner is not None:
        msg["winner"] = winner
    if done:
        msg["done"] = True
    return P.encode(msg) + _frame_bytes(FRAME_W, FRAME_H, fill)


def _ack(obj):
    return P.encode(obj)


def _walls_bytes(map_id="maps/center_block.json", tile_id=7, dims=None, columns=None):
    """A strict-JSON walls message blob in the exact Unity WallMessage shape (string keys)."""
    if dims is None:
        dims = {"minX": -1, "maxX": 1, "minY": -1, "maxY": 1}
    if columns is None:
        columns = {"0": [0, 1], "-1": [1]}
    return P.encode(
        {
            "type": "walls",
            "map_id": map_id,
            "tileID": tile_id,
            "dims": dims,
            "columns": columns,
        }
    )


def _flat_state(value=0.0):
    return [float(value)] * S.STATE_LEN


class ScriptedTransport:
    """In-process socket-like fake: emits queued recv blobs, records sendall bytes.

    ``recv(n)`` returns the head of the next queued blob capped at ``n`` (a real socket never
    returns more than requested); a leftover tail is re-queued. When the queue is exhausted
    ``recv`` returns ``b""`` (peer closed) — which ``Connection`` turns into ``ConnectionError``.
    """

    def __init__(self, blobs):
        self.blobs = list(blobs)
        self.sent = bytearray()
        self.closed = False

    def sendall(self, data):
        self.sent += data

    def recv(self, bufsize):
        if not self.blobs:
            return b""
        blob = self.blobs[0]
        if len(blob) <= bufsize:
            return self.blobs.pop(0)
        self.blobs[0] = blob[bufsize:]
        return blob[:bufsize]

    def close(self):
        self.closed = True


def _reset_blobs(first_state, *, fill=(10, 20, 30)):
    """The byte script for a successful reset handshake: restart ack, start ack, state+frame."""
    return [
        _ack({"restart": True}),
        _ack({"starting": True}),
        _state_and_frame_bytes(first_state, fill=fill),
    ]


def _reset_blobs_with_walls(first_state, *, walls=None, fill=(10, 20, 30)):
    """A reset script that includes the optional walls message after the start ack.

    Mirrors the real handshake ordering: restart ack, start ack, WALLS, then state+frame.
    """
    return [
        _ack({"restart": True}),
        _ack({"starting": True}),
        walls if walls is not None else _walls_bytes(),
        _state_and_frame_bytes(first_state, fill=fill),
    ]


def _make_env(blobs, **kwargs):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(connection=conn, **kwargs)
    return env, transport


# --- construction / spaces --------------------------------------------------------------


def test_requires_exactly_one_connection_source():
    transport = ScriptedTransport([])
    conn = P.Connection(transport)
    with pytest.raises(ValueError):
        TankEnv()  # neither
    with pytest.raises(ValueError):
        TankEnv(connection=conn, connection_factory=lambda: P.Connection(ScriptedTransport([])))


def test_observation_and_action_spaces():
    env, _ = _make_env([])
    assert env.observation_space.shape == DEFAULT_FRAME_SHAPE
    assert env.observation_space.dtype == np.uint8
    assert env.action_space.shape == (ACTION_DIM,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_rejects_bad_frame_shape():
    transport = ScriptedTransport([])
    with pytest.raises(ValueError):
        TankEnv(connection=P.Connection(transport), frame_shape=(8, 8))
    with pytest.raises(ValueError):
        TankEnv(connection=P.Connection(transport), frame_shape=(8, 8, 4))


# --- reset ------------------------------------------------------------------------------


def test_reset_returns_frame_obs_and_state_in_info():
    raw = [float(i) for i in range(S.STATE_LEN)]
    env, _ = _make_env(_reset_blobs(raw, fill=(7, 8, 9)))
    obs, info = env.reset(seed=0)

    # Observation is the (H, W, 3) uint8 pixel frame.
    assert obs.shape == DEFAULT_FRAME_SHAPE
    assert obs.dtype == np.uint8
    # frombuffer reshape + flipud preserves the constant fill.
    assert np.array_equal(obs[0, 0], [7, 8, 9])
    # The 52-float state is in info, NOT in the observation.
    assert info["state"] == raw
    assert len(info["state"]) == S.STATE_LEN


def test_reset_tracks_walls_message_and_reads_real_first_state():
    # Handshake delivers WALLS before the first state: the env parses + tracks the layout and
    # the first observation/state is the REAL first state (frame/state flow undisturbed).
    raw = [float(i) for i in range(S.STATE_LEN)]
    blobs = _reset_blobs_with_walls(raw, fill=(7, 8, 9))
    env, _ = _make_env(blobs)
    obs, info = env.reset(seed=0)

    # The walls message is parsed into the tracked current map.
    assert isinstance(env.current_map, P.WallLayout)
    assert env.current_map.map_id == "maps/center_block.json"
    assert env.current_map.tile_id == 7
    assert env.current_map.occupied == frozenset({(0, 0), (0, 1), (-1, 1)})
    # Same WallLayout object surfaced in reset info.
    assert info["map"] is env.current_map
    # The first observation/state is the REAL first state, not the walls message.
    assert info["state"] == raw
    assert obs.shape == DEFAULT_FRAME_SHAPE
    assert np.array_equal(obs[0, 0], [7, 8, 9])


def test_reset_without_walls_message_leaves_current_map_none():
    # No walls message (no arena configured): current_map stays None and the first state is
    # read correctly (the optional-message handling does not consume a state by mistake).
    raw = [float(i) for i in range(S.STATE_LEN)]
    env, _ = _make_env(_reset_blobs(raw, fill=(3, 4, 5)))
    obs, info = env.reset(seed=0)

    assert env.current_map is None
    assert info["map"] is None
    assert info["state"] == raw
    assert np.array_equal(obs[0, 0], [3, 4, 5])


def test_step_after_walls_reset_returns_five_tuple_with_map():
    # A normal step after a walls-carrying reset still returns the right 5-tuple, and the
    # tracked map persists into step info.
    raw0 = _flat_state(0.0)
    raw1 = [float(i) for i in range(S.STATE_LEN)]
    blobs = _reset_blobs_with_walls(raw0) + [_state_and_frame_bytes(raw1, fill=(1, 2, 3))]
    env, _ = _make_env(blobs)
    env.reset(seed=0)

    out = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == DEFAULT_FRAME_SHAPE
    assert np.array_equal(obs[0, 0], [1, 2, 3])
    assert info["state"] == raw1
    # The map tracked at reset persists into step info.
    assert info["map"] is env.current_map
    assert isinstance(env.current_map, P.WallLayout)
    assert terminated is False
    assert truncated is False


def test_reset_raises_on_unexpected_start_ack():
    raw = _flat_state()
    blobs = [
        _ack({"restart": True}),
        _ack({"not_an_ack": True}),  # missing the "starting" substring
        _state_and_frame_bytes(raw),
    ]
    env, _ = _make_env(blobs)
    with pytest.raises(RuntimeError):
        env.reset(seed=0)


def test_reset_sends_restart_then_start_handshake():
    raw = _flat_state()
    env, transport = _make_env(_reset_blobs(raw))
    env.reset(seed=0)
    sent = bytes(transport.sent)
    assert P.encode({"restart": True}) in sent
    assert P.encode({"start": True}) in sent


# --- step -------------------------------------------------------------------------------


def test_step_returns_five_tuple_continuing():
    raw0 = _flat_state(0.0)
    raw1 = [float(i) for i in range(S.STATE_LEN)]
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1, fill=(1, 2, 3))]
    env, _ = _make_env(blobs)
    env.reset(seed=0)

    out = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == DEFAULT_FRAME_SHAPE
    assert np.array_equal(obs[0, 0], [1, 2, 3])
    assert info["state"] == raw1
    # Continuing step under the default reward: only the per-step time penalty accrues.
    expected = RewardConfig().time_total / EnvConfig().max_steps
    assert reward == pytest.approx(expected)
    assert terminated is False
    assert truncated is False


def test_step_sends_integer_keyed_action_message():
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    transport.sent.clear()
    env.step(np.array([0.5, -0.5, 0.0, 0.0, 1.0], dtype=np.float32))
    sent = bytes(transport.sent).decode("utf-8")
    # Integer keys 1/2 are coerced to "1"/"2" on the wire (the frozen step contract).
    assert '"1"' in sent and '"2"' in sent


def test_step_terminates_on_winner_p1_win():
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1, winner=S.PLAYER_1)]
    env, _ = _make_env(blobs)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["winner"] == S.PLAYER_1
    assert info["outcome"] == "win"
    # Win terminal ADDED on top of the per-step time penalty.
    tp = RewardConfig().time_total / EnvConfig().max_steps
    assert reward == pytest.approx(RewardConfig().win_reward + tp)


def test_step_terminates_on_player2_win():
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1, winner=1)]
    env, _ = _make_env(blobs)
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is True
    assert info["outcome"] == "loss"
    tp = RewardConfig().time_total / EnvConfig().max_steps
    assert reward == pytest.approx(RewardConfig().loss_reward + tp)


def test_step_draw_terminates_with_draw_outcome():
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1, winner=-1)]
    env, _ = _make_env(blobs)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is True
    assert info["outcome"] == "draw"


def test_step_truncates_at_max_steps():
    raw0 = _flat_state()
    # max_steps=1: the FIRST step hits the cap and truncates (no winner reported).
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, _ = _make_env(blobs, env_config=EnvConfig(max_steps=1))
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is False
    assert truncated is True


def test_step_lost_connection_truncates_with_flag():
    raw0 = _flat_state()
    # No step blob queued: recv returns b"" -> Connection raises ConnectionError on step.
    env, _ = _make_env(_reset_blobs(raw0))
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["lost_connection"] is True
    # The observation falls back to the last good frame (still a valid obs shape).
    assert obs.shape == DEFAULT_FRAME_SHAPE


def test_lost_connection_reconnects_via_factory():
    raw0 = _flat_state(0.0)
    raw_again = _flat_state(2.0)

    # First connection: reset succeeds, then step drops (no step blob).
    # Second connection (from the factory): a fresh reset script.
    scripts = iter(
        [
            _reset_blobs(raw0),  # connection #1 (constructed)
            _reset_blobs(raw_again, fill=(4, 5, 6)),  # connection #2 (reconnect)
        ]
    )

    def factory():
        return P.Connection(ScriptedTransport(next(scripts)))

    env = TankEnv(connection_factory=factory)
    env.reset(seed=0)
    _, _, _, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert truncated is True
    assert info["lost_connection"] is True
    # After the reconnect, a fresh reset uses connection #2 and reads its state.
    _, info2 = env.reset(seed=0)
    assert info2["state"] == raw_again


def test_seeded_reset_makes_player2_action_reproducible():
    # Two DEFAULT-player2 envs reset with the same seed must send byte-identical key "2"
    # actions: this is the frozen-seam reproducibility contract (the built-in player2 draws
    # from the env's single seeded np_random, exactly as the prior direct uniform draw did).
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)

    def run():
        blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
        env, transport = _make_env(blobs)
        env.reset(seed=123)
        transport.sent.clear()
        env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        return P.decode(bytes(transport.sent))["2"]

    assert run() == run()


# --- player2 injection ------------------------------------------------------------------


class _FixedPlayer2:
    """A test player2 returning a constant action (no ``agents`` import)."""

    def __init__(self, action):
        self.action = list(action)

    def act(self, obs):
        return self.action


class _RecordingPlayer2:
    """A test player2 that records the obs it was handed and returns a fixed action."""

    def __init__(self, action):
        self.action = list(action)
        self.seen_obs = None

    def act(self, obs):
        self.seen_obs = obs
        return self.action


def test_injected_player2_drives_key2_action_and_info():
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    fixed = [0.2, -0.4, 0.6, -0.8, 1.0]
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs, player2=_FixedPlayer2(fixed))
    env.reset(seed=0)
    transport.sent.clear()
    p1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    _, _, _, _, info = env.step(p1)

    # The wire "2" action is exactly the injected agent's fixed action.
    sent = P.decode(bytes(transport.sent))
    assert sent["2"] == pytest.approx(fixed)
    # Surfaced both actions: p2 is the agent's; p1 is the action passed to step.
    assert info["p2_action"] == pytest.approx(fixed)
    assert info["p1_action"] == pytest.approx(p1.tolist())
    # Captured on the env too.
    assert env.last_p2_action == pytest.approx(fixed)
    assert env.last_p1_action == pytest.approx(p1.tolist())


def test_injected_player2_receives_flipped_state_view():
    # player2 must be handed its FLIPPED 52-float view of the state it is reacting to (the
    # state present BEFORE this step), proving the perspective flip is wired correctly.
    raw0 = [float(i) for i in range(S.STATE_LEN)]
    raw1 = _flat_state(7.0)
    recorder = _RecordingPlayer2([0.0, 0.0, 0.0, 0.0, 0.0])
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, _ = _make_env(blobs, player2=recorder)
    env.reset(seed=0)
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    expected_view = S.split_state_for_opponent(np.asarray(raw0))
    assert np.array_equal(recorder.seen_obs, expected_view)


def test_injected_player2_action_is_clipped_and_length_fitted():
    # An out-of-range / wrong-length player2 action is coerced to the [-1,1] length-5 wire shape.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs, player2=_FixedPlayer2([5.0, -5.0, 0.3]))
    env.reset(seed=0)
    transport.sent.clear()
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    sent = P.decode(bytes(transport.sent))
    assert sent["2"] == pytest.approx([1.0, -1.0, 0.3, 0.0, 0.0])


# --- self-play perspective helpers ------------------------------------------------------


def test_player2_frame_swaps_rb_channels():
    raw0 = _flat_state()
    env, _ = _make_env(_reset_blobs(raw0, fill=(11, 22, 33)))
    obs, _ = env.reset(seed=0)
    view = env.player2_frame()
    # R and B are swapped relative to the obs; G is unchanged.
    assert np.array_equal(view[0, 0], [33, 22, 11])
    assert np.array_equal(view[..., 1], obs[..., 1])


def test_player2_state_swaps_halves_and_is_none_before_reset():
    transport = ScriptedTransport([])
    env = TankEnv(connection=P.Connection(transport))
    assert env.player2_state() is None  # no state yet

    raw = [float(i) for i in range(S.STATE_LEN)]
    env2, _ = _make_env(_reset_blobs(raw))
    env2.reset(seed=0)
    view = env2.player2_state()
    assert np.array_equal(view[:26], np.asarray(raw[26:]))
    assert np.array_equal(view[26:], np.asarray(raw[:26]))


# --- close lifecycle --------------------------------------------------------------------


def test_close_is_idempotent_and_releases_transport():
    raw0 = _flat_state()
    # Give close() enough acks to drain its best-effort restart/end handshake.
    blobs = _reset_blobs(raw0) + [_ack({"restart": True}), _ack({"end": True})]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    env.close()
    assert transport.closed is True
    assert env.conn is None
    env.close()  # second close is a no-op


def test_close_swallows_transport_errors():
    # No acks queued for the end handshake -> recv b"" -> ConnectionError, swallowed.
    raw0 = _flat_state()
    env, transport = _make_env(_reset_blobs(raw0))
    env.reset(seed=0)
    env.close()  # must not raise despite the missing end-handshake acks
    assert env.conn is None
    assert transport.closed is True


def test_close_swallows_transport_close_oserror():
    class CloseRaises(ScriptedTransport):
        def close(self):
            raise OSError("close failed")

    transport = CloseRaises([])
    env = TankEnv(connection=P.Connection(transport))
    env.close()  # the OSError from transport.close() is suppressed
    assert env.conn is None


# --- misc lifecycle ---------------------------------------------------------------------


def test_render_not_implemented():
    env, _ = _make_env([])
    with pytest.raises(NotImplementedError):
        env.render()


def test_default_seed_used_when_reset_seed_omitted():
    # Two default-player2 envs constructed with the same default seed and reset() (no seed arg)
    # must produce identical key "2" actions, proving the constructor seed is the fallback.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)

    def run():
        blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
        env, transport = _make_env(blobs, seed=99)
        env.reset()  # no seed -> falls back to the constructor's default
        transport.sent.clear()
        env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        return P.decode(bytes(transport.sent))["2"]

    assert run() == run()


def test_reset_reconnects_and_retries_once_on_dropped_handshake():
    raw_ok = _flat_state(3.0)
    # Connection #1 drops mid-handshake (only the restart ack, then b""); the factory hands
    # back connection #2 with a full, valid reset script.
    scripts = iter(
        [
            [_ack({"restart": True})],  # connection #1: drops after restart ack
            _reset_blobs(raw_ok),  # connection #2: clean reset on retry
        ]
    )

    def factory():
        return P.Connection(ScriptedTransport(next(scripts)))

    env = TankEnv(connection_factory=factory)
    _, info = env.reset(seed=0)
    assert info["state"] == raw_ok
