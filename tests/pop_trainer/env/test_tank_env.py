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
from pop_trainer.env.tank_env import (
    ACTION_DIM,
    DEFAULT_FRAME_SHAPE,
    DEFAULT_MAX_STEPS,
    TankEnv,
)

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


def _safety_cap_reset_blobs(first_state, *, stray_state=None, fill=(10, 20, 30)):
    """Reset script for the SAFETY-CAP corner case: Unity emits ONE stray mid-round state (+ frame)
    BEFORE the restart ack (because per-step it sends state then reads the restart), then the normal
    restart ack, start ack, and first state+frame. The env must drain the stray pair and still
    complete the handshake.
    """
    if stray_state is None:
        stray_state = _flat_state(7.0)
    return [
        _state_and_frame_bytes(stray_state, fill=(9, 9, 9)),  # stray mid-round state+frame
        _ack({"restarting": True}),  # the real restart ack, AFTER the stray pair
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


def _switch_reset_blobs(first_state, *, switch_walls=None, start_walls=None, fill=(10, 20, 30)):
    """Byte script for a switch_arena reset: restart ack, arena_switched ack (+ optional walls),
    start ack (+ optional walls), then state+frame.

    Mirrors the Unity flow: the switch branch and the start branch BOTH emit an optional walls
    message after their respective acks (present only when the new arena has a "Walls" block).
    """
    blobs = [
        _ack({"restart": True}),
        _ack({"arena_switched": True}),
    ]
    if switch_walls is not None:
        blobs.append(switch_walls)
    blobs.append(_ack({"starting": True}))
    if start_walls is not None:
        blobs.append(start_walls)
    blobs.append(_state_and_frame_bytes(first_state, fill=fill))
    return blobs


def _make_env(blobs, **kwargs):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(connection=conn, **kwargs)
    return env, transport


def _decode_framed(sent):
    """Decode the FIRST length-prefixed message in ``sent`` (the env's outbound wire format).

    Connection.send length-prefixes each control/step JSON (4-byte big-endian length + JSON);
    tests that decode a single sent message strip that prefix here.
    """
    n = int.from_bytes(bytes(sent[: P.SEND_LENGTH_PREFIX_LEN]), "big")
    start = P.SEND_LENGTH_PREFIX_LEN
    return P.decode(bytes(sent[start : start + n]))


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


# --- switch_arena map rotation (additive, reset-time only) ------------------------------


def test_reset_switch_arena_with_walls_tracks_switched_map():
    # Switching to a walls arena: Unity emits walls after BOTH the arena_switched ack AND the
    # start ack. The env drains both and current_map / info["map"] reflect the switched arena.
    raw = [float(i) for i in range(S.STATE_LEN)]
    new_walls = _walls_bytes(
        map_id="maps/four_pillars.json",
        tile_id=9,
        columns={"2": [2], "-2": [-2]},
    )
    blobs = _switch_reset_blobs(raw, switch_walls=new_walls, start_walls=new_walls, fill=(7, 8, 9))
    env, transport = _make_env(blobs)
    obs, info = env.reset(seed=0, options={"switch_arena": "maps/four_pillars.json"})

    # The switch request was sent on the wire with the arena path as a string value.
    sent = bytes(transport.sent)
    assert P.encode({"switch_arena": "maps/four_pillars.json"}) in sent
    # The switched arena's walls are tracked (the LAST walls message wins; both name the arena).
    assert isinstance(env.current_map, P.WallLayout)
    assert env.current_map.map_id == "maps/four_pillars.json"
    assert env.current_map.tile_id == 9
    assert env.current_map.occupied == frozenset({(2, 2), (-2, -2)})
    assert info["map"] is env.current_map
    # The first observation/state is the REAL first state, not a walls message.
    assert info["state"] == raw
    assert np.array_equal(obs[0, 0], [7, 8, 9])


def test_reset_switch_arena_without_walls_does_not_consume_state():
    # Switching to a walls-ABSENT arena: NO walls after either ack. The env must read straight
    # through to the first state without mis-consuming it.
    raw = [float(i) for i in range(S.STATE_LEN)]
    blobs = _switch_reset_blobs(raw, switch_walls=None, start_walls=None, fill=(3, 4, 5))
    env, _ = _make_env(blobs)
    obs, info = env.reset(seed=0, options={"switch_arena": "maps/empty.json"})

    # No walls arrived -> current_map stays whatever it was (None here).
    assert env.current_map is None
    assert info["map"] is None
    # The first state is intact (not eaten by the optional-walls routing).
    assert info["state"] == raw
    assert np.array_equal(obs[0, 0], [3, 4, 5])


def test_reset_switch_arena_sends_switch_in_ingame_window_order():
    # The switch must be sent AFTER restart and BEFORE start (Unity's !ingame window).
    raw = _flat_state()
    blobs = _switch_reset_blobs(raw, switch_walls=_walls_bytes(), start_walls=_walls_bytes())
    env, transport = _make_env(blobs)
    env.reset(seed=0, options={"switch_arena": "maps/center_block.json"})
    sent = bytes(transport.sent)
    i_restart = sent.find(P.encode({"restart": True}))
    i_switch = sent.find(P.encode({"switch_arena": "maps/center_block.json"}))
    i_start = sent.find(P.encode({"start": True}))
    assert i_restart != -1 and i_switch != -1 and i_start != -1
    assert i_restart < i_switch < i_start


def test_reset_without_switch_sends_only_framed_restart_then_start():
    # SEAM RE-PROOF: a reset WITHOUT a switch target sends EXACTLY the framed restart + framed
    # start (no switch_arena), regardless of options. The SHAPE is the frozen restart/start
    # handshake; only the TIMING discipline + the length prefix changed.
    raw = _flat_state()
    env_a, transport_a = _make_env(_reset_blobs(raw))
    env_a.reset(seed=0)

    env_b, transport_b = _make_env(_reset_blobs(raw))
    env_b.reset(seed=0, options={"unrelated": "ignored"})

    expected = P.encode_framed({"restart": True}) + P.encode_framed({"start": True})
    assert bytes(transport_a.sent) == expected
    assert bytes(transport_b.sent) == expected
    # No switch_arena byte leaks onto the wire on a plain reset.
    assert b"switch_arena" not in bytes(transport_a.sent)
    assert b"switch_arena" not in bytes(transport_b.sent)


def test_step_after_switch_reset_persists_switched_map_in_info():
    # A normal step after a switch reset still returns the 5-tuple and info["map"] reflects the
    # switched arena (the per-step wire is unchanged: integer-keyed {1,2}).
    raw0 = _flat_state(0.0)
    raw1 = [float(i) for i in range(S.STATE_LEN)]
    new_walls = _walls_bytes(map_id="maps/four_pillars.json", tile_id=9, columns={"2": [2]})
    blobs = _switch_reset_blobs(raw0, switch_walls=new_walls, start_walls=new_walls) + [
        _state_and_frame_bytes(raw1, fill=(1, 2, 3))
    ]
    env, transport = _make_env(blobs)
    env.reset(seed=0, options={"switch_arena": "maps/four_pillars.json"})
    transport.sent.clear()

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    # The step wire is still the integer-keyed step message (frozen seam).
    sent = _decode_framed(transport.sent)
    assert set(sent) == {"1", "2"}
    assert info["state"] == raw1
    assert info["map"] is env.current_map
    assert env.current_map.map_id == "maps/four_pillars.json"
    assert terminated is False
    assert truncated is False


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
    # Continuing step under the default reward: only the per-step time penalty accrues. The env's
    # no-config safety cap is DEFAULT_MAX_STEPS, so the budget is spread over that.
    expected = RewardConfig().time_total / DEFAULT_MAX_STEPS
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
    # Integer keys 1/2 are coerced to "1"/"2" on the (now length-prefixed) wire (frozen seam).
    sent = _decode_framed(transport.sent)
    assert set(sent) == {"1", "2"}


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
    # Win terminal ADDED on top of the per-step time penalty (budget over DEFAULT_MAX_STEPS).
    tp = RewardConfig().time_total / DEFAULT_MAX_STEPS
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
    tp = RewardConfig().time_total / DEFAULT_MAX_STEPS
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


# --- round-boundary discipline: Python NEVER sends a raw mid-round restart -----------------


def _iter_framed(sent):
    """Yield each decoded length-prefixed message in ``sent`` (the env's outbound wire format)."""
    buf = bytes(sent)
    i = 0
    while i + P.SEND_LENGTH_PREFIX_LEN <= len(buf):
        n = int.from_bytes(buf[i : i + P.SEND_LENGTH_PREFIX_LEN], "big")
        i += P.SEND_LENGTH_PREFIX_LEN
        yield P.decode(buf[i : i + n])
        i += n


def test_default_max_steps_safety_cap_is_above_unity_round():
    # The env's no-config default is the GENEROUS safety cap (DEFAULT_MAX_STEPS=600), strictly
    # above Unity's ~300-step round so it never preempts Unity's own done.
    env, _ = _make_env([])
    assert env.max_steps == DEFAULT_MAX_STEPS
    assert DEFAULT_MAX_STEPS > 300


def test_step_does_not_send_restart_within_a_step():
    # The wedge invariant: step() NEVER puts a restart on the wire. A continuing step sends ONLY
    # the integer-keyed {1,2} action message; no control restart leaks from the step path.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    transport.sent.clear()
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    msgs = list(_iter_framed(transport.sent))
    assert all("restart" not in m for m in msgs)
    assert [set(m) for m in msgs] == [{"1", "2"}]


def test_max_steps_safety_truncation_does_not_emit_raw_restart():
    # CRITICAL: when the max_steps SAFETY cap fires mid-round (Unity reported no winner/done), the
    # truncating step must NOT itself send a restart. The step puts only the {1,2} action on the
    # wire; the subsequent reset's restart advances Unity through the waiting-state handshake.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)  # NO winner / NO done: an undecided round
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs, env_config=EnvConfig(max_steps=1))
    env.reset(seed=0)
    transport.sent.clear()
    _, _, terminated, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is False
    assert truncated is True  # the safety cap tripped
    # The ONLY thing sent during the truncating step is the action message -- never a raw restart.
    msgs = list(_iter_framed(transport.sent))
    assert [set(m) for m in msgs] == [{"1", "2"}]
    assert all("restart" not in m for m in msgs)


def test_step_terminates_on_done_without_winner():
    # A bare Unity ``done`` (no winner key) terminates the episode -- this is the NORMAL boundary
    # the env drives off (Unity's clock), distinct from a max_steps truncation.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1, done=True)]
    env, _ = _make_env(blobs)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "draw"  # done with no winner key


def test_reset_after_done_sends_restart_only_as_full_handshake():
    # After a done-terminated step, reset() sends restart -- but ONLY as the leading half of the
    # restart -> start handshake (the documented round-over -> restart sequence), never a lone
    # mid-round restart. The restart is immediately followed by start, with nothing between.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    raw2 = _flat_state(2.0)
    blobs = (
        _reset_blobs(raw0)
        + [_state_and_frame_bytes(raw1, winner=S.PLAYER_1)]  # decided round -> terminated
        + _reset_blobs(raw2)  # the next reset's handshake script
    )
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    _, _, terminated, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert terminated is True
    transport.sent.clear()
    env.reset(seed=0)
    # The reset emits EXACTLY framed restart then framed start (the full handshake), in order.
    msgs = list(_iter_framed(transport.sent))
    assert msgs == [{"restart": True}, {"start": True}]


def test_reset_drains_stray_midround_state_before_restart_ack():
    # SAFETY-CAP corner case: after a max_steps truncation, reset() sends restart while Unity is
    # technically still mid-round, so Unity emits ONE stray state (+frame) BEFORE the restart ack.
    # reset() must DRAIN that stray pair and still complete the handshake cleanly -- and it must
    # still send only the framed restart + start (never a raw mid-round restart).
    raw_first = _flat_state(3.0)
    env, transport = _make_env(_safety_cap_reset_blobs(raw_first, fill=(1, 2, 3)))
    obs, info = env.reset(seed=0)
    # The handshake completed: the first real state is read, NOT the stray one.
    assert info["state"] == raw_first
    assert np.array_equal(obs[0, 0], [1, 2, 3])
    # Only the framed restart + start went out (no raw mid-round restart).
    msgs = list(_iter_framed(transport.sent))
    assert msgs == [{"restart": True}, {"start": True}]


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


# --- pure transport: both actions come from the caller ----------------------------------


def test_step_transports_both_caller_actions_on_wire_and_info():
    # THE key new proof: step(a1, a2) sends BOTH on the wire under keys "1"/"2" and captures both
    # in info + on the env. The env owns neither player — a2 is the caller's opponent_action.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    transport.sent.clear()
    p1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    a2 = [0.2, -0.4, 0.6, -0.8, 1.0]
    _, _, _, _, info = env.step(p1, a2)

    sent = _decode_framed(transport.sent)
    assert sent["1"] == pytest.approx(p1.tolist())
    assert sent["2"] == pytest.approx(a2)
    # Both actions surfaced in info and on the env.
    assert info["p1_action"] == pytest.approx(p1.tolist())
    assert info["p2_action"] == pytest.approx(a2)
    assert env.last_p1_action == pytest.approx(p1.tolist())
    assert env.last_p2_action == pytest.approx(a2)


def test_step_default_opponent_action_sends_zero_a2():
    # With opponent_action omitted the env sends a well-formed no-op zero action for player2.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    transport.sent.clear()
    _, _, _, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    sent = _decode_framed(transport.sent)
    assert sent["2"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert info["p2_action"] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_step_opponent_action_is_clipped_and_length_fitted():
    # An out-of-range / wrong-length opponent_action is coerced to the [-1,1] length-5 wire shape.
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs)
    env.reset(seed=0)
    transport.sent.clear()
    env.step(np.zeros(ACTION_DIM, dtype=np.float32), [5.0, -5.0, 0.3])
    sent = _decode_framed(transport.sent)
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
    # The constructor seed is the fallback when reset() is called with no seed: the handshake +
    # step must not crash and must send a well-formed message (the env owns neither player, so
    # the default a2 is the zero no-op regardless of seed).
    raw0 = _flat_state()
    raw1 = _flat_state(1.0)
    blobs = _reset_blobs(raw0) + [_state_and_frame_bytes(raw1)]
    env, transport = _make_env(blobs, seed=99)
    env.reset()  # no seed -> falls back to the constructor's default
    transport.sent.clear()
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    sent = _decode_framed(transport.sent)
    assert sent["1"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert sent["2"] == [0.0, 0.0, 0.0, 0.0, 0.0]


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


# --- lazy lifecycle / release / reap hook / kill-old-first reconnect ---------------------


class _FakeProc:
    """A subprocess.Popen stand-in: records terminate/kill; poll reports running until reaped.

    Copied locally (not imported from the rl test) so the env test owns no rl dependency.
    """

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self._dead = False

    def poll(self):
        return 0 if self._dead else None

    def terminate(self):
        self.terminated = True
        self._dead = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._dead = True


def test_factory_not_called_at_construction_and_env_not_running():
    # LAZY: a connection_factory is NOT invoked in __init__; the env is "not running" until reset.
    calls = []

    def factory():
        calls.append(1)
        return P.Connection(ScriptedTransport(_reset_blobs(_flat_state())))

    env = TankEnv(connection_factory=factory)
    assert calls == []  # never called at construction
    assert env.is_running is False
    assert env.conn is None


def test_first_reset_lazily_launches_via_factory():
    # The factory IS called exactly once on the first reset, and is_running flips True.
    raw = [float(i) for i in range(S.STATE_LEN)]
    calls = []

    def factory():
        calls.append(1)
        return P.Connection(ScriptedTransport(_reset_blobs(raw)))

    env = TankEnv(connection_factory=factory)
    assert env.is_running is False
    _, info = env.reset(seed=0)
    assert calls == [1]  # launched exactly once on the first reset
    assert env.is_running is True
    assert info["state"] == raw


def test_bare_connection_is_running_from_construction():
    # The bare-connection seam is UNCHANGED: running from construction, factory path untouched.
    env, _ = _make_env([])
    assert env.is_running is True
    assert env.conn is not None


def test_release_reaps_proc_closes_transport_and_stops_running():
    # RELEASE on a running env: reaps the stashed proc, closes the transport, is_running -> False.
    raw = _flat_state()
    transport = ScriptedTransport(_reset_blobs(raw))
    proc = _FakeProc()

    def factory(_transport=transport, _proc=proc):
        conn = P.Connection(_transport)
        conn._launch_proc = _proc
        return conn

    reaped = []
    env = TankEnv(connection_factory=factory, reap=lambda p: reaped.append(p))
    env.reset(seed=0)
    assert env.is_running is True

    env.release()
    assert reaped == [proc]  # the stashed proc was reaped via the hook
    assert transport.closed is True  # the transport was closed
    assert env.is_running is False
    assert env.conn is None


def test_release_is_idempotent():
    # A second release is a no-op: the reap hook is NOT called again.
    raw = _flat_state()
    proc = _FakeProc()

    def factory():
        conn = P.Connection(ScriptedTransport(_reset_blobs(raw)))
        conn._launch_proc = proc
        return conn

    reaped = []
    env = TankEnv(connection_factory=factory, reap=lambda p: reaped.append(p))
    env.reset(seed=0)
    env.release()
    assert len(reaped) == 1
    env.release()  # already not running -> no-op
    assert len(reaped) == 1
    assert env.is_running is False


def test_reset_after_release_relaunches_via_factory():
    # The env OBJECT stays usable: a reset after release re-launches via the factory.
    raw_first = _flat_state(0.0)
    raw_again = _flat_state(2.0)
    scripts = iter([_reset_blobs(raw_first), _reset_blobs(raw_again)])
    calls = []

    def factory():
        calls.append(1)
        conn = P.Connection(ScriptedTransport(next(scripts)))
        conn._launch_proc = _FakeProc()
        return conn

    env = TankEnv(connection_factory=factory, reap=lambda p: p.terminate())
    _, info1 = env.reset(seed=0)
    assert info1["state"] == raw_first
    env.release()
    assert env.is_running is False
    _, info2 = env.reset(seed=0)  # re-launches via the factory
    assert calls == [1, 1]  # factory invoked again on the post-release reset
    assert env.is_running is True
    assert info2["state"] == raw_again


def test_release_with_reap_none_closes_transport_without_proc_kill():
    # reap=None: release closes the transport, does not crash, and attempts no proc-kill.
    raw = _flat_state()
    transport = ScriptedTransport(_reset_blobs(raw))
    proc = _FakeProc()

    def factory(_transport=transport, _proc=proc):
        conn = P.Connection(_transport)
        conn._launch_proc = _proc  # a proc is stashed, but reap=None must ignore it
        return conn

    env = TankEnv(connection_factory=factory)  # reap defaults to None
    env.reset(seed=0)
    env.release()
    assert transport.closed is True
    assert proc.terminated is False  # no proc-kill attempted with reap=None
    assert env.is_running is False


def test_reconnect_kills_old_proc_before_launching_new():
    # KILL-OLD-FIRST: the old (possibly stalled) instance is reaped BEFORE the new build launches.
    raw = _flat_state()
    order = []

    old_proc = _FakeProc()

    def factory():
        order.append("launch")
        conn = P.Connection(ScriptedTransport(_reset_blobs(raw)))
        conn._launch_proc = _FakeProc()
        return conn

    def reap(_proc):
        order.append("reap")

    env = TankEnv(connection_factory=factory, reap=reap)
    env.reset(seed=0)  # launch #1
    assert order == ["launch"]
    # Stash a known OLD proc on the current conn, then force a reconnect.
    env.conn._launch_proc = old_proc
    order.clear()
    env._reconnect()
    # The old proc was reaped FIRST, THEN the new build was launched.
    assert order == ["reap", "launch"]
    assert env.is_running is True


def test_reconnect_with_no_factory_is_noop():
    # A bare connection (no factory) cannot reconnect: documented no-op, conn left in place.
    raw = _flat_state()
    env, _ = _make_env(_reset_blobs(raw))
    env.reset(seed=0)
    before = env.conn
    env._reconnect()  # no factory -> no-op
    assert env.conn is before
    assert env.is_running is True


def test_step_lost_connection_reconnect_reaps_old_then_relaunches():
    # The lost-connection step path uses kill-old-first reconnect: with reap + factory the old
    # proc is reaped and a fresh conn is created; the next reset reads the new connection's state.
    raw0 = _flat_state(0.0)
    raw_again = _flat_state(5.0)
    scripts = iter([_reset_blobs(raw0), _reset_blobs(raw_again, fill=(4, 5, 6))])
    reaped = []

    def factory():
        conn = P.Connection(ScriptedTransport(next(scripts)))
        conn._launch_proc = _FakeProc()
        return conn

    env = TankEnv(connection_factory=factory, reap=lambda p: reaped.append(p))
    env.reset(seed=0)
    old_proc = env.conn._launch_proc
    _, _, _, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert truncated is True
    assert info["lost_connection"] is True
    assert reaped == [old_proc]  # the old build was reaped on reconnect
    _, info2 = env.reset(seed=0)
    assert info2["state"] == raw_again
