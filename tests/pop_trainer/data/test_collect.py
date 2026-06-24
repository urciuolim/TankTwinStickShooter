"""Tests for the collection driver, driving a real ``TankEnv`` over an in-process fake transport.

NO live socket is opened here (the contract: do not unit-test live collection). The transport is
a scripted in-process fake feeding canned handshake acks, state JSON, and length-prefixed pixel
frames through ``core.protocol.Connection`` — exactly the shape the env tests use — so
``run_episode`` / ``collect_to_shards`` are exercised end to end through the SAME gym observation
pipeline RL trains on. Collection pairs a ``player1`` agent + a ``player2`` agent (both REAL
:mod:`pop_trainer.agents`); ``player2`` is injected into the env at construction.
"""

import math

import numpy as np
import pytest

from pop_trainer import agents
from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig
from pop_trainer.data import collect, schema, shards
from pop_trainer.env.tank_env import DEFAULT_FRAME_SHAPE, TankEnv

FRAME_H, FRAME_W = DEFAULT_FRAME_SHAPE[0], DEFAULT_FRAME_SHAPE[1]


# --- wire-byte builders (mirrors tests/pop_trainer/env/test_tank_env.py) -----------------


def _frame_bytes(w, h, fill):
    """A length-prefixed RGB24 frame whose every pixel is ``fill`` (an (r, g, b))."""
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


def _step_episode_blobs(states, *, fills=None, winner_last=None, done_last=False):
    """Byte script for one episode: a reset handshake on ``states[0]`` then a step per remaining.

    ``states[i]`` (i >= 1) is the state the env reads on step ``i``; the last carries
    ``winner``/``done`` if asked. ``fills[i]`` colours frame ``i`` (defaults to a per-index fill).
    """
    fills = fills or [(i, i, i) for i in range(len(states))]
    blobs = _reset_blobs(states[0], fill=fills[0])
    for i in range(1, len(states)):
        last = i == len(states) - 1
        blobs.append(
            _state_and_frame_bytes(
                states[i],
                fill=fills[i],
                winner=winner_last if last else None,
                done=done_last and last,
            )
        )
    return blobs


def _make_env(blobs, *, player2=None, **kwargs):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(connection=conn, player2=player2, **kwargs)
    return env, transport


def _positioned_state(p1_xy, p2_xy):
    """A 52-float state with P1/P2 at the given (x, y); everything else zero."""
    s = [0.0] * S.STATE_LEN
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_X] = float(p1_xy[0])
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_Y] = float(p1_xy[1])
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = float(p2_xy[0])
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = float(p2_xy[1])
    return s


# --- run_episode -------------------------------------------------------------------------


def test_run_episode_records_all_steps_until_cap():
    # 5 states, cap at 5 -> records 5 samples, last carries the zero action.
    states = [_flat_state(i) for i in range(5)]
    fills = [(i + 1, i + 1, i + 1) for i in range(5)]
    blobs = _step_episode_blobs(states, fills=fills)
    env, transport = _make_env(blobs, player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=5))
    ep = collect.run_episode(
        env,
        player1=agents.IdleAgent(),
        map_id=3,
        episode_id=0,
        max_steps=5,
        seed=0,
    )
    assert len(ep) == 5
    assert not ep.ended_done
    # step indices are 0..4
    assert [s.step_idx for s in ep.samples] == [0, 1, 2, 3, 4]
    # frames are time-aligned with the scripted stream (the per-step constant fill).
    for i, s in enumerate(ep.samples):
        assert np.array_equal(s.frame[0, 0], [i + 1, i + 1, i + 1])
    # states are time-aligned too.
    for i, s in enumerate(ep.samples):
        np.testing.assert_array_equal(s.state, np.array(states[i], dtype=np.float32))
    # 4 action messages went on the wire (one per transition; none after the capped step).
    sent = bytes(transport.sent).decode("utf-8")
    assert sent.count('"1"') == 4
    # the final recorded step carries the zero action
    np.testing.assert_array_equal(
        ep.samples[-1].action, np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)
    )


def test_run_episode_stops_on_done():
    states = [_flat_state(0), _flat_state(1)]
    env, _ = _make_env(_step_episode_blobs(states, done_last=True), player2=agents.IdleAgent())
    ep = collect.run_episode(
        env,
        player1=agents.IdleAgent(),
        map_id=0,
        episode_id=0,
        max_steps=100,  # cap not reached; the env's terminated boundary stops it
        seed=0,
    )
    assert len(ep) == 2
    assert ep.ended_done


def test_run_episode_stops_on_winner_terminal():
    # A reported winner terminates the env; the episode stops there before the cap.
    states = [_flat_state(0), _flat_state(1)]
    env, _ = _make_env(
        _step_episode_blobs(states, winner_last=S.PLAYER_1), player2=agents.IdleAgent()
    )
    ep = collect.run_episode(
        env,
        player1=agents.IdleAgent(),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
    )
    assert len(ep) == 2
    assert ep.ended_done


def test_run_episode_captures_both_player_actions():
    # THE KEY NEW ASSERTION: both action slots are recorded. player1 is a ConstantAgent and
    # player2 is a ConstantAgent with a DISTINCT non-zero action; a non-boundary sample must
    # carry player1's action in slot 0 and player2's actual (non-zero) action in slot 1 — the
    # zeroed-player2 gap is fixed.
    a1 = [1.0, 0.0, 0.0, 0.0, 1.0]
    a2 = [0.0, -1.0, 1.0, 0.0, 1.0]
    states = [_flat_state(0), _flat_state(1)]
    env, transport = _make_env(
        _step_episode_blobs(states, done_last=True), player2=agents.ConstantAgent(a2)
    )
    ep = collect.run_episode(
        env,
        player1=agents.ConstantAgent(a1),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
    )
    np.testing.assert_array_equal(ep.samples[0].action[0], np.array(a1, dtype=np.float32))
    # player2's slot is its REAL action, NOT zero.
    np.testing.assert_array_equal(ep.samples[0].action[1], np.array(a2, dtype=np.float32))
    assert not np.array_equal(ep.samples[0].action[1], np.zeros(schema.ACTION_LEN, np.float32))
    # On the wire both actions were sent in their own slots.
    sent = P.decode(bytes(transport.sent[bytes(transport.sent).index(b'{"1"') :]))
    assert sent["1"] == a1
    assert sent["2"] == a2


def test_run_episode_records_applied_player1_action():
    # The action recorded on a non-terminal sample's player1 slot matches info["p1_action"].
    a1 = [0.5, -0.5, 0.0, 0.0, 1.0]
    states = [_flat_state(0), _flat_state(1)]
    env, transport = _make_env(
        _step_episode_blobs(states, done_last=True), player2=agents.IdleAgent()
    )
    ep = collect.run_episode(
        env,
        player1=agents.ConstantAgent(a1),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
    )
    np.testing.assert_array_equal(ep.samples[0].action[0], np.array(a1, dtype=np.float32))
    # player2 is idle: its slot is the zero action it actually sent (still its REAL action).
    np.testing.assert_array_equal(
        ep.samples[0].action[1], np.zeros(schema.ACTION_LEN, dtype=np.float32)
    )
    sent = P.decode(bytes(transport.sent[bytes(transport.sent).index(b'{"1"') :]))
    assert sent["1"] == a1


def test_run_episode_supports_scripted_cycle_player1():
    # ScriptedCycleAgent advances an internal counter each act(); its actions cycle across steps.
    cycle = agents.ScriptedCycleAgent([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(
        _step_episode_blobs(states), player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=3)
    )
    ep = collect.run_episode(
        env,
        player1=cycle,
        map_id=0,
        episode_id=0,
        max_steps=3,
        seed=0,
    )
    np.testing.assert_array_equal(ep.samples[0].action[0], np.array([1, 0, 0, 0, 0], np.float32))
    np.testing.assert_array_equal(ep.samples[1].action[0], np.array([0, 1, 0, 0, 0], np.float32))


def test_aim_agent_as_player1_through_run_episode():
    # AimAtPlayer2Agent as player1 over >= 3 steps. P1 at origin, P2 at (3, 4): player1 aims the
    # unit vector toward P2 -> (0.6, 0.8) and fires. Static positions keep the vector checkable at
    # every step (and exercise step index >= 2).
    state = _positioned_state((0.0, 0.0), (3.0, 4.0))
    states = [state for _ in range(4)]
    env, _ = _make_env(
        _step_episode_blobs(states), player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=4)
    )
    ep = collect.run_episode(
        env,
        player1=agents.AimAtPlayer2Agent(),  # default PLAYER_1 on its own view
        map_id=0,
        episode_id=0,
        max_steps=4,
        seed=0,
    )
    assert len(ep) == 4
    assert [s.step_idx for s in ep.samples] == [0, 1, 2, 3]
    for sample in ep.samples[:-1]:  # the final/capped sample carries the zero action
        a1 = sample.action[0]
        assert math.isclose(a1[2], 0.6, abs_tol=1e-6)
        assert math.isclose(a1[3], 0.8, abs_tol=1e-6)
        assert a1[4] == 1.0


def test_run_episode_validates_state_width():
    # A short first state surfaces in info["state"] and must fail the schema validator.
    bad = [0.0] * (S.STATE_LEN - 1)
    env, _ = _make_env(_reset_blobs(bad), player2=agents.IdleAgent())
    with pytest.raises(ValueError):
        collect.run_episode(
            env,
            player1=agents.IdleAgent(),
            map_id=0,
            episode_id=0,
            max_steps=1,
            seed=0,
        )


def test_run_episode_is_deterministic_in_seed():
    # Same (player1 agent + seed, player2 agent + seed) -> identical wire bytes across two runs.
    # The agents and the env are rebuilt each run; run_episode resets the resettable agents.
    states = [_flat_state(i) for i in range(3)]

    def run():
        env, transport = _make_env(
            _step_episode_blobs(states),
            player2=agents.RandomAgent(seed=7),
            env_config=EnvConfig(max_steps=3),
        )
        collect.run_episode(
            env,
            player1=agents.RandomAgent(seed=99),
            map_id=0,
            episode_id=0,
            max_steps=3,
            seed=123,
        )
        return bytes(transport.sent)

    assert run() == run()


def test_run_episode_propagates_agent_typeerror():
    # A buggy player1 whose act() raises must propagate the error, not swallow it.
    class BuggyAgent:
        def act(self, obs):
            raise TypeError("genuine bug inside the agent body")

    states = [_flat_state(0), _flat_state(1)]
    env, _ = _make_env(
        _step_episode_blobs(states), player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=5)
    )
    with pytest.raises(TypeError, match="genuine bug"):
        collect.run_episode(env, player1=BuggyAgent(), map_id=0, episode_id=0, max_steps=5, seed=0)


# --- samples_to_shard --------------------------------------------------------------------


def _sample(value, map_id=0, episode_id=0, step_idx=0):
    frame = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
    state = np.full(S.STATE_LEN, float(value), dtype=np.float32)
    action = np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)
    return collect.Sample(frame, state, action, map_id, episode_id, step_idx)


def test_samples_to_shard_stacks_parallel_arrays():
    samples = [_sample(i, map_id=i, episode_id=0, step_idx=i) for i in range(3)]
    shard = collect.samples_to_shard(samples)
    assert len(shard) == 3
    assert shard.actions is not None
    assert shard.actions.shape == (3, schema.NUM_PLAYERS, schema.ACTION_LEN)
    np.testing.assert_array_equal(shard.map_ids, np.array([0, 1, 2], dtype=np.int32))


def test_samples_to_shard_can_drop_actions():
    samples = [_sample(0)]
    shard = collect.samples_to_shard(samples, with_actions=False)
    assert shard.actions is None


def test_samples_to_shard_rejects_empty():
    with pytest.raises(ValueError):
        collect.samples_to_shard([])


# --- collect_to_shards -------------------------------------------------------------------


def test_collect_to_shards_writes_and_round_trips(tmp_path):
    # Two episodes on two maps, each reset() starting a fresh round on the same env. The env
    # script holds both episodes back to back; each run_episode replays one reset handshake.
    ep0 = _step_episode_blobs([_flat_state(0), _flat_state(1), _flat_state(2)], done_last=True)
    ep1 = _step_episode_blobs([_flat_state(10), _flat_state(11), _flat_state(12)], done_last=True)
    env, _ = _make_env(ep0 + ep1, player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=3))
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        map_ids=[5, 7],
        player1=agents.IdleAgent(),
        max_steps=3,
        seed=0,
        shard_prefix="shard_w0",
        shard_size=10_000,
    )
    assert len(written) == 1  # both episodes fit one shard
    out = shards.read_shard(written[0])
    # 3 samples per episode, two episodes -> 6 samples
    assert out[schema.ARRAY_MAP_IDS].shape[0] == 6
    np.testing.assert_array_equal(
        out[schema.ARRAY_MAP_IDS], np.array([5, 5, 5, 7, 7, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        out[schema.ARRAY_EPISODE_IDS], np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        out[schema.ARRAY_STEP_IDXS], np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    )


def test_collect_to_shards_respects_shard_size(tmp_path):
    # shard_size=2 over a 4-step episode -> multiple shards.
    states = [_flat_state(i) for i in range(4)]
    env, _ = _make_env(
        _step_episode_blobs(states), player2=agents.IdleAgent(), env_config=EnvConfig(max_steps=4)
    )
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        map_ids=[1],
        player1=agents.IdleAgent(),
        max_steps=4,
        seed=0,
        shard_size=2,
    )
    assert len(written) == 2
    total = sum(shards.shard_length(p) for p in written)
    assert total == 4


# --- collect_parallel (single-spec in-process path) --------------------------------------


def test_collect_parallel_single_spec_in_process(tmp_path):
    # The single-spec path runs in-process (no spawn), exercising run_worker via an injected
    # env_factory that builds a real TankEnv (WITH its player2) over a fake transport — still
    # NO live socket.
    states = [_flat_state(i) for i in range(3)]

    def env_factory(spec):
        transport = ScriptedTransport(_step_episode_blobs(states))
        return TankEnv(
            connection=P.Connection(transport),
            player2=agents.IdleAgent(),
            env_config=EnvConfig(max_steps=3),
        )

    def player1_factory(spec):
        return agents.IdleAgent()

    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        map_ids=[2],
        max_steps=3,
        seed=0,
        env_factory=env_factory,
        player1_factory=player1_factory,
    )
    results = collect.collect_parallel([spec])
    assert len(results) == 1
    assert results[0]["worker_id"] == 0
    assert results[0]["num_shards"] == 1
    shard_path = tmp_path / results[0]["shards"][0]
    assert shards.shard_length(shard_path) == 3


def test_collect_parallel_empty_specs_returns_empty():
    assert collect.collect_parallel([]) == []


def test_run_worker_requires_env_factory(tmp_path):
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        map_ids=[0],
        max_steps=1,
        player1_factory=lambda spec: agents.IdleAgent(),
    )
    with pytest.raises(ValueError, match="env_factory"):
        collect.run_worker(spec)


def test_run_worker_requires_player1_factory(tmp_path):
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        map_ids=[0],
        max_steps=1,
        env_factory=lambda spec: None,
    )
    with pytest.raises(ValueError, match="player1_factory"):
        collect.run_worker(spec)
