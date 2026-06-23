"""Tests for the collection driver's PURE step loop against an in-process fake connection.

NO live socket is opened here (the contract: do not unit-test live collection). The transport
is injected as a scripted fake, so ``run_episode`` / ``collect_to_shards`` are exercised end to
end against deterministic policies.
"""

import math

import numpy as np
import pytest

from pop_trainer.core import state as S
from pop_trainer.data import collect, policies, schema, shards


class FakeConnection:
    """In-process Connection stand-in: scripts the (state, frame) pairs each step yields.

    Constructed with a list of (state_dict, frame) the build "would" return. The first pair is
    consumed by the handshake (the caller passes it as first_state/first_frame); subsequent
    ``receive_state_and_frame`` calls pop the rest in order. ``send`` records the action messages
    so a test can assert the policy outputs were sent.
    """

    def __init__(self, pairs):
        self._pairs = list(pairs)
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def receive_state_and_frame(self):
        if not self._pairs:
            raise AssertionError("FakeConnection ran out of scripted pairs")
        return self._pairs.pop(0)


def _state_dict(value, done=False):
    s = {"state": [float(value)] * S.STATE_LEN}
    if done:
        s["done"] = True
    return s


def _frame(value, h=6, w=8):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_run_episode_records_all_steps_until_cap():
    # 5 steps total, cap at 5 -> records 5 samples, last one carries the zero action.
    first_state, first_frame = _state_dict(0), _frame(0)
    rest = [(_state_dict(i), _frame(i)) for i in range(1, 5)]
    conn = FakeConnection(rest)
    ep = collect.run_episode(
        conn,
        first_state=first_state,
        first_frame=first_frame,
        p1_policy=policies.idle_policy,
        p2_policy=policies.idle_policy,
        map_id=3,
        episode_id=0,
        max_steps=5,
    )
    assert len(ep) == 5
    assert not ep.ended_done
    # step indices are 0..4
    assert [s.step_idx for s in ep.samples] == [0, 1, 2, 3, 4]
    # frames are time-aligned with the scripted pairs (0,1,2,3,4)
    for i, s in enumerate(ep.samples):
        np.testing.assert_array_equal(s.frame, _frame(i))
    # 4 actions were sent (one per transition; none after the final/capped step)
    assert len(conn.sent) == 4
    # the final recorded step carries the zero action
    np.testing.assert_array_equal(
        ep.samples[-1].action, np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)
    )


def test_run_episode_stops_on_done():
    first_state, first_frame = _state_dict(0), _frame(0)
    # the second state is "done"
    rest = [(_state_dict(1, done=True), _frame(1))]
    conn = FakeConnection(rest)
    ep = collect.run_episode(
        conn,
        first_state=first_state,
        first_frame=first_frame,
        p1_policy=policies.idle_policy,
        p2_policy=policies.idle_policy,
        map_id=0,
        episode_id=0,
        max_steps=100,  # cap not reached; done stops it
    )
    assert len(ep) == 2
    assert ep.ended_done
    # one action was sent (to advance from step 0 to the done step), none after done
    assert len(conn.sent) == 1


def test_run_episode_records_applied_action():
    # A constant policy: the action recorded with a non-terminal sample is the policy output.
    a1 = [1.0, 0.0, 0.0, 0.0, 1.0]
    a2 = [0.0, -1.0, 0.0, 0.0, 0.0]
    first_state, first_frame = _state_dict(0), _frame(0)
    rest = [(_state_dict(1, done=True), _frame(1))]
    conn = FakeConnection(rest)
    ep = collect.run_episode(
        conn,
        first_state=first_state,
        first_frame=first_frame,
        p1_policy=policies.ConstantPolicy(a1),
        p2_policy=policies.ConstantPolicy(a2),
        map_id=0,
        episode_id=0,
        max_steps=100,
    )
    np.testing.assert_array_equal(ep.samples[0].action, np.array([a1, a2], dtype=np.float32))
    # and that exact action was sent on the wire
    assert conn.sent == [{1: a1, 2: a2}]


def test_run_episode_supports_scripted_cycle_policy():
    # ScriptedCyclePolicy takes (state, step) — the loop must adapt to that signature.
    cycle = policies.ScriptedCyclePolicy([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    first_state, first_frame = _state_dict(0), _frame(0)
    rest = [(_state_dict(i), _frame(i)) for i in range(1, 3)]
    conn = FakeConnection(rest)
    ep = collect.run_episode(
        conn,
        first_state=first_state,
        first_frame=first_frame,
        p1_policy=cycle,
        p2_policy=cycle,
        map_id=0,
        episode_id=0,
        max_steps=3,
    )
    # step 0 -> cycle[0], step 1 -> cycle[1]
    np.testing.assert_array_equal(ep.samples[0].action[0], np.array([1, 0, 0, 0, 0], np.float32))
    np.testing.assert_array_equal(ep.samples[1].action[0], np.array([0, 1, 0, 0, 0], np.float32))


def _positioned_state(p1_xy, p2_xy, done=False):
    """A 52-float state dict with P1/P2 at the given (x, y); everything else zero."""
    s = [0.0] * S.STATE_LEN
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_X] = float(p1_xy[0])
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_Y] = float(p1_xy[1])
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = float(p2_xy[0])
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = float(p2_xy[1])
    d = {"state": s}
    if done:
        d["done"] = True
    return d


def test_aim_policy_through_run_episode_both_slots():
    # The aim policy must work as a REAL collection policy via run_episode in BOTH slots, across
    # >= 3 steps (the old per-step crash point was step index >= 2). P1 at origin, P2 at (3, 4):
    # P1 aims toward P2 -> (0.6, 0.8); P2 aims toward P1 -> (-0.6, -0.8). Static positions keep
    # the expected unit vectors checkable at every step.
    p1_xy, p2_xy = (0.0, 0.0), (3.0, 4.0)
    pairs = [(_positioned_state(p1_xy, p2_xy), _frame(i)) for i in range(4)]
    first_state, first_frame = pairs[0]
    conn = FakeConnection(pairs[1:])

    p1_policy = policies.aim_at_opponent_policy  # 1-arg, default PLAYER_1
    p2_policy = policies.AimAtOpponentPolicy(player=S.PLAYER_2)  # player-bound, 1-arg

    ep = collect.run_episode(
        conn,
        first_state=first_state,
        first_frame=first_frame,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        map_id=0,
        episode_id=0,
        max_steps=4,
    )
    # (a) no crash, all 4 steps recorded, including step indices >= 2 (the old crash point).
    assert len(ep) == 4
    assert [s.step_idx for s in ep.samples] == [0, 1, 2, 3]
    # (c) the action sent at step index 2 exists and is correct (old code raised here).
    assert len(conn.sent) == 3  # transitions for steps 0,1,2 (none after the capped step 3)
    # (b) every applied action aims correctly toward the opponent for that slot's perspective.
    for sample in ep.samples[:-1]:  # the final/capped sample carries the zero action
        a1 = sample.action[0]
        a2 = sample.action[1]
        assert math.isclose(a1[2], 0.6, abs_tol=1e-6)
        assert math.isclose(a1[3], 0.8, abs_tol=1e-6)
        assert a1[4] == 1.0
        assert math.isclose(a2[2], -0.6, abs_tol=1e-6)
        assert math.isclose(a2[3], -0.8, abs_tol=1e-6)
        assert a2[4] == 1.0
    # the wire messages carry exactly those per-slot aims.
    for msg in conn.sent:
        assert math.isclose(msg[1][2], 0.6, abs_tol=1e-6)
        assert math.isclose(msg[2][2], -0.6, abs_tol=1e-6)


def test_aim_factory_closure_through_run_episode():
    # The closure form aims its bound slot correctly too (P2 aiming at P1).
    pairs = [(_positioned_state((0.0, 0.0), (3.0, 4.0)), _frame(i)) for i in range(3)]
    conn = FakeConnection(pairs[1:])
    p2_policy = policies.aim_at_opponent_factory(S.PLAYER_2)
    ep = collect.run_episode(
        conn,
        first_state=pairs[0][0],
        first_frame=pairs[0][1],
        p1_policy=policies.idle_policy,
        p2_policy=p2_policy,
        map_id=0,
        episode_id=0,
        max_steps=3,
    )
    a2 = ep.samples[0].action[1]
    assert math.isclose(a2[2], -0.6, abs_tol=1e-6)
    assert math.isclose(a2[3], -0.8, abs_tol=1e-6)


def test_policy_arity_classification():
    # 1-arg shapes -> takes_step False; (state, step) and *args -> True; non-introspectable
    # defensively -> False (1-arg, the safe shape that never passes an unexpected step).
    assert collect._policy_takes_step(policies.idle_policy) is False
    assert collect._policy_takes_step(policies.ConstantPolicy([0, 0, 0, 0, 0])) is False
    assert collect._policy_takes_step(policies.aim_at_opponent_policy) is False
    assert collect._policy_takes_step(policies.AimAtOpponentPolicy(player=S.PLAYER_2)) is False
    assert collect._policy_takes_step(policies.ScriptedCyclePolicy([[0, 0, 0, 0, 0]])) is True

    def varargs_policy(*args):
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    assert collect._policy_takes_step(varargs_policy) is True
    # A callable whose signature can't be introspected (raises ValueError) falls back to 1-arg.
    assert collect._policy_takes_step(range) is False


def test_apply_policy_does_not_mask_internal_typeerror():
    # A 1-arg policy that raises TypeError INSIDE its body must propagate, not be retried/masked.
    def buggy_policy(state):
        raise TypeError("genuine bug inside the policy body")

    conn = FakeConnection([(_state_dict(1), _frame(1))])
    with pytest.raises(TypeError, match="genuine bug"):
        collect.run_episode(
            conn,
            first_state=_state_dict(0),
            first_frame=_frame(0),
            p1_policy=buggy_policy,
            p2_policy=policies.idle_policy,
            map_id=0,
            episode_id=0,
            max_steps=5,
        )


def test_collect_to_shards_writes_and_round_trips(tmp_path):
    # Two episodes on two maps via a fake handshake; assert shards on disk carry the right maps.
    def make_pairs(start, n, done_last=True):
        pairs = [(_state_dict(start), _frame(start))]  # first (consumed by handshake)
        for i in range(1, n):
            last = i == n - 1
            pairs.append((_state_dict(start + i, done=done_last and last), _frame(start + i)))
        return pairs

    # Per episode the connection must yield its own scripted stream. We drive episodes serially,
    # so build a connection whose pairs cover both episodes; the handshake pops the first of each.
    ep0 = make_pairs(0, 3)
    ep1 = make_pairs(10, 3)

    class MultiEpisodeConn(FakeConnection):
        def __init__(self):
            super().__init__([])
            self._episodes = [ep0, ep1]
            self._cursor = None

        def begin(self, episode_pairs):
            # handshake hands back the first pair; the rest stream via receive_state_and_frame
            self._pairs = list(episode_pairs[1:])
            return episode_pairs[0]

    conn = MultiEpisodeConn()
    episodes = iter([ep0, ep1])

    def handshake(c, map_id):
        return c.begin(next(episodes))

    written = collect.collect_to_shards(
        conn,
        out_dir=tmp_path,
        handshake=handshake,
        map_ids=[5, 7],
        p1_policy=policies.idle_policy,
        p2_policy=policies.idle_policy,
        max_steps=3,
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


def test_collect_to_shards_respects_shard_size(tmp_path):
    # shard_size=2 over a 4-step episode -> multiple shards.
    pairs = [(_state_dict(i), _frame(i)) for i in range(4)]
    conn = FakeConnection(pairs[1:])

    def handshake(c, map_id):
        return pairs[0]

    written = collect.collect_to_shards(
        conn,
        out_dir=tmp_path,
        handshake=handshake,
        map_ids=[1],
        p1_policy=policies.idle_policy,
        p2_policy=policies.idle_policy,
        max_steps=4,
        shard_size=2,
    )
    assert len(written) == 2
    total = sum(shards.shard_length(p) for p in written)
    assert total == 4


def test_collect_parallel_single_spec_in_process(tmp_path):
    # The single-spec path runs in-process (no spawn), exercising run_worker via injected
    # factories with a fake connection — still NO live socket.
    pairs = [(_state_dict(i), _frame(i)) for i in range(3)]

    def transport_factory(spec):
        return FakeConnection(pairs[1:])

    def handshake_factory(spec):
        def handshake(conn, map_id):
            return pairs[0]

        return handshake

    def policy_factory(spec):
        return policies.idle_policy, policies.idle_policy

    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        map_ids=[2],
        max_steps=3,
        transport_factory=transport_factory,
        handshake_factory=handshake_factory,
        policy_factory=policy_factory,
    )
    results = collect.collect_parallel([spec])
    assert len(results) == 1
    assert results[0]["worker_id"] == 0
    assert results[0]["num_shards"] == 1
    shard_path = tmp_path / results[0]["shards"][0]
    assert shards.shard_length(shard_path) == 3


def test_run_episode_validates_state_width():
    bad = {"state": [0.0] * (S.STATE_LEN - 1)}
    conn = FakeConnection([])
    with pytest.raises(ValueError):
        collect.run_episode(
            conn,
            first_state=bad,
            first_frame=_frame(0),
            p1_policy=policies.idle_policy,
            p2_policy=policies.idle_policy,
            map_id=0,
            episode_id=0,
            max_steps=1,
        )
