"""Tests for the collection driver, driving a real ``TankEnv`` over an in-process fake transport.

NO live socket is opened here (the contract: do not unit-test live collection). The transport is
a scripted in-process fake feeding canned handshake acks, state JSON, and length-prefixed pixel
frames through ``core.protocol.Connection`` — exactly the shape the env tests use — so
``run_episode`` / ``collect_to_shards`` are exercised end to end through the SAME gym observation
pipeline RL trains on. Collection pairs a ``player1`` agent + a ``player2`` agent (both REAL
:mod:`pop_trainer.agents`); the env is a bare pure transport and BOTH agents are driven by the
collection loop (player2 acts on its driver-computed flipped view).
"""

import math
import sys

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


def _walls_for(map_id, *, tile_id=7, columns=None):
    """A strict-JSON walls message echoing ``map_id`` (the switched arena path Unity loaded)."""
    return P.encode(
        {
            "type": "walls",
            "map_id": map_id,
            "tileID": tile_id,
            "dims": {"minX": -1, "maxX": 1, "minY": -1, "maxY": 1},
            "columns": columns if columns is not None else {"0": [0]},
        }
    )


def _switch_reset_blobs(first_state, *, switch_walls=None, start_walls=None, fill=(10, 20, 30)):
    """Byte script for a switch_arena reset (mirrors the env test): restart ack, arena_switched ack
    (+ optional walls), start ack (+ optional walls), then state+frame."""
    blobs = [_ack({"restart": True}), _ack({"arena_switched": True})]
    if switch_walls is not None:
        blobs.append(switch_walls)
    blobs.append(_ack({"starting": True}))
    if start_walls is not None:
        blobs.append(start_walls)
    blobs.append(_state_and_frame_bytes(first_state, fill=fill))
    return blobs


def _switch_episode_blobs(states, *, switch_walls=None, start_walls=None, done_last=True):
    """A switch-reset handshake on ``states[0]`` then a step per remaining state."""
    blobs = _switch_reset_blobs(states[0], switch_walls=switch_walls, start_walls=start_walls)
    for i in range(1, len(states)):
        last = i == len(states) - 1
        blobs.append(_state_and_frame_bytes(states[i], fill=(i, i, i), done=done_last and last))
    return blobs


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


class _FixedAgent:
    """A minimal deterministic player: a constant length-5 action, ignores the obs.

    Stands in for the deleted fixed/constant agents now that the data-collection surface is the
    coverage family + RandomAgent; collection's step loop only needs an ``act`` returning a
    length-5 action plus an OPTIONAL ``reset``.
    """

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = [float(x) for x in action]

    def act(self, obs):  # noqa: ARG002
        return list(self._action)

    def reset(self, *, seed=None):  # noqa: ARG002
        pass


class _RecordingAgent:
    """Records the obs handed to its last ``act`` and returns a fixed action."""

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = [float(x) for x in action]
        self.seen_obs = None

    def act(self, obs):
        self.seen_obs = obs
        return list(self._action)

    def reset(self, *, seed=None):  # noqa: ARG002
        pass


class _CycleAgent:
    """Cycles a fixed list of actions on an internal counter (deterministic across steps)."""

    def __init__(self, actions):
        self._actions = [[float(x) for x in a] for a in actions]
        self._step = 0

    def act(self, obs):  # noqa: ARG002
        action = list(self._actions[self._step % len(self._actions)])
        self._step += 1
        return action

    def reset(self, *, seed=None):  # noqa: ARG002
        self._step = 0


def _make_env(blobs, **kwargs):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(connection=conn, **kwargs)
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
    env, transport = _make_env(blobs, env_config=EnvConfig(max_steps=5))
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
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
    env, _ = _make_env(_step_episode_blobs(states, done_last=True))
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
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
    env, _ = _make_env(_step_episode_blobs(states, winner_last=S.PLAYER_1))
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
    )
    assert len(ep) == 2
    assert ep.ended_done


class _MapRecordingAgent:
    """A map-aware player1 whose ``set_map`` records the layout it was handed."""

    def __init__(self):
        self.got_map = None

    def act(self, obs):  # noqa: ARG002
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def set_map(self, layout):
        self.got_map = layout


def _walls_bytes():
    """A strict-JSON walls message in the Unity WallMessage shape (string column keys)."""
    return P.encode(
        {
            "type": "walls",
            "map_id": "maps/center_block.json",
            "tileID": 7,
            "dims": {"minX": -1, "maxX": 1, "minY": -1, "maxY": 1},
            "columns": {"0": [0]},
        }
    )


def test_run_episode_hands_map_to_player1_set_map():
    # When the handshake delivers a walls layout, run_episode forwards info["map"] to a
    # map-aware player1's OPTIONAL set_map hook. The reset script splices a walls blob after the
    # start ack, before the first state (the env's handshake routing).
    states = [_flat_state(0), _flat_state(1)]
    blobs = _step_episode_blobs(states, done_last=True)
    blobs.insert(2, _walls_bytes())  # restart ack, start ack, [walls], first state+frame, ...
    env, _ = _make_env(blobs)
    agent1 = _MapRecordingAgent()
    collect.run_episode(
        env, player1=agent1, player2=_FixedAgent(), map_id=0, episode_id=0, max_steps=100, seed=0
    )
    assert agent1.got_map is not None
    assert agent1.got_map.map_id == "maps/center_block.json"


def test_run_episode_hands_map_to_player2_set_map():
    # The driver owns player2's map hook too (the env no longer does): a walls layout reaches a
    # map-aware player2's OPTIONAL set_map.
    states = [_flat_state(0), _flat_state(1)]
    blobs = _step_episode_blobs(states, done_last=True)
    blobs.insert(2, _walls_bytes())
    env, _ = _make_env(blobs)
    player2 = _MapRecordingAgent()
    collect.run_episode(
        env, player1=_FixedAgent(), player2=player2, map_id=0, episode_id=0, max_steps=100, seed=0
    )
    assert player2.got_map is not None
    assert player2.got_map.map_id == "maps/center_block.json"


def test_run_episode_hands_flipped_view_to_player2():
    # The perspective flip moved from the env to the DRIVER: player2 must be handed
    # split_state_for_opponent(vec) of the state it reacts to (the first state, before any step).
    raw0 = [float(i) for i in range(S.STATE_LEN)]
    raw1 = _flat_state(7.0)
    recorder = _RecordingAgent()
    env, _ = _make_env(_step_episode_blobs([raw0, raw1], done_last=True))
    collect.run_episode(
        env, player1=_FixedAgent(), player2=recorder, map_id=0, episode_id=0, max_steps=100, seed=0
    )
    expected_view = S.split_state_for_opponent(np.asarray(raw0))
    np.testing.assert_array_equal(recorder.seen_obs, expected_view)


def test_run_episode_captures_both_player_actions():
    # THE KEY NEW ASSERTION: both action slots are recorded. player1 returns a fixed action and
    # player2 returns a DISTINCT non-zero action; a non-boundary sample must carry player1's
    # action in slot 0 and player2's actual (non-zero) action in slot 1 — the zeroed-player2 gap
    # is fixed.
    a1 = [1.0, 0.0, 0.0, 0.0, 1.0]
    a2 = [0.0, -1.0, 1.0, 0.0, 1.0]
    states = [_flat_state(0), _flat_state(1)]
    env, transport = _make_env(_step_episode_blobs(states, done_last=True))
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(a1),
        player2=_FixedAgent(a2),
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
    env, transport = _make_env(_step_episode_blobs(states, done_last=True))
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(a1),
        player2=_FixedAgent(),
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


def test_run_episode_supports_cycling_player1():
    # A cycling player1 advances an internal counter each act(); its actions cycle across steps.
    cycle = _CycleAgent([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(_step_episode_blobs(states), env_config=EnvConfig(max_steps=3))
    ep = collect.run_episode(
        env,
        player1=cycle,
        player2=_FixedAgent(),
        map_id=0,
        episode_id=0,
        max_steps=3,
        seed=0,
    )
    np.testing.assert_array_equal(ep.samples[0].action[0], np.array([1, 0, 0, 0, 0], np.float32))
    np.testing.assert_array_equal(ep.samples[1].action[0], np.array([0, 1, 0, 0, 0], np.float32))


def test_shadower_aim_as_player1_through_run_episode():
    # opponent-shadower as player1 over >= 3 steps. P1 at origin, P2 at (3, 4): the aim layer
    # points the unit vector toward P2 -> (0.6, 0.8) every step (the aim is opponent-tracking, not
    # the swept aim). Static positions keep the vector checkable at every step.
    state = _positioned_state((0.0, 0.0), (3.0, 4.0))
    states = [state for _ in range(4)]
    env, _ = _make_env(_step_episode_blobs(states), env_config=EnvConfig(max_steps=4))
    ep = collect.run_episode(
        env,
        player1=agents.CoverageAgent.opponent_shadower(seed=0),  # PLAYER_1 on its own view
        player2=_FixedAgent(),
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
    # Fire is periodic (a modest cadence) — at least one of the recorded steps fired.
    assert any(sample.action[0][4] == 1.0 for sample in ep.samples[:-1])


def test_run_episode_validates_state_width():
    # A short first state surfaces in info["state"] and must fail the schema validator.
    bad = [0.0] * (S.STATE_LEN - 1)
    env, _ = _make_env(_reset_blobs(bad))
    with pytest.raises(ValueError):
        collect.run_episode(
            env,
            player1=_FixedAgent(),
            player2=_FixedAgent(),
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
            env_config=EnvConfig(max_steps=3),
        )
        collect.run_episode(
            env,
            player1=agents.RandomAgent(seed=99),
            player2=agents.RandomAgent(seed=7),
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
    env, _ = _make_env(_step_episode_blobs(states), env_config=EnvConfig(max_steps=5))
    with pytest.raises(TypeError, match="genuine bug"):
        collect.run_episode(
            env,
            player1=BuggyAgent(),
            player2=_FixedAgent(),
            map_id=0,
            episode_id=0,
            max_steps=5,
            seed=0,
        )


# --- switch_arena + tag-from-echo (F5) ---------------------------------------------------


def test_run_episode_no_switch_is_byte_identical_to_today():
    # SEAM RE-PROOF: run_episode with switch_arena=None (the default) sends EXACTLY the no-switch
    # bytes — bare restart + start per reset, NO switch_arena — exactly as before this change.
    states = [_flat_state(0), _flat_state(1)]
    env_a, transport_a = _make_env(_step_episode_blobs(states, done_last=True))
    collect.run_episode(
        env_a,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
    )
    # The handshake portion (before the first step) carries restart + start and NO switch_arena.
    sent = bytes(transport_a.sent)
    assert P.encode({"restart": True}) in sent
    assert P.encode({"start": True}) in sent
    assert b"switch_arena" not in sent


def test_run_episode_switch_sends_target_and_tags_from_echo():
    # The switch target reaches the wire AND the echoed arena (NOT the target) drives the int tag.
    states = [_flat_state(0), _flat_state(1)]
    walls = _walls_for("Arenas/four_pillars.json")
    blobs = _switch_episode_blobs(states, switch_walls=walls, start_walls=walls, done_last=True)
    env, transport = _make_env(blobs)
    map_index = {"Arenas/center_block.json": 0, "Arenas/four_pillars.json": 6}
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=0,  # the INTENDED int (the fallback) — must lose to the echo
        episode_id=0,
        max_steps=100,
        seed=0,
        switch_arena="Arenas/four_pillars.json",
        map_index=map_index,
    )
    assert P.encode({"switch_arena": "Arenas/four_pillars.json"}) in bytes(transport.sent)
    # Every sample is tagged from the ECHO (index 6), not the intended map_id 0.
    assert ep.tag_source == "echo"
    assert ep.echoed_map_id == "Arenas/four_pillars.json"
    assert all(s.map_id == 6 for s in ep.samples)


def test_run_episode_echo_wins_over_a_mismatched_intended_target():
    # A DELIBERATE desync: the runner targeted center_block (intended int 0) but Unity echoes
    # four_pillars (int 6). The ECHO wins — the samples carry 6, never the intended 0.
    states = [_flat_state(0), _flat_state(1)]
    walls = _walls_for("Arenas/four_pillars.json")
    blobs = _switch_episode_blobs(states, switch_walls=walls, start_walls=walls, done_last=True)
    env, _ = _make_env(blobs)
    map_index = {"Arenas/center_block.json": 0, "Arenas/four_pillars.json": 6}
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=0,  # intended center_block
        episode_id=0,
        max_steps=100,
        seed=0,
        switch_arena="Arenas/center_block.json",  # asked for center_block...
        map_index=map_index,
    )
    # ...but Unity loaded four_pillars; the echo (6) wins, the intended (0) loses.
    assert ep.tag_source == "echo"
    assert all(s.map_id == 6 for s in ep.samples)


def test_run_episode_falls_back_when_echo_missing():
    # A walls-ABSENT switch arena: no echo (info["map"] is None) -> the intended int is used and
    # the fallback is FLAGGED (fallback_no_echo), so a desync is detectable rather than silent.
    states = [_flat_state(0), _flat_state(1)]
    blobs = _switch_episode_blobs(states, switch_walls=None, start_walls=None, done_last=True)
    env, _ = _make_env(blobs)
    map_index = {"Arenas/center_block.json": 0, "Arenas/empty.json": 8}
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=8,  # intended empty (walls-absent arena)
        episode_id=0,
        max_steps=100,
        seed=0,
        switch_arena="Arenas/empty.json",
        map_index=map_index,
    )
    assert ep.tag_source == "fallback_no_echo"
    assert ep.echoed_map_id is None
    assert all(s.map_id == 8 for s in ep.samples)


def test_run_episode_flags_unknown_echo_fallback():
    # An echo whose arena is NOT in the index (a desync): fall back to the intended int but FLAG it
    # as fallback_unknown_echo so the run is treated as suspect, never silently mis-tagged.
    states = [_flat_state(0), _flat_state(1)]
    walls = _walls_for("Arenas/some_unmapped_arena.json")
    blobs = _switch_episode_blobs(states, switch_walls=walls, start_walls=walls, done_last=True)
    env, _ = _make_env(blobs)
    map_index = {"Arenas/center_block.json": 0}
    ep = collect.run_episode(
        env,
        player1=_FixedAgent(),
        player2=_FixedAgent(),
        map_id=0,
        episode_id=0,
        max_steps=100,
        seed=0,
        switch_arena="Arenas/center_block.json",
        map_index=map_index,
    )
    assert ep.tag_source == "fallback_unknown_echo"
    assert ep.echoed_map_id == "Arenas/some_unmapped_arena.json"
    assert all(s.map_id == 0 for s in ep.samples)


def test_resolve_map_tag_table():
    # The pure tag resolver: echo-wins, both fallbacks, and the no-index single-map path.
    idx = {"a.json": 1, "b.json": 2}
    layout_a = P.WallLayout(map_id="a.json", tile_id=0, dims=None, columns={})
    layout_x = P.WallLayout(map_id="x.json", tile_id=0, dims=None, columns={})
    assert collect.resolve_map_tag(layout_a, intended_map_id=9, map_index=idx) == (
        1,
        "echo",
        "a.json",
    )
    assert collect.resolve_map_tag(None, intended_map_id=9, map_index=idx) == (
        9,
        "fallback_no_echo",
        None,
    )
    assert collect.resolve_map_tag(layout_x, intended_map_id=9, map_index=idx) == (
        9,
        "fallback_unknown_echo",
        "x.json",
    )
    # No index (single-map / no rotation) -> always the intended int, source "intended".
    assert collect.resolve_map_tag(layout_a, intended_map_id=9, map_index=None) == (
        9,
        "intended",
        "a.json",
    )


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


# --- memory safety: byte-budget shard size + the pre-flight guard (the OOM class) ----------
#
# These reproduce the OOM-incident class in MILLISECONDS with NO real allocation: the estimator
# and the guard are pure functions over injected numbers, and the buffer-boundedness test only
# observes WRITTEN shard lengths over the fake-transport env. The 0.69 MB frame is the live
# 360x640x3 pixel frame whose count-based 10_000 buffer caused the 71% crash.

_INCIDENT_FRAME_NBYTES = 360 * 640 * 3  # 691200 B (~0.69 MB), the live pixel frame.


def test_estimate_peak_buffer_bytes_is_the_exact_product():
    # The estimator is frame_nbytes * shard_size * workers * SPIKE_FACTOR, exactly.
    frame_nbytes = 691200
    shard_size = 500
    workers = 4
    expected = frame_nbytes * shard_size * workers * collect.SPIKE_FACTOR
    assert collect.estimate_peak_buffer_bytes(frame_nbytes, shard_size, workers) == expected
    # SPIKE_FACTOR is the documented 2x compress transient.
    assert collect.SPIKE_FACTOR == 2


def test_default_shard_size_keeps_per_worker_buffer_under_budget():
    # The byte-budget default yields a shard_size whose per-worker buffer (frame_nbytes *
    # shard_size) stays at/under SHARD_BYTES_BUDGET for the live 0.69 MB frame — far under the
    # 10_000 that OOMed.
    shard_size = collect.default_shard_size(_INCIDENT_FRAME_NBYTES)
    assert shard_size >= 1
    assert shard_size * _INCIDENT_FRAME_NBYTES <= collect.SHARD_BYTES_BUDGET
    # And it is materially smaller than the count-based default that caused the crash.
    assert shard_size < 10_000
    # ~384 MB / 0.69 MB ~= 582 frames.
    assert shard_size == collect.SHARD_BYTES_BUDGET // _INCIDENT_FRAME_NBYTES


def test_default_shard_size_floors_to_one_and_rejects_nonpositive():
    # A frame larger than the whole budget still yields a usable shard_size of 1 (never 0).
    assert collect.default_shard_size(collect.SHARD_BYTES_BUDGET * 2) == 1
    with pytest.raises(ValueError):
        collect.default_shard_size(0)


def test_memory_guard_within_budget_returns_estimate_line():
    # Under budget: the guard does NOT raise and returns a human-readable estimate line carrying
    # the GB estimate, the available GB, and the budget threshold (the always-printed startup line).
    line = collect.check_memory_budget(
        frame_nbytes=_INCIDENT_FRAME_NBYTES,
        shard_size=collect.default_shard_size(_INCIDENT_FRAME_NBYTES),
        workers=8,
        available_bytes=64 * 1024**3,
        allow_oversized=False,
    )
    assert "memory estimate" in line
    assert "GB" in line
    assert "available" in line
    assert "budget" in line


def test_memory_guard_reproduces_the_oom_incident_and_aborts():
    # INCIDENT REPRODUCTION (no allocation): the EXACT crash config — 8 workers, the count-based
    # 10_000 buffer, the live 0.69 MB frame -> ~8 * 10000 * 691200 * 2 ~= 110 GB estimate — with
    # only 16 GB available MUST abort with an actionable MemoryError.
    workers = 8
    shard_size = 10_000
    frame_nbytes = 360 * 640 * 3  # the live pixel frame
    available_bytes = 16 * 1024**3  # 16 GB
    # Sanity: the estimate is the ~110 GB that overran the box.
    assert (
        collect.estimate_peak_buffer_bytes(frame_nbytes, shard_size, workers)
        == 8 * 10000 * 691200 * 2
    )
    with pytest.raises(MemoryError) as excinfo:
        collect.check_memory_budget(
            frame_nbytes=frame_nbytes,
            shard_size=shard_size,
            workers=workers,
            available_bytes=available_bytes,
            allow_oversized=False,
        )
    message = str(excinfo.value)
    assert "ABORT" in message
    assert "reduce --workers or --shard-size" in message
    assert "--allow-oversized" in message


def test_memory_guard_allow_oversized_overrides_abort():
    # The same overrunning config does NOT raise when allow_oversized is set; the estimate line is
    # still returned so the caller can print it.
    line = collect.check_memory_budget(
        frame_nbytes=360 * 640 * 3,
        shard_size=10_000,
        workers=8,
        available_bytes=16 * 1024**3,
        allow_oversized=True,
    )
    assert "memory estimate" in line


# --- collect_to_shards -------------------------------------------------------------------


def _plan(switch_arena, p1="p1", p2="p2", intended_map_id=0):
    return collect.EpisodePlan(
        switch_arena=switch_arena, player1=p1, player2=p2, intended_map_id=intended_map_id
    )


def test_collect_to_shards_writes_and_round_trips(tmp_path):
    # Two no-switch episodes (single-map mode), each reset() starting a fresh round on the same env.
    # The env script holds both episodes back to back; each run_episode replays one reset handshake.
    ep0 = _step_episode_blobs([_flat_state(0), _flat_state(1), _flat_state(2)], done_last=True)
    ep1 = _step_episode_blobs([_flat_state(10), _flat_state(11), _flat_state(12)], done_last=True)
    env, _ = _make_env(ep0 + ep1, env_config=EnvConfig(max_steps=3))
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[_plan(None, intended_map_id=5), _plan(None, intended_map_id=7)],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=3,
        seed=0,
        shard_prefix="shard_w0",
        shard_size=10_000,
    )
    assert len(written) == 1  # both episodes fit one shard
    out = shards.read_shard(written[0])
    # 3 samples per episode, two episodes -> 6 samples
    assert out[schema.ARRAY_MAP_IDS].shape[0] == 6
    # No map_index supplied -> the intended ids are used (5, then 7).
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
    env, _ = _make_env(_step_episode_blobs(states), env_config=EnvConfig(max_steps=4))
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[_plan(None, intended_map_id=1)],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=4,
        seed=0,
        shard_size=2,
    )
    assert len(written) == 2
    total = sum(shards.shard_length(p) for p in written)
    assert total == 4


def test_collect_to_shards_buffer_stays_bounded_by_shard_size(tmp_path, monkeypatch):
    # OOM-CLASS (boundedness): with a small shard_size the in-RAM buffer never grows without bound.
    # Observe every chunk handed to samples_to_shard: each WRITTEN shard must be <= shard_size, and
    # the live buffer never exceeds shard_size + one max-len episode (the residual carried between
    # flushes). Many short episodes accumulate, so without the flush the buffer would grow with the
    # run; the bound proves it does not. No real OOM — only lengths are observed.
    shard_size = 3
    n_episodes = 6
    ep_len = 2  # each episode contributes 2 samples (done_last on the 2nd state)
    plan = []
    blobs = []
    for i in range(n_episodes):
        blobs += _step_episode_blobs([_flat_state(i), _flat_state(i + 1)], done_last=True)
        plan.append(_plan(None, intended_map_id=0))
    env, _ = _make_env(blobs, env_config=EnvConfig(max_steps=100))

    chunk_lengths: list[int] = []
    real_samples_to_shard = collect.samples_to_shard

    def spy(samples, **kwargs):
        chunk_lengths.append(len(samples))
        return real_samples_to_shard(samples, **kwargs)

    monkeypatch.setattr(collect, "samples_to_shard", spy)
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=plan,
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=100,
        seed=0,
        shard_size=shard_size,
    )
    # Every full-shard flush wrote exactly shard_size; the final residual flush is <= shard_size.
    full_flushes = chunk_lengths[:-1]
    residual = chunk_lengths[-1]
    assert all(length == shard_size for length in full_flushes)
    assert residual <= shard_size
    # Boundedness ceiling: no chunk ever exceeds shard_size + one max-len episode.
    assert all(length <= shard_size + ep_len for length in chunk_lengths)
    # All samples were written (no loss): 6 episodes x 2 = 12 samples.
    assert sum(shards.shard_length(p) for p in written) == n_episodes * ep_len


def test_collect_to_shards_auto_shard_size_bounds_buffer_by_byte_budget(tmp_path, monkeypatch):
    # The DEFAULT (shard_size=None) derives the buffer bound from the first frame's byte budget, so
    # the per-worker buffer stays bounded with no caller knob. Use a tiny budget so the small test
    # frames trip the auto bound, and assert every written shard length <= the derived bound.
    monkeypatch.setattr(collect, "SHARD_BYTES_BUDGET", FRAME_H * FRAME_W * 3 * 2)
    derived = collect.default_shard_size(FRAME_H * FRAME_W * 3)  # == 2 frames per shard
    n_episodes = 4
    blobs = []
    plan = []
    for i in range(n_episodes):
        blobs += _step_episode_blobs([_flat_state(i), _flat_state(i + 1)], done_last=True)
        plan.append(_plan(None, intended_map_id=0))
    env, _ = _make_env(blobs, env_config=EnvConfig(max_steps=100))
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=plan,
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=100,
        seed=0,
        # shard_size omitted -> auto from the byte budget.
    )
    assert derived == 2
    assert all(shards.shard_length(p) <= derived for p in written)
    # No samples lost: 4 episodes x 2 = 8.
    assert sum(shards.shard_length(p) for p in written) == n_episodes * 2


def test_collect_to_shards_switches_arena_per_episode(tmp_path):
    # Two episodes, each switching to a DISTINCT arena. The env scripts a switch-reset per episode
    # whose walls echo the switched arena; map_index decodes the echo to the on-disk int.
    walls_a = _walls_for("Arenas/center_block.json")
    walls_b = _walls_for("Arenas/empty.json")
    ep0 = _switch_episode_blobs(
        [_flat_state(0), _flat_state(1)], switch_walls=walls_a, start_walls=walls_a, done_last=True
    )
    ep1 = _switch_episode_blobs(
        [_flat_state(2), _flat_state(3)], switch_walls=walls_b, start_walls=walls_b, done_last=True
    )
    env, transport = _make_env(ep0 + ep1, env_config=EnvConfig(max_steps=2))
    map_index = {"Arenas/center_block.json": 3, "Arenas/empty.json": 8}
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[
            _plan("Arenas/center_block.json", intended_map_id=3),
            _plan("Arenas/empty.json", intended_map_id=8),
        ],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=2,
        seed=0,
        map_index=map_index,
    )
    out = shards.read_shard(written[0])
    # The switch targets reached the wire for both episodes.
    sent = bytes(transport.sent)
    assert P.encode({"switch_arena": "Arenas/center_block.json"}) in sent
    assert P.encode({"switch_arena": "Arenas/empty.json"}) in sent
    # Each episode tagged from its echoed arena's index (3 then 8).
    np.testing.assert_array_equal(out[schema.ARRAY_MAP_IDS], np.array([3, 3, 8, 8], dtype=np.int32))


def test_collect_to_shards_raises_on_missing_pool_selector(tmp_path):
    env, _ = _make_env(
        _step_episode_blobs([_flat_state(0), _flat_state(1)]), env_config=EnvConfig(max_steps=2)
    )
    with pytest.raises(KeyError):
        collect.collect_to_shards(
            env,
            out_dir=tmp_path,
            episode_plan=[_plan(None, p1="missing")],
            agent_pool={"p2": _FixedAgent()},
            max_steps=2,
            seed=0,
        )


# --- collect_parallel (single-spec in-process path) --------------------------------------


def test_collect_parallel_single_spec_in_process(tmp_path):
    # The single-spec path runs in-process (no spawn), exercising run_worker via an injected
    # env_factory that builds a BARE TankEnv over a fake transport — still NO live socket. The
    # agents are driver-side, built by agent_pool_factory.
    states = [_flat_state(i) for i in range(3)]

    def env_factory(spec):
        transport = ScriptedTransport(_step_episode_blobs(states))
        return TankEnv(
            connection=P.Connection(transport),
            env_config=EnvConfig(max_steps=3),
        )

    def agent_pool_factory(spec):
        return {"p1": _FixedAgent(), "p2": _FixedAgent()}

    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[_plan(None, intended_map_id=2)],
        max_steps=3,
        seed=0,
        env_factory=env_factory,
        agent_pool_factory=agent_pool_factory,
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
        episode_plan=[_plan(None)],
        max_steps=1,
        agent_pool_factory=lambda spec: {"p1": _FixedAgent(), "p2": _FixedAgent()},
    )
    with pytest.raises(ValueError, match="env_factory"):
        collect.run_worker(spec)


def test_run_worker_requires_agent_pool_factory(tmp_path):
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[_plan(None)],
        max_steps=1,
        env_factory=lambda spec: None,
    )
    with pytest.raises(ValueError, match="agent_pool_factory"):
        collect.run_worker(spec)


# --- progress reporting (the live bar is a pure observability side-channel) ---------------
#
# These exercise the progress plumbing WITHOUT spawning real workers or opening a live socket:
# the per-episode callback counting (driven through the fake-transport env), the TTY-disable
# logic on the bar helper, and the spawn-safety invariants on CollectionSpec. The byte-identical
# guarantee is proven by collecting twice (callback vs no callback) and diffing the shard arrays.


def test_on_episode_done_fires_once_per_episode_with_sample_count(tmp_path):
    # Drive collect_to_shards over the fake transport with an on_episode_done that appends the
    # reported count to a list. Two episodes of 3 samples each (cap=3, no early done) -> two calls,
    # each carrying that episode's len(ep) == 3.
    ep0 = _step_episode_blobs([_flat_state(0), _flat_state(1), _flat_state(2)])
    ep1 = _step_episode_blobs([_flat_state(10), _flat_state(11), _flat_state(12)])
    env, _ = _make_env(ep0 + ep1, env_config=EnvConfig(max_steps=3))
    reported: list[int] = []
    collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[_plan(None, intended_map_id=0), _plan(None, intended_map_id=0)],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=3,
        seed=0,
        on_episode_done=reported.append,
    )
    assert reported == [3, 3]


def test_on_episode_done_count_matches_len_ep_on_early_done(tmp_path):
    # An episode that ends on the env's own boundary records fewer than the cap; the reported count
    # must equal that actual len(ep), not the cap.
    ep0 = _step_episode_blobs([_flat_state(0), _flat_state(1)], done_last=True)  # 2 samples
    env, _ = _make_env(ep0, env_config=EnvConfig(max_steps=100))
    reported: list[int] = []
    collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[_plan(None, intended_map_id=0)],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=100,
        seed=0,
        on_episode_done=reported.append,
    )
    assert reported == [2]


def test_on_episode_done_default_is_noop(tmp_path):
    # The default on_episode_done=None must not raise and must collect normally (today's behavior).
    env, _ = _make_env(
        _step_episode_blobs([_flat_state(0), _flat_state(1)], done_last=True),
        env_config=EnvConfig(max_steps=100),
    )
    written = collect.collect_to_shards(
        env,
        out_dir=tmp_path,
        episode_plan=[_plan(None, intended_map_id=0)],
        agent_pool={"p1": _FixedAgent(), "p2": _FixedAgent()},
        max_steps=100,
        seed=0,
    )
    assert len(written) == 1


def test_progress_bar_disabled_off_tty(monkeypatch):
    # The bar helper disables tqdm when sys.stderr is NOT a TTY: a redirected/captured run emits
    # nothing. The helper reads sys.stderr.isatty() at construction.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    progress = collect._Progress(total=5)
    try:
        assert progress.bar.disable is True
    finally:
        progress.close()


def test_progress_bar_enabled_on_tty(monkeypatch):
    # The bar helper ENABLES tqdm when sys.stderr.isatty() is True.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    progress = collect._Progress(total=5)
    try:
        assert progress.bar.disable is False
    finally:
        progress.close()


def _spy_progress(monkeypatch, captured):
    # Patch collect._Progress with a spy that records the instance AND its construction-time
    # ``disable`` flag (tqdm flips ``disable`` True on close, so the post-run flag is unreliable).
    real_cls = collect._Progress

    def spy(total):
        progress = real_cls(total)
        captured["progress"] = progress
        captured["disable_at_construction"] = progress.bar.disable
        return progress

    monkeypatch.setattr(collect, "_Progress", spy)


def _single_spec_factories(states, max_steps=3):
    def env_factory(spec):  # noqa: ARG001
        return TankEnv(
            connection=P.Connection(ScriptedTransport(_step_episode_blobs(states))),
            env_config=EnvConfig(max_steps=max_steps),
        )

    def agent_pool_factory(spec):  # noqa: ARG001
        return {"p1": _FixedAgent(), "p2": _FixedAgent()}

    return env_factory, agent_pool_factory


def test_collect_parallel_single_spec_disables_bar_off_tty(tmp_path, monkeypatch):
    # End-to-end through the single-spec in-process path with a non-TTY stderr: the run completes
    # and the constructed bar was DISABLED (no progress output leaks into a captured / redirected
    # run). _Progress reads sys.stderr.isatty() at construction.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    env_factory, agent_pool_factory = _single_spec_factories([_flat_state(i) for i in range(3)])
    captured = {}
    _spy_progress(monkeypatch, captured)
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[_plan(None, intended_map_id=0)],
        max_steps=3,
        seed=0,
        env_factory=env_factory,
        agent_pool_factory=agent_pool_factory,
    )
    results = collect.collect_parallel([spec])
    assert results[0]["num_shards"] == 1
    # The bar was disabled off-TTY and still folded the episode's 3 samples into its running total.
    assert captured["disable_at_construction"] is True
    assert captured["progress"].total_samples == 3


def test_collect_parallel_single_spec_advances_bar_on_tty(tmp_path, monkeypatch):
    # On a TTY the single-spec path ENABLES the bar and advances it once per episode; the running
    # sample total folds in each episode's len(ep).
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    env_factory, agent_pool_factory = _single_spec_factories([_flat_state(i) for i in range(3)])
    captured = {}
    _spy_progress(monkeypatch, captured)
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[_plan(None, intended_map_id=0)],
        max_steps=3,
        seed=0,
        env_factory=env_factory,
        agent_pool_factory=agent_pool_factory,
    )
    collect.collect_parallel([spec])
    assert captured["disable_at_construction"] is False
    assert captured["progress"].bar.n == 1  # one episode ticked
    assert captured["progress"].total_samples == 3


def test_single_spec_clears_worker_progress_global(tmp_path, monkeypatch):
    # The single-spec path installs a shim into the module global; after the run the global must be
    # restored to None so no live handle lingers in the module.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    states = [_flat_state(0), _flat_state(1)]

    def env_factory(spec):  # noqa: ARG001
        return TankEnv(
            connection=P.Connection(ScriptedTransport(_step_episode_blobs(states, done_last=True))),
            env_config=EnvConfig(max_steps=100),
        )

    def agent_pool_factory(spec):  # noqa: ARG001
        return {"p1": _FixedAgent(), "p2": _FixedAgent()}

    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[_plan(None, intended_map_id=0)],
        max_steps=100,
        seed=0,
        env_factory=env_factory,
        agent_pool_factory=agent_pool_factory,
    )
    collect.collect_parallel([spec])
    assert collect._WORKER_PROGRESS_QUEUE is None


# --- spawn-safety: the spec stays plain/picklable, no live queue handle leaks into it ------


def test_collection_spec_pickles_round_trip():
    # A spec like build_specs makes (module-level factories, plain extra) must pickle + unpickle.
    import pickle

    from pop_trainer.data import collect_runner

    spec = collect.CollectionSpec(
        worker_id=1,
        out_dir="out/worker_1",
        episode_plan=[_plan("Arenas/center_block.json", intended_map_id=0)],
        max_steps=800,
        seed=10_000,
        map_index={"Arenas/center_block.json": 0},
        env_factory=collect_runner.env_factory,
        agent_pool_factory=collect_runner.agent_pool_factory,
        extra={"exe": "build.exe", "config": "demo.json", "base_port": 50000, "pool": ["random"]},
    )
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.worker_id == 1
    assert restored.episode_plan[0].switch_arena == "Arenas/center_block.json"
    assert restored.map_index == {"Arenas/center_block.json": 0}


def test_collection_spec_carries_no_queue_or_live_handle():
    # No field of CollectionSpec is a multiprocessing queue / live handle: the spec is plain data,
    # and the progress queue rides the Pool initializer, never the spec.
    from dataclasses import fields

    field_names = {f.name for f in fields(collect.CollectionSpec)}
    assert "queue" not in field_names
    assert "progress" not in field_names
    spec = collect.CollectionSpec(
        worker_id=0,
        out_dir="out",
        episode_plan=[_plan(None)],
        max_steps=1,
    )
    for f in fields(collect.CollectionSpec):
        value = getattr(spec, f.name)
        assert not hasattr(value, "get_nowait"), f"{f.name} looks like a queue handle"


# --- byte-unchanged shards: the callback path does not perturb the artifact ----------------


def test_shards_byte_identical_with_and_without_callback(tmp_path):
    # Collect the SAME deterministic episodes twice over identical fake transports: once with
    # on_episode_done=None and once with a real recording callback. Every shard array (frames,
    # states, actions, map_ids, episode_ids, step_idxs) must read back array-equal — the callback
    # is a pure observability side-channel and touches no shard byte.
    states_a = [_flat_state(0), _flat_state(1), _flat_state(2)]
    states_b = [_flat_state(5), _flat_state(6), _flat_state(7)]

    def collect_once(out_dir, on_episode_done):
        env, _ = _make_env(
            _step_episode_blobs(states_a) + _step_episode_blobs(states_b),
            env_config=EnvConfig(max_steps=3),
        )
        return collect.collect_to_shards(
            env,
            out_dir=out_dir,
            episode_plan=[_plan(None, intended_map_id=4), _plan(None, intended_map_id=9)],
            agent_pool={"p1": _FixedAgent((1.0, 0.0, 0.0, 0.0, 1.0)), "p2": _FixedAgent()},
            max_steps=3,
            seed=0,
            shard_prefix="shard_w0",
            shard_size=10_000,
            on_episode_done=on_episode_done,
        )

    reported: list[int] = []
    without = collect_once(tmp_path / "without", None)
    with_cb = collect_once(tmp_path / "with", reported.append)
    assert reported == [3, 3]  # the callback really ran
    assert len(without) == len(with_cb) == 1

    a = shards.read_shard(without[0])
    b = shards.read_shard(with_cb[0])
    assert set(a) == set(b)
    for name in (
        schema.ARRAY_FRAMES,
        schema.ARRAY_STATES,
        schema.ARRAY_ACTIONS,
        schema.ARRAY_MAP_IDS,
        schema.ARRAY_EPISODE_IDS,
        schema.ARRAY_STEP_IDXS,
    ):
        np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} differs with callback")
