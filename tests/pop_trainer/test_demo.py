"""Tests for ``pop_trainer.demo`` — the episode loop driven over an in-process fake transport.

NO live socket and NO subprocess: a scripted in-process fake feeds canned handshake acks, a
walls message, state JSON, and length-prefixed pixel frames through ``core.protocol.Connection``
(the same shape the env / collect tests use), so :func:`run_demo_episode` is exercised end to
end through a REAL ``TankEnv``. ``main`` (subprocess launch + socket connect) is not exercised.
"""

import numpy as np
import pytest

from pop_trainer import agents, demo
from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig
from pop_trainer.env.tank_env import TankEnv

FRAME_H, FRAME_W = demo.FRAME_HEIGHT, demo.FRAME_WIDTH


# --- wire-byte builders (mirror tests/pop_trainer/env/test_tank_env.py) -------------------


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


def _walls_bytes(map_id="Arenas/custom1.json", tile_id=7, dims=None, columns=None):
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


def _episode_blobs_with_walls(states, *, walls=None, winner_last=None, done_last=False):
    """A full-episode byte script: reset handshake INCLUDING a walls message, then one
    state+frame per remaining state, the last carrying ``winner`` / ``done``."""
    blobs = [
        _ack({"restart": True}),
        _ack({"starting": True}),
        walls if walls is not None else _walls_bytes(),
        _state_and_frame_bytes(states[0], fill=(0, 0, 0)),
    ]
    for i in range(1, len(states)):
        last = i == len(states) - 1
        blobs.append(
            _state_and_frame_bytes(
                states[i],
                fill=(i, i, i),
                winner=winner_last if last else None,
                done=done_last and last,
            )
        )
    return blobs


class _FixedAgent:
    """A minimal deterministic player for tests: a constant action, ignores the obs.

    Stands in for the deleted fixed/constant agents now that the data-collection surface is the
    coverage family + RandomAgent; the demo loop only needs an ``act`` that returns a length-5
    action.
    """

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = [float(x) for x in action]

    def act(self, obs):  # noqa: ARG002
        return list(self._action)


def _make_env(blobs, *, max_steps=100):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(
        connection=conn,
        frame_shape=demo.FRAME_SHAPE,
        env_config=EnvConfig(max_steps=max_steps),
    )
    return env, transport


class _MapRecordingAgent:
    """A map-aware player whose ``set_map`` records the layout it was handed."""

    def __init__(self):
        self.got_map = None

    def act(self, obs):  # noqa: ARG002
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def set_map(self, layout):
        self.got_map = layout


class _RecordingAgent:
    """Records the obs handed to its last ``act`` and returns a fixed action."""

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = [float(x) for x in action]
        self.seen_obs = None

    def act(self, obs):
        self.seen_obs = obs
        return list(self._action)


# --- run_demo_episode --------------------------------------------------------------------


def test_run_demo_episode_hands_map_to_player1_set_map():
    # When the env tracks a walls layout, run_demo_episode forwards it to a map-aware player1.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent1 = _MapRecordingAgent()
    result = demo.run_demo_episode(env, agent1, _FixedAgent(), max_steps=100)
    assert result.map is not None
    assert agent1.got_map is result.map  # the tracked layout reached the agent's set_map


def test_run_demo_episode_hands_map_to_player2_set_map():
    # Both agents are driver-side now: a tracked walls layout reaches a map-aware agent2's set_map.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent2 = _MapRecordingAgent()
    result = demo.run_demo_episode(env, _FixedAgent(), agent2, max_steps=100)
    assert result.map is not None
    assert agent2.got_map is result.map


def test_run_demo_episode_hands_flipped_view_to_player2():
    # The perspective flip lives in the driver: agent2 is handed split_state_for_opponent(vec) of
    # the first state (the state it reacts to before any step).
    raw0 = [float(i) for i in range(S.STATE_LEN)]
    states = [raw0, _flat_state(7.0)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent2 = _RecordingAgent()
    demo.run_demo_episode(env, _FixedAgent(), agent2, max_steps=100)
    expected_view = S.split_state_for_opponent(np.asarray(raw0))
    np.testing.assert_array_equal(agent2.seen_obs, expected_view)


def test_run_demo_episode_reaches_winner_terminal_and_tracks_map():
    # Full episode: reset (with walls) + 3 steps, the last reporting player1 as the winner.
    states = [_flat_state(i) for i in range(4)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, winner_last=S.PLAYER_1),
        max_steps=100,
    )
    result = demo.run_demo_episode(
        env, _FixedAgent(), agents.CoverageAgent.aggressive(seed=0), max_steps=100
    )

    assert result.steps == 3  # one step per state after the first
    assert result.terminated
    assert not result.truncated
    assert result.winner == S.PLAYER_1
    assert result.outcome == "win"
    # The walls message in the handshake was tracked into info["map"].
    assert result.map is not None
    assert result.map.map_id == "Arenas/custom1.json"
    assert result.map.occupied  # at least one wall cell


def test_run_demo_episode_player2_loss_outcome():
    # The terminal reports player2 as the winner -> player1's outcome is a loss.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, winner_last=S.PLAYER_2),
        max_steps=100,
    )
    result = demo.run_demo_episode(env, _FixedAgent(), _FixedAgent(), max_steps=100)
    assert result.terminated
    assert result.winner == S.PLAYER_2
    assert result.outcome == "loss"


def test_run_demo_episode_truncates_at_max_steps():
    # No terminal in the script; the loop stops at max_steps with truncated semantics.
    states = [_flat_state(i) for i in range(4)]
    env, _ = _make_env(_episode_blobs_with_walls(states), max_steps=3)
    result = demo.run_demo_episode(env, _FixedAgent(), _FixedAgent(), max_steps=3)
    assert result.steps == 3
    assert not result.terminated
    assert result.truncated
    assert result.winner is None
    assert result.outcome is None


def test_run_demo_episode_drives_player1_actions_on_the_wire():
    # opponent-shadower as player1: P1 at origin, P2 at (3, 4) -> the aim layer points the unit
    # vector (0.6, 0.8) at the opponent and the first step fires. The first action sent on the
    # wire must carry that aim in slot "1".
    state = [0.0] * S.STATE_LEN
    state[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = 3.0
    state[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = 4.0
    states = [state, state, state]
    env, transport = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    demo.run_demo_episode(
        env, agents.CoverageAgent.opponent_shadower(seed=0), _FixedAgent(), max_steps=100
    )
    # Decode ONLY the first step message off the wire (sent holds several back to back); a
    # Connection over the sent bytes reads exactly one top-level object via its brace scan.
    step_start = bytes(transport.sent).index(b'{"1"')
    reader = P.Connection(ScriptedTransport([bytes(transport.sent[step_start:])]))
    sent = reader.receive()
    a1 = sent["1"]
    assert a1[2] == pytest.approx(0.6, abs=1e-6)
    assert a1[3] == pytest.approx(0.8, abs=1e-6)
    assert a1[4] == 1.0


# --- agent selectors ---------------------------------------------------------------------


def test_default_selectors_pick_coverage_and_shadower():
    assert demo.DEFAULT_PLAYER1 == "aggressive-coverage"
    assert demo.DEFAULT_PLAYER2 == "opponent-shadower"
    assert isinstance(demo.make_agent(demo.DEFAULT_PLAYER1), agents.CoverageAgent)
    assert isinstance(demo.make_agent(demo.DEFAULT_PLAYER2), agents.CoverageAgent)


def test_default_selectors_are_visibly_different_policies():
    # Both are the coverage family, but the default pairing is distinct presets (the player2
    # shadower aims at the opponent while player1 sweeps aim) — a visibly different demo.
    assert demo.DEFAULT_PLAYER1 != demo.DEFAULT_PLAYER2


def test_selector_surface_is_the_new_family_plus_random():
    assert set(demo.AGENT_SELECTORS) == {
        "aggressive-coverage",
        "wall-hugger",
        "opponent-shadower",
        "random",
    }


def test_no_deleted_selectors_remain():
    for gone in ("aim-at-player2", "explorer", "aim-sweep", "spray", "perimeter", "idle"):
        assert gone not in demo.AGENT_SELECTORS


def test_every_selector_builds_a_valid_agent():
    for name in demo.AGENT_SELECTORS:
        agent = demo.make_agent(name, seed=0)
        assert hasattr(agent, "act")


def test_make_agent_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        demo.make_agent("not-a-real-agent")


def test_seed_is_threaded_into_seeded_agents():
    # A seeded selector replays identically; two same-seed builds produce the same first action.
    a = demo.make_agent("random", seed=11)
    b = demo.make_agent("random", seed=11)
    np.testing.assert_array_equal(a.act(None), b.act(None))


# --- import-boundary guard ---------------------------------------------------------------


def test_demo_imports_no_forbidden_packages():
    # The demo is an application over env / agents / core; it must not reach into the model /
    # data / training layers or tank_twin. Parse the module's actual import STATEMENTS (an AST
    # walk, so a docstring mention of a name is never mistaken for an import).
    import ast
    import inspect

    forbidden = (
        "pop_trainer.models",
        "pop_trainer.data",
        "pop_trainer.rl",
        "pop_trainer.pretraining",
        "tank_twin",
    )

    def _is_forbidden(module: str) -> bool:
        return any(module == f or module.startswith(f + ".") for f in forbidden)

    tree = ast.parse(inspect.getsource(demo))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    offenders = [m for m in imported if _is_forbidden(m)]
    assert not offenders, f"demo must not import: {offenders}"
