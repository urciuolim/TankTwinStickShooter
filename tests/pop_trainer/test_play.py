"""Tests for ``pop_trainer.play`` — the episode loop driven over an in-process fake transport.

NO live socket and NO subprocess: a scripted in-process fake feeds canned handshake acks, a
walls message, state JSON, and length-prefixed pixel frames through ``core.protocol.Connection``
(the same shape the env / collect tests use), so :func:`run_play_episode` is exercised end to
end through a REAL ``TankEnv``. ``main`` (subprocess launch + socket connect) is not exercised.
RL players are driven with a STUBBED ``predict`` — no real SB3 / no real ``.zip``.
"""

import numpy as np
import pytest

from pop_trainer import agents, play
from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig
from pop_trainer.env.tank_env import TankEnv

FRAME_H, FRAME_W = play.FRAME_HEIGHT, play.FRAME_WIDTH


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


def _episode_blobs_with_walls(states, *, walls=None, winner_last=None, done_last=False, fills=None):
    """A full-episode byte script: reset handshake INCLUDING a walls message, then one
    state+frame per remaining state, the last carrying ``winner`` / ``done``.

    ``fills`` (OPTIONAL) is a per-state list of ``(r, g, b)`` frame fills; when ``None`` the first
    state is black and later states use ``(i, i, i)`` (the legacy fill scheme).
    """

    def _fill(i):
        if fills is not None:
            return fills[i]
        return (0, 0, 0) if i == 0 else (i, i, i)

    blobs = [
        _ack({"restart": True}),
        _ack({"starting": True}),
        walls if walls is not None else _walls_bytes(),
        _state_and_frame_bytes(states[0], fill=_fill(0)),
    ]
    for i in range(1, len(states)):
        last = i == len(states) - 1
        blobs.append(
            _state_and_frame_bytes(
                states[i],
                fill=_fill(i),
                winner=winner_last if last else None,
                done=done_last and last,
            )
        )
    return blobs


class _FixedAgent:
    """A minimal deterministic player for tests: a constant action, ignores the obs.

    Stands in for the deleted fixed/constant agents now that the data-collection surface is the
    coverage family + RandomAgent; the play loop only needs an ``act`` that returns a length-5
    action.
    """

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = [float(x) for x in action]

    def act(self, obs):  # noqa: ARG002
        return list(self._action)


def _state_adapter(agent, *, slot):
    """Wrap a raw test agent in the play StateAdapter for the given slot."""
    return play.StateAdapter(agent, slot=slot)


def _make_env(blobs, *, max_steps=100):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    env = TankEnv(
        connection=conn,
        frame_shape=play.FRAME_SHAPE,
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


class _RecordingPredict:
    """A fake SB3 ``predict``: records the frame it saw and returns a fixed (action, state).

    Stands in for ``model.predict`` so an RL :class:`play.PixelsAdapter` can be driven with NO real
    SB3 and NO real ``.zip``. The recorded frame lets a test assert the RIGHT perspective reached
    the model (unflipped for player1, ``player2_frame`` for player2).
    """

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self._action = np.asarray(action, dtype=np.float32)
        self.seen_frame = None
        self.calls = 0

    def __call__(self, obs, deterministic=False):  # noqa: ARG002
        self.seen_frame = np.asarray(obs)
        self.calls += 1
        return self._action, None


class _ModelWith:
    """A minimal model object exposing only ``.predict`` — the make_rl_player injection seam."""

    def __init__(self, predict):
        self.predict = predict


# --- run_play_episode --------------------------------------------------------------------


def test_run_play_episode_hands_map_to_player1_set_map():
    # When the env tracks a walls layout, run_play_episode forwards it to a map-aware player1.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent1 = _MapRecordingAgent()
    result = play.run_play_episode(
        env, _state_adapter(agent1, slot=S.PLAYER_1), _state_adapter(_FixedAgent(), slot=S.PLAYER_2)
    )
    assert result.map is not None
    assert agent1.got_map is result.map  # the tracked layout reached the agent's set_map


def test_run_play_episode_hands_map_to_player2_set_map():
    # Both players are driver-side: a tracked walls layout reaches a map-aware player2's set_map.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent2 = _MapRecordingAgent()
    result = play.run_play_episode(
        env, _state_adapter(_FixedAgent(), slot=S.PLAYER_1), _state_adapter(agent2, slot=S.PLAYER_2)
    )
    assert result.map is not None
    assert agent2.got_map is result.map


def test_run_play_episode_hands_flipped_state_view_to_player2():
    # The perspective flip lives in the StateAdapter: a player2 state agent is handed
    # split_state_for_opponent(vec) of the first state (the state it reacts to before any step).
    raw0 = [float(i) for i in range(S.STATE_LEN)]
    states = [raw0, _flat_state(7.0)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent2 = _RecordingAgent()
    play.run_play_episode(
        env, _state_adapter(_FixedAgent(), slot=S.PLAYER_1), _state_adapter(agent2, slot=S.PLAYER_2)
    )
    expected_view = S.split_state_for_opponent(np.asarray(raw0))
    np.testing.assert_array_equal(agent2.seen_obs, expected_view)


def test_run_play_episode_hands_unflipped_state_view_to_player1():
    # player1's StateAdapter passes the UNFLIPPED state through to the agent.
    raw0 = [float(i) for i in range(S.STATE_LEN)]
    states = [raw0, _flat_state(7.0)]
    env, _ = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    agent1 = _RecordingAgent()
    play.run_play_episode(
        env, _state_adapter(agent1, slot=S.PLAYER_1), _state_adapter(_FixedAgent(), slot=S.PLAYER_2)
    )
    np.testing.assert_array_equal(agent1.seen_obs, raw0)


def test_run_play_episode_reaches_winner_terminal_and_tracks_map():
    # Full episode: reset (with walls) + 3 steps, the last reporting player1 as the winner.
    states = [_flat_state(i) for i in range(4)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, winner_last=S.PLAYER_1),
        max_steps=100,
    )
    result = play.run_play_episode(
        env,
        _state_adapter(_FixedAgent(), slot=S.PLAYER_1),
        _state_adapter(agents.CoverageAgent.aggressive(seed=0), slot=S.PLAYER_2),
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


def test_run_play_episode_player2_loss_outcome():
    # The terminal reports player2 as the winner -> player1's outcome is a loss.
    states = [_flat_state(i) for i in range(3)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, winner_last=S.PLAYER_2),
        max_steps=100,
    )
    result = play.run_play_episode(
        env,
        _state_adapter(_FixedAgent(), slot=S.PLAYER_1),
        _state_adapter(_FixedAgent(), slot=S.PLAYER_2),
    )
    assert result.terminated
    assert result.winner == S.PLAYER_2
    assert result.outcome == "loss"


def test_run_play_episode_truncates_at_max_steps():
    # No terminal in the script; the loop stops at max_steps with truncated semantics.
    states = [_flat_state(i) for i in range(4)]
    env, _ = _make_env(_episode_blobs_with_walls(states), max_steps=3)
    result = play.run_play_episode(
        env,
        _state_adapter(_FixedAgent(), slot=S.PLAYER_1),
        _state_adapter(_FixedAgent(), slot=S.PLAYER_2),
        max_steps=3,
    )
    assert result.steps == 3
    assert not result.terminated
    assert result.truncated
    assert result.winner is None
    assert result.outcome is None


def test_run_play_episode_drives_player1_actions_on_the_wire():
    # opponent-shadower as player1: P1 at origin, P2 at (3, 4) -> the aim layer points the unit
    # vector (0.6, 0.8) at the opponent and the first step fires. The first action sent on the
    # wire must carry that aim in slot "1".
    state = [0.0] * S.STATE_LEN
    state[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = 3.0
    state[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = 4.0
    states = [state, state, state]
    env, transport = _make_env(_episode_blobs_with_walls(states, done_last=True), max_steps=100)
    play.run_play_episode(
        env,
        _state_adapter(agents.CoverageAgent.opponent_shadower(seed=0), slot=S.PLAYER_1),
        _state_adapter(_FixedAgent(), slot=S.PLAYER_2),
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


# --- RL pixels adapter (stubbed predict; no real SB3 / no real .zip) ----------------------


def test_rl_player1_sees_the_unflipped_current_frame():
    # An RL player1 adapter is handed the env's CURRENT frame as-is. Use an asymmetric fill so an
    # accidental R/B swap would be detectable; player1 must see it UNCHANGED.
    fill = (10, 20, 30)
    states = [_flat_state(0), _flat_state(1)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, done_last=True, fills=[fill, fill]),
        max_steps=100,
    )
    predict = _RecordingPredict()
    rl_player1 = play.make_rl_player("ignored.zip", slot=S.PLAYER_1, model=_ModelWith(predict))
    play.run_play_episode(env, rl_player1, _state_adapter(_FixedAgent(), slot=S.PLAYER_2))

    assert predict.calls >= 1
    seen = predict.seen_frame  # the FIRST step's frame (recorded on every call; first == this fill)
    assert seen.shape == play.FRAME_SHAPE
    # Unflipped: R==10, G==20, B==30 everywhere (the env's raw frame).
    assert int(seen[0, 0, 0]) == 10
    assert int(seen[0, 0, 1]) == 20
    assert int(seen[0, 0, 2]) == 30


def test_rl_player2_sees_the_flipped_frame():
    # An RL player2 adapter is handed env.player2_frame() — the R/B-swapped frame. With an
    # asymmetric fill the swap is observable: R and B are exchanged, G held.
    fill = (10, 20, 30)
    states = [_flat_state(0), _flat_state(1)]
    env, _ = _make_env(
        _episode_blobs_with_walls(states, done_last=True, fills=[fill, fill]),
        max_steps=100,
    )
    predict = _RecordingPredict()
    rl_player2 = play.make_rl_player("ignored.zip", slot=S.PLAYER_2, model=_ModelWith(predict))
    play.run_play_episode(env, _state_adapter(_FixedAgent(), slot=S.PLAYER_1), rl_player2)

    seen = predict.seen_frame
    assert seen.shape == play.FRAME_SHAPE
    # Flipped: the (10, 20, 30) source becomes (30, 20, 10) — R/B swapped, G held.
    assert int(seen[0, 0, 0]) == 30
    assert int(seen[0, 0, 1]) == 20
    assert int(seen[0, 0, 2]) == 10
    # And it equals flip_frame_perspective of the env's raw frame.
    raw = np.full(play.FRAME_SHAPE, 0, dtype=np.uint8)
    raw[:, :, 0], raw[:, :, 1], raw[:, :, 2] = fill
    np.testing.assert_array_equal(seen, S.flip_frame_perspective(raw))


def test_rl_adapter_acts_deterministically_via_predict():
    # The PixelsAdapter calls predict(frame, deterministic=True) and returns predict's action[0].
    predict = _RecordingPredict(action=(0.1, 0.2, 0.3, 0.4, 1.0))
    adapter = play.make_rl_player("ignored.zip", slot=S.PLAYER_1, model=_ModelWith(predict))
    frame = np.zeros(play.FRAME_SHAPE, dtype=np.uint8)
    action = adapter.act(frame, _flat_state(0))
    np.testing.assert_allclose(np.asarray(action), (0.1, 0.2, 0.3, 0.4, 1.0), rtol=1e-6)


# --- player-spec parsing -----------------------------------------------------------------


def test_parse_player_spec_human():
    spec = play.parse_player_spec("human")
    assert spec.is_human
    assert not spec.is_rl
    assert spec.selector is None
    assert spec.raw == "human"


def test_parse_player_spec_each_selector():
    for name in agents.AGENT_SELECTORS:
        spec = play.parse_player_spec(name)
        assert spec.selector == name
        assert not spec.is_human
        assert not spec.is_rl


def test_parse_player_spec_rl_prefix():
    spec = play.parse_player_spec("rl:some/path.zip")
    assert spec.is_rl
    assert spec.rl_path == play.Path("some/path.zip")
    assert not spec.is_human
    assert spec.selector is None


def test_parse_player_spec_bare_zip_path():
    spec = play.parse_player_spec("runs/m1/model.zip")
    assert spec.is_rl
    assert spec.rl_path == play.Path("runs/m1/model.zip")


def test_parse_player_spec_rejects_unknown():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="invalid player"):
        play.parse_player_spec("not-a-real-player")


def test_parse_player_spec_rejects_empty_rl_path():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="needs a checkpoint path"):
        play.parse_player_spec("rl:")


# --- player dispatch (build_players) ------------------------------------------------------


def test_build_players_dispatches_human_and_selector():
    # human -> StateAdapter over a HumanAgent on the shared state; a selector -> StateAdapter over
    # make_agent.
    state = agents.KeyboardState()
    p1, p2 = play.build_players(
        play.parse_player_spec("random"), play.parse_player_spec("human"), state, seed=0
    )
    assert isinstance(p1, play.StateAdapter)
    assert isinstance(p1.agent, agents.RandomAgent)
    assert isinstance(p2, play.StateAdapter)
    assert isinstance(p2.agent, agents.HumanAgent)
    assert p2.agent.keyboard_state is state


def test_build_players_shares_one_state_between_both_humans():
    # With BOTH players human, both HumanAgents read the SAME shared KeyboardState (identity).
    state = agents.KeyboardState()
    p1, p2 = play.build_players(
        play.parse_player_spec("human"), play.parse_player_spec("human"), state
    )
    assert isinstance(p1.agent, agents.HumanAgent)
    assert isinstance(p2.agent, agents.HumanAgent)
    assert p1.agent.keyboard_state is state
    assert p2.agent.keyboard_state is p1.agent.keyboard_state  # the SAME object
    assert p1.agent.mapping.name == "player1"
    assert p2.agent.mapping.name == "player2"


def test_build_players_dispatches_rl(tmp_path):
    # An rl:<path> spec dispatches to make_rl_player. A real *.zip on disk is required for the load
    # path, but loading SB3 here is heavy/unwanted — so assert the spec is classified as RL and the
    # missing-file guard fires (the adapter unit tests cover the predict seam with a fake model).
    spec = play.parse_player_spec("rl:does-not-exist.zip")
    assert spec.is_rl
    state = agents.KeyboardState()
    with pytest.raises(FileNotFoundError, match="RL checkpoint not found"):
        play.build_players(spec, play.parse_player_spec("noop"), state)


def test_make_rl_player_errors_on_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="RL checkpoint not found"):
        play.make_rl_player("nope/missing.zip", slot=S.PLAYER_1)


# --- selector surface (re-exported registry) ----------------------------------------------


def test_default_selectors_pick_coverage_and_shadower():
    assert play.DEFAULT_PLAYER1 == "aggressive-coverage"
    assert play.DEFAULT_PLAYER2 == "opponent-shadower"
    assert isinstance(play.make_agent(play.DEFAULT_PLAYER1), agents.CoverageAgent)
    assert isinstance(play.make_agent(play.DEFAULT_PLAYER2), agents.CoverageAgent)


def test_default_selectors_are_visibly_different_policies():
    # Both are the coverage family, but the default pairing is distinct presets (the player2
    # shadower aims at the opponent while player1 sweeps aim) — a visibly different match.
    assert play.DEFAULT_PLAYER1 != play.DEFAULT_PLAYER2


def test_selector_surface_is_the_new_family_plus_random_and_noop():
    # The registry lives in pop_trainer.agents; play re-exports it. The roster is the coverage
    # family, random, and the noop floor (a valid scripted opponent).
    assert set(play.AGENT_SELECTORS) == {
        "aggressive-coverage",
        "wall-hugger",
        "opponent-shadower",
        "random",
        "noop",
    }
    # The re-exported names are the SAME objects the registry owns (the de-dup is real).
    assert play.AGENT_SELECTORS is agents.AGENT_SELECTORS
    assert play.make_agent is agents.make_agent


def test_no_deleted_selectors_remain():
    for gone in ("aim-at-player2", "explorer", "aim-sweep", "spray", "perimeter", "idle"):
        assert gone not in play.AGENT_SELECTORS


def test_every_selector_builds_a_valid_agent():
    for name in play.AGENT_SELECTORS:
        agent = play.make_agent(name, seed=0)
        assert hasattr(agent, "act")


def test_make_agent_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        play.make_agent("not-a-real-agent")


def test_seed_is_threaded_into_seeded_agents():
    # A seeded selector replays identically; two same-seed builds produce the same first action.
    a = play.make_agent("random", seed=11)
    b = play.make_agent("random", seed=11)
    np.testing.assert_array_equal(a.act(None), b.act(None))


def test_human_is_a_player_form_but_not_a_selector():
    # "human" is a valid --player form but is NOT a seed-factory selector (it is a special
    # shared-listener path in build_players).
    assert play.parse_player_spec("human").is_human
    assert "human" not in play.AGENT_SELECTORS


# --- import-boundary / lazy-rl guard ------------------------------------------------------


def _module_level_imports(module):
    """The module's TOP-LEVEL import statements ONLY (the statements directly in the module body).

    Imports nested inside a function / class body (e.g. the LAZY ``from stable_baselines3 import
    PPO`` inside ``make_rl_player``) are EXCLUDED — that is the whole point: ``rl`` / sb3 / torch
    are permitted lazily but must not appear eagerly at module top level. A docstring mention of a
    name is never an import (an AST statement, not a string).
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    imported: list[str] = []
    for node in tree.body:  # only the module-body statements — not function / class interiors
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return imported


def test_play_top_level_imports_are_lazy_rl_and_forbidden_free():
    # play is a composition-root APP over env / agents / core. The CTO-approved relaxation lets it
    # import rl, but ONLY LAZILY (inside make_rl_player) — so torch/sb3/rl must NOT appear at module
    # top level. models / data / pretraining / tank_twin remain fully forbidden.
    forbidden = (
        "stable_baselines3",
        "torch",
        "pop_trainer.rl",
        "pop_trainer.models",
        "pop_trainer.data",
        "pop_trainer.pretraining",
        "tank_twin",
    )

    def _is_forbidden(module: str) -> bool:
        return any(module == f or module.startswith(f + ".") for f in forbidden)

    offenders = [m for m in _module_level_imports(play) if _is_forbidden(m)]
    assert not offenders, f"play must not import these at module level (rl is lazy): {offenders}"
