"""Tests for pop_trainer.rl.selfplay — the self-play opponent seam.

A pure-Python suite over a hand-rolled stub env + stub opponents — NO live Unity (the live smoke
is a later task). The stub env exposes the slice of the ``TankEnv`` contract the wrapper touches:
``reset`` returning ``(obs, {"state": [...52...], "map": <layout|None>})``, ``step(a1, a2)``
recording BOTH actions and returning a new state in info, and player1's ``action_space`` /
``observation_space``.

Coverage:

1. round-robin ``sample()`` ROTATES in order and wraps.
2. uniform ``sample()`` is reproducible under a fixed seed.
3. ``SelfPlayWrapper.step`` calls ``env.step`` with BOTH actions (a1 == caller's, a2 == opponent's).
4. the opponent RECEIVES player2's FLIPPED state (== ``split_state_for_opponent(state)`` exactly).
5. ``set_map`` IS called for a map-aware opponent at reset; NOT / no-op for a map-agnostic one or
   a ``None`` map.
6. a NEW opponent is sampled at reset, NOT mid-episode (one ``sample()`` across N steps; a second
   reset samples again).
7. ``_p2_obs`` RE-CACHES across steps, tracking each new flipped state.
8. a lost-connection step (``info`` has no ``"state"`` key) does NOT crash, passes the env's
   5-tuple through unchanged, and KEEPS the prior ``_p2_obs``.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from pop_trainer.core.maps import CURATED_ROTATION
from pop_trainer.core.state import STATE_LEN, split_state_for_opponent
from pop_trainer.rl.selfplay import (
    DEFAULT_ROSTER,
    MapProvider,
    Opponent,
    OpponentProvider,
    ScriptedOpponent,
    SelfPlayWrapper,
)

# --- stub doubles ----------------------------------------------------------------------


def _state(fill: float) -> list[float]:
    """A distinct 52-float state whose two halves differ, so the flip is observable."""
    s = [fill + i for i in range(STATE_LEN)]
    return s


class _StubBox(gymnasium.spaces.Space):
    """A stand-in space object; identity is all the wrapper-spaces test needs."""


class StubEnv(gymnasium.Env):
    """A minimal pure-transport env: records step args, hands out a fresh state per step.

    Subclasses :class:`gymnasium.Env` because :class:`SelfPlayWrapper` is a real
    :class:`gymnasium.Wrapper` and asserts its wrapped env is one — exactly as it wraps a real
    ``TankEnv``. It exposes only the slice the wrapper touches: ``reset`` -> ``(obs, info)`` with
    ``info["state"]`` / ``info["map"]``, ``step(a1, a2)`` recording BOTH args and returning a fresh
    state in ``info``, plus player1's spaces.

    ``states`` is the sequence of raw states emitted by reset (index 0) then each step (1, 2, ...).
    ``map`` is the static layout surfaced in every ``info``. ``step_calls`` records every
    ``(a1, a2)`` the wrapper passes through.
    """

    def __init__(self, states: list[list[float]], *, map_layout=None) -> None:
        super().__init__()
        self._states = states
        self._map = map_layout
        self._idx = 0
        self.action_space = _StubBox()
        self.observation_space = _StubBox()
        self.reset_calls: list[dict] = []
        self.step_calls: list[tuple] = []

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append({"seed": seed, "options": options})
        self._idx = 0
        return ("obs0", {"state": list(self._states[0]), "map": self._map})

    def step(self, action, opponent_action=None):
        self.step_calls.append((action, opponent_action))
        self._idx += 1
        state = list(self._states[self._idx])
        info = {"state": state, "map": self._map, "p1_action": action, "p2_action": opponent_action}
        return (f"obs{self._idx}", 0.0, False, False, info)


class RecordingOpponent:
    """An :class:`Opponent` that records every obs it acts on and every set_map / reset call.

    A concrete opponent ALWAYS exposes ``set_map`` / ``reset`` (exactly like ``ScriptedOpponent``);
    the map-agnostic / stateless distinction belongs to the wrapped AGENT one layer down, not the
    opponent. ``map_aware`` controls only whether ``set_map`` RECORDS the layout (a map-aware agent)
    or treats it as a no-op (a map-agnostic agent), simulating ``ScriptedOpponent``'s getattr-probe.
    """

    obs_kind = "state"

    def __init__(self, *, map_aware: bool, action=(0.1, 0.2, 0.3, 0.4, 0.5)) -> None:
        self._action = list(action)
        self._map_aware = map_aware
        self.acted_on: list = []
        self.set_map_calls: list = []
        self.reset_calls: list = []

    def act(self, obs):
        self.acted_on.append(np.asarray(obs).copy())
        return list(self._action)

    def set_map(self, layout) -> None:
        if layout is None or not self._map_aware:
            return
        self.set_map_calls.append(layout)

    def reset(self, seed=None) -> None:
        self.reset_calls.append(seed)


# --- 1. round-robin rotation -----------------------------------------------------------


def test_round_robin_rotates_in_order_and_wraps():
    a, b, c = (RecordingOpponent(map_aware=False) for _ in range(3))
    provider = OpponentProvider([a, b, c], strategy="round_robin")
    picks = [provider.sample() for _ in range(7)]
    assert picks == [a, b, c, a, b, c, a]


# --- 2. uniform reproducibility --------------------------------------------------------


def test_uniform_is_reproducible_under_a_fixed_seed():
    roster = [RecordingOpponent(map_aware=False) for _ in range(5)]
    p1 = OpponentProvider(roster, strategy="uniform", seed=123)
    p2 = OpponentProvider(roster, strategy="uniform", seed=123)
    seq1 = [p1.sample() for _ in range(20)]
    seq2 = [p2.sample() for _ in range(20)]
    assert seq1 == seq2
    # not vacuous: a uniform draw over 5 opponents across 20 picks touches more than one.
    assert len({id(o) for o in seq1}) > 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        OpponentProvider([RecordingOpponent(map_aware=False)], strategy="bogus")


def test_empty_roster_raises():
    with pytest.raises(ValueError, match="at least one opponent"):
        OpponentProvider([], strategy="round_robin")


# --- 3. step passes BOTH actions -------------------------------------------------------


def test_step_passes_both_actions_p1_caller_p2_opponent():
    env = StubEnv([_state(0.0), _state(100.0)])
    opp = RecordingOpponent(map_aware=False, action=(0.1, 0.2, 0.3, 0.4, 0.5))
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))
    wrapper.reset()

    caller_action = [9.0, 8.0, 7.0, 6.0, 5.0]
    wrapper.step(caller_action)

    assert len(env.step_calls) == 1
    a1, a2 = env.step_calls[0]
    assert a1 == caller_action  # player1 == the caller's action, byte-unchanged
    assert a2 == [0.1, 0.2, 0.3, 0.4, 0.5]  # player2 == the opponent's emitted action


# --- 4. opponent receives the FLIPPED player2 state ------------------------------------


def test_opponent_acts_on_flipped_player2_state():
    raw = _state(0.0)
    env = StubEnv([raw, _state(100.0)])
    opp = RecordingOpponent(map_aware=False)
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))
    wrapper.reset()
    wrapper.step([0.0] * 5)

    expected = split_state_for_opponent(np.asarray(raw))
    # the opponent's act() at this step saw the PRE-step flipped view of the reset state
    np.testing.assert_array_equal(opp.acted_on[0], expected)
    # the wrapper RE-cached the next step's view from the step's new state (the flip of states[1])
    np.testing.assert_array_equal(
        wrapper._p2_obs, split_state_for_opponent(np.asarray(env._states[1]))
    )


# --- 5. set_map forwarding -------------------------------------------------------------


def test_set_map_called_for_map_aware_opponent_with_info_map():
    layout = object()  # a stand-in WallLayout; identity is all we assert
    env = StubEnv([_state(0.0), _state(100.0)], map_layout=layout)
    opp = RecordingOpponent(map_aware=True)
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))
    wrapper.reset()

    assert opp.set_map_calls == [layout]


def test_set_map_noop_for_map_agnostic_opponent():
    """A ScriptedOpponent over a map-AGNOSTIC agent (no set_map) no-ops even with a layout present.

    The getattr-probe lives in ScriptedOpponent.set_map: an agent lacking the hook is left
    untouched, so the wrapper's unconditional ``self._opp.set_map(layout)`` raises nothing.
    """

    class _MapAgnosticAgent:
        def act(self, obs):
            return [0.0] * 5

    agent = _MapAgnosticAgent()
    assert not hasattr(agent, "set_map")
    layout = object()
    env = StubEnv([_state(0.0), _state(100.0)], map_layout=layout)
    wrapper = SelfPlayWrapper(
        env, OpponentProvider([ScriptedOpponent(agent)], strategy="round_robin")
    )
    # no AttributeError despite the layout being present
    wrapper.reset()


def test_set_map_noop_for_none_map():
    env = StubEnv([_state(0.0), _state(100.0)], map_layout=None)
    opp = RecordingOpponent(map_aware=True)
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))
    wrapper.reset()
    # a map-aware opponent is NOT called when the layout is None
    assert opp.set_map_calls == []


# --- 6. one opponent per episode, sampled only at reset --------------------------------


def test_one_opponent_sampled_per_episode_not_mid_episode():
    states = [_state(float(i) * 100.0) for i in range(6)]
    env = StubEnv(states)
    a = RecordingOpponent(map_aware=False)
    b = RecordingOpponent(map_aware=False)
    provider = OpponentProvider([a, b], strategy="round_robin")

    sample_count = {"n": 0}
    real_sample = provider.sample

    def counting_sample():
        sample_count["n"] += 1
        return real_sample()

    provider.sample = counting_sample  # type: ignore[method-assign]

    wrapper = SelfPlayWrapper(env, provider)
    wrapper.reset()
    assert sample_count["n"] == 1
    for _ in range(4):
        wrapper.step([0.0] * 5)
    # still one sample across all the steps within the episode
    assert sample_count["n"] == 1
    # the SAME opponent drove every step of the episode
    assert wrapper._opp is a
    assert len(a.acted_on) == 4  # act() runs once per step (4 steps); reset only caches the view

    # a second reset samples again, and round-robin advances to the next opponent
    wrapper.reset()
    assert sample_count["n"] == 2
    assert wrapper._opp is b


# --- 7. p2 view re-caches across steps -------------------------------------------------


def test_p2_obs_recaches_to_each_new_flipped_state():
    states = [_state(0.0), _state(100.0), _state(200.0), _state(300.0)]
    env = StubEnv(states)
    opp = RecordingOpponent(map_aware=False)
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))

    wrapper.reset()
    # cached PRE-step view is the flip of the reset state
    np.testing.assert_array_equal(wrapper._p2_obs, split_state_for_opponent(np.asarray(states[0])))

    for i in range(1, 4):
        wrapper.step([0.0] * 5)
        expected = split_state_for_opponent(np.asarray(states[i]))
        # the cache tracks the NEW state returned by this step
        np.testing.assert_array_equal(wrapper._p2_obs, expected)

    # act() runs once per step on the PRE-step cached view: step i (1..3) sees the flip of the
    # state BEFORE it, states[i-1]. 3 steps -> 3 acts on the flips of states[0..2].
    assert len(opp.acted_on) == 3
    for i in range(3):
        np.testing.assert_array_equal(
            opp.acted_on[i], split_state_for_opponent(np.asarray(states[i]))
        )


# --- 8. lost-connection step (no "state" in info) does not crash -----------------------


class LostConnectionEnv(StubEnv):
    """A :class:`StubEnv` whose ``step`` mimics ``TankEnv``'s ConnectionError/reconnect path.

    ``TankEnv.step`` returns ``info = {"lost_connection": True}`` with NO ``"state"`` key (and a
    truncated, reward-0 step) when the transport drops. ``reset`` still primes ``info["state"]``
    (the env reconnects), so only ``step`` diverges from the base stub.
    """

    def step(self, action, opponent_action=None):
        self.step_calls.append((action, opponent_action))
        return ("obs_drop", 0.0, False, True, {"lost_connection": True})


def test_step_survives_lost_connection_and_keeps_prior_p2_obs():
    env = LostConnectionEnv([_state(0.0), _state(100.0)])
    opp = RecordingOpponent(map_aware=False, action=(0.1, 0.2, 0.3, 0.4, 0.5))
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp], strategy="round_robin"))
    wrapper.reset()

    p2_obs_before = wrapper._p2_obs
    # 1. no KeyError despite info carrying no "state" key
    obs, reward, terminated, truncated, info = wrapper.step([0.0] * 5)

    # 2. the env's 5-tuple passes through unchanged
    assert obs == "obs_drop"
    assert reward == 0.0
    assert terminated is False
    assert truncated is True  # rewards.shaped_step_reward(lost_connection=True) -> truncated
    assert info == {"lost_connection": True}

    # 3. the cached p2 view is UNCHANGED across the lost-connection step (no re-cache)
    assert wrapper._p2_obs is p2_obs_before
    np.testing.assert_array_equal(
        wrapper._p2_obs, split_state_for_opponent(np.asarray(env._states[0]))
    )


# --- wiring: spaces + protocol + factory ----------------------------------------------


def test_wrapper_exposes_player1_spaces():
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    assert wrapper.action_space is env.action_space
    assert wrapper.observation_space is env.observation_space


def test_scripted_opponent_satisfies_protocol_and_delegates_act():
    class _Agent:
        def act(self, obs):
            return [1.0, 0.0, 0.0, 0.0, 0.0]

    opp = ScriptedOpponent(_Agent())
    assert isinstance(opp, Opponent)
    assert opp.obs_kind == "state"
    assert opp.act([0.0] * STATE_LEN) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_from_roster_builds_scripted_opponents_for_every_default_selector():
    provider = OpponentProvider.from_roster(seed=0)
    assert len(provider.opponents) == len(DEFAULT_ROSTER)
    for opp in provider.opponents:
        assert isinstance(opp, ScriptedOpponent)
        assert isinstance(opp, Opponent)
    # the roster's opponents act on a 52-float state and emit a 5-float action
    out = provider.opponents[0].act([0.0] * STATE_LEN)
    assert len(out) == 5


def test_scripted_opponent_forwards_reset_and_set_map_only_when_present():
    calls = {"reset": [], "set_map": []}

    class _Stateful:
        def act(self, obs):
            return [0.0] * 5

        def reset(self, *, seed=None):
            calls["reset"].append(seed)

        def set_map(self, layout):
            calls["set_map"].append(layout)

    opp = ScriptedOpponent(_Stateful())
    opp.reset(7)
    opp.set_map("layoutX")
    opp.set_map(None)  # None layout is a no-op even for a map-aware agent
    assert calls["reset"] == [7]
    assert calls["set_map"] == ["layoutX"]

    class _Pure:
        def act(self, obs):
            return [0.0] * 5

    pure = ScriptedOpponent(_Pure())
    pure.reset(7)  # no-op, no AttributeError
    pure.set_map("layoutY")  # no-op, no AttributeError


# --- map rotation: MapProvider (the per-episode arena sibling of OpponentProvider) ------------

_ARENAS = ("Arenas/a.json", "Arenas/b.json", "Arenas/c.json")


def test_map_provider_round_robin_cycles_in_order_and_wraps():
    provider = MapProvider(_ARENAS, strategy="round_robin")
    picks = [provider.sample() for _ in range(7)]
    assert picks == [
        "Arenas/a.json",
        "Arenas/b.json",
        "Arenas/c.json",
        "Arenas/a.json",
        "Arenas/b.json",
        "Arenas/c.json",
        "Arenas/a.json",
    ]


def test_map_provider_uniform_is_reproducible_under_a_fixed_seed():
    p1 = MapProvider(_ARENAS, strategy="uniform", seed=123)
    p2 = MapProvider(_ARENAS, strategy="uniform", seed=123)
    seq1 = [p1.sample() for _ in range(20)]
    seq2 = [p2.sample() for _ in range(20)]
    assert seq1 == seq2
    # not vacuous: a uniform draw over 3 arenas across 20 picks touches more than one.
    assert len(set(seq1)) > 1


def test_map_provider_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        MapProvider(_ARENAS, strategy="bogus")


def test_map_provider_empty_maps_raises():
    with pytest.raises(ValueError, match="at least one map"):
        MapProvider([], strategy="round_robin")


def test_map_provider_per_subproc_seeding_determinism():
    # Two providers with the SAME seed produce the SAME uniform sequence; with DIFFERENT seeds the
    # sequences differ (the per-subproc cfg.seed + i scheme -> each subproc rotates differently).
    same_a = MapProvider(_ARENAS, strategy="uniform", seed=7)
    same_b = MapProvider(_ARENAS, strategy="uniform", seed=7)
    diff = MapProvider(_ARENAS, strategy="uniform", seed=8)
    seq_a = [same_a.sample() for _ in range(30)]
    seq_b = [same_b.sample() for _ in range(30)]
    seq_diff = [diff.sample() for _ in range(30)]
    assert seq_a == seq_b  # same seed -> identical
    assert seq_a != seq_diff  # different seed -> not identical over a reasonable draw


def test_map_provider_from_curated_no_value_resolves_curated_rotation():
    # The flag passed with no value ([]) resolves to the curated rotation.
    provider = MapProvider.from_curated([])
    assert provider.maps == list(CURATED_ROTATION)


def test_map_provider_from_curated_absent_flag_raises():
    # A None value (flag ABSENT) resolves to None (single-arena) -> no rotation to build.
    with pytest.raises(ValueError, match="at least one map"):
        MapProvider.from_curated(None)


# --- map rotation: SelfPlayWrapper injects switch_arena (training) / never (eval) -------------


def test_selfplay_wrapper_with_map_provider_injects_switch_arena():
    env = StubEnv([_state(0.0), _state(100.0)])
    opp = RecordingOpponent(map_aware=False)
    maps = MapProvider(_ARENAS, strategy="round_robin")
    wrapper = SelfPlayWrapper(env, OpponentProvider([opp]), maps=maps)
    wrapper.reset()

    # the env.reset saw options carrying the first round-robin arena target.
    assert env.reset_calls[-1]["options"] == {"switch_arena": "Arenas/a.json"}


def test_selfplay_wrapper_map_target_advances_per_episode_across_resets():
    env = StubEnv([_state(0.0), _state(100.0)])
    maps = MapProvider(_ARENAS, strategy="round_robin")
    wrapper = SelfPlayWrapper(
        env, OpponentProvider([RecordingOpponent(map_aware=False)]), maps=maps
    )

    targets = []
    for _ in range(4):
        wrapper.reset()
        targets.append(env.reset_calls[-1]["options"]["switch_arena"])
    # round-robin advances per EPISODE (per reset), wrapping after the 3-arena rotation.
    assert targets == ["Arenas/a.json", "Arenas/b.json", "Arenas/c.json", "Arenas/a.json"]


def test_selfplay_wrapper_map_sampled_once_per_episode_not_mid_step():
    # The map is sampled at reset only (one switch_arena per episode); steps within the episode send
    # NO further switch_arena. Mirrors the one-opponent-per-episode contract.
    env = StubEnv([_state(float(i) * 100.0) for i in range(6)])
    maps = MapProvider(_ARENAS, strategy="round_robin")
    sample_count = {"n": 0}
    real_sample = maps.sample

    def counting_sample():
        sample_count["n"] += 1
        return real_sample()

    maps.sample = counting_sample  # type: ignore[method-assign]
    wrapper = SelfPlayWrapper(
        env, OpponentProvider([RecordingOpponent(map_aware=False)]), maps=maps
    )

    wrapper.reset()
    assert sample_count["n"] == 1
    for _ in range(4):
        wrapper.step([0.0] * 5)
    # still ONE map sample across all the steps within the episode (step never resamples).
    assert sample_count["n"] == 1
    # a second reset samples the next arena.
    wrapper.reset()
    assert sample_count["n"] == 2


def test_selfplay_wrapper_map_provider_preserves_caller_switch_arena():
    # A caller-supplied switch_arena WINS over the provider's sampled target (caller-wins).
    env = StubEnv([_state(0.0), _state(100.0)])
    maps = MapProvider(_ARENAS, strategy="round_robin")
    wrapper = SelfPlayWrapper(
        env, OpponentProvider([RecordingOpponent(map_aware=False)]), maps=maps
    )
    wrapper.reset(options={"switch_arena": "Arenas/explicit.json"})
    assert env.reset_calls[-1]["options"]["switch_arena"] == "Arenas/explicit.json"


def test_selfplay_wrapper_without_map_provider_sends_no_switch_arena():
    # BACKWARD-COMPAT / EVAL: maps=None (the default) injects NO switch_arena — reset options are
    # byte-identical to before (None passes through untouched).
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    assert wrapper.maps is None
    wrapper.reset()
    # plain reset() -> options is None (no dict injected, no switch_arena key).
    assert env.reset_calls[-1]["options"] is None


def test_selfplay_wrapper_no_map_provider_preserves_caller_options():
    # maps=None must NOT swallow caller-supplied options (it passes them through verbatim).
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    wrapper.reset(options={"switch_arena": "Arenas/caller.json"})
    assert env.reset_calls[-1]["options"] == {"switch_arena": "Arenas/caller.json"}
