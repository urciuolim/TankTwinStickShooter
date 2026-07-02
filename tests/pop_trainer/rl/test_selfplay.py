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
    MatchupProvider,
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


# --- matchup sampling: MatchupProvider (the joint opponent x map sampler) ----------------------


def _matchup_pairs(n=2):
    """n (selector, RecordingOpponent) pairs with distinct selector tags."""
    return [(f"opp{i}", RecordingOpponent(map_aware=False)) for i in range(n)]


def _one_hot(n, hot):
    return [1.0 if i == hot else 0.0 for i in range(n)]


def test_matchup_provider_cells_cross_product_and_initial_uniform():
    provider = MatchupProvider(_matchup_pairs(2), _ARENAS, seed=0)
    assert provider.cells == [
        ("opp0", "Arenas/a.json"),
        ("opp0", "Arenas/b.json"),
        ("opp0", "Arenas/c.json"),
        ("opp1", "Arenas/a.json"),
        ("opp1", "Arenas/b.json"),
        ("opp1", "Arenas/c.json"),
    ]
    # the well-defined initial distribution: all win-rates at the 0.5 prior -> uniform.
    assert provider.distribution == pytest.approx([1.0 / 6.0] * 6)


def test_matchup_provider_no_maps_cells_carry_none_boot_arena():
    provider = MatchupProvider(_matchup_pairs(2), None, seed=0)
    assert provider.cells == [("opp0", None), ("opp1", None)]
    _selector, _opp, arena = provider.sample()
    assert arena is None


def test_matchup_provider_seeded_determinism_same_seed_same_sequence():
    a = MatchupProvider(_matchup_pairs(3), _ARENAS, seed=7)
    b = MatchupProvider(_matchup_pairs(3), _ARENAS, seed=7)
    seq_a = [(sel, arena) for sel, _opp, arena in (a.sample() for _ in range(30))]
    seq_b = [(sel, arena) for sel, _opp, arena in (b.sample() for _ in range(30))]
    assert seq_a == seq_b
    # not vacuous: a uniform draw over 9 cells across 30 picks touches more than one.
    assert len(set(seq_a)) > 1


def test_matchup_provider_per_env_seed_offsets_differ():
    # The per-subproc cfg.seed + i scheme: different seeds -> different cell sequences.
    a = MatchupProvider(_matchup_pairs(3), _ARENAS, seed=7)
    b = MatchupProvider(_matchup_pairs(3), _ARENAS, seed=8)
    seq_a = [(sel, arena) for sel, _opp, arena in (a.sample() for _ in range(30))]
    seq_b = [(sel, arena) for sel, _opp, arena in (b.sample() for _ in range(30))]
    assert seq_a != seq_b


def test_matchup_provider_sample_respects_distribution():
    provider = MatchupProvider(_matchup_pairs(2), _ARENAS, seed=0)
    hot = 4  # ("opp1", "Arenas/b.json")
    provider.distribution = _one_hot(len(provider.cells), hot)
    for _ in range(10):
        selector, opp, arena = provider.sample()
        assert (selector, arena) == provider.cells[hot]
        # the sampled opponent is the one PAIRED with the cell's selector.
        assert opp is provider._entries[hot][1]


def test_matchup_provider_sample_keeps_selector_opponent_association():
    pairs = _matchup_pairs(3)
    provider = MatchupProvider(pairs, None, seed=3)
    by_selector = dict(pairs)
    for _ in range(20):
        selector, opp, _arena = provider.sample()
        assert opp is by_selector[selector]


def test_matchup_provider_validation():
    with pytest.raises(ValueError, match="at least one opponent"):
        MatchupProvider([], _ARENAS)
    with pytest.raises(ValueError, match="at least one map"):
        MatchupProvider(_matchup_pairs(1), [])


def test_matchup_provider_rejects_wrong_length_distribution():
    provider = MatchupProvider(_matchup_pairs(2), _ARENAS, seed=0)
    provider.distribution = [1.0]  # broadcast of the wrong shape
    with pytest.raises(ValueError, match="distribution length"):
        provider.sample()


def test_matchup_provider_from_roster_resolves_selectors():
    provider = MatchupProvider.from_roster(("noop", "random"), _ARENAS, seed=0)
    assert provider.cells[0] == ("noop", "Arenas/a.json")
    selector, opp, _arena = provider.sample()
    assert isinstance(opp, ScriptedOpponent)
    assert selector in ("noop", "random")
    # the resolved opponents speak the 52-float state -> 5-float action contract.
    assert len(opp.act([0.0] * STATE_LEN)) == 5


def test_matchup_provider_pickles_spawn_safe():
    # Plain-data state + a picklable RNG: the provider must survive the spawn boundary. The
    # pickled copy continues the SAME seeded cell sequence as the original.
    import pickle

    original = MatchupProvider.from_roster(("noop", "random"), _ARENAS, seed=11)
    clone = pickle.loads(pickle.dumps(original))
    seq_orig = [(sel, arena) for sel, _o, arena in (original.sample() for _ in range(15))]
    seq_clone = [(sel, arena) for sel, _o, arena in (clone.sample() for _ in range(15))]
    assert seq_orig == seq_clone
    assert clone.cells == original.cells
    assert clone.distribution == original.distribution


# --- matchup sampling: SelfPlayWrapper joint reset + terminal tagging --------------------------


class TerminalStubEnv(StubEnv):
    """A :class:`StubEnv` whose LAST scripted state ends the episode (terminated=True).

    ``outcome`` (when not ``None``) rides the terminal ``info`` as ``info["outcome"]`` — the
    token ``TankEnv.step`` surfaces on terminal steps. ``outcome=None`` models a done WITHOUT an
    outcome (truncation / time-limit), which the matchup tag must record as a 0.5 draw.
    """

    def __init__(self, states, *, outcome=None, map_layout=None) -> None:
        super().__init__(states, map_layout=map_layout)
        self._outcome = outcome

    def step(self, action, opponent_action=None):
        obs, reward, terminated, truncated, info = super().step(action, opponent_action)
        if self._idx == len(self._states) - 1:
            terminated = True
            if self._outcome is not None:
                info["outcome"] = self._outcome
        return obs, reward, terminated, truncated, info


def _wired_matchup_wrapper(env, *, maps=_ARENAS, hot=None, seed=0):
    """A SelfPlayWrapper whose MatchupProvider is optionally pinned one-hot to cell ``hot``."""
    pairs = _matchup_pairs(2)
    matchups = MatchupProvider(pairs, maps, seed=seed)
    if hot is not None:
        matchups.distribution = _one_hot(len(matchups.cells), hot)
    fallback = OpponentProvider([RecordingOpponent(map_aware=False)])
    return SelfPlayWrapper(env, fallback, matchups=matchups), pairs, matchups


def test_matchup_wrapper_injects_cell_arena_as_switch_arena():
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper, pairs, _m = _wired_matchup_wrapper(env, hot=4)  # ("opp1", "Arenas/b.json")
    wrapper.reset()
    assert env.reset_calls[-1]["options"] == {"switch_arena": "Arenas/b.json"}
    # the episode's opponent is the cell's paired opponent, not the fallback provider's.
    assert wrapper._opp is pairs[1][1]


def test_matchup_wrapper_no_maps_never_injects_switch_arena():
    # Boot-arena mode: cells carry None -> reset options pass through UNTOUCHED (byte-identical
    # handshake; opponent-only prioritization).
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper, _pairs, _m = _wired_matchup_wrapper(env, maps=None)
    wrapper.reset()
    assert env.reset_calls[-1]["options"] is None
    wrapper.reset(options={"other": 1})
    assert env.reset_calls[-1]["options"] == {"other": 1}
    assert "switch_arena" not in env.reset_calls[-1]["options"]


def test_matchup_wrapper_caller_switch_arena_wins():
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper, _pairs, _m = _wired_matchup_wrapper(env, hot=0)
    wrapper.reset(options={"switch_arena": "Arenas/explicit.json"})
    assert env.reset_calls[-1]["options"]["switch_arena"] == "Arenas/explicit.json"


def test_matchup_wrapper_overrides_sibling_samplers():
    # The joint provider is the SINGLE source of the episode's (opponent, arena): neither the
    # OpponentProvider nor the MapProvider sample() is consulted while it is active.
    env = StubEnv([_state(0.0), _state(100.0)])
    pairs = _matchup_pairs(2)
    matchups = MatchupProvider(pairs, _ARENAS, seed=0)
    opponents = OpponentProvider([RecordingOpponent(map_aware=False)])
    maps = MapProvider(_ARENAS, strategy="round_robin")
    counts = {"opp": 0, "map": 0}
    real_opp_sample, real_map_sample = opponents.sample, maps.sample

    def counting_opp():
        counts["opp"] += 1
        return real_opp_sample()

    def counting_map():
        counts["map"] += 1
        return real_map_sample()

    opponents.sample = counting_opp  # type: ignore[method-assign]
    maps.sample = counting_map  # type: ignore[method-assign]
    wrapper = SelfPlayWrapper(env, opponents, maps=maps, matchups=matchups)
    wrapper.reset()
    assert counts == {"opp": 0, "map": 0}


def test_matchup_wrapper_tags_terminal_info_with_cell_and_outcome():
    env = TerminalStubEnv([_state(0.0), _state(100.0)], outcome="win")
    wrapper, _pairs, _m = _wired_matchup_wrapper(env, hot=4)  # ("opp1", "Arenas/b.json")
    wrapper.reset()
    _obs, _r, terminated, _tr, info = wrapper.step([0.0] * 5)
    assert terminated is True
    assert info["matchup"] == {"opponent": "opp1", "map": "Arenas/b.json", "outcome": 1.0}


def test_matchup_wrapper_terminal_tag_outcome_convention():
    # loss -> 0.0; draw -> 0.5; done WITHOUT an outcome token -> 0.5 (never a win).
    for outcome, expected in (("loss", 0.0), ("draw", 0.5), (None, 0.5)):
        env = TerminalStubEnv([_state(0.0), _state(100.0)], outcome=outcome)
        wrapper, _pairs, _m = _wired_matchup_wrapper(env, maps=None, hot=0)
        wrapper.reset()
        _obs, _r, _term, _tr, info = wrapper.step([0.0] * 5)
        assert info["matchup"]["outcome"] == expected
        assert info["matchup"]["map"] is None


def test_matchup_wrapper_does_not_tag_non_terminal_steps():
    env = StubEnv([_state(0.0), _state(100.0), _state(200.0)])
    wrapper, _pairs, _m = _wired_matchup_wrapper(env)
    wrapper.reset()
    _obs, _r, _term, _tr, info = wrapper.step([0.0] * 5)
    assert "matchup" not in info


def test_wrapper_without_matchups_never_tags_terminal_info():
    # FLAG OFF: a terminal step's info carries NO "matchup" key — byte-identical to today.
    env = TerminalStubEnv([_state(0.0), _state(100.0)], outcome="win")
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    assert wrapper.matchups is None
    wrapper.reset()
    _obs, _r, terminated, _tr, info = wrapper.step([0.0] * 5)
    assert terminated is True
    assert "matchup" not in info
    # and the reset handshake stayed untouched (no options injected).
    assert env.reset_calls[-1]["options"] is None


def test_wrapper_with_pinned_single_map_provider_emits_no_matchup_tag():
    # The rotation-EVAL pin: ``rl.evaluate`` sets a single-map MapProvider on an eval wrapper per
    # phase (``matchups`` stays None). The pin injects switch_arena at every reset but must NOT
    # produce an ``info["matchup"]`` tag — eval episodes can never feed the matchup-sampling EMA
    # (the tag is emitted only from a MatchupProvider's joint draw).
    env = TerminalStubEnv([_state(0.0), _state(100.0)], outcome="win")
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    wrapper.maps = MapProvider(["Arenas/b.json"], strategy="round_robin")  # the eval-phase pin
    wrapper.reset()
    assert env.reset_calls[-1]["options"] == {"switch_arena": "Arenas/b.json"}
    _obs, _r, terminated, _tr, info = wrapper.step([0.0] * 5)
    assert terminated is True
    assert "matchup" not in info


def test_matchup_distribution_property_reads_and_broadcast_assigns_in_place():
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper, _pairs, matchups = _wired_matchup_wrapper(env)
    assert wrapper.matchup_distribution == matchups.distribution
    new_dist = _one_hot(len(matchups.cells), 2)
    rng_before = matchups._rng
    wrapper.matchup_distribution = new_dist  # the set_attr broadcast path
    assert matchups.distribution == new_dist
    # the provider object and its seeded RNG stream were NOT replaced.
    assert wrapper.matchups is matchups
    assert matchups._rng is rng_before


def test_matchup_distribution_setter_rejected_without_provider():
    env = StubEnv([_state(0.0), _state(100.0)])
    wrapper = SelfPlayWrapper(env, OpponentProvider([RecordingOpponent(map_aware=False)]))
    assert wrapper.matchup_distribution is None
    with pytest.raises(ValueError, match="without a MatchupProvider"):
        wrapper.matchup_distribution = [1.0]
