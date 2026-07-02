"""Tests for pop_trainer.rl.evaluate — the pure win_rate helper + parallel per-opponent eval.

No live Unity: hand-rolled stub ``gymnasium.Env``s whose ``info["outcome"]`` is scripted, driven
through ``evaluate_winrate``'s VEC path. ``evaluate_winrate`` now distributes ``n_episodes`` per
opponent across M eval envs (a VEC env, or a list of raw envs it auto-wraps in a ``DummyVecEnv`` of
``SelfPlayWrapper``s) and re-pins each env's opponent provider per selector via ``set_attr``. The
stub model's ``predict`` is greedy and returns a per-lane batched action (the outcome is
env-scripted, not policy-driven).

Coverage:

1. pure ``win_rate``: empty -> 0.0, all-win -> 1.0, mixed, unknown-token-to-denominator.
2. ``evaluate_winrate`` over the stubs: returned dict keys == opponents passed (order preserved),
   each value == the expected win fraction; a single env (M=1) and a list of M envs give the SAME
   per-opponent rate for deterministic scripted outcomes (the win-rate MATH is M-independent).
3. a done-without-outcome episode tallies as a DRAW (never a win).
4. episodes are distributed across the M envs (each lane is actually driven).
5. pure ``episode_spread``: deterministic floor/ceil split, sums to exactly N, N < n_maps edge.
6. pure ``map_short_name``: arena target -> stem; bare names / no extension.
7. pure pooling: per-opponent, per-map, and overall marginals from per-cell counts reconcile
   (weighted sums agree) even under an uneven spread; zero-episode groups are omitted.
8. rotation eval (``maps=...``): per-cell counts with the deterministic spread; every phase reset
   carries the pinned ``switch_arena``; zero-quota cells are skipped (never pinned).
9. no-rotation fallback (``maps=None``): the wrapper's ``maps`` attr is never pinned and no
   ``switch_arena`` is ever sent — byte-identical to the single-arena eval.
10. formatters: ``format_per_opponent_line`` / ``format_per_map_line`` contents; degenerate
    (empty) inputs never raise.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from pop_trainer.core.state import STATE_LEN
from pop_trainer.rl.evaluate import (
    DRAW,
    LOSS,
    WIN,
    episode_spread,
    evaluate_winrate,
    format_per_map_line,
    format_per_opponent_line,
    map_short_name,
    overall_win_rate,
    pool_by_map,
    pool_by_opponent,
    pooled_win_rate,
    win_rate,
)
from pop_trainer.rl.selfplay import OpponentProvider, SelfPlayWrapper

# --- 1. pure win_rate --------------------------------------------------------------------


def test_win_rate_empty_is_zero():
    assert win_rate([]) == 0.0


def test_win_rate_all_wins_is_one():
    assert win_rate([WIN, WIN, WIN]) == 1.0


def test_win_rate_mixed_counts_only_wins():
    # 2 wins out of 4 (a loss and a draw count to the denominator only).
    assert win_rate([WIN, LOSS, WIN, DRAW]) == 0.5


def test_win_rate_unknown_token_counts_to_denominator_only():
    # A stray token is not a win and never raises; it lowers the rate via the denominator.
    assert win_rate([WIN, "garbage"]) == 0.5


# --- stub doubles ------------------------------------------------------------------------


def _state(fill: float) -> list[float]:
    """A 52-float state (the wrapper's flip needs the full vector in info)."""
    return [fill + i for i in range(STATE_LEN)]


# Real Box spaces so DummyVecEnv can size its obs/action buffers (the vec path needs real shapes).
_OBS_SPACE = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
_ACTION_SPACE = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)


class FixedOutcomeEnv(gymnasium.Env):
    """A raw env stand-in whose EVERY episode terminates with one FIXED outcome.

    The outcome is constant per env instance, so the per-opponent win-rate is deterministic and
    M-independent regardless of how vec auto-reset distributes episodes across lanes.
    ``info["state"]`` (52 floats) is present so the ``SelfPlayWrapper`` flip works;
    ``info["outcome"]`` is set ONLY on the terminating step (never on a truncation), mirroring
    ``TankEnv``. ``None`` outcome -> the episode TRUNCATES with no outcome (a draw).
    """

    def __init__(self, outcome: str | None, *, steps_per_episode: int = 1) -> None:
        super().__init__()
        self._outcome = outcome
        self._steps_per_episode = steps_per_episode
        self._step_in_episode = 0
        self.action_space = _ACTION_SPACE
        self.observation_space = _OBS_SPACE

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        self._step_in_episode = 0
        return (_obs(), {"state": _state(0.0), "map": None})

    def step(self, action, opponent_action=None):
        self._step_in_episode += 1
        last = self._step_in_episode >= self._steps_per_episode
        info = {"state": _state(float(self._step_in_episode)), "map": None}
        if not last:
            return (_obs(), 0.0, False, False, info)
        if self._outcome is None:
            return (_obs(), 0.0, False, True, info)  # truncation, no outcome -> draw
        info["outcome"] = self._outcome
        return (_obs(), 0.0, True, False, info)


def _obs() -> np.ndarray:
    """A tiny obs the batched model.predict can stack (content is irrelevant to the stub)."""
    return np.zeros((1,), dtype=np.float32)


class GreedyStubModel:
    """A stub SB3-style model: ``predict(batch, deterministic=True)`` returns a per-lane action.

    The outcome is scripted by the env, so the action content is irrelevant — but the call must
    happen (greedy eval) and return a BATCHED ``(M, 5)`` action for the M-wide vec obs, like SB3's
    ``.predict`` over a vec env.
    """

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, obs, deterministic=False):
        self.predict_calls += 1
        assert deterministic is True  # eval must be greedy
        batch = int(np.asarray(obs).shape[0])
        return np.zeros((batch, 5), dtype=np.float32), None


# --- 2. evaluate_winrate per opponent (M=1 and M>1) --------------------------------------


def test_evaluate_winrate_returns_per_opponent_dict_in_order():
    # Two opponents; the env always wins, so both rates are 1.0 (order preserved).
    env = FixedOutcomeEnv(WIN)
    model = GreedyStubModel()
    result = evaluate_winrate(model, env, opponents=("noop", "random"), n_episodes=3, seed=0)
    assert list(result.keys()) == ["noop", "random"]
    assert result["noop"] == 1.0
    assert result["random"] == 1.0


def test_evaluate_winrate_m1_equals_mgt1_for_deterministic_outcomes():
    # 4 envs that each ALWAYS win -> rate 1.0 whether distributed across 4 lanes or 1.
    model = GreedyStubModel()
    single = evaluate_winrate(
        model, FixedOutcomeEnv(WIN), opponents=("noop",), n_episodes=8, seed=0
    )
    parallel = evaluate_winrate(
        model,
        [FixedOutcomeEnv(WIN) for _ in range(4)],
        opponents=("noop",),
        n_episodes=8,
        seed=0,
    )
    assert single["noop"] == parallel["noop"] == 1.0


def test_evaluate_winrate_parallel_half_win_half_loss():
    # 2 always-win lanes + 2 always-loss lanes, n_episodes=8 distributed across them. Vec auto-reset
    # streams episodes across lanes; the pooled win-rate is win_rate over the collected outcomes.
    envs = [
        FixedOutcomeEnv(WIN),
        FixedOutcomeEnv(WIN),
        FixedOutcomeEnv(LOSS),
        FixedOutcomeEnv(LOSS),
    ]
    model = GreedyStubModel()
    result = evaluate_winrate(model, envs, opponents=("noop",), n_episodes=8, seed=0)
    # 8 episodes across 4 lanes -> each lane runs 2 episodes; 2 win-lanes * 2 = 4 wins / 8 = 0.5.
    assert result["noop"] == 0.5


def test_evaluate_winrate_truncation_without_outcome_counts_as_draw():
    # An env that always truncates with no outcome -> every episode is a DRAW -> rate 0.0.
    env = FixedOutcomeEnv(None, steps_per_episode=2)
    model = GreedyStubModel()
    result = evaluate_winrate(model, env, opponents=("noop",), n_episodes=4, seed=0)
    assert result["noop"] == 0.0


def test_evaluate_winrate_all_wins_is_one():
    env = FixedOutcomeEnv(WIN, steps_per_episode=1)
    model = GreedyStubModel()
    result = evaluate_winrate(model, env, opponents=("noop",), n_episodes=2, seed=1)
    assert result["noop"] == 1.0


def test_evaluate_winrate_drives_every_lane():
    # Each of M envs records its step calls; with n_episodes >= M every lane is driven at least 1x.
    class _Counting(FixedOutcomeEnv):
        def __init__(self) -> None:
            super().__init__(WIN)
            self.steps = 0

        def step(self, action, opponent_action=None):
            self.steps += 1
            return super().step(action, opponent_action)

    envs = [_Counting() for _ in range(3)]
    model = GreedyStubModel()
    evaluate_winrate(model, envs, opponents=("noop",), n_episodes=6, seed=0)
    assert all(e.steps > 0 for e in envs)  # every lane was stepped


# --- 5. episode_spread -------------------------------------------------------------------


def test_episode_spread_divisible_is_even():
    assert episode_spread(100, 10) == [10] * 10


def test_episode_spread_non_divisible_is_deterministic_floor_ceil():
    # The first N % n_maps maps (rotation order) get the ceiling, the rest the floor.
    assert episode_spread(7, 3) == [3, 2, 2]
    assert episode_spread(11, 4) == [3, 3, 3, 2]
    for n, m in ((7, 3), (11, 4), (1, 1), (0, 3), (99, 10)):
        assert sum(episode_spread(n, m)) == n  # totals sum to exactly N per opponent


def test_episode_spread_fewer_episodes_than_maps():
    # N < n_maps: the first N maps get one episode each, the rest get zero.
    assert episode_spread(3, 5) == [1, 1, 1, 0, 0]


def test_episode_spread_invalid_inputs_raise():
    with pytest.raises(ValueError, match="n_maps"):
        episode_spread(5, 0)
    with pytest.raises(ValueError, match="n_episodes"):
        episode_spread(-1, 3)


# --- 6. map_short_name -------------------------------------------------------------------


def test_map_short_name_from_arena_target():
    assert map_short_name("Arenas/center_block.json") == "center_block"


def test_map_short_name_bare_name_and_no_directory():
    assert map_short_name("center_block") == "center_block"
    assert map_short_name("center_block.json") == "center_block"
    assert map_short_name("a/b/c.json") == "c"


# --- 7. pooled marginals reconcile -------------------------------------------------------


def test_pooled_marginals_reconcile_under_uneven_spread():
    # Uneven per-cell episode counts (a floor/ceil spread): all three aggregates are POOLED from
    # the same counts, so the weighted sums reconstruct the same total wins.
    cells = {
        ("noop", "Arenas/a.json"): (3, 4),
        ("noop", "Arenas/b.json"): (1, 3),
        ("random", "Arenas/a.json"): (0, 4),
        ("random", "Arenas/b.json"): (2, 3),
    }
    per_opp = pool_by_opponent(cells)
    per_map = pool_by_map(cells)
    overall = pooled_win_rate(cells)
    assert per_opp == {"noop": 4 / 7, "random": 2 / 7}
    assert per_map == {"Arenas/a.json": 3 / 8, "Arenas/b.json": 3 / 6}
    assert overall == 6 / 14
    # weighted sums agree: each marginal, weighted by its group's episodes, gives 6 total wins.
    assert sum(rate * 7 for rate in per_opp.values()) == pytest.approx(6)
    map_eps = {"Arenas/a.json": 8, "Arenas/b.json": 6}
    assert sum(rate * map_eps[k] for k, rate in per_map.items()) == pytest.approx(6)
    assert overall * 14 == pytest.approx(6)
    # equal per-opponent budgets -> the unweighted mean of the pooled per-opponent marginals
    # equals the pooled overall (the callback's overall scalar reconciles either way).
    assert overall_win_rate(per_opp) == pytest.approx(overall)


def test_pooling_omits_zero_episode_groups_and_empty_is_zero():
    cells = {
        ("noop", "Arenas/a.json"): (1, 1),
        ("noop", "Arenas/b.json"): (0, 0),  # zero-quota cell (N < n_maps)
    }
    assert pool_by_map(cells) == {"Arenas/a.json": 1.0}  # the unplayed map gets NO rate
    assert pool_by_opponent(cells) == {"noop": 1.0}
    assert pooled_win_rate({}) == 0.0
    assert pooled_win_rate({("noop", "Arenas/a.json"): (0, 0)}) == 0.0


# --- 8. rotation eval: per-cell counts + pinned switch_arena -----------------------------


class ArenaOutcomeEnv(FixedOutcomeEnv):
    """Wins ONLY on ``win_arena``: the outcome is decided by the ``switch_arena`` pinned at reset.

    Records every raw reset ``options`` so the tests can assert exactly which arenas were pinned
    (and that the no-rotation path pins none).
    """

    def __init__(self, win_arena: str) -> None:
        super().__init__(LOSS)
        self._win_arena = win_arena
        self.reset_options: list = []

    def reset(self, *, seed=None, options=None):
        self.reset_options.append(options)
        self._outcome = WIN if (options or {}).get("switch_arena") == self._win_arena else LOSS
        return super().reset(seed=seed, options=options)


_MAPS = ("Arenas/a.json", "Arenas/b.json", "Arenas/c.json")


def test_evaluate_winrate_with_maps_returns_per_cell_counts_with_spread():
    env = ArenaOutcomeEnv("Arenas/b.json")
    model = GreedyStubModel()
    cells = evaluate_winrate(
        model, env, opponents=("noop", "random"), n_episodes=5, seed=0, maps=_MAPS
    )
    # opponent-major key order (opponents outer, maps in rotation order inner).
    assert list(cells) == [(o, m) for o in ("noop", "random") for m in _MAPS]
    # spread of 5 over 3 maps -> [2, 2, 1]; the env wins only on arena b.
    for opp in ("noop", "random"):
        assert cells[(opp, "Arenas/a.json")] == (0, 2)
        assert cells[(opp, "Arenas/b.json")] == (2, 2)
        assert cells[(opp, "Arenas/c.json")] == (0, 1)
        assert sum(e for _w, e in (cells[(opp, m)] for m in _MAPS)) == 5  # exactly N per opponent
    # EVERY reset in a phase carried that phase's pinned arena, and all maps were visited.
    assert env.reset_options and all(
        opt is not None and "switch_arena" in opt for opt in env.reset_options
    )
    assert {opt["switch_arena"] for opt in env.reset_options} == set(_MAPS)
    # the marginals pooled from these counts reconcile.
    assert pool_by_opponent(cells) == {"noop": 2 / 5, "random": 2 / 5}
    assert pool_by_map(cells) == {
        "Arenas/a.json": 0.0,
        "Arenas/b.json": 1.0,
        "Arenas/c.json": 0.0,
    }
    assert pooled_win_rate(cells) == 4 / 10


def test_evaluate_winrate_with_maps_skips_zero_quota_cells():
    env = ArenaOutcomeEnv("Arenas/a.json")
    model = GreedyStubModel()
    cells = evaluate_winrate(model, env, opponents=("noop",), n_episodes=1, seed=0, maps=_MAPS)
    # spread of 1 over 3 maps -> [1, 0, 0]: only arena a is played; b and c record (0, 0).
    assert cells == {
        ("noop", "Arenas/a.json"): (1, 1),
        ("noop", "Arenas/b.json"): (0, 0),
        ("noop", "Arenas/c.json"): (0, 0),
    }
    # the zero-quota phases were NEVER pinned or reset.
    assert {opt["switch_arena"] for opt in env.reset_options} == {"Arenas/a.json"}
    assert list(pool_by_map(cells)) == ["Arenas/a.json"]  # unplayed maps get no marginal


def test_evaluate_winrate_empty_maps_raises():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_winrate(
            GreedyStubModel(), ArenaOutcomeEnv("x"), opponents=("noop",), n_episodes=1, maps=()
        )


# --- 9. no-rotation fallback: maps attr never pinned, no switch_arena --------------------


def test_evaluate_winrate_maps_none_never_pins_maps_attr_or_sends_switch_arena():
    # The byte-identical fallback: with maps=None the per-env `maps` attribute is NEVER touched
    # (stays None) and every raw reset sees options=None (no switch_arena, plain handshake).
    env = ArenaOutcomeEnv("Arenas/unused.json")
    placeholder = OpponentProvider.from_roster(["noop"], strategy="round_robin")
    wrapper = SelfPlayWrapper(env, placeholder)
    vec = DummyVecEnv([lambda: wrapper])
    result = evaluate_winrate(GreedyStubModel(), vec, opponents=("noop",), n_episodes=2, seed=0)
    assert result == {"noop": 0.0}  # boot arena only -> the stub never wins
    assert wrapper.maps is None  # the maps attr was never pinned
    assert env.reset_options and all(opt is None for opt in env.reset_options)


# --- 10. formatters ----------------------------------------------------------------------


def test_format_per_opponent_line_contents_and_accounting():
    line = format_per_opponent_line({"noop": 1.0, "random": 0.5}, episodes=10)
    assert line == "noop: 1.00 | random: 0.50 | OVERALL: 0.75 (2 opponents x 10 episodes)"


def test_format_per_opponent_line_empty_never_raises():
    assert format_per_opponent_line({}) == "OVERALL: 0.00"


def test_format_per_map_line_uses_short_names_in_order():
    line = format_per_map_line({"Arenas/center_block.json": 0.25, "empty": 1.0})
    assert line == "PER-MAP: center_block: 0.25 | empty: 1.00"


def test_format_per_map_line_empty_never_raises():
    assert format_per_map_line({}) == "PER-MAP: (none)"
