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
"""

from __future__ import annotations

import gymnasium
import numpy as np

from pop_trainer.core.state import STATE_LEN
from pop_trainer.rl.evaluate import DRAW, LOSS, WIN, evaluate_winrate, win_rate

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
