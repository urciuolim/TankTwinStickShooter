"""Tests for pop_trainer.rl.evaluate — the pure win_rate helper + per-opponent eval loop.

No live Unity: a hand-rolled stub ``gymnasium.Env`` whose ``info["outcome"]`` is scripted per
episode (like ``tests/.../test_selfplay.py``), driven through a REAL ``SelfPlayWrapper`` over an
``OpponentProvider.from_roster([selector])`` so the T3 seam is exercised. The stub model's
``predict`` is greedy and inert (the outcome is env-scripted, not policy-driven).

Coverage:

1. pure ``win_rate``: empty -> 0.0, all-win -> 1.0, mixed, unknown-token-to-denominator.
2. ``evaluate_winrate`` over the stub: returned dict keys == opponents passed (order preserved),
   each value == the expected win fraction for that opponent's scripted outcomes.
3. a truncated-without-outcome episode tallies as a DRAW (never a win).
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


class _StubBox(gymnasium.spaces.Space):
    """A stand-in space object; identity is all the wrapper needs."""


class ScriptedOutcomeEnv(gymnasium.Env):
    """A pure-transport stub whose episodes terminate with a SCRIPTED ``info["outcome"]``.

    ``episode_outcomes`` is a list of (outcome_token_or_None) per episode, consumed in order across
    ``reset`` calls. ``None`` means the episode TRUNCATES with no outcome (a time-limit episode) ->
    the evaluator must count it as a DRAW. Each episode runs a fixed number of steps then terminates
    (or truncates). ``step`` always returns ``info["state"]`` (52 floats) so the wrapper's flip
    works, plus ``info["outcome"]`` ONLY on a terminating step (never on truncation), mirroring
    ``TankEnv``.
    """

    def __init__(self, episode_outcomes: list[str | None], *, steps_per_episode: int = 2) -> None:
        super().__init__()
        self._outcomes = episode_outcomes
        self._steps_per_episode = steps_per_episode
        self._episode = -1
        self._step_in_episode = 0
        self.action_space = _StubBox()
        self.observation_space = _StubBox()

    def reset(self, *, seed=None, options=None):
        self._episode += 1
        self._step_in_episode = 0
        return ("obs", {"state": _state(0.0), "map": None})

    def step(self, action, opponent_action=None):
        self._step_in_episode += 1
        last = self._step_in_episode >= self._steps_per_episode
        info = {"state": _state(float(self._step_in_episode)), "map": None}
        if not last:
            return ("obs", 0.0, False, False, info)
        outcome = self._outcomes[self._episode]
        if outcome is None:
            # Truncation: NO outcome token in info (mirrors TankEnv on a time-limit).
            return ("obs", 0.0, False, True, info)
        info["outcome"] = outcome
        return ("obs", 0.0, True, False, info)


class GreedyStubModel:
    """A stub SB3-style model: ``predict(obs, deterministic=True)`` returns an inert action.

    The outcome is scripted by the env, so the action content is irrelevant — but the call must
    happen (deterministic greedy eval) and return ``(action, state)`` like SB3's ``.predict``.
    """

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, obs, deterministic=False):
        self.predict_calls += 1
        assert deterministic is True  # eval must be greedy
        return np.zeros(5, dtype=np.float32), None


# --- 2. evaluate_winrate per opponent ----------------------------------------------------


def test_evaluate_winrate_returns_per_opponent_dict_in_order():
    # Two opponents, three episodes each. Scripts: opp "noop" wins 2/3, opp "random" wins 1/3.
    # evaluate_winrate plays opponents SEQUENTIALLY (noop's 3 episodes, then random's 3), so a
    # single env scripted with [noop x3, random x3] yields the right per-opponent counts.
    env = ScriptedOutcomeEnv(
        [WIN, WIN, LOSS, LOSS, WIN, DRAW],
        steps_per_episode=2,
    )
    model = GreedyStubModel()

    result = evaluate_winrate(model, env, opponents=("noop", "random"), n_episodes=3, seed=0)

    # keys == the opponents passed, in order
    assert list(result.keys()) == ["noop", "random"]
    # noop: WIN, WIN, LOSS -> 2/3 ; random: LOSS, WIN, DRAW -> 1/3
    assert result["noop"] == 2 / 3
    assert result["random"] == 1 / 3
    # the greedy model was actually driven (2 steps * 6 episodes = 12 predicts)
    assert model.predict_calls == 12


def test_evaluate_winrate_truncation_without_outcome_counts_as_draw():
    # One opponent, two episodes: a clean WIN then a truncation (None -> no outcome token).
    env = ScriptedOutcomeEnv([WIN, None], steps_per_episode=2)
    model = GreedyStubModel()

    result = evaluate_winrate(model, env, opponents=("noop",), n_episodes=2, seed=0)

    # the truncated episode is a DRAW, not a win -> 1 win / 2 episodes.
    assert result["noop"] == 0.5


def test_evaluate_winrate_all_wins_is_one():
    env = ScriptedOutcomeEnv([WIN, WIN], steps_per_episode=1)
    model = GreedyStubModel()
    result = evaluate_winrate(model, env, opponents=("noop",), n_episodes=2, seed=1)
    assert result["noop"] == 1.0
