"""Unit tests for the PURE win-rate helper (M1, task 3.3).

These import ONLY ``tank_twin.evaluate`` (which keeps ``win_rate`` torch/sb3-free at
runtime) — no torch, no env, no Unity. They pin the counting contract the M1 metric
depends on. The end-to-end pipeline round-trip (env -> PPO -> checkpoint -> reload ->
evaluate) lives in the ``@pytest.mark.e2e`` test ``tests/test_train_e2e.py``.
"""

import pytest

from tank_twin.evaluate import DRAW, LOSS, WIN, win_rate


def test_all_wins_is_one():
    assert win_rate([WIN, WIN, WIN]) == 1.0


def test_no_wins_is_zero():
    assert win_rate([LOSS, LOSS, DRAW]) == 0.0


def test_empty_is_zero_not_division_error():
    # No episodes -> no wins -> 0.0 (must not raise ZeroDivisionError).
    assert win_rate([]) == 0.0


def test_half_wins():
    assert win_rate([WIN, LOSS, WIN, LOSS]) == 0.5


def test_draws_count_in_denominator_only():
    # 1 win out of 4 episodes (2 draws, 1 loss) -> 0.25 (draws are NOT wins).
    assert win_rate([WIN, DRAW, DRAW, LOSS]) == 0.25


def test_result_in_unit_interval_for_mixed_outcomes():
    outcomes = [WIN, LOSS, DRAW, WIN, WIN, LOSS, DRAW, LOSS, WIN, DRAW]
    rate = win_rate(outcomes)
    assert 0.0 <= rate <= 1.0
    assert rate == pytest.approx(4 / 10)


def test_unknown_token_counts_toward_denominator_only():
    # A stray/unknown outcome token must not raise and must not count as a win.
    assert win_rate([WIN, "weird", WIN]) == pytest.approx(2 / 3)


def test_outcome_tokens_match_env_strings():
    # Guard against drift between the evaluator's tokens and TankEnv.step's info values.
    assert (WIN, LOSS, DRAW) == ("win", "loss", "draw")
