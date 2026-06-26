"""Tests for pop_trainer.rl.elo — the pure ELO rating math.

Pure stdlib-math suite (no torch / sb3 / numpy): symmetry + known values of ``elo_prob``, and
deterministic ``elo_change`` updates including the per-side rounding behavior.
"""

from __future__ import annotations

import math

from pop_trainer.rl.elo import elo_change, elo_prob

# --- elo_prob ----------------------------------------------------------------------------


def test_elo_prob_equal_ratings_is_one_half():
    assert elo_prob(1500, 1500) == 0.5


def test_elo_prob_symmetry_sums_to_one():
    # P(A beats B) + P(B beats A) == 1 for any rating pair.
    assert math.isclose(elo_prob(1600, 1400) + elo_prob(1400, 1600), 1.0)


def test_elo_prob_known_400_point_gap():
    # A 400-point lead -> a 10:1 expected score -> 10/11 win probability (logistic base 10, /400).
    assert math.isclose(elo_prob(1800, 1400), 10.0 / 11.0)
    assert math.isclose(elo_prob(1400, 1800), 1.0 / 11.0)


def test_elo_prob_higher_rating_is_favored():
    assert elo_prob(1700, 1500) > 0.5
    assert elo_prob(1500, 1700) < 0.5


# --- elo_change --------------------------------------------------------------------------


def test_elo_change_equal_ratings_full_win_splits_K():
    # Equal ratings -> each expects 0.5. A wins (a_win_rate=1.0): A gains K*(1-0.5)=K/2,
    # B loses K*(0-0.5)=-K/2. With K=32 -> (+16, -16).
    assert elo_change(1500, 1500, 32, 1.0) == (16, -16)


def test_elo_change_equal_ratings_draw_is_zero():
    # Equal ratings, a_win_rate=0.5 -> both expect exactly their realized score -> no change.
    assert elo_change(1500, 1500, 32, 0.5) == (0, 0)


def test_elo_change_upset_underdog_gains_more():
    # The underdog (A at 1400 vs B at 1600) winning outright gains more than the favorite would.
    a_delta, b_delta = elo_change(1400, 1600, 32, 1.0)
    # A expected ~0.24, realized 1.0 -> gains round(32 * (1 - 0.2424...)) = round(24.24) = 24.
    assert a_delta == 24
    # B expected ~0.76, realized 0.0 -> loses round(32 * (0 - 0.7575...)) = round(-24.24) = -24.
    assert b_delta == -24


def test_elo_change_per_side_rounding_need_not_sum_to_zero():
    # Rounding is applied PER SIDE (not to a shared delta), so the pair can fail to sum to zero.
    # Pick a fractional case and assert the two rounds are computed independently.
    a_delta, b_delta = elo_change(1500, 1530, 32, 0.5)
    a_prob = elo_prob(1500, 1530)
    b_prob = elo_prob(1530, 1500)
    assert a_delta == round(32 * (0.5 - a_prob))
    assert b_delta == round(32 * (0.5 - b_prob))


def test_elo_change_is_deterministic():
    assert elo_change(1480, 1520, 24, 0.75) == elo_change(1480, 1520, 24, 0.75)
