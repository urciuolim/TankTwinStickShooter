"""Regression net for tank_twin.elo (migrated from PythonScripts/elo.py, A6).

These tests are the proof the move is byte-identical: they pin the exact 2021
math (logistic base-10/400 expectation; per-side rounded ELO deltas).
"""

import math

import pytest

from tank_twin.elo import elo_change, elo_prob

# --- elo_prob -------------------------------------------------------------


def test_elo_prob_equal_elos_is_half():
    assert elo_prob(1000, 1000) == 0.5
    assert elo_prob(0, 0) == 0.5
    assert elo_prob(2400, 2400) == 0.5


@pytest.mark.parametrize(
    "a,b",
    [(1000, 1200), (1500, 1000), (0, 400), (2400, 1800), (1234, 1233)],
)
def test_elo_prob_symmetry_sums_to_one(a, b):
    # p(a beats b) + p(b beats a) == 1 exactly.
    assert elo_prob(a, b) + elo_prob(b, a) == pytest.approx(1.0)


def test_elo_prob_monotonic_in_rating_gap():
    # Higher rating -> higher win probability; strictly increasing in own elo.
    base = elo_prob(1000, 1000)
    higher = elo_prob(1100, 1000)
    lower = elo_prob(900, 1000)
    assert lower < base < higher


def test_elo_prob_known_fixture_400_gap():
    # A 400-point edge is the canonical 10:1 odds -> p = 10/11.
    assert elo_prob(1400, 1000) == pytest.approx(10.0 / 11.0)
    assert elo_prob(1000, 1400) == pytest.approx(1.0 / 11.0)


def test_elo_prob_exact_formula_fixture():
    # Pin the literal 2021 expression so a refactor that changes the constant breaks.
    expected = 1.0 / (1.0 + math.pow(10.0, (1200 - 1000) / 400.0))
    assert elo_prob(1000, 1200) == expected


# --- elo_change -----------------------------------------------------------


def test_elo_change_equal_elos_a_wins_zero_sum():
    # Equal elo, A wins outright (win_rate=1): each side moves K/2, rounded.
    a, b = elo_change(1000, 1000, 32, 1.0)
    assert (a, b) == (16, -16)
    assert a + b == 0


def test_elo_change_equal_elos_draw_is_zero():
    # Equal elo, exactly expected result (win_rate=0.5) -> no change.
    assert elo_change(1000, 1000, 32, 0.5) == (0, 0)


def test_elo_change_equal_elos_a_loses():
    a, b = elo_change(1000, 1000, 32, 0.0)
    assert (a, b) == (-16, 16)


def test_elo_change_returns_rounded_ints():
    a, b = elo_change(1000, 1050, 32, 0.75)
    assert isinstance(a, int)
    assert isinstance(b, int)


def test_elo_change_per_side_rounding_fixture():
    # Underdog (A) wins: pin exact rounded deltas. Mismatched elos mean the two
    # sides round independently (the 2021 behavior), so the pair need NOT sum to 0.
    a_prob = elo_prob(1000, 1200)
    b_prob = elo_prob(1200, 1000)
    expected_a = round(32 * (1.0 - a_prob))
    expected_b = round(32 * (0.0 - b_prob))
    assert elo_change(1000, 1200, 32, 1.0) == (expected_a, expected_b)


def test_elo_change_k_scales_magnitude():
    a16, _ = elo_change(1000, 1000, 16, 1.0)
    a32, _ = elo_change(1000, 1000, 32, 1.0)
    assert a16 == 8
    assert a32 == 16
