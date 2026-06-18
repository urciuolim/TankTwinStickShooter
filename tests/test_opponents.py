"""Regression net for tank_twin.opponents (migrated from PythonScripts/tank_env.py, A6).

PFSP weighting + opponent selection. RNG is seeded via numpy's legacy global
state (the functions use ``numpy.random.choice``), so seeded draws are exact and
deterministic — we assert literal indices and weight ratios, never statistics.
"""

import numpy as np
import pytest

from tank_twin.opponents import choice_with_normalization, elo_based_choice, weight_func

# --- weight_func ----------------------------------------------------------


def test_weight_func_at_zero_is_one():
    assert weight_func(0, 35.0) == 1.0


def test_weight_func_within_distance_caps_at_one():
    # |x| <= D -> 1/(x/D)^2 >= 1, capped to 1.0. Boundary x == D is inclusive.
    assert weight_func(17.5, 35.0) == 1.0
    assert weight_func(-17.5, 35.0) == 1.0
    assert weight_func(35.0, 35.0) == 1.0
    assert weight_func(-35.0, 35.0) == 1.0


def test_weight_func_beyond_distance_decays_as_inverse_square():
    # x == 2D -> 1/(2)^2 == 0.25; x == 4D -> 1/16.
    assert weight_func(70.0, 35.0) == pytest.approx(0.25)
    assert weight_func(-70.0, 35.0) == pytest.approx(0.25)
    assert weight_func(140.0, 35.0) == pytest.approx(1.0 / 16.0)


def test_weight_func_symmetric_in_sign():
    assert weight_func(50.0, 35.0) == weight_func(-50.0, 35.0)


def test_weight_func_strictly_decreasing_beyond_d():
    assert weight_func(50.0, 35.0) > weight_func(80.0, 35.0) > weight_func(200.0, 35.0)


# --- choice_with_normalization -------------------------------------------


def test_choice_with_normalization_all_zero_weights_is_uniform_fallback():
    # All-zero weights must not divide by zero; falls back to a uniform draw and
    # always returns a valid index. Seed pins the exact fallback draw.
    np.random.seed(123)
    assert int(choice_with_normalization([0, 1, 2, 3], [0, 0, 0, 0])) == 2


def test_choice_with_normalization_zero_weight_never_chosen():
    # Only index 2 has weight -> it is always selected regardless of seed.
    for seed in range(20):
        np.random.seed(seed)
        assert int(choice_with_normalization([0, 1, 2], [0, 0, 5])) == 2


def test_choice_with_normalization_seeded_draw_is_exact():
    np.random.seed(0)
    assert int(choice_with_normalization([0, 1, 2, 3], [1, 1, 1, 1])) == 2


def test_choice_with_normalization_is_deterministic_under_same_seed():
    np.random.seed(0)
    first = int(choice_with_normalization([0, 1, 2, 3], [1, 1, 1, 1]))
    np.random.seed(0)
    second = int(choice_with_normalization([0, 1, 2, 3], [1, 1, 1, 1]))
    assert first == second


def test_choice_with_normalization_weight_ratio_honored():
    # Heavy weight on index 0 (10:1:1) -> majority of seeded draws land on 0.
    # Ratio assertion, not a flaky single draw: pin the seeded count.
    np.random.seed(0)
    draws = [int(choice_with_normalization([0, 1, 2], [10, 1, 1])) for _ in range(1000)]
    assert draws.count(0) == 824
    assert draws.count(1) == 80
    assert draws.count(2) == 96


# --- elo_based_choice -----------------------------------------------------


def test_elo_based_choice_seeded_draw_is_exact():
    elos = [800, 1000, 1050, 1300]
    np.random.seed(7)
    assert int(elo_based_choice(elos, 1000, 35.0)) == 1


def test_elo_based_choice_is_deterministic_under_same_seed():
    elos = [800, 1000, 1050, 1300]
    np.random.seed(7)
    first = int(elo_based_choice(elos, 1000, 35.0))
    np.random.seed(7)
    second = int(elo_based_choice(elos, 1000, 35.0))
    assert first == second


def test_elo_based_choice_prefers_opponents_near_center():
    # idx0 sits at center (weight 1), idx1 is 4000 elo away (tiny weight) ->
    # the near opponent is always chosen under every seed.
    elos = [1000, 5000]
    for seed in range(20):
        np.random.seed(seed)
        assert int(elo_based_choice(elos, 1000, 35.0)) == 0


def test_elo_based_choice_all_equal_elos_is_uniform_and_valid():
    # Every opponent at the center -> all weights 1.0 -> uniform over indices.
    elos = [1000, 1000, 1000]
    np.random.seed(0)
    idx = int(elo_based_choice(elos, 1000, 35.0))
    assert idx in (0, 1, 2)
