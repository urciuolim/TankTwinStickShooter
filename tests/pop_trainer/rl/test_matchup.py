"""Tests for pop_trainer.rl.matchup — the pure win-rate matchup-sampling math.

Stdlib-only functions, so the suite is torch/sb3/env-free:

1. cell enumeration: opponent-major cross product; maps=None -> one boot-arena (None) cell per
   opponent; empty inputs rejected.
2. the outcome convention: win=1.0 / loss=0.0 / draw=0.5; a done WITHOUT an outcome (None) and an
   unknown token both map to 0.5 (never a win).
3. the EMA fold: (1 - alpha) * wr + alpha * outcome; alpha bounds enforced; the unseen prior is
   INITIAL_WIN_RATE = 0.5.
4. the deficit distribution: sums to 1; the floor bounds every probability from below
   (>= floor / n); equal deficits -> uniform; ordering (higher deficit -> higher probability);
   the all-zero-deficit edge falls back to uniform (no NaN); validation errors.
5. entropy + worst-cells observability helpers.
"""

from __future__ import annotations

import math

import pytest

from pop_trainer.rl.matchup import (
    INITIAL_WIN_RATE,
    deficit_distribution,
    distribution_entropy,
    ema_update,
    matchup_cells,
    outcome_to_float,
    worst_cells,
)

# --- 1. cell enumeration ----------------------------------------------------------------


def test_matchup_cells_cross_product_opponent_major():
    cells = matchup_cells(["noop", "random"], ["Arenas/a.json", "Arenas/b.json"])
    assert cells == [
        ("noop", "Arenas/a.json"),
        ("noop", "Arenas/b.json"),
        ("random", "Arenas/a.json"),
        ("random", "Arenas/b.json"),
    ]


def test_matchup_cells_maps_none_yields_boot_arena_cells():
    # No rotation configured -> ONE cell per opponent, tagged with the None boot arena.
    cells = matchup_cells(["noop", "random"], None)
    assert cells == [("noop", None), ("random", None)]


def test_matchup_cells_empty_opponents_raises():
    with pytest.raises(ValueError, match="at least one opponent"):
        matchup_cells([], ["Arenas/a.json"])


def test_matchup_cells_empty_maps_raises():
    # An EMPTY rotation is a caller bug (None is the explicit no-rotation value).
    with pytest.raises(ValueError, match="at least one map"):
        matchup_cells(["noop"], [])


# --- 2. outcome convention ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("win", 1.0),
        ("loss", 0.0),
        ("draw", 0.5),
        (None, 0.5),  # done WITHOUT an outcome (truncation / time-limit / lost connection)
        ("bogus", 0.5),  # unknown token tolerated, never a win
    ],
)
def test_outcome_to_float_convention(token, expected):
    assert outcome_to_float(token) == expected


# --- 3. EMA fold + prior -----------------------------------------------------------------


def test_ema_update_math():
    # (1 - alpha) * wr + alpha * outcome, exactly.
    assert ema_update(0.5, 1.0, 0.1) == pytest.approx(0.55)
    assert ema_update(0.5, 0.0, 0.1) == pytest.approx(0.45)
    assert ema_update(0.8, 0.5, 0.5) == pytest.approx(0.65)


def test_ema_update_alpha_one_replaces_outright():
    assert ema_update(0.9, 0.0, 1.0) == 0.0
    assert ema_update(0.1, 1.0, 1.0) == 1.0


def test_ema_update_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha"):
        ema_update(0.5, 1.0, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        ema_update(0.5, 1.0, 1.5)
    with pytest.raises(ValueError, match="alpha"):
        ema_update(0.5, 1.0, -0.1)


def test_unseen_cell_prior_is_half():
    # All cells start at the 0.5 prior -> equal deficits -> the initial distribution is uniform.
    assert INITIAL_WIN_RATE == 0.5
    dist = deficit_distribution([INITIAL_WIN_RATE] * 4, 0.25)
    assert dist == pytest.approx([0.25] * 4)


# --- 4. deficit distribution --------------------------------------------------------------


def test_deficit_distribution_sums_to_one():
    dist = deficit_distribution([0.9, 0.5, 0.1, 0.7], 0.25)
    assert sum(dist) == pytest.approx(1.0)


def test_deficit_distribution_floor_bounds_every_cell():
    # Even a fully-won cell keeps at least floor / n probability.
    win_rates = [1.0, 0.0, 0.0, 0.0]
    floor = 0.25
    dist = deficit_distribution(win_rates, floor)
    assert min(dist) >= floor / len(win_rates) - 1e-12
    assert sum(dist) == pytest.approx(1.0)


def test_deficit_distribution_equal_deficits_uniform():
    dist = deficit_distribution([0.3, 0.3, 0.3], 0.25)
    assert dist == pytest.approx([1.0 / 3.0] * 3)


def test_deficit_distribution_ordering_respected():
    # A higher deficit (lower win-rate) always gets a strictly higher probability here.
    win_rates = [0.9, 0.5, 0.1]
    dist = deficit_distribution(win_rates, 0.25)
    assert dist[2] > dist[1] > dist[0]


def test_deficit_distribution_all_won_falls_back_to_uniform_no_nan():
    # All wr == 1.0 -> total deficit 0 -> uniform fallback, not NaN / division by zero.
    dist = deficit_distribution([1.0, 1.0, 1.0, 1.0], 0.25)
    assert dist == pytest.approx([0.25] * 4)
    assert all(not math.isnan(p) for p in dist)


def test_deficit_distribution_floor_one_is_uniform():
    dist = deficit_distribution([0.9, 0.1], 1.0)
    assert dist == pytest.approx([0.5, 0.5])


def test_deficit_distribution_floor_zero_is_pure_deficit():
    dist = deficit_distribution([0.75, 0.25], 0.0)
    # deficits 0.25 / 0.75 -> normalized 0.25 and 0.75.
    assert dist == pytest.approx([0.25, 0.75])


def test_deficit_distribution_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one cell"):
        deficit_distribution([], 0.25)
    with pytest.raises(ValueError, match="floor"):
        deficit_distribution([0.5], -0.1)
    with pytest.raises(ValueError, match="floor"):
        deficit_distribution([0.5], 1.1)


# --- 5. observability helpers -------------------------------------------------------------


def test_distribution_entropy_uniform_is_log_n():
    assert distribution_entropy([0.25] * 4) == pytest.approx(math.log(4))


def test_distribution_entropy_one_hot_is_zero():
    assert distribution_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_worst_cells_lowest_win_rates_first():
    cells = [("noop", None), ("random", None), ("wall-hugger", None)]
    win_rates = [0.9, 0.1, 0.5]
    worst = worst_cells(cells, win_rates, 2)
    assert worst == [(("random", None), 0.1), (("wall-hugger", None), 0.5)]


def test_worst_cells_k_larger_than_cells_returns_all():
    cells = [("noop", None)]
    assert worst_cells(cells, [0.5], 10) == [(("noop", None), 0.5)]


def test_worst_cells_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        worst_cells([("noop", None)], [0.5, 0.6], 1)
