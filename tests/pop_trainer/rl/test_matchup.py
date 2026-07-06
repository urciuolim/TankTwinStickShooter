"""Tests for pop_trainer.rl.matchup — the pure win-rate matchup-sampling math.

Stdlib-only functions, so the suite is torch/sb3/env-free:

1. cell enumeration: opponent-major cross product; maps=None -> one boot-arena (None) cell per
   opponent; empty inputs rejected.
2. the outcome convention: win=1.0 / loss=0.0 / draw=0.5; a done WITHOUT an outcome (None) and an
   unknown token both map to 0.5 (never a win).
3. the EMA fold: (1 - alpha) * wr + alpha * outcome; alpha bounds enforced; the unseen prior is
   INITIAL_WIN_RATE = 0.5.
4. the eval-signal feed: eval_cell_rates normalizes BOTH evaluate_winrate return shapes
   (per-cell counts with a rotation; per-opponent rates onto (selector, None) without one),
   omitting zero-episode cells; fold_eval_rates blends measured cells only and leaves the rest
   exactly unchanged.
5. the deficit distribution: sums to 1; the floor bounds every probability from below
   (>= floor / n); equal deficits -> uniform; ordering (higher deficit -> higher probability);
   the all-zero-deficit edge falls back to uniform (no NaN); validation errors.
6. entropy + worst-cells observability helpers.
"""

from __future__ import annotations

import math

import pytest

from pop_trainer.rl.matchup import (
    INITIAL_WIN_RATE,
    deficit_distribution,
    distribution_entropy,
    ema_update,
    eval_cell_rates,
    fold_eval_rates,
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


# --- 4. the eval-signal feed: eval_cell_rates + fold_eval_rates ----------------------------


def test_eval_cell_rates_from_per_cell_counts():
    # Rotation eval shape: {(selector, arena): (wins, episodes)} -> wins / episodes per cell.
    result = {
        ("noop", "Arenas/a.json"): (3, 4),
        ("noop", "Arenas/b.json"): (0, 2),
        ("random", "Arenas/a.json"): (1, 1),
    }
    assert eval_cell_rates(result) == {
        ("noop", "Arenas/a.json"): 0.75,
        ("noop", "Arenas/b.json"): 0.0,
        ("random", "Arenas/a.json"): 1.0,
    }


def test_eval_cell_rates_omits_zero_episode_cells():
    # A zero-quota cell was NOT measured this cycle; it must not appear (and so cannot fold).
    result = {("noop", "Arenas/a.json"): (0, 0), ("noop", "Arenas/b.json"): (1, 2)}
    assert eval_cell_rates(result) == {("noop", "Arenas/b.json"): 0.5}


def test_eval_cell_rates_from_per_opponent_rates_lands_on_boot_arena_cells():
    # Single-arena eval shape (maps=None): {selector: rate} -> the (selector, None) cell.
    result = {"noop": 0.1, "random": 0.9}
    assert eval_cell_rates(result) == {("noop", None): 0.1, ("random", None): 0.9}


def test_fold_eval_rates_alpha_blend_on_measured_cells():
    cells = [("noop", None), ("random", None)]
    folded = fold_eval_rates([0.5, 0.5], cells, {("noop", None): 1.0}, 0.4)
    # measured: 0.6 * 0.5 + 0.4 * 1.0 = 0.7; unmeasured: exactly unchanged.
    assert folded == pytest.approx([0.7, 0.5])
    assert folded[1] == 0.5


def test_fold_eval_rates_returns_new_list_and_leaves_input_untouched():
    cells = [("noop", None)]
    before = [0.5]
    folded = fold_eval_rates(before, cells, {("noop", None): 0.0}, 1.0)
    assert folded == [0.0]
    assert before == [0.5]


def test_fold_eval_rates_ignores_unknown_cells():
    # A rate for a cell outside the enumeration (roster / rotation mismatch) is ignored.
    cells = [("noop", None)]
    folded = fold_eval_rates([0.5], cells, {("not-a-cell", None): 1.0}, 0.4)
    assert folded == [0.5]


def test_fold_eval_rates_empty_result_is_identity():
    cells = [("noop", None), ("random", None)]
    assert fold_eval_rates([0.3, 0.8], cells, {}, 0.4) == [0.3, 0.8]


def test_fold_eval_rates_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        fold_eval_rates([0.5], [("noop", None), ("random", None)], {}, 0.4)


def test_eval_cell_rates_then_fold_composes_across_both_shapes():
    # The two helpers compose: a maps=None result folds onto the degenerate boot-arena grid.
    cells = [("noop", None), ("random", None)]
    rates = eval_cell_rates({"noop": 0.0, "random": 1.0})
    folded = fold_eval_rates([0.5, 0.5], cells, rates, 0.5)
    assert folded == pytest.approx([0.25, 0.75])


# --- 5. deficit distribution --------------------------------------------------------------


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


# --- 6. observability helpers -------------------------------------------------------------


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
