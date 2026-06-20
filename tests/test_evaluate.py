"""Unit tests for the PURE win-rate helper (M1, task 3.3).

These import ONLY ``tank_twin.evaluate`` (which keeps ``win_rate`` torch/sb3-free at
runtime) — no torch, no env, no Unity. They pin the counting contract the M1 metric
depends on. The end-to-end pipeline round-trip (env -> PPO -> checkpoint -> reload ->
evaluate) lives in the ``@pytest.mark.e2e`` test ``tests/test_train_e2e.py``.

Also pins the per-map breakdown plumbing added for the CTO map-eval diagnostic
(``python/eval-maps``): the PURE aggregation (:func:`overall_win_rate`) + table formatter
(:func:`format_per_map_table`), and that the new ``--config``/``--map`` +
``--maps``/``--map-rotation`` flags parse + resolve via the SHARED ``tank_twin.maps``
resolver (no torch / no Unity in these unit tests).
"""

from pathlib import Path

import pytest

from tank_twin.evaluate import (
    DRAW,
    LOSS,
    WIN,
    _build_parser,
    format_per_map_table,
    overall_win_rate,
    win_rate,
)
from tank_twin.maps import DEFAULT_MAPS_DIR, _resolve_map_rotation


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


# --- per-map aggregation (PURE: overall_win_rate) --------------------------------------


def test_overall_win_rate_is_mean_over_maps():
    # Uniform episodes-per-map -> simple mean of rates == pooled win-rate.
    per_map = {"a": 0.0, "b": 0.5, "c": 1.0}
    assert overall_win_rate(per_map) == pytest.approx(0.5)


def test_overall_win_rate_accepts_pairs_sequence():
    pairs = [("a", 0.2), ("b", 0.4)]
    assert overall_win_rate(pairs) == pytest.approx(0.3)


def test_overall_win_rate_empty_is_zero():
    # No maps -> 0.0 (mirrors win_rate on empty input; must not raise).
    assert overall_win_rate({}) == 0.0
    assert overall_win_rate([]) == 0.0


def test_overall_win_rate_single_map_is_that_rate():
    assert overall_win_rate({"custom1_2021": 0.30}) == pytest.approx(0.30)


# --- per-map breakdown formatter (PURE: format_per_map_table) --------------------------


def test_table_contains_each_map_name_and_rate_and_overall():
    per_map = {"custom1_2021": 0.30, "empty": 0.55}
    line = format_per_map_table(per_map, episodes=100)
    assert "custom1_2021: 0.30" in line
    assert "empty: 0.55" in line
    assert "OVERALL:" in line
    # OVERALL is the mean of the two rates (uniform episodes) -> 0.43 to 2dp.
    assert "OVERALL: 0.43" in line
    # Episode accounting suffix present (N maps x E episodes).
    assert "(2 maps x 100 episodes)" in line


def test_table_preserves_input_order():
    per_map = [("b", 0.1), ("a", 0.9)]  # NOT alphabetical -> must keep b before a
    line = format_per_map_table(per_map)
    assert line.index("b: 0.10") < line.index("a: 0.90")


def test_table_without_episodes_omits_suffix():
    line = format_per_map_table({"a": 0.5})
    assert "episodes)" not in line
    assert "a: 0.50" in line
    assert "OVERALL: 0.50" in line


def test_table_single_map_case():
    line = format_per_map_table({"custom1_2021": 0.30}, episodes=100)
    assert "custom1_2021: 0.30" in line
    assert "OVERALL: 0.30" in line
    assert "(1 maps x 100 episodes)" in line


def test_table_empty_yields_only_overall_token():
    # Degenerate breakdown never raises -> just the OVERALL token.
    assert format_per_map_table({}) == "OVERALL: 0.00"


# --- eval CLI flags parse + resolve via the SHARED tank_twin.maps resolver -------------
# Mirrors tests/test_train_features.py style: parse args, then run the shared resolver.
# No Unity / torch launched here.

_EVAL_BASE = ["--model-path", "m.zip", "--game-path", "g.exe"]


def test_eval_config_alias_writes_config_path():
    args = _build_parser().parse_args([*_EVAL_BASE, "--config", "exp-configs/maps/empty.json"])
    assert args.config_path == Path("exp-configs/maps/empty.json")
    # --map is an exact alias for --config (same dest).
    args2 = _build_parser().parse_args([*_EVAL_BASE, "--map", "exp-configs/maps/empty.json"])
    assert args2.config_path == Path("exp-configs/maps/empty.json")


def test_eval_maps_flag_absent_is_none():
    args = _build_parser().parse_args(_EVAL_BASE)
    assert args.maps is None
    assert args.config_path is None
    assert _resolve_map_rotation(args.maps) is None


def test_eval_maps_flag_no_value_resolves_to_all_ten():
    args = _build_parser().parse_args([*_EVAL_BASE, "--maps"])
    resolved = _resolve_map_rotation(args.maps)
    assert resolved is not None
    assert len(resolved) == 10
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))


def test_eval_maps_flag_directory_resolves_sorted():
    args = _build_parser().parse_args([*_EVAL_BASE, "--maps", str(DEFAULT_MAPS_DIR)])
    resolved = _resolve_map_rotation(args.maps)
    assert len(resolved) == 10
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))


def test_eval_map_rotation_alias_explicit_list_keeps_order():
    args = _build_parser().parse_args([*_EVAL_BASE, "--map-rotation", "b.json", "a.json"])
    resolved = _resolve_map_rotation(args.maps)
    assert [p.name for p in resolved] == ["b.json", "a.json"]  # not re-sorted
