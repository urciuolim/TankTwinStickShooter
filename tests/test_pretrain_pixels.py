"""Pure-logic tests for the pixel inverse-renderer pretraining helpers.

Covers ONLY the numpy-only, torch-free pieces: target encoding / sentinel masking,
the 12/10/40 split correctness, wall-grid lookup-by-map_id, normalization round-trip
(with bullet stats excluding sentinels), the WxH parser, and the stratified group
split. The trainer (torch) is NOT unit-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tank_twin.pretrain_pixels import (
    BULLET_BASE_INDICES,
    N_BULLET_POS,
    N_PLAYER,
    PLAYER_INDICES,
    SENTINEL,
    NormStats,
    bullet_position_indices,
    decode_targets,
    denormalize,
    encode_bullet_targets,
    encode_player_targets,
    fit_norm_stats,
    normalize,
    parse_input_res,
    stratified_group_split,
    wall_target_for,
    worker_id_from_shard,
)


def _make_state(p1_player, p2_player, p1_bullets, p2_bullets):
    """Assemble a 52-float state from blocks. Bullets are lists of 4-float slots
    (or None for an absent slot -> sentinel)."""
    s = np.full(52, SENTINEL, dtype=np.float32)
    s[0:6] = p1_player
    s[26:32] = p2_player
    for i, base in enumerate((6, 10, 14, 18, 22)):
        if p1_bullets[i] is not None:
            s[base : base + 4] = p1_bullets[i]
    for i, base in enumerate((32, 36, 40, 44, 48)):
        if p2_bullets[i] is not None:
            s[base : base + 4] = p2_bullets[i]
    return s


# --- index / split constants -------------------------------------------------
def test_player_indices_are_p1_0_6_then_p2_26_32():
    assert PLAYER_INDICES == (0, 1, 2, 3, 4, 5, 26, 27, 28, 29, 30, 31)
    assert N_PLAYER == 12


def test_bullet_position_indices_are_slot_major_4_each():
    idx = bullet_position_indices()
    assert len(idx) == N_BULLET_POS == 40
    # First slot is base 6 -> 6,7,8,9; P2's first is base 32 -> 32,33,34,35.
    assert idx[0:4] == [6, 7, 8, 9]
    assert idx[20:24] == [32, 33, 34, 35]
    # Slot bases recoverable as every 4th entry.
    assert tuple(idx[::4]) == BULLET_BASE_INDICES


def test_worker_id_from_shard_filename():
    assert worker_id_from_shard("shard_w0_0000.npz") == 0
    assert worker_id_from_shard("shard_w3_0247.npz") == 3
    with pytest.raises(ValueError):
        worker_id_from_shard("not_a_shard.npz")


# --- player target slicing ---------------------------------------------------
def test_encode_player_targets_picks_the_12_always_present_floats():
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [11, 12, 13, 14, 15, 16]
    s = _make_state(p1, p2, [None] * 5, [None] * 5)
    out = encode_player_targets(s[None, :])
    assert out.shape == (1, 12)
    np.testing.assert_array_equal(out[0], np.array(p1 + p2, dtype=np.float32))


# --- bullet presence + position (sentinel-aware) -----------------------------
def test_encode_bullet_targets_presence_and_positions():
    # P1: slots 0 and 2 present; P2: slot 4 present. Rest absent.
    p1b = [[1, 2, 3, 4], None, [5, 6, 7, 8], None, None]
    p2b = [None, None, None, None, [9, 9, 9, 9]]
    s = _make_state([0] * 6, [0] * 6, p1b, p2b)
    presence, pos = encode_bullet_targets(s[None, :])
    assert presence.shape == (1, 10)
    assert pos.shape == (1, 40)
    # presence bits: P1 slots 0,2 and P2 slot 4 (global index 9).
    expected_presence = np.zeros(10, dtype=np.float32)
    expected_presence[[0, 2, 9]] = 1.0
    np.testing.assert_array_equal(presence[0], expected_presence)
    # absent slots retain sentinel in positions (must be masked, never regressed).
    assert pos[0, 4] == SENTINEL  # slot 1 (absent) first float
    np.testing.assert_array_equal(pos[0, 0:4], np.array([1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(pos[0, 8:12], np.array([5, 6, 7, 8], dtype=np.float32))
    np.testing.assert_array_equal(pos[0, 36:40], np.array([9, 9, 9, 9], dtype=np.float32))


def test_presence_partial_sentinel_slot_counts_present():
    # A slot with 3 sentinels but one real value is NOT all -100 -> present.
    s = np.full(52, SENTINEL, dtype=np.float32)
    s[0:6] = 0
    s[26:32] = 0
    s[6:10] = [SENTINEL, SENTINEL, SENTINEL, 0.5]  # P1 slot 0: one real float
    presence, _ = encode_bullet_targets(s[None, :])
    assert presence[0, 0] == 1.0


def test_decode_targets_masks_absent_slot_positions_to_zero():
    s = np.full(52, SENTINEL, dtype=np.float32)
    s[0:6] = [1, 1, 1, 1, 1, 1]
    s[26:32] = [2, 2, 2, 2, 2, 2]
    # only P1 slot 0 present.
    s[6:10] = [3, 4, 5, 6]
    states = s[None, :]
    wall_grids = np.zeros((10, 12, 20), dtype=np.uint8)
    stats = fit_norm_stats(states)
    t = decode_targets(states, np.array([0]), wall_grids, stats)
    mask = t["bullet_pos_mask"][0]
    assert mask[0:4].sum() == 4  # present slot
    assert mask[4:].sum() == 0  # all other slots absent
    # masked positions are exactly zero (inert), never the sentinel.
    assert np.all(t["bullet_pos"][0, 4:] == 0.0)


# --- wall lookup -------------------------------------------------------------
def test_wall_target_lookup_by_map_id():
    wall_grids = np.zeros((10, 12, 20), dtype=np.uint8)
    wall_grids[3, 5, 7] = 1
    wall_grids[8, 0, 0] = 1
    map_ids = np.array([3, 8, 3])
    out = wall_target_for(map_ids, wall_grids)
    assert out.shape == (3, 12, 20)
    assert out[0, 5, 7] == 1.0
    assert out[1, 0, 0] == 1.0
    assert out[2, 5, 7] == 1.0
    assert out[0].sum() == 1.0


# --- normalization round-trip + sentinel exclusion ---------------------------
def test_normalize_denormalize_round_trip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 12)).astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = normalize(x, mean, std)
    back = denormalize(z, mean, std)
    np.testing.assert_allclose(back, x, rtol=1e-4, atol=1e-4)


def test_fit_norm_stats_player_matches_numpy():
    rng = np.random.default_rng(1)
    states = np.full((100, 52), SENTINEL, dtype=np.float32)
    states[:, 0:6] = rng.normal(size=(100, 6))
    states[:, 26:32] = rng.normal(size=(100, 6))
    stats = fit_norm_stats(states)
    player = encode_player_targets(states)
    np.testing.assert_allclose(stats.player_mean, player.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats.player_std, player.std(axis=0), atol=1e-5)


def test_fit_norm_stats_bullet_excludes_sentinels():
    # Build states where ONLY P1 slot 0 is ever present, with known values.
    n = 64
    states = np.full((n, 52), SENTINEL, dtype=np.float32)
    states[:, 0:6] = 0.0
    states[:, 26:32] = 0.0
    present_vals = np.arange(n, dtype=np.float32)
    states[:, 6] = present_vals  # slot 0, field 0 present on every row
    states[:, 7] = present_vals * 2
    states[:, 8] = present_vals * 3
    states[:, 9] = present_vals * 4
    stats = fit_norm_stats(states)
    # Field 0 of bullet positions == state index 6; stats over present (=all) rows.
    assert stats.bullet_mean[0] == pytest.approx(present_vals.mean(), abs=1e-4)
    assert stats.bullet_std[0] == pytest.approx(present_vals.std(), abs=1e-4)
    # An always-absent field (e.g. slot 1 field 0 -> position idx 4) must NOT see -100:
    # its stats fall back to mean 0 / std 1 (no present samples), not the sentinel.
    assert stats.bullet_mean[4] == pytest.approx(0.0)
    assert stats.bullet_std[4] == pytest.approx(1.0)


def test_norm_stats_json_round_trip():
    rng = np.random.default_rng(2)
    states = np.full((40, 52), SENTINEL, dtype=np.float32)
    states[:, 0:6] = rng.normal(size=(40, 6))
    states[:, 26:32] = rng.normal(size=(40, 6))
    states[:, 6:10] = rng.normal(size=(40, 4))  # P1 slot 0 always present
    stats = fit_norm_stats(states)
    back = NormStats.from_json(stats.to_json())
    np.testing.assert_allclose(back.player_mean, stats.player_mean, atol=1e-6)
    np.testing.assert_allclose(back.bullet_std, stats.bullet_std, atol=1e-6)


# --- input-res parser --------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "expected"),
    [("160x90", (160, 90)), (" 320X180 ", (320, 180)), ("80x45", (80, 45))],
)
def test_parse_input_res(spec, expected):
    assert parse_input_res(spec) == expected


@pytest.mark.parametrize("bad", ["160", "160*90", "x90", "0x90", "-1x10", "axb"])
def test_parse_input_res_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_input_res(bad)


# --- stratified group split --------------------------------------------------
def test_stratified_group_split_holds_out_per_map_and_no_overlap():
    # 4 maps, 10 groups each.
    group_keys = []
    map_of_group = {}
    for m in range(4):
        for e in range(10):
            g = (m, e)  # worker=m as a stand-in
            group_keys.append(g)
            map_of_group[g] = m
    train, val = stratified_group_split(group_keys, map_of_group, val_frac=0.1, seed=0)
    assert train.isdisjoint(val)
    assert train | val == set(group_keys)
    # ~10% of 10 = 1 per map -> 4 val groups, one per map.
    assert len(val) == 4
    val_maps = sorted(map_of_group[g] for g in val)
    assert val_maps == [0, 1, 2, 3]


def test_stratified_group_split_is_deterministic_by_seed():
    group_keys = [(0, e) for e in range(20)] + [(1, e) for e in range(20)]
    map_of_group = {g: g[0] for g in group_keys}
    a = stratified_group_split(group_keys, map_of_group, val_frac=0.2, seed=7)
    b = stratified_group_split(group_keys, map_of_group, val_frac=0.2, seed=7)
    c = stratified_group_split(group_keys, map_of_group, val_frac=0.2, seed=8)
    assert a == b
    assert a != c  # different seed -> different partition (with high probability)


def test_split_single_group_map_goes_to_train_not_val():
    # A map with one group can't be split -> it must land in train (val needs >=2).
    group_keys = [(0, 0)] + [(1, e) for e in range(10)]
    map_of_group = {(0, 0): 0, **{(1, e): 1 for e in range(10)}}
    train, val = stratified_group_split(group_keys, map_of_group, val_frac=0.1, seed=0)
    assert (0, 0) in train
    assert (0, 0) not in val
