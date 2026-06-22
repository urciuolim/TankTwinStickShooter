"""Pure-logic tests for the pixel inverse-renderer pretraining helpers.

Covers ONLY the numpy-only, torch-free pieces: objective config -> head sizing /
field selection (incl. a disabled group being fully dropped and re-enabling restoring
it), bullet position vs direction separation (the 20/20 split of the old 40), target
encoding / sentinel masking, wall-grid lookup, normalization round-trip, the WxH
parser, the stratified group split, the 3-way map-aware split (name->id holdout,
no leakage, determinism), the interior-wall mask (exactly 128 cells), and F1/IoU.
The trainer (torch) is NOT unit-tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tank_twin.pretrain_pixels import (
    BULLET_BASE_INDICES,
    DEFAULT_OBJECTIVES,
    N_BULLET_DIR,
    N_BULLET_POS,
    N_PLAYER,
    OBJECTIVE_GROUPS,
    PLAYER_INDICES,
    PLAYER_SUBGROUP_VEC_INDICES,
    SENTINEL,
    MapAwareSplit,
    NormStats,
    ObjectiveConfig,
    aim_angular_error_deg,
    binary_f1,
    bullet_direction_indices,
    bullet_position_indices,
    cosine_warmup_lr_multiplier,
    decode_targets,
    denormalize,
    encode_bullet_targets,
    encode_player_targets,
    fit_norm_stats,
    head_sizes,
    interior_wall_mask,
    interior_wall_pos_weight,
    iou_score,
    map_aware_split,
    normalize,
    parse_input_res,
    parse_objectives,
    parse_pos_weight_arg,
    per_player_cosine_distance,
    player_head_size,
    player_head_vec_indices,
    presence_pos_weight,
    resolve_map_ids,
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


def test_bullet_position_and_direction_indices_split_the_old_40():
    pos = bullet_position_indices()
    direction = bullet_direction_indices()
    assert len(pos) == N_BULLET_POS == 20
    assert len(direction) == N_BULLET_DIR == 20
    # Disjoint and together cover all 40 bullet floats.
    assert set(pos).isdisjoint(direction)
    assert sorted(pos + direction) == sorted(b + o for b in BULLET_BASE_INDICES for o in range(4))
    # First slot (base 6): pos = 6,7 ; dir = 8,9. P2 first slot (base 32): pos 32,33 ; dir 34,35.
    assert pos[0:2] == [6, 7]
    assert direction[0:2] == [8, 9]
    assert pos[10:12] == [32, 33]
    assert direction[10:12] == [34, 35]


def test_worker_id_from_shard_filename():
    assert worker_id_from_shard("shard_w0_0000.npz") == 0
    assert worker_id_from_shard("shard_w3_0247.npz") == 3
    with pytest.raises(ValueError):
        worker_id_from_shard("not_a_shard.npz")


# =============================================================================
# Objective config -> head sizing / field selection / drop-and-restore
# =============================================================================
def test_default_objectives_disable_velocity_keep_aim():
    cfg = ObjectiveConfig.default()
    assert cfg.is_on("player_position")
    assert cfg.is_on("player_aim")
    assert not cfg.is_on("player_velocity")  # velocity OFF by default
    assert set(DEFAULT_OBJECTIVES) == set(OBJECTIVE_GROUPS) - {"player_velocity"}
    assert len(cfg.enabled) == 6


def test_head_sizes_default_drops_velocity_from_player_head():
    cfg = ObjectiveConfig.default()
    sizes = head_sizes(cfg)
    # player head = position(4) + aim(4) = 8 ; velocity dropped.
    assert sizes["player"] == 8
    assert player_head_size(cfg) == 8
    assert sizes["bullet_presence"] == 10
    assert sizes["bullet_position"] == 20
    assert sizes["bullet_direction"] == 20
    assert sizes["wall"] == 240


def test_disabled_group_fully_dropped_from_heads_and_reenable_restores():
    # Disable bullet_direction + walls explicitly.
    cfg = ObjectiveConfig.from_set(
        ["player_position", "player_aim", "bullet_presence", "bullet_position"]
    )
    sizes = head_sizes(cfg)
    assert "bullet_direction" not in sizes  # dropped: no head
    assert "wall" not in sizes
    assert not cfg.is_on("bullet_direction")
    assert not cfg.is_on("walls")
    # Re-enabling restores them identically.
    cfg2 = ObjectiveConfig.from_set(
        [
            "player_position",
            "player_aim",
            "bullet_presence",
            "bullet_position",
            "bullet_direction",
            "walls",
        ]
    )
    sizes2 = head_sizes(cfg2)
    assert sizes2["bullet_direction"] == 20
    assert sizes2["wall"] == 240


def test_player_head_vec_indices_only_enabled_subgroups_in_order():
    # position + aim (default-style) -> 0,1,6,7 (pos) then 4,5,10,11 (aim).
    cfg = ObjectiveConfig.from_set(["player_position", "player_aim"])
    assert player_head_vec_indices(cfg) == [0, 1, 6, 7, 4, 5, 10, 11]
    # velocity-only -> just velocity indices.
    cfg2 = ObjectiveConfig.from_set(["player_velocity"])
    assert player_head_vec_indices(cfg2) == [2, 3, 8, 9]
    # The sub-group index map is consistent.
    assert PLAYER_SUBGROUP_VEC_INDICES["player_position"] == (0, 1, 6, 7)
    assert PLAYER_SUBGROUP_VEC_INDICES["player_velocity"] == (2, 3, 8, 9)
    assert PLAYER_SUBGROUP_VEC_INDICES["player_aim"] == (4, 5, 10, 11)


def test_no_player_groups_means_no_player_head():
    cfg = ObjectiveConfig.from_set(["walls"])
    sizes = head_sizes(cfg)
    assert "player" not in sizes
    assert player_head_size(cfg) == 0
    assert not cfg.any_player()


def test_parse_objectives_allowlist_and_disable_and_default():
    # Default (neither flag): all except velocity.
    assert parse_objectives(None, None).enabled == ObjectiveConfig.default().enabled
    # Allowlist.
    cfg = parse_objectives("walls,bullet_presence", None)
    assert cfg.enabled == frozenset({"walls", "bullet_presence"})
    # Disable subtracts from default.
    cfg2 = parse_objectives(None, "walls")
    assert "walls" not in cfg2.enabled
    assert "player_velocity" not in cfg2.enabled  # still off (default)
    assert cfg2.enabled == ObjectiveConfig.default().enabled - {"walls"}


def test_parse_objectives_rejects_unknown_and_both_flags_and_empty():
    with pytest.raises(ValueError):
        parse_objectives("bogus_group", None)
    with pytest.raises(ValueError):
        parse_objectives(None, "bogus_group")
    with pytest.raises(ValueError):
        parse_objectives("walls", "walls")  # both flags
    with pytest.raises(ValueError):
        parse_objectives("", None)  # empty allowlist
    with pytest.raises(ValueError):
        # disabling every default group leaves nothing.
        parse_objectives(None, ",".join(DEFAULT_OBJECTIVES))


def test_objective_config_json_round_trip_and_ordering():
    cfg = ObjectiveConfig.default()
    d = cfg.to_json()
    assert d["enabled"] == list(cfg.ordered())
    # ordered() follows canonical OBJECTIVE_GROUPS order.
    assert d["enabled"] == [g for g in OBJECTIVE_GROUPS if g != "player_velocity"]


# --- player target slicing ---------------------------------------------------
def test_encode_player_targets_picks_the_12_always_present_floats():
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [11, 12, 13, 14, 15, 16]
    s = _make_state(p1, p2, [None] * 5, [None] * 5)
    out = encode_player_targets(s[None, :])
    assert out.shape == (1, 12)
    np.testing.assert_array_equal(out[0], np.array(p1 + p2, dtype=np.float32))


# --- bullet presence + position + direction (sentinel-aware) -----------------
def test_encode_bullet_targets_presence_position_direction_separation():
    # P1: slots 0 and 2 present; P2: slot 4 present. Rest absent.
    p1b = [[1, 2, 3, 4], None, [5, 6, 7, 8], None, None]
    p2b = [None, None, None, None, [9, 9, 8, 8]]
    s = _make_state([0] * 6, [0] * 6, p1b, p2b)
    presence, pos, direction = encode_bullet_targets(s[None, :])
    assert presence.shape == (1, 10)
    assert pos.shape == (1, 20)
    assert direction.shape == (1, 20)
    expected_presence = np.zeros(10, dtype=np.float32)
    expected_presence[[0, 2, 9]] = 1.0
    np.testing.assert_array_equal(presence[0], expected_presence)
    # Slot 0: pos floats 1,2 ; dir floats 3,4. (pos = offsets 0,1 ; dir = offsets 2,3)
    np.testing.assert_array_equal(pos[0, 0:2], np.array([1, 2], dtype=np.float32))
    np.testing.assert_array_equal(direction[0, 0:2], np.array([3, 4], dtype=np.float32))
    # Slot 2 (pos index 4,5 in the 20-vec): floats 5,6 ; dir 7,8.
    np.testing.assert_array_equal(pos[0, 4:6], np.array([5, 6], dtype=np.float32))
    np.testing.assert_array_equal(direction[0, 4:6], np.array([7, 8], dtype=np.float32))
    # P2 slot 4 -> global slot 9 -> pos vec index 18,19.
    np.testing.assert_array_equal(pos[0, 18:20], np.array([9, 9], dtype=np.float32))
    np.testing.assert_array_equal(direction[0, 18:20], np.array([8, 8], dtype=np.float32))
    # Absent slot 1 retains sentinel in BOTH pos and dir (must be masked, never regressed).
    assert pos[0, 2] == SENTINEL
    assert direction[0, 2] == SENTINEL


def test_presence_partial_sentinel_slot_counts_present():
    s = np.full(52, SENTINEL, dtype=np.float32)
    s[0:6] = 0
    s[26:32] = 0
    s[6:10] = [SENTINEL, SENTINEL, SENTINEL, 0.5]  # P1 slot 0: one real float
    presence, _, _ = encode_bullet_targets(s[None, :])
    assert presence[0, 0] == 1.0


def test_decode_targets_masks_absent_slot_positions_and_directions_to_zero():
    s = np.full(52, SENTINEL, dtype=np.float32)
    s[0:6] = [1, 1, 1, 1, 1, 1]
    s[26:32] = [2, 2, 2, 2, 2, 2]
    s[6:10] = [3, 4, 5, 6]  # only P1 slot 0 present
    states = s[None, :]
    wall_grids = np.zeros((10, 12, 20), dtype=np.uint8)
    stats = fit_norm_stats(states)
    t = decode_targets(states, np.array([0]), wall_grids, stats)
    mask = t["bullet_slot_mask"][0]  # (20,)
    assert mask[0:2].sum() == 2  # slot 0 present (2 floats)
    assert mask[2:].sum() == 0  # everything else absent
    assert np.all(t["bullet_position"][0, 2:] == 0.0)  # masked positions inert
    assert np.all(t["bullet_direction"][0, 2:] == 0.0)  # masked directions inert


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


# --- interior wall mask ------------------------------------------------------
def test_interior_wall_mask_is_exactly_128_cells_rows_2_9_cols_2_17():
    mask = interior_wall_mask()
    assert mask.shape == (12, 20)
    assert mask.dtype == bool
    assert int(mask.sum()) == 128  # 8 rows x 16 cols
    # Rows 2..9 inclusive, cols 2..17 inclusive are True; the 2-deep border is False.
    assert mask[2, 2] and mask[9, 17]
    assert not mask[1, 2] and not mask[10, 2]  # border rows
    assert not mask[2, 1] and not mask[2, 18]  # border cols
    # Every True cell lies inside the interior rectangle.
    rows, cols = np.where(mask)
    assert rows.min() == 2 and rows.max() == 9
    assert cols.min() == 2 and cols.max() == 17


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


def test_fit_norm_stats_bullet_pos_and_dir_exclude_sentinels():
    n = 64
    states = np.full((n, 52), SENTINEL, dtype=np.float32)
    states[:, 0:6] = 0.0
    states[:, 26:32] = 0.0
    present_vals = np.arange(n, dtype=np.float32)
    states[:, 6] = present_vals  # slot 0 pos_x
    states[:, 7] = present_vals * 2  # slot 0 pos_y
    states[:, 8] = present_vals * 3  # slot 0 vec_x (direction)
    states[:, 9] = present_vals * 4  # slot 0 vec_y (direction)
    stats = fit_norm_stats(states)
    # Bullet POSITION field 0 == state idx 6.
    assert stats.bullet_pos_mean[0] == pytest.approx(present_vals.mean(), abs=1e-3)
    assert stats.bullet_pos_std[0] == pytest.approx(present_vals.std(), abs=1e-3)
    # Bullet DIRECTION field 0 == state idx 8.
    assert stats.bullet_dir_mean[0] == pytest.approx((present_vals * 3).mean(), abs=1e-3)
    assert stats.bullet_dir_std[0] == pytest.approx((present_vals * 3).std(), abs=1e-3)
    # An always-absent field falls back to 0/1, never the sentinel.
    assert stats.bullet_pos_mean[2] == pytest.approx(0.0)  # slot 1 absent
    assert stats.bullet_pos_std[2] == pytest.approx(1.0)
    assert stats.bullet_dir_mean[2] == pytest.approx(0.0)
    assert stats.bullet_dir_std[2] == pytest.approx(1.0)


def test_norm_stats_json_round_trip():
    rng = np.random.default_rng(2)
    states = np.full((40, 52), SENTINEL, dtype=np.float32)
    states[:, 0:6] = rng.normal(size=(40, 6))
    states[:, 26:32] = rng.normal(size=(40, 6))
    states[:, 6:10] = rng.normal(size=(40, 4))  # P1 slot 0 always present
    stats = fit_norm_stats(states)
    back = NormStats.from_json(stats.to_json())
    np.testing.assert_allclose(back.player_mean, stats.player_mean, atol=1e-6)
    np.testing.assert_allclose(back.bullet_pos_std, stats.bullet_pos_std, atol=1e-6)
    np.testing.assert_allclose(back.bullet_dir_mean, stats.bullet_dir_mean, atol=1e-6)


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
    group_keys = []
    map_of_group = {}
    for m in range(4):
        for e in range(10):
            g = (m, e)
            group_keys.append(g)
            map_of_group[g] = m
    train, val = stratified_group_split(group_keys, map_of_group, val_frac=0.1, seed=0)
    assert train.isdisjoint(val)
    assert train | val == set(group_keys)
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
    assert a != c


def test_split_single_group_map_goes_to_train_not_val():
    group_keys = [(0, 0)] + [(1, e) for e in range(10)]
    map_of_group = {(0, 0): 0, **{(1, e): 1 for e in range(10)}}
    train, val = stratified_group_split(group_keys, map_of_group, val_frac=0.1, seed=0)
    assert (0, 0) in train
    assert (0, 0) not in val


# =============================================================================
# Map-id resolution + 3-way map-aware split
# =============================================================================
# The canonical name->id mapping (from wall_grids_meta / _resolve_maps('all')).
_NAME_TO_ID = {
    "center_block": 0,
    "central_cross": 1,
    "chokepoint": 2,
    "custom1_2021": 3,
    "diagonal_pillars": 4,
    "empty": 5,
    "four_pillars": 6,
    "opposing_l": 7,
    "ring_fragments": 8,
    "scattered": 9,
}


def test_resolve_map_ids_by_name_and_rejects_unknown():
    assert resolve_map_ids(["diagonal_pillars", "ring_fragments"], _NAME_TO_ID) == [4, 8]
    with pytest.raises(ValueError):
        resolve_map_ids(["no_such_map"], _NAME_TO_ID)


def _ten_map_groups(groups_per_map=10):
    group_keys = []
    map_of_group = {}
    for m in range(10):
        for e in range(groups_per_map):
            g = (m, e)
            group_keys.append(g)
            map_of_group[g] = m
    return group_keys, map_of_group


def test_map_aware_split_holds_out_two_whole_maps_no_leakage():
    group_keys, map_of_group = _ten_map_groups()
    holdout = resolve_map_ids(["diagonal_pillars", "ring_fragments"], _NAME_TO_ID)  # [4, 8]
    split = map_aware_split(group_keys, map_of_group, holdout, eval_frac=0.1, seed=0)
    assert isinstance(split, MapAwareSplit)
    assert split.holdout_map_ids == (4, 8)
    assert set(split.train_map_ids) == set(range(10)) - {4, 8}
    assert len(split.train_map_ids) == 8
    # Validation = every group on maps 4 and 8 (10 each = 20).
    assert all(map_of_group[g] in (4, 8) for g in split.validation)
    assert len(split.validation) == 20
    # Train + eval live only on the 8 training maps.
    assert all(map_of_group[g] not in (4, 8) for g in split.train | split.eval)
    # No group appears in more than one set (no frame leakage across the 3 sets).
    assert split.train.isdisjoint(split.eval)
    assert split.train.isdisjoint(split.validation)
    assert split.eval.isdisjoint(split.validation)
    # The three sets partition exactly the full group set.
    assert split.train | split.eval | split.validation == set(group_keys)
    # 90/10 within the 8 training maps -> 1 eval group per map = 8 eval groups.
    assert len(split.eval) == 8
    assert all(map_of_group[g] not in (4, 8) for g in split.eval)


def test_map_aware_split_deterministic_by_seed():
    group_keys, map_of_group = _ten_map_groups()
    holdout = [4, 8]
    a = map_aware_split(group_keys, map_of_group, holdout, eval_frac=0.2, seed=3)
    b = map_aware_split(group_keys, map_of_group, holdout, eval_frac=0.2, seed=3)
    c = map_aware_split(group_keys, map_of_group, holdout, eval_frac=0.2, seed=4)
    assert a == b
    assert a.train != c.train or a.eval != c.eval


def test_map_aware_split_json_serializable_and_inspectable():
    group_keys, map_of_group = _ten_map_groups(groups_per_map=4)
    split = map_aware_split(group_keys, map_of_group, [4, 8], eval_frac=0.25, seed=1)
    d = split.to_json()
    assert d["holdout_map_ids"] == [4, 8]
    assert len(d["train_map_ids"]) == 8
    # group lists are JSON-friendly [worker, episode] pairs.
    assert all(isinstance(g, list) and len(g) == 2 for g in d["validation_groups"])


# =============================================================================
# F1 / IoU metric helpers
# =============================================================================
def test_binary_f1_hand_checked():
    # pred:   1 1 0 0 1
    # target: 1 0 0 1 1
    # TP=2 (idx 0,4), FP=1 (idx 1), FN=1 (idx 3).
    pred = np.array([1, 1, 0, 0, 1])
    target = np.array([1, 0, 0, 1, 1])
    r = binary_f1(pred, target)
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_binary_f1_all_negative_predictions():
    pred = np.zeros(5)
    target = np.array([1, 0, 1, 0, 0])
    r = binary_f1(pred, target)
    assert r["precision"] == 0.0  # no positive predictions
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_binary_f1_perfect():
    pred = np.array([1, 0, 1, 1, 0])
    target = np.array([1, 0, 1, 1, 0])
    r = binary_f1(pred, target)
    assert r["f1"] == pytest.approx(1.0)


def test_interior_wall_f1_on_a_tiny_grid_example():
    # Build a (1,12,20) pred/target where the border is all-1 (would inflate naive acc)
    # but the interior differs by exactly known cells. Verify masking to interior.
    mask = interior_wall_mask()
    target = np.ones((12, 20), dtype=np.int64)  # everything wall
    pred = np.ones((12, 20), dtype=np.int64)
    # Flip 2 interior target cells to 0 and predict 1 there -> 2 FP on interior.
    target[2, 2] = 0
    target[3, 3] = 0
    # Flip 1 interior pred cell to 0 where target is 1 -> 1 FN.
    pred[4, 4] = 0
    pi = pred[mask].ravel()
    ti = target[mask].ravel()
    r = binary_f1(pi, ti)
    # interior cells = 128; TP = 128 - 2(target0) - 1(pred0, target1) = 125.
    assert r["tp"] == 125
    assert r["fp"] == 2  # predicted 1 where interior target 0
    assert r["fn"] == 1  # predicted 0 where interior target 1
    expected_iou = 125 / (125 + 2 + 1)
    assert iou_score(pi, ti) == pytest.approx(expected_iou)


def test_iou_both_empty_is_one():
    assert iou_score(np.zeros(10), np.zeros(10)) == 1.0


# =============================================================================
# Class-imbalance BCE pos_weights (CHANGE 1 + CHANGE 2 pure helpers)
# =============================================================================
def test_presence_pos_weight_known_positive_rate():
    # 100 slots, exactly 20 present -> #zeros/#ones = 80/20 = 4.0.
    presence = np.zeros((10, 10), dtype=np.float32)  # (N, 10) = 100 slots
    presence.ravel()[:20] = 1.0
    assert presence_pos_weight(presence) == pytest.approx(4.0)
    # Realistic ~7.8% present -> ratio ~11.8 (the measured-data regime).
    pres2 = np.zeros(1000, dtype=np.float32)
    pres2[:78] = 1.0  # 78 present / 922 absent
    assert presence_pos_weight(pres2) == pytest.approx(922 / 78)


def test_presence_pos_weight_is_train_only_input():
    # The helper takes ONLY the (train) array it is given; it cannot see eval/val.
    train = np.zeros(50, dtype=np.float32)
    train[:10] = 1.0  # 40 absent / 10 present = 4.0 on TRAIN
    # A hypothetical eval array with a different rate must NOT change the result.
    assert presence_pos_weight(train) == pytest.approx(4.0)


def test_presence_pos_weight_fallback_when_no_present_slots():
    presence = np.zeros((5, 10), dtype=np.float32)  # all absent
    assert presence_pos_weight(presence) == pytest.approx(1.0)  # default fallback
    assert presence_pos_weight(presence, fallback=7.0) == pytest.approx(7.0)


def test_interior_wall_pos_weight_uses_interior_only_ignores_border():
    # (N,12,20) with a KNOWN number of interior wall cells AND some border walls.
    n = 3
    walls = np.zeros((n, 12, 20), dtype=np.float32)
    mask = interior_wall_mask()  # 128 interior cells
    interior_rc = list(zip(*np.where(mask), strict=True))
    # Set 8 interior wall cells in EACH of the n grids (interior cells: rows 2..9 cols 2..17).
    n_interior_walls_per = 8
    for i in range(n):
        for r, c in interior_rc[:n_interior_walls_per]:
            walls[i, r, c] = 1.0
    # Add BORDER walls everywhere on the border for grid 0 (must NOT affect the ratio).
    border = ~mask
    walls[0][border] = 1.0
    total_interior = n * 128
    total_interior_walls = n * n_interior_walls_per
    expected = (total_interior - total_interior_walls) / total_interior_walls
    assert interior_wall_pos_weight(walls) == pytest.approx(expected)
    # Sanity: flipping ALL border cells on/off leaves the ratio unchanged.
    walls_no_border = walls.copy()
    walls_no_border[..., border] = 0.0
    assert interior_wall_pos_weight(walls_no_border) == pytest.approx(expected)


def test_interior_wall_pos_weight_fallback_when_no_interior_walls():
    # Only border walls; zero interior walls -> fallback (avoid div-by-zero).
    walls = np.zeros((2, 12, 20), dtype=np.float32)
    border = ~interior_wall_mask()
    walls[..., border] = 1.0  # border fully wall, interior fully free
    assert interior_wall_pos_weight(walls) == pytest.approx(1.0)
    assert interior_wall_pos_weight(walls, fallback=3.5) == pytest.approx(3.5)


def test_parse_pos_weight_arg_auto_vs_float():
    assert parse_pos_weight_arg("auto") is None
    assert parse_pos_weight_arg("AUTO") is None
    assert parse_pos_weight_arg(None) is None
    assert parse_pos_weight_arg("12.5") == pytest.approx(12.5)
    assert parse_pos_weight_arg(3.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        parse_pos_weight_arg("not_a_number")


# --- Interior-masked + pos-weighted wall BCE: border contributes 0 (torch) ---
def test_interior_masked_wall_loss_border_is_inert_and_walls_upweighted():
    """The wall loss must (a) ignore border cells entirely and (b) up-weight interior
    wall (positive) cells via pos_weight. Uses torch; skipped if torch is absent."""
    torch = pytest.importorskip("torch")
    from tank_twin._pretrain_pixels_train import InverseRenderer, compute_losses
    from tank_twin.pretrain_pixels import ObjectiveConfig

    mask = interior_wall_mask()
    cfg = ObjectiveConfig.from_set(["walls"])
    model = InverseRenderer(36, 60, cfg, embedding_dim=8)

    def wall_loss(pred_grid, tgt_grid, wall_pw):
        preds = {"wall": torch.tensor(pred_grid, dtype=torch.float32).reshape(1, 12, 20)}
        targets = {"wall": torch.tensor(tgt_grid, dtype=torch.float32).reshape(1, 12, 20)}
        lambdas = dict.fromkeys(("player", "presence", "bullet_pos", "bullet_dir", "wall"), 1.0)
        _total, parts = compute_losses(preds, targets, model, lambdas, 1.0, wall_pw)
        return parts["wall"]

    # Baseline: zero logits, interior target free everywhere.
    base_pred = np.zeros((12, 20), dtype=np.float32)
    base_tgt = np.zeros((12, 20), dtype=np.float32)
    loss_base = wall_loss(base_pred, base_tgt, 1.0)

    # (a) Changing a BORDER target/pred must NOT change the loss (border is masked out).
    border = ~mask
    br, bc = np.argwhere(border)[0]
    pred_border = base_pred.copy()
    tgt_border = base_tgt.copy()
    pred_border[br, bc] = 5.0  # large border logit
    tgt_border[br, bc] = 1.0  # flip border target
    loss_border = wall_loss(pred_border, tgt_border, 1.0)
    assert loss_border == pytest.approx(loss_base, abs=1e-6)

    # (b) An interior WALL (positive) cell is up-weighted vs an interior FREE cell.
    ir, ic = np.argwhere(mask)[0]
    pred_one = base_pred.copy()
    pred_one[ir, ic] = -3.0  # wrong sign for both cases; isolate the target/weight effect
    # Case free: interior target 0 at that cell (negative class, weight 1).
    tgt_free = base_tgt.copy()
    loss_free = wall_loss(pred_one, tgt_free, 5.0)
    # Case wall: interior target 1 at that cell (positive class, weight = wall_pw = 5).
    tgt_wall = base_tgt.copy()
    tgt_wall[ir, ic] = 1.0
    loss_wall = wall_loss(pred_one, tgt_wall, 5.0)
    # The positive cell is up-weighted, so the wall case incurs strictly more loss.
    assert loss_wall > loss_free


# =============================================================================
# Aim cosine-distance loss + angular-error metric (per player; pixel Stage-2)
# =============================================================================
def _unit(angle_deg: float) -> tuple[float, float]:
    """Unit 2D vector at ``angle_deg`` (degrees, CCW from +x)."""
    r = np.radians(angle_deg)
    return float(np.cos(r)), float(np.sin(r))


def test_per_player_cosine_distance_known_angles():
    # Both players identical direction -> distance 0.
    pred = np.array([[1.0, 0.0, 0.0, 1.0]])
    tgt = np.array([[1.0, 0.0, 0.0, 1.0]])
    assert per_player_cosine_distance(pred, tgt) == pytest.approx(0.0, abs=1e-7)
    # Both players opposite -> 1 - (-1) = 2 per player -> mean 2.
    pred_opp = np.array([[1.0, 0.0, 1.0, 0.0]])
    tgt_opp = np.array([[-1.0, 0.0, -1.0, 0.0]])
    assert per_player_cosine_distance(pred_opp, tgt_opp) == pytest.approx(2.0, abs=1e-7)
    # Both players orthogonal (90 deg) -> 1 - 0 = 1 per player -> mean 1.
    pred_o = np.array([[1.0, 0.0, 1.0, 0.0]])
    tgt_o = np.array([[0.0, 1.0, 0.0, 1.0]])
    assert per_player_cosine_distance(pred_o, tgt_o) == pytest.approx(1.0, abs=1e-7)
    # Known intermediate: 60 deg apart -> 1 - cos60 = 0.5 per player -> mean 0.5.
    p1x, p1y = _unit(0.0)
    t1x, t1y = _unit(60.0)
    pred_60 = np.array([[p1x, p1y, p1x, p1y]])
    tgt_60 = np.array([[t1x, t1y, t1x, t1y]])
    assert per_player_cosine_distance(pred_60, tgt_60) == pytest.approx(0.5, abs=1e-7)


def test_per_player_cosine_distance_is_per_player_not_4d():
    # P1 identical (cos 1 -> dist 0), P2 opposite (cos -1 -> dist 2).
    # Per-player mean distance = (0 + 2) / 2 = 1.0.
    pred = np.array([[1.0, 0.0, 1.0, 0.0]])
    tgt = np.array([[1.0, 0.0, -1.0, 0.0]])
    per_player = per_player_cosine_distance(pred, tgt)
    assert per_player == pytest.approx(1.0, abs=1e-7)
    # A single 4D cosine over the concatenation [1,0,1,0]·[1,0,-1,0] = 0 -> dist 1.0
    # numerically coincides here, so use a SECOND case that separates them cleanly:
    # P1 identical (dist 0), P2 orthogonal (dist 1) -> per-player mean = 0.5.
    pred2 = np.array([[1.0, 0.0, 1.0, 0.0]])
    tgt2 = np.array([[1.0, 0.0, 0.0, 1.0]])
    pp2 = per_player_cosine_distance(pred2, tgt2)
    assert pp2 == pytest.approx(0.5, abs=1e-7)
    # The equivalent 4D cosine: [1,0,1,0]·[1,0,0,1] = 1, |a|=|b|=sqrt2 -> cos=0.5,
    # dist_4d = 0.5; that EQUALS pp2 by coincidence, so add a THIRD asymmetric case
    # where 4D and per-player genuinely differ.
    # P1 at 0 deg vs target 0 deg (dist 0); P2 |pred|=2 magnitude but same dir as target.
    pred3 = np.array([[1.0, 0.0, 2.0, 0.0]])  # P2 pred magnitude 2, dir +x
    tgt3 = np.array([[1.0, 0.0, 1.0, 0.0]])  # P2 target dir +x
    pp3 = per_player_cosine_distance(pred3, tgt3)
    # Per-player cosine is magnitude-invariant -> both players cos 1 -> dist 0.
    assert pp3 == pytest.approx(0.0, abs=1e-6)
    # A naive 4D cosine: [1,0,2,0]·[1,0,1,0]=3, |a|=sqrt5,|b|=sqrt2 -> cos=3/sqrt10≈0.9487,
    # dist_4d≈0.0513 != 0. So the helper is provably per-player, not a 4D cosine.
    a = np.array([1.0, 0.0, 2.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 0.0])
    cos_4d = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    dist_4d = 1.0 - cos_4d
    assert dist_4d == pytest.approx(0.05131, abs=1e-4)
    assert pp3 != pytest.approx(dist_4d, abs=1e-3)


def test_per_player_cosine_distance_zero_pred_is_finite():
    # A zero-magnitude prediction must yield a finite loss (eps in the denominator).
    pred = np.array([[0.0, 0.0, 0.0, 0.0]])
    tgt = np.array([[1.0, 0.0, 0.0, 1.0]])
    d = per_player_cosine_distance(pred, tgt)
    assert np.isfinite(d)
    # cos -> 0 as |pred| -> 0, so distance -> 1 per player (not NaN/inf).
    assert d == pytest.approx(1.0, abs=1e-5)


def test_aim_angular_error_known_angles():
    # Identical -> ~0 deg, cos ~1 (the eps in the norm denominator makes cos just
    # under 1, so arccos gives a tiny nonzero angle ~0.01 deg; both are near-perfect).
    pred = np.array([[1.0, 0.0, 0.0, 1.0]])
    tgt = np.array([[1.0, 0.0, 0.0, 1.0]])
    m = aim_angular_error_deg(pred, tgt)
    assert m["deg"] == pytest.approx(0.0, abs=0.05)
    assert m["cos"] == pytest.approx(1.0, abs=1e-6)
    # Opposite -> ~180 deg, cos ~-1 (eps makes cos just above -1, so arccos lands a
    # hair under 180 deg ~179.99; near-perfect anti-alignment).
    m_opp = aim_angular_error_deg(
        np.array([[1.0, 0.0, 1.0, 0.0]]), np.array([[-1.0, 0.0, -1.0, 0.0]])
    )
    assert m_opp["deg"] == pytest.approx(180.0, abs=0.05)
    assert m_opp["cos"] == pytest.approx(-1.0, abs=1e-6)
    # Orthogonal -> 90 deg, cos 0.
    m_o = aim_angular_error_deg(np.array([[1.0, 0.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 0.0, 1.0]]))
    assert m_o["deg"] == pytest.approx(90.0, abs=1e-4)
    assert m_o["cos"] == pytest.approx(0.0, abs=1e-7)


def test_aim_angular_error_random_baseline_is_near_90_deg():
    # Random unit pred vs random unit target -> ~90 deg mean error, cos ~0 (chance).
    rng = np.random.default_rng(0)
    n = 20000
    ang_p = rng.uniform(-np.pi, np.pi, size=(n, 2))  # 2 players
    ang_t = rng.uniform(-np.pi, np.pi, size=(n, 2))
    pred = np.empty((n, 4))
    tgt = np.empty((n, 4))
    pred[:, 0], pred[:, 1] = np.cos(ang_p[:, 0]), np.sin(ang_p[:, 0])
    pred[:, 2], pred[:, 3] = np.cos(ang_p[:, 1]), np.sin(ang_p[:, 1])
    tgt[:, 0], tgt[:, 1] = np.cos(ang_t[:, 0]), np.sin(ang_t[:, 0])
    tgt[:, 2], tgt[:, 3] = np.cos(ang_t[:, 1]), np.sin(ang_t[:, 1])
    m = aim_angular_error_deg(pred, tgt)
    assert m["deg"] == pytest.approx(90.0, abs=2.0)  # mean angular error band
    assert m["cos"] == pytest.approx(0.0, abs=0.03)
    # The matching cosine-distance baseline lands near 1.0 (1 - cos ~ 1).
    assert per_player_cosine_distance(pred, tgt) == pytest.approx(1.0, abs=0.03)


def test_aim_angular_error_zero_pred_is_finite_near_90():
    # A zero predictor (the MSE collapse point) sits exactly at chance: cos 0, 90 deg.
    pred = np.zeros((100, 4))
    rng = np.random.default_rng(1)
    ang = rng.uniform(-np.pi, np.pi, size=(100, 2))
    tgt = np.empty((100, 4))
    tgt[:, 0], tgt[:, 1] = np.cos(ang[:, 0]), np.sin(ang[:, 0])
    tgt[:, 2], tgt[:, 3] = np.cos(ang[:, 1]), np.sin(ang[:, 1])
    m = aim_angular_error_deg(pred, tgt)
    assert np.isfinite(m["deg"]) and np.isfinite(m["cos"])
    assert m["cos"] == pytest.approx(0.0, abs=1e-6)
    assert m["deg"] == pytest.approx(90.0, abs=1e-4)


# =============================================================================
# FEATURE 1: --weight-decay arg plumbing (pure; torch-free)
# =============================================================================
def test_weight_decay_arg_default_zero_and_passthrough():
    from tank_twin.pretrain_pixels import _parse_args

    # Default is 0.0 (behavior-preserving: AdamW(weight_decay=0) ~ the old Adam).
    args = _parse_args(["--out", "runs/x"])
    assert isinstance(args.weight_decay, float)
    assert args.weight_decay == 0.0
    # A passed value plumbs through as a float.
    args2 = _parse_args(["--out", "runs/x", "--weight-decay", "0.01"])
    assert args2.weight_decay == pytest.approx(0.01)


def test_lr_schedule_arg_default_constant_and_accepts_cosine():
    from tank_twin.pretrain_pixels import _parse_args

    assert _parse_args(["--out", "runs/x"]).lr_schedule == "constant"
    assert _parse_args(["--out", "runs/x", "--lr-schedule", "cosine"]).lr_schedule == "cosine"
    assert _parse_args(["--out", "runs/x", "--lr-schedule", "constant"]).lr_schedule == "constant"


def test_lr_schedule_arg_rejects_invalid_choice():
    from tank_twin.pretrain_pixels import _parse_args

    # argparse rejects an out-of-choices value with SystemExit (exit code 2).
    with pytest.raises(SystemExit):
        _parse_args(["--out", "runs/x", "--lr-schedule", "linear"])


# =============================================================================
# FEATURE 2: cosine + linear-warmup LR multiplier (pure; torch-free)
# =============================================================================
def test_cosine_warmup_multiplier_curve_shape():
    total = 1000
    warmup_frac = 0.05
    warmup_steps = round(warmup_frac * total)  # 50
    # Reaches exactly 1.0 at the end of warmup.
    assert cosine_warmup_lr_multiplier(warmup_steps, total, warmup_frac) == pytest.approx(1.0)
    # Step 0 is strictly positive (never exactly 0) and < 1.0.
    m0 = cosine_warmup_lr_multiplier(0, total, warmup_frac)
    assert 0.0 < m0 < 1.0
    # Monotone increasing on [0, warmup_steps].
    warm = [cosine_warmup_lr_multiplier(s, total, warmup_frac) for s in range(warmup_steps + 1)]
    assert all(b > a for a, b in zip(warm[:-1], warm[1:], strict=True))
    # Monotone decreasing strictly after warmup, through to the end.
    post = [
        cosine_warmup_lr_multiplier(s, total, warmup_frac) for s in range(warmup_steps, total + 1)
    ]
    assert all(b < a for a, b in zip(post[:-1], post[1:], strict=True))
    # Final step decays to ~0.
    assert cosine_warmup_lr_multiplier(total, total, warmup_frac) == pytest.approx(0.0, abs=1e-9)
    # Cosine-region midpoint is exactly 0.5: progress == 0.5 at step = warmup + (total-warmup)/2.
    mid = warmup_steps + (total - warmup_steps) // 2
    assert cosine_warmup_lr_multiplier(mid, total, warmup_frac) == pytest.approx(0.5, abs=1e-9)


def test_cosine_warmup_multiplier_bounds_in_unit_interval():
    total = 333
    for s in range(total + 1):
        m = cosine_warmup_lr_multiplier(s, total, warmup_frac=0.05)
        assert 0.0 <= m <= 1.0


def test_cosine_warmup_multiplier_second_known_pair():
    # A second (step, total) pair with a different total + larger warmup_frac.
    total = 200
    warmup_frac = 0.1
    warmup_steps = round(warmup_frac * total)  # 20
    assert cosine_warmup_lr_multiplier(warmup_steps, total, warmup_frac) == pytest.approx(1.0)
    assert cosine_warmup_lr_multiplier(total, total, warmup_frac) == pytest.approx(0.0, abs=1e-9)
    # Quarter into the cosine region: progress == 0.25 -> 0.5*(1+cos(pi/4)).
    q = warmup_steps + (total - warmup_steps) // 4
    expected = 0.5 * (1.0 + np.cos(np.pi * 0.25))
    assert cosine_warmup_lr_multiplier(q, total, warmup_frac) == pytest.approx(expected, abs=1e-9)


def test_cosine_warmup_multiplier_degenerate_total_steps():
    # total_steps <= 1 must not raise / divide-by-zero; warmup spans the whole run.
    assert 0.0 <= cosine_warmup_lr_multiplier(0, 1, 0.05) <= 1.0
    assert 0.0 <= cosine_warmup_lr_multiplier(0, 0, 0.05) <= 1.0
    # A warmup_frac that rounds warmup to the full run still degrades gracefully.
    assert cosine_warmup_lr_multiplier(10, 10, 1.0) == pytest.approx(1.0)
