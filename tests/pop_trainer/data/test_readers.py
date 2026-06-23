"""Tests for map-aware (group-key) splits: no map leaks across splits + determinism."""

import numpy as np
import pytest

from pop_trainer.data import readers, schema, shards


def _group_ids(per_map_counts):
    """Build a per-sample map-id array: map m repeated per_map_counts[m] times."""
    parts = [np.full(c, m, dtype=np.int32) for m, c in enumerate(per_map_counts)]
    return np.concatenate(parts)


def test_split_no_map_leak_across_splits():
    # 10 maps, varying sample counts.
    ids = _group_ids([20, 5, 13, 8, 30, 2, 17, 9, 11, 6])
    sp = readers.split_groups(ids, val_frac=0.2, test_frac=0.2, seed=42)

    # Every sample lands in exactly one split, and all of them are covered.
    all_idx = np.concatenate([sp.train, sp.val, sp.test])
    assert sorted(all_idx.tolist()) == list(range(len(ids)))

    # The group sets are disjoint: no map id appears in two splits.
    tg, vg, eg = (
        set(sp.train_groups.tolist()),
        set(sp.val_groups.tolist()),
        set(sp.test_groups.tolist()),
    )
    assert tg.isdisjoint(vg)
    assert tg.isdisjoint(eg)
    assert vg.isdisjoint(eg)
    # Their union is all maps.
    assert tg | vg | eg == set(np.unique(ids).tolist())

    # The KEY contract: each map's samples are ENTIRELY within one split.
    for split_idx, split_groups in (
        (sp.train, sp.train_groups),
        (sp.val, sp.val_groups),
        (sp.test, sp.test_groups),
    ):
        maps_in_split = set(ids[split_idx].tolist())
        assert maps_in_split == set(split_groups.tolist())


def test_split_deterministic_given_seed():
    ids = _group_ids([10] * 8)
    a = readers.split_groups(ids, val_frac=0.25, test_frac=0.25, seed=7)
    b = readers.split_groups(ids, val_frac=0.25, test_frac=0.25, seed=7)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.test, b.test)
    np.testing.assert_array_equal(a.train_groups, b.train_groups)


def test_split_different_seed_changes_assignment():
    ids = _group_ids([10] * 12)
    a = readers.split_groups(ids, val_frac=0.25, test_frac=0.25, seed=1)
    b = readers.split_groups(ids, val_frac=0.25, test_frac=0.25, seed=2)
    # Highly unlikely the group assignment is identical for two seeds on 12 groups.
    assert not np.array_equal(np.sort(a.val_groups), np.sort(b.val_groups)) or not np.array_equal(
        np.sort(a.test_groups), np.sort(b.test_groups)
    )


def test_split_canonical_order_independent_of_input_order():
    # The same set of (map -> count) samples, but presented in a shuffled row order, must give
    # the SAME group->split assignment (the split keys off the group id, not row position).
    counts = [7, 3, 9, 4, 6]
    ids = _group_ids(counts)
    rng = np.random.default_rng(0)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    a = readers.split_groups(ids, val_frac=0.2, test_frac=0.2, seed=3)
    b = readers.split_groups(shuffled, val_frac=0.2, test_frac=0.2, seed=3)
    np.testing.assert_array_equal(np.sort(a.train_groups), np.sort(b.train_groups))
    np.testing.assert_array_equal(np.sort(a.val_groups), np.sort(b.val_groups))
    np.testing.assert_array_equal(np.sort(a.test_groups), np.sort(b.test_groups))


def test_split_train_keeps_at_least_one_group_with_extreme_fracs():
    ids = _group_ids([5, 5, 5])  # only 3 groups
    sp = readers.split_groups(ids, val_frac=0.45, test_frac=0.45, seed=0)
    assert sp.train_groups.shape[0] >= 1  # train never empties


def test_split_two_groups_overflow_trim_keeps_train():
    # 2 groups, both fracs forced high -> val(1)+test(1)=2 > g-1=1 -> trim, train keeps 1.
    ids = _group_ids([5, 5])
    sp = readers.split_groups(ids, val_frac=0.4, test_frac=0.4, seed=0)
    assert sp.train_groups.shape[0] >= 1
    # Disjoint + complete despite the trim.
    tg = set(sp.train_groups.tolist())
    vg = set(sp.val_groups.tolist())
    eg = set(sp.test_groups.tolist())
    assert tg.isdisjoint(vg) and tg.isdisjoint(eg) and vg.isdisjoint(eg)
    assert tg | vg | eg == {0, 1}


def test_split_rejects_bad_fractions():
    ids = _group_ids([5, 5])
    with pytest.raises(ValueError):
        readers.split_groups(ids, val_frac=0.6, test_frac=0.6)
    with pytest.raises(ValueError):
        readers.split_groups(ids, val_frac=-0.1, test_frac=0.1)


def test_split_empty_groups():
    sp = readers.split_groups(np.empty(0, dtype=np.int32))
    assert sp.train.shape[0] == 0 and sp.val.shape[0] == 0 and sp.test.shape[0] == 0


def _write_shards(tmp_path, n_shards=3, per_shard=5, h=8, w=8):
    """Write a few shards with known map ids and return (dir, all map ids)."""
    rng = np.random.default_rng(0)
    all_ids = []
    for si in range(n_shards):
        frames = rng.integers(0, 256, size=(per_shard, h, w, 3), dtype=np.uint8)
        states = rng.standard_normal((per_shard, schema.STATE_LEN)).astype(np.float32)
        # Each shard concentrates on two map ids so the grouping is meaningful.
        map_ids = np.array([si, si + 10] * per_shard, dtype=np.int32)[:per_shard]
        episode_ids = np.full(per_shard, si, dtype=np.int32)
        step_idxs = np.arange(per_shard, dtype=np.int32)
        shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs)
        shards.write_shard(tmp_path / f"shard_w0_{si:04d}.npz", shard)
        all_ids.append(map_ids)
    return tmp_path, np.concatenate(all_ids)


def test_build_index_over_shard_dir(tmp_path):
    shard_dir, all_ids = _write_shards(tmp_path)
    idx = readers.build_index(shard_dir)
    assert len(idx) == all_ids.shape[0]
    np.testing.assert_array_equal(idx.map_ids, all_ids)
    assert idx.frame_hw == (8, 8)
    # sample_shard / sample_row locate each global sample back to its source.
    assert idx.sample_shard.min() == 0 and idx.sample_shard.max() == 2
    assert idx.sample_row.min() == 0
    # A split built from the index respects the group boundary.
    sp = idx.split(val_frac=0.2, test_frac=0.2, seed=0)
    for split_idx, groups in (
        (sp.train, sp.train_groups),
        (sp.val, sp.val_groups),
        (sp.test, sp.test_groups),
    ):
        assert set(idx.map_ids[split_idx].tolist()) == set(groups.tolist())


def test_build_index_rejects_mismatched_frame_sizes(tmp_path):
    _write_shards(tmp_path, n_shards=1, h=8, w=8)
    # Write one more shard at a different frame size.
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    states = np.zeros((3, schema.STATE_LEN), dtype=np.float32)
    shard = shards.Shard(
        frames, states, np.zeros(3, np.int32), np.zeros(3, np.int32), np.arange(3, dtype=np.int32)
    )
    shards.write_shard(tmp_path / "shard_w0_0099.npz", shard)
    with pytest.raises(ValueError):
        readers.build_index(tmp_path)


def test_build_index_no_shards_raises(tmp_path):
    with pytest.raises(ValueError):
        readers.build_index(tmp_path)


def test_even_group_targets_sums_to_total():
    parts = readers.even_group_targets(7, 100)
    assert sum(parts) == 100
    assert max(parts) - min(parts) <= 1


def test_per_map_quota():
    assert readers.per_map_quota(10, 95) == 10  # ceil(95/10)
    assert readers.per_map_quota(4, 8) == 2
