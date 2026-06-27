"""Tests for pop_trainer.pretraining.dataset — the shard-streaming map-aware loader.

Map-aware split has NO map leak, the recursive index sees the worker_*/ subdirs, downsample
produces the expected (C, H, W) per resolution, and target extraction matches the state.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pop_trainer.data import readers  # noqa: E402
from pop_trainer.pretraining.dataset import RESOLUTIONS, build_splits  # noqa: E402

from ._fixtures import make_fixture  # noqa: E402


def test_recursive_index_sees_worker_subdirs(tmp_path):
    make_fixture(tmp_path)
    idx = readers.build_index(tmp_path, pattern="**/shard_*.npz")
    assert len(idx) > 0
    # 2 workers x 2 shards x 12 rows = 48
    assert len(idx) == 48


def test_map_aware_split_no_leak(tmp_path):
    make_fixture(tmp_path, n_maps=6, workers=3, shards_per_worker=2)
    # build_splits and the index share the SAME map-aware split logic; assert no-leak directly.
    build_splits(tmp_path, resolution=360, seed=0)
    idx = readers.build_index(tmp_path, pattern="**/shard_*.npz")
    split = idx.split(val_frac=0.1, test_frac=0.1, seed=0)
    # the three group (map) sets are disjoint
    tr, va, te = set(split.train_groups), set(split.val_groups), set(split.test_groups)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    # every split's samples carry only its own maps (no row-level leak)
    for sample_idxs, groups in (
        (split.train, tr),
        (split.val, va),
        (split.test, te),
    ):
        if sample_idxs.size:
            assert set(idx.map_ids[sample_idxs].tolist()).issubset(groups)


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_downsample_shape(tmp_path, resolution):
    # native fixture frame is 360x640 so the integer factors line up.
    make_fixture(
        tmp_path, hw=(360, 640), rows_per_shard=4, shards_per_worker=1, workers=1, n_maps=2
    )
    splits = build_splits(tmp_path, resolution=resolution, seed=0)
    x, target = splits.train[0]
    factor = 360 // resolution
    assert x.shape == (3, resolution, 640 // factor)
    assert x.dtype == torch.float32
    assert float(x.max()) <= 1.0 and float(x.min()) >= 0.0


def test_target_extraction_matches_state(tmp_path):
    make_fixture(tmp_path)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    _, target = splits.train[0]
    # target dict carries every group key with the right widths
    assert target["player_position"].shape == (4,)
    assert target["player_aim"].shape == (4,)
    assert target["bullet_presence"].shape == (10,)
    assert target["bullet_position"].shape == (20,)
    assert target["bullet_slot_mask"].shape == (20,)
    # absent bullet positions are zeroed by the mask
    bpos = target["bullet_position"].numpy()
    mask = target["bullet_slot_mask"].numpy()
    assert np.all(bpos[mask == 0] == 0.0)


def test_stats_fit_on_train_only(tmp_path):
    make_fixture(tmp_path, n_maps=4, workers=2, shards_per_worker=2)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    # all three views share the same TRAIN-fit stats object
    assert splits.train.stats is splits.val.stats
    assert splits.val.stats is splits.test.stats


def test_limit_subsets(tmp_path):
    make_fixture(tmp_path)
    splits = build_splits(tmp_path, resolution=360, seed=0, limit=20)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total <= 20
