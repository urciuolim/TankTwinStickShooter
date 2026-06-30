"""Tests for pop_trainer.pretraining.dataset — the shard-streaming map-aware loader.

Map-aware split has NO map leak, the recursive index sees the worker_*/ subdirs, downsample
produces the expected (C, H, W) per resolution, and target extraction matches the state.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pop_trainer.data import readers  # noqa: E402
from pop_trainer.pretraining.dataset import (  # noqa: E402
    NATIVE_RESOLUTION,
    build_splits,
    downsampled_hw,
)

from ._fixtures import make_fixture, make_square_fixture  # noqa: E402

# The 16:9 heights that evenly divide the native-360 fixture (the byte-identical legacy path).
WIDESCREEN_RESOLUTIONS = (360, 180, 90)


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


@pytest.mark.parametrize("resolution", WIDESCREEN_RESOLUTIONS)
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
    # SplitDatasets surfaces the native frame_hw + the derived training input_hw.
    assert splits.frame_hw == (360, 640)
    assert splits.input_hw == (resolution, 640 // factor)


@pytest.mark.parametrize("resolution", (64, NATIVE_RESOLUTION))
def test_square_64x64_dataset_loads_square(tmp_path, resolution):
    # a 64x64-native dataset trains square at (64, 64): resolution 64 -> factor 1, 0 -> native.
    make_square_fixture(
        tmp_path, rows_per_shard=4, shards_per_worker=1, workers=1, n_maps=2
    )
    splits = build_splits(tmp_path, resolution=resolution, seed=0)
    x, target = splits.train[0]
    assert x.shape == (3, 64, 64)
    assert x.dtype == torch.float32
    assert float(x.max()) <= 1.0 and float(x.min()) >= 0.0
    assert splits.frame_hw == (64, 64)
    assert splits.input_hw == (64, 64)
    # target extraction is unchanged at the square resolution.
    assert target["player_position"].shape == (4,)
    assert target["bullet_presence"].shape == (10,)


def test_native_resolution_no_downsample(tmp_path):
    # resolution=0 (native sentinel) keeps the dataset's frame_hw byte-for-byte (factor 1).
    make_fixture(
        tmp_path, hw=(360, 640), rows_per_shard=4, shards_per_worker=1, workers=1, n_maps=2
    )
    splits = build_splits(tmp_path, resolution=NATIVE_RESOLUTION, seed=0)
    x, _ = splits.train[0]
    assert x.shape == (3, 360, 640)
    assert splits.input_hw == (360, 640)


def test_downsampled_hw_pure():
    # pure helper: 16:9 downsample, square no-op, and the native sentinel.
    assert downsampled_hw((360, 640), 180) == (180, 320)
    assert downsampled_hw((64, 64), 64) == (64, 64)
    assert downsampled_hw((64, 64), NATIVE_RESOLUTION) == (64, 64)


def test_downsample_rejects_indivisible_height(tmp_path):
    make_fixture(
        tmp_path, hw=(360, 640), rows_per_shard=4, shards_per_worker=1, workers=1, n_maps=2
    )
    # 64 does not divide native 360 -> the loader refuses (the 16:9 datasets cannot go square).
    with pytest.raises(ValueError, match="not divisible"):
        build_splits(tmp_path, resolution=64, seed=0)


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


def test_build_splits_logs_counts_and_warns_on_thin_split(tmp_path, caplog):
    # 4 maps with the default 0.2/0.2: val/test each seat round(0.2*4)=1 map (<2) -> warning.
    make_fixture(tmp_path, n_maps=4, workers=2, shards_per_worker=2)
    with caplog.at_level(logging.INFO, logger="pop_trainer.pretraining.dataset"):
        build_splits(tmp_path, resolution=360, seed=0)
    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    # per-split map counts + sample fractions are surfaced at INFO
    assert any("split val:" in m and "maps" in m for m in info_msgs)
    assert any("split train:" in m for m in info_msgs)
    # the single-map val/test splits raise a WARNING
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("seats only 1 map" in m for m in warnings)


def test_limit_subsets(tmp_path):
    make_fixture(tmp_path)
    splits = build_splits(tmp_path, resolution=360, seed=0, limit=20)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total <= 20
