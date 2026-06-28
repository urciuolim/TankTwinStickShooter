"""Tests for the windowed multi-shard train sampler and its effect on shard decompressions.

Two things must hold together: batches mix MANY shards (cross-shard diversity, what makes the
heatmap-CE decoder learn) WITHOUT re-decompressing a shard per sample (the loader perf
pathology). The KEY guard is :func:`test_one_epoch_reads_each_shard_exactly_once`: iterating one
epoch through the train loader (built the way train.py builds it, bounded cache = window + 1)
triggers exactly #train-shards shard reads, NOT ~#samples. Plus the sampler is a true per-epoch
permutation that varies across epochs, and full-window batches span the whole window.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from pop_trainer.data import shards  # noqa: E402
from pop_trainer.pretraining import dataset as ds_mod  # noqa: E402
from pop_trainer.pretraining.dataset import build_splits  # noqa: E402
from pop_trainer.pretraining.sampler import (  # noqa: E402
    ShardWindowBatchSampler,
    cache_capacity_for_window,
)
from pop_trainer.pretraining.train import build_train_loader  # noqa: E402

from ._fixtures import make_fixture  # noqa: E402


def test_cache_capacity_for_window_has_grace_slot():
    assert cache_capacity_for_window(1) == 2
    assert cache_capacity_for_window(8) == 9
    with pytest.raises(ValueError):
        cache_capacity_for_window(0)


def test_one_epoch_reads_each_shard_exactly_once(tmp_path, monkeypatch):
    # Many more samples than shards: 4 workers x 4 shards x 60 rows = 16 shards, 960 samples.
    make_fixture(tmp_path, n_maps=8, workers=4, shards_per_worker=4, rows_per_shard=60)
    splits = build_splits(tmp_path, resolution=360, seed=0)

    n_train_shards = len(np.unique(splits.train.sample_shards()))
    n_train_samples = len(splits.train)
    window = 5
    assert n_train_shards > window  # the window is genuinely smaller than the shard count
    assert n_train_samples > 10 * n_train_shards  # the read distinction is sharp

    reads = {"count": 0}
    real_read = shards.read_shard

    def counting_read(path):
        reads["count"] += 1
        return real_read(path)

    # the dataset's cache calls shards.read_shard via the module reference.
    monkeypatch.setattr(ds_mod.shards, "read_shard", counting_read)

    loader, _ = build_train_loader(splits.train, batch_size=16, window=window, seed=0)
    for _ in loader:
        pass

    # Bounded cache (capacity window + 1) + window <= capacity => one read per shard per epoch.
    assert reads["count"] == n_train_shards
    assert reads["count"] < n_train_samples


@pytest.mark.parametrize("drop_last", [False, True])
def test_epoch_is_a_permutation(tmp_path, drop_last):
    make_fixture(tmp_path, n_maps=6, workers=3, shards_per_worker=2, rows_per_shard=40)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    n = len(splits.train)
    batch_size = 7
    sampler = ShardWindowBatchSampler(
        splits.train.sample_shards(),
        batch_size=batch_size,
        window=4,
        seed=0,
        drop_last=drop_last,
    )
    sampler.set_epoch(0)
    batches = [list(b) for b in sampler]
    flat = [i for b in batches for i in b]

    if drop_last:
        # only the final short batch is dropped; every full batch is exactly batch_size
        assert all(len(b) == batch_size for b in batches)
        kept = (n // batch_size) * batch_size
        assert len(flat) == kept
        assert len(set(flat)) == len(flat)  # no dups among kept
        assert set(flat) <= set(range(n))
    else:
        assert sorted(flat) == list(range(n))  # every local index exactly once


def test_determinism_and_variation(tmp_path):
    make_fixture(tmp_path, n_maps=6, workers=3, shards_per_worker=2, rows_per_shard=40)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    shard_of = splits.train.sample_shards()

    def flat(sampler, epoch):
        sampler.set_epoch(epoch)
        return [i for b in sampler for i in b]

    a = ShardWindowBatchSampler(shard_of, batch_size=9, window=4, seed=0)
    b = ShardWindowBatchSampler(shard_of, batch_size=9, window=4, seed=0)
    # epoch 0 reproducible across two fresh samplers with the same (seed, epoch)
    assert flat(a, 0) == flat(b, 0)
    # different epochs reshuffle differently
    assert flat(a, 0) != flat(a, 1)
    # a different seed reshuffles differently too
    c = ShardWindowBatchSampler(shard_of, batch_size=9, window=4, seed=1)
    assert flat(a, 0) != flat(c, 0)


def test_cross_shard_diversity(tmp_path):
    # 24 shards, 60 rows each; window 8, batch 16 -> a full-window batch spans all 8 window shards.
    make_fixture(tmp_path, n_maps=12, workers=6, shards_per_worker=4, rows_per_shard=60)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    shard_of = splits.train.sample_shards()
    n_train_shards = len(np.unique(shard_of))
    window, batch_size = 8, 16
    assert n_train_shards >= 2 * window  # plenty of shards to keep the window full

    sampler = ShardWindowBatchSampler(shard_of, batch_size=batch_size, window=window, seed=0)
    sampler.set_epoch(0)
    distinct = [len({int(shard_of[i]) for i in b}) for b in sampler]

    # Round-robin over a full window of W shards yields exactly W distinct shards per batch
    # (more in a batch where shards drain mid-batch and new ones are admitted). Diversity only
    # tapers in the epoch tail once the shard order runs out and the window shrinks. So assert the
    # window's worth of diversity holds for the bulk of the epoch, never collapsing to one shard.
    assert max(distinct) >= window
    full = sum(d >= window for d in distinct)
    assert full >= len(distinct) - window  # only ~the last <window batches taper
    # the old single-shard sampler gave distinct == 1 for EVERY batch; that must be gone
    assert min(distinct[:-window]) >= window
    assert np.mean(distinct) > window * 0.8


def test_len_matches_emitted_batches(tmp_path):
    make_fixture(tmp_path, n_maps=6, workers=3, shards_per_worker=2, rows_per_shard=25)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    shard_of = splits.train.sample_shards()
    for drop_last in (False, True):
        sampler = ShardWindowBatchSampler(
            shard_of, batch_size=8, window=4, seed=0, drop_last=drop_last
        )
        sampler.set_epoch(0)
        assert len(sampler) == sum(1 for _ in sampler)
