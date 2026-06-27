"""Tests for the shard-grouped train sampler and its effect on shard decompressions.

The KEY guard: iterating one training epoch through the train loader (built the way train.py
builds it) must trigger ~#shards shard reads, NOT ~#samples — the whole point of the fix. Plus
the sampler is a true per-epoch permutation that varies across epochs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from pop_trainer.data import shards  # noqa: E402
from pop_trainer.pretraining import dataset as ds_mod  # noqa: E402
from pop_trainer.pretraining.dataset import build_splits  # noqa: E402
from pop_trainer.pretraining.sampler import ShardGroupedBatchSampler  # noqa: E402

from ._fixtures import make_fixture  # noqa: E402


def _train_loader(splits, batch_size, seed):
    """Build the train loader exactly as train.run does (shard-grouped batch sampler)."""
    sampler = ShardGroupedBatchSampler(
        splits.train.sample_shards(), batch_size=batch_size, seed=seed
    )
    return DataLoader(splits.train, batch_sampler=sampler, num_workers=0), sampler


def test_one_epoch_reads_each_shard_about_once(tmp_path, monkeypatch):
    # Clearly more samples than shards: 3 workers x 3 shards x 60 rows = 9 shards, 540 samples.
    make_fixture(tmp_path, n_maps=4, workers=3, shards_per_worker=3, rows_per_shard=60)
    splits = build_splits(tmp_path, resolution=360, seed=0)

    n_train_shards = len(np.unique(splits.train.sample_shards()))
    n_train_samples = len(splits.train)
    assert n_train_samples > 3 * n_train_shards  # the distinction is sharp

    reads = {"count": 0}
    real_read = shards.read_shard

    def counting_read(path):
        reads["count"] += 1
        return real_read(path)

    # the dataset's cache calls shards.read_shard via the module reference.
    monkeypatch.setattr(ds_mod.shards, "read_shard", counting_read)

    loader, _ = _train_loader(splits, batch_size=16, seed=0)
    for _ in loader:
        pass

    # Single-entry cache + shard-grouped order => one decompression per shard per epoch.
    assert reads["count"] == n_train_shards
    assert reads["count"] < n_train_samples


def test_epoch_is_a_permutation_and_varies(tmp_path):
    make_fixture(tmp_path, n_maps=4, workers=2, shards_per_worker=2, rows_per_shard=40)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    n = len(splits.train)

    sampler = ShardGroupedBatchSampler(splits.train.sample_shards(), batch_size=7, seed=0)

    def flat(epoch):
        sampler.set_epoch(epoch)
        return [i for batch in sampler for i in batch]

    e0 = flat(0)
    # every local index exactly once (a permutation, no drops/dups)
    assert sorted(e0) == list(range(n))

    e1 = flat(1)
    assert sorted(e1) == list(range(n))
    # different epochs reshuffle to a different order
    assert e0 != e1


def test_batches_never_span_shard_boundary(tmp_path):
    make_fixture(tmp_path, n_maps=4, workers=2, shards_per_worker=2, rows_per_shard=30)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    shard_of = splits.train.sample_shards()
    sampler = ShardGroupedBatchSampler(shard_of, batch_size=8, seed=3)
    sampler.set_epoch(0)
    for batch in sampler:
        assert len({int(shard_of[i]) for i in batch}) == 1


def test_len_matches_emitted_batches(tmp_path):
    make_fixture(tmp_path, n_maps=4, workers=2, shards_per_worker=2, rows_per_shard=25)
    splits = build_splits(tmp_path, resolution=360, seed=0)
    sampler = ShardGroupedBatchSampler(splits.train.sample_shards(), batch_size=8, seed=0)
    sampler.set_epoch(0)
    assert len(sampler) == sum(1 for _ in sampler)
