"""Shard-grouped batch sampler for the streaming train loader.

The training :class:`~pop_trainer.pretraining.dataset.DecodeDataset` serves one row per
``__getitem__`` by decompressing that row's backing shard; a single-entry cache makes a RUN of
same-shard rows reuse one decompression. A globally-shuffled order defeats the cache — consecutive
indices land in different shards, so every row re-decompresses a whole shard. This sampler keeps
the SGD benefit of shuffling while reading each shard ~once per epoch:

* shuffle the ORDER of shards each epoch (seeded, epoch-varying),
* shuffle the ROWS within each shard,
* emit indices GROUPED by shard, so consecutive ``__getitem__`` calls hit the same shard.

One epoch is a full PERMUTATION of the split's local indices (every sample exactly once, no
drops/dups) — just shard-grouped instead of globally shuffled. Yields lists of local indices, so
it plugs into a ``DataLoader`` via ``batch_sampler=``. Seed off ``cfg.seed`` and bump ``set_epoch``
each epoch so epochs differ; default epoch 0 keeps a fresh sampler deterministic.

numpy only (no torch); lives in ``pretraining`` and reasons over the per-index shard-id array a
``DecodeDataset`` exposes — nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["ShardGroupedBatchSampler"]


class ShardGroupedBatchSampler:
    """Yield batches of local indices grouped by backing shard, reshuffled per epoch.

    ``sample_shards[i]`` is the shard id of local index ``i`` (e.g.
    ``DecodeDataset.sample_shards()``). Each iteration shuffles shard order and within-shard rows
    with an RNG seeded by ``(seed, epoch)``, then walks the rows shard-by-shard, cutting batches of
    ``batch_size``. Batches never span a shard boundary, so the dataset's single-entry cache
    decompresses each shard once per epoch. ``drop_last`` drops the final short batch of EACH shard
    when set (default keeps them, so the epoch stays a full permutation).
    """

    def __init__(
        self,
        sample_shards: np.ndarray,
        *,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._shards = np.asarray(sample_shards, dtype=np.int64)
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        # Per-shard local-index buckets, in canonical (sorted shard id) order.
        unique = np.unique(self._shards)
        self._buckets: list[np.ndarray] = [
            np.nonzero(self._shards == s)[0].astype(np.int64) for s in unique
        ]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so the next iteration reshuffles differently (call before each epoch)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """Number of batches one epoch yields (depends on ``drop_last``)."""
        if self._drop_last:
            return sum(len(b) // self._batch_size for b in self._buckets)
        return sum((len(b) + self._batch_size - 1) // self._batch_size for b in self._buckets)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self._seed, self._epoch))
        order = rng.permutation(len(self._buckets))
        for bi in order:
            rows = self._buckets[bi].copy()
            rng.shuffle(rows)
            n = len(rows)
            stop = (n // self._batch_size) * self._batch_size if self._drop_last else n
            for start in range(0, stop, self._batch_size):
                yield rows[start : start + self._batch_size].tolist()
