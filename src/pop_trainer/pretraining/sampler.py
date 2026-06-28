"""Windowed multi-shard batch sampler for the streaming train loader.

The training :class:`~pop_trainer.pretraining.dataset.DecodeDataset` serves one row per
``__getitem__`` by decompressing that row's backing shard. The heatmap-CE localization decoder
only learns when batches carry cross-shard positional diversity: a batch confined to ONE shard
(one episode/map) has almost no positional spread and starves the heatmap gradient. But a globally
shuffled order re-decompresses a whole ``.npz`` PER SAMPLE, which is the loader perf pathology.

This sampler resolves the tension with a sliding WINDOW of ``window`` (W) resident shards:

* shuffle the ORDER of shards each epoch (seeded, epoch-varying) and the ROWS within each shard,
* keep W shards "in flight" at once and draw each batch's rows ROUND-ROBIN across those W shards,
  so a full-window batch mixes ~W distinct shards (hence W maps after the shuffle),
* when a shard's rows drain, drop it and admit the next shard from the shuffled order, so at most
  W shards are ever open at once.

Because at most W shards are open simultaneously, pairing this with the dataset's bounded LRU
shard cache at capacity ``C >= W + 1`` (see :func:`cache_capacity_for_window`) keeps an active
shard resident for its whole lifetime: a shard is decompressed exactly once — when it enters the
window — and evicted only after it fully drains. Decompressions per epoch == number of shards, not
number of samples.

One epoch is a full PERMUTATION of the split's local indices (every sample exactly once, no
drops/dups unless ``drop_last``). Yields lists of local indices, so it plugs into a ``DataLoader``
via ``batch_sampler=``. Seed off ``cfg.seed`` and bump ``set_epoch`` each epoch so epochs differ;
a fresh sampler at epoch 0 is deterministic.

RAM is the cost of W: each decode-v2 shard is ~580 frames at 360x640x3 uint8 (~0.4 GB resident
once decompressed), and the cache holds C = W + 1 of them, so the window costs ~(W + 1) x 0.4 GB
of system RAM. A batch's distinct-shard count is exactly W, and escaping the heatmap-CE
predict-center basin needs that diversity to approach what a full on-disk reshuffle gives (a 256-row
batch over 207 shards draws ~150 distinct shards). Empirically W=8 (~3.4 GB) does NOT escape (val
stays flat at the single-shard ~4.06 wu baseline) but W=32 does (val drops to ~0.3-0.4 wu in a few
epochs), so the default is W=32 (~13 GB). Diversity is per-EPISODE: decode-v2 has only ~10 maps but
207 shards (~one episode each), and one episode's trajectory is positionally correlated, so the
batch needs MANY episodes, not merely many maps. Raise W for more diversity (more RAM) or lower it
if RAM-bound.

numpy only (no torch); lives in ``pretraining`` and reasons over the per-index shard-id array a
``DecodeDataset`` exposes — nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["DEFAULT_SHARD_WINDOW", "ShardWindowBatchSampler", "cache_capacity_for_window"]

# Default window size: 32 shards in flight => 32-distinct-shard (-episode) batches, the diversity
# that escapes the heatmap-CE predict-center basin (W=8 does not), at ~(32 + 1) x 0.4 GB ~= 13 GB
# resident in the bounded shard cache.
DEFAULT_SHARD_WINDOW = 32


def cache_capacity_for_window(window: int) -> int:
    """Bounded-cache capacity C that keeps every windowed shard resident: ``C = window + 1``.

    The +1 is a grace slot. The cache is LRU; an OPEN shard (still in the window) is touched in
    every batch, so it is always more-recently-used than any DRAINED shard. With C = W + 1, a
    cache that is full holds at most W - 1 open shards plus the one being loaded, so the LRU
    victim is always a drained shard — never an active one. Hence each shard is decompressed
    exactly once per epoch. C = W can evict an active shard at a window transition and re-read it,
    so the wiring (``train.build_train_loader``) sets C from this function and guards ``C >= W``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return int(window) + 1


class ShardWindowBatchSampler:
    """Yield batches of local indices drawn round-robin across a sliding window of W shards.

    ``sample_shards[i]`` is the shard id of local index ``i`` (e.g.
    ``DecodeDataset.sample_shards()``). Each iteration shuffles shard order and within-shard rows
    with an RNG seeded by ``(seed, epoch)``, admits the first W shards into the window, then walks
    the window round-robin, emitting one row per shard in turn and cutting batches of
    ``batch_size``. When a shard drains it is dropped and the next shard from the shuffled order is
    admitted, so the window holds W shards until the order runs out (then it shrinks toward the
    epoch tail). At most W shards are open at once. ``drop_last`` drops only the final short batch
    of the epoch when set (default keeps it, so the epoch stays a full permutation).
    """

    def __init__(
        self,
        sample_shards: np.ndarray,
        *,
        batch_size: int,
        window: int = DEFAULT_SHARD_WINDOW,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._shards = np.asarray(sample_shards, dtype=np.int64)
        self._batch_size = int(batch_size)
        self._window = int(window)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._epoch = 0
        # Per-shard local-index buckets, in canonical (sorted shard id) order.
        unique = np.unique(self._shards)
        self._buckets: list[np.ndarray] = [
            np.nonzero(self._shards == s)[0].astype(np.int64) for s in unique
        ]
        self._n = int(self._shards.shape[0])

    @property
    def window(self) -> int:
        """Number of shards kept resident in the window (W)."""
        return self._window

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so the next iteration reshuffles differently (call before each epoch)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        """Number of batches one epoch yields (depends on ``drop_last``)."""
        if self._drop_last:
            return self._n // self._batch_size
        return (self._n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self._seed, self._epoch))
        shard_order = rng.permutation(len(self._buckets))

        # Each window slot is [rows, pos]: a shuffled row array and the next-to-emit position.
        window: list[list] = []
        next_admit = 0

        def admit() -> None:
            nonlocal next_admit
            while len(window) < self._window and next_admit < len(shard_order):
                rows = self._buckets[shard_order[next_admit]].copy()
                rng.shuffle(rows)
                window.append([rows, 0])
                next_admit += 1

        admit()
        batch: list[int] = []
        rr = 0  # round-robin cursor into ``window``
        while window:
            rows, pos = window[rr]
            batch.append(int(rows[pos]))
            window[rr][1] = pos + 1
            if window[rr][1] >= len(rows):
                # Shard drained: drop it and admit the next from the shuffled order. The slot that
                # followed shifts into ``rr``, so leaving ``rr`` put points at the next shard.
                window.pop(rr)
                admit()
                rr = rr % len(window) if window else 0
            else:
                rr = (rr + 1) % len(window)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch and not self._drop_last:
            yield batch
