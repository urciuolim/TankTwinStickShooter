"""Shard-streaming dataset for the single-frame decoder (torch ``Dataset``).

Builds a :class:`~pop_trainer.data.readers.DatasetIndex` over a decode-v1 directory, splits it
map-aware (group key = ``map_id``, 80/10/10, no leak), and serves ``(frame, target_dict)`` rows
by lazily fetching one frame from its shard. The last-read shard's arrays are cached so a run of
rows in the same shard does not re-open the ``.npz`` every ``__getitem__`` — but frames are
never bulk-loaded into RAM.

decode-v1 shards live in ``worker_*/`` SUBDIRS, so the index is built with a RECURSIVE glob
(``**/shard_*.npz``); a flat-layout dir works too. Frames are ``uint8 (H, W, 3)`` -> float32
``[0, 1]`` NCHW ``(3, h, w)``.

LOAD-TIME downsample: ``resolution`` is the target HEIGHT (one of 360 / 180 / 90); the WIDTH
scales by the SAME integer factor as the height (native 360 -> factor 1/2/4 -> 360/180/90), so
the native frame aspect is preserved. Resize is deterministic area-interpolation.

Targets are extracted per-row from the 52-float state via
:mod:`pop_trainer.pretraining.targets`, normalized by a :class:`~...targets.NormStats` that is
fit on the TRAIN split ONLY (no val/test leak). Build the three split views with
:func:`build_splits` so they share one index and one stats object.

torch + numpy + :mod:`pop_trainer.data` + :mod:`pop_trainer.pretraining.targets`; nothing from
``env`` / ``rl``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from pop_trainer.data import readers, schema, shards
from pop_trainer.pretraining import targets as T

__all__ = ["RESOLUTIONS", "DecodeDataset", "SplitDatasets", "build_splits"]

# Allowed target heights for the load-time downsample (native frame height is 360).
RESOLUTIONS: tuple[int, ...] = (360, 180, 90)
_NATIVE_HEIGHT = 360


def _downsample_factor(native_h: int, resolution: int) -> int:
    """Integer downsample factor mapping ``native_h`` to ``resolution`` (must divide evenly)."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution}")
    if native_h % resolution != 0:
        raise ValueError(f"native height {native_h} is not divisible by resolution {resolution}")
    return native_h // resolution


def _frame_to_chw(frame: np.ndarray, factor: int) -> torch.Tensor:
    """``uint8 (H, W, 3)`` -> float32 ``[0,1]`` NCHW ``(3, H/factor, W/factor)`` (area resize)."""
    t = torch.from_numpy(np.ascontiguousarray(frame)).to(torch.float32).div_(255.0)
    chw = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    if factor > 1:
        h, w = chw.shape[2] // factor, chw.shape[3] // factor
        chw = F.interpolate(chw, size=(h, w), mode="area")
    return chw.squeeze(0)


def _resolve_shard_paths(shard_dir: Path, shard_files: list[str], pattern: str) -> list[Path]:
    """Map the index's basenames back to full paths (shards live in ``worker_*/`` subdirs).

    ``DatasetIndex.shard_files`` stores only basenames, but the recursive glob walks
    subdirectories, so the full path is recovered by globbing ``pattern`` and indexing by
    basename. Basenames are unique across decode-v1's ``worker_*`` layout.
    """
    by_name = {p.name: p for p in shard_dir.glob(pattern)}
    missing = [f for f in shard_files if f not in by_name]
    if missing:
        raise ValueError(f"index references shards not found under {shard_dir}: {missing}")
    return [by_name[f] for f in shard_files]


class _ShardCache:
    """Single-entry cache of one shard's arrays so consecutive same-shard rows reuse the read."""

    def __init__(self, shard_paths: list[Path]) -> None:
        self._paths = shard_paths
        self._idx: int | None = None
        self._arrays: dict[str, np.ndarray] | None = None

    def arrays(self, shard_idx: int) -> dict[str, np.ndarray]:
        if shard_idx != self._idx:
            self._arrays = shards.read_shard(self._paths[shard_idx])
            self._idx = shard_idx
        assert self._arrays is not None
        return self._arrays


class DecodeDataset(Dataset):
    """One split view over a shared :class:`DatasetIndex`: serves ``(frame, target_dict)``.

    ``sample_indices`` are positions into the shared index (one split's rows). ``stats`` is the
    TRAIN-fit normalization applied to every row's targets. Construct via :func:`build_splits`,
    which wires the index, split, and stats consistently across train/val/test.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        index: readers.DatasetIndex,
        sample_indices: np.ndarray,
        stats: T.NormStats,
        resolution: int,
    ) -> None:
        self._index = index
        self._samples = np.asarray(sample_indices, dtype=np.int64)
        self._stats = stats
        self._factor = _downsample_factor(index.frame_hw[0], resolution)
        self._cache = _ShardCache(shard_paths)
        self.resolution = resolution

    @property
    def stats(self) -> T.NormStats:
        """The TRAIN-fit normalization stats this view applies (shared across splits)."""
        return self._stats

    def sample_shards(self) -> np.ndarray:
        """``(len(self),)`` int64 shard id per LOCAL index ``i`` (for shard-grouped sampling).

        Lets a sampler group local indices by their backing shard so consecutive ``__getitem__``
        calls hit the same ``.npz`` and the single-entry cache decompresses each shard once per
        pass — without reaching into this view's private split positions.
        """
        return self._index.sample_shard[self._samples].astype(np.int64)

    def __len__(self) -> int:
        return int(self._samples.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = int(self._samples[i])
        shard_idx = int(self._index.sample_shard[sample])
        row = int(self._index.sample_row[sample])
        arrays = self._cache.arrays(shard_idx)
        frame = arrays[schema.ARRAY_FRAMES][row]
        state = arrays[schema.ARRAY_STATES][row]
        x = _frame_to_chw(frame, self._factor)
        tgt = T.extract_targets(state[None, :], self._stats)
        target = {k: torch.from_numpy(v[0]) for k, v in tgt.items()}
        return x, target


class SplitDatasets:
    """The three map-aware split views (``train`` / ``val`` / ``test``) + the shared stats."""

    def __init__(
        self, train: DecodeDataset, val: DecodeDataset, test: DecodeDataset, stats: T.NormStats
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.stats = stats


def _collect_train_states(
    shard_paths: list[Path], index: readers.DatasetIndex, train_idx: np.ndarray
) -> np.ndarray:
    """Gather the TRAIN-split states (only the small 52-float vectors) to fit norm stats."""
    cache = _ShardCache(shard_paths)
    order = np.argsort(index.sample_shard[train_idx], kind="stable")
    out = np.empty((train_idx.shape[0], schema.STATE_LEN), dtype=np.float32)
    for pos in order:
        sample = int(train_idx[pos])
        shard_idx = int(index.sample_shard[sample])
        row = int(index.sample_row[sample])
        out[pos] = cache.arrays(shard_idx)[schema.ARRAY_STATES][row]
    return out


def build_splits(
    data_dir: str | Path,
    *,
    resolution: int = 180,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    limit: int | None = None,
    pattern: str = "**/shard_*.npz",
) -> SplitDatasets:
    """Build the map-aware train/val/test :class:`DecodeDataset` views over ``data_dir``.

    Indexes ``data_dir`` recursively (``**/shard_*.npz`` covers the ``worker_*/`` layout),
    splits the index map-aware (group key = ``map_id``, no leak), fits :class:`NormStats` on the
    TRAIN split ONLY, and returns the three views sharing that index + stats. ``limit`` caps the
    TOTAL number of indexed samples (a deterministic stride subset) for smoke runs. Raises
    ``ValueError`` if the recursive glob finds no shards.
    """
    data_dir = Path(data_dir)
    index = readers.build_index(data_dir, pattern=pattern)
    if len(index) == 0:
        raise ValueError(f"empty index over {data_dir} with pattern {pattern!r}")

    if limit is not None and limit < len(index):
        index = _subset_index(index, limit)

    shard_paths = _resolve_shard_paths(data_dir, index.shard_files, pattern)
    split = index.split(val_frac=val_frac, test_frac=test_frac, seed=seed)
    train_states = _collect_train_states(shard_paths, index, split.train)
    stats = T.fit_norm_stats(train_states)

    def view(idxs: np.ndarray) -> DecodeDataset:
        return DecodeDataset(shard_paths, index, idxs, stats, resolution)

    return SplitDatasets(view(split.train), view(split.val), view(split.test), stats)


def _subset_index(index: readers.DatasetIndex, limit: int) -> readers.DatasetIndex:
    """Deterministic shard-local subset of ``<= limit`` samples (keeps map coverage, no thrash).

    Distributes the budget evenly across the shards present in the index and keeps a CONTIGUOUS
    per-shard PREFIX of rows from each (the index is built in shard order, so a shard's samples
    are a contiguous block). Reading the subset still touches each shard once per pass instead of
    striding across every shard per row. Every shard contributes at least one row whenever the
    budget allows, preserving the map coverage the split needs.
    """
    shard_ids = index.sample_shard
    present = np.unique(shard_ids)
    quotas = readers.even_group_targets(present.shape[0], limit)
    keep_parts: list[np.ndarray] = []
    for shard, quota in zip(present, quotas, strict=True):
        rows = np.nonzero(shard_ids == shard)[0]  # contiguous, ascending (shard-ordered index)
        keep_parts.append(rows[:quota])
    keep = (
        np.sort(np.concatenate(keep_parts)).astype(np.int64)
        if keep_parts
        else np.empty(0, dtype=np.int64)
    )
    return readers.DatasetIndex(
        shard_files=index.shard_files,
        map_ids=index.map_ids[keep],
        episode_ids=index.episode_ids[keep],
        step_idxs=index.step_idxs[keep],
        sample_shard=index.sample_shard[keep],
        sample_row=index.sample_row[keep],
        frame_hw=index.frame_hw,
    )
