"""Dataset readers: map-aware (group-key) train/val/test splits over the shards.

The split contract for the inverse-render / supervised-decode dataset: samples are grouped by
their MAP (the group key = map id), and a whole map's samples land in EXACTLY ONE split. A
random per-sample split would leak the same map's frames across train and val (the same arena
geometry seen in training would inflate val), so we split by GROUP, not by row. This is the
standard "grouped split" guard against group leakage.

Everything here is PURE: it takes sample -> group metadata (an array of map ids, plus optional
shard membership) and returns index lists per split. No disk I/O is required to compute a split
(:func:`split_groups` works on a plain id array), so it is fully testable with fake metadata.
:func:`build_index` is the only function that touches disk — it scans a directory of shards to
assemble the per-sample group ids — and it is a thin loop over :mod:`pop_trainer.data.shards`'
cheap-metadata reads, kept separate from the pure split logic.

DETERMINISM: :func:`split_groups` takes a ``seed``; the same (group ids, fractions, seed) yields
the same split every time (groups are sorted to a canonical order, then a seeded
``numpy.random.default_rng`` permutation assigns whole groups to splits). No global RNG state is
touched.

stdlib + numpy + :mod:`pop_trainer.data.{schema,shards}` only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pop_trainer.data import schema, shards

__all__ = ["Split", "split_groups", "DatasetIndex", "build_index"]

_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """A grouped split: per-split sample-index arrays plus the group (map) ids in each split.

    * ``train`` / ``val`` / ``test`` — int64 arrays of SAMPLE indices (positions in the original
      per-sample group-id array), sorted ascending within each split.
    * ``train_groups`` / ``val_groups`` / ``test_groups`` — the group (map) ids assigned to each
      split. The three group sets are DISJOINT (the no-leak guarantee), so any given map id is in
      exactly one of them.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_groups: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    val_groups: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    test_groups: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def as_dict(self) -> dict[str, np.ndarray]:
        """``{"train": idx, "val": idx, "test": idx}`` sample-index view."""
        return {"train": self.train, "val": self.val, "test": self.test}

    def group_dict(self) -> dict[str, np.ndarray]:
        """``{"train": groups, "val": groups, "test": groups}`` group-id view."""
        return {"train": self.train_groups, "val": self.val_groups, "test": self.test_groups}


def split_groups(
    group_ids: Sequence[int] | np.ndarray,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Split:
    """Group-aware train/val/test split: whole groups (maps) never cross split boundaries.

    ``group_ids[i]`` is the group (map id) of sample ``i``. Returns a :class:`Split` whose
    per-split sample indices are exactly the samples whose group landed in that split. The
    GROUP sets are disjoint, so no map leaks across splits.

    Assignment is deterministic in ``seed``: the unique groups are sorted to a canonical order,
    a seeded ``default_rng`` permutes them, and the permuted groups are partitioned by count
    into test, then val, then the remainder to train. With ``G`` groups the val/test sizes are
    ``round(val_frac*G)`` / ``round(test_frac*G)`` but each is clamped so train keeps at least
    one group whenever ``G >= 1`` (and val/test get at least one group when their fraction is
    > 0 and there are spare groups). Raises ``ValueError`` if the fractions are invalid.
    """
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError(
            f"need val_frac>=0, test_frac>=0, val_frac+test_frac<1; "
            f"got val_frac={val_frac}, test_frac={test_frac}"
        )
    ids = np.asarray(group_ids)
    if ids.ndim != 1:
        raise ValueError(f"group_ids must be 1-D; got shape {ids.shape}")

    unique = np.unique(ids)  # sorted, canonical order independent of input order
    g = unique.shape[0]
    if g == 0:
        empty = np.empty(0, dtype=np.int64)
        return Split(empty, empty.copy(), empty.copy())

    rng = np.random.default_rng(seed)
    perm = rng.permutation(g)
    shuffled = unique[perm]

    n_test = _clamp_count(round(test_frac * g), test_frac, g)
    # Leave at least one group for train: val + test may take at most g-1 groups total.
    n_val = _clamp_count(round(val_frac * g), val_frac, g)
    if n_test + n_val > g - 1:
        # Trim val first, then test, so train always retains >= 1 group.
        overflow = n_test + n_val - (g - 1)
        take_from_val = min(n_val, overflow)
        n_val -= take_from_val
        overflow -= take_from_val
        n_test -= overflow

    test_groups = np.sort(shuffled[:n_test])
    val_groups = np.sort(shuffled[n_test : n_test + n_val])
    train_groups = np.sort(shuffled[n_test + n_val :])

    return Split(
        train=_indices_for_groups(ids, train_groups),
        val=_indices_for_groups(ids, val_groups),
        test=_indices_for_groups(ids, test_groups),
        train_groups=train_groups.astype(np.int64),
        val_groups=val_groups.astype(np.int64),
        test_groups=test_groups.astype(np.int64),
    )


def _clamp_count(n: int, frac: float, g: int) -> int:
    """Clamp a split's group count to ``[0, g]``; force >= 1 when ``frac > 0`` and groups spare."""
    n = max(0, min(int(n), g))
    if frac > 0 and n == 0 and g > 1:
        n = 1
    return n


def _indices_for_groups(ids: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sorted int64 sample indices whose group id is in ``groups``."""
    if groups.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    mask = np.isin(ids, groups)
    return np.nonzero(mask)[0].astype(np.int64)


@dataclass(frozen=True)
class DatasetIndex:
    """A flat index over a directory of shards: per-sample map id + provenance for splitting.

    Concatenates the small per-sample arrays across shards (NOT the frames) so a split can be
    computed without decompressing any frame. ``sample_shard`` / ``sample_row`` locate each
    global sample back to its (shard file, row) for a streaming loader to fetch later.

    * ``shard_files`` — shard basenames in scan order (sorted).
    * ``map_ids`` — (total,) int32 group key per sample (the split group key).
    * ``episode_ids`` / ``step_idxs`` — (total,) provenance.
    * ``sample_shard`` — (total,) int32 index into ``shard_files`` for each sample.
    * ``sample_row`` — (total,) int32 row WITHIN its shard.
    * ``frame_hw`` — the common ``(H, W)`` (validated identical across shards).
    """

    shard_files: list[str]
    map_ids: np.ndarray
    episode_ids: np.ndarray
    step_idxs: np.ndarray
    sample_shard: np.ndarray
    sample_row: np.ndarray
    frame_hw: tuple[int, int]

    def __len__(self) -> int:
        return int(self.map_ids.shape[0])

    def split(self, *, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> Split:
        """Map-aware split over this index (group key = ``map_ids``). See :func:`split_groups`."""
        return split_groups(self.map_ids, val_frac=val_frac, test_frac=test_frac, seed=seed)


def build_index(shard_dir: str | Path, pattern: str = "shard_*.npz") -> DatasetIndex:
    """Scan ``shard_dir`` for shards matching ``pattern`` and build a :class:`DatasetIndex`.

    Reads each shard's small arrays (``map_ids`` / ``episode_ids`` / ``step_idxs``) and the
    frame ``(H, W)`` header via :mod:`pop_trainer.data.shards` — it does NOT load frame bytes.
    Shards are processed in SORTED filename order (stable, reproducible global indices). Raises
    ``ValueError`` if no shards match or if shards disagree on ``(H, W)``.
    """
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob(pattern))
    if not paths:
        raise ValueError(f"no shards matching {pattern!r} in {shard_dir}")

    files: list[str] = []
    map_id_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    shard_idx_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    frame_hw: tuple[int, int] | None = None

    for si, p in enumerate(paths):
        arrays = shards.read_shard(p)
        n = int(arrays[schema.ARRAY_MAP_IDS].shape[0])
        hw = (int(arrays[schema.ARRAY_FRAMES].shape[1]), int(arrays[schema.ARRAY_FRAMES].shape[2]))
        if frame_hw is None:
            frame_hw = hw
        elif hw != frame_hw:
            raise ValueError(f"shard {p.name} frame size {hw} != dataset frame size {frame_hw}")
        files.append(p.name)
        map_id_parts.append(arrays[schema.ARRAY_MAP_IDS].astype(np.int32))
        episode_parts.append(arrays[schema.ARRAY_EPISODE_IDS].astype(np.int32))
        step_parts.append(arrays[schema.ARRAY_STEP_IDXS].astype(np.int32))
        shard_idx_parts.append(np.full(n, si, dtype=np.int32))
        row_parts.append(np.arange(n, dtype=np.int32))

    return DatasetIndex(
        shard_files=files,
        map_ids=_concat(map_id_parts, np.int32),
        episode_ids=_concat(episode_parts, np.int32),
        step_idxs=_concat(step_parts, np.int32),
        sample_shard=_concat(shard_idx_parts, np.int32),
        sample_row=_concat(row_parts, np.int32),
        frame_hw=frame_hw if frame_hw is not None else (0, 0),
    )


def _concat(parts: list[np.ndarray], dtype) -> np.ndarray:
    """Concatenate index parts, returning an empty typed array when there are none."""
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype)


def even_group_targets(num_groups: int, total: int) -> list[int]:
    """Split ``total`` samples as evenly as possible across ``num_groups`` (parts sum to total).

    A small pure helper the collector can use to set per-map quotas so the dataset stays
    balanced across maps. Kept here next to the split logic since both reason about groups.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be >= 1")
    base = total // num_groups
    rem = total % num_groups
    return [base + (1 if i < rem else 0) for i in range(num_groups)]


def per_map_quota(num_maps: int, sub_target: int) -> int:
    """Ceil-split quota per map for one collection worker (``ceil(sub_target / num_maps)``)."""
    if num_maps <= 0:
        raise ValueError("num_maps must be >= 1")
    return math.ceil(sub_target / num_maps)
