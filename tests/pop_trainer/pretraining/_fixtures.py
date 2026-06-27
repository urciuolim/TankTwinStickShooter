"""Tiny in-temp-dir decode-v1-style shard fixture for the loader / smoke tests.

Writes a worker_*/shard_*.npz layout with a handful of frames across a few maps, so the
map-aware split, recursive index, and end-to-end train step can run on a few-second CPU fixture
without the real dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pop_trainer.core import state as st
from pop_trainer.data import shards


def _make_states(n: int, map_id: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, 52)`` states: random tank kinematics + a few present bullets, map-correlated."""
    s = np.full((n, st.STATE_LEN), st.ABSENT_BULLET_SENTINEL, dtype=np.float32)
    for player in (st.PLAYER_1, st.PLAYER_2):
        base = player * st.PLAYER_STRIDE
        s[:, base + st.POS_X] = rng.uniform(-7, 7, n) + map_id  # map-correlated position
        s[:, base + st.POS_Y] = rng.uniform(-3, 3, n)
        s[:, base + st.VEC_X] = rng.uniform(-1, 1, n)
        s[:, base + st.VEC_Y] = rng.uniform(-1, 1, n)
        ang = rng.uniform(0, 2 * np.pi, n)
        s[:, base + st.AIM_X] = np.cos(ang)
        s[:, base + st.AIM_Y] = np.sin(ang)
    # make P1 slot 0 present in ~half the rows
    idx = list(st.bullet_field_indices(st.PLAYER_1))[0]
    present = rng.random(n) < 0.5
    s[present, idx + st.BULLET_POS_X] = rng.uniform(0, 5, present.sum())
    s[present, idx + st.BULLET_POS_Y] = rng.uniform(0, 5, present.sum())
    s[present, idx + st.BULLET_VEC_X] = rng.uniform(-1, 1, present.sum())
    s[present, idx + st.BULLET_VEC_Y] = rng.uniform(-1, 1, present.sum())
    return s


def make_fixture(
    root: Path,
    *,
    n_maps: int = 4,
    workers: int = 2,
    rows_per_shard: int = 12,
    shards_per_worker: int = 2,
    hw: tuple[int, int] = (360, 640),
    seed: int = 0,
) -> Path:
    """Write a tiny worker_*/shard_*.npz dataset under ``root``; returns ``root``.

    Frames are tiny synthetic RGB grids whose red channel tracks the (clamped) P1 x position, so
    the frame carries a weak signal the decoder can latch onto in the smoke train.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    map_id = 0
    for w in range(workers):
        for sidx in range(shards_per_worker):
            states = _make_states(rows_per_shard, map_id % n_maps, rng)
            h, ww = hw
            frames = np.zeros((rows_per_shard, h, ww, 3), dtype=np.uint8)
            # weak position signal into the frame: red intensity from P1 x.
            p1x = states[:, st.POS_X]
            red = np.clip((p1x + 8) / 16 * 255, 0, 255).astype(np.uint8)
            frames[:, :, :, 0] = red[:, None, None]
            frames[:, :, :, 1] = rng.integers(0, 64, (rows_per_shard, h, ww), dtype=np.uint8)
            map_ids = np.full(rows_per_shard, map_id % n_maps, dtype=np.int32)
            episode_ids = np.full(rows_per_shard, w * 100 + sidx, dtype=np.int32)
            step_idxs = np.arange(rows_per_shard, dtype=np.int32)
            shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs)
            shards.write_shard(root / f"worker_{w}" / f"shard_w{w}_{sidx:04d}.npz", shard)
            map_id += 1
    return root
