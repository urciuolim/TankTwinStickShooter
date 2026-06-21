"""Pixel inverse-renderer PRETRAINING (Stage 2): supervised, plain PyTorch.

Trains a NatureCNN-style encoder to invert the synthetic-pixel renderer: from a
downsampled RGB frame it must recover the underlying simulator state (player
kinematics, bullets, and the static wall layout). The reusable artifact is the
**encoder** (conv stack + FC projection), saved standalone as ``encoder.pt`` for
later pixel-RL. The three supervised heads (player / bullet / wall) exist only to
*force* that information into the shared embedding; they are discarded after
pretraining.

This is OFFLINE supervised learning over the existing ``datasets/pixel_1m`` shards
(read-only). It does NOT touch the RL seam (DriverController / GameController
52-float layout / protocol.py). It is plain PyTorch, NOT Stable-Baselines3.

State layout (52 floats), used to build the targets:
* Player block (always present): P1 ``[0:6]`` and P2 ``[26:32]`` =
  ``[pos_x, pos_y, vel_x, vel_y, aim_x, aim_y]`` -> 12 player floats.
* Bullets: stride 4, 5 slots/player. P1 at indices 6,10,14,18,22 ; P2 at
  32,36,40,44,48. Each slot = ``[pos_x, pos_y, vec_x, vec_y]``. An ABSENT slot is
  all-``-100.0`` (sentinel) -> 10 presence bits + 40 position floats.
Walls: static per map, looked up by ``map_id`` in ``wall_grids.npy`` (10,12,20).

Targets are per-field standardized (player + bullet positions; bullet stats over
PRESENT slots only) computed on the TRAIN split; errors are reported back in WORLD
units. Walls are binary. The split is by episode GROUP ``(worker_id, episode_id)``,
stratified across the 10 maps, to avoid frame-level leakage.

The pure target-encoding / sentinel-masking / normalization helpers are numpy-only
so they stay unit-testable without torch. Torch is imported lazily inside the
trainer entry points so the pure helpers (and their tests) do not require it.

Run::

    uv run python -m tank_twin.pretrain_pixels --subset 4000 --epochs 3 \
        --out runs/pixel_stage2_smoke
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BULLET_BASE_INDICES",
    "PLAYER_INDICES",
    "SENTINEL",
    "NormStats",
    "bullet_position_indices",
    "decode_targets",
    "denormalize",
    "encode_bullet_targets",
    "encode_player_targets",
    "fit_norm_stats",
    "main",
    "normalize",
    "parse_input_res",
    "stratified_group_split",
    "wall_target_for",
]

# --- Repo / dataset locations (repo-relative defaults) -----------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "datasets" / "pixel_1m"

# --- State-layout constants (EXACT; see module docstring) --------------------
SENTINEL = -100.0
# Player floats: P1[0:6] then P2[26:32] -> 12 values.
PLAYER_INDICES: tuple[int, ...] = tuple(range(0, 6)) + tuple(range(26, 32))
# Bullet slot base indices: 5 P1 then 5 P2; each slot spans base..base+4.
BULLET_BASE_INDICES: tuple[int, ...] = (6, 10, 14, 18, 22, 32, 36, 40, 44, 48)
N_PLAYER = 12
N_BULLET_SLOTS = 10
N_BULLET_POS = N_BULLET_SLOTS * 4  # 40
WALL_H, WALL_W = 12, 20
N_WALL = WALL_H * WALL_W  # 240

_SHARD_RE = re.compile(r"shard_w(\d+)_(\d+)\.npz$")


# =============================================================================
# Pure helpers (numpy-only; unit-tested without torch)
# =============================================================================
def parse_input_res(spec: str) -> tuple[int, int]:
    """Parse a ``WxH`` resolution string into ``(W, H)`` ints (e.g. ``160x90``)."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--input-res must look like WxH (e.g. 160x90), got {spec!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"--input-res must be positive, got {spec!r}")
    return w, h


def worker_id_from_shard(name: str) -> int:
    """Extract the integer worker id encoded in a shard filename ``shard_w{w}_{idx}.npz``."""
    m = _SHARD_RE.search(name)
    if not m:
        raise ValueError(f"not a pixel shard filename: {name!r}")
    return int(m.group(1))


def bullet_position_indices() -> list[int]:
    """The 40 state indices holding bullet position floats, slot-major (4 per slot)."""
    out: list[int] = []
    for base in BULLET_BASE_INDICES:
        out.extend(range(base, base + 4))
    return out


def encode_player_targets(states: np.ndarray) -> np.ndarray:
    """Slice the 12 always-present player floats (P1[0:6] + P2[26:32]) from states.

    ``states`` is ``(N, 52)``; returns ``(N, 12)`` float32.
    """
    states = np.asarray(states)
    return states[:, list(PLAYER_INDICES)].astype(np.float32, copy=False)


def encode_bullet_targets(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build bullet presence + position targets from states (sentinel-aware).

    Returns ``(presence, positions)``:
    * ``presence`` ``(N, 10)`` float32 in {0,1}: a slot is PRESENT iff its 4-float
      block is not the all-``-100`` sentinel (any non-sentinel value counts present).
    * ``positions`` ``(N, 40)`` float32: the raw bullet position floats, slot-major.
      Absent-slot positions remain the ``-100`` sentinel here; callers MUST mask the
      position loss to present slots (the sentinel must never be a regression target).
    """
    states = np.asarray(states)
    pos = states[:, bullet_position_indices()].astype(np.float32, copy=False)  # (N, 40)
    # Per-slot presence: reshape to (N, 10, 4); a slot is absent iff ALL 4 == sentinel.
    per_slot = pos.reshape(states.shape[0], N_BULLET_SLOTS, 4)
    absent = np.all(per_slot == SENTINEL, axis=2)  # (N, 10) bool
    presence = (~absent).astype(np.float32)
    return presence, pos


def bullet_position_mask(presence: np.ndarray) -> np.ndarray:
    """Expand a ``(N, 10)`` presence array to a ``(N, 40)`` per-float mask (4 per slot)."""
    presence = np.asarray(presence)
    return np.repeat(presence, 4, axis=1).astype(np.float32)


def wall_target_for(map_ids: np.ndarray, wall_grids: np.ndarray) -> np.ndarray:
    """Look up the ``(12,20)`` wall grid for each map_id -> ``(N, 12, 20)`` float32."""
    map_ids = np.asarray(map_ids).astype(np.int64)
    return wall_grids[map_ids].astype(np.float32, copy=False)


@dataclass(frozen=True)
class NormStats:
    """Per-field standardization stats for the player + bullet-position targets.

    Bullet stats are computed over PRESENT slots only (sentinels excluded). Persisted
    alongside the model so inference can de-standardize predictions to world units.
    """

    player_mean: np.ndarray  # (12,)
    player_std: np.ndarray  # (12,)
    bullet_mean: np.ndarray  # (40,)
    bullet_std: np.ndarray  # (40,)

    def to_json(self) -> dict:
        return {
            "convention": (
                "Per-field standardize player[12] and bullet_pos[40]. Bullet stats over "
                "PRESENT slots only (state==-100 sentinel excluded). z = (x-mean)/std; "
                "x = z*std + mean. Walls binary (no normalization)."
            ),
            "player_indices": list(PLAYER_INDICES),
            "bullet_position_indices": bullet_position_indices(),
            "player_mean": self.player_mean.tolist(),
            "player_std": self.player_std.tolist(),
            "bullet_mean": self.bullet_mean.tolist(),
            "bullet_std": self.bullet_std.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> NormStats:
        return cls(
            player_mean=np.asarray(d["player_mean"], dtype=np.float32),
            player_std=np.asarray(d["player_std"], dtype=np.float32),
            bullet_mean=np.asarray(d["bullet_mean"], dtype=np.float32),
            bullet_std=np.asarray(d["bullet_std"], dtype=np.float32),
        )


def fit_norm_stats(states: np.ndarray, *, eps: float = 1e-6) -> NormStats:
    """Fit per-field mean/std for player + bullet-position targets from raw states.

    Player stats use all rows (player block is always present). Bullet-position stats
    are computed per-field over PRESENT slots only: absent slots (sentinel ``-100``)
    are masked out so the sentinel never skews the mean/std. A field with no present
    samples falls back to mean 0 / std 1. ``std`` is floored at ``eps`` to avoid
    divide-by-zero on constant fields.
    """
    player = encode_player_targets(states)  # (N, 12)
    player_mean = player.mean(axis=0).astype(np.float32)
    player_std = np.maximum(player.std(axis=0), eps).astype(np.float32)

    presence, pos = encode_bullet_targets(states)  # (N,10), (N,40)
    mask = bullet_position_mask(presence).astype(bool)  # (N, 40)
    bullet_mean = np.zeros(N_BULLET_POS, dtype=np.float32)
    bullet_std = np.ones(N_BULLET_POS, dtype=np.float32)
    for j in range(N_BULLET_POS):
        col = pos[mask[:, j], j]
        if col.size > 0:
            bullet_mean[j] = float(col.mean())
            bullet_std[j] = float(max(col.std(), eps))
    return NormStats(
        player_mean=player_mean,
        player_std=player_std,
        bullet_mean=bullet_mean,
        bullet_std=bullet_std,
    )


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize ``(x - mean) / std`` (per-field; broadcasts over the leading axis)."""
    return (np.asarray(x, dtype=np.float32) - mean) / std


def denormalize(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize`: ``z * std + mean`` (recovers world units)."""
    return np.asarray(z, dtype=np.float32) * std + mean


def decode_targets(
    states: np.ndarray,
    map_ids: np.ndarray,
    wall_grids: np.ndarray,
    stats: NormStats,
) -> dict[str, np.ndarray]:
    """Build the full normalized target dict for a batch of states.

    Returns keys:
    * ``player`` ``(N,12)`` normalized player floats.
    * ``bullet_presence`` ``(N,10)`` {0,1}.
    * ``bullet_pos`` ``(N,40)`` normalized bullet positions (absent slots get 0 after
      normalization is irrelevant — they are masked out by ``bullet_pos_mask``).
    * ``bullet_pos_mask`` ``(N,40)`` {0,1} per-float mask (present slots only).
    * ``wall`` ``(N,12,20)`` binary wall target.
    """
    player = normalize(encode_player_targets(states), stats.player_mean, stats.player_std)
    presence, raw_pos = encode_bullet_targets(states)
    mask = bullet_position_mask(presence)
    pos = normalize(raw_pos, stats.bullet_mean, stats.bullet_std)
    # Zero out absent-slot positions post-normalize so they are inert and never -100-ish.
    pos = pos * mask
    return {
        "player": player.astype(np.float32),
        "bullet_presence": presence.astype(np.float32),
        "bullet_pos": pos.astype(np.float32),
        "bullet_pos_mask": mask.astype(np.float32),
        "wall": wall_target_for(map_ids, wall_grids).astype(np.float32),
    }


def stratified_group_split(
    group_keys: Sequence[tuple[int, int]],
    map_of_group: dict[tuple[int, int], int],
    *,
    val_frac: float,
    seed: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Split episode GROUPS into (train, val) sets, stratified by map.

    ``group_keys`` is the set/sequence of ``(worker_id, episode_id)`` groups;
    ``map_of_group`` maps each group to its map_id (a group lives on one map). Within
    each map, ``round(val_frac * n_groups)`` groups are held out (at least 1 if the
    map has >=2 groups and ``val_frac>0``). Deterministic given ``seed``.

    Returns ``(train_groups, val_groups)`` as sets of group keys.
    """
    rng = np.random.default_rng(seed)
    by_map: dict[int, list[tuple[int, int]]] = {}
    for g in dict.fromkeys(group_keys):  # de-dup, preserve order
        by_map.setdefault(map_of_group[g], []).append(g)

    train: set[tuple[int, int]] = set()
    val: set[tuple[int, int]] = set()
    for _map_id, groups in sorted(by_map.items()):
        groups_sorted = sorted(groups)
        perm = rng.permutation(len(groups_sorted))
        n_val = int(round(val_frac * len(groups_sorted)))
        if val_frac > 0 and len(groups_sorted) >= 2:
            n_val = max(1, min(n_val, len(groups_sorted) - 1))
        for rank, idx in enumerate(perm):
            (val if rank < n_val else train).add(groups_sorted[idx])
    return train, val


# =============================================================================
# Data loading / downsample (numpy; torch used only in the trainer)
# =============================================================================
def downsample_frames_np(frames: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Area-style downsample ``(N,360,640,3)`` uint8 -> ``(N,out_h,out_w,3)`` uint8.

    Pure-numpy block-mean used by the on-the-fly SMOKE streamer and the cache builder
    when an exact integer block factor exists (640/out_w and 360/out_h integral). Falls
    back to strided nearest-neighbour subsampling otherwise. The full run uses torch
    GPU interpolate for speed; this keeps the smoke and cache torch-free.
    """
    frames = np.asarray(frames)
    n, h, w, c = frames.shape
    if h % out_h == 0 and w % out_w == 0:
        fy, fx = h // out_h, w // out_w
        # Block-mean: reshape into blocks and average (area downsample).
        blocks = frames.reshape(n, out_h, fy, out_w, fx, c)
        return blocks.mean(axis=(2, 4)).round().astype(np.uint8)
    ys = (np.arange(out_h) * (h / out_h)).astype(np.int64)
    xs = (np.arange(out_w) * (w / out_w)).astype(np.int64)
    return frames[:, ys][:, :, xs].astype(np.uint8)


def list_shards(data_dir: Path) -> list[Path]:
    """All pixel shards in ``data_dir``, sorted by (worker, index)."""
    shards = sorted(
        data_dir.glob("shard_w*_*.npz"),
        key=lambda p: tuple(int(g) for g in _SHARD_RE.search(p.name).groups()),
    )
    if not shards:
        raise FileNotFoundError(f"no shard_w*_*.npz under {data_dir}")
    return shards


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    # Torch-dependent work lives in _run so `main`/helpers stay importable torch-free
    # for the pure-logic tests.
    from tank_twin._pretrain_pixels_train import run_training

    return run_training(args)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m tank_twin.pretrain_pixels",
        description="Pixel inverse-renderer pretraining (Stage 2; plain PyTorch).",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="pixel_1m dataset dir")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="memmap cache dir from prepare_pixel_cache.py (full run). Omit to STREAM shards.",
    )
    p.add_argument("--input-res", type=str, default="160x90", help="downsample target WxH")
    p.add_argument("--embedding-dim", type=int, default=512, help="encoder embedding width")
    p.add_argument("--lambda-player", type=float, default=1.0)
    p.add_argument("--lambda-presence", type=float, default=1.0)
    p.add_argument("--lambda-bullet-pos", type=float, default=1.0)
    p.add_argument("--lambda-wall", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", type=Path, required=True, help="output dir for checkpoints/stats")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of GROUPS held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="limit total pairs (SMOKE; streams a subset spanning maps)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="auto picks cuda if available",
    )
    p.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (0 = main process)"
    )
    p.add_argument("--log-every", type=int, default=20, help="log a train batch line every N steps")
    args = p.parse_args(argv)
    # Validate input-res eagerly so a bad value fails before any heavy import.
    args.input_wh = parse_input_res(args.input_res)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
