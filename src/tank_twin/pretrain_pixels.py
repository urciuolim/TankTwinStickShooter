"""Pixel inverse-renderer PRETRAINING (Stage 2): supervised, plain PyTorch.

Trains a NatureCNN-style encoder to invert the synthetic-pixel renderer: from a
downsampled RGB frame it must recover the underlying simulator state (player
kinematics, bullets, and the static wall layout). The reusable artifact is the
**encoder** (conv stack + FC projection), saved standalone as ``encoder.pt`` for
later pixel-RL. The supervised heads (player / bullet-presence / bullet-position /
bullet-direction / wall) exist only to *force* that information into the shared
embedding; they are discarded after pretraining.

This is OFFLINE supervised learning over the existing ``datasets/pixel_1m`` shards
(read-only). It does NOT touch the RL seam (DriverController / GameController
52-float layout / protocol.py). It is plain PyTorch, NOT Stable-Baselines3.

State layout (52 floats), used to build the targets:
* Player block (always present): P1 ``[0:6]`` and P2 ``[26:32]`` =
  ``[pos_x, pos_y, vel_x, vel_y, aim_x, aim_y]`` -> 12 player floats. In the
  ``PLAYER_INDICES`` ordering the 12-vector is
  ``[P1 pos_x,pos_y,vel_x,vel_y,aim_x,aim_y, P2 ...]`` so the sub-groups are
  position = vec idx {0,1,6,7}, velocity = {2,3,8,9}, aim = {4,5,10,11}.
* Bullets: stride 4, 5 slots/player. P1 at indices 6,10,14,18,22 ; P2 at
  32,36,40,44,48. Each slot = ``[pos_x, pos_y, vec_x, vec_y]``. An ABSENT slot is
  all-``-100.0`` (sentinel) -> 10 presence bits, 20 position floats (offsets 0,1
  within each slot), and 20 direction floats (offsets 2,3).
Walls: static per map, looked up by ``map_id`` in ``wall_grids.npy`` (10,12,20).

Objectives are configurable (see :class:`ObjectiveConfig`): a disabled group is
FULLY dropped (no head output, no loss term, eval reports ``"N/A"``); the shared
ENCODER is independent of the objective config, so ``encoder.pt`` is byte-stable.

The data split is 3-way and map-aware: 2 whole maps are held out for final
out-of-distribution VALIDATION; the other 8 maps are split 90/10 (TRAIN / in-
distribution EVAL) by episode GROUP ``(worker_id, episode_id)`` so there is no
frame-level leakage.

The pure target-encoding / sentinel-masking / normalization / split / mask /
metric helpers are numpy-only so they stay unit-testable without torch. Torch is
imported lazily inside the trainer entry points so the pure helpers (and their
tests) do not require it.

Run (SMOKE)::

    uv run python -m tank_twin.pretrain_pixels --subset 6000 --epochs 2 \
        --out runs/pixel_stage2_smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BULLET_BASE_INDICES",
    "OBJECTIVE_GROUPS",
    "PLAYER_INDICES",
    "PLAYER_SUBGROUP_VEC_INDICES",
    "SENTINEL",
    "MapAwareSplit",
    "NormStats",
    "ObjectiveConfig",
    "binary_f1",
    "bullet_direction_indices",
    "bullet_position_indices",
    "decode_targets",
    "denormalize",
    "encode_bullet_targets",
    "encode_player_targets",
    "fit_norm_stats",
    "interior_wall_mask",
    "iou_score",
    "main",
    "map_aware_split",
    "normalize",
    "parse_input_res",
    "player_head_size",
    "player_head_vec_indices",
    "resolve_map_ids",
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
N_BULLET_FLOATS = N_BULLET_SLOTS * 4  # 40 (pos + direction interleaved)
N_BULLET_POS = N_BULLET_SLOTS * 2  # 20 position floats
N_BULLET_DIR = N_BULLET_SLOTS * 2  # 20 direction floats
WALL_H, WALL_W = 12, 20
N_WALL = WALL_H * WALL_W  # 240

# Player sub-group -> indices into the 12-vector PLAYER_INDICES ordering.
PLAYER_SUBGROUP_VEC_INDICES: dict[str, tuple[int, ...]] = {
    "player_position": (0, 1, 6, 7),
    "player_velocity": (2, 3, 8, 9),
    "player_aim": (4, 5, 10, 11),
}

# --- Objective groups --------------------------------------------------------
OBJECTIVE_GROUPS: tuple[str, ...] = (
    "player_position",
    "player_velocity",
    "player_aim",
    "bullet_presence",
    "bullet_position",
    "bullet_direction",
    "walls",
)
_PLAYER_GROUPS: tuple[str, ...] = ("player_position", "player_velocity", "player_aim")
# DEFAULT = all 7 EXCEPT player_velocity (velocity disabled by default; aim stays on).
DEFAULT_OBJECTIVES: tuple[str, ...] = tuple(g for g in OBJECTIVE_GROUPS if g != "player_velocity")

# --- Wall-grid interior region (exclude the constant 2-deep border) ----------
# wall_grids orientation: row = maxY - y, col = x - minX, minX=-10, maxY=5.
# Interior floor = x in [-8,7] AND y in [-4,3] -> rows 2..9, cols 2..17.
WALL_INTERIOR_ROWS = (2, 9)  # inclusive
WALL_INTERIOR_COLS = (2, 17)  # inclusive
N_WALL_INTERIOR = 8 * 16  # 128

_SHARD_RE = re.compile(r"shard_w(\d+)_(\d+)\.npz$")


# =============================================================================
# Objective configuration (pure)
# =============================================================================
@dataclass(frozen=True)
class ObjectiveConfig:
    """Which of the 7 named objective groups are trained.

    A disabled group is FULLY dropped: its head is not built, it is not in the loss,
    and eval reports it as ``"N/A"``. Head sizes are derived from the enabled groups.
    The shared ENCODER is independent of this config (only heads change).
    """

    enabled: frozenset[str]

    @classmethod
    def from_set(cls, names: Sequence[str]) -> ObjectiveConfig:
        bad = [n for n in names if n not in OBJECTIVE_GROUPS]
        if bad:
            raise ValueError(
                f"unknown objective group(s): {bad}. Valid groups: {list(OBJECTIVE_GROUPS)}"
            )
        return cls(enabled=frozenset(names))

    @classmethod
    def default(cls) -> ObjectiveConfig:
        return cls.from_set(DEFAULT_OBJECTIVES)

    def is_on(self, group: str) -> bool:
        if group not in OBJECTIVE_GROUPS:
            raise ValueError(f"unknown objective group: {group!r}")
        return group in self.enabled

    def enabled_player_subgroups(self) -> tuple[str, ...]:
        """Enabled player sub-groups in canonical order (position, velocity, aim)."""
        return tuple(g for g in _PLAYER_GROUPS if g in self.enabled)

    def any_player(self) -> bool:
        return bool(self.enabled & set(_PLAYER_GROUPS))

    def ordered(self) -> tuple[str, ...]:
        """Enabled groups in canonical OBJECTIVE_GROUPS order."""
        return tuple(g for g in OBJECTIVE_GROUPS if g in self.enabled)

    def to_json(self) -> dict:
        return {"enabled": list(self.ordered()), "all_groups": list(OBJECTIVE_GROUPS)}


def parse_objectives(objectives: str | None, disable: str | None) -> ObjectiveConfig:
    """Resolve the ``--objectives`` / ``--disable`` flags into an :class:`ObjectiveConfig`.

    Exactly one mechanism may be used: ``--objectives a,b,c`` is an explicit allowlist;
    ``--disable g1,g2`` subtracts from the default set. With neither, the default config
    (all groups except ``player_velocity``) is used. Unknown group names raise ValueError.
    """
    if objectives is not None and disable is not None:
        raise ValueError("pass only one of --objectives / --disable, not both")
    if objectives is not None:
        names = [s.strip() for s in objectives.split(",") if s.strip()]
        if not names:
            raise ValueError("--objectives resolved to an empty set")
        return ObjectiveConfig.from_set(names)
    if disable is not None:
        drop = [s.strip() for s in disable.split(",") if s.strip()]
        bad = [n for n in drop if n not in OBJECTIVE_GROUPS]
        if bad:
            raise ValueError(
                f"unknown objective group(s) in --disable: {bad}. "
                f"Valid groups: {list(OBJECTIVE_GROUPS)}"
            )
        keep = [g for g in DEFAULT_OBJECTIVES if g not in drop]
        if not keep:
            raise ValueError("--disable removed every group; nothing left to train")
        return ObjectiveConfig.from_set(keep)
    return ObjectiveConfig.default()


def player_head_vec_indices(cfg: ObjectiveConfig) -> list[int]:
    """Indices into the 12-vector that the player head must output (enabled sub-groups).

    Order follows the enabled sub-groups (position, velocity, aim), and within each
    sub-group the canonical (P1 xy, P2 xy) order. Empty if no player group is enabled.
    """
    out: list[int] = []
    for sub in cfg.enabled_player_subgroups():
        out.extend(PLAYER_SUBGROUP_VEC_INDICES[sub])
    return out


def player_head_size(cfg: ObjectiveConfig) -> int:
    """Number of float outputs in the player head (4 per enabled player sub-group)."""
    return len(player_head_vec_indices(cfg))


def head_sizes(cfg: ObjectiveConfig) -> dict[str, int]:
    """Output width of each ENABLED head. Disabled heads are absent from the dict.

    Keys: ``player`` (4 * enabled player sub-groups), ``bullet_presence`` (10 logits),
    ``bullet_position`` (20), ``bullet_direction`` (20), ``wall`` (240 logits).
    """
    sizes: dict[str, int] = {}
    if cfg.any_player():
        sizes["player"] = player_head_size(cfg)
    if cfg.is_on("bullet_presence"):
        sizes["bullet_presence"] = N_BULLET_SLOTS
    if cfg.is_on("bullet_position"):
        sizes["bullet_position"] = N_BULLET_POS
    if cfg.is_on("bullet_direction"):
        sizes["bullet_direction"] = N_BULLET_DIR
    if cfg.is_on("walls"):
        sizes["wall"] = N_WALL
    return sizes


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


def _bullet_offset_indices(offsets: Sequence[int]) -> list[int]:
    """State indices for the given within-slot offsets, slot-major across all 10 slots."""
    out: list[int] = []
    for base in BULLET_BASE_INDICES:
        out.extend(base + o for o in offsets)
    return out


def bullet_all_indices() -> list[int]:
    """All 40 bullet state indices, slot-major (4 per slot)."""
    return _bullet_offset_indices((0, 1, 2, 3))


def bullet_position_indices() -> list[int]:
    """The 20 state indices holding bullet POSITION floats (offsets 0,1 per slot)."""
    return _bullet_offset_indices((0, 1))


def bullet_direction_indices() -> list[int]:
    """The 20 state indices holding bullet DIRECTION (vec) floats (offsets 2,3 per slot)."""
    return _bullet_offset_indices((2, 3))


def encode_player_targets(states: np.ndarray) -> np.ndarray:
    """Slice the 12 always-present player floats (P1[0:6] + P2[26:32]) from states.

    ``states`` is ``(N, 52)``; returns ``(N, 12)`` float32.
    """
    states = np.asarray(states)
    return states[:, list(PLAYER_INDICES)].astype(np.float32, copy=False)


def encode_bullet_targets(
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build bullet presence + position + direction targets (sentinel-aware).

    Returns ``(presence, positions, directions)``:
    * ``presence`` ``(N, 10)`` float32 in {0,1}: a slot is PRESENT iff its 4-float
      block is not the all-``-100`` sentinel (any non-sentinel value counts present).
    * ``positions`` ``(N, 20)`` float32: raw bullet position floats (offsets 0,1),
      slot-major. Absent-slot positions remain ``-100`` here; callers MUST mask.
    * ``directions`` ``(N, 20)`` float32: raw bullet direction (vec) floats (offsets
      2,3), slot-major. Absent-slot directions remain ``-100`` here; callers MUST mask.
    """
    states = np.asarray(states)
    n = states.shape[0]
    full = states[:, bullet_all_indices()].astype(np.float32, copy=False)  # (N, 40)
    per_slot = full.reshape(n, N_BULLET_SLOTS, 4)
    absent = np.all(per_slot == SENTINEL, axis=2)  # (N, 10) bool
    presence = (~absent).astype(np.float32)
    pos = states[:, bullet_position_indices()].astype(np.float32, copy=False)  # (N, 20)
    direction = states[:, bullet_direction_indices()].astype(np.float32, copy=False)  # (N, 20)
    return presence, pos, direction


def bullet_slot_mask(presence: np.ndarray) -> np.ndarray:
    """Expand a ``(N, 10)`` presence array to a ``(N, 20)`` per-float mask (2 per slot)."""
    presence = np.asarray(presence)
    return np.repeat(presence, 2, axis=1).astype(np.float32)


def wall_target_for(map_ids: np.ndarray, wall_grids: np.ndarray) -> np.ndarray:
    """Look up the ``(12,20)`` wall grid for each map_id -> ``(N, 12, 20)`` float32."""
    map_ids = np.asarray(map_ids).astype(np.int64)
    return wall_grids[map_ids].astype(np.float32, copy=False)


def interior_wall_mask() -> np.ndarray:
    """Boolean ``(12, 20)`` mask selecting the interior floor cells (no constant border).

    Interior = rows 2..9 (8 rows) x cols 2..17 (16 cols) = exactly 128 cells, per the
    wall_grids orientation (row = maxY - y, col = x - minX). The constant 2-deep border
    is excluded so wall metrics measure the per-map interior layout, not free wins.
    """
    mask = np.zeros((WALL_H, WALL_W), dtype=bool)
    r0, r1 = WALL_INTERIOR_ROWS
    c0, c1 = WALL_INTERIOR_COLS
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


@dataclass(frozen=True)
class NormStats:
    """Per-field standardization stats for the player + bullet-position + direction targets.

    Bullet stats are computed over PRESENT slots only (sentinels excluded). Persisted
    alongside the model so inference can de-standardize predictions to world units.
    Player stats cover the full 12-vector; bullet stats cover 20 pos + 20 dir floats.
    """

    player_mean: np.ndarray  # (12,)
    player_std: np.ndarray  # (12,)
    bullet_pos_mean: np.ndarray  # (20,)
    bullet_pos_std: np.ndarray  # (20,)
    bullet_dir_mean: np.ndarray  # (20,)
    bullet_dir_std: np.ndarray  # (20,)

    def to_json(self) -> dict:
        return {
            "convention": (
                "Per-field standardize player[12], bullet_pos[20], bullet_dir[20]. "
                "Bullet stats over PRESENT slots only (state==-100 sentinel excluded). "
                "z = (x-mean)/std; x = z*std + mean. Walls binary (no normalization)."
            ),
            "player_indices": list(PLAYER_INDICES),
            "bullet_position_indices": bullet_position_indices(),
            "bullet_direction_indices": bullet_direction_indices(),
            "player_mean": self.player_mean.tolist(),
            "player_std": self.player_std.tolist(),
            "bullet_pos_mean": self.bullet_pos_mean.tolist(),
            "bullet_pos_std": self.bullet_pos_std.tolist(),
            "bullet_dir_mean": self.bullet_dir_mean.tolist(),
            "bullet_dir_std": self.bullet_dir_std.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> NormStats:
        return cls(
            player_mean=np.asarray(d["player_mean"], dtype=np.float32),
            player_std=np.asarray(d["player_std"], dtype=np.float32),
            bullet_pos_mean=np.asarray(d["bullet_pos_mean"], dtype=np.float32),
            bullet_pos_std=np.asarray(d["bullet_pos_std"], dtype=np.float32),
            bullet_dir_mean=np.asarray(d["bullet_dir_mean"], dtype=np.float32),
            bullet_dir_std=np.asarray(d["bullet_dir_std"], dtype=np.float32),
        )


def _fit_masked_stats(
    values: np.ndarray, present_mask: np.ndarray, *, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-field mean/std over present samples; fall back to 0/1 where no samples exist."""
    n_fields = values.shape[1]
    mean = np.zeros(n_fields, dtype=np.float32)
    std = np.ones(n_fields, dtype=np.float32)
    for j in range(n_fields):
        col = values[present_mask[:, j], j]
        if col.size > 0:
            mean[j] = float(col.mean())
            std[j] = float(max(col.std(), eps))
    return mean, std


def fit_norm_stats(states: np.ndarray, *, eps: float = 1e-6) -> NormStats:
    """Fit per-field mean/std for player + bullet position + bullet direction targets.

    Player stats use all rows (player block is always present). Bullet stats are computed
    per-field over PRESENT slots only: absent slots (sentinel ``-100``) are masked out so
    the sentinel never skews mean/std. A field with no present samples falls back to mean
    0 / std 1. ``std`` is floored at ``eps`` to avoid divide-by-zero on constant fields.
    """
    player = encode_player_targets(states)  # (N, 12)
    player_mean = player.mean(axis=0).astype(np.float32)
    player_std = np.maximum(player.std(axis=0), eps).astype(np.float32)

    presence, pos, direction = encode_bullet_targets(states)  # (N,10),(N,20),(N,20)
    slot_mask = bullet_slot_mask(presence).astype(bool)  # (N, 20)
    pos_mean, pos_std = _fit_masked_stats(pos, slot_mask, eps=eps)
    dir_mean, dir_std = _fit_masked_stats(direction, slot_mask, eps=eps)
    return NormStats(
        player_mean=player_mean,
        player_std=player_std,
        bullet_pos_mean=pos_mean,
        bullet_pos_std=pos_std,
        bullet_dir_mean=dir_mean,
        bullet_dir_std=dir_std,
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
    """Build the full normalized target dict for a batch of states (all 7 groups).

    Returns keys (the trainer drops disabled ones when building the loss):
    * ``player`` ``(N,12)`` normalized player floats (all sub-groups; the model head
      selects the enabled sub-set via :func:`player_head_vec_indices`).
    * ``bullet_presence`` ``(N,10)`` {0,1}.
    * ``bullet_position`` ``(N,20)`` normalized bullet positions (absent slots zeroed).
    * ``bullet_direction`` ``(N,20)`` normalized bullet directions (absent slots zeroed).
    * ``bullet_slot_mask`` ``(N,20)`` {0,1} per-float mask (present slots only).
    * ``wall`` ``(N,12,20)`` binary wall target.
    """
    player = normalize(encode_player_targets(states), stats.player_mean, stats.player_std)
    presence, raw_pos, raw_dir = encode_bullet_targets(states)
    mask = bullet_slot_mask(presence)
    pos = normalize(raw_pos, stats.bullet_pos_mean, stats.bullet_pos_std) * mask
    direction = normalize(raw_dir, stats.bullet_dir_mean, stats.bullet_dir_std) * mask
    return {
        "player": player.astype(np.float32),
        "bullet_presence": presence.astype(np.float32),
        "bullet_position": pos.astype(np.float32),
        "bullet_direction": direction.astype(np.float32),
        "bullet_slot_mask": mask.astype(np.float32),
        "wall": wall_target_for(map_ids, wall_grids).astype(np.float32),
    }


# =============================================================================
# Map id resolution + 3-way map-aware split (pure)
# =============================================================================
def resolve_map_ids(names: Sequence[str], name_to_id: dict[str, int]) -> list[int]:
    """Resolve map NAMES to integer ids via the canonical name->id mapping.

    Raises ValueError on any unknown name (the IDs come from wall_grids_meta / the
    canonical resolver; do NOT hardcode them).
    """
    bad = [n for n in names if n not in name_to_id]
    if bad:
        raise ValueError(f"unknown map name(s): {bad}. Known maps: {sorted(name_to_id)}")
    return [name_to_id[n] for n in names]


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


@dataclass(frozen=True)
class MapAwareSplit:
    """A reproducible 3-way map-aware partition over episode groups.

    * ``train`` / ``eval`` groups live on the 8 training maps (90/10 stratified split).
    * ``validation`` groups live on the 2 held-out maps (out-of-distribution).
    * ``holdout_map_ids`` are the 2 validation maps; ``train_map_ids`` the other 8.
    The three group sets are mutually disjoint (no frame leakage across them).
    """

    train: frozenset[tuple[int, int]]
    eval: frozenset[tuple[int, int]]
    validation: frozenset[tuple[int, int]]
    holdout_map_ids: tuple[int, ...]
    train_map_ids: tuple[int, ...]

    def to_json(self) -> dict:
        return {
            "holdout_map_ids": list(self.holdout_map_ids),
            "train_map_ids": list(self.train_map_ids),
            "train_groups": [list(g) for g in sorted(self.train)],
            "eval_groups": [list(g) for g in sorted(self.eval)],
            "validation_groups": [list(g) for g in sorted(self.validation)],
        }


def map_aware_split(
    group_keys: Sequence[tuple[int, int]],
    map_of_group: dict[tuple[int, int], int],
    holdout_map_ids: Sequence[int],
    *,
    eval_frac: float,
    seed: int,
) -> MapAwareSplit:
    """3-way map-aware split: hold out whole maps, then 90/10 the rest by group.

    ``holdout_map_ids`` (the validation maps) are excluded from train AND eval entirely;
    every group on them becomes VALIDATION. The remaining maps' groups are split
    train/eval via :func:`stratified_group_split` (``eval_frac`` held out per map).
    Deterministic given ``seed``. No group appears in more than one of the three sets.
    """
    holdout = set(int(m) for m in holdout_map_ids)
    train_keys: list[tuple[int, int]] = []
    val_keys: list[tuple[int, int]] = []
    for g in dict.fromkeys(group_keys):
        (val_keys if map_of_group[g] in holdout else train_keys).append(g)

    sub_map_of_group = {g: map_of_group[g] for g in train_keys}
    train_groups, eval_groups = stratified_group_split(
        train_keys, sub_map_of_group, val_frac=eval_frac, seed=seed
    )
    train_map_ids = tuple(sorted({map_of_group[g] for g in train_keys}))
    return MapAwareSplit(
        train=frozenset(train_groups),
        eval=frozenset(eval_groups),
        validation=frozenset(val_keys),
        holdout_map_ids=tuple(sorted(holdout)),
        train_map_ids=train_map_ids,
    )


# =============================================================================
# Metric helpers (pure; numpy)
# =============================================================================
def binary_f1(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 for binary {0,1} arrays (flattened). Returns a dict.

    TP/FP/FN are summed over all elements. Precision/recall default to 0.0 with no
    positive predictions / targets; F1 is 0.0 when precision+recall is 0.
    """
    pred = (np.asarray(pred) > 0.5).astype(np.int64).ravel()
    target = (np.asarray(target) > 0.5).astype(np.int64).ravel()
    tp = int(np.sum((pred == 1) & (target == 1)))
    fp = int(np.sum((pred == 1) & (target == 0)))
    fn = int(np.sum((pred == 0) & (target == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection-over-union for binary {0,1} arrays (flattened). 1.0 when both empty."""
    pred = (np.asarray(pred) > 0.5).astype(np.int64).ravel()
    target = (np.asarray(target) > 0.5).astype(np.int64).ravel()
    inter = int(np.sum((pred == 1) & (target == 1)))
    union = int(np.sum((pred == 1) | (target == 1)))
    if union == 0:
        return 1.0
    return inter / union


def normalized_mse(pred_norm: np.ndarray, target_norm: np.ndarray) -> float:
    """Mean squared error in NORMALIZED space (~0 learned, ~1 = no better than the mean)."""
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    target_norm = np.asarray(target_norm, dtype=np.float64)
    if pred_norm.size == 0:
        return 0.0
    return float(np.mean((pred_norm - target_norm) ** 2))


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


def load_map_name_to_id(data_dir: Path) -> dict[str, int]:
    """Canonical map name -> id mapping from ``wall_grids_meta.json`` in the data dir.

    Falls back to the live resolver (``collect_pixels._resolve_maps('all')``) if the
    meta file is missing. Never hardcodes integers.
    """
    meta_path = Path(data_dir) / "wall_grids_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {m["name"]: int(m["id"]) for m in meta["maps"]}
    # Fallback: derive from the canonical resolver (sorted map-config stems).
    from tank_twin.collect_pixels import _resolve_maps

    _arena_paths, names, _config_paths = _resolve_maps("all")
    return {name: i for i, name in enumerate(names)}


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
    # --- Objective selection (one mechanism; default = all except player_velocity) ---
    p.add_argument(
        "--objectives",
        type=str,
        default=None,
        help=(
            "comma-list of objective groups to TRAIN (allowlist). Valid: "
            + ",".join(OBJECTIVE_GROUPS)
            + ". Mutually exclusive with --disable."
        ),
    )
    p.add_argument(
        "--disable",
        type=str,
        default=None,
        help="comma-list of objective groups to DROP from the default set.",
    )
    p.add_argument("--lambda-player", type=float, default=1.0)
    p.add_argument("--lambda-presence", type=float, default=1.0)
    p.add_argument("--lambda-bullet-pos", type=float, default=1.0)
    p.add_argument("--lambda-bullet-dir", type=float, default=1.0)
    p.add_argument("--lambda-wall", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="BASE output path; a unique run dir _{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM} is appended",
    )
    p.add_argument(
        "--eval-frac",
        type=float,
        default=0.1,
        help="fraction of GROUPS (within the 8 training maps) held out for in-distribution EVAL",
    )
    p.add_argument(
        "--holdout-maps",
        type=str,
        default="diagonal_pillars,ring_fragments",
        help="comma-list of map NAMES held out as out-of-distribution VALIDATION (2 maps)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="limit total pairs (SMOKE; streams a map-balanced subset spanning all 10 maps)",
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
    # Resolve objectives eagerly (fails fast on unknown group / both flags).
    args.objective_config = parse_objectives(args.objectives, args.disable)
    # Parse holdout map names (resolved to ids inside the trainer where data-dir is known).
    args.holdout_map_names = [s.strip() for s in args.holdout_maps.split(",") if s.strip()]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
