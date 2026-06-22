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
import io
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BULLET_BASE_INDICES",
    "ENCODER_OUTPUT_CHOICES",
    "HEAD_CHOICES",
    "HEATMAP_LOCALIZED_GROUPS",
    "OBJECTIVE_GROUPS",
    "PLAYER_INDICES",
    "PLAYER_POSITION_VEC_INDICES",
    "PLAYER_SUBGROUP_VEC_INDICES",
    "SENTINEL",
    "SPATIAL_OBJECTIVE_NCHAN",
    "SPATIAL_OBJECTIVES",
    "WORLD_MAX_Y",
    "WORLD_MIN_X",
    "MapAwareSplit",
    "NormStats",
    "ObjectiveConfig",
    "aim_angular_error_deg",
    "binary_f1",
    "bullet_direction_indices",
    "bullet_occupancy_field",
    "bullet_position_indices",
    "cosine_warmup_lr_multiplier",
    "decode_targets",
    "denormalize",
    "encode_bullet_targets",
    "encode_player_targets",
    "fit_norm_stats",
    "focal_occupancy_loss_np",
    "grid_to_world",
    "heatmap_point_channels",
    "interior_wall_mask",
    "interior_wall_pos_weight",
    "iou_score",
    "keypoint_field",
    "main",
    "map_aware_split",
    "normalize",
    "parse_channel_map",
    "parse_encoder_output",
    "parse_head",
    "parse_input_res",
    "parse_pos_weight_arg",
    "parse_spatial_shape",
    "per_player_cosine_distance",
    "player_head_size",
    "png_decode",
    "png_encode",
    "presence_pos_weight",
    "player_head_vec_indices",
    "resolve_map_ids",
    "stratified_group_split",
    "validate_channel_map",
    "wall_target_for",
    "world_to_grid",
]

# Epsilon floored into the L2-norm denominator of the aim cosine term, so a
# zero-magnitude prediction yields a finite loss/metric (no NaN/inf) and the
# gradient does not blow up as |pred| -> 0. Shared by the numpy helpers here and
# the mirrored torch loss in _pretrain_pixels_train.compute_losses.
AIM_COSINE_EPS = 1e-8

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
# Player POSITION sub-group as (P1_x, P1_y, P2_x, P2_y) within the 12-vec ordering. The
# heatmap head localizes the 2 player POINTS (P1, P2), so it groups these into 2 (x,y)
# pairs: P1 = (idx 0, 1), P2 = (idx 6, 7).
PLAYER_POSITION_VEC_INDICES: tuple[int, ...] = PLAYER_SUBGROUP_VEC_INDICES["player_position"]

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

# --- World->grid bounds (the SAME mapping wall_grids uses; see orientation above) ----
# The wall grid is W=20 cols x H=12 rows with col = x - MIN_X, row = MAX_Y - y. So the
# world spans x in [MIN_X, MIN_X+WALL_W) = [-10, 10) and y in (MAX_Y-WALL_H, MAX_Y] =
# (-7, 5]. The spatial-mode field targets (bullets / keypoints) map world (x,y) into a
# configured HxW grid by SCALING this same box: col = (x - MIN_X)/WALL_W * W,
# row = (MAX_Y - y)/WALL_H * H. This keeps the bullet/keypoint grid axis-consistent with
# the wall grid (and therefore with the synthetic renderer the encoder inverts).
WORLD_MIN_X = -10.0
WORLD_MAX_Y = 5.0

_SHARD_RE = re.compile(r"shard_w(\d+)_(\d+)\.npz$")

# --- Localization head selection (Part B: fc vs heatmap) ---------------------
# The READOUT for the POINT-localization targets. 'fc' (default) regresses points from
# the flatten->FC embedding (current behavior, byte-identical encoder). 'heatmap' predicts
# a per-point heatmap channel off the SPATIAL conv feature map, spatial-softmax ->
# soft-argmax -> (x,y) -> a learned affine into the SAME normalized target space the fc
# head regresses (so the per-objective metrics stay apples-to-apples). Only the point
# targets below switch; aim/velocity/presence/walls do NOT become heatmaps.
HEAD_CHOICES: tuple[str, ...] = ("fc", "heatmap")
# The objective groups whose POINTS are localized by the heatmap head (player POSITION =
# 2 points P1/P2; bullet POSITION = 10 points). These are the ONLY groups affected by the
# --head switch. Everything else keeps its current head type in both modes.
HEATMAP_LOCALIZED_GROUPS: tuple[str, ...] = ("player_position", "bullet_position")

# --- Encoder OUTPUT format (the NEW top-level axis: flat vs spatial field) ------------
# A higher-level axis than --head. 'flat' (default) = the ENTIRE existing world: the conv
# trunk pools/flattens to a flat embedding -> the slot/regression heads, with --head
# {fc,heatmap} choosing the point READOUT. Byte-identical to today. 'spatial' = a NEW
# field-based paradigm: the conv trunk projects to a configurable C x H x W field, each
# training objective is pinned to a channel (or channel range) of that field via
# --channel-map, and each channel IS that objective's prediction (occupancy/keypoint
# heatmaps + a wall grid) supervised against a FIELD target resized to H x W. In spatial
# mode --head is N/A (the point readout is intrinsic to the heatmaps). The conv TRUNK
# (cnn.0/2/4) is IDENTICAL across both modes; only the output projection / heads differ.
ENCODER_OUTPUT_CHOICES: tuple[str, ...] = ("flat", "spatial")

# The objectives a spatial channel-map may assign, and how many channels EACH requires.
# bullets -> 1 occupancy heatmap channel (order-free, count-agnostic; SUBSUMES the
# slot-based presence+position+direction objectives). walls -> 1 channel (the resized
# 20x12 wall grid; BCE / interior-F1 as today). keypoints -> 2 channels (per-tank
# heatmaps P1, P2, distinct, one channel EACH; soft-argmax read-out). Unassigned channels
# are free learned capacity (no supervision). velocity/aim are DISABLED by default in
# spatial mode and are NOT assignable here (out of scope).
SPATIAL_OBJECTIVES: tuple[str, ...] = ("bullets", "walls", "keypoints")
SPATIAL_OBJECTIVE_NCHAN: dict[str, int] = {"bullets": 1, "walls": 1, "keypoints": 2}


def parse_head(value: str | None) -> str:
    """Resolve the ``--head`` flag to a value in :data:`HEAD_CHOICES` (default ``"fc"``).

    ``None`` -> ``"fc"`` (the current behavior). A known head name passes through verbatim.
    Raises ValueError on any other string (argparse ``choices`` also guards the CLI, but
    this keeps the helper standalone-usable / unit-testable).
    """
    if value is None:
        return "fc"
    v = value.strip().lower()
    if v not in HEAD_CHOICES:
        raise ValueError(f"--head must be one of {list(HEAD_CHOICES)}, got {value!r}")
    return v


def heatmap_point_channels(cfg: ObjectiveConfig) -> dict[str, int]:
    """Number of heatmap CHANNELS (= localized points) per enabled localized group.

    Only the groups in :data:`HEATMAP_LOCALIZED_GROUPS` that are ENABLED appear:
    ``player_position`` -> 2 (P1, P2); ``bullet_position`` -> 10 (slots). Each channel is
    one (h,w) heatmap whose soft-argmax yields one (x,y). Empty if neither is enabled.
    Used only by the heatmap head; the fc head ignores it.
    """
    out: dict[str, int] = {}
    if cfg.is_on("player_position"):
        out["player_position"] = 2
    if cfg.is_on("bullet_position"):
        out["bullet_position"] = N_BULLET_SLOTS
    return out


# =============================================================================
# Spatial encoder-output config (pure; channel-map parse + validation)
# =============================================================================
def parse_encoder_output(value: str | None) -> str:
    """Resolve the ``--encoder-output`` flag to a value in :data:`ENCODER_OUTPUT_CHOICES`.

    ``None`` -> ``"flat"`` (the entire existing behavior). A known name passes through
    verbatim (case/whitespace tolerant). Raises ValueError on anything else (argparse
    ``choices`` also guards the CLI, but this keeps the helper standalone-usable).
    """
    if value is None:
        return "flat"
    v = value.strip().lower()
    if v not in ENCODER_OUTPUT_CHOICES:
        raise ValueError(
            f"--encoder-output must be one of {list(ENCODER_OUTPUT_CHOICES)}, got {value!r}"
        )
    return v


def parse_spatial_shape(spec: str) -> tuple[int, int, int]:
    """Parse a ``CxHxW`` spatial-field shape string into ``(C, H, W)`` positive ints.

    e.g. ``"4x24x40"`` -> ``(4, 24, 40)``. ``C`` is the channel count of the projected
    field, ``H``/``W`` its spatial resolution (the resolution the field targets are built
    at). All three must be positive. Raises ValueError on a malformed / non-positive spec.
    """
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--spatial-shape must look like CxHxW (e.g. 4x24x40), got {spec!r}")
    c, h, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"--spatial-shape C,H,W must all be positive, got {spec!r}")
    return c, h, w


def parse_channel_map(spec: str) -> dict[str, tuple[int, ...]]:
    """Parse a ``--channel-map`` string into ``{objective: (channel indices,...)}``.

    FORMAT: a comma-separated list of ``name=idx`` or ``name=idx0,idx1,...`` assignments,
    where a name owns every index up to the NEXT name. Examples::

        "bullets=0,walls=1,keypoints=2,3"   # bullets->(0,), walls->(1,), keypoints->(2,3)
        "keypoints=0,1,bullets=2,walls=3"   # order-independent

    Indices are parsed in declaration order; an index with no preceding ``name=`` extends
    the most recent objective's range. This is a PARSE-ONLY helper: it returns the raw
    assignment WITHOUT range/overlap/count checks (use :func:`validate_channel_map` for
    that, which also needs the field channel count C). Raises ValueError on syntax errors
    (empty, unknown objective name, a leading bare index, a non-integer index, a repeated
    objective name).
    """
    if spec is None or not spec.strip():
        raise ValueError("--channel-map is empty")
    assignment: dict[str, list[int]] = {}
    current: str | None = None
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            raise ValueError(f"--channel-map has an empty token in {spec!r}")
        if "=" in tok:
            name, _, idx_str = tok.partition("=")
            name = name.strip().lower()
            if name not in SPATIAL_OBJECTIVES:
                raise ValueError(
                    f"--channel-map: unknown objective {name!r}. Valid: {list(SPATIAL_OBJECTIVES)}"
                )
            if name in assignment:
                raise ValueError(f"--channel-map: objective {name!r} assigned more than once")
            assignment[name] = []
            current = name
            idx_str = idx_str.strip()
            if not idx_str:
                raise ValueError(f"--channel-map: objective {name!r} has no channel index")
            _append_index(assignment[current], idx_str, spec)
        else:
            if current is None:
                raise ValueError(
                    f"--channel-map: leading channel index {tok!r} has no objective "
                    f"(expected name=idx first) in {spec!r}"
                )
            _append_index(assignment[current], tok, spec)
    return {k: tuple(v) for k, v in assignment.items()}


def _append_index(dst: list[int], tok: str, spec: str) -> None:
    """Parse a single channel-index token into ``dst`` (raises ValueError on non-int)."""
    try:
        dst.append(int(tok))
    except ValueError as e:
        raise ValueError(
            f"--channel-map: channel index {tok!r} is not an integer in {spec!r}"
        ) from e


def validate_channel_map(
    channel_map: dict[str, tuple[int, ...]], n_channels: int
) -> dict[str, tuple[int, ...]]:
    """Validate a parsed channel-map against the field channel count ``C`` (n_channels).

    Rules (a violation raises a CLEAN ValueError, never a mid-train crash):
    * every index must be in range ``[0, C)``;
    * NO overlap between objectives (each channel owned by at most one objective);
    * each ASSIGNED objective gets EXACTLY the channels it requires
      (:data:`SPATIAL_OBJECTIVE_NCHAN`: bullets=1, walls=1, keypoints=2);
    * at least one objective must be assigned.

    Unassigned channels in ``[0, C)`` are allowed (free learned capacity). Returns the
    channel_map unchanged on success (so callers can chain). Negative indices are rejected
    by the range check (``< 0``).
    """
    if not channel_map:
        raise ValueError("--channel-map assigned no objectives")
    seen: dict[int, str] = {}
    for name, idxs in channel_map.items():
        need = SPATIAL_OBJECTIVE_NCHAN[name]
        if len(idxs) != need:
            raise ValueError(
                f"--channel-map: objective {name!r} needs exactly {need} channel(s), "
                f"got {len(idxs)} ({list(idxs)})"
            )
        if len(set(idxs)) != len(idxs):
            raise ValueError(
                f"--channel-map: objective {name!r} has duplicate channels {list(idxs)}"
            )
        for ch in idxs:
            if ch < 0 or ch >= n_channels:
                raise ValueError(
                    f"--channel-map: channel {ch} (objective {name!r}) out of range "
                    f"[0,{n_channels}) for spatial-shape C={n_channels}"
                )
            if ch in seen:
                raise ValueError(
                    f"--channel-map: channel {ch} assigned to both {seen[ch]!r} and {name!r} "
                    "(channels must not overlap)"
                )
            seen[ch] = name
    return channel_map


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


# =============================================================================
# Class-imbalance pos_weights (pure; numpy; TRAIN-split only — torch-free)
# =============================================================================
def presence_pos_weight(presence: np.ndarray, *, fallback: float = 1.0) -> float:
    """Positive-class BCE weight for the bullet-presence head: (#absent)/(#present).

    Bullet slots are overwhelmingly absent (~7.8% present), so plain BCE collapses to
    "always absent" (presence F1 = 0). A ``pos_weight`` of (#zeros)/(#ones) re-balances
    the loss so present slots are not drowned out. MUST be computed on the TRAIN split
    only (the caller passes the train presence targets) to avoid eval/validation leakage.

    ``presence`` is a ``(N, 10)`` (or any-shape) {0,1} array. Returns the float ratio of
    absent-to-present slots; falls back to ``fallback`` if there are zero present slots
    (avoids divide-by-zero).
    """
    arr = np.asarray(presence)
    present = float((arr > 0.5).sum())
    if present <= 0.0:
        return float(fallback)
    absent = float(arr.size) - present
    return absent / present


def interior_wall_pos_weight(walls: np.ndarray, *, fallback: float = 1.0) -> float:
    """Positive-class BCE weight for the INTERIOR wall cells: (#interior free)/(#interior wall).

    The full 20x12 grid is ~50% walls (a constant 2-deep border), so a GLOBAL wall
    pos_weight is ~1.0 (a no-op). The real imbalance lives in the interior (only ~6.6%
    walls), so this restricts to :func:`interior_wall_mask` cells and returns the ratio of
    interior FREE cells to interior WALL cells there. Border cells never enter the count.
    MUST be computed on the TRAIN split only (the caller passes train wall targets).

    ``walls`` is a ``(N, 12, 20)`` {0,1} array. Returns the float ratio; falls back to
    ``fallback`` if there are zero interior wall cells (avoids divide-by-zero).
    """
    arr = np.asarray(walls)
    mask = interior_wall_mask()  # (12, 20) bool
    interior = arr[..., mask] > 0.5  # (N, 128)
    wall = float(interior.sum())
    if wall <= 0.0:
        return float(fallback)
    free = float(interior.size) - wall
    return free / wall


def parse_pos_weight_arg(value: str | float | None) -> float | None:
    """Resolve a ``--*-pos-weight`` CLI value: ``None``/``"auto"`` -> ``None`` (compute),
    else a parsed/passed float (used verbatim).

    Returns ``None`` to signal "auto-compute from the train split", or the explicit float
    override. Raises ValueError on a non-numeric, non-"auto" string.
    """
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    s = value.strip().lower()
    if s == "auto":
        return None
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"pos-weight must be 'auto' or a float, got {value!r}") from e


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


# --- Aim (unit-vector direction) loss + metric (per player) ------------------
# Aim is a UNIT vector per player. MSE's least-bad answer on a ~uniform-on-the-
# circle unit target is to collapse to (0,0) (norm_mse == 1.0), burying any weak
# directional signal. A per-player COSINE distance forces a directional commitment
# instead. The aim slice is laid out [P1_x, P1_y, P2_x, P2_y] (12-vec idx 4,5,10,11);
# the cosine is computed on each player's 2D vector SEPARATELY (NOT one 4D cosine
# over the concatenation) and averaged over the 2 players and the batch.
def _per_player_cos_sim(pred_aim: np.ndarray, target_aim: np.ndarray, *, eps: float) -> np.ndarray:
    """Per-player cosine SIMILARITY for an aim slice ``(N, 4)`` -> ``(N, 2)``.

    Columns are ``[P1_x, P1_y, P2_x, P2_y]``; reshaped to ``(N, 2, 2)`` (player, xy).
    Each player's 2D vectors are L2-normalized with ``eps`` floored into the
    denominator (so a zero-magnitude vector is finite, not NaN/inf) before the dot.
    """
    pred = np.asarray(pred_aim, dtype=np.float64).reshape(-1, 2, 2)
    tgt = np.asarray(target_aim, dtype=np.float64).reshape(-1, 2, 2)
    pn = np.sqrt((pred**2).sum(axis=2)) + eps  # (N, 2)
    tn = np.sqrt((tgt**2).sum(axis=2)) + eps  # (N, 2)
    dot = (pred * tgt).sum(axis=2)  # (N, 2)
    return dot / (pn * tn)


def per_player_cosine_distance(
    pred_aim: np.ndarray, target_aim: np.ndarray, *, eps: float = AIM_COSINE_EPS
) -> float:
    """Mean per-player cosine DISTANCE ``1 - cos`` for an aim slice ``(N, 4)``.

    ``pred_aim`` / ``target_aim`` are ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]``. The
    cosine is taken on each PLAYER's 2D vector separately (not a single 4D cosine over
    the concatenation) and the ``1 - cos`` distance is averaged over the 2 players and
    the batch. Identical directions -> 0; orthogonal -> 1; opposite -> 2. ``eps`` is
    floored into the L2-norm denominator so a zero-magnitude prediction is finite.
    """
    if np.asarray(pred_aim).size == 0:
        return 0.0
    cos = _per_player_cos_sim(pred_aim, target_aim, eps=eps)  # (N, 2)
    return float(np.mean(1.0 - cos))


def aim_angular_error_deg(
    pred_aim: np.ndarray, target_aim: np.ndarray, *, eps: float = AIM_COSINE_EPS
) -> dict[str, float]:
    """Mean per-player angular error (DEGREES) + mean cosine similarity for aim.

    ``pred_aim`` / ``target_aim`` are ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]``. Returns
    ``{"deg": mean angular error in degrees, "cos": mean cosine similarity}``, both
    averaged per player over the batch. The angle is a clamped ``arccos`` of the cosine
    similarity (clamped to ``[-1, 1]`` for numerical safety). Reference baselines: a
    random / no-signal predictor -> ~90 deg, cos ~0; perfect -> 0 deg, cos 1; opposite
    -> 180 deg, cos -1.
    """
    if np.asarray(pred_aim).size == 0:
        return {"deg": 0.0, "cos": 0.0}
    cos = _per_player_cos_sim(pred_aim, target_aim, eps=eps)  # (N, 2)
    clamped = np.clip(cos, -1.0, 1.0)
    deg = np.degrees(np.arccos(clamped))
    return {"deg": float(np.mean(deg)), "cos": float(np.mean(cos))}


# =============================================================================
# Spatial-mode FIELD targets (pure; numpy): occupancy + keypoint heatmaps, wall resize
# =============================================================================
# All built AT the configured H x W. World (x,y) maps into the grid with the SAME box the
# wall grid uses (WORLD_MIN_X / WORLD_MAX_Y), scaled to H,W. Coordinates returned/consumed
# are FRACTIONAL grid coords (col_frac in [0,W], row_frac in [0,H]) so a sub-cell location
# is preserved; the soft-argmax read-out reports in the same fractional grid frame.
def world_to_grid(pos_xy: np.ndarray, grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Map world ``(x, y)`` positions into FRACTIONAL ``(col, row)`` coords of an H x W grid.

    ``pos_xy`` is ``(..., 2)`` = world ``[x, y]``. Returns ``(col, row)`` arrays (each the
    leading shape of ``pos_xy``) using the canonical wall-grid box scaled to ``grid_w`` /
    ``grid_h``: ``col = (x - WORLD_MIN_X) / WALL_W * grid_w``,
    ``row = (WORLD_MAX_Y - y) / WALL_H * grid_h``. Fractional (not floored) so sub-cell
    position is preserved for the Gaussian splat. Out-of-box positions map outside
    ``[0, grid]`` (the caller's Gaussian simply contributes ~0 inside the grid).
    """
    pos_xy = np.asarray(pos_xy, dtype=np.float64)
    x = pos_xy[..., 0]
    y = pos_xy[..., 1]
    col = (x - WORLD_MIN_X) / WALL_W * grid_w
    row = (WORLD_MAX_Y - y) / WALL_H * grid_h
    return col, row


def _gaussian_splat(field: np.ndarray, col: float, row: float, sigma: float) -> None:
    """Add a unit-peak 2D Gaussian centered at fractional ``(col, row)`` into ``field`` (H,W).

    In place; ``exp(-((c-col)^2 + (r-row)^2) / (2 sigma^2))`` over all cells (cell centers
    at integer indices). Peaks at 1.0 at the nearest cell to the center. Overlapping
    Gaussians ACCUMULATE (so the field is the sum-of-Gaussians the CenterNet focal loss
    expects); the read-out is per-peak, so accumulation does not change the peak locations.
    """
    h, w = field.shape
    rr = np.arange(h, dtype=np.float64).reshape(h, 1)
    cc = np.arange(w, dtype=np.float64).reshape(1, w)
    field += np.exp(-(((cc - col) ** 2) + ((rr - row) ** 2)) / (2.0 * sigma**2))


def bullet_occupancy_field(
    states: np.ndarray, grid_h: int, grid_w: int, *, sigma: float = 1.0
) -> np.ndarray:
    """Build the bullet OCCUPANCY heatmap target ``(N, grid_h, grid_w)`` (CenterNet style).

    For each frame, every PRESENT bullet (across both players' 10 slots; absent = the
    ``-100`` sentinel) is splatted as a unit-peak 2D Gaussian at its IMAGE location (world
    position mapped into the grid by :func:`world_to_grid`), and the per-frame field is the
    sum/max of those Gaussians clamped to ``[0, 1]`` so each true bullet is a ~1.0 peak.

    ORDER-FREE (no slots): permuting the bullets within a frame yields the SAME field.
    COUNT-AGNOSTIC: 0 bullets -> an ~all-zero field; N bullets -> N peaks. This REPLACES
    the slot-based presence + position + direction objectives in spatial mode. ``sigma`` is
    the Gaussian spread (in grid cells); the field values are floored at 0 and capped at 1.

    The cell NEAREST each bullet is snapped to EXACTLY 1.0 (the CenterNet positive "peak"):
    a fractional Gaussian center otherwise tops out a hair below 1.0, which would leave the
    focal-loss positive set (``target == 1``) empty and let the loss be satisfied by
    predicting all-zero. The Gaussian skirts (<1) are kept for the penalty-reduction term.
    """
    states = np.asarray(states)
    n = states.shape[0]
    pos = states[:, bullet_position_indices()].reshape(n, N_BULLET_SLOTS, 2)  # (N,10,2) world xy
    presence, _, _ = encode_bullet_targets(states)  # (N, 10) {0,1}
    out = np.zeros((n, grid_h, grid_w), dtype=np.float32)
    col_all, row_all = world_to_grid(pos, grid_h, grid_w)  # (N,10) each
    for i in range(n):
        field = np.zeros((grid_h, grid_w), dtype=np.float64)
        peaks: list[tuple[int, int]] = []
        for s in range(N_BULLET_SLOTS):
            if presence[i, s] <= 0.5:
                continue
            c, r = float(col_all[i, s]), float(row_all[i, s])
            _gaussian_splat(field, c, r, sigma)
            rc = int(round(r)), int(round(c))
            if 0 <= rc[0] < grid_h and 0 <= rc[1] < grid_w:
                peaks.append(rc)
        field = np.clip(field, 0.0, 1.0)
        for r, c in peaks:  # snap the nearest cell to an EXACT 1.0 positive peak
            field[r, c] = 1.0
        out[i] = field.astype(np.float32)
    return out


def keypoint_field(
    states: np.ndarray, grid_h: int, grid_w: int, *, sigma: float = 1.0
) -> np.ndarray:
    """Build the per-tank keypoint heatmap target ``(N, 2, grid_h, grid_w)`` (P1, P2 channels).

    Channel 0 = a single unit-peak Gaussian at P1's grid location, channel 1 = at P2's
    (both always present). Distinct channels (one tank EACH), soft-argmax read-out ->
    coords. Built at the configured H x W via :func:`world_to_grid`; ``sigma`` is the
    Gaussian spread in grid cells.
    """
    states = np.asarray(states)
    n = states.shape[0]
    # Player positions: P1 = state idx 0,1 ; P2 = idx 26,27.
    p1 = states[:, [0, 1]]
    p2 = states[:, [26, 27]]
    pts = np.stack([p1, p2], axis=1)  # (N, 2, 2) world xy per tank
    out = np.zeros((n, 2, grid_h, grid_w), dtype=np.float32)
    col_all, row_all = world_to_grid(pts, grid_h, grid_w)  # (N,2) each
    for i in range(n):
        for k in range(2):
            field = np.zeros((grid_h, grid_w), dtype=np.float64)
            _gaussian_splat(field, float(col_all[i, k]), float(row_all[i, k]), sigma)
            out[i, k] = field.astype(np.float32)
    return out


def grid_to_world(col: np.ndarray, row: np.ndarray, grid_h: int, grid_w: int):
    """Inverse of :func:`world_to_grid`: fractional ``(col, row)`` -> world ``(x, y)``.

    ``x = col / grid_w * WALL_W + WORLD_MIN_X``, ``y = WORLD_MAX_Y - row / grid_h * WALL_H``.
    Used to convert a soft-argmax grid read-out back to world units for the keypoint /
    occupancy world-error metric so spatial metrics stay comparable to flat's world units.
    """
    col = np.asarray(col, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64)
    x = col / grid_w * WALL_W + WORLD_MIN_X
    y = WORLD_MAX_Y - row / grid_h * WALL_H
    return x, y


def focal_occupancy_loss_np(
    pred_prob: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> float:
    """CenterNet penalty-reduced focal loss (numpy mirror; for unit tests + reference).

    ``pred_prob`` and ``target`` are ``(..., H, W)`` in ``[0, 1]`` (``pred_prob`` already a
    sigmoid output, ``target`` the sum-of-Gaussians occupancy field with ~1.0 peaks). The
    loss (Law & Deng, CenterNet) is, per cell::

        peak  (target == 1):  -(1 - p)^alpha * log(p)
        else  (0<=target<1):  -(1 - target)^beta * p^alpha * log(1 - p)

    averaged over the number of POSITIVE (peak) cells (>=1), so it is count-normalized. The
    ``(1 - target)^beta`` factor REDUCES the penalty for negatives NEAR a peak (where
    ``target`` is close to 1), exactly as specified. A perfect prediction -> ~0; a
    confident-wrong prediction -> large. ``eps`` clamps ``p`` away from 0/1 for a finite log.
    """
    p = np.clip(np.asarray(pred_prob, dtype=np.float64), eps, 1.0 - eps)
    t = np.asarray(target, dtype=np.float64)
    pos = t >= 1.0 - 1e-12
    pos_loss = -((1.0 - p) ** alpha) * np.log(p) * pos
    neg_loss = -((1.0 - t) ** beta) * (p**alpha) * np.log(1.0 - p) * (~pos)
    n_pos = float(pos.sum())
    total = float(pos_loss.sum() + neg_loss.sum())
    return total / max(1.0, n_pos)


# =============================================================================
# LR schedule: linear warmup -> cosine anneal (pure; torch-free)
# =============================================================================
def cosine_warmup_lr_multiplier(step: int, total_steps: int, warmup_frac: float = 0.05) -> float:
    """LR MULTIPLIER in [0, 1] for a linear-warmup -> cosine-anneal schedule.

    Drives a ``torch.optim.lr_scheduler.LambdaLR`` (the lambda just calls this), so the
    schedule is exactly unit-testable without torch. ``step`` is the optimizer-step index
    (0-based); ``total_steps`` is the total number of optimizer steps over the whole run
    (steps-per-epoch * epochs). The curve is:

    * WARMUP: ``warmup_steps = max(1, round(warmup_frac * total_steps))``. The multiplier
      ramps LINEARLY from ``1 / warmup_steps`` at ``step == 0`` (strictly positive, never 0)
      up to exactly ``1.0`` at ``step == warmup_steps`` (the base ``--lr`` is reached at the
      end of warmup). The ramp is strictly increasing on ``[0, warmup_steps]``.
    * COSINE: for ``step > warmup_steps`` the multiplier follows
      ``0.5 * (1 + cos(pi * progress))`` with
      ``progress = (step - warmup_steps) / (total_steps - warmup_steps)`` clamped to
      ``[0, 1]`` — i.e. 1.0 right after warmup, decaying monotonically to ~0 at
      ``step == total_steps``. The cosine midpoint (``progress == 0.5``) is exactly 0.5.

    ``warmup_frac=0.05`` (5%) is the default so the aggressive ``1e-3`` arm stays stable.
    With ``total_steps <= 1`` (or warmup spanning the whole run) it degrades gracefully to
    the warmup ramp. The returned multiplier is in ``[0, 1]``.
    """
    total = max(1, int(total_steps))
    warmup_steps = max(1, round(warmup_frac * total))
    warmup_steps = min(warmup_steps, total)  # never warm up past the end of the run
    if step <= warmup_steps:
        # Linear ramp from 1/(warmup_steps+1) at step 0 to exactly 1.0 at warmup_steps:
        # f(s) = (s + 1) / (warmup_steps + 1) for s < warmup_steps, capped at 1.0.
        # Strictly increasing on [0, warmup_steps]; f(warmup_steps) == 1.0.
        if step >= warmup_steps:
            return 1.0
        return (step + 1) / (warmup_steps + 1)
    denom = total - warmup_steps
    if denom <= 0:
        return 1.0
    progress = (step - warmup_steps) / denom
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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


# --- PNG round-trip helpers (LOSSLESS; used by the PNG-in-LMDB data path) -----
# Imported by scripts/transcode_pixel_png_lmdb.py (writer) and LmdbPngPixelDataset
# (reader). Pillow only (a pinned dep); NO torch — kept here in the pure module so the
# round-trip is unit-testable without building an LMDB and without importing the trainer.
def png_encode(arr: np.ndarray) -> bytes:
    """LOSSLESS PNG-encode a ``(H,W,3)`` uint8 RGB array to PNG bytes in memory.

    PNG is always lossless, so ``png_decode(png_encode(arr))`` is bit-identical to
    ``arr`` (asserted by the unit test). No resize / colorspace change happens here.
    """
    from PIL import Image

    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"png_encode expects (H,W,3) uint8, got shape {arr.shape}")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def png_decode(data: bytes) -> np.ndarray:
    """Decode PNG bytes back to a ``(H,W,3)`` uint8 RGB array (inverse of png_encode).

    Returns a WRITABLE copy (PIL's buffer is read-only; ``torch.from_numpy`` on a
    read-only array warns), so the dataset can wrap it in a tensor without a copy-warning.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8, copy=True)


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
    p.add_argument(
        "--frames-backend",
        type=str,
        default="memmap",
        choices=("memmap", "lmdb_png"),
        help=(
            "frame storage backend for --cache-dir (full run). 'memmap' (default) reads the "
            "flat uint8 memmap from prepare_pixel_cache.py; 'lmdb_png' reads PNG-encoded "
            "frames from the LMDB built by scripts/transcode_pixel_png_lmdb.py. No-op when "
            "--cache-dir is omitted (the SMOKE stream path is unaffected)."
        ),
    )
    p.add_argument("--input-res", type=str, default="160x90", help="downsample target WxH")
    p.add_argument("--embedding-dim", type=int, default=512, help="encoder embedding width")
    # --- TOP-LEVEL output-format switch (flat = entire existing world; spatial = NEW field) ---
    p.add_argument(
        "--encoder-output",
        type=str,
        default="flat",
        choices=ENCODER_OUTPUT_CHOICES,
        help=(
            "encoder OUTPUT format. 'flat' (default) = the entire EXISTING world: conv "
            "trunk -> flat embedding -> slot/regression heads, with --head choosing the "
            "point readout (BYTE-IDENTICAL to today). 'spatial' = NEW field paradigm: conv "
            "trunk -> a configurable C x H x W field (--spatial-shape); each objective is "
            "pinned to channel(s) via --channel-map and supervised against a FIELD target "
            "(bullet occupancy + keypoint heatmaps + resized wall grid). The conv TRUNK "
            "(cnn.0/2/4 = encoder.pt) is IDENTICAL in both modes; only the projection/heads "
            "differ. In spatial mode --head is N/A (recorded as null in config)."
        ),
    )
    p.add_argument(
        "--spatial-shape",
        type=str,
        default="4x24x40",
        help=(
            "spatial-mode FIELD shape CxHxW (e.g. 4x24x40). C = channels of the projected "
            "field, HxW = the resolution the field targets are built at (finer HxW helps "
            "bullet/keypoint localization; walls are fine coarse). No-op for --encoder-output flat."
        ),
    )
    p.add_argument(
        "--channel-map",
        type=str,
        default="bullets=0,walls=1,keypoints=2,3",
        help=(
            "spatial-mode channel assignment 'name=idx[,idx...]' (comma-list; a name owns "
            "indices up to the next name). Valid objectives: bullets (1 channel, occupancy "
            "heatmap), walls (1, resized grid), keypoints (2, per-tank P1/P2). e.g. "
            "'bullets=0,walls=1,keypoints=2,3'. Indices must be in [0,C), non-overlapping, "
            "and exactly match each objective's channel count. Unassigned channels are free "
            "capacity. No-op for --encoder-output flat."
        ),
    )
    p.add_argument(
        "--occupancy-sigma",
        type=float,
        default=1.0,
        help=(
            "Gaussian spread (in grid cells) for the spatial-mode bullet-occupancy and "
            "keypoint heatmap TARGETS. No-op for --encoder-output flat."
        ),
    )
    p.add_argument(
        "--head",
        type=str,
        default="fc",
        choices=HEAD_CHOICES,
        help=(
            "localization READOUT for the POINT targets (player_position + bullet_position). "
            "'fc' (default) = flatten->Linear(embedding) point regression (CURRENT behavior; "
            "encoder.pt byte-identical). 'heatmap' = per-point heatmap channel off the spatial "
            "conv feature map -> spatial-softmax -> soft-argmax -> learned affine into the SAME "
            "normalized target space, and the encoder GLOBAL-POOLs the feature map (far fewer "
            "params). Aim/velocity/presence stay regressed; walls use a spatial conv head."
        ),
    )
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
    # --- Aim sub-objective: cosine-distance loss (default) vs the old per-field MSE ---
    p.add_argument(
        "--aim-loss",
        type=str,
        default="cosine",
        choices=("cosine", "mse"),
        help=(
            "loss for the player_aim sub-objective. 'cosine' (default) = per-player "
            "cosine distance on RAW unit aim (forces a directional commitment); 'mse' "
            "reproduces the OLD per-field MSE on the standardized aim slice. No-op when "
            "player_aim is disabled."
        ),
    )
    p.add_argument(
        "--lambda-aim",
        type=float,
        default=1.0,
        help="weight on the player_aim loss term (no-op when player_aim is disabled).",
    )
    # --- Class-imbalance pos_weights (BCE). "auto" => computed from the TRAIN split. ---
    p.add_argument(
        "--presence-pos-weight",
        type=str,
        default="auto",
        help=(
            "BCE pos_weight for bullet_presence. 'auto' (default) computes "
            "(#absent)/(#present) over the TRAIN split (~12); or pass a float to use verbatim."
        ),
    )
    p.add_argument(
        "--wall-pos-weight",
        type=str,
        default="auto",
        help=(
            "BCE pos_weight for INTERIOR wall cells. 'auto' (default) computes "
            "(#interior free)/(#interior wall) over the TRAIN split (~14); or pass a float."
        ),
    )
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "AdamW decoupled weight decay. Default 0.0 makes AdamW behave like the old "
            "Adam (behavior-preserving). Set >0 to regularize."
        ),
    )
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "cosine"),
        help=(
            "LR schedule. 'constant' (default) holds --lr (byte-equivalent to the old "
            "behavior). 'cosine' = linear warmup (~5%% of total optimizer steps) then "
            "cosine anneal toward ~0 over the remaining steps."
        ),
    )
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
    # Normalize the head choice (argparse choices already guard; keep it canonical).
    args.head = parse_head(args.head)
    # Normalize the encoder-output mode (top-level flat vs spatial).
    args.encoder_output = parse_encoder_output(args.encoder_output)
    # Resolve the spatial field config eagerly so a bad --spatial-shape / --channel-map
    # fails CLEANLY before any heavy import or training (never mid-train). Only validated
    # in spatial mode; in flat mode they are inert (parsed but unused).
    if args.encoder_output == "spatial":
        args.spatial_shape = parse_spatial_shape(args.spatial_shape)
        args.channel_map = validate_channel_map(
            parse_channel_map(args.channel_map), args.spatial_shape[0]
        )
        # In spatial mode --head is N/A; force it None so config records head: null and the
        # flat-only point-readout switch can't be silently mis-applied.
        args.head = None
    else:
        # flat mode: keep the raw strings (inert); --head stays as resolved above.
        args.spatial_shape = None
        args.channel_map = None
    # Parse holdout map names (resolved to ids inside the trainer where data-dir is known).
    args.holdout_map_names = [s.strip() for s in args.holdout_maps.split(",") if s.strip()]
    return args


if __name__ == "__main__":
    raise SystemExit(main())
