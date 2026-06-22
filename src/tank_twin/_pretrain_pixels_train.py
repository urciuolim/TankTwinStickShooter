"""Torch trainer for the pixel inverse-renderer pretraining (Stage 2).

Imported lazily from :mod:`tank_twin.pretrain_pixels` so the pure target-encoding /
normalization / split / metric helpers (and their unit tests) never need torch.
Contains the model (reusable encoder + configurable supervised heads), the streaming
and memmap datasets, the loss, the train/eval/validation loop with per-objective
metrics, and checkpoint save/reload.

The ENCODER (conv stack + FC projection) is the reusable artifact and is INDEPENDENT
of the objective config: its state_dict is saved standalone as ``encoder.pt`` and can
be loaded into a fresh ``PixelEncoder`` without any head, regardless of which
objectives were trained. Only the heads change with the objective config.

TERMINOLOGY:
* "eval" = between-epoch, IN-DISTRIBUTION (held-out EPISODES of the 8 training maps).
* "validation" = final, OUT-OF-DISTRIBUTION (the 2 held-out MAPS), run ONCE.
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tank_twin.pretrain_pixels import (
    AIM_COSINE_EPS,
    N_BULLET_DIR,
    N_BULLET_POS,
    PLAYER_SUBGROUP_VEC_INDICES,
    WALL_H,
    WALL_W,
    NormStats,
    ObjectiveConfig,
    aim_angular_error_deg,
    cosine_warmup_lr_multiplier,
    decode_targets,
    denormalize,
    downsample_frames_np,
    fit_norm_stats,
    head_sizes,
    interior_wall_mask,
    interior_wall_pos_weight,
    list_shards,
    load_map_name_to_id,
    map_aware_split,
    parse_pos_weight_arg,
    player_head_vec_indices,
    png_decode,
    presence_pos_weight,
    resolve_map_ids,
    worker_id_from_shard,
)

# Player sub-group -> (name, position?) and its slice within the player HEAD output.
_PLAYER_SUBGROUPS = ("player_position", "player_velocity", "player_aim")
_PLAYER_POSITION_GROUPS = {"player_position"}
# The 12-vec indices of the aim sub-group, [P1_x, P1_y, P2_x, P2_y].
_AIM_VEC_INDICES = PLAYER_SUBGROUP_VEC_INDICES["player_aim"]


def _aim_head_columns(model: InverseRenderer) -> list[int] | None:
    """Columns of the player-head output that hold the aim slice (4,5,10,11), in
    ``[P1_x, P1_y, P2_x, P2_y]`` order; ``None`` if aim is not a head output."""
    vi = model.player_vec_indices
    if not all(k in vi for k in _AIM_VEC_INDICES):
        return None
    return [vi.index(k) for k in _AIM_VEC_INDICES]


def _player_non_aim_columns(model: InverseRenderer) -> list[int]:
    """Columns of the player-head output that are NOT aim (position + velocity)."""
    aim_cols = set(_aim_head_columns(model) or [])
    return [c for c in range(len(model.player_vec_indices)) if c not in aim_cols]


def _destandardize_aim(z_aim: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """De-standardize a ``(N, 4)`` aim slice (12-vec idx 4,5,10,11) back to RAW units.

    The player target is z-standardized (decode_targets divides each aim component by
    ~0.707, distorting the angle), so the cosine term must compare RAW directions. This
    inverts ``z -> z*std + mean`` for the aim indices, on the slice's device/dtype.
    """
    idx = list(_AIM_VEC_INDICES)
    mean = torch.as_tensor(stats.player_mean[idx], dtype=z_aim.dtype, device=z_aim.device)
    std = torch.as_tensor(stats.player_std[idx], dtype=z_aim.dtype, device=z_aim.device)
    return z_aim * std + mean


def _torch_per_player_cosine_distance(
    pred_aim: torch.Tensor, target_aim: torch.Tensor, *, eps: float = AIM_COSINE_EPS
) -> torch.Tensor:
    """Torch mirror of :func:`pretrain_pixels.per_player_cosine_distance`.

    ``pred_aim`` / ``target_aim`` are ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]``; the
    cosine is taken per PLAYER on the 2D vector (reshape to ``(N, 2, 2)``) with ``eps``
    floored into each L2-norm denominator, then ``1 - cos`` is averaged over the 2
    players and the batch. Identical -> 0, orthogonal -> 1, opposite -> 2.
    """
    pred = pred_aim.reshape(-1, 2, 2)
    tgt = target_aim.reshape(-1, 2, 2)
    pn = torch.sqrt((pred**2).sum(dim=2)) + eps  # (N, 2)
    tn = torch.sqrt((tgt**2).sum(dim=2)) + eps  # (N, 2)
    dot = (pred * tgt).sum(dim=2)  # (N, 2)
    cos = dot / (pn * tn)
    return (1.0 - cos).mean()


# =============================================================================
# Model: reusable encoder + configurable heads on the shared embedding
# =============================================================================
class PixelEncoder(nn.Module):
    """NatureCNN-style encoder: 3 convs -> flatten -> Linear(embedding_dim) -> ReLU.

    Input is channel-first ``(B, 3, H, W)`` float in [0,1]. This conv+FC stack is the
    REUSABLE artifact; ``state_dict`` is what gets saved as ``encoder.pt``. Its
    architecture / keys do NOT depend on the objective config.
    """

    def __init__(self, in_h: int, in_w: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.in_h, self.in_w, self.embedding_dim = in_h, in_w, embedding_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, 3, in_h, in_w)).shape[1]
        self.flatten_dim = int(flat)
        self.linear = nn.Sequential(nn.Linear(self.flatten_dim, embedding_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class InverseRenderer(nn.Module):
    """Encoder + the heads for the ENABLED objective groups (read the shared embedding).

    Disabled groups get no head at all (no parameters, no output, no compute). The
    player head outputs only the enabled player sub-groups (4 floats each); bullet
    position / direction / presence and wall heads are present only when enabled.
    """

    def __init__(
        self, in_h: int, in_w: int, cfg: ObjectiveConfig, embedding_dim: int = 512
    ) -> None:
        super().__init__()
        self.encoder = PixelEncoder(in_h, in_w, embedding_dim)
        self.cfg = cfg
        self.sizes = head_sizes(cfg)
        self.heads = nn.ModuleDict()
        for name, size in self.sizes.items():
            self.heads[name] = nn.Linear(embedding_dim, size)
        # Indices into the 12-vector that the player head predicts (enabled sub-groups).
        self.player_vec_indices = player_head_vec_indices(cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            y = head(z)
            if name == "wall":
                y = y.reshape(-1, WALL_H, WALL_W)
            out[name] = y
        return out


def compute_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    model: InverseRenderer,
    lambdas: dict[str, float],
    presence_pw: float | torch.Tensor,
    wall_pw: float | torch.Tensor,
    *,
    stats: NormStats | None = None,
    aim_loss: str = "cosine",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total + per-component loss over ENABLED objectives only.

    * player: MSE on the normalized enabled-sub-group slice of the 12-vector target,
      EXCLUDING aim (position + velocity only when their sub-groups are enabled).
    * aim (when ``player_aim`` enabled): a SEPARATE term. Default ``aim_loss="cosine"``
      uses a per-player cosine distance on the RAW (de-standardized) unit aim, which
      forces a directional commitment (MSE collapses to (0,0) on a ~uniform unit target).
      ``aim_loss="mse"`` reproduces the OLD behavior: per-field MSE on the standardized
      aim slice, folded into the ``player`` part. ``stats`` is required for cosine (to
      de-standardize aim back to raw); it is unused for mse.
    * bullet_presence: BCEWithLogits on 10 slot bits, with ``pos_weight=presence_pw`` to
      counter the slot-level imbalance (~7.8% present) that collapses plain BCE to F1=0.
    * bullet_position / bullet_direction: MSE on 20 normalized floats, MASKED to present
      slots (absent slots contribute 0; mean over present floats only).
    * walls: BCEWithLogits restricted to the INTERIOR cells only (the constant 2-deep
      border is information-free and is dropped from the loss), with ``pos_weight=wall_pw``
      on interior wall cells. The head stays 240 logits; border cells contribute exactly 0.

    ``presence_pw`` / ``wall_pw`` are TRAIN-level scalars (float or 0-dim tensor); they are
    materialized as scalar tensors on the preds' device/dtype inside this function.
    """
    mse = nn.functional.mse_loss
    bce = nn.functional.binary_cross_entropy_with_logits
    cfg = model.cfg

    terms: list[torch.Tensor] = []
    parts: dict[str, float] = {}

    def add(name: str, loss: torch.Tensor, weight: float) -> None:
        terms.append(loss * weight)
        parts[name] = float(loss.detach())

    if cfg.any_player():
        idx = model.player_vec_indices
        pred_player = preds["player"]
        tgt_player = targets["player"][:, idx]
        aim_cols = _aim_head_columns(model)
        use_cosine = aim_loss == "cosine" and aim_cols is not None
        if use_cosine:
            # Aim leaves the player MSE term: regress position(+velocity) only.
            non_aim = _player_non_aim_columns(model)
            if non_aim:
                add(
                    "player",
                    mse(pred_player[:, non_aim], tgt_player[:, non_aim]),
                    lambdas["player"],
                )
            if stats is None:  # pragma: no cover - run_training always threads stats
                raise ValueError("aim_loss='cosine' requires stats to de-standardize aim")
            pred_aim_raw = _destandardize_aim(pred_player[:, aim_cols], stats)
            tgt_aim_raw = _destandardize_aim(tgt_player[:, aim_cols], stats)
            add(
                "aim",
                _torch_per_player_cosine_distance(pred_aim_raw, tgt_aim_raw),
                lambdas["aim"],
            )
        else:
            # OLD behavior (aim_loss='mse', or no aim sub-group): MSE over the whole
            # enabled player slice, aim included, as a single 'player' term.
            add("player", mse(pred_player, tgt_player), lambdas["player"])

    if cfg.is_on("bullet_presence"):
        p = preds["bullet_presence"]
        pw = torch.as_tensor(presence_pw, dtype=p.dtype, device=p.device)
        add(
            "presence",
            bce(p, targets["bullet_presence"], pos_weight=pw),
            lambdas["presence"],
        )

    mask = targets.get("bullet_slot_mask")
    if cfg.is_on("bullet_position"):
        sq = (preds["bullet_position"] - targets["bullet_position"]) ** 2 * mask
        add("bullet_pos", sq.sum() / mask.sum().clamp_min(1.0), lambdas["bullet_pos"])

    if cfg.is_on("bullet_direction"):
        sq = (preds["bullet_direction"] - targets["bullet_direction"]) ** 2 * mask
        add("bullet_dir", sq.sum() / mask.sum().clamp_min(1.0), lambdas["bullet_dir"])

    if cfg.is_on("walls"):
        wp = preds["wall"]  # (B, 12, 20) logits
        wt = targets["wall"]  # (B, 12, 20) {0,1}
        # Interior-only BCE: select the 128 interior cells so the constant border
        # (identical on every map, info-free) contributes EXACTLY 0 to the loss.
        interior = _interior_wall_index(wp.device)  # (12,20) bool on device
        wpw = torch.as_tensor(wall_pw, dtype=wp.dtype, device=wp.device)
        per_cell = bce(wp, wt, pos_weight=wpw, reduction="none")  # (B,12,20)
        sel = per_cell[:, interior]  # (B, 128)
        add("wall", sel.mean(), lambdas["wall"])

    if not terms:  # pragma: no cover - parse_objectives forbids an empty config
        raise ValueError("no objectives enabled; nothing to optimize")
    total = terms[0] if len(terms) == 1 else torch.stack(terms).sum()
    parts["total"] = float(total.detach())
    return total, parts


# Cache the interior-wall boolean mask as a torch tensor per device (the numpy mask is
# constant; this avoids rebuilding it every batch).
_INTERIOR_WALL_CACHE: dict[torch.device, torch.Tensor] = {}


def _interior_wall_index(device: torch.device) -> torch.Tensor:
    """The (12,20) interior-cell boolean mask as a torch tensor on ``device`` (cached)."""
    cached = _INTERIOR_WALL_CACHE.get(device)
    if cached is None:
        cached = torch.from_numpy(interior_wall_mask()).to(device)
        _INTERIOR_WALL_CACHE[device] = cached
    return cached


# =============================================================================
# Datasets
# =============================================================================
class _BaseRows:
    """Holds resolved rows: frames (downsampled uint8), states, map_ids; targets cached."""

    def __init__(
        self,
        frames: np.ndarray,  # (N, H, W, 3) uint8
        states: np.ndarray,  # (N, 52) f32
        map_ids: np.ndarray,  # (N,) int
        wall_grids: np.ndarray,  # (10,12,20)
        stats: NormStats,
    ) -> None:
        self.frames = frames
        self.states = states.astype(np.float32, copy=False)
        self.map_ids = map_ids.astype(np.int64, copy=False)
        self.wall_grids = wall_grids
        self.stats = stats
        self._cache = decode_targets(self.states, self.map_ids, self.wall_grids, self.stats)

    def __len__(self) -> int:
        return self.frames.shape[0]


def _row_item(frame: torch.Tensor, cache: dict[str, np.ndarray], i: int) -> dict[str, torch.Tensor]:
    return {
        "frame": frame,
        "player": torch.from_numpy(cache["player"][i]),
        "bullet_presence": torch.from_numpy(cache["bullet_presence"][i]),
        "bullet_position": torch.from_numpy(cache["bullet_position"][i]),
        "bullet_direction": torch.from_numpy(cache["bullet_direction"][i]),
        "bullet_slot_mask": torch.from_numpy(cache["bullet_slot_mask"][i]),
        "wall": torch.from_numpy(cache["wall"][i]),
    }


class InMemoryPixelDataset(Dataset, _BaseRows):
    """In-memory dataset (SMOKE streamer or a small slice). Targets precomputed."""

    def __init__(self, *a, **kw) -> None:
        Dataset.__init__(self)
        _BaseRows.__init__(self, *a, **kw)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        frame = torch.from_numpy(self.frames[i]).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        return _row_item(frame, self._cache, i)


class MemmapPixelDataset(Dataset):
    """Reads pre-resized frames from a uint8 memmap + aligned states/map_ids.

    The cache dir holds ``frames_u8.dat`` (memmap, (N,H,W,3)), ``states.npy``,
    ``map_ids.npy``, ``group_keys.npy`` (N,2), and ``index.json`` with shape/res.
    """

    def __init__(
        self,
        cache_dir: Path,
        row_idx: np.ndarray,
        wall_grids: np.ndarray,
        stats: NormStats,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        n, h, w, c = meta["shape"]
        self.frames = np.memmap(
            self.cache_dir / "frames_u8.dat", dtype=np.uint8, mode="r", shape=(n, h, w, c)
        )
        self.states = np.load(self.cache_dir / "states.npy")
        self.map_ids = np.load(self.cache_dir / "map_ids.npy")
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.wall_grids = wall_grids
        self.stats = stats
        sel_states = self.states[self.row_idx]
        sel_maps = self.map_ids[self.row_idx]
        self._cache = decode_targets(sel_states, sel_maps, self.wall_grids, self.stats)

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.row_idx[i])
        frame = torch.from_numpy(np.array(self.frames[r], copy=True)).permute(2, 0, 1).contiguous()
        return _row_item(frame, self._cache, i)


# Process-local LMDB env cache, keyed by (pid, resolved-path). LMDB refuses to open the
# SAME env file twice within one process, and the train/eval/val datasets all read the
# same file — so they must SHARE one handle per process. Keying by pid (not just path)
# makes it spawn-safe: a freshly spawned DataLoader worker starts with an empty dict and
# opens its OWN handle (the parent's handle never survives pickling). lock=False allows
# the many concurrent readers; the file is read-only here.
_LMDB_ENV_CACHE: dict[tuple[int, str], object] = {}


def _open_lmdb_env(lmdb_path: Path):
    """Return THIS process's shared read-only env for ``lmdb_path`` (opened once per pid)."""
    import lmdb

    key = (os.getpid(), str(lmdb_path))
    env = _LMDB_ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=0,
        )
        _LMDB_ENV_CACHE[key] = env
    return env


class LmdbPngPixelDataset(Dataset):
    """Reads PNG-encoded frames from an LMDB env + aligned states/map_ids (PNG-in-LMDB).

    Parallel to :class:`MemmapPixelDataset`: SAME ``__init__`` signature, SAME target
    cache, and SAME ``(frame, targets)`` return — it yields a uint8 ``(3,H,W)`` frame
    tensor (NOT pre-divided; ``_move_batch`` divides by 255 downstream), bit-identical to
    what the memmap path returns for the same row. The cache dir holds
    ``frames_png.lmdb`` (LMDB env written by scripts/transcode_pixel_png_lmdb.py),
    ``states.npy``, ``map_ids.npy``, ``group_keys.npy`` (N,2), and ``index.json`` with
    ``backend == "lmdb_png"`` and the key convention.

    Windows / spawn safety: an LMDB env handle does NOT survive pickling to a spawned
    DataLoader worker, so the env is opened LAZILY on first ``__getitem__`` via the
    process-local :data:`_LMDB_ENV_CACHE` (keyed by ``(pid, path)``). Each worker — and the
    main process — therefore opens exactly ONE read-only handle and SHARES it across the
    train/eval/val datasets (LMDB forbids opening the same env file twice per process). The
    env is ``readonly=True, lock=False`` (many concurrent readers, no writer).
    """

    def __init__(
        self,
        cache_dir: Path,
        row_idx: np.ndarray,
        wall_grids: np.ndarray,
        stats: NormStats,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        n, h, w, c = meta["shape"]
        self.shape = (int(n), int(h), int(w), int(c))
        self.lmdb_path = self.cache_dir / meta.get("lmdb_file", "frames_png.lmdb")
        self.key_format = meta.get("key_format", "{:08d}")
        self.states = np.load(self.cache_dir / "states.npy")
        self.map_ids = np.load(self.cache_dir / "map_ids.npy")
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.wall_grids = wall_grids
        self.stats = stats
        sel_states = self.states[self.row_idx]
        sel_maps = self.map_ids[self.row_idx]
        self._cache = decode_targets(sel_states, sel_maps, self.wall_grids, self.stats)

    def _key(self, r: int) -> bytes:
        return self.key_format.format(r).encode("ascii")

    def _get_env(self):
        """Return this process's shared read-only LMDB env (opened lazily once per pid)."""
        return _open_lmdb_env(self.lmdb_path)

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.row_idx[i])
        with self._get_env().begin(write=False) as txn:
            data = txn.get(self._key(r))
        if data is None:
            raise KeyError(f"LMDB key for row {r} ({self._key(r)!r}) missing in {self.lmdb_path}")
        arr = png_decode(data)  # (H, W, 3) uint8 (writable copy)
        frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return _row_item(frame, self._cache, i)


# =============================================================================
# Streaming loader for the SMOKE: map-BALANCED so all 10 maps are surfaced
# =============================================================================
def _stream_map_balanced(
    data_dir: Path,
    out_w: int,
    out_h: int,
    subset: int,
    n_maps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Pull ~``subset`` pairs with a per-map QUOTA so every one of ``n_maps`` appears.

    Walks shards (interleaved across workers), bucketing rows by map_id until each map
    reaches its quota (``subset // n_maps``, >=1). This guarantees the smoke 3-way split
    can surface the 8 train maps + 2 validation maps with nonzero counts each. Returns
    ``(frames_ds, states, map_ids, group_keys)`` where group_keys are
    ``(worker_id, episode_id)`` aligned to rows.
    """
    shards = list_shards(data_dir)
    rng = np.random.default_rng(seed)
    by_worker: dict[int, list[Path]] = {}
    for s in shards:
        by_worker.setdefault(worker_id_from_shard(s.name), []).append(s)
    order: list[Path] = []
    idxs = {w: 0 for w in by_worker}
    while len(order) < len(shards):
        for w in sorted(by_worker):
            if idxs[w] < len(by_worker[w]):
                order.append(by_worker[w][idxs[w]])
                idxs[w] += 1

    per_map_quota = max(1, subset // max(1, n_maps))
    # Accumulators bucketed by map_id.
    frames_by_map: dict[int, list[np.ndarray]] = {}
    states_by_map: dict[int, list[np.ndarray]] = {}
    groups_by_map: dict[int, list[np.ndarray]] = {}
    counts: dict[int, int] = {}

    def remaining_maps() -> set[int]:
        return {m for m in range(n_maps) if counts.get(m, 0) < per_map_quota}

    for s in order:
        if not remaining_maps():
            break
        wid = worker_id_from_shard(s.name)
        with np.load(s) as d:
            mids = d["map_ids"].astype(np.int64)
            want = remaining_maps()
            in_want = np.isin(mids, list(want))
            if not in_want.any():
                continue
            for m in want:
                rows = np.flatnonzero(mids == m)
                if rows.size == 0:
                    continue
                need = per_map_quota - counts.get(m, 0)
                take = min(need, rows.size)
                sel = np.sort(rng.choice(rows, size=take, replace=False))
                fr = downsample_frames_np(d["frames"][sel], out_w, out_h)
                st = d["states"][sel].astype(np.float32)
                ep = d["episode_ids"][sel].astype(np.int64)
                frames_by_map.setdefault(m, []).append(fr)
                states_by_map.setdefault(m, []).append(st)
                groups_by_map.setdefault(m, []).append(
                    np.stack([np.full(take, wid, dtype=np.int64), ep], axis=1)
                )
                counts[m] = counts.get(m, 0) + take

    frames_acc: list[np.ndarray] = []
    states_acc: list[np.ndarray] = []
    maps_acc: list[np.ndarray] = []
    groups_acc: list[np.ndarray] = []
    for m in sorted(frames_by_map):
        fr = np.concatenate(frames_by_map[m], axis=0)
        frames_acc.append(fr)
        states_acc.append(np.concatenate(states_by_map[m], axis=0))
        maps_acc.append(np.full(fr.shape[0], m, dtype=np.int64))
        groups_acc.append(np.concatenate(groups_by_map[m], axis=0))

    frames = np.concatenate(frames_acc, axis=0)
    states = np.concatenate(states_acc, axis=0)
    map_ids = np.concatenate(maps_acc, axis=0)
    groups_arr = np.concatenate(groups_acc, axis=0)
    group_keys = [(int(w), int(e)) for w, e in groups_arr]
    missing = [m for m in range(n_maps) if counts.get(m, 0) == 0]
    if missing:
        raise RuntimeError(
            f"smoke stream could not surface map_ids {missing} from {data_dir}; "
            f"increase --subset (got {subset}, quota/map={per_map_quota})"
        )
    return frames, states, map_ids, group_keys


# =============================================================================
# Train / eval loop
# =============================================================================
def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out["frame"] = out["frame"].float() / 255.0  # uint8 (B,3,H,W) -> float [0,1]
    return out


def _na_breakdown(cfg: ObjectiveConfig) -> dict[str, object]:
    """Per-objective dict skeleton with all 7 groups; disabled groups -> "N/A"."""
    return {
        g: "N/A"
        for g in (
            "player_position",
            "player_velocity",
            "player_aim",
            "bullet_presence",
            "bullet_position",
            "bullet_direction",
            "walls",
        )
        if not cfg.is_on(g)
    }


@torch.no_grad()
def _evaluate(
    model: InverseRenderer,
    loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device,
    stats: NormStats,
    use_amp: bool,
    desc: str,
    presence_pw: float | torch.Tensor,
    wall_pw: float | torch.Tensor,
    aim_loss: str = "cosine",
) -> dict[str, object]:
    """Run the loss + the full per-objective breakdown over ``loader``.

    Disabled groups report ``"N/A"``. POSITION groups also report world-unit error.
    bullet_presence -> precision/recall/F1; walls -> INTERIOR F1/precision/recall/IoU
    (the wall loss is interior-only, so the border head output is unsupervised /
    meaningless and ``overall_acc`` is intentionally NOT reported).
    """
    model.eval()
    cfg = model.cfg
    interior = interior_wall_mask()  # (12,20) bool, 128 cells

    loss_total = 0.0
    n_batches = 0

    # Per-field normalized-MSE accumulators (sum of squared err / count).
    pl_sse = np.zeros(12)
    pl_world_sse = np.zeros(12)
    pl_n = 0
    # Aim angular-error accumulators (per-player, on RAW de-standardized aim).
    aim_deg_sum = 0.0
    aim_cos_sum = 0.0
    aim_n = 0  # number of (sample, player) cosine measurements summed
    bpos_sse = np.zeros(N_BULLET_POS)
    bpos_world_sae = np.zeros(N_BULLET_POS)
    bpos_cnt = np.zeros(N_BULLET_POS)
    bdir_sse = np.zeros(N_BULLET_DIR)
    bdir_cnt = np.zeros(N_BULLET_DIR)
    # Presence: collect predictions/targets for F1.
    pres_tp = pres_fp = pres_fn = 0
    # Walls: interior F1/IoU (no overall accuracy — border is unsupervised now).
    wall_tp = wall_fp = wall_fn = wall_inter = wall_union = 0

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        b = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(b["frame"])
            total, _parts = compute_losses(
                preds, b, model, lambdas, presence_pw, wall_pw, stats=stats, aim_loss=aim_loss
            )
        loss_total += float(total.detach())
        n_batches += 1

        if cfg.any_player():
            idx = model.player_vec_indices
            pred = preds["player"].float().cpu().numpy()  # (B, k) normalized
            tgt = b["player"][:, idx].float().cpu().numpy()
            pl_sse[idx] += ((pred - tgt) ** 2).sum(axis=0)
            pw = denormalize(pred, stats.player_mean[idx], stats.player_std[idx])
            tw = denormalize(tgt, stats.player_mean[idx], stats.player_std[idx])
            pl_world_sse[idx] += ((pw - tw) ** 2).sum(axis=0)
            pl_n += pred.shape[0]
            if cfg.is_on("player_aim"):
                # Aim metric on RAW (de-standardized) aim: pull the aim columns out of
                # the head output and de-standardize both pred and target to unit space.
                aim_cols = _aim_head_columns(model)
                aim_vi = list(_AIM_VEC_INDICES)
                pred_aim_raw = denormalize(
                    pred[:, aim_cols], stats.player_mean[aim_vi], stats.player_std[aim_vi]
                )
                tgt_aim_raw = denormalize(
                    tgt[:, aim_cols], stats.player_mean[aim_vi], stats.player_std[aim_vi]
                )
                m = aim_angular_error_deg(pred_aim_raw, tgt_aim_raw)
                # Weight each batch's mean by its (sample * 2-player) measurement count.
                bm = pred.shape[0] * 2
                aim_deg_sum += m["deg"] * bm
                aim_cos_sum += m["cos"] * bm
                aim_n += bm

        mask = b["bullet_slot_mask"].float().cpu().numpy() if "bullet_slot_mask" in b else None
        if cfg.is_on("bullet_position"):
            pred = preds["bullet_position"].float().cpu().numpy()
            tgt = b["bullet_position"].float().cpu().numpy()
            bpos_sse += ((pred - tgt) ** 2 * mask).sum(axis=0)
            pw = denormalize(pred, stats.bullet_pos_mean, stats.bullet_pos_std)
            tw = denormalize(tgt, stats.bullet_pos_mean, stats.bullet_pos_std)
            bpos_world_sae += (np.abs(pw - tw) * mask).sum(axis=0)
            bpos_cnt += mask.sum(axis=0)
        if cfg.is_on("bullet_direction"):
            pred = preds["bullet_direction"].float().cpu().numpy()
            tgt = b["bullet_direction"].float().cpu().numpy()
            bdir_sse += ((pred - tgt) ** 2 * mask).sum(axis=0)
            bdir_cnt += mask.sum(axis=0)
        if cfg.is_on("bullet_presence"):
            pred = (preds["bullet_presence"].float().cpu().numpy() > 0).astype(np.int64)
            tgt = b["bullet_presence"].float().cpu().numpy().astype(np.int64)
            pres_tp += int(((pred == 1) & (tgt == 1)).sum())
            pres_fp += int(((pred == 1) & (tgt == 0)).sum())
            pres_fn += int(((pred == 0) & (tgt == 1)).sum())
        if cfg.is_on("walls"):
            pred = (preds["wall"].float().cpu().numpy() > 0).astype(np.int64)  # (B,12,20)
            tgt = b["wall"].float().cpu().numpy().astype(np.int64)
            pi = pred[:, interior].ravel()
            ti = tgt[:, interior].ravel()
            wall_tp += int(((pi == 1) & (ti == 1)).sum())
            wall_fp += int(((pi == 1) & (ti == 0)).sum())
            wall_fn += int(((pi == 0) & (ti == 1)).sum())
            wall_inter += int(((pi == 1) & (ti == 1)).sum())
            wall_union += int(((pi == 1) | (ti == 1)).sum())

    out: dict[str, object] = {"total_loss": loss_total / max(1, n_batches)}
    breakdown: dict[str, object] = _na_breakdown(cfg)

    # Player sub-groups.
    for sub in _PLAYER_SUBGROUPS:
        if not cfg.is_on(sub):
            continue
        if sub == "player_aim":
            # Aim reports ANGULAR error (deg) + cosine similarity instead of norm_mse:
            # on a ~uniform unit target, norm_mse pins at ~1.0 regardless of signal, so
            # it cannot distinguish "learning direction" from "collapsed to (0,0)".
            breakdown[sub] = {
                "aim_angular_error_deg": float(aim_deg_sum / max(1, aim_n)),
                "aim_cosine": float(aim_cos_sum / max(1, aim_n)),
            }
            continue
        vi = list(PLAYER_SUBGROUP_VEC_INDICES[sub])
        nmse = float((pl_sse[vi] / max(1, pl_n)).mean())
        entry: dict[str, float] = {"norm_mse": nmse}
        if sub in _PLAYER_POSITION_GROUPS:
            entry["world_rmse"] = float(np.sqrt((pl_world_sse[vi] / max(1, pl_n)).mean()))
        breakdown[sub] = entry

    if cfg.is_on("bullet_presence"):
        prec = pres_tp / (pres_tp + pres_fp) if (pres_tp + pres_fp) else 0.0
        rec = pres_tp / (pres_tp + pres_fn) if (pres_tp + pres_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        breakdown["bullet_presence"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": pres_tp,
            "fp": pres_fp,
            "fn": pres_fn,
        }
    if cfg.is_on("bullet_position"):
        nmse = float((bpos_sse / np.clip(bpos_cnt, 1, None)).mean())
        world_mae = float(bpos_world_sae.sum() / max(1.0, bpos_cnt.sum()))
        breakdown["bullet_position"] = {"norm_mse": nmse, "world_mae": world_mae}
    if cfg.is_on("bullet_direction"):
        nmse = float((bdir_sse / np.clip(bdir_cnt, 1, None)).mean())
        breakdown["bullet_direction"] = {"norm_mse": nmse}
    if cfg.is_on("walls"):
        prec = wall_tp / (wall_tp + wall_fp) if (wall_tp + wall_fp) else 0.0
        rec = wall_tp / (wall_tp + wall_fn) if (wall_tp + wall_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = wall_inter / wall_union if wall_union else 1.0
        breakdown["walls"] = {
            "interior_f1": f1,
            "interior_precision": prec,
            "interior_recall": rec,
            "interior_iou": iou,
        }

    out["per_objective"] = breakdown
    return out


def _fmt_breakdown(b: dict[str, object]) -> str:
    """One-line human summary of the per-objective breakdown (N/A for disabled)."""
    parts = []
    for g in ("player_position", "player_velocity", "player_aim"):
        v = b[g]
        if v == "N/A":
            parts.append(f"{g}=N/A")
        elif g == "player_aim":
            parts.append(f"{g}=ang{v['aim_angular_error_deg']:.1f}deg/cos{v['aim_cosine']:.3f}")
        else:
            extra = f"/world={v['world_rmse']:.3f}" if "world_rmse" in v else ""
            parts.append(f"{g}=nmse{v['norm_mse']:.3f}{extra}")
    for g in ("bullet_presence",):
        v = b[g]
        parts.append(f"{g}=N/A" if v == "N/A" else f"{g}=F1{v['f1']:.3f}")
    bp = b["bullet_position"]
    parts.append(
        "bullet_position=N/A"
        if bp == "N/A"
        else f"bullet_position=nmse{bp['norm_mse']:.3f}/world{bp['world_mae']:.3f}"
    )
    bd = b["bullet_direction"]
    parts.append(
        "bullet_direction=N/A" if bd == "N/A" else f"bullet_direction=nmse{bd['norm_mse']:.3f}"
    )
    w = b["walls"]
    if w == "N/A":
        parts.append("walls=N/A")
    else:
        parts.append(f"walls=intF1{w['interior_f1']:.3f}/IoU{w['interior_iou']:.3f}")
    return "  ".join(parts)


# =============================================================================
# Dataset assembly: 3-way map-aware split over both data paths
# =============================================================================
def _build_in_memory_datasets(
    args: Namespace, wall_grids: np.ndarray, holdout_ids: list[int], n_maps: int
):
    """SMOKE path: map-balanced stream, then 3-way map-aware split; fit stats on TRAIN."""
    out_w, out_h = args.input_wh
    subset = args.subset if args.subset is not None else 8000
    frames, states, map_ids, group_keys = _stream_map_balanced(
        args.data_dir, out_w, out_h, subset, n_maps, args.seed
    )
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids, strict=True)}
    split = map_aware_split(
        group_keys, map_of_group, holdout_ids, eval_frac=args.eval_frac, seed=args.seed
    )
    train_mask = np.array([g in split.train for g in group_keys])
    eval_mask = np.array([g in split.eval for g in group_keys])
    val_mask = np.array([g in split.validation for g in group_keys])
    stats = fit_norm_stats(states[train_mask])

    def mk(m: np.ndarray) -> InMemoryPixelDataset:
        return InMemoryPixelDataset(frames[m], states[m], map_ids[m], wall_grids, stats)

    return mk(train_mask), mk(eval_mask), mk(val_mask), stats, split


# Maps the --frames-backend flag to the cache-backed Dataset class. Both classes share
# the IDENTICAL __init__(cache_dir, row_idx, wall_grids, stats) contract and return the
# same uint8 (3,H,W) frame + targets, so only the class differs — the split/stats code in
# _build_cache_datasets is shared (not copy-pasted).
_CACHE_DATASET_CLASSES = {
    "memmap": MemmapPixelDataset,
    "lmdb_png": LmdbPngPixelDataset,
}


def _build_cache_datasets(
    args: Namespace, wall_grids: np.ndarray, holdout_ids: list[int], frames_backend: str
):
    """FULL path: read cache sidecars, 3-way map-aware split by group, fit stats on TRAIN.

    Backend-agnostic: the sidecars (states/map_ids/group_keys/index.json) are IDENTICAL
    across the memmap and lmdb_png caches, so the split + stats are computed once here and
    the only difference is which Dataset class reads the frames (selected by
    ``frames_backend`` via :data:`_CACHE_DATASET_CLASSES`).
    """
    ds_cls = _CACHE_DATASET_CLASSES[frames_backend]
    cache = Path(args.cache_dir)
    group_keys_arr = np.load(cache / "group_keys.npy")  # (N, 2)
    map_ids = np.load(cache / "map_ids.npy")
    states = np.load(cache / "states.npy")
    n = group_keys_arr.shape[0]
    if args.subset is not None:
        n = min(n, args.subset)
    all_rows = np.arange(n)
    group_keys = [(int(w), int(e)) for w, e in group_keys_arr[:n]]
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids[:n], strict=True)}
    split = map_aware_split(
        group_keys, map_of_group, holdout_ids, eval_frac=args.eval_frac, seed=args.seed
    )
    train_rows = all_rows[[g in split.train for g in group_keys]]
    eval_rows = all_rows[[g in split.eval for g in group_keys]]
    val_rows = all_rows[[g in split.validation for g in group_keys]]
    stats = fit_norm_stats(states[train_rows])
    ds_train = ds_cls(cache, train_rows, wall_grids, stats)
    ds_eval = ds_cls(cache, eval_rows, wall_grids, stats)
    ds_val = ds_cls(cache, val_rows, wall_grids, stats)
    return ds_train, ds_eval, ds_val, stats, split


def _resolve_run_dir(out_base: Path, out_w: int, out_h: int, emb: int, batch: int) -> Path:
    """Append a unique ``_{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM}`` suffix. Never overwrite."""
    stamp = datetime.now().strftime("%m%d-%H%M")
    name = f"{out_base.name}_{out_w}x{out_h}_e{emb}_b{batch}_{stamp}"
    run_dir = out_base.parent / name
    if run_dir.exists():
        raise SystemExit(
            f"[pretrain] refusing to overwrite existing run dir: {run_dir}. "
            "Pick a different --out base or wait a minute (timestamp differs)."
        )
    return run_dir


def run_training(args: Namespace) -> int:
    """Build data, train, eval each epoch, validate once, save artifacts. Returns exit code."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg: ObjectiveConfig = args.objective_config
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"[pretrain] device={device} amp={use_amp} torch={torch.__version__}")
    print(f"[pretrain] objectives ENABLED: {list(cfg.ordered())}")
    from tank_twin.pretrain_pixels import OBJECTIVE_GROUPS

    disabled = [g for g in OBJECTIVE_GROUPS if not cfg.is_on(g)]
    print(f"[pretrain] objectives DISABLED (reported N/A): {disabled}")

    out_w, out_h = args.input_wh
    run_dir = _resolve_run_dir(Path(args.out), out_w, out_h, args.embedding_dim, args.batch)
    print(f"[pretrain] resolved run dir -> {run_dir}")

    wall_grids = np.load(args.data_dir / "wall_grids.npy")
    n_maps = wall_grids.shape[0]
    name_to_id = load_map_name_to_id(args.data_dir)
    holdout_ids = resolve_map_ids(args.holdout_map_names, name_to_id)
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(
        f"[pretrain] holdout (validation) maps: {[(id_to_name[i], i) for i in sorted(holdout_ids)]}"
    )

    # frames_backend only applies to the cache (--cache-dir) path; the SMOKE stream is
    # unaffected. Default "memmap" preserves the current behavior.
    frames_backend = getattr(args, "frames_backend", "memmap")
    t0 = time.time()
    if args.cache_dir is not None:
        ds_train, ds_eval, ds_val, stats, split = _build_cache_datasets(
            args, wall_grids, holdout_ids, frames_backend
        )
        mode = frames_backend
    else:
        ds_train, ds_eval, ds_val, stats, split = _build_in_memory_datasets(
            args, wall_grids, holdout_ids, n_maps
        )
        mode = "stream"

    train_map_names = [id_to_name.get(i, str(i)) for i in split.train_map_ids]
    val_map_names = [id_to_name.get(i, str(i)) for i in split.holdout_map_ids]
    print(
        f"[pretrain] mode={mode} res={out_w}x{out_h} | 3-way split: "
        f"train={len(ds_train)} rows ({len(split.train)} groups) "
        f"eval={len(ds_eval)} rows ({len(split.eval)} groups) "
        f"validation={len(ds_val)} rows ({len(split.validation)} groups)"
    )
    print(
        f"[pretrain] train maps ({len(train_map_names)}): {train_map_names} | "
        f"validation maps ({len(val_map_names)}): {val_map_names} "
        f"(data prep {time.time() - t0:.1f}s)"
    )

    # --- Resolve class-imbalance BCE pos_weights (TRAIN split ONLY; no leakage) ------
    # ds_train._cache holds the TRAIN split's decoded targets (presence (N,10), wall
    # (N,12,20)); use those so the weights are train-level, computed ONCE here and
    # threaded into every loss call (NOT recomputed per-batch).
    presence_override = parse_pos_weight_arg(args.presence_pos_weight)
    wall_override = parse_pos_weight_arg(args.wall_pos_weight)
    train_presence = ds_train._cache["bullet_presence"]  # (N, 10) {0,1}
    train_walls = ds_train._cache["wall"]  # (N, 12, 20) {0,1}
    presence_pw = (
        presence_override if presence_override is not None else presence_pos_weight(train_presence)
    )
    wall_pw = wall_override if wall_override is not None else interior_wall_pos_weight(train_walls)
    presence_src = "override" if presence_override is not None else "auto(train)"
    wall_src = "override" if wall_override is not None else "auto(train)"
    print(
        f"[pretrain] BCE pos_weights -> presence={presence_pw:.4f} ({presence_src}), "
        f"wall(interior)={wall_pw:.4f} ({wall_src})"
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    eval_loader = DataLoader(
        ds_eval,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = InverseRenderer(out_h, out_w, cfg, args.embedding_dim).to(device)
    # AdamW = decoupled weight decay (the correct form); weight_decay=0.0 (default) makes
    # this behave like the old Adam, so `--lr 3e-4 --lr-schedule constant` is unchanged.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- LR schedule (per-optimizer-step LambdaLR over the WHOLE run) ----------------
    # Total optimizer steps = steps-per-epoch * epochs. train_loader uses drop_last=False,
    # so steps-per-epoch == len(train_loader) (number of batches). For 'constant' the
    # multiplier is a no-op 1.0 (byte-equivalent to no scheduler); for 'cosine' it is the
    # pure linear-warmup -> cosine-anneal multiplier. The scheduler is stepped ONCE per
    # SUCCESSFUL optimizer step (guarded against AMP-skipped steps), so its internal step
    # counter == the optimizer-step index passed to the multiplier helper.
    warmup_frac = 0.05
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(max(1, round(warmup_frac * total_steps)), total_steps)
    if args.lr_schedule == "cosine":

        def _lr_lambda(opt_step: int) -> float:
            return cosine_warmup_lr_multiplier(opt_step, total_steps, warmup_frac)
    else:

        def _lr_lambda(opt_step: int) -> float:  # constant: no-op multiplier
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    print(
        f"[pretrain] lr={args.lr} schedule={args.lr_schedule} weight_decay={args.weight_decay} "
        f"(total_steps={total_steps}, warmup_steps={warmup_steps})"
    )
    lambdas = {
        "player": args.lambda_player,
        "aim": args.lambda_aim,
        "presence": args.lambda_presence,
        "bullet_pos": args.lambda_bullet_pos,
        "bullet_dir": args.lambda_bullet_dir,
        "wall": args.lambda_wall,
    }
    aim_loss = args.aim_loss
    if cfg.is_on("player_aim"):
        print(f"[pretrain] player_aim loss={aim_loss} lambda_aim={args.lambda_aim}")
    print(
        f"[pretrain] encoder flatten_dim={model.encoder.flatten_dim} "
        f"embedding_dim={model.encoder.embedding_dim} heads={list(model.sizes.items())} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    eval_history: list[dict] = []

    global_step = 0
    seen = 0
    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_total = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in pbar:
            b = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(b["frame"])
                total, parts = compute_losses(
                    preds, b, model, lambdas, presence_pw, wall_pw, stats=stats, aim_loss=aim_loss
                )
            scaler.scale(total).backward()
            # AMP can SKIP an optimizer step on inf/nan grads. Detect a real step by
            # comparing the GradScaler scale across update(): on a skipped step the scaler
            # lowers the scale (scale_after < scale_before); on a real step it is unchanged
            # or raised. Only advance the LR schedule when the optimizer actually stepped,
            # so a skipped step does not consume a slot of the warmup/cosine schedule.
            # (With AMP disabled the scale is constant 1.0, so the step always counts.)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
            if stepped:
                scheduler.step()
            ep_total += float(total.detach())
            n_batches += 1
            global_step += 1
            seen += b["frame"].shape[0]
            pbar.set_postfix(total=f"{float(total.detach()):.3f}")
            if global_step % args.log_every == 0:
                tqdm.write(
                    f"  [e{epoch} s{global_step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in parts.items() if k != "total")
                    + f" total={parts['total']:.4f}"
                )
        ep_total /= max(1, n_batches)
        ev = _evaluate(
            model,
            eval_loader,
            lambdas,
            device,
            stats,
            use_amp,
            desc="  eval",
            presence_pw=presence_pw,
            wall_pw=wall_pw,
            aim_loss=aim_loss,
        )
        ev["epoch"] = epoch
        ev["train_total_loss"] = ep_total
        eval_history.append(ev)
        print(
            f"[epoch {epoch}] TRAIN total={ep_total:.4f} || "
            f"EVAL total={ev['total_loss']:.4f} | {_fmt_breakdown(ev['per_objective'])}"
        )

    train_secs = time.time() - train_t0
    pairs_per_sec = seen / max(1e-9, train_secs)
    print(
        f"[pretrain] trained {seen} pair-steps in {train_secs:.1f}s "
        f"({pairs_per_sec:.0f} pairs/s, {global_step / max(1e-9, train_secs):.1f} steps/s)"
    )

    # --- Final out-of-distribution VALIDATION (the 2 held-out maps), run ONCE ----
    print("[pretrain] running FINAL out-of-distribution VALIDATION (2 held-out maps)...")
    validation = _evaluate(
        model,
        val_loader,
        lambdas,
        device,
        stats,
        use_amp,
        desc="  validation",
        presence_pw=presence_pw,
        wall_pw=wall_pw,
        aim_loss=aim_loss,
    )
    print(
        f"[validation] OOD total={validation['total_loss']:.4f} | "
        f"{_fmt_breakdown(validation['per_objective'])}"
    )

    # --- Save artifacts ------------------------------------------------------
    encoder_path = run_dir / "encoder.pt"
    full_path = run_dir / "model.pt"
    stats_path = run_dir / "norm_stats.json"
    split_path = run_dir / "split.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.state_dict(), full_path)
    stats_path.write_text(json.dumps(stats.to_json(), indent=2), encoding="utf-8")
    split_payload = split.to_json()
    split_payload.update(
        {
            "train_map_names": train_map_names,
            "validation_map_names": val_map_names,
            "eval_frac": args.eval_frac,
            "holdout_maps": args.holdout_map_names,
            "seed": args.seed,
        }
    )
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    config_dict = {
        "input_res": f"{out_w}x{out_h}",
        "embedding_dim": args.embedding_dim,
        "flatten_dim": model.encoder.flatten_dim,
        "objectives": cfg.to_json(),
        "head_sizes": model.sizes,
        "lambdas": lambdas,
        "aim_loss": aim_loss,
        "pos_weights": {
            "presence": float(presence_pw),
            "wall_interior": float(wall_pw),
            "presence_source": presence_src,
            "wall_source": wall_src,
        },
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "lr_schedule_warmup_frac": warmup_frac if args.lr_schedule == "cosine" else None,
        "lr_schedule_warmup_steps": warmup_steps if args.lr_schedule == "cosine" else None,
        "lr_schedule_total_steps": total_steps if args.lr_schedule == "cosine" else None,
        "epochs": args.epochs,
        "eval_frac": args.eval_frac,
        "holdout_maps": args.holdout_map_names,
        "seed": args.seed,
        "mode": mode,
        "frames_backend": frames_backend,
        "device": str(device),
    }
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    metrics_payload = {
        "config": config_dict,
        "eval": eval_history,
        "validation": validation,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"[pretrain] saved encoder -> {encoder_path}")
    print(f"[pretrain] saved full model -> {full_path}")
    print(f"[pretrain] saved norm stats -> {stats_path}")
    print(f"[pretrain] saved 3-way split -> {split_path}")
    print(f"[pretrain] saved metrics (eval history + validation) -> {metrics_path}")

    # --- Verify the encoder reloads STANDALONE into a fresh PixelEncoder -----
    # (the encoder is objective-config-independent; this check does not touch heads.)
    fresh = PixelEncoder(out_h, out_w, args.embedding_dim)
    sd = torch.load(encoder_path, map_location="cpu", weights_only=True)
    fresh.load_state_dict(sd)  # strict=True: raises on any mismatch
    fresh.eval()
    with torch.no_grad():
        emb = fresh(torch.zeros(2, 3, out_h, out_w))
    assert emb.shape == (2, args.embedding_dim), emb.shape
    print(
        f"[pretrain] encoder reload OK: fresh PixelEncoder forward -> {tuple(emb.shape)} "
        f"(expected (2, {args.embedding_dim})); encoder keys={sorted(sd.keys())[:2]}... "
        f"(objective-config-independent)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(1)
