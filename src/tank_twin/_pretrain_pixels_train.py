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
    N_BULLET_DIR,
    N_BULLET_POS,
    WALL_H,
    WALL_W,
    NormStats,
    ObjectiveConfig,
    decode_targets,
    denormalize,
    downsample_frames_np,
    fit_norm_stats,
    head_sizes,
    interior_wall_mask,
    list_shards,
    load_map_name_to_id,
    map_aware_split,
    player_head_vec_indices,
    resolve_map_ids,
    worker_id_from_shard,
)

# Player sub-group -> (name, position?) and its slice within the player HEAD output.
_PLAYER_SUBGROUPS = ("player_position", "player_velocity", "player_aim")
_PLAYER_POSITION_GROUPS = {"player_position"}


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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total + per-component loss over ENABLED objectives only.

    * player: MSE on the normalized enabled-sub-group slice of the 12-vector target.
    * bullet_presence: BCEWithLogits on 10 slot bits.
    * bullet_position / bullet_direction: MSE on 20 normalized floats, MASKED to present
      slots (absent slots contribute 0; mean over present floats only).
    * walls: per-cell BCEWithLogits over the (12,20) grid.
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
        add("player", mse(preds["player"], targets["player"][:, idx]), lambdas["player"])

    if cfg.is_on("bullet_presence"):
        add(
            "presence",
            bce(preds["bullet_presence"], targets["bullet_presence"]),
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
        add("wall", bce(preds["wall"], targets["wall"]), lambdas["wall"])

    if not terms:  # pragma: no cover - parse_objectives forbids an empty config
        raise ValueError("no objectives enabled; nothing to optimize")
    total = terms[0] if len(terms) == 1 else torch.stack(terms).sum()
    parts["total"] = float(total.detach())
    return total, parts


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
) -> dict[str, object]:
    """Run the loss + the full per-objective breakdown over ``loader``.

    Disabled groups report ``"N/A"``. POSITION groups also report world-unit error.
    bullet_presence -> precision/recall/F1; walls -> INTERIOR F1/IoU (+ overall acc).
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
    bpos_sse = np.zeros(N_BULLET_POS)
    bpos_world_sae = np.zeros(N_BULLET_POS)
    bpos_cnt = np.zeros(N_BULLET_POS)
    bdir_sse = np.zeros(N_BULLET_DIR)
    bdir_cnt = np.zeros(N_BULLET_DIR)
    # Presence: collect predictions/targets for F1.
    pres_tp = pres_fp = pres_fn = 0
    # Walls: interior F1/IoU + overall accuracy.
    wall_tp = wall_fp = wall_fn = wall_inter = wall_union = 0
    wall_correct = wall_total = 0

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        b = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(b["frame"])
            total, _parts = compute_losses(preds, b, model, lambdas)
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
            wall_correct += int((pred == tgt).sum())
            wall_total += int(tgt.size)
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
    from tank_twin.pretrain_pixels import PLAYER_SUBGROUP_VEC_INDICES

    for sub in _PLAYER_SUBGROUPS:
        if not cfg.is_on(sub):
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
            "overall_acc": wall_correct / max(1, wall_total),
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
        parts.append(
            f"walls=intF1{w['interior_f1']:.3f}/IoU{w['interior_iou']:.3f}/acc{w['overall_acc']:.3f}"
        )
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


def _build_memmap_datasets(args: Namespace, wall_grids: np.ndarray, holdout_ids: list[int]):
    """FULL path: read cache, 3-way map-aware split by group, fit stats on TRAIN rows."""
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
    ds_train = MemmapPixelDataset(cache, train_rows, wall_grids, stats)
    ds_eval = MemmapPixelDataset(cache, eval_rows, wall_grids, stats)
    ds_val = MemmapPixelDataset(cache, val_rows, wall_grids, stats)
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

    t0 = time.time()
    if args.cache_dir is not None:
        ds_train, ds_eval, ds_val, stats, split = _build_memmap_datasets(
            args, wall_grids, holdout_ids
        )
        mode = "memmap"
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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lambdas = {
        "player": args.lambda_player,
        "presence": args.lambda_presence,
        "bullet_pos": args.lambda_bullet_pos,
        "bullet_dir": args.lambda_bullet_dir,
        "wall": args.lambda_wall,
    }
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
                total, parts = compute_losses(preds, b, model, lambdas)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
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
        ev = _evaluate(model, eval_loader, lambdas, device, stats, use_amp, desc="  eval")
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
    validation = _evaluate(model, val_loader, lambdas, device, stats, use_amp, desc="  validation")
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
        "batch": args.batch,
        "lr": args.lr,
        "epochs": args.epochs,
        "eval_frac": args.eval_frac,
        "holdout_maps": args.holdout_map_names,
        "seed": args.seed,
        "mode": mode,
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
