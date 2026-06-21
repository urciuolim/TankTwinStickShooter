"""Torch trainer for the pixel inverse-renderer pretraining (Stage 2).

Imported lazily from :mod:`tank_twin.pretrain_pixels` so the pure target-encoding /
normalization helpers (and their unit tests) never need torch. Contains the model
(reusable encoder + 3 supervised heads), the streaming and memmap datasets, the
loss, the train/val loop, and checkpoint save/reload.

The ENCODER (conv stack + FC projection) is the reusable artifact: its state_dict is
saved standalone as ``encoder.pt`` and can be loaded into a fresh ``PixelEncoder``
without any head.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tank_twin.pretrain_pixels import (
    N_BULLET_POS,
    N_BULLET_SLOTS,
    N_PLAYER,
    N_WALL,
    WALL_H,
    WALL_W,
    NormStats,
    decode_targets,
    denormalize,
    downsample_frames_np,
    fit_norm_stats,
    list_shards,
    stratified_group_split,
    worker_id_from_shard,
)


# =============================================================================
# Model: reusable encoder + three heads on the shared embedding
# =============================================================================
class PixelEncoder(nn.Module):
    """NatureCNN-style encoder: 3 convs -> flatten -> Linear(embedding_dim) -> ReLU.

    Input is channel-first ``(B, 3, H, W)`` float in [0,1]. This conv+FC stack is the
    REUSABLE artifact; ``state_dict`` is what gets saved as ``encoder.pt``.
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
    """Encoder + three heads (player / bullet / wall) reading the SHARED embedding."""

    def __init__(self, in_h: int, in_w: int, embedding_dim: int = 512) -> None:
        super().__init__()
        self.encoder = PixelEncoder(in_h, in_w, embedding_dim)
        self.player_head = nn.Linear(embedding_dim, N_PLAYER)  # 12
        self.bullet_presence_head = nn.Linear(embedding_dim, N_BULLET_SLOTS)  # 10 logits
        self.bullet_pos_head = nn.Linear(embedding_dim, N_BULLET_POS)  # 40
        self.wall_head = nn.Linear(embedding_dim, N_WALL)  # 240 logits

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "player": self.player_head(z),
            "bullet_presence": self.bullet_presence_head(z),
            "bullet_pos": self.bullet_pos_head(z),
            "wall": self.wall_head(z).reshape(-1, WALL_H, WALL_W),
        }


def compute_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total + per-component loss.

    * player: MSE on normalized 12 floats.
    * presence: BCEWithLogits on 10 slot bits.
    * bullet_pos: MSE on normalized 40 floats, MASKED to present slots (absent slots
      contribute 0; mean is over present floats only).
    * wall: per-cell BCEWithLogits over the (12,20) grid.
    """
    mse = nn.functional.mse_loss
    bce = nn.functional.binary_cross_entropy_with_logits

    l_player = mse(preds["player"], targets["player"])
    l_presence = bce(preds["bullet_presence"], targets["bullet_presence"])

    mask = targets["bullet_pos_mask"]  # (B, 40) {0,1}
    sq = (preds["bullet_pos"] - targets["bullet_pos"]) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    l_bpos = sq.sum() / denom  # mean over PRESENT floats only

    l_wall = bce(preds["wall"], targets["wall"])

    total = (
        lambdas["player"] * l_player
        + lambdas["presence"] * l_presence
        + lambdas["bullet_pos"] * l_bpos
        + lambdas["wall"] * l_wall
    )
    parts = {
        "player": float(l_player.detach()),
        "presence": float(l_presence.detach()),
        "bullet_pos": float(l_bpos.detach()),
        "wall": float(l_wall.detach()),
    }
    return total, parts


# =============================================================================
# Datasets
# =============================================================================
class _BaseRows:
    """Holds resolved rows: frames (downsampled uint8), states, map_ids, group-key idx."""

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


class InMemoryPixelDataset(Dataset, _BaseRows):
    """In-memory dataset (SMOKE streamer or a small slice). Targets precomputed."""

    def __init__(self, *a, **kw) -> None:
        Dataset.__init__(self)
        _BaseRows.__init__(self, *a, **kw)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        frame = torch.from_numpy(self.frames[i]).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        return {
            "frame": frame,
            "player": torch.from_numpy(self._cache["player"][i]),
            "bullet_presence": torch.from_numpy(self._cache["bullet_presence"][i]),
            "bullet_pos": torch.from_numpy(self._cache["bullet_pos"][i]),
            "bullet_pos_mask": torch.from_numpy(self._cache["bullet_pos_mask"][i]),
            "wall": torch.from_numpy(self._cache["wall"][i]),
        }


# =============================================================================
# Streaming loader for the SMOKE (decode + resize a subset in-memory)
# =============================================================================
def _stream_subset(
    data_dir: Path,
    wall_grids: np.ndarray,
    out_w: int,
    out_h: int,
    subset: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Pull ~``subset`` pairs spanning maps from the shards, downsampling on the fly.

    Walks shards round-robin per worker and samples rows so multiple maps appear.
    Returns ``(frames_ds, states, map_ids, group_keys)`` where group_keys are
    ``(worker_id, episode_id)`` aligned to rows.
    """
    shards = list_shards(data_dir)
    rng = np.random.default_rng(seed)
    # Interleave shards across workers so the subset spans maps/workers.
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

    frames_acc: list[np.ndarray] = []
    states_acc: list[np.ndarray] = []
    maps_acc: list[np.ndarray] = []
    groups_acc: list[np.ndarray] = []
    collected = 0
    # Take a slice from each shard until we hit `subset`, spreading across many shards.
    per_shard = max(1, subset // max(1, min(len(order), 64)))
    for s in order:
        if collected >= subset:
            break
        wid = worker_id_from_shard(s.name)
        with np.load(s) as d:
            n = d["frames"].shape[0]
            take = min(per_shard, subset - collected, n)
            sel = np.sort(rng.choice(n, size=take, replace=False))
            fr = downsample_frames_np(d["frames"][sel], out_w, out_h)
            st = d["states"][sel].astype(np.float32)
            mp = d["map_ids"][sel].astype(np.int64)
            ep = d["episode_ids"][sel].astype(np.int64)
        frames_acc.append(fr)
        states_acc.append(st)
        maps_acc.append(mp)
        groups_acc.append(np.stack([np.full(take, wid, dtype=np.int64), ep], axis=1))
        collected += take

    frames = np.concatenate(frames_acc, axis=0)
    states = np.concatenate(states_acc, axis=0)
    map_ids = np.concatenate(maps_acc, axis=0)
    groups_arr = np.concatenate(groups_acc, axis=0)
    group_keys = [(int(w), int(e)) for w, e in groups_arr]
    return frames, states, map_ids, group_keys


# =============================================================================
# Memmap loader for the FULL run (cache built by scripts/prepare_pixel_cache.py)
# =============================================================================
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
        # Precompute normalized targets for the selected rows (cheap vs frames).
        sel_states = self.states[self.row_idx]
        sel_maps = self.map_ids[self.row_idx]
        self._cache = decode_targets(sel_states, sel_maps, self.wall_grids, self.stats)

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.row_idx[i])
        # .copy() yields a writable, contiguous array (the memmap row is read-only),
        # avoiding torch's non-writable-tensor UserWarning.
        frame = torch.from_numpy(np.array(self.frames[r], copy=True)).permute(2, 0, 1)
        return {
            "frame": frame.contiguous(),
            "player": torch.from_numpy(self._cache["player"][i]),
            "bullet_presence": torch.from_numpy(self._cache["bullet_presence"][i]),
            "bullet_pos": torch.from_numpy(self._cache["bullet_pos"][i]),
            "bullet_pos_mask": torch.from_numpy(self._cache["bullet_pos_mask"][i]),
            "wall": torch.from_numpy(self._cache["wall"][i]),
        }


# =============================================================================
# Train / eval loop
# =============================================================================
def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True)
    # Frame: uint8 (B,3,H,W) -> float [0,1].
    out["frame"] = out["frame"].float() / 255.0
    return out


@torch.no_grad()
def _evaluate(
    model: InverseRenderer,
    loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device,
    stats: NormStats,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {
        "total": 0.0,
        "player": 0.0,
        "presence": 0.0,
        "bullet_pos": 0.0,
        "wall": 0.0,
    }
    # World-unit error accumulators.
    player_sse = np.zeros(N_PLAYER)
    player_n = 0
    bpos_sae = 0.0
    bpos_n = 0
    wall_correct = 0
    wall_total = 0
    presence_correct = 0
    presence_total = 0
    n_batches = 0
    for batch in loader:
        b = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(b["frame"])
            total, parts = compute_losses(preds, b, lambdas)
        agg["total"] += float(total.detach())
        for k, v in parts.items():
            agg[k] += v
        n_batches += 1
        # World-unit metrics.
        pl_pred = denormalize(
            preds["player"].float().cpu().numpy(), stats.player_mean, stats.player_std
        )
        pl_tgt = denormalize(b["player"].float().cpu().numpy(), stats.player_mean, stats.player_std)
        player_sse += ((pl_pred - pl_tgt) ** 2).sum(axis=0)
        player_n += pl_pred.shape[0]

        mask = b["bullet_pos_mask"].float().cpu().numpy()
        bp_pred = denormalize(
            preds["bullet_pos"].float().cpu().numpy(), stats.bullet_mean, stats.bullet_std
        )
        bp_tgt = denormalize(
            b["bullet_pos"].float().cpu().numpy(), stats.bullet_mean, stats.bullet_std
        )
        bpos_sae += float((np.abs(bp_pred - bp_tgt) * mask).sum())
        bpos_n += int(mask.sum())

        wall_pred = (preds["wall"].float().cpu().numpy() > 0).astype(np.float32)
        wall_tgt = b["wall"].float().cpu().numpy()
        wall_correct += int((wall_pred == wall_tgt).sum())
        wall_total += int(wall_tgt.size)

        pres_pred = (preds["bullet_presence"].float().cpu().numpy() > 0).astype(np.float32)
        pres_tgt = b["bullet_presence"].float().cpu().numpy()
        presence_correct += int((pres_pred == pres_tgt).sum())
        presence_total += int(pres_tgt.size)

    for k in agg:
        agg[k] /= max(1, n_batches)
    agg["player_rmse_world"] = float(np.sqrt(player_sse.sum() / max(1, player_n * N_PLAYER)))
    agg["bullet_mae_world"] = float(bpos_sae / max(1, bpos_n))
    agg["wall_acc"] = float(wall_correct / max(1, wall_total))
    agg["presence_acc"] = float(presence_correct / max(1, presence_total))
    return agg


def _build_in_memory_datasets(args: Namespace, wall_grids: np.ndarray):
    """SMOKE / streaming path: pull a subset, fit stats on train, split by group."""
    out_w, out_h = args.input_wh
    subset = args.subset if args.subset is not None else 8000
    frames, states, map_ids, group_keys = _stream_subset(
        args.data_dir, wall_grids, out_w, out_h, subset, args.seed
    )
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids, strict=True)}
    train_groups, val_groups = stratified_group_split(
        group_keys, map_of_group, val_frac=args.val_frac, seed=args.seed
    )
    train_mask = np.array([g in train_groups for g in group_keys])
    val_mask = ~train_mask
    # Fit normalization on TRAIN only.
    stats = fit_norm_stats(states[train_mask])
    ds_train = InMemoryPixelDataset(
        frames[train_mask], states[train_mask], map_ids[train_mask], wall_grids, stats
    )
    ds_val = InMemoryPixelDataset(
        frames[val_mask], states[val_mask], map_ids[val_mask], wall_grids, stats
    )
    held_out = sorted(val_groups)
    return ds_train, ds_val, stats, held_out


def _build_memmap_datasets(args: Namespace, wall_grids: np.ndarray):
    """FULL path: read cache, split by group, fit stats on train rows."""
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    group_keys_arr = np.load(cache / "group_keys.npy")  # (N, 2)
    map_ids = np.load(cache / "map_ids.npy")
    states = np.load(cache / "states.npy")
    n = group_keys_arr.shape[0]
    if args.subset is not None:
        n = min(n, args.subset)
    all_rows = np.arange(n)
    group_keys = [(int(w), int(e)) for w, e in group_keys_arr[:n]]
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids[:n], strict=True)}
    train_groups, val_groups = stratified_group_split(
        group_keys, map_of_group, val_frac=args.val_frac, seed=args.seed
    )
    train_rows = all_rows[[g in train_groups for g in group_keys]]
    val_rows = all_rows[[g in val_groups for g in group_keys]]
    stats = fit_norm_stats(states[train_rows])
    ds_train = MemmapPixelDataset(cache, train_rows, wall_grids, stats)
    ds_val = MemmapPixelDataset(cache, val_rows, wall_grids, stats)
    _ = meta  # shape consumed inside MemmapPixelDataset
    return ds_train, ds_val, stats, sorted(val_groups)


def run_training(args: Namespace) -> int:
    """Build data, train, validate, and save the encoder + heads + stats. Returns exit code."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"[pretrain] device={device} amp={use_amp} torch={torch.__version__}")

    wall_grids = np.load(args.data_dir / "wall_grids.npy")

    out_w, out_h = args.input_wh
    t0 = time.time()
    if args.cache_dir is not None:
        ds_train, ds_val, stats, held_out = _build_memmap_datasets(args, wall_grids)
        mode = "memmap"
    else:
        ds_train, ds_val, stats, held_out = _build_in_memory_datasets(args, wall_grids)
        mode = "stream"
    print(
        f"[pretrain] mode={mode} res={out_w}x{out_h} "
        f"train={len(ds_train)} val={len(ds_val)} held_out_groups={len(held_out)} "
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
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = InverseRenderer(out_h, out_w, args.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lambdas = {
        "player": args.lambda_player,
        "presence": args.lambda_presence,
        "bullet_pos": args.lambda_bullet_pos,
        "wall": args.lambda_wall,
    }
    print(
        f"[pretrain] encoder flatten_dim={model.encoder.flatten_dim} "
        f"embedding_dim={model.encoder.embedding_dim} params="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    seen = 0
    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_parts = {"total": 0.0, "player": 0.0, "presence": 0.0, "bullet_pos": 0.0, "wall": 0.0}
        n_batches = 0
        for batch in train_loader:
            b = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(b["frame"])
                total, parts = compute_losses(preds, b, lambdas)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            ep_parts["total"] += float(total.detach())
            for k, v in parts.items():
                ep_parts[k] += v
            n_batches += 1
            global_step += 1
            seen += b["frame"].shape[0]
            if global_step % args.log_every == 0:
                print(
                    f"  [e{epoch} s{global_step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                    + f" total={float(total.detach()):.4f}"
                )
        for k in ep_parts:
            ep_parts[k] /= max(1, n_batches)
        val = _evaluate(model, val_loader, lambdas, device, stats, use_amp)
        print(
            f"[epoch {epoch}] "
            f"TRAIN total={ep_parts['total']:.4f} player={ep_parts['player']:.4f} "
            f"presence={ep_parts['presence']:.4f} bullet_pos={ep_parts['bullet_pos']:.4f} "
            f"wall={ep_parts['wall']:.4f} || "
            f"VAL total={val['total']:.4f} player={val['player']:.4f} "
            f"presence={val['presence']:.4f} bullet_pos={val['bullet_pos']:.4f} "
            f"wall={val['wall']:.4f} | "
            f"player_rmse_world={val['player_rmse_world']:.3f} "
            f"bullet_mae_world={val['bullet_mae_world']:.3f} "
            f"wall_acc={val['wall_acc']:.3f} presence_acc={val['presence_acc']:.3f}"
        )

    train_secs = time.time() - train_t0
    pairs_per_sec = seen / max(1e-9, train_secs)
    print(
        f"[pretrain] trained {seen} pair-steps in {train_secs:.1f}s "
        f"({pairs_per_sec:.0f} pairs/s, {global_step / max(1e-9, train_secs):.1f} steps/s)"
    )

    # --- Save artifacts ------------------------------------------------------
    encoder_path = out_dir / "encoder.pt"
    full_path = out_dir / "model.pt"
    stats_path = out_dir / "norm_stats.json"
    split_path = out_dir / "val_groups.json"
    config_path = out_dir / "config.json"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.state_dict(), full_path)
    stats_path.write_text(json.dumps(stats.to_json(), indent=2), encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "val_groups": [list(g) for g in held_out],
                "val_frac": args.val_frac,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "input_res": f"{out_w}x{out_h}",
                "embedding_dim": args.embedding_dim,
                "flatten_dim": model.encoder.flatten_dim,
                "lambdas": lambdas,
                "batch": args.batch,
                "lr": args.lr,
                "epochs": args.epochs,
                "seed": args.seed,
                "mode": mode,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[pretrain] saved encoder -> {encoder_path}")
    print(f"[pretrain] saved full model -> {full_path}")
    print(f"[pretrain] saved norm stats -> {stats_path}")
    print(f"[pretrain] saved val split -> {split_path}")

    # --- Verify the encoder reloads STANDALONE into a fresh PixelEncoder -----
    fresh = PixelEncoder(out_h, out_w, args.embedding_dim)
    sd = torch.load(encoder_path, map_location="cpu", weights_only=True)
    fresh.load_state_dict(sd)  # strict=True: raises on any mismatch
    fresh.eval()
    with torch.no_grad():
        emb = fresh(torch.zeros(2, 3, out_h, out_w))
    assert emb.shape == (2, args.embedding_dim), emb.shape
    print(
        f"[pretrain] encoder reload OK: fresh PixelEncoder forward -> {tuple(emb.shape)} "
        f"(expected (2, {args.embedding_dim}))"
    )
    return 0
