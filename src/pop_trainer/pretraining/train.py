"""Train CLI for the single-frame decoder: ``python -m pop_trainer.pretraining.train``.

Trains the encoder + spatial heads with the combined per-group loss AND the detached embed-probe
with its own loss, in TWO disjoint optimizer steps per batch (the probe reads ``embed.detach()``
so its gradient never reaches the encoder regardless). Evaluates on val + test, logging per-group
metrics for BOTH head families, then saves a checkpoint (encoder + decoder state) and a strict-JSON
results record.

Device-agnostic: ``--device auto`` picks cuda > mps > cpu, so the same command runs on a 4090
unchanged. Determinism is seeded (torch + numpy) where reasonable.

torch + numpy + the pretraining/models/data components; nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pop_trainer.models import EncoderConfig, build_encoder
from pop_trainer.pretraining import losses, metrics
from pop_trainer.pretraining.dataset import build_splits
from pop_trainer.pretraining.decoder import StateDecoder
from pop_trainer.pretraining.device import resolve_device
from pop_trainer.pretraining.progress import ProgressReporter
from pop_trainer.pretraining.sampler import ShardGroupedBatchSampler
from pop_trainer.pretraining.targets import NormStats, presence_pos_weight

__all__ = [
    "TrainConfig",
    "seed_everything",
    "should_eval_epoch",
    "train_one_epoch",
    "evaluate",
    "run",
    "main",
]

_NATIVE_HEIGHT = 360
_NATIVE_WIDTH = 640


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run (the CLI builds this)."""

    data_dir: str
    out_dir: str
    trunk: str = "nature"
    pooling: str = "gap"
    resolution: int = 180
    epochs: int = 1
    batch_size: int = 32
    lr: float = 3e-4
    seed: int = 0
    device: str = "auto"
    val_frac: float = 0.2
    test_frac: float = 0.2
    subset: int | None = None
    num_workers: int = 0
    progress: bool = True
    eval_every: int = 1


def should_eval_epoch(epoch: int, *, eval_every: int, total_epochs: int) -> bool:
    """PURE: should periodic val eval run at the end of ``epoch`` (0-based)?

    Returns True when ``eval_every > 0`` AND (``epoch`` is a multiple of ``eval_every``
    OR ``epoch`` is the final epoch ``total_epochs - 1``).

    When ``eval_every <= 0`` returns False for ALL epochs, including the final one — the
    periodic val curve is fully disabled and the existing end-of-run eval (called after the
    loop) is the sole evaluation. This makes the behavior of ``--eval-every 0`` identical to
    the pre-change code: no extra compute, ``val_trajectory`` stays empty.

    No I/O, no tensors.
    """
    if eval_every <= 0:
        return False
    return (epoch % eval_every == 0) or (epoch == total_epochs - 1)


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch RNGs for reproducible runs where reasonable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def input_hw(resolution: int) -> tuple[int, int]:
    """``(H, W)`` for a target height ``resolution`` keeping the native 360x640 aspect."""
    factor = _NATIVE_HEIGHT // resolution
    return resolution, _NATIVE_WIDTH // factor


def _to_device(batch_targets: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch_targets.items()}


def train_one_epoch(
    model: StateDecoder,
    loader: DataLoader,
    spatial_opt: torch.optim.Optimizer,
    probe_opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: float | None,
    on_batch: Callable[[int, float, float], None] | None = None,
) -> dict[str, float]:
    """One training pass: a spatial-loss step (trains encoder) + a probe-loss step per batch.

    Returns the mean total spatial loss and mean total probe loss over the epoch (for the loss
    trajectory). The probe reads ``embed.detach()``, so its optimizer step cannot move the
    encoder; the two steps are kept on disjoint parameter groups.

    ``on_batch`` is an OPTIONAL pure observability side-channel: when given, it is called after each
    batch with ``(step, running_spatial_loss, running_probe_loss)`` (``step`` is 1-based; the
    running losses are the epoch means so far, plain floats) so a reporter can show progress.
    ``None`` (the default) is a no-op and leaves this loop byte-identical — the callback receives
    only ints / floats and returns nothing, so it cannot touch tensors-with-grad, the optimizers, or
    the RNG.
    """
    model.train()
    spatial_sum = 0.0
    probe_sum = 0.0
    n_batches = 0
    for frames, targets in loader:
        frames = frames.to(device)
        targets = _to_device(targets, device)
        out = model(frames)

        spatial_total, _ = losses.combined_loss(
            out["spatial"], targets, presence_pos_weight=pos_weight
        )
        spatial_opt.zero_grad(set_to_none=True)
        probe_opt.zero_grad(set_to_none=True)
        spatial_total.backward()
        spatial_opt.step()

        probe_total, _ = losses.probe_loss(out["probe"], targets, presence_pos_weight=pos_weight)
        probe_opt.zero_grad(set_to_none=True)
        probe_total.backward()
        probe_opt.step()

        spatial_sum += float(spatial_total.detach())
        probe_sum += float(probe_total.detach())
        n_batches += 1
        if on_batch is not None:
            on_batch(n_batches, spatial_sum / n_batches, probe_sum / n_batches)

    denom = max(1, n_batches)
    return {"spatial_loss": spatial_sum / denom, "probe_loss": probe_sum / denom}


@torch.no_grad()
def evaluate(
    model: StateDecoder, loader: DataLoader, device: torch.device, stats: NormStats
) -> dict[str, dict]:
    """Evaluate per-group metrics for both head families over ``loader`` (concatenated).

    Accumulates predictions / targets across the loader and computes the metrics once, so
    presence F1 / world-unit errors are over the whole split. Returns
    ``{"spatial": {...}, "probe": {...}}``.
    """
    model.eval()
    fam_preds: dict[str, dict[str, list]] = {"spatial": {}, "probe": {}}
    tgt_acc: dict[str, list] = {}
    for frames, targets in loader:
        frames = frames.to(device)
        out = model(frames)
        for fam in ("spatial", "probe"):
            for k, v in out[fam].items():
                fam_preds[fam].setdefault(k, []).append(v.detach().cpu())
        for k, v in targets.items():
            tgt_acc.setdefault(k, []).append(v)

    if not tgt_acc:
        return {"spatial": {}, "probe": {}}
    targets_cat = {k: torch.cat(v, dim=0) for k, v in tgt_acc.items()}
    result: dict[str, dict] = {}
    for fam in ("spatial", "probe"):
        preds_cat = {k: torch.cat(v, dim=0) for k, v in fam_preds[fam].items()}
        result[fam] = metrics.group_metrics(preds_cat, targets_cat, stats)
    return result


def _fit_pos_weight(train_ds: Dataset, *, cap: int = 4096) -> float:
    """Bullet-presence pos_weight from up to ``cap`` TRAIN rows (no val/test leak)."""
    n = min(len(train_ds), cap)
    if n == 0:
        return 1.0
    presences = [train_ds[i][1]["bullet_presence"].numpy() for i in range(n)]
    return presence_pos_weight(np.stack(presences))


def _make_loader(ds: Dataset, batch_size: int, *, shuffle: bool, num_workers: int) -> DataLoader:
    """A spawn-safe DataLoader (``num_workers=0`` default; no fork)."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run(cfg: TrainConfig) -> dict:
    """Run training + eval per ``cfg`` and write the checkpoint + results record. Returns it.

    The results record is strict-JSON (NaN/Inf guarded). The checkpoint holds the encoder and
    full decoder state plus the normalization stats so a downstream consumer can de-normalize.
    """
    seed_everything(cfg.seed)
    device = torch.device(resolve_device(cfg.device))
    h, w = input_hw(cfg.resolution)

    splits = build_splits(
        cfg.data_dir,
        resolution=cfg.resolution,
        seed=cfg.seed,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        limit=cfg.subset,
    )
    encoder = build_encoder(EncoderConfig(trunk=cfg.trunk, pooling=cfg.pooling))
    model = StateDecoder(encoder, (h, w)).to(device)

    spatial_opt = torch.optim.Adam(model.spatial_parameters(), lr=cfg.lr)
    probe_opt = torch.optim.Adam(model.probe_parameters(), lr=cfg.lr)
    pos_weight = _fit_pos_weight(splits.train)

    train_sampler = ShardGroupedBatchSampler(
        splits.train.sample_shards(), batch_size=cfg.batch_size, seed=cfg.seed
    )
    train_loader = DataLoader(
        splits.train, batch_sampler=train_sampler, num_workers=cfg.num_workers
    )
    val_loader = _make_loader(
        splits.val, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = _make_loader(
        splits.test, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    reporter = ProgressReporter(total_epochs=cfg.epochs, enabled=cfg.progress)
    total_steps = len(train_loader)
    loss_trajectory: list[dict[str, float]] = []
    val_trajectory: list[dict] = []
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        reporter.epoch_start(epoch + 1, total_steps)
        epoch_losses = train_one_epoch(
            model,
            train_loader,
            spatial_opt,
            probe_opt,
            device,
            pos_weight=pos_weight,
            on_batch=reporter.on_batch,
        )
        reporter.epoch_close()
        loss_trajectory.append(epoch_losses)

        # Periodic val eval: train_one_epoch calls model.train() at its start, so evaluating
        # here (which calls model.eval() + no_grad) is safe — the next epoch re-enters train
        # mode unconditionally.
        if should_eval_epoch(epoch, eval_every=cfg.eval_every, total_epochs=cfg.epochs):
            ep_val = evaluate(model, val_loader, device, splits.stats)
            val_trajectory.append({"epoch": epoch, **_sanitize(ep_val)})
            # Print the val line to stderr regardless of --progress: it is low-frequency
            # (once per eval_every epochs) and is the primary convergence signal for the run.
            sp = ep_val.get("spatial", {})
            pos_err = sp.get("player_position", float("nan"))
            aim_err = sp.get("player_aim", float("nan"))
            pres_f1 = sp.get("bullet_presence", {})
            if isinstance(pres_f1, dict):
                pres_f1 = pres_f1.get("f1", float("nan"))
            print(
                f"[val] epoch {epoch}"
                f"  pos={_finite(pos_err):.4f}"
                f"  aim={_finite(aim_err):.2f}°"
                f"  presence_f1={_finite(pres_f1):.4f}",
                file=sys.stderr,
                flush=True,
            )

    val_metrics = evaluate(model, val_loader, device, splits.stats)
    test_metrics = evaluate(model, test_loader, device, splits.stats)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.state_dict(),
            "norm_stats": splits.stats.to_json(),
            "config": _config_json(cfg),
        },
        out_dir / "checkpoint.pt",
    )

    record = {
        "config": _config_json(cfg),
        "device": str(device),
        "input_hw": [h, w],
        "presence_pos_weight": _finite(pos_weight),
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "loss_trajectory": [{k: _finite(v) for k, v in e.items()} for e in loss_trajectory],
        "val_trajectory": _sanitize(val_trajectory),
        "val_metrics": _sanitize(val_metrics),
        "test_metrics": _sanitize(test_metrics),
        "norm_stats": splits.stats.to_json(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, allow_nan=False)
    return record


def _config_json(cfg: TrainConfig) -> dict:
    return {
        "data_dir": cfg.data_dir,
        "trunk": cfg.trunk,
        "pooling": cfg.pooling,
        "resolution": cfg.resolution,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "subset": cfg.subset,
        "eval_every": cfg.eval_every,
    }


def _finite(x: float) -> float:
    """Map NaN/Inf to 0.0 so the JSON record stays strict (no non-finite literals)."""
    return float(x) if math.isfinite(x) else 0.0


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with 0.0 in a metrics tree for strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return _finite(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    """Parse args, run training, print the loss trajectory + a per-group metric summary."""
    parser = argparse.ArgumentParser(description="Single-frame decoder pretraining.")
    parser.add_argument("--data", default="datasets/decode-v1")
    parser.add_argument("--out", default="runs/decode-smoke")
    parser.add_argument("--trunk", choices=("nature", "impala"), default="nature")
    parser.add_argument("--pooling", choices=("gap", "flatten"), default="gap")
    parser.add_argument("--resolution", type=int, choices=(360, 180, 90), default=180)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="map-aware val fraction by map COUNT; 0.2 seats >=2 maps at G=10 (default).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="map-aware test fraction by map COUNT; 0.2 seats >=2 maps at G=10 (default).",
    )
    parser.add_argument(
        "--subset",
        "--limit",
        dest="subset",
        type=int,
        default=None,
        help="cap the TOTAL indexed samples (smoke runs).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--progress",
        dest="progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show a live tqdm bar (TTY) or throttled progress lines (captured); on by default.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help=(
            "run val eval every K epochs and record a val_trajectory curve. "
            "Default 1 (every epoch). 0 disables periodic eval (end-of-run eval unchanged)."
        ),
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        trunk=args.trunk,
        pooling=args.pooling,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        subset=args.subset,
        num_workers=args.num_workers,
        progress=args.progress,
        eval_every=args.eval_every,
    )
    record = run(cfg)

    print("loss trajectory (per epoch):")
    for i, e in enumerate(record["loss_trajectory"]):
        print(f"  epoch {i}: spatial={e['spatial_loss']:.4f} probe={e['probe_loss']:.4f}")
    for split_name in ("val_metrics", "test_metrics"):
        print(f"{split_name}:")
        for fam in ("spatial", "probe"):
            print(f"  {fam}: {json.dumps(record[split_name][fam], allow_nan=False)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
