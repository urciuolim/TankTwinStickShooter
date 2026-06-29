"""Train CLI for the single-frame decoder: ``python -m pop_trainer.pretraining.train``.

Trains the encoder + spatial heads with the combined per-group loss (coord MSE + an auxiliary
heatmap cross-entropy on the score maps) AND the detached embed-probe with its own loss, in TWO
disjoint optimizer steps per batch (the probe reads ``embed.detach()`` so its gradient never
reaches the encoder regardless, and gets NO heatmap term). Evaluates on val + test, logging
per-group metrics for BOTH head families, then saves a checkpoint (encoder + decoder state) and a
strict-JSON results record.

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
from pop_trainer.pretraining.sampler import (
    DEFAULT_SHARD_WINDOW,
    ShardWindowBatchSampler,
    cache_capacity_for_window,
)
from pop_trainer.pretraining.targets import NormStats, presence_pos_weight

__all__ = [
    "TrainConfig",
    "parse_group_weights",
    "build_train_loader",
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
    shard_window: int = DEFAULT_SHARD_WINDOW
    heatmap_weight: float = losses.DEFAULT_HEATMAP_WEIGHT
    heatmap_sigma: float = losses.DEFAULT_HEATMAP_SIGMA
    group_weights: dict[str, float] | None = None
    presence_pos_weight: float | None = None


def parse_group_weights(items: list[str] | None) -> dict[str, float]:
    """PURE: parse ``["player_aim=5.0", "bullet_presence=2"]`` into ``{name: float}``.

    Each token is ``name=value``; ``name`` must be one of :data:`losses.DEFAULT_GROUP_WEIGHTS`
    and ``value`` must parse as a float. Empty / ``None`` input returns ``{}`` (the default-weight
    path). Raises ``ValueError`` on an unknown group name or a malformed token.
    """
    if not items:
        return {}
    out: dict[str, float] = {}
    for token in items:
        name, sep, raw = token.partition("=")
        if not sep or not name:
            raise ValueError(f"malformed group weight {token!r}; expected NAME=VALUE")
        if name not in losses.DEFAULT_GROUP_WEIGHTS:
            valid = ", ".join(sorted(losses.DEFAULT_GROUP_WEIGHTS))
            raise ValueError(f"unknown group {name!r}; valid groups: {valid}")
        try:
            out[name] = float(raw)
        except ValueError as exc:
            raise ValueError(f"malformed group weight {token!r}; value must be a float") from exc
    return out


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
    weights: dict[str, float] | None = None,
    heatmap_weight: float = losses.DEFAULT_HEATMAP_WEIGHT,
    heatmap_sigma: float = losses.DEFAULT_HEATMAP_SIGMA,
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
            out["spatial"],
            targets,
            weights=weights,
            presence_pos_weight=pos_weight,
            score_logits=out["spatial_score_logits"],
            heatmap_weight=heatmap_weight,
            heatmap_sigma=heatmap_sigma,
        )
        spatial_opt.zero_grad(set_to_none=True)
        probe_opt.zero_grad(set_to_none=True)
        spatial_total.backward()
        spatial_opt.step()

        probe_total, _ = losses.probe_loss(
            out["probe"], targets, weights=weights, presence_pos_weight=pos_weight
        )
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
def _accumulate(
    model: StateDecoder, loader: DataLoader, device: torch.device
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Run one forward pass over ``loader``, returning concatenated preds + targets.

    Returns ``({"spatial": {group: tensor}, "probe": {group: tensor}}, {group: tensor})`` with
    predictions detached on CPU. The families share the single forward pass per batch so callers
    can derive metrics (or calibrate a threshold) without re-running the model.
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
        return {"spatial": {}, "probe": {}}, {}
    targets_cat = {k: torch.cat(v, dim=0) for k, v in tgt_acc.items()}
    preds_cat: dict[str, dict[str, torch.Tensor]] = {}
    for fam in ("spatial", "probe"):
        preds_cat[fam] = {k: torch.cat(v, dim=0) for k, v in fam_preds[fam].items()}
    return preds_cat, targets_cat


def evaluate(
    model: StateDecoder, loader: DataLoader, device: torch.device, stats: NormStats
) -> dict[str, dict]:
    """Evaluate per-group metrics for both head families over ``loader`` (concatenated).

    Accumulates predictions / targets across the loader and computes the metrics once, so
    presence F1 / world-unit errors are over the whole split. Returns
    ``{"spatial": {...}, "probe": {...}}``.
    """
    fam_preds, targets_cat = _accumulate(model, loader, device)
    if not targets_cat:
        return {"spatial": {}, "probe": {}}
    return {fam: metrics.group_metrics(fam_preds[fam], targets_cat, stats) for fam in fam_preds}


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


def build_train_loader(
    train_ds,
    *,
    batch_size: int,
    window: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, ShardWindowBatchSampler]:
    """Build the windowed multi-shard train loader and size the dataset cache to match it.

    Single source of truth for the window/cache coupling: the cache capacity is
    ``cache_capacity_for_window(window)`` (= ``window + 1``) and the ``C >= window`` invariant is
    guarded here so it can never silently regress — a smaller cache than the window would re-read
    in-flight shards per sample (the loader perf pathology). Returns the loader and its sampler so
    the caller can ``set_epoch`` each epoch.
    """
    capacity = cache_capacity_for_window(window)
    if capacity < window:  # invariant guard: window shards must all stay resident
        raise ValueError(f"cache capacity {capacity} < window {window}; would re-read shards")
    train_ds.set_cache_capacity(capacity)
    sampler = ShardWindowBatchSampler(
        train_ds.sample_shards(), batch_size=batch_size, window=window, seed=seed
    )
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers)
    return loader, sampler


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
    pos_weight = (
        cfg.presence_pos_weight
        if cfg.presence_pos_weight is not None
        else _fit_pos_weight(splits.train)
    )

    train_loader, train_sampler = build_train_loader(
        splits.train,
        batch_size=cfg.batch_size,
        window=cfg.shard_window,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
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
            weights=cfg.group_weights,
            heatmap_weight=cfg.heatmap_weight,
            heatmap_sigma=cfg.heatmap_sigma,
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

    val_preds, val_targets = _accumulate(model, val_loader, device)
    test_preds, test_targets = _accumulate(model, test_loader, device)
    val_metrics = (
        {fam: metrics.group_metrics(val_preds[fam], val_targets, splits.stats) for fam in val_preds}
        if val_targets
        else {"spatial": {}, "probe": {}}
    )
    test_metrics = (
        {
            fam: metrics.group_metrics(test_preds[fam], test_targets, splits.stats)
            for fam in test_preds
        }
        if test_targets
        else {"spatial": {}, "probe": {}}
    )

    presence_calibration = _calibrate_presence(val_preds, val_targets, test_preds, test_targets)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.state_dict(),
            "norm_stats": splits.stats.to_json(),
            "grid_extent": splits.extent.to_json(),
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
        "presence_calibration": _sanitize(presence_calibration),
        "norm_stats": splits.stats.to_json(),
        "grid_extent": splits.extent.to_json(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, allow_nan=False)
    return record


def _calibrate_presence(
    val_preds: dict[str, dict[str, torch.Tensor]],
    val_targets: dict[str, torch.Tensor],
    test_preds: dict[str, dict[str, torch.Tensor]],
    test_targets: dict[str, torch.Tensor],
) -> dict:
    """Pick a presence logit threshold on VAL spatial presence and report it on val + test.

    Calibrates on the SPATIAL family only (the encoder readout we ship; the probe is diagnostic).
    The threshold maximizes F1 on VAL ``bullet_presence`` logits + targets, then the SAME threshold
    is applied to TEST — no val/test leak. Reported alongside the untouched threshold-0 presence F1
    in ``val_metrics`` / ``test_metrics``.
    """
    val_logits = val_preds.get("spatial", {}).get("bullet_presence")
    if val_logits is None or "bullet_presence" not in val_targets:
        return {
            "family": "spatial",
            "selected_threshold": 0.0,
            "val": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "test": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
    threshold, val_at = metrics.select_presence_threshold(
        val_logits, val_targets["bullet_presence"]
    )
    test_logits = test_preds.get("spatial", {}).get("bullet_presence")
    if test_logits is None or "bullet_presence" not in test_targets:
        test_at = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        test_at = metrics.presence_metrics_at_threshold(
            test_logits, test_targets["bullet_presence"], threshold
        )
    return {
        "family": "spatial",
        "selected_threshold": _finite(threshold),
        "val": val_at,
        "test": test_at,
    }


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
        "shard_window": cfg.shard_window,
        "heatmap_weight": cfg.heatmap_weight,
        "heatmap_sigma": cfg.heatmap_sigma,
        "group_weights": {**losses.DEFAULT_GROUP_WEIGHTS, **(cfg.group_weights or {})},
        "presence_pos_weight_override": cfg.presence_pos_weight,
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
        "--shard-window",
        type=int,
        default=DEFAULT_SHARD_WINDOW,
        help=(
            "number of shards kept in flight; each train batch draws rows round-robin across "
            f"them for cross-shard (cross-map) diversity. Default {DEFAULT_SHARD_WINDOW}. RAM is "
            "~(window + 1) x ~0.4 GB resident (the bounded shard cache holds window + 1 shards)."
        ),
    )
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
    parser.add_argument(
        "--heatmap-weight",
        type=float,
        default=losses.DEFAULT_HEATMAP_WEIGHT,
        help=(
            "scalar weight on the auxiliary heatmap cross-entropy (relative to unit coord MSE). "
            "Must be strong on a coarse grid; weak weights (<=1) backfire."
        ),
    )
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=losses.DEFAULT_HEATMAP_SIGMA,
        help="Gaussian target std in grid CELLS for the heatmap cross-entropy.",
    )
    parser.add_argument(
        "--group-weights",
        nargs="*",
        metavar="GROUP=W",
        default=None,
        help=(
            "up-weight specific per-group losses in BOTH head families, e.g. "
            "--group-weights player_aim=5.0 bullet_presence=2.0. Unlisted groups stay 1.0. "
            "Omit the flag for the default unit weighting."
        ),
    )
    parser.add_argument(
        "--presence-pos-weight",
        type=float,
        default=None,
        help=(
            "override the auto-fit bullet-presence pos_weight; when omitted it is auto-fit "
            "from TRAIN."
        ),
    )
    args = parser.parse_args(argv)
    group_weights = parse_group_weights(args.group_weights)

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
        shard_window=args.shard_window,
        heatmap_weight=args.heatmap_weight,
        heatmap_sigma=args.heatmap_sigma,
        group_weights=group_weights or None,
        presence_pos_weight=args.presence_pos_weight,
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
