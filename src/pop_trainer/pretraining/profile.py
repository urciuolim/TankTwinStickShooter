"""Compute profile: parameter count + forward-pass latency for an encoder config.

Reports the Phase-3 numbers for one ``(EncoderConfig, resolution, device)`` cell: total and
encoder-only parameter counts, and the mean forward-pass latency over a few timed iterations
(after warmup). The numbers are returned as a strict-JSON-safe dict and the CLI prints them.

torch + :mod:`pop_trainer.models` + :mod:`pop_trainer.pretraining`; nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pop_trainer.models import EncoderConfig, build_encoder
from pop_trainer.pretraining.decoder import StateDecoder
from pop_trainer.pretraining.device import resolve_device

__all__ = ["count_parameters", "profile_forward", "main"]

_NATIVE_HEIGHT = 360
_NATIVE_WIDTH = 640


def count_parameters(module: torch.nn.Module) -> int:
    """Total number of parameters (counting every tensor element) in ``module``."""
    return int(sum(p.numel() for p in module.parameters()))


def _input_hw(resolution: int) -> tuple[int, int]:
    """``(H, W)`` for a target height ``resolution`` keeping the native 360x640 aspect."""
    factor = _NATIVE_HEIGHT // resolution
    return resolution, _NATIVE_WIDTH // factor


def profile_forward(
    cfg: EncoderConfig,
    *,
    resolution: int = 180,
    device: str = "cpu",
    batch_size: int = 8,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Profile the decoder's forward pass for ``cfg`` at ``resolution`` on ``device``.

    Builds the encoder + decoder, runs ``warmup`` untimed forward passes then ``iters`` timed
    ones, and returns a strict-JSON-safe summary: parameter counts (total + encoder-only) and
    mean / per-sample forward latency in milliseconds.
    """
    dev = torch.device(resolve_device(device))
    h, w = _input_hw(resolution)
    encoder = build_encoder(cfg)
    model = StateDecoder(encoder, (h, w)).to(dev).eval()

    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)

    x = torch.zeros(batch_size, cfg.in_channels, h, w, device=dev)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(x)
        _sync(dev)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            model(x)
        _sync(dev)
        elapsed = time.perf_counter() - start

    mean_ms = elapsed / max(1, iters) * 1000.0
    return {
        "trunk": cfg.trunk,
        "pooling": cfg.pooling,
        "resolution": resolution,
        "input_hw": [h, w],
        "device": str(dev),
        "batch_size": batch_size,
        "iters": iters,
        "total_params": total_params,
        "encoder_params": encoder_params,
        "forward_ms_mean": round(mean_ms, 4),
        "forward_ms_per_sample": round(mean_ms / batch_size, 4),
    }


def _sync(device: torch.device) -> None:
    """Block until queued device work finishes (CUDA), so timing is wall-accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m pop_trainer.pretraining.profile`` — print the profile as strict JSON."""
    parser = argparse.ArgumentParser(description="Encoder compute profile (params + latency).")
    parser.add_argument("--trunk", choices=("nature", "impala"), default="nature")
    parser.add_argument("--pooling", choices=("gap", "flatten"), default="gap")
    parser.add_argument("--resolution", type=int, choices=(360, 180, 90), default=180)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args(argv)

    cfg = EncoderConfig(trunk=args.trunk, pooling=args.pooling)
    summary = profile_forward(
        cfg,
        resolution=args.resolution,
        device=args.device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
