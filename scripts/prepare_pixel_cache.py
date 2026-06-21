"""Pre-resize the pixel_1m shards ONCE into a uint8 memmap cache for the FULL run.

The pretraining loader re-reads every frame each epoch; decoding+resizing 918K
640x360 frames per epoch is the bottleneck. This builder downsamples each frame ONCE
to the configured input-res and writes a contiguous uint8 memmap
(``frames_u8.dat``, shape (N, H, W, 3)), plus the aligned ``states.npy``,
``map_ids.npy``, and ``group_keys.npy`` (N,2 = worker_id, episode_id), and a small
``index.json``. Epochs then mmap-read pre-resized frames (no decode/resize).

READ-ONLY over the shards (never modifies ``datasets/pixel_1m``). Writes a NEW cache
dir. GPU-accelerated resize via torch ``interpolate`` (area) when CUDA is available,
else a pure-numpy block-mean fallback (same result for integer block factors).

Disk cost = N * H * W * 3 bytes. At 160x90 over 918,000 pairs:
    918000 * 90 * 160 * 3 = 39,657,600,000 bytes ~= 36.9 GiB (~39.7 GB).
Scales linearly with H*W: e.g. 320x180 is 4x (~158 GB); 80x45 is 1/4 (~9.9 GB).

Run::

    uv run python scripts/prepare_pixel_cache.py \
        --data-dir datasets/pixel_1m --out datasets/pixel_cache_160x90 --input-res 160x90
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tank_twin.pretrain_pixels import (  # noqa: E402
    downsample_frames_np,
    list_shards,
    parse_input_res,
    worker_id_from_shard,
)

_SHARD_RE = re.compile(r"shard_w(\d+)_(\d+)\.npz$")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _resize_gpu(frames_u8: np.ndarray, out_w: int, out_h: int):
    """GPU area-resize a (N,360,640,3) uint8 batch -> (N,out_h,out_w,3) uint8.

    Returns None if torch/CUDA unavailable so the caller falls back to numpy.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    t = torch.from_numpy(frames_u8).cuda().permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="area")
    t = t.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    return t.cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="datasets/pixel_1m")
    ap.add_argument("--out", required=True, help="NEW cache dir to write")
    ap.add_argument("--input-res", default="160x90", help="downsample target WxH")
    ap.add_argument("--limit", type=int, default=None, help="cap total pairs (testing)")
    ap.add_argument("--cpu", action="store_true", help="force numpy block-mean (no GPU)")
    args = ap.parse_args(argv)

    out_w, out_h = parse_input_res(args.input_res)
    data_dir = _resolve(args.data_dir)
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = list_shards(data_dir)
    # First pass: total N (cheap header read).
    counts = []
    for s in tqdm(shards, desc="counting shards", unit="shard"):
        with np.load(s) as d:
            counts.append(int(d["map_ids"].shape[0]))
    total = sum(counts)
    if args.limit is not None:
        total = min(total, args.limit)

    nbytes = total * out_h * out_w * 3
    print(
        f"[cache] N={total} res={out_w}x{out_h} -> frames_u8.dat = {nbytes:,} bytes "
        f"({nbytes / 1024**3:.1f} GiB / {nbytes / 1e9:.1f} GB)"
    )

    frames_mm = np.memmap(
        out_dir / "frames_u8.dat", dtype=np.uint8, mode="w+", shape=(total, out_h, out_w, 3)
    )
    states = np.zeros((total, 52), dtype=np.float32)
    map_ids = np.zeros((total,), dtype=np.int32)
    group_keys = np.zeros((total, 2), dtype=np.int64)  # (worker_id, episode_id)

    row = 0
    t0 = time.time()
    pbar = tqdm(shards, desc="caching frames", unit="shard")
    for s in pbar:
        if row >= total:
            break
        wid = worker_id_from_shard(s.name)
        with np.load(s) as d:
            n = int(d["map_ids"].shape[0])
            take = min(n, total - row)
            fr = d["frames"][:take]
            ds = None if args.cpu else _resize_gpu(fr, out_w, out_h)
            if ds is None:
                ds = downsample_frames_np(fr, out_w, out_h)
            frames_mm[row : row + take] = ds
            states[row : row + take] = d["states"][:take].astype(np.float32)
            map_ids[row : row + take] = d["map_ids"][:take].astype(np.int32)
            group_keys[row : row + take, 0] = wid
            group_keys[row : row + take, 1] = d["episode_ids"][:take].astype(np.int64)
        row += take
        pbar.set_postfix(rows=f"{row}/{total}", rate=f"{row / max(1e-9, time.time() - t0):.0f}/s")

    frames_mm.flush()
    del frames_mm
    np.save(out_dir / "states.npy", states[:row])
    np.save(out_dir / "map_ids.npy", map_ids[:row])
    np.save(out_dir / "group_keys.npy", group_keys[:row])
    index = {
        "shape": [int(row), out_h, out_w, 3],
        "dtype": "uint8",
        "input_res": f"{out_w}x{out_h}",
        "source": str(data_dir),
        "frames_file": "frames_u8.dat",
        "group_key_axes": "(worker_id, episode_id)",
        "bytes": int(row) * out_h * out_w * 3,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[cache] wrote {row} rows to {out_dir} in {time.time() - t0:.1f}s")
    print(f"[cache] index.json: {json.dumps(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
