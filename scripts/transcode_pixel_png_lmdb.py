"""Transcode the pixel_1m shards ONCE into a PNG-in-LMDB cache for the FULL run.

Sibling of ``scripts/prepare_pixel_cache.py``. That builder writes a contiguous flat
uint8 memmap (N*H*W*3 bytes); at native 640x360 over 918K frames that is
``918000 * 360 * 640 * 3 = 634,521,600,000 bytes ~= 591 GiB`` — too big to keep on
disk. This builder instead PNG-encodes each frame (LOSSLESS — JPEG would blur the
1-2px bullets/tank features) and stores the bytes in an LMDB key/value env, so the
on-disk size is the sum of the compressed PNGs (typically a small fraction of the
flat memmap for these sparse synthetic frames). Epochs then pull ``png_bytes`` by
row index and decode on the CPU worker (see ``LmdbPngPixelDataset``).

READ-ONLY over the shards (never modifies ``datasets/pixel_1m``). Writes a NEW cache
dir. Mirrors prepare_pixel_cache's resize semantics: with the default
``--input-res 640x360`` (== native) NO resize happens (the native frame is encoded
verbatim); a smaller ``--input-res`` resizes first via the GPU ``interpolate`` (area)
path when CUDA is available, else the pure-numpy block-mean fallback.

LMDB KEY CONVENTION: the integer row index ``i`` (0-based, global across shards in
the same order prepare_pixel_cache fills its memmap) encoded as ``f"{i:08d}".encode()``
(zero-padded 8-digit ASCII). The aligned sidecars (``states.npy``, ``map_ids.npy``,
``group_keys.npy``) put row ``i`` at array index ``i``, so LMDB key ``f"{i:08d}"`` <->
sidecar row ``i``. ``LmdbPngPixelDataset`` MUST decode with the SAME convention.

SIDECARS (IDENTICAL semantics to prepare_pixel_cache, so the split logic is reusable
unchanged): ``states.npy`` (N,52 f32), ``map_ids.npy`` (N, int32), ``group_keys.npy``
(N,2 int64 = worker_id, episode_id), and ``index.json``. ``index.json`` carries the
record ``shape`` ``[N,H,W,3]``, ``dtype`` ``uint8``, ``input_res``, ``source``, the
``backend`` tag ``"lmdb_png"``, the ``lmdb_file`` / ``key_format`` / ``key_convention``,
and the frame ``count``.

RESUME / ATOMICITY: the run is resumable and skip-existing. On (re)start it reads the
LMDB ``stat()`` entry count and skips any row index ``< n_existing`` (rows are written
in strictly increasing index order, so the LMDB always holds a dense ``[0, n_existing)``
prefix). Already-encoded rows are NOT re-encoded. Each shard's frames are written in one
LMDB write txn (committed per shard) so a crash mid-run leaves a clean dense prefix, not a
half-written key. The sidecars + ``index.json`` are written LAST and ATOMICALLY (temp file
+ ``os.replace``); ``index.json`` existing therefore means the cache is complete. A partial
run has the LMDB populated but NO ``index.json`` — re-running resumes from the LMDB count
and finishes the sidecars. (Single-pass loop; no multiprocessing — keeping the dense-prefix
resume contract simple. The work is embarrassingly parallel across shards if a future
version wants to shard the LMDB, but that is out of scope here.)

Run (FULL, native 640x360)::

    uv run python scripts/transcode_pixel_png_lmdb.py \
        --data-dir datasets/pixel_1m --out datasets/pixel_cache_png_640x360 --input-res 640x360
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tank_twin.pretrain_pixels import (  # noqa: E402
    downsample_frames_np,
    list_shards,
    parse_input_res,
    png_encode,
    worker_id_from_shard,
)

# Native shard resolution (frames stored in the .npz at this size).
NATIVE_W, NATIVE_H = 640, 360

# LMDB key format: zero-padded 8-digit ASCII of the 0-based global row index.
KEY_FORMAT = "{:08d}"
BACKEND_TAG = "lmdb_png"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def row_key(i: int) -> bytes:
    """Canonical LMDB key for global row index ``i`` (zero-padded 8-digit ASCII)."""
    return KEY_FORMAT.format(i).encode("ascii")


def _resize_gpu(frames_u8: np.ndarray, out_w: int, out_h: int):
    """GPU area-resize a (N,360,640,3) uint8 batch -> (N,out_h,out_w,3) uint8.

    Returns None if torch/CUDA unavailable so the caller falls back to numpy. Mirrors
    prepare_pixel_cache._resize_gpu exactly (same area mode / round / clamp).
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


def _resize_batch(fr: np.ndarray, out_w: int, out_h: int, *, force_cpu: bool) -> np.ndarray:
    """Resize a (N,360,640,3) uint8 batch to (N,out_h,out_w,3); identity at native res."""
    if out_w == NATIVE_W and out_h == NATIVE_H:
        return np.ascontiguousarray(fr, dtype=np.uint8)  # native: encode verbatim, no resize
    ds = None if force_cpu else _resize_gpu(fr, out_w, out_h)
    if ds is None:
        ds = downsample_frames_np(fr, out_w, out_h)
    return ds


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + os.replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """np.save to a temp file then os.replace, so a reader never sees a half-written .npy."""
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _allocated_bytes(path: Path) -> int:
    """ACTUAL on-disk bytes of ``path`` (sparse-aware), not the apparent map_size.

    The LMDB data file is sized to map_size (~700 GiB apparent) but is SPARSE, so
    ``st_size`` wildly overstates real usage. On POSIX use ``st_blocks * 512``; on Windows
    use ``GetCompressedFileSize`` (which returns the allocated, sparse-aware size). Falls
    back to ``st_size`` only if neither is available.
    """
    st = path.stat()
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None:  # POSIX
        return int(blocks) * 512
    if os.name == "nt":  # Windows: allocated size of a sparse file
        try:
            import ctypes
            from ctypes import wintypes

            high = wintypes.DWORD(0)
            get_size = ctypes.windll.kernel32.GetCompressedFileSizeW
            low = get_size(str(path), ctypes.byref(high))
            if low != 0xFFFFFFFF:  # not INVALID_FILE_SIZE
                return (int(high.value) << 32) | int(low)
        except OSError:
            pass
    return int(st.st_size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="datasets/pixel_1m")
    ap.add_argument("--out", required=True, help="NEW cache dir to write")
    ap.add_argument("--input-res", default="640x360", help="encode target WxH (default native)")
    ap.add_argument("--limit", type=int, default=None, help="cap total pairs (testing)")
    ap.add_argument("--cpu", action="store_true", help="force numpy block-mean (no GPU resize)")
    ap.add_argument(
        "--map-size-gb",
        type=float,
        default=700.0,
        help="LMDB map_size in GiB (sparse on disk; size generously above the PNG total)",
    )
    args = ap.parse_args(argv)

    out_w, out_h = parse_input_res(args.input_res)
    data_dir = _resolve(args.data_dir)
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "frames_png.lmdb"

    shards = list_shards(data_dir)
    # First pass: total N (cheap header read), matching prepare_pixel_cache.
    counts = []
    for s in tqdm(shards, desc="counting shards", unit="shard"):
        with np.load(s) as d:
            counts.append(int(d["map_ids"].shape[0]))
    total = sum(counts)
    if args.limit is not None:
        total = min(total, args.limit)

    map_size = int(args.map_size_gb * 1024**3)
    print(
        f"[png-lmdb] N={total} res={out_w}x{out_h} -> {lmdb_path} "
        f"(map_size={args.map_size_gb:.0f} GiB, sparse) key={KEY_FORMAT!r}"
    )

    # Resume: open the env and read how many dense-prefix rows are already written.
    # subdir=False -> a single .lmdb data file (simpler to ship than a dir env).
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        subdir=False,
        readonly=False,
        meminit=False,
        max_dbs=0,
    )
    n_existing = env.stat()["entries"]
    if n_existing:
        print(f"[png-lmdb] RESUME: {n_existing} rows already in LMDB; skipping those re-encodes")
    if n_existing > total:
        env.close()
        raise SystemExit(
            f"[png-lmdb] LMDB already has {n_existing} rows > requested total {total}; "
            "point --out at a fresh dir or raise --limit."
        )

    states = np.zeros((total, 52), dtype=np.float32)
    map_ids = np.zeros((total,), dtype=np.int32)
    group_keys = np.zeros((total, 2), dtype=np.int64)  # (worker_id, episode_id)

    row = 0
    t0 = time.time()
    pbar = tqdm(shards, desc="encoding frames", unit="shard")
    for s in pbar:
        if row >= total:
            break
        wid = worker_id_from_shard(s.name)
        with np.load(s) as d:
            n = int(d["map_ids"].shape[0])
            take = min(n, total - row)
            # Always fill the sidecars (cheap; needed for the final atomic write).
            states[row : row + take] = d["states"][:take].astype(np.float32)
            map_ids[row : row + take] = d["map_ids"][:take].astype(np.int32)
            group_keys[row : row + take, 0] = wid
            group_keys[row : row + take, 1] = d["episode_ids"][:take].astype(np.int64)
            # Skip frames whose rows are already a dense prefix in the LMDB (resume).
            if row + take <= n_existing:
                row += take
                pbar.set_postfix(rows=f"{row}/{total}", skip="resume")
                continue
            fr = d["frames"][:take]
            ds = _resize_batch(fr, out_w, out_h, force_cpu=args.cpu)
            # One write txn per shard: a crash leaves a clean dense prefix (committed
            # shards), never a half-written key.
            with env.begin(write=True) as txn:
                for j in range(take):
                    gi = row + j
                    if gi < n_existing:
                        continue  # partial-shard resume boundary
                    txn.put(row_key(gi), png_encode(ds[j]), overwrite=True)
        row += take
        pbar.set_postfix(rows=f"{row}/{total}", rate=f"{row / max(1e-9, time.time() - t0):.0f}/s")

    env.sync()
    env.close()

    # Sidecars + index.json written LAST and ATOMICALLY: index.json existing == complete.
    _atomic_save_npy(out_dir / "states.npy", states[:row])
    _atomic_save_npy(out_dir / "map_ids.npy", map_ids[:row])
    _atomic_save_npy(out_dir / "group_keys.npy", group_keys[:row])
    # ACTUAL allocated bytes (the LMDB file is SPARSE: st_size == map_size, way overstated).
    on_disk = _allocated_bytes(lmdb_path) if lmdb_path.exists() else 0
    apparent = lmdb_path.stat().st_size if lmdb_path.exists() else 0
    index = {
        "shape": [int(row), out_h, out_w, 3],
        "dtype": "uint8",
        "input_res": f"{out_w}x{out_h}",
        "source": str(data_dir),
        "backend": BACKEND_TAG,
        "lmdb_file": "frames_png.lmdb",
        "key_format": KEY_FORMAT,
        "key_convention": (
            "global 0-based row index, zero-padded 8-digit ASCII; LMDB key f'{i:08d}' "
            "<-> sidecar row i (states/map_ids/group_keys index i)."
        ),
        "group_key_axes": "(worker_id, episode_id)",
        "count": int(row),
        "lmdb_bytes_allocated": int(on_disk),
        "lmdb_bytes_apparent": int(apparent),
    }
    _atomic_write_bytes(out_dir / "index.json", json.dumps(index, indent=2).encode("utf-8"))
    print(
        f"[png-lmdb] wrote {row} rows to {out_dir} in {time.time() - t0:.1f}s "
        f"(lmdb allocated {on_disk / 1e6:.1f} MB on disk; apparent/map_size "
        f"{apparent / 1e9:.0f} GB sparse)"
    )
    print(f"[png-lmdb] index.json: {json.dumps(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
