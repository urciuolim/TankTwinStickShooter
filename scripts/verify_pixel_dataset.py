"""Salvage-verify the pixel (frame, state) dataset and synthesize a manifest.

The 1M collection crashed at an episode-boundary handshake (a protocol desync, NOT a
capture), so the already-flushed shards should be clean. This proves it: it counts every
shard cheaply (map_ids only), sweeps a frame SAMPLE for degenerate/anomalous readbacks (the
signature a GPU-lagged bad frame would leave), sanity-checks the 52-float states + sentinels,
saves a few (PNG,state) eyeball pairs spanning all workers, and writes a manifest.json for the
salvaged shards.

Run: uv run python scripts/verify_pixel_dataset.py --dir datasets/pixel_1m
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def map_names():
    """map_id -> name, by sorted map-config stem (matches the harness rotation order)."""
    cfgs = sorted((REPO / "exp-configs" / "maps").glob("*.json"))
    return [c.stem for c in cfgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datasets/pixel_1m")
    ap.add_argument("--sample-shards", type=int, default=40)
    args = ap.parse_args()

    d = (REPO / args.dir).resolve() if not Path(args.dir).is_absolute() else Path(args.dir)
    shards = sorted(d.glob("shard_w*_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {d}")
    names = map_names()
    nmaps = len(names)

    total = 0
    per_worker = {}
    per_map = np.zeros(nmaps, dtype=np.int64)
    shard_meta = []
    for sp in shards:
        wid = int(sp.stem.split("_")[1][1:])
        with np.load(sp) as z:  # lazy; only touch the small int array
            mids = z["map_ids"]
        n = int(mids.shape[0])
        total += n
        per_worker[wid] = per_worker.get(wid, 0) + n
        u, c = np.unique(mids, return_counts=True)
        for uu, cc in zip(u, c, strict=True):
            if 0 <= int(uu) < nmaps:
                per_map[int(uu)] += int(cc)
        shard_meta.append({"file": sp.name, "worker": wid, "n": n})

    # Frame-health + state sanity on a sample spread across the shard list.
    idxs = np.linspace(0, len(shards) - 1, min(args.sample_shards, len(shards))).astype(int)
    f_std, f_min, f_max, degenerate = [], 999, 0, 0
    bad_state = 0
    samples_dir = d / "verify_samples"
    samples_dir.mkdir(exist_ok=True)
    saved = 0
    seen_workers = set()
    for k in idxs:
        sp = shards[int(k)]
        wid = int(sp.stem.split("_")[1][1:])
        with np.load(sp) as z:
            frames, states = z["frames"], z["states"]
            mids = z["map_ids"]
        # check up to 8 frames per sampled shard
        for j in np.linspace(0, frames.shape[0] - 1, min(8, frames.shape[0])).astype(int):
            fr = frames[j]
            s = float(fr.std())
            f_std.append(s)
            f_min, f_max = min(f_min, int(fr.min())), max(f_max, int(fr.max()))
            if s < 5.0:  # near-constant frame => failed/blank readback
                degenerate += 1
            st = states[j]
            ok = st.shape[0] == 52 and np.all(np.isfinite(st)) and (st[6:26] == -100.0).any()
            if not ok:
                bad_state += 1
        # one eyeball pair per new worker
        if wid not in seen_workers and saved < 6:
            nm = names[int(mids[0])] if 0 <= int(mids[0]) < nmaps else f"map{int(mids[0])}"
            Image.fromarray(frames[0], "RGB").save(samples_dir / f"w{wid}_{nm}_{sp.stem}.png")
            (samples_dir / f"w{wid}_{nm}_{sp.stem}.state.json").write_text(
                json.dumps({"state": states[0].tolist(), "map": nm}, indent=2), encoding="utf-8"
            )
            seen_workers.add(wid)
            saved += 1

    f_std = np.array(f_std)
    manifest = {
        "note": "SALVAGED after a handshake-desync crash; manifest synthesized post-hoc.",
        "resolution": {"w": 640, "h": 360, "c": 3},
        "frame_dtype": "uint8",
        "state_len": 52,
        "total_pairs": total,
        "per_worker": per_worker,
        "maps": [{"id": i, "name": names[i], "count": int(per_map[i])} for i in range(nmaps)],
        "num_shards": len(shards),
        "shards": shard_meta,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"DATASET: {d}")
    print(f"  shards={len(shards)}  total_pairs={total}")
    print(f"  per_worker={per_worker}")
    print("  per_map:")
    for i in range(nmaps):
        print(f"    {names[i]:>16}: {int(per_map[i])}")
    print(f"  frame sample n={len(f_std)}  std mean={f_std.mean():.1f} min={f_std.min():.1f} "
          f"max={f_std.max():.1f}  px_min={f_min} px_max={f_max}")
    print(f"  degenerate(std<5) frames in sample: {degenerate}/{len(f_std)}")
    print(f"  bad states in sample: {bad_state}/{len(f_std)}")
    print(f"  eyeball samples -> {samples_dir}")
    print(f"  manifest -> {d / 'manifest.json'}")


if __name__ == "__main__":
    main()
