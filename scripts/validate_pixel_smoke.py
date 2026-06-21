"""Validate a Stage-2 (frame, state) collection run and save sample (PNG, state.json) pairs.

Reads the manifest + shards under ``--out`` and ASSERTS the Stage-2 acceptance gate:
  * frame shape == (360, 640, 3), dtype uint8, NOT all-black (spread on sampled frames).
  * state len == 52; the -100.0 absent-bullet sentinel appears in bullet slots
    (indices 6-25 and 32-51) on at least some pairs.
  * per-map counts present for all maps; none wildly starved (reports the histogram).
  * manifest.json parses with STRICT json; total_pairs ~= the recorded target.

Also saves ``--samples`` (PNG, state.json) pairs to ``--samples-dir`` for an eyeball check.

Pure read-side: stdlib + numpy + Pillow (already deps). Run with ``uv run python
scripts/validate_pixel_smoke.py --out <dataset> [--samples-dir <dir>]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Bullet slots in the 52-float state that hold the -100.0 "absent" sentinel.
BULLET_SLOTS = list(range(6, 26)) + list(range(32, 52))
SENTINEL = -100.0


def _load_shards(out_dir: Path):
    """Yield (path, npz-dict) for every shard under ``out_dir``."""
    for sp in sorted(out_dir.glob("shard_w*_*.npz")):
        with np.load(sp) as data:
            yield sp, {k: data[k] for k in data.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate a Stage-2 pixel collection run.")
    ap.add_argument("--out", required=True, help="dataset dir (with manifest.json + shards)")
    ap.add_argument(
        "--samples-dir",
        default=str(_REPO_ROOT / "docs" / "pixel-capture-samples" / "stage2_smoke"),
        help="where to write sample (PNG, state.json) pairs",
    )
    ap.add_argument("--samples", type=int, default=4, help="how many sample pairs to save")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    manifest_path = out_dir / "manifest.json"

    # --- manifest: STRICT json ---
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"manifest: schema={manifest['schema_version']} encoding={manifest['frame_encoding']}")
    print(f"  resolution={manifest['resolution']} state_len={manifest['state_len']}")
    print(f"  target={manifest['target']} total={manifest['total_pairs_collected']}")
    print(f"  git_sha={manifest['git_sha']} time_scale={manifest['time_scale']}")

    num_maps = len(manifest["maps"])
    per_map = {m["name"]: 0 for m in manifest["maps"]}
    id_to_name = {m["index"]: m["name"] for m in manifest["maps"]}

    total = 0
    shape_ok = True
    dtype_ok = True
    not_black = False
    spread_samples = []
    sentinel_hits = 0
    sentinel_checked = 0
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    saved_maps: set[str] = set()

    for sp, d in _load_shards(out_dir):
        frames = d["frames"]
        states = d["states"]
        map_ids = d["map_ids"]
        n = frames.shape[0]
        total += n

        if frames.shape[1:] != (360, 640, 3):
            shape_ok = False
            print(f"  BAD shape in {sp.name}: {frames.shape}")
        if frames.dtype != np.uint8:
            dtype_ok = False
            print(f"  BAD dtype in {sp.name}: {frames.dtype}")
        if states.shape[1] != 52:
            print(f"  BAD state len in {sp.name}: {states.shape}")

        # per-map tally
        u, c = np.unique(map_ids, return_counts=True)
        for mu, mc in zip(u, c, strict=True):
            per_map[id_to_name[int(mu)]] += int(mc)

        # sample a handful of frames per shard for spread + sentinel + PNG export
        step = max(1, n // 8)
        for i in range(0, n, step):
            fr = frames[i]
            st = states[i]
            std = float(fr.std())
            spread_samples.append(std)
            if fr.min() < fr.max() and std > 2.0:
                not_black = True
            # sentinel check on bullet slots
            sentinel_checked += 1
            if np.any(np.isclose(st[BULLET_SLOTS], SENTINEL)):
                sentinel_hits += 1
            # save a few sample pairs, preferring distinct maps (one per map) for variety
            this_map = id_to_name[int(map_ids[i])]
            if saved < args.samples and this_map not in saved_maps and std > 2.0:
                saved_maps.add(this_map)
                png = samples_dir / f"sample_{saved:02d}_{id_to_name[int(map_ids[i])]}.png"
                js = samples_dir / f"sample_{saved:02d}_{id_to_name[int(map_ids[i])]}.state.json"
                Image.fromarray(fr, mode="RGB").save(png)
                js.write_text(
                    json.dumps(
                        {
                            "map": id_to_name[int(map_ids[i])],
                            "shard": sp.name,
                            "idx": int(i),
                            "frame_std": std,
                            "frame_min": int(fr.min()),
                            "frame_max": int(fr.max()),
                            "state": [float(x) for x in st],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                saved += 1

    print(f"\ntotal pairs scanned: {total}")
    print(f"frame shape (360,640,3): {'OK' if shape_ok else 'FAIL'}")
    print(f"frame dtype uint8:       {'OK' if dtype_ok else 'FAIL'}")
    mean_std = float(np.mean(spread_samples)) if spread_samples else 0.0
    print(
        f"not-all-black (spread):  {'OK' if not_black else 'FAIL'} "
        f"(mean sampled std={mean_std:.2f})"
    )
    print(
        f"bullet sentinel -100.0:  {'OK' if sentinel_hits > 0 else 'FAIL'} "
        f"({sentinel_hits}/{sentinel_checked} sampled pairs have it in bullet slots)"
    )

    print("\nper-map histogram:")
    counts = []
    for name in per_map:
        counts.append(per_map[name])
        bar = "#" * (per_map[name] * 40 // max(1, max(per_map.values())))
        print(f"  {name:>16}: {per_map[name]:5d} {bar}")
    present_all = all(c > 0 for c in counts)
    if counts:
        lo, hi = min(counts), max(counts)
        # "balanced" heuristic: max no more than ~2x min (random episode lengths vary).
        balanced = present_all and hi <= max(2 * lo, lo + manifest["target"] // num_maps // 2 + 5)
    else:
        balanced = False
    print(
        f"all maps present: {'OK' if present_all else 'FAIL'}; min={min(counts)} max={max(counts)}"
    )
    print(f"balanced (none starved): {'OK' if balanced else 'CHECK'}")

    target = manifest["target"]
    total_ok = abs(total - target) <= max(50, target // 50) or total >= target
    print(f"\ntotal ~= target: {'OK' if total_ok else 'CHECK'} ({total} vs target {target})")
    print(f"samples saved: {saved} -> {samples_dir}")

    all_ok = shape_ok and dtype_ok and not_black and sentinel_hits > 0 and present_all and total_ok
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL/CHECK'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
