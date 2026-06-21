"""Build the wall-occupancy sidecar for a pixel dataset (ADDITIVE; no shard/model change).

Walls are static per map and perfectly grid-aligned, so a layout is fully described by a
binary 20(W) x 12(H) occupancy grid (1=wall [tileID 1], 0=free). Only 10 unique grids exist
for the whole 918K dataset, so storing a per-pair copy is 918K redundant duplicates. Instead
we write a sidecar ``wall_grids.npy`` of shape (10, H=12, W=20) keyed by the SAME ``map_id``
the shards already carry -- the harness's ``_resolve_maps('all')`` order. The training loader
does ``wall = wall_grids[map_id]``.

Orientation: row 0 = top = maxY, to match the top-left-origin frames, so the wall target is
spatially aligned with the input image (no flip for the model to learn).

Run: uv run python scripts/build_wall_grids.py --dir datasets/pixel_1m
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tank_twin.collect_pixels import _resolve_maps  # noqa: E402  (canonical map_id order)


def wall_grid(arena_path):
    w = json.loads(Path(arena_path).read_text(encoding="utf-8"))["Walls"]
    d = w["dims"]
    minX, maxX, minY, maxY = d["minX"], d["maxX"], d["minY"], d["maxY"]
    W, H = maxX - minX + 1, maxY - minY + 1
    g = np.zeros((H, W), dtype=np.uint8)
    for k, ys in w.items():
        if k in ("dims", "tileID"):
            continue
        x = int(k)
        for y in ys:
            g[maxY - int(y), x - minX] = 1  # row 0 = top (maxY), col 0 = minX
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datasets/pixel_1m")
    args = ap.parse_args()
    d = REPO / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)

    arena_paths, names, _ = _resolve_maps("all")
    grids = np.stack([wall_grid(a) for a in arena_paths]).astype(np.uint8)  # (10, 12, 20)
    np.save(d / "wall_grids.npy", grids)
    meta = {
        "shape": list(grids.shape),
        "axes": "(map_id, H=12 [y: maxY..minY, row0=top], W=20 [x: minX..maxX])",
        "value": "1=wall, 0=free",
        "derived_from": "exp-configs/maps/Arenas/*.json via _resolve_maps('all')",
        "maps": [
            {"id": i, "name": names[i], "wall_cells": int(grids[i].sum())}
            for i in range(len(names))
        ],
    }
    (d / "wall_grids_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {d / 'wall_grids.npy'}  shape={grids.shape} (map_id, H, W)")
    for i, n in enumerate(names):
        print(f"  id={i:>2}  {n:>16}  wall_cells={int(grids[i].sum()):>3}/240")
    show = names.index("diagonal_pillars") if "diagonal_pillars" in names else 0
    print(f"\n  ASCII '{names[show]}' (row0=top, should match the frame: pillars low-left->up-right):")
    for row in grids[show]:
        print("   " + "".join("#" if c else "." for c in row))


if __name__ == "__main__":
    main()
