"""Per-field decomposition of a trained pixel inverse-renderer on its val set.

Answers: is the mediocre-looking player/bullet MSE because POSITION is weak, or because
unobservable fields (velocity, bullet direction) sit at full variance and drag the average?
Loads the saved model, runs the val split, reports per-field-group NORMALIZED MSE
(~0 = learned, ~1 = no better than predicting the mean) and position error in world units.

Run: uv run python scripts/eval_per_field.py --run runs/pixel_stage2_full --cache-dir datasets/pixel_cache_160x90
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tank_twin._pretrain_pixels_train import InverseRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/pixel_stage2_full")
    ap.add_argument("--cache-dir", default="datasets/pixel_cache_160x90")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    run, cache = Path(args.run), Path(args.cache_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.loads((run / "config.json").read_text())
    ns = json.loads((run / "norm_stats.json").read_text())
    p_idx = np.array(ns["player_indices"])  # 12
    b_idx = np.array(ns["bullet_position_indices"])  # 40
    p_mean, p_std = np.array(ns["player_mean"]), np.array(ns["player_std"])
    b_mean, b_std = np.array(ns["bullet_mean"]), np.array(ns["bullet_std"])
    w, h = (int(x) for x in cfg["input_res"].split("x"))

    model = InverseRenderer(h, w, cfg["embedding_dim"]).to(dev)
    model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
    model.eval()

    idx = json.loads((cache / "index.json").read_text())
    n = idx["shape"][0]
    frames = np.memmap(cache / "frames_u8.dat", dtype=np.uint8, mode="r", shape=tuple(idx["shape"]))
    states = np.load(cache / "states.npy")
    gkeys = np.load(cache / "group_keys.npy")  # (N,2) worker,episode
    val_set = {tuple(g) for g in json.loads((run / "val_groups.json").read_text())["val_groups"]}
    val_rows = np.array([i for i in range(n) if tuple(gkeys[i]) in val_set])

    # Accumulate squared error in NORMALIZED space per field; positions also in world units.
    p_sse = np.zeros(12)
    b_sse = np.zeros(40)
    b_cnt = np.zeros(40)
    p_world_sse = np.zeros(12)
    nseen = 0
    with torch.no_grad():
        for s in range(0, len(val_rows), args.batch):
            rows = val_rows[s : s + args.batch]
            fr = torch.from_numpy(np.ascontiguousarray(frames[rows])).to(dev).float().div_(255)
            fr = fr.permute(0, 3, 1, 2)
            out = model(fr)
            pp = out["player"].cpu().numpy()  # normalized
            bp = out["bullet_pos"].cpu().numpy()
            st = states[rows]
            pt = (st[:, p_idx] - p_mean) / p_std  # normalized target
            p_sse += ((pp - pt) ** 2).sum(0)
            p_world_sse += ((pp * p_std + p_mean - st[:, p_idx]) ** 2).sum(0)
            bt_raw = st[:, b_idx]
            present = bt_raw != -100.0
            bt = (bt_raw - b_mean) / b_std
            b_sse += np.where(present, (bp - bt) ** 2, 0.0).sum(0)
            b_cnt += present.sum(0)
            nseen += len(rows)

    p_nmse = p_sse / nseen
    p_wrmse = np.sqrt(p_world_sse / nseen)
    b_nmse = b_sse / np.clip(b_cnt, 1, None)

    # Player field order per side: [pos_x,pos_y, vel_x,vel_y, aim_x,aim_y]; P1=0..5, P2=6..11.
    def grp(arr, a, b):
        return float(arr[[a, b, a + 6, b + 6]].mean())

    print(f"val pairs={nseen}  (normalized MSE: ~0 learned, ~1.0 = no better than the mean)\n")
    print("PLAYER (avg over P1+P2):")
    print(f"  position   norm_mse={grp(p_nmse, 0, 1):.3f}   world_rmse={grp(p_wrmse, 0, 1):.3f} units")
    print(f"  velocity   norm_mse={grp(p_nmse, 2, 3):.3f}   world_rmse={grp(p_wrmse, 2, 3):.3f} units")
    print(f"  aim        norm_mse={grp(p_nmse, 4, 5):.3f}   world_rmse={grp(p_wrmse, 4, 5):.3f} units")
    # Bullets: within each 4-stride, offsets 0,1 = pos ; 2,3 = vec (direction).
    pos_mask = np.array([i % 4 in (0, 1) for i in range(40)])
    print("\nBULLETS (present slots only):")
    print(f"  position(pos_x,pos_y)  norm_mse={b_nmse[pos_mask].mean():.3f}")
    print(f"  direction(vec_x,vec_y) norm_mse={b_nmse[~pos_mask].mean():.3f}")


if __name__ == "__main__":
    main()
