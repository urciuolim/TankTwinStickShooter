"""Unit tests for the pixel-pretrain SWEEP orchestrator (scripts/sweep.py).

Torch-free, GPU-free, NO subprocess launches: these exercise ONLY the pure seam
(grid generation, argv building, metrics parsing, sorting, CSV / summary formatting).
The orchestrator launches the trainer in a child process; that path is NOT tested here
(it needs a GPU + the cache) -- it is exercised by the operator's smoke run instead.

``scripts/sweep.py`` is not part of the importable ``tank_twin`` package, so it is loaded
by absolute path via importlib.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SWEEP_PATH = _REPO_ROOT / "scripts" / "sweep.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location("_sweep_under_test", _SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep()


# =============================================================================
# generate_grid
# =============================================================================
def test_grid_has_exactly_twelve_unique_configs():
    grid = sweep.generate_grid()
    assert len(grid) == 12
    seen = {(c["lr"], c["embedding_dim"], c["batch"]) for c in grid}
    assert len(seen) == 12  # all unique


def test_grid_is_the_expected_cartesian_product():
    grid = sweep.generate_grid()
    expected = {
        (lr, emb, batch) for lr in (3e-4, 1e-3) for emb in (64, 32) for batch in (256, 512, 1024)
    }
    got = {(c["lr"], c["embedding_dim"], c["batch"]) for c in grid}
    assert got == expected


def test_grid_order_is_deterministic():
    # lr outermost, then embedding_dim, then batch innermost.
    grid = sweep.generate_grid()
    assert grid[0] == {"lr": 3e-4, "embedding_dim": 64, "batch": 256}
    assert grid[1] == {"lr": 3e-4, "embedding_dim": 64, "batch": 512}
    assert grid[2] == {"lr": 3e-4, "embedding_dim": 64, "batch": 1024}
    assert grid[3] == {"lr": 3e-4, "embedding_dim": 32, "batch": 256}
    assert grid[-1] == {"lr": 1e-3, "embedding_dim": 32, "batch": 1024}
    # generate_grid is pure: same output every call.
    assert sweep.generate_grid() == grid


# =============================================================================
# build_trainer_argv
# =============================================================================
def _fixed(**over):
    base = {
        "out_base": Path("runs/sweep/lr3e-04_e64_b256"),
        "cache_dir": Path("datasets/pixel_cache_160x90"),
        "input_res": "160x90",
        "epochs": 6,
    }
    base.update(over)
    return base


def _argv_pairs(argv):
    """Map a flag -> its following value for assertions (assumes flag/value pairs)."""
    out = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--") and i + 1 < len(argv):
            out[tok] = argv[i + 1]
    return out


def test_argv_starts_with_module_invocation():
    argv = sweep.build_trainer_argv({"lr": 3e-4, "embedding_dim": 64, "batch": 256}, _fixed())
    assert argv[:3] == [sys.executable, "-m", "tank_twin.pretrain_pixels"]
    assert all(isinstance(tok, str) for tok in argv)


def test_argv_carries_fixed_and_swept_flags():
    config = {"lr": 1e-3, "embedding_dim": 32, "batch": 512}
    argv = sweep.build_trainer_argv(config, _fixed(epochs=6))
    pairs = _argv_pairs(argv)
    # Fixed regularization / objective flags.
    assert pairs["--weight-decay"] == "1e-4"
    assert pairs["--lr-schedule"] == "cosine"
    assert pairs["--disable"] == "player_aim"
    assert pairs["--seed"] == "0"
    assert pairs["--cache-dir"] == str(Path("datasets/pixel_cache_160x90"))
    assert pairs["--input-res"] == "160x90"
    assert pairs["--epochs"] == "6"
    # Swept per-config flags.
    assert float(pairs["--lr"]) == pytest.approx(1e-3)
    assert pairs["--embedding-dim"] == "32"
    assert pairs["--batch"] == "512"


def test_argv_does_not_pass_holdout_or_pos_weights():
    # Default 2-map holdout and `auto` pos-weights must be LEFT at trainer defaults.
    argv = sweep.build_trainer_argv({"lr": 3e-4, "embedding_dim": 64, "batch": 256}, _fixed())
    assert "--holdout-maps" not in argv
    assert "--presence-pos-weight" not in argv
    assert "--wall-pos-weight" not in argv
    # player_aim is disabled (not in --objectives allowlist form).
    assert "--objectives" not in argv


def test_argv_optional_subset_and_device():
    argv = sweep.build_trainer_argv(
        {"lr": 3e-4, "embedding_dim": 64, "batch": 256},
        _fixed(subset=12000, device="cuda"),
    )
    pairs = _argv_pairs(argv)
    assert pairs["--subset"] == "12000"
    assert pairs["--device"] == "cuda"


# =============================================================================
# parse_metrics_json
# =============================================================================
def _synthetic_metrics(*, with_validation=True):
    """A small metrics.json dict matching the real trainer schema."""
    per_obj_eval = {
        "player_velocity": "N/A",  # disabled-group sentinel
        "player_aim": "N/A",  # disabled in the sweep
        "player_position": {"norm_mse": 0.02, "world_rmse": 0.31},
        "bullet_presence": {"precision": 0.81, "recall": 0.74, "f1": 0.775},
        "bullet_position": {"norm_mse": 0.12, "world_mae": 0.45},
        "bullet_direction": {"norm_mse": 0.33},
        "walls": {
            "interior_f1": 0.95,
            "interior_precision": 0.96,
            "interior_recall": 0.94,
            "interior_iou": 0.91,
        },
    }
    per_obj_val = {
        "player_velocity": "N/A",
        "player_aim": "N/A",
        "player_position": {"norm_mse": 0.05, "world_rmse": 0.40},
        "bullet_presence": {"precision": 0.70, "recall": 0.60, "f1": 0.648},
        "bullet_position": {"norm_mse": 0.20, "world_mae": 0.62},
        "bullet_direction": {"norm_mse": 0.40},
        "walls": {
            "interior_f1": 0.88,
            "interior_precision": 0.90,
            "interior_recall": 0.86,
            "interior_iou": 0.79,
        },
    }
    metrics = {
        "config": {
            "lr": 0.0003,
            "embedding_dim": 64,
            "batch": 256,
            "weight_decay": 0.0001,
            "lr_schedule": "cosine",
            "input_res": "160x90",
            "epochs": 6,
            "seed": 0,
            "holdout_maps": ["diagonal_pillars", "ring_fragments"],
        },
        # Two epochs: parser must take the LAST one.
        "eval": [
            {
                "epoch": 0,
                "total_loss": 9.9,
                "per_objective": {"bullet_position": {"norm_mse": 9.9}},
            },
            {"epoch": 1, "total_loss": 1.1, "per_objective": per_obj_eval},
        ],
    }
    if with_validation:
        metrics["validation"] = {"total_loss": 1.5, "per_objective": per_obj_val}
    return metrics


def test_parse_metrics_pulls_final_epoch_and_validation():
    row = sweep.parse_metrics_json(_synthetic_metrics())
    # Identity from config.
    assert row["lr"] == 0.0003
    assert row["embedding_dim"] == 64
    assert row["batch"] == 256
    # FINAL-epoch eval (epoch 1), NOT epoch 0.
    assert row["eval_bullet_position_norm_mse"] == pytest.approx(0.12)
    assert row["eval_bullet_position_world_mae"] == pytest.approx(0.45)
    assert row["eval_bullet_presence_f1"] == pytest.approx(0.775)
    assert row["eval_walls_interior_f1"] == pytest.approx(0.95)
    assert row["eval_bullet_direction_norm_mse"] == pytest.approx(0.33)
    # OOD validation.
    assert row["val_bullet_position_world_mae"] == pytest.approx(0.62)
    assert row["val_bullet_presence_f1"] == pytest.approx(0.648)
    assert row["val_walls_interior_f1"] == pytest.approx(0.88)


def test_parse_metrics_handles_na_and_missing_groups():
    metrics = _synthetic_metrics()
    # player_aim is "N/A" in both eval + val: its columns are not even surfaced, but the
    # parser must not crash and the disabled groups stay absent / blank.
    row = sweep.parse_metrics_json(metrics)
    # player_position IS present.
    assert row["eval_player_position_norm_mse"] == pytest.approx(0.02)
    # A group entirely absent from the dict -> None, no crash.
    del metrics["validation"]["per_objective"]["bullet_position"]
    row2 = sweep.parse_metrics_json(metrics)
    assert row2["val_bullet_position_world_mae"] is None
    assert row2["val_bullet_position_norm_mse"] is None


def test_parse_metrics_partial_file_does_not_crash():
    # No 'validation', no 'eval': must return a row with blank metric columns.
    row = sweep.parse_metrics_json({"config": {"lr": 1e-3, "embedding_dim": 32, "batch": 1024}})
    assert row["lr"] == 1e-3
    assert row["embedding_dim"] == 32
    assert row["batch"] == 1024
    assert row["eval_bullet_presence_f1"] is None
    assert row["val_bullet_position_world_mae"] is None


def test_parse_metrics_from_file(tmp_path):
    import json

    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(_synthetic_metrics()), encoding="utf-8")
    row = sweep.parse_metrics_json(p)
    assert row["eval_bullet_position_world_mae"] == pytest.approx(0.45)


# =============================================================================
# sort_leaderboard
# =============================================================================
def _row(lr, emb, batch, *, wmae=None, bpres_f1=None, wall_f1=None, status="done"):
    return {
        "lr": lr,
        "embedding_dim": emb,
        "batch": batch,
        "status": status,
        "val_bullet_position_world_mae": wmae,
        "val_bullet_presence_f1": bpres_f1,
        "val_walls_interior_f1": wall_f1,
    }


def test_sort_primary_is_world_mae_ascending():
    rows = [
        _row(3e-4, 64, 256, wmae=0.50, bpres_f1=0.6, wall_f1=0.9),
        _row(1e-3, 32, 512, wmae=0.20, bpres_f1=0.6, wall_f1=0.9),
        _row(3e-4, 32, 256, wmae=0.35, bpres_f1=0.6, wall_f1=0.9),
    ]
    out = sweep.sort_leaderboard(rows)
    assert [r["val_bullet_position_world_mae"] for r in out] == [0.20, 0.35, 0.50]


def test_sort_tiebreak_presence_f1_then_wall_f1_descending():
    rows = [
        _row(3e-4, 64, 256, wmae=0.30, bpres_f1=0.60, wall_f1=0.80),
        _row(1e-3, 64, 256, wmae=0.30, bpres_f1=0.70, wall_f1=0.70),
        _row(3e-4, 32, 256, wmae=0.30, bpres_f1=0.70, wall_f1=0.90),
    ]
    out = sweep.sort_leaderboard(rows)
    # Same wmae: higher presence_f1 first; among equal presence_f1, higher wall_f1 first.
    assert out[0]["val_bullet_presence_f1"] == 0.70
    assert out[0]["val_walls_interior_f1"] == 0.90
    assert out[1]["val_walls_interior_f1"] == 0.70
    assert out[2]["val_bullet_presence_f1"] == 0.60


def test_sort_puts_failed_and_missing_rows_last():
    rows = [
        _row(3e-4, 64, 256, status="failed"),  # no metrics
        _row(1e-3, 32, 512, wmae=0.40, bpres_f1=0.6, wall_f1=0.9),
        _row(3e-4, 32, 256, status="timeout"),  # no metrics
        _row(1e-3, 64, 512, wmae=0.10, bpres_f1=0.6, wall_f1=0.9),
    ]
    out = sweep.sort_leaderboard(rows)
    # The two rows WITH metrics come first (sorted by wmae), failures last.
    assert out[0]["val_bullet_position_world_mae"] == 0.10
    assert out[1]["val_bullet_position_world_mae"] == 0.40
    assert {out[2]["status"], out[3]["status"]} == {"failed", "timeout"}


# =============================================================================
# write_csv / format_summary
# =============================================================================
def test_csv_header_and_row_roundtrip(tmp_path):
    row = sweep.parse_metrics_json(_synthetic_metrics())
    row["status"] = "done"
    row["wall_time_s"] = 1234.5
    row["run_dir"] = "runs/sweep/lr3e-04_e64_b256_160x90_e64_b256_0621-2200"
    out = tmp_path / "leaderboard.csv"
    sweep.write_csv([row], out)

    text = out.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    assert reader.fieldnames == list(sweep.LEADERBOARD_COLUMNS)
    parsed = list(reader)
    assert len(parsed) == 1
    rec = parsed[0]
    # Sort keys are the first three columns.
    assert reader.fieldnames[:3] == [
        "val_bullet_position_world_mae",
        "val_bullet_presence_f1",
        "val_walls_interior_f1",
    ]
    # Round-trip a few values (CSV stores strings).
    assert float(rec["val_bullet_position_world_mae"]) == pytest.approx(0.62)
    assert float(rec["eval_bullet_presence_f1"]) == pytest.approx(0.775)
    assert int(rec["embedding_dim"]) == 64
    assert int(rec["batch"]) == 256
    assert rec["status"] == "done"
    assert float(rec["wall_time_s"]) == pytest.approx(1234.5)
    assert rec["run_dir"].endswith("0621-2200")


def test_csv_blank_cells_for_missing_metrics(tmp_path):
    # A failed config: identity + status present, all metric cells blank.
    row = {
        "lr": 1e-3,
        "embedding_dim": 32,
        "batch": 1024,
        "status": "failed",
        "wall_time_s": 12.0,
        "run_dir": None,
    }
    out = tmp_path / "lb.csv"
    sweep.write_csv([row], out)
    rec = next(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))
    assert rec["status"] == "failed"
    assert rec["val_bullet_position_world_mae"] == ""  # blank, did not crash
    assert rec["eval_bullet_presence_f1"] == ""
    assert rec["run_dir"] == ""


def test_format_summary_is_sorted_and_contains_configs():
    rows = [
        _row(3e-4, 64, 256, wmae=0.50, bpres_f1=0.6, wall_f1=0.9),
        _row(1e-3, 32, 512, wmae=0.20, bpres_f1=0.6, wall_f1=0.9),
    ]
    rows_sorted = sweep.sort_leaderboard(rows)
    text = sweep.format_summary(rows_sorted)
    assert "LEADERBOARD" in text
    # The better (lower wmae) config's row should appear before the worse one.
    assert text.index("0.2") < text.index("0.5")


# =============================================================================
# config_label
# =============================================================================
def test_config_label_is_filesystem_safe_and_unique():
    grid = sweep.generate_grid()
    labels = [sweep.config_label(c) for c in grid]
    assert len(set(labels)) == 12  # unique per config
    for lbl in labels:
        # No path separators / spaces; embeds emb + batch for run-dir disambiguation.
        assert "/" not in lbl and "\\" not in lbl and " " not in lbl
        assert "_e" in lbl and "_b" in lbl


# =============================================================================
# estimate_config_seconds (labeled-assumption ETA)
# =============================================================================
def test_estimate_scales_with_epochs_and_rows():
    s1 = sweep.estimate_config_seconds(1, 256, train_rows=100_000, pairs_per_sec=1000.0)
    s6 = sweep.estimate_config_seconds(6, 256, train_rows=100_000, pairs_per_sec=1000.0)
    assert s6 == pytest.approx(6 * s1)
    # 100k rows * 1 epoch / 1000 pps = 100s, plus ~12% overhead.
    assert s1 == pytest.approx(100.0 * 1.12)
