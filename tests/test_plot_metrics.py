"""Unit tests for the PURE CSV->series extraction in tank_twin.plot_metrics.

These tests exercise ONLY the pure path (stdlib csv -> numeric series). They do NOT import
matplotlib, torch, or Unity, and they need no display: matplotlib is imported lazily inside
plot_metrics.render_dashboard, never by extract_series / read_progress_rows. We assert:

* correct columns are extracted and paired with time/total_timesteps,
* missing columns are omitted (not crashed on),
* empty / partial cells are tolerated (the bad point is dropped, others survive),
* an empty CSV / no x-axis column yields {} rather than an exception,
* the train/entropy_loss -> train/std fallback resolves.
"""

import csv

from tank_twin.plot_metrics import (
    PANELS,
    X_KEY,
    extract_series,
    read_progress_rows,
)

# A canonical header covering all six panels + the x-axis, mirroring SB3 PPO's progress.csv.
_FULL_HEADER = [
    X_KEY,
    "rollout/ep_rew_mean",
    "train/explained_variance",
    "train/approx_kl",
    "train/entropy_loss",
    "train/clip_fraction",
    "time/fps",
]


def _write_csv(path, header, rows):
    """Write a progress.csv-shaped file; rows are lists aligned to header."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_extracts_all_panels_with_correct_points(tmp_path):
    path = _write_csv(
        tmp_path / "progress.csv",
        _FULL_HEADER,
        [
            [256, 1.0, 0.10, 0.01, -0.5, 0.2, 120.0],
            [512, 2.0, 0.30, 0.02, -0.4, 0.3, 130.0],
        ],
    )
    rows = read_progress_rows(path)
    series = extract_series(rows)

    # Every panel resolved.
    assert set(series.keys()) == {key for key, _title in PANELS}

    rew = series["rollout/ep_rew_mean"]
    assert rew["column"] == "rollout/ep_rew_mean"
    assert rew["x"] == [256.0, 512.0]
    assert rew["y"] == [1.0, 2.0]
    # Titles come from PANELS and are carried through.
    assert rew["title"] == dict(PANELS)["rollout/ep_rew_mean"]


def test_missing_columns_are_omitted_not_crashed(tmp_path):
    # Only ep_rew_mean + x present; the other five panels have no column.
    header = [X_KEY, "rollout/ep_rew_mean"]
    path = _write_csv(tmp_path / "progress.csv", header, [[256, 1.5], [512, 2.5]])
    series = extract_series(read_progress_rows(path))

    assert set(series.keys()) == {"rollout/ep_rew_mean"}
    assert series["rollout/ep_rew_mean"]["y"] == [1.5, 2.5]


def test_partial_and_empty_cells_are_tolerated(tmp_path):
    # SB3 leaves train/* cells empty on the first rollout (before any optimizer step).
    path = _write_csv(
        tmp_path / "progress.csv",
        _FULL_HEADER,
        [
            # First row: train/* blank, ep_rew_mean present.
            [256, 1.0, "", "", "", "", 100.0],
            # Second row: fully populated.
            [512, 2.0, 0.25, 0.02, -0.4, 0.3, 110.0],
            # Third row: a non-numeric junk cell for approx_kl -> that point dropped.
            [768, 3.0, 0.40, "nan?", -0.3, 0.4, 120.0],
        ],
    )
    series = extract_series(read_progress_rows(path))

    # ep_rew_mean has all three points.
    assert series["rollout/ep_rew_mean"]["x"] == [256.0, 512.0, 768.0]
    # explained_variance drops the first (blank) row.
    assert series["train/explained_variance"]["x"] == [512.0, 768.0]
    assert series["train/explained_variance"]["y"] == [0.25, 0.40]
    # approx_kl drops both the blank first row and the junk third row.
    assert series["train/approx_kl"]["x"] == [512.0]
    assert series["train/approx_kl"]["y"] == [0.02]


def test_empty_csv_yields_empty_mapping(tmp_path):
    # Header only, no data rows.
    path = _write_csv(tmp_path / "progress.csv", _FULL_HEADER, [])
    assert extract_series(read_progress_rows(path)) == {}


def test_no_x_axis_column_yields_empty_mapping(tmp_path):
    # No time/total_timesteps -> nothing plottable against the shared x-axis.
    header = ["rollout/ep_rew_mean", "train/approx_kl"]
    path = _write_csv(tmp_path / "progress.csv", header, [[1.0, 0.01], [2.0, 0.02]])
    assert extract_series(read_progress_rows(path)) == {}


def test_entropy_loss_falls_back_to_std(tmp_path):
    # A continuous-policy run that logs train/std instead of train/entropy_loss.
    header = [X_KEY, "rollout/ep_rew_mean", "train/std"]
    path = _write_csv(tmp_path / "progress.csv", header, [[256, 1.0, 0.9], [512, 2.0, 0.8]])
    series = extract_series(read_progress_rows(path))

    # The entropy_loss panel resolves via the std fallback.
    assert "train/entropy_loss" in series
    assert series["train/entropy_loss"]["column"] == "train/std"
    assert series["train/entropy_loss"]["y"] == [0.9, 0.8]


def test_panel_with_column_but_all_cells_blank_is_omitted(tmp_path):
    # approx_kl column exists but every cell is empty -> no usable points -> omitted.
    header = [X_KEY, "rollout/ep_rew_mean", "train/approx_kl"]
    path = _write_csv(tmp_path / "progress.csv", header, [[256, 1.0, ""], [512, 2.0, ""]])
    series = extract_series(read_progress_rows(path))

    assert "rollout/ep_rew_mean" in series
    assert "train/approx_kl" not in series
