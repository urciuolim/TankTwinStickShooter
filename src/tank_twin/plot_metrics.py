"""Render the SB3 ``progress.csv`` of a run into a PNG "is it learning?" dashboard.

The trainer (:mod:`tank_twin.train`) attaches an SB3 logger that writes
``runs/<run_name>/progress.csv`` (+ TensorBoard ``tfevents``) every rollout. This module
reads that CSV and plots the six headline metrics against ``time/total_timesteps`` so the
board can eyeball whether a run is improving without spinning up TensorBoard.

The CSV -> series extraction is factored into the PURE :func:`extract_series` (stdlib
``csv`` only — NO matplotlib, NO numpy, NO display), so it is unit-testable headless. The
matplotlib import is deferred INTO :func:`render_dashboard` (Agg backend, no display) and
:func:`plot_run`, keeping the pure path matplotlib-free for callers that only want series.

CLI::

    python -m tank_twin.plot_metrics --run-name <name> [--out <path>] [--boardroom]

``--boardroom`` also copies the PNG to ``boardroom/metrics.png`` (stdlib ``shutil``, no
``os.system``) so the boardroom server (port 8777, serves ``boardroom/``) loads it at
``/metrics.png``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "PANELS",
    "X_KEY",
    "extract_series",
    "read_progress_rows",
    "render_dashboard",
    "plot_run",
    "main",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"
BOARDROOM_PNG = _REPO_ROOT / "boardroom" / "metrics.png"

# The shared x-axis for every panel: cumulative environment steps.
X_KEY = "time/total_timesteps"

# The six "is it learning?" panels, in render order: (CSV column, human title).
# train/entropy_loss falls back to train/std when the policy logs std instead (handled in
# extract_series via the fallback list); the rest map 1:1 to SB3 PPO log keys.
PANELS: tuple[tuple[str, str], ...] = (
    ("rollout/ep_rew_mean", "Episode reward (mean)"),
    ("train/explained_variance", "Explained variance"),
    ("train/approx_kl", "Approx KL"),
    ("train/entropy_loss", "Entropy loss"),
    ("train/clip_fraction", "Clip fraction"),
    ("time/fps", "FPS"),
)

# Per-key fallbacks: if the primary column is absent, try these (first present wins).
# PPO logs train/std for a continuous policy; entropy_loss may be absent in some configs.
_FALLBACKS: dict[str, tuple[str, ...]] = {
    "train/entropy_loss": ("train/std",),
}


def read_progress_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Read ``progress.csv`` into a list of raw string-valued row dicts (stdlib csv).

    Pure: no plotting, no numeric coercion. Returns ``[]`` for an empty file. Raises
    ``FileNotFoundError`` if the path does not exist (caller surfaces a clean message).
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str | None) -> float | None:
    """Coerce a raw CSV cell to float, or ``None`` for blank / non-numeric cells.

    SB3 leaves a metric's cell EMPTY on rollouts where it was not computed (e.g. ``train/*``
    keys are blank until the first optimizer step), so partial rows are normal — we drop
    those points rather than crash.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_column(key: str, fieldnames: Iterable[str]) -> str | None:
    """Return ``key`` if present, else the first available fallback, else ``None``."""
    present = set(fieldnames)
    if key in present:
        return key
    for alt in _FALLBACKS.get(key, ()):
        if alt in present:
            return alt
    return None


def extract_series(
    rows: list[dict[str, str]],
    *,
    x_key: str = X_KEY,
    panels: tuple[tuple[str, str], ...] = PANELS,
) -> dict[str, dict[str, object]]:
    """Extract per-panel (x, y) numeric series from raw progress.csv rows. PURE.

    For each requested panel this pairs the panel's metric column (or its fallback) with the
    x-axis column (``time/total_timesteps``), keeping only rows where BOTH are numeric. This
    tolerates:

    * missing columns        -> the panel is OMITTED from the result entirely,
    * empty / partial cells  -> that single point is skipped (common for ``train/*`` early on),
    * an empty CSV / no rows -> returns ``{}``.

    Args:
        rows: output of :func:`read_progress_rows` (list of str-valued dicts).
        x_key: the shared x-axis column name.
        panels: ``(csv_column, title)`` pairs to extract.

    Returns:
        Mapping ``csv_column -> {"title": str, "column": <resolved col>, "x": [float...],
        "y": [float...]}`` for every panel that resolved to a present column AND yielded at
        least one (x, y) point. Panels with no usable data are absent from the mapping.
    """
    if not rows:
        return {}

    fieldnames = rows[0].keys()
    if x_key not in fieldnames:
        # No x-axis column -> nothing plottable against timesteps.
        return {}

    series: dict[str, dict[str, object]] = {}
    for key, title in panels:
        column = _resolve_column(key, fieldnames)
        if column is None:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x = _to_float(row.get(x_key))
            y = _to_float(row.get(column))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        series[key] = {"title": title, "column": column, "x": xs, "y": ys}
    return series


def render_dashboard(
    series: dict[str, dict[str, object]],
    out_path: str | Path,
    *,
    run_name: str = "",
) -> Path:
    """Render extracted series into a grid PNG at ``out_path`` (matplotlib, Agg backend).

    Imports matplotlib lazily (Agg, no display) so the pure path stays matplotlib-free.
    Always writes a figure even when ``series`` is empty (an annotated "no metrics yet"
    placeholder) so the boardroom never points at a missing file.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; no display / no GUI backend.
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    suptitle = f"Training metrics — {run_name}" if run_name else "Training metrics"

    if not series:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No metrics yet\n(progress.csv has no plottable rows)",
            ha="center",
            va="center",
            fontsize=14,
        )
        fig.suptitle(suptitle)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out

    # Plot in PANELS order for stable layout, but only the panels that have data.
    ordered = [(key, title) for key, title in PANELS if key in series]
    n = len(ordered)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, (key, _title) in zip(flat, ordered, strict=False):
        s = series[key]
        ax.plot(s["x"], s["y"], marker=".", linewidth=1.4)
        ax.set_title(str(s["title"]))
        ax.set_xlabel(X_KEY)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes in the last row.
    for ax in flat[n:]:
        ax.set_axis_off()

    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_run(
    run_name: str,
    *,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    out: str | Path | None = None,
    boardroom: bool = False,
) -> Path:
    """Read ``runs/<run_name>/progress.csv``, render the dashboard, return the PNG path.

    Args:
        run_name: run subdirectory under ``runs_dir``.
        runs_dir: runs root (defaults to the repo ``runs/``).
        out: output PNG path (defaults to ``runs/<run_name>/metrics.png``).
        boardroom: if True, also copy the PNG to ``boardroom/metrics.png`` (stdlib shutil)
            so the boardroom server serves it at ``/metrics.png``.
    """
    run_dir = Path(runs_dir) / run_name
    csv_path = run_dir / "progress.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No progress.csv at {csv_path} — run training with metrics logging enabled first."
        )

    rows = read_progress_rows(csv_path)
    series = extract_series(rows)

    out_path = Path(out) if out is not None else run_dir / "metrics.png"
    render_dashboard(series, out_path, run_name=run_name)

    if boardroom:
        BOARDROOM_PNG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, BOARDROOM_PNG)

    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tank_twin.plot_metrics",
        description="Render a run's progress.csv into a PNG metrics dashboard.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run name -> reads runs/<run-name>/progress.csv.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Runs root directory (default: repo runs/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: runs/<run-name>/metrics.png).",
    )
    parser.add_argument(
        "--boardroom",
        action="store_true",
        help="Also copy the PNG to boardroom/metrics.png (served at /metrics.png).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m tank_twin.plot_metrics --run-name <name> [...]``."""
    args = _build_parser().parse_args(argv)
    out_path = plot_run(
        args.run_name,
        runs_dir=args.runs_dir,
        out=args.out,
        boardroom=args.boardroom,
    )
    print(f"Wrote metrics dashboard to {out_path}")
    if args.boardroom:
        print(f"Copied to {BOARDROOM_PNG} (boardroom server serves /metrics.png)")


if __name__ == "__main__":
    main()
