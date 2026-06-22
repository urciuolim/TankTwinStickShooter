"""Hyperparameter SWEEP orchestrator for the pixel inverse-renderer pretrainer.

A THIN wrapper over the existing trainer CLI (``python -m tank_twin.pretrain_pixels``).
It does NOT reimplement any training logic: every config is one ``subprocess`` launch
of the trainer with a different ``--lr`` / ``--embedding-dim`` / ``--batch``, holding
the rest fixed (160x90 cache, ``player_aim`` disabled, ``--weight-decay 1e-4``,
``--lr-schedule cosine``, ``--seed 0``). After every run it (re)parses all known
``metrics.json`` files and rewrites a ``leaderboard.csv`` + ``summary.txt``.

THE GRID (12 configs): ``--lr {3e-4, 1e-3}`` x ``--embedding-dim {64, 32}`` x
``--batch {256, 512, 1024}`` (Cartesian product = 12).

Execution is SERIAL (one GPU; runs cannot share it), FAILURE-TOLERANT (a crash / OOM
is logged and the sweep continues), RESUMABLE (a config whose run dir already has a
completed ``metrics.json`` is skipped and folded into the leaderboard), INCREMENTAL
(the leaderboard is rewritten after every run, so an interrupted sweep still leaves a
valid partial leaderboard), and TIMEOUT-guarded (a per-run wall-clock cap kills the
child PROCESS TREE cross-platform and moves on).

The trainer auto-names its ``--out`` dir by appending
``_{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM}`` to the ``--out`` BASE in ``out_base.parent``
(see ``_pretrain_pixels_train._resolve_run_dir``), so the orchestrator cannot predict
the exact dir name; it gives each config a unique ``--out`` base and, after the run,
globs ``out_base.parent`` for ``out_base.name + "_*"`` matching this config's emb/batch
suffix and picks the newest dir holding a completed ``metrics.json``.

Pure, GPU-free helpers (``generate_grid`` / ``build_trainer_argv`` /
``parse_metrics_json`` / ``sort_leaderboard`` / ``write_csv`` / ``format_summary``) live
at module scope and are unit-tested in ``tests/test_sweep.py`` without torch or a GPU.

Run (FULL overnight sweep; the real cache, epochs 6)::

    uv run python scripts/sweep.py --epochs 6 \
        --cache-dir datasets/pixel_cache_160x90 \
        --out-root runs/pixel_stage2_sweep --per-run-timeout 7200

Run (SMOKE end-to-end, fast)::

    uv run python scripts/sweep.py --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Repo root: scripts/sweep.py -> parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable when run as a plain script (uv run also works).
sys.path.insert(0, str(_REPO_ROOT / "src"))

# --- The trainer module run as `python -m tank_twin.pretrain_pixels` ----------
_TRAINER_MODULE = "tank_twin.pretrain_pixels"

# --- Fixed sweep axes (THE GRID) ---------------------------------------------
_GRID_LR: tuple[float, ...] = (3e-4, 1e-3)
_GRID_EMBEDDING_DIM: tuple[int, ...] = (64, 32)
_GRID_BATCH: tuple[int, ...] = (256, 512, 1024)

# --- Fixed trainer args applied to EVERY run ---------------------------------
# input-res, cache-dir, epochs come from the orchestrator CLI; these are constant.
_FIXED_INPUT_RES = "160x90"
_FIXED_DISABLE = "player_aim"  # player_aim DISABLED; player_velocity is default-off.
_FIXED_WEIGHT_DECAY = "1e-4"
_FIXED_LR_SCHEDULE = "cosine"
_FIXED_SEED = "0"

# --- Wall-time estimate: throughput ASSUMPTION (labeled, to be refined) -------
# Coarse, DEFENSIBLE placeholder: on an RTX 3080 at 160x90 with AMP, the NatureCNN
# encoder + small heads processes on the order of ~3000 pair-steps/sec for the bigger
# batches. This is a deliberately conservative round number used ONLY for the up-front
# ETA; the orchestrator REFINES it from run 1's measured wall-time after the first run.
_ASSUMED_PAIRS_PER_SEC = 3000.0  # ASSUMPTION (3080, 160x90, AMP) -- refine from run 1.

# Effective TRAIN rows after the 2-map holdout (~8/10 of rows on the 8 train maps) and
# the 90/10 in-distribution eval split: 918000 * 0.8 * 0.9 ~= 660960. Coarse; the real
# per-map group split shifts this a little, but it sizes the estimate.
_FULL_CACHE_ROWS = 918_000
_HOLDOUT_FRAC = 0.8  # fraction of rows on the 8 training maps (2 of 10 maps held out)
_EVAL_KEEP_FRAC = 0.9  # 90/10 train/eval split within the training maps


# =============================================================================
# Pure helpers (GPU-free; unit-tested in tests/test_sweep.py)
# =============================================================================
def generate_grid() -> list[dict]:
    """The 12-config grid: ``lr`` x ``embedding_dim`` x ``batch`` Cartesian product.

    Deterministic order: lr outermost, then embedding_dim, then batch innermost (so the
    list is stable across runs). Each dict is ``{"lr", "embedding_dim", "batch"}``.
    Returns exactly ``len(_GRID_LR) * len(_GRID_EMBEDDING_DIM) * len(_GRID_BATCH)`` == 12
    unique configs.
    """
    grid: list[dict] = []
    for lr in _GRID_LR:
        for emb in _GRID_EMBEDDING_DIM:
            for batch in _GRID_BATCH:
                grid.append({"lr": lr, "embedding_dim": emb, "batch": batch})
    return grid


def config_label(config: dict) -> str:
    """Short, filesystem-safe label for a config, e.g. ``lr3e-04_e64_b256``.

    Used for the per-config ``--out`` BASE name and the per-config ``.log`` filename so
    each config maps to a stable, unique on-disk identity (the trainer then appends its
    own ``_{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM}`` suffix to the run dir).
    """
    lr = config["lr"]
    emb = config["embedding_dim"]
    batch = config["batch"]
    # Compact lr token (e.g. 0.0003 -> "3e-04", 0.001 -> "1e-03") so it is FS-safe.
    lr_token = f"{lr:.0e}".replace("+", "")
    return f"lr{lr_token}_e{emb}_b{batch}"


def build_trainer_argv(config: dict, fixed: dict) -> list[str]:
    """Build the trainer launch arg-list for one config (NO shell string).

    ``config`` carries the swept ``lr`` / ``embedding_dim`` / ``batch``; ``fixed`` carries
    the run-constant fields (``out_base``, ``cache_dir``, ``input_res``, ``epochs``,
    ``device``, optional ``subset``). The returned list starts with
    ``[sys.executable, "-m", "tank_twin.pretrain_pixels", ...]`` so the child inherits the
    SAME interpreter / venv as the orchestrator (no hardcoded python path). The fixed
    objective/regularization flags (``--disable player_aim``, ``--weight-decay 1e-4``,
    ``--lr-schedule cosine``, ``--seed 0``) are always present; the 2-map holdout and
    ``auto`` pos-weights are left at their trainer defaults (NOT passed).
    """
    argv: list[str] = [
        sys.executable,
        "-m",
        _TRAINER_MODULE,
        "--cache-dir",
        str(fixed["cache_dir"]),
        "--input-res",
        str(fixed.get("input_res", _FIXED_INPUT_RES)),
        "--disable",
        _FIXED_DISABLE,
        "--weight-decay",
        _FIXED_WEIGHT_DECAY,
        "--lr-schedule",
        _FIXED_LR_SCHEDULE,
        "--seed",
        _FIXED_SEED,
        "--epochs",
        str(fixed["epochs"]),
        "--lr",
        repr(float(config["lr"])),
        "--embedding-dim",
        str(config["embedding_dim"]),
        "--batch",
        str(config["batch"]),
        "--out",
        str(fixed["out_base"]),
    ]
    if fixed.get("device"):
        argv += ["--device", str(fixed["device"])]
    if fixed.get("subset") is not None:
        argv += ["--subset", str(fixed["subset"])]
    return argv


def _objective_entry(per_objective: dict | None, group: str) -> dict:
    """Return the per-objective sub-dict for ``group``, or ``{}`` if missing / "N/A"."""
    if not isinstance(per_objective, dict):
        return {}
    entry = per_objective.get(group)
    if not isinstance(entry, dict):  # "N/A" string, None, or absent -> no metrics
        return {}
    return entry


def _flatten_eval(prefix: str, per_objective: dict | None) -> dict:
    """Flatten the per-objective metrics we surface in the leaderboard under ``prefix``.

    ``prefix`` is ``"eval"`` (final-epoch, in-distribution) or ``"val"`` (OOD validation).
    Pulls the canonical fields for every objective the leaderboard reports; a disabled /
    missing group leaves its columns blank (``None``). Handles ``"N/A"`` gracefully.
    """
    out: dict = {}
    bp = _objective_entry(per_objective, "bullet_position")
    out[f"{prefix}_bullet_position_norm_mse"] = bp.get("norm_mse")
    out[f"{prefix}_bullet_position_world_mae"] = bp.get("world_mae")

    bpr = _objective_entry(per_objective, "bullet_presence")
    out[f"{prefix}_bullet_presence_f1"] = bpr.get("f1")
    out[f"{prefix}_bullet_presence_precision"] = bpr.get("precision")
    out[f"{prefix}_bullet_presence_recall"] = bpr.get("recall")

    w = _objective_entry(per_objective, "walls")
    out[f"{prefix}_walls_interior_f1"] = w.get("interior_f1")
    out[f"{prefix}_walls_interior_iou"] = w.get("interior_iou")

    bd = _objective_entry(per_objective, "bullet_direction")
    out[f"{prefix}_bullet_direction_norm_mse"] = bd.get("norm_mse")

    pp = _objective_entry(per_objective, "player_position")
    out[f"{prefix}_player_position_norm_mse"] = pp.get("norm_mse")
    out[f"{prefix}_player_position_world_rmse"] = pp.get("world_rmse")
    return out


# Leaderboard column order: sort keys + identity first, then eval/val per-objective.
# Kept as a module constant so write_csv and the tests agree on the header.
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    # --- default-sort keys (weak objectives, OOD validation) ---
    "val_bullet_position_world_mae",
    "val_bullet_presence_f1",
    "val_walls_interior_f1",
    # --- config identity ---
    "lr",
    "embedding_dim",
    "batch",
    "status",
    "wall_time_s",
    # --- in-distribution EVAL (final epoch) ---
    "eval_bullet_position_norm_mse",
    "eval_bullet_position_world_mae",
    "eval_bullet_presence_f1",
    "eval_bullet_presence_precision",
    "eval_bullet_presence_recall",
    "eval_walls_interior_f1",
    "eval_walls_interior_iou",
    "eval_bullet_direction_norm_mse",
    "eval_player_position_norm_mse",
    "eval_player_position_world_rmse",
    # --- OOD VALIDATION (full set; the first 3 are also the sort keys above) ---
    "val_bullet_position_norm_mse",
    "val_bullet_presence_precision",
    "val_bullet_presence_recall",
    "val_walls_interior_iou",
    "val_bullet_direction_norm_mse",
    "val_player_position_norm_mse",
    "val_player_position_world_rmse",
    # --- provenance ---
    "run_dir",
)


def parse_metrics_json(path_or_dict: Path | str | dict) -> dict:
    """Flatten a trainer ``metrics.json`` into one leaderboard ROW dict.

    Accepts a path (str/Path) to the file or an already-loaded dict. Pulls the swept
    config (``lr`` / ``embedding_dim`` / ``batch``) from ``metrics["config"]``, the
    FINAL-EPOCH eval entry (``metrics["eval"][-1]["per_objective"]``), and the OOD
    ``metrics["validation"]["per_objective"]``. Missing / partial files / ``"N/A"``
    groups leave their metric columns ``None`` instead of raising, so the parser never
    crashes the leaderboard on a half-written run. The returned dict has ``status`` set
    to ``"done"`` here (callers may override for skipped/failed/timeout rows).
    """
    if isinstance(path_or_dict, dict):
        metrics = path_or_dict
    else:
        metrics = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))

    config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    row: dict = {
        "lr": config.get("lr"),
        "embedding_dim": config.get("embedding_dim"),
        "batch": config.get("batch"),
        "status": "done",
        "wall_time_s": None,
        "run_dir": None,
    }

    eval_hist = metrics.get("eval") if isinstance(metrics, dict) else None
    final_eval = eval_hist[-1] if isinstance(eval_hist, list) and eval_hist else {}
    row.update(_flatten_eval("eval", final_eval.get("per_objective")))

    validation = metrics.get("validation") if isinstance(metrics, dict) else None
    val_per_obj = validation.get("per_objective") if isinstance(validation, dict) else None
    row.update(_flatten_eval("val", val_per_obj))
    return row


def _is_complete_metrics(metrics: dict) -> bool:
    """True iff a parsed metrics dict is a COMPLETED run (has a validation block)."""
    if not isinstance(metrics, dict):
        return False
    validation = metrics.get("validation")
    return isinstance(validation, dict) and "per_objective" in validation


def _sort_value(row: dict, key: str, *, descending: bool) -> tuple[int, float]:
    """Sort tuple for one key: present rows first, then the (sign-adjusted) value.

    Missing values sort LAST regardless of direction. For an ascending key the raw value
    is used; for a descending key the negated value is used so a single ``sorted`` over
    the tuple gives the right order.
    """
    v = row.get(key)
    if v is None:
        return (1, 0.0)  # group 1 = missing, always after present (group 0)
    fv = float(v)
    return (0, -fv if descending else fv)


def sort_leaderboard(rows: list[dict]) -> list[dict]:
    """Sort rows by the WEAK-objective OOD key, failed/missing rows last.

    Primary: ``val_bullet_position_world_mae`` ASC (lower error better). Tie-break:
    ``val_bullet_presence_f1`` DESC, then ``val_walls_interior_f1`` DESC (position / walls
    are near-maxed; bullet-position error + the two F1s are what is being pushed). A row
    with no validation metrics (failed / timed-out / missing) sorts after every row that
    has them; a final ``config_label`` tie-break keeps the order stable / deterministic.
    """

    def key(row: dict) -> tuple:
        return (
            _sort_value(row, "val_bullet_position_world_mae", descending=False),
            _sort_value(row, "val_bullet_presence_f1", descending=True),
            _sort_value(row, "val_walls_interior_f1", descending=True),
            config_label(
                {
                    "lr": row.get("lr") or 0.0,
                    "embedding_dim": row.get("embedding_dim") or 0,
                    "batch": row.get("batch") or 0,
                }
            ),
        )

    return sorted(rows, key=key)


def _fmt(v: object) -> str:
    """Format a leaderboard cell for the CSV / summary (blank for None)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the leaderboard rows to ``path`` as CSV with the canonical header.

    Columns follow :data:`LEADERBOARD_COLUMNS` (sort keys + identity first). Missing
    fields are written blank. Rows are written in the order given (callers pass the
    already-sorted list); the header round-trips through :func:`csv.DictReader`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(LEADERBOARD_COLUMNS)
        for row in rows:
            writer.writerow([_fmt(row.get(col)) for col in LEADERBOARD_COLUMNS])


def format_summary(rows: list[dict]) -> str:
    """Render a human-readable leaderboard table (sorted) as a string for ``summary.txt``.

    Shows the sort keys + identity + the headline weak-objective metrics per config, in
    sorted order. Blank cells for missing metrics. Intended for an at-a-glance ranking;
    the full per-objective detail lives in ``leaderboard.csv``.
    """
    cols = [
        ("lr", "lr", 8),
        ("embedding_dim", "emb", 4),
        ("batch", "batch", 6),
        ("status", "status", 8),
        ("wall_time_s", "wall_s", 8),
        ("val_bullet_position_world_mae", "val_bpos_wmae", 13),
        ("val_bullet_presence_f1", "val_bpres_f1", 12),
        ("val_walls_interior_f1", "val_wall_f1", 11),
        ("eval_bullet_position_world_mae", "ev_bpos_wmae", 12),
        ("eval_bullet_presence_f1", "ev_bpres_f1", 11),
    ]
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("PIXEL STAGE-2 SWEEP LEADERBOARD")
    lines.append(
        "sort: val bullet_position world_mae ASC, then val bullet_presence f1 DESC, "
        "then val walls interior_f1 DESC"
    )
    lines.append("=" * 110)
    header = " | ".join(f"{label:>{w}}" for _key, label, w in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        cells = []
        for key, _label, w in cols:
            val = row.get(key)
            if key == "wall_time_s" and val is not None:
                cells.append(f"{float(val):>{w}.0f}")
            else:
                cells.append(f"{_fmt(val):>{w}}")
        lines.append(" | ".join(cells))
    lines.append("=" * 110)
    return "\n".join(lines) + "\n"


# =============================================================================
# Wall-time estimate (pure; uses the labeled throughput ASSUMPTION)
# =============================================================================
def estimate_config_seconds(
    epochs: int,
    batch: int,
    *,
    train_rows: float,
    pairs_per_sec: float = _ASSUMED_PAIRS_PER_SEC,
) -> float:
    """Coarse per-config wall-time estimate (SECONDS) from a throughput assumption.

    The dominant cost is the forward/backward over ``train_rows * epochs`` pair-steps at
    ``pairs_per_sec`` (a 3080/160x90/AMP ASSUMPTION; see :data:`_ASSUMED_PAIRS_PER_SEC`).
    ``batch`` is accepted (and folds in a small per-epoch eval/validation overhead) so the
    signature can be refined later, but throughput is treated as roughly batch-independent
    on the GPU at these sizes. Returns the assumed wall-clock seconds for ONE config.
    """
    pair_steps = float(train_rows) * float(epochs)
    train_secs = pair_steps / max(1.0, pairs_per_sec)
    # ~12% overhead for the per-epoch eval pass + the one OOD validation pass.
    return train_secs * 1.12


def _full_train_rows(cache_rows: int, subset: int | None) -> float:
    """Effective TRAIN-split row count after the 2-map holdout + 90/10 eval split."""
    n = cache_rows if subset is None else min(cache_rows, subset)
    return n * _HOLDOUT_FRAC * _EVAL_KEEP_FRAC


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as ``HhMMmSSs`` (e.g. ``2h05m30s``)."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


# =============================================================================
# Run-dir resolution: locate the trainer's auto-named dir for a config
# =============================================================================
def find_run_dir(out_base: Path, config: dict, *, require_complete: bool) -> Path | None:
    """Locate the trainer-produced run dir for ``config`` under ``out_base.parent``.

    The trainer appends ``_{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM}`` to ``out_base.name``
    (see ``_resolve_run_dir``), so we glob ``out_base.name + "_*"`` and keep dirs whose
    suffix contains this config's ``_e{emb}_b{batch}_`` token (disambiguates configs that
    share a base parent). If ``require_complete`` is True, only dirs holding a parse-able,
    COMPLETED ``metrics.json`` (has a ``validation`` block) qualify. Returns the NEWEST
    qualifying dir (by mtime) or ``None`` if there is none.
    """
    parent = out_base.parent
    if not parent.is_dir():
        return None
    emb = config["embedding_dim"]
    batch = config["batch"]
    token = f"_e{emb}_b{batch}_"
    candidates: list[Path] = []
    for d in parent.glob(out_base.name + "_*"):
        if not d.is_dir() or token not in d.name:
            continue
        metrics_path = d / "metrics.json"
        if require_complete:
            if not metrics_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not _is_complete_metrics(metrics):
                continue
        candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# =============================================================================
# Cross-platform process-tree kill (for the per-run timeout)
# =============================================================================
def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` AND any child processes it spawned, cross-platform.

    The trainer can spawn DataLoader workers / CUDA helper processes, so killing only the
    direct child can orphan them. On Windows we use ``taskkill /T /F`` against the PID to
    take down the whole tree (the child is launched with ``CREATE_NEW_PROCESS_GROUP``); on
    POSIX the child is launched with ``start_new_session=True`` and we ``killpg`` the
    process group. Both paths fall back to ``proc.kill()`` and always reap with ``wait``.
    """
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                check=False,
                capture_output=True,
            )
        else:
            import os
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=30)


def _popen_kwargs() -> dict:
    """Popen kwargs that put the child in its OWN process group / session.

    Windows: ``CREATE_NEW_PROCESS_GROUP`` so ``taskkill /T`` reliably reaches the subtree.
    POSIX: ``start_new_session=True`` so ``os.killpg`` can signal the whole group. No
    ``fork``/``forkserver`` is involved (we exec a fresh interpreter via subprocess).
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# =============================================================================
# Per-config run + the orchestration loop
# =============================================================================
def _run_one(
    config: dict,
    fixed: dict,
    log_path: Path,
    *,
    per_run_timeout: float | None,
) -> tuple[str, float]:
    """Launch the trainer for one config, capturing stdout+stderr to ``log_path``.

    Returns ``(status, wall_time_s)`` where status is ``"done"`` (exit 0),
    ``"failed"`` (nonzero exit), or ``"timeout"`` (killed at ``per_run_timeout``). The
    child is timed with ``perf_counter`` (the trainer's metrics.json does not store
    wall-time). On timeout the whole process tree is killed via :func:`_kill_tree`.
    """
    argv = build_trainer_argv(config, fixed)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("$ " + " ".join(argv) + "\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(_REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            **_popen_kwargs(),
        )
        try:
            returncode = proc.wait(timeout=per_run_timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            elapsed = time.perf_counter() - t0
            log_fh.write(f"\n[sweep] TIMEOUT after {elapsed:.0f}s -> killed process tree\n")
            return "timeout", elapsed
    elapsed = time.perf_counter() - t0
    return ("done" if returncode == 0 else "failed"), elapsed


def _missing_row(config: dict, status: str, wall_time_s: float | None) -> dict:
    """A leaderboard row for a config with NO metrics (failed / timeout / missing)."""
    return {
        "lr": config["lr"],
        "embedding_dim": config["embedding_dim"],
        "batch": config["batch"],
        "status": status,
        "wall_time_s": wall_time_s,
        "run_dir": None,
    }


def _row_for_config(
    config: dict,
    fixed: dict,
    status: str,
    wall_time_s: float | None,
) -> dict:
    """Build the leaderboard row for a config: parse its metrics.json if a COMPLETE run
    dir exists, else emit a metric-less row carrying ``status``."""
    run_dir = find_run_dir(fixed["out_base"], config, require_complete=True)
    if run_dir is None:
        return _missing_row(config, status if status != "done" else "missing", wall_time_s)
    try:
        row = parse_metrics_json(run_dir / "metrics.json")
    except (json.JSONDecodeError, OSError, KeyError):
        return _missing_row(config, "missing", wall_time_s)
    # Trust the swept axes from the orchestrator's config (not the file) for identity.
    row["lr"] = config["lr"]
    row["embedding_dim"] = config["embedding_dim"]
    row["batch"] = config["batch"]
    row["status"] = status if status in ("done", "skipped") else "done"
    row["wall_time_s"] = wall_time_s
    row["run_dir"] = str(run_dir)
    return row


def _rewrite_leaderboard(rows_by_label: dict[str, dict], sweep_dir: Path) -> None:
    """Sort the accumulated rows and (re)write leaderboard.csv + summary.txt."""
    rows = sort_leaderboard(list(rows_by_label.values()))
    write_csv(rows, sweep_dir / "leaderboard.csv")
    (sweep_dir / "summary.txt").write_text(format_summary(rows), encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> int:
    """Drive the full (or smoke) sweep: serial runs, incremental leaderboard, resume."""
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (_REPO_ROOT / cache_dir).resolve()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (_REPO_ROOT / out_root).resolve()

    grid = generate_grid()
    if args.smoke:
        # 2 configs, 1 epoch, tiny subset: just prove the pipeline end-to-end.
        grid = [grid[0], grid[-1]]

    sweep_dir = out_root
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.is_dir():
        print(f"[sweep] WARNING: cache dir does not exist: {cache_dir}", flush=True)

    # --- Up-front wall-time ESTIMATE (labeled assumption) --------------------
    train_rows = _full_train_rows(_FULL_CACHE_ROWS, args.subset)
    per_cfg_est = estimate_config_seconds(args.epochs, grid[0]["batch"], train_rows=train_rows)
    total_est = per_cfg_est * len(grid)
    print("=" * 80, flush=True)
    print(f"[sweep] {len(grid)} configs | epochs={args.epochs} | cache={cache_dir}", flush=True)
    print(
        f"[sweep] ESTIMATE (ASSUMPTION ~{_ASSUMED_PAIRS_PER_SEC:.0f} pairs/sec on a 3080, "
        f"160x90, AMP -- to be refined from run 1):",
        flush=True,
    )
    print(
        f"[sweep]   train_rows~={train_rows:,.0f} (918k * {_HOLDOUT_FRAC} holdout * "
        f"{_EVAL_KEEP_FRAC} eval-split)",
        flush=True,
    )
    print(
        f"[sweep]   per-config ~{per_cfg_est:.0f}s ({_fmt_hms(per_cfg_est)}); "
        f"total {len(grid)} configs ~{total_est:.0f}s ({_fmt_hms(total_est)})",
        flush=True,
    )
    print("=" * 80, flush=True)

    rows_by_label: dict[str, dict] = {}
    measured_per_cfg: float | None = None

    for i, config in enumerate(grid, start=1):
        label = config_label(config)
        out_base = sweep_dir / label
        fixed = {
            "out_base": out_base,
            "cache_dir": cache_dir,
            "input_res": _FIXED_INPUT_RES,
            "epochs": args.epochs,
            "device": args.device,
            "subset": args.subset,
        }

        # --- RESUME: skip a config whose run dir already has a COMPLETE metrics.json ---
        existing = find_run_dir(out_base, config, require_complete=True)
        if existing is not None:
            print(
                f"[sweep] ({i}/{len(grid)}) resume: skipping {label} "
                f"(found completed {existing.name})",
                flush=True,
            )
            rows_by_label[label] = _row_for_config(config, fixed, "skipped", None)
            _rewrite_leaderboard(rows_by_label, sweep_dir)
            continue

        # --- Optional forced-failure test hook (smoke evidence point 5) ----------
        run_fixed = dict(fixed)
        if args.smoke_fail_first and i == 1:
            run_fixed["cache_dir"] = cache_dir.parent / "__nonexistent_cache__"
            print(
                f"[sweep] ({i}/{len(grid)}) {label}: FORCED-FAIL hook active "
                "(cache-dir points at a nonexistent dir)",
                flush=True,
            )

        argv = build_trainer_argv(config, run_fixed)
        log_path = log_dir / f"{label}.log"
        print(
            f"[sweep] ({i}/{len(grid)}) launching {label}: "
            f"lr={config['lr']} emb={config['embedding_dim']} batch={config['batch']}",
            flush=True,
        )
        print(f"[sweep]   $ {' '.join(argv)}", flush=True)
        print(f"[sweep]   log -> {log_path}", flush=True)

        status, wall_s = _run_one(config, run_fixed, log_path, per_run_timeout=args.per_run_timeout)

        if status == "done":
            print(f"[sweep]   -> done in {wall_s:.0f}s ({_fmt_hms(wall_s)})", flush=True)
        elif status == "timeout":
            print(
                f"[sweep]   -> TIMEOUT after {wall_s:.0f}s (killed); see {log_path}",
                flush=True,
            )
        else:
            print(
                f"[sweep]   -> FAILED (nonzero exit) after {wall_s:.0f}s; see {log_path}",
                flush=True,
            )

        rows_by_label[label] = _row_for_config(config, fixed, status, wall_s)
        _rewrite_leaderboard(rows_by_label, sweep_dir)

        # --- REFINE the ETA from run 1's measured wall-time ----------------------
        if measured_per_cfg is None and status == "done":
            measured_per_cfg = wall_s
            remaining = len(grid) - i
            refined_total = measured_per_cfg * remaining
            print(
                f"[sweep] REFINED ETA from run 1 (measured {wall_s:.0f}s/config): "
                f"{remaining} configs remaining ~{refined_total:.0f}s ({_fmt_hms(refined_total)})",
                flush=True,
            )

    rows = sort_leaderboard(list(rows_by_label.values()))
    print("\n" + format_summary(rows), flush=True)
    print(f"[sweep] leaderboard -> {sweep_dir / 'leaderboard.csv'}", flush=True)
    print(f"[sweep] summary     -> {sweep_dir / 'summary.txt'}", flush=True)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/sweep.py",
        description=(
            "Serial hyperparameter sweep (12 configs: lr x embedding_dim x batch) over the "
            "pixel inverse-renderer pretrainer. Failure-tolerant, resumable, incremental "
            "leaderboard. Thin wrapper -- launches `python -m tank_twin.pretrain_pixels`."
        ),
    )
    p.add_argument(
        "--cache-dir",
        default="datasets/pixel_cache_160x90",
        help="memmap cache dir passed to the trainer's --cache-dir (the 160x90 cache).",
    )
    p.add_argument(
        "--out-root",
        default="runs/pixel_stage2_sweep",
        help="sweep output dir (leaderboard + per-config logs + run-dir bases live here).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="epochs per config (the ranking horizon). Default 6.",
    )
    p.add_argument(
        "--per-run-timeout",
        type=float,
        default=None,
        help="per-config wall-clock cap (SECONDS); on timeout the child tree is killed and "
        "the sweep continues. Omit for no cap.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=("auto", "cuda", "cpu"),
        help="trainer --device (the box has a 3080). Default cuda.",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="trainer --subset N (limits rows; for fast smokes). Omit for the full cache.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="SMOKE mode: 2 configs, 1 epoch. Combine with --subset and --epochs 1.",
    )
    p.add_argument(
        "--smoke-fail-first",
        action="store_true",
        help="TEST HOOK: point the FIRST config's --cache-dir at a nonexistent dir so it "
        "fails, proving the sweep logs the failure and continues.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
