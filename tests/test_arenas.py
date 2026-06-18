"""Regression net for tank_twin.arenas (extracted from PythonScripts/tank_env.py, M1 1.2).

``load_level`` + the obs-grid geometry. Pins:

* the ``(36, 60, 3)`` observation shape at p=3 for the canonical ``default.json``,
* a GOLDEN wall-channel signature (the G-channel grid is the CnnPolicy's wall map),
* that the STRICT loader RAISES on a trailing-comma arena (guards the migration
  blocker: a tolerant parser would silently return a degraded grid).

numpy-only; no torch/sb3/gym/Unity.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tank_twin.arenas import load_level

R, G, B = (0, 1, 2)

# Repo-root-relative paths (this file lives in tests/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARENA = REPO_ROOT / "Assets" / "Arenas" / "default.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --- shape + dims geometry ------------------------------------------------


def test_default_arena_obs_shape_is_36x60x3_at_p3():
    # (maxY-minY+1)*p = (5-(-6)+1)*3 = 36 ; (maxX-minX+1)*p = (9-(-10)+1)*3 = 60.
    arena = load_level(DEFAULT_ARENA, p=3)
    assert arena.obs_shape == (36, 60, 3)
    assert arena.state.shape == (36, 60, 3)


def test_default_arena_dims_round_trip():
    arena = load_level(DEFAULT_ARENA, p=3)
    assert arena.dims == {"minX": -10, "maxX": 9, "minY": -6, "maxY": 5}


def test_geometry_scales_with_p():
    # p doubles -> grid doubles in each spatial dim, channels unchanged.
    a1 = load_level(DEFAULT_ARENA, p=3)
    a2 = load_level(DEFAULT_ARENA, p=6)
    assert a2.obs_shape == (a1.obs_shape[0] * 2, a1.obs_shape[1] * 2, 3)
    assert a2.p == 6


# --- golden wall channel --------------------------------------------------


def test_wall_grid_dtype_and_channels():
    arena = load_level(DEFAULT_ARENA, p=3)
    assert arena.state.dtype == np.uint8
    # Walls live ONLY in green; red/blue are untouched by load_level.
    assert arena.state[:, :, R].max() == 0
    assert arena.state[:, :, B].max() == 0
    assert set(np.unique(arena.state[:, :, G]).tolist()) == {0, 255}


def test_wall_grid_golden_signature():
    # GOLDEN: pin the exact wall map the CnnPolicy sees. The full-height left/right
    # border columns (x=-10,-9,8,9 -> 4 cols * 3 px = 12 px each, 36 px tall) plus
    # the top/bottom rails. Pin both the saturated-pixel count and the integer sum.
    arena = load_level(DEFAULT_ARENA, p=3)
    g = arena.state[:, :, G]
    assert int((g == 255).sum()) == 1008
    assert int(g.astype(np.int64).sum()) == 257040
    # Far-left wall column (x=-10) spans every row -> a full 255 column at x in 0..2.
    assert bool((g[:, 0] == 255).all())
    assert bool((g[:, 1] == 255).all())
    assert bool((g[:, 2] == 255).all())
    # Far-right wall column (x=9) -> a full 255 column at the last 3 px columns.
    assert bool((g[:, -1] == 255).all())


def test_wall_grid_matches_recomputed_reference():
    # Independent recomputation of the wall-baking loop must match load_level byte
    # for byte (catches an accidental off-by-one in the p-stride or the y-flip).
    with open(DEFAULT_ARENA) as f:
        level = json.load(f)
    dims = level["Walls"]["dims"]
    p = 3
    width = (dims["maxX"] - dims["minX"] + 1) * p
    height = (dims["maxY"] - dims["minY"] + 1) * p
    ref = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(dims["minX"], dims["maxX"] + 1):
        x_p = (x - dims["minX"]) * p
        for y in level["Walls"][str(x)]:
            y_p = (y - dims["minY"]) * p
            ref[y_p : y_p + p, x_p : x_p + p, G] = 255
    arena = load_level(DEFAULT_ARENA, p=3)
    assert np.array_equal(arena.state, ref)


# --- strict loader RAISES on malformed JSON (the migration blocker) --------


def test_strict_loader_raises_on_trailing_comma():
    # Newtonsoft (Unity side) tolerates a trailing comma; Python's strict json must
    # NOT — a tolerant fallback would silently return a degraded arena. Guard it.
    bad = FIXTURES / "arena_trailing_comma.json"
    with pytest.raises(json.JSONDecodeError):
        load_level(bad, p=3)
