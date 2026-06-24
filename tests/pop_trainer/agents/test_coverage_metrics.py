"""Tests for the coverage-measurement harness (``pop_trainer.agents.coverage_metrics``).

Validates the PRODUCTION free-cell derivation (Walls bbox minus occupied), the kinematic rollout
mechanics, the metric definitions, and determinism given a seed. No socket / env / data import.
"""

import json
from pathlib import Path

import pytest

from pop_trainer.agents import RandomAgent
from pop_trainer.agents.coverage import CoverageAgent, free_cells_from_layout
from pop_trainer.agents.coverage_metrics import (
    CoverageReport,
    build_grid,
    measure_coverage,
)
from pop_trainer.core.protocol import WallDims, WallLayout

ARENA_DIR = Path(__file__).resolve().parents[3] / "exp-configs" / "maps" / "Arenas"


def _layout_from_arena(name: str) -> WallLayout:
    d = json.loads((ARENA_DIR / f"{name}.json").read_text())
    w = d["Walls"]
    dims = WallDims(
        min_x=w["dims"]["minX"],
        max_x=w["dims"]["maxX"],
        min_y=w["dims"]["minY"],
        max_y=w["dims"]["maxY"],
    )
    columns = {int(k): tuple(v) for k, v in w.items() if k not in ("dims", "tileID")}
    return WallLayout(map_id=name, tile_id=w["tileID"], dims=dims, columns=columns)


# --- free-cell derivation (the sim-vs-real seam) ---------------------------------------------


def test_free_cells_are_bbox_minus_occupied():
    layout = WallLayout(
        map_id="t",
        tile_id=0,
        dims=WallDims(min_x=0, max_x=2, min_y=0, max_y=2),
        columns={1: (1,)},  # the centre is a wall
    )
    free = free_cells_from_layout(layout)
    assert (1, 1) not in free
    assert len(free) == 9 - 1  # 3x3 bbox minus one wall
    assert (0, 0) in free and (2, 2) in free


def test_free_cells_matches_arena_floor_minus_walls():
    # The production derivation (Walls bbox - occupied) must equal the arena's Floor-minus-walls
    # free set on a real map — that the border ring is fully occupied is what makes them agree.
    name = "center_block"
    d = json.loads((ARENA_DIR / f"{name}.json").read_text())
    walls = {(int(k), y) for k, v in d["Walls"].items() if k not in ("dims", "tileID") for y in v}
    floor = {(int(k), y) for k, v in d["Floor"].items() if k not in ("dims", "tileID") for y in v}
    floor_free = floor - walls
    layout = _layout_from_arena(name)
    assert free_cells_from_layout(layout) == frozenset(floor_free)


def test_build_grid_restricts_to_reachable_component():
    layout = _layout_from_arena("center_block")
    grid = build_grid(layout)
    free = free_cells_from_layout(layout)
    # Every reachable cell is free, and the spawn is inside the reachable component.
    assert grid.reach <= free
    assert grid.spawn in grid.reach
    assert len(grid.reach) > 0


def test_build_grid_rejects_empty_map():
    # A fully-occupied bounding box has no free cells.
    layout = WallLayout(
        map_id="full",
        tile_id=0,
        dims=WallDims(min_x=0, max_x=1, min_y=0, max_y=1),
        columns={0: (0, 1), 1: (0, 1)},
    )
    with pytest.raises(ValueError):
        build_grid(layout)


# --- metric definitions ----------------------------------------------------------------------


def test_measure_coverage_report_shape_and_ranges():
    layout = _layout_from_arena("center_block")
    report = measure_coverage(CoverageAgent.aggressive(seed=0), layout)
    assert isinstance(report, CoverageReport)
    assert 0.0 <= report.coverage_fraction <= 1.0
    assert 0.0 <= report.occupancy_entropy <= 1.0
    assert 0.0 <= report.ess_ratio <= 1.0
    assert report.n_reachable > 0
    assert report.n_visited == round(report.coverage_fraction * report.n_reachable)


def test_coverage_fraction_higher_for_family_than_random():
    layout = _layout_from_arena("chokepoint")
    fam = measure_coverage(CoverageAgent.aggressive(seed=0), layout).coverage_fraction
    rnd = measure_coverage(RandomAgent(seed=0), layout).coverage_fraction
    assert fam > rnd


def test_occupancy_entropy_higher_for_family_than_random():
    layout = _layout_from_arena("scattered")
    fam = measure_coverage(CoverageAgent.aggressive(seed=0), layout).occupancy_entropy
    rnd = measure_coverage(RandomAgent(seed=0), layout).occupancy_entropy
    assert fam > rnd


def test_shorter_rollout_runs():
    # The decisions count is a knob; a short rollout still produces a valid report.
    layout = _layout_from_arena("empty")
    report = measure_coverage(CoverageAgent.aggressive(seed=0), layout, decisions=30)
    assert 0.0 <= report.coverage_fraction <= 1.0


# --- determinism -----------------------------------------------------------------------------


def test_measure_coverage_is_deterministic_given_seed():
    layout = _layout_from_arena("center_block")
    a = measure_coverage(CoverageAgent.aggressive(seed=5), layout)
    b = measure_coverage(CoverageAgent.aggressive(seed=5), layout)
    assert a == b


def test_measure_coverage_random_deterministic_given_seed():
    layout = _layout_from_arena("center_block")
    a = measure_coverage(RandomAgent(seed=11), layout)
    b = measure_coverage(RandomAgent(seed=11), layout)
    assert a.coverage_fraction == b.coverage_fraction
    assert a.occupancy_entropy == b.occupancy_entropy
