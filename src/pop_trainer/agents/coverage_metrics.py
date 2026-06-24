"""Coverage-measurement harness — validate a movement policy against a map, deterministically.

The productionized spike harness: build the free/reachable grid from a
:class:`pop_trainer.core.protocol.WallLayout` (the PRODUCTION derivation — Walls bbox minus
occupied, NOT the arena ``Floor`` block), run a kinematic tank rollout driving an injected
agent's MOVEMENT decision, and report the coverage metrics. This is the tool that reproduces the
research sim result in-tree and validates the coverage family against ``RandomAgent``.

The harness drives the agent exactly as the env will: each decision it calls ``agent.act(state)``
with the current 52-float state and uses ``action[0:2]`` (``move_x`` / ``move_y``) as the
heading; the kinematic tank integrates that heading over the confirmed mechanics. The map is
handed to the agent via ``set_map`` (the same hook the env uses), so a map-aware agent is
measured map-aware. Deterministic given a seed.

Metrics:

* **coverage-fraction** = visited reachable cells / reachable cells. The DESIGN objective.
* **normalized occupancy-entropy** = entropy of the visit distribution over reachable cells,
  divided by ``log(n_reachable)``. The DESIGN objective (a flat visit distribution -> 1.0).
* **ESS ratio** = a GUARDRAIL ONLY. It measures trajectory decorrelation (jitter), NOT coverage,
  and is exposed solely to detect a degenerate still / stuck agent. NEVER present it as a
  coverage reward — it rewards jitter.

Confirmed kinematic mechanics (from the spike, matching the Unity sim): tank speed 3 u/s,
velocity ramps toward ``unit(move) * speed`` via per-component ``MoveTowards`` at accel 6/s,
``dt`` 0.02, 10 physics steps per decision (~0.2 s), axis-separable slide collision (a component
that would enter a non-free cell is rejected and that axis velocity zeroed), continuous position
clamped to the free bounding box. ~300 decisions per rollout.

Boundaries: imports :mod:`pop_trainer.core` (``WallLayout``, the state schema) +
:mod:`pop_trainer.agents.coverage` (grid helpers) + numpy + stdlib only. No torch; nothing from
``env`` / ``data`` / ``pretraining`` / ``rl`` / ``models``; nothing from ``tank_twin``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pop_trainer.agents.coverage import (
    Cell,
    cell_of,
    free_cells_from_layout,
    reachable_from,
)
from pop_trainer.core import state as state_schema

if TYPE_CHECKING:
    from pop_trainer.core.protocol import WallLayout

__all__ = [
    "CoverageReport",
    "build_grid",
    "measure_coverage",
]

# --- confirmed kinematic mechanics (match the Unity sim) --------------------------------
SPEED = 3.0
ACCEL = 6.0
DT = 0.02
PHYS_STEPS_PER_DECISION = 10
DEFAULT_DECISIONS = 300


@dataclass(frozen=True)
class CoverageReport:
    """The coverage metrics for one rollout on one map.

    * ``coverage_fraction`` — visited reachable cells / reachable cells (DESIGN objective).
    * ``occupancy_entropy`` — normalized visit-distribution entropy over reachable cells
      (DESIGN objective; flat distribution -> ~1.0).
    * ``ess_ratio`` — GUARDRAIL ONLY: trajectory decorrelation, never a coverage reward.
    * ``n_reachable`` — reachable free-cell count (the coverage denominator).
    * ``n_visited`` — distinct reachable cells the tank visited.
    """

    coverage_fraction: float
    occupancy_entropy: float
    ess_ratio: float
    n_reachable: int
    n_visited: int


@dataclass(frozen=True)
class _Grid:
    """The derived free/reachable grid + spawn for a rollout (internal)."""

    free: frozenset[Cell]
    reach: set[Cell]
    spawn: Cell


def build_grid(layout: WallLayout, *, spawn: Cell | None = None) -> _Grid:
    """Build the free + reachable grid from ``layout`` (PRODUCTION derivation).

    ``free = dims bbox - occupied``; ``reach`` is the BFS-reachable component from ``spawn``
    (snapped to the nearest free cell). ``spawn`` defaults to the left-edge cell ``(min_x+2,
    -1)`` if free, else the centroid of free cells — a deterministic seed independent of any
    agent. Raises ``ValueError`` if the map has no free cells.
    """
    free = free_cells_from_layout(layout)
    if not free:
        raise ValueError("layout has no free cells")
    if spawn is None:
        spawn = _default_spawn(free, layout)
    reach = reachable_from(free, spawn)
    snapped = spawn if spawn in reach else min(reach, key=lambda c: _d2(c, spawn))
    return _Grid(free=free, reach=reach, spawn=snapped)


def _default_spawn(free: frozenset[Cell], layout: WallLayout) -> Cell:
    """A deterministic spawn cell: prefer a left-edge interior cell, else the free centroid."""
    candidate = (layout.dims.min_x + 2, -1)
    if candidate in free:
        return candidate
    cx = round(sum(c[0] for c in free) / len(free))
    cy = round(sum(c[1] for c in free) / len(free))
    centroid = (cx, cy)
    if centroid in free:
        return centroid
    return min(free, key=lambda c: _d2(c, centroid))


def _d2(a: Cell, b: Cell) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# --- kinematic tank (axis-separable slide collision) ------------------------------------


class _Tank:
    """Continuous-position tank with inertia + axis-separable wall-slide collision."""

    def __init__(self, spawn: Cell, free: frozenset[Cell]):
        self.x = float(spawn[0])
        self.y = float(spawn[1])
        self.vx = 0.0
        self.vy = 0.0
        self._free = free
        xs = [c[0] for c in free]
        ys = [c[1] for c in free]
        self._lo_x, self._hi_x = float(min(xs)), float(max(xs))
        self._lo_y, self._hi_y = float(min(ys)), float(max(ys))

    def step(self, move_x: float, move_y: float) -> None:
        dvx = move_x * SPEED
        dvy = move_y * SPEED
        self.vx = _move_towards(self.vx, dvx, ACCEL * DT)
        self.vy = _move_towards(self.vy, dvy, ACCEL * DT)
        nx = self.x + self.vx * DT
        if not self._blocked(nx, self.y):
            self.x = nx
        else:
            self.vx = 0.0
        ny = self.y + self.vy * DT
        if not self._blocked(self.x, ny):
            self.y = ny
        else:
            self.vy = 0.0
        self.x = min(max(self.x, self._lo_x), self._hi_x)
        self.y = min(max(self.y, self._lo_y), self._hi_y)

    def _blocked(self, wx: float, wy: float) -> bool:
        return cell_of(wx, wy) not in self._free


def _move_towards(cur: float, tgt: float, max_delta: float) -> float:
    d = tgt - cur
    if abs(d) <= max_delta:
        return tgt
    return cur + math.copysign(max_delta, d)


def measure_coverage(
    agent,
    layout: WallLayout,
    *,
    decisions: int = DEFAULT_DECISIONS,
    spawn: Cell | None = None,
) -> CoverageReport:
    """Run a kinematic rollout of ``agent`` on ``layout`` and report the coverage metrics.

    Hands the agent the map via ``set_map`` (when it exposes it) and resets it via ``reset`` (so
    a seeded agent is deterministic), then drives ``decisions`` decisions: each decision calls
    ``agent.act(state)`` with the current 52-float state (the tank's own position in the PLAYER_1
    slot, the rest zero), takes ``action[0:2]`` as the movement heading, and integrates the tank
    over :data:`PHYS_STEPS_PER_DECISION` physics steps. Deterministic given the agent's seed and
    the (deterministic) spawn.
    """
    grid = build_grid(layout, spawn=spawn)
    _hand_map(agent, layout)

    tank = _Tank(grid.spawn, grid.free)
    visited: set[Cell] = set()
    occ: dict[Cell, int] = {}
    traj = np.empty((decisions, 2), dtype=np.float64)

    for i in range(decisions):
        action = agent.act(_state_at(tank.x, tank.y))
        move_x, move_y = float(action[0]), float(action[1])
        for _ in range(PHYS_STEPS_PER_DECISION):
            tank.step(move_x, move_y)
        c = cell_of(tank.x, tank.y)
        if c in grid.reach:
            visited.add(c)
            occ[c] = occ.get(c, 0) + 1
        traj[i, 0] = tank.x
        traj[i, 1] = tank.y

    n_reach = len(grid.reach)
    cov = len(visited) / n_reach if n_reach else 0.0
    entropy = _normalized_occupancy_entropy(occ, n_reach)
    ess = _ess_ratio(traj)
    return CoverageReport(
        coverage_fraction=cov,
        occupancy_entropy=entropy,
        ess_ratio=ess,
        n_reachable=n_reach,
        n_visited=len(visited),
    )


def _hand_map(agent, layout: WallLayout) -> None:
    """Reset the agent (if seeded) then hand it the map via ``set_map`` (if map-aware)."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()
    set_map = getattr(agent, "set_map", None)
    if callable(set_map):
        set_map(layout)


def _state_at(x: float, y: float) -> list[float]:
    """A 52-float state with the tank at ``(x, y)`` in the PLAYER_1 slot; everything else zero."""
    state = [0.0] * state_schema.STATE_LEN
    base = state_schema.PLAYER_1 * state_schema.PLAYER_STRIDE
    state[base + state_schema.POS_X] = x
    state[base + state_schema.POS_Y] = y
    return state


def _normalized_occupancy_entropy(occ: dict[Cell, int], n_reach: int) -> float:
    """Normalized Shannon entropy of the visit distribution over reachable cells ([0, 1])."""
    total = sum(occ.values())
    if total <= 0 or n_reach <= 1:
        return 0.0
    ps = np.array([v / total for v in occ.values()], dtype=np.float64)
    entropy = float(-np.sum(ps * np.log(ps)))
    return entropy / math.log(n_reach)


def _ess_ratio(traj: np.ndarray) -> float:
    """ESS / N over the trajectory (GUARDRAIL ONLY — decorrelation, not coverage). In [0, 1]."""
    n = len(traj)
    if n < 2:
        return 0.0
    tx = _integrated_autocorr_time(traj[:, 0])
    ty = _integrated_autocorr_time(traj[:, 1])
    tau = 0.5 * (tx + ty)
    ess = n / (2.0 * tau)
    return min(ess / n, 1.0)


def _integrated_autocorr_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time via Sokal's truncated sum (FFT autocorrelation)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    var = float(np.dot(x, x) / n)
    if var < 1e-12:
        return float(n)  # constant series -> maximally correlated
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return tau
