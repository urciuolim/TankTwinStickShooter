"""Map-aware occupancy-biased coverage — the one data-collection movement family + presets.

A single policy family fills the data-collection surface: an occupancy-biased coverage walk
that drives the tank to sweep the free floor of a map, parameterized by named presets for
population diversity. The design (load-bearing, from the 2026 research report):

* Maintain a **visit-count grid** over the FREE cells.
* Each re-pick, choose the **NEAREST least-recently-visited** free cell as the waypoint, via a
  plain BFS distance field FROM the current cell (greedy short hops — preferring far cells
  scored worse than baseline, so we do not).
* Navigate toward the waypoint via a plain BFS distance field FROM the waypoint (a
  ``collections.deque`` flood, NO A* / heap — the grid is ~128 cells), descending the gradient
  to the free 4-neighbour with the smallest distance-to-waypoint.
* Add a **reactive wall-repulsion** nudge (push away from adjacent OCCUPIED cells), combined as
  ``h = gradient + 0.3 * repulsion`` (gradient dominant). A reactive nudge, NOT a raw potential
  field — those trap in local minima.
* **eps noise** for decorrelation: with probability ``eps`` emit a random heading; on a
  zero/degenerate combined heading, fall back to a random heading.
* Re-pick the waypoint only every ``K`` steps OR when reached, so the twice-per-repick BFS
  stays cheap inside ``act``.

THE SIM-VS-REAL SEAM (free-cell derivation). The production family gets its grid from a
:class:`pop_trainer.core.protocol.WallLayout` ALONE: ``free = {all cells in the dims bounding
box} - occupied``. This is NOT the spike's arena ``Floor`` block (which the wire does not
carry). The visit grid is restricted to the BFS-reachable component from the spawn cell so
unreachable border cells never poison the least-visited pick; the tank's current continuous
position is snapped to the nearest free cell when it is off-grid. When ``set_map`` is never
called (no arena configured), the family degrades to a safe map-blind random-heading fallback —
it never crashes.

ORTHOGONAL AIM/FIRE. The movement family is POSITION-only (it sets ``move_x`` / ``move_y``); an
:class:`AimFireSchedule` mixin layers a cheap aim sweep + periodic fire so the aim and bullet
pos/dir state channels are also exercised. ``opponent-shadower`` swaps the sweep for aiming at
the opponent (read from the 52-float state).

Boundaries: imports :mod:`pop_trainer.core` only (the ``WallLayout`` type for annotations, the
state schema) plus numpy + stdlib. No torch; nothing from ``env`` / ``data`` / ``pretraining``
/ ``rl`` / ``models``; nothing from ``tank_twin``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from pop_trainer.agents.base import (
    A_AIM_X,
    A_AIM_Y,
    A_FIRE,
    A_MOVE_X,
    A_MOVE_Y,
    ACTION_LEN,
    as_state_vector,
)
from pop_trainer.core import state as state_schema

if TYPE_CHECKING:
    from pop_trainer.core.protocol import WallLayout

__all__ = [
    "Cell",
    "free_cells_from_layout",
    "reachable_from",
    "bfs_distance_field",
    "cell_of",
    "AimFireSchedule",
    "CoverageAgent",
]

Cell = tuple[int, int]

# Continuous world position -> integer grid cell (mirrors the spike's cell_of).
_NEIGHBOR_DELTAS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def cell_of(wx: float, wy: float) -> Cell:
    """Snap a continuous world position to its integer grid cell ``(round(x), round(y))``."""
    return (int(round(wx)), int(round(wy)))


def _neighbors4(c: Cell) -> list[Cell]:
    x, y = c
    return [(x + dx, y + dy) for dx, dy in _NEIGHBOR_DELTAS]


def free_cells_from_layout(layout: WallLayout) -> frozenset[Cell]:
    """Derive the FREE cells from a :class:`~pop_trainer.core.protocol.WallLayout`.

    ``free = {all (x, y) in the dims bounding box} - layout.occupied``. This is the production
    derivation (the wire carries Walls + dims only, never the arena Floor block).
    """
    d = layout.dims
    occupied = layout.occupied
    return frozenset(
        (x, y)
        for x in range(d.min_x, d.max_x + 1)
        for y in range(d.min_y, d.max_y + 1)
        if (x, y) not in occupied
    )


def reachable_from(free: frozenset[Cell], start: Cell) -> set[Cell]:
    """The BFS-reachable free component from ``start`` (snapped to the nearest free cell).

    Restricts the visit grid to one connected component so unreachable border cells never
    poison the least-visited pick. ``start`` is snapped to the nearest free cell if off-grid.
    Returns the empty set when ``free`` is empty.
    """
    if not free:
        return set()
    if start not in free:
        start = _nearest_free(free, start)
    seen = {start}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in _neighbors4(c):
            if n in free and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def bfs_distance_field(free: set[Cell], source: Cell) -> dict[Cell, int]:
    """Plain single-source BFS distance over ``free`` from ``source`` -> ``{cell: hops}``.

    A ``collections.deque`` flood (NO heap / priority queue — the grid is ~128 cells). Cells
    not reachable from ``source`` within ``free`` are absent from the result.
    """
    if source not in free:
        return {}
    dist = {source: 0}
    q = deque([source])
    while q:
        c = q.popleft()
        for n in _neighbors4(c):
            if n in free and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def _nearest_free(free: frozenset[Cell] | set[Cell], target: Cell) -> Cell:
    """The free cell nearest ``target`` by squared distance (``free`` must be non-empty)."""
    tx, ty = target
    return min(free, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)


class AimFireSchedule:
    """Orthogonal aim-sweep + fire cadence layered on the POSITION-only movement family.

    The movement family leaves aim/fire free; this fills the aim and bullet pos/dir state
    channels. Each step it advances an aim phase by ``2*pi / aim_period`` (sweeping the unit
    circle) and fires once every ``fire_period`` steps. The cadence is deliberately MODEST:
    bullets do not bounce (they die on wall/tank contact), so firing every step would have new
    bullets immediately self-block; a periodic fire spaces them out.

    ``opponent-shadower`` overrides the sweep by passing an explicit aim direction to
    :meth:`aim_fire` (the unit vector toward the opponent read from the state); the fire cadence
    is unchanged.
    """

    def __init__(self, *, aim_period: int = 16, fire_period: int = 6):
        if aim_period < 1:
            raise ValueError(f"aim_period must be >= 1, got {aim_period}")
        if fire_period < 1:
            raise ValueError(f"fire_period must be >= 1, got {fire_period}")
        self._aim_period = int(aim_period)
        self._fire_period = int(fire_period)
        self._aim_phase = 0
        self._fire_tick = 0

    def reset(self) -> None:
        """Zero the aim phase + fire cadence so the schedule restarts deterministically."""
        self._aim_phase = 0
        self._fire_tick = 0

    def aim_fire(self, aim: tuple[float, float] | None = None) -> tuple[float, float, float]:
        """Advance the schedule one step -> ``(aim_x, aim_y, fire)``.

        With ``aim=None`` the aim is the swept unit vector at the current phase; pass an explicit
        ``(aim_x, aim_y)`` (e.g. toward the opponent) to override the sweep. ``fire`` is ``1.0``
        on the periodic fire step, else ``0.0``.
        """
        if aim is None:
            angle = (2.0 * math.pi * self._aim_phase) / self._aim_period
            ax, ay = math.cos(angle), math.sin(angle)
        else:
            ax, ay = aim
        self._aim_phase = (self._aim_phase + 1) % self._aim_period
        fire = 1.0 if self._fire_tick == 0 else 0.0
        self._fire_tick = (self._fire_tick + 1) % self._fire_period
        return ax, ay, fire


class CoverageAgent:
    """Occupancy-biased map coverage — the one movement family + an aim/fire layer.

    Movement (position-only): a visit-count grid over the reachable free cells; each re-pick
    selects the nearest least-visited cell as a waypoint and the tank descends a BFS gradient
    toward it, nudged away from adjacent walls, with eps decorrelation noise. Aim/fire come from
    an :class:`AimFireSchedule`. Construct presets via the classmethods
    :meth:`aggressive` / :meth:`wall_hugger` / :meth:`opponent_shadower`.

    Map hook: :meth:`set_map` (re)builds the grid from a
    :class:`~pop_trainer.core.protocol.WallLayout`. Until it is called — or when the layout has
    no reachable free cells — the agent degrades to a map-blind random heading (it never
    crashes). The reachability seed is the tank's first observed position (snapped); the grid is
    built lazily on the first ``act`` once a position is known.

    Knobs:

    * ``K`` — re-pick the waypoint every ``K`` steps (or on reach). Smaller = more responsive.
    * ``eps`` — probability of a random decorrelation heading each step.
    * ``border_bias`` — when ``True`` (the wall-hugger preset) the least-visited tiebreak prefers
      free cells ADJACENT to an occupied/out-of-bounds cell, biasing the walk toward wall-hugging
      / position-extreme coverage. When ``False`` (the default) ties break on nearest hop only.
    * ``shadow_opponent`` — when ``True`` the aim layer aims at the opponent (read from the
      52-float state, self-play convention: the agent is PLAYER_1 of its own view) instead of
      sweeping.
    """

    REPULSION_WEIGHT = 0.3

    def __init__(
        self,
        *,
        seed: int | None = None,
        K: int = 4,
        eps: float = 0.02,
        border_bias: bool = False,
        shadow_opponent: bool = False,
        aim_period: int = 16,
        fire_period: int = 6,
    ):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self._seed = seed
        self._K = int(K)
        self._eps = float(eps)
        self._border_bias = bool(border_bias)
        self._shadow_opponent = bool(shadow_opponent)
        self._rng = np.random.default_rng(seed)
        self._aimfire = AimFireSchedule(aim_period=aim_period, fire_period=fire_period)

        # Map-derived state, (re)built by set_map / lazily on the first act with a position.
        self._free: frozenset[Cell] | None = None  # full free set (for wall-repulsion tests)
        self._reach: set[Cell] = set()  # the reachable component (the visit grid domain)
        self._border: frozenset[Cell] = frozenset()  # reachable cells adjacent to a wall
        self._visits: dict[Cell, int] = {}
        self._waypoint: Cell | None = None
        self._field: dict[Cell, int] = {}
        self._since = 0
        self._grid_ready = False  # reachability resolved from a known position

    # --- preset constructors -------------------------------------------------------------

    @classmethod
    def aggressive(cls, *, seed: int | None = None) -> CoverageAgent:
        """Max-coverage preset: the tuned K=4 / eps=0.02 baseline with an aim sweep + fire."""
        return cls(seed=seed, K=4, eps=0.02, border_bias=False, shadow_opponent=False)

    @classmethod
    def wall_hugger(cls, *, seed: int | None = None) -> CoverageAgent:
        """Wall-adjacent / position-extreme preset: least-visited ties prefer border cells.

        The knob turned is ``border_bias=True`` — among the least-visited candidates, ones
        ADJACENT to a wall (occupied / out-of-bounds) are preferred, so the walk hugs the
        boundary and samples the position extremes a centre-biased walk misses.
        """
        return cls(seed=seed, K=4, eps=0.02, border_bias=True, shadow_opponent=False)

    @classmethod
    def opponent_shadower(cls, *, seed: int | None = None) -> CoverageAgent:
        """Same coverage movement, but the aim layer aims at the opponent and fires.

        Subsumes the old aim-at-player2 rule: the movement still maximises coverage while the aim
        tracks the opponent (read from the state), so the trajectory exercises both coverage and
        a purposeful aim signal.
        """
        return cls(seed=seed, K=4, eps=0.02, border_bias=False, shadow_opponent=True)

    # --- the map hook --------------------------------------------------------------------

    def set_map(self, layout: WallLayout) -> None:
        """(Re)build the free grid from ``layout`` and clear the per-episode visit state.

        The reachable component is resolved lazily on the first :meth:`act` (it needs a spawn
        position from the state). A ``None`` layout, or one with no free cells, leaves the agent
        in the map-blind fallback.
        """
        self._reset_episode_state()
        if layout is None:
            self._free = None
            return
        free = free_cells_from_layout(layout)
        self._free = free if free else None

    def reset(self, *, seed: int | None = None) -> None:
        """Re-seed the RNG and clear per-episode walk state; keep the current map grid.

        ``seed=None`` restores the construction seed for an exact replay. The free grid set by
        :meth:`set_map` is retained (a new episode on the SAME map keeps the geometry), but the
        visit counts / waypoint / aim-fire phase are cleared so the walk restarts.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._aimfire.reset()
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._reach = set()
        self._border = frozenset()
        self._visits = {}
        self._waypoint = None
        self._field = {}
        self._since = 0
        self._grid_ready = False

    # --- the policy ----------------------------------------------------------------------

    def act(self, obs) -> list[float]:
        """Return ``[move_x, move_y, aim_x, aim_y, fire]`` for the current 52-float state view.

        Movement is the occupancy-biased coverage heading; aim/fire come from the schedule (a
        sweep, or aimed at the opponent for the shadower preset). With no usable map the move is
        a random heading (the map-blind fallback) while aim/fire still run.
        """
        vec = as_state_vector(obs)
        sx, sy = state_schema.position(vec, state_schema.PLAYER_1)
        mx, my = self._movement_heading(float(sx), float(sy))

        aim = self._aim_at_opponent(vec) if self._shadow_opponent else None
        ax, ay, fire = self._aimfire.aim_fire(aim)

        action = [0.0] * ACTION_LEN
        action[A_MOVE_X] = mx
        action[A_MOVE_Y] = my
        action[A_AIM_X] = ax
        action[A_AIM_Y] = ay
        action[A_FIRE] = fire
        return action

    def _movement_heading(self, sx: float, sy: float) -> tuple[float, float]:
        """The occupancy-biased coverage heading, or a random fallback with no usable map."""
        if self._free is None:
            return self._random_heading()
        if not self._grid_ready:
            self._build_reachable(cell_of(sx, sy))
            if not self._reach:
                return self._random_heading()

        cur = cell_of(sx, sy)
        if cur not in self._reach:
            cur = _nearest_free(self._reach, cur)
        self._visits[cur] = self._visits.get(cur, 0) + 1

        if self._waypoint is None or self._since >= self._K or cur == self._waypoint:
            self._pick_waypoint(cur)
        self._since += 1

        gx, gy = self._descend_gradient(cur)
        rx, ry = self._wall_repulsion(cur)
        hx = gx + self.REPULSION_WEIGHT * rx
        hy = gy + self.REPULSION_WEIGHT * ry

        if self._rng.uniform(0.0, 1.0) < self._eps:
            return self._random_heading()
        return self._normalize(hx, hy)

    def _build_reachable(self, spawn: Cell) -> None:
        """Resolve the reachable component from the spawn cell and seed the visit grid."""
        assert self._free is not None  # noqa: S101 (guarded by the caller)
        self._reach = reachable_from(self._free, spawn)
        self._visits = {c: 0 for c in self._reach}
        self._border = frozenset(c for c in self._reach if self._is_border(c))
        self._grid_ready = True

    def _is_border(self, c: Cell) -> bool:
        """Whether ``c`` has at least one adjacent cell that is occupied / out of the free set."""
        free = self._free
        return any((c[0] + dx, c[1] + dy) not in free for dx, dy in _NEIGHBOR_DELTAS)

    def _pick_waypoint(self, cur: Cell) -> None:
        """Pick the NEAREST least-visited reachable cell as the waypoint; build its BFS field."""
        minv = min(self._visits.values())
        cands = [c for c, v in self._visits.items() if v == minv]
        if self._border_bias:
            border_cands = [c for c in cands if c in self._border]
            if border_cands:
                cands = border_cands
        from_cur = bfs_distance_field(self._reach, cur)
        wp = min(
            (c for c in cands if c in from_cur),
            key=lambda c: from_cur[c],
            default=None,
        )
        if wp is None:
            wp = cands[int(self._rng.integers(0, len(cands)))]
        self._waypoint = wp
        self._field = bfs_distance_field(self._reach, wp)
        self._since = 0

    def _descend_gradient(self, cur: Cell) -> tuple[float, float]:
        """Step toward the lowest distance-to-waypoint free neighbour; random if none lower."""
        best: Cell | None = None
        best_d = self._field.get(cur, math.inf)
        for n in _neighbors4(cur):
            d = self._field.get(n)
            if d is not None and d < best_d:
                best_d = d
                best = n
        if best is None:
            return self._random_heading()
        return float(best[0] - cur[0]), float(best[1] - cur[1])

    def _wall_repulsion(self, cur: Cell) -> tuple[float, float]:
        """A local 4-neighbour nudge pushing away from adjacent occupied / off-grid cells."""
        free = self._free
        rx = ry = 0.0
        for dx, dy in _NEIGHBOR_DELTAS:
            if (cur[0] + dx, cur[1] + dy) not in free:
                rx -= dx
                ry -= dy
        return rx, ry

    def _aim_at_opponent(self, vec) -> tuple[float, float]:
        """Unit vector from PLAYER_1 (self, own view) toward PLAYER_2 (opponent); zero if same."""
        sx, sy = state_schema.position(vec, state_schema.PLAYER_1)
        ox, oy = state_schema.position(vec, state_schema.PLAYER_2)
        dx, dy = float(ox) - float(sx), float(oy) - float(sy)
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            return (0.0, 0.0)
        return (dx / dist, dy / dist)

    def _random_heading(self) -> tuple[float, float]:
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        return math.cos(angle), math.sin(angle)

    def _normalize(self, hx: float, hy: float) -> tuple[float, float]:
        n = math.hypot(hx, hy)
        if n < 1e-9:
            return self._random_heading()
        return hx / n, hy / n

    def __repr__(self) -> str:
        return (
            f"CoverageAgent(K={self._K}, eps={self._eps}, "
            f"border_bias={self._border_bias}, shadow_opponent={self._shadow_opponent})"
        )
