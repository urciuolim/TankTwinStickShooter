"""Coverage-oriented agents — each maximises coverage of one state variable.

These model-free :class:`pop_trainer.core.agent.Agent` drivers exist to fill a recorded
trajectory with DIVERSE states, so a downstream supervised decoder sees the full range of each
52-float field. Each agent targets one variable:

* :class:`ExplorerAgent` — POSITION. Cycles movement HEADINGS (the 8 compass directions plus a
  centre-return) on a dwell schedule, so the tank physically sweeps the whole floor.
* :class:`AimSweepAgent` — AIM. Rotates the aim unit vector through a full 360 over a fixed
  period, sweeping the AIM_X / AIM_Y channels around the unit circle.
* :class:`SprayAgent` — BULLET position/direction. Fires every step while moving and sweeping
  aim, so bullets spawn at many positions with many directions.
* :class:`PerimeterAgent` — position EXTREMES + wall-adjacent states. Right-hand wall-follows the
  four edges so the tank hugs the boundary and reaches the corners.

DESIGN RATIONALE (read before expecting closed-loop bounds). :mod:`pop_trainer.core` exposes NO
arena-bounds constant, so coverage of POSITION-space cannot be planned against absolute
coordinates. The correct strategy given the contract is therefore heading/phase-driven
(open-loop sweeps): cycling movement DIRECTIONS makes the tank's position change to span the
floor without any agent inventing a bounds constant the core does not provide. Where an agent
CAN read its own state to sharpen coverage it does so via :mod:`pop_trainer.core.state` (e.g.
:class:`ExplorerAgent` advances its waypoint early once its observed velocity shows it has been
committed to a heading) — but it never fabricates bounds.

Per the package's self-play convention, a state-reading agent treats ``PLAYER_1`` of the view it
receives as ITSELF. All these agents act as ``PLAYER_1`` on their own view.

stdlib + numpy compatible; imports :mod:`pop_trainer.core.state` for the schema accessors only.
"""

from __future__ import annotations

import math

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

__all__ = [
    "ExplorerAgent",
    "AimSweepAgent",
    "SprayAgent",
    "PerimeterAgent",
]

# The 8 compass move headings (unit vectors) plus a centre-return (zero) heading. Cycling these
# sweeps the tank outward in every direction; the zero heading lets it settle before re-aiming.
_COMPASS_HEADINGS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),  # E
    (math.sqrt(0.5), math.sqrt(0.5)),  # NE
    (0.0, 1.0),  # N
    (-math.sqrt(0.5), math.sqrt(0.5)),  # NW
    (-1.0, 0.0),  # W
    (-math.sqrt(0.5), -math.sqrt(0.5)),  # SW
    (0.0, -1.0),  # S
    (math.sqrt(0.5), -math.sqrt(0.5)),  # SE
    (0.0, 0.0),  # centre-return / settle
)

# The 4 cardinal move headings for wall-following (E, N, W, S), turned through clockwise.
_CARDINAL_HEADINGS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),  # E
    (0.0, 1.0),  # N
    (-1.0, 0.0),  # W
    (0.0, -1.0),  # S
)


class ExplorerAgent:
    """POSITION coverage — cycle movement headings on a dwell schedule to sweep the whole floor.

    Coverage comes from cycling movement HEADINGS (which change the tank's position), not from
    absolute coordinates (which the core does not bound). Each ``act`` drives the current heading;
    after ``dwell`` steps — or earlier once the observed velocity confirms the tank is committed to
    the heading — it advances to the next of the eight compass directions plus a centre-return, so
    over a full tour the tank physically reaches every part of the arena. Optional RNG jitter
    perturbs the heading angle so repeated tours are not identical. Aim follows the move heading;
    fire is held off. Stateful: :meth:`reset` restarts the tour deterministically.
    """

    def __init__(
        self,
        *,
        dwell: int = 12,
        jitter: float = 0.0,
        seed: int | None = None,
    ):
        if dwell < 1:
            raise ValueError(f"dwell must be >= 1, got {dwell}")
        self._dwell = int(dwell)
        self._jitter = float(jitter)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._ticks = 0

    def act(self, obs) -> list[float]:
        vec = as_state_vector(obs)
        hx, hy = _COMPASS_HEADINGS[self._index % len(_COMPASS_HEADINGS)]

        # Closed-loop early advance: if we are clearly moving along this heading, the dwell can be
        # cut short so the tour visits more directions per episode. The centre-return (zero) heading
        # has no velocity signal, so it only times out on the dwell budget.
        committed = False
        if hx or hy:
            vx, vy = state_schema.velocity(vec, state_schema.PLAYER_1)
            committed = (float(vx) * hx + float(vy) * hy) > 0.5

        if self._jitter and (hx or hy):
            angle = math.atan2(hy, hx) + float(self._rng.uniform(-self._jitter, self._jitter))
            hx, hy = math.cos(angle), math.sin(angle)

        action = [0.0] * ACTION_LEN
        action[A_MOVE_X] = hx
        action[A_MOVE_Y] = hy
        action[A_AIM_X] = hx
        action[A_AIM_Y] = hy

        self._ticks += 1
        if self._ticks >= self._dwell or (committed and self._ticks >= self._dwell // 2):
            self._index += 1
            self._ticks = 0
        return action

    def reset(self, *, seed: int | None = None) -> None:
        """Restart the tour; re-seed the jitter RNG (``None`` keeps the construction seed)."""
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._index = 0
        self._ticks = 0

    def __repr__(self) -> str:
        return f"ExplorerAgent(dwell={self._dwell}, jitter={self._jitter})"


class AimSweepAgent:
    """AIM coverage — rotate the aim unit vector through a full 360 while moving.

    Each ``act`` advances an internal phase by ``2*pi / period`` and emits an aim unit vector at
    that angle, so over one period the AIM_X / AIM_Y channels trace the whole unit circle. A
    nonzero move heading keeps the tank in motion (so the swept aim is sampled at varied positions).
    Fire defaults off (set ``fire=True`` for a firing sweep). Stateful: :meth:`reset` zeroes the
    phase, making the first post-reset action deterministic.
    """

    def __init__(
        self,
        *,
        period: int = 16,
        move: tuple[float, float] = (1.0, 0.0),
        fire: bool = False,
    ):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self._period = int(period)
        self._move = (float(move[0]), float(move[1]))
        self._fire = 1.0 if fire else 0.0
        self._phase = 0

    def act(self, obs) -> list[float]:  # noqa: ARG002 (open-loop sweep: ignores obs)
        angle = (2.0 * math.pi * self._phase) / self._period
        action = [0.0] * ACTION_LEN
        action[A_MOVE_X] = self._move[0]
        action[A_MOVE_Y] = self._move[1]
        action[A_AIM_X] = math.cos(angle)
        action[A_AIM_Y] = math.sin(angle)
        action[A_FIRE] = self._fire
        self._phase = (self._phase + 1) % self._period
        return action

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002 (no entropy to seed)
        """Zero the sweep phase so the aim restarts at angle 0."""
        self._phase = 0

    def __repr__(self) -> str:
        return f"AimSweepAgent(period={self._period}, fire={bool(self._fire)})"


class SprayAgent:
    """BULLET position/direction coverage — fire every step while moving and sweeping aim.

    Bullets inherit the tank's position and the aim direction at spawn, so firing continuously
    while the tank moves AND the aim sweeps the unit circle spawns bullets at many positions with
    many directions — broad coverage of the bullet pos/vec state variables. Each ``act`` advances
    an aim phase (like :class:`AimSweepAgent`), moves along a heading, and fires. Stateful:
    :meth:`reset` zeroes the phase.
    """

    def __init__(
        self,
        *,
        period: int = 16,
        move: tuple[float, float] = (1.0, 0.0),
    ):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self._period = int(period)
        self._move = (float(move[0]), float(move[1]))
        self._phase = 0

    def act(self, obs) -> list[float]:  # noqa: ARG002 (open-loop spray: ignores obs)
        angle = (2.0 * math.pi * self._phase) / self._period
        action = [0.0] * ACTION_LEN
        action[A_MOVE_X] = self._move[0]
        action[A_MOVE_Y] = self._move[1]
        action[A_AIM_X] = math.cos(angle)
        action[A_AIM_Y] = math.sin(angle)
        action[A_FIRE] = 1.0
        self._phase = (self._phase + 1) % self._period
        return action

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002 (no entropy to seed)
        """Zero the aim-sweep phase so the spray restarts at angle 0."""
        self._phase = 0

    def __repr__(self) -> str:
        return f"SprayAgent(period={self._period})"


class PerimeterAgent:
    """Position EXTREMES + wall-adjacent coverage — right-hand wall-follow the four edges.

    Drives one cardinal direction for a dwell, then turns 90 clockwise to the next edge, cycling E
    -> N -> W -> S so the tank hugs the boundary and reaches the position extremes / corners that
    centre-biased policies never sample. Aim follows the move heading; fire is held off. Stateful:
    :meth:`reset` restarts the wall-follow.
    """

    def __init__(self, *, dwell: int = 16):
        if dwell < 1:
            raise ValueError(f"dwell must be >= 1, got {dwell}")
        self._dwell = int(dwell)
        self._index = 0
        self._ticks = 0

    def act(self, obs) -> list[float]:  # noqa: ARG002 (open-loop wall-follow: ignores obs)
        hx, hy = _CARDINAL_HEADINGS[self._index % len(_CARDINAL_HEADINGS)]
        action = [0.0] * ACTION_LEN
        action[A_MOVE_X] = hx
        action[A_MOVE_Y] = hy
        action[A_AIM_X] = hx
        action[A_AIM_Y] = hy

        self._ticks += 1
        if self._ticks >= self._dwell:
            self._index += 1
            self._ticks = 0
        return action

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002 (no entropy to seed)
        """Restart the wall-follow at the first cardinal edge."""
        self._index = 0
        self._ticks = 0

    def __repr__(self) -> str:
        return f"PerimeterAgent(dwell={self._dwell})"
