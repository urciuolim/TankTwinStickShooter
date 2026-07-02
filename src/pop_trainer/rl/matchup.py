"""Pure math for win-rate-based (opponent x map) matchup sampling — import-light (stdlib only).

The ``--matchup-sampling winrate`` curriculum replaces the two independent per-episode samplers
(opponent rotation + map rotation) with ONE joint sampler over cells = (opponent selector x arena
target). This module holds the sampler's arithmetic as pure functions so the unit suite exercises
it with no torch / sb3 / env imports:

* :func:`matchup_cells` — the canonical CELL ENUMERATION (opponent-major cross product). Every
  consumer (the per-env :class:`~pop_trainer.rl.selfplay.MatchupProvider`, the main-process
  aggregation callback, the sidecar) derives its cell order from this ONE function, so a
  broadcast distribution indexes the same cell everywhere.
* :func:`outcome_to_float` — the outcome convention (mirrors eval's win logic in
  ``rl.evaluate``): ``"win"`` -> 1.0, ``"loss"`` -> 0.0, ``"draw"`` -> 0.5, and a done WITHOUT an
  outcome token (truncation / time-limit / lost connection) or an unknown token -> 0.5 (a draw,
  never a win), so a degenerate episode cannot inflate a cell's win-rate.
* :func:`ema_update` — the per-cell win-rate EMA fold, ``wr <- (1 - alpha) * wr + alpha *
  outcome``. Unseen cells start at :data:`INITIAL_WIN_RATE` (0.5).
* :func:`deficit_distribution` — ``P(cell) = floor * uniform + (1 - floor) *
  normalize(deficit)`` with ``deficit(cell) = 1 - wr(cell)``. The floor guarantees every cell
  keeps being sampled (min prob >= ``floor / n_cells``); an all-zero deficit (every cell fully
  won) falls back to uniform instead of dividing by zero.
* :func:`distribution_entropy` / :func:`worst_cells` — observability helpers for the compact
  rollout-boundary summary line.

Boundary: stdlib only (``math``); imported by ``rl.selfplay`` / ``rl.callbacks`` / the tests. No
cycles, nothing from ``data`` / ``pretraining``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "INITIAL_WIN_RATE",
    "matchup_cells",
    "outcome_to_float",
    "ema_update",
    "deficit_distribution",
    "distribution_entropy",
    "worst_cells",
]

# A cell tag: (opponent selector, arena target or None). None is the boot arena — the single-arena
# mode where the launched game config's arena is the only one and NO switch_arena is ever sent.
Cell = tuple[str, str | None]

# The prior for a never-played cell: an even chance of winning. All cells starting here makes the
# initial deficit vector constant, so the initial distribution is exactly uniform.
INITIAL_WIN_RATE = 0.5

# The outcome tokens TankEnv.step surfaces in info["outcome"] on TERMINAL steps, mapped to the
# scalar the EMA folds. Anything else (missing / unknown) maps to the draw value 0.5.
_OUTCOME_FLOATS = {"win": 1.0, "loss": 0.0, "draw": 0.5}


def matchup_cells(opponents: Sequence[str], maps: Sequence[str] | None = None) -> list[Cell]:
    """The canonical cell enumeration: opponent-major cross product of selectors x arenas (PURE).

    With ``maps is None`` (no rotation configured) each opponent forms ONE cell tagged with the
    ``None`` boot arena, so the sampler prioritizes over opponents only and never injects a
    ``switch_arena``. A non-``None`` ``maps`` must be non-empty. Every consumer of the joint
    sampler derives its cell ORDER from this function so a broadcast distribution indexes the
    same cell in every process.
    """
    selectors = list(opponents)
    if not selectors:
        raise ValueError("matchup_cells needs at least one opponent selector")
    if maps is not None and not list(maps):
        raise ValueError("matchup_cells needs at least one map (or None for the boot arena)")
    arenas: tuple[str | None, ...] = (None,) if maps is None else tuple(maps)
    return [(selector, arena) for selector in selectors for arena in arenas]


def outcome_to_float(token: str | None) -> float:
    """Map a terminal ``info["outcome"]`` token to the scalar the win-rate EMA folds (PURE).

    ``"win"`` -> 1.0, ``"loss"`` -> 0.0, ``"draw"`` -> 0.5. A ``None`` token — a done step WITHOUT
    an outcome (truncation / time-limit / lost connection) — and any unknown token map to 0.5:
    the same never-a-win convention eval uses (``rl.evaluate`` records such episodes as draws),
    so a degenerate episode can neither inflate nor crater a cell's win-rate.
    """
    return _OUTCOME_FLOATS.get(token, 0.5)


def ema_update(win_rate: float, outcome: float, alpha: float) -> float:
    """One exponential-moving-average fold: ``(1 - alpha) * win_rate + alpha * outcome`` (PURE).

    ``alpha`` in ``(0, 1]`` is the smoothing weight of the newest episode; ``alpha == 1`` replaces
    the running value outright.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return (1.0 - alpha) * win_rate + alpha * outcome


def deficit_distribution(win_rates: Sequence[float], floor: float) -> list[float]:
    """The sampling distribution over cells from their win-rates (PURE; sums to 1, never NaN).

    ``P(cell) = floor * uniform + (1 - floor) * normalize(deficit)`` with ``deficit(cell) =
    max(1 - wr, 0)``. Properties:

    * every probability is at least ``floor / n`` (the exploration floor keeps every cell
      sampled, so a cell's win-rate can always recover);
    * equal win-rates (including the all-0.5 initial state) yield the uniform distribution;
    * a higher deficit (lower win-rate) always gets at least as much probability;
    * an all-zero deficit total (every cell fully won) falls back to uniform instead of
      producing NaN.
    """
    n = len(win_rates)
    if n == 0:
        raise ValueError("deficit_distribution needs at least one cell")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    deficits = [max(1.0 - wr, 0.0) for wr in win_rates]
    total = sum(deficits)
    uniform = 1.0 / n
    if total <= 0.0:
        # All cells fully won: nothing to prioritize — uniform, not a division by zero.
        return [uniform] * n
    return [floor * uniform + (1.0 - floor) * (d / total) for d in deficits]


def distribution_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy of a distribution in nats (PURE); zero-probability terms contribute 0.

    Uniform over ``n`` cells gives ``log(n)``; a one-hot distribution gives ``0``. Used as the
    single aggregate TensorBoard scalar summarizing how concentrated the curriculum is.
    """
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def worst_cells(
    cells: Sequence[Cell], win_rates: Sequence[float], k: int
) -> list[tuple[Cell, float]]:
    """The ``k`` cells with the LOWEST win-rate, as ``(cell, win_rate)`` pairs (PURE).

    Ties keep the enumeration order (stable sort); ``k`` larger than the cell count returns all
    cells. This is the observability slice logged at each rollout-boundary update.
    """
    if len(cells) != len(win_rates):
        raise ValueError(
            f"cells and win_rates must have equal length, got {len(cells)} != {len(win_rates)}"
        )
    ranked = sorted(zip(cells, win_rates, strict=True), key=lambda pair: pair[1])
    return ranked[: max(k, 0)]
