"""Greedy per-opponent win-rate evaluation of a trained PPO agent (M1).

The M1 metric is win-rate: the fraction of greedy (deterministic) episodes the trained agent
wins. Unlike a fixed-opponent eval, this evaluates the policy against EACH selector in a roster
SEPARATELY (one fixed opponent per batch, via the self-play seam), so the breakdown shows where
the policy is strong / weak across the scripted family.

This module ships:

* :func:`win_rate` — the PURE counting helper (``Sequence[str] -> float``). No torch / no env,
  so the counting logic is unit-testable on its own (stdlib only).
* :func:`evaluate_winrate` — for EACH opponent selector, wraps the raw ``TankEnv`` in a
  :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` pinned to that ONE opponent, plays
  ``n_episodes`` greedy episodes, and returns ``{selector: win_rate}`` (insertion order matches
  the ``opponents`` argument). It reads the per-episode result from ``info["outcome"]`` (surfaced
  by ``TankEnv.step`` on TERMINAL steps only), NOT from the reward sign — the win signal is
  unambiguous regardless of reward shaping.
* :func:`overall_win_rate` / :func:`format_per_map_table` — OPTIONAL pure aggregation helpers
  over a per-opponent breakdown.

Only :func:`evaluate_winrate` touches sb3 / the env (lazily / type-only); the pure ``win_rate``
and aggregation helpers stay import-light (stdlib only) so the torch-free unit suite can exercise
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:  # type-only imports: keep the pure helpers torch/sb3-free at runtime
    import gymnasium
    from stable_baselines3.common.base_class import BaseAlgorithm

__all__ = [
    "WIN",
    "LOSS",
    "DRAW",
    "win_rate",
    "evaluate_winrate",
    "overall_win_rate",
    "format_per_map_table",
]

# Outcome tokens — mirror ``TankEnv.step``'s ``info["outcome"]`` values.
WIN = "win"
LOSS = "loss"
DRAW = "draw"


def win_rate(outcomes: Sequence[str]) -> float:
    """Fraction of episodes won, in ``[0, 1]`` (PURE; no torch / no env).

    Counts ``"win"`` entries over the total number of outcomes. Draws and losses count toward the
    denominator but not the numerator (win-rate, not win-or-draw-rate). An empty sequence returns
    ``0.0`` (no episodes -> no wins). Unknown tokens are tolerated (they count toward the
    denominator only), so a stray ``info`` value never raises here.

    Args:
        outcomes: per-episode result tokens, each typically one of ``"win"`` / ``"loss"`` /
            ``"draw"`` (see :data:`WIN` / :data:`LOSS` / :data:`DRAW`).

    Returns:
        ``wins / len(outcomes)`` as a float in ``[0, 1]`` (``0.0`` for an empty input).
    """
    total = len(outcomes)
    if total == 0:
        return 0.0
    wins = sum(1 for o in outcomes if o == WIN)
    return wins / total


def evaluate_winrate(
    model: BaseAlgorithm,
    eval_env: gymnasium.Env,
    opponents: Sequence[str] = DEFAULT_ROSTER,
    n_episodes: int = 100,
    seed: int | None = None,
) -> dict[str, float]:
    """Play ``n_episodes`` greedy episodes vs. EACH opponent selector; return per-selector win-rate.

    For EACH selector in ``opponents`` (drawn from ``DEFAULT_ROSTER`` by default), the raw
    ``eval_env`` is wrapped in a :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` pinned to that
    ONE opponent (a single-entry round-robin roster), and ``n_episodes`` deterministic episodes
    (``model.predict(obs, deterministic=True)``) are played to completion. The per-episode result
    is taken from ``info["outcome"]`` on the TERMINAL step (``TankEnv`` surfaces ``"win"`` /
    ``"loss"`` / ``"draw"`` on ``terminated`` only); a truncated-without-outcome episode (time-limit
    / lost connection) is recorded as a ``"draw"`` (never a win) so it cannot inflate the metric.

    Wrapping ONE raw ``eval_env`` in successive ``SelfPlayWrapper``s across selectors is safe — the
    wrapper holds no destructive state on the base env — so each selector reuses the same connected
    env. ``model`` and the env are touched lazily / only here; the pure ``win_rate`` above stays
    import-light.

    Args:
        model: a built SB3 algorithm (e.g. a loaded ``PPO``) with ``.predict``.
        eval_env: the UNDERLYING raw ``TankEnv`` (a gymnasium env, NOT a ``SelfPlayWrapper``).
        opponents: a sequence of ``agents`` selector strings (e.g. ``("noop", "random", ...)``).
        n_episodes: number of greedy episodes per opponent.
        seed: optional seed threaded into each opponent's ``OpponentProvider`` (agent +
            sampling reproducibility); also passed to each batch's first ``reset``.

    Returns:
        ``{selector: win_rate}`` in the order of ``opponents`` (a regular insertion-ordered dict).
    """
    # Lazy import: the self-play wrapper pulls in env/agents — keep the pure helpers above cheap.
    from pop_trainer.rl.selfplay import OpponentProvider, SelfPlayWrapper

    results: dict[str, float] = {}
    for selector in opponents:
        provider = OpponentProvider.from_roster([selector], strategy="round_robin", seed=seed)
        wrapped = SelfPlayWrapper(eval_env, provider)
        outcomes: list[str] = []
        for episode in range(n_episodes):
            # Seed only the FIRST reset of each batch (opponent reproducibility); subsequent
            # episodes advance the env's RNG so each episode differs. NO switch_arena option is
            # ever passed (caller-driven rotation is suppressed during eval — see callbacks.py).
            reset_seed = seed if (episode == 0 and seed is not None) else None
            obs, _info = wrapped.reset(seed=reset_seed)
            outcome = DRAW
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = wrapped.step(action)
                if terminated:
                    # Decided game: trust the env's unambiguous outcome token.
                    outcome = info.get("outcome", DRAW)
            outcomes.append(outcome)
        results[selector] = win_rate(outcomes)
    return results


def _as_pairs(
    per_opponent: Mapping[str, float] | Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize the per-opponent input to an ordered list of ``(name, win_rate)`` pairs (PURE).

    Accepts either an ordered mapping (e.g. a ``dict``, insertion order preserved) or an
    already-ordered sequence of ``(name, rate)`` tuples. Shared by :func:`overall_win_rate` and
    :func:`format_per_map_table` so both agree on ordering.
    """
    if isinstance(per_opponent, Mapping):
        return list(per_opponent.items())
    return [(name, rate) for name, rate in per_opponent]


def overall_win_rate(
    per_opponent: Mapping[str, float] | Sequence[tuple[str, float]],
) -> float:
    """Overall win-rate across opponents: the UNWEIGHTED MEAN of the per-opponent rates (PURE).

    Equals the pooled win-rate (total wins / total episodes) when every opponent is played the
    SAME number of episodes (the eval contract). No torch / no env — unit-testable on a plain
    mapping. An empty input returns ``0.0``, mirroring :func:`win_rate` on an empty input.
    """
    pairs = _as_pairs(per_opponent)
    if not pairs:
        return 0.0
    return sum(rate for _name, rate in pairs) / len(pairs)


def format_per_map_table(
    per_opponent: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    episodes: int | None = None,
) -> str:
    """Format the per-opponent win-rate breakdown as a single human-readable line (PURE).

    Produces ``"<opp>: <rate> | ... | OVERALL: <overall>"`` with each rate to two decimals, in the
    input's order. When ``episodes`` is given an accounting suffix ``" (N opponents x E episodes)"``
    is appended. The OVERALL value is :func:`overall_win_rate`. An empty input yields just
    ``"OVERALL: 0.00"`` so the formatter never raises on a degenerate breakdown.
    """
    pairs = _as_pairs(per_opponent)
    overall = overall_win_rate(pairs)
    cells = [f"{name}: {rate:.2f}" for name, rate in pairs]
    cells.append(f"OVERALL: {overall:.2f}")
    line = " | ".join(cells)
    if episodes is not None:
        line += f" ({len(pairs)} opponents x {episodes} episodes)"
    return line
