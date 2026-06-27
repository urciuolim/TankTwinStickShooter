"""Greedy per-opponent win-rate evaluation of a trained PPO agent (M1).

The M1 metric is win-rate: the fraction of greedy (deterministic) episodes the trained agent
wins. Unlike a fixed-opponent eval, this evaluates the policy against EACH selector in a roster
SEPARATELY (one fixed opponent per batch, via the self-play seam), so the breakdown shows where
the policy is strong / weak across the scripted family.

This module ships:

* :func:`win_rate` — the PURE counting helper (``Sequence[str] -> float``). No torch / no env,
  so the counting logic is unit-testable on its own (stdlib only).
* :func:`evaluate_winrate` — for EACH opponent selector, plays ``n_episodes`` greedy episodes and
  returns ``{selector: win_rate}`` (insertion order matches the ``opponents`` argument). It reads
  the per-episode result from ``info["outcome"]`` (surfaced by ``TankEnv.step`` on TERMINAL steps
  only), NOT from the reward sign — the win signal is unambiguous regardless of reward shaping.
  The eval is PARALLELIZED across ``M`` eval envs: ``n_episodes`` are DISTRIBUTED across the envs
  and played concurrently (the M lanes step together, batched through ``model.predict``), then
  tallied per opponent. ``M == 1`` (a single env / ``DummyVecEnv``) is the sequential path and is
  bit-equivalent to the old behaviour; ``M > 1`` gives the SAME per-opponent rates for the same
  scripted outcomes (only faster — the win-rate MATH is unchanged). The eval env(s) may be passed
  as a single raw ``TankEnv``, a list of raw envs, or a vec env of ``M`` envs.
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


def _as_eval_vec(eval_env):
    """Normalize the eval-env arg into a VEC env of ``M`` ``SelfPlayWrapper``-wrapped envs.

    Accepts three shapes so the caller can pass whatever it holds and ALWAYS drives the same
    parallel vec path:

    * a VEC env (exposes ``num_envs`` + ``set_attr`` + ``step`` + ``reset``) -> used AS-IS. Each of
      its envs must be a :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` (its ``opponents``
      provider is re-pinned per selector via ``set_attr``). This is the LIVE path: at ``M > 1`` it
      is a ``SubprocVecEnv`` whose workers step concurrently across processes — true parallel eval.
    * a single raw env -> wrapped in a one-env ``DummyVecEnv`` of a ``SelfPlayWrapper`` (the
      sequential ``M == 1`` path).
    * a LIST/TUPLE of M raw envs -> a ``DummyVecEnv`` of M ``SelfPlayWrapper``s (in-process
      parallel lanes; the unit-test seam).

    Returns ``(vec, owned)`` — ``owned`` is ``True`` when this function BUILT the vec (raw / list
    input), a throwaway ``DummyVecEnv`` the caller must not close (the raw envs are owned
    elsewhere). For a passed-in vec, ``owned`` is ``False``.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from pop_trainer.rl.selfplay import OpponentProvider, SelfPlayWrapper

    if hasattr(eval_env, "num_envs") and hasattr(eval_env, "set_attr"):
        return eval_env, False

    raw_envs = list(eval_env) if isinstance(eval_env, list | tuple) else [eval_env]
    # A throwaway placeholder provider; evaluate_winrate re-pins each env per selector via set_attr.
    placeholder = OpponentProvider.from_roster([DEFAULT_ROSTER[0]], strategy="round_robin")
    # default-arg capture so each closure binds its OWN raw env (not the loop variable).
    fns = [(lambda r=r: SelfPlayWrapper(r, placeholder)) for r in raw_envs]
    return DummyVecEnv(fns), True


def _eval_opponent_vec(model, vec, selector: str, n_episodes: int, seed) -> float:
    """Play ``n_episodes`` greedy episodes vs. ONE opponent across the vec's M envs; win-rate.

    Pins every env's :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` to a single-selector
    ``OpponentProvider`` for ``selector`` (``vec.set_attr("opponents", ...)`` — fanned to every
    worker; the provider is picklable), then drives the vec: ``reset`` once, then repeatedly
    ``predict(obs, deterministic=True)`` + ``vec.step(actions)``. The M envs step TOGETHER each
    iteration (a ``SubprocVecEnv`` runs its workers concurrently across processes -> true parallel
    eval), and SB3's vec AUTO-RESETS any finished env, so episodes stream across the lanes. The
    first ``n_episodes`` outcomes observed (env-index order within a step) are tallied — the
    win-rate MATH is :func:`win_rate` over exactly ``n_episodes`` outcomes, so it is INDEPENDENT of
    M (M==1 reproduces the sequential result for the same scripted outcomes).

    The per-episode outcome is ``info["outcome"]`` on the step where ``done`` is set (``TankEnv``
    surfaces it on ``terminated`` only); a done WITHOUT an ``"outcome"`` (truncation / time-limit /
    lost connection) is a DRAW, never a win. The per-env providers carry ``seed`` for opponent +
    first-episode reproducibility. NO ``switch_arena`` is ever sent (eval-rotation suppression —
    plain ``reset``/``step`` only; see ``callbacks.py``).
    """
    from pop_trainer.rl.selfplay import OpponentProvider

    # Pin EACH env to its OWN single-selector provider (a fresh provider per env so their
    # RNG/index are independent). SB3 set_attr sets ONE value across the given indices, so set
    # per-env by index (indices=i fans to worker i for a SubprocVecEnv; in-process for Dummy).
    for i in range(vec.num_envs):
        provider = OpponentProvider.from_roster([selector], strategy="round_robin", seed=seed)
        vec.set_attr("opponents", provider, indices=i)

    # SB3 VecEnv.reset takes no seed by contract; the per-env providers carry the seed and reset
    # re-primes each env, so opponent + first-episode reproducibility comes through the providers.
    obs = vec.reset()
    outcomes: list[str] = []
    while len(outcomes) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = vec.step(actions)
        for done, info in zip(dones, infos, strict=True):
            if not done:
                continue
            if len(outcomes) >= n_episodes:
                break
            # On a vec auto-reset the step's info is the TERMINAL step's info (outcome preserved);
            # a missing outcome (truncation / lost connection) is a draw.
            outcomes.append(info.get("outcome", DRAW))
    return win_rate(outcomes)


def evaluate_winrate(
    model: BaseAlgorithm,
    eval_env,
    opponents: Sequence[str] = DEFAULT_ROSTER,
    n_episodes: int = 100,
    seed: int | None = None,
) -> dict[str, float]:
    """Play ``n_episodes`` greedy episodes vs. EACH opponent selector; return per-selector win-rate.

    For EACH selector in ``opponents`` (drawn from ``DEFAULT_ROSTER`` by default), ``n_episodes``
    deterministic episodes (``model.predict(obs, deterministic=True)``) are played and the win-rate
    is tallied. The per-episode result is taken from ``info["outcome"]`` on the done step
    (``TankEnv`` surfaces ``"win"`` / ``"loss"`` / ``"draw"`` on ``terminated`` only); a
    done-without-outcome episode (time-limit / lost connection) is recorded as a ``"draw"`` (never a
    win) so it cannot inflate the metric.

    PARALLEL across M eval envs (:func:`_eval_opponent_vec`): the eval env is a VEC env of M
    :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper`-wrapped envs. For each selector every env's
    ``opponents`` provider is re-pinned to that ONE opponent (``vec.set_attr`` — fanned to each
    worker), then the M envs are stepped TOGETHER (one batched ``predict`` + one ``vec.step`` per
    iteration) with SB3 auto-resetting finished envs, so ``n_episodes`` stream across the lanes. At
    ``M > 1`` (a ``SubprocVecEnv``) the workers step concurrently across processes — true parallel
    eval; the win-rate MATH is unchanged (M==1 reproduces the sequential result for the same
    scripted outcomes). ``model`` and the env are touched lazily / only here; the pure ``win_rate``
    above stays import-light.

    Args:
        model: a built SB3 algorithm (e.g. a loaded ``PPO``) with ``.predict``.
        eval_env: the eval env(s). One of: a VEC env of M ``SelfPlayWrapper``-wrapped envs (the LIVE
            path; used as-is); a single raw env; or a list/tuple of M raw envs (each auto-wrapped in
            a throwaway ``DummyVecEnv`` of ``SelfPlayWrapper``s — the convenience / unit-test seam).
        opponents: a sequence of ``agents`` selector strings (e.g. ``("noop", "random", ...)``).
        n_episodes: number of greedy episodes per opponent (distributed across the M envs).
        seed: optional seed threaded into each opponent's ``OpponentProvider`` (agent +
            sampling reproducibility).

    Returns:
        ``{selector: win_rate}`` in the order of ``opponents`` (a regular insertion-ordered dict).
    """
    vec, _owned = _as_eval_vec(eval_env)
    return {
        selector: _eval_opponent_vec(model, vec, selector, n_episodes, seed)
        for selector in opponents
    }


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
