"""Greedy win-rate evaluation of a trained PPO agent, per opponent and (optionally) per map.

The M1 metric is win-rate: the fraction of greedy (deterministic) episodes the trained agent
wins. Unlike a fixed-opponent eval, this evaluates the policy against EACH selector in a roster
SEPARATELY (one fixed opponent per phase, via the self-play seam), so the breakdown shows where
the policy is strong / weak across the scripted family. When a map rotation is supplied, the eval
additionally covers EACH rotation arena: the per-opponent episode budget is spread
deterministically across the maps and every (opponent, map) cell is played as its own pinned
phase, yielding per-cell win/episode counts that pool into per-opponent, per-map, and overall
marginals which reconcile by construction.

This module ships:

* :func:`win_rate` — the PURE counting helper (``Sequence[str] -> float``). No torch / no env,
  so the counting logic is unit-testable on its own (stdlib only).
* :func:`episode_spread` / :func:`map_short_name` — PURE helpers: the deterministic floor/ceil
  split of the per-opponent episode budget across the eval maps, and the arena-target -> short
  TensorBoard tag name derivation.
* :func:`evaluate_winrate` — the eval driver. With ``maps=None`` (the default) it plays
  ``n_episodes`` greedy episodes per opponent selector and returns ``{selector: win_rate}``
  (insertion order matches the ``opponents`` argument) — exactly the single-arena behavior. With
  a map rotation it iterates (opponent, map) phases and returns per-cell ``(wins, episodes)``
  counts instead (see the function docstring). It reads the per-episode result from
  ``info["outcome"]`` (surfaced by ``TankEnv.step`` on TERMINAL steps only), NOT from the reward
  sign — the win signal is unambiguous regardless of reward shaping. The eval is PARALLELIZED
  across ``M`` eval envs: each phase's episodes are DISTRIBUTED across the envs and played
  concurrently (the M lanes step together, batched through ``model.predict``). ``M == 1`` (a
  single env / ``DummyVecEnv``) is the sequential path; ``M > 1`` gives the SAME rates for the
  same scripted outcomes (only faster — the win-rate MATH is unchanged). The eval env(s) may be
  passed as a single raw ``TankEnv``, a list of raw envs, or a vec env of ``M`` envs.
* :func:`pool_by_opponent` / :func:`pool_by_map` / :func:`pooled_win_rate` — PURE aggregation
  over the per-cell counts. All three POOL wins/episodes (never average rates), so the marginals
  and the overall reconcile exactly even under an uneven floor/ceil episode spread.
* :func:`overall_win_rate` / :func:`format_per_opponent_line` / :func:`format_per_map_line` —
  PURE helpers over a per-opponent / per-map breakdown (the overall mean and the final-summary
  lines).

Only :func:`evaluate_winrate` touches sb3 / the env (lazily / type-only); the pure counting,
spread, and aggregation helpers stay import-light (stdlib only) so the torch-free unit suite can
exercise them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:  # type-only imports: keep the pure helpers torch/sb3-free at runtime
    from stable_baselines3.common.base_class import BaseAlgorithm

__all__ = [
    "WIN",
    "LOSS",
    "DRAW",
    "win_rate",
    "episode_spread",
    "map_short_name",
    "evaluate_winrate",
    "pool_by_opponent",
    "pool_by_map",
    "pooled_win_rate",
    "overall_win_rate",
    "format_per_opponent_line",
    "format_per_map_line",
]

# Outcome tokens — mirror ``TankEnv.step``'s ``info["outcome"]`` values.
WIN = "win"
LOSS = "loss"
DRAW = "draw"

# Per-cell eval counts: (opponent selector, arena target) -> (wins, episodes). The insertion
# order is opponent-major (opponents outer, maps in rotation order inner).
CellCounts = dict[tuple[str, str], tuple[int, int]]


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


def episode_spread(n_episodes: int, n_maps: int) -> list[int]:
    """Deterministically spread ``n_episodes`` across ``n_maps`` maps (PURE; sums to exactly N).

    The first ``n_episodes % n_maps`` maps (in rotation order) get ``n_episodes // n_maps + 1``
    episodes; the rest get ``n_episodes // n_maps``. So the per-opponent episode budget is
    unchanged by the map rotation (the quotas sum to exactly ``n_episodes``) and the spread is
    reproducible (no randomness). With ``n_episodes < n_maps`` the first ``n_episodes`` maps get
    one episode each and the rest get zero.

    Args:
        n_episodes: the per-opponent episode budget (``>= 0``).
        n_maps: the number of eval maps (``> 0``).

    Returns:
        A list of ``n_maps`` quotas in rotation order, summing to ``n_episodes``.
    """
    if n_maps <= 0:
        raise ValueError(f"n_maps must be > 0, got {n_maps}")
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0, got {n_episodes}")
    base, extra = divmod(n_episodes, n_maps)
    return [base + 1 if i < extra else base for i in range(n_maps)]


def map_short_name(target: str) -> str:
    """The short map name for TensorBoard tags: the arena target's path stem (PURE).

    ``"Arenas/center_block.json"`` -> ``"center_block"``; a bare ``"center_block"`` or a
    ``"center_block.json"`` without a directory also yield ``"center_block"``. Arena targets are
    canonical forward-slash strings, so the split is POSIX regardless of host OS.
    """
    return PurePosixPath(target).stem


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
    # A throwaway placeholder provider; evaluate_winrate re-pins each env per phase via set_attr.
    placeholder = OpponentProvider.from_roster([DEFAULT_ROSTER[0]], strategy="round_robin")
    # default-arg capture so each closure binds its OWN raw env (not the loop variable).
    fns = [(lambda r=r: SelfPlayWrapper(r, placeholder)) for r in raw_envs]
    return DummyVecEnv(fns), True


def _eval_phase_vec(
    model, vec, selector: str, n_episodes: int, seed, arena: str | None = None
) -> list[str]:
    """Play ``n_episodes`` greedy episodes for ONE pinned (opponent[, arena]) phase; outcomes.

    Pins every env's :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` to a single-selector
    ``OpponentProvider`` for ``selector`` (``vec.set_attr("opponents", ...)`` — fanned to every
    worker; the provider is picklable), then drives the vec: ``reset`` once, then repeatedly
    ``predict(obs, deterministic=True)`` + ``vec.step(actions)``. The M envs step TOGETHER each
    iteration (a ``SubprocVecEnv`` runs its workers concurrently across processes -> true parallel
    eval), and SB3's vec AUTO-RESETS any finished env, so episodes stream across the lanes. The
    first ``n_episodes`` outcomes observed (env-index order within a step) are collected and
    returned — the tally is INDEPENDENT of M (M==1 reproduces the sequential result for the same
    scripted outcomes).

    Arena pinning (``arena`` not ``None``): each env is ALSO pinned to a single-map
    :class:`~pop_trainer.rl.selfplay.MapProvider` holding just ``arena``
    (``vec.set_attr("maps", ...)``), so EVERY reset in the phase — the initial one and each vec
    auto-reset — injects ``reset(options={"switch_arena": arena})`` through the wrapper. That is
    the SAME reset-time channel training rotation uses: the build handles ``switch_arena`` only
    while ``!ingame``, and a reset-option switch lands exactly there, so the deliberate pinned
    switching is safe where the historical UNCONTROLLED eval rotation (an arbitrary arena per
    auto-reset, unaccounted by the eval math) was not. With ``arena is None`` the ``maps`` attr is
    NEVER touched and plain ``reset()``s send no ``switch_arena`` — byte-identical to the
    single-arena eval.

    The per-episode outcome is ``info["outcome"]`` on the step where ``done`` is set (``TankEnv``
    surfaces it on ``terminated`` only); a done WITHOUT an ``"outcome"`` (truncation / time-limit /
    lost connection) is a DRAW, never a win. The per-env providers carry ``seed`` for opponent +
    first-episode reproducibility.
    """
    from pop_trainer.rl.selfplay import MapProvider, OpponentProvider

    # Pin EACH env to its OWN single-selector provider (a fresh provider per env so their
    # RNG/index are independent). SB3 set_attr sets ONE value across the given indices, so set
    # per-env by index (indices=i fans to worker i for a SubprocVecEnv; in-process for Dummy).
    for i in range(vec.num_envs):
        provider = OpponentProvider.from_roster([selector], strategy="round_robin", seed=seed)
        vec.set_attr("opponents", provider, indices=i)
        if arena is not None:
            # A single-map round_robin provider deterministically yields `arena` at every reset,
            # so the whole phase plays one pinned arena. Never touched on the no-rotation path.
            vec.set_attr("maps", MapProvider([arena], strategy="round_robin"), indices=i)

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
    return outcomes


def evaluate_winrate(
    model: BaseAlgorithm,
    eval_env,
    opponents: Sequence[str] = DEFAULT_ROSTER,
    n_episodes: int = 100,
    seed: int | None = None,
    maps: Sequence[str] | None = None,
) -> dict[str, float] | CellCounts:
    """Play ``n_episodes`` greedy episodes vs. EACH opponent (spread across ``maps`` if given).

    For EACH selector in ``opponents`` (drawn from ``DEFAULT_ROSTER`` by default), ``n_episodes``
    deterministic episodes (``model.predict(obs, deterministic=True)``) are played. The
    per-episode result is taken from ``info["outcome"]`` on the done step (``TankEnv`` surfaces
    ``"win"`` / ``"loss"`` / ``"draw"`` on ``terminated`` only); a done-without-outcome episode
    (time-limit / lost connection) is recorded as a ``"draw"`` (never a win) so it cannot inflate
    the metric.

    Map coverage (``maps`` not ``None``): the eval covers EVERY arena in ``maps``. Each
    opponent's ``n_episodes`` budget is spread across the maps by :func:`episode_spread`
    (deterministic floor/ceil, summing to exactly ``n_episodes`` — the eval cost is unchanged by
    the rotation), and each (opponent, map) cell runs as its own phase with the opponent AND a
    single-map provider pinned per env (:func:`_eval_phase_vec` — deliberate, phase-scoped
    ``switch_arena`` injected only at reset). A zero-quota cell (``n_episodes < len(maps)``) is
    skipped entirely (no pinning, no reset) and recorded as ``(0, 0)``. With ``maps=None`` (the
    default) the behavior is exactly the single-arena eval: no ``switch_arena`` is ever sent and
    the per-env ``maps`` attribute is never touched.

    PARALLEL across M eval envs (:func:`_eval_phase_vec`): the eval env is a VEC env of M
    :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper`-wrapped envs. For each phase every env is
    re-pinned (``vec.set_attr`` — fanned to each worker), then the M envs are stepped TOGETHER
    (one batched ``predict`` + one ``vec.step`` per iteration) with SB3 auto-resetting finished
    envs, so the phase's episodes stream across the lanes. At ``M > 1`` (a ``SubprocVecEnv``) the
    workers step concurrently across processes — true parallel eval; the win-rate MATH is
    unchanged (M==1 reproduces the sequential result for the same scripted outcomes). ``model``
    and the env are touched lazily / only here; the pure helpers above stay import-light.

    Args:
        model: a built SB3 algorithm (e.g. a loaded ``PPO``) with ``.predict``.
        eval_env: the eval env(s). One of: a VEC env of M ``SelfPlayWrapper``-wrapped envs (the LIVE
            path; used as-is); a single raw env; or a list/tuple of M raw envs (each auto-wrapped in
            a throwaway ``DummyVecEnv`` of ``SelfPlayWrapper``s — the convenience / unit-test seam).
        opponents: a sequence of ``agents`` selector strings (e.g. ``("noop", "random", ...)``).
        n_episodes: number of greedy episodes per opponent (distributed across the M envs, and —
            with ``maps`` — spread across the eval maps by :func:`episode_spread`).
        seed: optional seed threaded into each opponent's ``OpponentProvider`` (agent +
            sampling reproducibility).
        maps: optional eval map rotation (arena-target strings, e.g. ``"Arenas/empty.json"``).
            ``None`` = single-arena eval (today's exact behavior); non-empty = per-cell eval.

    Returns:
        With ``maps=None``: ``{selector: win_rate}`` in the order of ``opponents`` (a regular
        insertion-ordered dict) — unchanged for existing callers. With ``maps``:
        ``{(selector, arena): (wins, episodes)}`` per-cell counts, opponent-major (opponents
        outer, maps in rotation order inner) — pool with :func:`pool_by_opponent` /
        :func:`pool_by_map` / :func:`pooled_win_rate`.
    """
    vec, _owned = _as_eval_vec(eval_env)
    if maps is None:
        return {
            selector: win_rate(_eval_phase_vec(model, vec, selector, n_episodes, seed))
            for selector in opponents
        }
    rotation = list(maps)
    if not rotation:
        raise ValueError("maps must be non-empty (or None for the single-arena eval)")
    quotas = episode_spread(n_episodes, len(rotation))
    cells: CellCounts = {}
    for selector in opponents:
        for arena, quota in zip(rotation, quotas, strict=True):
            if quota == 0:
                cells[(selector, arena)] = (0, 0)
                continue
            outcomes = _eval_phase_vec(model, vec, selector, quota, seed, arena=arena)
            wins = sum(1 for o in outcomes if o == WIN)
            cells[(selector, arena)] = (wins, len(outcomes))
    return cells


def _pool(pairs: Iterable[tuple[str, tuple[int, int]]]) -> dict[str, float]:
    """Pool ``(key, (wins, episodes))`` pairs into ``{key: wins / episodes}`` (PURE).

    Wins and episodes are SUMMED per key before dividing (pooled, never a mean of means), so
    grouped marginals reconcile with the overall by construction. Keys keep first-appearance
    order. A key whose pooled episode count is zero (all its cells were zero-quota) is OMITTED —
    it was not evaluated, so reporting a rate for it would be misleading.
    """
    wins: dict[str, int] = {}
    episodes: dict[str, int] = {}
    for key, (w, e) in pairs:
        wins[key] = wins.get(key, 0) + w
        episodes[key] = episodes.get(key, 0) + e
    return {key: wins[key] / episodes[key] for key in wins if episodes[key] > 0}


def pool_by_opponent(cells: Mapping[tuple[str, str], tuple[int, int]]) -> dict[str, float]:
    """Per-opponent win-rate marginals pooled over maps from per-cell counts (PURE).

    ``{selector: total wins / total episodes}`` across that selector's cells, in the cells'
    opponent order. Pooled from counts (see :func:`_pool`), so with every opponent playing the
    same total budget the unweighted mean of these marginals equals :func:`pooled_win_rate`.
    """
    return _pool(((selector, count) for (selector, _arena), count in cells.items()))


def pool_by_map(cells: Mapping[tuple[str, str], tuple[int, int]]) -> dict[str, float]:
    """Per-map win-rate marginals pooled over opponents from per-cell counts (PURE).

    ``{arena_target: total wins / total episodes}`` across that arena's cells, in rotation order
    (first appearance). Keys are the FULL arena targets; derive TensorBoard tag names with
    :func:`map_short_name`. Maps with zero evaluated episodes are omitted (see :func:`_pool`).
    """
    return _pool(((arena, count) for (_selector, arena), count in cells.items()))


def pooled_win_rate(cells: Mapping[tuple[str, str], tuple[int, int]]) -> float:
    """The overall pooled win-rate: total wins / total episodes over ALL cells (PURE).

    Zero total episodes returns ``0.0`` (mirroring :func:`win_rate` on an empty input). Because
    it pools raw counts, it reconciles exactly with the weighted sums of both
    :func:`pool_by_opponent` and :func:`pool_by_map` even under an uneven episode spread.
    """
    total_wins = sum(w for w, _e in cells.values())
    total_episodes = sum(e for _w, e in cells.values())
    if total_episodes == 0:
        return 0.0
    return total_wins / total_episodes


def _as_pairs(
    per_group: Mapping[str, float] | Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize a breakdown input to an ordered list of ``(name, win_rate)`` pairs (PURE).

    Accepts either an ordered mapping (e.g. a ``dict``, insertion order preserved) or an
    already-ordered sequence of ``(name, rate)`` tuples. Shared by :func:`overall_win_rate` and
    the formatters so all agree on ordering.
    """
    if isinstance(per_group, Mapping):
        return list(per_group.items())
    return [(name, rate) for name, rate in per_group]


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


def format_per_opponent_line(
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


def format_per_map_line(
    per_map: Mapping[str, float] | Sequence[tuple[str, float]],
) -> str:
    """Format the per-map win-rate breakdown as a single human-readable line (PURE).

    Produces ``"PER-MAP: <short>: <rate> | ..."`` with each rate to two decimals, in the input's
    order; names are shortened via :func:`map_short_name` (full arena targets and bare names both
    work). No OVERALL cell — the overall belongs to :func:`format_per_opponent_line` (an
    unweighted mean over maps would not pool correctly under an uneven episode spread). An empty
    input yields ``"PER-MAP: (none)"`` so the formatter never raises on a degenerate breakdown.
    """
    pairs = _as_pairs(per_map)
    if not pairs:
        return "PER-MAP: (none)"
    return "PER-MAP: " + " | ".join(f"{map_short_name(name)}: {rate:.2f}" for name, rate in pairs)
