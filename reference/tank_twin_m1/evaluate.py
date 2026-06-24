"""Greedy win-rate evaluation of a trained PPO agent vs. the random opponent (M1, task 3.2).

The M1 metric is win-rate: the fraction of greedy (deterministic) episodes the trained
agent wins against the random opponent. This module ships:

* :func:`win_rate` — the PURE counting helper (``list[str] -> float``). No torch / no env,
  so the counting logic is unit-testable on its own (see ``tests/test_evaluate.py``).
* :func:`evaluate_winrate` — plays ``n_episodes`` greedy episodes through a built
  ``TankEnv`` with a built SB3 model and returns the win-rate. It reads the per-episode
  result from ``info["outcome"]`` (surfaced by ``TankEnv.step`` on terminal steps), NOT
  from the reward sign — the win signal is unambiguous regardless of reward shaping.
* a ``python -m tank_twin.evaluate`` CLI that loads a saved ``.zip`` model, launches a
  real Unity build, and prints the win-rate over K episodes (the Wave-4 GPU smoke uses
  this with K ~= 100).

Only this module + the trainer import torch / sb3 in the eval path; the pure ``win_rate``
helper stays import-light (stdlib only) so the torch-free unit suite can exercise it.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from tank_twin.maps import (
    ALL_MAPS_SENTINEL,  # noqa: F401  (re-exported for callers/tests symmetry with train)
    DEFAULT_MAPS_DIR,  # noqa: F401  (re-exported for callers/tests symmetry with train)
    _resolve_map_rotation,
)

if TYPE_CHECKING:  # type-only imports: keep the pure helper torch/sb3-free at runtime
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
    "main",
]

# Outcome tokens — mirror ``TankEnv.step``'s ``info["outcome"]`` values.
WIN = "win"
LOSS = "loss"
DRAW = "draw"


def win_rate(outcomes: Sequence[str]) -> float:
    """Fraction of episodes won, in ``[0, 1]`` (PURE; no torch / no env).

    Counts ``"win"`` entries over the total number of outcomes. Draws and losses count
    toward the denominator but not the numerator (win-rate, not win-or-draw-rate). An
    empty sequence returns ``0.0`` (no episodes -> no wins). Unknown tokens are tolerated
    (they count toward the denominator only), so a stray ``info`` value never raises here.

    Args:
        outcomes: per-episode result tokens, each typically one of ``"win"`` / ``"loss"``
            / ``"draw"`` (see :data:`WIN` / :data:`LOSS` / :data:`DRAW`).

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
    env: gymnasium.Env,
    n_episodes: int = 100,
    seed: int | None = None,
) -> float:
    """Play ``n_episodes`` greedy episodes of ``model`` in ``env`` and return win-rate.

    Each episode is run deterministically (``model.predict(obs, deterministic=True)``)
    until ``terminated or truncated``. The per-episode result is taken from
    ``info["outcome"]`` on the terminal step (``TankEnv`` surfaces ``"win"`` / ``"loss"``
    / ``"draw"``); a truncated-without-outcome episode (time-limit / lost connection) is
    recorded as a ``"draw"`` (not a win) so it never inflates the metric.

    Reproducibility: the FIRST ``reset`` is seeded with ``seed`` so the random opponent's
    draws are deterministic across runs; subsequent episodes advance the env's RNG (no
    re-seed) so each episode differs.

    Args:
        model: a built SB3 algorithm (e.g. a loaded ``PPO``) with ``.predict``.
        env: a built ``TankEnv`` (already connected to Unity or a fake transport).
        n_episodes: number of greedy episodes to play.
        seed: optional seed for the first ``reset`` (opponent reproducibility).

    Returns:
        Win-rate in ``[0, 1]`` over the played episodes.
    """
    outcomes: list[str] = []
    for episode in range(n_episodes):
        reset_seed = seed if (episode == 0 and seed is not None) else None
        obs, _info = env.reset(seed=reset_seed)
        outcome = DRAW
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated:
                # Decided game: trust the env's unambiguous outcome token.
                outcome = info.get("outcome", DRAW)
        outcomes.append(outcome)
    return win_rate(outcomes)


def _as_pairs(
    per_map: Mapping[str, float] | Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize the per-map input to an ordered list of ``(map_name, win_rate)`` pairs.

    Accepts either an ordered mapping (e.g. a ``dict``, insertion order preserved) or an
    already-ordered sequence of ``(name, rate)`` tuples. PURE: no torch / no env. Shared by
    :func:`overall_win_rate` and :func:`format_per_map_table` so both agree on ordering.
    """
    if isinstance(per_map, Mapping):
        return list(per_map.items())
    return [(name, rate) for name, rate in per_map]


def overall_win_rate(
    per_map: Mapping[str, float] | Sequence[tuple[str, float]],
) -> float:
    """Overall win-rate across maps: the UNWEIGHTED MEAN of the per-map rates (PURE).

    Returns the simple arithmetic mean of the per-map win-rates. This equals the POOLED
    win-rate (total wins / total episodes) BECAUSE the evaluator plays the SAME number of
    greedy episodes (``--episodes``) on every map — with a uniform episodes-per-map count
    the mean of the rates is identical to the pooled rate. If episodes-per-map ever became
    non-uniform this would diverge from the pooled rate and would need re-weighting; today
    it does not, so the simple mean is correct and is what the CTO breakdown reports.

    No torch / no env / no Unity — unit-testable on a plain mapping. An empty input returns
    ``0.0`` (no maps -> no measured win-rate), mirroring :func:`win_rate` on an empty input.

    Args:
        per_map: ordered ``{map_name: win_rate}`` mapping OR a sequence of ``(name, rate)``
            pairs. Each ``win_rate`` is a float in ``[0, 1]`` (e.g. from
            :func:`evaluate_winrate`).

    Returns:
        The mean per-map win-rate in ``[0, 1]`` (``0.0`` for an empty input).
    """
    pairs = _as_pairs(per_map)
    if not pairs:
        return 0.0
    return sum(rate for _name, rate in pairs) / len(pairs)


def format_per_map_table(
    per_map: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    episodes: int | None = None,
) -> str:
    """Format the per-map win-rate breakdown as a single human-readable line (PURE).

    Produces ``"<map>: <rate> | <map>: <rate> | ... | OVERALL: <overall>"`` with each rate
    formatted to two decimals, in the input's order. When ``episodes`` is given an episode
    accounting suffix ``" (N maps x E episodes)"`` is appended (the CTO diagnostic header).
    The OVERALL value is :func:`overall_win_rate` (the uniform-episodes mean == pooled rate).

    PURE: no torch / no env — unit-testable. An empty input yields just the OVERALL token
    (``"OVERALL: 0.00"``) so the formatter never raises on a degenerate breakdown.

    Args:
        per_map: ordered ``{map_name: win_rate}`` mapping OR ``(name, rate)`` pairs.
        episodes: greedy episodes played PER MAP (uniform); appended as the accounting
            suffix when given. ``None`` omits the suffix.

    Returns:
        The breakdown line, e.g.
        ``"custom1_2021: 0.30 | empty: 0.55 | OVERALL: 0.42 (2 maps x 100 episodes)"``.
    """
    pairs = _as_pairs(per_map)
    overall = overall_win_rate(pairs)
    cells = [f"{name}: {rate:.2f}" for name, rate in pairs]
    cells.append(f"OVERALL: {overall:.2f}")
    line = " | ".join(cells)
    if episodes is not None:
        line += f" ({len(pairs)} maps x {episodes} episodes)"
    return line


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tank_twin.evaluate",
        description="Evaluate a trained PPO model's win-rate vs. the random opponent.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved SB3 model (.zip) to evaluate.",
    )
    parser.add_argument(
        "--game-path",
        type=Path,
        required=True,
        help="Path to the Unity build executable to launch and play against.",
    )
    # --- map selection (mirrors train.py exactly) -------------------------------------
    # --config / --map (dest config_path): pin ONE map (the config single-source). --maps /
    # --map-rotation (dest maps, nargs="*"): evaluate a SET of maps with a PER-MAP breakdown.
    # Both ABSENT -> today's default-arena behavior (custom1, single number). Eval pins each
    # map via config_path on its OWN single-map env (NO switch_arena during eval).
    parser.add_argument(
        "--config",
        "--map",
        dest="config_path",
        type=Path,
        default=None,
        help=(
            "External game config.json: pin a SINGLE map (the config single-source - "
            "forwarded to the build AND read for the obs grid; --map is an alias). "
            "Default: the build's StreamingAssets config (custom1)."
        ),
    )
    parser.add_argument(
        "--maps",
        "--map-rotation",
        dest="maps",
        nargs="*",
        default=None,
        metavar="MAP_CONFIG",
        help=(
            "Evaluate over a SET of map configs with a PER-MAP win-rate breakdown (each map "
            "gets --episodes greedy episodes on its OWN single-map env; NO switch_arena). "
            "No value -> the shipped 10 exp-configs/maps; a directory -> its *.json configs "
            "(sorted); a list of paths -> that order. Absent -> default arena (single number)."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of greedy episodes to play (the M1 GPU smoke uses ~100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the first reset (opponent reproducibility).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for inference (auto -> cuda if available, else cpu).",
    )
    return parser


def _resolve_device(device: str) -> str:
    """Map ``auto`` -> ``cuda`` when available, else ``cpu``; pass others through."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _evaluate_per_map(
    model: BaseAlgorithm,
    *,
    game_path: str | Path,
    map_configs: Sequence[Path],
    n_episodes: int,
    seed: int | None,
    base_game_port: int = 50000,
) -> list[tuple[str, float]]:
    """Evaluate ``model`` on EACH map separately and return ordered ``(name, rate)`` pairs.

    Robust per-map env handling (avoids the rollout-boundary ``switch_arena`` desync): every
    map is evaluated on its OWN single-map ``TankEnv(game_path=..., config_path=<that map>)``
    — single map pinned via the config single-source, NO ``map_rotation``, so NO mid-eval
    ``switch_arena`` is ever sent. Each map: build relaunch -> ``evaluate_winrate`` for
    ``n_episodes`` greedy episodes -> ``env.close()`` (reaps the build cleanly) -> next map.

    A FRESH ``game_port`` is used per map (``base_game_port + i``; the env derives its own bind
    port as ``game_port + 1``) so sequential relaunches do not collide on the Windows
    TIME_WAIT quirk where the just-closed port lingers. The map NAME is the config file stem
    (e.g. ``custom1_2021.json`` -> ``custom1_2021``). The seed is passed to each map's first
    ``reset`` so the random opponent's draws are reproducible per map.

    Imports ``TankEnv`` lazily (the caller, :func:`main`, already imported torch/sb3) so this
    module stays torch-free at import time. Returns ordered pairs (input map order preserved)
    for :func:`overall_win_rate` / :func:`format_per_map_table`.
    """
    from tank_twin.env import TankEnv

    results: list[tuple[str, float]] = []
    for index, map_config in enumerate(map_configs):
        map_name = Path(map_config).stem
        env = TankEnv(
            game_path=str(game_path),
            config_path=map_config,
            game_port=base_game_port + index,
        )
        try:
            rate = evaluate_winrate(model, env, n_episodes=n_episodes, seed=seed)
        finally:
            env.close()
        results.append((map_name, rate))
    return results


def main(argv: Sequence[str] | None = None) -> float:
    """CLI entry point: load a model, launch Unity, print + return the win-rate.

    Three modes (mirrors the trainer's map interface):

    * BOTH ``--config``/``--map`` AND ``--maps``/``--map-rotation`` ABSENT -> today's
      behavior: evaluate the build's default arena (custom1) and print a SINGLE win-rate.
      Returns that rate.
    * ``--maps``/``--map-rotation`` (resolves to a map list) -> evaluate EACH map separately
      (``--episodes`` greedy episodes per map, each on its OWN single-map env, NO
      ``switch_arena``) and print a PER-MAP breakdown table plus the OVERALL average. Returns
      the OVERALL average.
    * ``--config``/``--map`` (single map) -> pin that one map via the config single-source and
      evaluate it (per-map path, one-entry list). Returns that map's rate.

    Imports torch / sb3 / the env lazily so ``import tank_twin.evaluate`` (and the pure
    ``win_rate`` / aggregation / formatter unit tests) stays cheap and torch-free.
    """
    # Lazy imports: heavy deps only when actually running an evaluation.
    from stable_baselines3 import PPO

    from tank_twin.env import TankEnv

    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    model = PPO.load(str(args.model_path), device=device)

    # Resolve the map selection. --maps/--map-rotation wins (a SET -> per-map breakdown);
    # else a single --config/--map (pin one map); else the default arena (single number).
    resolved_rotation = _resolve_map_rotation(args.maps)
    if resolved_rotation is not None:
        map_configs: list[Path] = resolved_rotation
    elif args.config_path is not None:
        map_configs = [Path(args.config_path)]
    else:
        map_configs = []

    if map_configs:
        per_map = _evaluate_per_map(
            model,
            game_path=args.game_path,
            map_configs=map_configs,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(format_per_map_table(per_map, episodes=args.episodes))
        return overall_win_rate(per_map)

    # Default arena (no map flags): today's single-number behavior, unchanged.
    env = TankEnv(game_path=str(args.game_path))
    try:
        rate = evaluate_winrate(model, env, n_episodes=args.episodes, seed=args.seed)
    finally:
        env.close()

    print(f"win_rate={rate:.4f} over {args.episodes} episodes (model={args.model_path.name})")
    return rate


if __name__ == "__main__":
    main()
