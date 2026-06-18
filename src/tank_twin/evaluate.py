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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only imports: keep the pure helper torch/sb3-free at runtime
    import gymnasium
    from stable_baselines3.common.base_class import BaseAlgorithm

__all__ = ["WIN", "LOSS", "DRAW", "win_rate", "evaluate_winrate", "main"]

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


def main(argv: Sequence[str] | None = None) -> float:
    """CLI entry point: load a model, launch Unity, print + return the win-rate.

    Imports torch / sb3 / the env lazily so ``import tank_twin.evaluate`` (and the pure
    ``win_rate`` unit test) stays cheap and torch-free.
    """
    # Lazy imports: heavy deps only when actually running an evaluation.
    from stable_baselines3 import PPO

    from tank_twin.env import TankEnv

    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    model = PPO.load(str(args.model_path), device=device)
    env = TankEnv(game_path=str(args.game_path))
    try:
        rate = evaluate_winrate(model, env, n_episodes=args.episodes, seed=args.seed)
    finally:
        env.close()

    print(f"win_rate={rate:.4f} over {args.episodes} episodes (model={args.model_path.name})")
    return rate


if __name__ == "__main__":
    main()
