"""Opponent selection / PFSP weighting (pure functions, numpy-only).

Migrated verbatim from module-level helpers in the 2021
``PythonScripts/tank_env.py`` (M0 task A6). Behavior is byte-identical to the
original; ``tank_env.py`` now imports these back so ``TankEnv`` selection logic
is unchanged, and ``ai_matchmaker.py`` / ``human_matchmaking.py`` re-import them
through that shim.

Uses ``numpy.random.choice`` (the legacy free function), so any caller that
seeds via ``numpy.random.seed(...)`` gets identical draws. No sb3/gym/torch.
"""

import math

from numpy.random import choice


def choice_with_normalization(elements, weights):
    """Pick one of ``elements`` with probability proportional to ``weights``.

    All-zero weights fall back to a uniform draw over ``elements``.
    """
    if sum(weights) == 0:
        return choice(elements, p=[1 / len(elements) for _ in elements])
    return choice(elements, p=[x / sum(weights) for x in weights])


def weight_func(x, D):
    """Return 1.0 if ``x`` is within ``D`` of 0, else ``1/(x/D)^2`` (capped at 1.0)."""
    if x == 0:
        return 1.0
    x = 1.0 / math.pow(x / D, 2)
    return x if x <= 1.0 else 1.0


def elo_based_choice(opponent_elos, center_elo, D):
    """Choose an opponent index by ELO proximity to ``center_elo`` (width ``D``)."""
    weights = [weight_func(elo - center_elo, D) for elo in opponent_elos]
    return choice_with_normalization([i for i in range(len(opponent_elos))], weights)
