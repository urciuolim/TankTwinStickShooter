"""ELO rating math (pure, stdlib-only).

The rating logic lives in ``rl`` (not ``core``): self-play opponent management and the ELO
ladder are a WRAPPER over the trainer. Pure ``math`` — no sb3 / gym / torch / numpy, so the
rating arithmetic is unit-testable on its own.
"""

import math


def elo_prob(elo1, elo2):
    """Expected win probability for ``elo1`` against ``elo2`` (logistic, base 10, /400)."""
    return 1.0 / (1.0 + math.pow(10.0, (elo2 - elo1) / 400.0))


def elo_change(elo_a, elo_b, K, a_win_rate):
    """Rounded ELO deltas for A and B given A's realized win rate.

    Returns ``(round(a_delta), round(b_delta))``. Rounding is applied PER SIDE, so the pair is
    not guaranteed to sum to zero.
    """
    a_prob = elo_prob(elo_a, elo_b)
    b_prob = elo_prob(elo_b, elo_a)

    a_elo_change = K * (a_win_rate - a_prob)
    b_elo_change = K * ((1 - a_win_rate) - b_prob)

    return round(a_elo_change), round(b_elo_change)
