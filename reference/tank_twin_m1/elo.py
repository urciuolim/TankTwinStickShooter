"""ELO rating math (pure, stdlib-only).

Migrated verbatim from the 2021 ``PythonScripts/elo.py`` (M0 task A6). Behavior
is byte-identical to the original; ``PythonScripts/elo.py`` is now a shim that
re-exports these so legacy importers (``consolidate.py``, ``ai_matchmaker.py``,
``replace.py``, ...) keep working unchanged.

No sb3/gym/torch imports — pure ``math``.
"""

import math


def elo_prob(elo1, elo2):
    """Expected win probability for ``elo1`` against ``elo2`` (logistic, base 10, /400)."""
    return 1.0 / (1.0 + math.pow(10.0, (elo2 - elo1) / 400.0))


def elo_change(elo_a, elo_b, K, a_win_rate):
    """Rounded zero-ish-sum ELO deltas for A and B given A's realized win rate.

    Returns ``(round(a_delta), round(b_delta))``. Rounding is applied per side
    (matching the 2021 behavior), so the pair is not guaranteed to sum to zero.
    """
    a_prob = elo_prob(elo_a, elo_b)
    b_prob = elo_prob(elo_b, elo_a)

    a_elo_change = K * (a_win_rate - a_prob)
    b_elo_change = K * ((1 - a_win_rate) - b_prob)

    return round(a_elo_change), round(b_elo_change)
