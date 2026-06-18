"""State transforms for the opponent's perspective (pure functions, numpy-only).

Migrated verbatim from the 2021 ``PythonScripts/tank_env.py`` (M0 task A6):
``flip_state`` (the image-observation R<->B channel swap) and the raw 52-float
state split that hands the opponent its own first-person view. Behavior is
byte-identical; ``tank_env.py`` imports these back so ``TankEnv`` is unchanged.

dtype is preserved EXACTLY (``flip_state`` allocates with ``state.dtype``; the
split is a plain ``np.concatenate`` so it inherits the input dtype). This is the
RL observation contract — do not change the math without sign-off. No sb3/gym/torch.
"""

import numpy as np


def flip_state(state):
    """Swap the R (self) and B (opponent) channels of an image state; keep G (walls).

    Returns a new array of the same shape and dtype. ``flip_state(flip_state(s))``
    is identity for the R/B channels.
    """
    R, G, B = (0, 1, 2)
    new_state = np.zeros(state.shape, dtype=state.dtype)
    new_state[:, :, R] = state[:, :, B].copy()
    new_state[:, :, G] = state[:, :, G].copy()
    new_state[:, :, B] = state[:, :, R].copy()
    return new_state


def split_state_for_opponent(state):
    """Re-order a raw 52-float state into the opponent's first-person perspective.

    Swaps the two 26-float halves: ``concatenate([state[26:], state[:26]])``.
    dtype is inherited from ``state`` (plain ``np.concatenate``). Applying it
    twice round-trips back to the original ordering.
    """
    return np.concatenate([state[26:], state[:26]])
