"""Shared action contract for the model-free agents — width, channel indices, validation.

Every agent in :mod:`pop_trainer.agents` emits the env's 5-float action
``[move_x, move_y, aim_x, aim_y, fire]`` whose channel ordering mirrors the Unity wire
action (the same list the step message ``{1: a1, 2: a2}`` carries). The move/aim channels
live in ``[-1, 1]``; ``fire`` is ``0.0`` / ``1.0``.

This module owns the action width (:data:`ACTION_LEN`), the named channel indices
(:data:`A_MOVE_X` ... :data:`A_FIRE`), the :func:`validate_action` coercer, and the
:func:`as_state_vector` adapter that lets a state-reading agent accept either a raw
52-float state sequence or an inbound state dict (``{"state": [...]}``).

stdlib + numpy compatible (``validate_action`` accepts any iterable, including a numpy
array); imports nothing internal.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "ACTION_LEN",
    "A_MOVE_X",
    "A_MOVE_Y",
    "A_AIM_X",
    "A_AIM_Y",
    "A_FIRE",
    "validate_action",
    "as_state_vector",
]

ACTION_LEN = 5

# Action channel indices (mirror the Unity wire action ordering).
A_MOVE_X = 0
A_MOVE_Y = 1
A_AIM_X = 2
A_AIM_Y = 3
A_FIRE = 4


def validate_action(action) -> list[float]:
    """Coerce ``action`` to a length-5 list of floats; raise ``ValueError`` on a bad length.

    Shared validator for agent outputs and usable by a caller before sending. ``fire`` is NOT
    clamped to {0, 1} here (the wire tolerates any float); callers that want a hard binary fire
    pass 0.0 / 1.0 explicitly. Accepts any iterable, including a numpy array.
    """
    vals = [float(x) for x in action]
    if len(vals) != ACTION_LEN:
        raise ValueError(f"action must have {ACTION_LEN} floats, got {len(vals)}")
    return vals


def as_state_vector(obs) -> Sequence:
    """Return the raw 52-float state sequence from ``obs``.

    A state-reading agent's ``act`` is handed its own first-person view; this adapter accepts
    either that view as a bare 52-float sequence OR as an inbound state dict
    (``{"state": [...]}``) and returns the bare vector. Does NOT copy — the caller owns the
    sequence.
    """
    if isinstance(obs, dict):
        vec = obs.get("state")
        if vec is None:
            raise ValueError("state dict has no 'state' key")
        return vec
    return obs
