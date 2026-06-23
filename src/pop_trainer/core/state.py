"""The 52-float game-state schema — single source of truth for the wire state.

This module DEFINES the meaning of the 52-float state vector the Unity simulator
emits on the TCP-JSON wire (``GameController.UpdateState`` — the frozen RL seam). The state
is the supervised-decode OBJECTIVE, NOT the policy observation — the observation is the real
rendered pixel frame from Unity (see :mod:`pop_trainer.core.protocol`). Every consumer (the
protocol's inbound state, the perspective transforms) reads field positions from the named
constants here instead of hardcoding magic indices.

Layout (52 floats total): Player 1 occupies indices 0..25, Player 2 occupies indices
26..51. Each 26-float player block is::

    [pos_x, pos_y, vec_x, vec_y, aim_x, aim_y,
     bullet0_pos_x, bullet0_pos_y, bullet0_vec_x, bullet0_vec_y,
     bullet1_pos_x, ... bullet4_vec_y]   # 5 bullets x 4 floats = 20 floats

So per player: 6 floats of tank state (pos/vec/aim 2D each) + 5 bullets * 4 floats.
P1 bullet fields start at index 6 (``range(6, 26, 4)``); P2 bullet fields start at
index 32 (``range(32, 52, 4)``).

An ABSENT bullet is signalled by a sentinel: its ``pos_x`` is set to ``-100``. Because
that is well off-board and negative, consumers test ``pos_x >= 0`` (or
:func:`bullet_present`) and skip absent bullets.

The accessors are pure and accept ANY indexable sequence (a Python list or a numpy
array) — numpy is NOT required to read the schema. The perspective transforms
(:func:`flip_frame_perspective`, :func:`split_state_for_opponent`) ARE numpy operations
and are part of the state contract (the RL self-play seam). stdlib + numpy only; nothing
internal.
"""

from __future__ import annotations

import numpy as np

# --- player block layout ---------------------------------------------------------------
STATE_LEN = 52  # total floats in a wire state vector
PLAYER_STRIDE = 26  # floats per player block
NUM_PLAYERS = 2

PLAYER_1 = 0  # index of player 1's block: state[0:26]
PLAYER_2 = 1  # index of player 2's block: state[26:52]

# Field offsets WITHIN a player block (add PLAYER_STRIDE * player to get the absolute idx).
POS_X = 0
POS_Y = 1
VEC_X = 2
VEC_Y = 3
AIM_X = 4
AIM_Y = 5

# --- bullets ----------------------------------------------------------------------------
NUM_BULLETS = 5  # bullet slots per player
BULLET_STRIDE = 4  # floats per bullet: (pos_x, pos_y, vec_x, vec_y)
BULLET_BLOCK_OFFSET = 6  # first bullet field, relative to the start of a player block

# Bullet sub-field offsets within a 4-float bullet record.
BULLET_POS_X = 0
BULLET_POS_Y = 1
BULLET_VEC_X = 2
BULLET_VEC_Y = 3

# Absolute starting indices of the first bullet field, per player. The full per-player
# bullet range is range(start, start + NUM_BULLETS * BULLET_STRIDE, BULLET_STRIDE):
#   P1 -> range(6, 26, 4)   (indices 6, 10, 14, 18, 22)
#   P2 -> range(32, 52, 4)  (indices 32, 36, 40, 44, 48)
BULLET_START = (
    PLAYER_1 * PLAYER_STRIDE + BULLET_BLOCK_OFFSET,
    PLAYER_2 * PLAYER_STRIDE + BULLET_BLOCK_OFFSET,
)

# An absent bullet's pos_x is set to this sentinel; consumers skip when pos_x < 0.
ABSENT_BULLET_SENTINEL = -100.0


def _check_player(player: int) -> None:
    """Raise ``ValueError`` unless ``player`` is PLAYER_1 (0) or PLAYER_2 (1)."""
    if player not in (PLAYER_1, PLAYER_2):
        raise ValueError(
            f"player must be PLAYER_1 ({PLAYER_1}) or PLAYER_2 ({PLAYER_2}), got {player!r}"
        )


def _player_base(player: int) -> int:
    """Absolute index of the start of ``player``'s 26-float block (PLAYER_1 / PLAYER_2)."""
    _check_player(player)
    return player * PLAYER_STRIDE


def validate(state) -> None:
    """Raise ``ValueError`` unless ``state`` has exactly :data:`STATE_LEN` (52) elements.

    Pure check on any sized sequence (list / tuple / 1-D numpy array). Does not inspect
    field values — only the length contract that makes the accessors meaningful.
    """
    n = len(state)
    if n != STATE_LEN:
        raise ValueError(f"state must have {STATE_LEN} floats, got {n}")


def position(state, player: int):
    """Return ``(pos_x, pos_y)`` for ``player`` (PLAYER_1 / PLAYER_2)."""
    base = _player_base(player)
    return state[base + POS_X], state[base + POS_Y]


def velocity(state, player: int):
    """Return the velocity vector ``(vec_x, vec_y)`` for ``player``."""
    base = _player_base(player)
    return state[base + VEC_X], state[base + VEC_Y]


def aim(state, player: int):
    """Return the aim direction ``(aim_x, aim_y)`` for ``player``."""
    base = _player_base(player)
    return state[base + AIM_X], state[base + AIM_Y]


def bullet_field_indices(player: int) -> range:
    """Return the ``range`` of absolute starting indices for ``player``'s bullet records.

    P1 -> ``range(6, 26, 4)``, P2 -> ``range(32, 52, 4)``. Each yielded index ``i`` is the
    ``pos_x`` of one bullet; the record is ``state[i : i + 4]``.
    """
    _check_player(player)
    start = BULLET_START[player]
    return range(start, start + NUM_BULLETS * BULLET_STRIDE, BULLET_STRIDE)


def bullet_present(pos_x) -> bool:
    """Whether a bullet slot is occupied: ``True`` iff its ``pos_x`` is on-board (>= 0).

    The absent-bullet sentinel sets ``pos_x`` to :data:`ABSENT_BULLET_SENTINEL` (-100),
    so any negative ``pos_x`` marks an empty slot.
    """
    return pos_x >= 0


def iter_bullets(state, player: int, *, include_absent: bool = False):
    """Yield ``(pos_x, pos_y, vec_x, vec_y)`` for each of ``player``'s bullet slots.

    By default absent slots (``pos_x`` < 0, the sentinel) are skipped via the
    :func:`bullet_present` rule. Pass ``include_absent=True`` to yield all five raw records.
    """
    for i in bullet_field_indices(player):
        pos_x = state[i + BULLET_POS_X]
        if not include_absent and not bullet_present(pos_x):
            continue
        yield (
            pos_x,
            state[i + BULLET_POS_Y],
            state[i + BULLET_VEC_X],
            state[i + BULLET_VEC_Y],
        )


# --- perspective transforms (part of the RL self-play state contract) ------------------


def flip_frame_perspective(state):
    """Swap the R (self) and B (opponent) channels of a rendered RGB frame; keep G.

    A self-play perspective op on the REAL rendered pixel frame from Unity (an
    ``(H, W, 3)`` array, e.g. from :meth:`pop_trainer.core.protocol.Connection.receive_frame`):
    returns a new array of the same shape and dtype with the red/blue channels exchanged so
    the opponent's view puts "self" back in R. ``flip_frame_perspective`` is an involution on
    the R/B channels — ``flip_frame_perspective(flip_frame_perspective(s))`` restores the
    original.
    """
    r, g, b = (0, 1, 2)
    new_state = np.zeros(state.shape, dtype=state.dtype)
    new_state[:, :, r] = state[:, :, b].copy()
    new_state[:, :, g] = state[:, :, g].copy()
    new_state[:, :, b] = state[:, :, r].copy()
    return new_state


def split_state_for_opponent(state):
    """Re-order a raw 52-float state into the opponent's first-person perspective.

    Swaps the two 26-float halves: ``concatenate([state[26:], state[:26]])``. The dtype is
    inherited from ``state`` (plain ``np.concatenate``). Applying it twice round-trips back
    to the original ordering (an involution).
    """
    return np.concatenate([state[PLAYER_STRIDE:], state[:PLAYER_STRIDE]])
