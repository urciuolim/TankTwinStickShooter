"""Per-group decode targets + per-field normalization (PURE numpy, torch-free).

The single-frame decoder's OBJECTIVE is the 52-float wire state, carved into the named field
groups of :mod:`pop_trainer.core.state`. This module turns a batch of raw states into the
per-group target arrays the heads regress / classify, and fits the per-field normalization the
loss runs in (so a unit-scale aim target and a world-scale position target are comparable).

Field groups (read via the :mod:`pop_trainer.core.state` accessors / index constants — never a
magic index):

* ``player_position`` ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]`` world position.
* ``player_velocity`` ``(N, 4)`` = ``[P1_vx, P1_vy, P2_vx, P2_vy]``.
* ``player_aim`` ``(N, 4)`` = ``[P1_ax, P1_ay, P2_ax, P2_ay]`` (a unit-ish direction).
* ``bullet_presence`` ``(N, 10)`` {0,1}: slot present iff its ``pos_x`` is on-board
  (:func:`pop_trainer.core.state.bullet_present`); P1's 5 slots then P2's 5 slots.
* ``bullet_position`` ``(N, 20)`` = ``[slot0_x, slot0_y, slot1_x, ...]`` world position;
  absent slots carry the sentinel here (the loss / stats MASK them out).
* ``bullet_slot_mask`` ``(N, 20)`` {0,1}: the per-float presence mask (2 floats per slot).

NORMALIZATION is per-field standardize ``z = (x - mean) / std``. Stats are fit on the TRAIN
split ONLY (no val/test leak). Position / velocity stats use every row; bullet-position stats
use PRESENT slots only (the sentinel is masked out so it never skews the mean/std). Aim is a
unit direction trained by a cosine loss, so it is NOT normalized. Persist the stats
(:meth:`NormStats.to_json`) so metrics can de-normalize predictions back to world units.

stdlib + numpy + :mod:`pop_trainer.core.state` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pop_trainer.core import state as st

__all__ = [
    "GROUPS",
    "PLAYER_GROUPS",
    "NormStats",
    "player_position_targets",
    "player_velocity_targets",
    "player_aim_targets",
    "bullet_targets",
    "extract_targets",
    "fit_norm_stats",
    "presence_pos_weight",
    "normalize",
    "denormalize",
]

# Canonical group order used for logging / records.
GROUPS: tuple[str, ...] = (
    "player_position",
    "player_velocity",
    "player_aim",
    "bullet_presence",
    "bullet_position",
)
PLAYER_GROUPS: tuple[str, ...] = ("player_position", "player_velocity", "player_aim")

_PLAYERS = (st.PLAYER_1, st.PLAYER_2)
_N_BULLET_SLOTS = st.NUM_PLAYERS * st.NUM_BULLETS  # 10
_N_BULLET_POS = _N_BULLET_SLOTS * 2  # 20


def _player_pair(states: np.ndarray, x_off: int, y_off: int) -> np.ndarray:
    """``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]`` for a tank field at offsets ``x_off``/``y_off``."""
    states = np.asarray(states)
    cols = []
    for player in _PLAYERS:
        base = player * st.PLAYER_STRIDE
        cols.extend((base + x_off, base + y_off))
    return states[:, cols].astype(np.float32, copy=False)


def player_position_targets(states: np.ndarray) -> np.ndarray:
    """``(N, 4)`` tank position ``[P1_x, P1_y, P2_x, P2_y]`` (world units)."""
    return _player_pair(states, st.POS_X, st.POS_Y)


def player_velocity_targets(states: np.ndarray) -> np.ndarray:
    """``(N, 4)`` tank velocity ``[P1_vx, P1_vy, P2_vx, P2_vy]``."""
    return _player_pair(states, st.VEC_X, st.VEC_Y)


def player_aim_targets(states: np.ndarray) -> np.ndarray:
    """``(N, 4)`` tank aim direction ``[P1_ax, P1_ay, P2_ax, P2_ay]`` (unit-ish)."""
    return _player_pair(states, st.AIM_X, st.AIM_Y)


def _bullet_slot_indices() -> list[int]:
    """Absolute ``pos_x`` indices of all 10 bullet slots, P1's 5 then P2's 5."""
    out: list[int] = []
    for player in _PLAYERS:
        out.extend(st.bullet_field_indices(player))
    return out


def bullet_targets(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(presence, positions, slot_mask)`` for the 10 bullet slots (sentinel-aware).

    * ``presence`` ``(N, 10)`` float32 {0,1}: slot present iff its ``pos_x`` is on-board.
    * ``positions`` ``(N, 20)`` float32 raw ``[x, y]`` per slot, slot-major; absent slots
      still carry the sentinel ``pos_x`` (the caller masks them out).
    * ``slot_mask`` ``(N, 20)`` float32 {0,1}: presence repeated 2x (one flag per pos float).
    """
    states = np.asarray(states)
    n = states.shape[0]
    pos = np.zeros((n, _N_BULLET_POS), dtype=np.float32)
    presence = np.zeros((n, _N_BULLET_SLOTS), dtype=np.float32)
    for slot, idx in enumerate(_bullet_slot_indices()):
        px = states[:, idx + st.BULLET_POS_X]
        py = states[:, idx + st.BULLET_POS_Y]
        pos[:, slot * 2] = px
        pos[:, slot * 2 + 1] = py
        presence[:, slot] = st.bullet_present(px).astype(np.float32)
    slot_mask = np.repeat(presence, 2, axis=1).astype(np.float32)
    return presence, pos, slot_mask


@dataclass(frozen=True)
class NormStats:
    """Per-field standardization stats for the regression groups (TRAIN-fit, no leak).

    Holds mean/std for position / velocity (over every row) and bullet position (over PRESENT
    slots only). Aim is unnormalized (a cosine loss target), so it has no stats. Serialized
    alongside the checkpoint so metrics can de-normalize predictions back to world units.
    """

    position_mean: np.ndarray  # (4,)
    position_std: np.ndarray  # (4,)
    velocity_mean: np.ndarray  # (4,)
    velocity_std: np.ndarray  # (4,)
    bullet_pos_mean: np.ndarray  # (20,)
    bullet_pos_std: np.ndarray  # (20,)

    def to_json(self) -> dict:
        """Strict-JSON-serializable view (plain lists; no numpy / NaN / Inf)."""
        return {
            "convention": (
                "z = (x - mean) / std; x = z*std + mean. position/velocity over all rows, "
                "bullet_pos over PRESENT slots only. aim is unnormalized (cosine loss)."
            ),
            "position_mean": _finite_list(self.position_mean),
            "position_std": _finite_list(self.position_std),
            "velocity_mean": _finite_list(self.velocity_mean),
            "velocity_std": _finite_list(self.velocity_std),
            "bullet_pos_mean": _finite_list(self.bullet_pos_mean),
            "bullet_pos_std": _finite_list(self.bullet_pos_std),
        }

    @classmethod
    def from_json(cls, d: dict) -> NormStats:
        """Rebuild from :meth:`to_json` output."""
        f = lambda k: np.asarray(d[k], dtype=np.float32)  # noqa: E731
        return cls(
            position_mean=f("position_mean"),
            position_std=f("position_std"),
            velocity_mean=f("velocity_mean"),
            velocity_std=f("velocity_std"),
            bullet_pos_mean=f("bullet_pos_mean"),
            bullet_pos_std=f("bullet_pos_std"),
        )


def _finite_list(arr: np.ndarray) -> list[float]:
    """``arr`` as a list of finite floats (NaN/Inf -> 0.0) for strict-JSON safety."""
    a = np.asarray(arr, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    return [float(x) for x in a]


def _mean_std(values: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-field mean/std over every row; std floored at ``eps`` (constant fields)."""
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), eps).astype(np.float32)
    return mean, std


def _masked_mean_std(
    values: np.ndarray, present_mask: np.ndarray, *, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-field mean/std over PRESENT entries; 0/1 fallback where a field has no samples."""
    n_fields = values.shape[1]
    mean = np.zeros(n_fields, dtype=np.float32)
    std = np.ones(n_fields, dtype=np.float32)
    for j in range(n_fields):
        col = values[present_mask[:, j], j]
        if col.size > 0:
            mean[j] = float(col.mean())
            std[j] = float(max(col.std(), eps))
    return mean, std


def fit_norm_stats(states: np.ndarray, *, eps: float = 1e-6) -> NormStats:
    """Fit per-field normalization stats for the regression groups (TRAIN split only).

    Position / velocity stats use every row. Bullet-position stats use PRESENT slots only —
    the sentinel ``pos_x`` of absent slots is masked out so it never skews mean/std. ``std`` is
    floored at ``eps`` to keep division safe on constant fields. Call this with the TRAIN
    states ONLY so val/test never leak into the normalization.
    """
    states = np.asarray(states)
    pos_mean, pos_std = _mean_std(player_position_targets(states), eps=eps)
    vel_mean, vel_std = _mean_std(player_velocity_targets(states), eps=eps)
    _, bullet_pos, slot_mask = bullet_targets(states)
    bp_mean, bp_std = _masked_mean_std(bullet_pos, slot_mask.astype(bool), eps=eps)
    return NormStats(
        position_mean=pos_mean,
        position_std=pos_std,
        velocity_mean=vel_mean,
        velocity_std=vel_std,
        bullet_pos_mean=bp_mean,
        bullet_pos_std=bp_std,
    )


def presence_pos_weight(presence: np.ndarray, *, fallback: float = 1.0) -> float:
    """BCE positive-class weight for bullet presence: ``(#absent) / (#present)``.

    Bullet slots are overwhelmingly absent (~3.3% present), so plain BCE collapses to
    "always absent". A ``pos_weight`` of absent/present re-balances the loss. Compute on the
    TRAIN presence targets ONLY (no leak). Falls back to ``fallback`` if no slots are present.
    """
    arr = np.asarray(presence)
    present = float((arr > 0.5).sum())
    if present <= 0.0:
        return float(fallback)
    absent = float(arr.size) - present
    return absent / present


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(x - mean) / std`` (per-field; broadcasts over the leading axis)."""
    return (np.asarray(x, dtype=np.float32) - mean) / std


def denormalize(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize`: ``z * std + mean`` (back to world units)."""
    return np.asarray(z, dtype=np.float32) * std + mean


def extract_targets(states: np.ndarray, stats: NormStats) -> dict[str, np.ndarray]:
    """Build the full per-group target dict for a batch of states, normalized via ``stats``.

    Position / velocity are normalized; aim is left raw (cosine loss); bullet positions are
    normalized over PRESENT slots and the absent slots are ZEROED (so a masked loss ignores
    them and a zero is a clean no-op). Returns keys:
    ``player_position`` ``(N,4)``, ``player_velocity`` ``(N,4)``, ``player_aim`` ``(N,4)``,
    ``bullet_presence`` ``(N,10)``, ``bullet_position`` ``(N,20)``, ``bullet_slot_mask``
    ``(N,20)``.
    """
    states = np.asarray(states)
    pos = normalize(player_position_targets(states), stats.position_mean, stats.position_std)
    vel = normalize(player_velocity_targets(states), stats.velocity_mean, stats.velocity_std)
    aim = player_aim_targets(states)
    presence, raw_bpos, slot_mask = bullet_targets(states)
    bpos = normalize(raw_bpos, stats.bullet_pos_mean, stats.bullet_pos_std) * slot_mask
    return {
        "player_position": pos.astype(np.float32),
        "player_velocity": vel.astype(np.float32),
        "player_aim": aim.astype(np.float32),
        "bullet_presence": presence.astype(np.float32),
        "bullet_position": bpos.astype(np.float32),
        "bullet_slot_mask": slot_mask.astype(np.float32),
    }
