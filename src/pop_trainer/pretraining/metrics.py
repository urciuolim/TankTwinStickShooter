"""Per-group decode metrics (PURE numpy; reported in interpretable units).

Each metric takes a head's predictions and the (normalized) targets plus the :class:`NormStats`
needed to de-normalize back to world units, so the numbers are meaningful regardless of the
normalization the loss ran in. Every metric has the SAME signature whether the predictions come
from the spatial heads or the embed-probe, so the trainer reports both with one code path.

* position / velocity -> mean L2 error in WORLD units (de-normalized first).
* aim -> mean per-player angular error in DEGREES (clamped arccos of the per-player cosine).
* bullet presence -> accuracy + precision / recall / F1.
* bullet position -> mean L2 error over PRESENT slots only (de-normalized).

Inputs accept torch tensors or numpy arrays (converted via ``np.asarray`` after a detached
``.cpu().numpy()`` when needed); outputs are plain Python floats so the records are strict-JSON
safe. numpy only; imports :mod:`pop_trainer.pretraining.targets` for the stats type + denorm.
"""

from __future__ import annotations

import numpy as np

from pop_trainer.pretraining.targets import NormStats, denormalize

__all__ = [
    "to_numpy",
    "position_error_world",
    "velocity_error_world",
    "aim_angular_error_deg",
    "presence_metrics",
    "bullet_position_error_world",
    "group_metrics",
]


def to_numpy(x) -> np.ndarray:
    """Convert a torch tensor / array-like to a detached float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _pairwise_l2(pred_xy: np.ndarray) -> np.ndarray:
    """Mean L2 error already as ``(N, K)`` per-2D-point deltas -> ``(N, K)`` distances."""
    return np.sqrt((pred_xy**2).sum(axis=-1))


def _l2_error_world(
    pred: np.ndarray, target: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> float:
    """Mean per-point L2 error in world units after de-normalizing both pred and target."""
    pred = to_numpy(pred)
    target = to_numpy(target)
    if pred.size == 0:
        return 0.0
    pw = denormalize(pred, mean, std).reshape(pred.shape[0], -1, 2)
    tw = denormalize(target, mean, std).reshape(target.shape[0], -1, 2)
    return float(_pairwise_l2(pw - tw).mean())


def position_error_world(pred: np.ndarray, target: np.ndarray, stats: NormStats) -> float:
    """Mean tank-position L2 error in WORLD units (de-normalized)."""
    return _l2_error_world(pred, target, stats.position_mean, stats.position_std)


def velocity_error_world(pred: np.ndarray, target: np.ndarray, stats: NormStats) -> float:
    """Mean tank-velocity L2 error in WORLD units (de-normalized)."""
    return _l2_error_world(pred, target, stats.velocity_mean, stats.velocity_std)


def aim_angular_error_deg(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    """Mean per-player aim angular error in DEGREES (clamped arccos of per-player cosine).

    ``pred`` / ``target`` are ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]``. Identical directions
    -> ~0 deg, orthogonal -> ~90 deg, opposite -> ~180 deg. Aim is unnormalized, so no stats.
    """
    p = to_numpy(pred).reshape(-1, 2, 2)
    t = to_numpy(target).reshape(-1, 2, 2)
    if p.size == 0:
        return 0.0
    pn = np.sqrt((p**2).sum(axis=2)) + eps
    tn = np.sqrt((t**2).sum(axis=2)) + eps
    cos = np.clip((p * t).sum(axis=2) / (pn * tn), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def presence_metrics(logits: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Accuracy + precision / recall / F1 for bullet presence (logits thresholded at 0).

    ``logits`` ``(N, 10)`` raw head outputs (a logit > 0 predicts present); ``target`` {0,1}.
    Returns ``{"accuracy", "precision", "recall", "f1"}`` as floats.
    """
    pred = (to_numpy(logits) > 0.0).astype(np.int64).ravel()
    tgt = (to_numpy(target) > 0.5).astype(np.int64).ravel()
    if pred.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    accuracy = float((pred == tgt).mean())
    tp = int(np.sum((pred == 1) & (tgt == 1)))
    fp = int(np.sum((pred == 1) & (tgt == 0)))
    fn = int(np.sum((pred == 0) & (tgt == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def bullet_position_error_world(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, stats: NormStats
) -> float:
    """Mean bullet-position L2 error in WORLD units over PRESENT slots only (de-normalized).

    ``pred`` / ``target`` are ``(N, 20)`` normalized positions; ``mask`` ``(N, 20)`` is the
    per-float presence mask. Errors are taken only where a slot is present; returns 0.0 if no
    slot is present.
    """
    pred = to_numpy(pred)
    target = to_numpy(target)
    mask = to_numpy(mask)
    if pred.size == 0:
        return 0.0
    pw = denormalize(pred, stats.bullet_pos_mean, stats.bullet_pos_std)
    tw = denormalize(target, stats.bullet_pos_mean, stats.bullet_pos_std)
    slot_mask = mask.reshape(mask.shape[0], -1, 2)[..., 0]  # (N, 10): one flag per slot
    dist = _pairwise_l2((pw - tw).reshape(pred.shape[0], -1, 2))  # (N, 10)
    denom = slot_mask.sum()
    if denom <= 0:
        return 0.0
    return float((dist * slot_mask).sum() / denom)


def group_metrics(
    preds: dict[str, np.ndarray], targets: dict[str, np.ndarray], stats: NormStats
) -> dict[str, dict[str, float] | float]:
    """All per-group metrics for one head family's outputs (spatial OR probe).

    Returns ``{group: value}`` where position/velocity/bullet_position are float world-unit
    errors, aim is a float angular error (deg), and presence is the metric sub-dict. A group
    absent from ``preds`` is skipped.
    """
    out: dict[str, dict[str, float] | float] = {}
    if "player_position" in preds:
        out["player_position"] = position_error_world(
            preds["player_position"], targets["player_position"], stats
        )
    if "player_velocity" in preds:
        out["player_velocity"] = velocity_error_world(
            preds["player_velocity"], targets["player_velocity"], stats
        )
    if "player_aim" in preds:
        out["player_aim"] = aim_angular_error_deg(preds["player_aim"], targets["player_aim"])
    if "bullet_presence" in preds:
        out["bullet_presence"] = presence_metrics(
            preds["bullet_presence"], targets["bullet_presence"]
        )
    if "bullet_position" in preds:
        out["bullet_position"] = bullet_position_error_world(
            preds["bullet_position"],
            targets["bullet_position"],
            targets["bullet_slot_mask"],
            stats,
        )
    return out
