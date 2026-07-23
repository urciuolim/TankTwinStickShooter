"""Per-group decode losses (torch; operate on prediction/target tensors).

One loss per field group, plus a :func:`combined_loss` that sums the enabled groups. The
losses are pure tensor functions (no module state) so they unit-test on hand-built tensors:

* position / velocity -> MSE in NORMALIZED space.
* aim -> per-player COSINE distance (NOT MSE: MSE collapses a unit target toward 0). The
  ``(N, 4)`` aim slice ``[P1_x, P1_y, P2_x, P2_y]`` is reshaped to ``(N, 2, 2)``, each
  player's 2D vector is eps-floored-L2-normalized, and ``1 - cos`` is averaged over the 2
  players and the batch.
* bullet presence -> BCE-with-logits, with an optional ``pos_weight`` for the present-slot
  imbalance.
* bullet position -> MASKED MSE: absent slots are excluded from the mean via the per-float
  slot mask (so an absent slot contributes neither error nor denominator).

The SAME group losses run on the spatial-head outputs (with :func:`combined_loss`) and on the
detached embed-probe outputs (with :func:`probe_loss`); both take a head-output dict and a
target dict.

torch + stdlib only; nothing internal.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "AIM_COSINE_EPS",
    "DEFAULT_GROUP_WEIGHTS",
    "DEFAULT_HEATMAP_WEIGHT",
    "DEFAULT_HEATMAP_SIGMA",
    "position_loss",
    "velocity_loss",
    "aim_cosine_loss",
    "presence_bce_loss",
    "bullet_position_masked_mse",
    "heatmap_ce_loss",
    "group_losses",
    "combined_loss",
    "probe_loss",
]

# Floored into the aim L2-norm denominator so a zero-magnitude prediction yields a finite
# loss (no NaN / blow-up as |pred| -> 0).
AIM_COSINE_EPS = 1e-8

# Auxiliary heatmap cross-entropy defaults. The CE rides ON TOP of the soft-argmax coord MSE; on
# the coarse feature grid a STRONG weight (>= the unit coord scale) is required — weak weights
# (<=1) backfire. Both knobs are grid-resolution dependent, so both are configurable.
DEFAULT_HEATMAP_WEIGHT = 6.0
DEFAULT_HEATMAP_SIGMA = 0.7  # Gaussian std in grid CELLS (per-axis-normalized at render time)

# Sensible default per-group weights for the combined objective. Presence is weighted up a
# touch (rare positives); the rest are unit-weighted ("good not best").
DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "player_position": 1.0,
    "player_velocity": 1.0,
    "player_aim": 1.0,
    "bullet_presence": 1.0,
    "bullet_position": 1.0,
}


def position_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE for the normalized position group ``(N, 4)``."""
    return F.mse_loss(pred, target)


def velocity_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE for the normalized velocity group ``(N, 4)``."""
    return F.mse_loss(pred, target)


def aim_cosine_loss(pred: Tensor, target: Tensor, *, eps: float = AIM_COSINE_EPS) -> Tensor:
    """Mean per-player cosine distance ``1 - cos`` for an aim slice ``(N, 4)``.

    Columns are ``[P1_x, P1_y, P2_x, P2_y]``; reshaped to ``(N, 2, 2)`` and each player's 2D
    vector L2-normalized with ``eps`` floored into the denominator before the dot. Identical
    directions -> 0, orthogonal -> 1, opposite -> 2. Averaged over the 2 players and the batch.
    """
    p = pred.reshape(-1, 2, 2)
    t = target.reshape(-1, 2, 2)
    pn = torch.sqrt((p * p).sum(dim=2)) + eps  # (N, 2)
    tn = torch.sqrt((t * t).sum(dim=2)) + eps  # (N, 2)
    cos = (p * t).sum(dim=2) / (pn * tn)  # (N, 2)
    return (1.0 - cos).mean()


def presence_bce_loss(logits: Tensor, target: Tensor, *, pos_weight: float | None = None) -> Tensor:
    """BCE-with-logits for bullet presence ``(N, 10)``, with optional ``pos_weight``.

    ``pos_weight`` (absent/present ratio, fit on TRAIN) up-weights the rare present class so
    the loss does not collapse to "always absent".
    """
    pw = (
        None
        if pos_weight is None
        else torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    )
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def bullet_position_masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Masked MSE for bullet position ``(N, 20)``: absent slots excluded from the mean.

    ``mask`` is the per-float ``(N, 20)`` {0,1} presence mask. The squared error is summed
    over PRESENT floats and divided by the present-float count, so absent slots contribute
    neither error nor denominator. Returns a zero tensor (grad-connected) when no slot is
    present in the batch.
    """
    se = (pred - target) ** 2 * mask
    denom = mask.sum()
    if denom <= 0:
        return (pred * mask).sum() * 0.0
    return se.sum() / denom


def heatmap_ce_loss(
    score_logits: Tensor,
    target_grid: Tensor,
    present_mask: Tensor,
    *,
    sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> Tensor:
    """Cross-entropy between per-keypoint score maps and a Gaussian target, masked to present.

    ``score_logits`` ``(B, K, h, w)`` are the raw per-keypoint score maps; ``target_grid``
    ``(B, K, 2)`` holds each keypoint's TRUE location as fractional grid coords in ``[0, 1]``;
    ``present_mask`` ``(B, K)`` {0,1} flags which keypoints exist (players always; bullets only
    where present).

    The target distribution per keypoint is a 2D Gaussian centered at its grid location, rendered
    on the ``(h, w)`` cell grid (cell ``(i, j)`` center ``((j+0.5)/w, (i+0.5)/h)`` — the same
    convention as :func:`~pop_trainer.pretraining.decoder.soft_argmax`) and normalized to sum 1 by
    a spatial softmax. ``sigma`` is the std in grid CELLS, converted per-axis to normalized units
    (``sfx = sigma/w``, ``sfy = sigma/h``). The loss is the CE
    ``-(target * log_softmax(score_logits)).sum(grid)`` averaged over PRESENT keypoints only.
    Computed as CE (NOT ``F.kl_div``: dropping the constant target-entropy term keeps the gradient
    identical while avoiding its NaN). Returns a grad-connected zero when no keypoint is present.
    """
    b, k, h, w = score_logits.shape
    device = score_logits.device
    dtype = score_logits.dtype
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w  # (w,) column centers
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h  # (h,) row centers
    tx = target_grid[..., 0].to(dtype)  # (B, K)
    ty = target_grid[..., 1].to(dtype)  # (B, K)
    sfx = sigma / w
    sfy = sigma / h
    dx = (xs.view(1, 1, 1, w) - tx.view(b, k, 1, 1)) / sfx  # (B, K, 1, w)
    dy = (ys.view(1, 1, h, 1) - ty.view(b, k, 1, 1)) / sfy  # (B, K, h, 1)
    log_t = -0.5 * (dx * dx + dy * dy)  # (B, K, h, w): unnormalized Gaussian log-density
    target = torch.softmax(log_t.reshape(b, k, h * w), dim=-1)  # normalized to sum 1 over the grid
    log_p = torch.log_softmax(score_logits.reshape(b, k, h * w), dim=-1)
    ce = -(target * log_p).sum(dim=-1)  # (B, K)
    mask = present_mask.to(dtype)
    denom = mask.sum()
    if denom <= 0:
        return (ce * mask).sum() * 0.0
    return (ce * mask).sum() / denom


def group_losses(
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    presence_pos_weight: float | None = None,
) -> dict[str, Tensor]:
    """Compute every per-group loss present in ``preds`` (each tensor a scalar).

    Keys mirror the field groups: ``player_position``, ``player_velocity``, ``player_aim``,
    ``bullet_presence``, ``bullet_position``. A group absent from ``preds`` is skipped.
    """
    out: dict[str, Tensor] = {}
    if "player_position" in preds:
        out["player_position"] = position_loss(preds["player_position"], targets["player_position"])
    if "player_velocity" in preds:
        out["player_velocity"] = velocity_loss(preds["player_velocity"], targets["player_velocity"])
    if "player_aim" in preds:
        out["player_aim"] = aim_cosine_loss(preds["player_aim"], targets["player_aim"])
    if "bullet_presence" in preds:
        out["bullet_presence"] = presence_bce_loss(
            preds["bullet_presence"], targets["bullet_presence"], pos_weight=presence_pos_weight
        )
    if "bullet_position" in preds:
        out["bullet_position"] = bullet_position_masked_mse(
            preds["bullet_position"], targets["bullet_position"], targets["bullet_slot_mask"]
        )
    return out


def _weighted_sum(
    per_group: dict[str, Tensor], weights: dict[str, float]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of the per-group scalars (default weight 1.0 for unlisted groups)."""
    total: Tensor | None = None
    for name, value in per_group.items():
        term = value * float(weights.get(name, 1.0))
        total = term if total is None else total + term
    if total is None:
        # No groups: a zero scalar (kept on the right device by the first available tensor).
        any_t = next(iter(per_group.values()), None)
        total = torch.zeros((), device=any_t.device) if any_t is not None else torch.zeros(())
    return total, per_group


def combined_loss(
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    weights: dict[str, float] | None = None,
    presence_pos_weight: float | None = None,
    score_logits: Tensor | None = None,
    heatmap_weight: float = DEFAULT_HEATMAP_WEIGHT,
    heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of the enabled group losses for the encoder + spatial heads.

    Returns ``(total, per_group)`` where ``per_group`` is the un-weighted per-group scalar
    dict (for logging). ``weights`` defaults to :data:`DEFAULT_GROUP_WEIGHTS`.

    When ``score_logits`` is given AND ``targets`` carries ``keypoint_grid`` + ``keypoint_present``,
    an AUXILIARY heatmap cross-entropy term (:func:`heatmap_ce_loss`) is added on top of the coord
    losses: ``total += heatmap_weight * ce`` (its own scalar weight, NOT routed through
    ``weights``), and the UNWEIGHTED ``ce`` is recorded in ``per_group["heatmap"]`` for logging.
    With ``score_logits=None`` the behavior is byte-identical to the coord-only objective.
    """
    weights = DEFAULT_GROUP_WEIGHTS if weights is None else weights
    per_group = group_losses(preds, targets, presence_pos_weight=presence_pos_weight)
    total, per_group = _weighted_sum(per_group, weights)
    if score_logits is not None and "keypoint_grid" in targets and "keypoint_present" in targets:
        ce = heatmap_ce_loss(
            score_logits, targets["keypoint_grid"], targets["keypoint_present"], sigma=heatmap_sigma
        )
        total = total.to(ce.device) + heatmap_weight * ce
        per_group["heatmap"] = ce
    return total, per_group


def probe_loss(
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    weights: dict[str, float] | None = None,
    presence_pos_weight: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The SAME weighted group-loss sum, applied to the detached embed-probe outputs.

    Identical math to :func:`combined_loss`; named separately because it trains a disjoint
    parameter set (the probe heads) and its inputs are read from ``embed.detach()`` so it
    never backprops into the encoder.
    """
    return combined_loss(preds, targets, weights=weights, presence_pos_weight=presence_pos_weight)
