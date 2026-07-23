"""The single-frame decoder: an :class:`~pop_trainer.models.Encoder` + disposable heads.

A supervised INVERSE-RENDERER. It wraps a config-built encoder and attaches two head families
that both decode the per-group state targets, so we can compare them:

* **spatial heads** read ``Encoder.features`` (the ``(B, C, h, w)`` map) and LOCALIZE on it.
  Locatable groups (player / bullet position) are decoded by a heatmap + soft-argmax head:
  a small spatial-context conv (3x3 -> ReLU -> 1x1) produces one score map per keypoint, a
  spatial softmax turns it into a distribution over the feature grid, and the expected grid
  coordinate is the predicted location. A learnable per-keypoint affine maps grid coords into
  the normalized target space the loss runs in. Bullet presence is a confidence read-out off
  the same score maps. Aim / velocity are directions / derivatives (not locations), so they use
  a position-preserving flatten-then-FC head. The whole family's gradient flows INTO the
  encoder, so it is what trains the reusable artifact.
* the **embed-probe** reads ``Encoder.embed`` and DETACHES it (``embed.detach()``) before its
  own per-group FC heads, so the probe NEVER backprops into the encoder. It measures how much
  state survives pooling (GAP vs Flatten), reported alongside the spatial heads.

Both families produce the same per-group outputs (position 4, velocity 4, aim 4, presence 10
logits, bullet position 20). :meth:`forward` returns ``{"spatial": {...}, "probe": {...},
"spatial_score_logits": (B,12,h,w)}`` — the raw score maps surfaced top-level for the auxiliary
heatmap cross-entropy loss (top-level so the val-split eval concat never accumulates them).

Head input widths are sized from the encoder at build time by probing the trunk with a single
zero frame at the chosen resolution (device-correct), so any trunk/pooling/resolution composes.

torch + :mod:`pop_trainer.models` only; nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pop_trainer.models import Encoder

__all__ = ["GROUP_OUTPUT_SIZES", "soft_argmax", "StateDecoder"]

# Output width of each per-group head (both families share these).
GROUP_OUTPUT_SIZES: dict[str, int] = {
    "player_position": 4,
    "player_velocity": 4,
    "player_aim": 4,
    "bullet_presence": 10,
    "bullet_position": 20,
}

# Locatable keypoints, in the order the score-map conv emits them: 2 player keypoints
# (P1, P2) then 10 bullet keypoints (P1's 5 slots, then P2's 5), matching the target layout.
_N_PLAYER_KEYPOINTS = 2
_N_BULLET_KEYPOINTS = 10
_N_KEYPOINTS = _N_PLAYER_KEYPOINTS + _N_BULLET_KEYPOINTS  # 12

# Hidden-channel width of the score head's spatial-context (3x3) conv; gives each keypoint a
# receptive field so it can localize, rather than scoring from a single feature cell.
_SCORE_HIDDEN = 64


def soft_argmax(scores: Tensor) -> Tensor:
    """Differentiable soft-argmax over a ``(B, K, h, w)`` score map -> ``(B, K, 2)`` coords.

    A spatial softmax over the ``(h, w)`` dims turns each keypoint's scores into a probability
    distribution over the feature grid; the expected ``(x, y)`` under that distribution is the
    predicted location. Grid convention: cell CENTERS linearly spaced in ``[0, 1]`` — column
    centers along width give ``x``, row centers along height give ``y``. So a one-hot at cell
    ``(i, j)`` returns ``((j + 0.5) / w, (i + 0.5) / h)`` and a uniform map returns the grid
    center ``(~0.5, ~0.5)``. Fully differentiable, so gradient flows to the conv and the encoder.
    """
    b, k, h, w = scores.shape
    flat = scores.reshape(b, k, h * w)
    prob = torch.softmax(flat, dim=-1).reshape(b, k, h, w)
    device = scores.device
    dtype = scores.dtype
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w  # (w,) column centers
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h  # (h,) row centers
    x = (prob.sum(dim=2) * xs).sum(dim=-1)  # (B, K): marginalize over rows, weight by columns
    y = (prob.sum(dim=3) * ys).sum(dim=-1)  # (B, K): marginalize over columns, weight by rows
    return torch.stack((x, y), dim=-1)  # (B, K, 2)


class _CoordAffine(nn.Module):
    """Per-keypoint learnable affine: maps soft-argmax grid coords into normalized space.

    The targets are per-run z-scored world coordinates whose mean/std are unknown at build time,
    so a learnable scale + bias per ``(keypoint, coordinate)`` lets each keypoint land in the
    same normalized space the MSE loss expects. ``scale`` / ``bias`` are shape ``(k, 2)``;
    applied to a ``(B, k, 2)`` coord tensor, they broadcast over the batch dim.

    Initialized to map the soft-argmax grid (coords in ``[0, 1]``, center ``0.5``) onto a unit
    z-score range: ``scale = 4`` spans roughly ``[-2, 2]`` and ``bias = -2`` centers the grid
    middle at ``0`` (the target mean). Starting in-range avoids a slow scale warm-up so the
    keypoint gradient reaches the encoder from the first steps.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((k, 2), 4.0))
        self.bias = nn.Parameter(torch.full((k, 2), -2.0))

    def forward(self, coords: Tensor) -> Tensor:
        return coords * self.scale + self.bias


class _SpatialHeads(nn.Module):
    """The localization head family read off the trunk's ``(B, C, h, w)`` feature map.

    * a spatial-context conv (3x3 -> ReLU -> 1x1) emits ``_N_KEYPOINTS`` score maps; the 3x3
      gives each keypoint a receptive field so it can localize. Soft-argmax localizes each
      keypoint, and separate player / bullet per-keypoint affines map grid coords into the
      normalized target space.
    * a presence read-out takes each bullet keypoint's score-map confidence (logsumexp over the
      grid) through a per-bullet learnable scale + bias to a presence logit.
    * aim / velocity are directions / derivatives (not locations): a position-preserving
      flatten of the feature map feeds a per-group FC head.
    """

    def __init__(
        self, channels: int, flat_dim: int, hidden: int, *, hidden_score: int = _SCORE_HIDDEN
    ) -> None:
        super().__init__()
        self.score_conv = nn.Sequential(
            nn.Conv2d(channels, hidden_score, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_score, _N_KEYPOINTS, kernel_size=1),
        )
        self.player_affine = _CoordAffine(_N_PLAYER_KEYPOINTS)
        self.bullet_affine = _CoordAffine(_N_BULLET_KEYPOINTS)
        # Presence confidence: per-bullet scalar (logsumexp over the grid) -> learnable affine.
        self.presence_scale = nn.Parameter(torch.ones(_N_BULLET_KEYPOINTS))
        self.presence_bias = nn.Parameter(torch.zeros(_N_BULLET_KEYPOINTS))
        # Direction / derivative groups: position-preserving flatten -> FC.
        self.aim_head = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, GROUP_OUTPUT_SIZES["player_aim"]),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, GROUP_OUTPUT_SIZES["player_velocity"]),
        )

    def forward(self, feat: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        b = feat.shape[0]
        scores = self.score_conv(feat)  # (B, K, h, w)
        coords = soft_argmax(scores)  # (B, K, 2) in [0, 1]
        player_coords = self.player_affine(coords[:, :_N_PLAYER_KEYPOINTS])  # (B, 2, 2)
        bullet_coords = self.bullet_affine(coords[:, _N_PLAYER_KEYPOINTS:])  # (B, 10, 2)
        player_position = player_coords.reshape(b, -1)  # (B, 4) [P1x,P1y,P2x,P2y]
        bullet_position = bullet_coords.reshape(b, -1)  # (B, 20) slot-major

        # Presence: per-bullet score-map confidence (logsumexp = soft-max activation) -> logit.
        bullet_scores = scores[:, _N_PLAYER_KEYPOINTS:]  # (B, 10, h, w)
        conf = torch.logsumexp(bullet_scores.reshape(b, _N_BULLET_KEYPOINTS, -1), dim=-1)  # (B,10)
        presence = conf * self.presence_scale + self.presence_bias  # (B, 10) logits

        flat = feat.reshape(b, -1)  # position-preserving flatten
        groups = {
            "player_position": player_position,
            "player_velocity": self.velocity_head(flat),
            "player_aim": self.aim_head(flat),
            "bullet_presence": presence,
            "bullet_position": bullet_position,
        }
        return groups, scores  # scores: the (B, K, h, w) logits for heatmap supervision


class _ProbeHeads(nn.Module):
    """A bank of independent FC heads, one per field group, off the pooled ``(B, D)`` embedding.

    Reads ``embed.detach()`` so it measures how much state survives pooling WITHOUT training the
    encoder. Disjoint from the spatial family in every parameter.
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, out),
                )
                for name, out in GROUP_OUTPUT_SIZES.items()
            }
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


class StateDecoder(nn.Module):
    """Encoder + localization spatial heads + detached embed-probe for single-frame decoding.

    :meth:`forward` returns ``{"spatial": per_group, "probe": per_group,
    "spatial_score_logits": (B,12,h,w)}``. The spatial family localizes on the trunk feature map
    (heatmap soft-argmax for positions, confidence for presence, flatten-FC for aim / velocity),
    so its gradient trains the encoder; the score logits are surfaced top-level for the auxiliary
    heatmap cross-entropy loss. The probe runs per-group FC heads on ``embed.detach()``
    (encoder-frozen). Head widths are discovered from a single zero-frame probe at ``input_hw`` so
    any encoder / resolution fits.
    """

    def __init__(self, encoder: Encoder, input_hw: tuple[int, int], *, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_hw = input_hw
        channels, flat_dim, embed_dim = self._discover_dims(input_hw)
        self.spatial_heads = _SpatialHeads(channels, flat_dim, hidden)
        self.probe_heads = _ProbeHeads(embed_dim, hidden)

    def _discover_dims(self, input_hw: tuple[int, int]) -> tuple[int, int, int]:
        """Probe the encoder with one zero frame to size the spatial + embed head inputs.

        Returns ``(channels, flat_dim, embed_dim)``: the feature-map channel count (for the
        score conv), the flattened feature-map width ``C*h*w`` (for the aim / velocity heads),
        and the pooled embedding width (for the probe heads).
        """
        h, w = input_hw
        device = next(self.encoder.parameters()).device
        probe = torch.zeros(1, self.encoder.trunk.in_channels, h, w, device=device)
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            feat = self.encoder.features(probe)
            emb = self.encoder.embed(probe)
        if was_training:
            self.encoder.train()
        channels = int(feat.shape[1])
        flat_dim = int(feat.shape[1] * feat.shape[2] * feat.shape[3])  # C*h*w
        embed_dim = int(emb.shape[1])
        return channels, flat_dim, embed_dim

    def forward(self, x: Tensor) -> dict[str, object]:
        """Decode per-group state for both head families.

        Returns ``{"spatial": ..., "probe": ..., "spatial_score_logits": ...}``. The score logits
        ``(B, 12, h, w)`` sit at TOP LEVEL (not inside ``spatial``) so :func:`evaluate`, which
        concatenates every key of ``out["spatial"]`` / ``out["probe"]`` over the whole val split,
        never accumulates them — eval memory stays flat while the heatmap loss can still read them.
        """
        feat = self.encoder.features(x)  # (B, C, h, w) — grad flows to the encoder
        spatial, scores = self.spatial_heads(feat)

        emb = self.encoder.embed(x).detach()  # (B, D); detach -> probe never trains the encoder
        probe = self.probe_heads(emb)
        return {"spatial": spatial, "probe": probe, "spatial_score_logits": scores}

    def spatial_parameters(self):
        """Parameters trained by the combined loss: the encoder + the spatial heads."""
        return list(self.encoder.parameters()) + list(self.spatial_heads.parameters())

    def probe_parameters(self):
        """Parameters trained by the probe loss: the probe heads ONLY (encoder is detached)."""
        return list(self.probe_heads.parameters())
