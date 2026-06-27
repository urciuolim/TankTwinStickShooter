"""The single-frame decoder: an :class:`~pop_trainer.models.Encoder` + disposable heads.

A supervised INVERSE-RENDERER. It wraps a config-built encoder and attaches two head families
that both decode the per-group state targets, so we can compare them:

* **spatial heads** read ``Encoder.features`` (the ``(B, C, h, w)`` map). Global-average-pooled
  to ``(B, C)``, then a per-group FC head. These heads' gradient flows INTO the encoder, so
  they are what trains the reusable artifact (heatmap read-out deferred; "good not best").
* the **embed-probe** reads ``Encoder.embed`` and DETACHES it (``embed.detach()``) before its
  own per-group FC heads, so the probe NEVER backprops into the encoder. It measures how much
  state survives pooling (GAP vs Flatten), reported alongside the spatial heads.

Both families produce the same per-group outputs (position 4, velocity 4, aim 4, presence 10
logits, bullet position 20). :meth:`forward` returns ``{"spatial": {...}, "probe": {...}}``.

Head input widths are sized from the encoder at build time by probing the trunk with a single
zero frame at the chosen resolution (device-correct), so any trunk/pooling/resolution composes.

torch + :mod:`pop_trainer.models` only; nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pop_trainer.models import Encoder

__all__ = ["GROUP_OUTPUT_SIZES", "StateDecoder"]

# Output width of each per-group head (both families share these).
GROUP_OUTPUT_SIZES: dict[str, int] = {
    "player_position": 4,
    "player_velocity": 4,
    "player_aim": 4,
    "bullet_presence": 10,
    "bullet_position": 20,
}


class _GroupHeads(nn.Module):
    """A bank of independent FC heads, one per field group, off a shared ``(B, D)`` input."""

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
    """Encoder + spatial heads + detached embed-probe for single-frame state decoding.

    :meth:`forward` returns ``{"spatial": per_group, "probe": per_group}``. The spatial heads
    GAP the trunk feature map to ``(B, C)`` then run per-group FC heads (encoder-training). The
    probe runs per-group FC heads on ``embed.detach()`` (encoder-frozen). Head widths are
    discovered from a single zero-frame probe at ``input_hw`` so any encoder/resolution fits.
    """

    def __init__(self, encoder: Encoder, input_hw: tuple[int, int], *, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_hw = input_hw
        spatial_dim, embed_dim = self._discover_dims(input_hw)
        self.spatial_heads = _GroupHeads(spatial_dim, hidden)
        self.probe_heads = _GroupHeads(embed_dim, hidden)

    def _discover_dims(self, input_hw: tuple[int, int]) -> tuple[int, int]:
        """Probe the encoder with one zero frame to size the spatial (GAP) + embed inputs."""
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
        spatial_dim = int(feat.shape[1])  # GAP collapses (B, C, h, w) -> (B, C)
        embed_dim = int(emb.shape[1])
        return spatial_dim, embed_dim

    def forward(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        """Decode per-group state for both head families: ``{"spatial": ..., "probe": ...}``."""
        feat = self.encoder.features(x)  # (B, C, h, w) — grad flows to the encoder
        pooled = feat.mean(dim=(2, 3))  # GAP -> (B, C)
        spatial = self.spatial_heads(pooled)

        emb = self.encoder.embed(x).detach()  # (B, D); detach -> probe never trains the encoder
        probe = self.probe_heads(emb)
        return {"spatial": spatial, "probe": probe}

    def spatial_parameters(self):
        """Parameters trained by the combined loss: the encoder + the spatial heads."""
        return list(self.encoder.parameters()) + list(self.spatial_heads.parameters())

    def probe_parameters(self):
        """Parameters trained by the probe loss: the probe heads ONLY (encoder is detached)."""
        return list(self.probe_heads.parameters())
