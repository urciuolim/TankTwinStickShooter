"""``EncoderExtractor`` — the seam between the env's pixel obs and the SB3 policy/value heads.

An :class:`EncoderExtractor` is a Stable-Baselines3 ``BaseFeaturesExtractor`` that wraps the
shared vision :class:`~pop_trainer.models.Encoder`, so SB3's ``CnnPolicy`` reads the SAME flat
embedding the supervised pretraining produced. The architecture is FIXED for Phase 1 — a
NatureCNN trunk with flatten pooling (``EncoderConfig(trunk="nature", pooling="flatten")``) at the
canonical 360x640 frame. The trunk / pooling / resolution ablation lives in the ``models``
benchmark, NOT here, so this extractor exposes no architecture knobs.

The extractor OWNS the pretrained-encoder load: given a ``checkpoint`` it ``torch.load``s a raw
encoder ``state_dict`` and applies it (there is no ``models.from_pretrained``). ``freeze`` stops the
encoder from learning by clearing ``requires_grad`` on its parameters — SB3 builds its optimizer
over ALL policy params, so freezing works through the grad path (a frozen param keeps a ``None``
grad and is never updated), NOT by excluding params from the optimizer.

Boundary: imports ``pop_trainer.models`` (the encoder factory) plus torch / gymnasium / SB3 /
stdlib. Imports nothing from ``data`` or ``pretraining``; no cycles.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor

from pop_trainer.models import EncoderConfig, build_encoder

__all__ = ["EncoderExtractor"]


class EncoderExtractor(BaseFeaturesExtractor):
    """SB3 features extractor wrapping the shared NatureCNN+flatten :class:`Encoder`.

    Constructed from SB3's channels-first ``(C, H, W)`` pixel observation space. The encoder is
    built with ``in_channels`` taken from the obs space (the canonical RGB frame is 3) and the
    ``features_dim`` is the encoder's flat embedding size probed at the obs ``H x W``.

    Args:
        observation_space: the env's pixel obs space, a channels-first ``Box`` with
            ``shape == (C, H, W)``.
        checkpoint: optional path to a raw encoder ``state_dict`` (saved via
            ``torch.save(encoder.state_dict(), path)``); when given it is loaded into the encoder.
        freeze: when true, clears ``requires_grad`` on every encoder parameter so the pretrained
            features stay fixed during RL (the frozen params keep ``None`` grads under SB3's
            optimizer, never updating).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        checkpoint: Path | None = None,
        freeze: bool = False,
    ) -> None:
        # SB3 hands a channels-first pixel Box: shape == (C, H, W).
        channels, height, width = observation_space.shape

        encoder = build_encoder(
            EncoderConfig(trunk="nature", pooling="flatten", in_channels=int(channels))
        )
        # Flatten embedding size at this obs resolution (an int for a known H x W).
        features_dim = encoder.embedding_dim(input_hw=(int(height), int(width)))
        if not isinstance(features_dim, int):
            raise ValueError(
                "encoder embedding_dim is not statically knowable at "
                f"({height}, {width}); got {features_dim!r}"
            )

        super().__init__(observation_space, features_dim)
        # Register the encoder AFTER super().__init__ runs nn.Module's __init__.
        self.encoder = encoder

        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            self.encoder.load_state_dict(state)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, observations: Tensor) -> Tensor:
        """Return the flat embedding ``(N, features_dim)`` for SB3's ``(N, C, H, W)`` obs.

        SB3's preprocessing has already cast the uint8 frame to float and divided by 255, so the
        observations arrive in ``[0, 1]`` — do not re-normalize.
        """
        return self.encoder(observations)
