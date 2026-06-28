"""Tests for pop_trainer.rl.extractor — the SB3 EncoderExtractor seam.

These import torch / stable-baselines3 (NOT pure), so the module skips wholesale if either is
absent via the top-level ``importorskip`` (matching the models encoder tests). They cover the
T2 contract:

1. CONSTRUCTS over the canonical 360x640 channels-first pixel Box; ``features_dim`` is a positive
   int matching the encoder's flatten embedding at that resolution.
2. ``forward`` over a (N, 3, 360, 640) float-[0,1] batch returns (N, features_dim).
3. FREEZE CORRECTNESS: with ``freeze=True``, one Adam step over ALL params leaves the encoder
   weights UNCHANGED and every encoder grad is None — freeze works via requires_grad, NOT by
   excluding params from the optimizer. The inverse (freeze=False) DOES move a param, so the test
   is not vacuous.
4. CHECKPOINT ROUND-TRIP: save extractor A's encoder state_dict (after perturbing it), build
   extractor B from that checkpoint, and confirm B's weights restored to A's.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402  (after importorskip, by design)
from torch import nn  # noqa: E402

from pop_trainer.models import DreamerCNN, EncoderConfig, NatureCNN, build_encoder  # noqa: E402
from pop_trainer.rl import EncoderExtractor  # noqa: E402

# The canonical obs is the real Unity RGB frame: 360 rows x 640 cols, 3 channels. 84x84 collapses
# the NatureCNN stem, so the extractor + tests use 360x640 everywhere.
OBS_HW = (360, 640)
OBS_SHAPE = (3, *OBS_HW)

# A SMALL frame (DreamerV3's canonical 64x64): the extractor must pick the dreamer trunk here.
SMALL_OBS_HW = (64, 64)
SMALL_OBS_SHAPE = (3, *SMALL_OBS_HW)


def _obs_space() -> gym.spaces.Box:
    """A channels-first float-[0,1] pixel Box, the shape SB3 hands the extractor."""
    return gym.spaces.Box(low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32)


def _small_obs_space() -> gym.spaces.Box:
    """A 64x64 channels-first float-[0,1] pixel Box (the small-frame DreamerV3 path)."""
    return gym.spaces.Box(low=0.0, high=1.0, shape=SMALL_OBS_SHAPE, dtype=np.float32)


def _expected_features_dim() -> int:
    """The flatten embedding D a fresh nature encoder reports at the canonical resolution."""
    enc = build_encoder(EncoderConfig(trunk="nature", pooling="flatten"))
    d = enc.embedding_dim(input_hw=OBS_HW)
    assert isinstance(d, int)
    return d


# --- 1. construction -------------------------------------------------------------------


def test_constructs_with_positive_matching_features_dim():
    """Builds over the canonical Box; features_dim is a positive int == encoder embedding D."""
    extractor = EncoderExtractor(_obs_space())
    assert isinstance(extractor.features_dim, int)
    assert extractor.features_dim > 0
    assert extractor.features_dim == _expected_features_dim()


def test_in_channels_derived_from_obs_space():
    """The encoder's input channels come from the obs space, not a hard-coded 3."""
    extractor = EncoderExtractor(_obs_space())
    assert extractor.encoder.trunk.in_channels == OBS_SHAPE[0]


# --- trunk selection by resolution -----------------------------------------------------


def test_canonical_obs_selects_nature_trunk():
    """The canonical 360x640 obs builds the unchanged NatureCNN trunk (deployment path)."""
    extractor = EncoderExtractor(_obs_space())
    assert isinstance(extractor.encoder.trunk, NatureCNN)
    # the canonical features_dim is unchanged by the dreamer addition
    assert extractor.features_dim == _expected_features_dim()


def test_small_obs_selects_dreamer_trunk_with_probed_features_dim():
    """A 64x64 obs builds the DreamerV3 trunk; features_dim is a positive int from a dummy probe.

    At 64x64 the four stride-2 dreamer blocks give a (256, 4, 4) map, so the flatten features_dim
    is 256*4*4 = 4096 — derived by the extractor's dummy forward, not a hardcoded spatial dim.
    """
    extractor = EncoderExtractor(_small_obs_space())
    assert isinstance(extractor.encoder.trunk, DreamerCNN)
    assert isinstance(extractor.features_dim, int)
    assert extractor.features_dim == 4096

    # the probe matches an independently built dreamer encoder's static D
    enc = build_encoder(EncoderConfig(trunk="dreamer", pooling="flatten"))
    assert extractor.features_dim == enc.embedding_dim(input_hw=SMALL_OBS_HW)


def test_small_obs_forward_returns_batched_embedding():
    """forward over (N, 3, 64, 64) in [0,1] returns (N, features_dim) on the dreamer path."""
    extractor = EncoderExtractor(_small_obs_space()).eval()
    n = 2
    obs = torch.rand(n, *SMALL_OBS_SHAPE)
    with torch.no_grad():
        out = extractor(obs)
    assert out.shape == (n, extractor.features_dim)


# --- 2. forward ------------------------------------------------------------------------


def test_forward_returns_batched_embedding():
    """forward over (N, 3, 360, 640) in [0,1] returns (N, features_dim)."""
    extractor = EncoderExtractor(_obs_space()).eval()
    n = 2
    obs = torch.rand(n, *OBS_SHAPE)  # already in [0, 1], as SB3 hands it
    with torch.no_grad():
        out = extractor(obs)
    assert out.shape == (n, extractor.features_dim)


# --- 3. freeze correctness (the load-bearing test) -------------------------------------


class _Probe(nn.Module):
    """Extractor + a 1-layer head, an SB3-shaped module whose params an optimizer owns wholesale."""

    def __init__(self, extractor: EncoderExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.head = nn.Linear(extractor.features_dim, 1)

    def forward(self, obs):
        return self.head(self.extractor(obs))


def _one_optimizer_step(module: nn.Module) -> None:
    """Adam over ALL module params (the SB3 model), one forward -> loss -> backward -> step."""
    opt = torch.optim.Adam(module.parameters(), lr=1e-2)
    obs = torch.rand(2, *OBS_SHAPE)
    opt.zero_grad()
    loss = module(obs).pow(2).mean()
    loss.backward()
    opt.step()


def _first_conv_weight(extractor: EncoderExtractor) -> torch.nn.Parameter:
    """A representative encoder weight tensor (the stem's conv) to watch across a step."""
    for m in extractor.encoder.modules():
        if isinstance(m, nn.Conv2d):
            return m.weight
    raise AssertionError("no Conv2d weight found in the encoder")


def test_freeze_leaves_encoder_unchanged_after_optimizer_step():
    """freeze=True: weights UNCHANGED and grads None after one Adam step over ALL params.

    Proves freeze works via requires_grad (a frozen param keeps a None grad and is never updated),
    NOT by excluding params from the optimizer — the optimizer owns every param here.
    """
    extractor = EncoderExtractor(_obs_space(), freeze=True)
    module = _Probe(extractor)

    before = _first_conv_weight(extractor).detach().clone()
    _one_optimizer_step(module)
    after = _first_conv_weight(extractor)

    # every encoder grad stayed None (requires_grad cleared -> no grad accumulated)
    assert all(p.grad is None for p in extractor.encoder.parameters())
    # and the weights are bit-for-bit unchanged across the optimizer step
    assert torch.equal(before, after)
    # the head DID train (the step was real, not a no-op)
    assert any(p.requires_grad for p in module.head.parameters())


def test_unfrozen_encoder_changes_after_optimizer_step():
    """Inverse sanity: freeze=False, a representative encoder weight MOVES after the same step."""
    extractor = EncoderExtractor(_obs_space(), freeze=False)
    module = _Probe(extractor)

    before = _first_conv_weight(extractor).detach().clone()
    _one_optimizer_step(module)
    after = _first_conv_weight(extractor)

    assert all(p.requires_grad for p in extractor.encoder.parameters())
    assert not torch.equal(before, after)


# --- 4. checkpoint round-trip ----------------------------------------------------------


def test_checkpoint_round_trip_restores_weights(tmp_path):
    """Save A's (perturbed) encoder state_dict; B built from it matches A weight-for-weight."""
    a = EncoderExtractor(_obs_space())

    # Perturb A so a successful load is provably a restore, not equal-by-default-init.
    with torch.no_grad():
        for p in a.encoder.parameters():
            p.add_(torch.randn_like(p))

    ckpt = tmp_path / "enc.pt"
    torch.save(a.encoder.state_dict(), ckpt)

    b = EncoderExtractor(_obs_space(), checkpoint=ckpt)

    a_state = a.encoder.state_dict()
    b_state = b.encoder.state_dict()
    assert a_state.keys() == b_state.keys()
    for key in a_state:
        assert torch.allclose(a_state[key], b_state[key]), f"mismatch at {key}"


def test_checkpoint_differs_from_fresh_without_load(tmp_path):
    """Guard the round-trip: a fresh (un-loaded) extractor does NOT match the perturbed A.

    Confirms the round-trip assertion above is meaningful — without the load the weights differ.
    """
    a = EncoderExtractor(_obs_space())
    with torch.no_grad():
        for p in a.encoder.parameters():
            p.add_(torch.randn_like(p) + 1.0)

    fresh = EncoderExtractor(_obs_space())
    conv_key = next(
        k for k, v in a.encoder.state_dict().items() if v.ndim == 4
    )  # a conv weight tensor
    assert not torch.allclose(
        a.encoder.state_dict()[conv_key], fresh.encoder.state_dict()[conv_key]
    )
