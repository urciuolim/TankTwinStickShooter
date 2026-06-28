"""Tests for pop_trainer.pretraining.decoder — the StateDecoder head families.

Soft-argmax localization correctness + its gradient flow; output shapes for both head families
on a tiny config + frame; and that the embed-probe reads ``embed.detach()`` (a probe-loss
backward leaves the encoder grads at zero) while the spatial family DOES train the encoder.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.models import EncoderConfig, build_encoder  # noqa: E402
from pop_trainer.pretraining import losses  # noqa: E402
from pop_trainer.pretraining.decoder import (  # noqa: E402
    _N_BULLET_KEYPOINTS,
    _N_KEYPOINTS,
    _N_PLAYER_KEYPOINTS,
    GROUP_OUTPUT_SIZES,
    StateDecoder,
    _CoordAffine,
    soft_argmax,
)

SMALL_HW = (180, 320)  # survives the NatureCNN stem+trio


def _decoder(trunk="nature", pooling="gap"):
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling))
    return StateDecoder(enc, SMALL_HW, hidden=32)


def test_soft_argmax_one_hot_returns_cell_center():
    # A sharply-peaked map at cell (i, j) -> the cell-center normalized coord ((j+.5)/w, (i+.5)/h).
    h, w = 4, 5
    i, j = 2, 3
    scores = torch.full((1, 1, h, w), -50.0)
    scores[0, 0, i, j] = 50.0  # near one-hot after spatial softmax
    coords = soft_argmax(scores)
    assert coords.shape == (1, 1, 2)
    expected = torch.tensor([(j + 0.5) / w, (i + 0.5) / h])
    assert torch.allclose(coords[0, 0], expected, atol=1e-3)


def test_soft_argmax_uniform_returns_grid_center():
    h, w = 4, 6
    scores = torch.zeros(1, 1, h, w)  # uniform -> distribution mean is the grid center
    coords = soft_argmax(scores)
    center = torch.tensor([w / 2.0, h / 2.0]) / torch.tensor([float(w), float(h)])
    assert torch.allclose(coords[0, 0], center, atol=1e-6)


def test_soft_argmax_is_differentiable():
    scores = torch.randn(2, 3, 4, 5, requires_grad=True)
    coords = soft_argmax(scores)
    coords.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_coord_affine_is_per_keypoint_and_broadcasts_over_batch():
    # Per-keypoint affine: scale/bias are (k, 2), and a (B, k, 2) coord forward broadcasts the
    # params over the batch dim, returning (B, k, 2).
    for k in (_N_PLAYER_KEYPOINTS, _N_BULLET_KEYPOINTS):
        affine = _CoordAffine(k)
        assert affine.scale.shape == (k, 2)
        assert affine.bias.shape == (k, 2)
        coords = torch.rand(4, k, 2)
        out = affine(coords)
        assert out.shape == (4, k, 2)


def test_spatial_heads_use_per_keypoint_affines():
    model = _decoder()
    assert model.spatial_heads.player_affine.scale.shape == (_N_PLAYER_KEYPOINTS, 2)
    assert model.spatial_heads.bullet_affine.scale.shape == (_N_BULLET_KEYPOINTS, 2)


@pytest.mark.parametrize("trunk", ["nature", "impala"])
@pytest.mark.parametrize("pooling", ["gap", "flatten"])
def test_head_output_shapes_both_families(trunk, pooling):
    model = _decoder(trunk=trunk, pooling=pooling)
    x = torch.zeros(2, 3, *SMALL_HW)
    out = model(x)
    assert set(out) == {"spatial", "probe", "spatial_score_logits"}
    for fam in ("spatial", "probe"):
        for name, width in GROUP_OUTPUT_SIZES.items():
            assert out[fam][name].shape == (2, width), (fam, name)
    # the score logits ride at TOP LEVEL (never inside a family dict, so eval never concats them).
    feat = model.encoder.features(x)
    fh, fw = feat.shape[2], feat.shape[3]
    assert out["spatial_score_logits"].shape == (2, _N_KEYPOINTS, fh, fw)


def test_probe_loss_does_not_backprop_into_encoder():
    model = _decoder()
    x = torch.zeros(3, 3, *SMALL_HW)
    targets = {
        "player_position": torch.zeros(3, 4),
        "player_velocity": torch.zeros(3, 4),
        "player_aim": torch.full((3, 4), 0.5),
        "bullet_presence": torch.zeros(3, 10),
        "bullet_position": torch.zeros(3, 20),
        "bullet_slot_mask": torch.ones(3, 20),
    }
    model.zero_grad(set_to_none=True)
    out = model(x)
    total, _ = losses.probe_loss(out["probe"], targets, presence_pos_weight=2.0)
    total.backward()
    # encoder params received NO gradient (detached embed); probe heads did.
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads == []
    probe_grads = [p.grad for p in model.probe_heads.parameters() if p.grad is not None]
    assert len(probe_grads) > 0


def test_spatial_loss_does_backprop_into_encoder():
    model = _decoder()
    x = torch.zeros(3, 3, *SMALL_HW)
    targets = {
        "player_position": torch.ones(3, 4),
        "player_velocity": torch.zeros(3, 4),
        "player_aim": torch.full((3, 4), 0.5),
        "bullet_presence": torch.zeros(3, 10),
        "bullet_position": torch.zeros(3, 20),
        "bullet_slot_mask": torch.ones(3, 20),
    }
    model.zero_grad(set_to_none=True)
    out = model(x)
    total, _ = losses.combined_loss(out["spatial"], targets, presence_pos_weight=2.0)
    total.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0  # spatial heads DO train the encoder


def test_softargmax_position_path_trains_encoder():
    # The localization (soft-argmax) path alone must reach the encoder: restrict the loss to
    # ONLY the position groups (player_position + bullet_position), which are produced purely by
    # the score-conv -> spatial-softmax -> soft-argmax -> affine path (no aim/velocity/presence
    # head involved). A grad on the encoder then proves the localization path trains the artifact.
    model = _decoder()
    x = torch.randn(3, 3, *SMALL_HW)  # nonzero so the score conv has signal to localize on
    out = model(x)
    pos_only = {
        "player_position": out["spatial"]["player_position"],
        "bullet_position": out["spatial"]["bullet_position"],
    }
    targets = {
        "player_position": torch.ones(3, 4),
        "bullet_position": torch.ones(3, 20),
        "bullet_slot_mask": torch.ones(3, 20),
    }
    model.zero_grad(set_to_none=True)
    total, per_group = losses.combined_loss(pos_only, targets)
    # The combined loss carries ONLY the two position groups (nothing else can leak gradient).
    assert set(per_group) == {"player_position", "bullet_position"}
    total.backward()

    # The encoder received gradient strictly via the soft-argmax localization path.
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0
    assert all(torch.isfinite(g).all() for g in enc_grads)
    # The score conv (the head of the localization path) trained; the aim/velocity flatten heads
    # and the presence affine did NOT (they were not in the loss), confirming the path isolation.
    assert model.spatial_heads.score_conv[-1].weight.grad is not None
    assert model.spatial_heads.aim_head[0].weight.grad is None
    assert model.spatial_heads.velocity_head[0].weight.grad is None
    assert model.spatial_heads.presence_scale.grad is None


def test_heatmap_only_loss_trains_encoder():
    # The auxiliary heatmap CE alone (no coord groups) must reach the encoder via the score maps,
    # proving the new term trains the reusable artifact.
    model = _decoder()
    x = torch.randn(3, 3, *SMALL_HW)  # nonzero signal for the score conv to localize on
    out = model(x)
    targets = {
        "keypoint_grid": torch.rand(3, _N_KEYPOINTS, 2),
        "keypoint_present": torch.ones(3, _N_KEYPOINTS),
    }
    model.zero_grad(set_to_none=True)
    total, per_group = losses.combined_loss({}, targets, score_logits=out["spatial_score_logits"])
    # ONLY the heatmap term contributes (no coord groups passed).
    assert set(per_group) == {"heatmap"}
    total.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0
    assert all(torch.isfinite(g).all() for g in enc_grads)
    # the score conv (head of the localization path) trained from the heatmap term.
    assert model.spatial_heads.score_conv[-1].weight.grad is not None
