"""Tests for pop_trainer.pretraining.decoder — the StateDecoder head families.

Output shapes for both head families on a tiny config + frame, and that the embed-probe reads
``embed.detach()`` (a probe-loss backward leaves the encoder grads at zero).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.models import EncoderConfig, build_encoder  # noqa: E402
from pop_trainer.pretraining import losses  # noqa: E402
from pop_trainer.pretraining.decoder import GROUP_OUTPUT_SIZES, StateDecoder  # noqa: E402

SMALL_HW = (180, 320)  # survives the NatureCNN stem+trio


def _decoder(trunk="nature", pooling="gap"):
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling))
    return StateDecoder(enc, SMALL_HW, hidden=32)


@pytest.mark.parametrize("pooling", ["gap", "flatten"])
def test_head_output_shapes_both_families(pooling):
    model = _decoder(pooling=pooling)
    x = torch.zeros(2, 3, *SMALL_HW)
    out = model(x)
    assert set(out) == {"spatial", "probe"}
    for fam in ("spatial", "probe"):
        for name, width in GROUP_OUTPUT_SIZES.items():
            assert out[fam][name].shape == (2, width), (fam, name)


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
