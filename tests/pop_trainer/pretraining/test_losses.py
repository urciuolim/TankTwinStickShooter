"""Tests for pop_trainer.pretraining.losses — per-group decode losses (torch).

Hand-built tiny tensors verify each loss: aim cosine = 0 for identical directions, masked bullet
position ignores absent slots, presence BCE responds to pos_weight, and the combined sum adds the
enabled groups.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.pretraining import losses  # noqa: E402


def test_aim_cosine_zero_for_identical_directions():
    aim = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # P1=(1,0), P2=(0,1)
    loss = losses.aim_cosine_loss(aim, aim)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_aim_cosine_orthogonal_is_one_opposite_is_two():
    pred = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    ortho = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    opposite = torch.tensor([[-1.0, 0.0, -1.0, 0.0]])
    assert float(losses.aim_cosine_loss(pred, ortho)) == pytest.approx(1.0, abs=1e-5)
    assert float(losses.aim_cosine_loss(pred, opposite)) == pytest.approx(2.0, abs=1e-5)


def test_bullet_position_masked_mse_ignores_absent_slots():
    # 1 row, 2 slots (4 floats). Slot 0 present, slot 1 absent. Make slot1 pred wildly wrong;
    # it must not affect the loss because mask=0 there.
    pred = torch.tensor([[1.0, 1.0, 99.0, 99.0]])
    target = torch.tensor([[1.5, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss = losses.bullet_position_masked_mse(pred, target, mask)
    # only present floats: ((1-1.5)^2 + (1-1)^2)/2 = 0.25/2 = 0.125
    assert float(loss) == pytest.approx(0.125, abs=1e-6)


def test_bullet_position_masked_mse_no_present_is_zero():
    pred = torch.tensor([[5.0, 5.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[0.0, 0.0]])
    assert float(losses.bullet_position_masked_mse(pred, target, mask)) == 0.0


def test_presence_bce_responds_to_pos_weight():
    # All-zero logit (p=0.5). One positive target slot. Higher pos_weight -> higher loss.
    logits = torch.zeros(1, 4)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    low = float(losses.presence_bce_loss(logits, target, pos_weight=1.0))
    high = float(losses.presence_bce_loss(logits, target, pos_weight=10.0))
    assert high > low


def test_combined_loss_sums_enabled_groups():
    preds = {
        "player_position": torch.zeros(1, 4),
        "player_aim": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    }
    targets = {
        "player_position": torch.ones(1, 4),  # mse = 1.0
        "player_aim": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),  # cosine dist = 1.0
    }
    total, per_group = losses.combined_loss(preds, targets)
    assert set(per_group) == {"player_position", "player_aim"}
    assert float(total) == pytest.approx(2.0, abs=1e-5)
