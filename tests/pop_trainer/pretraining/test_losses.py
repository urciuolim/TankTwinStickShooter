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


# --- heatmap cross-entropy -------------------------------------------------------------------

H, W, K = 4, 5, 12


def _cell_center(i, j):
    return torch.tensor([(j + 0.5) / W, (i + 0.5) / H])


def test_heatmap_ce_is_finite():
    logits = torch.randn(2, K, H, W)
    grid = torch.rand(2, K, 2)
    present = torch.ones(2, K)
    loss = losses.heatmap_ce_loss(logits, grid, present, sigma=0.7)
    assert torch.isfinite(loss).all()
    assert float(loss) > 0.0


def test_heatmap_ce_rewards_matching_the_target_at_the_right_cell():
    # Forward CE against a SOFT Gaussian target is mass-COVERING: it rewards a prediction that
    # covers the target's spread, not an over-sharp delta (a near-delta assigns ~0 mass to the
    # target's neighbor cells and can score WORSE than uniform — correct CE behavior, not a bug).
    # So assert the two meaningful properties: (1) a prediction matching the target Gaussian beats
    # a flat/uniform map; (2) a peak at the CORRECT cell beats a peak at the WRONG cell.
    i, j = 2, 3
    grid = _cell_center(i, j).reshape(1, 1, 2)
    present = torch.ones(1, 1)

    flat = torch.zeros(1, 1, H, W)
    ce_flat = float(losses.heatmap_ce_loss(flat, grid, present, sigma=0.7))

    # matched: logits == the target log-density, so softmax(logits) == the target Gaussian (same
    # per-axis sigma normalization + cell-center convention as heatmap_ce_loss).
    sfx, sfy = 0.7 / W, 0.7 / H
    xs = (torch.arange(W) + 0.5) / W
    ys = (torch.arange(H) + 0.5) / H
    dx = (xs.view(1, W) - grid[0, 0, 0]) / sfx
    dy = (ys.view(H, 1) - grid[0, 0, 1]) / sfy
    matched = (-0.5 * (dx * dx + dy * dy)).reshape(1, 1, H, W)
    ce_matched = float(losses.heatmap_ce_loss(matched, grid, present, sigma=0.7))
    assert ce_matched < ce_flat

    # a peak at the true cell beats the same peak at a wrong cell (directional localization signal).
    correct = torch.full((1, 1, H, W), -10.0)
    correct[0, 0, i, j] = 10.0
    wrong = torch.full((1, 1, H, W), -10.0)
    wrong[0, 0, 0, 0] = 10.0
    ce_correct = float(losses.heatmap_ce_loss(correct, grid, present, sigma=0.7))
    ce_wrong = float(losses.heatmap_ce_loss(wrong, grid, present, sigma=0.7))
    assert ce_correct < ce_wrong


def test_heatmap_ce_masks_absent_keypoints():
    # two keypoints, only the first present; garbage logits on the absent one must not move loss.
    grid = torch.stack([_cell_center(1, 1), _cell_center(0, 0)]).reshape(1, 2, 2)
    present = torch.tensor([[1.0, 0.0]])
    base = torch.randn(1, 2, H, W)
    loss_a = losses.heatmap_ce_loss(base, grid, present, sigma=0.7)
    garbled = base.clone()
    garbled[0, 1] = torch.randn(H, W) * 100.0  # only the ABSENT keypoint changes
    loss_b = losses.heatmap_ce_loss(garbled, grid, present, sigma=0.7)
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)


def test_heatmap_ce_no_present_is_grad_safe_zero():
    logits = torch.randn(1, 2, H, W, requires_grad=True)
    grid = torch.rand(1, 2, 2)
    present = torch.zeros(1, 2)
    loss = losses.heatmap_ce_loss(logits, grid, present, sigma=0.7)
    assert float(loss) == 0.0
    loss.backward()  # grad-connected zero: backward does not raise
    assert logits.grad is not None


def test_heatmap_ce_gradient_flows_to_logits():
    logits = torch.randn(2, K, H, W, requires_grad=True)
    grid = torch.rand(2, K, 2)
    present = torch.ones(2, K)
    losses.heatmap_ce_loss(logits, grid, present, sigma=0.7).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_combined_loss_adds_heatmap_with_score_logits():
    preds = {"player_position": torch.zeros(1, 4)}  # mse vs ones = 1.0
    base_targets = {"player_position": torch.ones(1, 4)}
    kp_targets = {
        **base_targets,
        "keypoint_grid": torch.rand(1, K, 2),
        "keypoint_present": torch.ones(1, K),
    }
    logits = torch.randn(1, K, H, W)
    without_total, without_pg = losses.combined_loss(preds, base_targets)
    with_total, with_pg = losses.combined_loss(
        preds, kp_targets, score_logits=logits, heatmap_weight=6.0, heatmap_sigma=0.7
    )
    # WITH score_logits: a "heatmap" per_group key (unweighted ce) and a strictly larger total.
    assert "heatmap" in with_pg
    assert float(with_pg["heatmap"]) > 0.0
    assert float(with_total) > float(without_total)
    # the heatmap term is weighted by its own scalar, NOT folded into per_group's unweighted ce.
    expected = float(without_total) + 6.0 * float(with_pg["heatmap"])
    assert float(with_total) == pytest.approx(expected, abs=1e-5)
    # WITHOUT score_logits: byte-identical to today, no "heatmap" key.
    assert "heatmap" not in without_pg


def test_probe_loss_never_adds_heatmap():
    # probe_loss calls combined_loss WITHOUT score_logits, so keypoint targets are ignored.
    preds = {"player_position": torch.zeros(1, 4)}
    targets = {
        "player_position": torch.ones(1, 4),
        "keypoint_grid": torch.rand(1, K, 2),
        "keypoint_present": torch.ones(1, K),
    }
    _, per_group = losses.probe_loss(preds, targets)
    assert "heatmap" not in per_group
