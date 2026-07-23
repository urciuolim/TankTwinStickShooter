"""Tests for pop_trainer.pretraining.metrics — per-group metrics (pure numpy).

Position error in world units after denorm, aim angular error (~0 identical / ~90 orthogonal),
presence F1, and masked bullet-position error.
"""

from __future__ import annotations

import numpy as np

from pop_trainer.pretraining import metrics
from pop_trainer.pretraining.targets import NormStats


def _stats() -> NormStats:
    # position std=2 mean=0 so a 0.5 normalized delta == 1.0 world unit per axis.
    four = np.zeros(4, dtype=np.float32)
    std4 = np.full(4, 2.0, dtype=np.float32)
    twenty = np.zeros(20, dtype=np.float32)
    std20 = np.full(20, 2.0, dtype=np.float32)
    return NormStats(four, std4, four.copy(), std4.copy(), twenty, std20)


def test_position_error_world_denormalizes():
    stats = _stats()
    # normalized delta of 0.5 on x only -> world delta = 0.5*std(2) = 1.0 -> L2 = 1.0 per point.
    pred = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.zeros((1, 4), dtype=np.float32)
    err = metrics.position_error_world(pred, target, stats)
    # two points (P1, P2); P1 off by 1.0, P2 by 0 -> mean = 0.5
    assert err == 0.5


def test_aim_angular_error_identical_and_orthogonal():
    # the eps floor in the cosine denominator leaves a hair of error at exact alignment; ~0.
    pred = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    assert metrics.aim_angular_error_deg(pred, pred) < 0.1
    ortho = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    assert abs(metrics.aim_angular_error_deg(pred, ortho) - 90.0) < 0.1
    opposite = np.array([[-1.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    assert abs(metrics.aim_angular_error_deg(pred, opposite) - 180.0) < 0.1


def test_presence_metrics_f1():
    # logits>0 predicts present. target present at idx 0,1. pred present at idx 0,2.
    logits = np.array([[5.0, -5.0, 5.0, -5.0]])
    target = np.array([[1.0, 1.0, 0.0, 0.0]])
    m = metrics.presence_metrics(logits, target)
    # tp=1 (idx0), fp=1 (idx2), fn=1 (idx1) -> precision=recall=0.5 -> f1=0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_presence_metrics_at_threshold_matches_zero():
    logits = np.array([[5.0, -5.0, 5.0, -5.0]])
    target = np.array([[1.0, 1.0, 0.0, 0.0]])
    assert metrics.presence_metrics_at_threshold(logits, target, 0.0) == metrics.presence_metrics(
        logits, target
    )


def test_presence_metrics_at_threshold_shifts_predictions():
    # logits 1,2,3,4; targets present at the two HIGH slots (3,4).
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    target = np.array([[0.0, 0.0, 1.0, 1.0]])
    # threshold 0 -> predicts all present: recall 1.0, precision 0.5.
    low = metrics.presence_metrics_at_threshold(logits, target, 0.0)
    assert low["recall"] == 1.0
    assert low["precision"] == 0.5
    # threshold 2.5 -> predicts only the two high slots present: perfect.
    high = metrics.presence_metrics_at_threshold(logits, target, 2.5)
    assert high["precision"] == 1.0
    assert high["recall"] == 1.0
    assert high["f1"] == 1.0


def test_select_presence_threshold_lifts_f1():
    # threshold-0 F1 is poor (many false positives); a positive threshold separates cleanly.
    logits = np.array([[1.0, 1.2, 0.8, 5.0, 6.0, 5.5]])
    target = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    f1_at_zero = metrics.presence_metrics(logits, target)["f1"]
    thr, sel = metrics.select_presence_threshold(logits, target)
    assert sel["f1"] > f1_at_zero
    assert sel["f1"] == 1.0
    # the selected threshold lands between the negative and positive clusters.
    assert 1.2 <= thr < 5.0
    # applying the SAME threshold to a held-out array with identical structure works.
    held = np.array([[0.9, 1.1, 5.2, 6.1]])
    held_tgt = np.array([[0.0, 0.0, 1.0, 1.0]])
    held_m = metrics.presence_metrics_at_threshold(held, held_tgt, thr)
    assert held_m["f1"] == 1.0


def test_select_presence_threshold_empty():
    thr, m = metrics.select_presence_threshold(np.zeros((0, 10)), np.zeros((0, 10)))
    assert thr == 0.0
    assert m == {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_select_presence_threshold_tiebreak_smallest():
    # all-present target: every candidate threshold below the data yields perfect F1; the
    # ascending scan keeps only STRICTLY greater F1, so the smallest tying threshold wins.
    logits = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[1.0, 1.0, 1.0]])
    thr, m = metrics.select_presence_threshold(logits, target)
    assert m["f1"] == 1.0
    # candidates are [min-1, 1, 2, 3]; min-1 == 0.0 already predicts all present (perfect),
    # so the tie-break picks it.
    assert thr == 0.0


def test_bullet_position_error_present_only():
    stats = _stats()  # 20-wide bullet stats, std=2
    # 1 row, 10 slots (20 floats). slot0 present, off by normalized 0.5 on x -> 1.0 world.
    # every other slot absent with a wildly wrong prediction that must be masked out.
    pred = np.full((1, 20), 9.0, dtype=np.float32)
    pred[0, 0] = 0.5
    pred[0, 1] = 0.0
    target = np.zeros((1, 20), dtype=np.float32)
    mask = np.zeros((1, 20), dtype=np.float32)
    mask[0, 0] = 1.0
    mask[0, 1] = 1.0
    # only slot0 counts -> world L2 = 1.0
    err = metrics.bullet_position_error_world(pred, target, mask, stats)
    assert err == 1.0


def test_metric_signature_works_for_both_families():
    # group_metrics takes the same args regardless of head family.
    stats = _stats()
    preds = {
        "player_position": np.zeros((2, 4), dtype=np.float32),
        "player_aim": np.tile([1.0, 0.0, 1.0, 0.0], (2, 1)).astype(np.float32),
        "bullet_presence": np.full((2, 10), -5.0, dtype=np.float32),
        "bullet_position": np.zeros((2, 20), dtype=np.float32),
    }
    targets = {
        "player_position": np.zeros((2, 4), dtype=np.float32),
        "player_aim": preds["player_aim"].copy(),
        "bullet_presence": np.zeros((2, 10), dtype=np.float32),
        "bullet_position": np.zeros((2, 20), dtype=np.float32),
        "bullet_slot_mask": np.zeros((2, 20), dtype=np.float32),
    }
    out = metrics.group_metrics(preds, targets, stats)
    assert out["player_position"] == 0.0
    assert out["player_aim"] < 0.1
    assert "f1" in out["bullet_presence"]
