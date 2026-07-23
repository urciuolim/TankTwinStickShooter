"""Tests for ``should_eval_epoch`` (pure cadence helper) and val_trajectory in results.json.

Covers:
- Unit tests for every contract branch of should_eval_epoch.
- Smoke-train: val_trajectory entry count + structure for several (epochs, eval_every) combos.
- Training-unchanged guard: eval_every=0 produces val_trajectory=[] and loss_trajectory is
  present with the right length.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.pretraining.train import TrainConfig, run, should_eval_epoch  # noqa: E402

from ._fixtures import make_fixture  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests — should_eval_epoch (pure, no I/O)
# ---------------------------------------------------------------------------

EXPECTED_GROUP_KEYS = {"player_position", "player_aim", "bullet_presence", "bullet_position"}


class TestShouldEvalEpoch:
    def test_disabled_zero_never_returns_true(self):
        """eval_every=0: False for every epoch including the last."""
        for epoch in range(5):
            assert should_eval_epoch(epoch, eval_every=0, total_epochs=5) is False

    def test_disabled_negative_never_returns_true(self):
        """eval_every<0: False for every epoch."""
        for epoch in range(4):
            assert should_eval_epoch(epoch, eval_every=-1, total_epochs=4) is False

    def test_every_one_returns_true_for_all_epochs(self):
        """eval_every=1: True at every epoch (all are multiples of 1)."""
        for epoch in range(6):
            assert should_eval_epoch(epoch, eval_every=1, total_epochs=6) is True

    def test_every_two_hits_even_epochs_and_last(self):
        """eval_every=2, total=5: True at 0, 2, 4 (multiples) and 4 (last) — {0,2,4}."""
        total = 5
        hits = {e for e in range(total) if should_eval_epoch(e, eval_every=2, total_epochs=total)}
        assert hits == {0, 2, 4}

    def test_every_three_hits_multiples_and_last(self):
        """eval_every=3, total=7: multiples {0,3,6} plus last=6 -> {0,3,6}."""
        total = 7
        hits = {e for e in range(total) if should_eval_epoch(e, eval_every=3, total_epochs=total)}
        assert hits == {0, 3, 6}

    def test_last_epoch_always_hit_when_enabled(self):
        """Final epoch (total-1) is always True when eval_every>0, even if not a multiple."""
        # eval_every=3, total=5 -> multiples 0,3; last=4 which is not a multiple of 3
        total = 5
        assert should_eval_epoch(4, eval_every=3, total_epochs=total) is True

    def test_last_epoch_not_hit_when_disabled(self):
        """Final epoch is False when eval_every=0."""
        assert should_eval_epoch(4, eval_every=0, total_epochs=5) is False

    def test_non_multiple_mid_epoch_not_hit(self):
        """eval_every=3, total=7: epoch 1 is not a multiple and not the last."""
        assert should_eval_epoch(1, eval_every=3, total_epochs=7) is False

    def test_single_epoch_run_eval_every_one(self):
        """total=1: epoch 0 is both 0%1==0 and the last epoch."""
        assert should_eval_epoch(0, eval_every=1, total_epochs=1) is True

    def test_single_epoch_run_eval_every_two(self):
        """total=1, eval_every=2: epoch 0 is the last epoch, so True."""
        assert should_eval_epoch(0, eval_every=2, total_epochs=1) is True

    def test_large_eval_every_only_hits_zero_and_last(self):
        """eval_every=100 with total=5: only epoch 0 (multiple) and epoch 4 (last)."""
        total = 5
        hits = {e for e in range(total) if should_eval_epoch(e, eval_every=100, total_epochs=total)}
        assert hits == {0, 4}


# ---------------------------------------------------------------------------
# Smoke integration tests — val_trajectory in results.json
# ---------------------------------------------------------------------------


def _run_fixture(
    tmp_path,
    *,
    epochs: int,
    eval_every: int,
    seed: int = 0,
) -> dict:
    data_dir = tmp_path / "data"
    make_fixture(
        data_dir, n_maps=4, workers=2, shards_per_worker=2, hw=(360, 640), rows_per_shard=10
    )
    out_dir = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        trunk="cnn",
        pooling="gap",
        resolution=180,
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        seed=seed,
        device="cpu",
        eval_every=eval_every,
        progress=False,
    )
    return run(cfg)


def test_val_trajectory_every_epoch(tmp_path):
    """epochs=3, eval_every=1: 3 entries at epochs 0, 1, 2."""
    record = _run_fixture(tmp_path, epochs=3, eval_every=1)
    vt = record["val_trajectory"]
    assert len(vt) == 3
    assert [e["epoch"] for e in vt] == [0, 1, 2]


def test_val_trajectory_every_two(tmp_path):
    """epochs=3, eval_every=2: hits at 0 (multiple) and 2 (last) -> 2 entries."""
    record = _run_fixture(tmp_path, epochs=3, eval_every=2)
    vt = record["val_trajectory"]
    assert [e["epoch"] for e in vt] == [0, 2]


def test_val_trajectory_every_five_three_epochs(tmp_path):
    """epochs=3, eval_every=5: epoch 0 fires (0%5==0) and epoch 2 fires (last) -> 2 entries."""
    record = _run_fixture(tmp_path, epochs=3, eval_every=5)
    vt = record["val_trajectory"]
    # epoch 0 fires (0 % 5 == 0); epoch 2 fires (last); epoch 1 does not.
    assert [e["epoch"] for e in vt] == [0, 2]


def test_val_trajectory_entry_has_both_families(tmp_path):
    """Each val_trajectory entry carries both spatial and probe families with required groups."""
    record = _run_fixture(tmp_path, epochs=2, eval_every=1)
    for entry in record["val_trajectory"]:
        assert "spatial" in entry and "probe" in entry
        for fam in ("spatial", "probe"):
            keys = set(entry[fam])
            assert EXPECTED_GROUP_KEYS <= keys, f"{fam} missing keys: {EXPECTED_GROUP_KEYS - keys}"


def test_val_trajectory_bullet_presence_is_subdict(tmp_path):
    """bullet_presence within each family is a sub-dict with accuracy/precision/recall/f1."""
    record = _run_fixture(tmp_path, epochs=2, eval_every=1)
    for entry in record["val_trajectory"]:
        for fam in ("spatial", "probe"):
            bp = entry[fam]["bullet_presence"]
            assert isinstance(bp, dict)
            assert {"accuracy", "precision", "recall", "f1"} <= set(bp)


def test_val_trajectory_written_to_json(tmp_path):
    """val_trajectory is present in the strict-JSON results file."""
    data_dir = tmp_path / "data"
    make_fixture(
        data_dir, n_maps=4, workers=2, shards_per_worker=2, hw=(360, 640), rows_per_shard=10
    )
    out_dir = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        trunk="cnn",
        pooling="gap",
        resolution=180,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        seed=0,
        device="cpu",
        eval_every=1,
        progress=False,
    )
    run(cfg)
    loaded = json.loads((out_dir / "results.json").read_text())
    assert "val_trajectory" in loaded
    assert len(loaded["val_trajectory"]) == 2


# ---------------------------------------------------------------------------
# Training-unchanged guard — eval_every=0
# ---------------------------------------------------------------------------


def test_eval_every_zero_produces_empty_val_trajectory(tmp_path):
    """eval_every=0: val_trajectory is empty; loss_trajectory still has one entry per epoch."""
    record = _run_fixture(tmp_path, epochs=3, eval_every=0)
    assert record["val_trajectory"] == []
    assert len(record["loss_trajectory"]) == 3


def test_eval_every_zero_loss_trajectory_matches_shape(tmp_path):
    """A seeded eval_every=0 run has the same loss_trajectory shape as a same-seed eval_every=1
    run (same number of epochs, same keys per entry). Training steps are identical; eval is
    purely additive.
    """
    record_0 = _run_fixture(tmp_path / "r0", epochs=3, eval_every=0, seed=7)
    record_1 = _run_fixture(tmp_path / "r1", epochs=3, eval_every=1, seed=7)
    lt0 = record_0["loss_trajectory"]
    lt1 = record_1["loss_trajectory"]
    assert len(lt0) == len(lt1)
    for e0, e1 in zip(lt0, lt1):
        assert set(e0) == set(e1)
        # Losses should be numerically identical (eval is read-only; train_one_epoch re-enters
        # model.train() so the eval call between epochs cannot perturb the optimizer state).
        for k in e0:
            assert abs(e0[k] - e1[k]) < 1e-6, (
                f"epoch loss diverged between eval_every=0 and eval_every=1: "
                f"key={k} diff={abs(e0[k] - e1[k])}"
            )


def test_eval_every_zero_json_has_empty_val_trajectory(tmp_path):
    """val_trajectory key is present in the JSON but is an empty list when eval_every=0."""
    data_dir = tmp_path / "data"
    make_fixture(
        data_dir, n_maps=4, workers=2, shards_per_worker=2, hw=(360, 640), rows_per_shard=10
    )
    out_dir = tmp_path / "out"
    cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        trunk="cnn",
        pooling="gap",
        resolution=180,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        seed=0,
        device="cpu",
        eval_every=0,
        progress=False,
    )
    run(cfg)
    loaded = json.loads((out_dir / "results.json").read_text())
    assert loaded["val_trajectory"] == []
    assert "val_metrics" in loaded
    assert "test_metrics" in loaded
