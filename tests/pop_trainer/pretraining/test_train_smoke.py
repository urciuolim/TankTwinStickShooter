"""Fast smoke-train tests for pop_trainer.pretraining.train.

A tiny-tensor train loop asserting the combined loss DECREASES over a few steps, plus a full
end-to-end `run` on the temp-dir fixture proving the loop runs and writes a strict-JSON record.
Kept CPU + tiny so it runs in a few seconds.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.models import EncoderConfig, build_encoder  # noqa: E402
from pop_trainer.pretraining import losses  # noqa: E402
from pop_trainer.pretraining import train as train_mod  # noqa: E402
from pop_trainer.pretraining.decoder import StateDecoder  # noqa: E402
from pop_trainer.pretraining.train import TrainConfig, run  # noqa: E402

from ._fixtures import make_fixture  # noqa: E402

SMALL_HW = (180, 320)


def test_train_config_split_frac_defaults():
    cfg = TrainConfig(data_dir="d", out_dir="o")
    assert cfg.val_frac == 0.2
    assert cfg.test_frac == 0.2


def test_main_parses_split_frac_args(monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        empty_fam = {"spatial": {}, "probe": {}}
        return {"loss_trajectory": [], "val_metrics": empty_fam, "test_metrics": empty_fam}

    monkeypatch.setattr(train_mod, "run", fake_run)
    rc = train_mod.main(["--val-frac", "0.3", "--test-frac", "0.25"])
    assert rc == 0
    assert captured["cfg"].val_frac == 0.3
    assert captured["cfg"].test_frac == 0.25
    # default when omitted
    train_mod.main([])
    assert captured["cfg"].val_frac == 0.2
    assert captured["cfg"].test_frac == 0.2


def test_parse_group_weights_valid():
    parsed = train_mod.parse_group_weights(["player_aim=5.0", "bullet_presence=2"])
    assert parsed == {"player_aim": 5.0, "bullet_presence": 2.0}


def test_parse_group_weights_empty_and_none():
    assert train_mod.parse_group_weights([]) == {}
    assert train_mod.parse_group_weights(None) == {}


def test_parse_group_weights_unknown_group():
    with pytest.raises(ValueError, match="unknown group"):
        train_mod.parse_group_weights(["not_a_group=1.0"])


def test_parse_group_weights_malformed_token():
    with pytest.raises(ValueError, match="malformed"):
        train_mod.parse_group_weights(["player_aim"])
    with pytest.raises(ValueError, match="malformed"):
        train_mod.parse_group_weights(["player_aim=abc"])


def test_main_parses_group_and_pos_weight_args(monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        empty_fam = {"spatial": {}, "probe": {}}
        return {"loss_trajectory": [], "val_metrics": empty_fam, "test_metrics": empty_fam}

    monkeypatch.setattr(train_mod, "run", fake_run)
    train_mod.main(["--group-weights", "player_aim=5.0", "--presence-pos-weight", "12.5"])
    assert captured["cfg"].group_weights == {"player_aim": 5.0}
    assert captured["cfg"].presence_pos_weight == 12.5
    # omitting both leaves the default (byte-identical) path: None / None.
    train_mod.main([])
    assert captured["cfg"].group_weights is None
    assert captured["cfg"].presence_pos_weight is None


def test_combined_loss_decreases_over_steps():
    torch.manual_seed(0)
    model = StateDecoder(build_encoder(EncoderConfig("cnn", "gap")), SMALL_HW, hidden=32)
    opt = torch.optim.Adam(model.spatial_parameters(), lr=1e-2)
    # one fixed batch the model should overfit: a constant frame -> constant targets.
    x = torch.rand(4, 3, *SMALL_HW)
    targets = {
        "player_position": torch.randn(4, 4),
        "player_velocity": torch.randn(4, 4),
        "player_aim": torch.nn.functional.normalize(torch.randn(4, 2, 2), dim=2).reshape(4, 4),
        "bullet_presence": (torch.rand(4, 10) < 0.3).float(),
        "bullet_position": torch.randn(4, 20),
        "bullet_slot_mask": torch.ones(4, 20),
    }
    first = last = None
    model.train()
    for step in range(20):
        out = model(x)
        total, _ = losses.combined_loss(out["spatial"], targets, presence_pos_weight=2.0)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step == 0:
            first = float(total.detach())
        last = float(total.detach())
    assert last < first  # the encoder + spatial heads learned the fixed batch


def test_run_end_to_end_writes_strict_json(tmp_path):
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
        resolution=180,  # cnn underflows a 90-row frame; 180 is the smallest cnn-valid res
        epochs=2,
        batch_size=8,
        lr=1e-3,
        seed=0,
        device="cpu",
    )
    record = run(cfg)
    # checkpoint + results written
    assert (out_dir / "checkpoint.pt").exists()
    results = out_dir / "results.json"
    assert results.exists()
    # strict JSON: no NaN/Inf (json.loads rejects them only if we did not write them)
    loaded = json.loads(results.read_text())
    assert "loss_trajectory" in loaded
    assert "spatial" in loaded["val_metrics"] and "probe" in loaded["val_metrics"]
    # both head families reported every group
    for fam in ("spatial", "probe"):
        groups = set(loaded["test_metrics"][fam])
        assert {"player_position", "player_aim", "bullet_presence", "bullet_position"} <= groups
    # loss trajectory has one entry per epoch
    assert len(record["loss_trajectory"]) == 2
    # presence calibration: spatial family, a selected threshold, and val/test metric dicts.
    cal = loaded["presence_calibration"]
    assert cal["family"] == "spatial"
    assert isinstance(cal["selected_threshold"], float)
    for split in ("val", "test"):
        assert {"accuracy", "precision", "recall", "f1"} <= set(cal[split])
    # self-describing config carries the effective per-group weights + the pos_weight override.
    cfg_json = loaded["config"]
    assert set(cfg_json["group_weights"]) == set(losses.DEFAULT_GROUP_WEIGHTS)
    assert cfg_json["presence_pos_weight_override"] is None
