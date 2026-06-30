"""Tests for the pure model-card assembly (pop_trainer.pretraining.card). No torch."""

from __future__ import annotations

import json
import math

import pytest

from pop_trainer.pretraining import card


def _architecture(**overrides) -> dict:
    base = {
        "trunk": "cnn",
        "pooling": "gap",
        "resolution": 180,
        "input_hw": [180, 320],
        "trunk_class": "CnnTrunk",
        "pooling_class": "GlobalAveragePool",
        "decoder_class": "StateDecoder",
        "encoder_params": 123456,
        "head_params": 7890,
        "total_params": 131346,
        "embedding_dim": 64,
    }
    base.update(overrides)
    return base


def _hyperparameters(**overrides) -> dict:
    base = {
        "lr": 3e-4,
        "batch_size": 32,
        "epochs": 8,
        "seed": 0,
        "heatmap_weight": 10.0,
        "heatmap_sigma": 1.5,
        "presence_pos_weight": 30.0,
        "optimizer": "Adam",
        "device": "cuda",
    }
    base.update(overrides)
    return base


def _dataset(**overrides) -> dict:
    base = {
        "name": "decode-v1",
        "manifest_id": "a" * 64,
        "path": "datasets/decode-v1",
    }
    base.update(overrides)
    return base


def _metrics(**overrides) -> dict:
    base = {
        "val": {"spatial": {"player_position": 0.5}, "probe": {"player_position": 0.7}},
        "test": {"spatial": {"player_position": 0.6}, "probe": {"player_position": 0.8}},
    }
    base.update(overrides)
    return base


def _provenance(**overrides) -> dict:
    base = {
        "git_commit": "deadbeef",
        "git_dirty": False,
        "weights_filename": "checkpoint.pt",
        "weights_sha256": "f" * 64,
    }
    base.update(overrides)
    return base


def _machine(**overrides) -> dict:
    base = {
        "hostname": "trainbox",
        "platform": "Windows-11",
        "system": "Windows",
        "release": "11",
        "arch": "AMD64",
        "processor": "Intel64",
        "cpu_count": 16,
        "ram_total_gb": 64.0,
        "python": "3.12.0",
    }
    base.update(overrides)
    return base


def _build(**overrides) -> dict:
    kwargs = {
        "name": "decode-smoke",
        "created_utc": "2026-06-29T00:00:00+00:00",
        "architecture": _architecture(),
        "hyperparameters": _hyperparameters(),
        "dataset": _dataset(),
        "metrics": _metrics(),
        "provenance": _provenance(),
        "machine": _machine(),
    }
    kwargs.update(overrides)
    return card.build_card(**kwargs)


def test_top_level_key_set():
    c = _build()
    assert set(c) == {
        "schema_version",
        "created_utc",
        "name",
        "descriptions",
        "architecture",
        "hyperparameters",
        "dataset",
        "metrics",
        "provenance",
        "machine",
    }
    assert c["schema_version"] == card.CARD_SCHEMA_VERSION
    assert card.CARD_NAME == "model_card.json"


def test_section_key_sets():
    c = _build()
    assert set(c["architecture"]) == {
        "trunk",
        "pooling",
        "resolution",
        "input_hw",
        "trunk_class",
        "pooling_class",
        "decoder_class",
        "encoder_params",
        "head_params",
        "total_params",
        "embedding_dim",
    }
    assert set(c["hyperparameters"]) == {
        "lr",
        "batch_size",
        "epochs",
        "seed",
        "heatmap_weight",
        "heatmap_sigma",
        "presence_pos_weight",
        "optimizer",
        "device",
    }
    assert set(c["dataset"]) == {"name", "manifest_id", "path"}
    assert set(c["metrics"]) == {"val", "test"}
    assert set(c["provenance"]) == {
        "git_commit",
        "git_dirty",
        "weights_filename",
        "weights_sha256",
    }
    assert set(c["machine"]) == {
        "hostname",
        "platform",
        "system",
        "release",
        "arch",
        "processor",
        "cpu_count",
        "ram_total_gb",
        "python",
    }


def test_strict_json_round_trip():
    c = _build()
    assert json.loads(json.dumps(c, allow_nan=False)) == c


def test_nonfinite_metric_nullified_real_zero_kept():
    c = _build(
        metrics={
            "val": {"a": float("nan"), "b": 0.0, "c": float("inf")},
            "test": {"d": float("-inf"), "e": 0.0},
        }
    )
    assert c["metrics"]["val"]["a"] is None
    assert c["metrics"]["val"]["b"] == 0.0
    assert c["metrics"]["val"]["c"] is None
    assert c["metrics"]["test"]["d"] is None
    assert c["metrics"]["test"]["e"] == 0.0
    # Result still strict-JSON-serializable after nullification.
    json.dumps(c, allow_nan=False)


def test_nonfinite_architecture_raises():
    with pytest.raises(ValueError, match="architecture"):
        _build(architecture=_architecture(encoder_params=123456, head_params=float("inf")))


def test_nonfinite_hyperparameter_raises():
    with pytest.raises(ValueError, match="hyperparameters"):
        _build(hyperparameters=_hyperparameters(lr=float("inf")))


def test_missing_required_field_raises():
    bad = _hyperparameters()
    del bad["lr"]
    with pytest.raises(ValueError, match="hyperparameters.*lr"):
        _build(hyperparameters=bad)


def test_missing_section_dict_raises():
    with pytest.raises(ValueError, match="dataset"):
        _build(dataset={"name": "x", "path": "y"})  # no manifest_id key


def test_null_manifest_id_passes():
    c = _build(dataset=_dataset(manifest_id=None))
    assert c["dataset"]["manifest_id"] is None


def test_null_git_provenance_passes():
    c = _build(provenance=_provenance(git_commit=None, git_dirty=None))
    assert c["provenance"]["git_commit"] is None
    assert c["provenance"]["git_dirty"] is None


def test_non_str_name_raises():
    with pytest.raises(ValueError, match="name"):
        _build(name=123)


def test_non_str_created_utc_raises():
    with pytest.raises(ValueError, match="created_utc"):
        _build(created_utc=123)


def test_descriptions_default_empty_list():
    c = _build()
    assert c["descriptions"] == []


def test_descriptions_passthrough():
    entries = [{"author": "human", "text": "note", "added_utc": "t0"}]
    c = _build(descriptions=entries)
    assert c["descriptions"] == entries
    assert c["descriptions"] is not entries  # copied to a fresh list


def test_real_zero_metric_stays_finite_not_nullified():
    c = _build(metrics={"val": {"f1": 0.0}, "test": {"f1": 1.0}})
    assert c["metrics"]["val"]["f1"] == 0.0
    assert c["metrics"]["test"]["f1"] == 1.0
    assert math.isfinite(c["metrics"]["val"]["f1"])


def test_add_description_appends_to_existing():
    c = _build(descriptions=[{"author": "human", "text": "a", "added_utc": "t0"}])
    out = card.add_description(c, author="claude", text="b", added_utc="t1")
    assert out["descriptions"] == [
        {"author": "human", "text": "a", "added_utc": "t0"},
        {"author": "claude", "text": "b", "added_utc": "t1"},
    ]


def test_add_description_creates_list_when_missing():
    bare = {"name": "x"}  # no descriptions key
    out = card.add_description(bare, author="human", text="hi", added_utc="t0")
    assert out["descriptions"] == [{"author": "human", "text": "hi", "added_utc": "t0"}]
    assert "descriptions" not in bare  # input untouched


def test_add_description_does_not_mutate_input():
    original = [{"author": "human", "text": "a", "added_utc": "t0"}]
    c = _build(descriptions=original)
    inner_before = c["descriptions"]
    out = card.add_description(c, author="claude", text="b", added_utc="t1")
    # Neither the input card, its nested list, nor the original list is mutated.
    assert len(c["descriptions"]) == 1
    assert c["descriptions"] is inner_before
    assert len(original) == 1
    assert out["descriptions"] is not inner_before
