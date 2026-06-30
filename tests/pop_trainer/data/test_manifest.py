"""Tests for the PURE manifest assembly: shape, JSON round-trip, validation, normalization."""

import json

import pytest

from pop_trainer.data import manifest


def _collection(**overrides):
    base = {
        "seed": 7,
        "command": ["collect", "--seed", "7"],
        "workers": 2,
        "episodes": 4,
        "max_steps": 1000,
        "maps": ["Arena.unity", "Arena2.unity"],
        "pairings": [("scripted", "stationary")],
        "total_shards": 5,
        "total_samples": 1234,
        "per_map_samples": {"Arena.unity": 600, "Arena2.unity": 634},
    }
    base.update(overrides)
    return base


def _provenance(**overrides):
    base = {
        "git_commit": "abc123",
        "git_dirty": False,
        "build": {
            "path": "/builds/tank.app",
            "mtime_utc": "2026-06-25T00:00:00+00:00",
            "size_bytes": 4096,
        },
    }
    base.update(overrides)
    return base


def _machine(**overrides):
    base = {
        "hostname": "host",
        "platform": "macOS",
        "system": "Darwin",
        "release": "24.6.0",
        "arch": "arm64",
        "processor": "arm",
        "cpu_count": 10,
        "ram_total_gb": 32.0,
        "python": "3.12.0",
    }
    base.update(overrides)
    return base


def _build(**kwargs):
    return manifest.build_manifest(
        dataset=kwargs.get("dataset", "run_001"),
        collection=kwargs.get("collection", _collection()),
        provenance=kwargs.get("provenance", _provenance()),
        machine=kwargs.get("machine", _machine()),
        created_utc=kwargs.get("created_utc", "2026-06-25T12:00:00+00:00"),
        descriptions=kwargs.get("descriptions"),
    )


def test_happy_path_shape():
    result = _build()
    assert set(result) == {
        "schema_version",
        "created_utc",
        "dataset",
        "descriptions",
        "collection",
        "provenance",
        "machine",
    }
    assert result["schema_version"] == manifest.MANIFEST_SCHEMA_VERSION
    assert result["created_utc"] == "2026-06-25T12:00:00+00:00"
    assert result["dataset"] == "run_001"
    assert set(result["collection"]) == {
        "seed",
        "command",
        "workers",
        "episodes",
        "max_steps",
        "maps",
        "pairings",
        "total_shards",
        "total_samples",
        "per_map_samples",
    }
    assert set(result["provenance"]) == {"git_commit", "git_dirty", "build"}
    assert set(result["provenance"]["build"]) == {"path", "mtime_utc", "size_bytes"}
    assert set(result["machine"]) == {
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


def test_strict_json_round_trip_is_identity():
    result = _build()
    assert json.loads(json.dumps(result)) == result


def test_pairings_normalized_to_lists():
    result = _build(collection=_collection(pairings=[("a", "b"), ("c", "d")]))
    assert result["collection"]["pairings"] == [["a", "b"], ["c", "d"]]


def test_per_map_samples_keys_are_strings():
    result = _build(collection=_collection(per_map_samples={0: 10, 1: 20}))
    assert result["collection"]["per_map_samples"] == {"0": 10, "1": 20}
    assert all(isinstance(k, str) for k in result["collection"]["per_map_samples"])


def test_null_git_fields_pass_through():
    result = _build(provenance=_provenance(git_commit=None, git_dirty=None))
    assert result["provenance"]["git_commit"] is None
    assert result["provenance"]["git_dirty"] is None


def test_missing_collection_key_raises():
    bad = _collection()
    del bad["total_samples"]
    with pytest.raises(ValueError, match="collection missing required keys"):
        _build(collection=bad)


def test_missing_build_key_raises():
    bad = _provenance()
    del bad["build"]["size_bytes"]
    with pytest.raises(ValueError, match="provenance.build missing required keys"):
        _build(provenance=bad)


def test_missing_machine_key_raises():
    bad = _machine()
    del bad["hostname"]
    with pytest.raises(ValueError, match="machine missing required keys"):
        _build(machine=bad)


def test_non_dict_sub_dict_raises():
    with pytest.raises(ValueError, match="machine must be a dict"):
        _build(machine=["not", "a", "dict"])


def test_schema_version_is_2():
    assert manifest.MANIFEST_SCHEMA_VERSION == 2


def test_descriptions_default_is_empty_list():
    result = _build()
    assert result["descriptions"] == []


def test_descriptions_passed_through_and_round_trips():
    entries = [
        {"author": "human", "text": "first run", "added_utc": "2026-06-25T12:00:00+00:00"},
        {"author": "claude", "text": "notes", "added_utc": "2026-06-25T13:00:00+00:00"},
    ]
    result = manifest.build_manifest(
        dataset="run_001",
        collection=_collection(),
        provenance=_provenance(),
        machine=_machine(),
        created_utc="2026-06-25T12:00:00+00:00",
        descriptions=entries,
    )
    assert result["descriptions"] == entries
    assert json.loads(json.dumps(result)) == result


def test_add_description_appends_to_existing():
    m = _build(descriptions=[{"author": "human", "text": "a", "added_utc": "t0"}])
    out = manifest.add_description(m, author="claude", text="b", added_utc="t1")
    assert out["descriptions"] == [
        {"author": "human", "text": "a", "added_utc": "t0"},
        {"author": "claude", "text": "b", "added_utc": "t1"},
    ]


def test_add_description_creates_list_on_v1_manifest():
    v1 = {"schema_version": 1, "dataset": "old"}  # no descriptions key
    out = manifest.add_description(v1, author="human", text="hi", added_utc="t0")
    assert out["descriptions"] == [{"author": "human", "text": "hi", "added_utc": "t0"}]
    assert "descriptions" not in v1  # input untouched


def test_add_description_does_not_mutate_input():
    original = [{"author": "human", "text": "a", "added_utc": "t0"}]
    m = _build(descriptions=original)
    inner_before = m["descriptions"]
    out = manifest.add_description(m, author="claude", text="b", added_utc="t1")
    # Neither the input manifest, its nested list, nor the original list is mutated.
    assert len(m["descriptions"]) == 1
    assert m["descriptions"] is inner_before
    assert len(original) == 1
    assert out["descriptions"] is not inner_before
