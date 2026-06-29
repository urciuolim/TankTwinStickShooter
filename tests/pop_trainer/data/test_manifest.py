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
        "manifest_id",
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


def test_schema_version_is_3():
    assert manifest.MANIFEST_SCHEMA_VERSION == 3


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


def test_manifest_id_deterministic():
    # Two builds with identical inputs produce identical ids; the id is stable across calls.
    a = _build()
    b = _build()
    assert a["manifest_id"] == b["manifest_id"]
    assert manifest.manifest_id(a) == manifest.manifest_id(a)


def test_manifest_id_golden():
    # Golden fingerprint over the canonical fixture's build_manifest output.
    #
    # Regenerate procedure: this hex is RECOMPUTED from build_manifest()'s output over the fixture
    # assembled by _build() (the _collection/_provenance/_machine defaults above). To re-pin, print
    # _build()["manifest_id"] and paste it here. Only re-pin when the canonical content or the
    # type-pinning in build_manifest legitimately changes (a surprise change means a regression).
    result = _build()
    assert (
        result["manifest_id"] == "45edcc53d444ef3099cb15cd9aa3b3da54ec8c51369f07697ff0c1c611b9d027"
    )


def test_manifest_id_embedded_matches_recompute():
    result = _build()
    assert result["manifest_id"] == manifest.manifest_id(result)


def test_manifest_id_invariant_under_add_description():
    m = _build()
    before = m["manifest_id"]
    out = manifest.add_description(m, author="claude", text="note", added_utc="t1")
    # Annotating leaves the recomputed id AND the stored embedded id unchanged.
    assert manifest.manifest_id(out) == manifest.manifest_id(m)
    assert out["manifest_id"] == before


def test_manifest_id_self_exclusion_idempotent():
    m = _build()
    without = {k: v for k, v in m.items() if k != "manifest_id"}
    injected = {**without, "manifest_id": "deadbeef" * 8}
    assert manifest.manifest_id(without) == manifest.manifest_id(injected)


def test_manifest_id_key_order_independent():
    m = _build()
    reordered = dict(reversed(list(m.items())))
    assert manifest.manifest_id(reordered) == manifest.manifest_id(m)


def test_manifest_id_v2_backcompat():
    # A hand-built v2-shaped dict (no manifest_id key) fingerprints cleanly and matches the id of
    # the same content (absence of the key is handled by the exclusion filter).
    v2 = {
        "schema_version": 2,
        "created_utc": "2026-06-25T12:00:00+00:00",
        "dataset": "old_run",
        "descriptions": [],
        "collection": {"seed": 7, "total_samples": 100},
        "provenance": {"git_commit": "abc123"},
        "machine": {"hostname": "host"},
    }
    assert "manifest_id" not in v2
    computed = manifest.manifest_id(v2)
    assert len(computed) == 64
    assert computed == manifest.manifest_id(dict(v2))


def test_manifest_id_type_pinning():
    # Two builds differing only in int-vs-float of a coerced field hash identically.
    int_build = _build(machine=_machine(ram_total_gb=32))
    float_build = _build(machine=_machine(ram_total_gb=32.0))
    assert int_build["manifest_id"] == float_build["manifest_id"]
