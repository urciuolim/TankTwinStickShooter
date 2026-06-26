"""Tests for the describe CLI: manifest annotation round-trip + v1 backward-compat + listing."""

import json

from pop_trainer.data import describe
from pop_trainer.data.manifest import MANIFEST_NAME


def _write_manifest(dataset_dir, manifest):
    path = dataset_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _read_manifest(dataset_dir):
    return json.loads((dataset_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_add_description_round_trip(tmp_path):
    _write_manifest(tmp_path, {"schema_version": 2, "dataset": "run", "descriptions": []})

    rc = describe.main([str(tmp_path), "--text", "hello", "--author", "claude"])
    assert rc == 0

    manifest = _read_manifest(tmp_path)
    assert len(manifest["descriptions"]) == 1
    entry = manifest["descriptions"][0]
    assert entry["author"] == "claude"
    assert entry["text"] == "hello"
    assert isinstance(entry["added_utc"], str) and entry["added_utc"]


def test_add_description_on_v1_manifest_without_key(tmp_path):
    # A v1 manifest lacking a descriptions key must remain annotatable (backward-compat).
    _write_manifest(tmp_path, {"schema_version": 1, "dataset": "old"})

    rc = describe.main([str(tmp_path), "--text", "annotate the old one"])
    assert rc == 0

    manifest = _read_manifest(tmp_path)
    assert manifest["descriptions"] == [
        {
            "author": "human",
            "text": "annotate the old one",
            "added_utc": manifest["descriptions"][0]["added_utc"],
        }
    ]
    # Default author is human; the rest of the manifest is preserved.
    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "old"


def test_default_author_is_human(tmp_path):
    _write_manifest(tmp_path, {"schema_version": 2, "dataset": "run", "descriptions": []})
    describe.main([str(tmp_path), "--text", "x"])
    assert _read_manifest(tmp_path)["descriptions"][0]["author"] == "human"


def test_list_prints_existing_descriptions(tmp_path, capsys):
    _write_manifest(
        tmp_path,
        {
            "schema_version": 2,
            "dataset": "run",
            "descriptions": [
                {"author": "human", "text": "first", "added_utc": "2026-06-25T12:00:00+00:00"}
            ],
        },
    )
    rc = describe.main([str(tmp_path), "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "human" in out
    assert "first" in out
    assert "2026-06-25T12:00:00+00:00" in out


def test_missing_manifest_returns_2(tmp_path):
    assert describe.main([str(tmp_path), "--text", "x"]) == 2


def test_no_text_without_list_returns_2(tmp_path):
    _write_manifest(tmp_path, {"schema_version": 2, "dataset": "run", "descriptions": []})
    assert describe.main([str(tmp_path)]) == 2
