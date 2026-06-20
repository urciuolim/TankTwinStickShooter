"""Unit tests for the SHARED map-rotation resolver (``tank_twin.maps``).

The canonical ``_resolve_map_rotation`` + ``DEFAULT_MAPS_DIR`` / ``ALL_MAPS_SENTINEL`` were
factored out of ``tank_twin.train`` into ``tank_twin.maps`` so BOTH the trainer and the
evaluator share ONE contract. ``tests/test_train_features.py`` already pins the resolver via
the ``tank_twin.train`` re-export (that import path must stay green); this file re-asserts the
SAME four branches imported DIRECTLY from ``tank_twin.maps`` (belt-and-suspenders) and that
the train re-export is the SAME object (byte-identical contract, no drift). Torch-free /
Unity-free: stdlib + pathlib only.
"""

from pathlib import Path

from tank_twin.maps import ALL_MAPS_SENTINEL, DEFAULT_MAPS_DIR, _resolve_map_rotation


def test_absent_flag_is_none():
    # None (flag ABSENT) -> None: no rotation.
    assert _resolve_map_rotation(None) is None


def test_no_value_resolves_to_sorted_default_dir():
    # [] (flag passed with no value) -> sorted DEFAULT_MAPS_DIR/*.json (the shipped 10).
    resolved = _resolve_map_rotation([])
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))
    assert len(resolved) == 10


def test_sentinel_resolves_to_sorted_default_dir():
    # [ALL_MAPS_SENTINEL] is equivalent to [] -> sorted DEFAULT_MAPS_DIR/*.json.
    assert _resolve_map_rotation([ALL_MAPS_SENTINEL]) == sorted(DEFAULT_MAPS_DIR.glob("*.json"))


def test_single_directory_resolves_sorted():
    # A single entry that is a DIRECTORY -> sorted its *.json.
    resolved = _resolve_map_rotation([str(DEFAULT_MAPS_DIR)])
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))
    assert len(resolved) == 10


def test_explicit_list_keeps_given_order():
    # An explicit list -> the given order (NOT re-sorted).
    resolved = _resolve_map_rotation(["b.json", "a.json", "c.json"])
    assert [p.name for p in resolved] == ["b.json", "a.json", "c.json"]
    assert all(isinstance(p, Path) for p in resolved)


def test_train_reexport_is_same_object():
    # The train re-export MUST be the SAME objects (no drift between caller contracts).
    from tank_twin import train

    assert train._resolve_map_rotation is _resolve_map_rotation
    assert train.DEFAULT_MAPS_DIR is DEFAULT_MAPS_DIR
    assert train.ALL_MAPS_SENTINEL is ALL_MAPS_SENTINEL
