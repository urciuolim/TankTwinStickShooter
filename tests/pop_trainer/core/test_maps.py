"""Contract tests for ``pop_trainer.core.maps`` (the map-rotation resolver)."""

from pathlib import Path

from pop_trainer.core.maps import (
    ALL_MAPS_SENTINEL,
    DEFAULT_MAPS_DIR,
    resolve_map_rotation,
)


def test_default_maps_dir_points_at_repo_exp_configs():
    # This file is tests/pop_trainer/core/test_maps.py -> repo root is parents[3].
    repo_root = Path(__file__).resolve().parents[3]
    assert DEFAULT_MAPS_DIR == repo_root / "exp-configs" / "maps"
    assert DEFAULT_MAPS_DIR.is_dir()


def test_none_returns_none():
    assert resolve_map_rotation(None) is None


def test_empty_and_sentinel_return_sorted_default_maps():
    expected = sorted(DEFAULT_MAPS_DIR.glob("*.json"))
    assert expected, "expected the shipped exp-configs maps to exist"
    assert resolve_map_rotation([]) == expected
    assert resolve_map_rotation([ALL_MAPS_SENTINEL]) == expected


def test_single_directory_returns_sorted_maps(tmp_path):
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    result = resolve_map_rotation([str(tmp_path)])
    assert result == sorted([tmp_path / "a.json", tmp_path / "b.json"])


def test_explicit_list_preserves_order():
    values = ["z.json", "a.json", "m.json"]
    result = resolve_map_rotation(values)
    assert result == [Path("z.json"), Path("a.json"), Path("m.json")]
