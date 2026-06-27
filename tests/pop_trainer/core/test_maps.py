"""Contract tests for ``pop_trainer.core.maps`` (the map-rotation resolver)."""

from pop_trainer.core.maps import (
    ALL_MAPS_SENTINEL,
    CURATED_ROTATION,
    resolve_map_rotation,
)

# The exact 10 curated arena targets, in alphabetical-by-name order. Hard-coded here as the
# NO-REGRESSION guarantee (NOT derived from the constant under test): the same names in the same
# order the prior sorted-glob default behavior produced.
_EXPECTED_ROTATION = [
    "Arenas/center_block.json",
    "Arenas/central_cross.json",
    "Arenas/chokepoint.json",
    "Arenas/custom1_2021.json",
    "Arenas/diagonal_pillars.json",
    "Arenas/empty.json",
    "Arenas/four_pillars.json",
    "Arenas/opposing_l.json",
    "Arenas/ring_fragments.json",
    "Arenas/scattered.json",
]


def test_none_returns_none():
    assert resolve_map_rotation(None) is None


def test_empty_and_sentinel_return_curated_rotation():
    expected = list(CURATED_ROTATION)
    assert resolve_map_rotation([]) == expected
    assert resolve_map_rotation([ALL_MAPS_SENTINEL]) == expected


def test_empty_yields_exactly_the_ten_maps_in_order():
    # NO-REGRESSION: the sentinel/empty case yields EXACTLY these 10 arena targets in THIS order
    # (same names, same order as the prior sorted-glob behavior). The literal IS the guarantee.
    assert resolve_map_rotation([]) == _EXPECTED_ROTATION
    assert resolve_map_rotation([ALL_MAPS_SENTINEL]) == _EXPECTED_ROTATION


def test_single_directory_returns_sorted_arena_targets(tmp_path):
    # A dir of map-config wrappers resolves to each config's arena_path, sorted by config filename.
    (tmp_path / "b.json").write_text('{"arena_path": "Arenas/b.json"}', encoding="utf-8")
    (tmp_path / "a.json").write_text('{"arena_path": "Arenas/a.json"}', encoding="utf-8")
    result = resolve_map_rotation([str(tmp_path)])
    assert result == ["Arenas/a.json", "Arenas/b.json"]


def test_explicit_list_preserves_order():
    # Explicit arena targets pass through verbatim, in the given order (not re-sorted).
    values = ["Arenas/z.json", "Arenas/a.json", "Arenas/m.json"]
    result = resolve_map_rotation(values)
    assert result == ["Arenas/z.json", "Arenas/a.json", "Arenas/m.json"]
