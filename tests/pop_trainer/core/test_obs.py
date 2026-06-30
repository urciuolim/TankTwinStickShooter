"""Contract tests for ``pop_trainer.core.obs`` — DERIVE the pixel frame_shape from a game config.

Pure (stdlib + tmp files); no live Unity, no socket. Covers:

* :func:`frame_shape_from_config` -> ``(64, 64, 3)`` for a 64x64 config, ``(360, 640, 3)`` for a
  640x360 config, and the fallback ``(360, 640, 3)`` for a config WITHOUT ``obs_pixels_*``.
* a malformed (non-int) value is a clear ``ValueError``.
* :func:`validate_frame_shape` is a no-op on a match and raises NAMING BOTH shapes on a mismatch.
"""

from __future__ import annotations

import json

import pytest

from pop_trainer.core.obs import (
    DEFAULT_FRAME_SHAPE,
    frame_shape_from_config,
    validate_frame_shape,
)


def _write_config(path, **overrides):
    """Write a minimal RAW Unity-style game config JSON (strict) with the given overrides."""
    config = {
        "connectionIP": "127.0.0.1",
        "connectionPort": 50000,
        "arena_path": "Arenas/custom1.json",
        "obs_pixels": True,
    }
    config.update(overrides)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def test_derives_square_64_shape(tmp_path):
    cfg = _write_config(tmp_path / "c64.json", obs_pixels_width=64, obs_pixels_height=64)
    assert frame_shape_from_config(cfg) == (64, 64, 3)


def test_derives_640x360_shape_height_first(tmp_path):
    # Channels-last (H, W, 3): HEIGHT first, WIDTH second.
    cfg = _write_config(tmp_path / "c.json", obs_pixels_width=640, obs_pixels_height=360)
    assert frame_shape_from_config(cfg) == (360, 640, 3)


def test_non_square_keeps_height_first(tmp_path):
    # A non-square config still maps (width=W, height=H) -> (H, W, 3), not transposed.
    cfg = _write_config(tmp_path / "wide.json", obs_pixels_width=128, obs_pixels_height=96)
    assert frame_shape_from_config(cfg) == (96, 128, 3)


def test_missing_obs_pixels_keys_falls_back_to_default(tmp_path):
    # A config that declares NO obs_pixels_* keys falls back to the canonical default (no crash).
    cfg = _write_config(tmp_path / "nopix.json")
    assert frame_shape_from_config(cfg) == DEFAULT_FRAME_SHAPE == (360, 640, 3)


def test_malformed_non_int_value_raises(tmp_path):
    cfg = _write_config(tmp_path / "bad.json", obs_pixels_width="64", obs_pixels_height=64)
    with pytest.raises(ValueError, match="obs_pixels_width"):
        frame_shape_from_config(cfg)


def test_non_positive_value_raises(tmp_path):
    cfg = _write_config(tmp_path / "zero.json", obs_pixels_width=0, obs_pixels_height=64)
    with pytest.raises(ValueError, match="obs_pixels_width"):
        frame_shape_from_config(cfg)


def test_bool_value_rejected(tmp_path):
    # bool is an int subclass in Python but is never a valid pixel dimension.
    cfg = _write_config(tmp_path / "boolw.json", obs_pixels_width=True, obs_pixels_height=64)
    with pytest.raises(ValueError, match="obs_pixels_width"):
        frame_shape_from_config(cfg)


def test_validate_frame_shape_noop_on_match(tmp_path):
    cfg = _write_config(tmp_path / "c64.json", obs_pixels_width=64, obs_pixels_height=64)
    # No raise on a match.
    assert validate_frame_shape((64, 64, 3), cfg) is None


def test_validate_frame_shape_raises_naming_both_shapes(tmp_path):
    cfg = _write_config(tmp_path / "c64.json", obs_pixels_width=64, obs_pixels_height=64)
    with pytest.raises(ValueError) as exc:
        validate_frame_shape((360, 640, 3), cfg)
    message = str(exc.value)
    # BOTH shapes are named so the desync is diagnosable from the message alone.
    assert "(360, 640, 3)" in message
    assert "(64, 64, 3)" in message
