"""Observation-resolution contract: DERIVE the pixel ``frame_shape`` from the game-config JSON.

The Unity build sizes its RenderTexture / Texture2D from ``obs_pixels_width`` /
``obs_pixels_height`` in the game config at RUNTIME (``DriverController`` ->
``FrameCapture.Init(w, h)``) and ships actual W/H on the wire. The Python env's pixel
``frame_shape`` therefore MUST match that config so the socket byte-read is correct. This module
makes the config the ONE source of truth for both sides — no hand-synced hardcoded resolution that
silently desyncs.

The game config is a RAW Unity config (e.g. ``unity/Assets/StreamingAssets/train_config.json``), NOT
one of the typed :mod:`pop_trainer.core.config` dataclasses (those reject unknown keys and are the
wrong surface for ``obs_pixels_*``). So the read is a plain STRICT-``json`` parse of the two keys.

stdlib only (``json`` + ``pathlib``) so this stays in the pure import path — no numpy, no env, no
torch.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DEFAULT_FRAME_SHAPE", "frame_shape_from_config", "validate_frame_shape"]

# The canonical default pixel-frame shape (H, W, 3) — the 640x360 16:9 frame the Unity
# DriverController defaults to when a config declares no obs_pixels_* keys. gymnasium pixel obs is
# channels-LAST (H, W, 3): HEIGHT first, WIDTH second.
DEFAULT_FRAME_SHAPE: tuple[int, int, int] = (360, 640, 3)


def _require_int(value: object, key: str, config_path: Path) -> int:
    """Coerce a present ``obs_pixels_*`` value to a positive int or raise a clear ``ValueError``.

    A bool is rejected (``isinstance(True, int)`` is True in Python, but a bool is never a valid
    pixel dimension). A non-int / non-positive value is a clear config error — not a silently
    defaulted shape.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"game config {config_path} has a non-int {key!r}: {value!r} "
            f"(must be a positive integer pixel dimension)"
        )
    if value <= 0:
        raise ValueError(
            f"game config {config_path} has a non-positive {key!r}: {value!r} "
            f"(must be a positive integer pixel dimension)"
        )
    return value


def frame_shape_from_config(config_path: str | Path) -> tuple[int, int, int]:
    """Derive the channels-last pixel ``(H, W, 3)`` frame shape from a RAW game-config JSON.

    STRICT-``json`` reads ``config_path`` and pulls ``obs_pixels_width`` / ``obs_pixels_height``,
    returning ``(obs_pixels_height, obs_pixels_width, 3)`` — HEIGHT first, WIDTH second (gymnasium
    pixel obs is channels-LAST). This is the SAME config the Unity build launches with, so the
    derived shape matches the bytes the build ships on the wire (one source of truth).

    Missing keys fall back to :data:`DEFAULT_FRAME_SHAPE` (the build's own DriverController default)
    — an absent ``obs_pixels_*`` is NOT an error. A present-but-malformed value (non-int /
    non-positive) IS a clear ``ValueError`` (a silent default there would re-introduce the desync
    this module exists to prevent). Both keys must be present together to override the default; if
    only one is present the other is taken from the default.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(
            f"game config {config_path} must be a JSON object, got {type(config).__name__}"
        )

    default_h, default_w, _ = DEFAULT_FRAME_SHAPE
    width = config.get("obs_pixels_width")
    height = config.get("obs_pixels_height")
    w = default_w if width is None else _require_int(width, "obs_pixels_width", config_path)
    h = default_h if height is None else _require_int(height, "obs_pixels_height", config_path)
    return (h, w, 3)


def validate_frame_shape(frame_shape: tuple[int, int, int], config_path: str | Path) -> None:
    """Guard against a silent env<->config desync: raise if ``frame_shape`` != the config's shape.

    Derives the expected ``(H, W, 3)`` from ``config_path`` via :func:`frame_shape_from_config` and
    raises a CLEAR ``ValueError`` naming BOTH shapes (and the config) when they differ. A no-op on a
    match. This is the anti-silent-desync guard: the env reads exactly ``prod(frame_shape)`` bytes
    per frame, so a mismatch with the build's actual W/H would corrupt every frame — fail fast
    instead.
    """
    expected = frame_shape_from_config(config_path)
    actual = tuple(int(d) for d in frame_shape)
    if actual != expected:
        raise ValueError(
            f"frame_shape {actual} does not match the game config {Path(config_path)} "
            f"obs_pixels shape {expected} (H, W, 3) — the env byte-read would desync from the "
            f"build's rendered frame. Derive frame_shape from the launched config "
            f"(frame_shape_from_config) so the two stay in sync."
        )
