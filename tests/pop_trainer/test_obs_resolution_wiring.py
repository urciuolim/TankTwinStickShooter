"""Cross-component wiring tests: the env frame_shape DERIVES from the launched game config.

These assert the anti-desync contract end to end without live Unity:

* the real StreamingAssets configs derive the shape their ``obs_pixels_*`` declare (a 64x64 config
  -> ``(64, 64, 3)``; a 640x360 config -> ``(360, 640, 3)``) — the SAME derivation play / train /
  collect now use, so setting ``obs_pixels_*`` in a config auto-matches the env with no flag;
* ``train_config_64.json`` (this task's square 64x64 config) parses as STRICT json with both
  ``obs_pixels_*`` == 64 and derives ``(64, 64, 3)``;
* each consumer (play / train / collect) wires the derivation through to the env's frame_shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from pop_trainer.core.obs import frame_shape_from_config

# The StreamingAssets dir holding the launchable game configs (repo-rooted, four parents up from
# this test file: tests/pop_trainer/ -> tests -> repo).
_STREAMING_ASSETS = Path(__file__).resolve().parents[2] / "unity" / "Assets" / "StreamingAssets"


def _declared_shape(config_path: Path) -> tuple[int, int, int]:
    """The (H, W, 3) the config's obs_pixels_* literally declare (independent recomputation)."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return (data["obs_pixels_height"], data["obs_pixels_width"], 3)


def test_train_config_64_is_strict_square_64():
    # The 64x64 SQUARE config this task adds: STRICT json, both obs_pixels_* == 64.
    path = _STREAMING_ASSETS / "train_config_64.json"
    data = json.loads(path.read_text(encoding="utf-8"))  # strict parse (raises on a trailing comma)
    assert data["obs_pixels_width"] == 64
    assert data["obs_pixels_height"] == 64
    assert data["obs_pixels"] is True


def test_train_config_64_derives_square_64_shape():
    path = _STREAMING_ASSETS / "train_config_64.json"
    assert frame_shape_from_config(path) == (64, 64, 3)


def test_real_configs_derive_their_declared_obs_pixels():
    # For every shipped config that declares obs_pixels_*, the derived (H, W, 3) matches the file's
    # own declaration — the one-source-of-truth guarantee, robust to whatever resolution each config
    # currently declares (640x360 or 64x64).
    for name in ("train_config.json", "train_config_64.json", "demo_config.json"):
        path = _STREAMING_ASSETS / name
        assert frame_shape_from_config(path) == _declared_shape(path), name


def test_train_main_path_derives_frame_shape_from_config():
    # rl.train.main DERIVES frame_shape from --config: build the TrainConfig the way main() does and
    # assert the shape matches the config (the wiring, no live launch).
    from pop_trainer.rl import train

    cfg_path = _STREAMING_ASSETS / "train_config_64.json"
    assert train.frame_shape_from_config(cfg_path) == (64, 64, 3)
    # A TrainConfig built with that derived shape + that game_config is internally consistent (the
    # validation guard is a no-op), proving the derive-then-validate path agrees.
    cfg = train.TrainConfig(
        total_timesteps=1,
        game_config=cfg_path,
        run_dir=Path("unused"),
        frame_shape=train.frame_shape_from_config(cfg_path),
    )
    assert cfg.frame_shape == (64, 64, 3)
    # The validation guard agrees (no raise) because the shape matches the config.
    train.validate_frame_shape(cfg.frame_shape, cfg.game_config)


def test_collect_runner_build_specs_derives_frame_shape_from_boot_config(tmp_path):
    # build_specs DERIVES extra["frame_shape"] from the boot config's obs_pixels_* (no override).
    from pop_trainer.data import collect_runner as R

    boot = tmp_path / "boot.json"
    boot.write_text(
        json.dumps(
            {
                "arena_path": "Arenas/custom1.json",
                "obs_pixels": True,
                "obs_pixels_width": 64,
                "obs_pixels_height": 64,
            }
        ),
        encoding="utf-8",
    )
    specs = R.build_specs(
        map_values=None,
        pairings=[("aggressive-coverage", "opponent-shadower")],
        map_name="custom1",
        config=boot,
        exe=R.DEFAULT_EXE,
        episodes=1,
        max_steps=10,
        out_dir=str(tmp_path / "out"),
        workers=1,
        base_port=50000,
        seed=0,
    )
    assert tuple(specs[0].extra["frame_shape"]) == (64, 64, 3)


def test_collect_runner_build_specs_explicit_frame_shape_overrides_derivation(tmp_path):
    # An explicit frame_shape (the test seam) overrides the boot-config derivation.
    from pop_trainer.data import collect_runner as R

    boot = tmp_path / "boot.json"
    boot.write_text(
        json.dumps(
            {
                "arena_path": "Arenas/custom1.json",
                "obs_pixels": True,
                "obs_pixels_width": 64,
                "obs_pixels_height": 64,
            }
        ),
        encoding="utf-8",
    )
    specs = R.build_specs(
        map_values=None,
        pairings=[("aggressive-coverage", "opponent-shadower")],
        map_name="custom1",
        config=boot,
        exe=R.DEFAULT_EXE,
        episodes=1,
        max_steps=10,
        out_dir=str(tmp_path / "out"),
        workers=1,
        base_port=50000,
        seed=0,
        frame_shape=(360, 640, 3),
    )
    assert tuple(specs[0].extra["frame_shape"]) == (360, 640, 3)
