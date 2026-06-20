"""Tests for tank_twin.config.RewardConfig + the train.py reward/manifest plumbing.

RewardConfig is the CTO's budget-based reward, parameterized by full-episode budgets.
These pin: the new defaults, strict-JSON loading (reject malformed / unknown keys), the
round-trip dict view, the CLI/file/default override precedence in train._resolve_reward_config,
and that train._build_manifest records the RESOLVED RewardConfig (built WITHOUT running SB3).

Also pins the Change-1 trainer wiring: train._build_parser exposes --config and its --map
alias (both -> the same config_path dest) and train._build_manifest records that config_path
(the arena single-source the run trained on) — all WITHOUT running SB3.
"""

import json

import pytest

from tank_twin.config import RewardConfig


def test_defaults_are_the_cto_reward():
    rc = RewardConfig()
    assert rc.win_reward == 1.0
    assert rc.loss_reward == -1.0
    assert rc.time_total == -1.0
    assert rc.action_total == -0.1
    assert rc.action_norm == 5.0


def test_to_dict_round_trips_through_from_dict():
    rc = RewardConfig(win_reward=2.0, loss_reward=-2.0, time_total=-0.5, action_total=-0.2)
    assert RewardConfig.from_dict(rc.to_dict()) == rc


def test_to_dict_is_all_floats():
    # Stable JSON: every value is a float (so a manifest never carries stray ints).
    assert all(isinstance(v, float) for v in RewardConfig().to_dict().values())


def test_from_dict_missing_keys_use_defaults():
    rc = RewardConfig.from_dict({"win_reward": 5.0})
    assert rc.win_reward == 5.0
    assert rc.loss_reward == -1.0  # default
    assert rc.action_norm == 5.0  # default


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError):
        RewardConfig.from_dict({"win_reward": 1.0, "bogus_knob": 9.0})


def test_from_json_parses_strict_object():
    rc = RewardConfig.from_json('{"win_reward": 3.0, "time_total": -0.25}')
    assert rc.win_reward == 3.0
    assert rc.time_total == -0.25


def test_from_json_rejects_trailing_comma():
    # STRICT json: a trailing comma (which Unity's Newtonsoft tolerates) RAISES.
    with pytest.raises(json.JSONDecodeError):
        RewardConfig.from_json('{"win_reward": 1.0,}')


def test_from_json_rejects_non_object_top_level():
    with pytest.raises(TypeError):
        RewardConfig.from_json("[1, 2, 3]")


def test_load_reads_strict_json_file(tmp_path):
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"win_reward": 4.0, "action_total": -0.3}), encoding="utf-8")
    rc = RewardConfig.load(path)
    assert rc.win_reward == 4.0
    assert rc.action_total == -0.3


def test_config_module_is_stdlib_only_no_heavy_deps():
    # config.py must stay in the pure import path: no numpy/sb3/torch pulled in.
    import sys

    import tank_twin.config  # noqa: F401

    # The module itself imports only json + dataclasses + pathlib.
    mod = sys.modules["tank_twin.config"]
    src = mod.__doc__ or ""
    assert "stdlib-only" in src
    # It must not have imported torch as a side effect of importing config.
    # (We can't assert global sys.modules cleanliness — other tests import torch — but we
    # CAN assert config has no torch/numpy attribute bound.)
    assert not hasattr(mod, "np")
    assert not hasattr(mod, "torch")


# --- train.py reward plumbing: precedence + manifest (no SB3 training) ------------------


def _ns(**overrides):
    """A minimal argparse.Namespace stand-in carrying just the reward knobs (+ reward_config)."""
    import argparse

    base = {
        "reward_config": None,
        "win_reward": None,
        "loss_reward": None,
        "time_penalty_total": None,
        "action_cost_total": None,
        "action_norm": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_reward_config_defaults_when_nothing_set():
    from tank_twin.train import _resolve_reward_config

    assert _resolve_reward_config(_ns()) == RewardConfig()


def test_resolve_reward_config_file_overrides_defaults(tmp_path):
    from tank_twin.train import _resolve_reward_config

    path = tmp_path / "r.json"
    path.write_text(json.dumps({"win_reward": 7.0, "time_total": -0.5}), encoding="utf-8")
    rc = _resolve_reward_config(_ns(reward_config=path))
    assert rc.win_reward == 7.0
    assert rc.time_total == -0.5
    assert rc.loss_reward == -1.0  # untouched default


def test_resolve_reward_config_cli_overrides_file(tmp_path):
    from tank_twin.train import _resolve_reward_config

    path = tmp_path / "r.json"
    path.write_text(json.dumps({"win_reward": 7.0, "time_total": -0.5}), encoding="utf-8")
    # --win-reward on the CLI overrides the file's win_reward; the file's time_total stays.
    rc = _resolve_reward_config(_ns(reward_config=path, win_reward=2.0))
    assert rc.win_reward == 2.0  # CLI wins
    assert rc.time_total == -0.5  # from file


def test_resolve_reward_config_cli_overrides_defaults_without_file():
    from tank_twin.train import _resolve_reward_config

    rc = _resolve_reward_config(_ns(action_cost_total=-0.25, action_norm=4.0))
    assert rc.action_total == -0.25
    assert rc.action_norm == 4.0
    assert rc.win_reward == 1.0  # default


def test_build_manifest_records_resolved_reward_config():
    # The manifest dict (built WITHOUT running SB3) carries the resolved RewardConfig.
    from tank_twin.train import _build_manifest

    rc = RewardConfig(win_reward=2.0, action_total=-0.2)
    manifest = _build_manifest(
        run_name="r",
        timesteps=100,
        seed=0,
        frozen=True,
        device="cpu",
        n_steps=32,
        batch_size=16,
        learning_rate=3e-4,
        game_path=None,
        reward_config=rc,
    )
    assert manifest["reward_config"] == rc.to_dict()
    assert manifest["reward_config"]["win_reward"] == 2.0
    assert manifest["reward_config"]["action_total"] == -0.2
    # config_path defaults to None (no map given) but the key is ALWAYS present.
    assert "config_path" in manifest
    assert manifest["config_path"] is None


def test_build_manifest_records_config_path_when_given():
    # The manifest records WHICH map (arena single-source) the run trained on, as a string.
    from tank_twin.train import _build_manifest

    manifest = _build_manifest(
        run_name="r",
        timesteps=100,
        seed=0,
        frozen=True,
        device="cpu",
        n_steps=32,
        batch_size=16,
        learning_rate=3e-4,
        game_path="C:/builds/game.exe",
        config_path="C:/arenas/custom1/config.json",
        reward_config=RewardConfig(),
    )
    assert manifest["config_path"] == "C:/arenas/custom1/config.json"


def test_build_manifest_is_strict_json_serializable():
    from tank_twin.train import _build_manifest

    manifest = _build_manifest(
        run_name="r",
        timesteps=100,
        seed=0,
        frozen=True,
        device="cpu",
        n_steps=32,
        batch_size=16,
        learning_rate=3e-4,
        game_path="C:/builds/game.exe",
        reward_config=RewardConfig(),
    )
    # Serializes cleanly (the manifest is written as strict JSON).
    text = json.dumps(manifest)
    assert json.loads(text)["reward_config"]["time_total"] == -1.0


# --- train.py CLI: --config / --map arena single-source flag (Change 1 trainer wiring) ---


def test_parser_config_flag_sets_config_path():
    from pathlib import Path

    from tank_twin.train import _build_parser

    args = _build_parser().parse_args(
        ["--game-path", "C:/builds/game.exe", "--config", "C:/arenas/custom1/config.json"]
    )
    assert args.config_path == Path("C:/arenas/custom1/config.json")


def test_parser_map_alias_resolves_to_same_dest_as_config():
    # --map is an explicit alias for --config: both write the SAME config_path dest.
    from tank_twin.train import _build_parser

    via_config = _build_parser().parse_args(
        ["--game-path", "C:/b/g.exe", "--config", "C:/arenas/m/config.json"]
    )
    via_map = _build_parser().parse_args(
        ["--game-path", "C:/b/g.exe", "--map", "C:/arenas/m/config.json"]
    )
    assert via_config.config_path == via_map.config_path
    assert via_map.config_path is not None


def test_parser_config_path_defaults_to_none():
    # No --config / --map -> config_path is None (env falls back to the build's config).
    from tank_twin.train import _build_parser

    args = _build_parser().parse_args(["--game-path", "C:/builds/game.exe"])
    assert args.config_path is None
