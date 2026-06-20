"""Unit-level plumbing tests for the trainer's new CLI flags + manifest fields.

These stay UNIT-LEVEL: they exercise the argparse parser and the pure ``_build_manifest``
helper WITHOUT running SB3 / PPO / Unity (the heavy round-trip is the e2e test). Covers:

* Feature 1: ``--maps`` / ``--map-rotation`` resolution + the manifest ``map_rotation`` field;
* Feature 2: ``--checkpoint-freq`` + ``--resume-from`` flags flow to the parser, and
  ``_build_manifest`` records the ``resume_from`` lineage;
* Feature 3: ``--eval-freq`` + ``--eval-episodes`` flags + the manifest eval fields.

The Feature-3 eval-CALLBACK behavior (rollout-boundary gating + buffer repair) is pinned in
``tests/test_callbacks.py``; this file pins the flag/manifest plumbing.
"""

from pathlib import Path

from tank_twin.config import RewardConfig
from tank_twin.train import (
    DEFAULT_MAPS_DIR,
    _build_manifest,
    _build_parser,
    _resolve_map_rotation,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(**overrides):
    """Build a manifest with sane required defaults, applying any overrides."""
    base = dict(
        run_name="r",
        timesteps=1,
        seed=0,
        frozen=True,
        device="cpu",
        n_steps=1,
        batch_size=1,
        learning_rate=1e-3,
        game_path=None,
        reward_config=RewardConfig(),
    )
    base.update(overrides)
    return _build_manifest(**base)


# --- Feature 1: --maps / --map-rotation parser + manifest ------------------------------


def test_maps_flag_absent_is_none():
    args = _build_parser().parse_args(["--game-path", "g.exe"])
    assert args.maps is None
    assert _resolve_map_rotation(args.maps) is None


def test_maps_flag_no_value_resolves_to_all_ten():
    args = _build_parser().parse_args(["--game-path", "g.exe", "--maps"])
    resolved = _resolve_map_rotation(args.maps)
    assert resolved is not None
    assert len(resolved) == 10
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))


def test_map_rotation_alias_explicit_list_keeps_order():
    args = _build_parser().parse_args(
        ["--game-path", "g.exe", "--map-rotation", "b.json", "a.json"]
    )
    resolved = _resolve_map_rotation(args.maps)
    assert [p.name for p in resolved] == ["b.json", "a.json"]  # not re-sorted


def test_maps_flag_directory_resolves_sorted():
    args = _build_parser().parse_args(["--game-path", "g.exe", "--maps", str(DEFAULT_MAPS_DIR)])
    resolved = _resolve_map_rotation(args.maps)
    assert len(resolved) == 10
    assert resolved == sorted(DEFAULT_MAPS_DIR.glob("*.json"))


def test_manifest_records_map_rotation_list():
    m = _manifest(map_rotation=[Path("a.json"), Path("b.json")])
    assert m["map_rotation"] == ["a.json", "b.json"]


def test_manifest_map_rotation_none_when_unset():
    assert _manifest()["map_rotation"] is None


# --- Feature 2: --checkpoint-freq + --resume-from parser + manifest --------------------


def test_checkpoint_freq_flag_flows_to_parser():
    args = _build_parser().parse_args(["--game-path", "g.exe", "--checkpoint-freq", "1234"])
    assert args.checkpoint_freq == 1234


def test_checkpoint_freq_default_matches_kwarg_default():
    args = _build_parser().parse_args(["--game-path", "g.exe"])
    assert args.checkpoint_freq == 50_000


def test_resume_from_flag_flows_to_parser():
    args = _build_parser().parse_args(["--game-path", "g.exe", "--resume-from", "models/ckpt.zip"])
    assert args.resume_from == Path("models/ckpt.zip")


def test_resume_from_default_is_none():
    args = _build_parser().parse_args(["--game-path", "g.exe"])
    assert args.resume_from is None


def test_manifest_records_resume_from_lineage():
    m = _manifest(resume_from=Path("models/ckpt.zip"))
    assert m["resume_from"] == str(Path("models/ckpt.zip"))


def test_manifest_resume_from_none_when_fresh():
    assert _manifest()["resume_from"] is None


# --- Feature 3: --eval-freq + --eval-episodes parser + manifest ------------------------


def test_eval_freq_default_is_off():
    args = _build_parser().parse_args(["--game-path", "g.exe"])
    assert args.eval_freq == 0  # 0 == OFF (no eval callback)


def test_eval_freq_and_episodes_flow_to_parser():
    args = _build_parser().parse_args(
        ["--game-path", "g.exe", "--eval-freq", "5000", "--eval-episodes", "25"]
    )
    assert args.eval_freq == 5000
    assert args.eval_episodes == 25


def test_eval_episodes_default():
    args = _build_parser().parse_args(["--game-path", "g.exe"])
    assert args.eval_episodes == 10


def test_manifest_records_eval_fields():
    m = _manifest(eval_freq=5000, eval_episodes=25)
    assert m["eval_freq"] == 5000
    assert m["eval_episodes"] == 25


def test_manifest_eval_defaults_when_unset():
    m = _manifest()
    assert m["eval_freq"] == 0
    assert m["eval_episodes"] == 10
