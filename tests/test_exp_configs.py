"""Pin the shipped experiment configs under exp-configs/.

Guards three things a hand-edit (or a regenerated map) could silently break:
* every map's arena loads to the standard (36, 60, 3) observation, so a model
  trained on one map loads on any other;
* every map's walls are POINT-SYMMETRIC under (x, y) -> (-1-x, -1-y), so neither
  side has a wall advantage;
* the three reward presets parse to exactly their documented values.
"""

import json
from pathlib import Path

import pytest

from tank_twin.arenas import load_level
from tank_twin.config import RewardConfig

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "exp-configs" / "maps"
REWARDS = REPO / "exp-configs" / "rewards"

EXPECTED_MAPS = [
    "custom1_2021",
    "empty",
    "center_block",
    "four_pillars",
    "central_cross",
    "opposing_l",
    "diagonal_pillars",
    "chokepoint",
    "scattered",
    "ring_fragments",
]

EXPECTED_REWARDS = {
    "sparse": (1.0, -1.0, 0.0, 0.0, 5.0),
    "classic_2021": (1.0, -1.0, -0.9, 0.0, 5.0),
    "shaped_default": (1.0, -1.0, -1.0, -0.1, 5.0),
}


def _wall_cells(arena_path):
    walls = json.loads(Path(arena_path).read_text(encoding="utf-8"))["Walls"]
    return {
        (int(col), int(y)) for col, ys in walls.items() if col not in ("dims", "tileID") for y in ys
    }


def test_all_expected_maps_present():
    found = sorted(p.stem for p in MAPS.glob("*.json"))
    assert found == sorted(EXPECTED_MAPS), found


@pytest.mark.parametrize("name", EXPECTED_MAPS)
def test_map_config_resolves_and_obs_shape(name):
    config = json.loads((MAPS / f"{name}.json").read_text(encoding="utf-8"))
    # arena_path is resolved relative to the config file's directory (mirrors
    # DriverController.ResolveArenaPath / TankEnv).
    arena_path = (MAPS / config["arena_path"]).resolve()
    assert arena_path.is_file(), arena_path
    arena = load_level(str(arena_path), p=3)
    assert arena.obs_shape == (36, 60, 3), (name, arena.obs_shape)


@pytest.mark.parametrize("name", EXPECTED_MAPS)
def test_map_is_point_symmetric(name):
    config = json.loads((MAPS / f"{name}.json").read_text(encoding="utf-8"))
    arena_path = (MAPS / config["arena_path"]).resolve()
    cells = _wall_cells(arena_path)
    missing = [c for c in cells if (-1 - c[0], -1 - c[1]) not in cells]
    assert not missing, f"{name}: walls without a (-1-x,-1-y) partner: {missing[:5]}"


@pytest.mark.parametrize("name,expected", EXPECTED_REWARDS.items())
def test_reward_preset_exact(name, expected):
    rc = RewardConfig.load(REWARDS / f"{name}.json")
    got = (rc.win_reward, rc.loss_reward, rc.time_total, rc.action_total, rc.action_norm)
    assert got == expected, (name, got)
