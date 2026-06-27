"""Contract + behaviour tests for the model-free agents in ``pop_trainer.agents``.

No socket, no env / data import — agents are exercised against bare 52-float state vectors and
``WallLayout``s built directly / from arena JSON.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import pop_trainer.core as core
from pop_trainer import agents
from pop_trainer.agents.coverage_metrics import measure_coverage
from pop_trainer.core import state as S
from pop_trainer.core.protocol import WallDims, WallLayout

ARENA_DIR = Path(__file__).resolve().parents[3] / "Assets" / "StreamingAssets" / "Arenas"
HARD_MAPS = ("center_block", "central_cross", "chokepoint", "scattered")


# --- test helpers ----------------------------------------------------------------------------


def _zero_state():
    return [0.0] * S.STATE_LEN


def _set_pos(state, player, x, y):
    base = player * S.PLAYER_STRIDE
    state[base + S.POS_X] = float(x)
    state[base + S.POS_Y] = float(y)


def _layout_from_arena(name: str) -> WallLayout:
    """Build a ``WallLayout`` from an arena JSON's ``Walls`` block (mirrors the wire shape).

    Reads the ``Walls`` dims + per-column occupied y-lists exactly as the Unity ``WallMessage``
    carries them, so the coverage harness measures agents on the PRODUCTION free-cell derivation.
    """
    d = json.loads((ARENA_DIR / f"{name}.json").read_text())
    w = d["Walls"]
    dims = WallDims(
        min_x=w["dims"]["minX"],
        max_x=w["dims"]["maxX"],
        min_y=w["dims"]["minY"],
        max_y=w["dims"]["maxY"],
    )
    columns = {int(k): tuple(v) for k, v in w.items() if k not in ("dims", "tileID")}
    return WallLayout(map_id=name, tile_id=w["tileID"], dims=dims, columns=columns)


def _small_layout() -> WallLayout:
    """A tiny 3x3 layout with a single interior wall cell (for cheap unit checks)."""
    return WallLayout(
        map_id="tiny",
        tile_id=0,
        dims=WallDims(min_x=-1, max_x=1, min_y=-1, max_y=1),
        columns={0: (0,)},  # the centre cell is a wall
    )


def _new_family_agents():
    return [
        agents.CoverageAgent.aggressive(seed=1),
        agents.CoverageAgent.wall_hugger(seed=1),
        agents.CoverageAgent.opponent_shadower(seed=1),
    ]


def _all_agents():
    return [agents.RandomAgent(seed=1), *_new_family_agents()]


# --- protocol + action contract --------------------------------------------------------------


@pytest.mark.parametrize("agent", _all_agents(), ids=lambda a: repr(a))
def test_agent_satisfies_protocol_and_returns_length5(agent):
    assert isinstance(agent, core.Agent)
    out = agent.act(_zero_state())
    assert len(out) == agents.ACTION_LEN


def test_package_exports_action_contract_and_family():
    for name in ("ACTION_LEN", "validate_action", "as_state_vector"):
        assert name in agents.__all__
    for name in (
        "RandomAgent",
        "CoverageAgent",
        "AimFireSchedule",
        "measure_coverage",
        "build_grid",
        "CoverageReport",
    ):
        assert name in agents.__all__


def test_deleted_agents_are_gone():
    # The pre-redesign per-variable sweeps + tactical rules were deleted (no back-compat).
    for gone in (
        "ExplorerAgent",
        "AimSweepAgent",
        "SprayAgent",
        "PerimeterAgent",
        "IdleAgent",
        "ConstantAgent",
        "ScriptedCycleAgent",
        "AimAtPlayer2Agent",
    ):
        assert not hasattr(agents, gone)
        assert gone not in agents.__all__


# --- the map hook (set_map) ------------------------------------------------------------------


def test_coverage_family_is_map_aware():
    # Every coverage preset exposes the OPTIONAL set_map hook and accepts a WallLayout.
    layout = _small_layout()
    for agent in _new_family_agents():
        assert hasattr(agent, "set_map")
        agent.set_map(layout)  # must not raise
        out = agent.act(_zero_state())
        assert len(out) == agents.ACTION_LEN


def test_stateful_agent_protocol_documents_set_map():
    # The map hook lives on the static-only StatefulAgent protocol (not runtime-checkable Agent).
    assert hasattr(core.StatefulAgent, "set_map")
    # A coverage agent statically presents the StatefulAgent surface: act + reset + set_map.
    agent = agents.CoverageAgent.aggressive(seed=0)
    for method in ("act", "reset", "set_map"):
        assert callable(getattr(agent, method))


def test_random_agent_is_not_map_aware():
    # The map-agnostic baseline does NOT implement set_map (never required to).
    assert not hasattr(agents.RandomAgent(seed=0), "set_map")


def test_coverage_degrades_without_set_map():
    # When set_map is never called (no arena), the family degrades to a safe map-blind fallback:
    # a length-5 action, never a crash / KeyError.
    agent = agents.CoverageAgent.aggressive(seed=0)
    for _ in range(30):
        out = agent.act(_zero_state())
        assert len(out) == agents.ACTION_LEN
        assert all(isinstance(float(x), float) for x in out)


def test_coverage_degrades_on_none_layout():
    agent = agents.CoverageAgent.aggressive(seed=0)
    agent.set_map(None)  # explicitly no map
    out = agent.act(_zero_state())
    assert len(out) == agents.ACTION_LEN


def test_coverage_movement_is_unit_or_zero_heading():
    # The movement heading is normalized (a unit vector) on a real map.
    agent = agents.CoverageAgent.aggressive(seed=0)
    agent.set_map(_layout_from_arena("center_block"))
    state = _zero_state()
    _set_pos(state, S.PLAYER_1, -8.0, -1.0)
    out = agent.act(state)
    mag = math.hypot(out[0], out[1])
    assert math.isclose(mag, 1.0, abs_tol=1e-6)


# --- the aim/fire layer ----------------------------------------------------------------------


def test_aggressive_aim_sweeps_and_fires_periodically():
    agent = agents.CoverageAgent.aggressive(seed=0)
    agent.set_map(_layout_from_arena("center_block"))
    state = _zero_state()
    _set_pos(state, S.PLAYER_1, -8.0, -1.0)
    aims = set()
    fires = []
    for _ in range(40):
        out = agent.act(state)
        aims.add((round(out[2], 3), round(out[3], 3)))  # aim channels
        fires.append(out[4])
    # The aim sweep visits many distinct aim vectors.
    assert len(aims) > 4
    # Fire is periodic: some steps fire, many do not (a modest cadence, not every step).
    assert any(f == 1.0 for f in fires)
    assert any(f == 0.0 for f in fires)


def test_opponent_shadower_aims_at_opponent():
    # The shadower's aim layer points the unit vector at PLAYER_2 (the opponent in its own view).
    agent = agents.CoverageAgent.opponent_shadower(seed=0)
    agent.set_map(_layout_from_arena("center_block"))
    state = _zero_state()
    _set_pos(state, S.PLAYER_1, 0.0, 0.0)
    _set_pos(state, S.PLAYER_2, 3.0, 4.0)  # unit vector (0.6, 0.8)
    out = agent.act(state)
    assert math.isclose(out[2], 0.6, abs_tol=1e-9)
    assert math.isclose(out[3], 0.8, abs_tol=1e-9)


def test_aim_fire_schedule_is_reusable_and_resettable():
    sched = agents.AimFireSchedule(aim_period=8, fire_period=4)
    first = sched.aim_fire()
    for _ in range(3):
        sched.aim_fire()
    sched.reset()
    assert sched.aim_fire() == first  # deterministic restart at phase 0


# --- determinism -----------------------------------------------------------------------------


def test_coverage_same_seed_same_action_sequence():
    layout = _layout_from_arena("center_block")
    a = agents.CoverageAgent.aggressive(seed=7)
    b = agents.CoverageAgent.aggressive(seed=7)
    a.set_map(layout)
    b.set_map(layout)
    state = _zero_state()
    _set_pos(state, S.PLAYER_1, -8.0, -1.0)
    seq_a = [a.act(state) for _ in range(25)]
    seq_b = [b.act(state) for _ in range(25)]
    assert seq_a == seq_b


def test_coverage_reset_replays_deterministically():
    layout = _layout_from_arena("center_block")
    a = agents.CoverageAgent.aggressive(seed=3)
    a.set_map(layout)
    state = _zero_state()
    _set_pos(state, S.PLAYER_1, -8.0, -1.0)
    first = [a.act(state) for _ in range(15)]
    a.reset(seed=3)
    a.set_map(layout)
    again = [a.act(state) for _ in range(15)]
    assert first == again


def test_coverage_rejects_bad_knobs():
    with pytest.raises(ValueError):
        agents.CoverageAgent(K=0)
    with pytest.raises(ValueError):
        agents.CoverageAgent(eps=2.0)


# --- the headline: coverage family beats RandomAgent on the hard maps ------------------------


@pytest.mark.parametrize("map_name", HARD_MAPS)
def test_family_beats_random_on_hard_maps(map_name):
    # The production reproduction of the research sim result: on the HARD maps the coverage
    # family's coverage-fraction distribution does NOT overlap RandomAgent's, with a wide margin.
    layout = _layout_from_arena(map_name)
    seeds = range(5)
    family = [
        measure_coverage(agents.CoverageAgent.aggressive(seed=s), layout).coverage_fraction
        for s in seeds
    ]
    random = [measure_coverage(agents.RandomAgent(seed=s), layout).coverage_fraction for s in seeds]
    # Distributions do not overlap: the family's WORST seed beats Random's BEST seed by a margin.
    assert min(family) > max(random) + 0.2
    # The family covers a large fraction of the reachable floor.
    assert min(family) > 0.7


def test_family_beats_random_on_all_arenas():
    # A broader sweep across every shipped arena (not just the hard ones) — the family is never
    # beaten by Random, and wins strictly wherever the arena has coverage HEADROOM. On a
    # degenerate arena whose reachable floor is tiny enough that BOTH agents saturate to full
    # coverage (e.g. the nowin_test game fixture), the result is a genuine tie, not a family
    # weakness — so strict superiority is required only when Random has not already saturated.
    for arena in ARENA_DIR.glob("*.json"):
        layout = _layout_from_arena(arena.stem)
        fam = np.mean(
            [
                measure_coverage(agents.CoverageAgent.aggressive(seed=s), layout).coverage_fraction
                for s in range(3)
            ]
        )
        rnd = np.mean(
            [
                measure_coverage(agents.RandomAgent(seed=s), layout).coverage_fraction
                for s in range(3)
            ]
        )
        if rnd >= 1.0:
            # Random already covers everything reachable — the family cannot beat a perfect
            # score; assert it at least matches (never worse).
            assert fam >= rnd, f"{arena.stem}: family {fam:.3f} < random {rnd:.3f} (saturated)"
        else:
            assert fam > rnd, f"{arena.stem}: family {fam:.3f} <= random {rnd:.3f}"


# --- RandomAgent (kept) ----------------------------------------------------------------------


def test_random_agent_deterministic_given_seed():
    a = agents.RandomAgent(seed=42)
    b = agents.RandomAgent(seed=42)
    seq_a = [a.act(None) for _ in range(20)]
    seq_b = [b.act(None) for _ in range(20)]
    assert seq_a == seq_b


def test_random_agent_actions_in_range_and_length():
    a = agents.RandomAgent(seed=7)
    for _ in range(50):
        act = a.act(None)
        assert len(act) == 5
        for ch in act[:4]:
            assert -1.0 <= ch <= 1.0
        assert act[4] in (0.0, 1.0)


def test_random_agent_reset_reproduces_sequence():
    a = agents.RandomAgent(seed=3)
    first = [a.act(None) for _ in range(10)]
    a.reset(seed=3)
    again = [a.act(None) for _ in range(10)]
    assert first == again


def test_random_agent_reset_without_seed_restores_construction_seed():
    a = agents.RandomAgent(seed=99)
    first = [a.act(None) for _ in range(5)]
    a.reset()
    again = [a.act(None) for _ in range(5)]
    assert first == again


def test_random_agent_different_seeds_differ():
    a = [agents.RandomAgent(seed=1).act(None) for _ in range(10)]
    b = [agents.RandomAgent(seed=2).act(None) for _ in range(10)]
    assert a != b


# --- NoOpAgent (the stationary curriculum floor) ---------------------------------------------


def test_noop_agent_satisfies_protocol():
    # act-only is the runtime-checkable Agent surface; the stateless NoOpAgent passes.
    assert isinstance(agents.NoOpAgent(), core.Agent)


@pytest.mark.parametrize(
    "obs",
    [None, np.zeros(S.STATE_LEN, dtype=np.float32), [0.0] * S.STATE_LEN, [1.0, 2.0, 3.0]],
    ids=["none", "ndarray", "state_list", "short_list"],
)
def test_noop_agent_returns_zero_action_for_any_obs(obs):
    # Obs-agnostic: the zero action [0,0,0,0,0] regardless of the observation type / contents.
    out = agents.NoOpAgent().act(obs)
    assert out == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert len(out) == agents.ACTION_LEN


def test_noop_agent_is_stateless_and_not_map_aware():
    # The floor has no reset / set_map (stateless): act alone, never required to track a map.
    agent = agents.NoOpAgent()
    assert not hasattr(agent, "set_map")
    assert not hasattr(agent, "reset")


def test_noop_agent_is_exported():
    assert "NoOpAgent" in agents.__all__


# --- validate_action -------------------------------------------------------------------------


def test_validate_action_coerces_floats():
    out = agents.validate_action(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(isinstance(x, float) for x in out)


def test_validate_action_rejects_bad_length():
    with pytest.raises(ValueError):
        agents.validate_action([0.0, 0.0, 0.0])
