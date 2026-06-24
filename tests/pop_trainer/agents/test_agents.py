"""Contract + behaviour tests for the model-free agents in ``pop_trainer.agents``.

No socket, no env / data import — agents are exercised against bare 52-float state vectors.
"""

import math

import numpy as np
import pytest

import pop_trainer.core as core
from pop_trainer import agents
from pop_trainer.core import state as S


def _zero_state():
    return [0.0] * S.STATE_LEN


def _set_pos(state, player, x, y):
    base = player * S.PLAYER_STRIDE
    state[base + S.POS_X] = float(x)
    state[base + S.POS_Y] = float(y)


def _set_vel(state, player, x, y):
    base = player * S.PLAYER_STRIDE
    state[base + S.VEC_X] = float(x)
    state[base + S.VEC_Y] = float(y)


def _quadrant(x, y):
    """A coarse direction bucket: the sign pair of (x, y). Distinct buckets == distinct headings."""
    return (1 if x > 1e-9 else -1 if x < -1e-9 else 0, 1 if y > 1e-9 else -1 if y < -1e-9 else 0)


# --- every agent satisfies the Agent Protocol ------------------------------------------------


def _all_agents():
    return [
        agents.RandomAgent(seed=1),
        agents.IdleAgent(),
        agents.ConstantAgent([0.1, 0.2, 0.3, 0.4, 1.0]),
        agents.ScriptedCycleAgent([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]]),
        agents.AimAtPlayer2Agent(),
        agents.ExplorerAgent(),
        agents.AimSweepAgent(),
        agents.SprayAgent(),
        agents.PerimeterAgent(),
    ]


@pytest.mark.parametrize("agent", _all_agents(), ids=lambda a: type(a).__name__)
def test_agent_satisfies_protocol(agent):
    assert isinstance(agent, core.Agent)
    out = agent.act(_zero_state())
    assert len(out) == agents.ACTION_LEN


def test_package_exports_action_contract():
    for name in ("ACTION_LEN", "validate_action", "as_state_vector"):
        assert name in agents.__all__
    for cls in (
        "RandomAgent",
        "IdleAgent",
        "ConstantAgent",
        "ScriptedCycleAgent",
        "AimAtPlayer2Agent",
        "ExplorerAgent",
        "AimSweepAgent",
        "SprayAgent",
        "PerimeterAgent",
    ):
        assert cls in agents.__all__


# --- RandomAgent -----------------------------------------------------------------------------


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


# --- AimAtPlayer2Agent -----------------------------------------------------------------------


def test_aim_at_player2_unit_vector_and_fire():
    s = _zero_state()
    _set_pos(s, S.PLAYER_1, 0.0, 0.0)
    _set_pos(s, S.PLAYER_2, 3.0, 4.0)  # unit vector (0.6, 0.8), distance 5
    a = agents.AimAtPlayer2Agent().act(s)
    assert a[0] == 0.0 and a[1] == 0.0  # no move
    assert math.isclose(a[2], 0.6, abs_tol=1e-9)
    assert math.isclose(a[3], 0.8, abs_tol=1e-9)
    assert a[4] == 1.0
    assert math.isclose(math.hypot(a[2], a[3]), 1.0, abs_tol=1e-9)


def test_aim_at_player2_coincident_holds_fire():
    a = agents.AimAtPlayer2Agent().act(_zero_state())  # both tanks at origin
    assert a == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_aim_at_player2_accepts_state_dict():
    s = _zero_state()
    _set_pos(s, S.PLAYER_2, 1.0, 0.0)
    assert agents.AimAtPlayer2Agent().act({"state": s}) == agents.AimAtPlayer2Agent().act(s)


def test_aim_at_player2_player_bind_aims_other_slot():
    s = _zero_state()
    _set_pos(s, S.PLAYER_1, 0.0, 0.0)
    _set_pos(s, S.PLAYER_2, 3.0, 4.0)
    # Binding P2 aims toward P1 -> the negated unit vector (-0.6, -0.8).
    pol = agents.AimAtPlayer2Agent(player=S.PLAYER_2)
    assert pol.player == S.PLAYER_2
    a = pol.act(s)
    assert math.isclose(a[2], -0.6, abs_tol=1e-9)
    assert math.isclose(a[3], -0.8, abs_tol=1e-9)
    assert a[4] == 1.0


def test_aim_at_player2_rejects_bad_player():
    with pytest.raises(ValueError):
        agents.AimAtPlayer2Agent(player=5)


# --- Idle / Constant -------------------------------------------------------------------------


def test_idle_is_zero():
    a = agents.IdleAgent()
    assert a.act(_zero_state()) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert a.act(list(range(S.STATE_LEN))) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_constant_returns_fixed_action_fresh_copy():
    a = agents.ConstantAgent([1.0, 2.0, 3.0, 4.0, 5.0])
    out = a.act(_zero_state())
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0]
    out[0] = 999.0  # mutate the returned list
    assert a.act(_zero_state()) == [1.0, 2.0, 3.0, 4.0, 5.0]  # agent uncorrupted


def test_constant_rejects_bad_length():
    with pytest.raises(ValueError):
        agents.ConstantAgent([0.0, 0.0, 0.0])


# --- ScriptedCycleAgent ----------------------------------------------------------------------


def test_scripted_cycle_advances_internal_counter():
    cycle = [[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]]
    a = agents.ScriptedCycleAgent(cycle)
    assert len(a) == 3
    s = _zero_state()
    assert a.act(s) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert a.act(s) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert a.act(s) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert a.act(s) == [1.0, 0.0, 0.0, 0.0, 0.0]  # wraps


def test_scripted_cycle_reset_rewinds():
    a = agents.ScriptedCycleAgent([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    a.act(_zero_state())
    a.act(_zero_state())  # advance past the start
    a.reset()
    assert a.act(_zero_state()) == [1.0, 0.0, 0.0, 0.0, 0.0]  # back to the first action


def test_scripted_cycle_rejects_empty():
    with pytest.raises(ValueError):
        agents.ScriptedCycleAgent([])


# --- ExplorerAgent (POSITION coverage) -------------------------------------------------------


def test_explorer_sweeps_multiple_move_headings():
    a = agents.ExplorerAgent(dwell=4)
    s = _zero_state()
    headings = set()
    for _ in range(200):
        act = a.act(s)
        headings.add(_quadrant(act[0], act[1]))
    # The 8 compass directions + centre-return span many distinct heading buckets.
    assert len(headings) > 4


def test_explorer_reset_restarts_deterministically():
    a = agents.ExplorerAgent(dwell=4, jitter=0.3, seed=11)
    s = _zero_state()
    first = [a.act(s) for _ in range(30)]
    a.reset(seed=11)
    again = [a.act(s) for _ in range(30)]
    assert first == again


def test_explorer_early_advance_on_velocity():
    # With the tank already moving east, the explorer cuts its dwell short and advances sooner
    # than the full budget — proving it reads its own velocity to sharpen coverage.
    fast = agents.ExplorerAgent(dwell=10)
    moving = _zero_state()
    _set_vel(moving, S.PLAYER_1, 1.0, 0.0)  # committed along the first (east) heading
    steps_to_turn = 0
    for _ in range(20):
        act = fast.act(moving)
        steps_to_turn += 1
        if _quadrant(act[0], act[1]) != (1, 0):
            break
    assert steps_to_turn < 10  # advanced before the full dwell budget


# --- AimSweepAgent (AIM coverage) ------------------------------------------------------------


def test_aim_sweep_visits_multiple_quadrants_unit_length():
    period = 16
    a = agents.AimSweepAgent(period=period)
    s = _zero_state()
    quads = set()
    for _ in range(period):
        act = a.act(s)
        assert math.isclose(math.hypot(act[2], act[3]), 1.0, abs_tol=1e-9)
        quads.add(_quadrant(act[2], act[3]))
    assert len(quads) > 4  # sweeps around the full unit circle


def test_aim_sweep_moves_while_sweeping():
    a = agents.AimSweepAgent(period=8, move=(1.0, 0.0))
    act = a.act(_zero_state())
    assert (act[0], act[1]) == (1.0, 0.0)  # nonzero move heading


def test_aim_sweep_reset_zeroes_phase():
    a = agents.AimSweepAgent(period=12)
    s = _zero_state()
    first = a.act(s)
    for _ in range(5):
        a.act(s)
    a.reset()
    assert a.act(s) == first  # deterministic restart at phase 0


# --- SprayAgent (BULLET coverage) ------------------------------------------------------------


def test_spray_fires_on_every_step():
    a = agents.SprayAgent(period=16)
    s = _zero_state()
    fires = [a.act(s)[4] for _ in range(40)]
    assert all(f == 1.0 for f in fires)


def test_spray_aim_sweeps_quadrants():
    period = 16
    a = agents.SprayAgent(period=period)
    s = _zero_state()
    quads = {_quadrant(*a.act(s)[2:4]) for _ in range(period)}
    assert len(quads) > 4


def test_spray_reset_zeroes_phase():
    a = agents.SprayAgent(period=10)
    s = _zero_state()
    first = a.act(s)
    a.act(s)
    a.reset()
    assert a.act(s) == first


# --- PerimeterAgent (position EXTREMES) ------------------------------------------------------


def test_perimeter_visits_all_four_cardinals():
    dwell = 5
    a = agents.PerimeterAgent(dwell=dwell)
    s = _zero_state()
    headings = set()
    for _ in range(dwell * 4 + 2):
        act = a.act(s)
        headings.add(_quadrant(act[0], act[1]))
    assert headings == {(1, 0), (0, 1), (-1, 0), (0, -1)}


def test_perimeter_reset_restarts():
    a = agents.PerimeterAgent(dwell=3)
    s = _zero_state()
    first = [a.act(s) for _ in range(12)]
    a.reset()
    again = [a.act(s) for _ in range(12)]
    assert first == again


# --- validate_action -------------------------------------------------------------------------


def test_validate_action_coerces_floats():
    out = agents.validate_action(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(isinstance(x, float) for x in out)


def test_validate_action_rejects_bad_length():
    with pytest.raises(ValueError):
        agents.validate_action([0.0, 0.0, 0.0])
