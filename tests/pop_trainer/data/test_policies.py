"""Tests for the deterministic collection policies (pure ``state -> action_list`` callables)."""

import math

import numpy as np
import pytest

from pop_trainer.core import state as S
from pop_trainer.data import policies as P


def _zero_state():
    return [0.0] * S.STATE_LEN


def test_idle_policy_is_zero_and_deterministic():
    a = P.idle_policy(_zero_state())
    assert a == [0.0, 0.0, 0.0, 0.0, 0.0]
    # Same input -> same output, every call.
    assert P.idle_policy(_zero_state()) == a
    # Length matches the wire action width.
    assert len(a) == P.ACTION_LEN


def test_idle_policy_ignores_state_content():
    s1 = _zero_state()
    s2 = list(range(S.STATE_LEN))
    assert P.idle_policy(s1) == P.idle_policy([float(x) for x in s2])


def test_constant_policy_returns_fixed_action():
    pol = P.ConstantPolicy([0.5, -0.5, 1.0, 0.0, 1.0])
    assert pol(_zero_state()) == [0.5, -0.5, 1.0, 0.0, 1.0]
    # Deterministic across calls and independent of state.
    assert pol(list(range(S.STATE_LEN))) == [0.5, -0.5, 1.0, 0.0, 1.0]


def test_constant_policy_returns_fresh_copy_each_call():
    pol = P.ConstantPolicy([1.0, 2.0, 3.0, 4.0, 5.0])
    a = pol(_zero_state())
    a[0] = 999.0  # mutate the returned list
    b = pol(_zero_state())
    assert b == [1.0, 2.0, 3.0, 4.0, 5.0]  # the policy is uncorrupted


def test_constant_policy_rejects_bad_length():
    with pytest.raises(ValueError):
        P.ConstantPolicy([0.0, 0.0, 0.0])


def test_scripted_cycle_cycles_by_step():
    cycle = [[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]]
    pol = P.ScriptedCyclePolicy(cycle)
    assert len(pol) == 3
    s = _zero_state()
    assert pol(s, 0) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert pol(s, 1) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert pol(s, 2) == [0.0, 0.0, 1.0, 0.0, 0.0]
    # wraps around
    assert pol(s, 3) == pol(s, 0)
    assert pol(s, 7) == pol(s, 1)


def test_scripted_cycle_requires_at_least_one_action():
    with pytest.raises(ValueError):
        P.ScriptedCyclePolicy([])


def test_aim_at_opponent_points_unit_vector_and_fires():
    s = _zero_state()
    # P1 at origin, P2 at (3, 4) -> unit vector (0.6, 0.8), distance 5.
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_X] = 0.0
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_Y] = 0.0
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = 3.0
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = 4.0
    a = P.aim_at_opponent_policy(s, player=S.PLAYER_1)
    assert a[0] == 0.0 and a[1] == 0.0  # no move
    assert math.isclose(a[2], 0.6, abs_tol=1e-9)
    assert math.isclose(a[3], 0.8, abs_tol=1e-9)
    assert a[4] == 1.0  # fire
    # unit length
    assert math.isclose(math.hypot(a[2], a[3]), 1.0, abs_tol=1e-9)


def test_aim_at_opponent_is_deterministic():
    s = _zero_state()
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = -2.0
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = 0.0
    assert P.aim_at_opponent_policy(s) == P.aim_at_opponent_policy(s)


def test_aim_at_opponent_degenerate_coincident_holds_fire():
    s = _zero_state()  # both tanks at origin
    a = P.aim_at_opponent_policy(s, player=S.PLAYER_1)
    assert a == [0.0, 0.0, 0.0, 0.0, 0.0]  # no divide-by-zero; fire held


def test_aim_accepts_state_dict():
    s = _zero_state()
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = 1.0
    via_dict = P.aim_at_opponent_policy({"state": s})
    via_vec = P.aim_at_opponent_policy(s)
    assert via_dict == via_vec


def test_aim_player_is_keyword_only():
    # The positional 2-arg call the collector uses must NOT bind an int to `player`; it raises
    # cleanly (TypeError: too many positional args) rather than silently mis-binding a step.
    s = _zero_state()
    with pytest.raises(TypeError):
        P.aim_at_opponent_policy(s, S.PLAYER_2)  # positional second arg rejected


def test_aim_bound_policy_p2_perspective():
    # The player-bound callable form aims the configured slot. P1 at origin, P2 at (3, 4):
    # binding P2 aims toward P1 -> the negated unit vector (-0.6, -0.8).
    s = _zero_state()
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_X] = 0.0
    s[S.PLAYER_1 * S.PLAYER_STRIDE + S.POS_Y] = 0.0
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_X] = 3.0
    s[S.PLAYER_2 * S.PLAYER_STRIDE + S.POS_Y] = 4.0
    pol = P.AimAtOpponentPolicy(player=S.PLAYER_2)
    assert pol.player == S.PLAYER_2
    a = pol(s)
    assert math.isclose(a[2], -0.6, abs_tol=1e-9)
    assert math.isclose(a[3], -0.8, abs_tol=1e-9)
    assert a[4] == 1.0
    # the factory closure agrees with the class.
    assert P.aim_at_opponent_factory(S.PLAYER_2)(s) == a


def test_aim_bound_forms_reject_bad_player():
    with pytest.raises(ValueError):
        P.AimAtOpponentPolicy(player=5)
    with pytest.raises(ValueError):
        P.aim_at_opponent_factory(5)


def test_validate_action_coerces_floats():
    out = P.validate_action(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    assert out == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(isinstance(x, float) for x in out)
