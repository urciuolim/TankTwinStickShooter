"""Unit tests for ``pop_trainer.env.rewards`` (the pure budget-based shaped reward).

The reward is a pure function (no socket / gym / numpy), so it is pinned in isolation:
the time penalty accrues every step, the win/loss terminal is ADDED on the decided step,
non-terminal steps get only the time penalty, and with ``time_penalty == 0`` the reward
reduces to the plain ±1 terminal (the budgets==0 -> ±1-terminal equivalence).
"""

import pytest

from pop_trainer.env.rewards import (
    PLAYER_1,
    shaped_step_reward,
    time_penalty_per_step,
)

# A round per-step penalty for exact-equality assertions (not the -1/300 default).
TP = -0.01


def test_time_penalty_per_step_converts_budget():
    assert time_penalty_per_step(-1.0, 300) == pytest.approx(-1.0 / 300)
    assert time_penalty_per_step(-3.0, 3) == pytest.approx(-1.0)
    # A zero budget is a legitimate (no shaping) configuration.
    assert time_penalty_per_step(0.0, 100) == 0.0


def test_time_penalty_per_step_rejects_nonpositive_max_steps():
    with pytest.raises(ValueError):
        time_penalty_per_step(-1.0, 0)
    with pytest.raises(ValueError):
        time_penalty_per_step(-1.0, -5)


def test_continuing_step_gets_only_time_penalty():
    reward, terminated, truncated = shaped_step_reward(winner=None, time_penalty=TP)
    assert reward == pytest.approx(TP)
    assert terminated is False
    assert truncated is False


def test_win_adds_win_reward_on_decided_step():
    reward, terminated, truncated = shaped_step_reward(
        winner=PLAYER_1, time_penalty=TP, win_reward=1.0, loss_reward=-1.0
    )
    # Terminal is ADDED on top of the accrued time penalty (a win late still pays the time).
    assert reward == pytest.approx(1.0 + TP)
    assert terminated is True
    assert truncated is False


def test_loss_adds_loss_reward_on_decided_step():
    reward, terminated, truncated = shaped_step_reward(
        winner=1, time_penalty=TP, win_reward=1.0, loss_reward=-1.0
    )
    assert reward == pytest.approx(-1.0 + TP)
    assert terminated is True
    assert truncated is False


def test_draw_contributes_only_time_penalty_but_terminates():
    # An explicit draw (winner == -1) terminates with NO terminal component.
    reward, terminated, truncated = shaped_step_reward(winner=-1, time_penalty=TP)
    assert reward == pytest.approx(TP)
    assert terminated is True
    assert truncated is False


def test_bare_done_terminates_with_only_time_penalty():
    # A winner-less ``done`` (round resolved without a winner key) terminates, no ±1.
    reward, terminated, truncated = shaped_step_reward(done=True, time_penalty=TP)
    assert reward == pytest.approx(TP)
    assert terminated is True
    assert truncated is False


def test_max_steps_truncates_undecided_step():
    reward, terminated, truncated = shaped_step_reward(
        winner=None, time_penalty=TP, max_steps_reached=True
    )
    assert reward == pytest.approx(TP)
    assert terminated is False
    assert truncated is True


def test_decided_step_takes_precedence_over_max_steps():
    # If the game decides on the very step max_steps is hit, terminated wins (not truncated).
    reward, terminated, truncated = shaped_step_reward(
        winner=PLAYER_1, time_penalty=TP, max_steps_reached=True
    )
    assert reward == pytest.approx(1.0 + TP)
    assert terminated is True
    assert truncated is False


def test_lost_connection_is_flat_zero_truncation():
    # No shaping accrues on a dropped connection; reward is exactly 0.0, truncated.
    reward, terminated, truncated = shaped_step_reward(
        lost_connection=True, time_penalty=TP, winner=PLAYER_1
    )
    assert reward == 0.0
    assert terminated is False
    assert truncated is True


def test_time_penalty_accrues_across_a_full_episode():
    # The per-step penalty accrues every step; summing N continuing steps == N * penalty.
    n = 50
    total = sum(shaped_step_reward(winner=None, time_penalty=TP)[0] for _ in range(n))
    assert total == pytest.approx(n * TP)


# --- the budgets==0 -> ±1-terminal equivalence (SANITY) ---------------------------------


def test_zero_time_penalty_reduces_to_plus_one_terminal_on_win():
    reward, terminated, truncated = shaped_step_reward(
        winner=PLAYER_1, time_penalty=0.0, win_reward=1.0, loss_reward=-1.0
    )
    assert (reward, terminated, truncated) == (1.0, True, False)


def test_zero_time_penalty_reduces_to_minus_one_terminal_on_loss():
    reward, terminated, truncated = shaped_step_reward(
        winner=1, time_penalty=0.0, win_reward=1.0, loss_reward=-1.0
    )
    assert (reward, terminated, truncated) == (-1.0, True, False)


def test_zero_time_penalty_continuing_is_zero():
    reward, terminated, truncated = shaped_step_reward(winner=None, time_penalty=0.0)
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_zero_time_penalty_draw_is_zero_terminal():
    reward, terminated, truncated = shaped_step_reward(winner=-1, time_penalty=0.0)
    assert (reward, terminated, truncated) == (0.0, True, False)


# --- survivor-mode flip (key-presence rule) ---------------------------------------------


def test_survivor_any_reported_winner_scores_loss_reward():
    # P1 "win" under survivor: a winner KEY was reported -> loss_reward (the asymmetry).
    reward, _, _ = shaped_step_reward(
        winner=PLAYER_1, survivor=True, time_penalty=TP, win_reward=1.0, loss_reward=-1.0
    )
    assert reward == pytest.approx(-1.0 + TP)
    # An explicit draw (winner == -1) also counts as a reported winner -> loss_reward.
    reward_draw, _, _ = shaped_step_reward(
        winner=-1, survivor=True, time_penalty=TP, loss_reward=-1.0
    )
    assert reward_draw == pytest.approx(-1.0 + TP)


def test_survivor_winner_absent_terminal_scores_win_reward():
    # A bare ``done`` with NO winner key -> win_reward (you survived).
    reward, terminated, _ = shaped_step_reward(
        done=True, survivor=True, time_penalty=TP, win_reward=1.0
    )
    assert reward == pytest.approx(1.0 + TP)
    assert terminated is True


def test_survivor_keeps_time_penalty_component():
    # The flip replaces only the terminal component; the accrued time penalty is KEPT.
    reward, _, _ = shaped_step_reward(done=True, survivor=True, time_penalty=TP, win_reward=2.0)
    assert reward == pytest.approx(2.0 + TP)
