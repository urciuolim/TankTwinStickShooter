"""Regression net for tank_twin.rewards (extracted from PythonScripts/tank_env.py, M1 1.1).

The 2021 ``TankEnv.step`` reward branch, upgraded to gymnasium semantics. These
tests are a TRUTH TABLE over (winner, survivor on/off, max_steps-hit,
lost_connection) and pin the exact ``(reward, terminated, truncated)`` triple plus
the reward NUMBERS. The legacy used a single ``done``; this asserts the
done -> (terminated, truncated) split — in particular that lost_connection and a
time-limit cutoff TRUNCATE (never terminate), while a winner TERMINATES.
"""

import pytest

from tank_twin.rewards import (
    L1_MAX,
    action_cost_per_step,
    shaped_step_reward,
    step_reward,
    time_penalty_per_step,
)

# --- ongoing round (no winner, not done) ----------------------------------


def test_ongoing_round_returns_time_reward_no_boundary():
    # No winner, not done -> shaping reward, neither terminated nor truncated.
    assert step_reward(winner=None, done=False, time_reward=0.0) == (0.0, False, False)


def test_ongoing_round_carries_nonzero_time_reward():
    assert step_reward(winner=None, done=False, time_reward=-0.01) == (-0.01, False, False)


# --- decided games: winner terminates -------------------------------------


def test_agent_win_is_plus_one_terminated():
    # winner == P1 (0) -> +1, terminated.
    assert step_reward(winner=0, time_reward=-0.01) == (1, True, False)


def test_opponent_win_is_minus_one_terminated():
    # winner == opponent (1) -> -1, terminated.
    assert step_reward(winner=1, time_reward=-0.01) == (-1, True, False)


def test_explicit_draw_minus_one_winner_is_time_reward_terminated():
    # winner == -1 is an explicit draw this step: keeps time_reward, still terminates.
    assert step_reward(winner=-1, time_reward=0.0) == (0.0, True, False)


def test_done_without_winner_is_time_reward_terminated():
    # The game declared the round over (legacy "done" in received) with no winner key.
    assert step_reward(winner=None, done=True, time_reward=0.0) == (0.0, True, False)


# --- survivor mode flips the terminal reward ------------------------------


def test_survivor_no_winner_terminal_is_plus_one():
    # Survivor + terminal + no decided winner -> +1 (you survived).
    assert step_reward(winner=None, done=True, survivor=True, time_reward=0.0) == (1, True, False)


def test_survivor_explicit_draw_scores_minus_one():
    # LEGACY-FAITHFUL: a winner==-1 draw still reports a winner KEY, and the legacy
    # survivor flip keyed off `"winner" in info` (key presence), NOT the value ->
    # scores -1. Only a winner-key-ABSENT terminal (bare done) scores +1.
    assert step_reward(winner=-1, survivor=True, time_reward=0.0) == (-1, True, False)


def test_survivor_agent_win_flips_to_minus_one():
    # Legacy quirk preserved: under survivor, ANY decided game (incl. a P1 win)
    # scores -1 because the winner branch ran first, then survivor overwrote it.
    assert step_reward(winner=0, survivor=True, time_reward=0.0) == (-1, True, False)


def test_survivor_opponent_win_is_minus_one():
    assert step_reward(winner=1, survivor=True, time_reward=0.0) == (-1, True, False)


def test_survivor_does_not_touch_ongoing_round():
    # Survivor only flips on a terminal step; an ongoing round is unaffected.
    result = step_reward(winner=None, done=False, survivor=True, time_reward=0.0)
    assert result == (0.0, False, False)


# --- max_steps reached truncates (NOT terminated) -------------------------


def test_max_steps_reached_truncates_with_time_reward():
    # Time-limit cutoff: truncated, not terminated; carries the step's shaping reward.
    assert step_reward(winner=None, max_steps_reached=True, time_reward=-0.01) == (
        -0.01,
        False,
        True,
    )


def test_max_steps_and_winner_same_step_prefers_termination():
    # A real terminal result wins over a coincident time-limit cutoff: terminated,
    # NOT truncated (gymnasium: a genuine end is a termination).
    assert step_reward(winner=0, max_steps_reached=True, time_reward=-0.01) == (1, True, False)


# --- lost connection truncates (explicitly NOT terminated) ----------------


def test_lost_connection_truncates_not_terminated():
    # Legacy returned (state, 0, done=True, {"lost_connection":True}); under
    # gymnasium that is a TRUNCATION with reward 0 — pin this hard.
    assert step_reward(winner=None, lost_connection=True, time_reward=-0.01) == (0.0, False, True)


def test_lost_connection_reward_is_zero_ignores_time_reward():
    # The transport-failure reward is a flat 0.0, regardless of shaping.
    reward, terminated, truncated = step_reward(lost_connection=True, time_reward=0.5)
    assert reward == 0.0
    assert terminated is False
    assert truncated is True


def test_lost_connection_overrides_winner_and_survivor():
    # A dropped connection is not a game result: even with a winner/survivor set,
    # it is a reward-0 truncation.
    assert step_reward(winner=0, survivor=True, lost_connection=True, time_reward=-0.01) == (
        0.0,
        False,
        True,
    )


# --- terminal/truncated are mutually consistent across the truth table ----


@pytest.mark.parametrize("winner", [-1, 0, 1, None])
@pytest.mark.parametrize("survivor", [False, True])
def test_terminated_and_truncated_never_both_true_for_decided_or_done(winner, survivor):
    # When the game is done/decided (and not a lost connection / time cutoff),
    # we terminate and do not truncate.
    done = winner is not None
    reward, terminated, truncated = step_reward(winner=winner, done=done, survivor=survivor)
    if done:
        assert terminated is True
        assert truncated is False


# === budget-based reward (CTO; the new env path) =======================================
#
# The legacy step_reward tests ABOVE are UNTOUCHED and stay green. These pin the new
# budget reward: per-step time + (L1-scaled) action penalties that ACCRUE every step, with
# the ±1 terminal ADDED (not overwritten). max_episode_length == the env max_steps (300).

MAX_EP = 300


# --- budget -> per-step conversion -----------------------------------------


def test_time_total_minus_one_converts_to_minus_one_over_300_per_step():
    # time_total -1.0 over 300 steps -> -1/300 per step.
    assert time_penalty_per_step(-1.0, MAX_EP) == -1.0 / 300


def test_full_episode_time_penalty_sums_to_time_total():
    per_step = time_penalty_per_step(-1.0, MAX_EP)
    assert per_step * MAX_EP == pytest.approx(-1.0)


# --- action-cost L1 scaling ------------------------------------------------


def test_action_cost_is_zero_at_zero_action():
    assert action_cost_per_step([0, 0, 0, 0, 0], -0.1, MAX_EP) == 0.0


def test_action_cost_at_constant_max_action_sums_to_action_total():
    # At a CONSTANT MAX action (L1 == L1_MAX == 5), a full episode totals action_total -0.1.
    max_action = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert sum(abs(a) for a in max_action) == L1_MAX
    per_step = action_cost_per_step(max_action, -0.1, MAX_EP)
    assert per_step * MAX_EP == pytest.approx(-0.1)


def test_action_cost_is_linear_in_l1_magnitude():
    # Half the L1 magnitude -> half the per-step cost (linear).
    full = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    half = action_cost_per_step([0.5, 0.5, 0.5, 0.5, 0.5], -0.1, MAX_EP)
    assert half == pytest.approx(full / 2)


def test_action_cost_is_sign_insensitive_l1():
    # L1 uses magnitudes: a negative-saturated action costs the same as positive-saturated.
    pos = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    neg = action_cost_per_step([-1.0, -1.0, -1.0, -1.0, -1.0], -0.1, MAX_EP)
    assert pos == neg


# --- backward-compat: budgets 0 + win/loss ±1 == today's numbers ------------


def test_budget_zero_continuing_matches_legacy_zero():
    # budgets 0, no winner -> 0.0, neither terminated nor truncated (legacy time_reward=0).
    assert shaped_step_reward(winner=None, done=False, time_penalty=0.0, action_cost=0.0) == (
        0.0,
        False,
        False,
    )


def test_budget_zero_win_is_plus_one_terminated_like_legacy():
    assert shaped_step_reward(winner=0, time_penalty=0.0, action_cost=0.0) == (1.0, True, False)


def test_budget_zero_loss_is_minus_one_terminated_like_legacy():
    assert shaped_step_reward(winner=1, time_penalty=0.0, action_cost=0.0) == (-1.0, True, False)


def test_budget_zero_draw_is_zero_terminated_like_legacy():
    assert shaped_step_reward(winner=-1, time_penalty=0.0, action_cost=0.0) == (0.0, True, False)


def test_budget_zero_done_without_winner_is_zero_terminated_like_legacy():
    assert shaped_step_reward(winner=None, done=True, time_penalty=0.0, action_cost=0.0) == (
        0.0,
        True,
        False,
    )


@pytest.mark.parametrize("winner", [-1, 0, 1, None])
@pytest.mark.parametrize("survivor", [False, True])
def test_budget_zero_matches_legacy_step_reward_exactly(winner, survivor):
    # The hard backward-compat contract: budgets 0 + win/loss ±1 -> EXACTLY what the legacy
    # step_reward returns (with its time_reward=0), across the whole winner/survivor table.
    done = winner is not None
    legacy = step_reward(winner=winner, done=done, survivor=survivor, time_reward=0.0)
    shaped = shaped_step_reward(
        winner=winner,
        done=done,
        survivor=survivor,
        time_penalty=0.0,
        action_cost=0.0,
        win_reward=1.0,
        loss_reward=-1.0,
    )
    assert shaped == legacy


# --- penalties ACCRUE; ±1 is ADDED (not overwritten) ------------------------


def test_continuing_step_returns_accrued_penalties():
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, truncated = shaped_step_reward(
        winner=None, done=False, time_penalty=tp, action_cost=ac
    )
    assert reward == pytest.approx(tp + ac)
    assert terminated is False
    assert truncated is False


def test_decided_loss_adds_terminal_onto_penalties_not_overwrite():
    # A late loss step: -1 terminal ADDED on top of the per-step time + action penalties.
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, _ = shaped_step_reward(
        winner=1, time_penalty=tp, action_cost=ac, win_reward=1.0, loss_reward=-1.0
    )
    assert terminated is True
    assert reward == pytest.approx(-1.0 + tp + ac)
    # And strictly worse than a bare -1 (the penalties pushed it below the terminal).
    assert reward < -1.0


def test_decided_win_adds_terminal_onto_penalties():
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, _ = shaped_step_reward(
        winner=0, time_penalty=tp, action_cost=ac, win_reward=1.0, loss_reward=-1.0
    )
    assert terminated is True
    assert reward == pytest.approx(1.0 + tp + ac)


def test_worst_case_late_loss_episode_return_is_minus_two_point_one():
    # The CTO's worst case: a full episode at CONSTANT MAX action that ends in a loss on the
    # final step. Episode RETURN = 300*time/step + 300*action/step + loss = -1 -0.1 -1 = -2.1.
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    total = 0.0
    for step in range(1, MAX_EP + 1):
        if step < MAX_EP:
            r, _, _ = shaped_step_reward(winner=None, done=False, time_penalty=tp, action_cost=ac)
        else:
            r, _, _ = shaped_step_reward(
                winner=1, time_penalty=tp, action_cost=ac, win_reward=1.0, loss_reward=-1.0
            )
        total += r
    assert total == pytest.approx(-2.1)


# --- truncation / lost-connection semantics carry over ----------------------


def test_budget_max_steps_truncates_with_accrued_penalty():
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, truncated = shaped_step_reward(
        winner=None, max_steps_reached=True, time_penalty=tp, action_cost=ac
    )
    assert terminated is False
    assert truncated is True
    assert reward == pytest.approx(tp + ac)


def test_budget_lost_connection_is_flat_zero_truncation():
    # No shaping accrues on a dropped connection (matches step_reward): flat reward 0.0.
    reward, terminated, truncated = shaped_step_reward(
        winner=0, lost_connection=True, time_penalty=-0.5, action_cost=-0.5
    )
    assert reward == 0.0
    assert terminated is False
    assert truncated is True


def test_budget_survivor_no_winner_terminal_adds_win_reward_to_penalties():
    # Survivor + winner-key-absent terminal -> win_reward ADDED to accrued penalties.
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, _ = shaped_step_reward(
        winner=None, done=True, survivor=True, time_penalty=tp, action_cost=ac
    )
    assert terminated is True
    assert reward == pytest.approx(1.0 + tp + ac)


def test_budget_survivor_reported_winner_adds_loss_reward_to_penalties():
    # Survivor + any reported winner (incl. a P1 win) -> loss_reward ADDED (legacy quirk).
    tp = time_penalty_per_step(-1.0, MAX_EP)
    ac = action_cost_per_step([1.0, 1.0, 1.0, 1.0, 1.0], -0.1, MAX_EP)
    reward, terminated, _ = shaped_step_reward(
        winner=0, survivor=True, time_penalty=tp, action_cost=ac
    )
    assert terminated is True
    assert reward == pytest.approx(-1.0 + tp + ac)


# --- configurable win/loss --------------------------------------------------


def test_configurable_win_loss_rewards():
    reward_win, _, _ = shaped_step_reward(winner=0, win_reward=2.5, loss_reward=-3.0)
    reward_loss, _, _ = shaped_step_reward(winner=1, win_reward=2.5, loss_reward=-3.0)
    assert reward_win == pytest.approx(2.5)
    assert reward_loss == pytest.approx(-3.0)
