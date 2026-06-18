"""Regression net for tank_twin.rewards (extracted from PythonScripts/tank_env.py, M1 1.1).

The 2021 ``TankEnv.step`` reward branch, upgraded to gymnasium semantics. These
tests are a TRUTH TABLE over (winner, survivor on/off, max_steps-hit,
lost_connection) and pin the exact ``(reward, terminated, truncated)`` triple plus
the reward NUMBERS. The legacy used a single ``done``; this asserts the
done -> (terminated, truncated) split — in particular that lost_connection and a
time-limit cutoff TRUNCATE (never terminate), while a winner TERMINATES.
"""

import pytest

from tank_twin.rewards import step_reward

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
