"""Unit net for tank_twin.determinism_check (M1 Workstream C, part 2).

The pure trajectory-diff that is the verdict half of the timeScale-determinism
gate. These tests exercise it on SYNTHETIC trajectories (no Unity, no socket, no
torch): identical trajectories -> zero diff + bitwise-identical; an injected
divergence -> the correct first-divergence step + max-diff + field group; and the
RL invariants (length / winner / total-reward mismatches) are detected.

The 52-float layout matches tank_twin.observation: P1 = indices 0..25, P2 = 26..51.
"""

from tank_twin.determinism_check import (
    MOTION_FLOOR,
    compute_invariants,
    diff_trajectories,
    motion_summary,
)


def _state(fill=0.0):
    """A single 52-float state, every element set to ``fill`` (a valid-width row)."""
    return [fill] * 52


def _trajectory(n_steps, fill=0.0):
    """An ``n_steps`` x 52 trajectory, every element ``fill``."""
    return [_state(fill) for _ in range(n_steps)]


# --- identical trajectories -> zero diff, bitwise identical -------------------


def test_identical_trajectories_zero_diff():
    traj = _trajectory(10, fill=1.5)
    result = diff_trajectories(traj, traj)
    assert result.bitwise_identical is True
    assert result.max_abs_diff == 0.0
    assert result.first_divergence_step is None
    assert result.compared_steps == 10
    assert all(v == 0.0 for v in result.per_group_max_diff.values())


def test_identical_distinct_objects_still_bitwise_identical():
    # Two SEPARATELY constructed but value-equal trajectories (the real harness
    # builds two independent recordings).
    a = _trajectory(5, fill=2.0)
    b = _trajectory(5, fill=2.0)
    result = diff_trajectories(a, b)
    assert result.bitwise_identical is True
    assert result.max_abs_diff == 0.0
    assert result.first_divergence_step is None


def test_empty_trajectories_are_identical():
    result = diff_trajectories([], [])
    assert result.bitwise_identical is True
    assert result.max_abs_diff == 0.0
    assert result.first_divergence_step is None
    assert result.compared_steps == 0


# --- injected divergence -> correct first step + max diff + group -------------


def test_injected_divergence_first_step_and_max_diff():
    a = _trajectory(8, fill=0.0)
    b = _trajectory(8, fill=0.0)
    # Perturb P1 pos_x (index 0) at step 3 by 0.25.
    b[3][0] = 0.25
    result = diff_trajectories(a, b)
    assert result.bitwise_identical is False
    assert result.first_divergence_step == 3
    assert result.max_abs_diff == 0.25
    assert result.per_group_max_diff["p1_tank"] == 0.25
    assert result.per_group_max_diff["p2_tank"] == 0.0


def test_divergence_attributed_to_p2_bullets_group():
    a = _trajectory(6, fill=0.0)
    b = _trajectory(6, fill=0.0)
    # P2 bullets occupy indices 32..51; perturb index 40 at step 5.
    b[5][40] = -3.0
    result = diff_trajectories(a, b)
    assert result.first_divergence_step == 5
    assert result.max_abs_diff == 3.0
    assert result.per_group_max_diff["p2_bullets"] == 3.0
    assert result.per_group_max_diff["p1_tank"] == 0.0
    assert result.per_group_max_diff["p2_tank"] == 0.0


def test_earliest_divergence_reported_when_multiple():
    a = _trajectory(10, fill=0.0)
    b = _trajectory(10, fill=0.0)
    b[7][2] = 1.0  # later, larger
    b[2][1] = 0.1  # earlier, smaller
    result = diff_trajectories(a, b)
    # first_divergence_step is the EARLIEST differing step (2), not the largest.
    assert result.first_divergence_step == 2
    # max_abs_diff is over the whole prefix (the larger 1.0).
    assert result.max_abs_diff == 1.0


# --- RL invariants: length / winner / total reward ----------------------------


def test_length_mismatch_detected_and_prefix_diffed():
    a = _trajectory(10, fill=1.0)
    b = _trajectory(7, fill=1.0)  # shorter
    result = diff_trajectories(a, b)
    assert result.compared_steps == 7  # diff runs over the overlap
    assert result.bitwise_identical is False  # length mismatch is not identical
    inv = result.invariants
    assert inv.same_length is False
    assert inv.length_a == 10
    assert inv.length_b == 7


def test_winner_invariant_match_and_mismatch():
    a = _trajectory(4)
    b = _trajectory(4)
    # winners reported only on the terminal step.
    inv_match = compute_invariants(
        a, b, winners_a=[None, None, None, 0], winners_b=[None, None, None, 0]
    )
    assert inv_match.same_winner is True
    assert inv_match.winner_a == 0
    assert inv_match.winner_b == 0
    assert inv_match.all_match is True

    inv_mismatch = compute_invariants(
        a, b, winners_a=[None, None, None, 0], winners_b=[None, None, None, 1]
    )
    assert inv_mismatch.same_winner is False
    assert inv_mismatch.winner_a == 0
    assert inv_mismatch.winner_b == 1
    assert inv_mismatch.all_match is False


def test_total_reward_invariant_reflects_winner():
    a = _trajectory(3)
    b = _trajectory(3)
    # P1 win -> +1; P1 loss -> -1; their total rewards differ.
    inv = compute_invariants(a, b, winners_a=[None, None, 0], winners_b=[None, None, 1])
    assert inv.total_reward_a == 1.0
    assert inv.total_reward_b == -1.0
    assert inv.same_total_reward is False


def test_no_winner_keys_means_draw_zero_reward():
    a = _trajectory(5)
    b = _trajectory(5)
    inv = compute_invariants(a, b, winners_a=None, winners_b=None)
    assert inv.winner_a is None
    assert inv.winner_b is None
    assert inv.total_reward_a == 0.0
    assert inv.total_reward_b == 0.0
    assert inv.same_winner is True
    assert inv.same_total_reward is True
    assert inv.all_match is True


def test_winner_is_last_nonnull_signal():
    # An explicit -1 (draw) reported on the terminal step is honored as the winner.
    a = _trajectory(3)
    inv = compute_invariants(a, a, winners_a=[None, None, -1], winners_b=[None, None, -1])
    assert inv.winner_a == -1
    assert inv.winner_b == -1
    # winner == -1 scores 0 reward (draw), via the real step_reward.
    assert inv.total_reward_a == 0.0


# --- serialization is strict-JSON-clean (no numpy scalars) --------------------


def test_to_dict_is_json_clean():
    import json

    a = _trajectory(4, fill=0.0)
    b = _trajectory(4, fill=0.0)
    b[2][0] = 0.5
    result = diff_trajectories(
        a, b, winners_a=[None, None, None, 0], winners_b=[None, None, None, 0]
    )
    d = result.to_dict()
    # Round-trips through strict json with no numpy types leaking through.
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["first_divergence_step"] == 2
    assert decoded["max_abs_diff"] == 0.5
    assert decoded["invariants"]["winner_a"] == 0
    assert decoded["invariants"]["all_match"] is True


def test_to_dict_first_divergence_none_when_identical():
    traj = _trajectory(3, fill=1.0)
    d = diff_trajectories(traj, traj).to_dict()
    assert d["first_divergence_step"] is None
    assert d["bitwise_identical"] is True


# --- motion / non-degeneracy: 0.0 must mean "identical DESPITE real motion" ----


def test_frozen_trajectory_is_degenerate():
    # Every step identical -> zero range, zero step delta -> degenerate.
    frozen = _trajectory(20, fill=3.0)
    m = motion_summary(frozen)
    assert m.p1_x_range == 0.0
    assert m.p1_y_range == 0.0
    assert m.p2_x_range == 0.0
    assert m.p2_y_range == 0.0
    assert m.p1_max_step_delta == 0.0
    assert m.p2_max_step_delta == 0.0
    assert m.non_degenerate is False


def test_moving_trajectory_is_non_degenerate():
    # Drive P1 x (index 0) and P2 y (index 27) across several world units.
    traj = _trajectory(10, fill=0.0)
    for i, state in enumerate(traj):
        state[0] = float(i)  # P1 pos_x: 0..9 -> range 9.0
        state[27] = float(2 * i)  # P2 pos_y: 0..18 -> range 18.0
    m = motion_summary(traj)
    assert m.p1_x_range == 9.0
    assert m.p1_y_range == 0.0
    assert m.p2_y_range == 18.0
    # Max step delta: P1 moves 1.0/step in x; P2 moves 2.0/step in y.
    assert m.p1_max_step_delta == 1.0
    assert m.p2_max_step_delta == 2.0
    assert m.non_degenerate is True
    assert m.motion_floor == MOTION_FLOOR


def test_motion_just_below_floor_is_degenerate():
    # Largest range strictly at/under the floor -> degenerate (conservative bool).
    traj = _trajectory(5, fill=0.0)
    for i, state in enumerate(traj):
        # Range over 5 steps = (4/4)*MOTION_FLOOR == MOTION_FLOOR exactly; not > floor.
        state[0] = (i / 4.0) * MOTION_FLOOR
    m = motion_summary(traj)
    assert m.p1_x_range == MOTION_FLOOR
    assert m.non_degenerate is False  # strictly-greater-than, so == floor fails


def test_motion_summary_single_step_is_degenerate():
    # Fewer than two steps cannot move.
    m = motion_summary(_trajectory(1, fill=5.0))
    assert m.non_degenerate is False
    assert m.p1_max_step_delta == 0.0


def test_motion_summary_empty_is_degenerate():
    m = motion_summary([])
    assert m.non_degenerate is False


def test_motion_step_delta_is_euclidean():
    # A diagonal move of (3, 4) per step has Euclidean delta 5.0 (3-4-5 triangle).
    traj = _trajectory(3, fill=0.0)
    for i, state in enumerate(traj):
        state[0] = 3.0 * i  # P1 pos_x
        state[1] = 4.0 * i  # P1 pos_y
    m = motion_summary(traj)
    assert m.p1_max_step_delta == 5.0


def test_motion_summary_to_dict_is_json_clean():
    import json

    traj = _trajectory(4, fill=0.0)
    for i, state in enumerate(traj):
        state[0] = float(i * 3)
    d = motion_summary(traj).to_dict()
    decoded = json.loads(json.dumps(d))
    assert decoded["p1_x_range"] == 9.0
    assert decoded["non_degenerate"] is True
    assert decoded["motion_floor"] == MOTION_FLOOR


def test_custom_motion_floor_respected():
    traj = _trajectory(4, fill=0.0)
    for i, state in enumerate(traj):
        state[0] = float(i)  # range 3.0
    assert motion_summary(traj, motion_floor=2.0).non_degenerate is True
    assert motion_summary(traj, motion_floor=10.0).non_degenerate is False


# --- NaN handling: a NaN that appears in only one trajectory is a divergence ---


def test_nan_mismatch_is_divergence():
    a = _trajectory(4, fill=0.0)
    b = _trajectory(4, fill=0.0)
    b[1][5] = float("nan")
    result = diff_trajectories(a, b)
    assert result.first_divergence_step == 1
    assert result.bitwise_identical is False
