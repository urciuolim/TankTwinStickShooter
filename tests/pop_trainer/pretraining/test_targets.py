"""Tests for pop_trainer.pretraining.targets — per-group extraction + normalization (pure numpy).

Covers that the per-group target extraction matches the core/state.py accessors on a hand-built
state, the bullet sentinel masking, and that the normalization stats are TRAIN-only (no leak) with
bullet-position stats over present slots only.
"""

from __future__ import annotations

import numpy as np

from pop_trainer.core import state as st
from pop_trainer.pretraining import targets as T


def _make_state(p1_pos, p1_vel, p1_aim, p2_pos, p2_vel, p2_aim, bullets=None) -> np.ndarray:
    """Build one 52-float state; ``bullets`` maps (player, slot) -> (x, y, vx, vy)."""
    s = np.full(st.STATE_LEN, st.ABSENT_BULLET_SENTINEL, dtype=np.float32)
    for player, (pos, vel, aim) in (
        (st.PLAYER_1, (p1_pos, p1_vel, p1_aim)),
        (st.PLAYER_2, (p2_pos, p2_vel, p2_aim)),
    ):
        base = player * st.PLAYER_STRIDE
        s[base + st.POS_X], s[base + st.POS_Y] = pos
        s[base + st.VEC_X], s[base + st.VEC_Y] = vel
        s[base + st.AIM_X], s[base + st.AIM_Y] = aim
    bullets = bullets or {}
    for (player, slot), (bx, by, bvx, bvy) in bullets.items():
        idx = list(st.bullet_field_indices(player))[slot]
        s[idx : idx + st.BULLET_STRIDE] = (bx, by, bvx, bvy)
    return s


def test_player_targets_match_state_accessors():
    s = _make_state(
        p1_pos=(1.0, 2.0),
        p1_vel=(0.1, 0.2),
        p1_aim=(1.0, 0.0),
        p2_pos=(-3.0, 4.0),
        p2_vel=(-0.5, 0.0),
        p2_aim=(0.0, -1.0),
    )
    states = s[None, :]
    pos = T.player_position_targets(states)[0]
    vel = T.player_velocity_targets(states)[0]
    aim = T.player_aim_targets(states)[0]
    assert tuple(pos) == (*st.position(s, st.PLAYER_1), *st.position(s, st.PLAYER_2))
    assert tuple(vel) == (*st.velocity(s, st.PLAYER_1), *st.velocity(s, st.PLAYER_2))
    assert tuple(aim) == (*st.aim(s, st.PLAYER_1), *st.aim(s, st.PLAYER_2))


def test_bullet_presence_and_sentinel_mask():
    # P1 slot 0 present; P2 slot 3 present; all others absent (sentinel).
    s = _make_state(
        p1_pos=(0, 0),
        p1_vel=(0, 0),
        p1_aim=(1, 0),
        p2_pos=(0, 0),
        p2_vel=(0, 0),
        p2_aim=(1, 0),
        bullets={(st.PLAYER_1, 0): (2.0, 3.0, 1.0, 0.0), (st.PLAYER_2, 3): (5.0, 6.0, 0.0, 1.0)},
    )
    presence, pos, mask = T.bullet_targets(s[None, :])
    presence = presence[0]
    # 10 slots: P1[0..4] then P2[0..4]. Present: index 0 and index 5+3=8.
    expected = np.zeros(10)
    expected[0] = 1.0
    expected[8] = 1.0
    assert np.array_equal(presence, expected)
    # present slot positions carried through; absent slots keep the sentinel pos_x.
    assert pos[0, 0] == 2.0 and pos[0, 1] == 3.0
    assert pos[0, 16] == 5.0 and pos[0, 17] == 6.0
    assert pos[0, 2] == st.ABSENT_BULLET_SENTINEL  # P1 slot1 absent
    # mask repeats presence twice per slot.
    assert mask[0, 0] == 1.0 and mask[0, 1] == 1.0
    assert mask[0, 2] == 0.0


def test_negative_x_bullet_is_present_not_absent():
    # Regression: the arena is centered on the origin, so a bullet at a real NEGATIVE x is a
    # valid on-board bullet. Only the -100 sentinel marks an absent slot.
    assert st.bullet_present(-3.0) is True
    # P1 slot 0 holds a left-half bullet (x=-3.0); P1 slot 1 is the absent sentinel.
    s = _make_state(
        p1_pos=(0, 0),
        p1_vel=(0, 0),
        p1_aim=(1, 0),
        p2_pos=(0, 0),
        p2_vel=(0, 0),
        p2_aim=(1, 0),
        bullets={(st.PLAYER_1, 0): (-3.0, -4.0, 1.0, 0.0)},
    )
    presence, pos, mask = T.bullet_targets(s[None, :])
    # Negative-x slot is PRESENT, its position carried through, and UNMASKED.
    assert presence[0, 0] == 1.0
    assert pos[0, 0] == -3.0 and pos[0, 1] == -4.0
    assert mask[0, 0] == 1.0 and mask[0, 1] == 1.0
    # Sentinel slot (P1 slot 1) is ABSENT and masked out.
    assert presence[0, 1] == 0.0
    assert mask[0, 2] == 0.0 and mask[0, 3] == 0.0


def test_fit_norm_stats_bullet_present_only_and_no_leak():
    # Train states: bullet present only in a couple of rows; absent sentinels must NOT skew stats.
    rng = np.random.default_rng(0)
    states = []
    for i in range(20):
        bullets = {(st.PLAYER_1, 0): (float(i), float(i) + 1, 0.0, 1.0)} if i < 5 else None
        states.append(
            _make_state(
                p1_pos=(rng.normal(), rng.normal()),
                p1_vel=(0, 0),
                p1_aim=(1, 0),
                p2_pos=(rng.normal(), rng.normal()),
                p2_vel=(0, 0),
                p2_aim=(0, 1),
                bullets=bullets,
            )
        )
    states = np.stack(states)
    stats = T.fit_norm_stats(states)
    # bullet pos slot-0 mean = mean of x over present rows (0..4) = 2.0, never near sentinel -100.
    assert abs(stats.bullet_pos_mean[0] - 2.0) < 1e-4
    assert stats.bullet_pos_mean[0] > -10.0  # sentinel excluded
    # round-trip json
    rt = T.NormStats.from_json(stats.to_json())
    assert np.allclose(rt.bullet_pos_mean, stats.bullet_pos_mean)


def test_extract_targets_zeros_absent_bullet_positions():
    s = _make_state(
        p1_pos=(1, 1),
        p1_vel=(0, 0),
        p1_aim=(1, 0),
        p2_pos=(2, 2),
        p2_vel=(0, 0),
        p2_aim=(0, 1),
        bullets={(st.PLAYER_1, 0): (3.0, 4.0, 1.0, 0.0)},
    )
    stats = T.fit_norm_stats(np.stack([s, s]))
    tgt = T.extract_targets(s[None, :], stats)
    # absent slots zeroed in the normalized bullet position target.
    bpos = tgt["bullet_position"][0]
    mask = tgt["bullet_slot_mask"][0]
    assert np.all(bpos[mask == 0] == 0.0)
    # aim is left raw (unit-ish), not normalized.
    assert np.allclose(tgt["player_aim"][0], (1, 0, 0, 1))


def test_presence_pos_weight():
    # 2 present out of 20 slots -> pos_weight = 18/2 = 9.0
    presence = np.zeros((2, 10))
    presence[0, 0] = 1.0
    presence[1, 5] = 1.0
    assert T.presence_pos_weight(presence) == 9.0
    # no present -> fallback
    assert T.presence_pos_weight(np.zeros((2, 10)), fallback=1.0) == 1.0


# --- grid extent + keypoint heatmap targets --------------------------------------------------


def _kp_states(rng, n=30):
    """``n`` states with bounded player positions + P1 slot-0 bullets present in half the rows."""
    states = []
    for i in range(n):
        bullets = {(st.PLAYER_1, 0): (float(i % 5), float(i % 3), 0.0, 1.0)} if i % 2 == 0 else None
        states.append(
            _make_state(
                p1_pos=(rng.uniform(-7, 7), rng.uniform(-3, 3)),
                p1_vel=(0, 0),
                p1_aim=(1, 0),
                p2_pos=(rng.uniform(-7, 7), rng.uniform(-3, 3)),
                p2_vel=(0, 0),
                p2_aim=(0, 1),
                bullets=bullets,
            )
        )
    return np.stack(states)


def test_fit_grid_extent_deterministic_and_excludes_absent_bullets():
    states = _kp_states(np.random.default_rng(0))
    a = T.fit_grid_extent(states)
    b = T.fit_grid_extent(states)
    assert (a.x_lo, a.x_hi, a.y_lo, a.y_hi) == (b.x_lo, b.x_hi, b.y_lo, b.y_hi)
    # the -100 sentinel of absent bullet slots must NOT drag the bounds toward -100.
    assert a.x_lo > -50.0 and a.y_lo > -50.0
    assert a.x_hi > a.x_lo and a.y_hi > a.y_lo


def test_fit_grid_extent_train_only_no_leak():
    # fitting on a subset uses ONLY the rows it is given (the build site passes TRAIN rows only).
    states = _kp_states(np.random.default_rng(1), n=40)
    full = T.fit_grid_extent(states)
    half = T.fit_grid_extent(states[:5])
    assert (full.x_lo, full.x_hi) != (half.x_lo, half.x_hi)


def test_world_to_grid_clamps_into_unit_range():
    ext = T.GridExtent(x_lo=-10.0, x_hi=10.0, y_lo=-5.0, y_hi=5.0)
    pts = np.array([[-100.0, -100.0], [0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    grid = ext.world_to_grid(pts)
    assert grid.shape == (3, 2)
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)
    assert np.allclose(grid[0], (0.0, 0.0))  # below-range clamps to 0
    assert np.allclose(grid[1], (0.5, 0.5))  # center maps to 0.5
    assert np.allclose(grid[2], (1.0, 1.0))  # above-range clamps to 1


def test_grid_extent_json_round_trip():
    ext = T.GridExtent(x_lo=-8.0, x_hi=8.5, y_lo=-3.25, y_hi=3.75)
    rt = T.GridExtent.from_json(ext.to_json())
    assert (rt.x_lo, rt.x_hi, rt.y_lo, rt.y_hi) == (ext.x_lo, ext.x_hi, ext.y_lo, ext.y_hi)


def test_keypoint_targets_shapes_order_and_presence():
    s = _make_state(
        p1_pos=(1.0, 2.0),
        p1_vel=(0, 0),
        p1_aim=(1, 0),
        p2_pos=(-3.0, 4.0),
        p2_vel=(0, 0),
        p2_aim=(0, 1),
        bullets={(st.PLAYER_1, 0): (2.0, 3.0, 1.0, 0.0), (st.PLAYER_2, 3): (5.0, 6.0, 0.0, 1.0)},
    )
    states = s[None, :]
    ext = T.fit_grid_extent(np.repeat(states, 4, axis=0))
    grid, present = T.keypoint_targets(states, ext)
    assert grid.shape == (1, 12, 2)
    assert present.shape == (1, 12)
    # order: players (P1, P2) then bullets (P1 slots 0-4, P2 slots 0-4).
    assert present[0, 0] == 1.0 and present[0, 1] == 1.0  # players always present
    assert present[0, 2] == 1.0  # P1 slot 0 present (keypoint 2)
    assert present[0, 5 + 3 + 2] == 1.0  # P2 slot 3 -> keypoint 2 + 5 + 3 = 10
    # all other bullet keypoints absent
    absent = [k for k in range(2, 12) if k not in (2, 10)]
    assert np.all(present[0, absent] == 0.0)
    # keypoint 0 (P1) grid coord matches world_to_grid of P1 world pos.
    expected_p1 = ext.world_to_grid(np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(grid[0, 0], expected_p1)
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)


def test_extract_targets_with_extent_adds_keypoint_keys():
    s = _make_state(
        p1_pos=(1, 1),
        p1_vel=(0, 0),
        p1_aim=(1, 0),
        p2_pos=(2, 2),
        p2_vel=(0, 0),
        p2_aim=(0, 1),
        bullets={(st.PLAYER_1, 0): (3.0, 4.0, 1.0, 0.0)},
    )
    states = np.stack([s, s])
    stats = T.fit_norm_stats(states)
    ext = T.fit_grid_extent(states)
    # without extent: no keypoint keys (back-compat).
    base = T.extract_targets(states, stats)
    assert "keypoint_grid" not in base and "keypoint_present" not in base
    # with extent: the two heatmap-supervision keys appear with the right shapes.
    tgt = T.extract_targets(states, stats, extent=ext)
    assert tgt["keypoint_grid"].shape == (2, 12, 2)
    assert tgt["keypoint_present"].shape == (2, 12)
