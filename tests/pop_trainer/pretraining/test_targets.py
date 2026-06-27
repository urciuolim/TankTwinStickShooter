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
