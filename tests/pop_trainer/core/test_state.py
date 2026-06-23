"""Contract tests for ``pop_trainer.core.state`` (the 52-float schema)."""

import numpy as np
import pytest

from pop_trainer.core import state as S


def _known_state():
    """A 52-float state with distinctive, position-coded values for accessor checks.

    Index ``i`` holds value ``float(i)`` EXCEPT bullet slots, which are set explicitly so the
    presence rule (sentinel pos_x) can be exercised.
    """
    raw = [float(i) for i in range(S.STATE_LEN)]
    # P1: make bullet slot 1 absent (sentinel pos_x), keep slot 0 present.
    raw[S.BULLET_START[S.PLAYER_1]] = 1.5  # bullet0 pos_x (present)
    raw[S.BULLET_START[S.PLAYER_1] + S.BULLET_STRIDE] = S.ABSENT_BULLET_SENTINEL  # bullet1 absent
    return raw


def test_state_len_is_52():
    assert S.STATE_LEN == 52
    assert S.PLAYER_STRIDE == 26
    assert S.NUM_BULLETS == 5
    assert S.PLAYER_1 == 0
    assert S.PLAYER_2 == 1


def test_position_velocity_aim_accessors():
    raw = [float(i) for i in range(S.STATE_LEN)]
    assert S.position(raw, S.PLAYER_1) == (0.0, 1.0)
    assert S.velocity(raw, S.PLAYER_1) == (2.0, 3.0)
    assert S.aim(raw, S.PLAYER_1) == (4.0, 5.0)
    # P2 block starts at 26.
    assert S.position(raw, S.PLAYER_2) == (26.0, 27.0)
    assert S.velocity(raw, S.PLAYER_2) == (28.0, 29.0)
    assert S.aim(raw, S.PLAYER_2) == (30.0, 31.0)


def test_bullet_field_indices_strides():
    assert list(S.bullet_field_indices(S.PLAYER_1)) == [6, 10, 14, 18, 22]
    assert list(S.bullet_field_indices(S.PLAYER_2)) == [32, 36, 40, 44, 48]


def test_accessors_accept_numpy_array():
    raw = np.arange(S.STATE_LEN, dtype=np.float32)
    assert S.position(raw, S.PLAYER_2)[0] == 26.0
    assert list(S.bullet_field_indices(S.PLAYER_2))[0] == 32


def test_sentinel_and_presence_helper():
    assert S.ABSENT_BULLET_SENTINEL == -100.0
    assert S.bullet_present(0.0) is True
    assert S.bullet_present(5.0) is True
    assert S.bullet_present(S.ABSENT_BULLET_SENTINEL) is False
    assert S.bullet_present(-0.1) is False


def test_iter_bullets_skips_absent_by_default():
    raw = _known_state()
    bullets = list(S.iter_bullets(raw, S.PLAYER_1))
    # Slot 1 was made absent; the other four slots have value-coded pos_x >= 0.
    assert len(bullets) == S.NUM_BULLETS - 1
    # First present bullet is slot 0 with the explicit pos_x we set.
    assert bullets[0][0] == 1.5


def test_iter_bullets_include_absent_returns_all():
    raw = _known_state()
    bullets = list(S.iter_bullets(raw, S.PLAYER_1, include_absent=True))
    assert len(bullets) == S.NUM_BULLETS


def test_iter_bullets_record_shape():
    raw = [float(i) for i in range(S.STATE_LEN)]
    first = next(iter(S.iter_bullets(raw, S.PLAYER_1)))
    # The first P1 bullet record is state[6:10].
    assert first == (6.0, 7.0, 8.0, 9.0)


def test_validate_accepts_52_rejects_wrong_length():
    S.validate([0.0] * 52)  # no raise
    with pytest.raises(ValueError):
        S.validate([0.0] * 51)
    with pytest.raises(ValueError):
        S.validate([0.0] * 53)


def test_player_arg_validated():
    raw = [0.0] * 52
    with pytest.raises(ValueError):
        S.position(raw, 2)
    with pytest.raises(ValueError):
        S.bullet_field_indices(5)


def test_flip_frame_perspective_is_involution_and_swaps_rb():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    flipped = S.flip_frame_perspective(img)
    # R<->B swapped, G unchanged.
    assert np.array_equal(flipped[:, :, 0], img[:, :, 2])
    assert np.array_equal(flipped[:, :, 2], img[:, :, 0])
    assert np.array_equal(flipped[:, :, 1], img[:, :, 1])
    # Involution.
    assert np.array_equal(S.flip_frame_perspective(flipped), img)
    # dtype preserved.
    assert flipped.dtype == img.dtype


def test_split_state_for_opponent_swaps_halves_and_is_involution():
    raw = np.arange(S.STATE_LEN, dtype=np.float64)
    swapped = S.split_state_for_opponent(raw)
    assert np.array_equal(swapped[:26], raw[26:])
    assert np.array_equal(swapped[26:], raw[:26])
    # Involution.
    assert np.array_equal(S.split_state_for_opponent(swapped), raw)
    # dtype inherited.
    assert swapped.dtype == raw.dtype
