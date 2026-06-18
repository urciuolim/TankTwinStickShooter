"""Regression net for tank_twin.state (migrated from PythonScripts/tank_env.py, A6).

The opponent-perspective transforms: ``flip_state`` (image R<->B channel swap,
G/walls preserved, dtype exact) and ``split_state_for_opponent`` (the raw 52-float
26/26 half swap). These are the RL observation contract — pin them hard.
"""

import numpy as np

from tank_twin.state import flip_state, split_state_for_opponent

R, G, B = (0, 1, 2)


def _sample_image(dtype=np.uint8):
    # Small HxWx3 grid (matches the synthetic ~36x60x3 observation shape family).
    rng = np.random.default_rng(0)
    arr = (rng.random((4, 5, 3)) * 200).astype(dtype)
    return arr


# --- flip_state -----------------------------------------------------------


def test_flip_state_swaps_red_and_blue_channels():
    s = _sample_image()
    f = flip_state(s)
    assert np.array_equal(f[:, :, R], s[:, :, B])
    assert np.array_equal(f[:, :, B], s[:, :, R])


def test_flip_state_preserves_green_walls_channel():
    s = _sample_image()
    f = flip_state(s)
    assert np.array_equal(f[:, :, G], s[:, :, G])


def test_flip_state_preserves_shape_and_dtype_uint8():
    s = _sample_image(dtype=np.uint8)
    f = flip_state(s)
    assert f.shape == s.shape
    assert f.dtype == np.uint8


def test_flip_state_preserves_dtype_float32():
    # dtype must be inherited from the input EXACTLY (allocates with state.dtype).
    s = _sample_image(dtype=np.float32)
    f = flip_state(s)
    assert f.dtype == np.float32
    assert np.array_equal(f[:, :, R], s[:, :, B])


def test_flip_state_double_flip_is_identity():
    s = _sample_image()
    assert np.array_equal(flip_state(flip_state(s)), s)


def test_flip_state_does_not_mutate_input():
    s = _sample_image()
    original = s.copy()
    flip_state(s)
    assert np.array_equal(s, original)


def test_flip_state_returns_new_array():
    s = _sample_image()
    f = flip_state(s)
    assert f is not s


# --- split_state_for_opponent --------------------------------------------


def test_split_state_swaps_52_float_halves():
    state = np.arange(52, dtype=np.float32)
    opp = split_state_for_opponent(state)
    assert np.array_equal(opp[:26], state[26:])
    assert np.array_equal(opp[26:], state[:26])


def test_split_state_matches_legacy_concatenate_expression():
    # Pin the exact 2021 expression that was inlined in TankEnv.step.
    state = np.arange(52, dtype=np.float32)
    expected = np.concatenate([state[26:], state[:26]])
    assert np.array_equal(split_state_for_opponent(state), expected)


def test_split_state_round_trips_under_double_apply():
    state = np.arange(52, dtype=np.float32)
    assert np.array_equal(split_state_for_opponent(split_state_for_opponent(state)), state)


def test_split_state_preserves_dtype():
    state = np.arange(52, dtype=np.float64)
    assert split_state_for_opponent(state).dtype == np.float64
    state32 = np.arange(52, dtype=np.float32)
    assert split_state_for_opponent(state32).dtype == np.float32


def test_split_state_preserves_length():
    state = np.arange(52, dtype=np.float32)
    assert split_state_for_opponent(state).shape == (52,)
