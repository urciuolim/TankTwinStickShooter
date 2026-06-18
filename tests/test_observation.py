"""Regression net for tank_twin.observation.draw_state (PythonScripts/tank_env.py, M1 1.3).

GOLDEN test: a fixed raw-52 vector + arena dims + p=3 -> a BYTE-EXACT RGB grid. The
pretrained CnnPolicy was trained on these exact bytes, so this pins the channel
placement (R=P1, G=walls, B=P2), the POS/VEC/AIM/BUL_POS intensities, the dtype
(uint8), and the [0,255] bounds. Also pins the accumulation rule (markers add via
``min(.,255)``) and the legacy uint8-wrap quirk under heavy overlap.

numpy-only; no torch/sb3/gym/Unity.
"""

import warnings
from pathlib import Path

import numpy as np

from tank_twin.arenas import load_level
from tank_twin.observation import draw_state

R, G, B = (0, 1, 2)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARENA = REPO_ROOT / "Assets" / "Arenas" / "default.json"


def _off_board_bullets(raw):
    """Park every bullet slot far off-board so unused slots are skipped (pos_x<0).

    Mirrors real gameplay: inactive bullets carry an off-board sentinel, so the
    ``pos_x >= 0`` guard drops them instead of piling onto pixel (0,0).
    """
    for i in list(range(6, 26, 4)) + list(range(32, 52, 4)):
        raw[i] = -1000.0
        raw[i + 1] = -1000.0
    return raw


def _golden_raw():
    # P1 at world(-5,-2) with vec +x, aim +y, one live bullet at world(-3,-2).
    # P2 at world(4,1) with vec -x, aim -y, one live bullet at world(6,1).
    raw = np.zeros(52, dtype=np.float32)
    _off_board_bullets(raw)
    raw[0], raw[1] = -5.0, -2.0
    raw[2], raw[3] = 1.0, 0.0
    raw[4], raw[5] = 0.0, 1.0
    raw[6], raw[7], raw[8], raw[9] = -3.0, -2.0, 0.0, 0.0
    raw[26], raw[27] = 4.0, 1.0
    raw[28], raw[29] = -1.0, 0.0
    raw[30], raw[31] = 0.0, -1.0
    raw[32], raw[33], raw[34], raw[35] = 6.0, 1.0, 0.0, 0.0
    return raw


def _render_golden():
    arena = load_level(DEFAULT_ARENA, p=3)
    return draw_state(_golden_raw(), arena.state, arena.dims, 3), arena


# --- shape / dtype / bounds -----------------------------------------------


def test_output_shape_dtype_bounds():
    out, arena = _render_golden()
    assert out.shape == (36, 60, 3)
    assert out.dtype == np.uint8
    assert out.shape == arena.state.shape
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_returns_new_array_does_not_mutate_last_state():
    arena = load_level(DEFAULT_ARENA, p=3)
    before = arena.state.copy()
    out = draw_state(_golden_raw(), arena.state, arena.dims, 3)
    assert out is not arena.state
    assert np.array_equal(arena.state, before)


# --- GOLDEN channel placement / intensities -------------------------------


def test_golden_red_channel_p1_markers():
    # R = player 1. pos=POS(120), vec=VEC(75), aim=AIM(30); the live bullet's
    # pos(BUL_POS=100) and vec(VEC=75) share a pixel (bullet vec==0) -> 175.
    out, _ = _render_golden()
    r = out[:, :, R]
    nz = sorted((int(y), int(x), int(r[y, x])) for y, x in zip(*np.nonzero(r)))
    assert nz == [(12, 15, 120), (12, 18, 75), (12, 21, 175), (15, 15, 30)]


def test_golden_blue_channel_p2_markers():
    # B = player 2. Same scheme as red, mirrored to P2's world coords.
    out, _ = _render_golden()
    b = out[:, :, B]
    nz = sorted((int(y), int(x), int(b[y, x])) for y, x in zip(*np.nonzero(b)))
    assert nz == [(18, 42, 30), (21, 39, 75), (21, 42, 120), (21, 48, 175)]


def test_golden_green_channel_is_walls_only():
    # G is copied straight from the arena's wall grid; P1/P2 never write green.
    out, arena = _render_golden()
    assert np.array_equal(out[:, :, G], arena.state[:, :, G])
    assert int((out[:, :, G] == 255).sum()) == 1008


def test_red_and_blue_are_disjoint_in_golden_case():
    # In this vector the two players occupy different cells; pin channel separation.
    out, _ = _render_golden()
    assert not np.any((out[:, :, R] > 0) & (out[:, :, B] > 0))


# --- accumulation + the legacy uint8-wrap quirk ---------------------------


def test_overlapping_markers_accumulate_under_255():
    # pos(120) and aim(30) forced onto the SAME pixel -> 150 (clean add, no wrap).
    arena = load_level(DEFAULT_ARENA, p=3)
    raw = _off_board_bullets(np.zeros(52, dtype=np.float32))
    raw[0], raw[1] = 0.0, 0.0  # pos world(0,0) -> pixel (18,30)
    raw[2], raw[3] = 1.0, 0.0  # vec -> (18,33)
    raw[4], raw[5] = 0.0, 0.0  # aim -> (18,30), same as pos
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # this case must NOT overflow
        out = draw_state(raw, arena.state, arena.dims, 3)
    assert int(out[18, 30, R]) == 150
    assert int(out[18, 33, R]) == 75


def test_legacy_uint8_wrap_quirk_is_preserved():
    # When many markers pile on one pixel the legacy expression `min(uint8+delta,255)`
    # WRAPS the uint8 add before the min, so it does NOT cleanly saturate to 255.
    # An all-zero raw vector parks pos/vec/aim AND all five bullet slots on the same
    # pixel world(0,0)=(18,30); the repeated +100/+75 bullet adds wrap the uint8 and
    # the legacy lands on 76 (not 255). Pin it so a "fix" to clean saturation is
    # caught as a behavior change.
    arena = load_level(DEFAULT_ARENA, p=3)
    raw = np.zeros(52, dtype=np.float32)  # all bullets at world(0,0) (NOT off-board)
    raw[0], raw[1] = 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # overflow warning is the legacy behavior
        out = draw_state(raw, arena.state, arena.dims, 3)
    assert int(out[18, 30, R]) == 76


def test_off_board_bullets_are_not_drawn():
    # A bullet parked off-board (pos_x clips negative) must be skipped entirely.
    arena = load_level(DEFAULT_ARENA, p=3)
    raw = _off_board_bullets(np.zeros(52, dtype=np.float32))
    raw[0], raw[1] = 0.0, 0.0  # P1 pos at world(0,0); vec/aim left zero -> same pixel
    out = draw_state(raw, arena.state, arena.dims, 3)
    # Exactly one red pixel (no stray off-board bullet pixels). pos+vec+aim stack on
    # (18,30) because the zero vec/aim vectors point at the pos cell: 120+75+30=225.
    assert int((out[:, :, R] > 0).sum()) == 1
    assert int(out[18, 30, R]) == 225
