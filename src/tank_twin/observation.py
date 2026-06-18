"""Synthetic RGB observation renderer (pure, numpy-only).

Extracted VERBATIM (logic-for-logic) from ``TankEnv.draw_state``
(``PythonScripts/tank_env.py``). The pretrained CnnPolicy was trained on the exact
bytes this produces, so the output MUST stay byte-identical — the only change is
that ``self.dims`` becomes an explicit ``dims`` argument (and ``last_state`` becomes
a plain argument), making the function pure.

Layout of the synthetic grid (channels R=self/P1, G=walls, B=opponent/P2):

* R channel  <- player 1: position (intensity POS=120), velocity tip (VEC=75),
  aim tip (AIM=30), and up to 5 bullets (BUL_POS=100 at the bullet, VEC at its tip).
* B channel  <- player 2: same scheme.
* G channel  <- walls, copied from ``last_state`` (>=255 pixels stay walls).

Intensities accumulate via ``min(existing + delta, 255)`` so overlapping markers
saturate rather than wrap. Pixel coords are ``np.clip``-ed: tanks to ``[0, dim-1]``,
bullets to ``[-100, dim-1]`` (the negative lower bound lets an off-board bullet's
``pos_x < 0`` skip drawing while its computed index stays in range for the guard).

Raw-52 state field map (per player, 26 floats): ``[pos_x, pos_y, vec_x, vec_y,
aim_x, aim_y, <5 bullets x (pos_x, pos_y, vec_x, vec_y)>]``; P1 = indices 0..25,
P2 = indices 26..51. Bullet strides ``range(6,26,4)`` / ``range(32,52,4)``.

numpy-only; no sb3/gym/torch.
"""

import numpy as np

R, G, B = (0, 1, 2)
POS, VEC, AIM = (120, 75, 30)
BUL_POS = 100


def draw_state(raw_state, last_state, dims, p):
    """Render the raw 52-float state into the synthetic RGB observation grid.

    Pure re-expression of ``TankEnv.draw_state``: ``dims`` (the ``{"minX","minY",...}``
    wall-bounds dict) is passed in instead of read off ``self``; ``last_state`` supplies
    the grid shape and the persisted wall (G) channel. Returns a NEW uint8 grid of
    ``last_state``'s shape. Output is byte-identical to the 2021 renderer.
    """
    state = np.zeros(last_state.shape, dtype=np.uint8)
    for y in range(last_state.shape[0]):
        for x in range(last_state.shape[1]):
            if last_state[y, x, G] >= 255:
                state[y, x, G] = 255

    # Parse player 1 position/velocity/aiming direction
    p1_pos_x = np.clip(int((raw_state[0] - dims["minX"]) * p), 0, state.shape[1] - 1)
    p1_pos_y = np.clip(int((raw_state[1] - dims["minY"]) * p), 0, state.shape[0] - 1)
    p1_vec_x = np.clip(p1_pos_x + int(raw_state[2] * p), 0, state.shape[1] - 1)
    p1_vec_y = np.clip(p1_pos_y + int(raw_state[3] * p), 0, state.shape[0] - 1)
    p1_aim_x = np.clip(p1_pos_x + int(raw_state[4] * p), 0, state.shape[1] - 1)
    p1_aim_y = np.clip(p1_pos_y + int(raw_state[5] * p), 0, state.shape[0] - 1)

    # Put that info into red channel
    state[p1_pos_y, p1_pos_x, R] = min(state[p1_pos_y, p1_pos_x, R] + POS, 255)
    state[p1_vec_y, p1_vec_x, R] = min(state[p1_vec_y, p1_vec_x, R] + VEC, 255)
    state[p1_aim_y, p1_aim_x, R] = min(state[p1_aim_y, p1_aim_x, R] + AIM, 255)
    # Parse each of five possible player 1 bullets, again putting info into red channel
    for i in range(6, 26, 4):
        p1_bullet_pos_x = np.clip(int((raw_state[i] - dims["minX"]) * p), -100, state.shape[1] - 1)
        p1_bullet_pos_y = np.clip(
            int((raw_state[i + 1] - dims["minY"]) * p), -100, state.shape[0] - 1
        )
        p1_bullet_vec_x = np.clip(
            p1_bullet_pos_x + int(raw_state[i + 2] * p), -100, state.shape[1] - 1
        )
        p1_bullet_vec_y = np.clip(
            p1_bullet_pos_y + int(raw_state[i + 3] * p), -100, state.shape[0] - 1
        )
        if p1_bullet_pos_x >= 0:
            state[p1_bullet_pos_y, p1_bullet_pos_x, R] = min(
                state[p1_bullet_pos_y, p1_bullet_pos_x, R] + BUL_POS, 255
            )
            state[p1_bullet_vec_y, p1_bullet_vec_x, R] = min(
                state[p1_bullet_vec_y, p1_bullet_vec_x, R] + VEC, 255
            )

    # Parse player 2 position/velocity/aiming direction
    p2_pos_x = np.clip(int((raw_state[26] - dims["minX"]) * p), 0, state.shape[1] - 1)
    p2_pos_y = np.clip(int((raw_state[27] - dims["minY"]) * p), 0, state.shape[0] - 1)
    p2_vec_x = np.clip(p2_pos_x + int(raw_state[28] * p), 0, state.shape[1] - 1)
    p2_vec_y = np.clip(p2_pos_y + int(raw_state[29] * p), 0, state.shape[0] - 1)
    p2_aim_x = np.clip(p2_pos_x + int(raw_state[30] * p), 0, state.shape[1] - 1)
    p2_aim_y = np.clip(p2_pos_y + int(raw_state[31] * p), 0, state.shape[0] - 1)
    # Put that info into blue channel
    state[p2_pos_y, p2_pos_x, B] = min(state[p2_pos_y, p2_pos_x, B] + POS, 255)
    state[p2_vec_y, p2_vec_x, B] = min(state[p2_vec_y, p2_vec_x, B] + VEC, 255)
    state[p2_aim_y, p2_aim_x, B] = min(state[p2_aim_y, p2_aim_x, B] + AIM, 255)
    # Parse each of five possible player 2 bullets, again putting info into blue channel
    for i in range(32, 52, 4):
        p2_bullet_pos_x = np.clip(int((raw_state[i] - dims["minX"]) * p), -100, state.shape[1] - 1)
        p2_bullet_pos_y = np.clip(
            int((raw_state[i + 1] - dims["minY"]) * p), -100, state.shape[0] - 1
        )
        p2_bullet_vec_x = np.clip(
            p2_bullet_pos_x + int(raw_state[i + 2] * p), -100, state.shape[1] - 1
        )
        p2_bullet_vec_y = np.clip(
            p2_bullet_pos_y + int(raw_state[i + 3] * p), -100, state.shape[0] - 1
        )
        if p2_bullet_pos_x >= 0:
            state[p2_bullet_pos_y, p2_bullet_pos_x, B] = min(
                state[p2_bullet_pos_y, p2_bullet_pos_x, B] + BUL_POS, 255
            )
            state[p2_bullet_vec_y, p2_bullet_vec_x, B] = min(
                state[p2_bullet_vec_y, p2_bullet_vec_x, B] + VEC, 255
            )

    return state
