"""Arena layout loading + observation-grid geometry (pure, numpy-only).

Extracted from ``TankEnv.load_level`` (``PythonScripts/tank_env.py``). Reads an
arena JSON (the same ``Assets/Arenas/*.json`` Unity authors) and builds the
wall-initialized synthetic-observation grid the CnnPolicy trains on: a
``(height, width, 3)`` uint8 array with the walls baked into the green channel.

Two upgrades over the legacy:

* STRICT ``json.load`` only — no tolerant/Newtonsoft-style fallback. A malformed
  arena (trailing comma, leading-dot float) RAISES instead of silently degrading.
  Unity's Newtonsoft tolerates such JSON; Python's ``json`` must not, and the env
  must not paper over it.
* No sb3 ``spaces.Box``: returns the grid SHAPE (``obs_shape``) plus the dims and
  the wall grid, so this module imports without gymnasium/sb3. The env constructs
  its ``observation_space`` from ``obs_shape``.

Grid geometry is byte-identical to 2021::

    width  = (maxX - minX + 1) * p
    height = (maxY - minY + 1) * p

with ``p`` pixels per game-grid square, walls read from ``json["Walls"]`` (the
``"dims"`` block plus one ``"<x>": [y, ...]`` list per column).

numpy-only; no sb3/gym/torch.
"""

import json
from dataclasses import dataclass

import numpy as np

# Channel indices: R=self (P1), G=walls, B=opponent (P2). Matches draw_state / state.py.
R, G, B = (0, 1, 2)


@dataclass(frozen=True)
class Arena:
    """Result of loading an arena: the obs grid shape, the wall dims, and the grid.

    * ``obs_shape``: ``(height, width, 3)`` — what the env's observation_space uses.
    * ``dims``: the raw ``{"minX","maxX","minY","maxY"}`` wall-bounds dict; ``draw_state``
      needs ``minX``/``minY`` to map world coords into pixels.
    * ``state``: the ``(height, width, 3)`` uint8 grid with walls baked into G.
    * ``p``: pixels-per-square the grid was built at (kept for the caller's records).
    """

    obs_shape: tuple
    dims: dict
    state: np.ndarray
    p: int


def load_level(level_path, p=3):
    """Load an arena JSON and build the wall-initialized observation grid.

    STRICT parse (``json.load``; raises ``json.JSONDecodeError`` on malformed JSON).
    Geometry and the wall-baking loop are byte-identical to ``TankEnv.load_level``.

    Returns an :class:`Arena`. The legacy returned ``(spaces.Box(...), state)`` and
    stashed ``self.dims``; here the shape and dims are surfaced on the dataclass so
    the env can rebuild the Box without this module depending on sb3.
    """
    with open(level_path) as level_file:
        level_json = json.load(level_file)
    dims = level_json["Walls"]["dims"]
    # p^2 = number of pixels to represent one grid square in game
    width = (dims["maxX"] - dims["minX"] + 1) * p
    height = (dims["maxY"] - dims["minY"] + 1) * p

    state = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(dims["minX"], dims["maxX"] + 1):
        x_p = (x - dims["minX"]) * p
        for y in level_json["Walls"][str(x)]:
            y_p = (y - dims["minY"]) * p
            state[y_p : y_p + p, x_p : x_p + p, G] = 255

    return Arena(obs_shape=state.shape, dims=dict(dims), state=state, p=p)
