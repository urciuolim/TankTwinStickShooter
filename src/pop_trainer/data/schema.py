"""The on-disk dataset schema: the (frame, state[, action, group]) record contract.

SINGLE SOURCE OF TRUTH for the shapes/dtypes/array-names that :mod:`pop_trainer.data.shards`
writes and reads. Every downstream consumer (the readers, the future streaming
loader in ``pretraining/``) reads these names/dtypes from here instead of hardcoding strings.

A SHARD is one ``numpy.savez_compressed`` ``.npz`` holding ``N`` time-aligned samples as
PARALLEL arrays (row ``i`` of every array describes the SAME captured step):

==============  =========  ========================  =====================================
array name      dtype      shape                     meaning
==============  =========  ========================  =====================================
``frames``      uint8      ``(N, H, W, 3)``          decoded RGB frame, top-left origin
``states``      float32    ``(N, 52)``               the paired 52-float wire state
``actions``     float32    ``(N, 2, 5)``             [p1, p2] x [mx, my, ax, ay, fire]
``map_ids``     int32      ``(N,)``                  index into the dataset ``maps`` list
``episode_ids`` int32      ``(N,)``                  monotonic per collection worker
``step_idxs``   int32      ``(N,)``                  step index within the episode
==============  =========  ========================  =====================================

* ``frames`` are stored RAW — the real rendered Unity frames; npz deflate keeps the corpus fine.
* ``actions`` is OPTIONAL on write (omitted when an inverse-render dataset only needs
  ``(frame, state)``); :data:`ARRAY_ACTIONS` is absent from such a shard and readers tolerate
  that. ``H`` / ``W`` are NOT fixed by this schema (the collector records the native frame
  size); a reader reads them from the array shape, and a whole-dataset reader checks all
  shards agree.
* The GROUP KEY for a map-aware split is :data:`ARRAY_MAP_IDS` — the same map's samples land
  in exactly one split. ``episode_ids`` / ``step_idxs`` are recorded for provenance / debugging
  and a finer (episode-level) grouping if a consumer wants it.

stdlib + numpy only; imports :mod:`pop_trainer.core.state` for ``STATE_LEN`` so the state
width is never re-hardcoded.
"""

from __future__ import annotations

from pop_trainer.core.state import STATE_LEN

__all__ = [
    "SCHEMA_VERSION",
    "STATE_LEN",
    "NUM_PLAYERS",
    "ACTION_LEN",
    "FRAME_CHANNELS",
    "ARRAY_FRAMES",
    "ARRAY_STATES",
    "ARRAY_ACTIONS",
    "ARRAY_MAP_IDS",
    "ARRAY_EPISODE_IDS",
    "ARRAY_STEP_IDXS",
    "REQUIRED_ARRAYS",
    "ALL_ARRAYS",
    "DTYPES",
]

# Bumped when the on-disk array layout changes incompatibly. Written into the dataset index.
SCHEMA_VERSION = 1

# A wire action is [move_x, move_y, aim_x, aim_y, fire]; two players per step.
NUM_PLAYERS = 2
ACTION_LEN = 5
FRAME_CHANNELS = 3

# Canonical array names inside a shard .npz. Read these, never the
# literal strings, so a rename is a one-line change here.
ARRAY_FRAMES = "frames"
ARRAY_STATES = "states"
ARRAY_ACTIONS = "actions"
ARRAY_MAP_IDS = "map_ids"
ARRAY_EPISODE_IDS = "episode_ids"
ARRAY_STEP_IDXS = "step_idxs"

# Arrays every shard MUST carry. ``actions`` is optional (a (frame, state)-only dataset omits
# it), so it is NOT in REQUIRED_ARRAYS.
REQUIRED_ARRAYS = (
    ARRAY_FRAMES,
    ARRAY_STATES,
    ARRAY_MAP_IDS,
    ARRAY_EPISODE_IDS,
    ARRAY_STEP_IDXS,
)
ALL_ARRAYS = (*REQUIRED_ARRAYS, ARRAY_ACTIONS)

# The dtype each array is stored as. ``shards.write_shard`` casts to these on write so the
# round-trip is dtype-exact regardless of the caller's input dtype.
DTYPES = {
    ARRAY_FRAMES: "uint8",
    ARRAY_STATES: "float32",
    ARRAY_ACTIONS: "float32",
    ARRAY_MAP_IDS: "int32",
    ARRAY_EPISODE_IDS: "int32",
    ARRAY_STEP_IDXS: "int32",
}
