"""Shard read/write: the on-disk ``.npz`` dataset artifact (PURE, round-trip exact).

A shard is one ``numpy.savez_compressed`` ``.npz`` of ``N`` time-aligned samples as the
parallel arrays defined in :mod:`pop_trainer.data.schema` (``frames`` / ``states`` /
optional ``actions`` / ``map_ids`` / ``episode_ids`` / ``step_idxs``). This module is the ONLY
place that touches the ``.npz`` byte layout; everything is pure (a path + arrays in, a path or
arrays out — no socket, no game) and fully testable on a temp dir.

Write is ATOMIC: the bytes go to a sibling ``*.npz.tmp`` then ``Path.replace`` into the final
name, so a crash mid-write never leaves a half-shard a reader/resume would trust. Arrays are
cast to the schema dtypes on write, so :func:`write_shard` round-trips ``read_shard`` EXACTLY
regardless of the caller's input dtype. ``read_shard`` returns plain numpy arrays in a dict
keyed by the schema array names; a missing optional ``actions`` is simply absent from the dict.

stdlib + numpy + :mod:`pop_trainer.data.schema` only.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from pop_trainer.data import schema

__all__ = ["Shard", "write_shard", "read_shard", "read_shard_meta", "shard_length"]


class Shard:
    """An in-memory shard: the parallel sample arrays before/after the ``.npz`` round-trip.

    Holds the schema arrays (``frames``, ``states``, optional ``actions``, ``map_ids``,
    ``episode_ids``, ``step_idxs``) cast to the schema dtypes. Validates on construction that
    all arrays share the same leading length ``N`` and the expected trailing shapes, so a
    malformed batch fails LOUDLY at build time rather than silently writing a corrupt shard.
    """

    def __init__(
        self,
        frames,
        states,
        map_ids,
        episode_ids,
        step_idxs,
        actions=None,
    ):
        self.frames = np.ascontiguousarray(frames, dtype=schema.DTYPES[schema.ARRAY_FRAMES])
        self.states = np.ascontiguousarray(states, dtype=schema.DTYPES[schema.ARRAY_STATES])
        self.map_ids = np.ascontiguousarray(map_ids, dtype=schema.DTYPES[schema.ARRAY_MAP_IDS])
        self.episode_ids = np.ascontiguousarray(
            episode_ids, dtype=schema.DTYPES[schema.ARRAY_EPISODE_IDS]
        )
        self.step_idxs = np.ascontiguousarray(
            step_idxs, dtype=schema.DTYPES[schema.ARRAY_STEP_IDXS]
        )
        self.actions = (
            None
            if actions is None
            else np.ascontiguousarray(actions, dtype=schema.DTYPES[schema.ARRAY_ACTIONS])
        )
        self._validate()

    def _validate(self) -> None:
        n = self.frames.shape[0]
        if self.frames.ndim != 4 or self.frames.shape[3] != schema.FRAME_CHANNELS:
            raise ValueError(
                f"frames must be (N, H, W, {schema.FRAME_CHANNELS}); got {self.frames.shape}"
            )
        if self.states.shape != (n, schema.STATE_LEN):
            raise ValueError(f"states must be ({n}, {schema.STATE_LEN}); got {self.states.shape}")
        for name, arr in (
            (schema.ARRAY_MAP_IDS, self.map_ids),
            (schema.ARRAY_EPISODE_IDS, self.episode_ids),
            (schema.ARRAY_STEP_IDXS, self.step_idxs),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must be ({n},); got {arr.shape}")
        if self.actions is not None and self.actions.shape != (
            n,
            schema.NUM_PLAYERS,
            schema.ACTION_LEN,
        ):
            raise ValueError(
                f"actions must be ({n}, {schema.NUM_PLAYERS}, {schema.ACTION_LEN}); "
                f"got {self.actions.shape}"
            )

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_hw(self) -> tuple[int, int]:
        """The ``(H, W)`` of this shard's frames."""
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    def to_arrays(self) -> dict[str, np.ndarray]:
        """The schema array dict (``actions`` present only if this shard carries it)."""
        out = {
            schema.ARRAY_FRAMES: self.frames,
            schema.ARRAY_STATES: self.states,
            schema.ARRAY_MAP_IDS: self.map_ids,
            schema.ARRAY_EPISODE_IDS: self.episode_ids,
            schema.ARRAY_STEP_IDXS: self.step_idxs,
        }
        if self.actions is not None:
            out[schema.ARRAY_ACTIONS] = self.actions
        return out


def write_shard(path: str | Path, shard: Shard) -> Path:
    """Atomically write ``shard`` to ``path`` as a compressed ``.npz``. Returns the final path.

    Writes to a sibling ``*.tmp`` then ``Path.replace`` (atomic on the same filesystem) so a
    crash mid-write never leaves a partial shard. Casting happened in :class:`Shard`, so the
    bytes on disk are exactly the schema dtypes.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **shard.to_arrays())
    tmp.replace(path)
    return path


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    """Read a shard ``.npz`` into a dict of schema-named numpy arrays. Pure (no game).

    Validates the required arrays are present and re-builds via :class:`Shard` so the same
    shape/length invariants hold on read as on write. A missing optional ``actions`` is simply
    absent from the returned dict. Arrays are materialized (copied out of the lazy ``NpzFile``)
    so the file handle can close.
    """
    path = Path(path)
    with np.load(path) as npz:
        present = set(npz.files)
        missing = [a for a in schema.REQUIRED_ARRAYS if a not in present]
        if missing:
            raise ValueError(f"shard {path.name} missing required arrays: {missing}")
        data = {name: np.array(npz[name]) for name in npz.files}
    shard = Shard(
        frames=data[schema.ARRAY_FRAMES],
        states=data[schema.ARRAY_STATES],
        map_ids=data[schema.ARRAY_MAP_IDS],
        episode_ids=data[schema.ARRAY_EPISODE_IDS],
        step_idxs=data[schema.ARRAY_STEP_IDXS],
        actions=data.get(schema.ARRAY_ACTIONS),
    )
    return shard.to_arrays()


# numpy .npy header is version-tagged; each version has its own header parser. We read only
# the header (a few hundred bytes), never the compressed array payload.
_NPY_HEADER_READERS = {
    (1, 0): npy_format.read_array_header_1_0,
    (2, 0): npy_format.read_array_header_2_0,
}


def _npz_member_shape(zf: zipfile.ZipFile, array_name: str) -> tuple[int, ...]:
    """Return the ``shape`` of a member array by parsing ONLY its ``.npy`` header in the zip.

    Opens the ``{array_name}.npy`` entry and reads the numpy magic + header (via
    :mod:`numpy.lib.format`), which carries ``shape`` in the first few hundred bytes — so the
    (possibly large, deflated) array payload is NEVER decompressed. Pure (stdlib zipfile + numpy
    format only); the caller owns the open :class:`zipfile.ZipFile`.
    """
    with zf.open(f"{array_name}.npy") as member:
        version = npy_format.read_magic(member)
        reader = _NPY_HEADER_READERS.get(version)
        if reader is None:
            raise ValueError(f"unsupported .npy header version {version} for {array_name!r}")
        shape, _fortran, _dtype = reader(member)
    return shape


def read_shard_meta(path: str | Path) -> dict:
    """Read only a shard's CHEAP metadata (no frame load): ``n``, ``frame_hw``, ``has_actions``.

    Parses array HEADERS straight from the ``.npz`` (a zip of ``.npy`` members) via stdlib
    ``zipfile`` + :mod:`numpy.lib.format`: ``n`` from the ``map_ids`` header and ``(H, W)`` from
    the ``frames`` header. Only each member's small ``.npy`` header is read — the (deflated) frame
    payload is never decompressed — so a directory of shards can be indexed without paying the
    frame-decompression cost. Returns ``{"n", "frame_hw", "has_actions"}``.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        members = set(zf.namelist())
        if f"{schema.ARRAY_MAP_IDS}.npy" not in members:
            raise ValueError(f"shard {path.name} missing {schema.ARRAY_MAP_IDS}")
        n = int(_npz_member_shape(zf, schema.ARRAY_MAP_IDS)[0])
        fshape = _npz_member_shape(zf, schema.ARRAY_FRAMES)
        has_actions = f"{schema.ARRAY_ACTIONS}.npy" in members
    return {
        "n": n,
        "frame_hw": (int(fshape[1]), int(fshape[2])),
        "has_actions": has_actions,
    }


def shard_length(path: str | Path) -> int:
    """Number of samples in a shard (cheap: reads only the ``map_ids`` array)."""
    path = Path(path)
    with np.load(path) as npz:
        return int(npz[schema.ARRAY_MAP_IDS].shape[0])
