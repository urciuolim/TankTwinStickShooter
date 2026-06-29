"""Reproducible, provenanced, split-respecting on-disk dataset SHUFFLE (out-of-core).

Turns a SOURCE dataset of episode-CONTIGUOUS ``.npz`` shards (the
:mod:`pop_trainer.data.schema` layout) into a NEW dataset whose rows are randomly
permuted, so a trivial shard-grouped streaming sampler sees a uniform mix of maps in
every batch. The shuffle is row-preserving: it is NOT a new collection, only a
re-ordering + re-chunking of the EXACT same frames/states/actions/ids.

Two split-modes:

* ``global`` (default) — permute ALL rows across the whole dataset into flat output
  shards ``shard_NNNNN.npz``; each output shard is a random draw from every map. The
  train/val/test split is applied later at LOAD time, per-sample by
  :func:`pop_trainer.data.readers.split_groups` (layout-independent), so this stage
  does not split.
* ``within-split`` — first apply :func:`readers.split_groups` over the source
  ``map_ids`` (REUSED, not re-implemented), then permute WITHIN each split and write
  separate ``train/`` / ``val/`` / ``test/`` shard subdirectories. Whole maps land in
  exactly one split, so no map leaks across splits.

THE MEMORY-BOUND INVARIANT (load-bearing — the full dataset is ~83 GB and will NOT fit
in RAM): the shuffle is a TWO-PASS BUCKETED out-of-core algorithm whose peak resident
frame memory is bounded to ~TWO output shards, regardless of dataset size.

* Pass 1 (scatter): stream source shards ONE AT A TIME; for each row draw a target
  output-bucket index from a SEEDED ``numpy.random.default_rng`` and append the row's
  bytes to that bucket's ON-DISK spill file (fixed-stride raw records — one combined
  spill file per bucket, ~one file handle per bucket). Only one source shard plus a
  transient per-bucket pack buffer (itself bounded by one source shard) is ever
  resident.
* Pass 2 (permute): load each bucket's spill (sized ~one output shard, fits in RAM),
  permute its rows with a per-bucket seeded rng, write the final ``.npz`` via
  :func:`pop_trainer.data.shards.write_shard`, then DELETE the spill — so peak DISK is
  bounded to ~the dataset plus one output shard, never doubled.

ROW ALIGNMENT: every per-frame array (``frames``/``states``/``map_ids``/
``episode_ids``/``step_idxs``/``actions``) is scattered and permuted by the SAME index
decision, so each output row keeps its own frame+state+map_id+action together.

DETERMINISM: identical ``(source, seed, split-mode, output-shard-size)`` yields
identical output. Only ``numpy.random.default_rng`` is used (no global RNG state); the
scatter draws over source shards in SORTED filename order and each bucket's permute
stream is derived from the seed in a fixed (group, bucket-index) order.

The PURE, tested core is :func:`shuffle_dataset` (the algorithm). The argparse / clock
read / stdout summary in :func:`main` is untested CLI glue, mirroring the style of
:mod:`pop_trainer.data.describe`. stdlib + numpy only; imports ``core`` (via the
schema) and sibling ``data`` modules, nothing from ``models`` / ``pretraining`` /
``rl`` / ``tank_twin``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pop_trainer.data import readers, schema, shards
from pop_trainer.data.manifest import MANIFEST_NAME, add_description, build_manifest

__all__ = ["DEFAULT_OUTPUT_SHARD_SIZE", "ShuffleResult", "shuffle_dataset", "main"]

DEFAULT_OUTPUT_SHARD_SIZE = 1024
MAPS_SIDECAR_NAME = "maps.json"
SHARD_PATTERN = "shard_*.npz"
SPLIT_NAMES = ("train", "val", "test")

# Datasets the shuffle must NEVER overwrite (the canonical original + the legacy
# scratch reshuffle), guarded by their conventional ``datasets/<name>`` location.
_FORBIDDEN_OUT_NAMES = ("decode-v2_original", "decode-v2_reshuffled")

# Group codes used to derive a fixed, seed-stable per-bucket permute stream. ``global``
# is its own group; within-split numbers the three splits.
_GROUP_GLOBAL = 0
_GROUP_CODE = {"global": _GROUP_GLOBAL, "train": 1, "val": 2, "test": 3}

# A resident-frame-bytes probe: ``probe(phase, resident_frame_bytes)``. A test seam that
# lets a caller assert the memory-bound invariant without parsing process RSS.
ResidentProbe = Callable[[str, int], None]


@dataclass(frozen=True)
class _RowLayout:
    """Byte layout of one packed spill record (fixed stride; arrays laid out in order).

    A spill record concatenates a single row's raw array bytes: the frame
    (``H*W*3`` uint8), the 52-float state, the three int32 ids, and (when the source
    carries it) the ``(2, 5)`` action — so a bucket spill is just ``n`` fixed-size
    records and reads back with one ``np.fromfile`` + reshape.
    """

    hw: tuple[int, int]
    has_actions: bool
    f0: int
    f1: int
    s0: int
    s1: int
    m0: int
    m1: int
    e0: int
    e1: int
    p0: int
    p1: int
    a0: int
    a1: int
    stride: int

    @property
    def frame_bytes(self) -> int:
        return self.f1 - self.f0


def _row_layout(hw: tuple[int, int], has_actions: bool) -> _RowLayout:
    """Compute the fixed spill-record byte offsets for ``(H, W)`` frames + optional actions."""
    h, w = hw
    frame_bytes = h * w * schema.FRAME_CHANNELS
    state_bytes = schema.STATE_LEN * 4
    id_bytes = 4
    action_bytes = schema.NUM_PLAYERS * schema.ACTION_LEN * 4 if has_actions else 0
    f0, f1 = 0, frame_bytes
    s0, s1 = f1, f1 + state_bytes
    m0, m1 = s1, s1 + id_bytes
    e0, e1 = m1, m1 + id_bytes
    p0, p1 = e1, e1 + id_bytes
    a0, a1 = p1, p1 + action_bytes
    return _RowLayout(hw, has_actions, f0, f1, s0, s1, m0, m1, e0, e1, p0, p1, a0, a1, a1)


@dataclass(frozen=True)
class ShuffleResult:
    """Summary of a completed shuffle (what was written + where)."""

    out: Path
    split_mode: str
    seed: int
    output_shard_size: int
    total_samples: int
    total_shards: int
    per_map_samples: dict[str, int]
    manifest_path: Path
    per_split: dict[str, dict] = field(default_factory=dict)


def _read_map_ids(path: Path) -> np.ndarray:
    """Read ONLY a shard's ``map_ids`` (int64), decompressing no frame payload."""
    with np.load(path) as npz:
        return np.asarray(npz[schema.ARRAY_MAP_IDS], dtype=np.int64)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _resolve_forbidden(out: Path) -> None:
    """Raise ValueError if ``out`` is a protected dataset dir (always refused)."""
    rp = out.resolve()
    if rp.name in _FORBIDDEN_OUT_NAMES and rp.parent.name == "datasets":
        raise ValueError(
            f"refusing to write to protected dataset directory {rp} "
            f"(decode-v2_original / decode-v2_reshuffled are never overwritable)"
        )


def _check_out_dir(out: Path, src: Path, *, force: bool) -> None:
    """Validate the destination: not the source, not protected, not non-empty unless force."""
    _resolve_forbidden(out)
    if out.resolve() == src.resolve():
        raise ValueError(f"output dir must differ from source dir ({src})")
    if out.exists() and any(out.iterdir()) and not force:
        raise ValueError(f"output dir {out} is not empty; pass force=True to overwrite into it")


def _scan_source(src: Path) -> tuple[list[Path], _RowLayout, dict[int, int], np.ndarray]:
    """Cheap first pass: discover shards + per-map counts + (H, W)/actions, no frame load.

    Returns ``(sorted_shard_paths, row_layout, per_map_counts, all_map_ids)``. Reads each
    shard's small header (``read_shard_meta``) and its ``map_ids`` only, so no frame bytes
    are decompressed. Validates every shard agrees on ``(H, W)`` and ``has_actions``.
    """
    paths = sorted(p for p in src.rglob(SHARD_PATTERN) if p.is_file())
    if not paths:
        raise ValueError(f"no shards matching {SHARD_PATTERN!r} under {src}")

    layout: _RowLayout | None = None
    per_map: dict[int, int] = {}
    all_ids: list[np.ndarray] = []
    for p in paths:
        meta = shards.read_shard_meta(p)
        hw = meta["frame_hw"]
        has_actions = meta["has_actions"]
        if layout is None:
            layout = _row_layout(hw, has_actions)
        elif (hw, has_actions) != (layout.hw, layout.has_actions):
            raise ValueError(
                f"shard {p.name} layout {(hw, has_actions)} != "
                f"dataset {(layout.hw, layout.has_actions)}"
            )
        mids = _read_map_ids(p)
        all_ids.append(mids)
        for mid, cnt in zip(*np.unique(mids, return_counts=True), strict=True):
            per_map[int(mid)] = per_map.get(int(mid), 0) + int(cnt)

    assert layout is not None  # paths non-empty guarantees a layout was built
    return paths, layout, per_map, np.concatenate(all_ids) if all_ids else np.empty(0, np.int64)


def _pack_rows(layout: _RowLayout, arrays: dict[str, np.ndarray], rows: np.ndarray) -> bytes:
    """Pack the selected ``rows`` of a shard into contiguous fixed-stride spill records."""
    k = int(rows.shape[0])
    buf = np.empty((k, layout.stride), dtype=np.uint8)
    buf[:, layout.f0 : layout.f1] = arrays[schema.ARRAY_FRAMES][rows].reshape(k, -1)
    buf[:, layout.s0 : layout.s1] = arrays[schema.ARRAY_STATES][rows].reshape(k, -1).view(np.uint8)
    buf[:, layout.m0 : layout.m1] = arrays[schema.ARRAY_MAP_IDS][rows].reshape(k, 1).view(np.uint8)
    buf[:, layout.e0 : layout.e1] = (
        arrays[schema.ARRAY_EPISODE_IDS][rows].reshape(k, 1).view(np.uint8)
    )
    buf[:, layout.p0 : layout.p1] = (
        arrays[schema.ARRAY_STEP_IDXS][rows].reshape(k, 1).view(np.uint8)
    )
    if layout.has_actions:
        buf[:, layout.a0 : layout.a1] = (
            arrays[schema.ARRAY_ACTIONS][rows].reshape(k, -1).view(np.uint8)
        )
    return buf.tobytes()


def _unpack_bucket(layout: _RowLayout, path: Path) -> shards.Shard:
    """Read a bucket's spill back into a :class:`Shard` (all arrays row-aligned)."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // layout.stride
    rec = raw.reshape(n, layout.stride)
    h, w = layout.hw
    frames = rec[:, layout.f0 : layout.f1].reshape(n, h, w, schema.FRAME_CHANNELS)
    states = np.ascontiguousarray(rec[:, layout.s0 : layout.s1]).view(np.float32).reshape(n, -1)
    map_ids = np.ascontiguousarray(rec[:, layout.m0 : layout.m1]).view(np.int32).reshape(n)
    episode_ids = np.ascontiguousarray(rec[:, layout.e0 : layout.e1]).view(np.int32).reshape(n)
    step_idxs = np.ascontiguousarray(rec[:, layout.p0 : layout.p1]).view(np.int32).reshape(n)
    actions = None
    if layout.has_actions:
        actions = (
            np.ascontiguousarray(rec[:, layout.a0 : layout.a1])
            .view(np.float32)
            .reshape(n, schema.NUM_PLAYERS, schema.ACTION_LEN)
        )
    return shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=actions)


class _BucketSpills:
    """Lazily-opened per-bucket spill files under ``spill_dir`` (one handle per bucket)."""

    def __init__(self, spill_dir: Path):
        self._dir = spill_dir
        self._files: dict[tuple[int, int], object] = {}
        self._paths: dict[tuple[int, int], Path] = {}
        spill_dir.mkdir(parents=True, exist_ok=True)

    def path(self, group_code: int, bucket: int) -> Path:
        return self._dir / f"spill_g{group_code}_b{bucket:05d}.bin"

    def write(self, group_code: int, bucket: int, payload: bytes) -> None:
        key = (group_code, bucket)
        fh = self._files.get(key)
        if fh is None:
            p = self.path(group_code, bucket)
            self._paths[key] = p
            # One long-lived handle per bucket spans the whole scatter pass (closed in
            # close()); a per-write context manager would defeat the streaming append.
            fh = p.open("wb")  # noqa: SIM115
            self._files[key] = fh
        fh.write(payload)

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def _split_assignment(
    all_map_ids: np.ndarray, *, val_frac: float, test_frac: float, seed: int
) -> tuple[dict[int, int], dict[str, list[int]]]:
    """Map each source map id to a split code (train=0/val=1/test=2) via ``split_groups``.

    Returns ``(map_to_code, split_to_maps)`` where ``split_to_maps`` lists the map ids in
    each split. REUSES :func:`readers.split_groups` so the split is identical to the one a
    loader computes for the same seed/fractions.
    """
    split = readers.split_groups(all_map_ids, val_frac=val_frac, test_frac=test_frac, seed=seed)
    groups = split.group_dict()
    map_to_code: dict[int, int] = {}
    split_to_maps: dict[str, list[int]] = {}
    for code, name in enumerate(SPLIT_NAMES):
        ids = [int(m) for m in groups[name].tolist()]
        split_to_maps[name] = ids
        for mid in ids:
            map_to_code[mid] = code
    return map_to_code, split_to_maps


def _scatter(
    paths: list[Path],
    layout: _RowLayout,
    spills: _BucketSpills,
    rng: np.random.Generator,
    *,
    route: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    probe: ResidentProbe | None,
) -> None:
    """Pass 1: stream each source shard once and append its rows to target bucket spills.

    ``route(map_ids)`` returns ``(group_codes, bucket_idxs)`` for the shard's rows. Holds
    only the current source shard plus a transient per-bucket pack buffer; the resident
    frame bytes reported to ``probe`` peak at ~one source shard (frames) plus the largest
    single-bucket pack buffer.
    """
    for p in paths:
        arrays = shards.read_shard(p)
        frame_bytes = int(arrays[schema.ARRAY_FRAMES].nbytes)
        if probe is not None:
            probe("scatter_loaded", frame_bytes)
        map_ids = arrays[schema.ARRAY_MAP_IDS].astype(np.int64)
        group_codes, bucket_idxs = route(map_ids)
        keys = np.stack([group_codes, bucket_idxs], axis=1)
        unique_keys = np.unique(keys, axis=0)
        for gc, b in unique_keys:
            rows = np.nonzero((group_codes == gc) & (bucket_idxs == b))[0]
            if probe is not None:
                probe("scatter_pack", frame_bytes + int(rows.shape[0]) * layout.frame_bytes)
            spills.write(int(gc), int(b), _pack_rows(layout, arrays, rows))
        del arrays
        if probe is not None:
            probe("scatter_released", 0)


def _write_group_shards(
    layout: _RowLayout,
    spills: _BucketSpills,
    out_dir: Path,
    group_code: int,
    n_buckets: int,
    seed: int,
    *,
    probe: ResidentProbe | None,
) -> tuple[int, dict[int, int]]:
    """Pass 2 for one output group: permute each bucket, write its ``.npz``, delete its spill.

    Buckets are visited in ascending index order; each gets a permute stream derived from
    ``(seed, group_code, bucket)`` so the output is deterministic. Returns
    ``(shards_written, per_map_counts)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_written = 0
    per_map: dict[int, int] = {}
    for b in range(n_buckets):
        spill_path = spills.path(group_code, b)
        if not spill_path.exists():
            continue
        shard = _unpack_bucket(layout, spill_path)
        if probe is not None:
            probe("bucket_loaded", int(shard.frames.nbytes))
        n = len(shard)
        perm = np.random.default_rng([seed, group_code, b]).permutation(n)
        out = shards.Shard(
            frames=shard.frames[perm],
            states=shard.states[perm],
            map_ids=shard.map_ids[perm],
            episode_ids=shard.episode_ids[perm],
            step_idxs=shard.step_idxs[perm],
            actions=None if shard.actions is None else shard.actions[perm],
        )
        shards.write_shard(out_dir / f"shard_{shards_written:05d}.npz", out)
        for mid, cnt in zip(*np.unique(out.map_ids, return_counts=True), strict=True):
            per_map[int(mid)] = per_map.get(int(mid), 0) + int(cnt)
        shards_written += 1
        spill_path.unlink()
        del shard, out
        if probe is not None:
            probe("bucket_released", 0)
    return shards_written, per_map


def _map_names(per_map_counts: dict[int, int], maps: list[str]) -> dict[str, int]:
    """Re-key per-map-id counts to map NAMES via the dataset ``maps`` list."""
    out: dict[str, int] = {}
    for mid, cnt in sorted(per_map_counts.items()):
        name = maps[mid] if 0 <= mid < len(maps) else str(mid)
        out[name] = out.get(name, 0) + cnt
    return out


def _data_card(
    *,
    src: Path,
    seed: int,
    split_mode: str,
    output_shard_size: int,
    total_samples: int,
    total_shards: int,
    per_split: dict[str, dict],
) -> str:
    """The strict-JSON-safe DATA CARD prose recorded in the output manifest descriptions."""
    lines = [
        f"DATA CARD - reproducible shuffle of {src.as_posix()}. "
        f"This dataset is a row-preserving, out-of-core SHUFFLE of the source collection "
        f"(same frames/states/actions/ids, re-ordered and re-chunked - NOT a new collection). "
        f"SOURCE: {src.as_posix()}. SEED: {seed}. "
        f"ALGORITHM: two-pass bucketed out-of-core shuffle "
        f"(pass 1 scatters each source row to a seeded target bucket's on-disk spill, "
        f"pass 2 permutes each bucket in RAM and writes the final shard, deleting its spill). "
        f"MEMORY-BOUND INVARIANT: peak resident frame memory stays bounded to ~two output "
        f"shards regardless of dataset size; only one source shard is ever loaded at a time. "
        f"SPLIT-MODE: {split_mode}. OUTPUT SHARD SIZE: {output_shard_size} samples. "
        f"TOTAL: {total_samples} samples across {total_shards} shards. "
        f"ROW-ALIGNMENT GUARANTEE: every per-frame array (frames/states/map_ids/episode_ids/"
        f"step_idxs/actions) is scattered and permuted by the SAME index decision, so each "
        f"output row keeps its own frame+state+map_id+action together. "
        f"DETERMINISM: identical (source, seed, split-mode, output-shard-size) yields identical "
        f"output; only numpy.random.default_rng is used (no global RNG state)."
    ]
    if split_mode == "global":
        lines.append(
            " The train/val/test split is layout-independent and applied at LOAD time per-sample "
            "by readers.split_groups (whole maps land in exactly one split regardless of which "
            "physical shard holds a frame)."
        )
    else:
        leak = "; ".join(
            f"{name}={sorted(per_split.get(name, {}).get('maps', []))}" for name in SPLIT_NAMES
        )
        lines.append(
            " NO-LEAK GUARANTEE: the split is computed once over the source map_ids via "
            "readers.split_groups and rows are routed into separate train/val/test shard "
            f"subdirectories by map; the per-split map sets are DISJOINT (maps -> {leak}), so no "
            "map leaks across splits."
        )
    lines.append(
        " NOTE: GitHub issue #13 will later add a content-hash manifest_id so the shuffle is "
        "visible in the manifest fingerprint; this card does not build that."
    )
    return "".join(lines)


def _build_output_manifest(
    *,
    src: Path,
    out: Path,
    seed: int,
    split_mode: str,
    output_shard_size: int,
    total_samples: int,
    total_shards: int,
    per_map_named: dict[str, int],
    per_split: dict[str, dict],
    command: list[str],
    author: str,
    created_utc: str,
) -> dict:
    """Assemble the output manifest: carry source provenance/machine/collection, recompute totals.

    Loads the SOURCE manifest, reuses its ``provenance`` + ``machine`` and the collection base
    fields (maps/pairings/workers/episodes/max_steps), RECOMPUTES the totals for the shuffled
    output, sets ``collection.command`` to ``command``, and appends the data card description.
    """
    with open(src / MANIFEST_NAME, encoding="utf-8") as fh:
        source_manifest = json.load(fh)

    src_collection = source_manifest["collection"]
    collection = {
        "seed": seed,
        "command": command,
        "workers": src_collection["workers"],
        "episodes": src_collection["episodes"],
        "max_steps": src_collection["max_steps"],
        "maps": list(src_collection["maps"]),
        "pairings": list(src_collection["pairings"]),
        "total_shards": total_shards,
        "total_samples": total_samples,
        "per_map_samples": per_map_named,
    }
    manifest = build_manifest(
        dataset=out.name,
        collection=collection,
        provenance=source_manifest["provenance"],
        machine=source_manifest["machine"],
        created_utc=created_utc,
        descriptions=source_manifest.get("descriptions"),
    )
    card = _data_card(
        src=src,
        seed=seed,
        split_mode=split_mode,
        output_shard_size=output_shard_size,
        total_samples=total_samples,
        total_shards=total_shards,
        per_split=per_split,
    )
    return add_description(manifest, author=author, text=card, added_utc=created_utc)


def _write_manifest(out: Path, manifest: dict) -> Path:
    """Atomically write ``manifest`` to ``<out>/manifest.json`` (tmp -> replace, strict JSON)."""
    path = out / MANIFEST_NAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)
    tmp.replace(path)
    return path


def shuffle_dataset(
    src: str | Path,
    out: str | Path,
    *,
    seed: int = 0,
    split_mode: str = "global",
    output_shard_size: int = DEFAULT_OUTPUT_SHARD_SIZE,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    force: bool = False,
    author: str = "claude",
    command: list[str] | None = None,
    created_utc: str | None = None,
    spill_dir: str | Path | None = None,
    _resident_probe: ResidentProbe | None = None,
) -> ShuffleResult:
    """Shuffle ``src`` into a new dataset at ``out`` (two-pass bucketed, out-of-core).

    See the module docstring for the algorithm and the memory-bound invariant. The shuffle
    is row-preserving and deterministic in ``seed``. ``split_mode`` is ``"global"`` (flat
    shards, split deferred to load time) or ``"within-split"`` (route maps into separate
    ``train/`` / ``val/`` / ``test/`` shard subdirs via :func:`readers.split_groups`).
    ``output_shard_size`` is the target rows per output shard (= bucket count is
    ``ceil(samples / output_shard_size)``). Refuses to write into a non-empty ``out``
    unless ``force``, and ALWAYS refuses the protected ``decode-v2_original`` /
    ``decode-v2_reshuffled`` dirs. Never mutates the source. ``_resident_probe`` is a test
    seam reporting resident frame bytes at each step.
    """
    if split_mode not in ("global", "within-split"):
        raise ValueError(f"split_mode must be 'global' or 'within-split'; got {split_mode!r}")
    if output_shard_size < 1:
        raise ValueError(f"output_shard_size must be >= 1; got {output_shard_size}")

    src = Path(src)
    out = Path(out)
    if not (src / MANIFEST_NAME).exists():
        raise ValueError(f"source dataset {src} has no {MANIFEST_NAME}")
    _check_out_dir(out, src, force=force)

    paths, layout, per_map_counts, all_map_ids = _scan_source(src)
    total_samples = int(sum(per_map_counts.values()))
    with open(src / MANIFEST_NAME, encoding="utf-8") as fh:
        maps = list(json.load(fh)["collection"]["maps"])

    out.mkdir(parents=True, exist_ok=True)
    spill_root = Path(spill_dir) if spill_dir is not None else out / ".shuffle_spill"
    spills = _BucketSpills(spill_root)
    scatter_rng = np.random.default_rng(seed)

    per_split: dict[str, dict] = {}
    if split_mode == "global":
        n_buckets = _ceil_div(total_samples, output_shard_size)

        def route(mids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gc = np.full(mids.shape[0], _GROUP_GLOBAL, dtype=np.int64)
            return gc, scatter_rng.integers(0, n_buckets, size=mids.shape[0])

        groups: list[tuple[str, int, int, Path]] = [("global", _GROUP_GLOBAL, n_buckets, out)]
    else:
        map_to_code, split_to_maps = _split_assignment(
            all_map_ids, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        max_mid = int(max(map_to_code)) if map_to_code else 0
        code_lut = np.full(max_mid + 1, -1, dtype=np.int64)
        for mid, code in map_to_code.items():
            code_lut[mid] = code
        split_samples = {name: 0 for name in SPLIT_NAMES}
        for mid, cnt in per_map_counts.items():
            split_samples[SPLIT_NAMES[map_to_code[mid]]] += cnt
        bucket_counts = {
            name: _ceil_div(split_samples[name], output_shard_size) for name in SPLIT_NAMES
        }

        def route(mids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            codes = code_lut[mids]
            gc = np.empty(mids.shape[0], dtype=np.int64)
            bk = np.empty(mids.shape[0], dtype=np.int64)
            for code, name in enumerate(SPLIT_NAMES):
                sel = np.nonzero(codes == code)[0]
                if sel.size == 0:
                    continue
                gc[sel] = _GROUP_CODE[name]
                bk[sel] = scatter_rng.integers(0, max(bucket_counts[name], 1), size=sel.size)
            return gc, bk

        groups = [
            (name, _GROUP_CODE[name], bucket_counts[name], out / name) for name in SPLIT_NAMES
        ]
        for name in SPLIT_NAMES:
            per_split[name] = {"maps": [maps[m] for m in sorted(split_to_maps[name])]}

    _scatter(paths, layout, spills, scatter_rng, route=route, probe=_resident_probe)
    spills.close()

    total_shards = 0
    output_per_map: dict[int, int] = {}
    for name, group_code, n_buckets, group_dir in groups:
        written, group_per_map = _write_group_shards(
            layout, spills, group_dir, group_code, n_buckets, seed, probe=_resident_probe
        )
        total_shards += written
        for mid, cnt in group_per_map.items():
            output_per_map[mid] = output_per_map.get(mid, 0) + cnt
        if name in per_split:
            per_split[name]["shards"] = written
            per_split[name]["samples"] = int(sum(group_per_map.values()))
        if group_dir != out:
            shutil.copy(src / MAPS_SIDECAR_NAME, group_dir / MAPS_SIDECAR_NAME)

    shutil.rmtree(spill_root, ignore_errors=True)
    shutil.copy(src / MAPS_SIDECAR_NAME, out / MAPS_SIDECAR_NAME)

    per_map_named = _map_names(output_per_map, maps)
    manifest = _build_output_manifest(
        src=src,
        out=out,
        seed=seed,
        split_mode=split_mode,
        output_shard_size=output_shard_size,
        total_samples=int(sum(output_per_map.values())),
        total_shards=total_shards,
        per_map_named=per_map_named,
        per_split=per_split,
        command=command if command is not None else ["shuffle_dataset", str(src), str(out)],
        author=author,
        created_utc=created_utc if created_utc is not None else datetime.now(UTC).isoformat(),
    )
    manifest_path = _write_manifest(out, manifest)

    return ShuffleResult(
        out=out,
        split_mode=split_mode,
        seed=seed,
        output_shard_size=output_shard_size,
        total_samples=int(sum(output_per_map.values())),
        total_shards=total_shards,
        per_map_samples=per_map_named,
        manifest_path=manifest_path,
        per_split=per_split,
    )


# --------------------------------------------------------------------------------------
# CLI glue (argparse / clock read / file I/O / stdout summary) — untested, mirrors describe.py.
# --------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.data.shuffle",
        description="Reproducible, split-respecting, out-of-core on-disk dataset shuffle.",
    )
    parser.add_argument("src", type=Path, help="source dataset directory (holds manifest.json)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default datasets/decode-v2_shuffled-s<seed>)",
    )
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    parser.add_argument(
        "--split-mode",
        choices=("global", "within-split"),
        default="global",
        help="global (default) flat shards, or within-split train/val/test subdirs",
    )
    parser.add_argument(
        "--output-shard-size",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_SIZE,
        help=f"target rows per output shard (default {DEFAULT_OUTPUT_SHARD_SIZE})",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="within-split val fraction")
    parser.add_argument("--test-frac", type=float, default=0.15, help="within-split test fraction")
    parser.add_argument(
        "--force", action="store_true", help="allow writing into a non-empty output dir"
    )
    parser.add_argument("--author", default="claude", help="data-card author (default 'claude')")
    parser.add_argument(
        "--spill-dir", type=Path, default=None, help="override the scratch spill directory"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the shuffle from the command line. Returns ``0`` on success, ``2`` on a usage error.

    GLUE: parses args, picks the default ``--out`` name, reads the clock for the manifest
    ``created_utc``, delegates to :func:`shuffle_dataset`, and prints a one-line summary.
    """
    args = _parse_args(argv)
    out = (
        args.out if args.out is not None else Path("datasets") / f"decode-v2_shuffled-s{args.seed}"
    )
    argv_record = ["python", "-m", "pop_trainer.data.shuffle", *(argv or [])]

    try:
        result = shuffle_dataset(
            args.src,
            out,
            seed=args.seed,
            split_mode=args.split_mode,
            output_shard_size=args.output_shard_size,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            force=args.force,
            author=args.author,
            command=argv_record,
            created_utc=datetime.now(UTC).isoformat(),
            spill_dir=args.spill_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2

    print(
        f"shuffled {result.total_samples} samples -> {result.total_shards} shards "
        f"({result.split_mode}) at {result.out}"
    )
    if result.per_split:
        for name in SPLIT_NAMES:
            info = result.per_split.get(name, {})
            print(
                f"  {name}: {info.get('samples', 0)} samples, {info.get('shards', 0)} shards, "
                f"maps={info.get('maps', [])}"
            )
    print(f"manifest -> {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
