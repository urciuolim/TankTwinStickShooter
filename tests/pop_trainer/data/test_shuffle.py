"""Tests for the out-of-core dataset shuffle (:mod:`pop_trainer.data.shuffle`).

Each test builds a SMALL synthetic source dataset in a temp dir (never a real 83 GB run)
whose rows carry a unique FINGERPRINT redundantly encoded across ``episode_ids``,
``step_idxs``, a frame pixel, a state element, and an action element. After a shuffle the
fingerprints must still agree within every output row (row alignment) and the multiset of
fingerprints must be preserved (conservation).
"""

from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest

from pop_trainer.data import readers, schema, shards, shuffle

HW = (24, 24)
NUM_MAPS = 8


def _encode_frame(rid: int) -> np.ndarray:
    """A frame whose pixel (0, 0) encodes ``rid`` in base-256 across its three channels."""
    h, w = HW
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, 0, 0] = rid & 0xFF
    frame[0, 0, 1] = (rid >> 8) & 0xFF
    frame[0, 0, 2] = (rid >> 16) & 0xFF
    return frame


def _decode_frame(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0]) | (int(frame[0, 0, 1]) << 8) | (int(frame[0, 0, 2]) << 16)


def _write_source(root, *, n_shards: int, rows_per_shard: int, has_actions: bool = True) -> int:
    """Write a synthetic source dataset (shards + maps.json + manifest.json). Returns total rows."""
    root.mkdir(parents=True, exist_ok=True)
    maps = [f"Arenas/map_{i}.json" for i in range(NUM_MAPS)]
    total = 0
    rid = 0
    for s in range(n_shards):
        k = rows_per_shard
        frames = np.stack([_encode_frame(rid + i) for i in range(k)])
        states = np.zeros((k, schema.STATE_LEN), dtype=np.float32)
        episode_ids = np.empty(k, dtype=np.int32)
        step_idxs = np.empty(k, dtype=np.int32)
        map_ids = np.empty(k, dtype=np.int32)
        actions = np.zeros((k, schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32)
        for i in range(k):
            r = rid + i
            states[i, 0] = float(r)
            episode_ids[i] = r
            step_idxs[i] = r
            actions[i, 0, 0] = float(r)
            map_ids[i] = r % NUM_MAPS
        shard = shards.Shard(
            frames=frames,
            states=states,
            map_ids=map_ids,
            episode_ids=episode_ids,
            step_idxs=step_idxs,
            actions=actions if has_actions else None,
        )
        shards.write_shard(root / f"shard_w0_{s:04d}.npz", shard)
        rid += k
        total += k

    (root / "maps.json").write_text(
        json.dumps({"schema_version": 1, "maps": maps}, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "dataset": "synthetic-src",
        "descriptions": [
            {"author": "test", "text": "source", "added_utc": "2026-01-01T00:00:00+00:00"}
        ],
        "collection": {
            "seed": 7,
            "command": ["collect", "--synthetic"],
            "workers": 1,
            "episodes": n_shards,
            "max_steps": rows_per_shard,
            "maps": maps,
            "pairings": [["a", "b"]],
            "total_shards": n_shards,
            "total_samples": total,
            "per_map_samples": {m: 0 for m in maps},
        },
        "provenance": {
            "git_commit": "deadbeef",
            "git_dirty": False,
            "build": {
                "path": "build/app",
                "mtime_utc": "2026-01-01T00:00:00+00:00",
                "size_bytes": 1,
            },
        },
        "machine": {
            "hostname": "test",
            "platform": "test",
            "system": "test",
            "release": "0",
            "arch": "x86_64",
            "processor": "test",
            "cpu_count": 1,
            "ram_total_gb": 1.0,
            "python": "3.12.0",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return total


def _collect_rows(shard_dir, pattern="shard_*.npz") -> list[dict[str, np.ndarray]]:
    return [shards.read_shard(p) for p in sorted(shard_dir.glob(pattern))]


def _assert_aligned_and_conserved(rows_list, expected_total: int) -> None:
    """Assert every output row's fingerprints agree and the rid multiset is 0..total-1 once each."""
    seen = []
    for arrays in rows_list:
        n = arrays[schema.ARRAY_MAP_IDS].shape[0]
        for i in range(n):
            rid = int(arrays[schema.ARRAY_EPISODE_IDS][i])
            assert int(arrays[schema.ARRAY_STEP_IDXS][i]) == rid
            assert _decode_frame(arrays[schema.ARRAY_FRAMES][i]) == rid
            assert int(arrays[schema.ARRAY_STATES][i, 0]) == rid
            assert int(arrays[schema.ARRAY_ACTIONS][i, 0, 0]) == rid
            seen.append(rid)
    assert sorted(seen) == list(range(expected_total))


def test_global_alignment_and_conservation(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    total = _write_source(src, n_shards=6, rows_per_shard=50)

    result = shuffle.shuffle_dataset(src, out, seed=3, output_shard_size=64)

    assert result.total_samples == total
    assert result.split_mode == "global"
    rows = _collect_rows(out)
    assert sum(a[schema.ARRAY_MAP_IDS].shape[0] for a in rows) == total
    _assert_aligned_and_conserved(rows, total)
    # per-map counts preserved vs source
    src_rows = _collect_rows(src)
    src_counts = np.bincount(
        np.concatenate([a[schema.ARRAY_MAP_IDS] for a in src_rows]), minlength=NUM_MAPS
    )
    out_counts = np.bincount(
        np.concatenate([a[schema.ARRAY_MAP_IDS] for a in rows]), minlength=NUM_MAPS
    )
    assert np.array_equal(src_counts, out_counts)


def test_global_actually_reorders(tmp_path):
    """A global shuffle must NOT leave the rows in source order (sanity that it permutes)."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    total = _write_source(src, n_shards=4, rows_per_shard=60)
    shuffle.shuffle_dataset(src, out, seed=1, output_shard_size=80)
    rids = np.concatenate([a[schema.ARRAY_EPISODE_IDS] for a in _collect_rows(out)])
    assert not np.array_equal(rids, np.arange(total))


def test_determinism_byte_equal(tmp_path):
    src = tmp_path / "src"
    _write_source(src, n_shards=5, rows_per_shard=40)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    kw = {"seed": 11, "output_shard_size": 50, "created_utc": "2026-01-02T00:00:00+00:00"}
    shuffle.shuffle_dataset(src, out_a, **kw)
    shuffle.shuffle_dataset(src, out_b, **kw)

    files_a = sorted(p.name for p in out_a.glob("shard_*.npz"))
    files_b = sorted(p.name for p in out_b.glob("shard_*.npz"))
    assert files_a == files_b and files_a
    for name in files_a:
        aa = shards.read_shard(out_a / name)
        bb = shards.read_shard(out_b / name)
        assert set(aa) == set(bb)
        for key in aa:
            assert np.array_equal(aa[key], bb[key]), key
        # raw bytes are also identical (savez is deterministic here)
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_within_split_no_leak(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    total = _write_source(src, n_shards=6, rows_per_shard=50)

    result = shuffle.shuffle_dataset(
        src, out, seed=5, split_mode="within-split", output_shard_size=40
    )

    # the split must match readers.split_groups for the same seed/fractions
    all_ids = np.concatenate([a[schema.ARRAY_MAP_IDS] for a in _collect_rows(src)])
    expected = readers.split_groups(all_ids, val_frac=0.15, test_frac=0.15, seed=5)
    expected_groups = {
        name: set(expected.group_dict()[name].tolist()) for name in shuffle.SPLIT_NAMES
    }

    seen_rids: list[int] = []
    split_maps: dict[str, set[int]] = {}
    for name in shuffle.SPLIT_NAMES:
        sub = out / name
        assert sub.is_dir()
        rows = _collect_rows(sub)
        ids = set()
        for arrays in rows:
            ids.update(int(m) for m in arrays[schema.ARRAY_MAP_IDS].tolist())
            seen_rids.extend(int(r) for r in arrays[schema.ARRAY_EPISODE_IDS].tolist())
        split_maps[name] = ids
        assert ids == expected_groups[name], name

    # pairwise disjoint map sets
    assert split_maps["train"].isdisjoint(split_maps["val"])
    assert split_maps["train"].isdisjoint(split_maps["test"])
    assert split_maps["val"].isdisjoint(split_maps["test"])
    # conservation across all splits
    assert sorted(seen_rids) == list(range(total))
    assert result.total_samples == total
    # each split subdir carries its own maps.json
    for name in shuffle.SPLIT_NAMES:
        assert (out / name / "maps.json").exists()


def test_memory_bound(tmp_path):
    """Peak resident frame memory stays bounded to ~two shards, not a full-dataset load."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    n_shards, rows_per_shard = 8, 60
    _write_source(src, n_shards=n_shards, rows_per_shard=rows_per_shard)

    frame_bytes_per_shard = rows_per_shard * HW[0] * HW[1] * schema.FRAME_CHANNELS
    total_frame_bytes = n_shards * frame_bytes_per_shard

    peak_resident = 0
    live_shards = 0
    max_live_shards = 0

    def probe(phase: str, resident_frame_bytes: int) -> None:
        nonlocal peak_resident, live_shards, max_live_shards
        peak_resident = max(peak_resident, resident_frame_bytes)
        if phase == "scatter_loaded":
            live_shards += 1
        elif phase == "scatter_released":
            live_shards -= 1
        max_live_shards = max(max_live_shards, live_shards)

    tracemalloc.start()
    shuffle.shuffle_dataset(
        src, out, seed=2, output_shard_size=rows_per_shard, _resident_probe=probe
    )
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # never more than one source shard's frames alive during scatter
    assert max_live_shards == 1
    # probe-reported resident frames bounded to ~2 shards (shard + pack buffer / one bucket)
    assert peak_resident <= 3 * frame_bytes_per_shard
    assert peak_resident < total_frame_bytes
    # real measured peak (tracemalloc sees numpy buffers here) is far below a full load
    assert tm_peak < total_frame_bytes


def test_no_actions_dataset(tmp_path):
    """A (frame, state)-only source (no actions) round-trips through the shuffle."""
    src = tmp_path / "src"
    out = tmp_path / "out"
    total = _write_source(src, n_shards=3, rows_per_shard=40, has_actions=False)
    shuffle.shuffle_dataset(src, out, seed=0, output_shard_size=50)
    rows = _collect_rows(out)
    seen = []
    for arrays in rows:
        assert schema.ARRAY_ACTIONS not in arrays
        n = arrays[schema.ARRAY_MAP_IDS].shape[0]
        for i in range(n):
            rid = int(arrays[schema.ARRAY_EPISODE_IDS][i])
            assert _decode_frame(arrays[schema.ARRAY_FRAMES][i]) == rid
            assert int(arrays[schema.ARRAY_STATES][i, 0]) == rid
            seen.append(rid)
    assert sorted(seen) == list(range(total))


def test_manifest_strict_json_and_data_card(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_source(src, n_shards=4, rows_per_shard=50)
    shuffle.shuffle_dataset(
        src, out, seed=42, output_shard_size=64, created_utc="2026-02-02T00:00:00+00:00"
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # strict serialize (no NaN/Inf) must succeed
    json.dumps(manifest, allow_nan=False)
    for key in (
        "schema_version",
        "created_utc",
        "dataset",
        "descriptions",
        "collection",
        "provenance",
        "machine",
    ):
        assert key in manifest
    assert manifest["collection"]["command"][0] == "shuffle_dataset" or manifest["collection"][
        "command"
    ][0].startswith("python")
    # the appended data card records the load-bearing facts
    card = manifest["descriptions"][-1]
    assert card["author"] == "claude"
    text = card["text"]
    assert "SEED: 42" in text
    assert "two-pass bucketed out-of-core shuffle" in text
    assert "SPLIT-MODE: global" in text
    assert "OUTPUT SHARD SIZE: 64" in text
    assert src.as_posix() in text
    assert "#13" in text
    # source descriptions carried over (lineage preserved)
    assert any(d.get("text") == "source" for d in manifest["descriptions"])
    # maps.json copied
    assert (out / "maps.json").exists()


def test_within_split_data_card_mentions_no_leak(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_source(src, n_shards=5, rows_per_shard=50)
    shuffle.shuffle_dataset(src, out, seed=9, split_mode="within-split", output_shard_size=40)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    text = manifest["descriptions"][-1]["text"]
    assert "SPLIT-MODE: within-split" in text
    assert "NO-LEAK GUARANTEE" in text


def test_refuses_protected_and_nonempty(tmp_path):
    src = tmp_path / "src"
    _write_source(src, n_shards=2, rows_per_shard=20)

    # protected dataset names are always refused
    protected = tmp_path / "datasets" / "decode-v2_original"
    with pytest.raises(ValueError, match="protected"):
        shuffle.shuffle_dataset(src, protected, seed=0)
    protected2 = tmp_path / "datasets" / "decode-v2_reshuffled"
    with pytest.raises(ValueError, match="protected"):
        shuffle.shuffle_dataset(src, protected2, seed=0)

    # out == src refused
    with pytest.raises(ValueError, match="differ from source"):
        shuffle.shuffle_dataset(src, src, seed=0)

    # non-empty out refused unless force
    out = tmp_path / "out"
    out.mkdir()
    (out / "stuff.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        shuffle.shuffle_dataset(src, out, seed=0)
    # force succeeds
    shuffle.shuffle_dataset(src, out, seed=0, force=True, output_shard_size=30)
    assert (out / "manifest.json").exists()


def test_bad_args(tmp_path):
    src = tmp_path / "src"
    _write_source(src, n_shards=1, rows_per_shard=10)
    with pytest.raises(ValueError, match="split_mode"):
        shuffle.shuffle_dataset(src, tmp_path / "o1", split_mode="nope")
    with pytest.raises(ValueError, match="output_shard_size"):
        shuffle.shuffle_dataset(src, tmp_path / "o2", output_shard_size=0)
    with pytest.raises(ValueError, match="manifest"):
        shuffle.shuffle_dataset(tmp_path / "missing", tmp_path / "o3")


def test_cli_main(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "cli_out"
    _write_source(src, n_shards=3, rows_per_shard=40)
    rc = shuffle.main([str(src), "--out", str(out), "--seed", "4", "--output-shard-size", "50"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert list(out.glob("shard_*.npz"))

    # spill scratch dir is cleaned up
    assert not (out / ".shuffle_spill").exists()


def test_spill_cleaned_up(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write_source(src, n_shards=3, rows_per_shard=40)
    shuffle.shuffle_dataset(src, out, seed=0, output_shard_size=50)
    assert not (out / ".shuffle_spill").exists()
