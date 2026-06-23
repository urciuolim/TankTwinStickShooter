"""Tests for shard npz read/write: round-trip EXACT, atomic write, schema validation."""

import numpy as np
import pytest

from pop_trainer.data import schema, shards


def _make_arrays(n=7, h=12, w=16, with_actions=True):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    states = rng.standard_normal((n, schema.STATE_LEN)).astype(np.float32)
    map_ids = rng.integers(0, 4, size=(n,), dtype=np.int32)
    episode_ids = np.arange(n, dtype=np.int32)
    step_idxs = np.arange(n, dtype=np.int32)
    actions = (
        rng.standard_normal((n, schema.NUM_PLAYERS, schema.ACTION_LEN)).astype(np.float32)
        if with_actions
        else None
    )
    return frames, states, map_ids, episode_ids, step_idxs, actions


def test_round_trip_exact_with_actions(tmp_path):
    frames, states, map_ids, episode_ids, step_idxs, actions = _make_arrays()
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=actions)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)
    assert path.exists()

    out = shards.read_shard(path)
    np.testing.assert_array_equal(out[schema.ARRAY_FRAMES], frames)
    np.testing.assert_array_equal(out[schema.ARRAY_STATES], states)
    np.testing.assert_array_equal(out[schema.ARRAY_MAP_IDS], map_ids)
    np.testing.assert_array_equal(out[schema.ARRAY_EPISODE_IDS], episode_ids)
    np.testing.assert_array_equal(out[schema.ARRAY_STEP_IDXS], step_idxs)
    np.testing.assert_array_equal(out[schema.ARRAY_ACTIONS], actions)
    # dtypes preserved exactly
    assert out[schema.ARRAY_FRAMES].dtype == np.uint8
    assert out[schema.ARRAY_STATES].dtype == np.float32
    assert out[schema.ARRAY_MAP_IDS].dtype == np.int32
    assert out[schema.ARRAY_ACTIONS].dtype == np.float32


def test_round_trip_without_actions(tmp_path):
    frames, states, map_ids, episode_ids, step_idxs, _ = _make_arrays(with_actions=False)
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=None)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)
    out = shards.read_shard(path)
    assert schema.ARRAY_ACTIONS not in out  # optional array absent
    np.testing.assert_array_equal(out[schema.ARRAY_FRAMES], frames)


def test_write_casts_to_schema_dtypes(tmp_path):
    # Feed wrong dtypes; the round-trip must come back as the schema dtypes, value-equal.
    n, h, w = 3, 8, 8
    frames = np.zeros((n, h, w, 3), dtype=np.int64)  # wrong dtype on purpose
    states = np.ones((n, schema.STATE_LEN), dtype=np.float64)
    map_ids = np.array([0, 1, 2], dtype=np.int64)
    episode_ids = np.array([0, 0, 0], dtype=np.int64)
    step_idxs = np.array([0, 1, 2], dtype=np.int64)
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs)
    out = shards.read_shard(shards.write_shard(tmp_path / "shard_w0_0000.npz", shard))
    assert out[schema.ARRAY_FRAMES].dtype == np.uint8
    assert out[schema.ARRAY_STATES].dtype == np.float32
    assert out[schema.ARRAY_MAP_IDS].dtype == np.int32
    np.testing.assert_array_equal(out[schema.ARRAY_STATES], np.ones((n, schema.STATE_LEN)))


def test_write_is_atomic_no_tmp_left(tmp_path):
    frames, states, map_ids, episode_ids, step_idxs, actions = _make_arrays()
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=actions)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)
    # No leftover *.tmp after a successful atomic write.
    assert list(tmp_path.glob("*.tmp")) == []
    assert path.name == "shard_w0_0000.npz"


def test_shard_rejects_mismatched_lengths():
    frames, states, map_ids, episode_ids, step_idxs, _ = _make_arrays(n=5)
    with pytest.raises(ValueError):
        shards.Shard(frames, states, map_ids[:3], episode_ids, step_idxs)


def test_shard_rejects_bad_state_width():
    n, h, w = 4, 8, 8
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    bad_states = np.zeros((n, schema.STATE_LEN - 1), dtype=np.float32)
    with pytest.raises(ValueError):
        shards.Shard(frames, bad_states, np.zeros(n), np.zeros(n), np.zeros(n))


def test_shard_rejects_bad_frame_channels():
    n, h, w = 4, 8, 8
    frames = np.zeros((n, h, w, 4), dtype=np.uint8)  # 4 channels
    states = np.zeros((n, schema.STATE_LEN), dtype=np.float32)
    with pytest.raises(ValueError):
        shards.Shard(frames, states, np.zeros(n), np.zeros(n), np.zeros(n))


def test_read_shard_meta_cheap(tmp_path):
    frames, states, map_ids, episode_ids, step_idxs, actions = _make_arrays(n=9, h=10, w=20)
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=actions)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)
    meta = shards.read_shard_meta(path)
    assert meta["n"] == 9
    assert meta["frame_hw"] == (10, 20)
    assert meta["has_actions"] is True
    assert shards.shard_length(path) == 9


def test_read_shard_meta_no_actions(tmp_path):
    frames, states, map_ids, episode_ids, step_idxs, _ = _make_arrays(
        n=4, h=8, w=12, with_actions=False
    )
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=None)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)
    meta = shards.read_shard_meta(path)
    assert meta == {"n": 4, "frame_hw": (8, 12), "has_actions": False}


def test_read_shard_meta_reads_header_only_not_frame_payload(tmp_path):
    # Prove read_shard_meta parses the .npy HEADER and never decompresses the frame payload:
    # corrupt the frames member's payload bytes (leaving its header intact) and confirm meta still
    # returns the correct (H, W) while a full read_shard chokes on the same file.
    import zipfile

    frames, states, map_ids, episode_ids, step_idxs, actions = _make_arrays(n=5, h=10, w=20)
    shard = shards.Shard(frames, states, map_ids, episode_ids, step_idxs, actions=actions)
    path = shards.write_shard(tmp_path / "shard_w0_0000.npz", shard)

    # Rewrite the zip with the frames member's payload truncated/garbled after its header.
    corrupt = tmp_path / "corrupt.npz"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(corrupt, "w", zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            raw = src.read(info.filename)
            if info.filename == f"{schema.ARRAY_FRAMES}.npy":
                # numpy v1.0 .npy: 10-byte magic+version+hlen, then the ASCII header dict, then
                # payload. Keep a generous header prefix; zero out the rest (the array payload).
                raw = raw[:128] + b"\x00" * (len(raw) - 128)
            dst.writestr(info, raw)

    meta = shards.read_shard_meta(corrupt)
    assert meta["frame_hw"] == (10, 20)  # header still parsed; payload never touched
    assert meta["n"] == 5


def test_read_shard_missing_required_array_raises(tmp_path):
    # Hand-write an npz lacking map_ids to prove the required-array guard fires.
    path = tmp_path / "broken.npz"
    np.savez_compressed(
        path,
        frames=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        states=np.zeros((2, schema.STATE_LEN), dtype=np.float32),
        episode_ids=np.zeros(2, dtype=np.int32),
        step_idxs=np.zeros(2, dtype=np.int32),
    )
    with pytest.raises(ValueError):
        shards.read_shard(path)


def test_samples_to_shard_via_collect(tmp_path):
    # Build a shard from collect.Sample objects and round-trip it.
    from pop_trainer.data import collect

    samples = []
    for i in range(4):
        samples.append(
            collect.Sample(
                frame=np.full((6, 6, 3), i, dtype=np.uint8),
                state=np.arange(schema.STATE_LEN, dtype=np.float32),
                action=np.zeros((schema.NUM_PLAYERS, schema.ACTION_LEN), dtype=np.float32),
                map_id=i % 2,
                episode_id=0,
                step_idx=i,
            )
        )
    shard = collect.samples_to_shard(samples)
    out = shards.read_shard(shards.write_shard(tmp_path / "shard_w0_0000.npz", shard))
    np.testing.assert_array_equal(out[schema.ARRAY_MAP_IDS], np.array([0, 1, 0, 1], dtype=np.int32))
    assert out[schema.ARRAY_FRAMES].shape == (4, 6, 6, 3)
