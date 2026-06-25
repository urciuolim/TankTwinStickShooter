"""Tier-1 LIVE e2e: drive a small real collection against a live Unity build.

This is a FUNCTIONAL / integration net — it proves the collection PIPELINE works end to end
against a real build (two workers, a 2-map rotation, real shards on disk with a valid schema and
both players' actions, and the startup memory-estimate line). It is NOT a scale / OOM test: the
OOM-incident class (the byte-budget shard size + the pre-flight memory guard) is covered by the
fast, no-allocation part-C tests in ``test_collect.py``.

It is marked ``e2e`` and so EXCLUDED from the default run (the runbook uses
``-m "not integration and not e2e"``). It AUTO-SKIPS when no Unity build is present
(``collect_runner.DEFAULT_EXE`` missing), so a buildless dev / CI run stays green (skipped, not
failed). Run it explicitly with a build present via ``-m e2e``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pop_trainer.core import maps as core_maps
from pop_trainer.core.state import STATE_LEN
from pop_trainer.data import collect_runner as R
from pop_trainer.data import schema, shards

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not R.DEFAULT_EXE.exists(),
        reason=f"no Unity build at {R.DEFAULT_EXE} (e2e collection needs a live build)",
    ),
]

# Two shipped map configs (exp-configs/maps) for the rotation, so at least one switch_arena
# happens and both arenas must echo-tag.
_MAP_A = core_maps.DEFAULT_MAPS_DIR / "center_block.json"
_MAP_B = core_maps.DEFAULT_MAPS_DIR / "empty.json"


def test_live_small_collection_two_workers_two_maps(tmp_path, capsys):
    out_dir = tmp_path / "ds"
    # 2 workers, 2 episodes each over a 2-map rotation (forces >=1 switch_arena), a small step cap
    # for a fast run. A SINGLE pairing keeps the grid = 2 cells (one per map), so with this small
    # episode budget the round-robin is guaranteed to cover BOTH maps (worker 0 -> map A, worker 1
    # -> map B). The boot config stays the obs_pixels-enabled --map (demo) config.
    exit_code = R.main(
        [
            "--maps",
            str(_MAP_A),
            str(_MAP_B),
            "--pairing",
            "aggressive-coverage:opponent-shadower",
            "--episodes",
            "2",
            "--max-steps",
            "30",
            "--workers",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    # The pre-flight memory-estimate line is ALWAYS printed at startup.
    captured = capsys.readouterr()
    assert "memory estimate" in captured.out

    # Shards were written for BOTH workers (worker_0 + worker_1 subdirs non-empty).
    worker_shards: dict[int, list] = {}
    for worker_id in (0, 1):
        worker_dir = out_dir / f"worker_{worker_id}"
        assert worker_dir.is_dir(), f"missing {worker_dir}"
        found = sorted(worker_dir.glob("shard_*.npz"))
        assert found, f"no shards written for worker {worker_id}"
        worker_shards[worker_id] = found

    # Schema valid per shard: the 6 arrays (incl. actions), schema dtypes, and the canonical shapes.
    all_map_ids: set[int] = set()
    saw_player2_action = False
    for shard_paths in worker_shards.values():
        for path in shard_paths:
            arrays = shards.read_shard(path)
            for name in schema.ALL_ARRAYS:
                assert name in arrays, f"{path.name} missing {name}"
            n = arrays[schema.ARRAY_MAP_IDS].shape[0]
            assert arrays[schema.ARRAY_STATES].shape == (n, STATE_LEN)
            assert arrays[schema.ARRAY_ACTIONS].shape == (
                n,
                schema.NUM_PLAYERS,
                schema.ACTION_LEN,
            )
            assert arrays[schema.ARRAY_FRAMES].dtype == np.dtype(schema.DTYPES[schema.ARRAY_FRAMES])
            assert arrays[schema.ARRAY_STATES].dtype == np.dtype(schema.DTYPES[schema.ARRAY_STATES])
            assert arrays[schema.ARRAY_ACTIONS].dtype == np.dtype(
                schema.DTYPES[schema.ARRAY_ACTIONS]
            )
            all_map_ids.update(int(m) for m in arrays[schema.ARRAY_MAP_IDS])
            # Player2 (action row 1) is a real driven player: at least one non-boundary sample has
            # a non-zero player2 action somewhere across the dataset.
            p2_rows = arrays[schema.ARRAY_ACTIONS][:, 1, :]
            if np.any(p2_rows != 0):
                saw_player2_action = True

    # BOTH maps are echo-tagged: the maps.json sidecar lists both arenas AND the union of on-disk
    # map_ids across shards covers both ints.
    sidecar = R.read_maps_sidecar(out_dir / R.MAPS_SIDECAR_NAME)
    assert len(sidecar) == 2
    arena_a = R.arena_path_for_config(_MAP_A)
    arena_b = R.arena_path_for_config(_MAP_B)
    assert set(sidecar) == {arena_a, arena_b}
    assert {sidecar.index(arena_a), sidecar.index(arena_b)} <= all_map_ids

    # Both players' actions are present (player2's row is not uniformly zero).
    assert saw_player2_action
