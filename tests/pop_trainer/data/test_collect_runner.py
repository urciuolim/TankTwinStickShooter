"""Tests for the collection runner (concrete factories + spec building + the launch glue).

Per the data contract the LIVE socket / subprocess path stays UNTESTED: no real socket is opened
and no real build is launched here. Two seams are exercised instead:

* the PURE pieces — ``build_specs`` (CLI args -> specs), the selector surface, ``player1_factory`` /
  ``player2_factory``, and ``build_env_from_connection`` (the bare env-build core) driven over an
  in-process fake transport exactly like ``test_collect.py``;
* the launch GLUE — ``env_factory`` with ``subprocess.Popen`` + ``core.launch.connect`` monkey-
  patched to fakes, asserting the wiring (port = base + worker_id, the proc stashed) and that
  closing the env REAPS the stashed proc.
"""

import collections

import numpy as np
import pytest

from pop_trainer import agents
from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.data import collect_runner as R
from pop_trainer.data import schema, shards
from pop_trainer.data.collect import CollectionSpec

FRAME_H, FRAME_W = R.FRAME_HEIGHT, R.FRAME_WIDTH


# --- wire-byte builders (mirror tests/pop_trainer/data/test_collect.py) --------------------


def _frame_bytes(w, h, fill):
    payload = bytes(fill) * (w * h)
    header = (
        bytes([P.FRAME_TAG])
        + len(payload).to_bytes(4, "big")
        + w.to_bytes(2, "big")
        + h.to_bytes(2, "big")
        + bytes([P.FRAME_CHANNELS])
    )
    return header + payload


def _state_and_frame_bytes(state_list, *, winner=None, done=False, fill=(10, 20, 30)):
    msg = {"state": list(state_list)}
    if winner is not None:
        msg["winner"] = winner
    if done:
        msg["done"] = True
    return P.encode(msg) + _frame_bytes(FRAME_W, FRAME_H, fill)


def _flat_state(value=0.0):
    return [float(value)] * S.STATE_LEN


class ScriptedTransport:
    """In-process socket-like fake: emits queued recv blobs, records sendall bytes."""

    def __init__(self, blobs):
        self.blobs = list(blobs)
        self.sent = bytearray()
        self.closed = False

    def sendall(self, data):
        self.sent += data

    def recv(self, bufsize):
        if not self.blobs:
            return b""
        blob = self.blobs[0]
        if len(blob) <= bufsize:
            return self.blobs.pop(0)
        self.blobs[0] = blob[bufsize:]
        return blob[:bufsize]

    def close(self):
        self.closed = True


def _reset_blobs(first_state, *, fill=(0, 0, 0)):
    return [
        P.encode({"restart": True}),
        P.encode({"starting": True}),
        _state_and_frame_bytes(first_state, fill=fill),
    ]


def _episode_blobs(states, *, done_last=True):
    blobs = _reset_blobs(states[0])
    for i in range(1, len(states)):
        last = i == len(states) - 1
        blobs.append(_state_and_frame_bytes(states[i], fill=(i, i, i), done=done_last and last))
    return blobs


# --- build_specs -------------------------------------------------------------------------


def _build(**overrides):
    kwargs = dict(
        map_values=None,  # single-map (no rotation) by default
        pairings=[("aggressive-coverage", "opponent-shadower")],
        map_name="custom1",
        config=R.MAP_CONFIGS["custom1"],
        exe=R.DEFAULT_EXE,
        episodes=3,
        max_steps=800,
        out_dir="out",
        workers=4,
        base_port=50000,
        seed=100,
    )
    kwargs.update(overrides)
    return R.build_specs(**kwargs)


def test_build_specs_produces_one_spec_per_worker_with_distinct_ids():
    specs = _build(workers=4)
    assert len(specs) == 4
    assert [s.worker_id for s in specs] == [0, 1, 2, 3]


def test_build_specs_seeds_are_distinct_per_worker():
    specs = _build(workers=4, seed=100)
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == 4  # distinct
    assert seeds[0] == 100  # worker 0 == base seed
    # The stride is wide enough that per-episode seeds (seed + episode_id) never collide either.
    gaps = [b - a for a, b in zip(seeds, seeds[1:])]
    assert all(gap >= 1000 for gap in gaps)


def test_build_specs_single_map_plan_has_no_switch_and_intended_zero():
    specs = _build(workers=2, episodes=3, max_steps=800, out_dir="root")
    for worker_id, spec in enumerate(specs):
        assert spec.out_dir.endswith(f"worker_{worker_id}")
        assert spec.max_steps == 800
        # One EpisodePlan per episode; single-map mode -> no switch, intended int 0.
        assert len(spec.episode_plan) == 3
        for plan in spec.episode_plan:
            assert plan.switch_arena is None
            assert plan.intended_map_id == 0
            assert plan.player1 == "aggressive-coverage"
            assert plan.player2 == "opponent-shadower"
        # No rotation -> empty map_index (the echo lookup is skipped, the intended int wins).
        assert spec.map_index == {}


def test_build_specs_wires_module_level_factories():
    # Spawn-safe: the wired factories MUST be the importable module-level functions, not closures.
    specs = _build()
    for spec in specs:
        assert spec.env_factory is R.env_factory
        assert spec.agent_pool_factory is R.agent_pool_factory


def test_build_specs_extra_carries_launch_params_and_pool():
    specs = _build(workers=1, base_port=50000)
    extra = specs[0].extra
    assert extra["base_port"] == 50000
    # The pool is the union of the pairing's selectors.
    assert set(extra["pool"]) == {"aggressive-coverage", "opponent-shadower"}
    assert extra["map"] == "custom1"
    assert extra["exe"] == str(R.DEFAULT_EXE)
    # Single-map boot config is the --map config.
    assert extra["config"] == str(R.MAP_CONFIGS["custom1"])
    # frame_shape is DERIVED from the boot config's obs_pixels_* (one source of truth), so it equals
    # whatever resolution that config declares — not a hardcoded constant.
    from pop_trainer.core.obs import frame_shape_from_config

    assert tuple(extra["frame_shape"]) == frame_shape_from_config(R.MAP_CONFIGS["custom1"])
    # The maps-list (int -> arena) is carried for the sidecar; single map -> the boot arena.
    assert extra["maps"] == ["Arenas/custom1.json"]


def test_build_specs_clamps_workers_to_max():
    specs = _build(workers=99)
    assert len(specs) == R.MAX_WORKERS == 8
    assert [s.worker_id for s in specs] == list(range(8))


def test_build_specs_clamps_workers_floor_to_one():
    specs = _build(workers=0)
    assert len(specs) == 1


def test_build_specs_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        _build(pairings=[("not-a-real-agent", "random")])


def test_build_specs_rejects_unknown_map():
    with pytest.raises(ValueError, match="unknown map"):
        _build(map_name="not-a-map")


def test_build_specs_rejects_zero_episodes():
    with pytest.raises(ValueError, match="episodes"):
        _build(episodes=0)


def test_build_specs_rejects_empty_pairings():
    with pytest.raises(ValueError, match="pairing"):
        _build(pairings=[])


# --- build_specs in ROTATION mode --------------------------------------------------------


def test_build_specs_rotation_mode_switches_and_indexes():
    # A rotation over two explicit arena targets x one pairing: every plan switches to a rotation
    # arena, the map_index decodes both, and the boot config is the obs_pixels --map config.
    arenas = ["Arenas/center_block.json", "Arenas/empty.json"]
    specs = _build(map_values=arenas, episodes=4, workers=2)
    # The rotation arenas are exactly the passed targets, indexed by rotation order.
    for spec in specs:
        assert spec.map_index == {"Arenas/center_block.json": 0, "Arenas/empty.json": 1}
        assert spec.extra["maps"] == ["Arenas/center_block.json", "Arenas/empty.json"]
        # The build boots on the obs_pixels-enabled --map config in BOTH modes; switch_arena
        # rotates the arena (the curated rotation is switch targets only).
        assert spec.extra["config"] == str(R.MAP_CONFIGS["custom1"])
        for plan in spec.episode_plan:
            assert plan.switch_arena in {"Arenas/center_block.json", "Arenas/empty.json"}
            assert plan.intended_map_id == spec.map_index[plan.switch_arena]


# --- selector surface --------------------------------------------------------------------


def test_selector_surface_is_the_coverage_family_plus_random_and_noop():
    # The registry now lives in pop_trainer.agents; collect_runner re-exports it. The roster is
    # the coverage family, random, and the noop floor (a valid scripted opponent).
    assert set(R.AGENT_SELECTORS) == {
        "aggressive-coverage",
        "wall-hugger",
        "opponent-shadower",
        "random",
        "noop",
    }
    # The re-exported names are the SAME objects the registry owns (the de-dup is real).
    assert R.AGENT_SELECTORS is agents.AGENT_SELECTORS
    assert R.make_agent is agents.make_agent


def test_every_selector_builds_a_valid_agent():
    for name in R.AGENT_SELECTORS:
        agent = R.make_agent(name, seed=0)
        assert hasattr(agent, "act")


def test_make_agent_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        R.make_agent("nope")


def test_make_agent_builds_the_right_types():
    assert isinstance(R.make_agent("aggressive-coverage"), agents.CoverageAgent)
    assert isinstance(R.make_agent("wall-hugger"), agents.CoverageAgent)
    assert isinstance(R.make_agent("opponent-shadower"), agents.CoverageAgent)
    assert isinstance(R.make_agent("random"), agents.RandomAgent)
    assert isinstance(R.make_agent("noop"), agents.NoOpAgent)


# --- agent_pool_factory ------------------------------------------------------------------


def _pool_spec(pool, seed=5):
    return CollectionSpec(
        worker_id=0,
        out_dir="out",
        episode_plan=[],
        max_steps=1,
        seed=seed,
        extra={"pool": pool},
    )


def test_agent_pool_factory_builds_one_agent_per_selector():
    pool = R.agent_pool_factory(_pool_spec(["random", "aggressive-coverage"]))
    assert set(pool) == {"random", "aggressive-coverage"}
    assert isinstance(pool["random"], agents.RandomAgent)
    assert isinstance(pool["aggressive-coverage"], agents.CoverageAgent)


def test_agent_pool_factory_seed_is_reproducible():
    a = R.agent_pool_factory(_pool_spec(["random"], seed=7))
    b = R.agent_pool_factory(_pool_spec(["random"], seed=7))
    np.testing.assert_array_equal(a["random"].act(None), b["random"].act(None))


# --- build_env_from_connection (the env-build core, fake transport) ----------------------


def test_build_env_from_connection_runs_an_episode():
    # The bare env-build core over an in-process fake transport: build the env and drive one episode
    # through the real collect.run_episode with both driver-side agents — NO live socket.
    from pop_trainer.data import collect

    states = [_flat_state(i) for i in range(3)]
    transport = ScriptedTransport(_episode_blobs(states, done_last=True))
    env = R.build_env_from_connection(
        P.Connection(transport),
        seed=0,
        max_steps=3,
        frame_shape=R.FRAME_SHAPE,
    )
    ep = collect.run_episode(
        env,
        player1=R.make_agent("random", seed=1),
        player2=R.make_agent("random", seed=2),
        map_id=0,
        episode_id=0,
        max_steps=3,
        seed=0,
    )
    assert len(ep) == 3
    assert ep.ended_done
    # The frame shape matches what we asked for (the build's rendered frame).
    assert ep.samples[0].frame.shape == R.FRAME_SHAPE


# --- env_factory wiring + proc reaping (Popen + connect monkeypatched) -------------------


class _FakeProc:
    """A fake subprocess.Popen: records terminate/kill/wait and reports liveness via poll()."""

    def __init__(self):
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):  # noqa: ARG002
        self._alive = False
        return 0


def _patch_launch(monkeypatch, *, blobs, capture):
    """Patch Popen (-> a recorded _FakeProc) and core.launch.connect (-> a fake socket).

    ``capture`` collects the launched cmd + the fake proc + the connected port so the test can
    assert the wiring. The "socket" returned is the ScriptedTransport itself (Connection only needs
    sendall/recv/close), so the env reads canned bytes with no real socket.
    """
    proc = _FakeProc()
    transport = ScriptedTransport(blobs)

    def fake_popen(cmd, *args, **kwargs):  # noqa: ARG001
        capture["cmd"] = cmd
        capture["proc"] = proc
        return proc

    def fake_connect(port, *args, **kwargs):  # noqa: ARG001
        capture["port"] = port
        return transport

    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(R.launch, "connect", fake_connect)
    return proc, transport


def _spec(worker_id=2, base_port=50000, max_steps=2):
    from pop_trainer.data.collect import EpisodePlan

    return CollectionSpec(
        worker_id=worker_id,
        out_dir="out",
        episode_plan=[
            EpisodePlan(switch_arena=None, player1="random", player2="random", intended_map_id=0)
        ],
        max_steps=max_steps,
        seed=0,
        env_factory=R.env_factory,
        agent_pool_factory=R.agent_pool_factory,
        extra={
            "exe": "build.exe",
            "config": "cfg.json",
            "base_port": base_port,
            "frame_shape": R.FRAME_SHAPE,
            "pool": ["random"],
            "map": "custom1",
        },
    )


def test_env_factory_computes_port_from_base_plus_worker_id(monkeypatch):
    capture = {}
    _patch_launch(
        monkeypatch, blobs=_episode_blobs([_flat_state(0), _flat_state(1)]), capture=capture
    )
    spec = _spec(worker_id=3, base_port=50000)
    env = R.env_factory(spec)
    try:
        assert capture["port"] == 50003
        # The chosen port is recorded back into spec.extra for inspection.
        assert spec.extra["port"] == 50003
        # args[1] of the launch cmd is the same port (positional, as the build parses it).
        assert capture["cmd"][1] == "50003"
    finally:
        env.close()


def test_env_factory_stashes_the_launch_proc(monkeypatch):
    capture = {}
    proc, _ = _patch_launch(
        monkeypatch, blobs=_episode_blobs([_flat_state(0), _flat_state(1)]), capture=capture
    )
    env = R.env_factory(_spec())
    try:
        assert env._launch_proc is proc
    finally:
        env.close()


def test_env_close_reaps_the_launch_proc(monkeypatch):
    capture = {}
    proc, _ = _patch_launch(
        monkeypatch, blobs=_episode_blobs([_flat_state(0), _flat_state(1)]), capture=capture
    )
    env = R.env_factory(_spec())
    assert not proc.terminated
    env.close()
    # Closing the env terminated the stashed build proc (the reaping wrapper).
    assert proc.terminated
    assert proc.poll() is not None


def test_run_worker_finally_reaps_the_launch_proc(monkeypatch, tmp_path):
    # The integration of the reaping with collect.run_worker's finally: drive a full worker over
    # the fake build + transport and assert the proc was reaped after run_worker returns.
    from pop_trainer.data import collect
    from pop_trainer.data.collect import EpisodePlan

    capture = {}
    proc, _ = _patch_launch(
        monkeypatch,
        blobs=_episode_blobs([_flat_state(0), _flat_state(1), _flat_state(2)], done_last=True),
        capture=capture,
    )
    spec = CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        episode_plan=[
            EpisodePlan(switch_arena=None, player1="random", player2="random", intended_map_id=0)
        ],
        max_steps=3,
        seed=0,
        env_factory=R.env_factory,
        agent_pool_factory=R.agent_pool_factory,
        extra={
            "exe": "build.exe",
            "config": "cfg.json",
            "base_port": 50000,
            "frame_shape": R.FRAME_SHAPE,
            "pool": ["random"],
            "map": "custom1",
        },
    )
    result = collect.run_worker(spec)
    assert result["num_shards"] == 1
    # run_worker's finally called env.close(), which reaped the build.
    assert proc.terminated


def test_env_factory_reaps_proc_when_connect_fails(monkeypatch):
    # If the connect fails after the build is launched, the proc must still be reaped (no leak).
    proc = _FakeProc()

    def fake_popen(cmd, *args, **kwargs):  # noqa: ARG001
        return proc

    def boom(port, *args, **kwargs):  # noqa: ARG001
        raise ConnectionError("never connected")

    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(R.launch, "connect", boom)
    with pytest.raises(ConnectionError):
        R.env_factory(_spec())
    assert proc.terminated


# --- round-robin scheduler (pure; deterministic; balanced worker spread) ------------------

_ROT = ["m0", "m1", "m2"]  # a 3-map rotation
_PAIRS = [("a", "b"), ("c", "d")]  # 2 pairings -> a 3x2 = 6-cell grid
_IDX = {"m0": 0, "m1": 1, "m2": 2}


def _cells(plan):
    """The (map, pairing) cells a plan visited (the schedule as an auditable multiset)."""
    return [(p.switch_arena, (p.player1, p.player2)) for p in plan]


def test_round_robin_plan_is_deterministic():
    a = R.round_robin_plan(_ROT, _PAIRS, episodes=5, worker_id=1, n_workers=3, map_index=_IDX)
    b = R.round_robin_plan(_ROT, _PAIRS, episodes=5, worker_id=1, n_workers=3, map_index=_IDX)
    assert _cells(a) == _cells(b)
    assert len(a) == 5


def test_round_robin_plan_single_worker_covers_the_whole_grid():
    # One worker over exactly G = 6 episodes visits every (map x pairing) cell once.
    plan = R.round_robin_plan(_ROT, _PAIRS, episodes=6, worker_id=0, n_workers=1, map_index=_IDX)
    grid = {(m, pr) for m in _ROT for pr in _PAIRS}
    assert set(_cells(plan)) == grid
    assert len(_cells(plan)) == 6  # each cell exactly once


def test_round_robin_plan_tags_intended_from_index():
    plan = R.round_robin_plan(_ROT, _PAIRS, episodes=6, worker_id=0, n_workers=1, map_index=_IDX)
    for p in plan:
        assert p.intended_map_id == _IDX[p.switch_arena]


def test_round_robin_union_across_workers_is_balanced_and_distinct():
    # Across N workers the UNION covers the grid evenly and no two workers do the same episode at
    # the same step (the spread guarantee). G = 6, N = 3, episodes = 4 -> N*episodes = 12 = 2*G,
    # so every cell appears EXACTLY twice across all workers.
    n_workers, episodes = 3, 4
    plans = [
        R.round_robin_plan(
            _ROT, _PAIRS, episodes=episodes, worker_id=w, n_workers=n_workers, map_index=_IDX
        )
        for w in range(n_workers)
    ]
    # Balanced union: each of the 6 cells appears N*episodes/G = 2 times.
    union = collections.Counter(c for plan in plans for c in _cells(plan))
    assert len(union) == 6
    assert set(union.values()) == {2}
    # Distinct work: at a fixed step i, all workers pick DIFFERENT cells.
    for i in range(episodes):
        step_cells = [_cells(plans[w])[i] for w in range(n_workers)]
        assert len(set(step_cells)) == n_workers


def test_round_robin_balanced_when_grid_does_not_divide_evenly():
    # G = 6, N = 2, episodes = 5 -> N*episodes = 10; cells get floor/ceil of 10/6 = {1, 2}.
    plans = [
        R.round_robin_plan(_ROT, _PAIRS, episodes=5, worker_id=w, n_workers=2, map_index=_IDX)
        for w in range(2)
    ]
    union = collections.Counter(c for plan in plans for c in _cells(plan))
    assert sum(union.values()) == 10
    assert set(union.values()) <= {1, 2}  # within one of perfectly balanced


def test_round_robin_rejects_empty_grid():
    with pytest.raises(ValueError):
        R.round_robin_plan([], _PAIRS, episodes=1, worker_id=0, n_workers=1, map_index={})


# --- rotation + map index ----------------------------------------------------------------


_CURATED_ROTATION = [
    "Arenas/center_block.json",
    "Arenas/central_cross.json",
    "Arenas/chokepoint.json",
    "Arenas/custom1_2021.json",
    "Arenas/diagonal_pillars.json",
    "Arenas/empty.json",
    "Arenas/four_pillars.json",
    "Arenas/opposing_l.json",
    "Arenas/ring_fragments.json",
    "Arenas/scattered.json",
]


def test_build_rotation_all_maps_yields_the_curated_rotation():
    # The ALL_MAPS sentinel (flag with no value) yields the curated 10-map rotation, in order.
    assert R.build_rotation([]) == _CURATED_ROTATION
    from pop_trainer.core.maps import ALL_MAPS_SENTINEL

    assert R.build_rotation([ALL_MAPS_SENTINEL]) == _CURATED_ROTATION


def test_build_rotation_none_is_single_map():
    assert R.build_rotation(None) == []


def test_build_map_index_is_position_in_rotation():
    idx = R.build_map_index(["Arenas/a.json", "Arenas/b.json", "Arenas/c.json"])
    assert idx == {"Arenas/a.json": 0, "Arenas/b.json": 1, "Arenas/c.json": 2}


def test_build_map_index_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        R.build_map_index(["Arenas/a.json", "Arenas/a.json"])


# --- pairing parsing ---------------------------------------------------------------------


def test_parse_pairing_valid():
    assert R.parse_pairing("aggressive-coverage:random") == ("aggressive-coverage", "random")


def test_parse_pairing_rejects_malformed():
    with pytest.raises(ValueError, match="pairing"):
        R.parse_pairing("just-one")


def test_parse_pairing_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        R.parse_pairing("random:not-a-real-agent")


def test_default_pairings_cover_coverage_vs_each_other_and_random():
    selectors = {s for pairing in R.DEFAULT_PAIRINGS for s in pairing}
    assert "random" in selectors
    # The coverage family is all present in the default mix.
    assert {"aggressive-coverage", "opponent-shadower", "wall-hugger"} <= selectors


# --- maps-list sidecar round-trip --------------------------------------------------------


def test_maps_sidecar_round_trips(tmp_path):
    rotation = ["Arenas/center_block.json", "Arenas/empty.json", "Arenas/four_pillars.json"]
    path = R.write_maps_sidecar(tmp_path, rotation)
    assert path.name == R.MAPS_SIDECAR_NAME
    decoded = R.read_maps_sidecar(path)
    assert decoded == rotation
    # The int -> path mapping is reversible: maps[i] decodes int i back to the arena path.
    for i, arena in enumerate(rotation):
        assert decoded[i] == arena


def test_maps_sidecar_is_strict_json(tmp_path):
    import json

    path = R.write_maps_sidecar(tmp_path, ["Arenas/empty.json"])
    payload = json.loads(path.read_text(encoding="utf-8"))  # strict json.loads must accept it
    assert payload["maps"] == ["Arenas/empty.json"]
    assert payload["schema_version"] == 1


def test_read_maps_sidecar_rejects_bad_payload(tmp_path):
    import json

    bad = tmp_path / "maps.json"
    bad.write_text(json.dumps({"not_maps": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="maps"):
        R.read_maps_sidecar(bad)


# --- summarize_collection (PURE end-of-run counting; no disk I/O) -------------------------

_SUMMARY_MAPS = ["Arenas/center_block.json", "Arenas/empty.json", "Arenas/four_pillars.json"]


def test_summarize_collection_totals_and_per_worker_counts():
    # Two workers; map 2 (four_pillars) gets ZERO samples from anyone -> a zero row in the table.
    worker_map_ids = [
        (0, np.array([0, 0, 1], dtype=np.int32)),  # 3 samples
        (1, np.array([0, 1, 1, 1], dtype=np.int32)),  # 4 samples
    ]
    s = R.summarize_collection(worker_map_ids, _SUMMARY_MAPS, total_shards=11)
    assert s.total_shards == 11
    assert s.total_samples == 7
    assert s.worker_samples == [(0, 3), (1, 4)]


def test_summarize_collection_per_map_aligned_to_maps_with_zero_row():
    worker_map_ids = [
        (0, np.array([0, 0, 1], dtype=np.int32)),
        (1, np.array([0, 1, 1, 1], dtype=np.int32)),
    ]
    s = R.summarize_collection(worker_map_ids, _SUMMARY_MAPS, total_shards=11)
    # ONE row per arena in the maps list, in maps order, including the zero-sample map.
    assert s.map_samples == [
        (0, "Arenas/center_block.json", 3),  # 2 from w0 + 1 from w1
        (1, "Arenas/empty.json", 4),  # 1 from w0 + 3 from w1
        (2, "Arenas/four_pillars.json", 0),  # never collected -> 0, still shown
    ]
    # The per-map table covers every map in the list (no map dropped, none invented).
    assert [m for _, m, _ in s.map_samples] == _SUMMARY_MAPS
    # Per-map counts reconcile with the grand total.
    assert sum(n for _, _, n in s.map_samples) == s.total_samples


def test_summarize_collection_worker_with_no_samples_is_zero_not_a_crash():
    # A worker that wrote nothing contributes an empty map_ids array -> (id, 0), no exception.
    worker_map_ids = [
        (0, np.array([0, 1, 2], dtype=np.int32)),
        (1, np.empty(0, dtype=np.int32)),  # wrote no shards
    ]
    s = R.summarize_collection(worker_map_ids, _SUMMARY_MAPS, total_shards=3)
    assert s.worker_samples == [(0, 3), (1, 0)]
    assert s.total_samples == 3
    assert s.map_samples == [
        (0, "Arenas/center_block.json", 1),
        (1, "Arenas/empty.json", 1),
        (2, "Arenas/four_pillars.json", 1),
    ]


def test_summarize_collection_single_map_run():
    # Single-map mode: one arena, all samples tagged int 0.
    worker_map_ids = [(0, np.zeros(5, dtype=np.int32)), (1, np.zeros(2, dtype=np.int32))]
    s = R.summarize_collection(worker_map_ids, ["Arenas/custom1.json"], total_shards=2)
    assert s.total_samples == 7
    assert s.map_samples == [(0, "Arenas/custom1.json", 7)]


# --- _read_worker_map_ids (the summary disk-read path; lazy per-member, NEVER touches frames) ---


def _write_real_shard(out_dir, *, prefix, index, n, map_ids, frame_hw=(180, 320)):
    """Write a real production shard with a deliberately heavy ``frames`` array via the writer.

    ``frames`` is sized so that decompressing it would dominate (n x H x W x 3 uint8), proving the
    summary read stays cheap only if it never inflates that member. The shard is built and written
    by the SAME ``Shard`` + ``write_shard`` path collection uses, so its byte layout is authentic.
    """
    h, w = frame_hw
    shard = shards.Shard(
        frames=np.zeros((n, h, w, schema.FRAME_CHANNELS), dtype=np.uint8),
        states=np.zeros((n, S.STATE_LEN), dtype=np.float32),
        map_ids=np.asarray(map_ids, dtype=np.int32),
        episode_ids=np.zeros(n, dtype=np.int32),
        step_idxs=np.arange(n, dtype=np.int32),
    )
    return shards.write_shard(out_dir / f"{prefix}_{index:04d}.npz", shard)


def test_read_worker_map_ids_returns_concatenated_ids_in_shard_order(tmp_path):
    # Two real shards (the runner's shard_w{id}_NNNN.npz names) -> ids concatenated in sorted order.
    _write_real_shard(tmp_path, prefix="shard_w0", index=0, n=3, map_ids=[0, 0, 1])
    _write_real_shard(tmp_path, prefix="shard_w0", index=1, n=2, map_ids=[2, 1])
    ids = R._read_worker_map_ids(tmp_path)
    assert ids.dtype == np.int32
    np.testing.assert_array_equal(ids, np.array([0, 0, 1, 2, 1], dtype=np.int32))


def test_read_worker_map_ids_empty_worker_returns_empty_not_raise(tmp_path):
    # A worker that wrote no shards -> empty int32 array, NOT a crash (summary must survive it).
    ids = R._read_worker_map_ids(tmp_path)
    assert ids.dtype == np.int32
    assert ids.shape == (0,)


def test_read_worker_map_ids_never_decompresses_frames(tmp_path, monkeypatch):
    # REGRESSION (perf): the summary read must inflate ONLY the tiny map_ids member, never the big
    # frames array. We spy on NpzFile member access and assert "frames" is NEVER requested while
    # "map_ids" IS. The frames array here is large enough that the OLD build_index path (which
    # materializes every member via read_shard) would record a "frames" access and fail this test.
    from numpy.lib.npyio import NpzFile

    _write_real_shard(tmp_path, prefix="shard_w0", index=0, n=400, map_ids=[7] * 400)

    accessed: list[str] = []
    orig_getitem = NpzFile.__getitem__

    def spy_getitem(self, key):
        accessed.append(key)
        return orig_getitem(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", spy_getitem)

    ids = R._read_worker_map_ids(tmp_path)

    # The spy actually fired (guards against a silently-broken patch passing the negative assert).
    assert schema.ARRAY_MAP_IDS in accessed
    # The whole point: the big frames member is NEVER inflated by the summary read path.
    assert schema.ARRAY_FRAMES not in accessed
    # Correctness, not just the I/O guard: the ids round-trip exactly.
    np.testing.assert_array_equal(ids, np.full(400, 7, dtype=np.int32))
