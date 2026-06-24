"""Tests for the collection runner (concrete factories + spec building + the launch glue).

Per the data contract the LIVE socket / subprocess path stays UNTESTED: no real socket is opened
and no real build is launched here. Two seams are exercised instead:

* the PURE pieces — ``build_specs`` (CLI args -> specs), the selector surface, ``player1_factory``,
  and ``build_env_from_connection`` (the env-build core) driven over an in-process fake transport
  exactly like ``test_collect.py``;
* the launch GLUE — ``env_factory`` with ``subprocess.Popen`` + ``core.launch.connect`` monkey-
  patched to fakes, asserting the wiring (port = base + worker_id, the proc stashed) and that
  closing the env REAPS the stashed proc.
"""

import numpy as np
import pytest

from pop_trainer import agents
from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.data import collect_runner as R
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
        player1="aggressive-coverage",
        player2="opponent-shadower",
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


def test_build_specs_per_worker_out_dir_and_map_ids_and_max_steps():
    specs = _build(workers=2, episodes=3, max_steps=800, out_dir="root")
    for worker_id, spec in enumerate(specs):
        assert spec.out_dir.endswith(f"worker_{worker_id}")
        assert spec.max_steps == 800
        # One episode per map_ids entry; the single Phase-A map (id 0) repeated per episode.
        assert spec.map_ids == [0, 0, 0]


def test_build_specs_wires_module_level_factories():
    # Spawn-safe: the wired factories MUST be the importable module-level functions, not closures.
    specs = _build()
    for spec in specs:
        assert spec.env_factory is R.env_factory
        assert spec.player1_factory is R.player1_factory


def test_build_specs_extra_carries_launch_params():
    specs = _build(workers=1, base_port=50000)
    extra = specs[0].extra
    assert extra["base_port"] == 50000
    assert extra["player1"] == "aggressive-coverage"
    assert extra["player2"] == "opponent-shadower"
    assert extra["map"] == "custom1"
    assert extra["exe"] == str(R.DEFAULT_EXE)
    assert extra["config"] == str(R.MAP_CONFIGS["custom1"])
    assert tuple(extra["frame_shape"]) == R.FRAME_SHAPE


def test_build_specs_clamps_workers_to_max():
    specs = _build(workers=99)
    assert len(specs) == R.MAX_WORKERS == 8
    assert [s.worker_id for s in specs] == list(range(8))


def test_build_specs_clamps_workers_floor_to_one():
    specs = _build(workers=0)
    assert len(specs) == 1


def test_build_specs_rejects_unknown_selector():
    with pytest.raises(ValueError, match="unknown agent selector"):
        _build(player1="not-a-real-agent")


def test_build_specs_rejects_unknown_map():
    with pytest.raises(ValueError, match="unknown map"):
        _build(map_name="not-a-map")


def test_build_specs_rejects_zero_episodes():
    with pytest.raises(ValueError, match="episodes"):
        _build(episodes=0)


# --- selector surface --------------------------------------------------------------------


def test_selector_surface_is_the_coverage_family_plus_random():
    assert set(R.AGENT_SELECTORS) == {
        "aggressive-coverage",
        "wall-hugger",
        "opponent-shadower",
        "random",
    }


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


# --- player1_factory ---------------------------------------------------------------------


def test_player1_factory_builds_the_selected_agent():
    spec = CollectionSpec(
        worker_id=0,
        out_dir="out",
        map_ids=[0],
        max_steps=1,
        seed=5,
        extra={"player1": "random"},
    )
    agent = R.player1_factory(spec)
    assert isinstance(agent, agents.RandomAgent)
    assert hasattr(agent, "act")


def test_player1_factory_seed_is_reproducible():
    spec = CollectionSpec(
        worker_id=0, out_dir="out", map_ids=[0], max_steps=1, seed=7, extra={"player1": "random"}
    )
    a = R.player1_factory(spec)
    b = R.player1_factory(spec)
    np.testing.assert_array_equal(a.act(None), b.act(None))


# --- build_env_from_connection (the env-build core, fake transport) ----------------------


def test_build_env_from_connection_runs_an_episode():
    # The env-build core over an in-process fake transport: build the env (with its player2) and
    # drive one episode through the real collect.run_episode — NO live socket.
    from pop_trainer.data import collect

    states = [_flat_state(i) for i in range(3)]
    transport = ScriptedTransport(_episode_blobs(states, done_last=True))
    env = R.build_env_from_connection(
        P.Connection(transport),
        player2_selector="random",
        seed=0,
        max_steps=3,
        frame_shape=R.FRAME_SHAPE,
    )
    ep = collect.run_episode(
        env, player1=R.make_agent("random", seed=1), map_id=0, episode_id=0, max_steps=3, seed=0
    )
    assert len(ep) == 3
    assert ep.ended_done
    # The frame shape matches what we asked for (the build's rendered frame).
    assert ep.samples[0].frame.shape == R.FRAME_SHAPE


def test_build_env_from_connection_injects_player2():
    states = [_flat_state(0), _flat_state(1)]
    transport = ScriptedTransport(_episode_blobs(states))
    env = R.build_env_from_connection(
        P.Connection(transport), player2_selector="aggressive-coverage", seed=0, max_steps=2
    )
    assert isinstance(env.player2, agents.CoverageAgent)


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
    return CollectionSpec(
        worker_id=worker_id,
        out_dir="out",
        map_ids=[0],
        max_steps=max_steps,
        seed=0,
        env_factory=R.env_factory,
        player1_factory=R.player1_factory,
        extra={
            "exe": "build.exe",
            "config": "cfg.json",
            "base_port": base_port,
            "frame_shape": R.FRAME_SHAPE,
            "player1": "random",
            "player2": "random",
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

    capture = {}
    proc, _ = _patch_launch(
        monkeypatch,
        blobs=_episode_blobs([_flat_state(0), _flat_state(1), _flat_state(2)], done_last=True),
        capture=capture,
    )
    spec = CollectionSpec(
        worker_id=0,
        out_dir=str(tmp_path),
        map_ids=[0],
        max_steps=3,
        seed=0,
        env_factory=R.env_factory,
        player1_factory=R.player1_factory,
        extra={
            "exe": "build.exe",
            "config": "cfg.json",
            "base_port": 50000,
            "frame_shape": R.FRAME_SHAPE,
            "player1": "random",
            "player2": "random",
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
