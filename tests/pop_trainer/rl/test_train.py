"""Tests for pop_trainer.rl.train — the Phase-1 PPO integrator.

NO live Unity / NO real ``.learn`` / NO real PPO model: the env-composition seam is exercised over
a STUB ``connection_factory`` (so ``TankEnv`` builds with no socket), the sidecar round-trip is
pure (no model), and the resume restore mocks ``PPO.load``. The slow/torch paths (building PPO, the
encoder, ``model.learn``) are never entered here — they are the Director's live smoke.

Coverage:

1. ``TrainConfig`` defaults + validation (each invalid field raises) + ``to_dict`` round-trip.
2. ``build_vec_env`` over a stub factory: the wrapper stack is
   ``VecFrameStack -> DummyVecEnv -> SelfPlayWrapper -> TankEnv``; the ``OpponentProvider`` carries
   ``cfg.opponents`` + ``cfg.opponent_strategy``; ``_build_policy_kwargs`` carries the
   ``EncoderExtractor`` + the checkpoint/freeze kwargs.
3. sidecar ``save -> load`` round-trip (provider position + ELO + cfg + num_timesteps survive).
4. resume restores cfg + sidecar: ``_latest_checkpoint`` picks the max-step zip; ``PPO.load`` is
   mocked and the provider ``_index`` + ELO are restored.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from pop_trainer.agents import AGENT_SELECTORS
from pop_trainer.core.protocol import Connection
from pop_trainer.env.tank_env import TankEnv
from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import DEFAULT_ROSTER, OpponentProvider, SelfPlayWrapper
from pop_trainer.rl.train import (
    BASE_ELO,
    MEMORY_MARGIN,
    UNITY_INSTANCE_BYTES,
    TrainConfig,
    _attempt_unity_log_path,
    _build_base_env,
    _build_policy_kwargs,
    _checkpoint_save_freq,
    _find_selfplay_wrapper,
    _initial_elo,
    _latest_checkpoint,
    _live_connection_factory_for_port,
    _make_self_play_env,
    _make_sidecar_callback,
    _parse_args,
    _restore_provider_position,
    _terminate,
    _training_env_factories,
    build_vec_env,
    check_rl_memory_budget,
    estimate_rl_memory_bytes,
    eval_ports,
    frame_nbytes,
    load_sidecar,
    main,
    save_sidecar,
    training_ports,
)

# --- stub connection seam (no socket / no Unity) ---------------------------------------------


class _StubConnection:
    """A do-nothing ``Connection`` stand-in: ``TankEnv.__init__`` stores it; no socket I/O.

    The composition test constructs the vec stack but never resets/steps, so the wire is never
    exercised. ``send`` / ``receive`` are inert no-ops so ``TankEnv.close``'s best-effort
    end-handshake on ``vec_env.close()`` does not raise; ``transport`` exposes a no-op ``close``.
    """

    def __init__(self) -> None:
        self.transport = self  # close() is a no-op below.

    def send(self, message) -> None:
        return None

    def receive(self):
        return {}

    def close(self) -> None:
        return None


def _stub_factory() -> Connection:
    """A zero-arg ``connection_factory`` returning a stub connection (no build, no socket)."""
    return _StubConnection()  # type: ignore[return-value]


def _stub_factory_for_port(port: int):  # noqa: ARG001 (port unused; the stub is port-agnostic)
    """The multi-env TEST seam: ``port -> a zero-arg stub factory`` (no Unity on any port)."""
    return _stub_factory


def _cfg(tmp_path: Path, **overrides) -> TrainConfig:
    """A minimal valid TrainConfig rooted at ``tmp_path`` (overridable per test)."""
    base = {
        "total_timesteps": 1_000,
        "game_config": tmp_path / "train_config.json",
        "run_dir": tmp_path / "run",
        # A tiny frame keeps any space introspection cheap (we never build the encoder here).
        "frame_shape": (4, 6, 3),
    }
    base.update(overrides)
    return TrainConfig(**base)


# --- 1. TrainConfig: defaults + validation + to_dict -----------------------------------------


def test_trainconfig_defaults(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.n_envs == 1
    assert cfg.frame_stack == 1
    assert cfg.opponents == DEFAULT_ROSTER
    assert cfg.opponent_strategy == "round_robin"
    assert cfg.encoder_checkpoint is None
    assert cfg.freeze_encoder is False
    assert cfg.resume is None
    assert cfg.seed == 0
    # the cadence defaults are a production-scale 10_000 (a smoke run overrides them downward).
    assert cfg.checkpoint_freq == 10_000
    assert cfg.eval_freq == 10_000
    # eval_port defaults to None -> the effective eval port is game_port + 1 (a separate socket).
    assert cfg.eval_port is None
    assert cfg.effective_eval_port == cfg.game_port + 1


def test_eval_port_default_is_after_training_range_at_multi_env(tmp_path):
    # 7 training envs -> ports 50000..50006; eval defaults to game_port + n_envs = 50007 (the first
    # port AFTER the training range, no collision).
    cfg = _cfg(tmp_path, game_port=50000, n_envs=7)
    assert cfg.eval_port is None
    assert cfg.effective_eval_port == 50007
    assert cfg.effective_eval_port not in set(training_ports(cfg))


def test_eval_port_explicit_override(tmp_path):
    cfg = _cfg(tmp_path, game_port=50000, eval_port=50007)
    assert cfg.eval_port == 50007
    assert cfg.effective_eval_port == 50007  # explicit override wins over game_port + n_envs


def test_eval_port_equal_to_game_port_rejected(tmp_path):
    with pytest.raises(ValueError, match="eval_port must differ from game_port"):
        _cfg(tmp_path, game_port=50000, eval_port=50000)


def test_eval_port_inside_training_range_rejected(tmp_path):
    # n_envs=7 -> training block [50000, 50006]; an eval block based at 50003 overlaps it.
    with pytest.raises(ValueError, match="overlaps the training port block"):
        _cfg(tmp_path, game_port=50000, n_envs=7, eval_port=50003)


def test_eval_block_overlapping_training_block_from_below_rejected(tmp_path):
    # eval block [49998, 50004] (n_envs=7) overlaps training [50000, 50006] from below.
    with pytest.raises(ValueError, match="overlaps the training port block"):
        _cfg(tmp_path, game_port=50000, n_envs=7, eval_port=49998)


def test_eval_port_just_outside_training_range_accepted(tmp_path):
    # 50007 is one past training high (50006) -> the eval block [50007, 50013] is disjoint.
    cfg = _cfg(tmp_path, game_port=50000, n_envs=7, eval_port=50007)
    assert cfg.effective_eval_port == 50007
    assert eval_ports(cfg) == list(range(50007, 50014))
    assert set(eval_ports(cfg)).isdisjoint(set(training_ports(cfg)))


def test_eval_ports_default_block_disjoint_from_training(tmp_path):
    # Default eval base = game_port + n_envs; the eval block sits entirely after training.
    cfg = _cfg(tmp_path, game_port=50000, n_envs=4)
    assert training_ports(cfg) == [50000, 50001, 50002, 50003]
    assert eval_ports(cfg) == [50004, 50005, 50006, 50007]
    assert set(eval_ports(cfg)).isdisjoint(set(training_ports(cfg)))


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("total_timesteps", 0),
        ("total_timesteps", -1),
        ("n_envs", 0),
        ("frame_stack", 0),
        ("opponent_strategy", "bogus"),
        ("opponents", ()),
        ("eval_freq", -1),
        ("checkpoint_freq", 0),
        ("frame_shape", (4, 6)),
        ("frame_shape", (4, 6, 4)),
    ],
)
def test_trainconfig_validation_rejects(tmp_path, field, bad):
    with pytest.raises(ValueError):
        _cfg(tmp_path, **{field: bad})


def test_trainconfig_to_dict_roundtrips_paths_and_tuples(tmp_path):
    cfg = _cfg(
        tmp_path,
        encoder_checkpoint=tmp_path / "enc.pt",
        resume=tmp_path / "prior",
        build_path=tmp_path / "build.exe",
    )
    d = cfg.to_dict()
    # Paths -> str
    assert d["game_config"] == str(cfg.game_config)
    assert d["run_dir"] == str(cfg.run_dir)
    assert d["encoder_checkpoint"] == str(cfg.encoder_checkpoint)
    assert d["resume"] == str(cfg.resume)
    assert d["build_path"] == str(cfg.build_path)
    # tuples -> list
    assert d["opponents"] == list(DEFAULT_ROSTER)
    assert d["frame_shape"] == [4, 6, 3]
    # eval_port serializes the RAW field value (None stays None).
    assert d["eval_port"] is None
    explicit = _cfg(tmp_path, eval_port=50123)
    assert explicit.to_dict()["eval_port"] == 50123
    # allow_oversized round-trips (default False; explicit True).
    assert d["allow_oversized"] is False
    assert _cfg(tmp_path, allow_oversized=True).to_dict()["allow_oversized"] is True
    # None encoder_checkpoint / resume / build_path serialize as null
    none_cfg = _cfg(tmp_path)
    nd = none_cfg.to_dict()
    assert nd["encoder_checkpoint"] is None
    assert nd["resume"] is None
    assert nd["build_path"] is None
    assert nd["eval_port"] is None
    # the dict is JSON-serializable (no Path / tuple leaks)
    import json

    json.dumps(d)


# --- 2. env composition over a stub factory --------------------------------------------------


def test_build_vec_env_stack_and_provider(tmp_path):
    cfg = _cfg(tmp_path, opponents=("noop", "random"), opponent_strategy="uniform", frame_stack=1)
    vec = build_vec_env(cfg, connection_factory=_stub_factory)
    try:
        # outermost is VecFrameStack -> DummyVecEnv -> SelfPlayWrapper -> TankEnv
        assert isinstance(vec, VecFrameStack)
        inner = vec.venv
        assert isinstance(inner, DummyVecEnv)
        wrapper = inner.envs[0]
        assert isinstance(wrapper, SelfPlayWrapper)
        assert isinstance(wrapper.env, TankEnv)
        # the OpponentProvider carries the roster + strategy from cfg.
        provider = wrapper.opponents
        assert isinstance(provider, OpponentProvider)
        assert provider.strategy == "uniform"
        assert len(provider.opponents) == 2  # one ScriptedOpponent per selector
    finally:
        vec.close()


def test_build_vec_env_frame_stack_depth(tmp_path):
    cfg = _cfg(tmp_path, frame_stack=4)
    vec = build_vec_env(cfg, connection_factory=_stub_factory)
    try:
        assert isinstance(vec, VecFrameStack)
        # VecFrameStack stacks along the channel axis: stub frame (4,6,3) -> SB3 transposes to
        # channels-first then stacks; the depth is recorded on the stacked-obs helper.
        assert vec.stacked_obs.n_stack == 4
    finally:
        vec.close()


def test_training_ports_distinct_and_contiguous(tmp_path):
    cfg = _cfg(tmp_path, game_port=50000, n_envs=7)
    ports = training_ports(cfg)
    assert ports == [50000, 50001, 50002, 50003, 50004, 50005, 50006]
    assert len(set(ports)) == 7  # all distinct


def test_training_env_factories_count_and_ports(tmp_path):
    # The SubprocVecEnv seam: n_envs=7 -> 7 factory callables over 7 distinct ports 50000..50006,
    # with eval (game_port + n_envs = 50007) BEYOND the range (no collision). Asserted on the pure
    # port list + factory list so NO SubprocVecEnv / process is spawned.
    cfg = _cfg(tmp_path, game_port=50000, n_envs=7)
    factories = _training_env_factories(cfg, connection_factory_for_port=_stub_factory_for_port)
    assert len(factories) == 7
    ports = training_ports(cfg)
    assert len(set(ports)) == 7
    assert ports == list(range(50000, 50007))
    # the eval port is outside the training port set (no collision).
    assert cfg.effective_eval_port == 50007
    assert cfg.effective_eval_port not in set(ports)


def test_build_vec_env_multi_env_builds_distinct_ports(tmp_path):
    # The multi-env path now BUILDS (no NotImplementedError). A stub-factory-for-port makes the vec
    # a DummyVecEnv of the factories (no subprocs spawned) so each env's recorded port is checkable.
    cfg = _cfg(tmp_path, game_port=52000, n_envs=3)
    vec = build_vec_env(cfg, port=cfg.game_port, connection_factory_for_port=_stub_factory_for_port)
    try:
        assert isinstance(vec, VecFrameStack)
        inner = vec.venv
        assert isinstance(inner, DummyVecEnv)
        assert inner.num_envs == 3
        recorded = sorted(w.env.env_config.game_port for w in inner.envs)
        assert recorded == [52000, 52001, 52002]
        # each env carries its own seeded provider (per-subproc rotation).
        for w in inner.envs:
            assert isinstance(w, SelfPlayWrapper)
            assert isinstance(w.opponents, OpponentProvider)
    finally:
        vec.close()


def test_build_vec_env_eval_single_even_at_multi_env(tmp_path):
    # The eval build is ALWAYS a single env (DummyVecEnv, num_envs == 1) even when cfg.n_envs > 1,
    # because evaluate_winrate needs the raw single TankEnv. single=True forces the one-env path.
    cfg = _cfg(tmp_path, game_port=53000, n_envs=4)
    eval_vec = build_vec_env(
        cfg, port=cfg.effective_eval_port, single=True, connection_factory=_stub_factory
    )
    try:
        assert isinstance(eval_vec, VecFrameStack)
        inner = eval_vec.venv
        assert isinstance(inner, DummyVecEnv)
        assert inner.num_envs == 1
        wrapper = _find_selfplay_wrapper(eval_vec)
        assert isinstance(wrapper.env, TankEnv)
        # eval env sits on game_port + n_envs (53004), outside the training range.
        assert wrapper.env.env_config.game_port == 53004
    finally:
        eval_vec.close()


def test_build_vec_env_port_threads_into_envconfig(tmp_path):
    # The training env builds on game_port; the eval env on the effective eval port. The built
    # TankEnv records its socket via EnvConfig.game_port, so two distinct ports are assertable
    # without launching Unity (the stub factory carries no proc).
    cfg = _cfg(tmp_path, game_port=51000)
    train_vec = build_vec_env(cfg, port=cfg.game_port, connection_factory=_stub_factory)
    eval_vec = build_vec_env(cfg, port=cfg.effective_eval_port, connection_factory=_stub_factory)
    try:
        train_tank = _find_selfplay_wrapper(train_vec).env
        eval_tank = _find_selfplay_wrapper(eval_vec).env
        assert train_tank.env_config.game_port == 51000
        assert eval_tank.env_config.game_port == 51001  # game_port + 1
        assert train_tank.env_config.game_port != eval_tank.env_config.game_port
    finally:
        train_vec.close()
        eval_vec.close()


def test_build_vec_env_monitor_present_on_training_absent_on_eval(tmp_path):
    cfg = _cfg(tmp_path)
    train_vec = build_vec_env(
        cfg, port=cfg.game_port, monitor=True, connection_factory=_stub_factory
    )
    eval_vec = build_vec_env(
        cfg, port=cfg.effective_eval_port, monitor=False, connection_factory=_stub_factory
    )
    try:
        # VecMonitor is OUTERMOST on the training stack (wrapping the VecFrameStack).
        assert isinstance(train_vec, VecMonitor)
        assert isinstance(train_vec.venv, VecFrameStack)
        # The eval stack is NOT monitored (eval uses evaluate_winrate's own loop).
        assert not isinstance(eval_vec, VecMonitor)
        assert isinstance(eval_vec, VecFrameStack)
    finally:
        train_vec.close()
        eval_vec.close()


def test_find_selfplay_wrapper_walks_monitor_and_framestack(tmp_path):
    # The walk must reach the SelfPlayWrapper through VecMonitor(VecFrameStack(DummyVecEnv([...]))).
    cfg = _cfg(tmp_path, frame_stack=2)
    vec = build_vec_env(cfg, port=cfg.game_port, monitor=True, connection_factory=_stub_factory)
    try:
        assert isinstance(vec, VecMonitor)
        wrapper = _find_selfplay_wrapper(vec)
        assert isinstance(wrapper, SelfPlayWrapper)
        assert isinstance(wrapper.env, TankEnv)
        assert isinstance(wrapper.opponents, OpponentProvider)
    finally:
        vec.close()


# --- reap hook + lazy lifecycle (the env OWNS the reap; rl injects reap=_terminate) ----------


class _FakeProc:
    """A subprocess.Popen stand-in: records terminate/kill; poll reports running until reaped."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self._dead = False

    def poll(self):
        return 0 if self._dead else None

    def terminate(self):
        self.terminated = True
        self._dead = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._dead = True


class _ProcStubConnection(_StubConnection):
    """A stub ``Connection`` carrying a live build ``Popen`` on ``_launch_proc`` (the LIVE-path
    shape the env's reap hook reads). Each instance owns a fresh :class:`_FakeProc`."""

    def __init__(self) -> None:
        super().__init__()
        self._launch_proc = _FakeProc()


def _proc_factory_calls() -> tuple:
    """A connection_factory that yields a fresh ``_ProcStubConnection`` per call + a record list.

    Returns ``(factory, conns)`` where ``conns`` accumulates every produced connection so a test can
    inspect the proc that each (lazy launch / reconnect) created.
    """
    conns: list[_ProcStubConnection] = []

    def factory() -> Connection:
        conn = _ProcStubConnection()
        conns.append(conn)
        return conn

    return factory, conns


def test_build_base_env_injects_terminate_reap_hook(tmp_path):
    # The env OWNS the reap: _build_base_env wires reap=_terminate so release()/reconnect can
    # hard-kill the live Unity child via the env's own machinery (no close-monkeypatch).
    cfg = _cfg(tmp_path)
    env = _build_base_env(cfg, _stub_factory, cfg.game_port)
    assert env._reap is _terminate


def test_make_self_play_env_injects_terminate_reap_hook(tmp_path):
    # The self-play env unit (over an injected stub factory) also carries reap=_terminate.
    cfg = _cfg(tmp_path)
    wrapper = _make_self_play_env(cfg, cfg.game_port, connection_factory=_stub_factory)
    assert wrapper.env._reap is _terminate


def test_lazy_construction_does_not_connect(tmp_path):
    # With a factory, _build_base_env launches nothing: conn is None / not running until reset.
    cfg = _cfg(tmp_path)
    factory, conns = _proc_factory_calls()
    env = _build_base_env(cfg, factory, cfg.game_port)
    assert env.conn is None
    assert env.is_running is False
    assert conns == []  # the factory was never invoked at construction


def test_release_reaps_the_current_live_build(tmp_path):
    # The MUST-FIX intent (adapted to lazy + reap-hook): the live build is reaped on teardown via
    # release(), which hard-kills the CURRENT connection's proc through the injected reap hook.
    cfg = _cfg(tmp_path)
    factory, conns = _proc_factory_calls()
    env = _build_base_env(cfg, factory, cfg.game_port)

    env._ensure_connected()  # lazily materialize the live connection (carrying _launch_proc)
    assert env.is_running is True
    current_proc = env.conn._launch_proc

    env.release()

    assert current_proc.terminated  # the live build was hard-killed via reap=_terminate
    assert env.conn is None and env.is_running is False  # env object stays alive, conn freed


def test_reconnect_swapped_proc_is_the_one_reaped(tmp_path):
    # The MUST-FIX reconnect intent: a reconnect swaps in a NEW connection/proc; a later release()
    # reaps THAT (current) proc, and the kill-old-first reconnect already reaped the original.
    cfg = _cfg(tmp_path)
    factory, conns = _proc_factory_calls()
    env = _build_base_env(cfg, factory, cfg.game_port)

    env._ensure_connected()  # first (original) live connection
    original_proc = env.conn._launch_proc

    env._reconnect()  # kill-old-first: reaps original, then relaunches a fresh connection
    assert original_proc.terminated  # the kill-old-first reconnect reaped the original build
    current_proc = env.conn._launch_proc
    assert current_proc is not original_proc

    env.release()  # reaps the CURRENT (reconnect-swapped) proc
    assert current_proc.terminated


def test_release_is_idempotent_and_noop_without_proc(tmp_path):
    # release() on a not-running env is a no-op; a stub conn without _launch_proc reaps nothing.
    cfg = _cfg(tmp_path)
    env = _build_base_env(cfg, _stub_factory, cfg.game_port)
    env.release()  # not running -> no-op (no raise)
    env._ensure_connected()  # stub conn (NO _launch_proc)
    assert not hasattr(env.conn, "_launch_proc")
    env.release()  # nothing to reap; just frees the transport
    assert env.conn is None
    env.release()  # idempotent second call


# --- logfile-per-attempt (no truncation across relaunches) -----------------------------------


def test_attempt_unity_log_path_is_distinct_per_attempt(tmp_path):
    # Each launch attempt gets a distinct unity-<role>-<port>-<attempt>.log so a relaunch never
    # truncates the prior instance's C# log.
    p0 = _attempt_unity_log_path(tmp_path, "train", 50000, 0)
    p1 = _attempt_unity_log_path(tmp_path, "train", 50000, 1)
    assert p0.name == "unity-train-50000-0.log"
    assert p1.name == "unity-train-50000-1.log"
    assert p0 != p1


def test_live_factory_relaunch_uses_distinct_logfiles(tmp_path, monkeypatch):
    # The live connection_factory increments an attempt counter per invocation, so each
    # (lazy launch / reconnect / respawn) points Unity's -logFile at a DISTINCT path -> a relaunch
    # never truncates the prior instance's log. We capture the unity_log_path build_launch_cmd sees
    # across successive factory calls and assert they differ. No Unity is launched (Popen/connect
    # are stubbed).
    import pop_trainer.rl.train as train_mod

    cfg = _cfg(tmp_path, log_dir=tmp_path / "logs")
    log_paths: list = []

    def _fake_build_cmd(exe, port, config, *, unity_log_path=None, **kw):
        log_paths.append(unity_log_path)
        return ["stub", str(port)]

    class _FakePopen:
        def __init__(self, cmd):
            self.cmd = cmd

        def poll(self):
            return None

    monkeypatch.setattr(train_mod.launch, "build_launch_cmd", _fake_build_cmd)
    monkeypatch.setattr(train_mod.subprocess, "Popen", lambda cmd: _FakePopen(cmd))
    monkeypatch.setattr(train_mod.launch, "connect", lambda port: object())
    monkeypatch.setattr(train_mod, "Connection", lambda sock, logger=None: _StubConnection())

    factory = _live_connection_factory_for_port(cfg, 50000, role="train")
    factory()  # first launch
    factory()  # relaunch (reconnect / respawn)
    factory()  # another relaunch

    assert len(log_paths) == 3
    assert len(set(str(p) for p in log_paths)) == 3  # all DISTINCT -> no truncation


def test_policy_kwargs_carry_extractor_and_checkpoint(tmp_path):
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, encoder_checkpoint=ckpt, freeze_encoder=True)
    pk = _build_policy_kwargs(cfg)
    assert pk["features_extractor_class"] is EncoderExtractor
    assert pk["features_extractor_kwargs"]["checkpoint"] == ckpt
    assert pk["features_extractor_kwargs"]["freeze"] is True


# --- 2b. rollout-buffer memory guard (pure; available_bytes INJECTED) ------------------------


def test_frame_nbytes_is_element_product():
    # uint8 = 1 byte/element, so the byte size is just the element product.
    assert frame_nbytes((360, 640, 3)) == 360 * 640 * 3
    assert frame_nbytes((4, 6, 3)) == 72


def test_estimate_rl_memory_bytes_formula():
    # n_steps x n_envs x frame_nbytes x frame_stack (uint8 rollout buffer).
    est = estimate_rl_memory_bytes(n_steps=512, n_envs=7, frame_nbytes=691_200, frame_stack=1)
    assert est == 512 * 7 * 691_200
    # frame_stack multiplies the stored channel depth.
    est4 = estimate_rl_memory_bytes(n_steps=512, n_envs=7, frame_nbytes=691_200, frame_stack=4)
    assert est4 == 512 * 7 * 691_200 * 4


def test_check_rl_memory_budget_passes_within_budget():
    # A tiny buffer with plenty of RAM -> returns the estimate line, no raise.
    line = check_rl_memory_budget(
        n_steps=128,
        n_envs=1,
        frame_nbytes=72,
        frame_stack=1,
        available_bytes=64 * 1024**3,  # 64 GB available
    )
    assert "memory estimate" in line
    assert "rollout buffer" in line


def test_check_rl_memory_budget_aborts_when_over_budget():
    # 7 envs x 2048 steps x 0.69 MB/frame ~ 9.2 GB buffer + ~8 GB Unity; on a small box this blows
    # the 60% budget -> MemoryError with actionable guidance.
    with pytest.raises(MemoryError, match="ABORT"):
        check_rl_memory_budget(
            n_steps=2048,
            n_envs=7,
            frame_nbytes=691_200,
            frame_stack=1,
            available_bytes=8 * 1024**3,  # only 8 GB available
        )


def test_check_rl_memory_budget_allow_oversized_overrides_abort():
    # Same oversized config, but allow_oversized=True -> returns the line, NO raise.
    line = check_rl_memory_budget(
        n_steps=2048,
        n_envs=7,
        frame_nbytes=691_200,
        frame_stack=1,
        available_bytes=8 * 1024**3,
        allow_oversized=True,
    )
    assert "memory estimate" in line


def test_check_rl_memory_budget_includes_unity_allowance():
    # The guard folds n_envs Unity instances into the total (NOT n_envs + 1 — the eval and training
    # SETS never coexist, so peak = max(N_train, M_eval) = n_envs). A buffer that fits alone but,
    # with the Unity allowance, exceeds the budget must ABORT.
    n_envs = 3
    buffer = estimate_rl_memory_bytes(
        n_steps=10, n_envs=n_envs, frame_nbytes=691_200, frame_stack=1
    )
    unity = n_envs * UNITY_INSTANCE_BYTES
    # available chosen so threshold (0.6*available) is above buffer but below buffer+unity.
    threshold_target = buffer + unity // 2
    available = int(threshold_target / MEMORY_MARGIN)
    assert int(available * MEMORY_MARGIN) >= buffer  # buffer alone would pass
    with pytest.raises(MemoryError):
        check_rl_memory_budget(
            n_steps=10,
            n_envs=n_envs,
            frame_nbytes=691_200,
            frame_stack=1,
            available_bytes=available,
        )


def test_check_rl_memory_budget_peak_is_n_envs_not_n_envs_plus_one():
    # The peak-instances formula change: a config sized to fit under n_envs Unity instances but NOT
    # under the OLD n_envs + 1 must now be ACCEPTED (the eval set never coexists with training).
    n_envs = 4
    buffer = estimate_rl_memory_bytes(n_steps=8, n_envs=n_envs, frame_nbytes=691_200, frame_stack=1)
    new_total = buffer + n_envs * UNITY_INSTANCE_BYTES  # current (max(N, M)) accounting
    old_total = buffer + (n_envs + 1) * UNITY_INSTANCE_BYTES  # the rejected-under accounting
    # Pick available so 0.6*available sits BETWEEN new_total and old_total: fits now, blew before.
    threshold_target = (new_total + old_total) // 2
    available = int(threshold_target / MEMORY_MARGIN)
    assert new_total <= int(available * MEMORY_MARGIN) < old_total
    # Now accepted (no raise) — the over-rejection under n_envs + 1 is gone.
    line = check_rl_memory_budget(
        n_steps=8, n_envs=n_envs, frame_nbytes=691_200, frame_stack=1, available_bytes=available
    )
    assert "memory estimate" in line
    # The printed estimate names n_envs (not n_envs + 1) Unity instances.
    assert f"{n_envs} Unity instance" in line


# --- 2c. sidecar cadence rides num_timesteps in lockstep with CheckpointCallback -------------


def test_checkpoint_save_freq_divides_by_n_envs():
    # checkpoint_freq is env-steps; the per-call save_freq is max(freq // n_envs, 1) so a checkpoint
    # lands every checkpoint_freq num_timesteps (num_timesteps = n_calls * n_envs).
    assert _checkpoint_save_freq(_cfg_n_envs(1)) == 10_000
    assert _checkpoint_save_freq(_cfg_n_envs(7, checkpoint_freq=7_000)) == 1_000
    # never below 1 (a freq smaller than n_envs floors to 1 call).
    assert _checkpoint_save_freq(_cfg_n_envs(7, checkpoint_freq=3)) == 1


def _cfg_n_envs(n_envs, **overrides):
    """A TrainConfig with n_envs set (game_port spread so eval avoids the training range)."""
    base = {
        "total_timesteps": 1_000,
        "game_config": Path("cfg.json"),
        "run_dir": Path("run"),
        "frame_shape": (4, 6, 3),
        "n_envs": n_envs,
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_sidecar_cadence_matches_checkpoint_at_multi_env(tmp_path):
    # At n_envs>1 the sidecar must fire on the SAME calls as SB3's CheckpointCallback. Both ride
    # the SAME transformed save_freq (max(checkpoint_freq // n_envs, 1)) and gate on
    # n_calls % save_freq, so state.json lands in lockstep with each model_<num_timesteps>.zip. We
    # mirror SB3's call bookkeeping (n_calls increments per env.step(); num_timesteps =
    # n_calls * n_envs) and assert the two gates fire on identical calls.
    from stable_baselines3.common.callbacks import CheckpointCallback

    n_envs = 7
    cfg = _cfg(tmp_path, n_envs=n_envs, game_port=54000, checkpoint_freq=7_000)
    save_freq = _checkpoint_save_freq(cfg)  # 7000 // 7 = 1000
    assert save_freq == 1_000

    provider = OpponentProvider.from_roster(cfg.opponents, cfg.opponent_strategy, seed=cfg.seed)
    sidecar_cb = _make_sidecar_callback(
        cfg, provider, _initial_elo(cfg.opponents), save_freq=save_freq
    )
    ckpt_cb = CheckpointCallback(save_freq=save_freq, save_path=str(tmp_path), name_prefix="model")

    # Both gate on n_calls % save_freq == 0. Walk a range of calls; the fire-sets must be identical.
    sidecar_fires = {c for c in range(1, 4_000) if c % sidecar_cb.save_freq == 0}
    ckpt_fires = {c for c in range(1, 4_000) if c % ckpt_cb.save_freq == 0}
    assert sidecar_fires == ckpt_fires
    # and a fire lands exactly at num_timesteps = save_freq * n_envs = checkpoint_freq (lockstep).
    first_fire_call = min(sidecar_fires)
    assert first_fire_call * n_envs == 7_000


def test_sidecar_save_with_none_provider_records_strategy_and_seed(tmp_path):
    # At n_envs>1 there is no live provider (per-subproc); save_sidecar(provider=None) records the
    # strategy + seed from cfg so the sidecar still fully describes the run.
    cfg = _cfg(tmp_path, opponents=("noop", "random"), opponent_strategy="uniform", seed=11)
    path = tmp_path / "state.json"
    save_sidecar(path, provider=None, elo=_initial_elo(cfg.opponents), cfg=cfg, num_timesteps=512)
    loaded = load_sidecar(path)
    assert loaded["provider"]["strategy"] == "uniform"
    assert loaded["provider"]["seed"] == 11
    assert "index" not in loaded["provider"]


# --- 3. sidecar save -> load round-trip ------------------------------------------------------


def test_sidecar_roundtrip_round_robin(tmp_path):
    cfg = _cfg(tmp_path, opponents=("noop", "random", "wall-hugger"))
    provider = OpponentProvider.from_roster(cfg.opponents, "round_robin", seed=cfg.seed)
    # advance the round-robin position so a non-zero index round-trips.
    provider.sample()
    provider.sample()
    assert provider._index == 2

    elo = {"noop": 1012.0, "random": 988.0, "wall-hugger": 1000.0}
    path = tmp_path / "state.json"
    save_sidecar(path, provider=provider, elo=elo, cfg=cfg, num_timesteps=4096)

    loaded = load_sidecar(path)
    assert loaded["num_timesteps"] == 4096
    assert loaded["provider"]["strategy"] == "round_robin"
    assert loaded["provider"]["index"] == 2
    assert loaded["elo"] == elo
    assert loaded["config"] == cfg.to_dict()


def test_sidecar_uniform_records_strategy_only(tmp_path):
    # uniform has no replayable position: only strategy is recorded (no index).
    cfg = _cfg(tmp_path, opponents=("noop", "random"), opponent_strategy="uniform")
    provider = OpponentProvider.from_roster(cfg.opponents, "uniform", seed=cfg.seed)
    provider.sample()
    path = tmp_path / "state.json"
    save_sidecar(path, provider=provider, elo=_initial_elo(cfg.opponents), cfg=cfg, num_timesteps=8)

    loaded = load_sidecar(path)
    assert loaded["provider"]["strategy"] == "uniform"
    assert "index" not in loaded["provider"]


def test_initial_elo_seeds_base_rating():
    elo = _initial_elo(("noop", "random"))
    assert elo == {"noop": BASE_ELO, "random": BASE_ELO}


# --- 4. resume: latest checkpoint + sidecar restore ------------------------------------------


def test_latest_checkpoint_picks_max_step(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    # CheckpointCallback names files model_<steps>_steps.zip.
    for steps in (1024, 8192, 2048):
        (run / f"model_{steps}_steps.zip").write_bytes(b"")
    latest = _latest_checkpoint(run)
    assert latest is not None
    assert latest.name == "model_8192_steps.zip"


def test_latest_checkpoint_none_when_empty(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert _latest_checkpoint(run) is None
    # a non-existent dir is also None (not an error).
    assert _latest_checkpoint(tmp_path / "nope") is None


def test_restore_provider_position_round_robin(tmp_path):
    cfg = _cfg(tmp_path, opponents=("noop", "random", "wall-hugger"))
    provider = OpponentProvider.from_roster(cfg.opponents, "round_robin", seed=cfg.seed)
    assert provider._index == 0
    sidecar = {"provider": {"strategy": "round_robin", "index": 5}, "elo": {}}
    _restore_provider_position(provider, sidecar)
    assert provider._index == 5


def test_restore_provider_position_uniform_noop(tmp_path):
    cfg = _cfg(tmp_path, opponents=("noop", "random"), opponent_strategy="uniform")
    provider = OpponentProvider.from_roster(cfg.opponents, "uniform", seed=cfg.seed)
    # uniform has no replayable index; restore is a no-op (no crash, _index untouched).
    _restore_provider_position(provider, {"provider": {"strategy": "uniform"}, "elo": {}})
    assert provider._index == 0


def test_resume_restores_provider_and_elo_with_mocked_ppo_load(tmp_path, monkeypatch):
    """The resume path loads the latest zip (mocked) and restores provider position + ELO.

    Mocks ``PPO.load`` so no real model is touched. Asserts: the latest checkpoint is selected, the
    provider ``_index`` is restored from the sidecar, the ELO dict is restored, and ``model.learn``
    is invoked with ``reset_num_timesteps=False``. The vec env is stubbed (no Unity).
    """
    import pop_trainer.rl.train as train_mod

    cfg = _cfg(
        tmp_path,
        opponents=("noop", "random", "wall-hugger"),
        resume=tmp_path / "run",
        total_timesteps=10,
        checkpoint_freq=5,
        eval_freq=0,
        eval_episodes=1,
    )

    # Lay down a prior run: a checkpoint zip + a sidecar with a non-zero index and custom ELO.
    run = tmp_path / "run"
    run.mkdir()
    (run / "model_2048_steps.zip").write_bytes(b"")
    provider_for_save = OpponentProvider.from_roster(cfg.opponents, "round_robin", seed=cfg.seed)
    provider_for_save._index = 7
    saved_elo = {"noop": 1010.0, "random": 990.0, "wall-hugger": 1005.0}
    save_sidecar(
        run / "state.json",
        provider=provider_for_save,
        elo=saved_elo,
        cfg=cfg,
        num_timesteps=2048,
    )

    # The vec env build is replaced with a stub-backed real stack (no Unity). train_local now
    # builds TWO vec envs (training on game_port, eval on eval_port) and passes port + monitor
    # kwargs; the stub factory serves BOTH ports (no Unity either way).
    def _stub_build_vec_env(c, *, port=None, ports=None, monitor=False, single=False, role="train"):
        # Serve BOTH the single path (connection_factory) and the multi-env eval block path
        # (connection_factory_for_port) with stubs so NO Unity launches on either branch.
        return build_vec_env(
            c,
            port=port,
            ports=ports,
            monitor=monitor,
            single=single,
            role=role,
            connection_factory=_stub_factory,
            connection_factory_for_port=_stub_factory_for_port,
        )

    monkeypatch.setattr(train_mod, "build_vec_env", _stub_build_vec_env)

    captured: dict = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.num_timesteps = 2048
            self.env = None  # set to the training vec by _fake_load (callback reads model.env)

        def learn(self, total_timesteps, callback=None, reset_num_timesteps=True):
            captured["reset_num_timesteps"] = reset_num_timesteps
            # surface the restored provider position + ELO at learn time.
            wrapper = train_mod._find_selfplay_wrapper(self._vec)
            captured["provider_index"] = wrapper.opponents._index

        def predict(self, obs, deterministic=False):
            import numpy as np

            return np.zeros(5, dtype=np.float32), None

    fake_model = _FakeModel()

    def _fake_load(path, env=None):
        captured["loaded_path"] = Path(path)
        fake_model._vec = env
        fake_model.env = env  # the callback is constructed with training_vec=model.env
        return fake_model

    # Patch PPO.load + the final-eval helper (we are not running a real eval here).
    from stable_baselines3 import PPO

    monkeypatch.setattr(PPO, "load", staticmethod(_fake_load))
    # train_local imports evaluate_winrate locally from its source module, so patch the SOURCE.
    import pop_trainer.rl.evaluate as eval_mod

    monkeypatch.setattr(
        eval_mod, "evaluate_winrate", lambda *a, **k: dict.fromkeys(cfg.opponents, 0.0)
    )

    # capture the ELO at the final save (post-restore).
    real_save = train_mod.save_sidecar
    final_elo: dict = {}

    def _capture_save(path, *, provider, elo, cfg, num_timesteps):
        final_elo.clear()
        final_elo.update(elo)
        real_save(path, provider=provider, elo=elo, cfg=cfg, num_timesteps=num_timesteps)

    monkeypatch.setattr(train_mod, "save_sidecar", _capture_save)

    out = train_mod.train_local(cfg)

    assert out == cfg.run_dir
    assert captured["loaded_path"].name == "model_2048_steps.zip"  # max-step zip selected
    assert captured["reset_num_timesteps"] is False  # resume continues, not resets
    assert captured["provider_index"] == 7  # provider position restored from sidecar
    # ELO restored from the sidecar (then possibly eval-nudged; the base values came from disk).
    assert set(final_elo) == set(cfg.opponents)
    # the restored ratings (not the BASE_ELO default) seeded the final ELO.
    assert final_elo["noop"] != BASE_ELO or final_elo["random"] != BASE_ELO


# --- 5. CLI: _parse_args + main namespace -> TrainConfig mapping ------------------------------


def _capture_cfg(monkeypatch) -> dict:
    """Patch ``train_local`` to capture the TrainConfig it receives (no training runs).

    Returns a dict that gets ``["cfg"]`` set to the captured config when ``main`` reaches
    ``train_local``. The lambda returns ``cfg.run_dir`` so ``main`` still yields a Path.
    """
    import pop_trainer.rl.train as train_mod

    captured: dict = {}

    def _fake_train_local(cfg):
        captured["cfg"] = cfg
        return cfg.run_dir

    monkeypatch.setattr(train_mod, "train_local", _fake_train_local)
    return captured


def test_parse_args_opponents_single():
    args = _parse_args(["--total-timesteps", "1000", "--run-dir", "out", "--opponents", "noop"])
    assert args.opponents == ("noop",)


def test_parse_args_opponents_multiple_in_order():
    args = _parse_args(
        [
            "--total-timesteps",
            "1000",
            "--run-dir",
            "out",
            "--opponents",
            "noop,random,aggressive-coverage",
        ]
    )
    assert args.opponents == ("noop", "random", "aggressive-coverage")


def test_parse_args_opponents_unknown_exits_listing_valid(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse_args(["--total-timesteps", "1000", "--run-dir", "out", "--opponents", "noop,bogus"])
    assert exc.value.code == 2  # argparse parser.error exits 2
    err = capsys.readouterr().err
    assert "bogus" in err  # names the bad selector
    # lists the valid selectors (every real one appears in the message)
    for sel in AGENT_SELECTORS:
        assert sel in err


def test_main_threads_new_flags_into_config(tmp_path, monkeypatch):
    captured = _capture_cfg(monkeypatch)
    out = main(
        [
            "--total-timesteps",
            "1000",
            "--run-dir",
            str(tmp_path),
            "--opponents",
            "noop,random",
            "--opponent-strategy",
            "uniform",
            "--eval-freq",
            "500",
            "--eval-episodes",
            "3",
            "--checkpoint-freq",
            "250",
            "--n-envs",
            "7",
            "--n-steps",
            "512",
            "--allow-oversized",
            "--port",
            "51234",
            "--eval-port",
            "51299",
        ]
    )
    cfg = captured["cfg"]
    assert out == cfg.run_dir
    assert cfg.opponents == ("noop", "random")
    assert cfg.opponent_strategy == "uniform"
    assert cfg.eval_freq == 500
    assert cfg.eval_episodes == 3
    assert cfg.checkpoint_freq == 250
    assert cfg.n_envs == 7
    assert cfg.n_steps == 512
    assert cfg.allow_oversized is True
    assert cfg.game_port == 51234
    assert cfg.eval_port == 51299
    assert cfg.effective_eval_port == 51299


def test_main_defaults_preserved_when_flags_omitted(tmp_path, monkeypatch):
    captured = _capture_cfg(monkeypatch)
    main(["--total-timesteps", "1000", "--run-dir", str(tmp_path)])
    cfg = captured["cfg"]
    # --opponents omitted -> the dataclass DEFAULT_ROSTER default owns it.
    assert cfg.opponents == DEFAULT_ROSTER
    assert cfg.opponent_strategy == "round_robin"
    assert cfg.eval_freq == 10_000
    assert cfg.eval_episodes == 10
    assert cfg.checkpoint_freq == 10_000
    assert cfg.n_envs == 1
    assert cfg.n_steps == 2048  # the TrainConfig default
    assert cfg.allow_oversized is False
    assert cfg.game_port == 50000
    assert cfg.eval_port is None


def test_main_equal_ports_rejected_through_cli(tmp_path, monkeypatch):
    # The equal-port rejection lives in TrainConfig.__post_init__; threading --port/--eval-port
    # through must still surface it (raises before train_local is reached).
    _capture_cfg(monkeypatch)
    with pytest.raises(ValueError, match="eval_port must differ from game_port"):
        main(
            [
                "--total-timesteps",
                "1000",
                "--run-dir",
                str(tmp_path),
                "--port",
                "50000",
                "--eval-port",
                "50000",
            ]
        )
