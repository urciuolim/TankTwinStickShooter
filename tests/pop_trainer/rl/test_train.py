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
    TrainConfig,
    _attach_launch_proc,
    _build_base_env,
    _build_policy_kwargs,
    _find_selfplay_wrapper,
    _initial_elo,
    _latest_checkpoint,
    _parse_args,
    _restore_provider_position,
    build_vec_env,
    load_sidecar,
    main,
    save_sidecar,
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


def test_eval_port_explicit_override(tmp_path):
    cfg = _cfg(tmp_path, game_port=50000, eval_port=50007)
    assert cfg.eval_port == 50007
    assert cfg.effective_eval_port == 50007  # explicit override wins over game_port + 1


def test_eval_port_equal_to_game_port_rejected(tmp_path):
    with pytest.raises(ValueError, match="eval_port must differ from game_port"):
        _cfg(tmp_path, game_port=50000, eval_port=50000)


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


def test_build_vec_env_rejects_multi_env_seam(tmp_path):
    cfg = _cfg(tmp_path, n_envs=2)
    with pytest.raises(NotImplementedError):
        build_vec_env(cfg, connection_factory=_stub_factory)


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


# --- reconnect-safe close-and-reap -----------------------------------------------------------


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


def test_close_reaps_current_connection_proc_not_original(tmp_path):
    # CHANGE 3: a mid-run reconnect swaps in a NEW Connection/_launch_proc; close must reap the
    # CURRENT conn's proc, not the originally stashed one.
    cfg = _cfg(tmp_path)
    env = _build_base_env(cfg, _stub_factory, cfg.game_port)

    original_proc = _FakeProc()
    _attach_launch_proc(env, original_proc)  # wires close-and-reap, stashing the original

    # Simulate a reconnect: the env now holds a fresh Connection whose _launch_proc is DIFFERENT.
    current_proc = _FakeProc()
    env.conn._launch_proc = current_proc  # type: ignore[attr-defined]

    env.close()

    assert current_proc.terminated  # the CURRENT connection's proc was reaped
    assert not original_proc.terminated  # the stale original was NOT reaped


def test_close_falls_back_to_original_proc_when_conn_has_none(tmp_path):
    # When the current conn carries no _launch_proc, close falls back to the originally stashed one.
    cfg = _cfg(tmp_path)
    env = _build_base_env(cfg, _stub_factory, cfg.game_port)

    original_proc = _FakeProc()
    _attach_launch_proc(env, original_proc)
    # The stub conn has no _launch_proc attribute -> getattr fallback to the original.
    assert not hasattr(env.conn, "_launch_proc")

    env.close()

    assert original_proc.terminated


def test_policy_kwargs_carry_extractor_and_checkpoint(tmp_path):
    ckpt = tmp_path / "enc.pt"
    cfg = _cfg(tmp_path, encoder_checkpoint=ckpt, freeze_encoder=True)
    pk = _build_policy_kwargs(cfg)
    assert pk["features_extractor_class"] is EncoderExtractor
    assert pk["features_extractor_kwargs"]["checkpoint"] == ckpt
    assert pk["features_extractor_kwargs"]["freeze"] is True


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
    monkeypatch.setattr(
        train_mod,
        "build_vec_env",
        lambda c, *, port=None, monitor=False: build_vec_env(
            c, port=port, monitor=monitor, connection_factory=_stub_factory
        ),
    )

    captured: dict = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.num_timesteps = 2048

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
