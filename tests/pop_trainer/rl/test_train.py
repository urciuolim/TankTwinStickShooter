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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from pop_trainer.core.protocol import Connection
from pop_trainer.env.tank_env import TankEnv
from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import DEFAULT_ROSTER, OpponentProvider, SelfPlayWrapper
from pop_trainer.rl.train import (
    BASE_ELO,
    TrainConfig,
    _build_policy_kwargs,
    _initial_elo,
    _latest_checkpoint,
    _restore_provider_position,
    build_vec_env,
    load_sidecar,
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
    # the cadence defaults are small enough that a few-K-step smoke triggers a checkpoint + eval.
    assert cfg.checkpoint_freq > 0
    assert cfg.eval_freq >= 0


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
    # None encoder_checkpoint / resume / build_path serialize as null
    none_cfg = _cfg(tmp_path)
    nd = none_cfg.to_dict()
    assert nd["encoder_checkpoint"] is None
    assert nd["resume"] is None
    assert nd["build_path"] is None
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

    # The vec env build is replaced with a stub-backed real stack (no Unity).
    monkeypatch.setattr(
        train_mod,
        "build_vec_env",
        lambda c: build_vec_env(c, connection_factory=_stub_factory),
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
