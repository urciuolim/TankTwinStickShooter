"""Tests for pop_trainer.rl.callbacks — EvalWinRateCallback.

No live Unity / no real SB3 model: a FAKE model exposing the slice the callback touches
(``get_env()`` -> a fake DummyVecEnv-like, ``_last_obs`` / ``_last_episode_starts``,
``num_timesteps``, ``predict``, ``logger``) and a fake logger (record / dump). The eval itself is
real (``evaluate_winrate`` over the fake env's raw TankEnv stand-in), so we exercise the whole
rollout-boundary -> eval -> log -> buffer-repair path.

Coverage (the Director's crux):

1. eval runs ONLY at a rollout boundary: ``_on_step()`` returns True and does NOT eval; eval fires
   in ``_on_rollout_end`` only when due (gated by ``eval_freq``); it does NOT fire when not yet due.
2. ``_last_obs`` / ``_last_episode_starts`` are RESTORED after eval to the FRESH ``vec_env.reset()``
   obs (all-True episode starts) — NOT the stale eval-corrupted obs.
3. the per-opponent ``eval/win_rate/<sel>`` keys (and the overall ``eval/win_rate``) are logged.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from pop_trainer.core.state import STATE_LEN
from pop_trainer.rl.callbacks import EvalWinRateCallback
from pop_trainer.rl.evaluate import WIN

# --- stub env / fakes --------------------------------------------------------------------


def _state(fill: float) -> list[float]:
    return [fill + i for i in range(STATE_LEN)]


class _StubBox(gymnasium.spaces.Space):
    pass


class AlwaysWinEnv(gymnasium.Env):
    """A raw-env stand-in (the ``.unwrapped`` TankEnv): every episode terminates with a WIN.

    One step per episode, terminating with ``info["outcome"] == "win"`` -> evaluate_winrate scores
    1.0 for every opponent. ``info["state"]`` (52 floats) is present so the SelfPlayWrapper flip
    works. Records resets so a test can assert NO ``switch_arena`` option is passed during eval.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = _StubBox()
        self.observation_space = _StubBox()
        self.reset_options: list = []

    @property
    def unwrapped(self):  # already the raw env
        return self

    def reset(self, *, seed=None, options=None):
        self.reset_options.append(options)
        return ("obs", {"state": _state(0.0), "map": None})

    def step(self, action, opponent_action=None):
        info = {"state": _state(1.0), "map": None, "outcome": WIN}
        return ("obs", 0.0, True, False, info)


# Distinct sentinels so we can prove which obs the model ends up holding.
STALE_OBS = np.full((1, 4), 7.0, dtype=np.float32)
FRESH_OBS = np.full((1, 4), 99.0, dtype=np.float32)


class FakeVecEnv:
    """A DummyVecEnv-like: ``.envs == [raw_env]``, ``.num_envs``, ``.reset()`` -> FRESH_OBS.

    ``reset()`` (the vec-level repair reset) returns a DISTINCT obs from anything eval produced, so
    a test can prove the model's ``_last_obs`` is the repaired/fresh one. Records reset count.
    """

    def __init__(self, raw_env) -> None:
        self.envs = [raw_env]
        self.num_envs = 1
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return FRESH_OBS.copy()


class FakeLogger:
    """A stand-in SB3 Logger: ``record`` accumulates, ``dump`` records the dumped timesteps."""

    def __init__(self) -> None:
        self.records: dict[str, float] = {}
        self.dumps: list[int] = []

    def record(self, key, value):
        self.records[key] = value

    def dump(self, step=0):
        self.dumps.append(step)


class FakeModel:
    """The slice of an SB3 algorithm EvalWinRateCallback + evaluate_winrate touch.

    ``get_env()`` -> the FakeVecEnv; ``predict`` is greedy + inert (the env scripts the outcome);
    ``_last_obs`` / ``_last_episode_starts`` start STALE so the test can prove the repair overwrote
    them. ``logger`` is the property source the callback reads via ``self.logger``.
    """

    def __init__(self, raw_env, logger) -> None:
        self._vec = FakeVecEnv(raw_env)
        self.logger = logger
        self.num_timesteps = 0
        self._last_obs = STALE_OBS.copy()
        self._last_episode_starts = np.zeros((1,), dtype=bool)

    def get_env(self):
        return self._vec

    def predict(self, obs, deterministic=False):
        assert deterministic is True
        return np.zeros(5, dtype=np.float32), None


def _wire(callback: EvalWinRateCallback, model: FakeModel) -> None:
    """Inject model/timesteps the way SB3's init_callback would (logger is model.logger)."""
    callback.model = model
    callback.num_timesteps = model.num_timesteps


# --- 1. eval gates to the rollout boundary -----------------------------------------------


def test_on_step_never_evals_and_returns_true():
    raw = AlwaysWinEnv()
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(eval_freq=100, eval_episodes=1, opponents=("noop",))
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 500  # well past eval_freq

    assert cb._on_step() is True
    # _on_step must NOT eval: no log records, no vec reset.
    assert logger.records == {}
    assert model._vec.reset_count == 0


def test_rollout_end_does_not_eval_when_not_yet_due():
    raw = AlwaysWinEnv()
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(eval_freq=1000, eval_episodes=1, opponents=("noop",))
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 999  # one short of eval_freq

    cb._on_rollout_end()
    # not due -> no eval, no log, no repair reset.
    assert logger.records == {}
    assert model._vec.reset_count == 0
    # the stale obs is untouched (no repair ran)
    np.testing.assert_array_equal(model._last_obs, STALE_OBS)


def test_eval_freq_zero_disables_eval():
    raw = AlwaysWinEnv()
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(eval_freq=0, eval_episodes=1, opponents=("noop",))
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 10_000

    cb._on_rollout_end()
    assert logger.records == {}
    assert model._vec.reset_count == 0


# --- 2. buffer repair (the crux) ---------------------------------------------------------


def test_rollout_end_evals_when_due_and_restores_last_obs():
    raw = AlwaysWinEnv()
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(eval_freq=100, eval_episodes=2, opponents=("noop", "random"))
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 100  # exactly due (>= eval_freq from 0)

    cb._on_rollout_end()

    # eval fired: the vec env was reset exactly once for the buffer repair.
    assert model._vec.reset_count == 1
    # THE CRUX: _last_obs is the FRESH vec reset obs, NOT the stale eval-corrupted obs.
    np.testing.assert_array_equal(model._last_obs, FRESH_OBS)
    # episode starts are all-True so the NEXT rollout begins clean.
    assert model._last_episode_starts.dtype == bool
    assert model._last_episode_starts.shape == (1,)
    assert bool(model._last_episode_starts.all())


# --- 3. per-opponent logging -------------------------------------------------------------


def test_logs_per_opponent_and_overall_win_rate_keys():
    raw = AlwaysWinEnv()  # every episode is a win -> every rate 1.0
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(
        eval_freq=100, eval_episodes=1, opponents=("noop", "aggressive-coverage")
    )
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # per-opponent keys logged, one per selector, plus an overall key.
    assert "eval/win_rate/noop" in logger.records
    assert "eval/win_rate/aggressive-coverage" in logger.records
    assert "eval/win_rate" in logger.records
    # always-win env -> every rate is 1.0
    assert logger.records["eval/win_rate/noop"] == 1.0
    assert logger.records["eval/win_rate/aggressive-coverage"] == 1.0
    assert logger.records["eval/win_rate"] == 1.0
    # the row was dumped at this timestep
    assert logger.dumps == [100]


def test_eval_resets_pass_no_switch_arena_option():
    # (B): rotation suppression is honored by NEVER sending switch_arena on any eval reset.
    raw = AlwaysWinEnv()
    logger = FakeLogger()
    model = FakeModel(raw, logger)
    cb = EvalWinRateCallback(eval_freq=100, eval_episodes=2, opponents=("noop",))
    _wire(cb, model)
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # every raw-env reset during eval passed options=None (no switch_arena -> no desync).
    assert raw.reset_options  # eval did reset the raw env
    assert all(opt is None for opt in raw.reset_options)
