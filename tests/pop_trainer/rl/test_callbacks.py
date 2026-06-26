"""Tests for pop_trainer.rl.callbacks — EvalWinRateCallback.

No live Unity / no real SB3 model: a FAKE model exposing the slice the callback touches
(``num_timesteps``, ``predict``, ``logger``, plus the training-env rollout sentinels ``_last_obs``
/ ``_last_episode_starts``) and a fake logger (record / dump). The eval itself is real
(``evaluate_winrate`` over a SEPARATE stub eval env's raw TankEnv stand-in), so we exercise the
whole rollout-boundary -> eval -> log path.

Coverage (the Director's crux):

1. eval runs ONLY at a rollout boundary: ``_on_step()`` returns True and does NOT eval; eval fires
   in ``_on_rollout_end`` only when due (gated by ``eval_freq``); it does NOT fire when not yet due.
2. THE CRUX (flipped): the callback uses the DEDICATED eval env and does NOT touch the training
   env — the model's ``_last_obs`` / ``_last_episode_starts`` stay the STALE sentinels after
   ``_on_rollout_end`` and ``model.get_env().reset()`` is never called for a repair.
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
    """A raw eval-env stand-in (the dedicated eval ``TankEnv``): every episode terminates WIN.

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


# Distinct sentinel the model holds before eval; it must remain unchanged after eval (the training
# env is never touched by the dedicated-eval-env design).
STALE_OBS = np.full((1, 4), 7.0, dtype=np.float32)


class FakeTrainingVecEnv:
    """The TRAINING vec env stand-in returned by ``model.get_env()``.

    ``reset()`` raises if the callback ever calls it — the dedicated-eval design must NEVER reset
    the training env for a repair. Records reset calls so the test can assert zero.
    """

    def __init__(self) -> None:
        self.num_envs = 1
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        raise AssertionError("the callback must not reset the training env (no buffer repair)")


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

    ``predict`` is greedy + inert (the eval env scripts the outcome); ``_last_obs`` /
    ``_last_episode_starts`` start STALE so the test can prove eval NEVER overwrites them.
    ``get_env()`` returns the training vec env whose ``reset()`` would raise if the callback tried a
    repair. ``logger`` is the property source the callback reads via ``self.logger``.
    """

    def __init__(self, logger) -> None:
        self._vec = FakeTrainingVecEnv()
        self.logger = logger
        self.num_timesteps = 0
        self._last_obs = STALE_OBS.copy()
        self._last_episode_starts = np.zeros((1,), dtype=bool)

    def get_env(self):
        return self._vec

    def predict(self, obs, deterministic=False):
        assert deterministic is True
        return np.zeros(5, dtype=np.float32), None


def _make(eval_freq, eval_episodes, opponents, eval_env):
    """Build a wired callback + model + logger over a SEPARATE stub eval env."""
    logger = FakeLogger()
    model = FakeModel(logger)
    cb = EvalWinRateCallback(
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        opponents=opponents,
        eval_env=eval_env,
    )
    cb.model = model
    cb.num_timesteps = model.num_timesteps
    return cb, model, logger


# --- 1. eval gates to the rollout boundary -----------------------------------------------


def test_on_step_never_evals_and_returns_true():
    eval_env = AlwaysWinEnv()
    cb, model, logger = _make(100, 1, ("noop",), eval_env)
    model.num_timesteps = cb.num_timesteps = 500  # well past eval_freq

    assert cb._on_step() is True
    # _on_step must NOT eval: no log records, no eval-env reset.
    assert logger.records == {}
    assert eval_env.reset_options == []


def test_rollout_end_does_not_eval_when_not_yet_due():
    eval_env = AlwaysWinEnv()
    cb, model, logger = _make(1000, 1, ("noop",), eval_env)
    model.num_timesteps = cb.num_timesteps = 999  # one short of eval_freq

    cb._on_rollout_end()
    # not due -> no eval, no log.
    assert logger.records == {}
    assert eval_env.reset_options == []
    # the stale obs is untouched (no eval ran).
    np.testing.assert_array_equal(model._last_obs, STALE_OBS)


def test_eval_freq_zero_disables_eval():
    eval_env = AlwaysWinEnv()
    cb, model, logger = _make(0, 1, ("noop",), eval_env)
    model.num_timesteps = cb.num_timesteps = 10_000

    cb._on_rollout_end()
    assert logger.records == {}
    assert eval_env.reset_options == []


# --- 2. the crux: training env / rollout state is NEVER touched --------------------------


def test_rollout_end_evals_on_dedicated_env_and_leaves_training_state_stale():
    eval_env = AlwaysWinEnv()
    cb, model, logger = _make(100, 2, ("noop", "random"), eval_env)
    model.num_timesteps = cb.num_timesteps = 100  # exactly due (>= eval_freq from 0)

    cb._on_rollout_end()

    # eval fired on the DEDICATED eval env (it was reset per episode).
    assert eval_env.reset_options  # eval drove resets on the eval env
    # THE CRUX (flipped): the training env was NEVER reset for a repair (its reset would raise).
    assert model._vec.reset_count == 0
    # the model's rollout sentinels stay STALE — eval never wrote them back.
    np.testing.assert_array_equal(model._last_obs, STALE_OBS)
    assert bool(model._last_episode_starts.any()) is False  # still all-False sentinel


# --- 3. per-opponent logging -------------------------------------------------------------


def test_logs_per_opponent_and_overall_win_rate_keys():
    eval_env = AlwaysWinEnv()  # every episode is a win -> every rate 1.0
    cb, model, logger = _make(100, 1, ("noop", "aggressive-coverage"), eval_env)
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
    # rotation suppression is honored by NEVER sending switch_arena on any eval reset.
    eval_env = AlwaysWinEnv()
    cb, model, logger = _make(100, 2, ("noop",), eval_env)
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # every raw eval-env reset during eval passed options=None (no switch_arena -> no desync).
    assert eval_env.reset_options  # eval did reset the eval env
    assert all(opt is None for opt in eval_env.reset_options)
