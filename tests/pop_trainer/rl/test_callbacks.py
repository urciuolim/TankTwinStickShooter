"""Tests for pop_trainer.rl.callbacks — EvalWinRateCallback (teardown -> eval -> respawn).

No live Unity / no real SB3 model. The callback now time-multiplexes the training and eval Unity
instances around each eval so they NEVER coexist:

    training_vec.env_method("release")  ->  parallel eval over the eval vec  ->
    eval_vec.env_method("release")      ->  training_vec.reset() + re-sync model._last_obs

The fakes assert that order, the no-coexist INVARIANT (a shared "live instances" ledger that never
holds train + eval at once), the rollout-state re-sync, and the per-opponent logging. The eval
itself is REAL (``evaluate_winrate`` over a real ``DummyVecEnv`` of scripted stub envs).

Coverage:

1. eval gates to the rollout boundary: ``_on_step`` returns True without evaluating; eval fires in
   ``_on_rollout_end`` only when due (``eval_freq``); not when not-yet-due / when disabled.
2. THE CYCLE: training released BEFORE eval instances spawn; eval released AFTER eval; training
   reset AFTER eval; ``model._last_obs`` re-synced to the post-respawn obs; ``_last_episode_starts``
   reset to all-True.
3. THE INVARIANT: no eval instance coexists with a training instance (shared ledger).
4. per-opponent ``eval/win_rate/<sel>`` keys (and overall ``eval/win_rate``) are logged + dumped.
"""

from __future__ import annotations

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from pop_trainer.core.state import STATE_LEN
from pop_trainer.rl.callbacks import EvalWinRateCallback
from pop_trainer.rl.evaluate import WIN
from pop_trainer.rl.selfplay import OpponentProvider, SelfPlayWrapper

# --- shared live-instance ledger + event trace -------------------------------------------


def _state(fill: float) -> list[float]:
    return [fill + i for i in range(STATE_LEN)]


_OBS_SPACE = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
_ACTION_SPACE = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)


class _EvalStubEnv(gymnasium.Env):
    """A raw eval-env stand-in: every episode wins. Touches a shared ledger so the invariant is
    assertable — its instance goes LIVE on reset and is removed by ``release``; while live it
    asserts NO training instance is also live.
    """

    def __init__(self, ledger: set, trace: list, name: str) -> None:
        super().__init__()
        self._ledger = ledger
        self._trace = trace
        self._name = name
        self.action_space = _ACTION_SPACE
        self.observation_space = _OBS_SPACE
        self.reset_options: list = []

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        self.reset_options.append(options)
        # Going live as an eval instance: a training instance must NOT be live right now.
        assert "train" not in self._ledger, "eval instance came up while training was still live"
        self._ledger.add("eval")
        return (np.zeros((1,), dtype=np.float32), {"state": _state(0.0), "map": None})

    def step(self, action, opponent_action=None):
        assert "train" not in self._ledger, "eval stepped while training was live"
        info = {"state": _state(1.0), "map": None, "outcome": WIN}
        return (np.zeros((1,), dtype=np.float32), 0.0, True, False, info)

    def release(self) -> None:
        self._trace.append(("eval_release", self._name))
        self._ledger.discard("eval")


def _make_eval_vec(ledger: set, trace: list, m: int) -> DummyVecEnv:
    """A real DummyVecEnv of M SelfPlayWrapper(_EvalStubEnv) — evaluate_winrate drives it."""
    placeholder = OpponentProvider.from_roster(["noop"], strategy="round_robin")

    def _fn(i):
        return lambda: SelfPlayWrapper(_EvalStubEnv(ledger, trace, f"eval{i}"), placeholder)

    return DummyVecEnv([_fn(i) for i in range(m)])


class FakeTrainingVec:
    """The TRAINING vec env handle (model.env stand-in). ``env_method('release')`` tears the
    training instances down (ledger -= 'train'); ``reset()`` respawns them (ledger += 'train') and
    returns a fresh obs. All calls are traced so the ORDER is assertable.
    """

    def __init__(self, ledger: set, trace: list, respawn_obs) -> None:
        self.num_envs = 1
        self._ledger = ledger
        self._trace = trace
        self._respawn_obs = respawn_obs
        self.reset_count = 0
        # Training comes up live at construction (mirrors model.learn launching the training set).
        ledger.add("train")

    def env_method(self, method_name, *args, **kwargs):
        self._trace.append(("train_env_method", method_name))
        if method_name == "release":
            self._ledger.discard("train")
        return [None] * self.num_envs

    def reset(self):
        self._trace.append(("train_reset", None))
        # Respawn: a training instance must NOT come up while an eval instance is still live.
        assert "eval" not in self._ledger, "training respawned while eval was still live"
        self._ledger.add("train")
        self.reset_count += 1
        return self._respawn_obs


# Distinct sentinels the model holds before eval; they must be overwritten by the respawn re-sync.
STALE_OBS = np.full((1, 4), 7.0, dtype=np.float32)
RESPAWN_OBS = np.full((1, 4), 3.0, dtype=np.float32)


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
    """The slice of an SB3 algorithm the callback + evaluate_winrate touch.

    ``predict`` is greedy + returns a per-lane batched action (the eval env scripts the outcome).
    ``_last_obs`` / ``_last_episode_starts`` start STALE so the test can prove the respawn re-syncs
    them. ``env`` is the training vec handle (passed to the callback as ``training_vec``).
    """

    def __init__(self, logger, training_vec) -> None:
        self.env = training_vec
        self.logger = logger
        self.num_timesteps = 0
        self._last_obs = STALE_OBS.copy()
        self._last_episode_starts = np.zeros((1,), dtype=bool)

    def predict(self, obs, deterministic=False):
        assert deterministic is True
        batch = int(np.asarray(obs).shape[0])
        return np.zeros((batch, 5), dtype=np.float32), None


def _make(eval_freq, eval_episodes, opponents, m=1):
    """Build a wired callback + model + logger + shared ledger/trace over an M-wide eval vec."""
    ledger: set = set()
    trace: list = []
    training_vec = FakeTrainingVec(ledger, trace, RESPAWN_OBS)
    eval_vec = _make_eval_vec(ledger, trace, m)
    logger = FakeLogger()
    model = FakeModel(logger, training_vec)
    cb = EvalWinRateCallback(
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        opponents=opponents,
        eval_env=eval_vec,
        training_vec=training_vec,
    )
    cb.model = model
    cb.num_timesteps = model.num_timesteps
    return cb, model, logger, training_vec, eval_vec, ledger, trace


# --- 1. eval gates to the rollout boundary -----------------------------------------------


def test_on_step_never_evals_and_returns_true():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 1, ("noop",))
    model.num_timesteps = cb.num_timesteps = 500  # well past eval_freq

    assert cb._on_step() is True
    # _on_step must NOT eval: no log records, no teardown/respawn calls.
    assert logger.records == {}
    assert trace == []
    assert training_vec.reset_count == 0


def test_rollout_end_does_not_eval_when_not_yet_due():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(1000, 1, ("noop",))
    model.num_timesteps = cb.num_timesteps = 999  # one short of eval_freq

    cb._on_rollout_end()
    assert logger.records == {}
    assert trace == []
    np.testing.assert_array_equal(model._last_obs, STALE_OBS)


def test_eval_freq_zero_disables_eval():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(0, 1, ("noop",))
    model.num_timesteps = cb.num_timesteps = 10_000

    cb._on_rollout_end()
    assert logger.records == {}
    assert trace == []


# --- 2. the cycle: teardown -> eval -> respawn + re-sync ---------------------------------


def test_rollout_end_runs_teardown_eval_respawn_in_order():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 2, ("noop", "random"))
    model.num_timesteps = cb.num_timesteps = 100  # exactly due

    cb._on_rollout_end()

    # The first action is the training teardown; it precedes ANY eval activity (eval reset/step).
    assert trace[0] == ("train_env_method", "release")
    # The training reset (respawn) happens, and comes AFTER the eval releases.
    first_train_reset = next(i for i, e in enumerate(trace) if e == ("train_reset", None))
    eval_releases = [i for i, e in enumerate(trace) if e[0] == "eval_release"]
    assert eval_releases, "eval instances must be released"
    assert max(eval_releases) < first_train_reset  # all eval releases precede the training respawn
    # the training env was respawned exactly once.
    assert training_vec.reset_count == 1


def test_rollout_end_resyncs_last_obs_and_episode_starts():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 1, ("noop",))
    model.num_timesteps = cb.num_timesteps = 100
    model._last_episode_starts[:] = False  # make the all-True reset observable

    cb._on_rollout_end()

    # _last_obs is overwritten with the fresh post-respawn obs (no longer the stale sentinel).
    np.testing.assert_array_equal(model._last_obs, RESPAWN_OBS)
    # _last_episode_starts reset to all-True (the abandoned in-flight episodes are truncated).
    assert bool(model._last_episode_starts.all()) is True


# --- 3. the no-coexist INVARIANT ---------------------------------------------------------


def test_no_eval_instance_coexists_with_a_training_instance():
    # The ledger asserts inside _EvalStubEnv.reset/step (eval up while train live -> AssertionError)
    # and inside FakeTrainingVec.reset (train up while eval live -> AssertionError). A clean cycle
    # leaves only the respawned training instance live.
    cb, model, logger, training_vec, eval_vec, ledger, trace = _make(
        100, 3, ("noop", "random"), m=3
    )
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()  # would raise if the two sets ever overlapped

    assert ledger == {"train"}  # ends with only training live (respawned), eval all torn down


def test_invariant_holds_at_multi_env_eval_width():
    # M>1 eval width: the M eval instances still never coexist with training.
    cb, model, logger, training_vec, eval_vec, ledger, trace = _make(100, 8, ("noop",), m=4)
    model.num_timesteps = cb.num_timesteps = 100
    cb._on_rollout_end()
    assert ledger == {"train"}


# --- 4. per-opponent logging -------------------------------------------------------------


def test_logs_per_opponent_and_overall_win_rate_keys():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(
        100, 1, ("noop", "aggressive-coverage")
    )
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    assert "eval/win_rate/noop" in logger.records
    assert "eval/win_rate/aggressive-coverage" in logger.records
    assert "eval/win_rate" in logger.records
    # always-win eval env -> every rate is 1.0
    assert logger.records["eval/win_rate/noop"] == 1.0
    assert logger.records["eval/win_rate/aggressive-coverage"] == 1.0
    assert logger.records["eval/win_rate"] == 1.0
    # the row was dumped at this timestep
    assert logger.dumps == [100]


def test_eval_resets_pass_no_switch_arena_option():
    # rotation suppression: NEVER send switch_arena on any eval reset.
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 2, ("noop",))
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # every raw eval-env reset during eval passed options=None (no switch_arena -> no desync).
    raw_envs = [w.env for w in eval_vec.envs]
    assert any(e.reset_options for e in raw_envs)  # eval did reset the eval envs
    assert all(opt is None for e in raw_envs for opt in e.reset_options)
