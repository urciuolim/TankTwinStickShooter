"""Eval-callback behavior tests (Feature 3) — rollout-boundary gating + buffer repair.

FAST + FAKE: NO Unity, NO real PPO, NO GPU. We construct
:class:`tank_twin.callbacks.EvalWinRateCallback` with a tiny fake model / vec env / logger
and monkeypatch ``evaluate_winrate`` to a constant, then assert the crux behaviors:

* eval runs only at a rollout boundary that is PAST ``eval_freq`` (below the boundary -> no
  eval, no log, no env reset);
* eval is run against the UNDERLYING raw env (``vec_env.envs[0]``), not the vec wrapper;
* ``eval/win_rate`` is logged into the model's logger (lands in progress.csv + tfevents);
* AFTER eval the model's rollout state is REPAIRED: ``_last_obs`` becomes the post-reset obs
  and ``_last_episode_starts`` is all-True (so the next rollout starts clean).

These pin the correctness contract WITHOUT spinning up SB3's training loop.
"""

import numpy as np
import pytest

import tank_twin.callbacks as callbacks_mod
from tank_twin.callbacks import EvalWinRateCallback


class FakeRawEnv:
    """Stand-in for the underlying gymnasium TankEnv (identity marker for the eval target).

    Models the slice of the env the callback now touches for the rotation-desync fix:
    ``.unwrapped`` (returns self, as a bare gymnasium.Env would) and
    ``set_rotation_enabled(bool)`` (records the on/off sequence so a test can assert eval
    disabled rotation around the eval block and restored it). The real ``vec_env.envs[0]``
    may be a ``Monitor`` wrapper, so the callback calls ``raw_env.unwrapped.set_rotation_*``.
    """

    def __init__(self):
        self.rotation_enabled = True
        # Ordered log of every set_rotation_enabled(value) call (for assertions).
        self.rotation_calls = []

    @property
    def unwrapped(self):
        return self

    def set_rotation_enabled(self, enabled):
        previous = self.rotation_enabled
        self.rotation_enabled = bool(enabled)
        self.rotation_calls.append(bool(enabled))
        return previous


class FakeVecEnv:
    """A DummyVecEnv-like fake: exposes ``.envs`` (raw envs), ``.num_envs``, and ``.reset()``.

    ``reset()`` returns a fixed sentinel obs and counts its invocations so a test can assert
    the buffer-repair reset happened exactly once per due eval.
    """

    def __init__(self, raw_env, reset_obs):
        self.envs = [raw_env]
        self.num_envs = 1
        self._reset_obs = reset_obs
        self.reset_count = 0
        # Rotation-enabled state captured at the MOMENT the buffer-repair reset runs, so a
        # test can prove the repair reset happened WHILE rotation was suppressed.
        self.rotation_enabled_at_reset = None

    def reset(self):
        self.reset_count += 1
        self.rotation_enabled_at_reset = self.envs[0].unwrapped.rotation_enabled
        return self._reset_obs


class FakeLogger:
    """Records ``record(key, value)`` and ``dump(step)`` calls for assertions."""

    def __init__(self):
        self.records = []  # (key, value)
        self.dumps = []  # step ints

    def record(self, key, value):
        self.records.append((key, value))

    def dump(self, step=None):
        self.dumps.append(step)


class FakeModel:
    """Minimal SB3-model stand-in the callback touches: get_env / logger / rollout-state attrs."""

    def __init__(self, vec_env, num_timesteps):
        self._vec_env = vec_env
        self.logger = FakeLogger()
        self.num_timesteps = num_timesteps
        # Stale rollout state the repair must overwrite.
        self._last_obs = None
        self._last_episode_starts = None

    def get_env(self):
        return self._vec_env


def _make_callback(eval_freq=1000, eval_episodes=7, num_timesteps=1000, win_rate=0.42):
    """Wire a callback to fakes; monkeypatch evaluate_winrate to a constant; return parts."""
    raw_env = FakeRawEnv()
    reset_obs = np.full((1, 36, 60, 3), 7, dtype=np.uint8)
    vec_env = FakeVecEnv(raw_env, reset_obs)
    model = FakeModel(vec_env, num_timesteps)

    cb = EvalWinRateCallback(eval_freq=eval_freq, eval_episodes=eval_episodes, seed=0)
    cb.model = model
    cb.num_timesteps = num_timesteps
    return cb, model, vec_env, raw_env, reset_obs, win_rate


def _patch_eval(monkeypatch, win_rate):
    """Patch tank_twin.callbacks.evaluate_winrate to record its call + return win_rate."""
    calls = []

    def fake_eval(model, env, n_episodes, seed=None):
        # Capture whether rotation was suppressed AT THE MOMENT eval ran (the fix: eval must
        # run with rotation disabled so it never drives switch_arena at a rollout boundary).
        rotation_enabled = env.unwrapped.rotation_enabled
        calls.append(
            {
                "model": model,
                "env": env,
                "n_episodes": n_episodes,
                "seed": seed,
                "rotation_enabled": rotation_enabled,
            }
        )
        return win_rate

    monkeypatch.setattr(callbacks_mod, "evaluate_winrate", fake_eval)
    return calls


# --- eval runs at a due rollout boundary ----------------------------------------------


def test_on_rollout_end_evaluates_at_boundary(monkeypatch):
    cb, model, vec_env, raw_env, reset_obs, win_rate = _make_callback(
        eval_freq=1000, num_timesteps=1000
    )
    calls = _patch_eval(monkeypatch, win_rate)

    cb._on_rollout_end()

    # (a) evaluate_winrate was called once, against the UNDERLYING raw env (not the vec env).
    assert len(calls) == 1
    assert calls[0]["env"] is raw_env
    assert calls[0]["env"] is vec_env.envs[0]
    assert calls[0]["model"] is model
    assert calls[0]["n_episodes"] == 7


def test_eval_logs_win_rate_into_logger(monkeypatch):
    cb, model, *_rest, win_rate = _make_callback(num_timesteps=1000, win_rate=0.42)
    _patch_eval(monkeypatch, win_rate)

    cb._on_rollout_end()

    # (c) logger.record("eval/win_rate", rate) was called, then dumped at num_timesteps.
    assert ("eval/win_rate", 0.42) in model.logger.records
    assert model.logger.dumps == [1000]


def test_eval_repairs_last_obs_and_episode_starts(monkeypatch):
    cb, model, vec_env, _raw, reset_obs, win_rate = _make_callback(num_timesteps=1000)
    _patch_eval(monkeypatch, win_rate)

    cb._on_rollout_end()

    # (b) buffer repair: vec env reset once, _last_obs == the post-reset obs, starts all-True.
    assert vec_env.reset_count == 1
    assert model._last_obs is reset_obs
    assert model._last_episode_starts.dtype == bool
    assert model._last_episode_starts.shape == (vec_env.num_envs,)
    assert model._last_episode_starts.all()


# --- gating: below the eval_freq boundary, NO eval ------------------------------------


def test_no_eval_below_freq_boundary(monkeypatch):
    # num_timesteps (500) < eval_freq (1000) and last_eval starts at 0 -> NOT due.
    cb, model, vec_env, *_ = _make_callback(eval_freq=1000, num_timesteps=500)
    calls = _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()

    # (d) no eval ran, nothing logged, no env reset (the rollout buffer is untouched).
    assert calls == []
    assert model.logger.records == []
    assert vec_env.reset_count == 0
    assert model._last_obs is None  # untouched


def test_eval_freq_zero_never_evaluates(monkeypatch):
    cb, model, vec_env, *_ = _make_callback(eval_freq=0, num_timesteps=10_000)
    calls = _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()

    assert calls == []
    assert vec_env.reset_count == 0


def test_eval_gating_advances_last_eval_timestep(monkeypatch):
    # First boundary at 1000 (due) -> evals + records last_eval=1000. A second boundary still
    # at 1000 is NOT due again; a later boundary at >=2000 IS due.
    cb, model, vec_env, *_ = _make_callback(eval_freq=1000, num_timesteps=1000)
    calls = _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()  # due -> 1 eval
    assert len(calls) == 1

    cb.num_timesteps = 1500  # only 500 since last eval -> NOT due
    cb._on_rollout_end()
    assert len(calls) == 1

    cb.num_timesteps = 2000  # 1000 since last eval -> due again
    cb._on_rollout_end()
    assert len(calls) == 2


def test_on_step_is_noop_returns_true():
    cb, *_ = _make_callback()
    # _on_step must NOT eval (eval is rollout-boundary only) and must return True.
    assert cb._on_step() is True


# --- the callback reads the seed it was constructed with ------------------------------


def test_eval_passes_constructed_seed(monkeypatch):
    cb, model, *_ = _make_callback(num_timesteps=1000)
    calls = _patch_eval(monkeypatch, 0.5)
    cb._on_rollout_end()
    assert calls[0]["seed"] == 0


@pytest.mark.parametrize(
    "freq,ts,expected_calls", [(1000, 999, 0), (1000, 1000, 1), (500, 1000, 1)]
)
def test_boundary_threshold_is_inclusive(monkeypatch, freq, ts, expected_calls):
    # Gate is `num_timesteps - last_eval >= eval_freq` (inclusive at exactly eval_freq).
    cb, model, vec_env, *_ = _make_callback(eval_freq=freq, num_timesteps=ts)
    calls = _patch_eval(monkeypatch, 0.5)
    cb._on_rollout_end()
    assert len(calls) == expected_calls


# --- rotation-desync fix: eval suppresses map rotation around the WHOLE eval block --------
#
# LIVE BUG (python/training-features): with map_rotation set, the eval callback's reset()
# (and the post-eval buffer-repair reset) would send switch_arena at a ROLLOUT BOUNDARY,
# where the Unity build is still mid-game (ingame). The build only handles switch_arena
# while !ingame, so the env read a stale `state` instead of the {"arena_switched":true} ack
# and crashed (RuntimeError), and the build flooded PlayerController.GetInput NREs. The fix:
# the callback wraps eval + the repair reset in set_rotation_enabled(False)..restore, so eval
# pins to the CURRENT arena (plain restart/start, proven safe at the boundary) and consumes
# no rotation slots. These pin that contract on the fake interface (they FAIL against the old
# callback, which never touched rotation).


def test_eval_runs_with_rotation_suppressed(monkeypatch):
    # The crux: at eval time rotation MUST be disabled (so evaluate_winrate's reset() does the
    # plain restart/start handshake on the current arena, not a switch_arena that desyncs).
    cb, model, vec_env, raw_env, *_ = _make_callback(num_timesteps=1000)
    calls = _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()

    assert len(calls) == 1
    assert calls[0]["rotation_enabled"] is False  # eval ran with rotation OFF


def test_buffer_repair_reset_runs_with_rotation_suppressed(monkeypatch):
    # The post-eval vec_env.reset() ALSO goes through TankEnv.reset, so it must be covered by
    # the suppression too (it must not switch_arena either at the boundary).
    cb, model, vec_env, raw_env, *_ = _make_callback(num_timesteps=1000)
    _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()

    assert vec_env.reset_count == 1
    assert vec_env.rotation_enabled_at_reset is False  # repair reset ran with rotation OFF


def test_rotation_restored_after_eval(monkeypatch):
    # After the eval block, rotation is RESTORED to its prior value so the NEXT training
    # rollout's first reset (the VecEnv auto-reset on episode end) rotates normally.
    cb, model, vec_env, raw_env, *_ = _make_callback(num_timesteps=1000)
    _patch_eval(monkeypatch, 0.5)

    assert raw_env.unwrapped.rotation_enabled is True  # starts enabled
    cb._on_rollout_end()
    assert raw_env.unwrapped.rotation_enabled is True  # restored after the eval block
    # And the sequence was exactly: disable for the block, then restore.
    assert raw_env.unwrapped.rotation_calls == [False, True]


def test_rotation_restored_even_if_eval_raises(monkeypatch):
    # try/finally: if evaluate_winrate raises mid-eval, rotation is STILL restored so a
    # subsequent training reset is not left wedged with rotation disabled.
    cb, model, vec_env, raw_env, *_ = _make_callback(num_timesteps=1000)

    def boom(model, env, n_episodes, seed=None):
        raise RuntimeError("eval blew up")

    monkeypatch.setattr(callbacks_mod, "evaluate_winrate", boom)

    with pytest.raises(RuntimeError, match="eval blew up"):
        cb._on_rollout_end()
    assert raw_env.unwrapped.rotation_enabled is True  # restored despite the exception
    assert raw_env.unwrapped.rotation_calls == [False, True]


def test_no_eval_below_freq_does_not_touch_rotation(monkeypatch):
    # Below the eval boundary: no eval, no rotation toggling at all (the env is untouched).
    cb, model, vec_env, raw_env, *_ = _make_callback(eval_freq=1000, num_timesteps=500)
    _patch_eval(monkeypatch, 0.5)

    cb._on_rollout_end()

    assert raw_env.unwrapped.rotation_calls == []  # never toggled
    assert raw_env.unwrapped.rotation_enabled is True
