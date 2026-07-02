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
5. rotation eval (``maps=...``): one ``eval/win_rate/map/<short>`` scalar per MAP (never per
   cell); phase resets carry the pinned ``switch_arena``; without ``maps`` NO map scalar exists
   and no ``switch_arena`` is ever sent (byte-identical fallback).
6. matchup-EMA isolation: an eval-style untagged terminal info leaves the aggregator untouched.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from pop_trainer.core.state import STATE_LEN
from pop_trainer.rl.callbacks import EvalWinRateCallback, MatchupSamplingCallback
from pop_trainer.rl.evaluate import WIN
from pop_trainer.rl.selfplay import MatchupProvider, OpponentProvider, SelfPlayWrapper

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


def _make(eval_freq, eval_episodes, opponents, m=1, maps=None):
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
        maps=maps,
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
    # the no-rotation fallback: NEVER send switch_arena on any eval reset when maps is None.
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 2, ("noop",))
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # every raw eval-env reset during eval passed options=None (plain handshake, byte-identical).
    raw_envs = [w.env for w in eval_vec.envs]
    assert any(e.reset_options for e in raw_envs)  # eval did reset the eval envs
    assert all(opt is None for e in raw_envs for opt in e.reset_options)
    # and the eval wrappers' maps attr was never pinned.
    assert all(w.maps is None for w in eval_vec.envs)


def test_no_rotation_records_no_map_scalars():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(100, 2, ("noop", "random"))
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    assert not any(key.startswith("eval/win_rate/map/") for key in logger.records)


# --- 5. rotation eval: per-map scalars + pinned switch_arena ------------------------------

_EVAL_MAPS = ("Arenas/a.json", "Arenas/b.json")


def test_rotation_eval_logs_one_scalar_per_map_with_short_names():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(
        100, 2, ("noop", "random"), maps=_EVAL_MAPS
    )
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # exactly one scalar per MAP (short tag names), never one per (opponent x map) cell.
    map_keys = sorted(k for k in logger.records if k.startswith("eval/win_rate/map/"))
    assert map_keys == ["eval/win_rate/map/a", "eval/win_rate/map/b"]
    # always-win eval stubs -> every marginal is 1.0 and they all reconcile.
    assert logger.records["eval/win_rate/map/a"] == 1.0
    assert logger.records["eval/win_rate/map/b"] == 1.0
    assert logger.records["eval/win_rate/noop"] == 1.0
    assert logger.records["eval/win_rate/random"] == 1.0
    assert logger.records["eval/win_rate"] == 1.0
    assert logger.dumps == [100]


def test_rotation_eval_resets_carry_the_pinned_switch_arena():
    cb, model, logger, training_vec, eval_vec, _ledger, trace = _make(
        100, 2, ("noop",), maps=_EVAL_MAPS
    )
    model.num_timesteps = cb.num_timesteps = 100

    cb._on_rollout_end()

    # every eval reset carried a pinned rotation arena, and both arenas were visited.
    raw_envs = [w.env for w in eval_vec.envs]
    opts = [opt for e in raw_envs for opt in e.reset_options]
    assert opts and all(opt is not None and opt.get("switch_arena") in _EVAL_MAPS for opt in opts)
    assert {opt["switch_arena"] for opt in opts} == set(_EVAL_MAPS)


def test_rotation_eval_preserves_the_no_coexist_invariant():
    # The teardown -> eval -> respawn cycle is unchanged by the map phases (the ledger would
    # raise if a pinned-arena reset ever overlapped a live training instance).
    cb, model, logger, training_vec, eval_vec, ledger, trace = _make(
        100, 4, ("noop", "random"), m=2, maps=_EVAL_MAPS
    )
    model.num_timesteps = cb.num_timesteps = 100
    cb._on_rollout_end()
    assert ledger == {"train"}


# --- MatchupSamplingCallback: accumulate -> fold at rollout end -> broadcast ---------------


class _MatchupEnv(gymnasium.Env):
    """A minimal env for wrapper construction in the broadcast tests (never stepped)."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = _ACTION_SPACE
        self.observation_space = _OBS_SPACE

    def reset(self, *, seed=None, options=None):
        return (np.zeros((1,), dtype=np.float32), {"state": _state(0.0), "map": None})

    def step(self, action, opponent_action=None):
        info = {"state": _state(1.0), "map": None}
        return (np.zeros((1,), dtype=np.float32), 0.0, False, False, info)


class _LoggerModel:
    """The slice of an SB3 algorithm MatchupSamplingCallback touches: just ``.logger``."""

    def __init__(self, logger) -> None:
        self.logger = logger


class _RecordingSysLogger:
    """A system-logger stand-in recording every (event, detail) the callback emits."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event, extra=None):
        self.events.append((event, (extra or {}).get("detail", {})))


def _matchup_cb(opponents=("noop", "random"), maps=None, *, alpha=0.5, floor=0.25, **kwargs):
    cb = MatchupSamplingCallback(opponents, maps, floor=floor, ema_alpha=alpha, **kwargs)
    cb.model = _LoggerModel(FakeLogger())
    return cb


def _terminal_locals(*tags):
    """Fake SB3 rollout locals: one done lane per tag (a None tag = a done with no matchup)."""
    dones = np.array([True] * len(tags))
    infos = [{} if tag is None else {"matchup": tag} for tag in tags]
    return {"dones": dones, "infos": infos}


def test_matchup_on_step_accumulates_without_changing_distribution():
    cb = _matchup_cb()
    before = list(cb.distribution)
    cb.locals = _terminal_locals({"opponent": "noop", "map": None, "outcome": 1.0})
    assert cb._on_step() is True
    # accumulate ONLY: the EMA and the distribution are untouched mid-rollout.
    assert cb.win_rates == [0.5, 0.5]
    assert cb.counts == [0, 0]
    assert cb.distribution == before
    assert cb._pending == [(0, 1.0)]


def test_matchup_on_step_ignores_non_terminal_untagged_and_unknown_cells():
    cb = _matchup_cb()
    cb.locals = {
        "dones": np.array([False, True, True]),
        "infos": [
            # not done: its tag must NOT be consumed (there should be none in practice).
            {"matchup": {"opponent": "noop", "map": None, "outcome": 1.0}},
            {},  # done with no matchup tag (feature-off env / eval) -> ignored
            {"matchup": {"opponent": "not-a-cell", "map": None, "outcome": 1.0}},  # unknown cell
        ],
    }
    cb._on_step()
    assert cb._pending == []


def test_matchup_on_step_without_locals_is_a_noop():
    cb = _matchup_cb()
    cb.locals = {}
    assert cb._on_step() is True
    assert cb._pending == []


def test_eval_style_untagged_terminal_info_leaves_the_ema_untouched():
    # Eval episodes cannot feed the EMA: eval wrappers carry NO MatchupProvider (even during
    # rotation eval, where only a single-map MapProvider is pinned), so their terminal infos
    # have no "matchup" tag — and an untagged done is ignored by the aggregator.
    cb = _matchup_cb()
    cb.locals = {
        "dones": np.array([True]),
        "infos": [{"outcome": WIN, "state": _state(1.0)}],  # a typical eval terminal info
    }
    assert cb._on_step() is True
    assert cb._pending == []
    cb._on_rollout_end()
    assert cb.win_rates == [0.5, 0.5]
    assert cb.counts == [0, 0]


def test_matchup_rollout_end_folds_ema_counts_and_records_scalars():
    cb = _matchup_cb(alpha=0.5, floor=0.25)
    cb.locals = _terminal_locals({"opponent": "noop", "map": None, "outcome": 1.0})
    cb._on_step()
    cb._on_rollout_end()

    # EMA fold: 0.5 * 0.5 + 0.5 * 1.0 = 0.75 for the winning cell; the other keeps the prior.
    assert cb.win_rates == pytest.approx([0.75, 0.5])
    assert cb.counts == [1, 0]
    assert cb._pending == []  # consumed
    # deficits (0.25, 0.5) -> base (1/3, 2/3); P = 0.25 * 0.5 + 0.75 * base.
    assert cb.distribution == pytest.approx([0.375, 0.625])
    # exactly the two aggregate scalars (never one per cell).
    records = cb.model.logger.records
    assert set(records) == {"matchup/distribution_entropy", "matchup/episodes"}
    assert records["matchup/episodes"] == 1


def test_matchup_rollout_end_broadcasts_into_providers_without_replacing_them():
    # A real DummyVecEnv of two wrappers, each with its OWN seeded provider: the broadcast lands
    # the SAME plain-data distribution inside both providers; neither the provider objects nor
    # their RNG streams are replaced.
    providers = [MatchupProvider.from_roster(("noop", "random"), None, seed=i) for i in range(2)]
    placeholder = OpponentProvider.from_roster(["noop"], strategy="round_robin")

    def _fn(i):
        return lambda: SelfPlayWrapper(_MatchupEnv(), placeholder, matchups=providers[i])

    vec = DummyVecEnv([_fn(0), _fn(1)])
    rngs_before = [p._rng for p in providers]

    cb = _matchup_cb(alpha=0.5, floor=0.25, training_vec=vec)
    cb.locals = _terminal_locals({"opponent": "random", "map": None, "outcome": 0.0})
    cb._on_step()
    cb._on_rollout_end()

    expected = cb.distribution
    assert expected != pytest.approx([0.5, 0.5])  # the loss actually skewed it
    for wrapper, provider, rng in zip(vec.envs, providers, rngs_before, strict=True):
        assert wrapper.matchups is provider  # provider NOT replaced
        assert provider._rng is rng  # RNG stream NOT replaced
        assert provider.distribution == pytest.approx(expected)


def test_matchup_training_start_broadcasts_restored_curriculum():
    # The resume seam: freshly-built providers hold uniform; _on_training_start pushes the
    # callback's (restored) EMA-derived distribution into them before the first rollout.
    provider = MatchupProvider.from_roster(("noop", "random"), None, seed=0)
    placeholder = OpponentProvider.from_roster(["noop"], strategy="round_robin")
    vec = DummyVecEnv([lambda: SelfPlayWrapper(_MatchupEnv(), placeholder, matchups=provider)])

    cb = _matchup_cb(alpha=0.5, floor=0.0, training_vec=vec)
    cb.restore(
        {
            "sampling": "winrate",
            "cells": [["noop", None], ["random", None]],
            "win_rates": [1.0, 0.0],
            "counts": [10, 10],
        }
    )
    cb._on_training_start()
    # deficits (0, 1) with no floor -> all probability on the losing cell.
    assert provider.distribution == pytest.approx([0.0, 1.0])


def test_matchup_state_restore_roundtrip_through_json():
    import json

    cb = _matchup_cb(("noop", "random"), ("Arenas/a.json",), alpha=0.5)
    cb.locals = _terminal_locals({"opponent": "noop", "map": "Arenas/a.json", "outcome": 1.0})
    cb._on_step()
    cb._on_rollout_end()

    # STRICT-JSON roundtrip (the sidecar path) then restore into a FRESH callback.
    state = json.loads(json.dumps(cb.state()))
    fresh = _matchup_cb(("noop", "random"), ("Arenas/a.json",), alpha=0.5)
    fresh.restore(state)
    assert fresh.win_rates == cb.win_rates
    assert fresh.counts == cb.counts
    assert fresh.distribution == pytest.approx(cb.distribution)


def test_matchup_restore_matches_by_cell_key_tolerating_roster_changes():
    state = {
        "sampling": "winrate",
        "cells": [["noop", None], ["random", None]],
        "win_rates": [0.9, 0.2],
        "counts": [5, 7],
    }
    # A resumed run with a CHANGED roster keeps the cells that still exist; new cells stay at
    # the 0.5 prior.
    cb = _matchup_cb(("random", "wall-hugger"))
    cb.restore(state)
    assert cb.win_rates == [0.2, 0.5]
    assert cb.counts == [7, 0]


def test_matchup_restore_malformed_block_is_a_noop():
    cb = _matchup_cb()
    cb.restore({})  # empty block
    cb.restore({"cells": [["noop", None]], "win_rates": [0.1]})  # mismatched lengths
    assert cb.win_rates == [0.5, 0.5]
    assert cb.counts == [0, 0]


def test_matchup_all_cells_won_yields_uniform_not_nan():
    cb = _matchup_cb(alpha=1.0)
    for _ in range(2):
        cb.locals = _terminal_locals(
            {"opponent": "noop", "map": None, "outcome": 1.0},
            {"opponent": "random", "map": None, "outcome": 1.0},
        )
        cb._on_step()
    cb._on_rollout_end()
    assert cb.win_rates == [1.0, 1.0]
    # total deficit 0 -> uniform fallback, and the recorded entropy is finite (no NaN).
    assert cb.distribution == pytest.approx([0.5, 0.5])
    entropy = cb.model.logger.records["matchup/distribution_entropy"]
    assert np.isfinite(entropy)


def test_matchup_rollout_end_logs_one_compact_summary():
    sys_logger = _RecordingSysLogger()
    cb = _matchup_cb(alpha=0.5, sys_logger=sys_logger, worst_k=1)
    cb.locals = _terminal_locals({"opponent": "random", "map": None, "outcome": 0.0})
    cb._on_step()
    cb._on_rollout_end()

    assert len(sys_logger.events) == 1  # ONE summary per rollout-boundary update
    event, detail = sys_logger.events[0]
    assert event == "matchup_update"
    assert detail["new_samples"] == 1
    assert detail["total_episodes"] == 1
    assert np.isfinite(detail["entropy"])
    # worst_k=1 names the single lowest-win-rate cell — the one that just lost.
    assert detail["worst_cells"] == [{"opponent": "random", "map": None, "win_rate": 0.25}]
