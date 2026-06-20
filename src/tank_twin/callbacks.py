"""SB3 callbacks for the M1 trainer — periodic greedy win-rate eval (Feature 3).

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the
training policy's GREEDY (``deterministic=True``) win-rate vs. the RANDOM opponent (the
training env already IS that env) and logs ``eval/win_rate`` into the model's existing SB3
logger, so it lands in both ``progress.csv`` and the TensorBoard ``tfevents`` the trainer
attaches via ``configure([..., "csv", "tensorboard"])``.

Why this is non-trivial (the crux):

* Eval runs at a ROLLOUT BOUNDARY (``_on_rollout_end``), NEVER mid-rollout — running
  ``env.step`` inside a rollout would poison the PPO experience buffer. ``_on_step`` only
  returns ``True``; the eval is gated by timesteps in ``_on_rollout_end``.
* Eval reuses the SINGLE training env / socket (one Unity build, one connection): we eval
  against the UNDERLYING gymnasium env (``vec_env.envs[0]`` for the ``DummyVecEnv`` SB3 wraps
  a bare env in), NOT the vec wrapper (``evaluate_winrate`` expects a raw gymnasium env). Eval
  drives ``env.reset()`` / ``env.step()`` on that shared env, leaving it mid/just-finished an
  eval episode and the model's ``_last_obs`` / ``_last_episode_starts`` STALE.
* So AFTER eval we REPAIR the rollout state: ``obs = vec_env.reset()`` (a fresh episode) and
  write ``model._last_obs = obs`` + ``model._last_episode_starts = all-True`` so the NEXT
  rollout starts clean.

Imports sb3 + numpy (it is a trainer-side module, not in the pure-logic import path);
``evaluate_winrate`` is reused from :mod:`tank_twin.evaluate` (NOT sb3 ``evaluate_policy``).
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from tank_twin.evaluate import evaluate_winrate

__all__ = ["EvalWinRateCallback"]


class EvalWinRateCallback(BaseCallback):
    """Periodically log greedy ``eval/win_rate`` vs. the random opponent at rollout boundaries.

    Every ``eval_freq`` TIMESTEPS (checked at the end of each rollout), runs ``eval_episodes``
    greedy episodes via :func:`tank_twin.evaluate.evaluate_winrate` against the UNDERLYING
    training env, logs ``eval/win_rate``, then repairs the model's rollout state so the next
    rollout is clean. See the module docstring for why this must be a rollout-boundary
    operation with a buffer repair afterward.

    Args:
        eval_freq: minimum env-steps between evaluations (gated against the last eval's
            timestep at each rollout boundary). ``<= 0`` disables eval (the trainer simply
            does not attach this callback in that case).
        eval_episodes: number of greedy episodes per evaluation.
        seed: optional seed passed to ``evaluate_winrate`` (its FIRST reset) for opponent
            reproducibility across evaluations.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int = 10,
        seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.seed = seed
        # Timestep of the last eval; -inf-like start so the FIRST boundary past eval_freq fires.
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        """No-op per-step hook: eval happens at the ROLLOUT boundary, not mid-rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """At a rollout boundary: if due, eval greedy win-rate, log it, and repair the buffer."""
        if self.eval_freq <= 0:
            return
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return
        self._last_eval_timestep = self.num_timesteps

        # The VecEnv SB3 wraps the bare TankEnv in; the raw gymnasium env is envs[0] for a
        # DummyVecEnv. evaluate_winrate expects the RAW env (it calls reset()/step() returning
        # the gymnasium 5-tuple), so we eval against the underlying env, NOT the vec wrapper.
        vec_env = self.model.get_env()
        raw_env = vec_env.envs[0]

        rate = evaluate_winrate(self.model, raw_env, n_episodes=self.eval_episodes, seed=self.seed)

        # Log into the model's existing logger so it lands in progress.csv AND tfevents.
        self.logger.record("eval/win_rate", rate)
        self.logger.dump(self.num_timesteps)

        # --- REPAIR the rollout state (the crux) -----------------------------------------
        # evaluate_winrate left the SHARED training env mid/just-finished an eval episode and
        # the model's _last_obs / _last_episode_starts stale. Reset the vec env to a fresh
        # episode and write the fresh obs back into the model so the NEXT rollout is clean.
        obs = vec_env.reset()
        self.model._last_obs = obs
        self.model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
