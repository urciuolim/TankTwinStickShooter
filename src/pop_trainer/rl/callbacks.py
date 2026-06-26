"""SB3 callbacks for the M1 trainer — periodic greedy per-opponent win-rate eval.

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the training
policy's GREEDY (``deterministic=True``) win-rate against each opponent in a roster (via the
self-play seam) and logs ``eval/win_rate/<selector>`` plus an overall ``eval/win_rate`` into the
model's existing SB3 logger, so they land in both ``progress.csv`` and TensorBoard ``tfevents``.

Why this is non-trivial (the crux):

* Eval runs at a ROLLOUT BOUNDARY (``_on_rollout_end``), NEVER mid-rollout — running ``env.step``
  inside a rollout would poison the PPO experience buffer. ``_on_step`` only returns ``True``; the
  eval is gated by timesteps in ``_on_rollout_end``.
* Eval reuses the SINGLE training env / socket (one Unity build, one connection): we eval against
  the UNDERLYING gymnasium env (``vec_env.envs[0].unwrapped`` — the raw ``TankEnv`` behind the
  ``DummyVecEnv`` / ``Monitor`` SB3 wraps it in), NOT the vec wrapper, because
  :func:`evaluate_winrate` re-wraps the raw env in its own per-opponent ``SelfPlayWrapper``. Eval
  drives ``reset`` / ``step`` on that shared env, leaving the model's ``_last_obs`` /
  ``_last_episode_starts`` STALE.
* So AFTER eval we REPAIR the rollout state: ``obs = vec_env.reset()`` (a fresh episode) and write
  ``model._last_obs = obs`` + ``model._last_episode_starts = all-True`` so the NEXT rollout starts
  clean. The repair is in a ``finally`` so it ALWAYS runs, even if eval raises.

Imports sb3 + numpy (a trainer-side module, not in the pure-logic import path); the eval logic is
reused from :func:`pop_trainer.rl.evaluate.evaluate_winrate` (NOT sb3 ``evaluate_policy``).
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pop_trainer.rl.evaluate import evaluate_winrate, overall_win_rate
from pop_trainer.rl.selfplay import DEFAULT_ROSTER

__all__ = ["EvalWinRateCallback"]


class EvalWinRateCallback(BaseCallback):
    """Periodically log greedy per-opponent ``eval/win_rate`` at rollout boundaries.

    Every ``eval_freq`` TIMESTEPS (checked at the end of each rollout), runs ``eval_episodes``
    greedy episodes per opponent via :func:`pop_trainer.rl.evaluate.evaluate_winrate` against the
    UNDERLYING training env, logs ``eval/win_rate/<selector>`` for each opponent plus an overall
    ``eval/win_rate``, then repairs the model's rollout state so the next rollout is clean. See the
    module docstring for why this must be a rollout-boundary operation with a buffer repair after.

    Args:
        eval_freq: minimum env-steps between evaluations (gated against the last eval's timestep at
            each rollout boundary). ``<= 0`` disables eval.
        eval_episodes: number of greedy episodes per opponent per evaluation.
        opponents: the roster of ``agents`` selector strings to evaluate against (default
            :data:`~pop_trainer.rl.selfplay.DEFAULT_ROSTER`).
        seed: optional seed passed to ``evaluate_winrate`` (opponent reproducibility across evals).
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int = 10,
        opponents=DEFAULT_ROSTER,
        seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponents = tuple(opponents)
        self.seed = seed
        # Timestep of the last eval; the FIRST boundary past eval_freq fires.
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        """No-op per-step hook: eval happens at the ROLLOUT boundary, not mid-rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """At a rollout boundary: if due, eval per-opponent win-rate, log, and repair the buffer."""
        if self.eval_freq <= 0:
            return
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return
        self._last_eval_timestep = self.num_timesteps

        # The VecEnv SB3 wraps the bare TankEnv in; reach the RAW gymnasium env. envs[0] may be a
        # Monitor wrapper, so .unwrapped gets the underlying TankEnv. evaluate_winrate re-wraps it
        # in its own per-opponent SelfPlayWrapper, so we hand it the raw env (NOT the vec wrapper).
        vec_env = self.model.get_env()
        raw_env = vec_env.envs[0].unwrapped

        try:
            # NOTE (rotation suppression): the reference suppressed map rotation during eval via an
            # env-owned `set_rotation_enabled(False)` toggle, because a mid-eval reset that sent
            # `switch_arena` desynced the build (the build only handles switch_arena while !ingame,
            # so the env reads a stale `state` instead of the arena-switch ack and crashes). Our
            # TankEnv does NOT own rotation — it rotates ONLY when the CALLER passes
            # reset(options={"switch_arena": ...}). So we honor the SAME intent simply by NEVER
            # passing switch_arena on any eval reset: evaluate_winrate calls plain reset()s, and the
            # buffer-repair reset below is a plain vec_env.reset(). No switch_arena is ever sent
            # mid-eval, so the documented desync cannot occur. There is no rotation-toggle API to
            # call here (calling a nonexistent env.set_rotation_enabled would be wrong).
            per_opponent = evaluate_winrate(
                self.model,
                raw_env,
                opponents=self.opponents,
                n_episodes=self.eval_episodes,
                seed=self.seed,
            )

            # Log per-opponent and overall into the model's existing logger (-> progress.csv AND
            # tfevents). Then dump so the row lands at this timestep.
            for selector, rate in per_opponent.items():
                self.logger.record(f"eval/win_rate/{selector}", rate)
            self.logger.record("eval/win_rate", overall_win_rate(per_opponent))
            self.logger.dump(self.num_timesteps)
        finally:
            # --- REPAIR the rollout state (the crux) -----------------------------------------
            # evaluate_winrate left the SHARED training env mid/just-finished an eval episode and
            # the model's _last_obs / _last_episode_starts stale. Reset the vec env to a fresh
            # episode (a plain reset — NO switch_arena) and write the fresh obs back into the model
            # so the NEXT rollout is clean. In the finally so the repair ALWAYS runs, even if eval
            # raised mid-batch.
            obs = vec_env.reset()
            self.model._last_obs = obs
            self.model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
