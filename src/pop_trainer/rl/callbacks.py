"""SB3 callbacks for the M1 trainer — periodic greedy per-opponent win-rate eval.

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the training
policy's GREEDY (``deterministic=True``) win-rate against each opponent in a roster (via the
self-play seam) and logs ``eval/win_rate/<selector>`` plus an overall ``eval/win_rate`` into the
model's existing SB3 logger, so they land in both ``progress.csv`` and TensorBoard ``tfevents``.

Why this is non-trivial (the crux):

* Eval runs against a DEDICATED eval env — a SEPARATE :class:`~pop_trainer.env.tank_env.TankEnv`
  on its OWN Unity build / socket, PASSED IN at construction. It is NOT ``self.model.get_env()``
  and NOT a re-wrap of the training vec stack. Because the training env / socket is never driven by
  eval, the model's rollout state (``_last_obs`` / ``_last_episode_starts``) is left untouched and
  there is NO buffer "repair" to undo — eval and training are fully isolated.
* Eval still runs at a ROLLOUT BOUNDARY (``_on_rollout_end``), gated by ``eval_freq`` timesteps,
  NEVER mid-rollout. ``_on_step`` only returns ``True``; the eval is gated in ``_on_rollout_end``.
  (The dedicated env makes this purely a cadence choice, not a correctness one — the training
  buffer can no longer be poisoned by eval — but the boundary cadence is kept so eval work is
  batched between rollouts rather than per-step.)
* :func:`evaluate_winrate` re-wraps the raw eval ``TankEnv`` in its own per-opponent
  ``SelfPlayWrapper`` and plays greedy (``deterministic=True``) episodes via ``model.predict``.

Imports sb3 (a trainer-side module, not in the pure-logic import path); the eval logic is reused
from :func:`pop_trainer.rl.evaluate.evaluate_winrate` (NOT sb3 ``evaluate_policy``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

from pop_trainer.rl.evaluate import evaluate_winrate, overall_win_rate
from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:
    import gymnasium

__all__ = ["EvalWinRateCallback"]


class EvalWinRateCallback(BaseCallback):
    """Periodically log greedy per-opponent ``eval/win_rate`` at rollout boundaries.

    Every ``eval_freq`` TIMESTEPS (checked at the end of each rollout), runs ``eval_episodes``
    greedy episodes per opponent via :func:`pop_trainer.rl.evaluate.evaluate_winrate` against the
    DEDICATED eval env (a separate ``TankEnv`` on its own build/socket, passed in), logs
    ``eval/win_rate/<selector>`` for each opponent plus an overall ``eval/win_rate``, then dumps the
    row. The training env / model rollout state is NEVER touched, so no buffer repair is needed —
    see the module docstring.

    Args:
        eval_freq: minimum env-steps between evaluations (gated against the last eval's timestep at
            each rollout boundary). ``<= 0`` disables eval.
        eval_episodes: number of greedy episodes per opponent per evaluation.
        opponents: the roster of ``agents`` selector strings to evaluate against (default
            :data:`~pop_trainer.rl.selfplay.DEFAULT_ROSTER`).
        seed: optional seed passed to ``evaluate_winrate`` (opponent reproducibility across evals).
        eval_env: the raw eval ``TankEnv`` (a gymnasium env, NOT a ``SelfPlayWrapper`` / vec stack)
            that ``evaluate_winrate`` wraps per-opponent. Lives on a SEPARATE socket from training.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int = 10,
        opponents=DEFAULT_ROSTER,
        seed: int | None = None,
        eval_env: gymnasium.Env | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponents = tuple(opponents)
        self.seed = seed
        self.eval_env = eval_env
        # Timestep of the last eval; the FIRST boundary past eval_freq fires.
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        """No-op per-step hook: eval happens at the ROLLOUT boundary, not mid-rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """At a rollout boundary: if due, eval per-opponent win-rate on the dedicated eval env."""
        if self.eval_freq <= 0:
            return
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return
        self._last_eval_timestep = self.num_timesteps

        # Eval against the DEDICATED eval env (its own build/socket). evaluate_winrate re-wraps this
        # raw env in its own per-opponent SelfPlayWrapper. The training env / model state is never
        # touched here — no buffer repair afterward.
        #
        # NOTE (rotation suppression): a mid-eval reset that sent `switch_arena` would desync the
        # build (it only handles switch_arena while !ingame, so the env reads a stale `state`
        # instead of the arena-switch ack and crashes). Our TankEnv rotates ONLY when the CALLER
        # passes reset(options={"switch_arena": ...}); evaluate_winrate calls plain reset()s, so no
        # switch_arena is ever sent during eval and the documented desync cannot occur. There is no
        # rotation-toggle API to call here.
        per_opponent = evaluate_winrate(
            self.model,
            self.eval_env,
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
