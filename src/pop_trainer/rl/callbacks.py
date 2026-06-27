"""SB3 callbacks for the M1 trainer — periodic greedy per-opponent win-rate eval.

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the training
policy's GREEDY (``deterministic=True``) win-rate against each opponent in a roster (via the
self-play seam) and logs ``eval/win_rate/<selector>`` plus an overall ``eval/win_rate`` into the
model's existing SB3 logger, so they land in both ``progress.csv`` and TensorBoard ``tfevents``.

THE INVARIANT (enforced structurally): NO eval Unity instance ever coexists with a training Unity
instance. The training builds and the eval builds use the SAME RAM budget (``M_eval == N_train``),
so they must be time-multiplexed. At each eval boundary the callback runs this sequence:

1. TEAR DOWN all training Unity instances: ``training_vec.env_method("release")`` (n=1: the
   ``DummyVecEnv`` calls ``release`` directly in-process; n>1: the ``SubprocVecEnv`` fans the call
   to every worker). ``release`` HARD-KILLS each training Unity child via the env's injected reap
   hook and frees its port — the worker PROCESSES stay alive, only the Unity children die.
2. EVAL: lazy-launch the M eval instances (``eval_vec.reset`` / the parallel eval's first step) and
   run ``eval_episodes`` per opponent DISTRIBUTED across them in parallel
   (:func:`pop_trainer.rl.evaluate.evaluate_winrate`), aggregating per opponent.
3. TEAR DOWN the eval instances: ``eval_vec.env_method("release")``.
4. RESPAWN training: ``training_vec.reset()`` (lazily re-launches each training instance via its
   factory) and OVERWRITE the model's rollout sentinels with the fresh obs
   (``model._last_obs = reset_obs``; ``model._last_episode_starts[:] = True``). In-flight training
   episodes are abandoned (truncation) — intended; the next rollout starts clean.

So at any instant the run holds EITHER the training set OR the eval set, never both. The callback is
handed BOTH vec env handles (and reaches the model via ``self.model``) at construction.

Eval runs at a ROLLOUT BOUNDARY (``_on_rollout_end``), gated by ``eval_freq`` timesteps, NEVER
mid-rollout. ``_on_step`` only returns ``True``. :func:`evaluate_winrate` re-wraps each raw eval
``TankEnv`` in its own per-opponent ``SelfPlayWrapper`` and plays greedy episodes via
``model.predict``.

Imports sb3 (a trainer-side module, not in the pure-logic import path); the eval logic is reused
from :func:`pop_trainer.rl.evaluate.evaluate_winrate` (NOT sb3 ``evaluate_policy``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

from pop_trainer.rl.evaluate import evaluate_winrate, overall_win_rate
from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

__all__ = ["EvalWinRateCallback"]


class EvalWinRateCallback(BaseCallback):
    """Periodically log greedy per-opponent ``eval/win_rate`` at rollout boundaries.

    Every ``eval_freq`` TIMESTEPS (checked at the end of each rollout), this time-multiplexes the
    training and eval Unity instances so the two sets NEVER coexist (see the module docstring): it
    tears down all training instances, runs ``eval_episodes`` greedy episodes per opponent
    distributed across the M eval instances in parallel
    (:func:`pop_trainer.rl.evaluate.evaluate_winrate`), tears the eval instances down, then respawns
    training and re-syncs the model's rollout sentinels to the post-respawn obs. It logs
    ``eval/win_rate/<selector>`` for each opponent plus an overall ``eval/win_rate``, then dumps the
    row.

    Args:
        eval_freq: minimum env-steps between evaluations (gated against the last eval's timestep at
            each rollout boundary). ``<= 0`` disables eval.
        eval_episodes: number of greedy episodes per opponent per evaluation (distributed across the
            M eval envs).
        opponents: the roster of ``agents`` selector strings to evaluate against (default
            :data:`~pop_trainer.rl.selfplay.DEFAULT_ROSTER`).
        seed: optional seed passed to ``evaluate_winrate`` (opponent reproducibility across evals).
        eval_env: the eval vec env (``M`` envs; ``DummyVecEnv`` at M=1, ``SubprocVecEnv`` at M>1).
            ``evaluate_winrate`` peels it to the raw ``TankEnv``s and re-wraps each per-opponent.
            Its instances are spawned for eval and torn down (``env_method("release")``) afterward.
        training_vec: the TRAINING vec env handle — the SAME vec env SB3 reads ``_last_obs`` from
            (the integrator passes ``model.env``, i.e. SB3's ``VecTransposeImage``-wrapped training
            stack, so the respawn ``reset()`` returns obs in the exact space ``_last_obs`` expects).
            Torn down (``env_method("release")``) BEFORE eval instances spawn and respawned
            (``reset()``) AFTER eval. Passed in explicitly rather than via ``self.model.get_env()``
            so the handle is unambiguous.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int = 10,
        opponents=DEFAULT_ROSTER,
        seed: int | None = None,
        eval_env: VecEnv | None = None,
        training_vec: VecEnv | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponents = tuple(opponents)
        self.seed = seed
        self.eval_env = eval_env
        self.training_vec = training_vec
        # Timestep of the last eval; the FIRST boundary past eval_freq fires.
        self._last_eval_timestep = 0

    def _on_step(self) -> bool:
        """No-op per-step hook: eval happens at the ROLLOUT boundary, not mid-rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """At a rollout boundary: if due, teardown training -> parallel eval -> respawn training."""
        if self.eval_freq <= 0:
            return
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return
        self._last_eval_timestep = self.num_timesteps

        per_opponent = self._run_eval_cycle()

        # Log per-opponent and overall into the model's existing logger (-> progress.csv AND
        # tfevents). Then dump so the row lands at this timestep.
        for selector, rate in per_opponent.items():
            self.logger.record(f"eval/win_rate/{selector}", rate)
        self.logger.record("eval/win_rate", overall_win_rate(per_opponent))
        self.logger.dump(self.num_timesteps)

    def _run_eval_cycle(self) -> dict[str, float]:
        """Teardown training -> parallel eval -> teardown eval -> respawn training (the INVARIANT).

        Returns the per-opponent win-rate. The order is load-bearing: the eval instances are spawned
        ONLY after every training instance is reaped, and the training instances are respawned ONLY
        after every eval instance is reaped, so the two sets never coexist.

        NOTE (eval-rotation suppression): a mid-eval reset that sent ``switch_arena`` would desync
        a build (it handles ``switch_arena`` only while ``!ingame``). ``TankEnv`` rotates ONLY when
        the CALLER passes ``reset(options={"switch_arena": ...})``; ``evaluate_winrate`` calls plain
        ``reset()``s, so no ``switch_arena`` is ever sent during eval and that desync cannot occur.
        """
        # 1. Tear down ALL training Unity instances (frees their ports; workers stay alive).
        self.training_vec.env_method("release")

        try:
            # 2. Parallel eval across the M eval instances (lazy-launched on first reset/step).
            per_opponent = evaluate_winrate(
                self.model,
                self.eval_env,
                opponents=self.opponents,
                n_episodes=self.eval_episodes,
                seed=self.seed,
            )
        finally:
            # 3. Tear down the eval instances regardless of eval outcome (invariant before respawn).
            self.eval_env.env_method("release")

        # 4. Respawn training: reset lazily re-launches each training instance; re-sync the model's
        # rollout sentinels to the fresh obs so the next rollout starts from a valid state. The
        # abandoned in-flight episodes truncate — intended.
        reset_obs = self.training_vec.reset()
        self.model._last_obs = reset_obs
        if getattr(self.model, "_last_episode_starts", None) is not None:
            self.model._last_episode_starts[:] = True

        return per_opponent
