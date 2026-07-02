"""SB3 callbacks for the M1 trainer — periodic greedy per-opponent win-rate eval + the
matchup-sampling aggregator.

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the training
policy's GREEDY (``deterministic=True``) win-rate against each opponent in a roster (via the
self-play seam) and logs ``eval/win_rate/<selector>`` plus an overall ``eval/win_rate`` into the
model's existing SB3 logger, so they land in both ``progress.csv`` and TensorBoard ``tfevents``.

Also ships :class:`MatchupSamplingCallback` (the ``--matchup-sampling winrate`` main-process
half): it accumulates the ``info["matchup"]`` cell/outcome tags the training
``SelfPlayWrapper``s emit on TERMINAL steps, folds them into a per-cell win-rate EMA at each
ROLLOUT BOUNDARY, recomputes the deficit sampling distribution, and broadcasts the plain-data
distribution to every training env via ``set_attr`` — the same delegation path eval uses to
re-pin opponents. The EMA matrix lives HERE (one process), so it is position-exact at any
``n_envs`` and is what the sidecar persists/restores across ``--resume``.

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
from pop_trainer.rl.matchup import (
    INITIAL_WIN_RATE,
    deficit_distribution,
    distribution_entropy,
    ema_update,
    matchup_cells,
    worst_cells,
)
from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence

    from stable_baselines3.common.vec_env import VecEnv

__all__ = ["EvalWinRateCallback", "MatchupSamplingCallback"]


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


class MatchupSamplingCallback(BaseCallback):
    """Aggregate training-episode outcomes per (opponent x map) cell and steer the joint sampler.

    The main-process half of ``--matchup-sampling winrate``. The per-env
    :class:`~pop_trainer.rl.selfplay.MatchupProvider`s draw each episode's (opponent, arena) cell
    from a distribution this callback owns:

    * ``_on_step`` CONSUMES the ``info["matchup"]`` tags the training ``SelfPlayWrapper``s emit
      on terminal steps (read from ``self.locals`` dones + infos) and accumulates them —
      accumulate ONLY, the distribution never changes mid-rollout;
    * ``_on_rollout_end`` folds the accumulated (cell, outcome) samples into the per-cell
      win-rate EMA (:func:`~pop_trainer.rl.matchup.ema_update`; unseen cells start at the 0.5
      prior), recomputes ``P(cell) = floor * uniform + (1 - floor) * normalize(1 - wr)``
      (:func:`~pop_trainer.rl.matchup.deficit_distribution`), and BROADCASTS the plain-data
      distribution to every training env via ``training_vec.set_attr("matchup_distribution",
      ...)`` — the same delegation path eval uses to re-pin opponents. Only the distribution
      travels: each worker's provider (and its seeded RNG stream) is never replaced;
    * ``_on_training_start`` broadcasts once up-front so a ``--resume``-restored EMA reaches the
      freshly-built (uniform-initialized) providers before the first rollout.

    The EMA matrix lives HERE, in one process, so it is position-exact at ANY ``n_envs`` (the
    per-env sampling RNGs remain per-subproc and reseed on resume, like the existing providers).
    :meth:`state` / :meth:`restore` are the sidecar seam: a plain-JSON dict of cells, win-rates,
    and sample counts that continues the curriculum across a resume.

    Observability: at each rollout-boundary update ONE compact summary lands on the system
    logger (the ``worst_k`` lowest-win-rate cells + the distribution entropy + sample counts),
    plus two aggregate SB3 scalars (``matchup/distribution_entropy``, ``matchup/episodes``) —
    never one scalar per cell.

    Args:
        opponents: the roster selector strings (the cell rows).
        maps: the arena rotation (the cell columns), or ``None`` for boot-arena cells.
        floor: the exploration floor ``eps`` in ``[0, 1]`` (min cell prob ``floor / n_cells``).
        ema_alpha: the EMA weight of the newest episode, in ``(0, 1]``.
        training_vec: the TRAINING vec env handle (the integrator passes ``model.env``); the
            ``set_attr`` broadcast target. ``None`` skips broadcasting (pure aggregation).
        sys_logger: optional system logger for the rollout-boundary summary record.
        worst_k: how many lowest-win-rate cells the summary names.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        opponents: Sequence[str],
        maps: Sequence[str] | None = None,
        *,
        floor: float = 0.25,
        ema_alpha: float = 0.05,
        training_vec: VecEnv | None = None,
        sys_logger: logging.Logger | None = None,
        worst_k: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.cells = matchup_cells(opponents, maps)
        self._cell_index = {cell: i for i, cell in enumerate(self.cells)}
        self.win_rates = [INITIAL_WIN_RATE] * len(self.cells)
        self.counts = [0] * len(self.cells)
        self.floor = floor
        self.ema_alpha = ema_alpha
        self.training_vec = training_vec
        self.sys_logger = sys_logger
        self.worst_k = worst_k
        # Terminal samples accumulated within the CURRENT rollout: (cell index, outcome float).
        self._pending: list[tuple[int, float]] = []

    @property
    def distribution(self) -> list[float]:
        """The current sampling distribution over :attr:`cells` (recomputed from the EMA)."""
        return deficit_distribution(self.win_rates, self.floor)

    def state(self) -> dict:
        """The plain-JSON resumable state the sidecar persists (cells + EMA + counts + config)."""
        return {
            "sampling": "winrate",
            "floor": float(self.floor),
            "ema_alpha": float(self.ema_alpha),
            "cells": [[selector, arena] for selector, arena in self.cells],
            "win_rates": [float(wr) for wr in self.win_rates],
            "counts": [int(c) for c in self.counts],
        }

    def restore(self, state: dict) -> None:
        """Continue a persisted curriculum: restore each cell's EMA + count by cell key.

        Matching is by cell tag, so a resume with a changed roster / rotation keeps the cells
        that still exist and leaves new cells at the 0.5 prior. A malformed / empty block is a
        no-op.
        """
        cells = state.get("cells") or []
        win_rates = state.get("win_rates") or []
        counts = state.get("counts") or []
        if not (len(cells) == len(win_rates) == len(counts)):
            return
        saved = {
            (cell[0], cell[1]): (float(wr), int(count))
            for cell, wr, count in zip(cells, win_rates, counts, strict=True)
        }
        for i, cell in enumerate(self.cells):
            if cell in saved:
                self.win_rates[i], self.counts[i] = saved[cell]

    def _broadcast(self) -> None:
        """Push the current distribution into every training env's provider (plain data only)."""
        if self.training_vec is not None:
            self.training_vec.set_attr("matchup_distribution", self.distribution)

    def _on_training_start(self) -> None:
        """Sync the providers to the callback's EMA before the first rollout (the resume seam).

        On a fresh run this re-sends the uniform distribution the providers already hold (a
        no-op in effect); after a ``--resume`` restore it is what carries the persisted
        curriculum into the freshly-built providers.
        """
        self._broadcast()

    def _on_step(self) -> bool:
        """Accumulate terminal (cell, outcome) samples — NEVER change the distribution here."""
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos, strict=True):
            if not done:
                continue
            tag = info.get("matchup")
            if not isinstance(tag, dict):
                continue
            index = self._cell_index.get((tag.get("opponent"), tag.get("map")))
            if index is None:
                continue
            self._pending.append((index, float(tag.get("outcome", 0.5))))
        return True

    def _on_rollout_end(self) -> None:
        """Fold the rollout's samples into the EMA, broadcast the new distribution, log ONE line."""
        new_samples = len(self._pending)
        for index, outcome in self._pending:
            self.win_rates[index] = ema_update(self.win_rates[index], outcome, self.ema_alpha)
            self.counts[index] += 1
        self._pending.clear()

        distribution = self.distribution
        self._broadcast()

        entropy = distribution_entropy(distribution)
        self.logger.record("matchup/distribution_entropy", entropy)
        self.logger.record("matchup/episodes", sum(self.counts))
        if self.sys_logger is not None:
            worst = worst_cells(self.cells, self.win_rates, self.worst_k)
            self.sys_logger.info(
                "matchup_update",
                extra={
                    "detail": {
                        "num_timesteps": self.num_timesteps,
                        "new_samples": new_samples,
                        "total_episodes": sum(self.counts),
                        "entropy": entropy,
                        "worst_cells": [
                            {"opponent": selector, "map": arena, "win_rate": wr}
                            for (selector, arena), wr in worst
                        ],
                    }
                },
            )
