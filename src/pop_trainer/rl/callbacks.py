"""SB3 callbacks for the M1 trainer — periodic greedy per-opponent win-rate eval + the
matchup-sampling aggregator.

Ships :class:`EvalWinRateCallback`: a ``BaseCallback`` that periodically evaluates the training
policy's GREEDY (``deterministic=True``) win-rate against each opponent in a roster (via the
self-play seam) and logs ``eval/win_rate/<selector>`` plus an overall ``eval/win_rate`` into the
model's existing SB3 logger, so they land in both ``progress.csv`` and TensorBoard ``tfevents``.
When the training map rotation is active (``maps`` given), each eval additionally covers EVERY
rotation arena — the per-opponent episode budget is spread deterministically across the maps
(:func:`~pop_trainer.rl.evaluate.episode_spread`; the eval cost is unchanged) — and the callback
also logs a per-map marginal ``eval/win_rate/map/<short-map-name>`` per arena. The per-opponent
scalars and the overall become marginals over maps, all POOLED from the same per-cell counts so
they reconcile; no per-(opponent x map) cell scalar is ever emitted.

Also ships :class:`MatchupSamplingCallback` (the ``--matchup-sampling winrate`` main-process
half). Its win-rate SIGNAL is EVALUATION: after each eval cycle, ``EvalWinRateCallback`` hands
the cycle's structured ``evaluate_winrate`` result to the matchup callback (the ``result_sink``
wiring in ``rl.train.train_local``), which folds the per-cell rates into its win-rate table at
the SAME rollout boundary (the eval callback runs FIRST in the ``CallbackList``, the matchup
callback LAST), recomputes the deficit sampling distribution, and broadcasts the plain-data
distribution to every training env via ``set_attr``. Training terminal outcomes (the
``info["matchup"]`` tags) feed only the per-cell play COUNTS — observability of what the sampler
actually played — never the win-rates. Rationale: a training-side win-rate measures the
STOCHASTIC policy, and at high entropy it inverts against the deterministic eval (training can
report a noop win-rate near 0.9 while eval reports near 0.1), steering the curriculum away from
the real weakness; eval measures the deterministic policy the run is judged on. The win-rate
table lives HERE (one process), so it is position-exact at any ``n_envs`` and is what the
sidecar persists/restores across ``--resume``.

THE INVARIANT (enforced structurally): NO eval Unity instance ever coexists with a training Unity
instance. The training builds and the eval builds use the SAME RAM budget (``M_eval == N_train``),
so they must be time-multiplexed. At each eval boundary the callback runs this sequence:

1. TEAR DOWN all training Unity instances: ``training_vec.env_method("release")`` (n=1: the
   ``DummyVecEnv`` calls ``release`` directly in-process; n>1: the ``SubprocVecEnv`` fans the call
   to every worker). ``release`` HARD-KILLS each training Unity child via the env's injected reap
   hook and frees its port — the worker PROCESSES stay alive, only the Unity children die.
2. EVAL: lazy-launch the M eval instances (``eval_vec.reset`` / the parallel eval's first step) and
   run ``eval_episodes`` per opponent DISTRIBUTED across them in parallel
   (:func:`pop_trainer.rl.evaluate.evaluate_winrate`), aggregating per opponent (and per map when
   the rotation is active).
3. TEAR DOWN the eval instances: ``eval_vec.env_method("release")``.
4. RESPAWN training: ``training_vec.reset()`` (lazily re-launches each training instance via its
   factory) and OVERWRITE the model's rollout sentinels with the fresh obs
   (``model._last_obs = reset_obs``; ``model._last_episode_starts[:] = True``). In-flight training
   episodes are abandoned (truncation) — intended; the next rollout starts clean.

So at any instant the run holds EITHER the training set OR the eval set, never both. The callback is
handed BOTH vec env handles (and reaches the model via ``self.model``) at construction.

Eval runs at a ROLLOUT BOUNDARY (``_on_rollout_end``), gated by ``eval_freq`` timesteps, NEVER
mid-rollout. ``_on_step`` only returns ``True``. :func:`evaluate_winrate` pins each eval env's
``SelfPlayWrapper`` per phase — the opponent always, plus a single-map provider when the map
rotation is active — and plays greedy episodes via ``model.predict``.

Imports sb3 (a trainer-side module, not in the pure-logic import path); the eval logic is reused
from :func:`pop_trainer.rl.evaluate.evaluate_winrate` (NOT sb3 ``evaluate_policy``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

from pop_trainer.rl.evaluate import (
    evaluate_winrate,
    map_short_name,
    overall_win_rate,
    pool_by_map,
    pool_by_opponent,
    pooled_win_rate,
)
from pop_trainer.rl.matchup import (
    INITIAL_WIN_RATE,
    deficit_distribution,
    distribution_entropy,
    eval_cell_rates,
    fold_eval_rates,
    matchup_cells,
    worst_cells,
)
from pop_trainer.rl.selfplay import DEFAULT_ROSTER

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Mapping, Sequence

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
    row. With ``maps`` given, the eval spreads each opponent's budget across those arenas (per-cell
    pinned phases) and additionally logs ``eval/win_rate/map/<short-map-name>`` per arena; all
    scalars are pooled from the same per-cell counts, so the marginals reconcile.

    Args:
        eval_freq: minimum env-steps between evaluations (gated against the last eval's timestep at
            each rollout boundary). ``<= 0`` disables eval.
        eval_episodes: number of greedy episodes per opponent per evaluation (distributed across the
            M eval envs, and — with ``maps`` — spread deterministically across the eval maps).
        opponents: the roster of ``agents`` selector strings to evaluate against (default
            :data:`~pop_trainer.rl.selfplay.DEFAULT_ROSTER`).
        maps: the eval map rotation (arena-target strings) — normally the TRAINING rotation, so
            eval covers the same arenas training plays. ``None`` (default) = single-arena eval:
            no ``switch_arena`` is ever sent, no per-map scalar is emitted, and the eval envs'
            ``maps`` attribute is never touched (byte-identical to the no-rotation behavior).
        seed: optional seed passed to ``evaluate_winrate`` (opponent reproducibility across evals).
        eval_env: the eval vec env (``M`` envs; ``DummyVecEnv`` at M=1, ``SubprocVecEnv`` at M>1).
            ``evaluate_winrate`` re-pins each wrapper's providers per phase via ``set_attr``.
            Its instances are spawned for eval and torn down (``env_method("release")``) afterward.
        training_vec: the TRAINING vec env handle — the SAME vec env SB3 reads ``_last_obs`` from
            (the integrator passes ``model.env``, i.e. SB3's ``VecTransposeImage``-wrapped training
            stack, so the respawn ``reset()`` returns obs in the exact space ``_last_obs`` expects).
            Torn down (``env_method("release")``) BEFORE eval instances spawn and respawned
            (``reset()``) AFTER eval. Passed in explicitly rather than via ``self.model.get_env()``
            so the handle is unambiguous.
        result_sink: optional callable receiving each completed eval cycle's structured
            ``evaluate_winrate`` result. This is the matchup curriculum's win-rate feed:
            ``train_local`` wires :meth:`MatchupSamplingCallback.submit_eval_result` here when
            ``--matchup-sampling winrate`` is on. ``None`` (the default) changes nothing —
            the logging path is identical either way.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int = 10,
        opponents=DEFAULT_ROSTER,
        maps: Sequence[str] | None = None,
        seed: int | None = None,
        eval_env: VecEnv | None = None,
        training_vec: VecEnv | None = None,
        result_sink: Callable[[Mapping], None] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponents = tuple(opponents)
        self.maps = None if maps is None else tuple(maps)
        self.seed = seed
        self.eval_env = eval_env
        self.training_vec = training_vec
        self.result_sink = result_sink
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

        result = self._run_eval_cycle()
        if self.result_sink is not None:
            # Hand the cycle's structured result to the matchup curriculum (its ONLY win-rate
            # feed). The fold + broadcast happen at THIS same rollout boundary: the matchup
            # callback runs after this one in train_local's CallbackList.
            self.result_sink(result)
        if self.maps is None:
            # Single-arena eval: result is the per-selector win-rate dict, logged exactly as
            # before (no per-map scalar exists to emit).
            per_opponent = result
            overall = overall_win_rate(per_opponent)
        else:
            # Rotation eval: result is per-(opponent x map) cell counts. Pool the marginals from
            # the SAME counts (never a mean of means) so per-opponent, per-map, and overall
            # reconcile; emit one scalar per MAP, never one per cell.
            per_opponent = pool_by_opponent(result)
            overall = pooled_win_rate(result)
            for target, rate in pool_by_map(result).items():
                self.logger.record(f"eval/win_rate/map/{map_short_name(target)}", rate)

        # Log per-opponent and overall into the model's existing logger (-> progress.csv AND
        # tfevents). Then dump so the row lands at this timestep.
        for selector, rate in per_opponent.items():
            self.logger.record(f"eval/win_rate/{selector}", rate)
        self.logger.record("eval/win_rate", overall)
        self.logger.dump(self.num_timesteps)

    def _run_eval_cycle(self):
        """Teardown training -> parallel eval -> teardown eval -> respawn training (the INVARIANT).

        Returns ``evaluate_winrate``'s result: the per-opponent win-rate dict (``maps is None``)
        or the per-(opponent x map) cell counts. The order is load-bearing: the eval instances are
        spawned ONLY after every training instance is reaped, and the training instances are
        respawned ONLY after every eval instance is reaped, so the two sets never coexist.

        NOTE (eval arena switching): the build handles ``switch_arena`` only while ``!ingame``, so
        the ONLY safe channel is the reset handshake — ``TankEnv`` sends ``switch_arena`` solely
        when the caller passes ``reset(options={"switch_arena": ...})``. Rotation eval uses exactly
        that channel, DELIBERATELY: ``evaluate_winrate`` pins ONE arena per (opponent, map) phase
        (a single-map provider set on each eval wrapper), so every reset in the phase injects the
        same known target at reset time, while ``!ingame``. That is safe where the historical
        uncontrolled rotation was not: eval envs once carrying a free-running rotation switched to
        an arbitrary arena at every auto-reset, unaccounted by the eval math, desyncing the run at
        the rollout boundary. With ``maps is None`` no ``switch_arena`` is ever sent (plain
        ``reset()``s, byte-identical to before).
        """
        # 1. Tear down ALL training Unity instances (frees their ports; workers stay alive).
        self.training_vec.env_method("release")

        try:
            # 2. Parallel eval across the M eval instances (lazy-launched on first reset/step).
            result = evaluate_winrate(
                self.model,
                self.eval_env,
                opponents=self.opponents,
                n_episodes=self.eval_episodes,
                seed=self.seed,
                maps=self.maps,
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

        return result


class MatchupSamplingCallback(BaseCallback):
    """Steer the joint (opponent x map) sampler from EVAL win-rates; count training episodes.

    The main-process half of ``--matchup-sampling winrate``. The per-env
    :class:`~pop_trainer.rl.selfplay.MatchupProvider`s draw each episode's (opponent, arena) cell
    from a distribution this callback owns. Two strictly separated data flows:

    * ``win_rates`` — WHAT EVAL MEASURED. :meth:`submit_eval_result` (wired as
      :class:`EvalWinRateCallback`'s ``result_sink`` in ``train_local``) queues each completed
      eval cycle's structured ``evaluate_winrate`` result; ``_on_rollout_end`` folds the queued
      per-cell rates into the win-rate table (:func:`~pop_trainer.rl.matchup.fold_eval_rates`),
      recomputes ``P(cell) = floor * uniform + (1 - floor) * normalize(1 - wr)``
      (:func:`~pop_trainer.rl.matchup.deficit_distribution`), BROADCASTS the plain-data
      distribution to every training env via ``training_vec.set_attr("matchup_distribution",
      ...)``, records the SB3 scalars, and logs ONE ``matchup_update`` summary. Because the eval
      callback runs FIRST in the ``CallbackList`` and this one LAST, the fold lands at the SAME
      rollout boundary the eval cycle completed at — never mid-rollout. Between eval cycles the
      distribution is FROZEN: no fold, no broadcast, no log. Before the first eval cycle every
      cell sits at the 0.5 prior, so the distribution is exactly uniform.
    * ``counts`` — WHAT TRAINING PLAYED. ``_on_step`` consumes the ``info["matchup"]`` tags the
      training ``SelfPlayWrapper``s emit on terminal steps and ``_on_rollout_end`` folds them
      into the per-cell play counts at every boundary — pure observability of what the sampler
      actually sampled. Training outcomes NEVER move ``win_rates``: the training-side win-rate
      measures the STOCHASTIC policy, and at high entropy it inverts against the deterministic
      eval (training can report a noop win-rate near 0.9 while eval reports near 0.1), steering
      the curriculum away from the real weakness.

    ``_on_training_start`` broadcasts once up-front so a ``--resume``-restored table reaches the
    freshly-built (uniform-initialized) providers before the first rollout.

    The win-rate table lives HERE, in one process, so it is position-exact at ANY ``n_envs`` (the
    per-env sampling RNGs remain per-subproc and reseed on resume, like the existing providers).
    :meth:`state` / :meth:`restore` are the sidecar seam: a plain-JSON dict of cells, win-rates,
    and play counts that continues the curriculum across a resume.

    Observability: at each post-eval update ONE compact summary lands on the system logger (the
    ``worst_k`` lowest-win-rate cells + the distribution entropy + play counts), plus two
    aggregate SB3 scalars (``matchup/distribution_entropy``, ``matchup/episodes``) — never one
    scalar per cell.

    Args:
        opponents: the roster selector strings (the cell rows).
        maps: the arena rotation (the cell columns), or ``None`` for boot-arena cells.
        floor: the exploration floor ``eps`` in ``[0, 1]`` (min cell prob ``floor / n_cells``).
        ema_alpha: the EMA weight of the newest eval cycle's per-cell rate, in ``(0, 1]``. The
            default 0.4 is sized for per-EVAL-CYCLE folds of ~10-episode cell estimates (a few
            cycles dominate the table); a per-episode-scale alpha like 0.05 would need ~20 eval
            cycles to move a cell off its prior.
        training_vec: the TRAINING vec env handle (the integrator passes ``model.env``); the
            ``set_attr`` broadcast target. ``None`` skips broadcasting (pure aggregation).
        sys_logger: optional system logger for the post-eval summary record.
        worst_k: how many lowest-win-rate cells the summary names.
        verbose: SB3 verbosity passed to ``BaseCallback``.
    """

    def __init__(
        self,
        opponents: Sequence[str],
        maps: Sequence[str] | None = None,
        *,
        floor: float = 0.25,
        ema_alpha: float = 0.4,
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
        # Cell indices of the training episodes terminated within the CURRENT rollout (feeds
        # counts only).
        self._pending: list[int] = []
        # Normalized per-cell rate dicts of the eval cycles completed since the last fold
        # (feeds win_rates; normally at most one — eval runs at the same boundary cadence).
        self._pending_eval: list[dict] = []

    @property
    def distribution(self) -> list[float]:
        """The current sampling distribution over :attr:`cells` (from the win-rate table)."""
        return deficit_distribution(self.win_rates, self.floor)

    def state(self) -> dict:
        """The plain-JSON resumable state the sidecar persists (cells + win-rates + counts).

        ``signal: "eval"`` marks the win-rate feed so a sidecar reader can tell the table holds
        deterministic eval measurements; :meth:`restore` tolerates its absence.
        """
        return {
            "sampling": "winrate",
            "signal": "eval",
            "floor": float(self.floor),
            "ema_alpha": float(self.ema_alpha),
            "cells": [[selector, arena] for selector, arena in self.cells],
            "win_rates": [float(wr) for wr in self.win_rates],
            "counts": [int(c) for c in self.counts],
        }

    def restore(self, state: dict) -> None:
        """Continue a persisted curriculum: restore each cell's win-rate + count by cell key.

        Matching is by cell tag, so a resume with a changed roster / rotation keeps the cells
        that still exist and leaves new cells at the 0.5 prior. A malformed / empty block is a
        no-op. A block without the ``signal`` field (an older sidecar whose table was fed from
        training outcomes) restores fine — the next eval folds overwrite it.
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

    def submit_eval_result(self, result: Mapping) -> None:
        """Queue one completed eval cycle's structured ``evaluate_winrate`` result (the signal).

        Called by :class:`EvalWinRateCallback` (the ``result_sink`` wiring) right after each
        eval cycle. The result is normalized to per-cell rates here
        (:func:`~pop_trainer.rl.matchup.eval_cell_rates` handles both return shapes — per-cell
        counts under a map rotation, per-opponent rates onto ``(selector, None)`` boot-arena
        cells without one) and folded at the next ``_on_rollout_end`` — the same boundary, given
        the ``CallbackList`` order — so the distribution never changes mid-rollout.
        """
        self._pending_eval.append(eval_cell_rates(result))

    def _broadcast(self) -> None:
        """Push the current distribution into every training env's provider (plain data only)."""
        if self.training_vec is not None:
            self.training_vec.set_attr("matchup_distribution", self.distribution)

    def _on_training_start(self) -> None:
        """Sync the providers to the callback's table before the first rollout (the resume seam).

        On a fresh run this re-sends the uniform distribution the providers already hold (a
        no-op in effect); after a ``--resume`` restore it is what carries the persisted
        curriculum into the freshly-built providers.
        """
        self._broadcast()

    def _on_step(self) -> bool:
        """Accumulate terminal training-episode cells — NEVER change the distribution here."""
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
            self._pending.append(index)
        return True

    def _on_rollout_end(self) -> None:
        """Fold counts every boundary; fold + broadcast win-rates ONLY after an eval cycle.

        The play counts absorb the rollout's terminal tags unconditionally (cheap
        observability). The win-rate table, the distribution broadcast, the SB3 scalars, and
        the one-line ``matchup_update`` summary move ONLY when an eval cycle completed at this
        boundary (the eval callback ran first and queued its result); otherwise the
        distribution stays frozen exactly as the workers last received it.
        """
        for index in self._pending:
            self.counts[index] += 1
        self._pending.clear()

        if not self._pending_eval:
            return

        cells_measured = 0
        for cell_rates in self._pending_eval:
            cells_measured += len(cell_rates)
            self.win_rates = fold_eval_rates(self.win_rates, self.cells, cell_rates, self.ema_alpha)
        eval_folds = len(self._pending_eval)
        self._pending_eval.clear()

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
                        "eval_folds": eval_folds,
                        "cells_measured": cells_measured,
                        "total_episodes": sum(self.counts),
                        "entropy": entropy,
                        "worst_cells": [
                            {"opponent": selector, "map": arena, "win_rate": wr}
                            for (selector, arena), wr in worst
                        ],
                    }
                },
            )
