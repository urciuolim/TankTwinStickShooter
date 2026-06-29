"""``train_local`` — the Phase-1 PPO integrator (compose T2-T4 into a runnable train loop).

This module is the INTEGRATOR: it composes the already-built RL seams into one SB3 PPO run over
the pixel env, with self-play opponents, periodic eval, checkpointing, and a resumable sidecar.
Nothing here re-implements those seams — it WIRES them:

* :class:`~pop_trainer.rl.extractor.EncoderExtractor` is handed to the ``CnnPolicy`` as the
  features extractor (it owns the pretrained-encoder checkpoint load + the freeze path);
* :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` +
  :class:`~pop_trainer.rl.selfplay.OpponentProvider` drive player2 behind the learner;
* :class:`~pop_trainer.rl.callbacks.EvalWinRateCallback` logs periodic per-opponent win-rate
  against a DEDICATED eval env (a second build/socket) — the training env is never touched;
* :func:`~pop_trainer.rl.evaluate.evaluate_winrate` +
  :func:`~pop_trainer.rl.evaluate.format_per_map_table` produce a final per-opponent summary line;
* :mod:`~pop_trainer.rl.elo` holds the ELO math the sidecar persists.

The env LAUNCH seam (the crux): :class:`~pop_trainer.env.tank_env.TankEnv` does NOT launch Unity —
it takes an injected ``connection`` / ``connection_factory``. The LIVE launch lives in ``data``,
which ``rl`` must NOT import (boundary). So the live ``connection_factory`` is re-derived HERE from
the SHARED, dependency-free :mod:`pop_trainer.core.launch` primitives (``rl -> core`` is allowed):
``build_launch_cmd`` + ``subprocess.Popen`` + ``connect`` + ``core.protocol.Connection``, with the
``Popen`` stashed so ``env.close()`` reaps the build (terminate -> wait -> kill). The unit tests
inject a STUB ``connection_factory`` so NO Unity is launched.

Multi-env (Phase 2): at ``n_envs > 1`` the training env is a ``SubprocVecEnv`` of ``n_envs`` Unity
builds, each on its OWN port (``game_port + i``) in its OWN process (``start_method="spawn"`` —
Windows-safe). The env factories are CLOSURES capturing only ``cfg`` + the int ``i`` and building
everything live INSIDE the subprocess (cloudpickle ships them), mirroring collection's per-worker
fan-out. A pre-flight memory guard sizes the PPO RolloutBuffer + the live Unity instances against
available RAM and aborts before launch if over budget.

Boundary: imports ``core`` (incl. ``core.launch`` / ``core.protocol`` / ``core.config``), ``env``,
``models`` / ``agents`` (transitively, via the extractor / self-play seams), ``psutil``, sb3 /
gymnasium / stdlib. Imports NOTHING from ``data`` or ``pretraining`` (the live launch is re-derived
from ``core.launch``, never imported from ``data``; the memory guard is the rl-side equivalent of
collection's, not an import of it). No cycles.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pop_trainer.agents import AGENT_SELECTORS
from pop_trainer.core import launch
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.core.logging_setup import (
    LOG_LEVEL_ENV_VAR,
    ROLE_EVAL,
    ROLE_TRAIN,
    level_from_env,
    setup_env_logger,
    setup_system_logger,
    unity_log_path,
)
from pop_trainer.core.obs import (
    DEFAULT_FRAME_SHAPE,
    frame_shape_from_config,
    validate_frame_shape,
)
from pop_trainer.core.protocol import Connection
from pop_trainer.env.tank_env import TankEnv
from pop_trainer.rl.elo import elo_change
from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import _STRATEGIES, DEFAULT_ROSTER, OpponentProvider, SelfPlayWrapper

if TYPE_CHECKING:  # type-only: keep the pure helpers / config import-light at module load
    from collections.abc import Callable, Sequence

    from stable_baselines3.common.vec_env import VecEnv

__all__ = [
    "TrainConfig",
    "train_local",
    "build_vec_env",
    "save_sidecar",
    "load_sidecar",
    "main",
    "frame_nbytes",
    "estimate_rl_memory_bytes",
    "check_rl_memory_budget",
    "training_ports",
    "eval_ports",
]

# This file is src/pop_trainer/rl/train.py: parents[3] is the repo root
# (core sibling -> rl -> pop_trainer -> src -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The default game/training config forwarded to the build launch (single-map, obs_pixels=True,
# timeScale<=5). Lives next to the other StreamingAssets configs.
DEFAULT_TRAIN_CONFIG = _REPO_ROOT / "unity" / "Assets" / "StreamingAssets" / "train_config.json"

# The channels-LAST (H, W, 3) pixel frame is DERIVED from the launched game config's obs_pixels_*
# (the one source of truth the Unity build also reads); core.DEFAULT_FRAME_SHAPE (640x360, the
# DriverController default) is only the fallback when a config declares no obs_pixels_* keys. SB3
# auto-applies VecTransposeImage so the extractor sees (N, 3, H, W).

# Phase-1 round cap (1-min / 300-step rounds — intentional, see the milestone notes).
DEFAULT_MAX_STEPS = 300

# Every roster opponent starts at this rating; Phase-1 persists the ELO structure so resume can
# restore it (a simple from-eval update is wired below but optional).
BASE_ELO = 1000.0

# The sidecar file written alongside each model_<steps>.zip checkpoint.
SIDECAR_NAME = "state.json"

# Valid PPO learning-rate schedules: "constant" passes the float lr unchanged; "linear" decays it
# to 0 over training via SB3's progress_remaining callable. Default "constant" == today's behavior.
LR_SCHEDULES = ("constant", "linear")

# Valid encoder-trunk overrides threaded to EncoderExtractor: "auto" keeps the size-based selection
# (today's behavior); the rest are explicit registry trunks. Kept in sync with models.TRUNKS.
TRUNK_CHOICES = ("auto", "cnn", "resnet", "gn-cnn")

# The explicit policy/value MLP-head default. SB3 silently uses [64, 64] when net_arch is unset;
# making it explicit keeps today's behavior while letting the CLI override it.
DEFAULT_NET_ARCH = [64, 64]

# --- multi-env rollout-buffer memory guard ---------------------------------------------------
# At n_envs > 1 the SB3 PPO RolloutBuffer is the OOM surface: it allocates
# (n_steps, n_envs, *obs_shape) of the obs dtype. The pixel obs is uint8 (tank_env: dtype=uint8),
# so the buffer stores uint8 — 1 byte/element — but n_steps x n_envs x frame is still large
# (e.g. 2048 x 7 x 691200 B ~ 9.2 GB). VecFrameStack(n_stack=k) multiplies the stored channel
# depth by k, so the per-element obs cost scales with frame_stack. We pre-flight this against the
# box's available RAM exactly as collection pre-flights its per-worker shard buffer.

# Coarse per-Unity-instance RAM allowance the rollout-buffer estimate does NOT model. Each live
# Unity build is its own process (~1 GB). The eval and training instance SETS never coexist (eval
# tears down training first, see EvalWinRateCallback), so peak concurrent instances is
# max(n_envs, n_envs) = n_envs, NOT n_envs + 1. Folded into the total so the guard reasons about
# builds + buffer, not just the buffer.
UNITY_INSTANCE_BYTES = 1024**3  # ~1 GB per live Unity instance (coarse).

# The estimate excludes the OS + torch/CUDA + the policy net, so the guard only lets the modelled
# total consume a conservative fraction of available RAM (mirrors collection's MARGIN).
MEMORY_MARGIN = 0.6


@dataclass(frozen=True)
class TrainConfig:
    """A single local PPO self-play training run (frozen; validated in ``__post_init__``).

    Fields split into the run topology, the encoder wiring, the self-play roster, the
    eval/checkpoint cadence, the resume hook, and the PPO hyperparameters.

    Cadence defaults (``eval_freq`` / ``checkpoint_freq``) are ``10_000`` env-steps — a
    production-scale cadence, NOT smoke-small. A short smoke run overrides them DOWNWARD so a
    few-thousand-step run still triggers at least one checkpoint + one eval.

    Args:
        total_timesteps: total env-steps to train (must be ``> 0``).
        game_config: the game/training config JSON forwarded to the build launch as ``--config``
            (single-map, ``obs_pixels=True``, ``timeScale<=5``).
        run_dir: the output directory for checkpoints / sidecar / tensorboard logs.
        n_envs: number of parallel TRAINING envs. ``1`` -> ``DummyVecEnv`` (single in-process env,
            position-exact ``round_robin`` resume). ``>1`` -> ``SubprocVecEnv`` of ``n_envs`` Unity
            builds, each on its OWN port (``game_port + i``). At ``>1`` the opponent rotation is
            PER-SUBPROC (each subproc has its own seeded provider) and resume RESEEDS it
            (approximate phase, like ``uniform``) — see :func:`build_vec_env`.
        frame_stack: VecFrameStack depth (``1`` = passthrough).
        encoder_checkpoint: optional pretrained-encoder ``state_dict`` path for the extractor.
        freeze_encoder: freeze the encoder weights during RL.
        opponents: the self-play roster (``agents`` selector strings; non-empty).
        opponent_strategy: ``"round_robin"`` (resumable) or ``"uniform"`` (seed-only resume).
        eval_freq: env-steps between periodic win-rate evals (``>= 0``; ``0`` disables).
        eval_episodes: greedy episodes per opponent per eval.
        checkpoint_freq: env-steps between checkpoints (``> 0``; the sidecar rides this cadence).
        resume: a prior ``run_dir`` to resume from (load the latest ``model_*.zip`` + sidecar).
        seed: the master seed (threaded into SB3, the env, and the opponent provider).
        learning_rate / n_steps / batch_size / n_epochs / gamma / gae_lambda / clip_range /
        ent_coef / vf_coef / max_grad_norm:
            PPO hyperparameters (the defaults keep SB3's own defaults exact, so an existing run
            reproduces bit-for-bit).
        lr_schedule: ``"constant"`` (pass ``learning_rate`` as a float, the default) or
            ``"linear"`` (decay ``learning_rate`` to 0 over training via SB3's
            ``progress_remaining`` schedule callable).
        net_arch: the policy/value MLP-head architecture passed through ``policy_kwargs``. ``None``
            (the default) becomes SB3's implicit ``[64, 64]`` made EXPLICIT; an SB3 ``net_arch``
            value (e.g. ``dict(pi=[64, 64], vf=[64, 64])``) overrides it.
        trunk: the encoder trunk override threaded to the :class:`EncoderExtractor`. ``"auto"``
            (the default) keeps the extractor's size-based selection; an explicit ``"cnn"`` /
            ``"resnet"`` / ``"gn-cnn"`` forces that trunk.
        frame_shape: the ``(H, W, 3)`` pixel-frame shape (channels-last; SB3 transposes it).
        game_port: the BASE TCP port. Training env ``i`` listens on ``game_port + i`` for
            ``i in 0..n_envs-1`` (``args[1]`` of each build's launch arg-list). The training BLOCK
            is ``[game_port, game_port + n_envs - 1]``.
        eval_port: the BASE TCP port for the eval block. There are ``n_envs`` eval builds
            (``M_eval == N_train``), each listening on ``eval_port_base + i`` for
            ``i in 0..n_envs-1``, so the eval BLOCK is
            ``[eval_port_base, eval_port_base + n_envs - 1]``. ``None`` (default) ->
            ``eval_port_base = game_port + n_envs`` (the eval block sits entirely AFTER the training
            block: ``[game_port + n_envs, game_port + 2*n_envs - 1]``, so at ``n_envs == 1`` it is
            ``game_port + 1`` as before). When set it must differ from ``game_port`` AND the WHOLE
            eval block must NOT overlap the training block (no eval build can collide with any
            training build). The eval and training builds NEVER run concurrently (the eval callback
            tears training down before spawning eval), but the disjoint port blocks keep a
            relaunched-but-not-yet-reaped instance from clashing on bind.
        allow_oversized: skip the pre-flight rollout-buffer memory ABORT (the estimate is still
            printed). Use only when the box has RAM the conservative guard does not model.
        build_path: the build binary; ``None`` -> :func:`core.launch.default_build_path`.
        log_dir: the directory the per-process observability logs (``training-system.log`` +
            ``env-<role>-<port>.log`` + the paired Unity ``unity-<role>-<port>.log``) are written
            to. ``None`` (default) -> ``run_dir / "logs"``. PURELY observational.
        debug_logging: crank ALL observability logs to DEBUG (the single switch; default ``False``
            = INFO). The CLI ``--debug`` flag / the ``POP_LOG_LEVEL`` env var set this.
    """

    total_timesteps: int
    game_config: Path
    run_dir: Path
    n_envs: int = 1
    frame_stack: int = 1
    encoder_checkpoint: Path | None = None
    freeze_encoder: bool = False
    opponents: tuple[str, ...] = DEFAULT_ROSTER
    opponent_strategy: str = "round_robin"
    eval_freq: int = 10_000
    eval_episodes: int = 10
    checkpoint_freq: int = 10_000
    resume: Path | None = None
    seed: int = 0
    log_dir: Path | None = None
    debug_logging: bool = False
    # PPO hyperparameters. Every default keeps SB3's own default exact, so a TrainConfig built
    # with no new flags reproduces today's run bit-for-bit.
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_schedule: str = "constant"
    # The policy/value MLP-head architecture. None at construction -> SB3's implicit [64, 64]
    # made EXPLICIT in __post_init__; otherwise an SB3 net_arch value (e.g.
    # dict(pi=[64, 64], vf=[64, 64])).
    net_arch: dict | list | None = None
    # The encoder trunk override threaded to EncoderExtractor ("auto" = the size rule).
    trunk: str = "auto"
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE
    game_port: int = 50000
    eval_port: int | None = None
    allow_oversized: bool = False
    build_path: Path | None = None

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {self.total_timesteps}")
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {self.n_envs}")
        if self.frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {self.frame_stack}")
        if self.opponent_strategy not in _STRATEGIES:
            valid = ", ".join(_STRATEGIES)
            raise ValueError(
                f"unknown opponent_strategy {self.opponent_strategy!r}; choose one of: {valid}"
            )
        if not self.opponents:
            raise ValueError("opponents must be non-empty")
        if self.eval_freq < 0:
            raise ValueError(f"eval_freq must be >= 0, got {self.eval_freq}")
        if self.checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be > 0, got {self.checkpoint_freq}")
        if len(self.frame_shape) != 3 or self.frame_shape[2] != 3:
            raise ValueError(f"frame_shape must be (H, W, 3), got {self.frame_shape!r}")
        if self.eval_port is not None and self.eval_port == self.game_port:
            raise ValueError(
                f"eval_port must differ from game_port (both {self.game_port}); the eval block "
                "binds its own sockets"
            )
        # The eval BLOCK [eval_port, eval_port + n_envs - 1] must not overlap the training block
        # [game_port, game_port + n_envs - 1] — no eval build may collide with a training build.
        # Two half-open ranges [a, a+n) and [b, b+n) overlap iff a < b+n and b < a+n.
        if self.eval_port is not None:
            train_lo, train_hi = self.game_port, self.game_port + self.n_envs - 1
            eval_lo, eval_hi = self.eval_port, self.eval_port + self.n_envs - 1
            if eval_lo <= train_hi and train_lo <= eval_hi:
                raise ValueError(
                    f"the eval port block [{eval_lo}, {eval_hi}] overlaps the training port block "
                    f"[{train_lo}, {train_hi}] (n_envs={self.n_envs}); no eval build can collide "
                    "with a training build — choose an eval_port whose block is disjoint (e.g. "
                    f">= {train_hi + 1})"
                )
        if self.lr_schedule not in LR_SCHEDULES:
            valid = ", ".join(LR_SCHEDULES)
            raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}; choose one of: {valid}")
        if self.trunk not in TRUNK_CHOICES:
            valid = ", ".join(TRUNK_CHOICES)
            raise ValueError(f"unknown trunk {self.trunk!r}; choose one of: {valid}")
        # Make SB3's silent [64, 64] head EXPLICIT when net_arch is unset (frozen -> setattr).
        if self.net_arch is None:
            object.__setattr__(self, "net_arch", list(DEFAULT_NET_ARCH))

    @property
    def effective_eval_port(self) -> int:
        """The BASE eval port (the eval block's low end): ``eval_port`` or ``game_port + n_envs``.

        The default places the eval block ENTIRELY after the training block
        ``[game_port, game_port + n_envs - 1]`` — eval is ``[game_port + n_envs, game_port +
        2*n_envs - 1]`` (so 7 train envs = ``50000..50006`` -> eval ``50007..50013``). At
        ``n_envs == 1`` this is ``game_port + 1`` exactly as before. The full eval port list is
        :func:`eval_ports`.
        """
        return self.eval_port if self.eval_port is not None else self.game_port + self.n_envs

    @property
    def effective_log_dir(self) -> Path:
        """The observability log directory: ``log_dir`` or, when ``None``, ``run_dir / "logs"``."""
        return self.log_dir if self.log_dir is not None else self.run_dir / "logs"

    @property
    def log_level(self) -> int:
        """The observability log level: ``DEBUG`` when ``debug_logging`` else ``INFO``."""
        return logging.DEBUG if self.debug_logging else logging.INFO

    def to_dict(self) -> dict:
        """JSON-ready plain-dict view (Paths -> str, tuples -> list)."""
        return {
            "total_timesteps": self.total_timesteps,
            "game_config": str(self.game_config),
            "run_dir": str(self.run_dir),
            "n_envs": self.n_envs,
            "frame_stack": self.frame_stack,
            "encoder_checkpoint": (
                None if self.encoder_checkpoint is None else str(self.encoder_checkpoint)
            ),
            "freeze_encoder": self.freeze_encoder,
            "opponents": list(self.opponents),
            "opponent_strategy": self.opponent_strategy,
            "eval_freq": self.eval_freq,
            "eval_episodes": self.eval_episodes,
            "checkpoint_freq": self.checkpoint_freq,
            "resume": None if self.resume is None else str(self.resume),
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "lr_schedule": self.lr_schedule,
            "net_arch": self.net_arch,
            "trunk": self.trunk,
            "frame_shape": list(self.frame_shape),
            "game_port": self.game_port,
            "eval_port": self.eval_port,
            "allow_oversized": self.allow_oversized,
            "build_path": None if self.build_path is None else str(self.build_path),
            "log_dir": None if self.log_dir is None else str(self.log_dir),
            "debug_logging": self.debug_logging,
        }


# --- pure rollout-buffer memory guard (no psutil here; available_bytes is INJECTED) ----------


def frame_nbytes(frame_shape: Sequence[int]) -> int:
    """The uncompressed byte size of ONE uint8 frame of ``frame_shape`` (the element product).

    uint8 is 1 byte/element, so a frame costs exactly ``prod(frame_shape)`` bytes. This is the
    per-(step, env) obs cost the rollout-buffer estimate multiplies up.
    """
    n = 1
    for dim in frame_shape:
        n *= int(dim)
    return n


def estimate_rl_memory_bytes(
    *, n_steps: int, n_envs: int, frame_nbytes: int, frame_stack: int
) -> int:
    """Estimate the SB3 PPO RolloutBuffer's peak obs bytes (PURE; no allocation).

    The buffer is ``(n_steps, n_envs, *obs_shape)`` of the uint8 obs dtype, and
    ``VecFrameStack(n_stack=frame_stack)`` multiplies the stored channel depth by ``frame_stack``.
    So the obs bytes are ``n_steps * n_envs * frame_nbytes * frame_stack``. (The buffer's rewards /
    returns / values / log-probs are O(n_steps * n_envs) floats — negligible beside the frames — so
    they are not modelled.) This is the quantity the pre-flight guard compares against RAM.
    """
    return int(n_steps) * int(n_envs) * int(frame_nbytes) * int(frame_stack)


def check_rl_memory_budget(
    *,
    n_steps: int,
    n_envs: int,
    frame_nbytes: int,
    frame_stack: int,
    available_bytes: int,
    allow_oversized: bool = False,
) -> str:
    """Pre-flight RL memory guard. Returns the estimate line; may raise ``MemoryError``.

    Adds the rollout-buffer obs estimate (:func:`estimate_rl_memory_bytes`) to a coarse Unity
    allowance of :data:`UNITY_INSTANCE_BYTES` per live instance. Peak concurrent instances is
    ``max(N_train, M_eval) = max(n_envs, n_envs) = n_envs``: the ``M_eval == n_envs`` eval builds
    NEVER coexist with the ``n_envs`` training builds (the eval callback tears all training builds
    down before spawning eval, and tears eval down before respawning training — see
    :class:`~pop_trainer.rl.callbacks.EvalWinRateCallback`). So the guard charges ``n_envs`` Unity
    instances, not ``n_envs + 1``. The TOTAL is compared to a conservative fraction
    (:data:`MEMORY_MARGIN`) of ``available_bytes`` (INJECTED by the caller — this pure function
    never touches psutil). When the total exceeds the threshold and ``allow_oversized`` is ``False``
    it raises ``MemoryError`` with an ACTIONABLE message (lower ``--n-steps`` or ``--n-envs``, or
    pass ``--allow-oversized``). Within budget — or overridden — it returns the estimate line so the
    caller can ALWAYS print it at startup.
    """
    buffer_bytes = estimate_rl_memory_bytes(
        n_steps=n_steps, n_envs=n_envs, frame_nbytes=frame_nbytes, frame_stack=frame_stack
    )
    # Peak concurrent Unity instances = max(N_train, M_eval) = n_envs (the sets never coexist).
    unity_instances = n_envs
    unity_bytes = unity_instances * UNITY_INSTANCE_BYTES
    total = buffer_bytes + unity_bytes
    threshold = int(available_bytes * MEMORY_MARGIN)
    gib = 1024**3
    line = (
        f"memory estimate: rollout buffer ~{buffer_bytes / gib:.2f} GB "
        f"(n_steps={n_steps} x n_envs={n_envs} x {frame_nbytes / 1024**2:.2f} MB/frame "
        f"x {frame_stack} frame_stack) + {unity_instances} Unity instance(s) "
        f"~{unity_bytes / gib:.2f} GB = total ~{total / gib:.2f} GB; "
        f"available {available_bytes / gib:.2f} GB, "
        f"budget {MEMORY_MARGIN:.0%} = {threshold / gib:.2f} GB"
    )
    if total > threshold and not allow_oversized:
        raise MemoryError(
            f"{line}. ABORT: estimated total ~{total / gib:.2f} GB exceeds the "
            f"{MEMORY_MARGIN:.0%} budget ({threshold / gib:.2f} GB of "
            f"{available_bytes / gib:.2f} GB available). lower --n-steps or --n-envs "
            "(or pass --allow-oversized to override)."
        )
    return line


# --- live launch seam (re-derived from core.launch; NEVER imported from data) ----------------


def _terminate(proc: subprocess.Popen) -> None:
    """Reap the build subprocess, escalating ``terminate -> wait -> kill`` on a stuck exit.

    Idempotent: a no-op once the process has already exited (``poll`` returns non-``None``).
    Mirrors the reap policy the collection runner uses, re-derived here so ``rl`` does not import
    ``data``.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


def _attempt_unity_log_path(base_log_dir, role: str, port: int, attempt: int):
    """The per-LAUNCH Unity ``-logFile`` path: ``unity-<role>-<port>-<attempt>.log`` (PURE).

    Unity's ``-logFile`` TRUNCATES its target on every launch, so a relaunch (lazy respawn /
    reconnect) reusing ONE path would wipe the prior (possibly hung) instance's C# log. Suffixing a
    monotonically incrementing ``attempt`` counter gives each launch within a factory a DISTINCT
    file (``...-0.log`` for the first launch, ``...-1.log`` for the first relaunch, ...), so a prior
    instance's log survives for post-mortem. Built off the shared :func:`unity_log_path` stem so the
    ``(role, port)`` pairing with the Python ``env-<role>-<port>.log`` is preserved.
    """
    stem = unity_log_path(base_log_dir, role, port)  # .../unity-<role>-<port>.log
    return stem.with_name(f"{stem.stem}-{attempt}{stem.suffix}")


def _live_connection_factory_for_port(
    cfg: TrainConfig, port: int, *, role: str = ROLE_TRAIN, logger: logging.Logger | None = None
) -> Callable[[], Connection]:
    """Build the LIVE ``connection_factory`` for ``port``: launch the build, connect, wrap socket.

    Returns a zero-arg callable the env invokes on the first ``reset``/``step`` (LAZY launch) and
    re-invokes on every reconnect / post-``release`` respawn. Each call: launches the build via
    :func:`core.launch.build_launch_cmd` + ``subprocess.Popen`` on ``port``, connects via
    :func:`core.launch.connect`, and wraps the socket in a :class:`core.protocol.Connection`
    carrying the SHARED per-connection ``logger`` (so protocol + env records for this socket land
    in one ``env-<role>-<port>.log``). The live ``Popen`` is stashed on the returned ``Connection``
    (``conn._launch_proc``) so the env's injected reap hook (``reap=_terminate``) can hard-kill it
    on :meth:`TankEnv.release` / the kill-old-first reconnect.

    Per-launch logfile (no truncation across respawns): a launch counter is captured in the closure
    and incremented every invocation, so each launch points Unity's ``-logFile`` at a DISTINCT
    ``unity-<role>-<port>-<attempt>.log`` (:func:`_attempt_unity_log_path`). A respawn therefore
    NEVER truncates the prior instance's C# log.

    The build is launched WINDOWED (never batchmode) on ``cfg.game_config``. Parameterizing the
    ``port`` lets each TRAINING / EVAL env launch its OWN build on its OWN socket. Re-derived from
    the shared ``core.launch`` primitives — ``rl`` does NOT import the ``data`` launch path.

    Observability: ``role`` tags the Unity log filename; ``logger`` (when given) is threaded into
    the ``Connection`` for wire-level evidence. Both default off — the no-logger path is identical.
    """
    exe = (
        cfg.build_path
        if cfg.build_path is not None
        else launch.default_build_path(_REPO_ROOT / "unity")
    )
    # Mutable launch counter captured by the closure: each (lazy launch / reconnect / respawn) gets
    # a distinct -logFile so no prior instance's log is truncated.
    attempt = {"n": 0}

    def factory() -> Connection:
        u_log_path = _attempt_unity_log_path(cfg.effective_log_dir, role, port, attempt["n"])
        attempt["n"] += 1
        cmd = launch.build_launch_cmd(exe, port, cfg.game_config, unity_log_path=u_log_path)
        proc = subprocess.Popen(cmd)  # noqa: S603 (arg-list, trusted local build path)
        try:
            sock = launch.connect(port)
            conn = Connection(sock, logger=logger)
        except BaseException:
            # The build is already running; if connect never completes, reap it so a failed launch
            # does not leak a process.
            _terminate(proc)
            raise
        # Stash the proc on the connection so the env's reap hook can hard-kill it on release.
        conn._launch_proc = proc  # type: ignore[attr-defined]
        return conn

    return factory


# --- env composition (the unit-test seam) ----------------------------------------------------


def _build_base_env(
    cfg: TrainConfig,
    connection_factory: Callable[[], Connection],
    port: int,
    *,
    role: str = ROLE_TRAIN,
    logger: logging.Logger | None = None,
) -> TankEnv:
    """Build the bare pure-transport :class:`TankEnv` over the given ``connection_factory``.

    Wires the env config (``max_steps`` from the Phase-1 round cap, ``game_port`` = ``port`` so
    the built env records WHICH socket it speaks on), the default reward, the channels-last
    ``frame_shape``, the seed, and the OPTIONAL observability ``logger`` / ``role`` (the env layer
    logs its milestones on the SAME logger the connection_factory hands its ``Connection``).

    LAZY: the factory is NOT invoked here — ``env.conn`` is ``None`` until the first
    ``reset``/``step``. The build the factory eventually launches (the LIVE path) is reaped by the
    env ITSELF via the injected ``reap=_terminate`` hook: :meth:`TankEnv.release` and the
    kill-old-first reconnect read the live ``Popen`` off
    ``getattr(self.conn, "_launch_proc", None)`` and call ``reap(proc)``. ``rl`` owns the reap
    primitive (:func:`_terminate`) and injects it so ``env`` need not import a subprocess reap
    (boundary: ``env`` imports ``core`` only). An injected
    STUB factory carries no proc, so the reap is a no-op (no Unity). ``logger=None`` (the default /
    test path) is behavior-identical to before.
    """
    return TankEnv(
        connection_factory=connection_factory,
        reap=_terminate,
        env_config=EnvConfig(max_steps=DEFAULT_MAX_STEPS, game_port=port),
        reward_config=RewardConfig(),
        frame_shape=cfg.frame_shape,
        seed=cfg.seed,
        logger=logger,
        role=role,
    )


def _make_self_play_env(
    cfg: TrainConfig,
    port: int,
    *,
    seed_offset: int = 0,
    connection_factory: Callable[[], Connection] | None = None,
    role: str = ROLE_TRAIN,
) -> SelfPlayWrapper:
    """Build one full ``SelfPlayWrapper(TankEnv)`` for ``port`` — the per-env construction unit.

    Builds the live ``connection_factory`` for ``port`` (re-derived from ``core.launch``) unless an
    explicit STUB ``connection_factory`` is injected (tests / the n_envs=1 seam), the base
    :class:`TankEnv`, and the :class:`OpponentProvider` (seeded ``cfg.seed + seed_offset`` so each
    subproc's roster RNG differs). Constructing the PROVIDER here — not before — is what makes the
    closure that wraps this spawn-safe: nothing live is captured, the provider is built INSIDE the
    subprocess. Returns the wrapper.

    Observability: on the LIVE path (no injected ``connection_factory``) this sets up the
    per-process env logger for ``(role, port)`` HERE — which, for a ``SubprocVecEnv`` worker, runs
    INSIDE the spawned subprocess, so the worker opens its OWN ``env-<role>-<port>.log`` handle —
    and threads that SAME logger into both the live ``Connection`` and the env layer. An injected
    STUB factory (the unit-test seam) gets NO logger (``None``), so the test path stays
    behavior-identical and never touches the filesystem.
    """
    if connection_factory is not None:
        factory = connection_factory
        logger = None
    else:
        logger = setup_env_logger(cfg.effective_log_dir, role, port, level=cfg.log_level)
        factory = _live_connection_factory_for_port(cfg, port, role=role, logger=logger)
    base = _build_base_env(cfg, factory, port, role=role, logger=logger)
    provider = OpponentProvider.from_roster(
        cfg.opponents, cfg.opponent_strategy, seed=cfg.seed + seed_offset
    )
    return SelfPlayWrapper(base, provider)


def training_ports(cfg: TrainConfig) -> list[int]:
    """The ``n_envs`` distinct training ports ``[game_port + i for i in range(n_envs)]`` (PURE)."""
    return [cfg.game_port + i for i in range(cfg.n_envs)]


def eval_ports(cfg: TrainConfig) -> list[int]:
    """The ``n_envs`` eval ports ``[effective_eval_port + i for i in range(n_envs)]`` (PURE).

    ``M_eval == N_train == n_envs`` eval builds, one per port in the eval block. The block is
    disjoint from :func:`training_ports` (validated in :meth:`TrainConfig.__post_init__`), so an
    eval build never collides with a training build even across a relaunch.
    """
    return [cfg.effective_eval_port + i for i in range(cfg.n_envs)]


def _env_factories_for_ports(
    cfg: TrainConfig,
    ports: list[int],
    *,
    role: str = ROLE_TRAIN,
    connection_factory_for_port: Callable[[int], Callable[[], Connection]] | None = None,
) -> list[Callable[[], SelfPlayWrapper]]:
    """``len(ports)`` zero-arg env factories — one per port (the SubprocVecEnv seam).

    Each factory ``i`` is a CLOSURE that builds env ``i`` on ``ports[i]`` with provider seed
    ``cfg.seed + i``. The closure captures ONLY ``cfg`` (a frozen, picklable dataclass) and the ints
    ``i`` / ``port`` — it constructs the provider / base env / live connection INSIDE its body, so
    nothing live crosses the spawn boundary (SB3 ships the ``env_fns`` via cloudpickle, which
    serializes the closure by its captured vars). This is why a closure is spawn-safe HERE where
    collection needed module-level functions: cloudpickle handles closures that capture only plain
    data.

    ``ports`` is the explicit port block: :func:`training_ports` for the training vec, or
    :func:`eval_ports` for the eval vec (the eval block is disjoint from training).
    ``connection_factory_for_port`` is a TEST seam: given a port it returns that port's STUB
    ``connection_factory`` (no Unity). ``None`` (the live path) -> each factory derives its own live
    connection from ``core.launch``. Returns the factory list (length ``len(ports)``).
    """
    factories: list[Callable[[], SelfPlayWrapper]] = []
    for i, port in enumerate(ports):

        def factory(i: int = i, port: int = port) -> SelfPlayWrapper:
            conn_factory = (
                None if connection_factory_for_port is None else connection_factory_for_port(port)
            )
            return _make_self_play_env(
                cfg, port, seed_offset=i, connection_factory=conn_factory, role=role
            )

        factories.append(factory)
    return factories


def _training_env_factories(
    cfg: TrainConfig,
    *,
    role: str = ROLE_TRAIN,
    connection_factory_for_port: Callable[[int], Callable[[], Connection]] | None = None,
) -> list[Callable[[], SelfPlayWrapper]]:
    """The ``n_envs`` training env factories (one per :func:`training_ports` port).

    Thin wrapper over :func:`_env_factories_for_ports` pinned to the TRAINING port block — kept as a
    named seam the unit tests assert on.
    """
    return _env_factories_for_ports(
        cfg,
        training_ports(cfg),
        role=role,
        connection_factory_for_port=connection_factory_for_port,
    )


def build_vec_env(
    cfg: TrainConfig,
    *,
    port: int | None = None,
    ports: list[int] | None = None,
    monitor: bool = False,
    single: bool = False,
    role: str = ROLE_TRAIN,
    connection_factory: Callable[[], Connection] | None = None,
    connection_factory_for_port: Callable[[int], Callable[[], Connection]] | None = None,
) -> VecEnv:
    """Compose the SB3 vec-env stack: ``[VecMonitor ->] VecFrameStack -> {Dummy,Subproc}VecEnv``.

    The per-env unit (:func:`_make_self_play_env`) is a base :class:`TankEnv` over its port's
    ``connection_factory`` wrapped in a :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` driving
    player2 from an :class:`~pop_trainer.rl.selfplay.OpponentProvider` (``cfg.opponents`` /
    ``cfg.opponent_strategy``). The vec stack around it:

    * **n_envs == 1 (or ``single=True``)** -> a ``DummyVecEnv`` of ONE env on ``port`` (default
      ``cfg.game_port``), provider seeded ``cfg.seed``. The provider is reachable for assertions /
      position-exact ``round_robin`` resume via :func:`_find_selfplay_wrapper` (``.envs`` is exposed
      in-process).
    * **n_envs > 1** -> a ``SubprocVecEnv`` of ``n_envs`` envs, one per port in ``ports`` (default
      :func:`training_ports`; the eval vec passes :func:`eval_ports`, a disjoint block), each
      launching its OWN Unity build in its OWN process with provider seeded ``cfg.seed + i``.
      ``start_method="spawn"`` is REQUIRED (Windows; no fork/forkserver). The env factories are
      CLOSURES that capture only ``cfg`` (a frozen, picklable dataclass) + the ints ``i`` / ``port``
      and build everything live INSIDE the subprocess — SB3 ships them via cloudpickle, so nothing
      live crosses the boundary. The provider lives PER-SUBPROC and is NOT reachable via
      :func:`_find_selfplay_wrapper` (``SubprocVecEnv`` exposes no ``.envs``); the caller treats
      opponent rotation as per-subproc + reseed-on-resume (see :func:`train_local`).

    Then ``VecFrameStack`` (``n_stack = cfg.frame_stack``; ``1`` = passthrough, still wrapped for a
    uniform stack), and OUTERMOST ``VecMonitor`` when ``monitor`` is set (the TRAINING env only — it
    logs ``rollout/ep_rew_mean`` / ``rollout/ep_len_mean``; the eval vec uses ``evaluate_winrate``'s
    own loop, so it is built ``monitor=False``).

    The TRAINING and EVAL vec envs are built the SAME way and at the SAME width (``M_eval ==
    N_train == cfg.n_envs``): training passes ``ports=training_ports`` (default) + ``role="train"``,
    eval passes ``ports=eval_ports`` + ``role="eval"``. ``single=True`` forces the ONE-env
    ``DummyVecEnv`` path regardless of ``cfg.n_envs`` (used by tests / a forced single env);
    ``port`` selects that single env's TCP port. ``ports`` overrides the multi-env port block (the
    eval vec uses it to bind the disjoint eval block instead of the training block).

    ``role`` (``"train"`` / ``"eval"``) tags the per-process observability log filenames the LIVE
    path opens (``env-<role>-<port>.log`` + the Unity ``unity-<role>-<port>.log``); the training env
    is built with ``role="train"`` and the eval env with ``role="eval"`` so eval evidence routes to
    its OWN file. It is purely observational and threads through to :func:`_make_self_play_env`.

    Test seams (no Unity): ``connection_factory`` injects a STUB for the SINGLE path; for the
    multi-env path ``connection_factory_for_port`` maps a port -> that port's STUB factory and the
    vec is built as a ``DummyVecEnv`` of the factories (subprocs are NOT spawned in unit tests).
    When a STUB is injected NO logger is set up (the test path stays behavior-identical, no files
    written).
    """
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
        VecMonitor,
    )

    if single or cfg.n_envs == 1:
        live_port = port if port is not None else cfg.game_port

        def make_wrapped(
            cf: Callable[[], Connection] | None = connection_factory,
        ) -> SelfPlayWrapper:
            return _make_self_play_env(
                cfg, live_port, seed_offset=0, connection_factory=cf, role=role
            )

        vec: VecEnv = DummyVecEnv([make_wrapped])
    else:
        block = ports if ports is not None else training_ports(cfg)
        factories = _env_factories_for_ports(
            cfg, block, role=role, connection_factory_for_port=connection_factory_for_port
        )
        if connection_factory_for_port is not None:
            # TEST path: stub factories build with no Unity, so run them in-process (DummyVecEnv) —
            # spawning real subprocs in a unit test would re-import + launch builds. The list /
            # ports are what the seam asserts.
            vec = DummyVecEnv(factories)
        else:
            # LIVE path: one Unity build per env in its OWN process. spawn is REQUIRED (Windows; no
            # fork/forkserver per CLAUDE.md).
            vec = SubprocVecEnv(factories, start_method="spawn")

    stacked: VecEnv = VecFrameStack(vec, n_stack=cfg.frame_stack)
    if monitor:
        # VecMonitor OUTERMOST so it sees episode boundaries on the stacked obs and logs SB3's
        # rollout/ep_rew_mean + rollout/ep_len_mean. Only the training env is monitored.
        stacked = VecMonitor(stacked)
    return stacked


def _build_policy_kwargs(cfg: TrainConfig) -> dict:
    """The ``policy_kwargs`` handed to PPO: wire the :class:`EncoderExtractor` + the head arch.

    Threads the extractor's ``checkpoint`` / ``freeze`` / ``trunk`` kwargs and the policy/value
    ``net_arch`` (the explicit ``[64, 64]`` default unless overridden). Pure (builds no encoder /
    no PPO) so the test can assert the extractor class and the kwargs without touching torch.
    """
    return {
        "features_extractor_class": EncoderExtractor,
        "features_extractor_kwargs": {
            "checkpoint": cfg.encoder_checkpoint,
            "freeze": cfg.freeze_encoder,
            "trunk": cfg.trunk,
        },
        "net_arch": cfg.net_arch,
    }


def _resolve_learning_rate(cfg: TrainConfig):
    """The PPO ``learning_rate`` arg: a float for ``"constant"`` or a schedule for ``"linear"``.

    ``"constant"`` (the default) returns ``cfg.learning_rate`` unchanged — bit-for-bit today's
    behavior. ``"linear"`` returns SB3's ``progress_remaining`` callable
    ``lambda progress_remaining: progress_remaining * lr`` so the rate decays from ``lr`` to 0
    across training (``progress_remaining`` runs 1 -> 0). Pure (no PPO / torch).
    """
    if cfg.lr_schedule == "linear":
        lr = cfg.learning_rate
        return lambda progress_remaining: progress_remaining * lr
    return cfg.learning_rate


def _find_selfplay_wrapper(vec_env: VecEnv) -> SelfPlayWrapper:
    """Reach the :class:`SelfPlayWrapper` inside the vec stack (the provider-access path).

    The training stack is ``VecMonitor(VecFrameStack(DummyVecEnv([SelfPlayWrapper(TankEnv)])))``;
    the eval stack drops the ``VecMonitor`` layer. Each vec WRAPPER (``VecMonitor`` /
    ``VecFrameStack``) exposes its inner vec env as ``.venv``, and ``DummyVecEnv`` exposes the gym
    envs as ``.envs``. Walk ``.venv`` down to the ``DummyVecEnv`` (the layer that has ``.envs``),
    then take ``.envs[0]`` — the ``SelfPlayWrapper`` whose ``.opponents`` is the
    :class:`OpponentProvider` the sidecar persists.
    """
    inner = vec_env
    while not hasattr(inner, "envs") and hasattr(inner, "venv"):
        inner = inner.venv  # VecMonitor -> VecFrameStack -> DummyVecEnv
    env0 = inner.envs[0]
    if not isinstance(env0, SelfPlayWrapper):
        raise TypeError(f"expected a SelfPlayWrapper at envs[0], got {type(env0).__name__}")
    return env0


# --- sidecar (resumable run state; pure save/load + a checkpoint-cadence callback) -----------


def save_sidecar(
    path: str | Path,
    *,
    provider: OpponentProvider | None,
    elo: dict[str, float],
    cfg: TrainConfig,
    num_timesteps: int,
) -> None:
    """Write the resumable run state to ``path`` as STRICT JSON (PURE; no model touched).

    Captures the opponent-provider position, the per-selector ELO dict, ``cfg.to_dict()``, and the
    current ``num_timesteps``. The provider position is replayable ONLY for ``round_robin`` (the
    ``_index``); for ``uniform`` there is no replayable position, so only ``strategy`` is recorded
    (resume continues the seeded RNG fresh — documented as the non-replayable strategy).

    ``provider is None`` is the n_envs>1 case: the providers live PER-SUBPROC (unreachable in this
    process), so only the strategy + seed (from ``cfg``) are recorded — resume RESEEDS the per-
    subproc rotation (approximate phase, like ``uniform``). The config carries ``opponent_strategy``
    and ``seed`` either way, so the sidecar still fully describes the run.
    """
    if provider is None:
        provider_state: dict = {"strategy": cfg.opponent_strategy, "seed": cfg.seed}
    else:
        provider_state = {"strategy": provider.strategy}
        if provider.strategy == "round_robin":
            provider_state["index"] = int(provider._index)
    payload = {
        "num_timesteps": int(num_timesteps),
        "provider": provider_state,
        "elo": {str(k): float(v) for k, v in elo.items()},
        "config": cfg.to_dict(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_sidecar(path: str | Path) -> dict:
    """Parse the sidecar back into a plain dict (STRICT ``json.loads``; PURE, no model)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _initial_elo(opponents: tuple[str, ...]) -> dict[str, float]:
    """Seed every roster selector at :data:`BASE_ELO` (the persisted-structure baseline)."""
    return {sel: BASE_ELO for sel in opponents}


def _restore_provider_position(provider: OpponentProvider | None, sidecar: dict) -> None:
    """Restore the opponent-provider position from a loaded sidecar (round_robin only).

    For ``round_robin`` the ``_index`` is restored so the rotation continues where it left off; for
    ``uniform`` there is nothing replayable to restore (the seeded RNG simply continues fresh).
    ``provider is None`` (n_envs>1) is a no-op: the per-subproc providers are unreachable here and
    resume RESEEDS them (approximate phase).
    """
    if provider is None:
        return
    provider_state = sidecar.get("provider", {})
    if provider.strategy == "round_robin" and "index" in provider_state:
        provider._index = int(provider_state["index"])


def _update_elo_from_eval(
    elo: dict[str, float], per_opponent: dict[str, float], *, k: float = 16.0
) -> dict[str, float]:
    """Nudge the learner-vs-opponent ELO from a per-opponent eval win-rate (PURE).

    Each opponent's rating moves against a notional learner rating (the mean of the current
    ratings) by the rounded :func:`pop_trainer.rl.elo.elo_change` delta for the opponent's side,
    given the LEARNER's realized win-rate. Returns a NEW dict (the input is not mutated). This is a
    light, optional Phase-1 wiring so the persisted ELO structure is non-trivial; the ladder proper
    lands with frozen-self opponents in P2.
    """
    if not elo:
        return dict(elo)
    learner_elo = sum(elo.values()) / len(elo)
    updated = dict(elo)
    for selector, learner_win_rate in per_opponent.items():
        if selector not in updated:
            continue
        # elo_change(A=learner, B=opponent, K, A's win-rate) -> (learner_delta, opponent_delta).
        _learner_delta, opp_delta = elo_change(learner_elo, updated[selector], k, learner_win_rate)
        updated[selector] = float(updated[selector] + opp_delta)
    return updated


def _checkpoint_save_freq(cfg: TrainConfig) -> int:
    """The per-env-CALL save_freq so a checkpoint lands every ``checkpoint_freq`` NUM_TIMESTEPS.

    SB3's ``CheckpointCallback._on_step`` gates on ``self.n_calls % self.save_freq == 0`` and does
    NOT divide ``save_freq`` by ``n_envs`` internally (it only *documents* that the caller should
    pass ``max(save_freq // n_envs, 1)``; verified in the installed source). Since
    ``num_timesteps == n_calls * n_envs``, gating on the RAW ``checkpoint_freq`` would fire every
    ``checkpoint_freq * n_envs`` timesteps at ``n_envs > 1`` — NOT the env-step cadence the cfg
    field promises. So we apply the documented ``max(checkpoint_freq // n_envs, 1)`` transform HERE
    and feed the SAME value to BOTH the ``CheckpointCallback`` and the ``SidecarCallback`` so they
    fire in lockstep at any ``n_envs`` and each ``state.json`` lands beside its ``model_*.zip``.
    """
    return max(cfg.checkpoint_freq // cfg.n_envs, 1)


def _make_sidecar_callback(
    cfg: TrainConfig, provider: OpponentProvider | None, elo: dict[str, float], *, save_freq: int
):
    """Build the SB3 callback that writes ``state.json`` in lockstep with each checkpoint.

    The callback is created lazily (SB3 imported here, not at module top) so the pure sidecar
    helpers above stay import-light. It rides the SAME transformed ``save_freq`` as the
    ``CheckpointCallback`` (see :func:`_checkpoint_save_freq`) so ``run_dir/state.json`` lands
    alongside each ``model_<steps>.zip`` at ANY ``n_envs``, capturing the provider position + ELO +
    cfg + ``num_timesteps``. ``provider`` is ``None`` for ``n_envs > 1`` (the per-subproc providers
    are unreachable; the sidecar records strategy + seed from ``cfg`` instead).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class SidecarCallback(BaseCallback):
        """Persist the resumable sidecar in lockstep with CheckpointCallback (same save_freq)."""

        def __init__(self, save_freq: int, sidecar_path: Path) -> None:
            super().__init__(verbose=0)
            self.save_freq = save_freq
            self.sidecar_path = sidecar_path
            self._provider = provider
            self._elo = elo

        def _on_step(self) -> bool:
            # Gate on the SAME (n_envs-transformed) cadence CheckpointCallback uses, so the sidecar
            # lands alongside the just-written model_<steps>.zip at any n_envs.
            if self.n_calls % self.save_freq == 0:
                save_sidecar(
                    self.sidecar_path,
                    provider=self._provider,
                    elo=self._elo,
                    cfg=cfg,
                    num_timesteps=self.num_timesteps,
                )
            return True

    return SidecarCallback(save_freq, cfg.run_dir / SIDECAR_NAME)


def _make_observability_callback(sys_logger: logging.Logger, *, checkpoint_save_freq: int):
    """An ADDITIVE SB3 callback logging rollout + checkpoint boundaries to the training-system log.

    Logs ``rollout_start`` / ``rollout_end`` (iteration, ``num_timesteps``, and ``fps`` pulled from
    SB3's own logger when available) at INFO on the ``_on_rollout_start`` / ``_on_rollout_end``
    hooks, and ``checkpoint_save`` on the SAME per-call cadence the ``CheckpointCallback`` /
    ``SidecarCallback`` ride (``n_calls % checkpoint_save_freq == 0``), so each ``model_*.zip``
    write leaves a system-log marker. It NEVER returns ``False`` and NEVER mutates the model /
    rollout — it is purely observational, so it does not change control flow. The SB3 import is
    local so the pure helpers above stay import-light.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class ObservabilityCallback(BaseCallback):
        """Log rollout iteration + checkpoint boundaries — additive, no control flow change."""

        def __init__(self, save_freq: int) -> None:
            super().__init__(verbose=0)
            self._iteration = 0
            self._save_freq = save_freq

        def _fps(self) -> float | None:
            # SB3 records time/fps on its logger's name_to_value once a rollout completes; read it
            # observationally (None before the first rollout / when unavailable).
            values = getattr(self.logger, "name_to_value", None)
            if isinstance(values, dict):
                fps = values.get("time/fps")
                if fps is not None:
                    return float(fps)
            return None

        def _on_rollout_start(self) -> None:
            self._iteration += 1
            sys_logger.info(
                "rollout_start",
                extra={
                    "detail": {"iteration": self._iteration, "num_timesteps": self.num_timesteps}
                },
            )

        def _on_rollout_end(self) -> None:
            sys_logger.info(
                "rollout_end",
                extra={
                    "detail": {
                        "iteration": self._iteration,
                        "num_timesteps": self.num_timesteps,
                        "fps": self._fps(),
                    }
                },
            )

        def _on_step(self) -> bool:
            if self._save_freq > 0 and self.n_calls % self._save_freq == 0:
                sys_logger.info(
                    "checkpoint_save", extra={"detail": {"num_timesteps": self.num_timesteps}}
                )
            return True

    return ObservabilityCallback(checkpoint_save_freq)


# --- resume ----------------------------------------------------------------------------------


def _latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the ``model_<steps>.zip`` with the MAX step count in ``run_dir`` (PURE).

    Parses the trailing integer from each ``model_*.zip`` basename and returns the max-step path;
    ``None`` when the directory has no parseable checkpoint. ``CheckpointCallback`` names files
    ``model_<steps>_steps.zip``, so the step count is the digit run before the ``_steps`` suffix.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for zip_path in run_dir.glob("model_*.zip"):
        stem = zip_path.stem  # e.g. "model_2048_steps"
        digits = [tok for tok in stem.split("_") if tok.isdigit()]
        if not digits:
            continue
        steps = int(digits[-1])
        if best is None or steps > best[0]:
            best = (steps, zip_path)
    return None if best is None else best[1]


# --- the integrator --------------------------------------------------------------------------


def train_local(cfg: TrainConfig) -> Path:
    """Run (or resume) one local PPO self-play training and return ``cfg.run_dir``.

    Composes the TRAINING vec-env stack at startup — a ``DummyVecEnv`` of ONE env at
    ``cfg.n_envs == 1`` or a ``SubprocVecEnv`` of ``cfg.n_envs`` builds (ports ``game_port + i``),
    ``VecMonitor``-wrapped so SB3 logs rollout episode stats — plus a DEDICATED, ALWAYS-SINGLE eval
    env on ``cfg.effective_eval_port`` (a separate Unity build / socket, ``single=True`` regardless
    of ``cfg.n_envs`` because ``evaluate_winrate`` needs the raw single ``TankEnv``). It builds (or
    loads) the PPO model with the :class:`EncoderExtractor` policy, attaches the eval + checkpoint +
    sidecar callbacks, runs ``model.learn``, then prints a final per-opponent win-rate line.

    A PRE-FLIGHT memory guard runs at startup: it reads ``psutil.virtual_memory().available``,
    estimates the rollout-buffer + Unity-instance bytes (:func:`check_rl_memory_budget`), ALWAYS
    prints the estimate, and ABORTS (raises ``MemoryError``) before any build launches if the
    estimate blows the budget — unless ``cfg.allow_oversized``.

    OPPONENT PROVIDER. At ``n_envs == 1`` the provider is reached in-process via
    :func:`_find_selfplay_wrapper` and resume restores its position EXACTLY (position-exact
    ``round_robin``). At ``n_envs > 1`` the providers live PER-SUBPROC (one per training build,
    seeded ``cfg.seed + i``) and are NOT reachable here; the sidecar records strategy + seed only
    and resume RESEEDS the rotation (approximate phase, like ``uniform``) — ``round_robin`` at
    ``n_envs > 1`` therefore degrades to per-subproc rotation with reseed-on-resume.

    Eval (periodic + final) runs against the dedicated eval env's raw :class:`TankEnv`, so the
    training env / model rollout state is NEVER touched by eval — no buffer "repair" is needed.
    BOTH env stacks are ALWAYS closed in a ``finally`` (each close suppressed independently so a
    failure to close one still closes the other) so all live Unity builds are reaped even on
    exception / KeyboardInterrupt.
    """
    import sys

    import psutil
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from pop_trainer.rl.callbacks import EvalWinRateCallback
    from pop_trainer.rl.evaluate import evaluate_winrate, format_per_map_table

    set_random_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # FAIL FAST on a frame_shape <-> game_config desync BEFORE any build launches: the env reads
    # exactly prod(frame_shape) bytes per frame, so a mismatch with the build's actual obs_pixels
    # W/H would corrupt every frame. This guards a directly-constructed TrainConfig too (main()
    # already derives the shape, but this is the anti-silent-desync invariant for any caller). Only
    # when the config file is actually present — an absent path is the launch's own failure surface
    # (and the test seam constructs fabricated config paths it never reads).
    if Path(cfg.game_config).exists():
        validate_frame_shape(cfg.frame_shape, cfg.game_config)

    # Set up the training-system logger (the main process) at the chosen level. INFO by default;
    # DEBUG when the single switch is set. This is purely observational.
    sys_logger = setup_system_logger(cfg.effective_log_dir, level=cfg.log_level)
    sys_logger.info(
        "run_config",
        extra={
            "detail": {
                "n_envs": cfg.n_envs,
                "training_ports": training_ports(cfg),
                "eval_port": cfg.effective_eval_port,
                "game_config": str(cfg.game_config),
                "obs_pixels": True,
                "n_steps": cfg.n_steps,
                "total_timesteps": cfg.total_timesteps,
                "frame_shape": list(cfg.frame_shape),
                "frame_stack": cfg.frame_stack,
                "opponents": list(cfg.opponents),
                "opponent_strategy": cfg.opponent_strategy,
                "seed": cfg.seed,
                "log_dir": str(cfg.effective_log_dir),
                "level": logging.getLevelName(cfg.log_level),
            }
        },
    )

    # PRE-FLIGHT MEMORY GUARD (psutil only here, the run glue). The PPO RolloutBuffer + the live
    # Unity instances are the OOM surface at high n_envs x n_steps. Estimate, ALWAYS print, then
    # abort BEFORE launching any build if it blows the budget (unless --allow-oversized).
    available = psutil.virtual_memory().available
    try:
        estimate_line = check_rl_memory_budget(
            n_steps=cfg.n_steps,
            n_envs=cfg.n_envs,
            frame_nbytes=frame_nbytes(cfg.frame_shape),
            frame_stack=cfg.frame_stack,
            available_bytes=available,
            allow_oversized=cfg.allow_oversized,
        )
    except MemoryError as exc:
        sys_logger.error("memory_abort", extra={"detail": {"message": str(exc)}})
        print(str(exc), file=sys.stderr)  # noqa: T201
        raise SystemExit(2) from exc
    sys_logger.info(
        "memory_estimate",
        extra={"detail": {"line": estimate_line, "available_bytes": available}},
    )
    print(estimate_line)  # noqa: T201

    # Training env: DummyVecEnv (n_envs=1) or SubprocVecEnv (n_envs>1) of builds on game_port + i,
    # VecMonitor-wrapped for rollout episode stats. Eval env: a vec of M_eval == n_envs builds on
    # the DISJOINT eval port block (eval_ports) — same width as training. The two SETS never
    # coexist: the eval callback tears the training instances down before lazy-launching the eval
    # instances and tears the eval instances down before respawning training (see
    # EvalWinRateCallback). With LAZY TankEnv construction, building these vec envs launches NO
    # Unity — the instances start only on the first reset/step (training: learn; eval: first cycle).
    vec_env = build_vec_env(cfg, port=cfg.game_port, monitor=True, role=ROLE_TRAIN)
    try:
        eval_vec_env = build_vec_env(
            cfg, port=cfg.effective_eval_port, ports=eval_ports(cfg), monitor=False, role=ROLE_EVAL
        )
        try:
            # n_envs=1: reach the in-process provider for position-exact round_robin resume.
            # n_envs>1: providers live per-subproc (unreachable here) -> None -> reseed-on-resume.
            provider = _find_selfplay_wrapper(vec_env).opponents if cfg.n_envs == 1 else None

            elo = _initial_elo(cfg.opponents)
            reset_num_timesteps = cfg.resume is None

            if cfg.resume is not None:
                # RESUME: load the latest checkpoint, restore the sidecar (provider position + ELO),
                # and continue from the recorded num_timesteps.
                latest = _latest_checkpoint(cfg.resume)
                if latest is None:
                    raise FileNotFoundError(
                        f"no model_*.zip checkpoint to resume from in {cfg.resume}"
                    )
                model = PPO.load(latest, env=vec_env)
                sidecar_path = Path(cfg.resume) / SIDECAR_NAME
                if sidecar_path.exists():
                    sidecar = load_sidecar(sidecar_path)
                    _restore_provider_position(provider, sidecar)
                    elo = {**elo, **{k: float(v) for k, v in sidecar.get("elo", {}).items()}}
            else:
                # FRESH: build PPO with the CnnPolicy + EncoderExtractor (it owns the checkpoint
                # load + freeze). SB3 builds the optimizer over all policy params; freeze works via
                # the extractor's requires_grad path, not optimizer membership.
                model = PPO(
                    "CnnPolicy",
                    vec_env,
                    policy_kwargs=_build_policy_kwargs(cfg),
                    seed=cfg.seed,
                    tensorboard_log=str(cfg.run_dir),
                    learning_rate=_resolve_learning_rate(cfg),
                    n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.n_epochs,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    clip_range=cfg.clip_range,
                    ent_coef=cfg.ent_coef,
                    vf_coef=cfg.vf_coef,
                    max_grad_norm=cfg.max_grad_norm,
                    verbose=1,
                )

            # Both CheckpointCallback and the sidecar ride the SAME n_envs-transformed save_freq so
            # a checkpoint + its state.json land every checkpoint_freq NUM_TIMESTEPS in lockstep at
            # any n_envs (see _checkpoint_save_freq).
            ckpt_save_freq = _checkpoint_save_freq(cfg)
            callbacks = CallbackList(
                [
                    EvalWinRateCallback(
                        cfg.eval_freq,
                        cfg.eval_episodes,
                        opponents=cfg.opponents,
                        seed=cfg.seed,
                        # The eval vec (M == n_envs) and the TRAINING vec handle SB3 reads
                        # _last_obs from (model.env, post-VecTransposeImage). The callback
                        # time-multiplexes the two SETS so no eval/training instance coexists.
                        eval_env=eval_vec_env,
                        training_vec=model.env,
                    ),
                    CheckpointCallback(
                        save_freq=ckpt_save_freq,
                        save_path=str(cfg.run_dir),
                        name_prefix="model",
                    ),
                    _make_sidecar_callback(cfg, provider, elo, save_freq=ckpt_save_freq),
                    _make_observability_callback(sys_logger, checkpoint_save_freq=ckpt_save_freq),
                ]
            )

            sys_logger.info(
                "learn_begin",
                extra={
                    "detail": {
                        "total_timesteps": cfg.total_timesteps,
                        "reset_num_timesteps": reset_num_timesteps,
                    }
                },
            )
            try:
                model.learn(
                    cfg.total_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=reset_num_timesteps,
                )
            except (EOFError, BrokenPipeError, ConnectionError) as exc:
                # The multi-env hang we are chasing surfaces as a SubprocVecEnv worker dying
                # mid-run (a closed pipe / EOF from a spawned worker). LOG the death with the
                # exception repr, then RE-RAISE unchanged — observability only, no swallow.
                sys_logger.error(
                    "worker_death",
                    extra={
                        "detail": {
                            "exc_type": type(exc).__name__,
                            "exc": repr(exc),
                            "num_timesteps": getattr(model, "num_timesteps", None),
                        }
                    },
                )
                raise
            sys_logger.info("learn_end", extra={"detail": {"num_timesteps": model.num_timesteps}})

            # Final per-opponent win-rate summary line (wires format_per_map_table -> the Director's
            # smoke gets one final per-opponent line). OBEY THE INVARIANT: tear the TRAINING
            # instances down FIRST (free their RAM/ports) so the M eval instances never coexist with
            # them; then run the PARALLEL eval over the eval vec. This is end-of-run, so no respawn
            # is needed afterward — everything is closed in the finally blocks below.
            sys_logger.info(
                "final_eval_begin",
                extra={"detail": {"episodes": cfg.eval_episodes, "opponents": list(cfg.opponents)}},
            )
            vec_env.env_method("release")
            per_opponent = evaluate_winrate(
                model,
                eval_vec_env,
                opponents=cfg.opponents,
                n_episodes=cfg.eval_episodes,
                seed=cfg.seed,
            )
            elo = _update_elo_from_eval(elo, per_opponent)
            sys_logger.info(
                "final_eval_end",
                extra={
                    "detail": {"win_rates": {str(k): float(v) for k, v in per_opponent.items()}}
                },
            )
            print(format_per_map_table(per_opponent, episodes=cfg.eval_episodes))  # noqa: T201

            # Persist the final sidecar (post-learn position + the eval-updated ELO).
            save_sidecar(
                cfg.run_dir / SIDECAR_NAME,
                provider=provider,
                elo=elo,
                cfg=cfg,
                num_timesteps=model.num_timesteps,
            )
        finally:
            # ALWAYS reap any live eval Unity instances. release() HARD-KILLS each live build via
            # the env's injected reap hook (close() alone only does the graceful end-handshake +
            # transport release, NOT the process kill); then close() tears down transports/workers.
            # Suppressed so a failure here still lets the training env tear down below.
            with contextlib.suppress(Exception):
                eval_vec_env.env_method("release")
            with contextlib.suppress(Exception):
                eval_vec_env.close()
    finally:
        # ALWAYS reap any live training Unity instances (hard-kill via the reap hook), then close,
        # even on exception / KeyboardInterrupt.
        with contextlib.suppress(Exception):
            vec_env.env_method("release")
        vec_env.close()

    return cfg.run_dir


# --- CLI -------------------------------------------------------------------------------------


def _parse_opponents(spec: str) -> tuple[str, ...]:
    """Split a comma-separated ``--opponents`` selector list into an ordered, stripped tuple.

    Pure: ``"noop"`` -> ``("noop",)``; ``"a, b ,c"`` -> ``("a", "b", "c")``. Validation against
    :data:`AGENT_SELECTORS` is the caller's job (so a parse error can route through
    ``parser.error`` for a clean exit-code-2 message).
    """
    return tuple(s.strip() for s in spec.split(","))


def _parse_net_arch(spec: str) -> dict[str, list[int]]:
    """Parse a ``--net-arch`` spec ``"pi=64,64:vf=64,64"`` into SB3's ``dict(pi=..., vf=...)``.

    The spec is colon-separated ``head=widths`` groups, each a comma-separated int width list, so
    ``"pi=64,64:vf=64,64"`` -> ``{"pi": [64, 64], "vf": [64, 64]}`` (the per-head SB3 net_arch
    form). Pure; raises :class:`ValueError` on a malformed group / non-int width (argparse turns it
    into a clean exit-code-2 message).
    """
    arch: dict[str, list[int]] = {}
    for group in spec.split(":"):
        head, _, widths = group.partition("=")
        head = head.strip()
        if not head or not widths.strip():
            raise ValueError(f"malformed net_arch group {group!r}; expected 'head=w1,w2,...'")
        try:
            arch[head] = [int(w) for w in widths.split(",")]
        except ValueError as exc:
            raise ValueError(f"net_arch widths must be ints in {group!r}") from exc
    return arch


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the train CLI (PURE: no side effects, returns the namespace)."""
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.rl.train",
        description="Local PPO self-play training over the Unity pixel env (Phase 1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="game/training config JSON forwarded to the build launch (single-map, obs_pixels).",
    )
    parser.add_argument(
        "--total-timesteps", type=int, required=True, help="total env-steps to train."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="output dir for checkpoints / sidecar / tb."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="a prior run_dir to resume (load latest model_*.zip + sidecar).",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="optional pretrained-encoder state_dict for the features extractor.",
    )
    parser.add_argument(
        "--freeze-encoder", action="store_true", help="freeze the encoder weights during RL."
    )
    parser.add_argument("--frame-stack", type=int, default=1, help="VecFrameStack depth (1=off).")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel training envs (1=DummyVecEnv; >1=SubprocVecEnv, one Unity build per env on "
        "game_port + i).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=TrainConfig.n_steps,
        help="PPO rollout length (the memory lever at n_envs>1; lower it for more envs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
        help="PPO batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainConfig.learning_rate,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default=TrainConfig.lr_schedule,
        help="learning-rate schedule: constant (default) or linear decay to 0.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=TrainConfig.n_epochs,
        help="PPO epochs per rollout.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=TrainConfig.gamma,
        help="discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=TrainConfig.gae_lambda,
        help="GAE lambda.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=TrainConfig.clip_range,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=TrainConfig.ent_coef,
        help="entropy coefficient.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=TrainConfig.vf_coef,
        help="value-function loss coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=TrainConfig.max_grad_norm,
        help="gradient-clipping max norm.",
    )
    parser.add_argument(
        "--net-arch",
        type=_parse_net_arch,
        default=None,
        help="policy/value MLP head, e.g. 'pi=64,64:vf=64,64' (default: [64, 64]).",
    )
    parser.add_argument(
        "--trunk",
        choices=TRUNK_CHOICES,
        default=TrainConfig.trunk,
        help="encoder trunk override: auto (size rule) or cnn/resnet/gn-cnn.",
    )
    parser.add_argument(
        "--allow-oversized",
        action="store_true",
        help="skip the pre-flight rollout-buffer memory abort (the estimate is still printed).",
    )
    parser.add_argument("--seed", type=int, default=0, help="master seed.")
    parser.add_argument(
        "--opponents",
        type=_parse_opponents,
        default=None,
        help="comma-separated self-play roster selectors (default: the full roster).",
    )
    parser.add_argument(
        "--opponent-strategy",
        choices=("round_robin", "uniform"),
        default="round_robin",
        help="opponent rotation: round_robin (resumable) or uniform (seed-only resume).",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000, help="env-steps between win-rate evals (0=off)."
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="greedy episodes per opponent per eval."
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10_000, help="env-steps between checkpoints."
    )
    parser.add_argument(
        "--port", type=int, default=50000, help="TCP port the training build listens on."
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=None,
        help="TCP port the eval build listens on (default: port + 1).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="dir for the per-process observability logs (default: run_dir/logs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"the SINGLE switch: crank ALL observability logs to DEBUG (default INFO). "
        f"Equivalent to {LOG_LEVEL_ENV_VAR}=DEBUG.",
    )
    args = parser.parse_args(argv)
    if args.opponents is not None:
        unknown = [sel for sel in args.opponents if sel not in AGENT_SELECTORS]
        if unknown:
            valid = ", ".join(sorted(AGENT_SELECTORS))
            parser.error(f"unknown opponent selector(s) {unknown}; choose from: {valid}")
    return args


def main(argv: list[str] | None = None) -> Path:
    """CLI entry: build a :class:`TrainConfig` from the args and run :func:`train_local`.

    The SINGLE observability switch is ``--debug`` OR the ``POP_LOG_LEVEL`` env var resolving to
    DEBUG (:func:`pop_trainer.core.logging_setup.level_from_env`); either cranks every layer's logs
    to DEBUG. The default is INFO (negligible overhead). ``--log-dir`` overrides where the
    per-process logs are written (default ``run_dir/logs``).
    """
    args = _parse_args(argv)
    # --opponents omitted (None) -> let TrainConfig's DEFAULT_ROSTER default apply.
    opponents_kwarg = {} if args.opponents is None else {"opponents": args.opponents}
    # The single DEBUG switch: --debug flag OR the POP_LOG_LEVEL env var (resolved purely).
    debug_logging = (
        level_from_env(debug=args.debug, env_value=os.environ.get(LOG_LEVEL_ENV_VAR))
        == logging.DEBUG
    )
    # DERIVE the pixel frame_shape from the LAUNCHED config's obs_pixels_* so the env byte-read
    # matches the build's rendered frame (one source of truth) — no hand-synced constant. Missing
    # keys fall back to core.DEFAULT_FRAME_SHAPE; a malformed value fails here.
    frame_shape = frame_shape_from_config(args.config)
    cfg = TrainConfig(
        total_timesteps=args.total_timesteps,
        game_config=args.config,
        run_dir=args.run_dir,
        frame_shape=frame_shape,
        n_envs=args.n_envs,
        frame_stack=args.frame_stack,
        encoder_checkpoint=args.encoder_checkpoint,
        freeze_encoder=args.freeze_encoder,
        resume=args.resume,
        seed=args.seed,
        opponent_strategy=args.opponent_strategy,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=args.net_arch,
        trunk=args.trunk,
        allow_oversized=args.allow_oversized,
        game_port=args.port,
        eval_port=args.eval_port,
        log_dir=args.log_dir,
        debug_logging=debug_logging,
        **opponents_kwarg,
    )
    return train_local(cfg)


if __name__ == "__main__":
    main()
