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

Boundary: imports ``core`` (incl. ``core.launch`` / ``core.protocol`` / ``core.config``), ``env``,
sb3 / gymnasium / stdlib. Imports NOTHING from ``data`` or ``pretraining`` (the live launch is
re-derived from ``core.launch``, never imported from ``data``). No cycles.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pop_trainer.core import launch
from pop_trainer.core.config import EnvConfig, RewardConfig
from pop_trainer.core.protocol import Connection
from pop_trainer.env.tank_env import TankEnv
from pop_trainer.rl.elo import elo_change
from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import _STRATEGIES, DEFAULT_ROSTER, OpponentProvider, SelfPlayWrapper

if TYPE_CHECKING:  # type-only: keep the pure helpers / config import-light at module load
    from collections.abc import Callable

    from stable_baselines3.common.vec_env import VecEnv

__all__ = ["TrainConfig", "train_local", "build_vec_env", "save_sidecar", "load_sidecar", "main"]

# This file is src/pop_trainer/rl/train.py: parents[3] is the repo root
# (core sibling -> rl -> pop_trainer -> src -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The default game/training config forwarded to the build launch (single-map, obs_pixels=True,
# timeScale<=5). Lives next to the other StreamingAssets configs.
DEFAULT_TRAIN_CONFIG = _REPO_ROOT / "Assets" / "StreamingAssets" / "train_config.json"

# The 640x360 RGB pixel frame Unity renders, channels-LAST (H, W, 3). SB3 auto-applies
# VecTransposeImage so the extractor sees (N, 3, 360, 640).
DEFAULT_FRAME_SHAPE = (360, 640, 3)

# Phase-1 round cap (1-min / 300-step rounds — intentional, see the milestone notes).
DEFAULT_MAX_STEPS = 300

# Every roster opponent starts at this rating; Phase-1 persists the ELO structure so resume can
# restore it (a simple from-eval update is wired below but optional).
BASE_ELO = 1000.0

# The sidecar file written alongside each model_<steps>.zip checkpoint.
SIDECAR_NAME = "state.json"


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
        n_envs: number of parallel envs (Phase-1 is ``1``; ``>1`` is a documented SEAM, not built).
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
        learning_rate / n_steps / batch_size / n_epochs / gamma / gae_lambda / clip_range:
            PPO hyperparameters (pixel-PPO defaults).
        frame_shape: the ``(H, W, 3)`` pixel-frame shape (channels-last; SB3 transposes it).
        game_port: the TCP port the TRAINING build listens on (``args[1]`` of the launch arg-list).
        eval_port: the TCP port the DEDICATED eval build listens on. ``None`` (default) -> the
            effective eval port is ``game_port + 1``. When set it must differ from ``game_port``
            (the two builds run side-by-side on separate sockets).
        build_path: the build binary; ``None`` -> :func:`core.launch.default_build_path`.
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
    # PPO hyperparameters (pixel-PPO defaults).
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE
    game_port: int = 50000
    eval_port: int | None = None
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
                f"eval_port must differ from game_port (both {self.game_port}); the eval build "
                "runs side-by-side on a separate socket"
            )

    @property
    def effective_eval_port(self) -> int:
        """The port the dedicated eval build listens on: ``eval_port`` or ``game_port + 1``."""
        return self.eval_port if self.eval_port is not None else self.game_port + 1

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
            "frame_shape": list(self.frame_shape),
            "game_port": self.game_port,
            "eval_port": self.eval_port,
            "build_path": None if self.build_path is None else str(self.build_path),
        }


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


def _attach_launch_proc(env: TankEnv, proc: subprocess.Popen) -> None:
    """Stash ``proc`` on ``env`` and wrap ``env.close`` so closing the env reaps the build.

    ``TankEnv.close`` releases only the transport (the socket) — it does NOT know about the build
    subprocess. So we record the proc on ``env._launch_proc`` (inspectable) and replace
    ``env.close`` with a wrapper that runs the original close (end-handshake + transport release)
    and THEN terminates the build. Idempotent (the second close is a no-op once reaped). The
    ``TankEnv`` class is untouched — this is per-instance caller-side wrapping. Mirrors the
    collection runner's pattern, re-derived here to keep ``rl`` off ``data``.

    Reconnect-safe: a mid-run socket drop makes the env RE-INVOKE the factory, producing a NEW
    ``Connection`` with a NEW ``_launch_proc`` for the relaunched build. So at close time we reap
    the CURRENT connection's proc (``env.conn._launch_proc``), falling back to the originally
    stashed ``proc`` only when the current conn carries none — the original (already-dead) build is
    a no-op reap, and the relaunched build can never be orphaned.
    """
    env._launch_proc = proc
    original_close = env.close

    def close_and_reap() -> None:
        # Read the CURRENT connection's launch proc before close tears the transport down; a
        # reconnect mid-run swapped in a fresh Connection/proc, so reap THAT, not the original.
        current = getattr(env.conn, "_launch_proc", proc)
        with contextlib.suppress(Exception):
            original_close()
        _terminate(current)

    env.close = close_and_reap  # type: ignore[method-assign]


def _live_connection_factory_for_port(cfg: TrainConfig, port: int) -> Callable[[], Connection]:
    """Build the LIVE ``connection_factory`` for ``port``: launch the build, connect, wrap socket.

    Returns a zero-arg callable the env invokes on construction (and re-invokes on reconnect). Each
    call: launches the build via :func:`core.launch.build_launch_cmd` + ``subprocess.Popen`` on
    ``port``, connects via :func:`core.launch.connect`, and wraps the socket in a
    :class:`core.protocol.Connection`. The live ``Popen`` is stashed on the returned ``Connection``
    (``conn._launch_proc``) so :func:`_build_base_env` can wire ``env.close`` to reap it.

    The build is launched WINDOWED (never batchmode) on ``cfg.game_config``. Parameterizing the
    ``port`` lets the TRAINING env (``cfg.game_port``) and the DEDICATED eval env
    (``cfg.effective_eval_port``) each launch their OWN build on their OWN socket. Re-derived from
    the shared ``core.launch`` primitives — ``rl`` does NOT import the ``data`` launch path.
    """
    exe = cfg.build_path if cfg.build_path is not None else launch.default_build_path(_REPO_ROOT)

    def factory() -> Connection:
        cmd = launch.build_launch_cmd(exe, port, cfg.game_config)
        proc = subprocess.Popen(cmd)  # noqa: S603 (arg-list, trusted local build path)
        try:
            sock = launch.connect(port)
            conn = Connection(sock)
        except BaseException:
            # The build is already running; if connect never completes, reap it so a failed launch
            # does not leak a process.
            _terminate(proc)
            raise
        # Stash the proc on the connection so the env-build step can reap it on close.
        conn._launch_proc = proc  # type: ignore[attr-defined]
        return conn

    return factory


# --- env composition (the unit-test seam) ----------------------------------------------------


def _build_base_env(
    cfg: TrainConfig, connection_factory: Callable[[], Connection], port: int
) -> TankEnv:
    """Build the bare pure-transport :class:`TankEnv` over the given ``connection_factory``.

    Wires the env config (``max_steps`` from the Phase-1 round cap, ``game_port`` = ``port`` so
    the built env records WHICH socket it speaks on), the default reward, the channels-last
    ``frame_shape``, and the seed. If the factory stashed a live build ``Popen`` on its produced
    ``Connection`` (the LIVE path), wrap ``env.close`` so closing the env reaps the build; an
    injected STUB factory carries no proc, so the reap-wrap is skipped (no Unity).
    """
    env = TankEnv(
        connection_factory=connection_factory,
        env_config=EnvConfig(max_steps=DEFAULT_MAX_STEPS, game_port=port),
        reward_config=RewardConfig(),
        frame_shape=cfg.frame_shape,
        seed=cfg.seed,
    )
    proc = getattr(env.conn, "_launch_proc", None)
    if proc is not None:
        _attach_launch_proc(env, proc)
    return env


def build_vec_env(
    cfg: TrainConfig,
    *,
    port: int | None = None,
    monitor: bool = False,
    connection_factory: Callable[[], Connection] | None = None,
) -> VecEnv:
    """Compose the SB3 vec-env stack: ``VecFrameStack`` -> ``DummyVecEnv`` -> ``SelfPlayWrapper``.

    The composition (the seam the unit test asserts):

    1. base :class:`TankEnv` over ``connection_factory`` (the LIVE launch on ``port`` when ``None``;
       the injected STUB in tests);
    2. wrapped in a :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` driving player2 from a
       :class:`~pop_trainer.rl.selfplay.OpponentProvider` built from ``cfg.opponents`` /
       ``cfg.opponent_strategy`` (seeded with ``cfg.seed``);
    3. boxed in a ``DummyVecEnv`` (Phase-1 ``n_envs == 1``);
    4. wrapped in ``VecFrameStack`` (``n_stack = cfg.frame_stack``; ``1`` = passthrough, still
       wrapped so the stack is uniform);
    5. when ``monitor`` is set, wrapped OUTERMOST in ``VecMonitor`` so SB3 logs
       ``rollout/ep_rew_mean`` / ``rollout/ep_len_mean`` (the TRAINING env only; the eval env uses
       ``evaluate_winrate``'s own loop, not SB3 episode stats, so it is built ``monitor=False``).

    ``port`` selects the live build's TCP port (defaults to ``cfg.game_port`` — the training env;
    the eval env passes ``cfg.effective_eval_port``). It also threads into ``EnvConfig.game_port``
    so each built env records its own socket. When a STUB ``connection_factory`` is injected NO
    Unity is launched on either port.

    ``n_envs > 1`` is a documented SEAM (``SubprocVecEnv`` + per-env distinct ports), NOT built in
    Phase-1 — it raises ``NotImplementedError``.

    The constructed ``OpponentProvider`` is reachable for assertions/resume through the vec stack:
    ``[VecMonitor ->] VecFrameStack -> DummyVecEnv -> envs[0]`` (the SelfPlayWrapper) ->
    ``.opponents``. :func:`_find_selfplay_wrapper` walks that path (skipping any ``.venv`` layer).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    if cfg.n_envs != 1:
        # SEAM: n>1 needs SubprocVecEnv with a distinct game_port per env (each launches its own
        # build). Phase-1 is n=1; the multi-env fan-out is deferred.
        raise NotImplementedError(
            f"n_envs > 1 is a deferred seam (SubprocVecEnv + per-env ports); got {cfg.n_envs}"
        )

    live_port = port if port is not None else cfg.game_port
    factory = (
        connection_factory
        if connection_factory is not None
        else _live_connection_factory_for_port(cfg, live_port)
    )

    def make_wrapped() -> SelfPlayWrapper:
        base = _build_base_env(cfg, factory, live_port)
        provider = OpponentProvider.from_roster(cfg.opponents, cfg.opponent_strategy, seed=cfg.seed)
        return SelfPlayWrapper(base, provider)

    vec = DummyVecEnv([make_wrapped])
    stacked: VecEnv = VecFrameStack(vec, n_stack=cfg.frame_stack)
    if monitor:
        # VecMonitor OUTERMOST so it sees episode boundaries on the stacked obs and logs SB3's
        # rollout/ep_rew_mean + rollout/ep_len_mean. Only the training env is monitored.
        stacked = VecMonitor(stacked)
    return stacked


def _build_policy_kwargs(cfg: TrainConfig) -> dict:
    """The ``policy_kwargs`` handed to PPO: wire the :class:`EncoderExtractor` + its load/freeze.

    Pure (builds no encoder / no PPO) so the test can assert the extractor class and the
    checkpoint/freeze kwargs without touching torch.
    """
    return {
        "features_extractor_class": EncoderExtractor,
        "features_extractor_kwargs": {
            "checkpoint": cfg.encoder_checkpoint,
            "freeze": cfg.freeze_encoder,
        },
    }


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
    provider: OpponentProvider,
    elo: dict[str, float],
    cfg: TrainConfig,
    num_timesteps: int,
) -> None:
    """Write the resumable run state to ``path`` as STRICT JSON (PURE; no model touched).

    Captures the opponent-provider position, the per-selector ELO dict, ``cfg.to_dict()``, and the
    current ``num_timesteps``. The provider position is replayable ONLY for ``round_robin`` (the
    ``_index``); for ``uniform`` there is no replayable position, so only ``strategy`` is recorded
    (resume continues the seeded RNG fresh — documented as the non-replayable strategy).
    """
    provider_state: dict = {"strategy": provider.strategy}
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


def _restore_provider_position(provider: OpponentProvider, sidecar: dict) -> None:
    """Restore the opponent-provider position from a loaded sidecar (round_robin only).

    For ``round_robin`` the ``_index`` is restored so the rotation continues where it left off; for
    ``uniform`` there is nothing replayable to restore (the seeded RNG simply continues fresh).
    """
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


def _make_sidecar_callback(cfg: TrainConfig, provider: OpponentProvider, elo: dict[str, float]):
    """Build the SB3 callback that writes ``state.json`` on the checkpoint cadence.

    The callback is created lazily (SB3 imported here, not at module top) so the pure sidecar
    helpers above stay import-light. It rides the SAME ``checkpoint_freq`` as
    ``CheckpointCallback``, writing ``run_dir/state.json`` alongside each ``model_<steps>.zip``,
    capturing the provider position + ELO + cfg + ``num_timesteps``.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class SidecarCallback(BaseCallback):
        """Persist the resumable sidecar on the checkpoint cadence (mirrors CheckpointCallback)."""

        def __init__(self, save_freq: int, sidecar_path: Path) -> None:
            super().__init__(verbose=0)
            self.save_freq = save_freq
            self.sidecar_path = sidecar_path
            self._provider = provider
            self._elo = elo

        def _on_step(self) -> bool:
            # Gate on the SAME cadence CheckpointCallback uses (per-env-call counting); write the
            # sidecar alongside the just-written model_<steps>.zip.
            if self.n_calls % self.save_freq == 0:
                save_sidecar(
                    self.sidecar_path,
                    provider=self._provider,
                    elo=self._elo,
                    cfg=cfg,
                    num_timesteps=self.num_timesteps,
                )
            return True

    return SidecarCallback(cfg.checkpoint_freq, cfg.run_dir / SIDECAR_NAME)


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

    Composes TWO vec-env stacks at startup — the TRAINING env on ``cfg.game_port`` (wrapped in
    ``VecMonitor`` so SB3 logs rollout episode stats) and a DEDICATED eval env on
    ``cfg.effective_eval_port`` (a SECOND Unity build / socket) — builds (or loads) the PPO model
    with the :class:`EncoderExtractor` policy, attaches the eval + checkpoint + sidecar callbacks,
    runs ``model.learn``, then prints a final per-opponent win-rate line.

    Eval (periodic + final) runs against the dedicated eval env's raw :class:`TankEnv`, so the
    training env / model rollout state is NEVER touched by eval — no buffer "repair" is needed.
    BOTH env stacks are ALWAYS closed in a ``finally`` (each close suppressed independently so a
    failure to close one still closes the other) so both live Unity builds are reaped even on
    exception / KeyboardInterrupt.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from pop_trainer.rl.callbacks import EvalWinRateCallback
    from pop_trainer.rl.evaluate import evaluate_winrate, format_per_map_table

    set_random_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # Training env on game_port (VecMonitor-wrapped for rollout episode stats); dedicated eval env
    # on the effective eval port (a separate build/socket — eval never touches the training env).
    vec_env = build_vec_env(cfg, port=cfg.game_port, monitor=True)
    try:
        eval_vec_env = build_vec_env(cfg, port=cfg.effective_eval_port, monitor=False)
        try:
            provider = _find_selfplay_wrapper(vec_env).opponents
            eval_raw_env = _find_selfplay_wrapper(eval_vec_env).env

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
                    learning_rate=cfg.learning_rate,
                    n_steps=cfg.n_steps,
                    batch_size=cfg.batch_size,
                    n_epochs=cfg.n_epochs,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    clip_range=cfg.clip_range,
                    verbose=1,
                )

            callbacks = CallbackList(
                [
                    EvalWinRateCallback(
                        cfg.eval_freq,
                        cfg.eval_episodes,
                        opponents=cfg.opponents,
                        seed=cfg.seed,
                        eval_env=eval_raw_env,
                    ),
                    CheckpointCallback(
                        save_freq=cfg.checkpoint_freq,
                        save_path=str(cfg.run_dir),
                        name_prefix="model",
                    ),
                    _make_sidecar_callback(cfg, provider, elo),
                ]
            )

            model.learn(
                cfg.total_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
            )

            # Final per-opponent win-rate summary line (wires format_per_map_table -> the
            # Director's smoke gets one final per-opponent line). Eval re-wraps the DEDICATED eval
            # env's raw TankEnv in its own per-opponent SelfPlayWrapper (the evaluate_winrate
            # contract) — the training env / model state is untouched.
            per_opponent = evaluate_winrate(
                model,
                eval_raw_env,
                opponents=cfg.opponents,
                n_episodes=cfg.eval_episodes,
                seed=cfg.seed,
            )
            elo = _update_elo_from_eval(elo, per_opponent)
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
            # ALWAYS reap the eval Unity build; suppress so a failure here still lets the training
            # env close below.
            with contextlib.suppress(Exception):
                eval_vec_env.close()
    finally:
        # ALWAYS reap the training Unity build (the close-and-reap wrapper), even on exception /
        # KeyboardInterrupt.
        vec_env.close()

    return cfg.run_dir


# --- CLI -------------------------------------------------------------------------------------


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
    parser.add_argument("--seed", type=int, default=0, help="master seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry: build a :class:`TrainConfig` from the args and run :func:`train_local`."""
    args = _parse_args(argv)
    cfg = TrainConfig(
        total_timesteps=args.total_timesteps,
        game_config=args.config,
        run_dir=args.run_dir,
        frame_stack=args.frame_stack,
        encoder_checkpoint=args.encoder_checkpoint,
        freeze_encoder=args.freeze_encoder,
        resume=args.resume,
        seed=args.seed,
    )
    return train_local(cfg)


if __name__ == "__main__":
    main()
