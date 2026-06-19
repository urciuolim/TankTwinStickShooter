"""Local single-agent PPO training: one agent learning vs. the random opponent (M1, task 3.1).

This is the M1 trainer. It COMPOSES the already-merged modules — it does NOT re-implement
the env, reward, obs, protocol, or feature-extractor logic:

* :class:`tank_twin.env.TankEnv` — the gymnasium env (image-based, ``env_p=3``,
  random opponent), launched against a real Unity build via ``game_path`` OR driven by an
  injected fake transport / env-factory in tests.
* :class:`tank_twin.features.PretrainedNatureCNN` — the 2021 pretrained vision weights
  loaded into an SB3 ``NatureCNN`` and FROZEN by default (board-approved "freeze first").

Frozen-CNN wiring (Route B, via ``policy_kwargs``): we pass ``PretrainedNatureCNN`` as the
policy's ``features_extractor_class`` rather than the ``load_pretrained_into_policy`` Route-A
helper. Route B is cleaner here because the load+freeze happens INSIDE policy construction,
BEFORE SB3 builds the optimizer over ``policy.parameters()`` — so the frozen
(``requires_grad=False``) submodules are naturally excluded from the optimizer with no
post-build surgery. Route A would mutate ``requires_grad`` AFTER the optimizer already
captured the params (frozen tensors still in the optimizer, just zero-grad), which is
correct but less clean. ``normalize_images`` is left at the SB3 default ``True`` (the
pretrain divided inputs by 255 — do NOT override that).

Single-core by board decision: a bare ``TankEnv`` (SB3 wraps it in a ``DummyVecEnv``
internally) — NOT ``SubprocVecEnv``. No ``fork`` / ``forkserver``. All process / path
construction uses ``pathlib`` + ``subprocess`` arg-lists (in the env), never ``os.system``.
Strict JSON only (the protocol module enforces it).

Imports torch + sb3 — this module is NOT in the pure-logic import path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from tank_twin.env import TankEnv
from tank_twin.features import PretrainedNatureCNN

__all__ = ["train_local", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"

# PPO defaults: sane single-env starting knobs (the GPU smoke / Wave-4 tuning overrides).
DEFAULT_N_STEPS = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_TIMESTEPS = 100_000


def _resolve_device(device: str) -> str:
    """Map ``auto`` -> ``cuda`` when CUDA is available, else ``cpu``; pass others through."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_local(
    *,
    timesteps: int = DEFAULT_TIMESTEPS,
    game_path: str | Path | None = None,
    env_factory: Callable[[], gymnasium.Env] | None = None,
    seed: int = 0,
    unfreeze: bool = False,
    run_name: str = "m1-local",
    device: str = "auto",
    n_steps: int = DEFAULT_N_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    checkpoint_freq: int = 50_000,
    verbose: int = 1,
    log_metrics: bool = True,
) -> PPO:
    """Train one PPO agent against the random opponent and return the built model.

    Builds a single ``TankEnv`` (image-based, ``env_p=3``, random opponent), constructs a
    PPO ``CnnPolicy`` whose feature extractor is the frozen (or, with ``unfreeze``, trainable)
    pretrained ``PretrainedNatureCNN``, trains for ``timesteps``, checkpoints into
    ``runs/<run_name>/``, and saves the final model ``.zip`` into ``models/``.

    Args:
        timesteps: total environment steps to train for.
        game_path: path to the Unity build (real-Unity path). Mutually exclusive with
            ``env_factory``; ignored when ``env_factory`` is given.
        env_factory: zero-arg callable returning a built env — the TEST seam (inject a
            ``TankEnv`` with a fake transport so no Unity / subprocess spawns). When given,
            ``game_path`` is not used.
        seed: master seed; threaded through ``set_random_seed`` (numpy / torch / python),
            the env ``reset``, and ``PPO(seed=...)`` for reproducibility.
        unfreeze: if True, the pretrained CNN is loaded but left TRAINABLE
            (``--unfreeze``); default False keeps it frozen (board-approved freeze-first).
        run_name: subdirectory under ``runs_dir`` for checkpoints + the run manifest.
        device: ``auto`` / ``cpu`` / ``cuda`` (auto -> cuda if available).
        n_steps / batch_size / learning_rate: PPO rollout / optimization knobs.
        runs_dir / models_dir: output roots (defaulting to the repo's ``runs`` / ``models``).
        checkpoint_freq: env-steps between checkpoints (``CheckpointCallback``).
        verbose: SB3 verbosity (1 -> the default logger prints ``ep_rew_mean``).
        log_metrics: if True (default), attach an SB3 logger that writes
            ``progress.csv`` + TensorBoard ``tfevents`` into ``runs/<run_name>/``
            AND keeps the stdout table (formats ``["stdout", "csv", "tensorboard"]``).
            This is a SINGLE logging mechanism — we do NOT also pass
            ``tensorboard_log=`` to PPO, so tfevents are written exactly once.

    Returns:
        The trained ``PPO`` model (also saved to ``<models_dir>/<run_name>.zip``).
    """
    if env_factory is None and game_path is None:
        raise ValueError(
            "train_local needs either game_path (real Unity) or env_factory (test seam)."
        )

    device = _resolve_device(device)

    # Seed everything BEFORE building the env / model so numpy/torch/python RNGs are
    # deterministic. set_random_seed also seeds torch CUDA when using_cuda is True.
    set_random_seed(seed, using_cuda=(device == "cuda"))

    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # --- env (single-core: bare env; SB3 wraps it in a DummyVecEnv internally) --------
    if env_factory is not None:
        env = env_factory()
    else:
        env = TankEnv(game_path=str(game_path), image_based=True, env_p=3, rand_opp=True)

    # --- frozen pretrained CNN wired via policy_kwargs (Route B; see module docstring) -
    policy_kwargs = {
        "features_extractor_class": PretrainedNatureCNN,
        "features_extractor_kwargs": {"freeze": not unfreeze},
        # normalize_images left at SB3 default True (pretrain divided by 255).
    }

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    # --- run manifest (light reproducibility: config the run was built with) ----------
    _write_manifest(
        run_dir / "manifest.json",
        {
            "run_name": run_name,
            "timesteps": timesteps,
            "seed": seed,
            "frozen": not unfreeze,
            "device": device,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "game_path": None if game_path is None else str(game_path),
        },
    )

    # --- metrics logging (single mechanism: set_logger with 3 formats) ----------------
    # configure(run_dir, ["stdout","csv","tensorboard"]) writes progress.csv AND a
    # tfevents file into runs/<run_name>/ while preserving the stdout table. We attach
    # it via set_logger and deliberately do NOT pass tensorboard_log= to PPO above,
    # so tfevents are written exactly once (no double-write). plot_metrics reads the
    # resulting progress.csv. The default-on flag keeps every real run observable.
    if log_metrics:
        new_logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(run_dir),
        name_prefix=run_name,
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_cb, progress_bar=False)

    final_path = Path(models_dir) / f"{run_name}.zip"
    model.save(str(final_path))
    return model


def _write_manifest(path: Path, data: dict) -> None:
    """Write a STRICT-JSON run manifest (light reproducibility record)."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tank_twin.train",
        description="Train one PPO agent vs. the random opponent (frozen pretrained CNN).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--game-path",
        type=Path,
        required=True,
        help="Path to the Unity build executable to train against.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed (reproducibility).")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Load the pretrained CNN but leave it TRAINABLE (default: frozen).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="m1-local",
        help="Run name -> runs/<run-name>/ checkpoints + models/<run-name>.zip.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device (auto -> cuda if available, else cpu).",
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="PPO rollout length.")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="PPO minibatch size."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--no-metrics",
        dest="log_metrics",
        action="store_false",
        help="Disable the progress.csv + tfevents metrics logger (default: enabled).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m tank_twin.train --game-path <build> [...]``."""
    args = _build_parser().parse_args(argv)
    train_local(
        timesteps=args.timesteps,
        game_path=args.game_path,
        seed=args.seed,
        unfreeze=args.unfreeze,
        run_name=args.run_name,
        device=args.device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_metrics=args.log_metrics,
    )


if __name__ == "__main__":
    main()
