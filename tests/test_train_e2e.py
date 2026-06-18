"""End-to-end pipeline round-trip test (M1, task 3.3) — CPU-only, fake-socket, TINY.

Proves the M1 training pipeline ROUND-TRIPS, NOT that PPO converges (convergence is the
GPU smoke, Wave 4). The whole loop runs with NO Unity and NO GPU:

    env(ScriptedUnity fake transport)
        -> PPO("CnnPolicy", PretrainedNatureCNN frozen, device="cpu")
        -> train a handful of timesteps
        -> CheckpointCallback into tmp_path
        -> PPO.load(final .zip)
        -> evaluate_winrate(reloaded model, fresh env, a few episodes)
        -> assert win-rate in [0,1] AND reloaded .predict(obs) -> action shape (5,) in space.

It imports torch / sb3 (so it is tagged ``@pytest.mark.e2e``) and SKIPS GRACEFULLY when
the pretrained ``.pth`` weights are absent — the rest of the suite still passes without
them. Reuses the ``ScriptedUnity`` fake transport from ``tests/test_env.py`` (no Unity).
"""

from pathlib import Path

import numpy as np
import pytest

from tank_twin.env import TankEnv
from tank_twin.features import DEFAULT_CNN_PATH, DEFAULT_LINEAR_PATH

# Reuse the canned-protocol fake transport (answers arbitrary reset/step counts) from the
# env contract test. Under pytest's default "prepend" import mode the tests dir is on
# sys.path, so the sibling module is importable by basename (no `tests/__init__.py`).
from test_env import ScriptedUnity

pytestmark = pytest.mark.e2e

_WEIGHTS_PRESENT = Path(DEFAULT_CNN_PATH).is_file() and Path(DEFAULT_LINEAR_PATH).is_file()
_SKIP_REASON = "pretrained CNN weights (models/1M_pretrained_*.pth) absent; skipping e2e round-trip"


def _make_scripted_env() -> TankEnv:
    """A TankEnv driven by the fake Unity transport (winner=0 -> short, winnable episodes)."""
    # win_after=2 keeps episodes tiny so a handful of timesteps spans several episodes.
    return TankEnv(game_path=None, transport=ScriptedUnity(win_after=2, winner=0))


@pytest.mark.skipif(not _WEIGHTS_PRESENT, reason=_SKIP_REASON)
def test_train_checkpoint_reload_evaluate_roundtrips(tmp_path: Path):
    """The full pipeline builds, trains a few steps, checkpoints, reloads, and evaluates."""
    from tank_twin.evaluate import evaluate_winrate
    from tank_twin.train import train_local

    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"

    # TINY run: enough timesteps to fill one PPO rollout (n_steps) and step the optimizer
    # once, on CPU, with the frozen pretrained CNN. This is a round-trip proof, not training.
    model = train_local(
        timesteps=64,
        env_factory=_make_scripted_env,
        seed=0,
        unfreeze=False,  # frozen pretrained CNN (board default)
        run_name="e2e",
        device="cpu",
        n_steps=32,
        batch_size=16,
        learning_rate=3e-4,
        runs_dir=runs_dir,
        models_dir=models_dir,
        checkpoint_freq=32,  # ensure at least one checkpoint is written
        verbose=0,
    )

    # Final model .zip landed in models_dir; at least one checkpoint landed in runs/<name>/.
    final_zip = models_dir / "e2e.zip"
    assert final_zip.is_file(), "train_local did not save the final model .zip"
    checkpoints = list((runs_dir / "e2e").glob("e2e_*_steps.zip"))
    assert checkpoints, "CheckpointCallback wrote no checkpoint into runs/<run-name>/"
    assert (runs_dir / "e2e" / "manifest.json").is_file(), "run manifest not written"

    # The frozen feature extractor really had its grads disabled (freeze-first).
    fx = model.policy.features_extractor
    assert all(not p.requires_grad for p in fx.cnn.parameters())
    assert all(not p.requires_grad for p in fx.linear.parameters())

    # Reload from disk (proves the .zip is a loadable artifact, not just an in-memory model).
    from stable_baselines3 import PPO

    reloaded = PPO.load(str(final_zip), device="cpu")

    # Evaluate the reloaded model over a few greedy episodes; metric must be a valid win-rate.
    eval_env = _make_scripted_env()
    rate = evaluate_winrate(reloaded, eval_env, n_episodes=3, seed=0)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0

    # The reloaded policy predicts a well-formed action in the action space.
    obs, _info = eval_env.reset(seed=0)
    action, _state = reloaded.predict(obs, deterministic=True)
    action = np.asarray(action)
    assert action.shape == (5,)
    assert eval_env.action_space.contains(action.astype(np.float32))
