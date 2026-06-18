"""Tests for tank_twin.features — the pretrained CNN feature extractor (M1, task 1.5).

These import torch + stable_baselines3 (NOT pure) — they exercise the real load of the
2021 pretrained weights (``models/1M_pretrained_*.pth``) into an SB3 NatureCNN at the
``(3, 36, 60)`` channel-first obs shape (env_p == 3). They are the early-warning net for
the day a future SB3 changes NatureCNN's ``.cnn`` / ``.linear`` layout or shapes.

Asserts:
  (a) a loaded conv weight equals the saved tensor (weights really loaded, not re-init);
  (b) ``freeze=True`` -> ``requires_grad == False`` on cnn + linear; ``freeze=False`` -> True;
  (c) ``forward(zeros((1, 3, 36, 60)))`` returns shape ``(1, 512)``.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from tank_twin.features import (
    DEFAULT_CNN_PATH,
    DEFAULT_LINEAR_PATH,
    PretrainedNatureCNN,
    load_pretrained_into_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CNN_PATH = REPO_ROOT / "models" / "1M_pretrained_cnn.pth"
LINEAR_PATH = REPO_ROOT / "models" / "1M_pretrained_linear.pth"

# Channel-first (3, 36, 60): what SB3 hands the extractor after VecTransposeImage
# transposes the (36, 60, 3) uint8 obs. env_p == 3 (20*3 x 12*3 -> 60 x 36).
OBS_SHAPE = (3, 36, 60)


def _obs_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8)


@pytest.fixture(scope="module")
def saved_cnn_sd():
    return th.load(str(CNN_PATH), map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def saved_linear_sd():
    return th.load(str(LINEAR_PATH), map_location="cpu", weights_only=True)


def test_default_paths_point_at_repo_models():
    """The module's default weight paths resolve to the real on-disk shards."""
    assert Path(DEFAULT_CNN_PATH) == CNN_PATH
    assert Path(DEFAULT_LINEAR_PATH) == LINEAR_PATH
    assert CNN_PATH.is_file()
    assert LINEAR_PATH.is_file()


def test_weights_actually_loaded(saved_cnn_sd, saved_linear_sd):
    """(a) A loaded conv weight (and the linear weight) equals the saved tensor."""
    extractor = PretrainedNatureCNN(_obs_space(), freeze=True)

    # First conv [32, 3, 8, 8] — the load-bearing shape that pins env_p == 3.
    th.testing.assert_close(extractor.cnn[0].weight, saved_cnn_sd["0.weight"])
    th.testing.assert_close(extractor.cnn[0].bias, saved_cnn_sd["0.bias"])
    # Last conv too, so we know the whole stack loaded, not just layer 0.
    th.testing.assert_close(extractor.cnn[4].weight, saved_cnn_sd["4.weight"])
    # The 256 -> 512 projection.
    th.testing.assert_close(extractor.linear[0].weight, saved_linear_sd["0.weight"])
    th.testing.assert_close(extractor.linear[0].bias, saved_linear_sd["0.bias"])


def test_freeze_true_disables_grad():
    """(b) freeze=True (the default) -> requires_grad False on cnn + linear."""
    extractor = PretrainedNatureCNN(_obs_space(), freeze=True)
    assert extractor.frozen is True
    assert all(not p.requires_grad for p in extractor.cnn.parameters())
    assert all(not p.requires_grad for p in extractor.linear.parameters())


def test_freeze_default_is_true():
    """Board-approved default: omitting freeze freezes."""
    extractor = PretrainedNatureCNN(_obs_space())
    assert extractor.frozen is True
    assert all(not p.requires_grad for p in extractor.cnn.parameters())


def test_no_freeze_leaves_grad_on():
    """(b) freeze=False (the --no-freeze path) -> requires_grad True on cnn + linear."""
    extractor = PretrainedNatureCNN(_obs_space(), freeze=False)
    assert extractor.frozen is False
    assert all(p.requires_grad for p in extractor.cnn.parameters())
    assert all(p.requires_grad for p in extractor.linear.parameters())


def test_forward_output_shape():
    """(c) forward(zeros((1, 3, 36, 60))) -> (1, 512)."""
    extractor = PretrainedNatureCNN(_obs_space(), freeze=True)
    extractor.eval()
    out = extractor(th.zeros((1, *OBS_SHAPE), dtype=th.float32))
    assert out.shape == (1, 512)


def test_wrong_features_dim_rejected():
    """features_dim must be 512 (the saved linear projection)."""
    with pytest.raises(ValueError, match="features_dim == 512"):
        PretrainedNatureCNN(_obs_space(), features_dim=256)


def test_load_pretrained_into_policy_route_a(saved_cnn_sd):
    """The literal 2021 helper loads + freezes a built CnnPolicy's extractor.

    Uses a tiny dummy env so PPO('CnnPolicy', ...) builds a NatureCNN at the (36,60,3)
    obs shape, mirroring preamble.py:88-90 on a live model.
    """
    from stable_baselines3 import PPO

    env = _PixelDummyEnv()
    model = PPO("CnnPolicy", env, n_steps=8, batch_size=8, device="cpu")

    # Pre-load, the conv weights are random init (not equal to the saved tensor).
    pre = model.policy.features_extractor.cnn[0].weight.detach().clone()
    assert not th.allclose(pre, saved_cnn_sd["0.weight"])

    load_pretrained_into_policy(model, freeze=True)

    th.testing.assert_close(model.policy.features_extractor.cnn[0].weight, saved_cnn_sd["0.weight"])
    assert all(not p.requires_grad for p in model.policy.features_extractor.cnn.parameters())
    assert all(not p.requires_grad for p in model.policy.features_extractor.linear.parameters())


class _PixelDummyEnv(gym.Env):
    """Minimal gymnasium env exposing the (36,60,3) uint8 pixel obs for PPO('CnnPolicy')."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(36, 60, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
