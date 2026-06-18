"""Pretrained CNN feature extractor for the SB3 PPO ``CnnPolicy`` (M1, task 1.5).

Loads the 2021 self-supervised pretrained vision weights into an SB3
``NatureCNN`` feature extractor and (by default) FREEZES them — the
board-approved "freeze first, fine-tune later" path. The weights come in two
files, mirroring how ``nature_cnn_pretrain.py`` saved them:

* ``models/1M_pretrained_cnn.pth``    -> the 3-conv stack (the ``.cnn`` submodule)
* ``models/1M_pretrained_linear.pth`` -> the flatten->512 projection (``.linear``)

This is the modern re-expression of the legacy load path
(``PythonScripts/preamble.py:88-90``)::

    model.policy.features_extractor.cnn.load_state_dict(T.load(pretrain + "_cnn.pth"))
    model.policy.features_extractor.linear.load_state_dict(T.load(pretrain + "_linear.pth"))

Route taken: **Route B** — a ``PretrainedNatureCNN(NatureCNN)`` subclass that
loads + freezes in ``__init__`` and is wired via ``policy_kwargs``. Subclassing
``NatureCNN`` (rather than re-declaring the conv stack) reuses SB3's own,
shape-derived architecture, so the ``.cnn`` / ``.linear`` submodule names and the
flatten dimension are exactly what the saved weights expect — no parallel arch to
keep in sync. A ``load_pretrained_into_policy`` helper (the literal 2021 path) is
also provided for callers who already hold a built ``PPO`` model.

Shape contract (FAIL LOUDLY if violated):
- obs is ``(36, 60, 3)`` uint8 channel-last; SB3 auto-transposes to channel-first
  ``(3, 36, 60)`` via ``VecTransposeImage`` before this extractor sees it. The saved
  first conv is ``[32, 3, 8, 8]`` and the saved linear in-features is 256 — that flatten
  dim only arises from the 36x60 grid, i.e. **env_p == 3**. At any other grid size the
  ``load_state_dict`` below raises (size mismatch), which is the intended early warning.

normalize_images: kept at the SB3 default ``True``. The pretrain divided inputs by
255, so the policy MUST feed [0,1] floats to match the training distribution. Do NOT
disable normalization.

This module imports torch + stable_baselines3 and is therefore NOT pure — it is
deliberately kept OUT of the pure-logic import path (``tank_twin/__init__`` does not
import it) so the numpy-only modules and their tests stay torch-free.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn

__all__ = ["PretrainedNatureCNN", "load_pretrained_into_policy"]

# Default on-disk locations of the 2021 pretrained weight shards (repo-relative).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CNN_PATH = _REPO_ROOT / "models" / "1M_pretrained_cnn.pth"
DEFAULT_LINEAR_PATH = _REPO_ROOT / "models" / "1M_pretrained_linear.pth"


def _assert_nature_cnn_shape(extractor: NatureCNN) -> None:
    """Fail loudly if SB3's NatureCNN no longer matches the pretrained contract.

    Guards the two assumptions the load relies on: the ``.cnn`` / ``.linear``
    submodule names, and the first-conv / linear-in shapes that only hold at the
    36x60 (env_p == 3) grid. If a future SB3 renames or restructures NatureCNN this
    raises here rather than silently loading garbage.
    """
    if not (hasattr(extractor, "cnn") and hasattr(extractor, "linear")):
        raise RuntimeError(
            "SB3 NatureCNN no longer exposes .cnn / .linear submodules; the pretrained "
            "load path (preamble.py:88-90) is broken. Re-validate against this SB3 version."
        )
    first_conv = extractor.cnn[0]
    if not isinstance(first_conv, nn.Conv2d) or tuple(first_conv.weight.shape) != (32, 3, 8, 8):
        raise RuntimeError(
            "NatureCNN first conv shape is "
            f"{tuple(getattr(first_conv, 'weight', th.empty(0)).shape)}, expected (32, 3, 8, 8). "
            "The pretrained _cnn.pth will not fit."
        )
    linear_layer = extractor.linear[0]
    if not isinstance(linear_layer, nn.Linear) or linear_layer.in_features != 256:
        raise RuntimeError(
            f"NatureCNN linear in_features is {getattr(linear_layer, 'in_features', None)}, "
            "expected 256. The pretrained _linear.pth (flatten dim) only fits the 36x60 grid "
            "(env_p == 3); this obs space is the wrong size."
        )


def _load_shards(
    extractor: NatureCNN,
    cnn_path: str | Path,
    linear_path: str | Path,
    *,
    freeze: bool,
) -> None:
    """Load the two weight shards into ``extractor.cnn`` / ``.linear`` and (maybe) freeze.

    ``load_state_dict`` uses ``strict=True`` (the default) so a key/shape mismatch raises
    — the deliberate early warning if the grid size or SB3 arch ever drifts.
    """
    _assert_nature_cnn_shape(extractor)
    cnn_sd = th.load(str(cnn_path), map_location="cpu", weights_only=True)
    linear_sd = th.load(str(linear_path), map_location="cpu", weights_only=True)
    extractor.cnn.load_state_dict(cnn_sd)  # strict=True: raises on any mismatch
    extractor.linear.load_state_dict(linear_sd)
    if freeze:
        extractor.cnn.requires_grad_(False)
        extractor.linear.requires_grad_(False)
    else:
        extractor.cnn.requires_grad_(True)
        extractor.linear.requires_grad_(True)


class PretrainedNatureCNN(NatureCNN):
    """SB3 ``NatureCNN`` pre-initialised with the 2021 pretrained vision weights.

    Same architecture as ``NatureCNN`` (3 convs 8/4 -> 4/2 -> 3/1, then
    ``Linear(flatten, 512) + ReLU``); the only difference is that ``__init__`` loads the
    pretrained ``.cnn`` / ``.linear`` weights and, when ``freeze`` is True (the default),
    sets ``requires_grad_(False)`` on both submodules.

    Wire it into a policy via ``policy_kwargs``::

        policy_kwargs = dict(
            features_extractor_class=PretrainedNatureCNN,
            features_extractor_kwargs=dict(freeze=True),  # or freeze=False for --no-freeze
        )
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)

    Keep the policy's ``normalize_images`` at its default ``True`` (the pretrain divided
    by 255). ``features_dim`` is fixed at 512 to match the saved linear projection.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        *,
        freeze: bool = True,
        cnn_path: str | Path = DEFAULT_CNN_PATH,
        linear_path: str | Path = DEFAULT_LINEAR_PATH,
        normalized_image: bool = False,
    ) -> None:
        if features_dim != 512:
            raise ValueError(
                f"PretrainedNatureCNN requires features_dim == 512 (the saved linear "
                f"projection), got {features_dim}."
            )
        super().__init__(
            observation_space,
            features_dim=features_dim,
            normalized_image=normalized_image,
        )
        _load_shards(self, cnn_path, linear_path, freeze=freeze)
        self._frozen = freeze

    @property
    def frozen(self) -> bool:
        """Whether the pretrained submodules were frozen at construction."""
        return self._frozen


def load_pretrained_into_policy(
    model,
    cnn_path: str | Path = DEFAULT_CNN_PATH,
    linear_path: str | Path = DEFAULT_LINEAR_PATH,
    *,
    freeze: bool = True,
):
    """Load the pretrained shards into an already-built PPO model's feature extractor.

    The literal 2021 path (``preamble.py:88-90``): call this AFTER
    ``PPO("CnnPolicy", env, ...)`` and BEFORE ``model.learn(...)``. Provided for callers
    who build the model first; ``PretrainedNatureCNN`` (via ``policy_kwargs``) is the
    preferred, more-testable route. Returns ``model`` for chaining.

    Raises if the policy's feature extractor is not a NatureCNN-shaped module exposing
    ``.cnn`` / ``.linear`` at the expected (env_p == 3) shapes.
    """
    extractor = model.policy.features_extractor
    if not isinstance(extractor, NatureCNN):
        raise TypeError(
            f"Expected a NatureCNN feature extractor (CnnPolicy), got {type(extractor).__name__}. "
            "Build the model with PPO('CnnPolicy', ...) before calling this."
        )
    _load_shards(extractor, cnn_path, linear_path, freeze=freeze)
    return model
