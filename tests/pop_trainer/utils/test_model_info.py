"""Tests for pop_trainer.utils.model_info — the SB3 PPO checkpoint inspector.

These import torch / stable-baselines3 (NOT pure), so the module skips wholesale if either is
absent via the top-level ``importorskip`` (matching the rl / models tests). The default path RUNS:
a TINY PPO is built on CPU over a trivial in-test gymnasium image env (NO Unity, NO socket), wired
with the production :class:`~pop_trainer.rl.extractor.EncoderExtractor` so the
``policy.features_extractor.encoder.trunk`` path the tool reports is genuinely exercised. A 64x64x3
pixel obs selects the cheap DreamerV3 trunk, so the model is small enough to build on CPU.

Coverage:

1. ``collect_model_info`` returns both spaces, the trunk class name, and an id-deduped policy total
   that EQUALS the sum of ``numel`` over the UNIQUE ``model.policy.parameters()`` (the shared
   features extractor counted exactly once — the load-bearing assertion).
2. the shared-extractor dedupe is non-vacuous: SB3 really does alias the extractor across the
   policy's pi/vf references, so a naive per-reference sum OVER-counts vs the deduped total.
3. round-trip through a saved checkpoint: ``load_ppo_cpu`` loads it on CPU and yields the SAME
   trunk + total.
4. ``main([ckpt])`` smoke — returns 0 and prints the spaces / trunk / counts (capsys).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402  (after importorskip, by design)
from stable_baselines3 import PPO  # noqa: E402

from pop_trainer.models import DreamerCNN  # noqa: E402
from pop_trainer.rl.extractor import EncoderExtractor  # noqa: E402
from pop_trainer.utils.model_info import (  # noqa: E402
    collect_model_info,
    format_model_info,
    load_ppo_cpu,
    main,
)

# A small pixel obs: 64x64x3 channels-LAST uint8 (the shape SB3 treats as an image and transposes
# to NCHW before the extractor). At 64x64 the EncoderExtractor picks the cheap DreamerV3 trunk.
OBS_HW = (64, 64)
OBS_SHAPE_HWC = (*OBS_HW, 3)


class _TinyImageEnv(gym.Env):
    """A trivial single-step image env: a uint8 64x64x3 obs + a tiny Box action, instant terminal.

    Enough for SB3 to build a ``CnnPolicy`` PPO over it — the inspector never steps the env, it only
    reads the loaded network — so reset/step are minimal (a zero frame, zero reward, immediate
    ``terminated``). The action space is a small continuous Box so the policy net is tiny.
    """

    metadata: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=OBS_SHAPE_HWC, dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        return np.zeros(OBS_SHAPE_HWC, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):  # noqa: ARG002 (the action is irrelevant to this stub env)
        return self._obs(), 0.0, True, False, {}


def _build_tiny_ppo() -> PPO:
    """Build a TINY CPU PPO with the production EncoderExtractor over the 64x64 image env.

    Mirrors the train integrator's policy wiring (``CnnPolicy`` + ``EncoderExtractor``) at a
    minimal scale so ``policy.features_extractor.encoder.trunk`` resolves to the DreamerV3 trunk and
    the param-count paths are exercised — but tiny ``n_steps`` / ``batch_size`` keep it cheap.
    """
    env = _TinyImageEnv()
    return PPO(
        "CnnPolicy",
        env,
        device="cpu",
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"features_extractor_class": EncoderExtractor},
        seed=0,
    )


def _unique_policy_param_numel(model: PPO) -> int:
    """Sum of ``numel`` over the id-deduped ``model.policy.parameters()`` (the truth to match)."""
    seen: set[int] = set()
    total = 0
    for p in model.policy.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


# --- 1. collect_model_info: spaces + trunk + id-deduped total ---------------------------


def test_collect_reports_spaces_trunk_and_deduped_total():
    """The returned info carries both spaces, the DreamerV3 trunk name, and a correct deduped total.

    The policy ``total`` must equal the sum of ``numel`` over the UNIQUE policy parameters — the
    shared features extractor is counted exactly once (the load-bearing id-dedupe).
    """
    model = _build_tiny_ppo()
    info = collect_model_info(model, checkpoint="tiny")

    assert info.observation_space == repr(model.observation_space)
    assert info.action_space == repr(model.action_space)
    assert info.trunk_class == DreamerCNN.__name__

    # total == sum of numel over id-deduped model.policy.parameters() (extractor counted once).
    assert info.policy.total == _unique_policy_param_numel(model)
    # trainable + frozen partition the total; a fresh PPO trains everything, so nothing is frozen.
    assert info.policy.trainable + info.policy.frozen == info.policy.total
    assert info.policy.frozen == 0
    assert info.policy.trainable == info.policy.total

    # the extractor section is non-empty and (deduped) no larger than the whole policy.
    assert info.extractor.total > 0
    assert info.extractor.total <= info.policy.total


# --- 2. the dedupe is non-vacuous (SB3 really aliases the extractor) --------------------


def test_dedupe_is_non_vacuous():
    """A naive per-reference sum OVER-counts vs the id-deduped total — proving the dedupe matters.

    SB3's ActorCriticPolicy references the one features-extractor instance under
    ``features_extractor`` / ``pi_features_extractor`` / ``vf_features_extractor``; summing each
    reference's params (with overlaps) exceeds the unique total, so the dedupe is load-bearing.
    """
    model = _build_tiny_ppo()
    policy = model.policy

    naive = 0
    for ref_name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        ref = getattr(policy, ref_name, None)
        if ref is not None:
            naive += sum(p.numel() for p in ref.parameters())

    deduped = _unique_policy_param_numel(model)
    # naive counts the shared extractor multiple times, so it strictly exceeds the deduped total.
    assert naive > deduped

    info = collect_model_info(model)
    assert info.policy.total == deduped


# --- 3. checkpoint round-trip via load_ppo_cpu ------------------------------------------


def test_round_trip_through_saved_checkpoint(tmp_path):
    """Save the tiny PPO, reload via load_ppo_cpu, and confirm the same trunk + deduped total."""
    model = _build_tiny_ppo()
    expected = collect_model_info(model)

    ckpt = tmp_path / "tiny_ppo.zip"
    model.save(str(ckpt))

    loaded = load_ppo_cpu(ckpt)
    info = collect_model_info(loaded, checkpoint=ckpt)

    assert info.trunk_class == expected.trunk_class
    assert info.policy.total == expected.policy.total
    # the reloaded model's own deduped truth matches the reported total, too.
    assert info.policy.total == _unique_policy_param_numel(loaded)
    assert info.checkpoint == str(ckpt)


# --- 4. main() smoke (capsys) -----------------------------------------------------------


def test_main_smoke_prints_report(tmp_path, capsys):
    """main([ckpt]) returns 0 and prints the spaces / trunk / param counts (no numeric scraping)."""
    model = _build_tiny_ppo()
    ckpt = tmp_path / "tiny_ppo.zip"
    model.save(str(ckpt))

    rc = main([str(ckpt)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "observation space:" in out
    assert "action space:" in out
    assert DreamerCNN.__name__ in out
    assert "trainable=" in out
    assert "frozen=" in out


def test_format_model_info_renders_sections():
    """format_model_info includes the trunk name and both param-count section names."""
    model = _build_tiny_ppo()
    text = format_model_info(collect_model_info(model))
    assert DreamerCNN.__name__ in text
    assert "policy:" in text
    assert "features_extractor:" in text
