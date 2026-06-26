"""``pop_trainer.rl`` — online RL training (SB3 PPO + self-play wiring).

The RL component runs Stable-Baselines3 PPO with a ``CnnPolicy`` over the env's pixel
observations, plus the self-play machinery (opponent roster, ELO) layered as WRAPPERS over the
trainer rather than woven into the PPO core. This Phase-1 slice ships only the policy<->encoder
seam:

* :class:`~pop_trainer.rl.extractor.EncoderExtractor` — an SB3 ``BaseFeaturesExtractor`` that wraps
  the shared :class:`~pop_trainer.models.Encoder`, owning the pretrained-encoder checkpoint load
  and the freeze path so the policy/value heads read the standardized vision embedding.

The PPO train loop, callbacks, opponent provider, and ELO land in later RL tasks.

Boundary: ``rl`` imports ``core`` / ``env`` / ``models`` / ``agents``; it does NOT import ``data``
or ``pretraining`` (the pretrained encoder is consumed as a loaded ``state_dict`` artifact — there
is no ``models.from_pretrained``). No cycles.
"""

from pop_trainer.rl.extractor import EncoderExtractor

__all__ = ["EncoderExtractor"]
