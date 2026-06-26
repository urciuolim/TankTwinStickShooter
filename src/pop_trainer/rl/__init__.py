"""``pop_trainer.rl`` — online RL training (SB3 PPO + self-play wiring).

The RL component runs Stable-Baselines3 PPO with a ``CnnPolicy`` over the env's pixel
observations, plus the self-play machinery (opponent roster, ELO) layered as WRAPPERS over the
trainer rather than woven into the PPO core. This Phase-1 slice ships:

* :class:`~pop_trainer.rl.extractor.EncoderExtractor` — an SB3 ``BaseFeaturesExtractor`` that wraps
  the shared :class:`~pop_trainer.models.Encoder`, owning the pretrained-encoder checkpoint load
  and the freeze path so the policy/value heads read the standardized vision embedding.
* the self-play opponent seam (:mod:`~pop_trainer.rl.selfplay`):
  :class:`~pop_trainer.rl.selfplay.SelfPlayWrapper` packages the symmetric pure-transport
  ``TankEnv`` as a 1-action gym env (SB3 supplies player1; the wrapper drives a scripted player2);
  :class:`~pop_trainer.rl.selfplay.OpponentProvider` rosters the opponents and samples one per
  episode (round-robin / uniform); :class:`~pop_trainer.rl.selfplay.ScriptedOpponent` /
  :class:`~pop_trainer.rl.selfplay.Opponent` are the player2 wrapper + its protocol, resolving the
  canonical ``agents`` selectors via :func:`pop_trainer.agents.make_agent`.

The PPO train loop, callbacks, ELO, and the live self-play smoke land in later RL tasks.

Boundary: ``rl`` imports ``core`` / ``env`` / ``models`` / ``agents``; it does NOT import ``data``
or ``pretraining`` (the pretrained encoder is consumed as a loaded ``state_dict`` artifact — there
is no ``models.from_pretrained``; the opponent seam re-uses ``agents`` selectors +
``core.state.split_state_for_opponent`` directly, never the ``data`` module). No cycles.
"""

from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import (
    Opponent,
    OpponentProvider,
    ScriptedOpponent,
    SelfPlayWrapper,
)

__all__ = [
    "EncoderExtractor",
    "Opponent",
    "OpponentProvider",
    "ScriptedOpponent",
    "SelfPlayWrapper",
]
