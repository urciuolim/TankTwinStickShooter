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
* the eval + ELO seam: :func:`~pop_trainer.rl.evaluate.evaluate_winrate` plays greedy episodes vs.
  each opponent in a roster and returns a per-opponent win-rate; ``win_rate`` is the pure counting
  helper; :class:`~pop_trainer.rl.callbacks.EvalWinRateCallback` logs that win-rate periodically at
  rollout boundaries; :mod:`~pop_trainer.rl.elo` holds the pure ELO math.

* the integrator (:mod:`~pop_trainer.rl.train`): :class:`~pop_trainer.rl.train.TrainConfig` +
  :func:`~pop_trainer.rl.train.train_local` compose the seams above into one runnable SB3 PPO
  self-play run — vec-env stack, the ``CnnPolicy`` + ``EncoderExtractor`` policy, periodic eval,
  checkpointing, and a resumable ``state.json`` sidecar. The live Unity ``connection_factory`` is
  re-derived from :mod:`pop_trainer.core.launch` (NEVER imported from ``data``).

Boundary: ``rl`` imports ``core`` (incl. ``core.launch`` / ``core.protocol`` / ``core.config``) /
``env`` / ``models`` / ``agents``; it does NOT import ``data`` or ``pretraining`` (the pretrained
encoder is consumed as a loaded ``state_dict`` artifact — there is no ``models.from_pretrained``;
the opponent seam re-uses ``agents`` selectors + ``core.state.split_state_for_opponent`` directly,
and the live launch is re-derived from ``core.launch``, never the ``data`` module). No cycles.
"""

from pop_trainer.rl.callbacks import EvalWinRateCallback
from pop_trainer.rl.evaluate import evaluate_winrate, win_rate
from pop_trainer.rl.extractor import EncoderExtractor
from pop_trainer.rl.selfplay import (
    DEFAULT_ROSTER,
    Opponent,
    OpponentProvider,
    ScriptedOpponent,
    SelfPlayWrapper,
)
from pop_trainer.rl.train import TrainConfig, train_local

__all__ = [
    "DEFAULT_ROSTER",
    "EncoderExtractor",
    "EvalWinRateCallback",
    "Opponent",
    "OpponentProvider",
    "ScriptedOpponent",
    "SelfPlayWrapper",
    "TrainConfig",
    "evaluate_winrate",
    "train_local",
    "win_rate",
]
