"""``pop_trainer.env`` — the simulator interface (the swap point).

The Gymnasium environment wrapping the Unity sim over the socket, plus the budget-based
shaped reward + episode-boundary logic. Everything downstream (``models`` / ``rl``) speaks
the gymnasium API, so a different simulator could implement the same interface unchanged.

Boundary: imports ``core`` only (plus ``gymnasium`` + ``numpy``). Imports NOTHING from
``models`` / ``data`` / ``pretraining`` / ``rl`` and NOTHING from ``tank_twin``. No torch,
no stable-baselines3.

The env owns NEITHER player — it is a PURE TRANSPORT. ``TankEnv.step(action,
opponent_action=None)`` sends ``{1: a1, 2: a2}`` where BOTH actions come from the CALLER
(``opponent_action=None`` defaults to a zero no-op). The DRIVER (collection / demo / the
future training wrapper) runs both policies and computes player2's perspective-flipped view
via :func:`pop_trainer.core.state.split_state_for_opponent`. BOTH actions are captured in
``info["p1_action"]`` / ``info["p2_action"]``.

Modules:

* :mod:`pop_trainer.env.tank_env` — :class:`TankEnv` (reset/step -> the gymnasium 5-tuple);
  the OBSERVATION is the real rendered pixel frame, the 52-float state is in ``info``.
* :mod:`pop_trainer.env.rewards` — the PURE budget-based shaped reward
  (:func:`shaped_step_reward` / :func:`time_penalty_per_step`): a per-step time penalty plus
  the win/loss terminal added on the decided step.
"""

from pop_trainer.env.rewards import shaped_step_reward, time_penalty_per_step
from pop_trainer.env.tank_env import (
    ACTION_DIM,
    DEFAULT_FRAME_SHAPE,
    TankEnv,
)

__all__ = [
    "TankEnv",
    "ACTION_DIM",
    "DEFAULT_FRAME_SHAPE",
    "shaped_step_reward",
    "time_penalty_per_step",
]
