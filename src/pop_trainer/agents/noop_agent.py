"""``NoOpAgent`` — the stationary curriculum floor (an obs-agnostic zero-action policy).

A model-free :class:`pop_trainer.core.agent.Agent` whose ``act`` ignores the observation and
returns the env's ZERO action — ``[move_x, move_y, aim_x, aim_y, fire]`` all ``0.0`` (no move,
no aim, no fire). It is the easiest / stationary TRAINING opponent: the curriculum FLOOR the RL
learner first faces. Stateless (no RNG / map state), so it needs neither ``reset`` nor
``set_map`` — ``act`` alone satisfies the runtime-checkable :class:`Agent` Protocol.
"""

from __future__ import annotations

from pop_trainer.agents.base import ACTION_LEN, validate_action

__all__ = ["NoOpAgent"]


class NoOpAgent:
    """The env's zero action every step, regardless of the observation.

    ``act`` ignores ``obs`` and returns the canonical validated 5-float zero action
    ``[0.0, 0.0, 0.0, 0.0, 0.0]``. A stationary opponent with no internal state.
    """

    def act(self, obs) -> list[float]:  # noqa: ARG002 (stationary by contract: ignores obs)
        return validate_action([0.0] * ACTION_LEN)

    def __repr__(self) -> str:
        return "NoOpAgent()"
