"""``RandomAgent`` — a seeded uniform-random policy (an exploration / baseline driver).

A model-free :class:`pop_trainer.core.agent.Agent` whose ``act`` ignores the observation and
draws a fresh uniform-random action each step: move/aim channels uniform in ``[-1, 1]`` and a
Bernoulli ``fire`` (1.0 when a uniform draw clears :attr:`RandomAgent.fire_prob`). The stream
is a :func:`numpy.random.default_rng` seeded at construction and re-seedable via
:meth:`RandomAgent.reset`, so a given seed yields a fully deterministic, reproducible action
sequence.

Random coverage is the broadest (least structured) form of exploration — it touches every
state variable diffusely rather than maximising any one, which makes it the natural baseline
alongside the targeted coverage agents.
"""

from __future__ import annotations

import numpy as np

from pop_trainer.agents.base import (
    A_FIRE,
    ACTION_LEN,
    validate_action,
)

__all__ = ["RandomAgent"]


class RandomAgent:
    """Uniform-random actions, seeded and deterministic given the seed.

    ``act`` ignores ``obs`` and returns ``[move_x, move_y, aim_x, aim_y, fire]`` with the four
    move/aim channels drawn uniformly from ``[-1, 1]`` and ``fire`` set to 1.0 with probability
    ``fire_prob`` (else 0.0). The RNG is fixed at construction; :meth:`reset` re-seeds it so two
    same-seed agents (or one agent reset to the same seed) replay an identical sequence.
    """

    def __init__(self, seed: int | None = None, *, fire_prob: float = 0.5):
        self.fire_prob = float(fire_prob)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs) -> list[float]:  # noqa: ARG002 (random by contract: ignores obs)
        action = [0.0] * ACTION_LEN
        # Four move/aim channels uniform in [-1, 1].
        for i in range(A_FIRE):
            action[i] = float(self._rng.uniform(-1.0, 1.0))
        action[A_FIRE] = 1.0 if self._rng.random() < self.fire_prob else 0.0
        return validate_action(action)

    def reset(self, *, seed: int | None = None) -> None:
        """Re-seed the RNG; ``seed=None`` restores the construction seed for an exact replay."""
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        return f"RandomAgent(seed={self._seed!r}, fire_prob={self.fire_prob})"
