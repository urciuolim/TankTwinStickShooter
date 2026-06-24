"""Rule-based agents — fixed / scripted / tactical decision-makers over the 52-float state.

Each class here is a model-free :class:`pop_trainer.core.agent.Agent` whose ``act`` is handed
its OWN first-person 52-float state view (or an inbound state dict; see
:func:`pop_trainer.agents.base.as_state_vector`). Per the self-play convention of this package
(documented in :mod:`pop_trainer.agents`), a state-reading agent treats ``PLAYER_1`` of the
view it receives as ITSELF — so :class:`AimAtPlayer2Agent` aiming at ``PLAYER_2`` of its view
is aiming at the OTHER player from its own perspective.

Contents:

* :class:`IdleAgent` — the zero action. The simplest fixed agent (low information; a
  ``player2`` option, not a primary data driver).
* :class:`ConstantAgent` — always the same fixed action (validated at construction).
* :class:`ScriptedCycleAgent` — cycles a fixed ordered list of actions, advancing an INTERNAL
  step counter on each ``act``; :meth:`ScriptedCycleAgent.reset` rewinds the counter to 0.
* :class:`AimAtPlayer2Agent` — the one tactical rule kept: aim the unit vector from the acting
  player toward the OTHER player and fire. Acts as ``PLAYER_1`` on its own view by default; a
  player-bind option drives the non-default slot.

stdlib + numpy compatible; imports :mod:`pop_trainer.core.state` for the schema accessors only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pop_trainer.agents.base import as_state_vector, validate_action
from pop_trainer.core import state as state_schema

__all__ = [
    "IdleAgent",
    "ConstantAgent",
    "ScriptedCycleAgent",
    "AimAtPlayer2Agent",
]


def _check_player(player: int) -> None:
    """Raise ``ValueError`` unless ``player`` is PLAYER_1 / PLAYER_2 (public schema constants)."""
    if player not in (state_schema.PLAYER_1, state_schema.PLAYER_2):
        raise ValueError(
            f"player must be PLAYER_1 ({state_schema.PLAYER_1}) or "
            f"PLAYER_2 ({state_schema.PLAYER_2}), got {player!r}"
        )


class IdleAgent:
    """Returns the zero action ``[0, 0, 0, 0, 0]`` every step — no move, no aim, no fire.

    Low information by design; kept as a ``player2`` option, not a primary data driver.
    """

    def act(self, obs) -> list[float]:  # noqa: ARG002 (idle by contract: ignores obs)
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def __repr__(self) -> str:
        return "IdleAgent()"


class ConstantAgent:
    """Always returns the same fixed action, ignoring the observation.

    The action is validated (length 5) and frozen at construction; each call returns a fresh
    copy so a caller mutating the returned list can never corrupt the agent. A low-information
    ``player2`` option, not a primary data driver.
    """

    def __init__(self, action: Sequence[float]):
        self._action = validate_action(action)

    def act(self, obs) -> list[float]:  # noqa: ARG002 (constant by contract)
        return list(self._action)

    def __repr__(self) -> str:
        return f"ConstantAgent({self._action!r})"


class ScriptedCycleAgent:
    """Cycle a fixed ordered list of actions, advancing an INTERNAL step counter each ``act``.

    Because :meth:`pop_trainer.core.agent.Agent.act` takes only the observation, this agent
    carries its own step counter: the n-th ``act`` returns ``actions[n % len(actions)]``, and
    :meth:`reset` rewinds the counter to 0. A single-element cycle degenerates to a constant
    agent. A low-information ``player2`` option, not a primary data driver.
    """

    def __init__(self, actions: Sequence[Sequence[float]]):
        seq = [validate_action(a) for a in actions]
        if not seq:
            raise ValueError("ScriptedCycleAgent needs at least one action")
        self._actions = seq
        self._step = 0

    def act(self, obs) -> list[float]:  # noqa: ARG002 (scripted by internal counter)
        action = list(self._actions[self._step % len(self._actions)])
        self._step += 1
        return action

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002 (no entropy to seed)
        """Rewind the internal step counter to 0 so the cycle restarts from the first action."""
        self._step = 0

    def __len__(self) -> int:
        return len(self._actions)

    def __repr__(self) -> str:
        return f"ScriptedCycleAgent(period={len(self._actions)})"


def _aim_at_other_player(state, player: int) -> list[float]:
    """The unit-vector-at-the-other-player action for ``player``; no move, fire on contact.

    Reads both tanks' positions from the 52-float state and returns ``[0, 0, aim_x, aim_y, 1.0]``
    where ``(aim_x, aim_y)`` is the unit vector from ``player`` toward the OTHER player. If the two
    tanks coincide (degenerate; zero distance) the aim is ``(0, 0)`` and fire is held (0.0) — a
    well-defined fallback rather than a divide-by-zero. Same (state, player) -> same action.
    """
    vec = as_state_vector(state)
    other = state_schema.PLAYER_2 if player == state_schema.PLAYER_1 else state_schema.PLAYER_1
    sx, sy = state_schema.position(vec, player)
    ox, oy = state_schema.position(vec, other)
    dx = float(ox) - float(sx)
    dy = float(oy) - float(sy)
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [0.0, 0.0, dx / dist, dy / dist, 1.0]


class AimAtPlayer2Agent:
    """Aim the unit vector toward the OTHER player and fire — the one tactical rule kept.

    By default the agent acts as ``PLAYER_1`` on its own first-person view, so it aims at
    ``PLAYER_2`` of that view: the other player. The optional ``player`` bind (PLAYER_1 /
    PLAYER_2, validated at construction) drives the non-default slot for callers that hand the
    agent an un-flipped shared state. Coincident positions hold fire. Stateless.
    """

    def __init__(self, player: int = state_schema.PLAYER_1):
        _check_player(player)
        self._player = player

    def act(self, obs) -> list[float]:
        return _aim_at_other_player(obs, self._player)

    @property
    def player(self) -> int:
        return self._player

    def __repr__(self) -> str:
        return f"AimAtPlayer2Agent(player={self._player})"
