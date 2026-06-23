"""Deterministic collection policies: PURE callables ``state -> action_list``.

The collection driver (:mod:`pop_trainer.data.collect`) feeds the current state to a policy
each step to compute the agent action it applies via ``TankEnv.step``. A policy here is a pure
function of the decoded wire state (a 52-float sequence, OR the inbound state dict) to a 5-float
action ``[move_x, move_y, aim_x, aim_y, fire]`` — no socket, no RNG side-channel, no global
mutable state — so a recorded ``(frame, state, action)`` trajectory is REPRODUCIBLE from the
seed/spec alone and the policy is unit-testable with no live game.

"Deterministic" means: same input -> same output, every call. A policy MAY carry a fixed
parameter (a constant action, a scripted cycle keyed off the step index) but draws NO entropy
internally — any randomness a caller wants is supplied as an explicit seeded parameter, never
hidden in the policy. This keeps the trajectory a deterministic function of (policy spec, seed)
exactly as the collection contract requires.

Action channel order matches the Unity wire action ``[move_x, move_y, aim_x, aim_y, fire]``
(the same 5-float list the step message ``{1: a1, 2: a2}`` carries). ``fire`` is 0.0 / 1.0.

Provided policies:

* :func:`idle_policy` — the zero action (no move, no aim, no fire). The simplest fixed policy.
* :func:`ConstantPolicy` — always returns a fixed action (validated at construction).
* :func:`ScriptedCyclePolicy` — cycles a fixed ordered list of actions, indexed by an explicit
  step counter the caller passes in (so the policy itself stays pure / stateless).
* :func:`aim_at_opponent_policy` — a 1-arg ``state -> action`` policy: aim the unit vector from
  PLAYER_1 toward the opponent and fire (a deterministic function of the state). The optional
  ``player`` is KEYWORD-ONLY so the collector's positional ``policy(state, step)`` call can never
  bind the step int to it — to aim a DIFFERENT slot (e.g. P2) through the collector, use the
  player-bound :class:`AimAtOpponentPolicy` / :func:`aim_at_opponent_factory` instead.
* :class:`AimAtOpponentPolicy` — a player-BOUND, 1-arg callable form of the aim policy (its
  ``__call__(state)`` aims the configured player at the opponent), so it plugs into the collector
  as the agent policy for either perspective. :func:`aim_at_opponent_factory` is the closure
  equivalent.

stdlib + numpy; imports :mod:`pop_trainer.core.state` for the schema accessors only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pop_trainer.core import state as state_schema

__all__ = [
    "ACTION_LEN",
    "idle_policy",
    "ConstantPolicy",
    "ScriptedCyclePolicy",
    "aim_at_opponent_policy",
    "AimAtOpponentPolicy",
    "aim_at_opponent_factory",
    "validate_action",
]

ACTION_LEN = 5

# Action channel indices (mirror the Unity wire action ordering).
A_MOVE_X = 0
A_MOVE_Y = 1
A_AIM_X = 2
A_AIM_Y = 3
A_FIRE = 4


def _check_player(player: int) -> None:
    """Raise ``ValueError`` unless ``player`` is PLAYER_1 / PLAYER_2 (public schema constants)."""
    if player not in (state_schema.PLAYER_1, state_schema.PLAYER_2):
        raise ValueError(
            f"player must be PLAYER_1 ({state_schema.PLAYER_1}) or "
            f"PLAYER_2 ({state_schema.PLAYER_2}), got {player!r}"
        )


def _as_state_vector(state) -> Sequence:
    """Accept either an inbound state dict (``{"state": [...]}``) or a raw 52-float sequence.

    Returns the raw 52-float sequence. Lets a policy be called with the decoded protocol dict
    OR the bare vector. Does NOT copy — the caller owns the sequence.
    """
    if isinstance(state, dict):
        vec = state.get("state")
        if vec is None:
            raise ValueError("state dict has no 'state' key")
        return vec
    return state


def validate_action(action) -> list[float]:
    """Coerce ``action`` to a length-5 list of floats; raise ``ValueError`` on a bad length.

    Pure validator shared by the policy constructors and usable by a caller before sending.
    ``fire`` is NOT clamped to {0, 1} here (the wire tolerates any float); callers that want a
    hard binary fire pass 0.0 / 1.0 explicitly.
    """
    vals = [float(x) for x in action]
    if len(vals) != ACTION_LEN:
        raise ValueError(f"action must have {ACTION_LEN} floats, got {len(vals)}")
    return vals


def idle_policy(state) -> list[float]:  # noqa: ARG001 (pure: ignores state by contract)
    """The zero action ``[0, 0, 0, 0, 0]`` — no move, no aim, no fire. Pure / stateless."""
    return [0.0, 0.0, 0.0, 0.0, 0.0]


class ConstantPolicy:
    """A policy that ALWAYS returns the same fixed action, ignoring the state.

    The action is validated (length 5) and frozen at construction; each call returns a fresh
    copy so a caller mutating the returned list can never corrupt the policy.
    """

    def __init__(self, action: Sequence[float]):
        self._action = validate_action(action)

    def __call__(self, state) -> list[float]:  # noqa: ARG002 (constant by contract)
        return list(self._action)

    def __repr__(self) -> str:
        return f"ConstantPolicy({self._action!r})"


class ScriptedCyclePolicy:
    """Cycle a fixed ordered list of actions, indexed by an explicit ``step`` the caller passes.

    The policy holds NO step counter of its own (it stays pure): the caller supplies the step
    index, and the action returned is ``actions[step % len(actions)]``. Same (actions, step) ->
    same action, always. A single-element cycle degenerates to a constant policy.
    """

    def __init__(self, actions: Sequence[Sequence[float]]):
        seq = [validate_action(a) for a in actions]
        if not seq:
            raise ValueError("ScriptedCyclePolicy needs at least one action")
        self._actions = seq

    def __call__(self, state, step: int) -> list[float]:  # noqa: ARG002 (scripted by step)
        return list(self._actions[step % len(self._actions)])

    def __len__(self) -> int:
        return len(self._actions)

    def __repr__(self) -> str:
        return f"ScriptedCyclePolicy(period={len(self._actions)})"


def _aim_at_opponent(state, player: int) -> list[float]:
    """Core aim math: the unit-vector-at-opponent action for ``player``; no move.

    Reads both tanks' positions from the 52-float state and returns
    ``[0, 0, aim_x, aim_y, 1.0]`` where ``(aim_x, aim_y)`` is the unit vector from ``player``
    toward the opponent. If the two tanks coincide (degenerate; zero distance) the aim is
    ``(0, 0)`` and fire is held (0.0) — a well-defined fallback rather than a divide-by-zero.
    Pure: same (state, player) -> same action. ``player`` is validated by the schema accessors.
    """
    vec = _as_state_vector(state)
    opponent = state_schema.PLAYER_2 if player == state_schema.PLAYER_1 else state_schema.PLAYER_1
    sx, sy = state_schema.position(vec, player)
    ox, oy = state_schema.position(vec, opponent)
    dx = float(ox) - float(sx)
    dy = float(oy) - float(sy)
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [0.0, 0.0, dx / dist, dy / dist, 1.0]


def aim_at_opponent_policy(state, *, player: int = state_schema.PLAYER_1) -> list[float]:
    """Aim the unit vector from ``player`` (default PLAYER_1) toward the opponent and fire.

    A deterministic, state-REACTIVE scripted policy and a 1-arg ``state -> action`` callable: it
    takes exactly the state positionally. ``player`` is KEYWORD-ONLY by design so the collector's
    positional ``policy(state, step)`` call can NEVER bind the step int to ``player`` (it raises a
    clean ``TypeError`` for the extra positional, which the loop's arity dispatch avoids by
    classifying this as a 1-arg policy). To drive a NON-default slot (the P2 perspective) through
    the collector, wire in :class:`AimAtOpponentPolicy` / :func:`aim_at_opponent_factory` — which
    are themselves 1-arg callables — rather than reaching for the keyword here.

    Pure: same ``state`` -> same action. See :func:`_aim_at_opponent` for the geometry.
    """
    return _aim_at_opponent(state, player)


class AimAtOpponentPolicy:
    """Player-BOUND aim policy: a 1-arg ``state -> action`` callable for a fixed slot.

    Binds ``player`` (PLAYER_1 / PLAYER_2, validated at construction) so ``__call__(state)`` aims
    that player at its opponent and fires. Its 1-arg signature plugs straight into the collector's
    ``policy(state)`` call, mirroring the :class:`ConstantPolicy` / :class:`ScriptedCyclePolicy`
    callable-class style. Pure / stateless.
    """

    def __init__(self, player: int = state_schema.PLAYER_1):
        # Validate the slot eagerly so a bad player fails at construction, not mid-episode.
        _check_player(player)
        self._player = player

    def __call__(self, state) -> list[float]:
        return _aim_at_opponent(state, self._player)

    @property
    def player(self) -> int:
        return self._player

    def __repr__(self) -> str:
        return f"AimAtOpponentPolicy(player={self._player})"


def aim_at_opponent_factory(player: int = state_schema.PLAYER_1):
    """Return a 1-arg ``state -> action`` closure that aims ``player`` at the opponent and fires.

    The closure equivalent of :class:`AimAtOpponentPolicy` for callers who prefer a function. The
    ``player`` is validated up front (so a bad slot raises here, not on first call) and captured;
    the returned callable takes exactly the state, so it plugs into the collector's 1-arg path.
    """
    _check_player(player)

    def _policy(state) -> list[float]:
        return _aim_at_opponent(state, player)

    return _policy
