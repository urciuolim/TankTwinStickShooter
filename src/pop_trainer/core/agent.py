"""The Agent Protocol — the shared decision-maker interface (``act(obs) -> action``).

This module DEFINES the structural contract that every decision-maker (``player1`` /
``player2``) satisfies, with NOTHING else: it is type-only. It lives in ``core`` precisely
so :mod:`pop_trainer.env` can be typed against an agent WITHOUT importing ``agents`` or
torch — ``agents`` implements the Protocol and ``env`` merely consumes it, which keeps
``env`` torch-free. Pure stdlib + ``typing``; no numpy, no torch, nothing internal.

The single required behaviour is :meth:`Agent.act`: given an observation ``obs`` (whatever
the caller hands the agent — a rendered pixel frame OR a 52-float state vector; the
Protocol deliberately constrains NEITHER its type) it returns an ``action``, the env's
5-float ``[move_x, move_y, aim_x, aim_y, fire]`` list. The action's length constant is NOT
owned here (it belongs to ``agents`` / ``data``); this module stays type-only.

Two protocols, on purpose, because of how :func:`typing.runtime_checkable` works:

* :class:`Agent` is ``@runtime_checkable`` and declares ONLY ``act``. A
  ``runtime_checkable`` Protocol's ``isinstance`` check tests merely that the named methods
  EXIST, so the runtime-checkable surface must be exactly the REQUIRED surface — ``act``.
  A pure, stateless agent that implements only ``act`` therefore passes
  ``isinstance(x, Agent)``.
* :class:`StatefulAgent` extends :class:`Agent` with the OPTIONAL ``reset`` method (for
  seeded / stateful agents). It is intentionally NOT ``runtime_checkable``: folding
  ``reset`` into the runtime-checkable Protocol would make a stateless ``act``-only agent
  FAIL ``isinstance``, which would wrongly reject a valid agent. ``reset`` thus lives in a
  separate static-only Protocol — documented and statically checkable, never required at
  runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Agent", "StatefulAgent"]


@runtime_checkable
class Agent(Protocol):
    """A decision-maker: maps an observation to an env action.

    The REQUIRED (and only runtime-checkable) surface. ``isinstance(x, Agent)`` is true for
    any object exposing an ``act`` method, which is why this Protocol declares ``act`` alone.
    """

    def act(self, obs):
        """Return the env action for ``obs``.

        ``obs`` is whatever the caller passes the agent (a rendered pixel frame OR a 52-float
        state vector — unconstrained here). The return value is the env's 5-float action list
        ``[move_x, move_y, aim_x, aim_y, fire]``.
        """
        ...


class StatefulAgent(Agent, Protocol):
    """An :class:`Agent` that also supports an OPTIONAL seeded :meth:`reset`.

    Static-only (NOT ``runtime_checkable``): stateful / seeded agents may implement
    ``reset`` to clear or re-seed internal state between episodes, but pure agents need not
    — they still satisfy :class:`Agent`. Use this Protocol for static typing where the
    caller relies on ``reset`` being present.
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Clear or re-seed any internal episode state; ``seed`` makes it deterministic."""
        ...
