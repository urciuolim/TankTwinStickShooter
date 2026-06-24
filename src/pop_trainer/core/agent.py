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
* :class:`StatefulAgent` extends :class:`Agent` with the OPTIONAL ``reset`` and ``set_map``
  methods (for seeded / stateful and map-aware agents). These live in a SEPARATE Protocol —
  NOT folded into :class:`Agent` — to keep the runtime-checkable surface exactly the REQUIRED
  surface (``act``): if ``reset`` / ``set_map`` were on :class:`Agent`, a stateless ``act``-only
  agent would FAIL ``isinstance(x, Agent)`` and be wrongly rejected. The optional methods are
  consumed by probing with ``getattr`` / ``hasattr`` at runtime — never via ``isinstance`` — so
  :class:`StatefulAgent` exists for STATIC typing.

The map hook (:meth:`StatefulAgent.set_map`) is how a map-aware agent receives the static
wall layout once per episode. The env / demo / collection call it via ``getattr`` ONLY when
the agent exposes it (it is never required), passing the
:class:`pop_trainer.core.protocol.WallLayout` from ``info["map"]``. Map-agnostic agents (e.g.
``RandomAgent``) simply do not implement it. ``WallLayout`` is referenced under
``TYPE_CHECKING`` only — importing it at runtime would create an ``agent`` -> ``protocol`` ->
``state`` cycle, and this module must stay import-free of the rest of ``core`` at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pop_trainer.core.protocol import WallLayout

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
    """An :class:`Agent` that also supports OPTIONAL seeded :meth:`reset` / :meth:`set_map`.

    A STATIC-typing Protocol — the optional methods are consumed via ``getattr`` / ``hasattr``,
    never ``isinstance``: stateful / seeded agents may implement ``reset`` to clear or re-seed
    internal state between episodes, and map-aware agents may implement ``set_map`` to receive
    the static wall layout — but pure / map-agnostic agents need neither and still satisfy
    :class:`Agent`. Use this Protocol for static typing where the caller relies on ``reset`` /
    ``set_map`` being present.
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Clear or re-seed any internal episode state; ``seed`` makes it deterministic."""
        ...

    def set_map(self, layout: WallLayout) -> None:
        """Receive the static :class:`~pop_trainer.core.protocol.WallLayout` for this episode.

        Called once per episode (on map load / change) for map-aware agents to (re)build any
        map-derived state — e.g. a coverage policy's free/reachable grid. The caller invokes
        it ONLY when the agent exposes it (``getattr``/``hasattr``); a map-agnostic agent need
        not implement it.
        """
        ...
