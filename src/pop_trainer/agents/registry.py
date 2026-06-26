"""The canonical agent-selector registry — the SINGLE source of truth for ``string -> agent``.

``agents`` is the natural home for "selector name to a :class:`pop_trainer.core.agent.Agent`",
so this module OWNS that map: :data:`AGENT_SELECTORS` (a selector name -> a seed-taking factory)
and :func:`make_agent` (look up + build, raising on an unknown name). Both the collection runner
(:mod:`pop_trainer.data.collect_runner`) and the demo (:mod:`pop_trainer.demo`) import it from
here rather than each duplicating it, and ``rl`` can reuse the SAME selectors without reaching
into ``data`` (a forbidden import).

The roster is the one map-aware coverage family (its three presets), the map-agnostic
``random`` baseline, and the stationary ``noop`` floor. Every factory takes an OPTIONAL seed so
the signature is uniform even for the stateless agents (``noop`` ignores it).

Boundary: a clean leaf — imports ONLY ``agents`` siblings (the agent classes) + ``core`` (the
``Agent`` type). Nothing from ``data`` / ``demo`` / ``env`` / ``models`` / ``rl`` /
``pretraining`` / ``tank_twin``.
"""

from __future__ import annotations

from collections.abc import Callable

from pop_trainer.agents.coverage import CoverageAgent
from pop_trainer.agents.noop_agent import NoOpAgent
from pop_trainer.agents.random_agent import RandomAgent
from pop_trainer.core import agent as core_agent

__all__ = ["AGENT_SELECTORS", "make_agent"]

# Selector name -> a factory taking an OPTIONAL seed. The coverage family's three presets, the
# random baseline, and the stationary noop floor. ``noop`` is stateless, so its lambda accepts
# and ignores the seed to keep the factory signature uniform.
AGENT_SELECTORS: dict[str, Callable[[int | None], core_agent.Agent]] = {
    "aggressive-coverage": lambda seed: CoverageAgent.aggressive(seed=seed),
    "wall-hugger": lambda seed: CoverageAgent.wall_hugger(seed=seed),
    "opponent-shadower": lambda seed: CoverageAgent.opponent_shadower(seed=seed),
    "random": lambda seed: RandomAgent(seed=seed),
    "noop": lambda seed: NoOpAgent(),  # noqa: ARG005 (stateless: ignores the uniform seed arg)
}


def make_agent(selector: str, *, seed: int | None = None) -> core_agent.Agent:
    """Build the agent named by ``selector`` (see :data:`AGENT_SELECTORS`).

    Raises ``ValueError`` on an unknown selector, listing the valid names.
    """
    factory = AGENT_SELECTORS.get(selector)
    if factory is None:
        valid = ", ".join(sorted(AGENT_SELECTORS))
        raise ValueError(f"unknown agent selector {selector!r}; choose one of: {valid}")
    return factory(seed)
