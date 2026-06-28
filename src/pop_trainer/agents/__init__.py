"""``pop_trainer.agents`` — the model-free decision-makers that fill ``player1`` / ``player2``.

Every agent here implements the :class:`pop_trainer.core.agent.Agent` Protocol — ``act(obs) ->
action`` — and stateful / seeded ones also implement ``reset(*, seed=None)``. Map-aware agents
additionally implement the OPTIONAL ``set_map(layout)`` hook (the static
:class:`pop_trainer.core.protocol.WallLayout` once per episode). The action is the env's 5-float
``[move_x, move_y, aim_x, aim_y, fire]`` (move/aim in ``[-1, 1]``, fire 0/1). No ``load`` /
``save`` / ``train`` hooks — those arrive later with neural-net agents; this package is
model-free only.

SELF-PLAY CONVENTION — "act as PLAYER_1 on own view". The env hands a player2 agent its OWN
first-person observation (the perspective-flipped 52-float state). After that flip, reading
``PLAYER_1`` out of the view returns the acting agent ITSELF and ``PLAYER_2`` returns the OTHER
player. So every state-reading agent here acts as ``PLAYER_1`` on whatever view it is handed —
which is why :meth:`CoverageAgent.opponent_shadower` aims ``PLAYER_1`` at ``PLAYER_2``.

THE COVERAGE FAMILY. The data-collection surface is ONE map-aware policy family,
:class:`CoverageAgent` (occupancy-biased coverage), parameterized by presets:

* ``CoverageAgent.aggressive`` — max coverage (tuned K=4 / eps=0.02) with an aim sweep + fire.
* ``CoverageAgent.wall_hugger`` — biases the least-visited tiebreak toward wall-adjacent cells
  for position-extreme / wall-hugging coverage.
* ``CoverageAgent.opponent_shadower`` — the same coverage movement, but the aim layer aims at the
  opponent and fires (subsuming the old aim-at-player2 rule).

:class:`RandomAgent` is the map-agnostic baseline; :class:`NoOpAgent` is the stationary
curriculum floor (the env's zero action every step). :mod:`pop_trainer.agents.coverage_metrics`
provides the kinematic-rollout coverage-measurement harness (coverage-fraction +
occupancy-entropy + an ESS guardrail) that validates the family against ``RandomAgent``.

THE SELECTOR REGISTRY. :mod:`pop_trainer.agents.registry` owns the canonical ``string -> agent``
map (:data:`AGENT_SELECTORS` + :func:`make_agent`) — the SINGLE source of truth re-exported here
that :mod:`pop_trainer.data.collect_runner` and :mod:`pop_trainer.play` import (rather than
duplicating). The roster is the coverage family's three presets, ``random``, and ``noop``.

:class:`HumanAgent` (:mod:`pop_trainer.agents.human_agent`) is a KEYBOARD-driven agent for live
human-vs-human play: two players share one keyboard via a single :class:`KeyboardListener` /
:class:`KeyboardState`, and two :class:`HumanAgent`s (P1 / P2 :class:`KeyMapping`) read that shared
state. ``pynput`` is an OPTIONAL, LAZY dependency (the ``human`` extra) imported only inside the
listener — so importing this package never imports ``pynput``.

Boundaries: imports :mod:`pop_trainer.core` only (the Agent Protocol, the state schema, the
``WallLayout`` type) plus numpy and stdlib (and ``pynput`` LAZILY, listener-only). No torch;
nothing from ``env`` / ``data`` / ``pretraining`` / ``rl`` / ``models``; nothing from ``tank_twin``.
"""

from __future__ import annotations

from pop_trainer.agents.base import (
    A_AIM_X,
    A_AIM_Y,
    A_FIRE,
    A_MOVE_X,
    A_MOVE_Y,
    ACTION_LEN,
    as_state_vector,
    validate_action,
)
from pop_trainer.agents.coverage import AimFireSchedule, CoverageAgent
from pop_trainer.agents.coverage_metrics import (
    CoverageReport,
    build_grid,
    measure_coverage,
)
from pop_trainer.agents.human_agent import (
    HumanAgent,
    KeyboardListener,
    KeyboardState,
    KeyMapping,
    player1_mapping,
    player2_mapping,
)
from pop_trainer.agents.noop_agent import NoOpAgent
from pop_trainer.agents.random_agent import RandomAgent
from pop_trainer.agents.registry import AGENT_SELECTORS, make_agent

__all__ = [
    # action contract
    "ACTION_LEN",
    "A_MOVE_X",
    "A_MOVE_Y",
    "A_AIM_X",
    "A_AIM_Y",
    "A_FIRE",
    "validate_action",
    "as_state_vector",
    # the coverage family + the baseline
    "CoverageAgent",
    "AimFireSchedule",
    "RandomAgent",
    "NoOpAgent",
    # the canonical agent-selector registry (string -> agent)
    "AGENT_SELECTORS",
    "make_agent",
    # coverage-measurement harness
    "measure_coverage",
    "build_grid",
    "CoverageReport",
    # keyboard-driven human play (pynput is a lazy optional 'human' extra)
    "HumanAgent",
    "KeyMapping",
    "KeyboardState",
    "KeyboardListener",
    "player1_mapping",
    "player2_mapping",
]
