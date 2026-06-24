"""``pop_trainer.agents`` — the model-free decision-makers that fill ``player1`` / ``player2``.

Every agent here implements the :class:`pop_trainer.core.agent.Agent` Protocol — ``act(obs) ->
action`` — and stateful / seeded ones also implement ``reset(*, seed=None)``. The action is the
env's 5-float ``[move_x, move_y, aim_x, aim_y, fire]`` (move/aim in ``[-1, 1]``, fire 0/1). No
``load`` / ``save`` / ``train`` hooks — those arrive later with neural-net agents; this package
is model-free only.

SELF-PLAY CONVENTION — "act as PLAYER_1 on own view". The env hands a player2 agent its OWN
first-person observation. For the state-reading agents here that observation is the
perspective-flipped 52-float state (the env adapter swaps the two 26-float player halves via the
``core.state`` perspective transform before calling ``player2.act``). After that flip, reading
``PLAYER_1`` out of the view returns the acting agent ITSELF and ``PLAYER_2`` returns the OTHER
player. Therefore EVERY state-reading agent here acts as ``PLAYER_1`` on
whatever view it is handed — this is the clean self-play convention, and it is why
:class:`AimAtPlayer2Agent` defaults to aiming ``PLAYER_1`` at ``PLAYER_2`` (the other player). The
aim agent keeps an optional player-bind for callers that hand it an un-flipped shared state and
want to drive the non-default slot.

COVERAGE RATIONALE. Beyond the rule policies, four agents are each designed to maximise coverage
of one 52-float state variable so a recorded trajectory exercises the full range of every field:

* :class:`ExplorerAgent` — POSITION: cycles movement headings to sweep the floor.
* :class:`AimSweepAgent` — AIM: rotates the aim vector through 360.
* :class:`SprayAgent` — BULLET pos/vec: fires continuously while moving + sweeping aim.
* :class:`PerimeterAgent` — position EXTREMES: wall-follows the four edges into the corners.

Because :mod:`pop_trainer.core` exposes no arena-bounds constant, the coverage strategy is
necessarily heading/phase-driven (open-loop sweeps), optionally sharpened by the agent's own
observed position/velocity — never by a fabricated bounds constant. See
:mod:`pop_trainer.agents.coverage` for the per-agent rationale.

Boundaries: imports :mod:`pop_trainer.core` only (the Agent Protocol + the state schema) plus
numpy and stdlib. No torch; nothing from ``env`` / ``data`` / ``pretraining`` / ``rl`` /
``models``; nothing from ``tank_twin``.
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
from pop_trainer.agents.coverage import (
    AimSweepAgent,
    ExplorerAgent,
    PerimeterAgent,
    SprayAgent,
)
from pop_trainer.agents.random_agent import RandomAgent
from pop_trainer.agents.rules import (
    AimAtPlayer2Agent,
    ConstantAgent,
    IdleAgent,
    ScriptedCycleAgent,
)

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
    # rule + random agents
    "RandomAgent",
    "IdleAgent",
    "ConstantAgent",
    "ScriptedCycleAgent",
    "AimAtPlayer2Agent",
    # coverage agents
    "ExplorerAgent",
    "AimSweepAgent",
    "SprayAgent",
    "PerimeterAgent",
]
