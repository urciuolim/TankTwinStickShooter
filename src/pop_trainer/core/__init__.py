"""``pop_trainer.core`` — the dependency-free contract layer.

The shared CONTRACT every other ``pop_trainer`` component depends on; the root of the
dependency graph. It imports NOTHING internal (nothing from ``models`` / ``env`` / ``data``
/ ``pretraining`` / ``rl``, nothing from ``tank_twin``) and uses stdlib + numpy ONLY (no
torch, no gymnasium, no stable-baselines3). All JSON is STRICT.

Modules:

* :mod:`pop_trainer.core.state` — the 52-float game-state schema (single source of truth):
  layout constants, pure accessors, bullet iteration, and the perspective transforms
  (``flip_frame_perspective`` / ``split_state_for_opponent``).
* :mod:`pop_trainer.core.protocol` — the strict TCP-JSON wire client over an injected
  transport: ``encode`` / ``decode``, frame-aware ``Connection``, and the additive
  length-prefixed REAL pixel-frame channel (the actual observation input path).
* :mod:`pop_trainer.core.config` — frozen, strict-JSON config dataclasses
  (``RunConfig`` / ``EnvConfig`` / ``RewardConfig``).
* :mod:`pop_trainer.core.maps` — the map-rotation resolution contract.
* :mod:`pop_trainer.core.agent` — the Agent Protocol (``act(obs) -> action``), the
  type-only decision-maker interface ``agents`` implements and ``env`` consumes (keeping
  ``env`` torch-free).
* :mod:`pop_trainer.core.launch` — stdlib-only live-build launch + socket-connect helpers
  (the windowed ``Popen`` arg-list + a bounded connect retry/backoff) shared by the play app and
  the collection runner.
* :mod:`pop_trainer.core.logging_setup` — stdlib-``logging``-only per-process structured-JSONL
  observability setup (per-process file routing + a strict-JSON line formatter) every layer
  routes to.
"""

from pop_trainer.core import agent, config, launch, logging_setup, maps, protocol, state
from pop_trainer.core.agent import Agent, StatefulAgent

__all__ = [
    "state",
    "protocol",
    "config",
    "maps",
    "agent",
    "launch",
    "logging_setup",
    "Agent",
    "StatefulAgent",
]
