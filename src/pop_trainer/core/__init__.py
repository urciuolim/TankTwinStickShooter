"""``pop_trainer.core`` — the dependency-free contract layer.

The shared CONTRACT every other ``pop_trainer`` component depends on; the root of the
dependency graph. It imports NOTHING internal (nothing from ``models`` / ``env`` / ``data``
/ ``pretraining`` / ``rl``, nothing from ``tank_twin``) and uses stdlib + numpy ONLY (no
torch, no gymnasium, no stable-baselines3). All JSON is STRICT.

Modules:

* :mod:`pop_trainer.core.state` — the 52-float game-state schema (single source of truth):
  layout constants, pure accessors, bullet iteration, and the perspective transforms
  (``flip_state`` / ``split_state_for_opponent``).
* :mod:`pop_trainer.core.protocol` — the strict TCP-JSON wire client over an injected
  transport: ``encode`` / ``decode``, frame-aware ``Connection``, and the additive
  length-prefixed REAL pixel-frame channel (the actual observation input path).
* :mod:`pop_trainer.core.config` — frozen, strict-JSON config dataclasses
  (``RunConfig`` / ``EnvConfig`` / ``RewardConfig``).
* :mod:`pop_trainer.core.maps` — the map-rotation resolution contract.
"""

from pop_trainer.core import config, maps, protocol, state

__all__ = ["state", "protocol", "config", "maps"]
