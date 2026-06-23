"""``pop_trainer.data`` — the datasets pipeline (collection + sharding + readers).

Collects time-aligned ``(frame, state[, action])`` samples for supervised pretraining and
organizes them into on-disk dataset artifacts. It drives the game DIRECTLY via
:mod:`pop_trainer.core.protocol` (the ``Connection`` + the length-prefixed pixel-frame channel)
with DETERMINISTIC policies; it does NOT go through ``env``'s gym wrapper (collection needs the
wire driver, not the RL reward machinery), and it communicates downstream ONLY via dataset
artifacts on disk.

Boundaries: imports ``pop_trainer.core`` ONLY (the protocol ``Connection``, the 52-float state
schema, ``EnvConfig``) plus numpy / stdlib. It imports NOTHING from
``env`` / ``models`` / ``pretraining`` / ``rl`` and nothing from ``tank_twin``. The package is
named ``data`` (NOT ``datasets`` — that collides with the git-ignored data dir).

Modules:

* :mod:`pop_trainer.data.schema` — the on-disk record schema (array names / dtypes / shapes).
* :mod:`pop_trainer.data.policies` — deterministic, pure collection policies (``state ->
  action_list``).
* :mod:`pop_trainer.data.shards` — atomic, round-trip-exact ``.npz`` shard read/write.
* :mod:`pop_trainer.data.readers` — map-aware (group-key) train/val/test splits + a dataset
  index over a shard directory.
* :mod:`pop_trainer.data.collect` — the collection driver: a pure, transport-injectable step
  loop plus spawn-based (never fork) parallel orchestration whose live socket path is isolated.
"""

from pop_trainer.data import collect, policies, readers, schema, shards

__all__ = ["schema", "policies", "shards", "readers", "collect"]
