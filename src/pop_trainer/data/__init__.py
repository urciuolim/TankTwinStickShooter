"""``pop_trainer.data`` — the datasets pipeline (collection + sharding + readers).

Collects time-aligned ``(frame, state[, action])`` samples for supervised pretraining and
organizes them into on-disk dataset artifacts. Collection routes through
:class:`pop_trainer.env.tank_env.TankEnv` — the SAME gymnasium observation pipeline RL trains
on — driven by DETERMINISTIC policies, so the captured ``(frame, state)`` rows match the RL
observations exactly. It communicates downstream ONLY via dataset artifacts on disk.

Boundaries: imports ``pop_trainer.core`` (the 52-float state schema) and ``pop_trainer.env``
(the ``TankEnv`` gym wrapper) plus numpy / stdlib, following the ``core <- env <- data``
direction. It imports NOTHING from ``models`` / ``pretraining`` / ``rl`` and nothing from
``tank_twin``. The package is named ``data`` (NOT ``datasets`` — that collides with the
git-ignored data dir).

Modules:

* :mod:`pop_trainer.data.schema` — the on-disk record schema (array names / dtypes / shapes).
* :mod:`pop_trainer.data.policies` — deterministic, pure collection policies (``state ->
  action_list``).
* :mod:`pop_trainer.data.shards` — atomic, round-trip-exact ``.npz`` shard read/write.
* :mod:`pop_trainer.data.readers` — map-aware (group-key) train/val/test splits + a dataset
  index over a shard directory.
* :mod:`pop_trainer.data.collect` — the collection driver: a pure, env-injectable step loop
  plus spawn-based (never fork) parallel orchestration whose live socket path is isolated.
"""

from pop_trainer.data import collect, policies, readers, schema, shards

__all__ = ["schema", "policies", "shards", "readers", "collect"]
