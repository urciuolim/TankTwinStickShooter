"""``pop_trainer.data`` — the datasets pipeline (collection + sharding + readers).

Collects time-aligned ``(frame, state[, action])`` samples for supervised pretraining and
organizes them into on-disk dataset artifacts. Collection routes through
:class:`pop_trainer.env.tank_env.TankEnv` — the SAME gymnasium observation pipeline RL trains
on — pairing a ``player1`` agent + a ``player2`` agent (both :mod:`pop_trainer.agents`
policies; ``player2`` is injected into the env), so the captured ``(frame, state)`` rows match
the RL observations exactly. It communicates downstream ONLY via dataset artifacts on disk.

Boundaries: imports ``pop_trainer.core`` (the 52-float state schema), ``pop_trainer.env``
(the ``TankEnv`` gym wrapper), and ``pop_trainer.agents`` (the policies that drive collection)
plus numpy / stdlib, following the ``core <- {env, agents} <- data`` direction. It imports
NOTHING from ``models`` / ``pretraining`` / ``rl`` and nothing from ``tank_twin``. The package
is named ``data`` (NOT ``datasets`` — that collides with the git-ignored data dir).

Modules:

* :mod:`pop_trainer.data.schema` — the on-disk record schema (array names / dtypes / shapes).
* :mod:`pop_trainer.data.shards` — atomic, round-trip-exact ``.npz`` shard read/write.
* :mod:`pop_trainer.data.readers` — map-aware (group-key) train/val/test splits + a dataset
  index over a shard directory.
* :mod:`pop_trainer.data.collect` — the collection driver: a pure, env-injectable step loop
  pairing a ``player1`` + ``player2`` agent, plus spawn-based (never fork) parallel
  orchestration whose live socket path is isolated.
* :mod:`pop_trainer.data.manifest` — pure assembly + validation of the run-root provenance +
  machine manifest (with free-form ``descriptions``) that makes a (non-byte-reproducible) dataset
  identifiable.
* :mod:`pop_trainer.data.describe` — stdlib-only CLI to add / list a manifest's free-form
  descriptions (delegates to :func:`manifest.add_description`).
* :mod:`pop_trainer.data.shuffle` — reproducible, split-respecting, out-of-core two-pass
  bucketed shuffle of a dataset into a new row-permuted dataset (+ data-card manifest).
"""

from pop_trainer.data import collect, describe, manifest, readers, schema, shards, shuffle

__all__ = ["schema", "shards", "readers", "collect", "manifest", "describe", "shuffle"]
