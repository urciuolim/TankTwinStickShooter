---
name: pop-data
description: Contract + boundaries for the data/ component of src/pop_trainer. Load when building or reviewing data/.
---

# Component: `data/` — the datasets pipeline

**Responsibility:** collection + sharding of `(frame, state[, action, next])` pairs/tuples for pretraining, plus dataset readers/index. Collection drives the game **through the `env` gym wrapper** (`TankEnv.reset` / `step`, with deterministic policies; capture the obs + `info["state"]` it yields) — so the data captured for pretraining is the EXACT same observation pipeline RL trains on, with no drift between pretraining inputs and RL observations. Produces dataset artifacts on disk.

**Contains:**
- the collection driver (drives `TankEnv` with deterministic policies; parallel-safe — **spawn not fork**; one env per worker).
- shard format read/write (npz) — the on-disk dataset artifact.
- dataset readers / index (map-aware splits, group keys) — describe and organize the shards.

**NOT here:** no pre-resize / memmap **cache creation**. Training **streams** from the dataset at train time; the streaming loader (decode + batch shards on demand) is **deferred to `pretraining/`**. `data/` only produces and describes the dataset.

**Boundaries:** imports `core` and `env` (the gym wrapper). Imports nothing from `models / pretraining / rl`. Dependency direction: `core ← env ← data`. Communicates downstream ONLY via dataset artifacts on disk. The package is named `data/` — **NOT `datasets/`** (collides with the git-ignored data dir).

**Inspiration (do NOT copy):** `src/tank_twin/collect_pixels.py`.
