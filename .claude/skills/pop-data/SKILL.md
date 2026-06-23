---
name: pop-data
description: Contract + boundaries for the data/ component of src/pop_trainer. Load when building or reviewing data/.
---

# Component: `data/` — the datasets pipeline

**Responsibility:** collection + sharding of `(frame, state[, action, next])` pairs/tuples for pretraining, plus dataset readers/index. Drives the game **directly via `core.protocol`** (the `Connection` + frame channel) with deterministic policies; produces dataset artifacts on disk. It does NOT go through `env`'s gym wrapper — collection needs the wire driver, not the RL reward machinery.

**Contains:**
- the collection driver (uses `core.protocol.Connection`; parallel-safe; **spawn not fork**; buffer-aware reads).
- shard format read/write (npz) — the on-disk dataset artifact.
- dataset readers / index (map-aware splits, group keys) — describe and organize the shards.

**NOT here (per CTO):** no pre-resize / memmap **cache creation**. Training **streams** from the dataset at train time instead. The streaming loader (decode + batch shards on demand) is **deferred to `pretraining/`**, built when the training loop needs it — `data/` only produces and describes the dataset.

**Boundaries:** imports `core` only (the protocol `Connection`, the state schema, `EnvConfig`) — never Unity specifics. Imports nothing from `env / models / pretraining / rl` (it drives the wire directly; it does not depend on the gym wrapper). Communicates downstream ONLY via dataset artifacts on disk. The package is named `data/` — **NOT `datasets/`** (collides with the git-ignored data dir).

**Inspiration (do NOT copy):** `src/tank_twin/collect_pixels.py`. (The old `prepare_pixel_cache.py` cache builder is explicitly NOT carried over.)
