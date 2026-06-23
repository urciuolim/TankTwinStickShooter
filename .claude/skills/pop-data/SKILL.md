---
name: pop-data
description: Contract + boundaries for the data/ component of src/pop_trainer. Load when building or reviewing data/.
---

# Component: `data/` — the datasets pipeline

**Responsibility:** collection, sharding, caching/streaming of `(frame, state[, action, next])` pairs/tuples for pretraining. Takes the env at the gym interface; produces dataset artifacts on disk.

**Contains:**
- the collection driver (parallel-safe; **spawn not fork**; buffer-aware socket reads).
- shard format read/write (npz).
- the cache / streaming layer (memmap / PNG-LMDB; open the LMDB env inside each worker — spawn-safe).
- dataset readers (map-aware splits, group keys).

**Boundaries:** imports `core` (state/obs spec) and drives `env` via its gym interface only — never Unity specifics. Imports nothing from `models / pretraining / rl`. Communicates downstream ONLY via dataset artifacts on disk. The package is named `data/` — **NOT `datasets/`** (that collides with the git-ignored data dir).

**Inspiration (do NOT copy):** `src/tank_twin/collect_pixels.py`, the cache/transcode scripts under `scripts/`.
