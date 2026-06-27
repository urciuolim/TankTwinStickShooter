---
name: pop-data
description: Contract + boundaries for the data/ component of src/pop_trainer. Load when building or reviewing data/.
---

# Component: `data/` — the datasets pipeline

**Responsibility:** collection + sharding of `(frame, state[, action, next])` pairs/tuples for pretraining, plus dataset readers/index. Collection drives the game **through the `env` gym wrapper** (`TankEnv.reset` / `step`, with deterministic policies; capture the obs + `info["state"]` it yields) — so the data captured for pretraining is the EXACT same observation pipeline RL trains on, with no drift between pretraining inputs and RL observations. Produces dataset artifacts on disk.

**Contains:**
- the collection driver (drives `TankEnv` pairing a `player1` agent + a `player2` agent — any `Agent`; captures obs + `info["state"]` + BOTH players' actions; parallel-safe — **spawn not fork**; one env per worker).
- shard format read/write (npz) — the on-disk dataset artifact.
- dataset readers / index (map-aware splits, group keys) — describe and organize the shards.

**NOT here:** no pre-resize / memmap **cache creation**. Training **streams** from the dataset at train time; the streaming loader (decode + batch shards on demand) is **deferred to `pretraining/`**. `data/` only produces and describes the dataset.

**Progress reporting (collection only):** collection MAY show a live progress bar (`tqdm`), but it is purely an observability side-channel — it is ORTHOGONAL to the artifacts and MUST NOT alter a single byte of any shard, sidecar, or per-worker result. Contract:
- **ONE aggregate bar**, never one-per-worker. Total = `sum(len(spec.episode_plan))` across all specs (known upfront). Granularity is **PER-EPISODE** (episode lengths vary with kills, so total steps is unbounded; episode count is the bounded, accurate unit). Postfix MAY show a running sample count.
- **Multi-worker is spawn-safe:** the bar lives in the PARENT. Workers push a per-episode "done" message onto a `multiprocessing` queue (created from the spawn context, passed via the `Pool` initializer or a worker arg — NEVER pickled into `CollectionSpec`, which stays a plain-data, picklable spec carrying no live handles). The parent drains the queue and advances the one bar; this REQUIRES async submission (`apply_async` / `imap_unordered`), not blocking `Pool.map`. The single-spec in-process path drives the same bar.
- **No context-flood:** the bar overwrites in place (tqdm's default `\r`, one line) AND is **disabled on a non-TTY** (`disable=not sys.stderr.isatty()`) so a redirected / captured run emits NOTHING. NO per-step/episode `print` may compete with the bar; end-of-run summary lines print after the bar closes.

Tests cover the progress plumbing WITHOUT live workers: the per-episode counting, and the TTY-disable logic (bar disabled when `sys.stderr.isatty()` is False, enabled when True — monkeypatch `isatty`). The byte-identical-shards guarantee is verifiable by running a collection with and without the bar and diffing the artifacts.

**Boundaries:** imports `core`, `env` (the gym wrapper), and `agents` (the policies it drives collection with — the rule policies live in `agents/`, not here). Imports nothing from `models / pretraining / rl`. Dependency direction: `core ← {env, agents} ← data`. Communicates downstream ONLY via dataset artifacts on disk. The package is named `data/` — **NOT `datasets/`** (collides with the git-ignored data dir).

**Inspiration (do NOT copy):** the 2021 data-collection scripts (retired to git history).
