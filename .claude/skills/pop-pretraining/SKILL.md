---
name: pop-pretraining
description: Contract + boundaries for the pretraining/ component of src/pop_trainer. Load when building or reviewing pretraining/.
---

# Component: `pretraining/` — encoder pretraining

**Responsibility:** train the `models/` encoder via disposable supervised heads (state / next-state prediction) and, later, JEPA. Produces the reusable encoder artifact RL trains on. The HEADS are disposable scaffolding; the ENCODER is the artifact.

**Contains:**
- the training loop + dataset→batch glue (reads `data/` artifacts).
- head wiring + losses (occupancy / keypoint / wall objectives; focal/pos-weighted as needed).
- the encoder-output configuration (flat embedding vs spatial field, channel pinning).

**Boundaries:** imports `core`, `data` (dataset readers), `models` (encoder + heads). Imports nothing from `env / rl`. Decoupled from `env` BY the dataset (offline training).

**Inspiration (do NOT copy):** the M1 `tank_twin` pixel-pretraining reference (`pretrain_pixels` + its train module), retired to git history. NOTE: that was a 53%-of-src, cycle-bearing, 2,112-line mass — this is built CLEAN and SPLIT: small cohesive modules, **no import cycle**, no 2,000-line files. Draw on it ONLY for field-grouping / per-head ideas; current-only comments (no "ported from", no legacy cross-refs).

## The single-frame decoder harness (issues #3 / #6, Phase 1) — BUILT

A supervised **inverse-renderer** that decodes the 52-float wire state from ONE pixel frame. Locked by CTO/Brian (PR #4). The ENCODER is the artifact; every head is disposable.

**Field groups** — read via `core/state.py` NAMED accessors / index constants, NEVER magic indices (`position`, `velocity`, `aim`, `bullet_field_indices`, `bullet_present`, `ABSENT_BULLET_SENTINEL`, `PLAYER_1`/`PLAYER_2`). Per group, the right loss + metric:
- tank **position** (pos_x,pos_y x2 players) → regression (MSE in normalized space); metric = position error in WORLD units.
- tank **velocity** (vec_x,vec_y x2) → regression; metric = velocity error.
- tank **aim** (aim_x,aim_y x2) → per-player COSINE/angular loss (NOT MSE — MSE collapses a unit target to 0); metric = angular error (deg).
- **bullet presence** (5 slots x2 players = 10 bits; a slot is absent ONLY when its `pos_x` is the `ABSENT_BULLET_SENTINEL` (-100), tested via `bullet_present` as `pos_x > -50` — NOT `pos_x >= 0`: the arena is origin-centered so left-half bullets have valid negative x) → BCE (pos-weighted for the ~3.3%-present imbalance); metric = accuracy + F1.
- **bullet position** (present slots only) → MASKED regression (absent slots zeroed + masked out of the loss); metric = position error over present slots.

**Two head families, BOTH reported:**
- **spatial heads** read `Encoder.features` (the `(B,C,h,w)` map). Locatable groups (player / bullet position) localize via a heatmap + soft-argmax head; aim / velocity (directions / derivatives, not locations) use a flatten-then-FC head; presence reads off the same score maps. These train the encoder.
- **embed-probe** reads `Encoder.embed.detach()` (the pooled `(B,D)` vector). LOAD-BEARING: it measures how much state survives pooling (GAP vs Flatten), so pooling shows a real accuracy delta. The `.detach()` is mandatory — the probe must NOT backprop into the encoder. Report its per-group metrics ALONGSIDE the spatial heads. Train the probe with its OWN loss/optimizer step (separate from the encoder+spatial loss).

**Loader** — shard-streaming `Dataset` over `data.readers` (`build_index` + `DatasetIndex.split` → map-aware split, group key = map_id, no leak) and `data.shards` (lazy per-row frame fetch). `uint8 → [0,1]` NCHW; **load-time resolution downsample** (360→180→90 as a loader param); per-group target extraction from the 52-float state + sensible per-field target normalization (fit on TRAIN only — no val/test leak). NOTE: decode-v1 shards live in `worker_*/` SUBDIRS, so index recursively (`**/shard_*.npz`) or per-worker-dir; do not assume flat layout.

**Sweep axes** (config/CLI; the grid runs on a 4090 later, NOT here): trunk × pooling {gap, flatten} × resolution {360,180,90}. The trunk keys are the repo-standard `cnn` / `resnet` / `gn-cnn` (model lineage: Nature-DQN / ResNet / DreamerV3-GroupNorm respectively). Build them as knobs; here only smoke-verify.

**Train CLI** (`python -m pop_trainer.pretraining.train`): **device-agnostic** `--device cuda|mps|cpu` (auto-detect default; must run on the 4090 unchanged — no hardcoded device), `--data` (default `datasets/decode-v1`), `--trunk {cnn,resnet,gn-cnn}`, `--pooling {gap,flatten}`, `--resolution {360,180,90}`, epochs/batch/lr defaults, `--seed` (deterministic where reasonable). Trains, evals on val + test, logs per-group metrics for spatial heads + embed-probe, saves a checkpoint + a STRICT-JSON results record.

**Compute profile** — a fn/CLI reporting parameter count + forward-pass latency for a config+device (the Phase-3 numbers).

**Build scope for the first cut:** harness CODE + tests + a SMOKE-TRAIN only. Smoke = a few steps on a tiny subset of decode-v1 (cpu/mps), proving the loop runs end-to-end and the loss decreases. Do NOT run the full sweep. Do NOT require decode-v2.

**Tests (pure + one smoke):** head output shapes; each per-group loss; each metric fn; the loader's split + target extraction + downsample; plus a fast smoke-train test (tiny tensors / tiny fixture) asserting the train step runs and loss decreases over a few steps. Strict JSON for any on-disk record.
