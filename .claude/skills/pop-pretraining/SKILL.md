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

**Inspiration (do NOT copy):** `reference/tank_twin_m1/pretrain_pixels.py` + `reference/tank_twin_m1/_pretrain_pixels_train.py`. NOTE: that is the legacy 53%-of-src, cycle-bearing, 2,112-line mass — build this CLEAN and SPLIT: small cohesive modules, **no import cycle**, no 2,000-line files.
