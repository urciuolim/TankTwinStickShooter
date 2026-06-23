---
name: pop-models
description: Contract + boundaries for the models/ component of src/pop_trainer. Load when building or reviewing models/.
---

# Component: `models/` — reusable nn.Modules

**Responsibility:** the torch model DEFINITIONS — the reusable artifacts pretraining produces and rl consumes. Definitions only; training loops live in `pretraining/` and `rl/`.

**Contains:**
- the encoder trunk (e.g. NatureCNN over pixel obs), with a configurable output (flat embedding vs spatial field).
- objective heads (supervised decode heads — disposable training scaffolding).
- policy / value networks for RL.
- a `from_pretrained`-style load path for the encoder artifact (LeRobot precedent) — the train→deploy/rl seam.

**Boundaries:** imports `core` only (e.g. the state schema for head output shapes). Imports nothing from `env / data / pretraining / rl`. No cycles.

**Inspiration (do NOT copy):** `src/tank_twin/features.py` (the CNN extractor), the pixel-pretrain heads in `src/tank_twin/_pretrain_pixels_train.py`. Build fresh.
