---
name: pop-rl
description: Contract + boundaries for the rl/ component of src/pop_trainer. Load when building or reviewing rl/.
---

# Component: `rl/` — PPO + self-play

**Responsibility:** the online RL training — SB3 PPO (CnnPolicy over pixels / the pretrained encoder) on the env, plus self-play wiring. Self-play is a WRAPPER over the trainer (ML-Agents `ghost/` precedent), **never woven into the PPO core**.

**Contains:**
- the PPO train loop + config; policy / feature-extractor wiring to `models`.
- callbacks / eval hooks.
- self-play opponent management + ELO (rating logic lives HERE, not in core).

**Boundaries:** imports `core`, `env`, `models`. Consumes the pretrained encoder as a loaded artifact via `models.from_pretrained` — does NOT import `pretraining` or `data`. No cycles.

**Inspiration (do NOT copy):** `reference/tank_twin_m1/{train,evaluate,callbacks,elo}.py`; the 2021 `PythonScripts/` PPO scripts.
