---
name: pop-env
description: Contract + boundaries for the env/ component of src/pop_trainer. Load when building or reviewing env/.
---

# Component: `env/` — the simulator interface (the swap point)

**Responsibility:** the Gymnasium environment wrapping the Unity sim + the socket driver that speaks the protocol. This is the SWAP POINT — another sim would implement the same gym interface and everything downstream is unchanged.

**Contains:**
- `TankEnv` — Gymnasium API (reset/step → 5-tuple `(obs, reward, terminated, truncated, info)`).
- the Unity socket driver (uses `core.protocol`).
- reward shaping — the budget-based shaped reward (env/domain logic, **not** core) + episode-boundary logic.

**Boundaries:** imports `core` only. Imports nothing from `models / data / pretraining / rl`. The Unity game is the implementation *behind* this interface (bridged only by `core.protocol`); never reach into Unity specifics elsewhere.

**Inspiration (do NOT copy):** `src/tank_twin/{env,rewards}.py`, the env-drive part of `collect_pixels.py`; 2021 `PythonScripts/tank_env.py`.
