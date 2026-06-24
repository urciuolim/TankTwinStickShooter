---
name: pop-env
description: Contract + boundaries for the env/ component of src/pop_trainer. Load when building or reviewing env/.
---

# Component: `env/` — the simulator interface (the swap point)

**Responsibility:** the Gymnasium environment wrapping the Unity sim + the socket driver that speaks the protocol. This is the SWAP POINT — another sim would implement the same gym interface and everything downstream is unchanged.

**Contains:**
- `TankEnv` — Gymnasium API (reset/step → 5-tuple `(obs, reward, terminated, truncated, info)`).
- the Unity socket driver (uses `core.protocol`).
- **`player2` is an injected `Agent`** (any `core.Agent`; default = a trivial built-in random player2 implemented in `env`, so the env never imports `agents`). The env computes `player2`'s own view (the perspective flip via `core.state`) and calls `player2.act()`; BOTH players' actions are captured. `player2`-as-learner / true 2-agent obs+reward is a FUTURE mode — leave the seam open, do not build it.
- reward shaping — the budget-based shaped reward (env/domain logic, **not** core) + episode-boundary logic.
- **map-state tracking** — the handshake consumes the OPTIONAL one-time `{"type":"walls",...}` message (after the start ack, before the first state; absent when no arena is configured), parses it via `core.protocol.parse_walls_message`, and stores `self.current_map: WallLayout | None`, surfaced in reset/step `info["map"]`. Route inbound objects by their `"type"` tag so a walls message is never mis-decoded as a state. Unity is the source of truth — never re-parse arena JSON here. NOT part of `observation_space`.

**Boundaries:** imports `core` only. Imports nothing from `models / data / pretraining / rl`. The Unity game is the implementation *behind* this interface (bridged only by `core.protocol`); never reach into Unity specifics elsewhere.

**Inspiration (do NOT copy):** `src/tank_twin/{env,rewards}.py`, the env-drive part of `collect_pixels.py`; 2021 `PythonScripts/tank_env.py`.
