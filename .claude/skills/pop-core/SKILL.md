---
name: pop-core
description: Contract + boundaries for the core/ component of src/pop_trainer. Load when building or reviewing core/.
---

# Component: `core/` — the dependency-free contract

**Responsibility:** the shared CONTRACT every other component depends on. Depends on NOTHING internal; stdlib + numpy only — **no torch, no gymnasium, no sb3**. It is the root of the dependency graph.

**Contains:**
- `state.py` — the game-state schema (the 52-float layout: per-player pos/vel/aim + bullet slots + sentinels). Single source of truth for the wire state.
- `protocol.py` — the TCP-JSON protocol client matching the Unity wire format (framing/handshake). STRICT JSON (no trailing commas / leading-dot floats — Python's `json` is strict; frame-aware reads, not `recv(1024)`).
- `observation.py` — the observation spec (state → pixel grid, `draw_state`) both data-gen and training agree on.
- `config.py` — typed config dataclasses (run / env / reward).
- `arenas.py`, `maps.py` — static arena/map reference data.

**Boundaries:** imports nothing from `models / env / data / pretraining / rl`. No torch. No cycles.

**Inspiration (do NOT copy):** `src/tank_twin/{state,protocol,observation,config,arenas,maps}.py`; the 2021 `PythonScripts/` originals. Build fresh — match the Unity wire contract, not the old code's structure. This is NEW code; it does not modify `src/tank_twin`'s frozen seam.
