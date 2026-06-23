---
name: pop-core
description: Contract + boundaries for the core/ component of src/pop_trainer. Load when building or reviewing core/.
---

# Component: `core/` — the dependency-free contract

**Responsibility:** the shared CONTRACT every other component depends on. Depends on NOTHING internal; stdlib + numpy only — **no torch, no gymnasium, no sb3**. It is the root of the dependency graph.

**Observation model:** the policy/encoder input is the **real rendered pixel frame** from Unity (received via `protocol`'s frame channel) — NOT a Python-drawn grid. There is **no `draw_state` / synthetic RGB obs and no `arenas.py`** (the 2021 approach; removed). The 52-float **state is a training OBJECTIVE** (the supervised decode target), not the observation.

**Contains:**
- `state.py` — the 52-float state schema (single source of truth; the supervised objective signal).
- `protocol.py` — the TCP-JSON wire client (framing/handshake) + the real pixel-frame channel (the actual obs input path).
- `config.py` — typed config dataclasses (run / env / reward).
- `maps.py` — map selection / rotation (which map to play).
- A small world-bounds CONSTANT may live here if a downstream objective needs world→grid coords — a constant, NOT an arena-file loader.

**Boundaries:** imports nothing from `models / env / data / pretraining / rl`. No torch. No cycles.

**Future (not built here):** the static map/wall layout will be sent by Unity as a one-time message on load/map-change and tracked as map-state — designed at `env/` integration with the Unity side, NOT via Python arena parsing.

**Inspiration (do NOT copy):** `src/tank_twin/{state,protocol,config,maps}.py`; the 2021 `PythonScripts/` originals. Build fresh — match the Unity wire contract exactly (52-float layout, JSON shape, big-endian frame header), but this is NEW code; it does not modify `src/tank_twin`'s frozen seam.
