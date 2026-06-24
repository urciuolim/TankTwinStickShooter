---
name: pop-agents
description: Contract + boundaries for the agents/ component of src/pop_trainer. Load when building or reviewing agents/.
---

# Component: `agents/` — pluggable decision-makers (policies)

**Responsibility:** the agents that fill `player1` / `player2` — decision-makers producing an action from an observation. One interface covers rule-based policies, a random agent, and (later) a neural-net policy in inference mode. **Definitions only** — no training loops (those are `rl`/`pretraining`).

**Interface:** implements the **`core.Agent` Protocol** — `act(obs) -> action` (the one required method), plus `reset(seed=...)` for stateful/seeded agents. Keep it minimal: **NO `load`/`save`/`train` hooks yet** (those arrive with the NN agent in `rl/`). The Protocol lives in `core` so `env` can consume it without importing `agents` or torch; `agents/` IMPLEMENTS it.

**Contains (light — model-free agents only for now):**
- `RandomAgent` — uniform random actions (seeded).
- Rule-based, **coverage-oriented** policies (the existing rules move out of `data/policies.py`; plus new ones), each chosen to maximize coverage of a state variable:
  - `aim-at-player2` — purposeful tactical states (existing).
  - `explorer` — drive to cover the whole arena (waypoint / least-recently-visited) → POSITION coverage.
  - `aim-sweep` — rotate aim through 360° while moving → AIM coverage.
  - `spray` — fire frequently while moving + sweeping → BULLET position/direction coverage.
  - `perimeter` — hug walls/edges → position extremes + wall-adjacent states.
  - low-information policies kept only as `player2` options, NOT data drivers: `idle`, `constant`, `scripted-cycle`.
- Agents read the 52-float state (positions/aim/bullets, via `core.state`) to decide actions.

**Boundaries:** imports `core` ONLY (the `Agent` Protocol, the state schema, the action spec). **NO torch** (model-free; the `models` dependency arrives only when an NN agent is added). Nothing from `env / data / pretraining / rl`. No cycles.

**Inspiration (do NOT copy):** `src/pop_trainer/data/policies.py` (the existing rules → move here); 2021 `PythonScripts/` opponent policies.
