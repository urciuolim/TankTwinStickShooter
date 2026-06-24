---
name: pop-agents
description: Contract + boundaries for the agents/ component of src/pop_trainer. Load when building or reviewing agents/.
---

# Component: `agents/` — map-aware coverage policies (one family) + the random baseline

**Responsibility:** the agents that fill `player1` / `player2` — decision-makers producing an
action from an observation. After the 2026 coverage redesign the data-collection surface is
ONE map-aware policy family (occupancy-biased coverage), parameterized by presets, plus the
map-agnostic `RandomAgent` baseline. **Definitions only** — no training loops (those are
`rl`/`pretraining`).

**Interface:** implements the **`core.Agent` Protocol** — `act(obs) -> action` (the one
required method), plus `reset(*, seed=...)` for stateful/seeded agents, plus an OPTIONAL
`set_map(layout)` map hook (see below). The Protocol lives in `core` so `env` can consume it
without importing `agents` or torch; `agents/` IMPLEMENTS it. Keep the Protocol minimal +
torch-free: NO `load`/`save`/`train` hooks (those arrive with the NN agent in `rl/`).

**The map hook (the ONE interface change):** the static wall layout is sent once per episode
(on load / map-change) and arrives as `core.protocol.WallLayout` on the env's `info["map"]`.
`core.Agent` gains an OPTIONAL `set_map(layout)` method. Map-aware agents implement it and
rebuild their grid; map-agnostic agents (`RandomAgent`) DO NOT implement it and are simply
never called with it (or no-op). `env` calls `player2.set_map(layout)` at reset IF the agent
exposes it (`getattr`/`hasattr` — never required); the demo / collection do the same for
`player1`. The hook is additive and torch-free; the env seam is otherwise unchanged.

**Contains:**
- `RandomAgent` — uniform random actions (seeded). The map-agnostic baseline; KEPT.
- `HumanAgent` (`agents/human_agent.py`) — a KEYBOARD-driven `core.Agent` for live human play
  (human-vs-human, two players on ONE keyboard). Contract:
  - Implements `core.Agent` (`act(obs) -> action`); **`act` IGNORES `obs`** — the action comes
    from the keyboard, not the observation. Returns the env's 5-float
    `[move_x, move_y, aim_x, aim_y, fire]` (move/aim in `[-1, 1]`, fire 0/1).
  - Constructed with a **`KeyMapping`** (this player's move/aim/fire keys) and a SHARED
    pressed-key source (a keyboard-state object both players read). `act` reads the CURRENT
    pressed-key set and maps THIS player's keys: `move_x = right−left`, `move_y = up−down`,
    same for aim; `fire = 1.0` iff the fire key is down. Diagonals may leave `(1,1)`
    un-normalized — the wire/`validate_action` coerces to `[-1, 1]`.
  - **Two players, ONE listener:** a SINGLE shared keyboard listener tracks the pressed-key
    set; TWO `HumanAgent`s (P1 keymap, P2 keymap) read the SAME state object. The demo builds
    one listener and both agents share it (start on play, stop on teardown).
  - **`pynput` is an OPTIONAL, LAZY dependency** (pyproject `[project.optional-dependencies]`
    extra `human`; pin `pynput>=1.8,<2`). Imported INSIDE the listener/HumanAgent constructor,
    NEVER at module top — so `import pop_trainer.agents` and `agents/human_agent` stay
    importable core-only WITHOUT pynput. Constructing the live listener (or a HumanAgent that
    needs it) without pynput raises a CLEAR, actionable error naming the `[human]` extra.
  - **Numpad-by-vk (load-bearing):** the P2 numpad keys must be detected by VIRTUAL-KEY CODE
    (`pynput` `KeyCode.vk`), NOT by character — so numpad-8/4/5/6 work regardless of Num Lock
    (with Num Lock OFF a numpad key arrives as an arrow/`Key`, not a digit char). Map the numpad
    set by `.vk`. This is the seam that MUST be fake-tested (inject a pressed-key set, no
    hardware). Windows VKs: NUMPAD4=100, NUMPAD5=101, NUMPAD6=102, NUMPAD8=104; numpad-Enter
    shares VK_RETURN (13) with main Enter on Windows and is NOT separable via pynput's public
    API — treat numpad-Enter == Enter (`Key.enter`) for P2 fire, and make the fire-key match
    configurable so the keymap, not hard-coded logic, decides it.
  - **Boundary:** imports `core` (Agent + action contract) + `pynput` (LAZY) ONLY. **NO torch.**
    The pressed-key→action MAPPING is a PURE function of (keymap, pressed-key set) and is unit
    tested without any listener/hardware. `agents/__init__` must stay importable without pynput;
    do NOT import `human_agent` eagerly in `agents/__init__` if that would force a pynput import
    (the lazy guard inside the module is the safeguard either way).
- **One coverage family** — occupancy-biased coverage (the design is load-bearing, do NOT
  re-derive; it comes from the 2026 research report):
  - Maintain a **visit-count grid** over the FREE cells.
  - Each re-pick, choose the **NEAREST least-recently-visited** free cell as the waypoint
    (greedy short hops via a BFS distance field FROM the current cell — preferring FARTHEST
    scored WORSE than baseline; do NOT do it).
  - Navigate toward the waypoint via a **plain BFS distance field** (NO A* / heap /
    priority-queue — the grid is ~128 cells), descending the gradient to the lowest-distance
    free neighbour.
  - **Reactive wall-repulsion**: a local 4-neighbour nudge under the moving waypoint (push
    away from adjacent occupied cells), NOT a raw potential field (those trap in local minima).
  - **ε / softmax noise** for decorrelation.
  - Core target ~80–150 LOC, pure stdlib + numpy. BFS is re-run twice per waypoint re-pick
    (from-current to pick the waypoint, from-waypoint for the descent field) — fine at ~128
    cells but it runs INSIDE `act()`, so keep it cheap (re-pick only every K steps / on reach).
  - **Presets** (parameterizations of the one family, for PBT/population diversity):
    `aggressive-coverage`, `wall-hugger`, `opponent-shadower`. `opponent-shadower` reads the
    opponent position from the 52-float state (`core.state`) to shadow / decorrelate — it
    SUBSUMES the old `aim-at-player2`.
- **Orthogonal aim/fire schedule layer** — the movement family is POSITION-only; layer a
  cheap aim-sweep + fire schedule on top so the bullet pos/dir + aim channels also get covered
  (the family leaves aim/fire free). Bullets do NOT bounce (die on wall/tank contact) — ground
  the fire cadence in that (don't fire so fast that bullets immediately self-block).

**Free-cell derivation (load-bearing; the sim-vs-real seam):** the production family gets its
grid from `WallLayout` ALONE — `free = {all cells in the WallLayout.dims bounding box} −
WallLayout.occupied`. This is NOT the spike's `Floor`-block free set (the spike read the arena
JSON's `Floor` directly; production only has `Walls` + `dims` on the wire). The two differ at
the border ring (the `Walls` dims are wider than the playable `Floor`). The family must be
robust to that: snap the tank's current cell to the nearest free cell when it is off-grid, and
restrict the visit grid to the BFS-reachable component from the spawn so unreachable border
cells never poison the least-visited pick. When `set_map` is never called (no arena), the
family must degrade to a safe map-blind fallback (e.g. random heading) rather than crash.

**Metric suite (productionized from the spike):** coverage-fraction + normalized
occupancy-entropy are the DESIGN objectives. **ESS / autocorrelation is a GUARDRAIL ONLY** —
it rewards jitter not coverage; use it only to detect a degenerate still agent, NEVER as a
coverage reward. Provide a way to MEASURE coverage on a map (a kinematic-rollout harness that
takes a `WallLayout`-derived grid + an agent and returns coverage-fraction + occupancy-entropy)
so the family can be validated against `RandomAgent`.

**Boundaries:** imports `core` ONLY (the `Agent` Protocol, `core.state`, `core.protocol`'s
`WallLayout`) plus numpy + stdlib. **NO torch** (model-free). Nothing from
`env / data / pretraining / rl / models`; nothing from `tank_twin`. No cycles.

**DELETED (no back-compat):** `agents/coverage.py` (explorer / aim-sweep / spray / perimeter)
and `agents/rules.py` (idle / constant / scripted-cycle / aim-at-player2). The old per-variable
open-loop sweeps and the tactical rules are SUBSUMED by the one family + its aim/fire layer +
the `opponent-shadower` preset. `RandomAgent` is the only pre-redesign agent kept.

**Inspiration (do NOT copy):** `.scratch/coverage_sim.py` (the `OccupancyBiased` spike, K=4,
eps=0.02, prefer-near — the production target) + `.scratch/tune.py` (the preset sweep); 2021
`PythonScripts/` opponent policies. The behaviour we trust is the ALGORITHM, not the spike code.
