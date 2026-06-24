# Architecture

The cross-component class-to-class interaction map for the built + GO'd slice of `pop_trainer`.

## Dependency direction

`core` is the **dependency-free root**; every component points inward toward it, never back out.
`models` imports nothing internal (a leaf alongside `core`). The full direction:

```
core  ←  { models, env, agents, data }  ←  demo
```

- [core](components/core.md) — stdlib + numpy only; imports nothing internal.
- [models](components/models.md) — torch only; imports nothing internal.
- [env](components/env.md) — imports `core` (+ gymnasium, numpy).
- [agents](components/agents.md) — imports `core` (+ numpy).
- [data](components/data.md) — imports `core`, `env`, `agents`.
- [demo](components/demo.md) — imports `core`, `env`, `agents`.

No import cycles: `data` and `demo` sit at the top, `core` at the bottom, `models` off to the
side. (`pretraining` / `rl` / `eval` / `population` / `deployment` / `imitation` are not built
yet and are omitted.)

## Class-to-class interaction map

```mermaid
graph TD
    subgraph core["core (contract layer — dep-free root)"]
        State["state.py<br/>52-float schema +<br/>split_state_for_opponent /<br/>flip_frame_perspective"]
        Protocol["protocol.py<br/>Connection, encode/decode,<br/>receive_frame, WallLayout,<br/>parse_walls_message"]
        Config["config.py<br/>RunConfig / EnvConfig /<br/>RewardConfig"]
        Maps["maps.py<br/>resolve_map_rotation"]
        AgentProto["agent.py<br/>Agent / StatefulAgent<br/>(Protocol)"]
        Launch["launch.py<br/>build_launch_cmd / connect<br/>(stdlib leaf)"]
    end

    subgraph models["models (torch; no internal deps)"]
        Encoder["Encoder = Trunk × Pooling<br/>build_encoder / export_onnx"]
    end

    subgraph agents["agents (model-free policies)"]
        AgentImpls["CoverageAgent (+ presets) /<br/>AimFireSchedule / RandomAgent /<br/>measure_coverage"]
    end

    subgraph env["env"]
        TankEnv["TankEnv<br/>(gymnasium.Env)"]
        Rewards["shaped_step_reward /<br/>time_penalty_per_step"]
    end

    subgraph data["data"]
        Runner["collect_runner.py<br/>env_factory / agent_pool_factory /<br/>round_robin_plan / build_specs / main (CLI)"]
        Collect["collect.py<br/>run_episode / collect_to_shards /<br/>resolve_map_tag / collect_parallel"]
        Shards["shards.py / schema.py"]
        Readers["readers.py<br/>split_groups / build_index"]
    end

    Demo["demo.py<br/>run_demo_episode / main"]

    Unity["Unity sim<br/>(GameController, DriverController,<br/>WallMessage, FrameCapture)"]

    %% env wiring
    TankEnv --> Protocol
    TankEnv -->|"reset(switch_arena)<br/>→ Connection.switch_arena"| Protocol
    TankEnv --> State
    TankEnv --> Config
    TankEnv --> AgentProto
    TankEnv --> Rewards
    TankEnv <-->|TCP-JSON + pixel frames + switch_arena| Unity

    %% agents wiring
    AgentImpls --> AgentProto
    AgentImpls --> State
    AgentImpls --> Protocol

    %% map hook: WallLayout reaches a map-aware agent via the optional set_map hook (driver-side)
    Collect -.->|set_map player1 + player2| AgentImpls
    Demo -.->|set_map player1 + player2| AgentImpls

    %% models wiring
    Encoder -.->|leaf, no internal import| core

    %% data wiring
    Runner --> Collect
    Runner --> Launch
    Runner -->|"resolve_map_rotation<br/>(rotation contract)"| Maps
    Runner --> TankEnv
    Runner --> AgentImpls
    Runner -.->|"round_robin_plan →<br/>EpisodePlan[]"| Collect
    Collect -->|"collect_to_shards →<br/>run_episode → reset(switch)"| TankEnv
    Collect --> AgentImpls
    Collect --> State
    Collect --> Shards
    Readers --> Shards

    %% demo wiring
    Demo --> TankEnv
    Demo --> AgentImpls
    Demo --> Protocol
    Demo --> Launch

    classDef root fill:#d4edda,stroke:#28a745;
    class core root;
```

## Reading the map

- **`TankEnv` is the hub.** It pulls every `core` module it needs (`protocol`, `state`,
  `config`, `agent`, plus its own `rewards`) and is the only thing that talks to Unity over the
  socket. It is a **pure symmetric transport** that owns neither player — `step(a1, a2)` takes both
  actions from the caller. Both `data.collect` and `demo` run their episodes through it.
- **The self-play seam** lives in `core.state` (`split_state_for_opponent` /
  `flip_frame_perspective`) and is owned **driver-side**: `data.collect` / `demo` flip player2's
  perspective to compute its action `a2`, then pass it to `env.step(a1, a2)`. Agents read `PLAYER_1`
  out of whatever (possibly flipped) view they're handed. `TankEnv` exposes `player2_frame()` /
  `player2_state()` helpers but does not call them during `step`.
- **The wall-message seam** flows `WallMessage` (Unity) → `protocol.parse_walls_message` →
  `WallLayout` → `TankEnv.current_map` → `info["map"]`, and from there into a map-aware agent via
  the OPTIONAL `set_map` hook — called **driver-side** on **both** players: `data.collect` /
  `demo` call `set_map` on player1 and player2 (all `getattr`-probed; the env notifies no agent). A
  [`CoverageAgent`](components/agents.md) rebuilds its coverage grid from the layout; `RandomAgent`
  does not implement the hook. See
  [core](components/core.md#the-wall-message--protocol-seam-walllayout--infomap).
- **`core.launch` is the shared live seam.** Both the `demo` and `data.collect_runner` launch the
  build + open the socket through `build_launch_cmd` / `connect`; it is stdlib-only so `core` stays
  the leaf. `collect_runner.env_factory` calls it once per worker (`port = base_port + worker_id`).
- **`data.collect_runner` is the collection CLI** (`python -m pop_trainer.data.collect_runner`):
  the concrete `env_factory` + `agent_pool_factory` (module-level, spawn-safe) + `build_specs` over
  `collect`'s spawn orchestration — multi-worker, single-map **or** a map × pairing rotation.
- **The rotation seam.** `--maps` resolves through `core.maps.resolve_map_rotation` (the shared
  rotation contract) to a set of arenas; `round_robin_plan` slices the (map × pairing) grid per
  worker (cell `(worker_id + i*n_workers) % G`) into `EpisodePlan[]`. Each episode rotates the
  long-lived build's arena via `collect.run_episode → env.reset(options={"switch_arena": ...}) →
  core.Connection.switch_arena` (the additive outbound seam), and `collect.resolve_map_tag` tags
  each sample's `map_id` from the arena Unity **echoed** (the F5 tag-from-echo), decoded through the
  `maps.json` sidecar / `map_index`. See
  [data](components/data.md#the-rotation-scheduler--map-tagging).
- **`models` is detached** from the live loop today — it's the shared vision backbone the
  future `pretraining` / `rl` will consume, and the deployable ONNX artifact.

---
[← back to index](README.md)
