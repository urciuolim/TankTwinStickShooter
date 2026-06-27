# Tank Twin Stick Shooter

A 2021 Unity 2D twin-stick tank game, revived in 2026 as a **reinforcement-learning research environment**. Unity 6.5 is the simulator; a Python driver (`pop_trainer`) drives it over a TCP-JSON socket, with agents observing the **real rendered pixel frame**.

<p align="center">
  <img src="docs/assets/gameplay.gif" width="480" alt="A map-aware coverage agent (blue) and a random opponent (red) playing the custom1 arena"><br>
  <em>A map-aware coverage agent (blue) vs. a random opponent (red) — real captured gameplay frames.</em>
</p>

## What this is

The game runs as a **Python-clocked simulator**: Unity renders and steps the physics, while Python supplies both players' actions each frame and reads back a 640×360 pixel frame plus a 52-float game state. That makes it a clean RL gym — train a policy on pixels (CnnPolicy), with the state vector as a supervised-decode objective.

The training stack (`src/pop_trainer/`) is rebuilt component-by-component, each independently gated:

| component | what it does |
|---|---|
| `core` | the contract: the 52-float state, the wire protocol, configs, maps, the Agent interface |
| `env` | `TankEnv` — a Gymnasium wrapper over the socket (the simulator swap point) |
| `agents` | map-aware coverage policies (+ a random baseline) that drive data collection |
| `models` | composable, ablation-ready vision encoders (ONNX / Unity-Sentis-clean) |
| `data` | a multi-worker collection runner — parallel builds, **map + pairing rotation**, echo-tagged shards |
| `demo` | launch the live build and watch two policies play |

### Multi-map data collection

The collector rotates arenas *and* policy pairings between episodes — one long-lived build, a `switch_arena` message at each reset — tagging every shard with the map Unity actually loaded:

<p align="center">
  <img src="docs/assets/rotation.gif" width="480" alt="The collector rotating through three arenas via switch_arena"><br>
  <em>One collector worker rotating through three arenas via <code>switch_arena</code>.</em>
</p>

## Quickstart

```bash
uv sync                                          # Python 3.12 env (managed by uv)
# build the Unity game -> unity/build/TankTwinStickShooter.exe  (see docs/runbook.md)
uv run python -m pop_trainer.demo                # watch two policies play one episode
uv run python -m pop_trainer.data.collect_runner --out-dir runs/demo --maps   # collect a rotating dataset
```

Full setup, the component map, and the class-to-class architecture live in **[`docs/`](docs/README.md)**.

## How it's built

This revival is developed by a team of AI coding subagents — research, engineering, and a read-only platform gate that documents at GO — under a human CTO. See **[`docs/agent-team.md`](docs/agent-team.md)**.

## Status

**M0** — playable two-human game ✅ · **M1** — single-agent PPO on real pixels (in progress) · **M2** — population-based self-play (next). Target training cluster: GCP.
