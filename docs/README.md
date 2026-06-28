# pop_trainer — documentation hub

`pop_trainer` is the Python reinforcement-learning driver for the Tank Twin Stick Shooter
revival. Unity 6.5 is the 2D twin-stick **simulator**; `pop_trainer` is the **trainer** that
drives it over a strict TCP-JSON socket (plus a binary pixel-frame channel). Agents observe
the **real rendered pixel frame** Unity emits each step (CnnPolicy), while the 52-float wire
state rides alongside in `info` as the supervised-decode objective.

This site is onboarding-grade: skim the map, drill into one component (each page lists what it
**pulls from** and **pushes to**, linked so the dependency graph is clickable), then run things
end to end via the [runbook](runbook.md).

## What's here today

The built + GO'd slice of the target architecture. The dependency direction is
**`core ← {models, env, data, agents} ← demo`** — `core` is the dependency-free root and every
component points inward toward it. The online-RL component [rl](components/rl.md) has its first two
seams (the policy↔encoder seam and the self-play opponent seam) and is documented.
(`pretraining` / `eval` / `population` / `deployment` / `imitation` are namespace placeholders, not
yet built — not documented here.)

## Component map at a glance

```mermaid
graph TD
    core["core<br/>contract layer (dep-free root)"]
    models["models<br/>vision encoders (torch)"]
    env["env<br/>TankEnv gym wrapper"]
    agents["agents<br/>map-aware coverage policies"]
    data["data<br/>dataset pipeline"]
    play["play<br/>runnable entry point<br/>(human / rule-based / rl:&lt;ckpt&gt;)"]
    rl["rl<br/>online RL (SB3) — encoder + self-play seams"]
    unity["Unity sim<br/>(TCP-JSON + pixel frames)"]

    env --> core
    agents --> core
    models -.->|no internal deps| core
    data --> core
    data --> env
    data --> agents
    play --> core
    play --> env
    play --> agents
    play -.->|"rl:&lt;ckpt&gt; only — lazy sb3 PPO.load"| rl
    rl -.->|wraps Encoder| models
    rl -.->|SelfPlayWrapper wraps| env
    rl -.->|make_agent roster| agents
    rl -.->|split_state_for_opponent| core
    env <-->|socket| unity

    classDef root fill:#d4edda,stroke:#28a745;
    class core root;
```

Solid arrows are import-time dependencies (A → B means "A imports B"). `models` imports nothing
internal — it only shares `core`'s place as a leaf; [rl](components/rl.md)'s `EncoderExtractor`
now wraps its `Encoder` (the seam exists), while the future `pretraining` will consume it too. `rl`
spans four internal deps: beyond `models`, its self-play seam wraps `env` (`SelfPlayWrapper`) and
reuses `agents` (`make_agent` roster) + `core` (`split_state_for_opponent`); it imports nothing from
`data` or `pretraining`.

## Navigation index

| Component | Responsibility |
|-----------|----------------|
| [core](components/core.md) | The dependency-free contract: state schema, wire protocol, configs, maps, Agent Protocol. |
| [models](components/models.md) | Composable, ablation-ready vision encoders + Sentis-clean ONNX export (torch). |
| [env](components/env.md) | `TankEnv` — the Gymnasium wrapper over the Unity socket; the simulator swap point. |
| [agents](components/agents.md) | The model-free decision-makers: the map-aware `CoverageAgent` family (+ presets) + the `RandomAgent` baseline + the coverage-measurement harness. |
| [data](components/data.md) | The dataset pipeline: a multi-worker collection CLI (`collect_runner`) → `(frame, state, action)` samples → shards → map-aware splits. |
| [play](components/play.md) | `python -m pop_trainer.play` — launch the live build and play one episode with any pairing of human / rule-based / trained `rl:<ckpt>` players. |
| [rl](components/rl.md) | Online RL (SB3 PPO); ships two seams — the policy↔encoder seam (`EncoderExtractor` wrapping the shared `Encoder`) and the self-play opponent seam (`SelfPlayWrapper` + `OpponentProvider` over the scripted-agent roster). |

- [Architecture](architecture.md) — the cross-component class-to-class interaction map.
- [Runbook](runbook.md) — clone → `uv sync` → build the Unity game → play an episode / collect.

## How this gets built

- [Agent team](agent-team.md) — how the research / engineering / platform teams (and the Director, under the CTO) build, gate, and document this repo.

## Unity side (still relevant)

- [Game architecture](game-architecture.md) — how the Unity build actually runs (Python-clocked simulator, no standalone human mode).
- [Input controls](input-controls.md) — the New Input System wiring for human tanks.
