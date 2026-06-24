# Runbook — end to end

Clone → set up the Python env → build the Unity game → watch two agents play. Every command
below is verified against this repo. Commands assume the repo root
(`C:\src\TankTwinStickShooter`). The Python env is managed by **`uv`** — never the system Python.

## 1. Python environment

```bash
uv sync
```

Creates / updates the isolated venv from `pyproject.toml` + `uv.lock` (Python 3.12, pinned).
Verify:

```bash
uv run python --version              # -> Python 3.12.x
uv run python -c "import pop_trainer; print('ok')"
```

Run the tests (pure-logic + contracts; Unity-needing tests are opt-in `integration`):

```bash
uv run pytest -m "not integration and not e2e"
```

## 2. Build the Unity game

The trainer/demo needs a standalone Windows build of the simulator. The headless build entry
point is [`BuildScript.BuildWindows`](../Assets/Editor/BuildScript.cs), invoked via the Unity
6.5 editor in batchmode:

```bash
"C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe" \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "C:\src\TankTwinStickShooter" \
  -executeMethod BuildScript.BuildWindows
```

Output goes to `build/TankTwinStickShooter.exe` (git-ignored). `StreamingAssets`
(`config.json` + `Arenas/*.json` + the demo config) ships into the player automatically.
`BuildScript` exits non-zero on failure so the batchmode caller can detect it.

> For how the built game runs (it is a **Python-clocked simulator** with no standalone human
> mode), see [game architecture](game-architecture.md).

## 3. Run the demo (live build + two agents)

With the build in place, watch two visibly-different policies play one episode:

```bash
uv run python -m pop_trainer.demo
```

This launches `build/TankTwinStickShooter.exe` **windowed** with
`Assets/StreamingAssets/demo_config.json` (which enables `obs_pixels` at 640×360 — required,
since the env always reads a pixel frame each step), connects over TCP, runs one episode of
`player1 = aggressive-coverage` vs `player2 = opponent-shadower`, prints a trace, and tears down.

See the full flag surface:

```bash
uv run python -m pop_trainer.demo --help
```

Useful flags (defaults shown): `--exe build/TankTwinStickShooter.exe`,
`--config Assets/StreamingAssets/demo_config.json`, `--port 50000`, `--player1` / `--player2`
(any of: `aggressive-coverage`, `wall-hugger`, `opponent-shadower`, `random`), `--max-steps 600`,
`--seed 0`. The demo exits `2` if the build exe or config is missing.

## 4. Collect a dataset (programmatic)

There is **no collection CLI entry point yet** — data collection is driven through the
[`data.collect`](../src/pop_trainer/data/collect.py) API
([`run_worker`](../src/pop_trainer/data/collect.py) /
[`collect_parallel`](../src/pop_trainer/data/collect.py)) with caller-supplied env + agent
factories, writing `.npz` shards to disk. See the [data component page](components/data.md) for
the shape of the pipeline. (A thin CLI lands with the agent redesign.)

> **Episode budget.** The coverage family fully sweeps a map only by ~800 decisions on the live
> engine, so set the collection `max_steps` long enough to cover the map (see the
> [agents operational note](components/agents.md#operational-note--coverage-is-step-budget-dependent-on-the-real-engine)).

## 5. Validate coverage (no Unity required)

The [`measure_coverage`](../src/pop_trainer/agents/coverage_metrics.py) harness runs a
deterministic kinematic rollout of an agent on a `WallLayout` and reports coverage-fraction +
occupancy-entropy — the in-tree check that the coverage family beats `RandomAgent`. No live build
needed:

```bash
uv run python -c "
from pop_trainer.core.protocol import WallLayout, WallDims
from pop_trainer.agents import CoverageAgent, RandomAgent, measure_coverage
layout = WallLayout(map_id='open', tile_id=0, dims=WallDims(min_x=-6, max_x=5, min_y=-6, max_y=5), columns={})
cov = measure_coverage(CoverageAgent.aggressive(seed=0), layout)
rnd = measure_coverage(RandomAgent(seed=0), layout)
print(f'coverage  cov={cov.coverage_fraction:.2f} entropy={cov.occupancy_entropy:.2f}')
print(f'random    cov={rnd.coverage_fraction:.2f} entropy={rnd.occupancy_entropy:.2f}')
"
```

On this 12×12 open map the coverage agent reaches `cov≈0.85` vs the baseline `cov≈0.33`. (The
kinematic harness is optimistic vs the live engine — see the operational note above.)

---
[← back to index](README.md)
