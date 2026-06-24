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

## 4. Collect a dataset (CLI)

The **multi-worker** collection runner:
[`python -m pop_trainer.data.collect_runner`](../src/pop_trainer/data/collect_runner.py)
([`main`](../src/pop_trainer/data/collect_runner.py)). It builds one
[`CollectionSpec`](../src/pop_trainer/data/collect.py) per worker and drives them through the
[`data.collect`](../src/pop_trainer/data/collect.py) orchestration
([`collect_parallel`](../src/pop_trainer/data/collect.py), spawn-based), each worker launching its
**own** Unity build + socket and writing `.npz` shards to a `worker_<id>/` subdir of `--out-dir`.

A run is either a **single map** (no `--maps`) or a **map × pairing rotation** (round-robin via the
runtime `switch_arena` seam). Single-map:

```bash
uv run python -m pop_trainer.data.collect_runner --out-dir runs/collect-demo
```

Rotate over the shipped 10 maps, default pairing mix:

```bash
uv run python -m pop_trainer.data.collect_runner --out-dir runs/collect-rot --maps
```

See the full flag surface:

```bash
uv run python -m pop_trainer.data.collect_runner --help
```

Flags (defaults shown), grounded in
[`_parse_args`](../src/pop_trainer/data/collect_runner.py):

| Flag | Default | Meaning |
|------|---------|---------|
| `--maps` / `--map-rotation` | *(absent)* | rotation set. **No value** → the shipped 10 `exp-configs/maps/*.json`; **a directory** → its `*.json` (sorted); **a list of paths** → that order. **Absent** → no rotation (single `--map`). |
| `--pairing P1:P2` | `DEFAULT_PAIRINGS` mix | a `(player1:player2)` selector pairing; **repeatable**. Omitted → the default coverage-family mix (coverage vs each other + vs random). |
| `--map` | `custom1` | single-map (no-rotation) config; the **boot** + only arena when `--maps` is absent. |
| `--episodes` | `1` | episodes **per worker** |
| `--max-steps` | `800` | per-episode step cap |
| `--out-dir` | *(required)* | root output dir; worker `w` writes shards to `worker_<w>/` under it |
| `--workers` | `2` | parallel workers, one build per worker (**clamped to `[1, MAX_WORKERS=8]`**) |
| `--exe` | `build/TankTwinStickShooter.exe` | path to the Unity build |
| `--base-port` | `50000` | base TCP port; worker `w` connects on `base_port + w` |
| `--seed` | `0` | base seed; worker `w` uses `base + w*10000` |

Selectors for `--pairing` (and `--map`'s default boot): `aggressive-coverage`, `wall-hugger`,
`opponent-shadower`, `random`.

### The `maps.json` sidecar (reversible map ids)

On-disk `map_ids` stay `int32`; `main` writes a **`maps.json`** sidecar (constant
`MAPS_SIDECAR_NAME`) so each int is reversible to the arena it names. It is written **per-worker-dir
AND at the run root**, with shape:

```json
{"schema_version": 1, "maps": ["Arenas/center_block.json", "..."]}
```

`maps[i]` is the arena for on-disk int `i`. The int is the **echo-tagged** id: it comes from the
arena path Unity actually loaded (echoed as `WallLayout.map_id`), decoded through the
`arena-path → int` index built from this same list (the F5 tag-from-echo; see
[data](components/data.md#the-rotation-scheduler--map-tagging)).

### The obs_pixels gotcha

Collection **must** receive pixel frames (the env reads a length-prefixed frame after every state;
a build without `obs_pixels` would leave the env blocking on bytes that never arrive). So the build
**boots** on the obs_pixels-enabled `--map` config (`custom1` → `Assets/StreamingAssets/demo_config.json`,
the **same** config the demo launches with — 640×360 pixels on, arena `Arenas/custom1.json`) and
rotates the arena via `switch_arena` at runtime. The shipped `exp-configs/maps` rotation configs
supply **switch targets only** (their `arena_path`) and do **not** enable `obs_pixels`, so they are
never used as the boot config. Either way the captured `(frame, state)` rows are byte-for-byte the
demo's / RL's observation pipeline. `main` exits `2` if the build exe or the resolved boot config is
missing.

> **Episode budget.** The coverage family fully sweeps a map only by ~800 decisions on the live
> engine — which is why `--max-steps` defaults to `800` — so keep it long enough to cover the map
> (see the
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
