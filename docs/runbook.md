# Runbook — end to end

Clone → set up the Python env → build the Unity game → watch two agents play. Every command
below is verified against this repo. Commands run from the repo root; Windows and macOS hosts are
both supported (the build step and the launcher are OS-aware — see §2 / §3). Replace the example
repo-root paths (`C:\src\TankTwinStickShooter` / `/Users/<you>/.../TankTwinStickShooter`) with
yours. The Python env is managed by **`uv`** — never the system Python.

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

Run the tests (pure-logic + contracts; Unity-needing tests are opt-in `integration` / `e2e`):

```bash
uv run pytest -m "not integration and not e2e"
```

The **Tier-1 live e2e** (`test_e2e_collection.py`) drives a small real collection against a live
Unity build. It is opt-in (the `e2e` marker, excluded above) and **auto-skips** when no build is
present at `collect_runner.DEFAULT_EXE`, so a buildless run stays green. Run it with a build present:

```bash
uv run pytest -m e2e
```

## 2. Build the Unity game

The trainer/demo needs a standalone build of the simulator for the host OS. The headless build
entry points live in [`BuildScript`](../Assets/Editor/BuildScript.cs): one method per platform,
each invoked via the Unity 6.5 editor in batchmode. Both build the enabled scenes, exit non-zero
on failure (so the batchmode caller can detect it), and ship `StreamingAssets` (`config.json` +
`Arenas/*.json` + the demo / human configs) into the player automatically.

### Windows

[`BuildScript.BuildWindows`](../Assets/Editor/BuildScript.cs) → `build/TankTwinStickShooter.exe`
(git-ignored):

```bash
"C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe" \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "C:\src\TankTwinStickShooter" \
  -executeMethod BuildScript.BuildWindows
```

### macOS

[`BuildScript.BuildOSX`](../Assets/Editor/BuildScript.cs) targets `StandaloneOSX` and writes the
`build/TankTwinStickShooter.app` bundle (git-ignored). The Unity 6.5 editor binary on macOS is
inside the editor `.app`:

```bash
/Applications/Unity/Hub/Editor/6000.5.0f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "/Users/<you>/path/to/TankTwinStickShooter" \
  -executeMethod BuildScript.BuildOSX
```

The launchable binary lives *inside* the bundle at `Contents/MacOS/`; you never type its name —
the OS-aware default resolver
([`core.launch.default_build_path`](../src/pop_trainer/core/launch.py)) globs the bundle and
hands the inner binary to the launcher (see §3).

> For how the built game runs (it is a **Python-clocked simulator** with no standalone human
> mode), see [game architecture](game-architecture.md).

## 3. Run the demo (live build + two agents)

With the build in place, watch two visibly-different policies play one episode:

```bash
uv run python -m pop_trainer.demo
```

This launches the build **windowed** with `Assets/StreamingAssets/demo_config.json` (which enables
`obs_pixels` at 640×360 — required, since the env always reads a pixel frame each step), connects
over TCP, runs one episode of `player1 = aggressive-coverage` vs `player2 = opponent-shadower`,
prints a trace, and tears down.

**No `--exe` needed on any OS.** The default executable is resolved for the current platform by
[`core.launch.default_build_path`](../src/pop_trainer/core/launch.py)
([`demo.DEFAULT_EXE`](../src/pop_trainer/demo.py)): on **Windows** it is
`build/TankTwinStickShooter.exe`; on **macOS** it is the binary *inside* the
`build/TankTwinStickShooter.app` bundle (`Contents/MacOS/`, discovered by globbing the bundle —
you don't type the inner name); on **Linux** the bare `build/TankTwinStickShooter` binary. So
`python -m pop_trainer.demo` works as-is once the platform build from §2 is present — pass `--exe`
only to point at a build elsewhere.

See the full flag surface:

```bash
uv run python -m pop_trainer.demo --help
```

Useful flags (defaults shown): `--exe` (the OS-resolved build path above),
`--config Assets/StreamingAssets/demo_config.json`, `--port 50000`, `--player1` / `--player2`
(any of: `aggressive-coverage`, `wall-hugger`, `opponent-shadower`, `random`, `human`),
`--max-steps 600`, `--seed 0`. The demo exits `2` if the build exe or config is missing.

### Human-vs-human play (two players, one keyboard)

Human play needs the optional `human` extra (installs `pynput`, the global keyboard hook):

```bash
uv sync --extra human
```

Then make either or both players keyboard-driven. Two humans share ONE keyboard:

```bash
uv run python -m pop_trainer.demo --player1 human --player2 human
```

One human + one bot is also valid, e.g. `--player1 human --player2 opponent-shadower`.

Controls:

- **P1** — move WASD, aim TFGH, fire LEFT SHIFT.
- **P2** — move IJKL, aim numpad 8/4/5/6, fire ENTER. (Numpad aim works with Num Lock ON or OFF.)

When a player is `human` the demo auto-selects `Assets/StreamingAssets/human_config.json` (real-time
`timeScale: 1`, a more forgiving cadence) instead of `demo_config.json` — unless you pass `--config`.

> **LOCAL-DEV ONLY.** `pynput` installs a GLOBAL keyboard hook that needs a real desktop session.
> Human play is NOT part of the headless GCP / cluster path — that path is bot-vs-bot collection /
> training only. **Do not add `--extra human` to the cluster / headless install.**

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
| `--shard-size` | *(auto)* | in-RAM **BUFFER** bound (samples per worker before a flush), **NOT a file-size knob**. Omitted → auto: derived from the byte budget so the per-worker buffer stays bounded regardless of frame resolution (the OOM-safe default). See [Memory safety](#memory-safety-the-pre-flight-guard). |
| `--allow-oversized` | *(off)* | skip the pre-flight memory **abort** (the estimate is still printed). Use only when the box has RAM the guard's conservative estimate does not account for. |

Selectors for `--pairing` (and `--map`'s default boot): `aggressive-coverage`, `wall-hugger`,
`opponent-shadower`, `random`.

### Memory safety (the pre-flight guard)

`--shard-size` is the **in-RAM BUFFER bound** (samples a worker buffers before a flush), **not a
file-size knob**. Each buffered sample holds an UNCOMPRESSED `(H, W, 3)` uint8 frame, so the
buffer — not the tiny on-disk shard — is the OOM surface (flat frames compress ~140×, so disk size
badly under-reports RAM cost). A smaller bound just yields **more, smaller** shards with
**byte-identical** on-disk contents (`collect.py:471-477`).

The peak across all workers is (`estimate_peak_buffer_bytes`, `collect.py:125-133`):

```
peak buffer ~= frame_bytes × shard_size × workers × spike   (spike = SPIKE_FACTOR = 2)
```

`SPIKE_FACTOR=2` folds in the `savez_compressed` flush transient (`collect.py:99-102`). Default
`--shard-size` is **auto**: a per-worker BYTE budget (`SHARD_BYTES_BUDGET ≈ 384 MB`,
`collect.py:97`) is divided by the frame size (`default_shard_size`, `collect.py:110-122`). For the
360×640×3 = 691200-byte (~0.69 MB) pixel frame this yields **~582 frames/shard** — far under the old
count-based default of `10000` (a ~6.9 GB/worker buffer that OOMed a multi-worker box,
`collect.py:93`).

`main` runs a **pre-flight guard** before launching any build (`collect_runner.py:718-739`): it
reads `psutil.virtual_memory().available`, computes the estimate, and **always prints the estimate
line at startup**. When the estimate exceeds `MARGIN` (`0.6` → 60%) of available RAM it **aborts
with exit code `2` BEFORE any build launches** (`check_memory_budget`, `collect.py:136-170`) —
unless `--allow-oversized` (which still prints the estimate, then proceeds). The margin leaves
headroom for the ~1 GB/worker live Unity instances + the OS, which the estimate does not model
(`collect.py:104-107`).

**Worker × RAM guideline.** More `--workers` × bigger `--shard-size` raises the peak **linearly**
(both are factors in the formula). If the guard aborts, reduce `--workers` or `--shard-size`
(or pass `--allow-oversized` only if you know the box has the headroom).

### Live progress

On a terminal (TTY) the runner shows a **single aggregate progress bar** (one line, tqdm) that
ticks **per completed episode** across ALL workers — its total is the run's total episode count, so
it carries an ETA / remaining-time and a running `samples=…` count
([`collect_parallel`](../src/pop_trainer/data/collect.py), `collect.py:574-612`). It **auto-silences**
when `stderr` is **not** a TTY (`disable=not sys.stderr.isatty()`, `collect.py:559`): a redirected /
captured / piped run emits **zero** progress bytes, so logs stay clean. The bar is a pure
observability side-channel — it does not affect a single shard byte.

### End-of-run summary

After collection finishes (the bar closes), `main` prints a **concise summary** — counts only, no
filename dumps (`collect_runner.py:856-869`). Three parts:

1. an **aggregate** line —
   `done: <N> shards, <N,NNN> samples, <K> workers -> <out-dir>`;
2. one **terse per-worker** line — `worker <id>: <n> shards, <n> samples` (no filenames);
3. a **per-map sample-count table** aligned to the `maps.json` sidecar — one row
   `map <id>  <arena>  <count>` per arena, **including arenas that got zero samples**.

The sample + per-map counts come from a **pure** counter
([`summarize_collection`](../src/pop_trainer/data/collect_runner.py), `collect_runner.py:613-648`)
over each worker's per-sample `map_ids` (read back by globbing each worker dir's `shard_*.npz` in
sorted order and lazily reading ONLY each shard's `map_ids` member — the big `frames` array is never
decompressed, `collect_runner.py:651-666`); the worker result dict supplies only the shard counts.

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

## 5. Run RL training (CLI)

Train a PPO policy against the scripted self-play roster over the live Unity **pixel** env. The
integrator is [`python -m pop_trainer.rl.train`](../src/pop_trainer/rl/train.py)
([`main`](../src/pop_trainer/rl/train.py) → [`train_local`](../src/pop_trainer/rl/train.py)). It
composes the RL seams (encoder extractor, self-play wrapper, eval callback) into one SB3 PPO run with
checkpointing and a resumable sidecar — see the [rl component page](components/rl.md#the-train_local-integrator-seam).

Like the demo (§3) and collection (§4), this **launches a live windowed Unity build** (needs the §2
build present) and reads a pixel frame each step — `train_config.json` enables `obs_pixels` at
640×360, required since the env always reads a pixel frame.

The two required args plus a run dir. This example **omits `--opponents`**, so it trains against the
full 5-selector `DEFAULT_ROSTER` (`noop`, `random`, `aggressive-coverage`, `wall-hugger`,
`opponent-shadower`):

```bash
uv run python -m pop_trainer.rl.train --total-timesteps 10000 --run-dir runs/train-smoke
```

Train against a **single opponent** (NoOp-only roster) by passing `--opponents`:

```bash
uv run python -m pop_trainer.rl.train --total-timesteps 10000 --run-dir runs/train-noop --opponents noop
```

`--opponents` takes a comma-separated list, so `--opponents noop,random` trains against two. An
unknown selector (e.g. `--opponents foo`) exits 2 with the valid list, **before** Unity launches.

Run **two trainings side-by-side** on separate sockets with `--port` / `--eval-port` (each training
build uses its `--port`, and its eval build uses `--eval-port`, defaulting to `--port` + 1 when
omitted — so leave a ≥2-port gap between runs):

```bash
# run A: training on 50000, eval on 50001 (the default port + 1)
uv run python -m pop_trainer.rl.train --total-timesteps 10000 --run-dir runs/train-a --port 50000
# run B: training on 50002, eval on 50003
uv run python -m pop_trainer.rl.train --total-timesteps 10000 --run-dir runs/train-b --port 50002 --eval-port 50003
```

See the full flag surface:

```bash
uv run python -m pop_trainer.rl.train --help
```

Flags (defaults shown), grounded in [`_parse_args`](../src/pop_trainer/rl/train.py)
(`train.py:700-771`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--total-timesteps` | *(required)* | total env-steps to train (`train.py:712-714`). |
| `--run-dir` | *(required)* | output dir for checkpoints / sidecar / TensorBoard (`train.py:715-717`). |
| `--config` | `Assets/StreamingAssets/train_config.json` | game/training config JSON forwarded to the build launch (`DEFAULT_TRAIN_CONFIG`, `train.py:60-62,706-711`). |
| `--resume` | *(absent)* | a prior `run_dir` to resume — load the latest `model_*.zip` + `state.json` (`train.py:718-723`). |
| `--encoder-checkpoint` | `None` | optional pretrained-encoder `state_dict` for the features extractor (`train.py:724-729`). |
| `--freeze-encoder` | *(off)* | freeze the encoder weights during RL (`train.py:730-732`). |
| `--frame-stack` | `1` | `VecFrameStack` depth (`1` = passthrough, `train.py:733`). |
| `--seed` | `0` | master seed — threaded into SB3, the env, and the opponent provider (`train.py:734`). |
| `--opponents` | *(absent → full `DEFAULT_ROSTER`)* | comma-separated self-play roster selectors (e.g. `noop,random`), parsed to an ordered tuple and **validated** against [`AGENT_SELECTORS`](../src/pop_trainer/agents/registry.py) (`aggressive-coverage`, `noop`, `opponent-shadower`, `random`, `wall-hugger`); an unknown selector exits 2 with the valid list **before any Unity launch**. **Omitting** the flag trains against the full 5-selector [`DEFAULT_ROSTER`](../src/pop_trainer/rl/selfplay.py) (the omit path passes no kwarg, so the dataclass default owns it) (`train.py:736-740,766-770`; omit plumbing `train.py:777-778`). |
| `--opponent-strategy` | `round_robin` | opponent rotation: `round_robin` (resumable) or `uniform` (seed-only resume) — the two `choices` (`train.py:741-746`). |
| `--eval-freq` | `10000` | env-steps between win-rate evals (`0` = off) (`train.py:747-749`). |
| `--eval-episodes` | `10` | greedy episodes per opponent per eval (`train.py:750-752`). |
| `--checkpoint-freq` | `10000` | env-steps between checkpoints (the sidecar rides this cadence) (`train.py:753-755`). |
| `--port` | `50000` | TCP port the **training** build listens on → `game_port` (`train.py:756-758`). |
| `--eval-port` | *(absent → `--port` + 1)* | TCP port the **dedicated eval** build listens on → `eval_port`. When omitted the effective eval port is `port + 1` (the `effective_eval_port` property, `train.py:169-172`). When set it **must differ from `--port`**; an equal value is rejected by `TrainConfig.__post_init__` with a `ValueError` (NOT duplicated in argparse) (`train.py:759-764`; rejection `train.py:163-167`). |

**Knobs NOT on the CLI** (they use `TrainConfig` dataclass defaults — override in code, not the
command line): `n_envs` (`1`), `frame_shape` (the 640×360×3 frame), the PPO hyperparameters
(`learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`), and
`build_path` (`None` → the OS-aware [`core.launch.default_build_path`](../src/pop_trainer/core/launch.py))
— see `train.py:116-141`.

### The training topology (`train_config.json`)

The build launches under [`train_config.json`](../Assets/StreamingAssets/train_config.json): a
**single-arena AI-vs-AI** pixel config — `obs_pixels: true` @ 640×360, both `player1_ai`/`player2_ai`
`true`, `game_maxTime: 60`, `player_maxHealth: 3`, one `arena_path` (`Arenas/custom1.json`, no
rotation) (`train_config.json:1-22`).

> **`timeScale` MUST stay ≤ 5 (hard operational rule).** `train_config.json` ships `timeScale: 5`
> (`train_config.json:5`). Collection at `timeScale ≤ 5` is bit-identical to 1×; **`timeScale ≥ 10`
> corrupts** frames under load. NEVER raise it to 10 or above.

### Resume

`--resume <prior run_dir>` continues a previous run: it picks the **highest-step** `model_*.zip` in
that dir (`_latest_checkpoint`, `train.py:532-551`), does `PPO.load(env=...)`, restores the opponent
position + ELO from that dir's `state.json` sidecar (`train.py:594-607`), and continues with
`reset_num_timesteps=False` (`train.py:592,649`). Resuming a dir with no parseable `model_*.zip`
raises `FileNotFoundError` (`train.py:598-601`).

> **CRITICAL — sub-cadence smoke runs are NOT resumable.** The checkpoint cadence (`--checkpoint-freq`,
> default **10_000** env-steps; `train.py:127`) controls when a `model_*.zip` is written. A run whose
> `--total-timesteps` is **below the checkpoint cadence** writes **no intermediate `model_*.zip`** —
> only the final `state.json` — and therefore **cannot be `--resume`d** (resume needs a checkpoint zip
> and will raise `FileNotFoundError`). For a resumable checkpoint, keep `--total-timesteps >=` the
> `--checkpoint-freq` you pass (with the default that means `--total-timesteps >= 10000`); treat
> anything below the cadence as a **smoke-only, non-resumable** run. You can lower `--checkpoint-freq`
> to force an earlier checkpoint on a short run.

## 6. Validate coverage (no Unity required)

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
