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

The trainer / play app needs a standalone build of the simulator for the host OS. The headless build
entry points live in [`BuildScript`](../unity/Assets/Editor/BuildScript.cs): one method per platform,
each invoked via the Unity 6.5 editor in batchmode. Both build the enabled scenes, exit non-zero
on failure (so the batchmode caller can detect it), and ship `StreamingAssets` (`config.json` +
`Arenas/*.json` + the `demo_config.json` / `human_config.json` configs) into the player automatically.

### Windows

[`BuildScript.BuildWindows`](../unity/Assets/Editor/BuildScript.cs) → `unity/build/TankTwinStickShooter.exe`
(git-ignored):

```bash
"C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe" \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "C:\src\TankTwinStickShooter\unity" \
  -executeMethod BuildScript.BuildWindows
```

### macOS

[`BuildScript.BuildOSX`](../unity/Assets/Editor/BuildScript.cs) targets `StandaloneOSX` and writes the
`unity/build/TankTwinStickShooter.app` bundle (git-ignored). The Unity 6.5 editor binary on macOS is
inside the editor `.app`:

```bash
/Applications/Unity/Hub/Editor/6000.5.0f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "/Users/<you>/path/to/TankTwinStickShooter/unity" \
  -executeMethod BuildScript.BuildOSX
```

The launchable binary lives *inside* the bundle at `Contents/MacOS/`; you never type its name —
the OS-aware default resolver
([`core.launch.default_build_path`](../src/pop_trainer/core/launch.py)) globs the bundle and
hands the inner binary to the launcher (see §3).

> For how the built game runs (it is a **Python-clocked simulator** with no standalone human
> mode), see [game architecture](game-architecture.md).

## 3. Play an episode (live build + any two players)

With the build in place, run one episode with any pairing of player forms:

```bash
uv run python -m pop_trainer.play
```

This launches the build **windowed** with `unity/Assets/StreamingAssets/demo_config.json` (which enables
`obs_pixels` at 640×360 — required, since the env always reads a pixel frame each step), connects
over TCP, runs one episode of the default `player1 = aggressive-coverage` vs `player2 = opponent-shadower`,
prints a trace, and tears down.

**Three player forms** on each of `--player1` / `--player2`:

- `human` — keyboard play (see below).
- a rule-based selector — one of `aggressive-coverage`, `wall-hugger`, `opponent-shadower`,
  `random`, `noop`.
- `rl:<checkpoint.zip>` — a trained Stable-Baselines3 PPO model loaded from that path (a bare
  `*.zip` path also works). Any matchup is valid: rl-vs-human, rl-vs-rulebased, rl-vs-rl, etc. The
  RL model is loaded predict-only (`PPO.load`); `stable_baselines3` / `torch` are imported **lazily**
  — only when an `rl:` player is requested — so human / rule-based play stays torch-free. Example:

```bash
uv run python -m pop_trainer.play --player1 rl:runs/train/model.zip --player2 opponent-shadower
```

**No `--exe` needed on any OS.** The default executable is resolved for the current platform by
[`core.launch.default_build_path`](../src/pop_trainer/core/launch.py)
([`play.DEFAULT_EXE`](../src/pop_trainer/play.py)): on **Windows** it is
`unity/build/TankTwinStickShooter.exe`; on **macOS** it is the binary *inside* the
`unity/build/TankTwinStickShooter.app` bundle (`Contents/MacOS/`, discovered by globbing the bundle —
you don't type the inner name); on **Linux** the bare `unity/build/TankTwinStickShooter` binary. So
`python -m pop_trainer.play` works as-is once the platform build from §2 is present — pass `--exe`
only to point at a build elsewhere.

See the full flag surface:

```bash
uv run python -m pop_trainer.play --help
```

Useful flags (defaults shown): `--exe` (the OS-resolved build path above),
`--config unity/Assets/StreamingAssets/demo_config.json`, `--port 50000`, `--player1` / `--player2`
(each: `human`, a selector above, or `rl:<checkpoint.zip>`),
`--max-steps 600`, `--seed 0`. Play exits `2` if the build exe, config, or an `rl:` checkpoint is missing.

### Human-vs-human play (two players, one keyboard)

Human play needs the optional `human` extra (installs `pynput`, the global keyboard hook):

```bash
uv sync --extra human
```

Then make either or both players keyboard-driven. Two humans share ONE keyboard:

```bash
uv run python -m pop_trainer.play --player1 human --player2 human
```

One human + one bot is also valid, e.g. `--player1 human --player2 opponent-shadower` (or play a
human against a trained agent: `--player1 human --player2 rl:runs/train/model.zip`).

Controls:

- **P1** — move WASD, aim TFGH, fire LEFT SHIFT.
- **P2** — move IJKL, aim numpad 8/4/5/6, fire ENTER. (Numpad aim works with Num Lock ON or OFF.)

When a player is `human` the play app auto-selects `unity/Assets/StreamingAssets/human_config.json` (real-time
`timeScale: 1`, a more forgiving cadence) instead of `demo_config.json` — unless you pass `--config`.
(An `rl:` player does **not** change the default cadence — it runs at the bots config unless a human
is also present or `--config` is given.)

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

Rotate over the curated 10-map rotation (`core.maps.CURATED_ROTATION`), default pairing mix —
pass `--maps` with **no value**:

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
| `--maps` / `--map-rotation` | *(absent)* | rotation set. **No value** → the curated 10-map rotation (defined in `core.maps`); **a directory** → its `*.json` configs' arena targets (sorted); **a list of arena targets** → that order. **Absent** → no rotation (single `--map`). |
| `--pairing P1:P2` | `DEFAULT_PAIRINGS` mix | a `(player1:player2)` selector pairing; **repeatable**. Omitted → the default coverage-family mix (coverage vs each other + vs random). |
| `--map` | `custom1` | single-map (no-rotation) config; the **boot** + only arena when `--maps` is absent. |
| `--episodes` | `1` | episodes **per worker** |
| `--max-steps` | `800` | per-episode step cap |
| `--out-dir` | *(required)* | root output dir; worker `w` writes shards to `worker_<w>/` under it |
| `--workers` | `2` | parallel workers, one build per worker (**clamped to `[1, MAX_WORKERS=8]`**) |
| `--exe` | `unity/build/TankTwinStickShooter.exe` | path to the Unity build |
| `--base-port` | `50000` | base TCP port; worker `w` connects on `base_port + w` |
| `--seed` | `0` | base seed; worker `w` uses `base + w*10000` |
| `--shard-size` | *(auto)* | in-RAM **BUFFER** bound (samples per worker before a flush), **NOT a file-size knob**. Omitted → auto: derived from the byte budget so the per-worker buffer stays bounded regardless of frame resolution (the OOM-safe default). See [Memory safety](#memory-safety-the-pre-flight-guard). |
| `--description TEXT` | *(none)* | seed ONE free-form description into the manifest at collection time (see [Descriptions](#descriptions-annotating-a-dataset)). |
| `--description-author` | `human` | author for `--description` (free string, e.g. `claude`). Content provenance — who wrote the note, not git/PR attribution. |
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

`main` runs a **pre-flight guard** before launching any build (`collect_runner.py:786-815`): it
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
filename dumps (`collect_runner.py:819-838`). Three parts:

1. an **aggregate** line —
   `done: <N> shards, <N,NNN> samples, <K> workers -> <out-dir>`;
2. one **terse per-worker** line — `worker <id>: <n> shards, <n> samples` (no filenames);
3. a **per-map sample-count table** aligned to the `maps.json` sidecar — one row
   `map <id>  <arena>  <count>` per arena, **including arenas that got zero samples**.

The sample + per-map counts come from a **pure** counter
([`summarize_collection`](../src/pop_trainer/data/collect_runner.py), `collect_runner.py:576-611`)
over each worker's per-sample `map_ids` (read back by globbing each worker dir's `shard_*.npz` in
sorted order and lazily reading ONLY each shard's `map_ids` member — the big `frames` array is never
decompressed, `collect_runner.py:614-629`); the worker result dict supplies only the shard counts.

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

### The `manifest.json` fingerprint (what identifies a dataset)

Each run also writes a single **`manifest.json`** at the **run root** (next to the root
`maps.json`, `collect_runner.py:893-955`). It records, under
[`schema_version`](../src/pop_trainer/data/manifest.py)` = 2`:

- **descriptions** — a list of free-form human/agent annotations (empty unless seeded with
  `--description`; see [Descriptions](#descriptions-annotating-a-dataset)). Each entry is exactly
  `{"author", "text", "added_utc"}`;
- **collection params** — `seed`, the full `command` (`sys.argv`), `workers`, `episodes`,
  `max_steps`, the `maps` list, the `pairings`, and the `total_shards` / `total_samples` /
  `per_map_samples` counts (reused from the same summary as the printout);
- **provenance** — `git_commit` (`git rev-parse HEAD`), `git_dirty` (`git status --porcelain`,
  `None` if git is absent), and a `build` fingerprint `{path, mtime_utc, size_bytes}` of the Unity
  binary that produced the run;
- **machine** — `hostname`, `platform`, `system`, `release`, `arch`, `processor`, `cpu_count`,
  `ram_total_gb`, `python`.

> **Not a reproduce-this recipe — it is the dataset's IDENTITY.** Datasets are **NOT
> byte-reproducible run-to-run**: Unity physics + GPU rendering vary across machines, so two runs
> of the same command on two boxes will differ. The manifest exists to **identify** a given dataset
> and make those discrepancies **explicable** (which commit, which build, which machine) — it does
> NOT let you regenerate a corpus byte-for-byte.

The shaping is a **pure, unit-tested** assembler
([`build_manifest`](../src/pop_trainer/data/manifest.py), `manifest.py:70-139`); the live gathering
of the provenance/machine values is `main`'s untested CLI glue. The file is written **atomically**
(tmp → `Path.replace`). See [data](components/data.md#the-run-root-manifestjson-dataset-fingerprint).

### Descriptions (annotating a dataset)

The manifest carries a `descriptions` list of free-form human/agent annotations so you can record
WHAT a dataset is. Each entry is exactly `{"author", "text", "added_utc"}` (`added_utc` ISO-8601
UTC); `author` is **content provenance** (who wrote the note — `human`, `claude`, …), NOT git/PR
attribution. `MANIFEST_SCHEMA_VERSION` is **2** for this field; a pre-existing v1 manifest with no
`descriptions` key is tolerated (treated as empty), so older datasets stay annotatable.

**At collection time** — seed one description as the manifest is written:

```bash
uv run python -m pop_trainer.data.collect_runner --out-dir runs/collect-demo \
  --description "first center-block sweep" --description-author claude
```

`--description-author` defaults to `human`. The seed entry's `added_utc` shares the manifest's
`created_utc` (one clock read).

**Post-hoc** — append (or list) descriptions on an already-collected run with
[`python -m pop_trainer.data.describe`](../src/pop_trainer/data/describe.py):

```bash
# append one annotation (rewrites manifest.json atomically; --author defaults to 'human')
uv run python -m pop_trainer.data.describe runs/collect-demo --text "looks clean" --author claude

# list the existing descriptions instead of adding one
uv run python -m pop_trainer.data.describe runs/collect-demo --list
```

`describe` takes the run **directory** holding `manifest.json`. It returns **0** on success and
**2** on a missing `manifest.json` or a missing `--text` (when not using `--list`).

### The obs_pixels gotcha

Collection **must** receive pixel frames (the env reads a length-prefixed frame after every state;
a build without `obs_pixels` would leave the env blocking on bytes that never arrive). So the build
**boots** on the obs_pixels-enabled `--map` config (`custom1` → `unity/Assets/StreamingAssets/demo_config.json`,
the **same** config the play app launches with — 640×360 pixels on, arena `Arenas/custom1.json`) and
rotates the arena via `switch_arena` at runtime. The curated rotation supplies **arena switch
targets directly** (the `Arenas/<name>.json` strings) and is never used as the boot config. Either
way the captured `(frame, state)` rows are byte-for-byte the
play / RL observation pipeline. `main` exits `2` if the build exe or the resolved boot config is
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

Like play (§3) and collection (§4), this **launches a live windowed Unity build** (needs the §2
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

Run **two trainings side-by-side** on separate sockets with `--port` / `--eval-port`. At the default
`--n-envs 1` each run uses **two** ports — its `--port` for the training build and `--port + 1` for
the eval build (the default `--eval-port`) — so leave a **≥2-port gap** between runs (at `--n-envs N`
a run uses `2N` ports — `[P, P+N-1]` train + `[P+N, P+2N-1]` eval — so leave a `≥2N` gap):

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
(`train.py:1261-1362`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--total-timesteps` | *(required)* | total env-steps to train (`train.py:1273-1275`). |
| `--run-dir` | *(required)* | output dir for checkpoints / sidecar / TensorBoard (`train.py:1276-1278`). |
| `--config` | `unity/Assets/StreamingAssets/train_config.json` | game/training config JSON forwarded to the build launch (`DEFAULT_TRAIN_CONFIG`, `train.py:94`; arg `train.py:1267-1272`). |
| `--resume` | *(absent)* | a prior `run_dir` to resume — load the latest `model_*.zip` + `state.json` (`train.py:1279-1284`). |
| `--encoder-checkpoint` | `None` | optional pretrained-encoder `state_dict` for the features extractor (`train.py:1285-1290`). |
| `--freeze-encoder` | *(off)* | freeze the encoder weights during RL (`train.py:1291-1293`). |
| `--frame-stack` | `1` | `VecFrameStack` depth (`1` = passthrough, `train.py:1294`). |
| `--n-envs` | `1` | parallel **training** envs. `1` → a single in-process `DummyVecEnv`. `N > 1` → a `SubprocVecEnv` of `N` Unity builds, each on its OWN port `game_port + i` (`i` in `0..N-1`) in its OWN process (`start_method="spawn"`, Windows-safe). The eval set is built at the SAME width (`M == N`, time-multiplexed). See [Multi-env training](#multi-env-training-n-envs--1). |
| `--n-steps` | `2048` | PPO rollout length — **THE memory lever at `--n-envs > 1`** (lower it for more envs; the rollout buffer scales `n_steps × n_envs`). Default `2048` (`TrainConfig.n_steps`, `train.py:206`). |
| `--allow-oversized` | *(off)* | skip the pre-flight rollout-buffer memory **abort** (the estimate is still printed). Use only when the box has RAM the conservative guard does not model (guard `check_rl_memory_budget`, `train.py:344-392`; it charges `n_envs` Unity instances, not `n_envs + 1`, since train/eval sets never coexist). |
| `--seed` | `0` | master seed — threaded into SB3, the env, and the opponent provider (`train.py:1313`). |
| `--opponents` | *(absent → full `DEFAULT_ROSTER`)* | comma-separated self-play roster selectors (e.g. `noop,random`), parsed to an ordered tuple and **validated** against [`AGENT_SELECTORS`](../src/pop_trainer/agents/registry.py) (`aggressive-coverage`, `noop`, `opponent-shadower`, `random`, `wall-hugger`); an unknown selector exits 2 with the valid list **before any Unity launch**. **Omitting** the flag trains against the full 5-selector [`DEFAULT_ROSTER`](../src/pop_trainer/rl/selfplay.py) (the omit path passes no kwarg, so the dataclass default owns it) (`train.py:1314-1319,1357-1361`; omit plumbing `train.py:1375`). |
| `--opponent-strategy` | `round_robin` | opponent rotation: `round_robin` (resumable) or `uniform` (seed-only resume) — the two `choices` (`train.py:1320-1325`). |
| `--eval-freq` | `10000` | env-steps between win-rate evals (`0` = off) (`train.py:1326-1328`). |
| `--eval-episodes` | `10` | greedy episodes per opponent per eval (`train.py:1329-1331`). |
| `--checkpoint-freq` | `10000` | env-steps between checkpoints (the sidecar rides this cadence) (`train.py:1332-1334`). |
| `--port` | `50000` | base TCP port → `game_port`. Training env `i` listens on `game_port + i` for `i` in `0..n_envs-1` (at `--n-envs 1` that is just `--port`); the training block is `[game_port, game_port + n_envs - 1]` (`training_ports`, `train.py:558-560`). |
| `--eval-port` | *(absent → `game_port + n_envs`)* | base TCP port for the **eval block** → `eval_port`. There are `M == n_envs` eval builds (same width as training) on `[eval_port, eval_port + n_envs - 1]`. When omitted the base is `game_port + n_envs` — the **first port AFTER the training range** `[game_port, game_port + n_envs - 1]` (so at `--n-envs 1` it is `port + 1` as before; at `--n-envs 4 --port 50000` the eval block is `50004..50007`) (`effective_eval_port`, `train.py:257-267`; `eval_ports`, `train.py:563-570`). When set it **must differ from `--port`** AND its **whole block must NOT overlap the training block** — either is rejected by `TrainConfig.__post_init__` with a `ValueError` (rejection `train.py:238-255`). The eval and training builds never run concurrently (time-multiplexed), but disjoint blocks keep a relaunched-but-not-yet-reaped instance from clashing on bind. |
| `--log-dir` | *(absent → `run_dir/logs`)* | directory for the per-process observability logs (the JSONL `training-system.log` + `env-<role>-<port>.log` + the paired per-launch Unity `unity-<role>-<port>-<attempt>.log`). See [Observability logs](#observability-logs) (`effective_log_dir`, `train.py:269-272`). |
| `--debug` | *(off → INFO)* | the SINGLE switch: crank ALL observability logs to DEBUG (per-step send/recv/step). Equivalent to setting `POP_LOG_LEVEL=DEBUG`; resolved via `level_from_env` (arg `train.py:1350-1355`, resolution `train.py:1377-1380`). |

**Knobs NOT on the CLI** (they use `TrainConfig` dataclass defaults — override in code, not the
command line): `frame_shape` (the 640×360×3 frame), the PPO hyperparameters other than `--n-steps`
(`learning_rate`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`), and `build_path`
(`None` → the OS-aware [`core.launch.default_build_path`](../src/pop_trainer/core/launch.py)) — see
`train.py:204-216`.

### Multi-env training (`--n-envs > 1`)

`--n-envs 1` (the default) runs one in-process training env (a `DummyVecEnv`). `--n-envs N` for
`N > 1` fans out to a **`SubprocVecEnv` of `N` Unity builds**, each launched in its OWN process
(`start_method="spawn"`, Windows-safe) on its OWN training port `game_port + i` (`i` in `0..N-1`).

The eval set is built at the **SAME width** (`M_eval == N_train == N`) on a **disjoint** port block
**auto-assigned** to start at the first port after the training range (`game_port + n_envs`) — you do
**not** pass `--eval-port` for the common case. The two SETS never run at once: at each eval boundary
the run **tears down all `N` training builds, lazy-launches the `N` eval builds, runs the eval in
parallel, tears the eval builds down, and respawns training** — so peak concurrent instances is `N`,
not `2N` (this is why the memory guard charges `N`, not `N + 1`).

Concrete example — **4 training envs + 4 (time-multiplexed) eval envs**, with a shorter rollout to
keep memory in budget:

```bash
uv run python -m pop_trainer.rl.train \
  --total-timesteps 200000 --run-dir runs/train-4x \
  --n-envs 4 --n-steps 512 --port 50000
```

With `--port 50000 --n-envs 4`:

- **Training port block** = `[50000, 50003]` — builds on `50000, 50001, 50002, 50003`
  (`training_ports`, `train.py:558-560`).
- **Eval port block** = `[50004, 50007]` — auto-assigned `game_port + n_envs` = `50000 + 4 = 50004`
  as its low end (the first port after the training range), `M == N == 4` eval builds on
  `50004..50007` (no `--eval-port` needed; `effective_eval_port`, `train.py:257-267`; `eval_ports`,
  `train.py:563-570`). The eval block is validated disjoint from the training block
  (`train.py:238-255`).

In general at `--port P --n-envs N`: training = `[P, P + N - 1]`, eval = `[P + N, P + 2N - 1]`.

> **Memory — the rollout buffer is the OOM surface at high `--n-envs`.** The SB3 PPO `RolloutBuffer`
> stores `(n_steps, n_envs, *obs_shape)` uint8 frames (1 byte/element), so its obs cost is
> approximately **`n_steps × n_envs × frame_bytes × frame_stack`** (`estimate_rl_memory_bytes`,
> `train.py:330-341`). **Lowering `--n-steps` is the primary lever** — halve it to roughly halve the
> buffer at a fixed `--n-envs` (which is why the 4-env example drops to `--n-steps 512`).
>
> A **pre-flight guard** runs at startup (`check_rl_memory_budget`, `train.py:344-392`; called in
> `train_local`, `train.py:1062-1080`): it reads `psutil.virtual_memory().available`, adds **~1 GB per
> live Unity instance** for **`n_envs` instances** — NOT `n_envs + 1` — because the eval and training
> instance SETS **never coexist** (the eval cycle tears the `N` training builds down before launching
> the `M == N` eval builds, so peak concurrent = `max(N_train, M_eval) = n_envs`;
> `UNITY_INSTANCE_BYTES = 1 GB`, `train.py:123`; `unity_instances = n_envs`, `train.py:372`). It adds
> that to the buffer estimate, **always prints the estimate line**, and **aborts with exit code `2`
> BEFORE any build launches** when the modelled total exceeds **60 % of available RAM**
> (`MEMORY_MARGIN = 0.6`, `train.py:127`) — unless `--allow-oversized` (which still prints the estimate,
> then proceeds). If the guard aborts, lower `--n-steps` or `--n-envs` (or pass `--allow-oversized`
> only if the box has RAM the conservative guard does not model). `timeScale` **MUST stay ≤ 5** for the
> training builds too — the collection cap applies (see below).

At `--n-envs > 1` the opponent rotation is **per-subproc** (each training build has its own seeded
`OpponentProvider`, seed `cfg.seed + i`, unreachable from the main process), so **`--resume` RESEEDS
the rotation** (approximate phase, like `uniform`) rather than restoring a position-exact
`round_robin` — see the [rl component page](components/rl.md#multi-env-n_envs--1). At `--n-envs 1`
resume stays position-exact `round_robin` as documented under [Resume](#resume).

> **Stall recovery — the run survives a lost connection, it does not prevent the stall.** Multi-env
> training intermittently hits a ~30 s instance stall (an underlying C# reset-region issue, **not yet
> fixed**). When a socket drops, `TankEnv` does a **kill-old-first reconnect** — it reaps the OLD
> (possibly stalled) instance to free its port BEFORE launching the replacement — so the run
> **recovers and continues** instead of wedging. The dropped step is counted as a reward-0 truncation
> and the next episode starts fresh. A relaunch points Unity's `-logFile` at the NEXT
> `unity-<role>-<port>-<attempt>.log`, so the stalled instance's prior C# log survives for
> post-mortem (see [Observability logs](#observability-logs)). If you see a worker reconnect (a
> `worker_death` or `step_lost_connection` record in the logs) mid-run, that is the recovery path, not
> a crash.

### The training topology (`train_config.json`)

The build launches under [`train_config.json`](../unity/Assets/StreamingAssets/train_config.json): a
**single-arena AI-vs-AI** pixel config — `obs_pixels: true` @ 640×360, both `player1_ai`/`player2_ai`
`true`, `game_maxTime: 60`, `player_maxHealth: 3`, one `arena_path` (`Arenas/custom1.json`, no
rotation) (`train_config.json:1-22`).

> **`timeScale` MUST stay ≤ 5 (hard operational rule).** `train_config.json` ships `timeScale: 5`
> (`train_config.json:5`). Collection at `timeScale ≤ 5` is bit-identical to 1×; **`timeScale ≥ 10`
> corrupts** frames under load. NEVER raise it to 10 or above.

### Resume

`--resume <prior run_dir>` continues a previous run: it picks the **highest-step** `model_*.zip` in
that dir (`_latest_checkpoint`, `train.py:967-986`), does `PPO.load(env=...)`, restores the opponent
position + ELO from that dir's `state.json` sidecar (`train.py:1110-1115`), and continues with
`reset_num_timesteps=False` (`train.py:1100,1176`). Resuming a dir with no parseable `model_*.zip`
raises `FileNotFoundError` (`train.py:1106-1109`).

> **CRITICAL — sub-cadence smoke runs are NOT resumable.** The checkpoint cadence (`--checkpoint-freq`,
> default **10_000** env-steps; `train.py:199`) controls when a `model_*.zip` is written. A run whose
> `--total-timesteps` is **below the checkpoint cadence** writes **no intermediate `model_*.zip`** —
> only the final `state.json` — and therefore **cannot be `--resume`d** (resume needs a checkpoint zip
> and will raise `FileNotFoundError`). For a resumable checkpoint, keep `--total-timesteps >=` the
> `--checkpoint-freq` you pass (with the default that means `--total-timesteps >= 10000`); treat
> anything below the cadence as a **smoke-only, non-resumable** run. You can lower `--checkpoint-freq`
> to force an earlier checkpoint on a short run.

### Observability logs

Every run writes a per-process **structured-JSONL** log trail into the log dir (default
`run_dir/logs`, override with `--log-dir`). This is the operator's guide to reading them when
diagnosing the multi-env hang. The logging is **purely observational** — it does not touch the TCP
wire, the 52-float state, message ordering, or control flow; the Python side is off by default on
any unwired path and the C# side is gated on the build's `verbose` config flag (see
[game architecture](game-architecture.md#observability-logging-verbose-gated-zero-wire-impact)).

**Per-process file routing.** A run is many processes (the main process, each `SubprocVecEnv`
worker, the eval env behind the eval callback), and each writes its OWN file so concurrent writers
never contend:

- `training-system.log` — the main process (`role=system`, `port=null`, `layer=train`).
- `env-<role>-<port>.log` — one per env connection (`role` = `train`/`eval`, `port` = that env's
  TCP port). This file is **SHARED** by the env layer (`layer=env`) and the protocol layer
  (`layer=protocol`) for that one socket. It is **NOT** suffixed per launch — the Python env log
  appends across respawns.
- `unity-<role>-<port>-<attempt>.log` — one per Unity **LAUNCH** (the build's `-logFile` output),
  paired to the Python `env-*.log` by the same `(role, port)`. The **`<attempt>` suffix** (`-0` for
  the first launch, `-1` for the first relaunch, …) is **load-bearing**: Unity's `-logFile`
  truncates its target on every launch, so giving each launch a DISTINCT file means a respawn / lazy
  re-launch / reconnect **never wipes the prior (possibly hung) instance's C# log** — prior logs
  survive for post-mortem (`_attempt_unity_log_path`, `train.py:416-427`).

So a **1-train + 1-eval run** (the default `--n-envs 1`, ports 50000/50001) that never relaunches
produces **5 files** (the Unity logs carry the `-0` first-launch suffix):

```
training-system.log
env-train-50000.log    unity-train-50000-0.log
env-eval-50001.log     unity-eval-50001-0.log
```

A reconnect / respawn on the train socket would add `unity-train-50000-1.log` (the prior `-0` is left
intact). General formula: **1 `env-*.log` per connection + 1 `unity-*-<attempt>.log` per launch of
that connection + the one system file**. (At `--n-envs N` that is `N` train connections + `N` eval
connections, since the eval set is the same width — each set's Unity instances start lazily on first
use and respawn across eval cycles, accumulating `-<attempt>` files.)

**INFO vs DEBUG.** Default level is **INFO** (handshake / reset / episode milestones + the
`run_config` / memory-estimate / `rollout_*` / `checkpoint_save` / `worker_death` system markers).
Pass `--debug` (or set `POP_LOG_LEVEL=DEBUG`) — the **single switch** — to crank ALL logs to
**DEBUG** (adds per-step send/recv/step).

**JSONL schema + the cross-stack merge key.** Every Python line is one strict-JSON object with at
least `ts_wall` (`time.time`), `ts_mono` (`time.monotonic`), `level`, `layer`, `role`, `port`,
`event`, plus any detail fields (`bytes` / `elapsed_ms` / `episode` / `timesteps` / `fps` / …). The
C# lines are `[tag] wall=<ISO-8601 UtcNow> k=v …`. **`ts_wall` (Python) and `wall=` (C#) are the
same wall-clock**, so they are how you **merge across files and across the Python/C# stack** when
chasing where a hang happened (e.g. line up a Python `step` against the C# `readpixels` / dead-zone
lines for that port).

**Eval isolation.** Each logger is set up with `propagate=False`, so eval records land ONLY in
`env-eval-<port>.log` and **never leak into `training-system.log`** — when scanning the system log
for the hang you are looking at training + system markers only, with no eval noise.

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
