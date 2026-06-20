# Experiments: any reward on any map (self-serve)

The trainer and the Unity build now read the **same map from one source**, so the agent's
observation matches the live game (no more "the obs and the simulator disagree"). And the
reward is **fully configurable** from the CLI or a JSON file. This guide is how you, the
CTO, run an experiment with any reward on any map — usually with **no rebuild**.

Trainer entry point (Windows, isolated `uv` venv):

```
uv run python -m tank_twin.train --game-path <abs path to build exe> [options]
```

`--game-path` is **required**. The default build exe is
`C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe`.

---

## 1. Configure a map

### Arena JSON format

An arena is one JSON object with a `"Floor"` block and a `"Walls"` block. Each block has:

- `"dims"`: `{minX, maxX, minY, maxY}` — the integer column/row bounds.
- `"tileID"`: which tile to draw (0 = floor, 1 = wall in the examples).
- One `"<x>": [y, y, ...]` list **per integer column** `x` in `[minX..maxX]`, giving the
  y-rows that have a tile in that column.

**The observation grid is built from `Walls` ONLY.** Its shape is
`((maxY-minY+1)*p, (maxX-minX+1)*p, 3)` with `p = 3`. The shipped `custom1` (and the
example below) use Walls dims `minX -10, maxX 9, minY -6, maxY 5` -> a 20x12 cell grid ->
a **`(36, 60, 3)` observation**.

> **Keep the Walls dims the same to reuse the `(36, 60, 3)` obs.** If you change the Walls
> dims, the observation **shape** changes, and a model trained on a different shape **will
> not load**. Change the interior wall layout freely; keep the dims fixed if you want to
> reuse a trained model.

**Spawns are NOT in the arena JSON.** The build reads spawn behavior from the **config**
keys: `player_randomStart`, `player_x_spawn_lim`, `player_y_spawn_lim`, `player_maxHealth`.

**Strict JSON only** — no trailing commas, no leading-dot floats (write `0.5`, not `.5`).
Unity's Newtonsoft tolerates loose JSON; Python's `json` does not, and the env will raise.

#### Worked example: a tiny arena (border + one interior pillar)

This keeps the standard Walls dims, so the obs stays `(36, 60, 3)` and a model trained on
it is interchangeable with `custom1`. It is a full outer wall ring plus a 2x2 pillar in the
middle (columns -1 and 0, rows -1 and 0). Copy it verbatim into `Arenas/tiny.json`:

```json
{
  "Floor": {
    "dims": {"minX": -8, "maxX": 7, "minY": -4, "maxY": 3},
    "tileID": 0,
    "-8": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-7": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-6": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-5": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-4": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-3": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-2": [-4, -3, -2, -1, 0, 1, 2, 3],
    "-1": [-4, -3, -2, -1, 0, 1, 2, 3],
    "0": [-4, -3, -2, -1, 0, 1, 2, 3],
    "1": [-4, -3, -2, -1, 0, 1, 2, 3],
    "2": [-4, -3, -2, -1, 0, 1, 2, 3],
    "3": [-4, -3, -2, -1, 0, 1, 2, 3],
    "4": [-4, -3, -2, -1, 0, 1, 2, 3],
    "5": [-4, -3, -2, -1, 0, 1, 2, 3],
    "6": [-4, -3, -2, -1, 0, 1, 2, 3],
    "7": [-4, -3, -2, -1, 0, 1, 2, 3]
  },
  "Walls": {
    "dims": {"minX": -10, "maxX": 9, "minY": -6, "maxY": 5},
    "tileID": 1,
    "-10": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "-9": [-6, 5],
    "-8": [-6, 5],
    "-7": [-6, 5],
    "-6": [-6, 5],
    "-5": [-6, 5],
    "-4": [-6, 5],
    "-3": [-6, 5],
    "-2": [-6, 5],
    "-1": [-6, -1, 0, 5],
    "0": [-6, -1, 0, 5],
    "1": [-6, 5],
    "2": [-6, 5],
    "3": [-6, 5],
    "4": [-6, 5],
    "5": [-6, 5],
    "6": [-6, 5],
    "7": [-6, 5],
    "8": [-6, 5],
    "9": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  }
}
```

(Here the `-10` / `9` columns are full vertical walls, every other column has the top
`-6` and bottom `5` border, and columns `-1` / `0` additionally carry the interior pillar
at rows `-1, 0`.)

### Run a training experiment on a custom map (external — NO rebuild)

External maps need **no rebuild**. The build was verified to load an external config +
arena over the socket with no rebuild. Lay out a config dir like this:

```
C:\experiments\tiny-map\
  config.json
  Arenas\
    tiny.json
```

The config's `arena_path` resolves **relative to the config file's directory** (an
absolute `arena_path` is used as-is), so a config dir and its `Arenas\` subdir travel
together. A minimal `config.json` (strict JSON):

```json
{
  "connectionIP": "127.0.0.1",
  "connectionPort": 50000,
  "game_maxTime": 60,
  "timeScale": 5,
  "verbose": false,
  "arena_path": "Arenas/tiny.json",
  "player1_ai": true,
  "player1_keyboard": false,
  "player2_ai": true,
  "player2_keyboard": false,
  "ai_async": false,
  "ai_actionFreq": 10,
  "ai_fixedDeltaTime": 0.02,
  "player_maxHealth": 1,
  "player_randomStart": false,
  "player_x_spawn_lim": 0.5,
  "player_y_spawn_lim": 1.0
}
```

Then train:

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --config C:\experiments\tiny-map\config.json ^
  --run-name tiny-map-exp
```

`--map` is an exact **alias** for `--config` (same effect) — type whichever reads better.

When you pass `--config`, that one file picks the arena for **both** the build (forwarded
on the launch arg-list as `--config <abspath>`) **and** the Python obs (the env reads the
same config's `arena_path`). One source -> **obs == game**.

**With no `--config`, you get `custom1` by default** — the env reads the build's
StreamingAssets `config.json` (whose `arena_path` is `Arenas/custom1.json`), the same map
the build runs. Still single-source; you just don't pick a custom map.

### Sanity-check a map without training

To eyeball that a map loads (no RL, no training), run the socket smoke host. Set
`"verbose": true` in the config so Unity logs the arena it loaded:

```
uv run python PythonScripts/play_local.py ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --config C:\experiments\tiny-map\config.json ^
  --port 50901 --verbose
```

Then check Unity's `Player.log` for `Arena loaded from: Arenas/tiny.json`. The log lives at:

```
%USERPROFILE%\AppData\LocalLow\DefaultCompany\Tank Twin Stick Shooter\Player.log
```

### Bake a map into the build (baked default — REBUILD required)

You only need a rebuild to **ship** a map inside the player (a new baked default). For
day-to-day experiments, prefer external `--config` (above) and skip this.

1. Add your arena to `C:\src\TankTwinStickShooter\Assets\StreamingAssets\Arenas\`.
2. Point a StreamingAssets `config.json`'s `arena_path` at it.
3. Rebuild via the editor build method `BuildScript.BuildWindows`
   (`C:\src\TankTwinStickShooter\Assets\Editor\BuildScript.cs`). It outputs
   `C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe` and copies the
   StreamingAssets `config.json` + `Arenas\*.json` automatically. Headless command:

```
"C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe" -batchmode -quit -accept-apiupdate -projectPath "C:\src\TankTwinStickShooter" -executeMethod BuildScript.BuildWindows -logFile -
```

**Rebuild is required only for:** changing the DEFAULT arena the build ships, or shipping
NEW arena files inside the player. **No rebuild** for external-config experiments.

> **Single-source guarantee:** whichever path you use (external `--config` or baked
> default), the trainer and the build read the **same** map, so the agent's observation
> matches the live game.

---

## 2. Configure a reward

The reward is **budget-based**: the knobs are full-episode totals, spread per-step over
`max_episode_length = 300` (the env's `max_steps`). Set it with per-knob CLI flags, a JSON
file, or both. **Precedence: CLI flags > `--reward-config` file > defaults.**

| CLI flag | Default | JSON key | What it does |
|---|---|---|---|
| `--win-reward` | `1.0` | `win_reward` | Terminal reward ADDED on a win. |
| `--loss-reward` | `-1.0` | `loss_reward` | Terminal reward ADDED on a loss. |
| `--time-penalty-total` | `-1.0` | `time_total` | **Full-episode** time budget; per-step = total/300, accrues EVERY step. |
| `--action-cost-total` | `-0.1` | `action_total` | **Full-episode** action-cost budget at constant MAX action; scales linearly with action magnitude. |
| `--action-norm` | `5.0` | `action_norm` | L1_MAX — all 5 action dims saturated sums to 5. |

Equivalent `--reward-config` JSON shape (strict):

```json
{
  "win_reward": 1.0,
  "loss_reward": -1.0,
  "time_total": -1.0,
  "action_total": -0.1,
  "action_norm": 5.0
}
```

### Episode-budget semantics (plain)

- **`time_total` is the full-episode time budget.** At the default `-1.0`, every step
  accrues `-1/300`, so a full 300-step episode accrues `-1.0` from time.
- **`action_total` is the full-episode cost at constant MAX action.** Per-step cost is
  `(action_total/300) * (L1(action)/action_norm)`. Zero action -> 0 cost; a constant max
  action over 300 steps -> the full `action_total` (default `-0.1`); it scales linearly
  with how hard the agent is pushing.
- **Penalties accrue EVERY step.** The `±1` win/loss is **ADDED** on the decided step (it
  does not overwrite the accrued penalties).
- **Worst-case late LOSS ~= -2.1**: `loss(-1) + time(-1) + action(-0.1)`. That asymmetry
  (a slow loss is worse than a fast one) is intended.

### Copy-paste examples

**(a) The new DEFAULT reward** — just omit all reward flags:

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --run-name default-reward
```

Budgets: `win +1`, `loss -1`, `time_total -1.0` (full episode), `action_total -0.1`
(at constant max action), `action_norm 5.0`.

**(b) SPARSE win/loss-only** — zero out the shaping; the classic `±1`-only terminal:

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --time-penalty-total 0 --action-cost-total 0 ^
  --run-name sparse-reward
```

This reproduces the legacy terminal-only reward: `+1` win, `-1` loss, nothing in between.

**(c) HEAVIER action cost** — make actions more expensive to encourage smoother, cheaper
control:

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --action-cost-total -0.5 ^
  --run-name heavy-action-cost
```

Now a full episode at max action costs `-0.5` (5x the default), so the agent is pushed
toward gentler, less wasteful actions.

**(d) From a file** — save the JSON above as `C:\experiments\myreward.json` (e.g. with
`action_total` set to `-0.5`) and pass it:

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --reward-config C:\experiments\myreward.json ^
  --run-name file-reward
```

Per-knob flags still override the file if you add them (CLI > file > defaults).

### Mix map + reward + the usual knobs

Combine freely. Other handy flags: `--timesteps`, `--seed`, `--device {auto,cpu,cuda}`.

```
uv run python -m tank_twin.train ^
  --game-path C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe ^
  --config C:\experiments\tiny-map\config.json ^
  --action-cost-total -0.5 --time-penalty-total -2.0 ^
  --timesteps 200000 --seed 7 --device cuda ^
  --run-name tiny-heavy-cost
```

---

## 3. Reproducibility

Every run is **self-describing**. The trainer writes
`runs/<run-name>/manifest.json` recording:

- the **resolved reward config** (all five budget knobs as actually used),
- the **map** it trained on (`config_path` — `null` means the `custom1` default),
- `seed`, `timesteps`, device, and the PPO knobs.

So a finished run carries exactly which map and which reward produced it, and is
re-runnable from its own manifest. `--run-name <name>` controls the outputs:
`runs/<name>/` (checkpoints + manifest + metrics) and `models/<name>.zip` (final model).
These dirs are git-ignored.
