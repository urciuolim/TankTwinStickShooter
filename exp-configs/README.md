# exp-configs — ready-made maps & rewards for experiments

Point the trainer at one map config and one reward config to run an experiment.
See `docs/experiments.md` for the full format reference; this dir is the curated
set to grab from.

```
exp-configs/
  maps/
    <name>.json          # game config (references its arena)
    Arenas/<name>.json   # the arena geometry (walls)
  rewards/
    <name>.json          # a RewardConfig preset
  _generate_maps.py      # regenerates the maps (symmetry guaranteed by construction)
```

## Maps (10)

Every map is **point-symmetric** under a 180° rotation about the arena centre
(`(x,y) -> (-1-x, -1-y)`) — the same flip the game uses on the two players' spawns —
so **neither side has a wall advantage**. All maps share the standard **(36, 60, 3)**
observation (same Walls dims), so a model trained on one map loads on any other.
Symmetry, obs-shape, and floor-connectivity are pinned by `tests/test_exp_configs.py`.

| map | layout |
|---|---|
| `custom1_2021` | **the 2021 training arena**, copied verbatim |
| `empty` | open arena, border only |
| `center_block` | a 2×2 block dead centre |
| `four_pillars` | four pillars at the corners of a central rectangle |
| `central_cross` | a plus/cross of walls through the middle |
| `opposing_l` | two mirrored L-walls |
| `diagonal_pillars` | pillars stepping along both diagonals |
| `chokepoint` | central vertical walls with a horizontal gap (a pinch point) |
| `scattered` | assorted symmetric pillars |
| `ring_fragments` | fragments of an inner ring at the corners |

Regenerate / add more: edit the seed lists in `_generate_maps.py` and run
`uv run python exp-configs/_generate_maps.py` — it auto-mirrors every wall you place,
so new maps stay symmetric by construction. (Keep the Walls dims fixed to keep the
(36, 60, 3) obs and cross-map model compatibility.)

## Rewards (3)

Budget-based (full-episode totals, spread per step over `max_episode_length = 300`):

| reward | meaning |
|---|---|
| `sparse` | just **+1 win / −1 loss**, no shaping |
| `classic_2021` | the **2021 reward**: +1/−1 plus a `−0.003`/step time penalty (budget `−0.9`), no action cost |
| `shaped_default` | the **current default**: +1/−1, time budget `−1.0`, action-cost budget `−0.1` |

## Run an experiment (PowerShell, one line each)

The 2021 map with the 2021 reward:

```powershell
uv run python -m tank_twin.train --game-path "C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe" --config "C:\src\TankTwinStickShooter\exp-configs\maps\custom1_2021.json" --reward-config "C:\src\TankTwinStickShooter\exp-configs\rewards\classic_2021.json" --run-name exp-2021 --device cuda
```

A chokepoint map with the current shaped reward:

```powershell
uv run python -m tank_twin.train --game-path "C:\src\TankTwinStickShooter\build\TankTwinStickShooter.exe" --config "C:\src\TankTwinStickShooter\exp-configs\maps\chokepoint.json" --reward-config "C:\src\TankTwinStickShooter\exp-configs\rewards\shaped_default.json" --run-name exp-chokepoint --device cuda
```

Per-knob reward flags (`--time-penalty-total`, `--action-cost-total`, …) still override
a `--reward-config` file if you add them. Every run records its resolved map + reward in
`runs/<run-name>/manifest.json`.
