# exp-configs — ready-made maps for experiments

Point the trainer at one map config to run an experiment.
See `docs/experiments.md` for the full format reference; this dir is the curated
set to grab from.

```
exp-configs/
  maps/
    <name>.json          # game config (references its arena)
  _generate_maps.py      # regenerates the maps (symmetry guaranteed by construction)
```

Each `maps/<name>.json` is the game-config wrapper (it carries an `arena_path`). The arena
**geometry** (walls/floor) is written by `_generate_maps.py` to `Assets/StreamingAssets/Arenas/`,
the single source of truth for that geometry.

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

## Rewards

The reward shaping lives in code, not here — see `src/pop_trainer/env/rewards.py`
(budget-based: full-episode totals spread per step). There are no reward-preset JSONs in this dir.

## Run an experiment

The previous `tank_twin.train` experiment commands were retired at M1 (the `tank_twin` package
was removed; see git history). A `pop_trainer` training entrypoint that consumes these map
configs supersedes them and is not yet wired.
