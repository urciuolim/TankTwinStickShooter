# exp-configs — symmetric arena geometry for experiments

This dir generates the curated set of symmetric **arena geometry** files. See
`docs/experiments.md` for the full arena-format reference.

```
exp-configs/
  _generate_maps.py      # (re)generates the arena geometry (symmetry guaranteed by construction)
```

`_generate_maps.py` writes the arena **geometry** (walls/floor) to
`unity/Assets/StreamingAssets/Arenas/`, the single source of truth for that geometry. The curated
10-map **training rotation** — which of those arenas the trainer rotates over, and in what order —
lives in `src/pop_trainer/core/maps.py` (the `CURATED_ROTATION` constant). There are no game-config
wrappers in this dir.

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
(36, 60, 3) obs and cross-map model compatibility.) If you add a new arena to the
training rotation, also add its `Arenas/<name>.json` target to `CURATED_ROTATION` in
`src/pop_trainer/core/maps.py`.

## Rewards

The reward shaping lives in code, not here — see `src/pop_trainer/env/rewards.py`
(budget-based: full-episode totals spread per step). There are no reward-preset JSONs in this dir.

## Run an experiment

The previous `tank_twin.train` experiment commands were retired at M1 (the `tank_twin` package
was removed; see git history). A `pop_trainer` training entrypoint that consumes the curated
rotation supersedes them and is not yet wired.
