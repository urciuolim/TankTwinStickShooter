# Arena (map) JSON format

Reference for the arena-config JSON the Unity build loads — the `"Floor"` / `"Walls"` geometry
every map under `unity/Assets/StreamingAssets/Arenas/` uses. It is the same `Walls` shape Unity's
[`WallMessage.Build`](../unity/Assets/Scripts/WallMessage.cs) emits and that
[`core.protocol.parse_walls_message`](../src/pop_trainer/core/protocol.py) parses into a `WallLayout`
(see the [core wall-message seam](components/core.md#the-wall-message-seam-walllayout--infomap)).

> The retired `tank_twin.train` experiment workflow that once lived on this page was dropped at M1
> (the `tank_twin` package was removed; the `pop_trainer` entrypoint supersedes it). Only the still-valid
> **arena-format reference** is kept — `exp-configs/README.md` links here for it.

## Format

An arena is one JSON object with a `"Floor"` block and a `"Walls"` block. Each block has:

- `"dims"`: `{minX, maxX, minY, maxY}` — the integer column/row bounds.
- `"tileID"`: which tile to draw (0 = floor, 1 = wall in the examples).
- One `"<x>": [y, y, ...]` list **per integer column** `x` in `[minX..maxX]` — the y-rows that hold a
  tile in that column.

Three things to keep straight:

- **`Walls` is geometry, NOT the observation.** Unity renders these walls and emits them once per map;
  `pop_trainer` parses them into a `WallLayout` surfaced as `info["map"]`. The observation is the
  **real rendered pixel frame `(360, 640, 3)`** ([env](components/env.md)), independent of arena dims —
  changing `Walls` dims changes the playfield, not the obs shape.
- **Spawns are NOT here.** The build reads spawn behavior from the **config** keys (`player_randomStart`,
  `player_x_spawn_lim`, `player_y_spawn_lim`, `player_maxHealth`), not the arena JSON.
- **Strict JSON only** — no trailing commas, no leading-dot floats (`0.5`, never `.5`). Unity's Newtonsoft
  tolerates loose JSON; Python's `json` does not, and the env will raise.

## Worked example: a tiny arena (border + one interior pillar)

A full outer wall ring plus a 2x2 pillar in the middle (columns -1 and 0, rows -1 and 0). The `-10` / `9`
columns are full vertical walls; every other column carries the top `-6` and bottom `5` border; columns
`-1` / `0` additionally carry the interior pillar at rows `-1, 0`. Copy verbatim into `Arenas/tiny.json`:

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

## Where the arenas + the rotation live

Two distinct things, in two distinct places:

- **Arena GEOMETRY** — `exp-configs/_generate_maps.py` (re)writes the 10 `Arenas/<name>.json` geometry
  files under `unity/Assets/StreamingAssets/Arenas/` (copies the 2021 `custom1` verbatim as
  `custom1_2021`, generates the other nine), asserting point-symmetry + floor connectivity on all 10
  (`_generate_maps.py:123-148`). Geometry only — no config-wrapper generation. Run from anywhere:
  `uv run python exp-configs/_generate_maps.py`.
- **The training ROTATION** — which arenas the rotation plays, and in what order, lives in
  [`core/maps.py`](../src/pop_trainer/core/maps.py) as the `CURATED_ROTATION` tuple (the single source of
  truth; `maps.py:41-52`), consumed via `core.maps.resolve_map_rotation` (see
  [data → rotation scheduler](components/data.md#the-rotation-scheduler--tag-from-echo)). The generator
  does NOT define the rotation.

---
[← back to index](README.md)
