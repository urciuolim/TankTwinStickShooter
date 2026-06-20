"""Generate the symmetric experiment maps under exp-configs/maps/.

Every map is POINT-SYMMETRIC under (x, y) -> (-1 - x, -1 - y) (a 180 deg rotation
about the wall-grid centre (-0.5, -0.5)). That is the same symmetry the shipped
`custom1` arena has and the same transform the game uses to flip the two players'
spawns -- so neither side gets a wall advantage.

All maps share the SAME Walls dims (minX -10, maxX 9, minY -6, maxY 5) so the
observation stays (36, 60, 3) and a model trained on one map loads on any other.
Only the interior wall layout varies. The standard 2-cell-thick border is kept on
every map. Interior obstacles stay in the central columns, away from the tank
spawns near the left/right edges (world x ~= +/-7), and never wall off a column.

Run from anywhere:  uv run python exp-configs/_generate_maps.py
It (re)writes exp-configs/maps/Arenas/*.json and exp-configs/maps/*.json, copies
the 2021 custom1 arena verbatim, and asserts symmetry on all 10.
"""

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "exp-configs" / "maps"
ARENAS = MAPS / "Arenas"
CUSTOM1_SRC = REPO / "Assets" / "StreamingAssets" / "Arenas" / "custom1.json"

WALL_DIMS = {"minX": -10, "maxX": 9, "minY": -6, "maxY": 5}
FLOOR_DIMS = {"minX": -8, "maxX": 7, "minY": -4, "maxY": 3}
SOLID_COLS = (-10, -9, 8, 9)  # full-height border columns
BORDER_ROWS = (-6, -5, 4, 5)  # top/bottom border rows on every other column

CONFIG_TEMPLATE = {
    "connectionIP": "127.0.0.1",
    "connectionPort": 50000,
    "game_maxTime": 60,
    "timeScale": 5,
    "verbose": False,
    "arena_path": None,
    "player1_ai": True,
    "player1_keyboard": False,
    "player2_ai": True,
    "player2_keyboard": False,
    "ai_async": False,
    "ai_actionFreq": 10,
    "ai_fixedDeltaTime": 0.02,
    "player_maxHealth": 1,
    "player_randomStart": False,
    "player_x_spawn_lim": 0.5,
    "player_y_spawn_lim": 1.0,
}


def mirror(cell):
    x, y = cell
    return (-1 - x, -1 - y)


def border_cells():
    cells = set()
    for x in range(WALL_DIMS["minX"], WALL_DIMS["maxX"] + 1):
        if x in SOLID_COLS:
            for y in range(WALL_DIMS["minY"], WALL_DIMS["maxY"] + 1):
                cells.add((x, y))
        else:
            for y in BORDER_ROWS:
                cells.add((x, y))
    return cells


def symmetrize(seeds):
    """Union each seed with its (-1-x,-1-y) partner -> guaranteed symmetric."""
    out = set()
    for s in seeds:
        c = (int(s[0]), int(s[1]))
        out.add(c)
        out.add(mirror(c))
    return out


def assert_symmetric(cells):
    for c in cells:
        assert mirror(c) in cells, f"asymmetry: {c} has no partner {mirror(c)}"


def cells_to_block(cells, dims, tile_id):
    block = {"dims": dims, "tileID": tile_id}
    for x in range(dims["minX"], dims["maxX"] + 1):
        block[str(x)] = sorted(cy for (cx, cy) in cells if cx == x)
    return block


def floor_cells():
    return {
        (x, y)
        for x in range(FLOOR_DIMS["minX"], FLOOR_DIMS["maxX"] + 1)
        for y in range(FLOOR_DIMS["minY"], FLOOR_DIMS["maxY"] + 1)
    }


# Interior seeds per map (the generator mirrors each seed automatically).
# A centred plus, built so it is symmetric by construction:
CROSS = [(x, y) for x in (-1, 0) for y in range(-3, 3)] + [
    (x, y) for x in range(-3, 3) for y in (-1, 0)
]

MAP_SEEDS = {
    "empty": [],
    "center_block": [(-1, -1), (-1, 0), (0, -1), (0, 0)],
    "four_pillars": [(-4, 2), (3, 2)],
    "central_cross": CROSS,
    "opposing_l": [(-5, 2), (-5, 1), (-4, 2)],
    "diagonal_pillars": [(-4, -3), (-3, -2), (-2, -1), (-1, 0)],
    "chokepoint": [(-1, 2), (-1, 3), (0, 2), (0, 3)],
    "scattered": [(-5, 3), (-5, -1), (-3, 1)],
    "ring_fragments": [(-5, 2), (-4, 2), (-5, 1), (4, 2), (3, 2), (4, 1)],
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_config(name):
    cfg = dict(CONFIG_TEMPLATE)
    cfg["arena_path"] = f"Arenas/{name}.json"
    write_json(MAPS / f"{name}.json", cfg)


def free_space_connected(wall_cells):
    """Flood-fill the floor interior; every non-wall floor cell must be reachable."""
    floor = floor_cells()
    free = {c for c in floor if c not in wall_cells}
    if not free:
        return False
    start = next(iter(free))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in free and (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return seen == free


def main():
    ARENAS.mkdir(parents=True, exist_ok=True)
    floor_block = cells_to_block(floor_cells(), FLOOR_DIMS, 0)

    names = []

    # Map #1: the 2021 map, copied verbatim, then symmetry-checked.
    shutil.copyfile(CUSTOM1_SRC, ARENAS / "custom1_2021.json")
    c1 = json.loads((ARENAS / "custom1_2021.json").read_text(encoding="utf-8"))
    c1_walls = {
        (int(x), y) for x, ys in c1["Walls"].items() if x not in ("dims", "tileID") for y in ys
    }
    assert_symmetric(c1_walls)
    assert free_space_connected(c1_walls), "custom1_2021 floor not connected"
    write_config("custom1_2021")
    names.append("custom1_2021")

    # Maps #2-10: generated, symmetric by construction.
    for name, seeds in MAP_SEEDS.items():
        walls = border_cells() | symmetrize(seeds)
        assert_symmetric(walls)
        assert free_space_connected(walls), f"{name} floor not connected"
        arena = {"Floor": floor_block, "Walls": cells_to_block(walls, WALL_DIMS, 1)}
        write_json(ARENAS / f"{name}.json", arena)
        write_config(name)
        names.append(name)

    print(f"wrote {len(names)} maps (symmetric + connected): {', '.join(names)}")


if __name__ == "__main__":
    main()
