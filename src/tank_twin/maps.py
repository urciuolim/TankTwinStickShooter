"""Shared map-rotation resolver + constants for the trainer AND the evaluator.

This is the SINGLE SOURCE of the ``--maps`` / ``--map-rotation`` resolution contract. It
was factored out of :mod:`tank_twin.train` so :mod:`tank_twin.evaluate` can take the SAME
map interface (single ``--config``/``--map`` map OR a rotation) without duplicating the
resolver — one definition, two callers.

Deliberately torch-free / sb3-free / env-free: stdlib + :mod:`pathlib` only. The trainer
imports + RE-EXPORTS these names (``from tank_twin.maps import ...``) so the long-standing
``from tank_twin.train import DEFAULT_MAPS_DIR, ALL_MAPS_SENTINEL, _resolve_map_rotation``
import path (and ``tests/test_train_features.py``) keeps working byte-for-byte. The
evaluator imports the SAME names from here.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_MAPS_DIR", "ALL_MAPS_SENTINEL", "_resolve_map_rotation"]

# Repo root is two parents up from src/tank_twin/maps.py (mirrors train.py's _REPO_ROOT).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The shipped 10-map experiment configs (exp-configs/maps/*.json). When the user passes the
# rotation flag with no value (the sentinel below), the resolver returns these, sorted.
DEFAULT_MAPS_DIR = _REPO_ROOT / "exp-configs" / "maps"
# argparse const for the rotation flag used with no value -> rotate over DEFAULT_MAPS_DIR.
ALL_MAPS_SENTINEL = "__all__"


def _resolve_map_rotation(values: list[str] | None) -> list[Path] | None:
    """Resolve the ``--maps`` flag value into a list of map-config paths (or ``None``).

    Precedence / semantics (matches the CLI contract). With ``argparse`` ``nargs="*"`` the
    flag passed with NO value yields an empty list ``[]`` (the "all maps" sentinel); the flag
    ABSENT keeps the ``default=None``:

    * ``None`` (flag ABSENT) -> ``None``: NO rotation (today's single-map behavior).
    * ``[]`` or ``[ALL_MAPS_SENTINEL]`` (flag passed with no value) -> every ``*.json`` map
      config in :data:`DEFAULT_MAPS_DIR`, SORTED (the shipped 10 exp-configs maps).
    * a single entry that is a DIRECTORY -> every ``*.json`` map config in it, SORTED.
    * otherwise -> the explicit list of map-config paths, IN THE GIVEN ORDER (rotation order
      is the user's order; not re-sorted).

    Each returned entry is a map CONFIG path (with an ``arena_path``); ``TankEnv`` resolves it
    to its arena via ``_arena_path_from_config`` (single source). Returns ``None`` for the
    absent flag so the env stays in single-map mode.
    """
    if values is None:
        return None
    if values == [] or values == [ALL_MAPS_SENTINEL]:
        return sorted(DEFAULT_MAPS_DIR.glob("*.json"))
    if len(values) == 1 and Path(values[0]).is_dir():
        return sorted(Path(values[0]).glob("*.json"))
    return [Path(v) for v in values]
