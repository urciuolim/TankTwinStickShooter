"""The map-rotation resolution contract: a ``--maps`` flag value -> list of map paths.

The SINGLE SOURCE of how a map-rotation request resolves to concrete map-config paths, so
the trainer and the evaluator share one definition. stdlib + :mod:`pathlib` only — no numpy,
no env, no torch.

Resolution semantics (matches the CLI contract; with ``argparse`` ``nargs="*"`` the flag
passed with NO value yields ``[]``, while an ABSENT flag keeps ``default=None``):

* ``None`` (flag ABSENT) -> ``None``: no rotation (single-map mode).
* ``[]`` or ``[ALL_MAPS_SENTINEL]`` (flag passed with no value) -> every ``*.json`` in
  :data:`DEFAULT_MAPS_DIR`, SORTED.
* a single entry that is a DIRECTORY -> every ``*.json`` in it, SORTED.
* otherwise -> the explicit list of paths, IN THE GIVEN ORDER (not re-sorted).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_MAPS_DIR", "ALL_MAPS_SENTINEL", "resolve_map_rotation"]

# This file lives at src/pop_trainer/core/maps.py, so the repo root is parents[3]:
#   parents[0] = .../core, [1] = .../pop_trainer, [2] = .../src, [3] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The shipped experiment map configs (exp-configs/maps/*.json). When the user passes the
# rotation flag with no value (the sentinel below), the resolver returns these, sorted.
DEFAULT_MAPS_DIR = _REPO_ROOT / "exp-configs" / "maps"

# argparse const for the rotation flag used with no value -> rotate over DEFAULT_MAPS_DIR.
ALL_MAPS_SENTINEL = "__all__"


def resolve_map_rotation(values: list[str] | None) -> list[Path] | None:
    """Resolve a ``--maps`` flag value into a list of map-config paths (or ``None``).

    See the module docstring for the precedence. Each returned entry is a map-config path
    (with an ``arena_path`` inside); ``None`` means single-map mode (no rotation).
    """
    if values is None:
        return None
    if values == [] or values == [ALL_MAPS_SENTINEL]:
        return sorted(DEFAULT_MAPS_DIR.glob("*.json"))
    if len(values) == 1 and Path(values[0]).is_dir():
        return sorted(Path(values[0]).glob("*.json"))
    return [Path(v) for v in values]
