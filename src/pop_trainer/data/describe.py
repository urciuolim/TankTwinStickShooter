"""Annotate a dataset's manifest with free-form descriptions (data-layer CLI GLUE).

``python -m pop_trainer.data.describe <dataset-dir> --text "..." [--author NAME]`` loads the run's
``manifest.json``, appends one ``{"author", "text", "added_utc"}`` annotation via the pure
:func:`pop_trainer.data.manifest.add_description`, and writes the manifest back ATOMICALLY (a
sibling ``*.tmp`` then ``Path.replace``, mirroring :func:`pop_trainer.data.shards.write_shard`).
The ``--list`` mode prints the existing descriptions instead of adding one.

``--author`` is free-form content provenance (``"human"`` by default, ``"claude"`` and any other
string allowed) — it describes WHO wrote the dataset annotation, NOT commit/PR attribution. A v1
manifest lacking a ``descriptions`` key is tolerated (``add_description`` treats it as empty).

This is GLUE: stdlib only (``json`` / ``pathlib`` / ``datetime`` / ``argparse``). The only pure
work it does is delegate to :func:`add_description`; everything else (argparse, file read/write, the
clock read for ``added_utc``) is side-effecting glue. No torch / numpy / models / env imports.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pop_trainer.data.manifest import MANIFEST_NAME, add_description, manifest_id

__all__ = ["main"]


def _load_manifest(dataset_dir: Path) -> dict:
    """Load ``<dataset-dir>/manifest.json`` (strict JSON)."""
    with open(dataset_dir / MANIFEST_NAME, encoding="utf-8") as fh:
        return json.load(fh)


def _write_manifest(dataset_dir: Path, manifest: dict) -> Path:
    """Atomically write ``manifest`` to ``<dataset-dir>/manifest.json`` (tmp -> replace, strict).

    Writes to a sibling ``*.tmp`` then ``Path.replace`` (atomic on the same filesystem) so a crash
    mid-write never leaves a half-written manifest. Returns the final path.
    """
    path = dataset_dir / MANIFEST_NAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    tmp.replace(path)
    return path


def _print_descriptions(manifest: dict) -> None:
    """Print the manifest's descriptions, one per line (author / added_utc / text)."""
    descriptions = manifest.get("descriptions") or []
    if not descriptions:
        print("no descriptions")
        return
    for entry in descriptions:
        author = entry.get("author", "")
        added_utc = entry.get("added_utc", "")
        text = entry.get("text", "")
        print(f"{author}  {added_utc}  {text}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.data.describe",
        description="Annotate a dataset's manifest with free-form descriptions.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="dataset run directory holding manifest.json",
    )
    parser.add_argument(
        "--text",
        default=None,
        metavar="TEXT",
        help="description text to append to the manifest",
    )
    parser.add_argument(
        "--author",
        default="human",
        help="annotation author (free string; default 'human', e.g. 'claude')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the manifest's existing descriptions instead of adding one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Add a description to (or list) a manifest. Returns ``0`` on success, ``2`` on error.

    ``--list`` prints the existing descriptions; otherwise ``--text`` is required and one annotation
    (author / text / ``datetime.now(UTC)``) is appended via :func:`add_description` and the manifest
    written back atomically.
    """
    args = _parse_args(argv)
    dataset_dir = args.dataset_dir

    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"error: manifest not found at {manifest_path}")
        return 2

    manifest = _load_manifest(dataset_dir)
    # Recompute the content fingerprint (never trust the stored field) and display it.
    print(f"manifest_id: {manifest_id(manifest)}")

    if args.list:
        _print_descriptions(manifest)
        return 0

    if args.text is None:
        print("error: --text is required (or use --list)")
        return 2

    updated = add_description(
        manifest,
        author=args.author,
        text=args.text,
        added_utc=datetime.now(UTC).isoformat(),
    )
    _write_manifest(dataset_dir, updated)
    print(f"added description by {args.author} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
