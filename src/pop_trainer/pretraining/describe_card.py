"""Pretty-print a trained checkpoint's model card.

``python -m pop_trainer.pretraining.describe_card <run-dir-or-model_card.json>`` loads the
strict-JSON ``model_card.json`` written next to a run's ``checkpoint.pt`` and prints a
human-readable rendering of its sections (architecture / hyperparameters / dataset / provenance /
machine / metrics / descriptions). Read-only: it never writes.

This is GLUE: stdlib only (``argparse`` / ``json`` / ``pathlib``) plus the pure ``card`` module for
the filename constant. No torch / numpy / models imports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pop_trainer.pretraining import card

__all__ = ["main"]


def _resolve_card_path(target: Path) -> Path:
    """Resolve ``target`` to a ``model_card.json`` path (accepts the file or its run dir)."""
    if target.is_dir():
        return target / card.CARD_NAME
    return target


def _print_section(title: str, section: dict) -> None:
    """Print one ``key: value`` block under a ``title`` header."""
    print(f"{title}:")
    for key, value in section.items():
        print(f"  {key}: {json.dumps(value, allow_nan=False)}")


def _print_descriptions(descriptions: list) -> None:
    """Print the card's descriptions, one per line (author / added_utc / text)."""
    print("descriptions:")
    if not descriptions:
        print("  (none)")
        return
    for entry in descriptions:
        author = entry.get("author", "")
        added_utc = entry.get("added_utc", "")
        text = entry.get("text", "")
        print(f"  {author}  {added_utc}  {text}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.pretraining.describe_card",
        description="Pretty-print a trained checkpoint's model card.",
    )
    parser.add_argument(
        "card_path",
        type=Path,
        help="model_card.json, or a run directory containing one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load a model card and print its sections. Returns ``0`` on success, ``2`` on error."""
    args = _parse_args(argv)
    card_path = _resolve_card_path(args.card_path)
    if not card_path.exists():
        print(f"error: model card not found at {card_path}")
        return 2

    with open(card_path, encoding="utf-8") as fh:
        doc = json.load(fh)

    print(f"name: {doc.get('name')}")
    print(f"schema_version: {doc.get('schema_version')}")
    print(f"created_utc: {doc.get('created_utc')}")
    for title in ("architecture", "hyperparameters", "dataset", "provenance", "machine"):
        section = doc.get(title)
        if isinstance(section, dict):
            _print_section(title, section)
    metrics = doc.get("metrics")
    if isinstance(metrics, dict):
        for split in ("val", "test"):
            print(f"metrics.{split}: {json.dumps(metrics.get(split), allow_nan=False)}")
    _print_descriptions(doc.get("descriptions") or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
