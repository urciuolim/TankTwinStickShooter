"""Dataset manifest: provenance + machine fingerprint for a collection run (PURE assembly).

A manifest is a single strict-JSON file written at the run root (next to the root ``maps.json``)
that makes a dataset identifiable and discrepancies explicable: datasets are NOT byte-reproducible
run-to-run (Unity physics + GPU rendering vary across machines), so the manifest records WHAT was
collected, WHICH commit/build produced it, and on WHAT machine.

:func:`build_manifest` is pure: it assembles AND validates the manifest dict from values the caller
INJECTS (no I/O, no env reads, no clock read — ``created_utc`` is passed in), normalizes the
pieces that JSON can't represent verbatim (pairing tuples -> 2-element lists; ``per_map_samples``
keys -> strings), and fails LOUDLY if a required field is missing or mistyped. The result is
strict-JSON-serializable (``json.dumps`` round-trips it). The live gathering of provenance/machine
values is the caller's untested CLI glue; this module only shapes + validates what it is given.

stdlib only.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MANIFEST_NAME", "MANIFEST_SCHEMA_VERSION", "build_manifest"]

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

_COLLECTION_KEYS = (
    "seed",
    "command",
    "workers",
    "episodes",
    "max_steps",
    "maps",
    "pairings",
    "total_shards",
    "total_samples",
    "per_map_samples",
)
_PROVENANCE_KEYS = ("git_commit", "git_dirty", "build")
_BUILD_KEYS = ("path", "mtime_utc", "size_bytes")
_MACHINE_KEYS = (
    "hostname",
    "platform",
    "system",
    "release",
    "arch",
    "processor",
    "cpu_count",
    "ram_total_gb",
    "python",
)


def _require_keys(name: str, mapping: Any, keys: tuple[str, ...]) -> None:
    """Raise ValueError unless ``mapping`` is a dict carrying every key in ``keys``."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a dict; got {type(mapping).__name__}")
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


def build_manifest(
    *,
    dataset: str,
    collection: dict,
    provenance: dict,
    machine: dict,
    created_utc: str,
) -> dict:
    """Assemble + validate the dataset manifest from INJECTED values (pure: no I/O, no clock).

    ``dataset`` is the run name; ``created_utc`` is an ISO-8601 UTC string the caller read from the
    clock. ``collection`` / ``provenance`` / ``machine`` are already-assembled sub-dicts whose
    required keys are validated here (a forgotten field fails loudly). ``pairings`` are normalized
    to 2-element lists and ``per_map_samples`` keys to strings so the result is strict-JSON-
    serializable with a stable round-trip. Returns the top-level manifest dict.
    """
    if not isinstance(dataset, str):
        raise ValueError(f"dataset must be a str; got {type(dataset).__name__}")
    if not isinstance(created_utc, str):
        raise ValueError(f"created_utc must be a str; got {type(created_utc).__name__}")

    _require_keys("collection", collection, _COLLECTION_KEYS)
    _require_keys("provenance", provenance, _PROVENANCE_KEYS)
    _require_keys("provenance.build", provenance["build"], _BUILD_KEYS)
    _require_keys("machine", machine, _MACHINE_KEYS)

    pairings = [list(pair) for pair in collection["pairings"]]
    per_map_samples = {str(k): int(v) for k, v in collection["per_map_samples"].items()}

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc,
        "dataset": dataset,
        "collection": {
            "seed": collection["seed"],
            "command": list(collection["command"]),
            "workers": collection["workers"],
            "episodes": collection["episodes"],
            "max_steps": collection["max_steps"],
            "maps": list(collection["maps"]),
            "pairings": pairings,
            "total_shards": collection["total_shards"],
            "total_samples": collection["total_samples"],
            "per_map_samples": per_map_samples,
        },
        "provenance": {
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "build": {
                "path": provenance["build"]["path"],
                "mtime_utc": provenance["build"]["mtime_utc"],
                "size_bytes": provenance["build"]["size_bytes"],
            },
        },
        "machine": {
            "hostname": machine["hostname"],
            "platform": machine["platform"],
            "system": machine["system"],
            "release": machine["release"],
            "arch": machine["arch"],
            "processor": machine["processor"],
            "cpu_count": machine["cpu_count"],
            "ram_total_gb": machine["ram_total_gb"],
            "python": machine["python"],
        },
    }
