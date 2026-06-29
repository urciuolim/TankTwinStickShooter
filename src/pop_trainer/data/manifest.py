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

The manifest carries a ``descriptions`` list of free-form human/agent annotations, each a
``{"author": str, "text": str, "added_utc": str}`` entry. :func:`build_manifest` seeds the list
(``None`` -> ``[]``); :func:`add_description` appends one annotation, returning an updated COPY
(it never mutates its input, so a manifest loaded off disk is safe to annotate). A v1 manifest
lacking the ``descriptions`` key is treated as having an empty list, so older datasets remain
annotatable.

stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "add_description",
    "manifest_id",
]

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 3

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


def manifest_id(manifest: dict) -> str:
    """Return a stable content fingerprint (bare 64-char lowercase sha256 hex) of ``manifest``.

    The id is computed over the manifest's CANONICAL content — every top-level key EXCEPT
    ``descriptions`` (free-form human/agent annotations that must not change identity) and
    ``manifest_id`` itself (the embedded value must not feed its own hash). The remaining content is
    serialized with strict, sorted, separator-pinned JSON (``sort_keys=True``, no whitespace,
    ``ensure_ascii=True``, ``allow_nan=False``) so the hash is insertion-order- and platform-
    independent. Pairs cleanly over v1/v2 manifests lacking a ``manifest_id`` key (the filter
    handles absence). Callers ALWAYS recompute via this function and never trust any stored field.
    """
    content = {k: v for k, v in manifest.items() if k not in ("descriptions", "manifest_id")}
    payload = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    descriptions: list[dict] | None = None,
) -> dict:
    """Assemble + validate the dataset manifest from INJECTED values (pure: no I/O, no clock).

    ``dataset`` is the run name; ``created_utc`` is an ISO-8601 UTC string the caller read from the
    clock. ``collection`` / ``provenance`` / ``machine`` are already-assembled sub-dicts whose
    required keys are validated here (a forgotten field fails loudly). ``pairings`` are normalized
    to 2-element lists and ``per_map_samples`` keys to strings so the result is strict-JSON-
    serializable with a stable round-trip. ``descriptions`` is the (optional) list of caller-shaped
    ``{"author", "text", "added_utc"}`` annotation entries — ``None`` -> ``[]`` — normalized to a
    fresh list so the result stays strict-JSON-serializable. Returns the top-level manifest dict.
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

    # Type-pin hashed numeric fields so e.g. 32 and 32.0 fingerprint identically (the JSON
    # serializer renders int 32 as "32" but float 32.0 as "32.0"). Nullable git fields and
    # string fields are left untouched.
    result = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc,
        "dataset": dataset,
        "descriptions": [] if descriptions is None else list(descriptions),
        "collection": {
            "seed": int(collection["seed"]),
            "command": list(collection["command"]),
            "workers": int(collection["workers"]),
            "episodes": int(collection["episodes"]),
            "max_steps": int(collection["max_steps"]),
            "maps": list(collection["maps"]),
            "pairings": pairings,
            "total_shards": int(collection["total_shards"]),
            "total_samples": int(collection["total_samples"]),
            "per_map_samples": per_map_samples,
        },
        "provenance": {
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "build": {
                "path": provenance["build"]["path"],
                "mtime_utc": provenance["build"]["mtime_utc"],
                "size_bytes": int(provenance["build"]["size_bytes"]),
            },
        },
        "machine": {
            "hostname": machine["hostname"],
            "platform": machine["platform"],
            "system": machine["system"],
            "release": machine["release"],
            "arch": machine["arch"],
            "processor": machine["processor"],
            "cpu_count": int(machine["cpu_count"]),
            "ram_total_gb": float(machine["ram_total_gb"]),
            "python": machine["python"],
        },
    }
    # Embed the content fingerprint as a top-level key. ``manifest_id`` excludes both
    # ``descriptions`` and ``manifest_id`` itself, so computing over the 7-key dict here equals
    # computing over the final 8-key dict (self-consistent). The embedded value is informational;
    # callers ALWAYS recompute via manifest_id() and never trust the stored field.
    result["manifest_id"] = manifest_id(result)
    return result


def add_description(manifest: dict, *, author: str, text: str, added_utc: str) -> dict:
    """Append one annotation to ``manifest`` and return an updated COPY (pure: no I/O, no clock).

    Builds a ``{"author": author, "text": text, "added_utc": added_utc}`` entry and appends it to a
    FRESH copy of the manifest's ``descriptions`` list, leaving the input dict AND its nested list
    untouched. A manifest without a ``descriptions`` key (a v1 manifest) is treated as having an
    empty list, so older datasets remain annotatable. ``added_utc`` is the ISO-8601 UTC string the
    caller read from the clock.
    """
    existing = manifest.get("descriptions") or []
    entry = {"author": author, "text": text, "added_utc": added_utc}
    return {**manifest, "descriptions": [*existing, entry]}
