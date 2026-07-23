"""Model card: provenance + architecture + metrics for a trained checkpoint (PURE assembly).

A model card is a single strict-JSON file written next to a run's ``checkpoint.pt`` that makes a
trained encoder identifiable and its metrics explicable: it records WHAT architecture +
hyperparameters produced the weights, WHICH commit + dataset (by recomputed ``manifest_id``) they
came from, on WHAT machine they were trained, and HOW they scored on val + test.

:func:`build_card` is pure: it assembles AND validates the card dict from values the caller INJECTS
(no I/O, no env reads, no clock read — ``created_utc`` is passed in), re-shapes each section into a
fresh dict carrying exactly its required keys, and fails LOUDLY if a required field is missing or
mistyped. The result is strict-JSON-serializable (``json.dumps(..., allow_nan=False)`` round-trips).

Non-finite policy lives in ONE place. The ``metrics`` section is the ONLY section where a non-finite
float (NaN / +-Inf) is tolerated: :func:`build_card` maps it to JSON ``null`` (a real ``0.0`` stays
``0.0``) so a degenerate metric never aborts a card write. EVERY OTHER section (architecture /
hyperparameters / dataset / provenance / machine) rejects a non-finite numeric leaf with a loud
``ValueError`` naming the offending section — a non-finite hyperparameter or param count is a real
bug, not a metric artifact.

The card carries a ``descriptions`` list of free-form human/agent annotations, each a
``{"author", "text", "added_utc"}`` entry. :func:`build_card` seeds the list (``None`` -> ``[]``);
:func:`add_description` appends one annotation, returning an updated COPY (it never mutates its
input or its nested list, so a card loaded off disk is safe to annotate).

stdlib only.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "CARD_NAME",
    "CARD_SCHEMA_VERSION",
    "build_card",
    "add_description",
]

CARD_NAME = "model_card.json"
CARD_SCHEMA_VERSION = 1

_ARCHITECTURE_KEYS = (
    "trunk",
    "pooling",
    "resolution",
    "input_hw",
    "trunk_class",
    "pooling_class",
    "decoder_class",
    "encoder_params",
    "head_params",
    "total_params",
    "embedding_dim",
)
_HYPERPARAMETER_KEYS = (
    "lr",
    "batch_size",
    "epochs",
    "seed",
    "heatmap_weight",
    "heatmap_sigma",
    "presence_pos_weight",
    "optimizer",
    "device",
)
_DATASET_KEYS = ("name", "manifest_id", "path")
_METRICS_KEYS = ("val", "test")
_PROVENANCE_KEYS = ("git_commit", "git_dirty", "weights_filename", "weights_sha256")
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


def _require_finite(section: Any, section_name: str) -> None:
    """Raise ValueError if any float leaf in ``section`` is non-finite (NaN / +-Inf).

    Recursively walks dicts/lists. Booleans and ints cannot be non-finite, so only ``float``
    leaves are checked. Used for every section EXCEPT ``metrics`` (whose non-finite policy is to
    nullify, not reject) — a non-finite leaf anywhere else signals a real bug.
    """
    if isinstance(section, dict):
        for k, v in section.items():
            _require_finite(v, f"{section_name}.{k}")
    elif isinstance(section, list | tuple):
        for i, v in enumerate(section):
            _require_finite(v, f"{section_name}[{i}]")
    elif isinstance(section, float) and not math.isfinite(section):
        raise ValueError(
            f"{section_name} is non-finite ({section}); only metrics may be non-finite"
        )


def _nullify_nonfinite(obj: Any) -> Any:
    """Return a copy of ``obj`` with every non-finite float leaf mapped to ``None`` (JSON null).

    Recursively walks dicts/lists. A finite float (including a real ``0.0``) is preserved exactly;
    only NaN / +-Inf become ``None``. This is the SINGLE null-mapping policy point for the card —
    applied to the ``metrics`` section only.
    """
    if isinstance(obj, dict):
        return {k: _nullify_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_nullify_nonfinite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def build_card(
    *,
    name: str,
    created_utc: str,
    architecture: dict,
    hyperparameters: dict,
    dataset: dict,
    metrics: dict,
    provenance: dict,
    machine: dict,
    descriptions: list[dict] | None = None,
) -> dict:
    """Assemble + validate the model card from INJECTED values (pure: no I/O, no clock).

    ``name`` is the run/model name; ``created_utc`` is an ISO-8601 UTC string the caller read from
    the clock. The six section dicts are already-assembled sub-dicts whose required keys are
    validated here (a forgotten field fails loudly). Each section is re-shaped into a fresh dict
    carrying EXACTLY its required keys so the output shape is pinned.

    Non-finite policy: the ``metrics`` section is nullified (non-finite float -> ``None``); every
    other section rejects a non-finite numeric leaf with a loud ``ValueError``.
    ``dataset.manifest_id`` and ``provenance.git_commit`` / ``git_dirty`` are nullable (``None`` is
    accepted). ``descriptions`` is the (optional) list of ``{"author", "text", "added_utc"}``
    entries — ``None`` -> ``[]`` — copied to a fresh list. The result is strict-JSON-serializable.
    """
    if not isinstance(name, str):
        raise ValueError(f"name must be a str; got {type(name).__name__}")
    if not isinstance(created_utc, str):
        raise ValueError(f"created_utc must be a str; got {type(created_utc).__name__}")

    _require_keys("architecture", architecture, _ARCHITECTURE_KEYS)
    _require_keys("hyperparameters", hyperparameters, _HYPERPARAMETER_KEYS)
    _require_keys("dataset", dataset, _DATASET_KEYS)
    _require_keys("metrics", metrics, _METRICS_KEYS)
    _require_keys("provenance", provenance, _PROVENANCE_KEYS)
    _require_keys("machine", machine, _MACHINE_KEYS)

    # Non-finite policy enforcement BEFORE re-shaping: reject a non-finite leaf in any section other
    # than metrics (a clean ValueError, ahead of any int()/float() cast that would raise opaquely).
    _require_finite(architecture, "architecture")
    _require_finite(hyperparameters, "hyperparameters")
    _require_finite(dataset, "dataset")
    _require_finite(provenance, "provenance")
    _require_finite(machine, "machine")

    architecture_block = {
        "trunk": architecture["trunk"],
        "pooling": architecture["pooling"],
        "resolution": int(architecture["resolution"]),
        "input_hw": list(architecture["input_hw"]),
        "trunk_class": architecture["trunk_class"],
        "pooling_class": architecture["pooling_class"],
        "decoder_class": architecture["decoder_class"],
        "encoder_params": int(architecture["encoder_params"]),
        "head_params": int(architecture["head_params"]),
        "total_params": int(architecture["total_params"]),
        "embedding_dim": architecture["embedding_dim"],
    }
    hyperparameters_block = {
        "lr": float(hyperparameters["lr"]),
        "batch_size": int(hyperparameters["batch_size"]),
        "epochs": int(hyperparameters["epochs"]),
        "seed": int(hyperparameters["seed"]),
        "heatmap_weight": float(hyperparameters["heatmap_weight"]),
        "heatmap_sigma": float(hyperparameters["heatmap_sigma"]),
        "presence_pos_weight": float(hyperparameters["presence_pos_weight"]),
        "optimizer": hyperparameters["optimizer"],
        "device": hyperparameters["device"],
    }
    dataset_block = {
        "name": dataset["name"],
        "manifest_id": dataset["manifest_id"],
        "path": dataset["path"],
    }
    provenance_block = {
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "weights_filename": provenance["weights_filename"],
        "weights_sha256": provenance["weights_sha256"],
    }
    machine_block = {
        "hostname": machine["hostname"],
        "platform": machine["platform"],
        "system": machine["system"],
        "release": machine["release"],
        "arch": machine["arch"],
        "processor": machine["processor"],
        "cpu_count": int(machine["cpu_count"]),
        "ram_total_gb": float(machine["ram_total_gb"]),
        "python": machine["python"],
    }

    # metrics is the ONLY section where a non-finite float is tolerated: map it to JSON null.
    metrics_block = {
        "val": _nullify_nonfinite(metrics["val"]),
        "test": _nullify_nonfinite(metrics["test"]),
    }

    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "created_utc": created_utc,
        "name": name,
        "descriptions": [] if descriptions is None else list(descriptions),
        "architecture": architecture_block,
        "hyperparameters": hyperparameters_block,
        "dataset": dataset_block,
        "metrics": metrics_block,
        "provenance": provenance_block,
        "machine": machine_block,
    }


def add_description(card: dict, *, author: str, text: str, added_utc: str) -> dict:
    """Append one annotation to ``card`` and return an updated COPY (pure: no I/O, no clock).

    Builds a ``{"author": author, "text": text, "added_utc": added_utc}`` entry and appends it to a
    FRESH copy of the card's ``descriptions`` list, leaving the input dict AND its nested list
    untouched. A card without a ``descriptions`` key is treated as having an empty list.
    ``added_utc`` is the ISO-8601 UTC string the caller read from the clock.
    """
    existing = card.get("descriptions") or []
    entry = {"author": author, "text": text, "added_utc": added_utc}
    return {**card, "descriptions": [*existing, entry]}
