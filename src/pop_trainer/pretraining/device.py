"""Device resolution: auto-detect cuda > mps > cpu, with no hardcoded device anywhere.

The harness must run on a 4090 (cuda) unchanged AND on this mac (mps) AND on a CI box (cpu).
``resolve_device`` maps the CLI ``--device`` to a concrete torch device string, defaulting to
the best available accelerator. torch only; nothing internal.
"""

from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(requested: str = "auto") -> str:
    """Resolve a ``--device`` request to a concrete torch device string.

    ``"auto"`` (the default) picks ``"cuda"`` if available, else ``"mps"`` if available, else
    ``"cpu"``. An explicit ``"cuda"`` / ``"mps"`` / ``"cpu"`` is returned verbatim (a request
    for an unavailable accelerator is left to torch to raise on use, so the choice is explicit).
    """
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req not in ("cuda", "mps", "cpu"):
        raise ValueError(f"--device must be auto/cuda/mps/cpu, got {requested!r}")
    return req
