"""Device resolution: auto-detect cuda > mps > cpu, with no hardcoded device anywhere.

The harness must run on a 4090 (cuda) unchanged AND on this mac (mps) AND on a CI box (cpu).
``resolve_device`` maps the CLI ``--device`` to a concrete torch device string, defaulting to
the best available accelerator. torch + os only; nothing internal.

The ``TT_DEVICE`` env var lets ``scripts/gpu_run.py`` steer device selection for the default
(``"auto"``) path -- e.g. it sets ``TT_DEVICE=cuda`` where it dispatches the job -- so training,
profiling, AND ad-hoc diagnostics land on the GPU instead of silently defaulting to MPS. An
explicit ``--device`` always wins over the env (the escape hatch).
"""

from __future__ import annotations

import os

import torch

__all__ = ["resolve_device"]


def resolve_device(requested: str = "auto") -> str:
    """Resolve a ``--device`` request to a concrete torch device string.

    ``"auto"`` (the default) honors the ``TT_DEVICE`` env var if it names a concrete device,
    otherwise picks ``"cuda"`` if available, else ``"mps"`` if available, else ``"cpu"``. An
    explicit ``"cuda"`` / ``"mps"`` / ``"cpu"`` is returned verbatim (a request for an
    unavailable accelerator is left to torch to raise on use, so the choice is explicit).
    """
    req = requested.strip().lower()
    if req == "auto":
        env = os.environ.get("TT_DEVICE", "").strip().lower()
        if env in ("cuda", "mps", "cpu"):
            return env
        if env and env != "auto":
            raise ValueError(f"TT_DEVICE must be auto/cuda/mps/cpu, got {env!r}")
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req not in ("cuda", "mps", "cpu"):
        raise ValueError(f"--device must be auto/cuda/mps/cpu, got {requested!r}")
    return req
