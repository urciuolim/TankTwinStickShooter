"""``pop_trainer.models`` — reusable torch model DEFINITIONS (the ablation-ready encoders).

Definitions only — no training loops, no losses, no datasets (those live in ``pretraining``
and ``rl``). The ENCODER is the reusable artifact: a composition of a swappable TRUNK and a
swappable POOLING, built ABLATION-READY so the architecture is decided by experiment. Every
encoder satisfies one interface (``features`` -> the spatial map, ``embed`` -> the flat
embedding) and exports to a Unity-Sentis-clean ONNX graph.

This slice ships the encoders only. The supervised objective heads, the policy/value nets,
and a ``from_pretrained`` load path are DEFERRED to their consumers (``pretraining`` / ``rl``).

Boundary: torch is allowed here; nothing internal is imported (``core`` is not needed by the
encoders). Imports nothing from ``env`` / ``data`` / ``pretraining`` / ``rl``. No cycles.

Modules:

* :mod:`pop_trainer.models.encoders` — the trunks (:class:`NatureCNN`, :class:`ImpalaResNet`),
  the poolings (:class:`GlobalAveragePool`, :class:`Flatten`), the composed :class:`Encoder`,
  the :class:`EncoderConfig` + :func:`build_encoder` factory, and :func:`export_onnx`.
"""

from pop_trainer.models.encoders import (
    CANONICAL_HW,
    POOLINGS,
    TRUNKS,
    Encoder,
    EncoderConfig,
    Flatten,
    GlobalAveragePool,
    ImpalaResNet,
    NatureCNN,
    Pooling,
    Trunk,
    build_encoder,
    export_onnx,
)

__all__ = [
    "Trunk",
    "NatureCNN",
    "ImpalaResNet",
    "Pooling",
    "GlobalAveragePool",
    "Flatten",
    "Encoder",
    "EncoderConfig",
    "build_encoder",
    "export_onnx",
    "CANONICAL_HW",
    "TRUNKS",
    "POOLINGS",
]
