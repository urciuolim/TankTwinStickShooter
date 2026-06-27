"""pretraining/ — the single-frame decoder harness (supervised inverse-renderer).

Trains the reusable :mod:`pop_trainer.models` encoder by decoding the 52-float wire state from
one pixel frame via disposable per-group heads (the ENCODER is the artifact; the heads are
scaffolding). See the ``pop-pretraining`` skill for the locked contract.

Modules:

* :mod:`~pop_trainer.pretraining.targets` — pure per-group target extraction + TRAIN-fit
  normalization stats (numpy, torch-free).
* :mod:`~pop_trainer.pretraining.losses` — per-group torch losses + the combined / probe sums.
* :mod:`~pop_trainer.pretraining.metrics` — per-group world-unit / angular / F1 metrics.
* :mod:`~pop_trainer.pretraining.decoder` — the :class:`~...decoder.StateDecoder` (encoder +
  spatial heads + detached embed-probe).
* :mod:`~pop_trainer.pretraining.dataset` — the shard-streaming map-aware split loader.
* :mod:`~pop_trainer.pretraining.train` — the device-agnostic training CLI.
* :mod:`~pop_trainer.pretraining.profile` — the parameter-count + forward-latency profile.

Boundary: imports ``core`` / ``data`` / ``models`` + torch + numpy; nothing from ``env`` / ``rl``.
"""
