# models

Reusable **torch model definitions** — the ablation-ready vision encoders. Definitions only: no
training loops, no losses, no datasets (those belong to the future `pretraining` / `rl`). The
encoder is the single standardized vision module the whole stack shares, and it exports to a
Unity-Sentis-clean ONNX graph at deploy.

**Boundary:** torch is allowed here; **nothing internal is imported** (it doesn't even need
`core`). No cycles. It shares `core`'s leaf position as something the future consumers build on.

## Key classes / entry points

All in [`models.encoders`](../../src/pop_trainer/models/encoders.py):

- An **encoder = a swappable trunk × a swappable pooling**, so the architecture is decided by
  experiment.
- **Trunks** (fully-convolutional spatial-map producers, `(B, 3, H, W) → (B, C, h, w)`), each
  with an early downsampling stem to stay CPU-cheap on the 640×360 frame:
  [`NatureCNN`](../../src/pop_trainer/models/encoders.py) and
  [`ImpalaResNet`](../../src/pop_trainer/models/encoders.py) (whose `residual` /
  `blocks_per_stage` knobs are themselves ablation axes).
- **Poolings** (`(B, C, h, w) → (B, D)`):
  [`GlobalAveragePool`](../../src/pop_trainer/models/encoders.py) (param-cheap, position-lossy)
  and [`Flatten`](../../src/pop_trainer/models/encoders.py) (position-preserving, large).
- [`Encoder`](../../src/pop_trainer/models/encoders.py) — the composed module with one
  interface: `features(x) → (B, C, h, w)` (the spatial map, for supervised heads) and
  `embed(x) → (B, D)` (the flat embedding, the RL artifact). `forward` IS `embed`.
- [`EncoderConfig`](../../src/pop_trainer/models/encoders.py) +
  [`build_encoder`](../../src/pop_trainer/models/encoders.py) — the config-driven one-liner
  ablation factory (`TRUNKS` / `POOLINGS` registries).
- [`export_onnx`](../../src/pop_trainer/models/encoders.py) — writes a Sentis-clean ONNX graph
  of `embed`: NCHW input, fixed spatial axes (only batch dynamic), TorchScript exporter
  (`dynamo=False`), `opset_version=15`, no Scan/Loop/GroupNorm. The encoder IS the exported
  artifact. (Requires the `onnx` package, per `pyproject.toml`.)

## Pulls from (upstream)

Nothing internal. Just `torch` (a leaf alongside [core](core.md) in the dependency graph).

## Pushes to (downstream)

Nothing internal **yet** — the consumers are not built:

- (Future `pretraining` reads `Encoder.features` for the supervised-decode heads.)
- (Future `rl` reads `Encoder.embed` as the policy/value feature extractor.)
- Unity-Sentis runs the exported ONNX graph at deploy.

## Where it sits in the run

Off to the side of the live loop today: the encoder is the standardized vision backbone that
the (not-yet-built) pretraining and RL phases will share, and the deployable ONNX artifact. It
consumes the pixel frames that [env](env.md) produces and [data](data.md) records, but only once
those consumers exist.

---
[← back to index](../README.md)
