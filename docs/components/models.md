# models

Reusable **torch model definitions** — the ablation-ready vision encoders. Definitions only: no
training loops, no losses, no datasets (those belong to the [`rl`](rl.md) trainer and the future
`pretraining` phase, never here). The encoder is the single standardized vision module the whole
stack shares, and it exports to a Unity-Sentis-clean ONNX graph at deploy.

**Boundary:** torch is allowed here; **nothing internal is imported** (it doesn't even need
`core`). No cycles. It shares `core`'s leaf position as something its consumers build on.

## Key classes / entry points

All in [`models.encoders`](../../src/pop_trainer/models/encoders.py):

- An **encoder = a swappable trunk × a swappable pooling**, so the architecture is decided by
  experiment.
- **Trunks** (fully-convolutional spatial-map producers, `(B, 3, H, W) → (B, C, h, w)`). They are
  named by **architecture** (registry keys `cnn` / `resnet` / `gn-cnn`, `encoders.py:417-421`) — the
  old lineage names (nature / impala / dreamer) are gone:
  - [`CnnTrunk`](../../src/pop_trainer/models/encoders.py) (key `cnn`) — a strided conv stack with an
    early `4×4 stride-4` downsampling **stem** (crushes the 640×360 frame 4× before the body) so the
    conv stack + exported ONNX stay CPU-cheap.
  - [`ResNetTrunk`](../../src/pop_trainer/models/encoders.py) (key `resnet`) — a residual conv stack,
    same early stride-4 stem; its `residual` / `blocks_per_stage` knobs are themselves ablation axes.
  - [`GroupNormCNN`](../../src/pop_trainer/models/encoders.py) (key `gn-cnn`) — a **DreamerV3-style
    small-frame** trunk: four `Conv(4×4, stride 2) → GroupNorm(1) → SiLU` blocks whose channels
    DOUBLE from `cnn_depth` (`32 → 64 → 128 → 256`) while the spatial dims halve each block
    (`64 → 32 → 16 → 8 → 4`). It downsamples **gently** (no stride-4 stem), so it survives a SMALL
    frame (e.g. 64×64) where the `cnn` stem would collapse it. One-group `GroupNorm` (a channel-wise
    LayerNorm), NOT BatchNorm — BatchNorm's running stats are an on-policy-RL footgun
    (`encoders.py:245-290`). The [rl](rl.md) extractor auto-selects it for ≤128px frames; see
    [Resolution-based trunk selection](#resolution-based-trunk-selection).
- **Poolings** (`(B, C, h, w) → (B, D)`):
  [`GlobalAveragePool`](../../src/pop_trainer/models/encoders.py) (param-cheap, position-lossy)
  and [`Flatten`](../../src/pop_trainer/models/encoders.py) (position-preserving, large).
- [`Encoder`](../../src/pop_trainer/models/encoders.py) — the composed module with one
  interface: `features(x) → (B, C, h, w)` (the spatial map, for supervised heads) and
  `embed(x) → (B, D)` (the flat embedding, the RL artifact). `forward` IS `embed`.
- [`EncoderConfig`](../../src/pop_trainer/models/encoders.py) +
  [`build_encoder`](../../src/pop_trainer/models/encoders.py) — the config-driven one-liner
  ablation factory (`TRUNKS` / `POOLINGS` registries). Trunk-specific fields are passed ONLY to
  their trunk: the resnet-only `residual` / `blocks_per_stage` reach `ResNetTrunk` only for
  `trunk="resnet"`, and the gn-cnn-only `cnn_depth` reaches `GroupNormCNN` only for `trunk="gn-cnn"`
  — inert (harmless defaults) for any other config (`encoders.py:467-495`).
- [`export_onnx`](../../src/pop_trainer/models/encoders.py) — writes a Sentis-clean ONNX graph
  of `embed`: NCHW input, fixed spatial axes (only batch dynamic), TorchScript exporter
  (`dynamo=False`), `opset_version=15` (`encoders.py:501-540`). The `cnn` / `resnet` trunks export
  to a plain Conv/Relu/Add/MaxPool/BatchNorm op set (no Scan/Loop/GroupNorm; GAP → ReduceMean,
  Flatten → Reshape). The `gn-cnn` trunk uses `GroupNorm` + `SiLU`, which the deploy target must
  support — it is the SMALL-frame RL trunk, not the ONNX-deploy default. The encoder IS the exported
  artifact. (Requires the `onnx` package, per `pyproject.toml`.)

## Resolution-based trunk selection

The trunk used by [rl](rl.md) is chosen by the **observation resolution**, not hard-coded:
[`EncoderExtractor`](rl.md#the-encoder-seam) auto-selects `gn-cnn` for a SMALL frame (max spatial
dim `≤ SMALL_FRAME_MAX_DIM = 128`, e.g. 64×64) and `cnn` for a larger frame (the canonical 360×640),
both with `flatten` pooling. An explicit `--trunk {cnn,resnet,gn-cnn}` (the CTO-signed-off override)
replaces the size rule. The pooling / depth ablation lives in the `models` benchmark, NOT in the
extractor. See [rl → the encoder seam](rl.md#the-encoder-seam). The pixel resolution that drives
this selection is itself config-derived (`core.obs.frame_shape_from_config`) — see
[core](core.md#the-observation-resolution-contract-coreobs).

## Pulls from (upstream)

Nothing internal. Just `torch` (a leaf alongside [core](core.md) in the dependency graph).

## Pushes to (downstream)

One internal consumer is built; the other is still future:

- [rl](rl.md) — `EncoderExtractor` wraps `build_encoder`/`Encoder` and reads its `embed` flat
  embedding as the SB3 policy/value feature extractor (it owns the `state_dict` load + freeze; there
  is no `models.from_pretrained`). **Built** — the M1 trainer.
- (Future `pretraining` reads `Encoder.features` for the supervised-decode heads.)
- Unity-Sentis runs the exported ONNX graph at deploy.

## Where it sits in the run

Off to the side of the live loop today: the encoder is the standardized vision backbone that the
RL phase already shares (via [rl](rl.md)'s `EncoderExtractor`) and the future pretraining phase
will share too, plus the deployable ONNX artifact. It consumes the pixel frames that [env](env.md)
produces and [data](data.md) records.

---
[← back to index](../README.md)
