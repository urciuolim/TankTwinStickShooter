# models

Reusable **torch model definitions** — the ablation-ready vision encoders, and *nothing else*. The
encoder is the single standardized vision module the whole stack shares: the [rl](rl.md) policy reads
its flat embedding, the [pretraining](pretraining.md) heads read its spatial map (and *train* it), and
Unity-Sentis runs its exported ONNX graph at deploy. Definitions only — no training loops, no losses,
no datasets.

**Boundary:** `models` is a leaf — torch is allowed, but **nothing internal is imported** (it doesn't
even need [core](core.md)). It shares `core`'s dependency-free position; its consumers build on it. No
cycles ([`encoders.py:55-56`](../../src/pop_trainer/models/encoders.py)).

## Key classes / entry points

An **encoder = a swappable trunk × a swappable pooling**, so the architecture is decided by
experiment. All in [`models.encoders`](../../src/pop_trainer/models/encoders.py):

- **Trunks** (spatial-map producers `(B,3,H,W) → (B,C,h,w)`), named by architecture in
  [`TRUNKS`, `encoders.py:417-421`](../../src/pop_trainer/models/encoders.py) — the old lineage names
  (nature / impala / dreamer) are gone:
  - [`CnnTrunk`](../../src/pop_trainer/models/encoders.py) (`cnn`) — a strided conv stack whose early
    `4×4 stride-4` **stem** crushes the 640×360 frame 4× before the body, keeping the conv stack +
    exported ONNX CPU-cheap ([`encoders.py:122-153`](../../src/pop_trainer/models/encoders.py)).
  - [`ResNetTrunk`](../../src/pop_trainer/models/encoders.py) (`resnet`) — a residual conv stack, same
    stride-4 stem; its `residual` / `blocks_per_stage` knobs are themselves ablation axes
    ([`encoders.py:204-242`](../../src/pop_trainer/models/encoders.py)).
  - [`GroupNormCNN`](../../src/pop_trainer/models/encoders.py) (`gn-cnn`) — a **DreamerV3-style
    small-frame** trunk: four `Conv(4×4, stride 2) → GroupNorm(1) → SiLU` blocks whose channels DOUBLE
    from `cnn_depth` (`32 → 64 → 128 → 256`) while the spatial dims halve (`64 → 32 → 16 → 8 → 4`). It
    downsamples **gently** (no stride-4 stem), so it survives a SMALL frame (e.g. 64×64) where the
    `cnn` stem would collapse it. One-group `GroupNorm` is a channel-wise LayerNorm, NOT BatchNorm —
    BatchNorm's running stats are an on-policy-RL footgun
    ([`encoders.py:245-290`](../../src/pop_trainer/models/encoders.py)). The [rl](rl.md) extractor
    auto-selects it for ≤128px frames — see
    [Resolution-based trunk selection](#resolution-based-trunk-selection).
- **Poolings** (`(B,C,h,w) → (B,D)`):
  [`GlobalAveragePool`](../../src/pop_trainer/models/encoders.py) (param-cheap, position-lossy; plain
  `mean` → ReduceMean) and [`Flatten`](../../src/pop_trainer/models/encoders.py) (position-preserving,
  large; plain `reshape` → Reshape)
  ([`encoders.py:316-349`](../../src/pop_trainer/models/encoders.py)).
- [`Encoder`](../../src/pop_trainer/models/encoders.py) — the composed module, one interface:
  `features(x) → (B,C,h,w)` (spatial map, for supervised heads) and `embed(x) → (B,D)` (flat
  embedding, the RL artifact). `forward` IS `embed`, so the exported graph is the deployable forward
  ([`encoders.py:355-411`](../../src/pop_trainer/models/encoders.py)).
- [`EncoderConfig`](../../src/pop_trainer/models/encoders.py) +
  [`build_encoder`](../../src/pop_trainer/models/encoders.py) — the config-driven one-liner ablation
  factory. Each trunk-specific field is passed ONLY to its trunk: the resnet-only `residual` /
  `blocks_per_stage` reach `ResNetTrunk` only for `trunk="resnet"`, the gn-cnn-only `cnn_depth` reaches
  `GroupNormCNN` only for `trunk="gn-cnn"` — inert (harmless defaults) otherwise
  ([`encoders.py:425-464,467-495`](../../src/pop_trainer/models/encoders.py)).
- [`export_onnx`](../../src/pop_trainer/models/encoders.py) — writes a Sentis-clean ONNX graph of
  `embed`: NCHW input, fixed spatial axes (only batch dynamic), TorchScript exporter (`dynamo=False`),
  `opset_version=15`. The `cnn` / `resnet` trunks export to a plain Conv/Relu/Add/MaxPool/BatchNorm op
  set (no Scan/Loop/GroupNorm). The `gn-cnn` trunk uses `GroupNorm` + `SiLU`, which the deploy target
  must support — it is the SMALL-frame RL trunk, not the ONNX-deploy default. The encoder IS the
  exported artifact (requires the `onnx` package)
  ([`encoders.py:501-540`](../../src/pop_trainer/models/encoders.py)).

## Resolution-based trunk selection

The trunk used by [rl](rl.md) is chosen by the **observation resolution**, not hard-coded:
[`EncoderExtractor`](rl.md#encoderextractor--the-policyencoder-seam) auto-selects `gn-cnn` for a
SMALL frame (max spatial dim `≤ SMALL_FRAME_MAX_DIM = 128`, e.g. 64×64) and `cnn` for a larger frame
(the canonical 360×640), both with `flatten` pooling. An explicit `--trunk {cnn,resnet,gn-cnn}` (the
CTO-signed-off override) replaces the size rule
([`_resolve_trunk`, `extractor.py:44-55`](../../src/pop_trainer/rl/extractor.py)). The pixel
resolution that drives this is itself config-derived — see
[core → the observation-resolution contract](core.md#the-observation-resolution-contract-coreobs).

## Pulls from (upstream)

Nothing internal — just `torch` (a leaf alongside [core](core.md))
([`encoders.py:55-56`](../../src/pop_trainer/models/encoders.py)).

## Pushes to (downstream)

- [rl](rl.md) — `EncoderExtractor` wraps `build_encoder` / `Encoder` and reads its `embed` flat
  embedding as the SB3 policy/value feature extractor (it owns the `state_dict` load + freeze; there
  is no `models.from_pretrained`). **Built** — the M1 trainer
  ([`extractor.py:95-99`](../../src/pop_trainer/rl/extractor.py)).
- [pretraining](pretraining.md) — the single-frame decoder reads `Encoder.features` for its
  encoder-training spatial heads and `Encoder.embed` (detached) for the pooling probe, via
  `build_encoder` / `EncoderConfig`. **Built** — it is the first real consumer that *trains* this
  encoder.
- [utils](utils.md) — `model_info` names the active trunk by reaching the loaded policy's
  `.encoder.trunk.__class__.__name__` (by attribute, not import).
- Unity-Sentis runs the exported ONNX graph at deploy.

## Where it sits in the run

Off to the side of the live loop: the encoder is the standardized vision backbone shared on BOTH
arms — the RL phase (via [rl](rl.md)'s `EncoderExtractor`) and the offline pretraining phase (via
[pretraining](pretraining.md)'s `StateDecoder`, which is what *trains* it) — plus the deployable ONNX
artifact. It consumes the pixel frames that [env](env.md) produces and [data](data.md) records (the
pretraining harness streams that recorded corpus).

---
[← back to index](../README.md)
