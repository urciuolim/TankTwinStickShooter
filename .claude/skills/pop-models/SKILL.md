---
name: pop-models
description: Contract + boundaries for the models/ component of src/pop_trainer. Load when building or reviewing models/.
---

# Component: `models/` — reusable nn.Modules (encoders + heads + policy nets)

**Responsibility:** torch model DEFINITIONS (definitions only — training loops live in `pretraining/` and `rl/`). The ENCODER is the reusable artifact, built **composable and ABLATION-READY** so the architecture is decided by experiment, not assumption.

**The Encoder — composable + standardized:**
- One interface every encoder satisfies: `features(x) -> (B, C, h, w)` (the spatial feature map — read by the supervised heads + the JEPA substrate later) and `embed(x) -> (B, D)` (the **flat embedding** — the standardized apples-to-apples interface + the reusable RL artifact).
- An encoder = a **TRUNK × a POOLING**, both swappable; AND each trunk's defining structure is itself ablatable — the architecture choices are experiment variables, never fiats.
  - **Pooling** (spatial map → flat embedding): **GAP** (param-cheap, position-lossy) and **flatten** (raw, position-preserving, large). Which one is an ablation variable; the data decides on our position-critical task. Registry key in `POOLINGS`: `gap` / `flatten`.
  - **`CnnTrunk`** (registry key `cnn`) — a strided fully-conv stack + a downsampling stem for 640×360. Ablation axis: the pooling.
  - **`ResNetTrunk`** (registry key `resnet`) — `[16,32,32]` stages (conv → maxpool-stride-2 → residual blocks) — **broken into its own ablation study:** the residual structure is configurable — **residual connections on/off** (full residual vs a *plain conv stack of the same shape*) and **configurable residual-block depth** per stage — so resnet-plain vs resnet-residual (± depth) isolates whether the residual structure earns its keep.
  - **`GroupNormCNN`** (registry key `gn-cnn`) — a gently-strided CNN with GroupNorm + SiLU for SMALL frames (e.g. 64×64), where the `cnn` stride-4 stem would collapse the map. `cnn_depth` sets the base width (output channels are `8*cnn_depth`). GroupNorm is NOT Sentis-clean, so this trunk is an RL/small-frame variant, not a deploy target.
- The ablation grid is thus `{cnn, resnet-plain, resnet-residual, depth variants} × {flatten, GAP}`, every cell a trivial `EncoderConfig` instantiation; `gn-cnn` is the small-frame variant.

**Also (built when their consumers need them, not up front):** the supervised objective heads (heatmap/keypoint/wall — disposable scaffolding read off `features()`); the policy/value nets (read `embed()`); a `from_pretrained`-style encoder load path.

**Deployment (HARD — every encoder must satisfy):** ONNX / Unity-Sentis-exportable + CPU-cheap. Downsample BEFORE the trunk; NCHW; fixed spatial axes at export (batch-only dynamic); legacy `torch.onnx.export` (NOT dynamo); opset 15; NO Scan/Loop/GroupNorm/exotic ops. A test must export each encoder and assert the op-set is Sentis-clean.

**Boundaries:** imports `core` only (e.g. the state schema for head output shapes). torch allowed here (NOT in core). Imports nothing from `env / data / pretraining / rl`. No cycles.

**Inspiration (do NOT copy):** the flat/spatial/heatmap paradigms from the retired M1 `tank_twin` pixel-pretraining reference (in git history); Unity ml-agents `encoders.py` (the residual conv-stack precedent + the proven ONNX→Sentis export path).
