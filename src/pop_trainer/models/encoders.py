"""Composable, ablation-ready vision encoders (the reusable RL/pretraining artifact).

An ENCODER is the single standardized vision module the whole stack shares: the supervised
pretraining heads read its spatial feature map, the RL policy/value nets read its flat
embedding, and Unity-Sentis runs its exported ONNX graph at deploy. An encoder is a
composition of two independently swappable pieces so the architecture is decided by
experiment:

* a **trunk** (a :class:`Trunk`) — a fully-convolutional spatial-map producer that turns an
  NCHW RGB frame into a ``(B, C, h, w)`` feature map. Three are provided: :class:`CnnTrunk`
  (a strided conv stack) and :class:`ResNetTrunk` (a residual conv stack), both of which
  downsample the input EARLY via a stem BEFORE the main body so the conv stack and the
  exported ONNX graph stay CPU-cheap on the 640x360 (HxW = 360x640) canonical frame; and
  :class:`GroupNormCNN` (a gently-strided GroupNorm + SiLU stack) for SMALL frames.
* a **pooling** (a :class:`Pooling`) — collapses the trunk's spatial map into the flat
  ``(B, D)`` embedding. Two are provided: :class:`GlobalAveragePool` (GAP — param-cheap but
  position-lossy) and :class:`Flatten` (position-preserving but large).

The ResNet trunk's residual structure is itself an ablation axis, driven by the config:

* ``residual=True`` keeps the skip connection (``x + conv(relu(conv(relu(x))))``);
  ``residual=False`` is the SAME conv stack — identical channel widths, conv-layer count, and
  stride-2 maxpool stages — with the skip add removed, isolating whether the skip earns its
  keep.
* ``blocks_per_stage`` sets the per-stage block depth, so depth variants are one-liners too.

Every encoder satisfies ONE interface (:class:`Encoder`):

* ``features(x) -> (B, C, h, w)`` — the trunk's spatial map (read by the supervised heads /
  the JEPA substrate later).
* ``embed(x) -> (B, D)`` — the flat embedding == ``pool(features(x))`` (the apples-to-apples
  interface + the reusable RL artifact). ``forward`` is ``embed`` so the EXPORTED ONNX graph
  is the deployable ``(B, D)`` forward.

Composition is config-driven: an :class:`EncoderConfig` names a trunk and a pooling (plus the
resnet-only residual knobs), and :func:`build_encoder` assembles them, so each ablation cell
is a one-liner::

    build_encoder(EncoderConfig(trunk="cnn", pooling="flatten"))  # "just the CNN trunk"
    build_encoder(EncoderConfig(trunk="cnn", pooling="gap"))
    build_encoder(EncoderConfig(trunk="resnet", pooling="gap"))                   # residual
    build_encoder(EncoderConfig(trunk="resnet", pooling="gap", residual=False))   # plain
    build_encoder(EncoderConfig(trunk="resnet", pooling="gap", blocks_per_stage=3))  # deeper

The resnet-only fields (``residual``, ``blocks_per_stage``) are consumed ONLY when
``trunk == "resnet"``; for ``trunk == "cnn"`` they are inert and the :class:`CnnTrunk` trunk
never sees them.

Deployment (the hard contract): :func:`export_onnx` writes a Unity-Sentis-clean ONNX graph
of ``embed`` — NCHW input, FIXED spatial axes (only the batch axis is dynamic), the
TorchScript exporter (``dynamo=False``), ``opset_version=15``, and an op set free of
Scan/Loop/GroupNorm/exotic ops (GAP is a plain ``mean``, flatten is a plain ``reshape``,
norm is BatchNorm). The encoder IS the exported artifact.

Boundary: torch is allowed here; nothing internal is imported. Imports nothing from ``env`` /
``data`` / ``pretraining`` / ``rl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "Trunk",
    "CnnTrunk",
    "ResNetTrunk",
    "GroupNormCNN",
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

# Canonical rendered-frame size as (H, W): 360 rows x 640 cols (the real Unity render the env
# captures). Used as the default export spatial axes; encoders are size-agnostic at build time.
CANONICAL_HW = (360, 640)

TrunkName = Literal["cnn", "resnet", "gn-cnn"]
PoolingName = Literal["gap", "flatten"]


# --- trunks (spatial-map producers; fully convolutional) -------------------------------


class Trunk(nn.Module):
    """A fully-convolutional spatial-map producer: ``(B, 3, H, W) -> (B, C, h, w)``.

    Subclasses set :attr:`out_channels` (the ``C`` of the produced map) and :attr:`in_channels`
    (the ``C`` consumed) at construction so a pooling can size its embedding and the export
    helper can size its example input without a forward pass. ``forward`` returns the spatial
    feature map; every trunk downsamples EARLY (a stem before the main body) to keep the conv
    stack and the exported ONNX graph CPU-cheap.
    """

    in_channels: int
    out_channels: int

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    """A ``Conv2d -> BatchNorm2d -> ReLU`` block (BatchNorm keeps the exported graph clean)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CnnTrunk(Trunk):
    """A strided fully-conv trunk with an early downsampling STEM.

    A stem ``Conv2d(3 -> stem_channels, 4x4, stride 4)`` crushes the 640x360 frame by 4x in
    each spatial dim BEFORE the main 3-conv stack (the "downsample before the trunk" deployment
    rule). The main stack is the strided conv trio (8x8 s4 -> 4x4 s2 -> 3x3 s1), each
    ``Conv2d -> BatchNorm2d -> ReLU``, producing a ``(B, out_channels, h, w)`` map. No
    flatten/linear here — that is the pooling's job, kept separate so any pooling composes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        channels: tuple[int, int, int] = (32, 64, 64),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.in_channels = in_channels
        # Stem: 4x4 stride-4 downsample (640x360 -> 160x90) before the trunk.
        self.stem = _conv_bn_relu(in_channels, stem_channels, kernel=4, stride=4, padding=0)
        # Strided conv trio, fully convolutional (no flatten).
        self.conv = nn.Sequential(
            _conv_bn_relu(stem_channels, c1, kernel=8, stride=4, padding=0),
            _conv_bn_relu(c1, c2, kernel=4, stride=2, padding=0),
            _conv_bn_relu(c2, c3, kernel=3, stride=1, padding=0),
        )
        self.out_channels = c3

    def forward(self, x: Tensor) -> Tensor:
        """Return the spatial feature map ``(B, out_channels, h, w)`` for NCHW ``x``."""
        return self.conv(self.stem(x))


class _ResNetBlock(nn.Module):
    """ReLU-pre-activation residual-style block, residual or plain.

    Two 3x3 stride-1 same-padding convs with a pre-activation ReLU before each; channels are
    preserved. When ``residual`` is true the input is added back
    (``x + conv(relu(conv(relu(x))))``); when false the same conv stack runs WITHOUT the add
    (``conv(relu(conv(relu(x))))``), so the two modes share every conv layer and differ only by
    the skip. No norm in the branch, so the exported graph stays plain Conv/Relu (+ Add only in
    the residual mode).
    """

    def __init__(self, channels: int, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv0(self.relu(x))
        h = self.conv1(self.relu(h))
        if self.residual:
            return x + h
        return h


class _ResNetStage(nn.Module):
    """One ResNet stage: ``conv -> maxpool(stride 2) -> N blocks``.

    The 3x3 same-padding conv projects to the stage's channel width, a 3x3 stride-2 maxpool
    halves the spatial dims, then ``num_blocks`` blocks refine at that resolution. ``residual``
    selects skip-connected vs plain blocks (the same convs either way).
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int = 2, residual: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            *(_ResNetBlock(out_channels, residual=residual) for _ in range(num_blocks))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.pool(self.conv(x)))


class ResNetTrunk(Trunk):
    """A residual conv trunk with an early downsampling STEM.

    A stem ``Conv2d(3 -> stem_channels, 4x4, stride 4)`` crushes the 640x360 frame by 4x first
    (the deployment "downsample before the trunk" rule), then a sequence of stages with channel
    widths ``[16, 32, 32]``. Each stage is ``conv -> maxpool(stride 2) -> blocks_per_stage
    blocks`` (a block = ReLU-pre-activation conv pairs). A final ReLU caps the trunk. Produces
    a ``(B, out_channels, h, w)`` map; pooling stays external.

    ``residual`` selects the block type: ``True`` adds the input back (skip connection),
    ``False`` is the same conv stack without the add. ``blocks_per_stage`` sets the per-stage
    block depth. Both leave the channel widths and stage count unchanged, so they isolate the
    residual structure as an ablation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 16,
        stage_channels: tuple[int, ...] = (16, 32, 32),
        blocks_per_stage: int = 2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Stem: 4x4 stride-4 downsample before any stage (640x360 -> 160x90).
        self.stem = nn.Conv2d(in_channels, stem_channels, kernel_size=4, stride=4, padding=0)
        stages: list[nn.Module] = []
        prev = stem_channels
        for width in stage_channels:
            stages.append(_ResNetStage(prev, width, num_blocks=blocks_per_stage, residual=residual))
            prev = width
        self.stages = nn.Sequential(*stages)
        self.out_relu = nn.ReLU(inplace=True)
        self.out_channels = prev

    def forward(self, x: Tensor) -> Tensor:
        """Return the spatial feature map ``(B, out_channels, h, w)`` for NCHW ``x``."""
        return self.out_relu(self.stages(self.stem(x)))


class GroupNormCNN(Trunk):
    """A strided GroupNorm + SiLU conv trunk that survives SMALL frames (e.g. 64x64).

    Where :class:`CnnTrunk` / :class:`ResNetTrunk` crush a 640x360 frame with a stride-4 stem
    before the body, this trunk takes an already-small frame and downsamples gently: a stack of
    **four** ``Conv2d(kernel=4, stride=2, padding=1) -> norm -> SiLU`` blocks whose channels
    DOUBLE from ``cnn_depth``. At ``cnn_depth=32`` that is ``32 -> 64 -> 128 -> 256`` while the
    spatial dims halve each block (``64 -> 32 -> 16 -> 8 -> 4``), producing the
    ``(B, 8*cnn_depth, h, w)`` map (``(B, 256, 4, 4)`` for a 64x64 input). No flatten/linear here
    — the FC to a fixed ``features_dim`` is the pooling+consumer's job, kept external so any
    pooling composes (with :class:`Flatten` the flat embedding is ``8*cnn_depth*h*w``; the SB3
    policy head supplies the FC).

    The norm is :class:`~torch.nn.GroupNorm` with one group (a channel-wise LayerNorm over the
    spatial map), NOT BatchNorm: BatchNorm's running stats are an on-policy-RL footgun. Activation
    is :class:`~torch.nn.SiLU`. ``cnn_depth`` sets the base width; the widths here are a tunable
    default, not a fixed constant.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_depth: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        blocks: list[nn.Module] = []
        prev = in_channels
        width = cnn_depth
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev, width, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(1, width),  # one-group GroupNorm == channel-wise LayerNorm
                    nn.SiLU(inplace=True),
                )
            )
            prev = width
            width *= 2
        self.conv = nn.Sequential(*blocks)
        self.out_channels = prev

    def forward(self, x: Tensor) -> Tensor:
        """Return the spatial feature map ``(B, out_channels, h, w)`` for NCHW ``x``."""
        return self.conv(x)


# --- pooling (spatial map -> flat embedding; swappable ablation variable) ---------------


class Pooling(nn.Module):
    """A spatial-map -> flat-embedding collapse: ``(B, C, h, w) -> (B, D)``.

    Implements :meth:`embedding_dim` so an encoder can report its ``D`` from the trunk's
    ``out_channels`` (and, for flatten, a known spatial ``(h, w)``) without a forward pass.
    """

    def embedding_dim(self, channels: int, spatial_hw: tuple[int, int] | None) -> int | None:
        """Return the embedding ``D`` for a trunk of ``channels`` (and ``spatial_hw`` if known).

        Returns ``None`` when ``D`` cannot be known statically (flatten without a known
        spatial size). ``spatial_hw`` is the trunk's output ``(h, w)`` when the input size is
        fixed; pass ``None`` if unknown.
        """
        raise NotImplementedError  # pragma: no cover - overridden

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


class GlobalAveragePool(Pooling):
    """Global average pool over the spatial dims: ``(B, C, h, w) -> (B, C)``.

    Implemented as ``x.mean(dim=(2, 3))`` (NOT adaptive pooling) so the exported ONNX op is a
    plain ReduceMean — Sentis-clean and opset-15 safe. Param-cheap and size-agnostic, but
    discards position; an ablation variable, not a default.
    """

    def embedding_dim(self, channels: int, spatial_hw: tuple[int, int] | None) -> int | None:
        return channels

    def forward(self, x: Tensor) -> Tensor:
        """Return the channel-wise spatial mean ``(B, C)`` of NCHW map ``x``."""
        return x.mean(dim=(2, 3))


class Flatten(Pooling):
    """Flatten the spatial map: ``(B, C, h, w) -> (B, C*h*w)`` (position-preserving).

    Implemented as ``x.reshape(B, -1)`` (a plain Reshape/Flatten op, not adaptive pooling) so
    the exported graph stays Sentis-clean. Position-preserving but large — the apt baseline on
    our position-critical task; an ablation variable, not a default. ``D`` is only known when
    the trunk's output spatial size is known.
    """

    def embedding_dim(self, channels: int, spatial_hw: tuple[int, int] | None) -> int | None:
        if spatial_hw is None:
            return None
        h, w = spatial_hw
        return channels * h * w

    def forward(self, x: Tensor) -> Tensor:
        """Return the flattened map ``(B, C*h*w)`` of NCHW map ``x``."""
        return x.reshape(x.shape[0], -1)


# --- encoder (trunk x pooling) ----------------------------------------------------------


class Encoder(nn.Module):
    """A vision encoder == a :class:`Trunk` composed with a :class:`Pooling`.

    The single standardized interface every architecture variant satisfies:

    * :meth:`features` -> ``(B, C, h, w)`` — the trunk's spatial map (for the supervised heads
      / JEPA substrate).
    * :meth:`embed` -> ``(B, D)`` — ``pool(features(x))``, the flat embedding (the RL artifact
      + the apples-to-apples interface). ``forward`` IS ``embed``, so the exported ONNX graph
      is the deployable ``(B, D)`` forward.

    Trunk and pooling are stored as independent submodules so any trunk composes with any
    pooling. Build via :func:`build_encoder` for the config-driven ablation matrix.
    """

    def __init__(self, trunk: Trunk, pooling: Pooling) -> None:
        super().__init__()
        self.trunk = trunk
        self.pooling = pooling

    @property
    def out_channels(self) -> int:
        """The channel count ``C`` of the trunk's spatial feature map."""
        return self.trunk.out_channels

    def features(self, x: Tensor) -> Tensor:
        """Return the trunk's spatial feature map ``(B, C, h, w)`` for NCHW ``x``."""
        return self.trunk(x)

    def embed(self, x: Tensor) -> Tensor:
        """Return the flat embedding ``(B, D) = pool(features(x))`` for NCHW ``x``."""
        return self.pooling(self.trunk(x))

    def forward(self, x: Tensor) -> Tensor:
        """Alias for :meth:`embed`, so the exported ONNX graph is the deployable forward."""
        return self.embed(x)

    def embedding_dim(self, input_hw: tuple[int, int] | None = None) -> int | None:
        """Return the embedding ``D``, or ``None`` if it is not statically knowable.

        For GAP, ``D == out_channels`` regardless of input size. For flatten, ``D`` depends on
        the trunk's output spatial size, so an ``input_hw`` (defaulting to nothing) is needed;
        when not supplied the trunk is probed with a single zero frame to derive ``(h, w)``.
        """
        spatial_hw: tuple[int, int] | None = None
        if input_hw is not None:
            h, w = input_hw
            device = next(self.parameters()).device
            probe = torch.zeros(1, 3, h, w, device=device)
            was_training = self.training
            self.eval()
            with torch.no_grad():
                feat = self.trunk(probe)
            if was_training:
                self.train()
            spatial_hw = (int(feat.shape[2]), int(feat.shape[3]))
        return self.pooling.embedding_dim(self.out_channels, spatial_hw)


# --- config-driven composition ----------------------------------------------------------

# Trunk / pooling registries: the swappable pieces of the {trunk} x {pooling} ablation grid.
TRUNKS: dict[str, type[Trunk]] = {
    "cnn": CnnTrunk,
    "resnet": ResNetTrunk,
    "gn-cnn": GroupNormCNN,
}
POOLINGS: dict[str, type[Pooling]] = {"gap": GlobalAveragePool, "flatten": Flatten}


@dataclass(frozen=True)
class EncoderConfig:
    """The composition contract: which trunk and which pooling make up an encoder.

    ``trunk`` is a key of :data:`TRUNKS` (``"cnn"`` / ``"resnet"`` / ``"gn-cnn"``); ``pooling``
    is a key of :data:`POOLINGS` (``"gap"`` / ``"flatten"``). ``in_channels`` is the input
    channel count (3 for the RGB frame).

    ``residual`` and ``blocks_per_stage`` drive the ResNet ablation: ``residual=True`` (the
    default) keeps the skip connections, ``residual=False`` is the same conv stack without the
    add, and ``blocks_per_stage`` sets the per-stage block depth. These two fields are
    RESNET-ONLY: :func:`build_encoder` consumes them only when ``trunk == "resnet"`` and never
    passes them to the :class:`CnnTrunk` trunk, so they are inert (harmless defaults) for a cnn
    config.

    ``cnn_depth`` is GN-CNN-ONLY: it sets the :class:`GroupNormCNN` base channel width that
    doubles per stride-2 block (output channels are ``8*cnn_depth``). :func:`build_encoder`
    passes it only when ``trunk == "gn-cnn"``; it is inert for cnn / resnet configs.

    Frozen + validated so a config is a stable, hashable record of one ablation cell.
    """

    trunk: TrunkName = "cnn"
    pooling: PoolingName = "gap"
    in_channels: int = 3
    residual: bool = True
    blocks_per_stage: int = 2
    cnn_depth: int = 32

    def __post_init__(self) -> None:
        if self.trunk not in TRUNKS:
            raise ValueError(f"unknown trunk {self.trunk!r}; choose from {sorted(TRUNKS)}")
        if self.pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {self.pooling!r}; choose from {sorted(POOLINGS)}")
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage must be >= 1, got {self.blocks_per_stage}")
        if self.cnn_depth < 1:
            raise ValueError(f"cnn_depth must be >= 1, got {self.cnn_depth}")


def build_encoder(cfg: EncoderConfig) -> Encoder:
    """Assemble the :class:`Encoder` named by ``cfg`` (the one-liner ablation factory).

    Looks the trunk and pooling classes up in :data:`TRUNKS` / :data:`POOLINGS` and composes
    them. The trunk-specific fields are passed ONLY to their trunk: the resnet-only fields
    (``residual``, ``blocks_per_stage``) reach :class:`ResNetTrunk` only when
    ``trunk == "resnet"``, and the gn-cnn-only ``cnn_depth`` reaches :class:`GroupNormCNN` only
    when ``trunk == "gn-cnn"``; the :class:`CnnTrunk` trunk receives neither. Each ablation
    cell is a single call::

        build_encoder(EncoderConfig("cnn", "flatten"))  # "just the CNN trunk"
        build_encoder(EncoderConfig("cnn", "gap"))
        build_encoder(EncoderConfig("resnet", "gap"))                    # residual
        build_encoder(EncoderConfig("resnet", "gap", residual=False))    # plain
        build_encoder(EncoderConfig("resnet", "gap", blocks_per_stage=3))  # deeper
        build_encoder(EncoderConfig("gn-cnn", "flatten"))               # small frames
    """
    if cfg.trunk == "resnet":
        trunk: Trunk = ResNetTrunk(
            in_channels=cfg.in_channels,
            blocks_per_stage=cfg.blocks_per_stage,
            residual=cfg.residual,
        )
    elif cfg.trunk == "gn-cnn":
        trunk = GroupNormCNN(in_channels=cfg.in_channels, cnn_depth=cfg.cnn_depth)
    else:
        trunk = TRUNKS[cfg.trunk](in_channels=cfg.in_channels)
    pooling = POOLINGS[cfg.pooling]()
    return Encoder(trunk, pooling)


# --- ONNX export (the deployment seam) --------------------------------------------------


def export_onnx(
    encoder: Encoder,
    path: str | Path,
    input_hw: tuple[int, int] = CANONICAL_HW,
    *,
    opset_version: int = 15,
) -> Path:
    """Export ``encoder.embed`` to a Unity-Sentis-clean ONNX graph at ``path``.

    The deployable forward is the ``(B, D)`` embedding, so the encoder is put in ``eval()``
    (BatchNorm in inference mode) and exported with a single NCHW input ``(1, in, H, W)``.
    Only the BATCH axis is dynamic — the spatial axes are FIXED at ``input_hw`` so the graph
    has static spatial shapes for Sentis. Uses the TorchScript exporter (``dynamo=False``),
    ``opset_version=15`` (default), and the op set is kept clean by construction (GAP ->
    ReduceMean, flatten -> Reshape, BatchNorm, Conv/Relu/Add/MaxPool — no Scan/Loop/GroupNorm).

    Returns the written :class:`~pathlib.Path`. NOTE: ``torch.onnx.export`` requires the
    ``onnx`` package to be installed to serialize the model proto; if it is absent this raises
    ``torch.onnx.OnnxExporterError`` from torch.
    """
    path = Path(path)
    h, w = input_hw
    example = torch.zeros(1, encoder.trunk.in_channels, h, w)
    was_training = encoder.training
    encoder.eval()
    try:
        torch.onnx.export(
            encoder,
            (example,),
            str(path),
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset_version,
            dynamo=False,
        )
    finally:
        if was_training:
            encoder.train()
    return path
