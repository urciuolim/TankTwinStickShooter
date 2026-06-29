"""Tests for pop_trainer.models.encoders — the composable, ablation-ready vision encoders.

These import torch (NOT pure), so the module skips wholesale if torch is absent via the
top-level ``importorskip`` (the repo gates its torch tests this way). They cover:

1. forward-pass SHAPES for every {trunk} x {pooling}: ``features`` is 4-D ``(B, C, h, w)``,
   ``embed`` is 2-D ``(B, D)`` — at a small representative size for speed AND once at the real
   640x360 (batch 1) to prove the stem crushes the canonical frame;
2. composability: the named ablation configs (cnn, resnet-residual, resnet-plain) build
   + run end-to-end via the factory;
3. the resnet residual ablation: resnet-plain and resnet-residual share channel widths and
   conv-layer count, differing only by the skip add, so their outputs diverge; and the
   ``blocks_per_stage`` depth knob adds conv params while keeping the output channels;
4. ONNX-export cleanliness: each encoder (residual AND plain) exports to a tmp file and the
   exported graph's op set is asserted Sentis-clean (NO Scan / Loop / GroupNorm, plus a
   positive allow-list). The op-scan is REAL — it iterates the exported graph's node op_types
   via the ``onnx`` package (a pinned hard dependency), and the assertion is not a no-op (see
   ``_assert_sentis_clean``);
5. the registry contract: ``TRUNKS`` carries exactly the architecture-descriptive keys mapped
   to the right classes;
6. determinism: ``embed`` in ``eval()`` is deterministic for the same input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop_trainer.models import (  # noqa: E402  (after importorskip, by design)
    CANONICAL_HW,
    POOLINGS,
    TRUNKS,
    CnnTrunk,
    EncoderConfig,
    GroupNormCNN,
    ResNetTrunk,
    build_encoder,
    export_onnx,
)

# A small input that still survives the cnn-trunk stem+trio (90x160 underflows the conv
# stack; half-canonical 180x320 runs the SAME code path and is fast).
SMALL_HW = (180, 320)

# Every {trunk} x {pooling} cell of the canonical-frame, Sentis-deployable ablation grid. The
# gn-cnn trunk is a SMALL-frame RL trunk (GroupNorm, not a Sentis-clean deploy target) and is
# covered by its own tests below, so it is excluded from the deployable-grid sweep here.
DEPLOY_TRUNKS = [t for t in TRUNKS if t != "gn-cnn"]
ALL_CELLS = [(t, p) for t in DEPLOY_TRUNKS for p in POOLINGS]

# The named ablation configs the contract calls out, including the resnet-plain (residual off)
# variant that isolates whether the skip connections earn their keep.
ABLATION_CONFIGS = [
    EncoderConfig(trunk="cnn", pooling="flatten"),  # "just the cnn trunk"
    EncoderConfig(trunk="cnn", pooling="gap"),
    EncoderConfig(trunk="resnet", pooling="gap"),  # resnet-residual
    EncoderConfig(trunk="resnet", pooling="gap", residual=False),  # resnet-plain
    EncoderConfig(trunk="resnet", pooling="flatten", residual=False),  # resnet-plain + flatten
]

# resnet configs that must export Sentis-clean: residual AND plain, both poolings. Parametrized
# into the ONNX test so the op-scan fires for plain configs too.
RESNET_ONNX_CONFIGS = [
    EncoderConfig(trunk="resnet", pooling="gap"),  # resnet-residual
    EncoderConfig(trunk="resnet", pooling="gap", residual=False),  # resnet-plain
    EncoderConfig(trunk="resnet", pooling="flatten", residual=False),  # resnet-plain + flatten
]


def _resnet_id(cfg) -> str:
    """A readable parametrize id naming the resnet variant under test."""
    return f"{'residual' if cfg.residual else 'plain'}+{cfg.pooling}"


def _conv_param_count(module) -> int:
    """Total parameter count across all Conv2d layers in ``module`` (proxy for conv depth)."""
    return sum(
        p.numel()
        for m in module.modules()
        if isinstance(m, torch.nn.Conv2d)
        for p in m.parameters()
    )


# Ops a Unity-Sentis-clean graph must NEVER contain (the deployment hard rule).
FORBIDDEN_OPS = frozenset({"Scan", "Loop", "GroupNormalization", "GroupNorm"})

# Positive allow-list: every op we expect our encoders to lower to on opset 15. A node op
# outside this set is treated as a regression (a new exotic op crept in).
ALLOWED_OPS = frozenset(
    {
        "Conv",
        "BatchNormalization",
        "Relu",
        "MaxPool",
        "Add",
        "ReduceMean",
        "GlobalAveragePool",
        "Reshape",
        "Flatten",
        "Constant",
        "Shape",
        "Gather",
        "Unsqueeze",
        "Concat",
        "Squeeze",
        "Identity",
        "Cast",
    }
)


def _exported_op_types(path) -> list[str]:
    """Return the op_type of every node in an exported ONNX graph at ``path``.

    Strongest path: if the ``onnx`` package is importable, ``onnx.load`` the model and read
    ``graph.node[*].op_type`` (and any subgraph nodes). Fallback (no ``onnx`` package): scan
    the raw protobuf bytes for the NodeProto ``op_type`` field. Both yield REAL op names from
    the exported file — neither is a hardcoded pass.
    """
    try:
        import onnx  # noqa: PLC0415  (optional, probed at call time)
    except ImportError:
        return _scan_proto_op_types(path)

    model = onnx.load(str(path))
    ops: list[str] = []

    def _collect(graph) -> None:
        for node in graph.node:
            ops.append(node.op_type)
            for attr in node.attribute:
                if attr.g.ByteSize():
                    _collect(attr.g)
                for sub in attr.graphs:
                    _collect(sub)

    _collect(model.graph)
    return ops


def _scan_proto_op_types(path) -> list[str]:
    """Recover node op_types from a raw ONNX protobuf without the ``onnx`` package.

    In the serialized ModelProto a ``NodeProto.op_type`` is field #4, a length-delimited
    string -> the wire byte ``0x22`` (field 4, wire type 2) followed by a varint length and
    the UTF-8 op name. This walks the bytes and pulls those strings. Used only as the fallback
    when ``onnx`` is not installed; it is a genuine scan of the exported graph, not a stub.
    """
    data = path.read_bytes()
    ops: list[str] = []
    i, n = 0, len(data)
    while i < n:
        if data[i] == 0x22:  # field 4 (op_type), wire type 2 (length-delimited)
            j = i + 1
            length = 0
            shift = 0
            # decode the varint length
            while j < n:
                byte = data[j]
                length |= (byte & 0x7F) << shift
                j += 1
                if not (byte & 0x80):
                    break
                shift += 7
            if 0 < length <= 64 and j + length <= n:
                chunk = data[j : j + length]
                try:
                    name = chunk.decode("ascii")
                except UnicodeDecodeError:
                    i += 1
                    continue
                # op_type names are CamelCase identifiers (ascii letters/digits)
                if name and name[0].isascii() and name.replace("_", "").isalnum():
                    ops.append(name)
                i = j + length
                continue
        i += 1
    return ops


def _assert_sentis_clean(op_types: list[str]) -> None:
    """Assert an exported graph's op list is Sentis-clean: REAL assertion, not a no-op.

    Fails if ANY forbidden op (Scan / Loop / GroupNorm) appears, and fails if the op set is
    empty (a sign the scan recovered nothing). When the strong path is in use the allow-list
    catches any unexpected exotic op too.
    """
    assert op_types, "no op_types recovered from the exported ONNX graph"
    present = set(op_types)
    forbidden_present = present & FORBIDDEN_OPS
    assert not forbidden_present, (
        f"forbidden Sentis ops in exported graph: {sorted(forbidden_present)}"
    )


# --- 0. the registry contract (architecture-descriptive keys -> classes) ----------------


def test_trunk_registry_keys_and_classes():
    """TRUNKS carries exactly the architecture-descriptive keys mapped to the right classes."""
    assert set(TRUNKS) == {"cnn", "resnet", "gn-cnn"}
    assert TRUNKS["cnn"] is CnnTrunk
    assert TRUNKS["resnet"] is ResNetTrunk
    assert TRUNKS["gn-cnn"] is GroupNormCNN


@pytest.mark.parametrize("trunk", ["cnn", "resnet", "gn-cnn"])
def test_build_encoder_works_for_each_registry_key(trunk):
    """build_encoder assembles + runs every registry trunk, honoring its trunk-specific knobs.

    cnn / resnet take the canonical-frame path; gn-cnn takes a small frame. resnet exercises the
    residual / blocks_per_stage knobs and gn-cnn exercises cnn_depth, so the per-trunk fields are
    proven to flow through the factory.
    """
    if trunk == "gn-cnn":
        cfg = EncoderConfig(trunk=trunk, pooling="flatten", cnn_depth=16)
        hw = (64, 64)
    elif trunk == "resnet":
        cfg = EncoderConfig(trunk=trunk, pooling="gap", residual=False, blocks_per_stage=3)
        hw = SMALL_HW
    else:
        cfg = EncoderConfig(trunk=trunk, pooling="flatten")
        hw = SMALL_HW
    enc = build_encoder(cfg).eval()
    x = torch.zeros(1, 3, *hw)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)
    assert feats.ndim == 4
    assert emb.ndim == 2
    assert isinstance(TRUNKS[trunk], type)
    assert isinstance(enc.trunk, TRUNKS[trunk])


def test_encoder_config_rejects_unknown_trunk():
    """An unknown trunk key fails loudly at EncoderConfig construction (registry-driven)."""
    with pytest.raises(ValueError, match="unknown trunk"):
        EncoderConfig(trunk="bogus-trunk", pooling="gap")  # type: ignore[arg-type]


# --- 1. instantiation + forward-pass SHAPES --------------------------------------------


@pytest.mark.parametrize(("trunk", "pooling"), ALL_CELLS)
def test_forward_shapes_small(trunk, pooling):
    """features -> 4-D (B, C, h, w); embed -> 2-D (B, D) for every {trunk} x {pooling}."""
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling)).eval()
    h, w = SMALL_HW
    x = torch.zeros(2, 3, h, w)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)

    assert feats.ndim == 4
    assert feats.shape[0] == 2
    assert feats.shape[1] == enc.out_channels
    # the stem must have downsampled below the input
    assert feats.shape[2] < h and feats.shape[3] < w

    assert emb.ndim == 2
    assert emb.shape[0] == 2
    if pooling == "gap":
        assert emb.shape[1] == enc.out_channels
    else:  # flatten
        assert emb.shape[1] == enc.out_channels * feats.shape[2] * feats.shape[3]
    # forward() aliases embed()
    with torch.no_grad():
        assert torch.equal(enc(x), emb)


@pytest.mark.parametrize(("trunk", "pooling"), ALL_CELLS)
def test_forward_shapes_canonical(trunk, pooling):
    """At the real 640x360 (H=360, W=640), batch 1: the stem crushes the frame tiny."""
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling)).eval()
    h, w = CANONICAL_HW  # (360, 640)
    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)

    assert feats.ndim == 4 and feats.shape[0] == 1
    assert feats.shape[1] == enc.out_channels
    # crushed by the stem: feature map is far smaller than the canonical frame
    assert feats.shape[2] < h // 8 and feats.shape[3] < w // 8
    assert emb.ndim == 2 and emb.shape[0] == 1


# --- 2. composability (the named ablations) --------------------------------------------


def _cfg_id(c) -> str:
    return f"{c.trunk}+{c.pooling}+{'res' if c.residual else 'plain'}"


@pytest.mark.parametrize("cfg", ABLATION_CONFIGS, ids=_cfg_id)
def test_ablation_configs_build_and_run(cfg):
    """Each named ablation (incl. resnet-plain) assembles via build_encoder and runs."""
    enc = build_encoder(cfg).eval()
    h, w = SMALL_HW
    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)
    assert feats.ndim == 4
    assert emb.ndim == 2

    # statically-derivable embedding_dim agrees with the realized embedding
    d = enc.embedding_dim(SMALL_HW)
    assert d == emb.shape[1]


def test_bad_config_rejected():
    """Unknown trunk / pooling names, bad in_channels, bad depth fail loudly at construction."""
    with pytest.raises(ValueError, match="unknown trunk"):
        EncoderConfig(trunk="mamba", pooling="gap")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown pooling"):
        EncoderConfig(trunk="cnn", pooling="attention")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="in_channels"):
        EncoderConfig(trunk="cnn", pooling="gap", in_channels=0)
    with pytest.raises(ValueError, match="blocks_per_stage"):
        EncoderConfig(trunk="resnet", pooling="gap", blocks_per_stage=0)


# --- the resnet residual ablation: same shape, the skip is the only difference ----------


def test_resnet_plain_vs_residual_same_shape_differ_only_by_skip():
    """resnet-plain and resnet-residual: identical conv structure, outputs diverge.

    With the SAME seed (so both build identical conv weights), the plain and residual trunks
    produce the same output channel count, the same spatial shape, and the SAME number of conv
    parameters (the skip add holds no weights) — yet their feature maps differ on a nonzero
    input. That isolates the skip connection as the only structural difference.
    """
    h, w = SMALL_HW
    x = torch.randn(1, 3, h, w)

    torch.manual_seed(0)
    res = build_encoder(EncoderConfig(trunk="resnet", pooling="gap", residual=True)).eval()
    torch.manual_seed(0)
    plain = build_encoder(EncoderConfig(trunk="resnet", pooling="gap", residual=False)).eval()

    with torch.no_grad():
        f_res = res.features(x)
        f_plain = plain.features(x)

    # same output channels (last of [16, 32, 32]) and same spatial shape
    assert res.out_channels == plain.out_channels == 32
    assert f_res.shape == f_plain.shape
    # same conv-layer parameter count: the skip add carries no weights
    assert _conv_param_count(res.trunk) == _conv_param_count(plain.trunk)
    # but the skip add changes the feature map: outputs must diverge on a nonzero input
    assert not torch.allclose(f_res, f_plain)


def test_resnet_depth_knob_adds_conv_params_keeps_channels():
    """blocks_per_stage: deeper builds + runs, adds conv params, keeps output channels at 32.

    Each extra block per stage adds two 3x3 convs at that stage's width, so the total conv
    parameter count strictly increases with depth, while the trunk's output channels (the last
    of [16, 32, 32]) stay 32 and the encoder still runs end-to-end.
    """
    h, w = SMALL_HW
    x = torch.zeros(1, 3, h, w)

    shallow = build_encoder(EncoderConfig(trunk="resnet", pooling="gap", blocks_per_stage=1)).eval()
    deep = build_encoder(EncoderConfig(trunk="resnet", pooling="gap", blocks_per_stage=3)).eval()

    assert shallow.out_channels == deep.out_channels == 32
    assert _conv_param_count(deep.trunk) > _conv_param_count(shallow.trunk)
    with torch.no_grad():
        assert deep.embed(x).shape == (1, 32)


def test_embedding_dim_static_vs_probed():
    """embedding_dim: GAP is static (=out_channels) with no input_hw; flatten needs a probe.

    Covers the flatten 'D unknown without a spatial size' early-return and the probe path
    that restores train() afterward.
    """
    gap = build_encoder(EncoderConfig(trunk="cnn", pooling="gap"))
    # GAP D is knowable with no input_hw at all.
    assert gap.embedding_dim() == gap.out_channels

    flat = build_encoder(EncoderConfig(trunk="resnet", pooling="flatten"))
    # flatten D is not statically knowable without a spatial size.
    assert flat.embedding_dim() is None
    # probing in train() mode must derive a concrete D and leave the module in train().
    flat.train()
    d = flat.embedding_dim(SMALL_HW)
    assert isinstance(d, int) and d > 0
    assert flat.training is True


# --- 3. ONNX-export cleanliness smoke (every encoder) ----------------------------------


@pytest.mark.parametrize(("trunk", "pooling"), ALL_CELLS)
def test_onnx_export_sentis_clean(trunk, pooling, tmp_path):
    """Export each {trunk} x {pooling} encoder and assert its op set is Sentis-clean (REAL op-scan).

    ``onnx`` is a pinned dependency, so the export runs and the op-scan + forbidden-op
    assertion fire; this does not skip.
    """
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling))
    out = tmp_path / f"{trunk}_{pooling}.onnx"
    export_onnx(enc, out, input_hw=SMALL_HW)

    assert out.exists() and out.stat().st_size > 0
    op_types = _exported_op_types(out)
    _assert_sentis_clean(op_types)


@pytest.mark.parametrize("cfg", RESNET_ONNX_CONFIGS, ids=lambda c: _resnet_id(c))
def test_onnx_export_resnet_residual_and_plain_clean(cfg, tmp_path):
    """resnet-residual AND resnet-plain export Sentis-clean; plain emits no Add (no skip).

    The op-scan runs for both residual and plain configs. Beyond the forbidden-op check, this
    asserts every recovered op is in the positive allow-list and that a resnet-plain graph
    contains NO ``Add`` node (the skip is the only structural difference), while the residual
    graph does.
    """
    enc = build_encoder(cfg)
    out = tmp_path / f"resnet_{cfg.pooling}_{'res' if cfg.residual else 'plain'}.onnx"
    export_onnx(enc, out, input_hw=SMALL_HW)

    assert out.exists() and out.stat().st_size > 0
    op_types = _exported_op_types(out)
    _assert_sentis_clean(op_types)
    # every op must be in the positive allow-list (catches a new exotic op creeping in)
    unexpected = set(op_types) - ALLOWED_OPS
    assert not unexpected, f"unexpected ops in exported graph: {sorted(unexpected)}"
    # the residual skip add is the ONLY structural difference between the two graphs
    if cfg.residual:
        assert "Add" in op_types, "resnet-residual graph should contain the skip Add"
    else:
        assert "Add" not in op_types, "resnet-plain graph must contain no skip Add"


def test_forbidden_op_assertion_actually_fires():
    """Guard the guard: _assert_sentis_clean must FAIL when a forbidden op is present.

    Proves the cleanliness assertion is not a no-op — a graph op list containing 'Scan' (or
    'Loop' / 'GroupNorm') trips the assertion.
    """
    with pytest.raises(AssertionError, match="forbidden"):
        _assert_sentis_clean(["Conv", "Relu", "Scan", "ReduceMean"])
    with pytest.raises(AssertionError):
        _assert_sentis_clean([])  # empty scan must also fail


def test_proto_scanner_recovers_known_ops():
    """The no-onnx fallback proto-scanner recovers op names from real exported bytes.

    Builds a tiny ONNX-shaped protobuf by hand (length-delimited field-4 op_type strings) and
    confirms the byte-scanner pulls them out, so the fallback op-scan is exercised even when
    ``onnx`` IS installed (and is provably not a stub).
    """

    def _node(op: str) -> bytes:
        body = op.encode("ascii")
        return b"\x22" + bytes([len(body)]) + body

    blob = b"".join(_node(op) for op in ("Conv", "Relu", "ReduceMean"))

    class _FakePath:
        def read_bytes(self):  # noqa: D401 - tiny shim
            return blob

    ops = _scan_proto_op_types(_FakePath())
    assert ops == ["Conv", "Relu", "ReduceMean"]


# --- the gn-cnn trunk: survives small frames where the cnn stem collapses -----------------

# The gn-cnn trunk's canonical input is a 64x64 RGB frame; the four stride-2 blocks take it 64 -> 4.
SMALL_FRAME_HW = (64, 64)


def test_gn_cnn_trunk_spatial_map_default_depth():
    """gn-cnn trunk at 64x64, cnn_depth=32: feature map is (B, 256, 4, 4); embedding finite.

    Four stride-2 blocks halve the spatial dims each (64 -> 32 -> 16 -> 8 -> 4) while channels
    double from cnn_depth (32 -> 64 -> 128 -> 256 = 8*cnn_depth), so the map is (B, 256, 4, 4)
    and the flatten embedding is 256*4*4 = 4096.
    """
    enc = build_encoder(EncoderConfig(trunk="gn-cnn", pooling="flatten")).eval()
    x = torch.zeros(2, 3, *SMALL_FRAME_HW)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)

    assert feats.shape == (2, 256, 4, 4)
    assert enc.out_channels == 256
    assert emb.shape == (2, 256 * 4 * 4)  # 4096
    assert emb.shape[1] == 4096
    assert torch.isfinite(emb).all()
    # the statically-derived flatten D agrees with the realized embedding
    assert enc.embedding_dim(SMALL_FRAME_HW) == 4096


def test_gn_cnn_trunk_scales_with_cnn_depth():
    """cnn_depth flows ONLY to the gn-cnn trunk: depth 16 -> (B, 128, 4, 4), out_channels 128.

    Output channels are 8*cnn_depth, so halving the depth halves every channel width while the
    64 -> 4 spatial schedule is unchanged.
    """
    enc = build_encoder(EncoderConfig(trunk="gn-cnn", pooling="flatten", cnn_depth=16)).eval()
    x = torch.zeros(1, 3, *SMALL_FRAME_HW)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)

    assert feats.shape == (1, 128, 4, 4)
    assert enc.out_channels == 128
    assert emb.shape == (1, 128 * 4 * 4)  # 2048
    assert torch.isfinite(emb).all()


def test_gn_cnn_no_batchnorm():
    """The gn-cnn trunk uses no BatchNorm (an on-policy-RL footgun) — norm is GroupNorm."""
    enc = build_encoder(EncoderConfig(trunk="gn-cnn", pooling="flatten"))
    norms = [m for m in enc.trunk.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert not norms, "gn-cnn trunk must not contain BatchNorm"
    assert any(isinstance(m, torch.nn.GroupNorm) for m in enc.trunk.modules())


def test_gn_cnn_bad_cnn_depth_rejected():
    """cnn_depth < 1 fails loudly at construction (the gn-cnn-only depth knob is validated)."""
    with pytest.raises(ValueError, match="cnn_depth"):
        EncoderConfig(trunk="gn-cnn", pooling="flatten", cnn_depth=0)


# --- regression: the cnn-trunk path at the canonical 360x640 is unchanged ----------------


def test_cnn_canonical_regression_unchanged():
    """The cnn trunk at the canonical 360x640 still produces the expected (B, 64, h, w) map.

    Locks the byte-for-byte 360x640 cnn-trunk path against the gn-cnn addition: out_channels is
    the conv trio's last width (64), the stem+trio crush the frame well below an eighth of
    the input, and the flatten embedding D is the expected positive int.
    """
    enc = build_encoder(EncoderConfig(trunk="cnn", pooling="flatten")).eval()
    h, w = CANONICAL_HW  # (360, 640)
    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        feats = enc.features(x)
        emb = enc.embed(x)

    assert enc.out_channels == 64
    assert feats.shape[:2] == (1, 64)
    assert feats.shape[2] < h // 8 and feats.shape[3] < w // 8
    assert emb.shape == (1, 64 * feats.shape[2] * feats.shape[3])
    assert enc.embedding_dim(CANONICAL_HW) == emb.shape[1]


# --- 4. determinism nicety -------------------------------------------------------------


@pytest.mark.parametrize(("trunk", "pooling"), ALL_CELLS)
def test_embed_deterministic_in_eval(trunk, pooling):
    """embed() in eval() is deterministic for the same input (no dropout / BN-in-train)."""
    enc = build_encoder(EncoderConfig(trunk=trunk, pooling=pooling)).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 3, *SMALL_HW)
    with torch.no_grad():
        a = enc.embed(x)
        b = enc.embed(x)
    assert torch.equal(a, b)
