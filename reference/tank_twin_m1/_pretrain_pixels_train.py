"""Torch trainer for the pixel inverse-renderer pretraining (Stage 2).

Imported lazily from :mod:`tank_twin.pretrain_pixels` so the pure target-encoding /
normalization / split / metric helpers (and their unit tests) never need torch.
Contains the model (reusable encoder + configurable supervised heads), the streaming
and memmap datasets, the loss, the train/eval/validation loop with per-objective
metrics, and checkpoint save/reload.

The ENCODER (conv stack + FC projection) is the reusable artifact and is INDEPENDENT
of the objective config: its state_dict is saved standalone as ``encoder.pt`` and can
be loaded into a fresh ``PixelEncoder`` without any head, regardless of which
objectives were trained. Only the heads change with the objective config.

TERMINOLOGY:
* "eval" = between-epoch, IN-DISTRIBUTION (held-out EPISODES of the 8 training maps).
* "validation" = final, OUT-OF-DISTRIBUTION (the 2 held-out MAPS), run ONCE.
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tank_twin.pretrain_pixels import (
    AIM_COSINE_EPS,
    N_BULLET_DIR,
    N_BULLET_POS,
    N_BULLET_SLOTS,
    PLAYER_POSITION_VEC_INDICES,
    PLAYER_SUBGROUP_VEC_INDICES,
    WALL_H,
    WALL_W,
    NormStats,
    ObjectiveConfig,
    aim_angular_error_deg,
    bullet_occupancy_field,
    cosine_warmup_lr_multiplier,
    decode_targets,
    denormalize,
    downsample_frames_np,
    fit_norm_stats,
    grid_to_world,
    head_sizes,
    heatmap_point_channels,
    interior_wall_mask,
    interior_wall_pos_weight,
    list_shards,
    load_map_name_to_id,
    map_aware_split,
    parse_head,
    parse_pos_weight_arg,
    player_head_vec_indices,
    png_decode,
    presence_pos_weight,
    resolve_map_ids,
    wall_target_for,
    worker_id_from_shard,
    world_to_grid,
)

# Player sub-group -> (name, position?) and its slice within the player HEAD output.
_PLAYER_SUBGROUPS = ("player_position", "player_velocity", "player_aim")
_PLAYER_POSITION_GROUPS = {"player_position"}
# The 12-vec indices of the aim sub-group, [P1_x, P1_y, P2_x, P2_y].
_AIM_VEC_INDICES = PLAYER_SUBGROUP_VEC_INDICES["player_aim"]


def _aim_head_columns(model: InverseRenderer) -> list[int] | None:
    """Columns of the player-head output that hold the aim slice (4,5,10,11), in
    ``[P1_x, P1_y, P2_x, P2_y]`` order; ``None`` if aim is not a head output."""
    vi = model.player_vec_indices
    if not all(k in vi for k in _AIM_VEC_INDICES):
        return None
    return [vi.index(k) for k in _AIM_VEC_INDICES]


def _player_non_aim_columns(model: InverseRenderer) -> list[int]:
    """Columns of the player-head output that are NOT aim (position + velocity)."""
    aim_cols = set(_aim_head_columns(model) or [])
    return [c for c in range(len(model.player_vec_indices)) if c not in aim_cols]


def _destandardize_aim(z_aim: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """De-standardize a ``(N, 4)`` aim slice (12-vec idx 4,5,10,11) back to RAW units.

    The player target is z-standardized (decode_targets divides each aim component by
    ~0.707, distorting the angle), so the cosine term must compare RAW directions. This
    inverts ``z -> z*std + mean`` for the aim indices, on the slice's device/dtype.
    """
    idx = list(_AIM_VEC_INDICES)
    mean = torch.as_tensor(stats.player_mean[idx], dtype=z_aim.dtype, device=z_aim.device)
    std = torch.as_tensor(stats.player_std[idx], dtype=z_aim.dtype, device=z_aim.device)
    return z_aim * std + mean


def _torch_per_player_cosine_distance(
    pred_aim: torch.Tensor, target_aim: torch.Tensor, *, eps: float = AIM_COSINE_EPS
) -> torch.Tensor:
    """Torch mirror of :func:`pretrain_pixels.per_player_cosine_distance`.

    ``pred_aim`` / ``target_aim`` are ``(N, 4)`` = ``[P1_x, P1_y, P2_x, P2_y]``; the
    cosine is taken per PLAYER on the 2D vector (reshape to ``(N, 2, 2)``) with ``eps``
    floored into each L2-norm denominator, then ``1 - cos`` is averaged over the 2
    players and the batch. Identical -> 0, orthogonal -> 1, opposite -> 2.
    """
    pred = pred_aim.reshape(-1, 2, 2)
    tgt = target_aim.reshape(-1, 2, 2)
    pn = torch.sqrt((pred**2).sum(dim=2)) + eps  # (N, 2)
    tn = torch.sqrt((tgt**2).sum(dim=2)) + eps  # (N, 2)
    dot = (pred * tgt).sum(dim=2)  # (N, 2)
    cos = dot / (pn * tn)
    return (1.0 - cos).mean()


# =============================================================================
# Spatial-softmax / soft-argmax (heatmap head; DIFFERENTIABLE; unit-tested)
# =============================================================================
def spatial_soft_argmax(heatmaps: torch.Tensor, *, normalized: str = "unit") -> torch.Tensor:
    """DIFFERENTIABLE soft-argmax over a stack of 2D heatmaps -> continuous (x, y).

    ``heatmaps`` is ``(B, K, h, w)`` of raw (unnormalized) scores. A spatial SOFTMAX over
    the ``(h, w)`` plane turns each channel into a probability map; the soft-argmax is the
    probability-weighted mean of the cell coordinates -> one continuous ``(x, y)`` per
    channel, giving ``(B, K, 2)`` with ``[..., 0] = x`` (column) and ``[..., 1] = y`` (row).

    Coordinate convention (``normalized``):
    * ``"unit"`` (default): cell centers span ``[0, 1]`` along each axis (col j ->
      ``(j + 0.5) / w``, row i -> ``(i + 0.5) / h``). A heatmap peaked at one cell returns
      that cell's center; a symmetric blob returns its centroid. Bounded in ``(0, 1)``.
    * ``"centered"``: same but mapped to ``[-1, 1]`` (``2 * unit - 1``).

    Fully differentiable: softmax + weighted sum, no argmax / indexing. Grad flows to the
    input heatmaps (asserted by the unit test). Factored out of the head so it is unit-
    testable on a one-hot / Gaussian heatmap without building the encoder.
    """
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, h * w)
    prob = torch.softmax(flat, dim=2).reshape(b, k, h, w)
    device, dtype = heatmaps.device, heatmaps.dtype
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w  # (w,) col centers in [0,1]
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h  # (h,) row centers in [0,1]
    exp_x = (prob.sum(dim=2) * xs).sum(dim=2)  # (B, K) sum over rows then weight cols
    exp_y = (prob.sum(dim=3) * ys).sum(dim=2)  # (B, K) sum over cols then weight rows
    coords = torch.stack([exp_x, exp_y], dim=2)  # (B, K, 2) -> (x, y) in [0,1]
    if normalized == "centered":
        return coords * 2.0 - 1.0
    return coords


def centernet_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on LOGITS (numerically stable).

    ``logits`` are RAW occupancy logits ``(B, H, W)`` (sigmoid applied INTERNALLY);
    ``target`` the sum-of-Gaussians occupancy field with ~1.0 peaks. Per cell::

        peak (target==1): -(1-p)^alpha * log(p)
        else:             -(1-target)^beta * p^alpha * log(1-p)

    averaged over the number of POSITIVE (peak) cells across the batch (>=1, count-
    normalized). The ``(1-target)^beta`` factor reduces the penalty for negatives NEAR a peak.

    STABILITY: the log terms use ``logsigmoid`` (``log p = logsigmoid(z)``,
    ``log(1-p) = logsigmoid(-z)``) rather than ``log(sigmoid(z).clamp(eps))``. The naive
    clamp-then-log form has a VANISHING gradient once logits saturate (``d p/d z = p(1-p)
    -> 0``), which lets the occupancy channel collapse to all-negative and get STUCK (a dead
    unit). The logsigmoid form keeps a finite gradient at saturation so the positive cells
    can always be pulled back up. Value-equivalent to the numpy reference (parity-tested).
    """
    logp = nn.functional.logsigmoid(logits)  # log(sigmoid(z)) = log p  (stable)
    log1mp = nn.functional.logsigmoid(-logits)  # log(sigmoid(-z)) = log(1-p)  (stable)
    p = torch.sigmoid(logits)
    pos = target >= 1.0 - 1e-12
    pos_loss = -((1.0 - p) ** alpha) * logp
    neg_loss = -((1.0 - target) ** beta) * (p**alpha) * log1mp
    loss = torch.where(pos, pos_loss, neg_loss)
    n_pos = pos.sum().clamp_min(1.0)
    return loss.sum() / n_pos


def extract_peaks(
    prob: np.ndarray, *, threshold: float = 0.3, nms: int = 1
) -> list[tuple[float, float]]:
    """Local-maximum peak extraction on a 2D occupancy heatmap (numpy; read-out for metrics).

    ``prob`` is a ``(H, W)`` heatmap in [0,1]. Returns the (col, row) fractional grid coords
    of each cell that is (a) ``>= threshold`` and (b) a strict local max in its
    ``(2*nms+1)`` neighbourhood (3x3 by default). This is the count + location read-out that
    makes the order-free occupancy field comparable to the slot-based count: ``len(peaks)``
    is the predicted bullet count, the coords the predicted locations. Pure numpy so it is
    unit-testable without torch.
    """
    h, w = prob.shape
    peaks: list[tuple[float, float]] = []
    for r in range(h):
        for c in range(w):
            v = prob[r, c]
            if v < threshold:
                continue
            r0, r1 = max(0, r - nms), min(h, r + nms + 1)
            c0, c1 = max(0, c - nms), min(w, c + nms + 1)
            if v >= prob[r0:r1, c0:c1].max():
                peaks.append((float(c), float(r)))
    return peaks


# =============================================================================
# Model: reusable encoder + configurable heads on the shared embedding
# =============================================================================
class PixelEncoder(nn.Module):
    """NatureCNN-style encoder; the embedding READOUT is head-aware (Part B).

    Input is channel-first ``(B, 3, H, W)`` float in [0,1]. The shared 3-conv trunk is the
    REUSABLE artifact; ``state_dict`` is what gets saved as ``encoder.pt``. Two readouts:

    * ``head="fc"`` (default, UNCHANGED): ``self.cnn`` is the 3 conv+ReLU layers FOLLOWED BY
      ``nn.Flatten()`` and ``self.linear`` is ``Linear(flatten_dim, embedding_dim)+ReLU``.
      ``forward`` -> ``self.linear(self.cnn(x))``. The module layout / state_dict keys
      (``cnn.0/2/4`` convs, ``linear.0``) and the forward are BYTE-IDENTICAL to the original
      encoder, so an ``encoder.pt`` saved in fc mode reloads into a fresh fc ``PixelEncoder``
      exactly as before.
    * ``head="heatmap"``: ``self.cnn`` is the 3 conv+ReLU layers WITHOUT the flatten (so the
      spatial ``(B, C, h, w)`` feature map is exposed for the heatmap head), and there is NO
      ``self.linear``. The embedding is a GLOBAL AVERAGE POOL of the feature map over
      ``(h, w)`` -> ``(B, C)``. This drops the (huge) flatten->FC matrix (12.8M params at 640)
      to ~0, which is the pooling win; ``embedding_dim`` is then forced to ``C`` (=64).

    ``head`` is recorded on the module so a fresh encoder can be rebuilt with the matching
    architecture for the standalone-reload self-check (the artifact is head-specific).

    SPATIAL encoder-output (``spatial_trunk=True``)
    -----------------------------------------------
    The conv TRUNK (``cnn.0/2/4``) is BUILT IDENTICALLY to the heatmap encoder (convs only,
    NO flatten, NO linear), so ``encoder.pt`` is the SAME 3-conv trunk and its ``cnn.0/2/4``
    keys+shapes match the flat-mode encoder. ``forward`` returns the spatial feature map
    ``(B, C, h, w)`` (the field projection to ``C x H x W`` lives in the InverseRenderer, NOT
    in the saved encoder, so ``encoder.pt`` stays a pure trunk). ``head`` is forced to
    ``None`` / inert in this mode.
    """

    def __init__(
        self,
        in_h: int,
        in_w: int,
        embedding_dim: int = 512,
        *,
        head: str = "fc",
        spatial_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_trunk = spatial_trunk
        self.head = None if spatial_trunk else parse_head(head)
        self.in_h, self.in_w = in_h, in_w
        if not spatial_trunk and self.head == "fc":
            # UNCHANGED: convs + Flatten in self.cnn, then Linear->ReLU in self.linear.
            self.embedding_dim = embedding_dim
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                flat = self.cnn(torch.zeros(1, 3, in_h, in_w)).shape[1]
            self.flatten_dim = int(flat)
            self.linear = nn.Sequential(nn.Linear(self.flatten_dim, embedding_dim), nn.ReLU())
        else:
            # heatmap OR spatial: convs only (NO flatten / NO linear) -> a clean trunk whose
            # cnn.0/2/4 keys are IDENTICAL across modes (the reusable encoder.pt artifact).
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                fm = self.cnn(torch.zeros(1, 3, in_h, in_w))  # (1, C, h, w)
            self.feature_channels = int(fm.shape[1])
            self.feature_hw = (int(fm.shape[2]), int(fm.shape[3]))
            self.flatten_dim = int(fm.numel())  # informational (what fc WOULD flatten to)
            # The pooled embedding width is the conv channel count (global pool over h,w).
            self.embedding_dim = self.feature_channels

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial conv feature map ``(B, C, h, w)`` (heatmap/spatial trunk; before any pool).

        Only valid when ``self.cnn`` is convs-only (heatmap head or spatial trunk); the fc
        ``self.cnn`` ends in a Flatten and returns ``(B, flatten_dim)``, so fc callers never
        invoke this.
        """
        return self.cnn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_trunk:
            # Return the spatial feature map; the field projection lives in the head.
            return self.cnn(x)
        if self.head == "fc":
            return self.linear(self.cnn(x))
        # heatmap: global average pool the (B, C, h, w) feature map over (h, w) -> (B, C).
        fm = self.cnn(x)
        return fm.mean(dim=(2, 3))


class InverseRenderer(nn.Module):
    """Encoder + the heads for the ENABLED objective groups. The localization READOUT is
    head-aware (Part B): ``head="fc"`` (default) is the ORIGINAL behavior; ``head="heatmap"``
    localizes the POINT targets (player_position, bullet_position) via per-point heatmaps.

    Disabled groups get no head at all (no parameters, no output, no compute).

    fc mode (UNCHANGED)
    -------------------
    Every enabled head is an ``nn.Linear(embedding_dim, size)`` off the flatten->FC
    embedding; the player head emits all enabled player sub-group columns; walls are a
    Linear(240)->reshape. Byte-identical to the original ``InverseRenderer``.

    heatmap mode
    ------------
    The encoder global-pools its spatial feature map into a ``(B, C)`` embedding (no
    flatten->FC). The POINT targets are localized off the SPATIAL map instead of regressed:

    * **player_position / bullet_position** (the point groups): a small conv on the feature
      map emits one heatmap channel per point (2 for players, 10 for bullets). Each channel
      is spatial-softmax'd and soft-argmax'd to a continuous ``(x, y)`` in normalized grid
      coords ``[0,1]``, then a PER-GROUP learned affine ``Linear(2, 2)`` maps grid->the SAME
      normalized (z-standardized) target space the fc head regresses, so the
      ``bullet_position``/``player_position`` metrics stay apples-to-apples. The player head
      ASSEMBLES the full ``player_vec_indices`` vector: position columns come from the
      heatmap path, velocity/aim columns (if enabled) from a small regression head off the
      pooled embedding (aim/velocity are directions/rates, NOT point localizations).
    * **walls**: a small SPATIAL CONV head off the feature map predicts the (B,12,20) logits
      (more natural than from a pooled vector; protects the spatial 20x12 wall structure
      from the global pool). The interior-only BCE loss/metric is unchanged.
    * **bullet_presence**: a small conv head off the feature map -> global-pool -> Linear(10)
      so presence keeps a SPATIAL source (bullet occupancy is spatial) and its F1 does not
      regress under the pooling. (bullet_direction is regressed from the pooled embedding.)

    In BOTH modes ``forward`` returns the SAME ``preds`` dict keys/shapes, so
    ``compute_losses`` and ``_evaluate`` are identical across heads — only the producers
    differ. The reusable ``encoder.pt`` is head-specific (fc: flatten->FC; heatmap: pooled).
    """

    def __init__(
        self,
        in_h: int,
        in_w: int,
        cfg: ObjectiveConfig,
        embedding_dim: int = 512,
        *,
        head: str = "fc",
        encoder_output: str = "flat",
        spatial_shape: tuple[int, int, int] | None = None,
        channel_map: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()
        self.encoder_output = encoder_output
        self.cfg = cfg
        if encoder_output == "spatial":
            # SPATIAL: trunk is convs-only (cnn.0/2/4 = encoder.pt); --head is N/A.
            self.head = None
            self.encoder = PixelEncoder(in_h, in_w, embedding_dim, spatial_trunk=True)
            self._init_spatial_heads(spatial_shape, channel_map)
            return
        self.head = parse_head(head)
        self.encoder = PixelEncoder(in_h, in_w, embedding_dim, head=self.head)
        self.sizes = head_sizes(cfg)
        # Indices into the 12-vector that the player head predicts (enabled sub-groups).
        self.player_vec_indices = player_head_vec_indices(cfg)
        # The pooled/FC embedding width (C in heatmap mode, embedding_dim in fc mode).
        emb = self.encoder.embedding_dim

        if self.head == "fc":
            self._init_fc_heads(emb)
        else:
            self._init_heatmap_heads(emb)

    # --- fc-mode heads (ORIGINAL) --------------------------------------------
    def _init_fc_heads(self, emb: int) -> None:
        self.heads = nn.ModuleDict()
        for name, size in self.sizes.items():
            self.heads[name] = nn.Linear(emb, size)

    # --- heatmap-mode heads --------------------------------------------------
    def _init_heatmap_heads(self, emb: int) -> None:
        cfg = self.cfg
        c = self.encoder.feature_channels
        # Which point groups are localized (player_position -> 2, bullet_position -> 10).
        self.point_channels = heatmap_point_channels(cfg)  # dict group -> n_points
        n_heat = sum(self.point_channels.values())
        # Regression heads off the POOLED embedding for the non-localized targets.
        self.heads = nn.ModuleDict()
        # Player velocity/aim columns (everything in player_vec_indices that is NOT position).
        self._player_pos_cols = [
            i for i, vi in enumerate(self.player_vec_indices) if vi in PLAYER_POSITION_VEC_INDICES
        ]
        self._player_reg_cols = [
            i for i in range(len(self.player_vec_indices)) if i not in self._player_pos_cols
        ]
        if cfg.any_player() and self._player_reg_cols:
            # Regress only the non-position player columns (velocity and/or aim).
            self.heads["player_reg"] = nn.Linear(emb, len(self._player_reg_cols))
        if cfg.is_on("bullet_direction"):
            self.heads["bullet_direction"] = nn.Linear(emb, self.sizes["bullet_direction"])
        # Heatmap conv: feature map -> n_heat heatmap channels (one per localized point).
        if n_heat > 0:
            self.heatmap_conv = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, n_heat, kernel_size=1),
            )
            # Per-GROUP learned affine grid(x,y)[0,1] -> normalized target space.
            self.point_affine = nn.ModuleDict({g: nn.Linear(2, 2) for g in self.point_channels})
        # bullet_presence: small conv -> global pool -> Linear(10) (spatial source).
        if cfg.is_on("bullet_presence"):
            self.presence_conv = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.presence_fc = nn.Linear(c, N_BULLET_SLOTS)
        # walls: small spatial conv head -> (B,1,12,20) logits via adaptive pool.
        if cfg.is_on("walls"):
            self.wall_conv = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, 1, kernel_size=1),
                nn.AdaptiveAvgPool2d((WALL_H, WALL_W)),
            )

    # --- spatial-mode heads (NEW field paradigm) -----------------------------
    def _init_spatial_heads(
        self,
        spatial_shape: tuple[int, int, int] | None,
        channel_map: dict[str, tuple[int, ...]] | None,
    ) -> None:
        """Project the trunk feature map to a ``C x H x W`` field; pin objectives to channels.

        A light 1x1 conv maps the trunk's ``feature_channels`` to ``C``; the field is then
        bilinearly resized to ``(H, W)`` (the configured grid the targets are built at). Each
        objective READS its assigned channel(s) of that field directly (light per-channel
        activation); there are NO separate per-objective head matrices beyond the shared 1x1
        projection. Unassigned channels are free learned capacity (no read-out, no loss).
        """
        if spatial_shape is None or channel_map is None:
            raise ValueError("spatial encoder-output requires spatial_shape and channel_map")
        c, h, w = spatial_shape
        self.spatial_c, self.spatial_h, self.spatial_w = c, h, w
        self.channel_map = dict(channel_map)
        trunk_c = self.encoder.feature_channels
        # Shared field projection: trunk (B, trunk_c, h0, w0) -> (B, C, h0, w0), then resize.
        self.field_proj = nn.Sequential(
            nn.Conv2d(trunk_c, trunk_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(trunk_c, c, kernel_size=1),
        )
        # CenterNet focal cold-start: the occupancy field is overwhelmingly NEGATIVE (a
        # handful of bullet peaks among C*H*W cells), so a zero-init logit collapses to
        # all-negative (the 959 negative cells out-push the 1 positive cell on the shared
        # bias). Bias the BULLETS output channel(s) strongly negative (sigmoid ~0.05 at init,
        # the prior), so the negative loss is small at init and the positive-cell gradient
        # can actually raise the peaks. (Law & Deng 2018; RetinaNet pi=0.01 init.) Only the
        # occupancy channel(s) are biased; walls/keypoints keep the default init.
        final = self.field_proj[-1]
        with torch.no_grad():
            final.bias.zero_()
            for ch in self.channel_map.get("bullets", ()):  # occupancy channel(s)
                final.bias[ch] = -2.94  # sigmoid(-2.94) ~ 0.05
        # head_sizes / player_vec_indices are flat-mode concepts; in spatial mode the
        # objectives are the channel-map ones. Expose an analogous `sizes` for logging.
        self.sizes = {name: len(idxs) for name, idxs in self.channel_map.items()}
        self.player_vec_indices = []  # no flat player head in spatial mode

    def _field(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk -> 1x1 projection -> bilinear resize to the configured ``(B, C, H, W)`` field."""
        fm = self.encoder.features(x)  # (B, trunk_c, h0, w0)
        proj = self.field_proj(fm)  # (B, C, h0, w0)
        return nn.functional.interpolate(
            proj, size=(self.spatial_h, self.spatial_w), mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.encoder_output == "spatial":
            return self._forward_spatial(x)
        if self.head == "fc":
            return self._forward_fc(x)
        return self._forward_heatmap(x)

    def _forward_spatial(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read each assigned channel of the C x H x W field as that objective's prediction.

        Returns RAW field outputs (no sigmoid) so the losses control activation:
        * ``occupancy`` ``(B, H, W)`` raw logits for the bullets channel (focal loss applies
          sigmoid internally; the metric extracts peaks from the sigmoid).
        * ``wall`` ``(B, 12, 20)`` logits = the walls channel adaptively pooled to the wall
          grid (so the interior-BCE loss/metric is unchanged).
        * ``keypoint_heat`` ``(B, 2, H, W)`` raw heatmaps for the 2 keypoint channels (the
          soft-argmax read-out / coords happen in the loss + metric).
        Only the keys for ASSIGNED objectives appear.
        """
        field = self._field(x)  # (B, C, H, W)
        out: dict[str, torch.Tensor] = {}
        cm = self.channel_map
        if "bullets" in cm:
            out["occupancy"] = field[:, cm["bullets"][0], :, :]  # (B, H, W) logits
        if "walls" in cm:
            wlog = field[:, cm["walls"][0] : cm["walls"][0] + 1, :, :]  # (B,1,H,W)
            pooled = nn.functional.adaptive_avg_pool2d(wlog, (WALL_H, WALL_W))  # (B,1,12,20)
            out["wall"] = pooled.squeeze(1)  # (B, 12, 20) logits
        if "keypoints" in cm:
            ch = list(cm["keypoints"])  # 2 channel indices, P1 then P2
            out["keypoint_heat"] = field[:, ch, :, :]  # (B, 2, H, W) raw heatmaps
        return out

    def _forward_fc(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            y = head(z)
            if name == "wall":
                y = y.reshape(-1, WALL_H, WALL_W)
            out[name] = y
        return out

    def _forward_heatmap(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fm = self.encoder.features(x)  # (B, C, h, w) spatial feature map
        z = fm.mean(dim=(2, 3))  # (B, C) pooled embedding (same as encoder.forward)
        out: dict[str, torch.Tensor] = {}

        # --- Localized points: heatmaps -> soft-argmax -> per-group affine -------
        point_xy: dict[str, torch.Tensor] = {}
        if self.point_channels:
            heat = self.heatmap_conv(fm)  # (B, n_heat, h, w)
            coords = spatial_soft_argmax(heat, normalized="unit")  # (B, n_heat, 2) in [0,1]
            off = 0
            for g, n in self.point_channels.items():
                grid_xy = coords[:, off : off + n, :]  # (B, n, 2)
                # Per-group affine grid->normalized target space, shared across the group's
                # points (applied to the last dim 2->2).
                point_xy[g] = self.point_affine[g](grid_xy)  # (B, n, 2)
                off += n

        # --- player: assemble position (heatmap) + velocity/aim (regression) -----
        if self.cfg.any_player():
            b = x.shape[0]
            player = x.new_zeros(b, len(self.player_vec_indices))
            if "player_position" in point_xy:
                # point_xy player_position is (B, 2, 2) = [[P1_x,P1_y],[P2_x,P2_y]] ->
                # flatten to [P1_x,P1_y,P2_x,P2_y]; scatter into the position columns.
                pos = point_xy["player_position"].reshape(b, -1)  # (B, 4)
                for j, col in enumerate(self._player_pos_cols):
                    player[:, col] = pos[:, j]
            if self._player_reg_cols:
                reg = self.heads["player_reg"](z)  # (B, len(reg_cols))
                for j, col in enumerate(self._player_reg_cols):
                    player[:, col] = reg[:, j]
            out["player"] = player

        # --- bullets: position (heatmap) + presence (conv) + direction (reg) -----
        if "bullet_position" in point_xy:
            out["bullet_position"] = point_xy["bullet_position"].reshape(x.shape[0], -1)  # (B,20)
        if self.cfg.is_on("bullet_presence"):
            pres = self.presence_conv(fm).mean(dim=(2, 3))  # (B, C)
            out["bullet_presence"] = self.presence_fc(pres)  # (B, 10)
        if self.cfg.is_on("bullet_direction"):
            out["bullet_direction"] = self.heads["bullet_direction"](z)  # (B, 20)

        # --- walls: spatial conv head -> (B, 12, 20) logits ----------------------
        if self.cfg.is_on("walls"):
            out["wall"] = self.wall_conv(fm).squeeze(1)  # (B, 12, 20)
        return out


def compute_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    model: InverseRenderer,
    lambdas: dict[str, float],
    presence_pw: float | torch.Tensor,
    wall_pw: float | torch.Tensor,
    *,
    stats: NormStats | None = None,
    aim_loss: str = "cosine",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total + per-component loss over ENABLED objectives only.

    * player: MSE on the normalized enabled-sub-group slice of the 12-vector target,
      EXCLUDING aim (position + velocity only when their sub-groups are enabled).
    * aim (when ``player_aim`` enabled): a SEPARATE term. Default ``aim_loss="cosine"``
      uses a per-player cosine distance on the RAW (de-standardized) unit aim, which
      forces a directional commitment (MSE collapses to (0,0) on a ~uniform unit target).
      ``aim_loss="mse"`` reproduces the OLD behavior: per-field MSE on the standardized
      aim slice, folded into the ``player`` part. ``stats`` is required for cosine (to
      de-standardize aim back to raw); it is unused for mse.
    * bullet_presence: BCEWithLogits on 10 slot bits, with ``pos_weight=presence_pw`` to
      counter the slot-level imbalance (~7.8% present) that collapses plain BCE to F1=0.
    * bullet_position / bullet_direction: MSE on 20 normalized floats, MASKED to present
      slots (absent slots contribute 0; mean over present floats only).
    * walls: BCEWithLogits restricted to the INTERIOR cells only (the constant 2-deep
      border is information-free and is dropped from the loss), with ``pos_weight=wall_pw``
      on interior wall cells. The head stays 240 logits; border cells contribute exactly 0.

    ``presence_pw`` / ``wall_pw`` are TRAIN-level scalars (float or 0-dim tensor); they are
    materialized as scalar tensors on the preds' device/dtype inside this function.
    """
    mse = nn.functional.mse_loss
    bce = nn.functional.binary_cross_entropy_with_logits
    cfg = model.cfg

    terms: list[torch.Tensor] = []
    parts: dict[str, float] = {}

    def add(name: str, loss: torch.Tensor, weight: float) -> None:
        terms.append(loss * weight)
        parts[name] = float(loss.detach())

    if cfg.any_player():
        idx = model.player_vec_indices
        pred_player = preds["player"]
        tgt_player = targets["player"][:, idx]
        aim_cols = _aim_head_columns(model)
        use_cosine = aim_loss == "cosine" and aim_cols is not None
        if use_cosine:
            # Aim leaves the player MSE term: regress position(+velocity) only.
            non_aim = _player_non_aim_columns(model)
            if non_aim:
                add(
                    "player",
                    mse(pred_player[:, non_aim], tgt_player[:, non_aim]),
                    lambdas["player"],
                )
            if stats is None:  # pragma: no cover - run_training always threads stats
                raise ValueError("aim_loss='cosine' requires stats to de-standardize aim")
            pred_aim_raw = _destandardize_aim(pred_player[:, aim_cols], stats)
            tgt_aim_raw = _destandardize_aim(tgt_player[:, aim_cols], stats)
            add(
                "aim",
                _torch_per_player_cosine_distance(pred_aim_raw, tgt_aim_raw),
                lambdas["aim"],
            )
        else:
            # OLD behavior (aim_loss='mse', or no aim sub-group): MSE over the whole
            # enabled player slice, aim included, as a single 'player' term.
            add("player", mse(pred_player, tgt_player), lambdas["player"])

    if cfg.is_on("bullet_presence"):
        p = preds["bullet_presence"]
        pw = torch.as_tensor(presence_pw, dtype=p.dtype, device=p.device)
        add(
            "presence",
            bce(p, targets["bullet_presence"], pos_weight=pw),
            lambdas["presence"],
        )

    mask = targets.get("bullet_slot_mask")
    if cfg.is_on("bullet_position"):
        sq = (preds["bullet_position"] - targets["bullet_position"]) ** 2 * mask
        add("bullet_pos", sq.sum() / mask.sum().clamp_min(1.0), lambdas["bullet_pos"])

    if cfg.is_on("bullet_direction"):
        sq = (preds["bullet_direction"] - targets["bullet_direction"]) ** 2 * mask
        add("bullet_dir", sq.sum() / mask.sum().clamp_min(1.0), lambdas["bullet_dir"])

    if cfg.is_on("walls"):
        wp = preds["wall"]  # (B, 12, 20) logits
        wt = targets["wall"]  # (B, 12, 20) {0,1}
        # Interior-only BCE: select the 128 interior cells so the constant border
        # (identical on every map, info-free) contributes EXACTLY 0 to the loss.
        interior = _interior_wall_index(wp.device)  # (12,20) bool on device
        wpw = torch.as_tensor(wall_pw, dtype=wp.dtype, device=wp.device)
        per_cell = bce(wp, wt, pos_weight=wpw, reduction="none")  # (B,12,20)
        sel = per_cell[:, interior]  # (B, 128)
        add("wall", sel.mean(), lambdas["wall"])

    if not terms:  # pragma: no cover - parse_objectives forbids an empty config
        raise ValueError("no objectives enabled; nothing to optimize")
    total = terms[0] if len(terms) == 1 else torch.stack(terms).sum()
    parts["total"] = float(total.detach())
    return total, parts


def compute_spatial_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    channel_map: dict[str, tuple[int, ...]],
    lambdas: dict[str, float],
    wall_pw: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total + per-objective loss for SPATIAL (field) mode over the ASSIGNED objectives.

    * ``bullets`` -> CenterNet penalty-reduced FOCAL loss on the occupancy channel
      (sigmoid(logits) vs the sum-of-Gaussians occupancy target). REPLACES the slot-based
      presence + position + direction terms (order-free, count-agnostic). Loss key
      ``occupancy``, ``lambdas['occupancy']``.
    * ``walls`` -> interior-only BCEWithLogits on the (B,12,20) channel (pooled), pos_weight
      ``wall_pw`` on interior wall cells; IDENTICAL loss to flat mode. Key ``wall``.
    * ``keypoints`` -> per-tank MSE between the soft-argmax of each keypoint channel and the
      target, in NORMALIZED [0,1] grid coords (NOT raw cells), averaged over the 2 tanks.
      Normalizing keeps the term O(1) — comparable in magnitude to the focal + BCE terms —
      so no single objective's gradient floods the shared field projection / trunk (a raw-
      cell MSE is O(H^2) and would dominate, collapsing occupancy). Key ``keypoint``,
      ``lambdas['keypoint']``.

    ``lambdas`` keys: ``occupancy``, ``wall``, ``keypoint`` (missing -> 1.0).
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    terms: list[torch.Tensor] = []
    parts: dict[str, float] = {}

    def add(name: str, loss: torch.Tensor, weight: float) -> None:
        terms.append(loss * weight)
        parts[name] = float(loss.detach())

    if "bullets" in channel_map:
        # centernet_focal_loss applies sigmoid internally (stable logsigmoid form).
        loss = centernet_focal_loss(preds["occupancy"], targets["occupancy"])
        add("occupancy", loss, lambdas.get("occupancy", 1.0))

    if "walls" in channel_map:
        wp = preds["wall"]  # (B, 12, 20) logits
        wt = targets["wall_grid"]  # (B, 12, 20) {0,1} at WALL_H x WALL_W
        interior = _interior_wall_index(wp.device)
        wpw = torch.as_tensor(wall_pw, dtype=wp.dtype, device=wp.device)
        per_cell = bce(wp, wt, pos_weight=wpw, reduction="none")
        add("wall", per_cell[:, interior].mean(), lambdas.get("wall", 1.0))

    if "keypoints" in channel_map:
        # soft-argmax over each keypoint channel -> (B, 2, 2) in [0,1] grid frac.
        # spatial_soft_argmax puts cell centers at (j+0.5)/[w,h]; keypoint_cr (via world_to_grid
        # / the splat) uses INTEGER indices. Undo the +0.5 so the read-out is in the SAME
        # integer-index frame as the target, then normalize by [w,h] so the MSE is O(1). Without
        # this the loss optimum sits a half-cell off the true splat location (the eval metric
        # then reads that as half-cell error). This is a CONSTANT +0.5-cell reframing of the
        # regression target: identical MSE curvature / scale, only the (correct) optimum location
        # shifts -- it does not change optimizer dynamics.
        heat = preds["keypoint_heat"]  # (B, 2, H, W)
        _b, _k, h, w = heat.shape
        coords01 = spatial_soft_argmax(heat, normalized="unit")  # (B, 2, 2) (x,y) in [0,1]
        scale = coords01.new_tensor([w, h])  # (col,row) -> [0,1] normalizer
        pred_idx01 = coords01 - 0.5 / scale  # (j+0.5)/[w,h] -> integer-index frame, in [0,1]
        tgt01 = targets["keypoint_cr"] / scale  # (B, 2, 2) (col,row) integer-index frac -> [0,1]
        add("keypoint", ((pred_idx01 - tgt01) ** 2).mean(), lambdas.get("keypoint", 1.0))

    if not terms:  # pragma: no cover - validate_channel_map forbids an empty map
        raise ValueError("no spatial objectives assigned; nothing to optimize")
    total = terms[0] if len(terms) == 1 else torch.stack(terms).sum()
    parts["total"] = float(total.detach())
    return total, parts


# Cache the interior-wall boolean mask as a torch tensor per device (the numpy mask is
# constant; this avoids rebuilding it every batch).
_INTERIOR_WALL_CACHE: dict[torch.device, torch.Tensor] = {}


def _interior_wall_index(device: torch.device) -> torch.Tensor:
    """The (12,20) interior-cell boolean mask as a torch tensor on ``device`` (cached)."""
    cached = _INTERIOR_WALL_CACHE.get(device)
    if cached is None:
        cached = torch.from_numpy(interior_wall_mask()).to(device)
        _INTERIOR_WALL_CACHE[device] = cached
    return cached


# =============================================================================
# Datasets
# =============================================================================
def decode_spatial_targets(
    states: np.ndarray,
    map_ids: np.ndarray,
    wall_grids: np.ndarray,
    spatial_shape: tuple[int, int, int],
    *,
    sigma: float,
) -> dict[str, np.ndarray]:
    """Build the SPATIAL field targets for a batch of states at the configured H x W.

    Returns:
    * ``occupancy`` ``(N, H, W)`` sum-of-Gaussians bullet occupancy (order-free,
      count-agnostic), built via :func:`bullet_occupancy_field`.
    * ``wall_grid`` ``(N, 12, 20)`` binary wall target (the wall head pools to WALL_H x
      WALL_W; resizing is unnecessary so the wall target stays the native grid). Resized
      copy not stored — the wall loss compares at the wall grid.
    * ``keypoint_cr`` ``(N, 2, 2)`` per-tank target in FRACTIONAL (col,row) grid coords (the
      soft-argmax read-out frame), P1 then P2.
    """
    _c, h, w = spatial_shape
    occ = bullet_occupancy_field(states, h, w, sigma=sigma)  # (N,H,W)
    walls = wall_target_for(map_ids, wall_grids)  # (N,12,20) binary
    # Keypoint targets as fractional (col,row) at H x W (matches the soft-argmax frame).
    p1 = states[:, [0, 1]]
    p2 = states[:, [26, 27]]
    pts = np.stack([p1, p2], axis=1)  # (N,2,2) world xy
    col, row = world_to_grid(pts, h, w)  # (N,2) each
    kp_cr = np.stack([col, row], axis=2).astype(np.float32)  # (N,2,2) (col,row)
    return {
        "occupancy": occ.astype(np.float32),
        "wall_grid": walls.astype(np.float32),
        "keypoint_cr": kp_cr,
    }


class _BaseRows:
    """Holds resolved rows: frames (downsampled uint8), states, map_ids; targets cached.

    ``spatial`` (optional ``(spatial_shape, sigma)``) switches the target cache to the
    SPATIAL field targets; otherwise the flat decode_targets cache is built (default).
    """

    def __init__(
        self,
        frames: np.ndarray,  # (N, H, W, 3) uint8
        states: np.ndarray,  # (N, 52) f32
        map_ids: np.ndarray,  # (N,) int
        wall_grids: np.ndarray,  # (10,12,20)
        stats: NormStats,
        spatial: tuple[tuple[int, int, int], float] | None = None,
    ) -> None:
        self.frames = frames
        self.states = states.astype(np.float32, copy=False)
        self.map_ids = map_ids.astype(np.int64, copy=False)
        self.wall_grids = wall_grids
        self.stats = stats
        self.spatial = spatial
        if spatial is None:
            self._cache = decode_targets(self.states, self.map_ids, self.wall_grids, self.stats)
        else:
            shape, sigma = spatial
            self._cache = decode_spatial_targets(
                self.states, self.map_ids, self.wall_grids, shape, sigma=sigma
            )

    def __len__(self) -> int:
        return self.frames.shape[0]


def _row_item(
    frame: torch.Tensor,
    cache: dict[str, np.ndarray],
    i: int,
    spatial: bool = False,
) -> dict[str, torch.Tensor]:
    if spatial:
        return {
            "frame": frame,
            "occupancy": torch.from_numpy(cache["occupancy"][i]),
            "wall_grid": torch.from_numpy(cache["wall_grid"][i]),
            "keypoint_cr": torch.from_numpy(cache["keypoint_cr"][i]),
        }
    return {
        "frame": frame,
        "player": torch.from_numpy(cache["player"][i]),
        "bullet_presence": torch.from_numpy(cache["bullet_presence"][i]),
        "bullet_position": torch.from_numpy(cache["bullet_position"][i]),
        "bullet_direction": torch.from_numpy(cache["bullet_direction"][i]),
        "bullet_slot_mask": torch.from_numpy(cache["bullet_slot_mask"][i]),
        "wall": torch.from_numpy(cache["wall"][i]),
    }


def _decode_cache(
    states: np.ndarray,
    map_ids: np.ndarray,
    wall_grids: np.ndarray,
    stats: NormStats,
    spatial: tuple[tuple[int, int, int], float] | None,
) -> dict[str, np.ndarray]:
    """Build the flat (default) OR spatial target cache for a set of rows (shared helper)."""
    if spatial is None:
        return decode_targets(states, map_ids, wall_grids, stats)
    shape, sigma = spatial
    return decode_spatial_targets(states, map_ids, wall_grids, shape, sigma=sigma)


class InMemoryPixelDataset(Dataset, _BaseRows):
    """In-memory dataset (SMOKE streamer or a small slice). Targets precomputed."""

    def __init__(self, *a, **kw) -> None:
        Dataset.__init__(self)
        _BaseRows.__init__(self, *a, **kw)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        frame = torch.from_numpy(self.frames[i]).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        return _row_item(frame, self._cache, i, spatial=self.spatial is not None)


class MemmapPixelDataset(Dataset):
    """Reads pre-resized frames from a uint8 memmap + aligned states/map_ids.

    The cache dir holds ``frames_u8.dat`` (memmap, (N,H,W,3)), ``states.npy``,
    ``map_ids.npy``, ``group_keys.npy`` (N,2), and ``index.json`` with shape/res.
    """

    def __init__(
        self,
        cache_dir: Path,
        row_idx: np.ndarray,
        wall_grids: np.ndarray,
        stats: NormStats,
        spatial: tuple[tuple[int, int, int], float] | None = None,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        n, h, w, c = meta["shape"]
        self.frames = np.memmap(
            self.cache_dir / "frames_u8.dat", dtype=np.uint8, mode="r", shape=(n, h, w, c)
        )
        self.states = np.load(self.cache_dir / "states.npy")
        self.map_ids = np.load(self.cache_dir / "map_ids.npy")
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.wall_grids = wall_grids
        self.stats = stats
        self.spatial = spatial
        sel_states = self.states[self.row_idx]
        sel_maps = self.map_ids[self.row_idx]
        self._cache = _decode_cache(sel_states, sel_maps, self.wall_grids, self.stats, spatial)

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.row_idx[i])
        frame = torch.from_numpy(np.array(self.frames[r], copy=True)).permute(2, 0, 1).contiguous()
        return _row_item(frame, self._cache, i, spatial=self.spatial is not None)


# Process-local LMDB env cache, keyed by (pid, resolved-path). LMDB refuses to open the
# SAME env file twice within one process, and the train/eval/val datasets all read the
# same file — so they must SHARE one handle per process. Keying by pid (not just path)
# makes it spawn-safe: a freshly spawned DataLoader worker starts with an empty dict and
# opens its OWN handle (the parent's handle never survives pickling). lock=False allows
# the many concurrent readers; the file is read-only here.
_LMDB_ENV_CACHE: dict[tuple[int, str], object] = {}


def _open_lmdb_env(lmdb_path: Path):
    """Return THIS process's shared read-only env for ``lmdb_path`` (opened once per pid)."""
    import lmdb

    key = (os.getpid(), str(lmdb_path))
    env = _LMDB_ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_dbs=0,
        )
        _LMDB_ENV_CACHE[key] = env
    return env


class LmdbPngPixelDataset(Dataset):
    """Reads PNG-encoded frames from an LMDB env + aligned states/map_ids (PNG-in-LMDB).

    Parallel to :class:`MemmapPixelDataset`: SAME ``__init__`` signature, SAME target
    cache, and SAME ``(frame, targets)`` return — it yields a uint8 ``(3,H,W)`` frame
    tensor (NOT pre-divided; ``_move_batch`` divides by 255 downstream), bit-identical to
    what the memmap path returns for the same row. The cache dir holds
    ``frames_png.lmdb`` (LMDB env written by scripts/transcode_pixel_png_lmdb.py),
    ``states.npy``, ``map_ids.npy``, ``group_keys.npy`` (N,2), and ``index.json`` with
    ``backend == "lmdb_png"`` and the key convention.

    Windows / spawn safety: an LMDB env handle does NOT survive pickling to a spawned
    DataLoader worker, so the env is opened LAZILY on first ``__getitem__`` via the
    process-local :data:`_LMDB_ENV_CACHE` (keyed by ``(pid, path)``). Each worker — and the
    main process — therefore opens exactly ONE read-only handle and SHARES it across the
    train/eval/val datasets (LMDB forbids opening the same env file twice per process). The
    env is ``readonly=True, lock=False`` (many concurrent readers, no writer).
    """

    def __init__(
        self,
        cache_dir: Path,
        row_idx: np.ndarray,
        wall_grids: np.ndarray,
        stats: NormStats,
        spatial: tuple[tuple[int, int, int], float] | None = None,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        n, h, w, c = meta["shape"]
        self.shape = (int(n), int(h), int(w), int(c))
        self.lmdb_path = self.cache_dir / meta.get("lmdb_file", "frames_png.lmdb")
        self.key_format = meta.get("key_format", "{:08d}")
        self.states = np.load(self.cache_dir / "states.npy")
        self.map_ids = np.load(self.cache_dir / "map_ids.npy")
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.wall_grids = wall_grids
        self.stats = stats
        self.spatial = spatial
        sel_states = self.states[self.row_idx]
        sel_maps = self.map_ids[self.row_idx]
        self._cache = _decode_cache(sel_states, sel_maps, self.wall_grids, self.stats, spatial)

    def _key(self, r: int) -> bytes:
        return self.key_format.format(r).encode("ascii")

    def _get_env(self):
        """Return this process's shared read-only LMDB env (opened lazily once per pid)."""
        return _open_lmdb_env(self.lmdb_path)

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.row_idx[i])
        with self._get_env().begin(write=False) as txn:
            data = txn.get(self._key(r))
        if data is None:
            raise KeyError(f"LMDB key for row {r} ({self._key(r)!r}) missing in {self.lmdb_path}")
        arr = png_decode(data)  # (H, W, 3) uint8 (writable copy)
        frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return _row_item(frame, self._cache, i, spatial=self.spatial is not None)


# =============================================================================
# Streaming loader for the SMOKE: map-BALANCED so all 10 maps are surfaced
# =============================================================================
def _stream_map_balanced(
    data_dir: Path,
    out_w: int,
    out_h: int,
    subset: int,
    n_maps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Pull ~``subset`` pairs with a per-map QUOTA so every one of ``n_maps`` appears.

    Walks shards (interleaved across workers), bucketing rows by map_id until each map
    reaches its quota (``subset // n_maps``, >=1). This guarantees the smoke 3-way split
    can surface the 8 train maps + 2 validation maps with nonzero counts each. Returns
    ``(frames_ds, states, map_ids, group_keys)`` where group_keys are
    ``(worker_id, episode_id)`` aligned to rows.
    """
    shards = list_shards(data_dir)
    rng = np.random.default_rng(seed)
    by_worker: dict[int, list[Path]] = {}
    for s in shards:
        by_worker.setdefault(worker_id_from_shard(s.name), []).append(s)
    order: list[Path] = []
    idxs = {w: 0 for w in by_worker}
    while len(order) < len(shards):
        for w in sorted(by_worker):
            if idxs[w] < len(by_worker[w]):
                order.append(by_worker[w][idxs[w]])
                idxs[w] += 1

    per_map_quota = max(1, subset // max(1, n_maps))
    # Accumulators bucketed by map_id.
    frames_by_map: dict[int, list[np.ndarray]] = {}
    states_by_map: dict[int, list[np.ndarray]] = {}
    groups_by_map: dict[int, list[np.ndarray]] = {}
    counts: dict[int, int] = {}

    def remaining_maps() -> set[int]:
        return {m for m in range(n_maps) if counts.get(m, 0) < per_map_quota}

    for s in order:
        if not remaining_maps():
            break
        wid = worker_id_from_shard(s.name)
        with np.load(s) as d:
            mids = d["map_ids"].astype(np.int64)
            want = remaining_maps()
            in_want = np.isin(mids, list(want))
            if not in_want.any():
                continue
            for m in want:
                rows = np.flatnonzero(mids == m)
                if rows.size == 0:
                    continue
                need = per_map_quota - counts.get(m, 0)
                take = min(need, rows.size)
                sel = np.sort(rng.choice(rows, size=take, replace=False))
                fr = downsample_frames_np(d["frames"][sel], out_w, out_h)
                st = d["states"][sel].astype(np.float32)
                ep = d["episode_ids"][sel].astype(np.int64)
                frames_by_map.setdefault(m, []).append(fr)
                states_by_map.setdefault(m, []).append(st)
                groups_by_map.setdefault(m, []).append(
                    np.stack([np.full(take, wid, dtype=np.int64), ep], axis=1)
                )
                counts[m] = counts.get(m, 0) + take

    frames_acc: list[np.ndarray] = []
    states_acc: list[np.ndarray] = []
    maps_acc: list[np.ndarray] = []
    groups_acc: list[np.ndarray] = []
    for m in sorted(frames_by_map):
        fr = np.concatenate(frames_by_map[m], axis=0)
        frames_acc.append(fr)
        states_acc.append(np.concatenate(states_by_map[m], axis=0))
        maps_acc.append(np.full(fr.shape[0], m, dtype=np.int64))
        groups_acc.append(np.concatenate(groups_by_map[m], axis=0))

    frames = np.concatenate(frames_acc, axis=0)
    states = np.concatenate(states_acc, axis=0)
    map_ids = np.concatenate(maps_acc, axis=0)
    groups_arr = np.concatenate(groups_acc, axis=0)
    group_keys = [(int(w), int(e)) for w, e in groups_arr]
    missing = [m for m in range(n_maps) if counts.get(m, 0) == 0]
    if missing:
        raise RuntimeError(
            f"smoke stream could not surface map_ids {missing} from {data_dir}; "
            f"increase --subset (got {subset}, quota/map={per_map_quota})"
        )
    return frames, states, map_ids, group_keys


# =============================================================================
# Train / eval loop
# =============================================================================
def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out["frame"] = out["frame"].float() / 255.0  # uint8 (B,3,H,W) -> float [0,1]
    return out


def _na_breakdown(cfg: ObjectiveConfig) -> dict[str, object]:
    """Per-objective dict skeleton with all 7 groups; disabled groups -> "N/A"."""
    return {
        g: "N/A"
        for g in (
            "player_position",
            "player_velocity",
            "player_aim",
            "bullet_presence",
            "bullet_position",
            "bullet_direction",
            "walls",
        )
        if not cfg.is_on(g)
    }


@torch.no_grad()
def _evaluate(
    model: InverseRenderer,
    loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device,
    stats: NormStats,
    use_amp: bool,
    desc: str,
    presence_pw: float | torch.Tensor,
    wall_pw: float | torch.Tensor,
    aim_loss: str = "cosine",
) -> dict[str, object]:
    """Run the loss + the full per-objective breakdown over ``loader``.

    Disabled groups report ``"N/A"``. POSITION groups also report world-unit error.
    bullet_presence -> precision/recall/F1; walls -> INTERIOR F1/precision/recall/IoU
    (the wall loss is interior-only, so the border head output is unsupervised /
    meaningless and ``overall_acc`` is intentionally NOT reported).
    """
    model.eval()
    cfg = model.cfg
    interior = interior_wall_mask()  # (12,20) bool, 128 cells

    loss_total = 0.0
    n_batches = 0

    # Per-field normalized-MSE accumulators (sum of squared err / count).
    pl_sse = np.zeros(12)
    pl_world_sse = np.zeros(12)
    pl_n = 0
    # Aim angular-error accumulators (per-player, on RAW de-standardized aim).
    aim_deg_sum = 0.0
    aim_cos_sum = 0.0
    aim_n = 0  # number of (sample, player) cosine measurements summed
    bpos_sse = np.zeros(N_BULLET_POS)
    bpos_world_sae = np.zeros(N_BULLET_POS)
    bpos_cnt = np.zeros(N_BULLET_POS)
    bdir_sse = np.zeros(N_BULLET_DIR)
    bdir_cnt = np.zeros(N_BULLET_DIR)
    # Presence: collect predictions/targets for F1.
    pres_tp = pres_fp = pres_fn = 0
    # Walls: interior F1/IoU (no overall accuracy — border is unsupervised now).
    wall_tp = wall_fp = wall_fn = wall_inter = wall_union = 0

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        b = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(b["frame"])
            total, _parts = compute_losses(
                preds, b, model, lambdas, presence_pw, wall_pw, stats=stats, aim_loss=aim_loss
            )
        loss_total += float(total.detach())
        n_batches += 1

        if cfg.any_player():
            idx = model.player_vec_indices
            pred = preds["player"].float().cpu().numpy()  # (B, k) normalized
            tgt = b["player"][:, idx].float().cpu().numpy()
            pl_sse[idx] += ((pred - tgt) ** 2).sum(axis=0)
            pw = denormalize(pred, stats.player_mean[idx], stats.player_std[idx])
            tw = denormalize(tgt, stats.player_mean[idx], stats.player_std[idx])
            pl_world_sse[idx] += ((pw - tw) ** 2).sum(axis=0)
            pl_n += pred.shape[0]
            if cfg.is_on("player_aim"):
                # Aim metric on RAW (de-standardized) aim: pull the aim columns out of
                # the head output and de-standardize both pred and target to unit space.
                aim_cols = _aim_head_columns(model)
                aim_vi = list(_AIM_VEC_INDICES)
                pred_aim_raw = denormalize(
                    pred[:, aim_cols], stats.player_mean[aim_vi], stats.player_std[aim_vi]
                )
                tgt_aim_raw = denormalize(
                    tgt[:, aim_cols], stats.player_mean[aim_vi], stats.player_std[aim_vi]
                )
                m = aim_angular_error_deg(pred_aim_raw, tgt_aim_raw)
                # Weight each batch's mean by its (sample * 2-player) measurement count.
                bm = pred.shape[0] * 2
                aim_deg_sum += m["deg"] * bm
                aim_cos_sum += m["cos"] * bm
                aim_n += bm

        mask = b["bullet_slot_mask"].float().cpu().numpy() if "bullet_slot_mask" in b else None
        if cfg.is_on("bullet_position"):
            pred = preds["bullet_position"].float().cpu().numpy()
            tgt = b["bullet_position"].float().cpu().numpy()
            bpos_sse += ((pred - tgt) ** 2 * mask).sum(axis=0)
            pw = denormalize(pred, stats.bullet_pos_mean, stats.bullet_pos_std)
            tw = denormalize(tgt, stats.bullet_pos_mean, stats.bullet_pos_std)
            bpos_world_sae += (np.abs(pw - tw) * mask).sum(axis=0)
            bpos_cnt += mask.sum(axis=0)
        if cfg.is_on("bullet_direction"):
            pred = preds["bullet_direction"].float().cpu().numpy()
            tgt = b["bullet_direction"].float().cpu().numpy()
            bdir_sse += ((pred - tgt) ** 2 * mask).sum(axis=0)
            bdir_cnt += mask.sum(axis=0)
        if cfg.is_on("bullet_presence"):
            pred = (preds["bullet_presence"].float().cpu().numpy() > 0).astype(np.int64)
            tgt = b["bullet_presence"].float().cpu().numpy().astype(np.int64)
            pres_tp += int(((pred == 1) & (tgt == 1)).sum())
            pres_fp += int(((pred == 1) & (tgt == 0)).sum())
            pres_fn += int(((pred == 0) & (tgt == 1)).sum())
        if cfg.is_on("walls"):
            pred = (preds["wall"].float().cpu().numpy() > 0).astype(np.int64)  # (B,12,20)
            tgt = b["wall"].float().cpu().numpy().astype(np.int64)
            pi = pred[:, interior].ravel()
            ti = tgt[:, interior].ravel()
            wall_tp += int(((pi == 1) & (ti == 1)).sum())
            wall_fp += int(((pi == 1) & (ti == 0)).sum())
            wall_fn += int(((pi == 0) & (ti == 1)).sum())
            wall_inter += int(((pi == 1) & (ti == 1)).sum())
            wall_union += int(((pi == 1) | (ti == 1)).sum())

    out: dict[str, object] = {"total_loss": loss_total / max(1, n_batches)}
    breakdown: dict[str, object] = _na_breakdown(cfg)

    # Player sub-groups.
    for sub in _PLAYER_SUBGROUPS:
        if not cfg.is_on(sub):
            continue
        if sub == "player_aim":
            # Aim reports ANGULAR error (deg) + cosine similarity instead of norm_mse:
            # on a ~uniform unit target, norm_mse pins at ~1.0 regardless of signal, so
            # it cannot distinguish "learning direction" from "collapsed to (0,0)".
            breakdown[sub] = {
                "aim_angular_error_deg": float(aim_deg_sum / max(1, aim_n)),
                "aim_cosine": float(aim_cos_sum / max(1, aim_n)),
            }
            continue
        vi = list(PLAYER_SUBGROUP_VEC_INDICES[sub])
        nmse = float((pl_sse[vi] / max(1, pl_n)).mean())
        entry: dict[str, float] = {"norm_mse": nmse}
        if sub in _PLAYER_POSITION_GROUPS:
            entry["world_rmse"] = float(np.sqrt((pl_world_sse[vi] / max(1, pl_n)).mean()))
        breakdown[sub] = entry

    if cfg.is_on("bullet_presence"):
        prec = pres_tp / (pres_tp + pres_fp) if (pres_tp + pres_fp) else 0.0
        rec = pres_tp / (pres_tp + pres_fn) if (pres_tp + pres_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        breakdown["bullet_presence"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": pres_tp,
            "fp": pres_fp,
            "fn": pres_fn,
        }
    if cfg.is_on("bullet_position"):
        nmse = float((bpos_sse / np.clip(bpos_cnt, 1, None)).mean())
        world_mae = float(bpos_world_sae.sum() / max(1.0, bpos_cnt.sum()))
        breakdown["bullet_position"] = {"norm_mse": nmse, "world_mae": world_mae}
    if cfg.is_on("bullet_direction"):
        nmse = float((bdir_sse / np.clip(bdir_cnt, 1, None)).mean())
        breakdown["bullet_direction"] = {"norm_mse": nmse}
    if cfg.is_on("walls"):
        prec = wall_tp / (wall_tp + wall_fp) if (wall_tp + wall_fp) else 0.0
        rec = wall_tp / (wall_tp + wall_fn) if (wall_tp + wall_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = wall_inter / wall_union if wall_union else 1.0
        breakdown["walls"] = {
            "interior_f1": f1,
            "interior_precision": prec,
            "interior_recall": rec,
            "interior_iou": iou,
        }

    out["per_objective"] = breakdown
    return out


def _fmt_breakdown(b: dict[str, object]) -> str:
    """One-line human summary of the per-objective breakdown (N/A for disabled)."""
    parts = []
    for g in ("player_position", "player_velocity", "player_aim"):
        v = b[g]
        if v == "N/A":
            parts.append(f"{g}=N/A")
        elif g == "player_aim":
            parts.append(f"{g}=ang{v['aim_angular_error_deg']:.1f}deg/cos{v['aim_cosine']:.3f}")
        else:
            extra = f"/world={v['world_rmse']:.3f}" if "world_rmse" in v else ""
            parts.append(f"{g}=nmse{v['norm_mse']:.3f}{extra}")
    for g in ("bullet_presence",):
        v = b[g]
        parts.append(f"{g}=N/A" if v == "N/A" else f"{g}=F1{v['f1']:.3f}")
    bp = b["bullet_position"]
    parts.append(
        "bullet_position=N/A"
        if bp == "N/A"
        else f"bullet_position=nmse{bp['norm_mse']:.3f}/world{bp['world_mae']:.3f}"
    )
    bd = b["bullet_direction"]
    parts.append(
        "bullet_direction=N/A" if bd == "N/A" else f"bullet_direction=nmse{bd['norm_mse']:.3f}"
    )
    w = b["walls"]
    if w == "N/A":
        parts.append("walls=N/A")
    else:
        parts.append(f"walls=intF1{w['interior_f1']:.3f}/IoU{w['interior_iou']:.3f}")
    return "  ".join(parts)


# =============================================================================
# Spatial-mode evaluation: per-objective metrics under the SAME 7-group keys as flat
# =============================================================================
def _match_peaks_to_truth(
    pred_cr: list[tuple[float, float]],
    true_cr: np.ndarray,
    *,
    radius: float,
) -> tuple[int, int, int, float, int]:
    """Greedy 1-1 match of predicted peaks to true bullet (col,row) within ``radius`` cells.

    Returns ``(tp, fp, fn, matched_dist_sum, n_matched)`` where the distance sum is in GRID
    cells over matched pairs (for a world-error conversion by the caller). A predicted peak
    matches the nearest unused true point within ``radius``; unmatched preds are FP, unmatched
    truths FN. ``true_cr`` is ``(M, 2)`` (col,row); empty arrays are handled.
    """
    used = np.zeros(len(true_cr), dtype=bool)
    tp = fp = 0
    dist_sum = 0.0
    n_matched = 0
    for pc, pr in pred_cr:
        if len(true_cr) == 0:
            fp += 1
            continue
        d = np.hypot(true_cr[:, 0] - pc, true_cr[:, 1] - pr)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            used[j] = True
            tp += 1
            dist_sum += float(d[j])
            n_matched += 1
        else:
            fp += 1
    fn = int((~used).sum())
    return tp, fp, fn, dist_sum, n_matched


@torch.no_grad()
def _evaluate_spatial(
    model: InverseRenderer,
    loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device,
    spatial_shape: tuple[int, int, int],
    use_amp: bool,
    desc: str,
    wall_pw: float | torch.Tensor,
) -> dict[str, object]:
    """SPATIAL eval: emits the SAME 7 per-objective group keys as :func:`_evaluate`.

    The bullet OCCUPANCY field SUBSUMES the slot objectives, so its peak read-out is mapped
    back onto the flat keys for an apples-to-apples comparison:
    * ``bullet_presence`` -> precision/recall/F1/tp/fp/fn from greedy peak<->truth matching
      (count-based: the occupancy peak count vs the true bullet count).
    * ``bullet_position`` -> ``world_mae`` (mean world-unit distance of matched peaks) +
      ``norm_mse`` (grid-frac MSE analog), the SAME sub-keys flat reports.
    * ``bullet_direction`` -> "N/A" (occupancy drops direction by design).
    * keypoints -> ``player_position`` ``{norm_mse, world_rmse}`` (soft-argmax coords vs the
      keypoint target, grid-frac MSE + world RMSE); ``player_velocity`` / ``player_aim`` N/A.
    * ``walls`` -> interior F1/precision/recall/IoU (IDENTICAL to flat).
    """
    model.eval()
    _c, gh, gw = spatial_shape
    cm = model.channel_map
    interior = interior_wall_mask()
    loss_total = 0.0
    n_batches = 0

    # Occupancy accumulators.
    occ_tp = occ_fp = occ_fn = 0
    occ_dist_sum = 0.0  # grid cells over matched peaks
    occ_matched = 0
    # Keypoint accumulators (grid-frac SSE + world SSE over the 2 tanks).
    kp_grid_sse = 0.0
    kp_world_sse = 0.0
    kp_n = 0
    # Wall accumulators.
    wall_tp = wall_fp = wall_fn = wall_inter = wall_union = 0

    # World-per-cell scale (for grid-cell distance -> world units): a cell spans
    # WALL_W/gw in x and WALL_H/gh in y world units; use the mean as an isotropic scale.
    world_per_col = WALL_W / gw
    world_per_row = WALL_H / gh

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        b = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(b["frame"])
            total, _parts = compute_spatial_losses(preds, b, cm, lambdas, wall_pw)
        loss_total += float(total.detach())
        n_batches += 1

        if "bullets" in cm:
            prob = torch.sigmoid(preds["occupancy"]).float().cpu().numpy()  # (B,H,W)
            tgt = b["occupancy"].float().cpu().numpy()  # (B,H,W)
            for bi in range(prob.shape[0]):
                pred_peaks = extract_peaks(prob[bi], threshold=0.3, nms=1)
                true_peaks = extract_peaks(tgt[bi], threshold=0.5, nms=1)
                true_cr = np.array(true_peaks, dtype=np.float64).reshape(-1, 2)
                tp, fp, fn, dsum, nm = _match_peaks_to_truth(pred_peaks, true_cr, radius=2.0)
                occ_tp += tp
                occ_fp += fp
                occ_fn += fn
                occ_dist_sum += dsum
                occ_matched += nm

        if "keypoints" in cm:
            heat = preds["keypoint_heat"]  # (B,2,H,W)
            coords01 = spatial_soft_argmax(heat, normalized="unit").float().cpu().numpy()  # (B,2,2)
            # spatial_soft_argmax puts cell centers at (j+0.5)/[w,h]; the splat / world_to_grid
            # (-> keypoint_cr) and grid_to_world put them at INTEGER indices. Undo the +0.5 so
            # the read-out lands in the SAME integer-index frame as the target -- otherwise a
            # heatmap peaked at the TRUE splat cell reads a fixed half-cell-too-far world error.
            pred_col = coords01[:, :, 0] * gw - 0.5  # integer-index grid frame
            pred_row = coords01[:, :, 1] * gh - 0.5
            tgt_cr = b["keypoint_cr"].float().cpu().numpy()  # (B,2,2) (col,row) integer-index
            d_col = pred_col - tgt_cr[:, :, 0]
            d_row = pred_row - tgt_cr[:, :, 1]
            kp_grid_sse += float((d_col**2 + d_row**2).sum())
            # World error anchored to GROUND TRUTH: map BOTH the prediction and the target out
            # of the grid via grid_to_world (the exact, round-trip-tested inverse of
            # world_to_grid that built keypoint_cr) -- not the ad-hoc (pred-tgt)*world_per_cell.
            pred_wx, pred_wy = grid_to_world(pred_col, pred_row, gh, gw)
            tgt_wx, tgt_wy = grid_to_world(tgt_cr[:, :, 0], tgt_cr[:, :, 1], gh, gw)
            wx = pred_wx - tgt_wx
            wy = pred_wy - tgt_wy
            kp_world_sse += float((wx**2 + wy**2).sum())
            kp_n += pred_col.shape[0] * 2

        if "walls" in cm:
            pred = (preds["wall"].float().cpu().numpy() > 0).astype(np.int64)  # (B,12,20)
            twt = b["wall_grid"].float().cpu().numpy().astype(np.int64)
            pi = pred[:, interior].ravel()
            ti = twt[:, interior].ravel()
            wall_tp += int(((pi == 1) & (ti == 1)).sum())
            wall_fp += int(((pi == 1) & (ti == 0)).sum())
            wall_fn += int(((pi == 0) & (ti == 1)).sum())
            wall_inter += int(((pi == 1) & (ti == 1)).sum())
            wall_union += int(((pi == 1) | (ti == 1)).sum())

    out: dict[str, object] = {"total_loss": loss_total / max(1, n_batches)}
    # Start with ALL 7 group keys = N/A, then fill in the assigned ones (same keys as flat).
    breakdown: dict[str, object] = {
        g: "N/A"
        for g in (
            "player_position",
            "player_velocity",
            "player_aim",
            "bullet_presence",
            "bullet_position",
            "bullet_direction",
            "walls",
        )
    }

    if "bullets" in cm:
        prec = occ_tp / (occ_tp + occ_fp) if (occ_tp + occ_fp) else 0.0
        rec = occ_tp / (occ_tp + occ_fn) if (occ_tp + occ_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        breakdown["bullet_presence"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": occ_tp,
            "fp": occ_fp,
            "fn": occ_fn,
        }
        # bullet_position world error from matched-peak grid distance -> world units.
        mean_grid_dist = occ_dist_sum / max(1, occ_matched)
        world_mae = mean_grid_dist * (world_per_col + world_per_row) / 2.0
        breakdown["bullet_position"] = {
            "norm_mse": float(mean_grid_dist**2),  # grid-frac analog of flat's norm_mse
            "world_mae": float(world_mae),
        }
        # bullet_direction stays N/A: occupancy is direction-free by design.

    if "keypoints" in cm:
        breakdown["player_position"] = {
            "norm_mse": float(kp_grid_sse / max(1, kp_n)),  # grid-frac MSE analog
            "world_rmse": float(np.sqrt(kp_world_sse / max(1, kp_n))),
        }

    if "walls" in cm:
        prec = wall_tp / (wall_tp + wall_fp) if (wall_tp + wall_fp) else 0.0
        rec = wall_tp / (wall_tp + wall_fn) if (wall_tp + wall_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = wall_inter / wall_union if wall_union else 1.0
        breakdown["walls"] = {
            "interior_f1": f1,
            "interior_precision": prec,
            "interior_recall": rec,
            "interior_iou": iou,
        }

    out["per_objective"] = breakdown
    return out


# =============================================================================
# Dataset assembly: 3-way map-aware split over both data paths
# =============================================================================
def _build_in_memory_datasets(
    args: Namespace,
    wall_grids: np.ndarray,
    holdout_ids: list[int],
    n_maps: int,
    spatial: tuple[tuple[int, int, int], float] | None = None,
):
    """SMOKE path: map-balanced stream, then 3-way map-aware split; fit stats on TRAIN."""
    out_w, out_h = args.input_wh
    subset = args.subset if args.subset is not None else 8000
    frames, states, map_ids, group_keys = _stream_map_balanced(
        args.data_dir, out_w, out_h, subset, n_maps, args.seed
    )
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids, strict=True)}
    split = map_aware_split(
        group_keys, map_of_group, holdout_ids, eval_frac=args.eval_frac, seed=args.seed
    )
    train_mask = np.array([g in split.train for g in group_keys])
    eval_mask = np.array([g in split.eval for g in group_keys])
    val_mask = np.array([g in split.validation for g in group_keys])
    stats = fit_norm_stats(states[train_mask])

    def mk(m: np.ndarray) -> InMemoryPixelDataset:
        return InMemoryPixelDataset(
            frames[m], states[m], map_ids[m], wall_grids, stats, spatial=spatial
        )

    return mk(train_mask), mk(eval_mask), mk(val_mask), stats, split


# Maps the --frames-backend flag to the cache-backed Dataset class. Both classes share
# the IDENTICAL __init__(cache_dir, row_idx, wall_grids, stats) contract and return the
# same uint8 (3,H,W) frame + targets, so only the class differs — the split/stats code in
# _build_cache_datasets is shared (not copy-pasted).
_CACHE_DATASET_CLASSES = {
    "memmap": MemmapPixelDataset,
    "lmdb_png": LmdbPngPixelDataset,
}


def _build_cache_datasets(
    args: Namespace,
    wall_grids: np.ndarray,
    holdout_ids: list[int],
    frames_backend: str,
    spatial: tuple[tuple[int, int, int], float] | None = None,
):
    """FULL path: read cache sidecars, 3-way map-aware split by group, fit stats on TRAIN.

    Backend-agnostic: the sidecars (states/map_ids/group_keys/index.json) are IDENTICAL
    across the memmap and lmdb_png caches, so the split + stats are computed once here and
    the only difference is which Dataset class reads the frames (selected by
    ``frames_backend`` via :data:`_CACHE_DATASET_CLASSES`).
    """
    ds_cls = _CACHE_DATASET_CLASSES[frames_backend]
    cache = Path(args.cache_dir)
    group_keys_arr = np.load(cache / "group_keys.npy")  # (N, 2)
    map_ids = np.load(cache / "map_ids.npy")
    states = np.load(cache / "states.npy")
    n = group_keys_arr.shape[0]
    if args.subset is not None:
        n = min(n, args.subset)
    all_rows = np.arange(n)
    group_keys = [(int(w), int(e)) for w, e in group_keys_arr[:n]]
    map_of_group = {g: int(m) for g, m in zip(group_keys, map_ids[:n], strict=True)}
    split = map_aware_split(
        group_keys, map_of_group, holdout_ids, eval_frac=args.eval_frac, seed=args.seed
    )
    train_rows = all_rows[[g in split.train for g in group_keys]]
    eval_rows = all_rows[[g in split.eval for g in group_keys]]
    val_rows = all_rows[[g in split.validation for g in group_keys]]
    stats = fit_norm_stats(states[train_rows])
    ds_train = ds_cls(cache, train_rows, wall_grids, stats, spatial=spatial)
    ds_eval = ds_cls(cache, eval_rows, wall_grids, stats, spatial=spatial)
    ds_val = ds_cls(cache, val_rows, wall_grids, stats, spatial=spatial)
    return ds_train, ds_eval, ds_val, stats, split


def _run_eval(
    model: InverseRenderer,
    loader: DataLoader,
    lambdas: dict[str, float],
    device: torch.device,
    stats: NormStats,
    use_amp: bool,
    desc: str,
    presence_pw: float | torch.Tensor,
    wall_pw: float | torch.Tensor,
    aim_loss: str,
    is_spatial: bool,
    spatial_shape: tuple[int, int, int] | None,
) -> dict[str, object]:
    """Dispatch to the flat or spatial evaluator (both emit the SAME 7 per-objective keys)."""
    if is_spatial:
        return _evaluate_spatial(
            model, loader, lambdas, device, spatial_shape, use_amp, desc, wall_pw
        )
    return _evaluate(
        model,
        loader,
        lambdas,
        device,
        stats,
        use_amp,
        desc=desc,
        presence_pw=presence_pw,
        wall_pw=wall_pw,
        aim_loss=aim_loss,
    )


def _resolve_run_dir(out_base: Path, out_w: int, out_h: int, emb: int, batch: int) -> Path:
    """Append a unique ``_{W}x{H}_e{emb}_b{batch}_{MMDD-HHMM}`` suffix. Never overwrite."""
    stamp = datetime.now().strftime("%m%d-%H%M")
    name = f"{out_base.name}_{out_w}x{out_h}_e{emb}_b{batch}_{stamp}"
    run_dir = out_base.parent / name
    if run_dir.exists():
        raise SystemExit(
            f"[pretrain] refusing to overwrite existing run dir: {run_dir}. "
            "Pick a different --out base or wait a minute (timestamp differs)."
        )
    return run_dir


def run_training(args: Namespace) -> int:
    """Build data, train, eval each epoch, validate once, save artifacts. Returns exit code."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg: ObjectiveConfig = args.objective_config
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"[pretrain] device={device} amp={use_amp} torch={torch.__version__}")
    print(f"[pretrain] objectives ENABLED: {list(cfg.ordered())}")
    from tank_twin.pretrain_pixels import OBJECTIVE_GROUPS

    disabled = [g for g in OBJECTIVE_GROUPS if not cfg.is_on(g)]
    print(f"[pretrain] objectives DISABLED (reported N/A): {disabled}")

    # Resolve the encoder-output mode + spatial config (validated already in _parse_args).
    encoder_output = getattr(args, "encoder_output", "flat")
    spatial_shape = getattr(args, "spatial_shape", None)
    channel_map = getattr(args, "channel_map", None)
    occupancy_sigma = getattr(args, "occupancy_sigma", 1.0)
    is_spatial = encoder_output == "spatial"
    spatial_spec = (spatial_shape, occupancy_sigma) if is_spatial else None
    if is_spatial:
        print(
            f"[pretrain] encoder-output=spatial field={spatial_shape[0]}x"
            f"{spatial_shape[1]}x{spatial_shape[2]} (CxHxW) channel_map={channel_map} "
            f"occupancy_sigma={occupancy_sigma}"
        )
    else:
        print(f"[pretrain] encoder-output=flat (head={getattr(args, 'head', 'fc')})")

    out_w, out_h = args.input_wh
    run_dir = _resolve_run_dir(Path(args.out), out_w, out_h, args.embedding_dim, args.batch)
    print(f"[pretrain] resolved run dir -> {run_dir}")

    wall_grids = np.load(args.data_dir / "wall_grids.npy")
    n_maps = wall_grids.shape[0]
    name_to_id = load_map_name_to_id(args.data_dir)
    holdout_ids = resolve_map_ids(args.holdout_map_names, name_to_id)
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(
        f"[pretrain] holdout (validation) maps: {[(id_to_name[i], i) for i in sorted(holdout_ids)]}"
    )

    # frames_backend only applies to the cache (--cache-dir) path; the SMOKE stream is
    # unaffected. Default "memmap" preserves the current behavior.
    frames_backend = getattr(args, "frames_backend", "memmap")
    t0 = time.time()
    if args.cache_dir is not None:
        ds_train, ds_eval, ds_val, stats, split = _build_cache_datasets(
            args, wall_grids, holdout_ids, frames_backend, spatial=spatial_spec
        )
        mode = frames_backend
    else:
        ds_train, ds_eval, ds_val, stats, split = _build_in_memory_datasets(
            args, wall_grids, holdout_ids, n_maps, spatial=spatial_spec
        )
        mode = "stream"

    train_map_names = [id_to_name.get(i, str(i)) for i in split.train_map_ids]
    val_map_names = [id_to_name.get(i, str(i)) for i in split.holdout_map_ids]
    print(
        f"[pretrain] mode={mode} res={out_w}x{out_h} | 3-way split: "
        f"train={len(ds_train)} rows ({len(split.train)} groups) "
        f"eval={len(ds_eval)} rows ({len(split.eval)} groups) "
        f"validation={len(ds_val)} rows ({len(split.validation)} groups)"
    )
    print(
        f"[pretrain] train maps ({len(train_map_names)}): {train_map_names} | "
        f"validation maps ({len(val_map_names)}): {val_map_names} "
        f"(data prep {time.time() - t0:.1f}s)"
    )

    # --- Resolve class-imbalance BCE pos_weights (TRAIN split ONLY; no leakage) ------
    # ds_train._cache holds the TRAIN split's decoded targets; the wall key differs by
    # mode ('wall' flat vs 'wall_grid' spatial). presence pos_weight is flat-only (spatial
    # occupancy uses the focal loss, which is self-balancing; no presence head). Computed
    # ONCE here and threaded into every loss call (NOT recomputed per-batch).
    presence_override = parse_pos_weight_arg(args.presence_pos_weight)
    wall_override = parse_pos_weight_arg(args.wall_pos_weight)
    wall_key = "wall_grid" if is_spatial else "wall"
    train_walls = ds_train._cache[wall_key]  # (N, 12, 20) {0,1}
    wall_pw = wall_override if wall_override is not None else interior_wall_pos_weight(train_walls)
    wall_src = "override" if wall_override is not None else "auto(train)"
    if is_spatial:
        presence_pw = 1.0  # unused in spatial mode (occupancy focal loss is self-balancing)
        presence_src = "n/a(spatial)"
        print(f"[pretrain] BCE pos_weights -> wall(interior)={wall_pw:.4f} ({wall_src})")
    else:
        train_presence = ds_train._cache["bullet_presence"]  # (N, 10) {0,1}
        presence_pw = (
            presence_override
            if presence_override is not None
            else presence_pos_weight(train_presence)
        )
        presence_src = "override" if presence_override is not None else "auto(train)"
        print(
            f"[pretrain] BCE pos_weights -> presence={presence_pw:.4f} ({presence_src}), "
            f"wall(interior)={wall_pw:.4f} ({wall_src})"
        )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    eval_loader = DataLoader(
        ds_eval,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    if is_spatial:
        head = None  # N/A in spatial mode (recorded as null in config)
        model = InverseRenderer(
            out_h,
            out_w,
            cfg,
            args.embedding_dim,
            encoder_output="spatial",
            spatial_shape=spatial_shape,
            channel_map=channel_map,
        ).to(device)
    else:
        head = parse_head(getattr(args, "head", "fc"))
        model = InverseRenderer(out_h, out_w, cfg, args.embedding_dim, head=head).to(device)
    # AdamW = decoupled weight decay (the correct form); weight_decay=0.0 (default) makes
    # this behave like the old Adam, so `--lr 3e-4 --lr-schedule constant` is unchanged.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- LR schedule (per-optimizer-step LambdaLR over the WHOLE run) ----------------
    # Total optimizer steps = steps-per-epoch * epochs. train_loader uses drop_last=False,
    # so steps-per-epoch == len(train_loader) (number of batches). For 'constant' the
    # multiplier is a no-op 1.0 (byte-equivalent to no scheduler); for 'cosine' it is the
    # pure linear-warmup -> cosine-anneal multiplier. The scheduler is stepped ONCE per
    # SUCCESSFUL optimizer step (guarded against AMP-skipped steps), so its internal step
    # counter == the optimizer-step index passed to the multiplier helper.
    warmup_frac = 0.05
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(max(1, round(warmup_frac * total_steps)), total_steps)
    if args.lr_schedule == "cosine":

        def _lr_lambda(opt_step: int) -> float:
            return cosine_warmup_lr_multiplier(opt_step, total_steps, warmup_frac)
    else:

        def _lr_lambda(opt_step: int) -> float:  # constant: no-op multiplier
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    print(
        f"[pretrain] lr={args.lr} schedule={args.lr_schedule} weight_decay={args.weight_decay} "
        f"(total_steps={total_steps}, warmup_steps={warmup_steps})"
    )
    lambdas = {
        "player": args.lambda_player,
        "aim": args.lambda_aim,
        "presence": args.lambda_presence,
        "bullet_pos": args.lambda_bullet_pos,
        "bullet_dir": args.lambda_bullet_dir,
        "wall": args.lambda_wall,
        # Spatial-mode loss-term weights reuse the flat lambdas where natural: occupancy
        # SUBSUMES presence+pos so it takes lambda_presence; keypoints take lambda_player.
        "occupancy": args.lambda_presence,
        "keypoint": args.lambda_player,
    }
    aim_loss = args.aim_loss
    if not is_spatial and cfg.is_on("player_aim"):
        print(f"[pretrain] player_aim loss={aim_loss} lambda_aim={args.lambda_aim}")
    # Encoder param count is the load-bearing A/B number (fc flatten->FC vs spatial/heatmap
    # global-pool trunk): record it for the report and persist it in config/metrics.
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[pretrain] mode={encoder_output} head={head} encoder flatten_dim="
        f"{model.encoder.flatten_dim} embedding_dim={model.encoder.embedding_dim} "
        f"heads={list(model.sizes.items())} "
        f"encoder_params={encoder_params:,} total_params={total_params:,}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    eval_history: list[dict] = []

    global_step = 0
    seen = 0
    train_t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_total = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in pbar:
            b = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(b["frame"])
                if is_spatial:
                    total, parts = compute_spatial_losses(preds, b, channel_map, lambdas, wall_pw)
                else:
                    total, parts = compute_losses(
                        preds,
                        b,
                        model,
                        lambdas,
                        presence_pw,
                        wall_pw,
                        stats=stats,
                        aim_loss=aim_loss,
                    )
            scaler.scale(total).backward()
            # AMP can SKIP an optimizer step on inf/nan grads. Detect a real step by
            # comparing the GradScaler scale across update(): on a skipped step the scaler
            # lowers the scale (scale_after < scale_before); on a real step it is unchanged
            # or raised. Only advance the LR schedule when the optimizer actually stepped,
            # so a skipped step does not consume a slot of the warmup/cosine schedule.
            # (With AMP disabled the scale is constant 1.0, so the step always counts.)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
            if stepped:
                scheduler.step()
            ep_total += float(total.detach())
            n_batches += 1
            global_step += 1
            seen += b["frame"].shape[0]
            pbar.set_postfix(total=f"{float(total.detach()):.3f}")
            if global_step % args.log_every == 0:
                tqdm.write(
                    f"  [e{epoch} s{global_step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in parts.items() if k != "total")
                    + f" total={parts['total']:.4f}"
                )
        ep_total /= max(1, n_batches)
        ev = _run_eval(
            model,
            eval_loader,
            lambdas,
            device,
            stats,
            use_amp,
            "  eval",
            presence_pw,
            wall_pw,
            aim_loss,
            is_spatial,
            spatial_shape,
        )
        ev["epoch"] = epoch
        ev["train_total_loss"] = ep_total
        eval_history.append(ev)
        print(
            f"[epoch {epoch}] TRAIN total={ep_total:.4f} || "
            f"EVAL total={ev['total_loss']:.4f} | {_fmt_breakdown(ev['per_objective'])}"
        )

    train_secs = time.time() - train_t0
    pairs_per_sec = seen / max(1e-9, train_secs)
    print(
        f"[pretrain] trained {seen} pair-steps in {train_secs:.1f}s "
        f"({pairs_per_sec:.0f} pairs/s, {global_step / max(1e-9, train_secs):.1f} steps/s)"
    )

    # --- Final out-of-distribution VALIDATION (the 2 held-out maps), run ONCE ----
    print("[pretrain] running FINAL out-of-distribution VALIDATION (2 held-out maps)...")
    validation = _run_eval(
        model,
        val_loader,
        lambdas,
        device,
        stats,
        use_amp,
        "  validation",
        presence_pw,
        wall_pw,
        aim_loss,
        is_spatial,
        spatial_shape,
    )
    print(
        f"[validation] OOD total={validation['total_loss']:.4f} | "
        f"{_fmt_breakdown(validation['per_objective'])}"
    )

    # --- Save artifacts ------------------------------------------------------
    encoder_path = run_dir / "encoder.pt"
    full_path = run_dir / "model.pt"
    stats_path = run_dir / "norm_stats.json"
    split_path = run_dir / "split.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.state_dict(), full_path)
    stats_path.write_text(json.dumps(stats.to_json(), indent=2), encoding="utf-8")
    split_payload = split.to_json()
    split_payload.update(
        {
            "train_map_names": train_map_names,
            "validation_map_names": val_map_names,
            "eval_frac": args.eval_frac,
            "holdout_maps": args.holdout_map_names,
            "seed": args.seed,
        }
    )
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    # Spatial-mode config block (channel-map as JSON-friendly lists; null in flat mode).
    spatial_config = None
    if is_spatial:
        spatial_config = {
            "spatial_shape": list(spatial_shape),  # [C, H, W]
            "channel_map": {k: list(v) for k, v in channel_map.items()},
            "occupancy_sigma": occupancy_sigma,
            "objectives": sorted(channel_map.keys()),
            "subsumes": "bullet occupancy heatmap REPLACES presence+position+direction",
        }
    config_dict = {
        "input_res": f"{out_w}x{out_h}",
        "encoder_output": encoder_output,
        "head": head,  # null in spatial mode (--head is N/A)
        "spatial": spatial_config,
        "embedding_dim": args.embedding_dim,
        "encoder_embedding_dim": model.encoder.embedding_dim,
        "flatten_dim": model.encoder.flatten_dim,
        "encoder_params": encoder_params,
        "total_params": total_params,
        "objectives": cfg.to_json(),
        "head_sizes": model.sizes,
        "lambdas": lambdas,
        "aim_loss": aim_loss,
        "pos_weights": {
            "presence": float(presence_pw),
            "wall_interior": float(wall_pw),
            "presence_source": presence_src,
            "wall_source": wall_src,
        },
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "lr_schedule_warmup_frac": warmup_frac if args.lr_schedule == "cosine" else None,
        "lr_schedule_warmup_steps": warmup_steps if args.lr_schedule == "cosine" else None,
        "lr_schedule_total_steps": total_steps if args.lr_schedule == "cosine" else None,
        "epochs": args.epochs,
        "eval_frac": args.eval_frac,
        "holdout_maps": args.holdout_map_names,
        "seed": args.seed,
        "mode": mode,
        "frames_backend": frames_backend,
        "device": str(device),
    }
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    metrics_payload = {
        "config": config_dict,
        "eval": eval_history,
        "validation": validation,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"[pretrain] saved encoder -> {encoder_path}")
    print(f"[pretrain] saved full model -> {full_path}")
    print(f"[pretrain] saved norm stats -> {stats_path}")
    print(f"[pretrain] saved 3-way split -> {split_path}")
    print(f"[pretrain] saved metrics (eval history + validation) -> {metrics_path}")

    # --- Verify the encoder reloads STANDALONE into a fresh PixelEncoder -----
    # The encoder is objective-config-independent but architecture-specific (fc: flatten->FC,
    # embedding=embedding_dim; heatmap: global-pool, embedding=C; spatial: convs-only trunk,
    # forward returns the (B,C,h,w) feature map). Build the fresh encoder the SAME way so the
    # keys match (strict load) and assert its forward shape. In spatial AND heatmap mode the
    # saved keys are the SAME cnn.0/2/4 trunk (the reusable artifact).
    sd = torch.load(encoder_path, map_location="cpu", weights_only=True)
    if is_spatial:
        fresh = PixelEncoder(out_h, out_w, args.embedding_dim, spatial_trunk=True)
        fresh.load_state_dict(sd)  # strict=True
        fresh.eval()
        with torch.no_grad():
            fm = fresh(torch.zeros(2, 3, out_h, out_w))  # (2, C, h, w) feature map
        assert fm.ndim == 4 and fm.shape[0] == 2, fm.shape
        print(
            f"[pretrain] encoder reload OK (spatial trunk): fresh PixelEncoder forward -> "
            f"{tuple(fm.shape)} (feature map); encoder keys={sorted(sd.keys())} "
            "(IDENTICAL cnn.0/2/4 trunk to flat-mode encoder; objective-config-independent)"
        )
    else:
        fresh = PixelEncoder(out_h, out_w, args.embedding_dim, head=head)
        fresh.load_state_dict(sd)  # strict=True: raises on any mismatch
        fresh.eval()
        expected_emb = fresh.embedding_dim
        with torch.no_grad():
            emb = fresh(torch.zeros(2, 3, out_h, out_w))
        assert emb.shape == (2, expected_emb), emb.shape
        print(
            f"[pretrain] encoder reload OK (head={head}): fresh PixelEncoder forward -> "
            f"{tuple(emb.shape)} (expected (2, {expected_emb})); "
            f"encoder keys={sorted(sd.keys())[:2]}... (objective-config-independent)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(1)
