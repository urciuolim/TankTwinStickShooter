# Plan of Record — Single-Frame Encoder State-Decoder Benchmark

**Status:** DRAFT for review (Mike + Brian). **Target branch:** `revival-2026`. **Component:** `pretraining/` (first content). **Tracking issue:** linked in the PR.

> This is a plan-of-record draft — **no implementation in this PR**. It exists so we agree the shape before building.

## 1. Goal

Build a **single-frame** image encoder that decodes the 52-float game state from one rendered pixel frame, and benchmark it. Deliver:

1. A **per-field accuracy breakdown** — explicitly surfacing which fields are decodable from a single frame and which are not (e.g. velocity).
2. **Forward-pass latency** at inference.
3. **Parameter count**.
4. A **comparison across input resolutions**.

This is the first real content of the `pretraining/` component (the inverse-renderer). It validates the encoder as an RL backbone **before** any RL is built on top.

## 2. What exists vs. what we build

| Piece | Status |
|---|---|
| Encoder **architecture** (`models/encoders.py`) | ✅ exists, **untrained** (random weights) |
| Trained encoder / checkpoint | ❌ none |
| Decode heads (embed → 52 floats) | ❌ build |
| Training loop + losses (`pretraining/`) | ❌ build (stub today) |
| `(frame, state)` dataset | ❌ collect (`datasets/` empty) |
| Measurement/benchmark harness | ❌ build |
| Reference training code (`reference/tank_twin_m1/pretrain_pixels.py`) | ✅ retired — inspiration only |

So the ask is **train an encoder + build the harness to verify efficacy** — only the encoder architecture is reused as-is.

## 3. The decode task & per-field story

Per `core/state.py`, each player's 26-float block splits into groups that are *expected-decodable* vs *expected-hard* from a single frame — which is the breakdown to surface:

| Field group | Fields (×2 players) | Single-frame decodable? | Metric |
|---|---|---|---|
| Tank **position** | pos_x, pos_y | ✅ yes | mean position error (world units) |
| Tank **velocity** | vec_x, vec_y | ❌ no (needs ≥2 frames) | velocity error (expect ≈ chance) |
| Tank **aim** | aim_x, aim_y | ✅ likely (barrel/aim rendered) | angular error (deg) |
| **Bullet presence** | 5 slots (`pos_x<0` sentinel) | ✅ yes | accuracy / F1 |
| **Bullet position** | pos_x, pos_y (present only) | ✅ yes | masked position error |
| **Bullet velocity** | vec_x, vec_y | ❌ no | velocity error (expect poor) |

The headline result: **positions / aim / bullet-presence decode well; velocities do not** — an honest demonstration of single-frame information content, and a signal for whether temporal context (frame-stacking / recurrence) is later warranted.

## 4. Plan (phased)

- **Phase 0 — Data.** Collect a `(frame, state)` dataset via `data.collect_runner` (now runs on macOS). Map-aware split (no leak) via `data.readers`.
- **Phase 1 — Decoder harness** in `pretraining/`: `models.Encoder` → per-group decode heads → per-group losses; a shard-streaming `Dataset` over `data.schema`/`readers` (lazy frame fetch).
- **Phase 2 — Benchmark.** Sweep {encoder variant × input resolution}; emit per-field accuracy + param count + forward-pass latency.

## 5. Design specifics (proposed)

- **Heads:** structured per-group on top of the encoder — regression (pos/vel), cosine/angular (aim), BCE (bullet presence), masked-MSE (bullet position, present slots only). A plain FC-to-52 baseline is an optional control.
- **Losses & metrics:** per group, as in the §3 table; report each separately (never a single blended number).
- **Loader:** stream shards via `data.readers.DatasetIndex` (frames lazy); honor the map-aware train/val/test split.
- **Normalization:** frame `uint8 → [0,1]`; state fields scaled to sane ranges; absent-bullet sentinel masked out of position/velocity losses.
- **Single-frame only** (per direction).

## 6. Benchmark dimensions

- **Encoders:** `NatureCNN` (start) → `ImpalaResNet` (stretch); pooling GAP vs Flatten (stretch).
- **Resolutions:** 360p (640×360 native), 180p (320×180), 90p (160×90).
- **Latency:** ms/frame + throughput, on **CPU and MPS** (no CUDA on the dev Mac; real training later targets GCP — this is a *relative* benchmark).
- **Params:** count per config.

## 7. Deliverables / acceptance

- A trained encoder+heads checkpoint per swept config.
- A **results report** (table + plots): per-field accuracy, param counts, latency, resolution comparison.
- Reproducible from a documented command; unit tests for the pure pieces (head shapes, metric/loss fns, loader split).

## 8. Open decisions for review (Mike + Brian)

| # | Decision | Recommended default |
|---|---|---|
| 1 | Data budget (frames / collection time) | a modest run first (size TBD) |
| 2 | Sweep breadth | NatureCNN + 3 resolutions; ImpalaResNet/pooling as stretch |
| 3 | Train where | MPS on the Mac now (relative benchmark); GCP later |
| 4 | Head design | structured per-group heads (FC-to-52 as optional control) |
| 5 | Output home | build in `pretraining/` (committed) with the report artifact |

## 9. Process & fit

- Lives in `pretraining/` (depends on `core`, `models`, `data`; `pop-pretraining` skill). Reuses `data.readers`/`schema` for the dataset and `models.Encoder` for the backbone.
- Built through the operating model: eng team (`builder`) → platform gate → documentarian.

## 10. Non-goals

- Not the competitive `eval/` framework (ELO / payoff matrix) — separate, paused.
- Not multi-frame / temporal (single-frame only).
- Not RL training (this validates the perception backbone first).
