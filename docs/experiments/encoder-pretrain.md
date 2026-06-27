# Encoder Pretraining — Design Doc

**Living source of truth** for the single-frame encoder decode benchmark: method · decisions · results in one place. Update this as decisions land and the sweep runs.

- **Status:** harness built + GPU-validated; **training config under investigation** (flat val curve, see §5); sweep not yet run.
- **Owner:** Brian (CTO-delegated eval/encoder direction).
- **Tracking:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) (parent) · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) (phases) · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) (follow-up). **Code:** PR #11. **Plan:** PR #4.
- **Last updated:** 2026-06-27.

---

## 1. Goal & research question

**Can a single-frame vision encoder recover the game state from one rendered frame — which fields, and what's the best decode-accuracy-per-compute encoder to hand to RL?**

- **Metric:** per-field val decode error — position (world units), aim (angular degrees), velocity, bullet-presence (F1), bullet-position — plus compute (params, forward latency). Reported for **both** head families (spatial + detached embed-probe).
- **Decision rule** *(to finalize before the sweep — senior/principal rigor):* pick the config on the accuracy↔compute **Pareto frontier** that clears the downstream-RL "good enough" bar. **[OPEN]**
- **Non-goals:** not best-possible (architecture exploration is later, [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)); not the competitive eval framework; single-frame only.

---

## 2. Method — the pretraining harness

Single-frame **inverse-renderer**. (Full code reference: [`docs/components/pretraining.md`](../components/pretraining.md).)

- **Encoder** (`models.build_encoder`): a NatureCNN / ImpalaResNet **trunk** × GAP / Flatten **pooling**, with an early ÷4 **stem** sized for the 640×360 native frame.
- **Two head families:**
  - **Spatial heads** on `Encoder.features` → gradient trains the encoder (the reusable artifact).
  - **Detached embed-probe** on `Encoder.embed.detach()` → measures how much state survives pooling; this is *why pooling earns a place in the sweep* (a real accuracy delta on the probe).
- **Five per-group targets/losses/metrics:** position & velocity (MSE → world-unit error), aim (cosine → angular degrees), bullet-presence (pos-weighted BCE → F1), bullet-position (masked MSE).
- **Data:** shard-streaming `DecodeDataset`, map-aware 80/10/10 split (no leak), **shard-grouped sampler** (each shard decompressed once per epoch), load-time downsample.
- **CLIs:** `train` (device-agnostic `--device cuda|mps|cpu`, `--eval-every K` periodic val, progress reporter) · `profile` (params + forward latency).

---

## 3. Experimental design (the sweep)

- **Grid:** `{nature, impala} × {gap, flatten} × {360, 180, 90}` = 12 → **10 cells** (`nature@90` dropped — NatureCNN's ÷4 stem underflows a 90-row frame).
- **Resolutions:** 360 = 640×360 (native) · 180 = 320×180 · 90 = 160×90.
- **Data:** `decode-v2` (~120k samples, 10 maps) — research-confirmed *generous* for a low-diversity supervised decode.
- **Protocol to lock:** seed(s) **[OPEN — ≥3 for the final comparison]**, epochs/cell **[OPEN — convergence probe]**, LR / batch / normalization fixed, map-aware split.
- **Rigor we're applying:** baseline + fixed protocol; one-factor attribution; the embed-probe as the pooling ablation; a data **learning curve**; a **pre-written decision rule**; exploration → confirmation; and the higher-value **encoder→RL transfer probe** as a check on the decode *proxy*.

---

## 4. Decision log

| Decision | Rationale | By | Status |
|---|---|---|---|
| Observation = real Unity pixels, single-frame | the synthetic-grid approach was dropped; decode the 52-float state from one real render | Mike | ✅ |
| **Structured per-group heads** (no FC-to-52 control) | heterogeneous fields need the right loss per type (aim=angular, presence=BCE, …); FC-to-52 would output garbage for some | Brian | ✅ |
| **Embed-probe on `embed.detach()`** (option B) | makes pooling (GAP vs Flatten) show a *real* accuracy delta; RL-relevant retention measure | Brian | ✅ |
| Resolutions **{360, 180, 90}**; `nature@90` dropped | the range the stemmed encoders support; NatureCNN underflows 90 | Brian | ✅ |
| **10-cell sweep**, pooling swept via the probe | full trunk × pooling × resolution comparison | Brian | ✅ |
| **Committed `pretraining/`** component (not scratch) | reusable, gated, shareable; it's a real planned component | Brian | ✅ |
| `decode-v2` ~120k; iterate against `decode-v1` pilot | research: 120k is generous for low-diversity supervised decode | Brian | ✅ |
| **Train on the 4090** (remote, Tailscale SSH) | CUDA (`cu130`), fast, free; MPS is partial/flaky | Brian | ✅ |
| Epochs / cell | — | — | ⏳ convergence probe |
| Sweep scope: full 10-cell **vs** principal cut | value-of-information | — | ⏳ |
| Seeds (≥3) + pre-written decision rule | variance reporting + pre-registration | — | ⏳ |
| Stem-less small-input encoder (field-norm 64–96px) | research follow-up; not this sweep | — | 📋 [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) |

---

## 5. Results

- **Pipeline validated** — smoke-train on `decode-v1` (MPS/CPU): loss decreased; both head families report per-group metrics.
- **Loader perf bug found + fixed** — a single-entry shard cache + global `shuffle=True` re-decompressed a whole ~580-frame `.npz` per sample (a 3k smoke hung **17 min** with the GPU idle). Replaced with a shard-grouped sampler → **~45s** (each shard read once/epoch). A `read_shard`-count regression test guards it. (PR #11)
- **GPU validated (4090, `torch 2.9.1+cu130`)** — `impala/180` ~20s/epoch; **~2,200 samples/sec** at `num_workers=8` / `batch=256` (~17× the `num_workers=0` default). Windows spawn works.
- **⚠️ OPEN — flat val curve (sweep blocker).** The convergence probe (`impala/180`, full `decode-v2`, 50 epochs, periodic-val) shows **no val improvement**: position ~5.5, aim ~88° (≈ chance), presence-F1 *frozen* at 0.233 over the first ~7 epochs, while train loss only creeps 5.0 → 4.6. Likely cause: learning rate too low, per-group loss weighting, or the presence head collapsing to base-rate. **Must be resolved before the sweep — a sweep on a non-learning setup is wasted compute.**
- **Sweep results** — *pending.* Table per cell to be filled: per-field accuracy (spatial + probe) × params × latency × resolution.

### Context from research (logged decisions inputs)
- **RL-from-pixels resolutions (~June 2026):** the field trains from-scratch encoders at **64–96px** (Atari 84×84, Procgen/Dreamer 64×64, Dreamer-Atari 96×96); ≥224px only with frozen pretrained encoders. Our smallest point (160×90) is ~2× the 84×84 norm; 640×360 is ~33× (off-map). → motivates [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12).
- **Dataset size:** ~120k labeled frames is **generous** for supervised decode on a low-diversity synthetic domain (analogs: NS-VQA de-render ~4k, DOPE pose ~120k on a *harder* domain). The binding constraints are **class imbalance** (bullet-presence ~8%) and **map diversity** (10), not total frame count. Confirm with a **learning curve** (the harness's `--subset` knob).

---

## 6. Open questions / risks / next steps

1. **[BLOCKER] Flat val** — diagnose why the model isn't learning (LR, per-group loss weighting, presence-head collapse) before any sweep.
2. **Proxy-metric risk** — decode accuracy ≠ RL usefulness (Goodhart). A thin **encoder→RL transfer probe** is higher value-of-information than cells 4–10 of the grid.
3. **Single-frame limit** — velocity is fundamentally undecodable from one frame (expected); points at frame-stacking / temporal input for the RL phase.
4. **Sweep scope** — full 10-cell (matches Mike's ask: per-field accuracy + latency + params + resolution) vs. the **principal cut** (2–3 representative cells + an RL probe).
5. **Levers > frame count** — class imbalance (presence ~8%) and held-out-map generalization (10 maps) move the result more than raw N.

---

## References
- **Code:** [`docs/components/pretraining.md`](../components/pretraining.md) · PR #11 (harness) · PR #4 (plan)
- **Research briefings:** RL-from-pixels resolutions; supervised dataset sizes *(produced this session; archive here as they're finalized)*
- **Issues:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) parent · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) phases · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) follow-up
- **Datasets:** `decode-v1` (~20k) · `decode-v2` (~120k) — both manifested (provenance + machine)
