# Encoder Pretraining — Design Doc

**Living source of truth** for the single-frame encoder decode benchmark: method · decisions · results in one place. Update this as decisions land and the sweep runs.

- **Status:** harness built + GPU-validated; **flat-val hit twice** — first the position-blind GAP heads (fixed by the soft-argmax decoder), then the soft-argmax decoder's own **predict-center basin** (coord-only escape gradient too weak; capacity fix insufficient, §6) → **heatmap-CE fix landed + validated** (§6/§10): spike-validated (~0.4–0.5 wu), then production was flat until the **shard-grouped sampler** (single-episode, no-diversity batches) was found to starve it — with cross-shard batch diversity (validated by a one-off on-disk reshuffle) val drops **4.06 → 0.40 wu in 3 epochs**. **Sweep 1 (3 of 10 cells) now landed** (§2 → Experimental log): the fix holds at full training scale (~0.30 wu), **`impala × flatten @ 180` leads**, with **aim (~41°) + presence (F1 ~0.46)** the architecture-independent bottlenecks. The **windowed multi-shard sampler** is now committed + **learning-validated on the pristine `decode-v2_original`** (val 2.15 → 0.31 wu) — but it's **NO-GO on perf** (per-worker cache replication at `num_workers=8` → ~36 GB / ~9× slow); a loader rework is planned (spike-gated, §6 To-do #1). Next = **land the loader perf fix → GO the sampler** → continue the sweep (resolution axis) + close the per-metric gaps (§6 To do).
- **Owner:** Brian (CTO-delegated eval/encoder direction).
- **Tracking:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) (parent) · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) (phases) · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) (follow-up). **Code:** PR #11. **Plan:** PR #4.
- **Last updated:** 2026-06-28.

---

## 1. Goal & research question

**Can a single-frame vision encoder recover the game state from one rendered frame — which fields, and what's the best decode-accuracy-per-compute encoder to hand to RL?**

- **Metric:** per-field val decode error — position (world units), aim (angular degrees), velocity, bullet-presence (F1), bullet-position — plus compute (params, forward latency). Reported for **both** head families (spatial + detached embed-probe).
- **Decision rule** *(to finalize before the sweep):* pick the config on the accuracy↔compute **Pareto frontier** that clears the downstream-RL "good enough" bar. **[OPEN]**
- **Non-goals:** not best-possible (architecture exploration is later, [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)); not the competitive eval framework; single-frame only.

### 1.1 Provisional target bands

These are **interim, physically-motivated sanity bars** — *not* the committed gate. The real decision criterion is the downstream **encoder→RL transfer probe** (§4); decode error is only a cheap proxy. The bands below answer "is a decode number tactically meaningful, or still near zero-skill?" so we can read the sweep without an RL run in the loop. All bands are derived from in-game physical constants (cited), not from external benchmarks.

**Grounding constants** (game scale this is all derived from):

| Constant | Value | Source |
|---|---|---|
| Arena extent | 16.72 × 7.95 wu | `runs/*/results.json` → `grid_extent` |
| Sim timestep `dt` | 0.02 s (50 Hz) | `exp-configs/maps/*.json:14` (`ai_fixedDeltaTime`) |
| Tank body | 0.75 × 0.5 wu (≈0.31 wu radius) | `Assets/Prefabs/Tank.prefab:225`; `tank.png` @64 PPU |
| Bullet | 0.08 × 0.04 wu | `Assets/Prefabs/Bullet.prefab:81` |
| Player speed | 3 wu/s → 0.06 wu/step | `Assets/Scripts/PlayerController.cs:12` |
| Bullet speed | 15 wu/s → 0.30 wu/step | `Assets/Scripts/BulletController.cs:8` |
| Pixel size @180 / @360 | ~0.048 / ~0.024 wu/px | arena ÷ frame rows |
| Presence imbalance | ~29:1 (absent:present) | `results.json` → `presence_pos_weight` 28.9 |

**Target bands** (lower better except F1; "Sweep-1 best" = best cell to date, `spatial / probe`):

| Metric | Zero-skill (chance) | 🟡 Good | 🟢 Excellent | Sweep-1 best | Basis & citation |
|---|---|---|---|---|---|
| **Position** (wu) | ~5.0 | ≤ 0.30 | ≤ 0.10 | 0.302 / 0.539 | *Good* = one tank radius (~0.31 wu) ≈ one bullet-step (0.30 wu) ≈ 6 px @180 → "knows which tank-width cell." *Excellent* ≈ 2 px @180, near the pixel floor. *Chance* = RMS distance from arena center over `grid_extent` (predict-the-mean). |
| **Aim** (°) | 90 | ≤ 10 | ≤ 3–5 | 41.3 / 41.2 | A tank of radius 0.31 wu at a ~5 wu engagement range subtends ±atan(0.31/5) ≈ **3.5°**, so sub-5° is "reliably on-target"; ≤10° lands at close range / leads roughly. *Chance* = E[\|θ\|] for θ∼Uniform(−180°,180°) = **90°** (derivation). |
| **Bullet-pos** (wu) | ~5.0 | ≤ 0.50 | ≤ 0.30 | 1.322 / 2.043 | To dodge, you must localize a bullet to within ~one tank radius / one step of its travel (0.30 wu/step). Bullets are tiny (0.08×0.04 wu) and fast, so this is strictly harder than player position. *Chance* as for position. |
| **Presence** (F1) | ~0.06 | ≥ 0.70 | ≥ 0.85 | 0.386 / 0.464 | Harsh under ~29:1 imbalance. *Chance* = always-predict-present → precision 1/30, recall 1 → F1 ≈ 0.065 (always-absent → 0). The deficit is **precision** (recall already ~0.94) and is **calibration, not discrimination** — Exp #2 measured **PR-AUC ~0.79** (~24× chance), so recalibration lifts F1 to ~0.74–0.76 and **clears Good on held-out test** (optimism gap ~0); not an encoder/resolution limit. |
| **Velocity** | *TBD* | *TBD* | *TBD* | 0.772 / 0.762 | **[OPEN]** — band deferred until the field's unit & normalization are pinned from `results.json` → `norm_stats` (stored as wu/s vs per-step displacement changes the scale). |

**Caveats.** (1) Single seed (0); bands are read against one run per cell. (2) Tank/bullet footprints are the *sprite draw size*, not a physics collider radius — a collider, if different, would shift the position/aim bands slightly. (3) These are **proxy** bars: a cell can clear every band and still underperform on RL transfer, or vice-versa — the transfer probe overrides. (4) Engagement range of ~5 wu for the aim band is an assumption (≈⅓ of arena width); closer fights tolerate looser aim, longer fights demand tighter.

---

## 2. Results

Per-cell decode performance from the 10-cell sweep. **Pending the decoder fix + a confirming re-probe** (see §6 Progress) — cells are scaffolded with the reporting columns; metrics fill in as the sweep runs. Each metric is reported for **both head families** as `spatial / embed-probe`.

- **Position / Velocity / Bullet-pos:** world-unit error (lower better). **Aim:** angular degrees (lower better). **Presence:** F1 (higher better). **Params / Latency:** forward compute.

| Config | Res | Params | Latency (ms) | Position err (wu) | Aim err (°) | Velocity err | Presence F1 | Bullet-pos err |
|---|---|---|---|---|---|---|---|---|
| nature × gap | 360 |  |  |  |  |  |  |  |
| nature × gap | 180 | 0.137M | — | 4.205 / 12.833 | 85.0 / 84.8 | 1.717 / 3.468 | 0.143 / 0.230 | 5.191 / 12.533 |
| nature × flatten | 360 |  |  |  |  |  |  |  |
| nature × flatten | 180 |  |  |  |  |  |  |  |
| impala × gap | 360 |  |  |  |  |  |  |  |
| impala × gap | 180 | 0.100M | — | 0.302 / 4.050 | 51.6 / 55.9 | 0.772 / 0.871 | 0.364 / 0.329 | 1.453 / 4.100 |
| impala × gap | 90 |  |  |  |  |  |  |  |
| impala × flatten | 360 |  |  |  |  |  |  |  |
| impala × flatten | 180 | 0.100M | — | 0.326 / 0.539 | 41.3 / 41.2 | 0.757 / 0.762 | 0.386 / 0.464 | 1.322 / 2.043 |
| impala × flatten | 90 |  |  |  |  |  |  |  |

*(`nature@90` dropped — NatureCNN's ÷4 stem underflows a 90-row frame.)*
*(Filled cells = test-set metrics from **Sweep 1**, below. **Params** = encoder-only (the reusable artifact); flatten's larger decoder/embed cost is noted in Sweep 1. **Latency** not yet profiled.)*

### Experimental log

A rolling registry of experiments on this benchmark — completed **and** planned. The §2 table above always holds the latest per-cell numbers; each entry below records, for a completed experiment, *what ran / what it showed / what it implies*, and for a planned one, the *goal / rationale / what will run / its gate*. Entries aren't deleted when done — their status flips, so the log stays a full history.

| # | Experiment | Status | Outcome / goal |
|---|---|---|---|
| 1 | **Sweep 1** — trunk × pooling @180 | ✅ Done (2026-06-28) | Leader = `impala × flatten @ 180`; aim (~41°) & presence (F1 ~0.46) are architecture-independent bottlenecks |
| 2 | **Presence calibration probe** — threshold sweep + PR-AUC on Sweep 1 checkpoints | ✅ Done (2026-06-28) | **Calibration-limited (val + test)** — leader PR-AUC ~0.79; recalibration lifts F1 ~0.49 → ~0.74–0.76, clears the ≥0.70 band on held-out test (gap ~0), no retrain/resolution needed |
| 3 | **Readout fixes on the leader** — aim reparam + presence loss + bullet-pos sharpen, re-run `impala × flatten @180` | 📋 Planned · needs sampler GO | Decloak the true metric ceiling at 180 before spending on resolution |
| 4 | **Resolution axis (gated)** — `impala × flatten @360` (+ `impala × gap @360` confirm), multi-seed winner | 📋 Planned · conditional on #3 | Test the one config (resolution) that can move aim/bullet-pos; lock the winner |
| 5 | **Principal cut + encoder→RL transfer probe** | 📋 Planned · high VOI | Validate the decode proxy against real RL usefulness (§7.1) |

#### 1 · Sweep 1 — trunk (impala vs nature) × pooling (gap vs flatten) at 180 — ✅ done (2026-06-28)

**What ran.** The first three of the ten cells, chosen to settle the two architecture axes (trunk, pooling) at a single mid resolution before spending compute on the resolution axis — and to confirm the diverse-batch sampler fix holds on full production runs (not just the one-off reshuffle spike, §6/§10). Cells: `impala × gap`, `impala × flatten`, `nature × gap`, all at 180 (320×180). Config: 12 epochs, batch 256, lr 3e-3, seed 0, `datasets/decode-v2_reshuffled` (70.8k train / 22.7k val / 26.4k test), heatmap-CE decoder (`heatmap_weight=6.0`, `heatmap_sigma=0.7`), on the local RTX 4090 (`--device cuda`). All three completed clean — no NaNs/errors. Runs: `runs/pc-{impala-gap,impala-flatten,nature-gap}-180`.

**Results** (test set, `spatial / embed-probe`; see §2 table for the full row):

| Cell | Position (wu) | Aim (°) | Presence F1 | Bullet-pos (wu) |
|---|---|---|---|---|
| impala × gap | **0.302** / 4.050 | 51.6 / 55.9 | 0.364 / 0.329 | 1.453 / 4.100 |
| impala × flatten | 0.326 / **0.539** | **41.3** / **41.2** | **0.386** / **0.464** | **1.322** / **2.043** |
| nature × gap | 4.205 / 12.833 | 85.0 / 84.8 | 0.143 / 0.230 | 5.191 / 12.533 |

**Gap to target & how to close it.** Bands from §1.1 (🟡 Good / 🟢 Excellent). "Leader best" = `impala × flatten @ 180` (the chosen direction), as `spatial / probe`. Gap is measured against 🟡 Good on the **embed-probe** (the RL-relevant artifact) unless noted.

| Metric | 🟡 Good / 🟢 Excellent | Leader best (spatial / probe) | Gap to Good | Next step to close the gap |
|---|---|---|---|---|
| **Position** (wu) | ≤ 0.30 / ≤ 0.10 | 0.326 / 0.539 | spatial **met**; probe ~1.8× over | Keep **flatten** (preserves location); push **resolution 360** for sub-pixel heatmap peaks + a few more epochs / LR-decay tail. Closest to done — likely clears Good with the 360 run already queued. |
| **Aim** (°) | ≤ 10 / ≤ 3–5 | 41.3 / 41.2 | ~4× over (need −31°) | **Biggest lever.** Re-parameterize aim as a `(sinθ, cosθ)` unit-vector regression with a von-Mises / cosine-distance loss, **up-weight the aim term**, and verify label correctness. The turret/barrel cue is tiny in pixels → **360 resolution** should help materially. Candidate for a dedicated angular head. |
| **Bullet-pos** (wu) | ≤ 0.50 / ≤ 0.30 | 1.322 / 2.043 | ~2.6× over (spatial) | Bullets are 0.08×0.04 wu and move 0.30 wu/step → **resolution-limited**: raise to **360**, **sharpen the heatmap** (↑`heatmap_weight`, ↓`heatmap_sigma`) for crisper peaks, and up-weight bullet-pos. Couples with presence (below) — only scored where a bullet is detected. |
| **Presence** (F1) | ≥ 0.70 / ≥ 0.85 | 0.386 / 0.464 | ~0.24 F1 short | **✅ Confirmed calibration-limited, val + test (Exp #2)** — PR-AUC ~0.79 (~24× chance); recalibration lifts F1 to **~0.74–0.76, clears the band on held-out test (optimism gap ~0), no retrain**. It's a precision problem (recall already ~0.94), F1-optimal threshold ~0.9 not 0.5. Fix: **lower `pos_weight`** toward single digits and/or pick the threshold on a val PR curve; focal loss optional. **Not** resolution-bound. |
| **Velocity** | *TBD (§1.1)* | 0.772 / 0.762 | *unknown* | First **pin the unit/normalization** from `norm_stats` to set a band. Caveat: single-frame velocity is **partially ill-posed** (no temporal signal beyond static motion cues) → may be inherently capped until a 2-frame input is considered. |

Three findings:

1. **Sampler fix confirmed in production.** Both impala cells land at ~0.3 wu spatial position — matching the reshuffle spike's 4.06 → ~0.40 wu prediction. No regression to the old flat-val basin. The diverse-batch productionization holds on full runs.
2. **impala ≫ nature, decisively.** `nature × gap` never escaped the predict-the-center basin (4.2 wu position, 85° aim, presence F1 0.14) — NatureCNN's shallow trunk lacks the capacity to localize here. impala is the trunk going forward.
3. **flatten beats gap where it counts.** On the spatial head the two are near-tied (gap edges position 0.302 vs 0.326; flatten wins aim, presence, bullet-pos). But on the **detached embed-probe** — the embedding RL actually reuses — it's not close: flatten retains position at **0.539 wu vs gap's 4.050 wu**, presence F1 0.464 vs 0.329. This is the expected "GAP averages away position" effect; gap's spatial head only looks good because it reads the pre-pooled feature map. Cost: flatten's decoder is ~3.6M params vs gap's ~1.2M (encoder is identical at 0.10M); the larger embedding is the price of spatial retention.

**Leader so far:** `impala × flatten @ 180` — best embed-probe retention, which is the property the downstream RL encoder needs.

**Open caveats.** Aim error (~41°) and presence F1 (~0.39) are still weak across the board — aim is genuinely hard and presence fights a ~29:1 class imbalance (high recall ~0.94, low precision ~0.23). Single seed (0) only; no resolution sweep yet.

**Proposed next steps.** Formalized as planned experiments **#2–#5** below (and tracked as actions in §6 To-do #5–#6).

#### 2 · Presence calibration probe — ✅ done (2026-06-28, val + test confirmed)

**Goal.** Determine whether the presence F1 gap (Sweep 1: ~0.39–0.49 vs the §1.1 ≥0.70 band) is **calibration**- or **discrimination**-limited, and recover F1 without a retrain.

**What ran.** Read-only diagnostic on the 3 Sweep 1 checkpoints (no training): rebuilt encoder+decoder from each saved `config` via the production builders, `eval()`+`no_grad()`, reproduced the exact production **val** split, took `sigmoid(bullet_presence logits)` over all 10 slots (micro-averaged), and computed PR-AUC (average precision), ROC-AUC, and an F1 threshold sweep (0.01–0.99). On the local 4090 (CUDA). **Sanity passed:** val size = 22,661 (= `n_val`); F1@0.5 reproduced each `results.json` val presence F1 to 4 dp; base rate = **3.26%** (matches §9.1). Scratch script (throwaway, no production edits): `.scratch/presence_calibration_probe.py`.

**Results (val split):**

| Checkpoint | Family | PR-AUC | ROC-AUC | F1@0.5 | best thr | F1@best | P/R@best | gain |
|---|---|---|---|---|---|---|---|---|
| **impala × flatten (leader)** | **spatial** | **0.790** | 0.986 | 0.483 | 0.92 | **0.746** | 0.74 / 0.76 | +0.263 |
| **impala × flatten (leader)** | **probe** | **0.778** | 0.985 | 0.491 | 0.87 | **0.730** | 0.69 / 0.78 | +0.239 |
| impala × gap | spatial | 0.794 | 0.983 | 0.425 | 0.94 | 0.743 | 0.77 / 0.71 | +0.317 |
| impala × gap | probe | 0.700 | 0.982 | 0.359 | 0.93 | 0.694 | 0.66 / 0.73 | +0.336 |
| nature × gap | spatial | 0.135 | 0.896 | 0.186 | 0.01 | 0.193 | 0.13 / 0.40 | +0.007 |
| nature × gap | probe | 0.171 | 0.916 | 0.264 | 0.01 | 0.264 | 0.15 / 0.94 | +0.000 |

**Verdict — calibration-limited (leader, decisively).** PR-AUC ≈ 0.79 against a 3.26% base rate is **~24× chance** (chance AP ≈ 0.033); ROC-AUC ≈ 0.99 — the encoder *already separates present/absent well at 180*. Pure recalibration lifts leader F1 **0.49 → 0.73–0.75, clearing the §1.1 "Good" band (≥0.70) with no retraining**. The F1-optimal threshold sits far out at **~0.87–0.92** (logit ≈ +2 to +2.4), not 0.5 — the textbook fingerprint of `pos_weight≈28.9` inflating positive logits to buy recall (R≈0.93/P≈0.33 at 0.5) at precision's cost. Both impala cells are calibration-limited; `nature × gap` is *discrimination*-limited (PR-AUC ~0.15 ≈ chance, thresholding can't help) — but that only re-confirms the dead trunk, it doesn't affect the decision.

**Test confirmation (val→test transfer) — ✅ holds.** The val-selected threshold was applied *fixed* to the held-out **test** split. Sanity passed: test size = 26,445 (= `n_test`); test F1@0.5 reproduced each `results.json` test F1 to 4 dp. Test base rate drifted modestly (3.26% val → **2.74%** test, the expected map-mix effect), but the optimal threshold barely moved, and the **optimism gap is negligible** (+0.0002 to +0.0023 F1) — the recalibration did not overfit val.

| Checkpoint | Family | val-thr | test PR-AUC | F1@val-thr (test) | F1@test-best | optimism gap |
|---|---|---|---|---|---|---|
| **impala × flatten (leader)** | **spatial** | 0.92 | 0.802 | **0.762** | 0.763 | +0.0002 |
| **impala × flatten (leader)** | **probe** | 0.87 | 0.799 | **0.739** | 0.739 | +0.0007 |
| impala × gap | spatial | 0.94 | 0.792 | 0.744 | 0.746 | +0.0023 |
| impala × gap | probe | 0.93 | 0.677 | 0.681 | 0.682 | +0.0009 |

The leader clears the §1.1 "Good" band (≥0.70) on truly held-out test in **both** families (spatial 0.762, probe 0.739). `impala × gap` probe lands just under (0.681), consistent with its weaker PR-AUC (0.68 — partly discrimination-bound). Scratch script: `.scratch/presence_threshold_transfer.py`.

**Implication for #3.** The presence fix is **"lower `pos_weight` toward single digits and/or pick the threshold on a val PR curve"** — **not** a resolution problem. Presence clears the band on recalibration alone (confirmed on held-out test), cleanly separating it from aim (~41°) and bullet-pos (~1.3 wu), the genuinely resolution-bound metrics. Focal loss is optional polish. **Caveat:** single-seed (seed 0), n-of-1 per cell; lowering `pos_weight` in #3 recenters the operating point near 0.5 rather than changing this conclusion.

#### 3 · Readout fixes on the leader — 📋 planned (needs sampler GO)

**Goal.** Decloak the true metric ceiling at 180 by fixing the architecture-independent readout problems *before* spending compute on resolution. **What runs:** apply aim re-parameterization (`(sinθ,cosθ)` + angular loss, up-weighted), the presence loss/threshold change from #2, and bullet-pos heatmap sharpening; re-run **only** `impala × flatten @ 180` (1 cell). **Why this order:** running the full grid through a broken readout confounds the fixes with config and forces a re-run (§4 one-factor attribution). **Gate:** aim/presence/bullet-pos vs the §1.1 bands at 180. Sits on the **productionized sampler** (in flight) so results land on the committed pipeline, not the one-off reshuffle.

#### 4 · Resolution axis (gated) — 📋 planned (conditional on #3)

**Goal.** Test the one config with a mechanistic reason to move aim/bullet-pos — resolution — and lock the winner. **What runs:** `impala × flatten @ 360` (and `@ 90` for the Pareto point); `impala × gap @ 360` to confirm the flatten≫gap embed-probe gap isn't resolution-specific; **≥3 seeds** on the eventual winner (§4). **Gate:** only run the cells #3 shows are still short of band; the **`nature` rows are likely droppable** (Sweep 1 shows nature dominated) — fill only for a complete-grid writeup. **Decision:** pick the accuracy↔compute Pareto cell that clears the bar (§1 decision rule).

#### 5 · Principal cut + encoder→RL transfer probe — 📋 planned (high VOI)

**Goal.** Validate that decode accuracy actually predicts RL usefulness — the proxy-metric (Goodhart) risk in §7.1. **Why:** higher value-of-information than sweep cells 4–10, and the encoder *weights* are throwaway once JEPA lands (§8) — the durable output is this probe + architecture intuition. **What runs:** a thin probe wiring the frozen leader encoder into PPO vs from-scratch. **Note:** touches the RL seam → needs Mike's sign-off (§8 phase 5). **Gate:** does the pretrained encoder help PPO? — the real verdict the decode bench is a proxy for.

---

## 3. Method — the pretraining harness

Single-frame **inverse-renderer**. (Full code reference: [`docs/components/pretraining.md`](../components/pretraining.md).)

- **Encoder** (`models.build_encoder`): a NatureCNN / ImpalaResNet **trunk** × GAP / Flatten **pooling**, with an early ÷4 **stem** sized for the 640×360 native frame.
- **Two head families:**
  - **Spatial heads** read `Encoder.features` (the `(B,C,h,w)` map) → gradient trains the encoder (the reusable artifact). **Readout being redesigned** — see §6 (the original GAP-then-FC readout was position-blind; replaced by a heatmap/soft-argmax localization decoder).
  - **Detached embed-probe** reads `Encoder.embed.detach()` → measures how much state survives pooling; *why pooling earns a place in the sweep* (a real accuracy delta on the probe).
- **Five per-group targets/losses/metrics:** position & velocity (MSE → world-unit error), aim (cosine → angular degrees), bullet-presence (pos-weighted BCE → F1), bullet-position (masked MSE).
- **Data:** shard-streaming `DecodeDataset`, map-aware 80/10/10 split (no leak), **shard-grouped sampler** (each shard decompressed once per epoch), load-time downsample.
- **CLIs:** `train` (device-agnostic, `--eval-every K` periodic val, progress reporter) · `profile` (params + latency).

---

## 4. Experimental design (the sweep)

- **Grid:** `{nature, impala} × {gap, flatten} × {360, 180, 90}` = 12 → **10 cells** (`nature@90` dropped — NatureCNN's ÷4 stem underflows a 90-row frame).
- **Resolutions:** 360 = 640×360 (native) · 180 = 320×180 · 90 = 160×90.
- **Data:** `decode-v2` (~120k samples, 10 maps) — research-confirmed *generous* for a low-diversity supervised decode.
- **Protocol to lock:** seed(s) **[OPEN — ≥3 for the final comparison]**, epochs/cell **[OPEN — re-probe after the decoder fix]**, LR / batch / normalization fixed (note: scale LR with batch size — `3e-4` was tuned for batch 32), map-aware split.
- **Rigor we're applying:** baseline + fixed protocol; one-factor attribution; the embed-probe as the pooling ablation; a data **learning curve**; a **pre-written decision rule**; exploration → confirmation; and the higher-value **encoder→RL transfer probe** as a check on the decode *proxy*.

---

## 5. Decision log

| Decision | Rationale | By | Status |
|---|---|---|---|
| Observation = real Unity pixels, single-frame | the synthetic-grid approach was dropped; decode the 52-float state from one real render | Mike | ✅ |
| **Structured per-group heads** (no FC-to-52 control) | heterogeneous fields need the right loss per type; FC-to-52 would output garbage for some | Brian | ✅ |
| **Embed-probe on `embed.detach()`** (option B) | makes pooling (GAP vs Flatten) show a *real* accuracy delta; RL-relevant retention measure | Brian | ✅ |
| **Spatial heads: heatmap + soft-argmax localization** (NOT GAP-then-FC) | GAP destroys position → coordinate regression is impossible (root cause of the flat val, §6); heatmap is the standard differentiable localization readout (matches the 2021 reference) | Brian | ✅ |
| **Add cross-entropy heatmap loss** (strong weight ~6–10, σ≈0.7), keep coord MSE for refinement | spike (§10) hit ~0.4–0.5 wu, ~5× below coord-only — but weak weights backfire and σ must match the coarse grid; CE supplies a direct localization gradient not dependent on the fragile soft-argmax escape | Brian | ✅ (pending production re-run, §10.2) |
| Resolutions **{360, 180, 90}**; `nature@90` dropped | the range the stemmed encoders support; NatureCNN underflows 90 | Brian | ✅ |
| **10-cell sweep**, pooling swept via the probe | full trunk × pooling × resolution comparison | Brian | ✅ |
| **Committed `pretraining/`** component (not scratch) | reusable, gated, shareable; a real planned component | Brian | ✅ |
| `decode-v2` ~120k; iterate against `decode-v1` pilot | research: 120k is generous for low-diversity supervised decode | Brian | ✅ |
| **Train on the 4090** (remote, Tailscale SSH) | CUDA (`cu130`), fast, free; MPS is partial/flaky | Brian | ✅ |
| Epochs / cell | — | — | ⏳ re-probe after fix |
| Sweep scope: full 10-cell **vs** principal cut | value-of-information | — | ⏳ |
| Seeds (≥3) + pre-written decision rule | variance reporting + pre-registration | — | ⏳ |
| Stem-less small-input encoder (field-norm 64–96px) | research follow-up; not this sweep | — | 📋 [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) |

---

## 6. Progress

Accomplishments to date (the harness build-out + the flat-val root-cause). Metrics-bearing sweep results live in §2; open next steps and decisions are tracked under **To do** below.

| Date | Milestone | Detail |
|---|---|---|
| 2026-06-27 | **Harness built + pipeline validated** | Single-frame state-decoder harness + embed-probe lands; runs end-to-end on CUDA, both head families report per-group metrics, progress + periodic-val stream over SSH. |
| 2026-06-27 | **Loader perf bug found + fixed** | A single-entry shard cache + global `shuffle=True` re-decompressed a whole ~580-frame `.npz` per sample (a 3k smoke hung **17 min** with the GPU idle). Replaced with a shard-grouped sampler → **~45s** (each shard read once/epoch); a `read_shard`-count regression test guards it. (PR #11) |
| 2026-06-27 | **GPU validated (4090, `torch 2.9.1+cu130`)** | `impala/180` ~20s/epoch; **~2,200 samples/sec** at `num_workers=8` / `batch=256` (~17× the default). Windows spawn works. |
| 2026-06-27 | **🔴 Root-caused — flat val (spatial heads destroyed position)** | The convergence probe showed **no val learning** — position pinned at the arena center (~5.5 err), aim ~88° (chance), presence-F1 *frozen* — at **both `lr=3e-4` and `lr=3e-3`**, train loss plateauing at ~4.55 immediately. **Cause:** the "spatial" decode heads **global-average-pool the feature map before the FC** (`decoder.py:97`, `feat.mean(dim=(2,3))`), destroying position — so the position/aim/bullet heads regress coordinates from a *position-invariant* vector and can only output the dataset mean. (This also made spatial and embed-probe families identical.) **Not LR** (10× didn't move it). The earlier small-data smoke's "loss decreased" was *overfitting* a 256-sample set, not learning. Caught by the val-curve probe + periodic-val *before* a wasted 10-cell sweep. |
| 2026-06-27 (in progress) | **Fix — heatmap / soft-argmax localization decoder** | Per locatable keypoint (tanks ×2, bullets ×10): a 1×1 conv → spatial softmax → **soft-argmax** (differentiable expected grid coordinate) → world coords. Position info is preserved and gradients teach the encoder to *fire at the entity's location*. Paired with a **confidence head** (bullet presence) and a **position-preserving regression head** (aim/velocity — directions, not locations; soft-argmax doesn't apply). *Caveat:* soft-argmax precision is bounded by feature-map resolution; our ÷4 stem makes it coarse (ties to [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)). |
| 2026-06-27 | **Heatmap decoder built; operation moved onto the 4090** | The soft-argmax localization decoder (above) is implemented with tests, replacing the position-blind GAP heads. Work now runs directly on the local RTX 4090 (`--device cuda`; `device.py` `resolve_device`/`TT_DEVICE` still routes device choice). The earlier remote-dispatch launcher (`scripts/gpu_run.py`) was removed — it existed to reach a Tailnet 4090 over SSH and is unnecessary now that the GPU is local. |
| 2026-06-27 | **🔴 Soft-argmax decoder ALSO flat on production — the "predict-center basin"** | A *second, distinct* flat-val (different cause than the GAP heads above). The heatmap/soft-argmax decoder with **coordinate-only** loss was run on the production trainer (impala/gap/180, 12 epochs, lr 3e-3): **still flat** — val position pinned at **5.54 wu**, aim ~88°, train loss stuck at the ~4.6 mean floor, spatial ≡ detached-probe (encoder learned nothing). An **overfit probe** (memorize a fixed 512-frame batch) showed the head *can* localize (4.54→0.61 wu) but only after ~500 repeated full-batch steps; a **3×3 score-head + per-keypoint-affine capacity fix overfit 512 yet STILL didn't generalize on the full 120k**. Conclusion: capacity is necessary but **not** sufficient — soft-argmax coordinate-only regression has a **weak escape gradient** when the score map is diffuse, so on noisy minibatches it never breaks symmetry out of the predict-the-mean basin. This motivated the heatmap-CE spike below. |
| 2026-06-27 | **✅ Spike — heatmap CE supervision works (strong yes, one caveat)** | A throwaway spike validated adding a **cross-entropy heatmap term** on top of coord regression. **Strong relative win:** at heatmap weight **≈6–10** (σ=0.7), val position error reached **~0.4–0.5 wu — ~5× below coord-only's ~2.2 floor** in the spike harness and **~13× below the ~5.5 trivial** (§9.4). `hm_ce` fell ~4.09 (uniform) → ~2.1, confirming the score map sharpened onto the true cell; the world→grid map (linear, x∈[−8,8], y∈[−4.1,4.1]) checked out — a trained soft-argmax lands **0.042 grid-cells** from the true player on the coarse 6×10 grid. See the two regime findings and the caveat in §10. |
| 2026-06-27 | **✅ Dataset audit → bullet-presence label bug fixed + split visibility shipped** | A read-only `decode-v2` audit (§9) found two defects gating the bullet fields and the eval. **(1) Bullet-presence label bug:** the `pos_x >= 0` presence rule mislabeled **51% of on-board bullets** (the left half of a centered arena, valid negative x) as *absent* — so the heatmap decoder would never be supervised to fire on the left. Fixed: `bullet_present` now keys on the −100 sentinel (`> -50`), single-sourced in `core/state.py` with `targets.py` delegating; regression test added; **no dataset regeneration** (raw states on disk are correct — a Python label-derivation bug only). **(2) One-map val split:** `build_splits`/`train.py` defaults 0.1→0.2 (val + test each seat ≥2 maps at 10 maps) plus INFO/WARN visibility on realized per-split map-counts + sample fractions; `split_groups` stays pure. Eng-manager + platform gate **GO**; RL seam confirmed untouched (52-float layout / sentinel / socket-actions). Commit `c936c19`. |
| 2026-06-28 | **✅ Heatmap-CE landed; production flat root-caused to the SAMPLER; fix validated by reshuffle** | Brought CE heatmap supervision into the production decoder (`losses.heatmap_ce_loss` + `targets.GridExtent`/`keypoint_targets`, fit TRAIN-only & serialized; score-logits surfaced top-level so val-eval concat stays flat; `--heatmap-weight`/`--heatmap-sigma`; coord MSE kept; **98 tests GO**). But the 12-epoch production learn-check was **FLAT (val 4.06 wu; `hm_ce` pinned at `ln 60` = the score map never sharpens)** — even the always-present player keypoints. **Not a decoder bug:** an isolation test on the *exact production code paths* but with RANDOM shuffled batches escaped (val → **0.39 wu**) while coord-only stayed stuck (~4.2) → heatmap-CE is both **necessary and sufficient-with-diversity**. **Root cause = the `ShardGroupedBatchSampler`:** it confines every batch to one shard (one episode/map), so a batch has ~no positional diversity and the heatmap gradient can't break symmetry out of the uniform map. **Validated end-to-end** by a one-off on-disk reshuffle (mix all 10 maps into every shard; `map_id` preserved → *identical* val maps, apples-to-apples): the same production trainer + heatmap-CE on the reshuffled data drops val **4.06 → 0.66 → 0.44 → 0.40 wu over epochs 0–2** (preliminary; full 12-epoch run in progress, §2 to be filled). Net fix = **heatmap-CE + cross-shard batch diversity**. |
| 2026-06-28 | **✅ Sweep 1 — trunk + pooling decided; full-run confirms the reshuffle win** | Ran the first **3 of 10 cells** (`impala × {gap, flatten}`, `nature × gap` @180, full 12 epochs, batch 256, lr 3e-3, seed 0, on `decode-v2_reshuffled`, local 4090). Three findings (full writeup + per-cell numbers in §2 → Experimental log → **Sweep 1**; target-band gap analysis added at **§1.1**): **(1)** the diverse-batch fix **holds at full training scale** — both impala cells reach val/test position **~0.30 wu** (spatial), no regression to the flat basin (matches the reshuffle preview). **(2) impala ≫ nature** — nature never left the predict-center basin (4.2 wu / 85° / F1 0.14); impala is the trunk. **(3) flatten ≫ gap on the embed-probe** (the RL-relevant artifact): **0.54 vs 4.05 wu** position, F1 0.46 vs 0.33 — gap's spatial head only looks competitive because it reads the *pre-pooled* map; once pooled, GAP averages position away. Cost: flatten decoder ~3.6M vs gap ~1.2M (encoder identical 0.10M). **Leader = `impala × flatten @ 180`.** Remaining bottlenecks are **architecture-independent**: aim ~41° (≫ ≤10° bar) and presence F1 ~0.46 (< 0.70 bar) cap every cell. §2 rows filled. |
| 2026-06-28 | **⚠️ Sampler productionized + validated on pristine data — LEARNING ✅ / PERF ❌ (NO-GO)** | Committed the windowed multi-shard sampler (`ShardWindowBatchSampler` + per-worker `_LruShardCache` capacity W+1; commit `906a974`; **sampler unit tests 7/7 pass**). Validated it through the committed pipeline on the **pristine `decode-v2_original`** (`impala × flatten @ 180`, 12 epochs, `--num-workers 8 --shard-window 8 --device cuda`; `runs/sampler-prod-validate`). **LEARNING ✅:** val position **2.15 → 0.31 wu** over 12 epochs (test pos **0.30**, aim 37.5°, presence-F1 0.55, bullet-pos 1.39) — the diverse-batch fix reproduces end-to-end on the committed sampler with **no manual reshuffle and no flat basin** → closes To-do #1's pristine-data validation. **PERF ❌ NO-GO:** **~2:55/epoch (~9× Sweep 1's ~20s)** at **~36 GB RAM**, GPU pinned at **0%** — root cause: with `num_workers=8` the LRU cache is **replicated per DataLoader worker** (8× copies + each shard decompressed ~8×, not once/epoch), so the "decompress once" guarantee only holds at `num_workers=0`. Verdict: **correctness GO, perf NO-GO** → loader rework planned (eng-manager + tech-lead, spike-gated: threaded-prefetch + GPU-side transform **vs** uncompressed-`.npy` + memmap so the OS page cache is the shared cross-process cache; GO bar < 60s/epoch + bounded RAM + learning preserved). |

### To do
Open next steps + decisions (was §7's "next steps / open questions") — all **📋 to do** unless flagged otherwise.

1. 🟠 **Productionize the diverse-batch fix — code done + learning-validated; blocked on a PERF NO-GO.** The windowed multi-shard sampler is committed (`906a974`, `ShardWindowBatchSampler` + LRU shard cache) and ✅ **validated on the pristine `decode-v2_original`** (val **2.15 → 0.31 wu**, §6 2026-06-28) — this closes both the original "(b) full-run confirm" and the "(a) reproduce the win on pristine data without a manual reshuffle." **BUT it is NOT GO on performance:** at `num_workers=8` the LRU cache replicates per DataLoader worker → **~36 GB RAM + ~9× slower (GPU-starved)** because the "decompress once/epoch" property only holds at `num_workers=0`. **Open blocker → loader rework** (eng-manager + tech-lead plan delivered; **spike-gated** between *Finalist A:* single-process threaded prefetch + shared cache + GPU-side transform [in-`pretraining/`-boundary], and *Finalist B:* uncompressed-`.npy` + `mmap` so the OS page cache is the shared cross-process cache [cleaner, but crosses into `data/` + needs a one-time `decode-v2` regen → **scope decision**]). GO bar: **< 60 s/epoch (≈ Sweep 1's ~20s target), RAM bounded, learning preserved (val 4.06 → ~0.40 wu)**. Still open too: (c) heads-up to Mike on PR #11 re: the decoder-approach change + the sampler finding. *(Validation artifacts: `datasets/decode-v2_reshuffled` (one-off shuffle), `datasets/decode-v2_original` (pristine), `runs/sampler-prod-validate` — all git-ignored, local only.)*
2. 📋 **Decide sweep scope** — full 10-cell (matches Mike's ask) vs. the **principal cut** (2–3 cells + an encoder→RL transfer probe, which is higher value-of-information than cells 4–10; see §7 proxy-metric risk). **[OPEN]**
3. ✅ **[DONE — data] Bullet-presence labels fixed** (commit `c936c19`). The `pos_x >= 0` rule that mislabeled **51% of on-board bullets** (left-half, §9.1) now keys on the −100 sentinel (`bullet_present` → `pos_x > -50`), single-sourced in `core/state.py` with `targets.py` delegating; regression test added; stats refit automatically. No dataset regeneration. Eng + platform gate GO. → the heatmap decoder is now supervised on the full arena.
4. ✅ **[DONE — eval] One-map val split mitigated** (commit `c936c19`). `build_splits`/`train.py` defaults raised 0.1→0.2 so val + test each seat ≥2 maps at 10 maps, plus INFO/WARN logging of realized per-split fractions (§9.2). The full **fractions-vs-k-fold eval policy for the final comparison remains [OPEN]** — fold into the seeds/scope pre-registration (#2 above, §4).
5. 🟠 **Continue the sweep — resolution axis for the leader** (next batch, "Sweep 2"). From Sweep 1's leader `impala × flatten @ 180`: (a) run `impala × flatten @ {360, 90}` — is 180 on the accuracy↔compute Pareto frontier, or does 360 buy enough to justify the cost? (b) `impala × gap @ 360` to confirm the flatten≫gap embed-probe gap isn't resolution-specific; (c) **multi-seed the eventual winner** (≥3 seeds, per §4) before locking the decision. The `nature` rows are likely droppable — Sweep 1 shows nature dominated — fill them only if a complete trunk comparison is wanted for the writeup.
6. 🟠 **Close the per-metric gaps** (architecture-independent; won't move with trunk/pooling — see the §2 Sweep 1 gap table + bands at §1.1).
   - **Aim (~41° vs ≤10° bar) — biggest lever.** Re-parameterize as a `(sinθ, cosθ)` unit-vector regression with a von-Mises / cosine-distance loss, up-weight the aim term, verify label correctness; the turret cue is tiny in pixels → expect **360 resolution** to help materially.
   - **Presence (F1 ~0.46 vs ≥0.70 bar) — ✅ confirmed calibration, *not* balance — val + test (Exp #2).** PR-AUC measured at **~0.79** (~24× chance) on the leader → discrimination is fine; the deficit is **precision** (recall already ~0.94), the fingerprint of *over*-correcting imbalance (`pos_weight≈29`, F1-optimal threshold ~0.9 not 0.5). **Recalibration alone lifts F1 to ~0.74–0.76, clearing the band on held-out test (optimism gap ~0) — no retrain, no resolution.** A "more balanced dataset" is the **wrong lever** (pushes further toward saturated recall + distorts the ~2.7–3.3% deploy prior). Action for #3: **lower `pos_weight`** toward single digits and/or **pick the threshold on a val PR curve**; focal loss optional.
   - **Bullet-pos (~1.3 wu vs ≤0.5 bar) — resolution-limited.** Bullets are 0.08×0.04 wu (~1–2 px @180): raise to **360**, **sharpen the heatmap** (↑`heatmap_weight`, ↓`heatmap_sigma`), up-weight bullet-pos; couples with presence (only scored where detected).

### Background
- **RL-from-pixels resolutions (~June 2026):** the field trains from-scratch encoders at **64–96px** — DrQ-v2 at 84×84 [2], DreamerV3 at 64×64 [3], all tracing back to the 84×84 Atari-DQN preprocessing norm [1]; ≥224px appears only with frozen pretrained encoders. Our smallest (160×90) is ~2× the 84×84 norm; 640×360 is ~33× (off-map) → motivates [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12).
- **Dataset size:** ~120k labeled frames is **generous** for supervised decode on a low-diversity synthetic domain (analogs: NS-VQA ~4k programs on CLEVR [4], DOPE ~120k synthetic frames on a harder 6-DoF-pose domain [5]). Binding constraints are **class imbalance** (bullet-presence **~3.3% of slots** once the §9.1 label bug is fixed — ~1.6% as currently labeled) and **map diversity** (10), not total frame count. Confirm with a **learning curve** (`--subset`).

**Sources**
1. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 2015 — 84×84 frame preprocessing (DQN/NatureCNN). [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
2. Yarats et al., "Mastering Visual Continuous Control (DrQ-v2)," 2021 — 84×84 from-scratch image RL. [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)
3. Hafner et al., "Mastering Diverse Domains through World Models (DreamerV3)," 2023 — 64×64 pixel observations. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
4. Yi et al., "Neural-Symbolic VQA," NeurIPS 2018 — symbolic decode on CLEVR from few annotated programs. [arXiv:1810.02338](https://arxiv.org/abs/1810.02338)
5. Tremblay et al., "Deep Object Pose Estimation (DOPE)," CoRL 2018 — synthetic-only training for 6-DoF pose. [arXiv:1809.10790](https://arxiv.org/abs/1809.10790)

---

## 7. Risks

*(Open next steps + decisions, including the two data/eval blockers, now live under §6 To do.)*

1. **Proxy-metric risk (Goodhart)** — decode accuracy ≠ RL usefulness. A thin **encoder→RL transfer probe** is higher value-of-information than cells 4–10 (drives the sweep-scope decision in §6 To do).
2. **Single-frame limit** — velocity is fundamentally undecodable from one frame (expected); points at frame-stacking / temporal input for RL.
3. **Localization precision vs feature-map resolution** — the ÷4 stem makes features coarse, bounding soft-argmax precision; lever is a lighter stem / higher-res features ([#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)).
4. **Levers > frame count** — class imbalance (slot-present **~3.3%** once the label bug §9.1 is fixed; ~1.6% as-labeled) and held-out-map generalization (10 maps) move the result more than raw N.

---

## 8. Roadmap to JEPA

The decode benchmark is **architecture selection + an eval harness**, not the final encoder. JEPA (Joint-Embedding Predictive Architecture [7][8]) swaps the **training objective** — supervised state-decode → **self-supervised latent prediction** (predict masked regions' *embeddings*, via a context encoder + an **EMA target encoder** + a predictor; no pixels, no labels, no negatives) — while **reusing this doc's data pipeline and the embed-probe as its evaluation metric**. The embed-probe ("how much state survives a frozen embedding") *is* the frozen-JEPA-encoder eval.

**Two destinations.** (A) a JEPA-trained **encoder feeding PPO** — representation learning only, control stays with RL; realistic near-term. (B) a JEPA **world model** with an action-conditioned predictor doing **latent-space planning** ([9], V-JEPA 2-AC) — a separate, larger research program. The phases below target (A), with (B) as the optional tail.

| Phase | Step | Notes / risk |
|---|---|---|
| 0 | **Finish the decode benchmark** (you're here) | Land the heatmap fix, run the sweep. Durable outputs = the **probe** + architecture intuition; the encoder *weights* are throwaway once JEPA lands — don't over-invest. |
| 1 | ⚠️ **Resolve ViT-vs-CNN** | Canonical JEPA is **ViT** (masks patch tokens, predicts *per-patch* targets); our trunks are conv. Choose a small ViT trunk or a conv masked-prediction variant. Determines how much of the sweep matters; ties to [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12). |
| 2 | **Single-frame (I-JEPA) vs video (V-JEPA)** | §7.2 — velocity is undecodable from one frame → favors **V-JEPA spatiotemporal** masking. Needs *clip* data (collection/dataset change), but **no labels** → far more frames usable. |
| 3 | **Build the JEPA pretrain harness** | Context encoder · EMA target · predictor · latent regression + masking. **Instrument collapse from day one** (embedding variance / rank + the decode probe) — the analog of the val-curve probe that caught the flat-val bug (§5/§6). |
| 4 | **Evaluate frozen JEPA encoder with the probe** | Run the embed-probe decode eval on the frozen JEPA encoder vs the supervised encoder — head-to-head, same metric. Add a linear/attentive probe. |
| 5 | ⚠️ **Wire frozen encoder into RL** | The "frozen vision-encoder embedding" deferred to M1. Replaces the pixel-obs path (`tank_env.draw_state` → CnnPolicy) with embedding obs. **Touches the RL seam (`DriverController` socket/actions + 52-float state) — requires Mike's sign-off**, not a unilateral change. |
| 6 | **Transfer probe (the verdict)** | §7.1 — does the JEPA encoder *actually help PPO* vs from-scratch? Decode accuracy ≠ RL usefulness (Goodhart). The gate that validates "JEPA-powered agent." |
| 7 | **(optional) Action-conditioned latent world model** | Destination (B): action-conditioned predictor → plan in latent space ([9]). A new milestone beyond M2, not a continuation of the encoder track. |

**Forks you can't defer past their phase:** ViT-vs-CNN (P1) · single-frame-vs-video (P2) · frozen-vs-fine-tuned encoder in RL (P5) · encoder-for-PPO (A) vs latent world model (B) (P7).

**Honest dependency:** P3–6 don't strictly need P0 to *finish* — the JEPA harness can start once P1 settles the architecture. P0's only durable outputs are the probe and the architecture intuition.

**Sources (JEPA)**
7. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)," CVPR 2023 — context/target encoders, EMA target, predictor, multi-block masking, latent-space prediction. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
8. Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)," 2024 — spatiotemporal masking / feature prediction on video. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
9. Meta AI, "V-JEPA 2," 2025 — action-conditioned predictor (V-JEPA 2-AC) for planning/control. *(verify exact ref before citing externally)*

---

## 9. Dataset report — `decode-v2` (measured)

Read-only diagnostics over all **207 shards / 119,869 samples / 10 maps** (states + map_ids only; frames never decompressed). These are the *measured* binding constraints + the trivial baselines that calibrate the flat-val story. Reproduce: `.scratch/decode_v2_diag.py` + `.scratch/bullet_probe.py` (seed 0, val=test=0.1 map-aware split; method described inline so the report stands alone). Run 2026-06-27.

### 9.1 ✅ Bullet-presence label bug — half the bullets were dropped (RESOLVED, commit `c936c19`)
The presence target was `pos_x >= 0` (`core/state.py:136 bullet_present`; duplicated in `targets.py:117`), written assuming "negative x = off-board." **False:** the arena is centered on the origin — tanks span x ∈ [−7.7, 7.7], and bullets span x ∈ [−7.97, 7.97]. So every left-half bullet has a valid negative `pos_x` and was mislabeled **absent**, its position masked out of the loss.

| raw bullet `pos_x` (1,198,690 slots) | count | note |
|---|---|---|
| `== -100` (sentinel, truly absent) | 1,159,597 | correct |
| `∈ [−7.97, 0)` (real, **mislabeled absent**) | **20,077** | ← the bug |
| `≥ 0` (labeled present) | 19,016 | correct |

- **51.4%** of on-board bullets (20,077 / 39,093) are labeled "no bullet."
- Corrected slot-present rate ≈ **3.26%** (vs 1.59% as-labeled); corrected BCE `pos_weight` ≈ **29.7** (vs 59.8).
- Impact: presence head learns "bullets only exist on the right"; bullet-position / heatmap heads are **never supervised on the left half**.
- **✅ Fixed:** `bullet_present` now tests the sentinel (`pos_x > -50` = `ABSENT_BULLET_SENTINEL / 2`) and is the **single source of truth** — `targets.py` delegates to it (the duplicated rule is gone); docstrings corrected; norm stats (incl. `pos_weight`) refit automatically at `build_splits` time. Regression test asserts a real negative-x bullet is present+unmasked and the sentinel slot absent+masked. No dataset regeneration — the raw 52-float states on disk were always correct.

### 9.2 ✅ Degenerate val/test split — one map each (MITIGATED, commit `c936c19`)
The map-aware split allocates by **map count**: `round(0.1 × 10) = 1` map to val, 1 to test. Seed 0 lands the **smallest** map in val and the **largest** in test, so realized sample fractions are nowhere near 10/10:

| split | samples | % | maps |
|---|---|---|---|
| train | 93,424 | 77.9% | 8 |
| **val** | **5,698** | **4.75%** | **1** (`four_pillars`) |
| **test** | **20,747** | **17.31%** | **1** (`diagonal_pillars`) |

No map leak (✔ disjoint), but **the val curve is one arena geometry** → high variance, not a generalization estimate. **✅ Mitigated:** defaults raised to `val_frac=test_frac=0.2` (val + test each seat ≥2 maps at 10 maps), and `build_splits` now logs realized per-split map-counts + sample fractions (INFO) and WARNs when a split seats <2 maps, so this degeneracy can never recur silently. **Still [OPEN]:** whether the final sweep comparison uses these fractions or k-fold over maps (fold into the seeds/scope pre-registration, §4).

### 9.3 Map balance
10 maps, **3.64× imbalance** (`diagonal_pillars` 20,747 / 17.3% → `four_pillars` 5,698 / 4.75%). Enough per-map signal everywhere; the imbalance matters mainly because it interacts with §9.2 (the smallest map being the *entire* val set).

### 9.4 Trivial baselines (what a real decoder must beat)
| field | trivial predictor | train | **val** | confirms |
|---|---|---|---|---|
| position | predict train-mean | 4.87 wu | **5.54 wu** | the doc's "~5.5 flat-val floor" exactly — arena ≈ 15.5 × 7.5 wu |
| aim | predict mean direction | 87.7° | 86.2° | the doc's "~88° = chance" |

A position decoder that doesn't beat **5.54 wu on val** has learned nothing — this is the number the re-probe must move.

### 9.5 Other field notes
- **Velocity is near-constant-magnitude:** tank speed median **1.000**, max 1.40, only **0.9%** stationary. Direction is undecodable from one frame (expected, §7.2), but the *magnitude* is ~trivially constant — a single-frame velocity head mostly learns "≈1.0."
- **Bullets are sparse & low-slot:** 85% of frames carry 0 (labeled) bullets, ≤3 ever on screen; only **slot 0** of each player is meaningfully used (slots 2–4 ≈ 0%). The 10-channel bullet heatmap has ~8 channels with almost no training signal — fixing §9.1 roughly doubles slot-0 occupancy but doesn't populate the high slots.
- **Aim** is exactly unit-norm (|aim| = 1.000 ± 0.000) → cosine loss is well-posed.

---

## 10. Heatmap-CE spike — findings, caveat, recommendation

Throwaway spike ("KL heatmap supervision basin escape", 2026-06-27) that validated cross-entropy heatmap supervision before bringing it into the production decoder. Headline result is in the §6 Progress row. Two findings change *how* we land it, plus one honest caveat.

### 10.1 Two non-obvious regime findings

1. **Heatmap weight is a regime switch, not a seasoning.** There's a **threshold**, and below it the term makes things *worse*. At σ=0.7, weights **≤1 stayed pinned at ~4.93 wu — worse than coord-only's 2.2** (barely above the ~5.5 trivial floor). A *weak* peaked-CE gradient is too feeble to teach localization but strong enough to perturb coord-only's fragile escape from the center basin and knock it back into center-collapse; localization only emerges once the **CE term dominates** (~6–10). The small-but-nonzero middle is strictly worse than either pure approach — **do not "sprinkle in a small auxiliary loss."** Either commit to it as the *primary* localization objective or leave it out.
2. **σ must match the coarse grid.** σ is the width of the target Gaussian blob *in grid cells*. On the 6×10 grid, σ≈1.5 is nearly uniform — its CE floor is `ln(60) ≈ 4.09`, i.e. **no gradient signal**. **σ≈0.7** (tight, peaked) is what worked. Corollary: escape near the threshold is **nondeterministic** — CUDA conv nondeterminism shifted weight=3.0's escape by ~150 steps between identical-seed runs.

**Terms.** *Heatmap weight* `w` — the multiplier in `total = coord_loss + w · heatmap_loss`; because the two losses live on different numeric scales, `w` sets their relative pull (finding: `w` must be large, ~6–10, so the heatmap term dominates). *σ* — target-blob width in cells; tight = "the tank is in *this* cell," broad = a smear that covers everything → no "here not there" signal on a coarse grid. *Escape / threshold* — "escape" is the step where training breaks out of the predict-center basin and error drops sharply; "threshold" is the minimum `w` for that to happen reliably. *CE floor `ln(60)`* — 6×10 = 60 cells, so a uniform prediction gives cross-entropy `ln(60) ≈ 4.09`; seeing `hm_ce` start there and fall confirms the map went from "no idea" to "peaked on the right cell." *Warmup/schedule* — start `w` high to guarantee the escape, then decay it so the coord MSE can refine precision once the blob is on the right cell.

### 10.2 🟡 Honest caveat — the spike did not reproduce the production stuck-at-5.5 baseline

The spike's coord-only baseline **escaped to ~2.2**, not the production "pinned at ~5.5" (§6 root-cause). Its in-memory minibatch loop is more escape-friendly than the production trainer (likely the difference vs true single-pass streaming + all loss groups pulling the encoder + LR handling). So the spike proves the **relative** win — heatmap CE reaches a ~5× sharper floor and supplies a *direct* localization gradient that doesn't depend on the fragile soft-argmax escape — but it does **not** independently prove it rescues the exact production failure. **That last step must be confirmed on the real trainer** (the §6-To-do re-probe).

### 10.3 Recommendation — bring heatmap CE into the production decoder, deliberately

- **Strong weight (~6–10, heatmap ≳ coord)**, **tight σ (~0.7 cells** for the 6×10 grid), keep the coord MSE + `_CoordAffine` for refinement, implement as **cross-entropy** (no NaN guard needed), and consider a **heatmap-dominant warmup** that decays.
- The **world→grid extent** (±8 / ±4, shared across all 10 maps) needs a clean home — most naturally **alongside the norm stats in `targets.py`, fit from the data**.
- **The real test:** re-run the production 12-epoch learn-check and confirm it beats the **actual** stuck-at-5.5 baseline.
- Bigger change than the last fix (touches the loss, targets/calibration, and config) → **route through the eng team**, and **heads-up to Mike on PR #11** about the decoder-approach change.

---

## References
- **Code:** [`docs/components/pretraining.md`](../components/pretraining.md) · PR #11 (harness) · PR #4 (plan)
- **Research briefings:** RL-from-pixels resolutions; supervised dataset sizes *(produced this session)*
- **Issues:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) parent · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) phases · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) follow-up
- **Datasets:** `decode-v1` (~20k) · `decode-v2` (~120k) — both manifested (provenance + machine)
