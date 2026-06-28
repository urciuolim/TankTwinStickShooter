# Encoder Pretraining — Design Doc

**Living source of truth** for the single-frame encoder decode benchmark: method · decisions · results in one place. Update this as decisions land and the sweep runs.

- **Status:** harness built + GPU-validated; **flat-val hit twice** — first the position-blind GAP heads (fixed by the soft-argmax decoder), then the soft-argmax decoder's own **predict-center basin** (coord-only escape gradient too weak; capacity fix insufficient, §6) → **heatmap-CE fix spike-validated** (§10: ~0.4–0.5 wu, ~5× below coord-only); next = land it in the production decoder **via the eng team (in progress)** + confirm on the real trainer (§10.2); sweep blocked on that.
- **Owner:** Brian (CTO-delegated eval/encoder direction).
- **Tracking:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) (parent) · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) (phases) · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) (follow-up). **Code:** PR #11. **Plan:** PR #4.
- **Last updated:** 2026-06-27.

---

## 1. Goal & research question

**Can a single-frame vision encoder recover the game state from one rendered frame — which fields, and what's the best decode-accuracy-per-compute encoder to hand to RL?**

- **Metric:** per-field val decode error — position (world units), aim (angular degrees), velocity, bullet-presence (F1), bullet-position — plus compute (params, forward latency). Reported for **both** head families (spatial + detached embed-probe).
- **Decision rule** *(to finalize before the sweep):* pick the config on the accuracy↔compute **Pareto frontier** that clears the downstream-RL "good enough" bar. **[OPEN]**
- **Non-goals:** not best-possible (architecture exploration is later, [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)); not the competitive eval framework; single-frame only.

---

## 2. Results

Per-cell decode performance from the 10-cell sweep. **Pending the decoder fix + a confirming re-probe** (see §6 Progress) — cells are scaffolded with the reporting columns; metrics fill in as the sweep runs. Each metric is reported for **both head families** as `spatial / embed-probe`.

- **Position / Velocity / Bullet-pos:** world-unit error (lower better). **Aim:** angular degrees (lower better). **Presence:** F1 (higher better). **Params / Latency:** forward compute.

| Config | Res | Params | Latency (ms) | Position err (wu) | Aim err (°) | Velocity err | Presence F1 | Bullet-pos err |
|---|---|---|---|---|---|---|---|---|
| nature × gap | 360 |  |  |  |  |  |  |  |
| nature × gap | 180 |  |  |  |  |  |  |  |
| nature × flatten | 360 |  |  |  |  |  |  |  |
| nature × flatten | 180 |  |  |  |  |  |  |  |
| impala × gap | 360 |  |  |  |  |  |  |  |
| impala × gap | 180 |  |  |  |  |  |  |  |
| impala × gap | 90 |  |  |  |  |  |  |  |
| impala × flatten | 360 |  |  |  |  |  |  |  |
| impala × flatten | 180 |  |  |  |  |  |  |  |
| impala × flatten | 90 |  |  |  |  |  |  |  |

*(`nature@90` dropped — NatureCNN's ÷4 stem underflows a 90-row frame.)*

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

### To do
Open next steps + decisions (was §7's "next steps / open questions") — all **📋 to do** unless flagged otherwise.

1. 📋 **Implement heatmap-CE in the production decoder, then verify → run the 10-cell sweep.** The spike (§10) proved the *relative* win but did **not** reproduce the production stuck-at-5.5 baseline (§10.2). Next: bring CE heatmap loss into the production decoder per the §10.3 recipe (strong weight ~6–10, σ≈0.7, world→grid extent in `targets.py`, keep coord MSE) **via the eng team**, then re-run the 12-epoch learn-check and confirm it beats the **actual ~5.54 wu** baseline (§9.4) before the sweep fills in §2. Heads-up to Mike on PR #11 re: the decoder-approach change.
2. 📋 **Decide sweep scope** — full 10-cell (matches Mike's ask) vs. the **principal cut** (2–3 cells + an encoder→RL transfer probe, which is higher value-of-information than cells 4–10; see §7 proxy-metric risk). **[OPEN]**
3. ✅ **[DONE — data] Bullet-presence labels fixed** (commit `c936c19`). The `pos_x >= 0` rule that mislabeled **51% of on-board bullets** (left-half, §9.1) now keys on the −100 sentinel (`bullet_present` → `pos_x > -50`), single-sourced in `core/state.py` with `targets.py` delegating; regression test added; stats refit automatically. No dataset regeneration. Eng + platform gate GO. → the heatmap decoder is now supervised on the full arena.
4. ✅ **[DONE — eval] One-map val split mitigated** (commit `c936c19`). `build_splits`/`train.py` defaults raised 0.1→0.2 so val + test each seat ≥2 maps at 10 maps, plus INFO/WARN logging of realized per-split fractions (§9.2). The full **fractions-vs-k-fold eval policy for the final comparison remains [OPEN]** — fold into the seeds/scope pre-registration (#2 above, §4).

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
