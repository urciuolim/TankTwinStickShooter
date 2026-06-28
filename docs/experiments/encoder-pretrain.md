# Encoder Pretraining — Design Doc

**Living source of truth** for the single-frame encoder decode benchmark: method · decisions · results in one place. Update this as decisions land and the sweep runs.

- **Status:** harness built + GPU-validated; **flat-val root-caused** (the spatial heads GAP away position) → **heatmap-localization-decoder fix in progress**; sweep blocked on the fix.
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

Accomplishments to date (the harness build-out + the flat-val root-cause). Metrics-bearing sweep results live in §2.

| Date | Milestone | Detail |
|---|---|---|
| 2026-06-27 | **Harness built + pipeline validated** | Single-frame state-decoder harness + embed-probe lands; runs end-to-end on CUDA, both head families report per-group metrics, progress + periodic-val stream over SSH. |
| 2026-06-27 | **Loader perf bug found + fixed** | A single-entry shard cache + global `shuffle=True` re-decompressed a whole ~580-frame `.npz` per sample (a 3k smoke hung **17 min** with the GPU idle). Replaced with a shard-grouped sampler → **~45s** (each shard read once/epoch); a `read_shard`-count regression test guards it. (PR #11) |
| 2026-06-27 | **GPU validated (4090, `torch 2.9.1+cu130`)** | `impala/180` ~20s/epoch; **~2,200 samples/sec** at `num_workers=8` / `batch=256` (~17× the default). Windows spawn works. |
| 2026-06-27 | **🔴 Root-caused — flat val (spatial heads destroyed position)** | The convergence probe showed **no val learning** — position pinned at the arena center (~5.5 err), aim ~88° (chance), presence-F1 *frozen* — at **both `lr=3e-4` and `lr=3e-3`**, train loss plateauing at ~4.55 immediately. **Cause:** the "spatial" decode heads **global-average-pool the feature map before the FC** (`decoder.py:97`, `feat.mean(dim=(2,3))`), destroying position — so the position/aim/bullet heads regress coordinates from a *position-invariant* vector and can only output the dataset mean. (This also made spatial and embed-probe families identical.) **Not LR** (10× didn't move it). The earlier small-data smoke's "loss decreased" was *overfitting* a 256-sample set, not learning. Caught by the val-curve probe + periodic-val *before* a wasted 10-cell sweep. |
| 2026-06-27 (in progress) | **Fix — heatmap / soft-argmax localization decoder** | Per locatable keypoint (tanks ×2, bullets ×10): a 1×1 conv → spatial softmax → **soft-argmax** (differentiable expected grid coordinate) → world coords. Position info is preserved and gradients teach the encoder to *fire at the entity's location*. Paired with a **confidence head** (bullet presence) and a **position-preserving regression head** (aim/velocity — directions, not locations; soft-argmax doesn't apply). *Caveat:* soft-argmax precision is bounded by feature-map resolution; our ÷4 stem makes it coarse (ties to [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)). |
| 2026-06-27 | **Heatmap decoder + CUDA-first execution infra built** | The soft-argmax localization decoder (above) is implemented with tests, replacing the position-blind GAP heads. Added `scripts/gpu_run.py` (CUDA-first launcher — local-CUDA → remote-SSH → refuse, audited to `logs/gpu_run.jsonl`), `device.py` `TT_DEVICE` routing, and the `CLAUDE.md` compute rule, so torch work never silently lands on MPS again. **Operation moving to the 4090 PC; next is verifying the fix there (the val curve must finally move) → then the sweep.** |
| — (pending) | **10-cell sweep** | Blocked on the decoder fix + a confirming re-probe. Fills in §2. |

### Background
- **RL-from-pixels resolutions (~June 2026):** the field trains from-scratch encoders at **64–96px** — DrQ-v2 at 84×84 [2], DreamerV3 at 64×64 [3], all tracing back to the 84×84 Atari-DQN preprocessing norm [1]; ≥224px appears only with frozen pretrained encoders. Our smallest (160×90) is ~2× the 84×84 norm; 640×360 is ~33× (off-map) → motivates [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12).
- **Dataset size:** ~120k labeled frames is **generous** for supervised decode on a low-diversity synthetic domain (analogs: NS-VQA ~4k programs on CLEVR [4], DOPE ~120k synthetic frames on a harder 6-DoF-pose domain [5]). Binding constraints are **class imbalance** (bullet-presence ~8%) and **map diversity** (10), not total frame count. Confirm with a **learning curve** (`--subset`).

**Sources**
1. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 2015 — 84×84 frame preprocessing (DQN/NatureCNN). [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
2. Yarats et al., "Mastering Visual Continuous Control (DrQ-v2)," 2021 — 84×84 from-scratch image RL. [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)
3. Hafner et al., "Mastering Diverse Domains through World Models (DreamerV3)," 2023 — 64×64 pixel observations. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
4. Yi et al., "Neural-Symbolic VQA," NeurIPS 2018 — symbolic decode on CLEVR from few annotated programs. [arXiv:1810.02338](https://arxiv.org/abs/1810.02338)
5. Tremblay et al., "Deep Object Pose Estimation (DOPE)," CoRL 2018 — synthetic-only training for 6-DoF pose. [arXiv:1809.10790](https://arxiv.org/abs/1809.10790)

---

## 7. Open questions / risks / next steps

1. ~~[BLOCKER] flat val~~ → **root-caused (position-blind GAP heads) + fix in progress** (heatmap-localization decoder). Sweep blocked until the fix lands **and a re-probe confirms the val curve actually moves** (position error drops below the trivial ~5.5).
2. **Proxy-metric risk** — decode accuracy ≠ RL usefulness (Goodhart). A thin **encoder→RL transfer probe** is higher value-of-information than cells 4–10.
3. **Single-frame limit** — velocity is fundamentally undecodable from one frame (expected); points at frame-stacking / temporal input for RL.
4. **Sweep scope** — full 10-cell (matches Mike's ask) vs. the **principal cut** (2–3 cells + an RL probe).
5. **Localization precision vs feature-map resolution** — the ÷4 stem makes features coarse, bounding soft-argmax precision; lever is a lighter stem / higher-res features ([#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12)).
6. **Levers > frame count** — class imbalance (presence ~8%) and held-out-map generalization (10 maps) move the result more than raw N.

---

## 8. Roadmap to JEPA

The decode benchmark is **architecture selection + an eval harness**, not the final encoder. JEPA (Joint-Embedding Predictive Architecture [7][8]) swaps the **training objective** — supervised state-decode → **self-supervised latent prediction** (predict masked regions' *embeddings*, via a context encoder + an **EMA target encoder** + a predictor; no pixels, no labels, no negatives) — while **reusing this doc's data pipeline and the embed-probe as its evaluation metric**. The embed-probe ("how much state survives a frozen embedding") *is* the frozen-JEPA-encoder eval.

**Two destinations.** (A) a JEPA-trained **encoder feeding PPO** — representation learning only, control stays with RL; realistic near-term. (B) a JEPA **world model** with an action-conditioned predictor doing **latent-space planning** ([9], V-JEPA 2-AC) — a separate, larger research program. The phases below target (A), with (B) as the optional tail.

| Phase | Step | Notes / risk |
|---|---|---|
| 0 | **Finish the decode benchmark** (you're here) | Land the heatmap fix, run the sweep. Durable outputs = the **probe** + architecture intuition; the encoder *weights* are throwaway once JEPA lands — don't over-invest. |
| 1 | ⚠️ **Resolve ViT-vs-CNN** | Canonical JEPA is **ViT** (masks patch tokens, predicts *per-patch* targets); our trunks are conv. Choose a small ViT trunk or a conv masked-prediction variant. Determines how much of the sweep matters; ties to [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12). |
| 2 | **Single-frame (I-JEPA) vs video (V-JEPA)** | §7.3 — velocity is undecodable from one frame → favors **V-JEPA spatiotemporal** masking. Needs *clip* data (collection/dataset change), but **no labels** → far more frames usable. |
| 3 | **Build the JEPA pretrain harness** | Context encoder · EMA target · predictor · latent regression + masking. **Instrument collapse from day one** (embedding variance / rank + the decode probe) — the analog of the val-curve probe that caught the flat-val bug (§5/§6). |
| 4 | **Evaluate frozen JEPA encoder with the probe** | Run the embed-probe decode eval on the frozen JEPA encoder vs the supervised encoder — head-to-head, same metric. Add a linear/attentive probe. |
| 5 | ⚠️ **Wire frozen encoder into RL** | The "frozen vision-encoder embedding" deferred to M1. Replaces the pixel-obs path (`tank_env.draw_state` → CnnPolicy) with embedding obs. **Touches the RL seam (`DriverController` socket/actions + 52-float state) — requires Mike's sign-off**, not a unilateral change. |
| 6 | **Transfer probe (the verdict)** | §7.2 — does the JEPA encoder *actually help PPO* vs from-scratch? Decode accuracy ≠ RL usefulness (Goodhart). The gate that validates "JEPA-powered agent." |
| 7 | **(optional) Action-conditioned latent world model** | Destination (B): action-conditioned predictor → plan in latent space ([9]). A new milestone beyond M2, not a continuation of the encoder track. |

**Forks you can't defer past their phase:** ViT-vs-CNN (P1) · single-frame-vs-video (P2) · frozen-vs-fine-tuned encoder in RL (P5) · encoder-for-PPO (A) vs latent world model (B) (P7).

**Honest dependency:** P3–6 don't strictly need P0 to *finish* — the JEPA harness can start once P1 settles the architecture. P0's only durable outputs are the probe and the architecture intuition.

**Sources (JEPA)**
7. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)," CVPR 2023 — context/target encoders, EMA target, predictor, multi-block masking, latent-space prediction. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
8. Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)," 2024 — spatiotemporal masking / feature prediction on video. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
9. Meta AI, "V-JEPA 2," 2025 — action-conditioned predictor (V-JEPA 2-AC) for planning/control. *(verify exact ref before citing externally)*

---

## References
- **Code:** [`docs/components/pretraining.md`](../components/pretraining.md) · PR #11 (harness) · PR #4 (plan)
- **Research briefings:** RL-from-pixels resolutions; supervised dataset sizes *(produced this session)*
- **Issues:** [#3](https://github.com/urciuolim/TankTwinStickShooter/issues/3) parent · [#5–8](https://github.com/urciuolim/TankTwinStickShooter/issues/5) phases · [#12](https://github.com/urciuolim/TankTwinStickShooter/issues/12) follow-up
- **Datasets:** `decode-v1` (~20k) · `decode-v2` (~120k) — both manifested (provenance + machine)
