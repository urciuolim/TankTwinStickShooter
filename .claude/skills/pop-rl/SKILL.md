---
name: pop-rl
description: Contract + boundaries for the rl/ component of src/pop_trainer. Load when building or reviewing rl/.
---

# Component: `rl/` — PPO + self-play

**Responsibility:** the online RL training — SB3 PPO (CnnPolicy over pixels / the pretrained encoder) on the env, plus self-play wiring. Self-play is a WRAPPER over the trainer (ML-Agents `ghost/` precedent), **never woven into the PPO core**.

**Contains:**
- the PPO train loop + config; policy / feature-extractor wiring to `models`.
- **`EncoderExtractor`** — an SB3 `BaseFeaturesExtractor` subclass wrapping `models.build_encoder`,
  the seam between the env's pixel obs and the SB3 policy/value heads.
- callbacks / eval hooks.
- the opponent roster (the `OpponentProvider` "frozen policy + version tag" seam — a WRAPPER over
  the trainer, NEVER in env or the PPO core), resolving the SAME `agents` selectors via the
  canonical `agents` registry (M1 = SCRIPTED only: coverage/random/noop; FrozenSelf deferred to P2).
- self-play opponent management + ELO (rating logic lives HERE, not in core).

**`EncoderExtractor` contract (T2, Phase 1):**
- An `stable_baselines3.common.torch_layers.BaseFeaturesExtractor` subclass. FIXED architecture:
  **NatureCNN trunk + flatten pooling at 360×640** (`EncoderConfig(trunk="nature", pooling="flatten")`).
  NO trunk/pooling/resolution config knobs in Phase 1 — the encoder/resolution ablation is a
  teammate's SEPARATE `models` benchmark; do not re-expose it here.
- `__init__(observation_space, *, checkpoint: Path | None = None, freeze: bool = False)`:
  - build the encoder via `models.build_encoder`;
  - compute `features_dim` = `encoder.embedding_dim(input_hw)` at the obs `H×W` (the obs space's
    spatial dims) and pass it to `super().__init__(observation_space, features_dim)`;
  - if `checkpoint` is not None → `encoder.load_state_dict(torch.load(checkpoint))` — THIS extractor
    OWNS the checkpoint load (`models.from_pretrained` does NOT exist; do NOT call it);
  - if `freeze` → set `requires_grad_(False)` on the encoder params.
- `forward(obs) -> (N, embed_dim)` over SB3's `(N, C, H, W)` float-`[0,1]` obs (SB3 normalizes the
  uint8 frame before the extractor; do not re-normalize).
- **CORRECTIONS (from [[rl-training-approach]] — they CONTRADICT the old reference trainer; apply
  them):**
  * Freezing does NOT remove params from SB3's optimizer (SB3 builds the optimizer over ALL policy
    params). Freeze works ONLY by `requires_grad=False` so the grad stays `None`. The freeze TEST
    asserts **encoder weights UNCHANGED after one optimizer step (or grads are `None`)**, NOT
    optimizer membership.
  * 84×84 FAILS this encoder (the stem collapses the map) — the obs is **360×640**.
  * `models.from_pretrained` does NOT exist — the extractor owns the `load_state_dict`.

**Boundaries:** imports `core`, `env`, `models`, `agents`. Does NOT import `pretraining` or `data`
(the pretrained encoder is consumed as a loaded checkpoint artifact — `EncoderExtractor` loads the
`state_dict` itself; there is NO `models.from_pretrained`). The opponent roster reaches the scripted
agents through the canonical `agents` registry, NOT through `data`. No cycles.

**Inspiration (do NOT copy):** `reference/tank_twin_m1/{train,evaluate,callbacks,elo}.py`; the 2021 `PythonScripts/` PPO scripts.
