# utils

A **leaf CLI toolbox** over the rest of the stack — self-contained developer/operator tools run as
`python -m pop_trainer.utils.<tool>`. The tools **inspect / report on** the artifacts the production
components produce; they are conveniences, **never part of any runtime code path**.

**Boundary (a one-way sink):** `utils` MAY import [core](core.md) / [models](models.md) / [rl](rl.md)
(and any other component) plus torch / stable-baselines3 / gymnasium / stdlib — but **NOTHING in
`pop_trainer` outside `utils` imports `pop_trainer.utils`**, so a tool can reach across components
without ever creating a cycle. The package `__init__` is intentionally import-light (importing the
package pulls in no heavy deps; each tool's imports live in its own module)
([`utils/__init__.py:8-13`](../../src/pop_trainer/utils/__init__.py)).

## Key entry points

[`utils.model_info`](../../src/pop_trainer/utils/model_info.py) — inspect a trained SB3 PPO
checkpoint's architecture + parameter budget. Run it as:

```bash
uv run python -m pop_trainer.utils.model_info <ckpt.zip>
```

- **Loads on CPU, schedules skipped.** [`load_ppo_cpu`](../../src/pop_trainer/utils/model_info.py)
  loads with inspect-only `custom_objects` (constant `learning_rate` / `lr_schedule` / `clip_range`),
  so no env / learning-rate context is needed
  ([`model_info.py:51-55,93-101`](../../src/pop_trainer/utils/model_info.py)).
- **Prints** the observation + action spaces, the full `model.policy` repr, the **active encoder
  trunk** class name, and id-deduped per-section parameter counts (total / trainable / frozen)
  ([`format_model_info`, `model_info.py:182-197`](../../src/pop_trainer/utils/model_info.py)).
- **Active-trunk resolution.** [`_resolve_trunk_class`](../../src/pop_trainer/utils/model_info.py)
  reads `policy.features_extractor.encoder.trunk.__class__.__name__` so the active [models](models.md)
  trunk (`CnnTrunk` / `ResNetTrunk` / `GroupNormCNN`) is visible at a glance; `None` (printed
  `(none)`) when the policy has no such path
  ([`model_info.py:134-144`](../../src/pop_trainer/utils/model_info.py)).
- **Param-count correctness (the load-bearing detail).** SB3's `ActorCriticPolicy` **shares the one
  features-extractor instance** across its `features_extractor` / `pi_features_extractor` /
  `vf_features_extractor` references, so a naive per-name sum **triple-counts** the encoder. Every
  count here is **deduped by `id(p)`** first
  ([`_dedup_by_id`, `model_info.py:104-117`](../../src/pop_trainer/utils/model_info.py)), so the
  policy total is exactly the sum over the UNIQUE policy params
  ([docstring, `model_info.py:13-19`](../../src/pop_trainer/utils/model_info.py)).
- **Pure core, testable.** The numeric work lives in
  [`collect_model_info`](../../src/pop_trainer/utils/model_info.py), which RETURNS a frozen
  [`ModelInfo`](../../src/pop_trainer/utils/model_info.py) (no I/O, no print) so the numbers are
  testable without scraping stdout; `format_model_info` renders it and `main` prints it
  ([`model_info.py:147-171`](../../src/pop_trainer/utils/model_info.py)).

## Pulls from (upstream)

- **[rl](rl.md)** — indirectly: it loads an SB3 PPO checkpoint produced by
  [`rl.train`](rl.md#the-train_local-integrator) and reads the
  [`EncoderExtractor`](rl.md#encoderextractor--the-policyencoder-seam) → [`Encoder`](models.md)
  `.trunk` to name the active vision trunk. (It imports `stable_baselines3.PPO` directly; the encoder
  path is reached by attribute, not import.)
- **[models](models.md)** — the encoder trunk classes (`CnnTrunk` / `ResNetTrunk` / `GroupNormCNN`)
  whose class name it surfaces (by attribute, not import).
- **`stable_baselines3`** (`PPO`, [`model_info.py:34`](../../src/pop_trainer/utils/model_info.py)) /
  torch (type-only) / stdlib.

## Pushes to (downstream)

Nothing — `utils` is a **one-way sink**. No component imports it; the only output is the printed
report. Adding a tool here can never introduce an import cycle.

## Where it sits in the run

Off the live loop entirely — a post-hoc inspection layer. After a [training run](rl.md) writes a
`model_*.zip`, point `model_info` at it to confirm the obs/action spaces, the active trunk, and the
(id-deduped) parameter budget — the quick "what is actually in this checkpoint?" check. See the
[runbook](../runbook.md#7-inspect-a-checkpoint).

---
[← back to index](../README.md)
