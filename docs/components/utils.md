# utils

A **leaf CLI toolbox** over the rest of the stack — self-contained developer/operator tools run
as `python -m pop_trainer.utils.<tool>`. The tools **inspect / report on** artifacts the
production components produce; they are conveniences, **never part of any runtime code path**.

**Boundary (a one-way sink):** `utils` MAY import [`core`](core.md) / [`models`](models.md) /
[`rl`](rl.md) (and any other component) plus torch / stable-baselines3 / gymnasium / stdlib — but
**NOTHING in `pop_trainer` outside `utils` imports `pop_trainer.utils`**. The stack never depends
on the tools, so a tool can reach across components without ever creating a cycle. The package
`__init__` is intentionally import-light (importing the package pulls in no heavy deps; each tool's
imports live in its own module — [`utils/__init__.py`](../../src/pop_trainer/utils/__init__.py)).

## Key entry points

- [`utils.model_info`](../../src/pop_trainer/utils/model_info.py) — inspect a trained SB3 PPO
  checkpoint's architecture + parameter counts. Run it as:

  ```bash
  uv run python -m pop_trainer.utils.model_info <ckpt.zip>
  ```

  It loads the checkpoint **on CPU** ([`load_ppo_cpu`](../../src/pop_trainer/utils/model_info.py),
  `model_info.py:93-101`) with the saved schedules **skipped** on load (the inspect-only
  `custom_objects` set a constant `learning_rate` / `lr_schedule` / `clip_range`, so no env /
  learning-rate context is needed, `model_info.py:51-55`), then prints:
  - the observation space and the action space (the gymnasium spaces' `repr`),
  - the full `model.policy` repr,
  - the **active encoder trunk** class name — resolved from
    `policy.features_extractor.encoder.trunk.__class__.__name__`
    ([`_resolve_trunk_class`](../../src/pop_trainer/utils/model_info.py), `model_info.py:134-144`),
    so the active vision trunk (`CnnTrunk` / `ResNetTrunk` / `GroupNormCNN`) is visible at a glance;
    `None` (printed `(none)`) when the policy has no such path,
  - **id-deduped per-section parameter counts** (total / trainable / frozen) for the whole policy
    and the features extractor.

  - **Param-count correctness (the load-bearing detail).** SB3's `ActorCriticPolicy` **shares the
    one features-extractor instance** across its `features_extractor` / `pi_features_extractor` /
    `vf_features_extractor` references, so a naive per-name sum **triple-counts** the encoder. Every
    count here is **deduped by `id(p)`** first ([`_dedup_by_id`](../../src/pop_trainer/utils/model_info.py),
    `model_info.py:104-117`), so the policy total is exactly
    `sum(p.numel() for unique p in model.policy.parameters())` and the extractor section is counted
    once (`model_info.py:13-19`).
  - **Pure core, testable.** The numeric work lives in
    [`collect_model_info`](../../src/pop_trainer/utils/model_info.py) (`model_info.py:147-171`),
    which RETURNS a frozen [`ModelInfo`](../../src/pop_trainer/utils/model_info.py) (no I/O, no
    print) so the numbers are testable without scraping stdout;
    [`format_model_info`](../../src/pop_trainer/utils/model_info.py) renders it and
    [`main`](../../src/pop_trainer/utils/model_info.py) prints it.

## Pulls from (upstream)

- [rl](rl.md) — indirectly: it loads an SB3 PPO checkpoint produced by
  [`rl.train`](rl.md#the-train_local-integrator-seam) and reads the
  [`EncoderExtractor`](rl.md#the-encoder-seam) → [`Encoder`](models.md) `.trunk` to name the active
  vision trunk. (It imports `stable_baselines3.PPO` directly; the encoder path is reached by
  attribute, not import.)
- [models](models.md) — the encoder trunk classes (`CnnTrunk` / `ResNetTrunk` / `GroupNormCNN`)
  whose class name it surfaces (by attribute, not import).
- Plus `stable_baselines3` / torch (type-only) / stdlib.

## Pushes to (downstream)

Nothing — `utils` is a **one-way sink**. No component imports it; the only output is the printed
report. Adding a tool here can never introduce an import cycle.

## Where it sits in the run

Off the live loop entirely — a post-hoc inspection layer. After a [training run](rl.md) writes a
`model_*.zip`, point `model_info` at it to confirm the obs/action spaces, the active trunk, and the
(id-deduped) parameter budget — the quick "what is actually in this checkpoint?" check. See the
[runbook](../runbook.md#7-inspect-a-checkpoint-cli).

---
[← back to index](../README.md)
