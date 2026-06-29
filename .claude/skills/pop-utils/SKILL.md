---
name: pop-utils
description: Contract + boundaries for the utils/ component of src/pop_trainer — a leaf CLI toolbox (model inspection etc.). Load when building or reviewing utils/.
---

# Component: `utils/` — leaf CLI toolbox (no production component depends on it)

**Responsibility:** small, runnable developer/operator TOOLS over the rest of the stack —
inspect / introspect / report on artifacts the production components produce (e.g. dump a trained
SB3 PPO checkpoint's architecture + param counts). Each tool is a self-contained CLI module run via
`python -m pop_trainer.utils.<tool>`. These are CONVENIENCES, never part of any production code path:
nothing the trainer / env / data pipeline does at runtime routes through here.

**Contains:**
- **`model_info`** — load an SB3 PPO checkpoint on CPU (schedules skipped via `custom_objects`),
  print the observation/action spaces, the full `model.policy` repr, the ACTIVE encoder trunk class
  name (`policy.features_extractor.encoder.trunk.__class__.__name__`), and per-section parameter
  counts (total / trainable / frozen) **DEDUPED by `id(p)`** — the features-extractor encoder is
  SHARED across the policy's features/pi/vf sub-modules, so a naive per-child sum triple-counts it.
  The numeric work returns a structured result (a dataclass/dict) so it is testable WITHOUT scraping
  stdout; a thin `main()` prints it.
- future inspection / reporting tools follow the same shape: pure compute that RETURNS data + a thin
  printing `main(argv)` + a `python -m` entry point. No import side-effects at module load.

**Boundaries:** `utils/` is a LEAF. It MAY import `core` / `models` / `rl` (and `env` / `agents` /
`data` / `pretraining` / `play` if a tool needs them) plus torch / stable-baselines3 / gymnasium /
stdlib. It MUST NOT be imported BY any production component: nothing in `src/pop_trainer/` OUTSIDE
`utils/` may `import pop_trainer.utils...` — not `core`, `models`, `env`, `agents`, `data`,
`pretraining`, `rl`, or `play`. This keeps `utils/` a one-way sink (tools depend on the stack; the
stack never depends on the tools), so it can freely reach across components without creating a cycle.
A repo-steward boundary check asserts "no production component imports `utils`."

**Inspiration (do NOT copy):** a retired scratchpad `load_model.py` that printed a checkpoint's
architecture by hand — the intent (eyeball a trained model's shape) survives; build the tool fresh,
and get the SHARED-extractor `id(p)` dedupe right (the old by-name sum over-counted the encoder).
