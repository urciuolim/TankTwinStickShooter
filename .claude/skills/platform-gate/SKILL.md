---
name: platform-gate
description: The platform team's GO/NO-GO rubric for reviewing/gating an engineering deliverable. Load when acting as infra-manager, code-reviewer, repo-steward, or evaluation.
---

# Platform GO / NO-GO rubric

The platform team is the cross-team adversarial gate on engineering. It emits VERDICTS, never patches. This rubric is iterable — bars can move.

## Hard gates — any one fails → NO-GO

**It works** (`evaluation`)
- Component tests are green (`uv run pytest`).
- Imports cleanly in the `uv` venv; no missing deps.
- Entry points smoke-run.
- Stated behavioral contracts hold (e.g. the 52-float round-trips; a claimed equivalence matches).

**It's correct** (`code-reviewer`)
- No correctness bug vs the contract/spec.
- No security issue.
- No hallucinated / nonexistent API or package (imports + symbols actually exist).
- No new/changed behavior without a test.
- No scope creep beyond the assigned component.

**It's structurally sound** (`repo-steward`)
- No dependency-boundary violation (`core ← {models, env, data} ← {pretraining, rl}`; `core` imports nothing internal).
- No import cycle.
- No duplication of existing logic; no superseded code left beside its replacement (add-X-delete-Y honored).
- Right package placement.

**Contract met** (all)
- Contains what the component's skill specifies, and nothing it forbids.
- The component's Definition-of-Done is satisfied.

## Unit tests & coverage
- Run tests WITH coverage measured: `uv run pytest --cov=pop_trainer.<component> --cov-report=term-missing` (requires the `pytest-cov` dev dep).
- **Coverage is reported on every gate** (the % + the uncovered lines) as `evaluation` evidence — always measured, never skipped.
- The test bar is **contract-aware**: pure functions/contracts MUST be covered; framework/trainer glue is exempt (we do not test the trainer). **Uncovered pure-contract logic is a blocking finding; uncovered glue is advisory.**
- Soft target (tunable): pure-logic components (e.g. `core`) ~90%+. Not a blunt global cutoff — tie a low number to *which* untested lines actually matter.

## Advisory (recorded, does NOT block)
Style/naming nits; soft cohesion/size flags (the ~1000-line *signal*, not a hard cap); nice-to-haves. Noted for later, never a NO-GO on their own.

## Aggregation (`infra-manager`)
1. **Adversarially vet every finding** — downgrade weak/unsupported ones (this is what prevents a false-positive NO-GO). Demand `file:line` + severity.
2. **GO** iff zero *valid* blockers across all four dimensions AND evidence (including the coverage report) was shown.
3. **NO-GO** = the valid blockers, each with `file:line` + why + fix direction → routed back to `eng-manager`.
