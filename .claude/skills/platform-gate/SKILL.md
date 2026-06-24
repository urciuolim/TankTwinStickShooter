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
- No legacy/historical narration in comments — comments describe only the CURRENT code (not what was removed or how it used to be). We build fresh.

**It's structurally sound** (`repo-steward`)
- No dependency-boundary violation — each component imports only what its component skill permits (e.g. `data → env → core`: collection routes through the gym env); `core` is the dependency-free root.
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

## C# / Unity deliverables (the same four gates apply)
- **Tests:** Unity Test Framework EditMode tests for pure logic (extracted from MonoBehaviours). `evaluation` runs them **headless**: `Unity.exe -batchmode -runTests -testPlatform EditMode -testResults <path> -quit` (editor at `C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe`). Expect minutes — C# gates are slower (editor launch).
- **Coverage:** measured + reported via the Unity Code Coverage package (same expectation as Python).
- **Structure (`repo-steward`):** asmdef boundaries respected; **no runtime `UnityEditor` imports in shipping code** (breaks standalone builds); pure logic extracted for testability.
- **Contract:** JSON output stays strict-compatible (no trailing commas / leading-dot floats); the protocol / 52-float state layout matches `pop_trainer.core`.

## Advisory (recorded, does NOT block)
Style/naming nits; soft cohesion/size flags (the ~1000-line *signal*, not a hard cap); nice-to-haves. Noted for later, never a NO-GO on their own.

## Aggregation (`infra-manager`)
1. **Adversarially vet every finding** — downgrade weak/unsupported ones (this is what prevents a false-positive NO-GO). Demand `file:line` + severity.
2. **GO** iff zero *valid* blockers across all four dimensions AND evidence (including the coverage report) was shown.
3. **NO-GO** = the valid blockers, each with `file:line` + why + fix direction → routed back to `eng-manager`.
4. **On GO, before returning: run the docs.** Dispatch the `documentarian` to update the affected `./docs/` for the deliverable, then accuracy-check its output against the code. Return GO **with** the docs updated + checked — GO means "clearing to commit," so the docs must match the code. (A NO-GO skips this.)
