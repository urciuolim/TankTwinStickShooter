# Revival agent team

Coding-agent org for the Tank Twin Stick Shooter 2026 revival. The **Director** is the orchestrating session (reports to the CTO); these files are the team it dispatches.

## Org

- **Research** — `research-lead` → `rl-pbt-analyst`, `engine-bridge-analyst`, `cloud-mlops-analyst`
- **Engineering** — `eng-lead` → `game-sim-engineer`, `training-engineer`, `platform-infra-engineer`
- **Quality** — `verification-lead` (runs it / proves it works) and `code-reviewer` (reads the diff for defects) — both independent → Director, neither edits code.

Ephemeral workers can be spun up per task on top of these standing roles.

## Orchestration pattern

Managers (`research-lead`, `eng-lead`) run in two modes. The Director sequences:

1. Run the manager to **scope** a goal into worker tasks.
2. Run the **workers** (the manager may dispatch them directly if nested agents are supported; otherwise the Director runs them).
3. Run the manager again to **integrate / check** and synthesize.

`verification-lead` and `code-reviewer` are intentionally separate from the builders and cannot edit code (separation of duties — builders don't grade their own homework). The two gates do different jobs: **`code-reviewer` reads the diff** for correctness/security/hallucinated-APIs/scope; **`verification-lead` runs the artifact** to prove the milestone works. Both feed the [Definition of Done](../../docs/definition-of-done.md).

## Standards

All agents follow `CLAUDE.md` (auto-loaded) and `docs/engineering-standards.md`. "Done" means a check passed — see `docs/definition-of-done.md`.

## Conventions

- Canonical project docs: `Plan.md`, `docs/revival-research-2026.md`, `docs/unity6-upgrade-runbook.md`.
- Milestones: M0 playable game (two humans, Xbox pads + shared keyboard) → M1 single-agent PPO → M2 population-based training.
- Always isolated venvs (never system Python); keep everything GCP-deployable and reproducible.
