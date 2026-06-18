# Definition of Done

Proportional to a revived research repo (one human + an agent team). Items **1–3, 5, 8 are must-haves on every change**; 4, 6, 7 scale up at milestone (M0/M1/M2) boundaries. Reserve the full gate for milestone-level / multi-file work — a one-sentence diff skips the ceremony.

A change is **Done** when:

1. **Acceptance criteria met** — the task's stated criteria (and the relevant `Plan.md` phase criteria) are satisfied.
2. **Tests pass and cover the change** — relevant `pytest` / Unity EditMode green (PlayMode too for gameplay-state changes); new behavior has tests, including named edge cases (protocol framing, rewards, ELO, paths, config).
3. **Format/lint clean** — `ruff` (Python) clean; no new analyzer warnings (C#).
4. **No unverified dependencies** — any new package exists on PyPI, is pinned in the lockfile, and isn't suspiciously new.
5. **Reviewer approved** — `code-reviewer` found no correctness / security / hallucination / scope gaps (or they're fixed).
6. **Verified to run** (milestone work) — `verification-lead` PASS with evidence (commands, exit codes, log excerpts).
7. **Docs updated** when behavior/interface changed (`Plan.md` / runbook / `CLAUDE.md`); artifacts git-ignored.
8. **Reviewable, single-concern, reversible commit** with the AI-authorship trailer.

NOT required here (overkill): 85–90% coverage gates, CI/CD (a current non-goal), SBOM tooling, staging deploys.
