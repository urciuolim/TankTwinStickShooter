# Agent team (v1) — charter

**Status: bootstrapping.** The v0 flat team is archived in `archive/agents-v0/`. This is the in-progress v1.

## Operating model — a matrix
- **Functional teams** (the "rows") each have a **manager** who owns the discipline's *practice*, defines ownership areas, dispatches workers, and is the **adversary** for their workers' output.
- **Ownership areas / components** (the "columns") are carried as **skills** — opt-in, scoped context. NOT per-directory `CLAUDE.md` (which loads globally into every agent and does not scope) and NOT per-component agents. In Claude Code, filesystem location is cosmetic; scoping is always *explicit* — skills (context), tools (capability), unique names (identity).
- **Adversarial chain at every tier:** workers ← manager ← Director. Nothing self-certifies. ("Done" = a check passed, not "looks done.")
- **Models:** all agents on Opus (quality over cost — Claude Max). Tune reasoning *effort* per task, not the model.

## Teams
| Team | Manager | Members | Status |
|---|---|---|---|
| **Research** | `research-manager` | `research-docs`, `research-source`, `research-empirical` | **staffed** |
| **Engineering** | `eng-manager` | `builder` (Python) + `unity-builder` (C#/Unity, `Assets/`); specialty builders on demand | **staffed** |
| **Infra / platform** | `infra-manager` | `code-reviewer`, `evaluation`, `repo-steward`, `documentarian` staffed; `ci-cd`, `cloud` deferred | **staffed** |

### Rules that distinguish the teams
- **Researchers** run code only as *throwaway spikes* (empirical lane); they return findings, not edits.
- **Builders** are the only agents that write production code — single-threaded writes, worktree isolation, component context via skills.
- **Infra / platform agents are read-only verdict-emitters** — they produce GO / NO-GO conclusions against explicit criteria and never write code; a NO-GO routes back to engineering. `repo-steward` earns its slot by *objective isolation*: a dedicated repo-health objective a builder would deprioritize. Deterministic boundary checks via CI tooling (e.g. import-linter) are its instrument, not its replacement.
- **The one writer on the platform team is the `documentarian`** — it writes documentation only (`./docs/` Markdown, never `src/`) as the post-GO tail of the pipeline: build → platform GO → docs update → commit. Separation of duties holds because it documents code it didn't write and that already passed the gate; the infra-manager accuracy-checks the docs against the source.

## This is a deliberate bet — so it's instrumented
The field is immature and the recent evidence is contested. We are establishing this org model ahead of settled practice, on purpose. Leading indicators we watch to know if it's working (else we collapse layers):
1. **Delegation actually happens** — large tasks go to the team, not the main session. (Small SWE tasks the Director may do directly.)
2. **Managers catch real things** in adversarial review — revision/rejection rate > 0; not rubber-stamping.
3. **Steward KPI moves** — the adds:deletes ratio improves; consolidation actually happens.
4. **Cost stays justified** — tokens/task stays sane.

## Provenance
Built from a research round (docs + source + Claude-Code-mechanics lanes) on AI-agent-team best practices, plus our own v0 retro. The engineering / infra teams and the component skills are detailed *after* the "how do we sustainably build this RL/PBT repo" research round.
