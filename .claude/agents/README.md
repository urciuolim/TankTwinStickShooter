# Agent team (v1) — charter

**Status: bootstrapping.** The v0 flat team was retired (it lives in git history); this is the in-progress v1.

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
- **The one writer on the platform team is the `documentarian`** — it writes documentation only (`./docs/` Markdown, never `src/`). It is the post-GO tail of the pipeline: build → platform gate, and **on GO the infra-manager dispatches the documentarian and accuracy-checks its docs, returning GO _with_ docs** → the Director commits code + docs together. Separation of duties holds: the documentarian documents code it didn't write and that already passed the gate; the infra-manager verifies the docs against the source. (Running the doc agent is the infra-manager's job at GO, not the Director's after it.)

## Live / e2e test runs — the time-budget ladder
Unit tests run on every change; **live runs** (anything that launches the Unity build — e2e smokes, collection validation) cost wall-clock, so they are tiered and budget-gated. **No agent autonomously starts a long run.**

**Test tiers**
- **Tier 0** — UTs + contracts. Every change, fast, in the default `pytest` run.
- **Tier 1** — fast e2e smoke (`@pytest.mark.e2e`, EXCLUDED from the default run, auto-skips with no `build/` exe): a small live collection (~2 workers, 1-2 episodes, one `switch_arena`), ~1-3 min. The FUNCTIONAL / integration net for the `data`/`env`/`protocol`/Unity live seam that UTs structurally can't reach. Run by `evaluation` at the gate for live-seam changes, and by the Director in the commit smoke.
- **Tier 2** — full / scale collection or soak (the ~1 hr kind). Ad-hoc, never automatic.

**An e2e is NOT a scale net.** A 2-worker smoke never reproduces a resource blow-up (e.g. the shard-buffer OOM). Scale / resource bugs are caught by a *static* net instead — a pre-flight memory-estimate guard plus fast OOM-class unit tests that mock low RAM. A green e2e ≠ "scales fine."

**The escalation ladder (wall-clock per live run).** An agent self-estimates (`workers × episodes × ~3 s/episode`) and checks BEFORE launching. A run estimated over the actor's tier STOPS and bubbles up a written ask (what changed, why a live run is warranted, est. time):
- **Worker ≤ 3 min** — autonomous.
- **Manager ≤ 10 min** — the dispatching manager signs off.
- **Director ≤ 30 min** — bubble up to the Director.
- **> 30 min** — the CTO signs off.

Budgets are wall-clock; tune as the engine's throughput changes.

## This is a deliberate bet — so it's instrumented
The field is immature and the recent evidence is contested. We are establishing this org model ahead of settled practice, on purpose. Leading indicators we watch to know if it's working (else we collapse layers):
1. **Delegation actually happens** — large tasks go to the team, not the main session. (Small SWE tasks the Director may do directly.)
2. **Managers catch real things** in adversarial review — revision/rejection rate > 0; not rubber-stamping.
3. **Steward KPI moves** — the adds:deletes ratio improves; consolidation actually happens.
4. **Cost stays justified** — tokens/task stays sane.

## Provenance
Built from a research round (docs + source + Claude-Code-mechanics lanes) on AI-agent-team best practices, plus our own v0 retro. The engineering / infra teams and the component skills are detailed *after* the "how do we sustainably build this RL/PBT repo" research round.
