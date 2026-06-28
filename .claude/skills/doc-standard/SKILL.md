---
name: doc-standard
description: The documentation rubric — how to build the navigable ./docs/ Markdown site (root hub → per-component pages with pulls-from/pushes-to → architecture map → runbook). Load when acting as the documentarian or accuracy-reviewing docs.
---

# Documentation standard — the navigable `./docs/` site

The docs are a **website-like, multi-page Markdown site** a newcomer navigates by clicking. It MUST work both in a local Markdown viewer (VS Code, etc.) AND rendered on GitHub. So: plain `.md` files, **relative links** between them, **mermaid** code-fences for diagrams (GitHub renders them natively). No build step, no HTML, no site generator.

## Audience + bar
Someone onboarding who wants to (a) grasp the whole system fast, then (b) drill into ONE component well enough to work on it — including **what it pulls from and what it pushes to** — then (c) run things end to end. **High-level and concise — onboarding-grade, NOT exhaustive API reference** (the code + its comments are the API). If a page can't be skimmed in ~2 minutes, it's too long.

## Site structure (hub-and-spoke under `./docs/`)
- **`docs/README.md` — the hub / root page.** A short high-level overview (what the project is) + a **component map at a glance** (a mermaid dependency graph) + a navigation index linking to every component page, the architecture page, and the runbook. This is the page GitHub shows when you open `docs/`.
- **`docs/components/<component>.md` — one page per component** (core, models, env, data, agents, …). Each page, in this order:
  1. **Responsibility** — 1–2 sentences.
  2. **Key classes / entry points** — the public surface, each linked to its source, e.g. `[TankEnv](../../src/pop_trainer/env/tank_env.py)`. Say what it is, not every method.
  3. **Pulls from (upstream)** — the components/classes it depends on, each **linked to that component's page** so the reader can navigate up the graph.
  4. **Pushes to (downstream)** — what consumes it, linked the same way.
  5. **Where it sits in the run** — one line tying it to the end-to-end flow.
  6. *(optional)* a small **mermaid** class-interaction diagram when it clarifies.
  7. A **"← back to index"** relative link to `../README.md`.
- **`docs/architecture.md`** — the cross-component **class-to-class interaction map** in one mermaid graph, plus the dependency direction (`core` is the dependency-free root; everything points inward). The big picture on one page.
- **`docs/runbook.md`** — the end-to-end path: clone → `uv sync` → build the Unity game → play an episode / collect. Copy-pasteable commands grounded in the actual entry points + config (READ them — `play.py`'s `main`, the `BuildScript` method, the config keys). Do NOT run them — the infra-manager accuracy-check executes them to verify.

## Navigation rules (the "website" feel)
- Every page links to its neighbours with **relative `.md` links** (`[env](env.md)`, `[back](../README.md)`) — clickable on GitHub AND followable in a local viewer.
- The **pulls-from / pushes-to links ARE the navigation**: a reader on `data` clicks straight through to `env` and `agents` because those are what it pulls from.
- Keep a consistent breadcrumb/back-to-index on every page.

## Accuracy (non-negotiable — a doc that lies is worse than none)
- Document only **built, GO'd** code. Ground every structural claim in `file:line` from the actual source — **read it, don't infer**.
- **Read + write only — never execute.** The documentarian has no run/build/test tools; running commands or the harness to confirm reality is the **infra-manager accuracy-check's** job, not yours.
- The infra-manager accuracy-checks the docs against the code before commit; the `file:line` grounding makes that fast.

## Scope + boundary
- Write **ONLY under `./docs/`**. NEVER touch `src/`, tests, or code comments (those are the builders' job under the comment policy).
- **Bootstrapping run:** build the whole site for the currently built+GO'd components, AND **prune** any existing `./docs/` page that is stale and unrelated to `src/pop_trainer` or the current agent team.
- **Per-GO run:** incremental — update only the touched component's page + any changed pulls-from/pushes-to edges, and the architecture diagram if an edge moved.
