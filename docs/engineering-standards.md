# Engineering standards (revival, 2026)

Research-derived standards for bringing this repo "up to snuff within reason" — toward production quality, proportional to a solo-maintained research repo, not enterprise gold-plating. Derived from a three-lane research sprint (Python, Unity/C#, agent-ops), verified against live 2026 sources (Anthropic best-practices docs, a CSA Apr-2026 slopsquatting research note, current PyPI/Unity docs). Each engineer applies their section as they reach that phase; `code-reviewer` and the [Definition of Done](definition-of-done.md) enforce it.

## Python (`PythonScripts/` → `tank_twin` package)

- **Env/deps:** `uv` (one tool: env + deps + Python pin). Commit `uv.lock`; `uv python pin 3.12`. Torch GPU wheel via `[tool.uv.sources]` (CUDA index) — the one fiddly step; verify the index URL against the installed driver at setup.
- **Layout:** `src/` layout, one installable package `tank_twin` (config, protocol, env, rewards, opponents, training, evaluation, arenas), thin `scripts/` entry points, `tests/`. PEP 621 `pyproject.toml`; console-scripts (`train-local`, `eval-local`). Migrate incrementally — pure functions (ELO, opponent-selection, reward, state-flip) first; leave Unity-coupled scripts as shims importing the package.
- **Lint/format:** `ruff` only (lint + format; black is retired). Modest rules `E,F,I,UP,B,SIM`, line length 100. Per-file ignores for `tests/` + legacy scripts during migration. pre-commit hook ids are `ruff-check` / `ruff-format`.
- **Types:** light, non-strict, on pure-logic modules only. `ignore_missing_imports` for sb3/torch/gymnasium. Not `--strict`.
- **Tests (`pytest`):** test pure functions + contracts, NOT the trainer. Targets: ELO math, opponent weighting/selection, reward logic, state-flip & 52-vector split, protocol framing (split/coalesced reads vs a fake socket), env contract via `check_env`, path/process construction (Windows vs POSIX). Coverage ~50–70% of the package, ~90% pure-logic, ~0 on torch/sb3 glue. Mark Unity-needing tests `integration` (opt-in).
- **Reproducibility (light):** `seed_everything` helper; per-run `runs/<id>/manifest.json` (git SHA+dirty, seed, resolved config, arena, lib versions). TensorBoard (free via SB3). No MLflow/W&B/Hydra yet.
- **Load-bearing prerequisite:** the gym→gymnasium migration (5-tuple `step`, `reset(seed=,options=)`, `terminated`/`truncated`) must land before `check_env` passes — stage tests: pure-logic first, env-contract after the port.
- **Overkill to avoid:** MLflow/Hydra, `--strict` mypy, ruff `D`/`ANN` rules, coverage gates >70%, release automation, testing that PPO converges.

## Unity / C# (`Assets/Scripts/`)

- **Assemblies:** add THREE asmdefs only — runtime `TankTwinStickShooter` (references Newtonsoft), `…EditModeTests`, `…PlayModeTests`. Skip the 6-module split (overkill for 4 scripts). The runtime asmdef turns the stray-`UnityEditor`-import bug class into a compile error.
- **Testing:** Unity Test Framework 1.7 (bundled; 2.x is experimental — don't). EditMode for pure logic (arena parse, config parse, the `float[52]` state encoder, protocol dispatch); a thin PlayMode smoke layer (~4 tests: scene boot, timer-end, player-death-end, bullet collision/despawn). Skip visual/UI and live-socket tests.
- **Testability:** extract pure logic out of MonoBehaviours into plain classes — `ArenaLayout` parser, `StateEncoder.Encode` (the RL observation contract — test it hard), typed `GameConfig`, `ProtocolHandler` (socket I/O stays in the MonoBehaviour). Same work as Plan Phase 7. Land EditMode tests capturing CURRENT behavior BEFORE refactoring.
- **Style:** `.editorconfig` (repo root; Unity 6 honors it for analyzer severity) + `Microsoft.Unity.Analyzers`. Rules as warning/suggestion, not error (don't break the build on existing inconsistent code). Nullable reference types only on extracted pure-logic files (`#nullable enable`), NOT project-wide (fights MonoBehaviour serialization).
- **VCS hygiene:** `.gitattributes` with UnityYAMLMerge for `*.unity`/`*.prefab`/`*.asset`. Serialization is already Force Text + Visible Meta (good). Skip Git LFS (small 2D game).
- **Unity-6 trap:** disabling Domain Reload for faster PlayMode iteration leaves the `static instance` singletons in `DriverController`/`GameController` un-reset between runs → stale-state bugs. Gate behind an explicit static reset (`[RuntimeInitializeOnLoadMethod]`) or leave domain reload on.
- **Overkill to avoid:** the 6-asmdef split, UPM packaging, StyleCop/large rulesets, project-wide NRT, analyzer rules as errors on day one.

## Agent-ops (how the team runs)

- **`CLAUDE.md`** at root, lean (<150 lines), auto-loaded every session: load-bearing operational rules only (commands, venv discipline, gotchas, etiquette, what-not-to-touch). Litmus per line: *"would removing this cause an agent to make a mistake?"* Narrative stays in `Plan.md`/`docs/`.
- **Two gates, distinct jobs:** `code-reviewer` (fresh context, reads the diff for defects) AND `verification-lead` (runs the artifact, proves it works). Neither edits code (separation of duties).
- **Definition of Done:** see [definition-of-done.md](definition-of-done.md). "Done" = a check passed, not "looks done."
- **Supply-chain guardrail (real, not theoretical):** ~20% of AI-generated code references nonexistent packages (slopsquatting; 43% reproduce on re-run). Commit lockfiles; every new dependency must exist on PyPI with a plausible age/maintainer; route new deps to the human.
- **Diffs:** one concern per change, branch-per-task, reversible, AI-authorship trailer, no secrets.
- **Multi-agent failure modes:** infinite handoff loops (Director owns sequencing/decisions), cascade failure + hallucination propagation (per-stage verify/review gates), cost blowup (keep the team small; don't route one-line fixes through the full gate).
- **Proportionality:** reserve the full scope→build→review→verify gate for milestone-level / multi-file work; a one-sentence diff skips the ceremony.
