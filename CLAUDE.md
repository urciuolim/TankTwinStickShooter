# CLAUDE.md

Operational guide for agents on the Tank Twin Stick Shooter revival. Load-bearing rules only — see `@Plan.md` (revival plan), `@docs/revival-research-2026.md` (2026 stack research), `@docs/engineering-standards.md` (coding standards), `@docs/definition-of-done.md`, and `@.claude/agents/README.md` (agent org) for the full picture.

## What this is
Unity 6.5 (2D twin-stick tank game) as the simulator + a Python 3.12 reinforcement-learning driver (Stable-Baselines3 PPO, population-based self-play) over a TCP-JSON protocol. Milestones: M0 playable game (two humans, Xbox pads + shared keyboard) → M1 single-agent PPO → M2 population training. Cluster target: GCP.

## Environment (YOU MUST)
- Always use an isolated venv (managed by `uv`); NEVER the system Python. Python 3.12.
- Cross-platform: no `fork`/`forkserver` (use `spawn` on Windows); no `os.system("zip"/"rm"/"cp"/"mv")` (use stdlib `zipfile`/`shutil`/`pathlib`); build commands with `subprocess` arg-lists, not shell strings.
- Unity 6.5 editor: `C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe`. Headless: `-batchmode -quit -accept-apiupdate -logFile -`.

## Project gotchas (an agent that only reads code will miss these)
- JSON consumed by Python must be STRICT (no trailing commas / leading-dot floats). Unity's Newtonsoft tolerates them; Python's `json` will not.
- The socket protocol is unframed today (`recv(1024)` assumes one JSON object per read) — don't rely on that; framing is planned.
- No runtime `UnityEditor` imports in shipping C# (breaks standalone builds).
- **DO NOT** change the RL seam without explicit sign-off: `DriverController`'s socket/`actions` path and `GameController.UpdateState()`'s 52-float state layout — M1 depends on them being stable.
- Observations are PIXELS: a synthetic RGB grid drawn from state by `tank_env.draw_state` (CnnPolicy), NOT a Unity render. (Future: frozen vision-encoder embedding — deferred to M1.)
- The game is a **Python-clocked simulator**: it blocks on `AcceptTcpClient` and Python drives the clock/opponent — there is NO standalone human-play mode yet. Human *input* is read locally by Unity; AI opponents are Python. Opening `Arena.unity` alone throws NREs (it needs the Driver scene). See `docs/game-architecture.md`.

## Standards — apply your phase's section of `@docs/engineering-standards.md`
- Python: `uv` + `ruff` (lint+format) + `pytest`; src-layout `pop_trainer` package; test pure functions/contracts, not the trainer.
- Unity/C#: asmdefs (runtime + 2 test); Test Framework 1.7; `.editorconfig` + Microsoft.Unity.Analyzers; extract pure logic from MonoBehaviours.

## Repo etiquette
- One concern per change; branch-per-task (`unity/…`, `python/…`, `fix/…`, `protocol/…`); keep it reversible.
- Verify every new dependency exists on PyPI and pin it (lockfile). Never commit secrets / GCP keys.
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Definition of Done: `@docs/definition-of-done.md`. "Done" = a check passed, not "looks done."

## Don't touch
- Generated `Library/`, and the `runs/ models/ logs/ datasets/` artifacts (git-ignored).
- The RL seam (above) without sign-off.
