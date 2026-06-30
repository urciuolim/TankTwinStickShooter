# CLAUDE.md

Operational guide for agents on the Tank Twin Stick Shooter revival. Load-bearing rules only — see `@Plan.md` (revival plan), `@docs/engineering-standards.md` (coding standards), `@docs/definition-of-done.md`, and `@.claude/agents/README.md` (agent org) for the full picture.

## What this is
Unity 6.5 (2D twin-stick tank game) as the simulator + a Python 3.12 reinforcement-learning driver (Stable-Baselines3 PPO, population-based self-play) over a TCP-JSON protocol. Milestones: M0 playable game (two humans, Xbox pads + shared keyboard) → M1 single-agent PPO → M2 population training. Cluster target: GCP.

## Environment (YOU MUST)
- Always use an isolated venv (managed by `uv`); NEVER the system Python. Python 3.12.
- Cross-platform: no `fork`/`forkserver` (use `spawn` on Windows); no `os.system("zip"/"rm"/"cp"/"mv")` (use stdlib `zipfile`/`shutil`/`pathlib`); build commands with `subprocess` arg-lists, not shell strings.
- Unity 6.5 editor: `C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe`. Headless: `-batchmode -quit -accept-apiupdate -logFile -`.

## Compute / GPU execution (YOU MUST)
- **Run torch work on the local CUDA GPU (RTX 4090).** Any torch work that trains/backprops or benchmarks GPU compute — full runs, sweeps, smokes, *and* throwaway diagnostics — MUST run on CUDA. Pass `--device cuda` (or `--device auto`, which detects CUDA first). MPS/CPU is a last resort, used only when **no** CUDA is available, and you MUST say so out loud when you fall back. Example: `uv run python -m pop_trainer.pretraining.train --epochs 8 --device cuda`.
- **Never hardcode `torch.device('cuda')`** (or `'mps'`/`'cpu'`). Use `resolve_device()` / `--device auto`, which honors the `TT_DEVICE` env override. Hardcoding the device is what once stranded a diagnostic on MPS while the GPU sat idle.
- **Carry this into delegations.** When you spawn a subagent that will touch torch, put the CUDA rule (`--device cuda`) in its task prompt — do NOT tell it to "smoke on cpu/mps."

## Project gotchas (an agent that only reads code will miss these)
- JSON consumed by Python must be STRICT (no trailing commas / leading-dot floats). Unity's Newtonsoft tolerates them; Python's `json` will not.
- The socket protocol is unframed today (`recv(1024)` assumes one JSON object per read) — don't rely on that; framing is planned.
- No runtime `UnityEditor` imports in shipping C# (breaks standalone builds).
- **DO NOT** change the RL seam without explicit sign-off: `DriverController`'s socket/`actions` path and `GameController.UpdateState()`'s 52-float state layout — M1 depends on them being stable.
- Observations are PIXELS: the REAL rendered Unity frame (CnnPolicy), captured from the live game render; the old synthetic RGB grid (`tank_env.draw_state`) is DROPPED. (Future: frozen vision-encoder embedding over that frame.)
- The game is a **Python-clocked simulator**: it blocks on `AcceptTcpClient` and Python drives the clock/opponent — there is NO standalone human-play mode yet. Human *input* is read locally by Unity; AI opponents are Python. Opening `Arena.unity` alone throws NREs (it needs the Driver scene). See `docs/game-architecture.md`.

## Standards — apply your phase's section of `@docs/engineering-standards.md`
- Python: `uv` + `ruff` (lint+format) + `pytest`; src-layout `pop_trainer` package; test pure functions/contracts, not the trainer.
- Unity/C#: asmdefs (runtime + 2 test); Test Framework 1.7; `.editorconfig` + Microsoft.Unity.Analyzers; extract pure logic from MonoBehaviours.

## Repo etiquette
- One concern per change; branch-per-task (`unity/…`, `python/…`, `fix/…`, `protocol/…`); keep it reversible.
- Verify every new dependency exists on PyPI and pin it (lockfile). Never commit secrets / GCP keys.
- Definition of Done: `@docs/definition-of-done.md`. "Done" = a check passed, not "looks done."

## The gate is NOT optional (YOU MUST)
The platform/infra team is the checks-and-balances, and it must NEVER be bypassed — not for time pressure, not for "technical difficulties" (agent deaths, app restarts, flaky tooling). The flow is always `build → gate → docs → commit`:
- Every deliverable goes through the infra-manager GATE: the independent read-only reviewers (code-reviewer / repo-steward / evaluation) emit GO/NO-GO, and at GO the documentarian updates `./docs/`. Only then does the Director commit.
- The Director's inline `ruff`/`pytest`/smoke checks SUPPLEMENT the gate — they NEVER replace the independent review or the docs step, and never license a self-certified commit.
- If ANY team/agent is ERRORING (research, engineering, or platform/infra — it may be a Claude Code app issue): STOP and FLAG it to the user. Do NOT bypass it, do NOT blindly re-run or grind, do NOT self-certify. The user decides how to proceed.
- The user MAY explicitly authorize a deviation (a bypass, or a specific way to use a team) — then follow the user's instruction. But NEVER suggest or propose deviating from the established path yourself.
- Seam changes additionally require a LIVE gated run + explicit CTO sign-off (see Don't touch).

A process abandoned under pressure is not a process. The whole point of the checks-and-balances is that they are not skippable.

## Don't touch
- Generated `Library/`, and the `runs/ models/ logs/ datasets/` artifacts (git-ignored).
- The RL seam (above) without sign-off.
