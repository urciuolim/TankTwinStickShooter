# Runbook — end to end

Clone → set up the Python env → build the Unity game → watch two agents play. Every command
below is verified against this repo. Commands assume the repo root
(`C:\src\TankTwinStickShooter`). The Python env is managed by **`uv`** — never the system Python.

## 1. Python environment

```bash
uv sync
```

Creates / updates the isolated venv from `pyproject.toml` + `uv.lock` (Python 3.12, pinned).
Verify:

```bash
uv run python --version              # -> Python 3.12.x
uv run python -c "import pop_trainer; print('ok')"
```

Run the tests (pure-logic + contracts; Unity-needing tests are opt-in `integration`):

```bash
uv run pytest -m "not integration and not e2e"
```

## 2. Build the Unity game

The trainer/demo needs a standalone Windows build of the simulator. The headless build entry
point is [`BuildScript.BuildWindows`](../Assets/Editor/BuildScript.cs), invoked via the Unity
6.5 editor in batchmode:

```bash
"C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe" \
  -batchmode -quit -accept-apiupdate -logFile - \
  -projectPath "C:\src\TankTwinStickShooter" \
  -executeMethod BuildScript.BuildWindows
```

Output goes to `build/TankTwinStickShooter.exe` (git-ignored). `StreamingAssets`
(`config.json` + `Arenas/*.json` + the demo config) ships into the player automatically.
`BuildScript` exits non-zero on failure so the batchmode caller can detect it.

> The editor upgrade itself is a one-time GUI step — see the
> [Unity 6 upgrade runbook](unity6-upgrade-runbook.md). For how the built game runs (it is a
> **Python-clocked simulator** with no standalone human mode), see
> [game architecture](game-architecture.md).

## 3. Run the demo (live build + two agents)

With the build in place, watch two visibly-different policies play one episode:

```bash
uv run python -m pop_trainer.demo
```

This launches `build/TankTwinStickShooter.exe` **windowed** with
`Assets/StreamingAssets/demo_config.json` (which enables `obs_pixels` at 640×360 — required,
since the env always reads a pixel frame each step), connects over TCP, runs one episode of
`player1 = aim-at-player2` vs `player2 = explorer`, prints a trace, and tears down.

See the full flag surface:

```bash
uv run python -m pop_trainer.demo --help
```

Useful flags (defaults shown): `--exe build/TankTwinStickShooter.exe`,
`--config Assets/StreamingAssets/demo_config.json`, `--port 50000`, `--player1` / `--player2`
(any of: `aim-at-player2`, `aim-sweep`, `constant`, `explorer`, `idle`, `perimeter`, `random`,
`scripted-cycle`, `spray`), `--max-steps 600`, `--seed 0`. The demo exits `2` if the build exe
or config is missing.

## 4. Collect a dataset (programmatic)

There is **no collection CLI entry point yet** — data collection is driven through the
[`data.collect`](../src/pop_trainer/data/collect.py) API
([`run_worker`](../src/pop_trainer/data/collect.py) /
[`collect_parallel`](../src/pop_trainer/data/collect.py)) with caller-supplied env + agent
factories, writing `.npz` shards to disk. See the [data component page](components/data.md) for
the shape of the pipeline. (A thin CLI lands with the agent redesign.)

---
[← back to index](README.md)
