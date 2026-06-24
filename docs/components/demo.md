# demo

The runnable **entry point**: `python -m pop_trainer.demo` launches the real Unity build
windowed, opens a TCP socket to it, wraps it in a [`Connection`](core.md), constructs a
[`TankEnv`](env.md) over it, and runs ONE episode with two visibly-different policies so a human
can watch the difference.

**Boundary:** an application. Imports [`env`](env.md), [`agents`](agents.md), and
[`core`](core.md) only — never `models` / `data` / `rl` / `pretraining`.

## Key entry points

All in [`demo.py`](../../src/pop_trainer/demo.py):

- [`main`](../../src/pop_trainer/demo.py) — the live path (not unit-tested): launches the build
  via `subprocess.Popen` over the shared [`core.launch.build_launch_cmd`](core.md) arg-list
  (windowed — `-batchmode` deliberately absent), connects with
  [`core.launch.connect`](core.md)'s bounded retry/backoff, builds the env, runs the episode,
  prints a trace, and tears down cleanly. (The launch + connect helpers moved to `core.launch` so
  the [data](data.md) collection runner shares them.)
- [`run_demo_episode`](../../src/pop_trainer/demo.py) — the **pure** episode loop (an
  already-built env + `agent1` in, a [`DemoResult`](../../src/pop_trainer/demo.py) out). Drives
  `player1` off `info["state"]`; the env owns and acts `player2`. Unit-testable against a
  fake-backed `TankEnv` — no subprocess, no live socket.
- [`AGENT_SELECTORS`](../../src/pop_trainer/demo.py) / `make_agent` — the `--player1` /
  `--player2` selector surface. Four names (`demo.py:77`): `aggressive-coverage`, `wall-hugger`,
  `opponent-shadower` (the three [`CoverageAgent`](agents.md) presets), and `random` (the
  map-agnostic baseline). Default pairing: `player1 = aggressive-coverage` (sweeps the arena,
  aim-sweeps + fires) vs `player2 = opponent-shadower` (the same coverage movement but the aim
  tracks player1) — two visibly different policies (`demo.py:84`).

## Pixels must be on

`TankEnv.reset` / `step` always read a length-prefixed pixel frame after each state JSON, but the
shipped `Assets/StreamingAssets/config.json` has no `obs_pixels` key (the build defaults it off
and sends no frame). So the demo launches the build with
[`Assets/StreamingAssets/demo_config.json`](../../Assets/StreamingAssets/demo_config.json), which
sets `"obs_pixels": true` at 640×360, and constructs the env with `frame_shape=(360, 640, 3)` to
match. That config lives inside `StreamingAssets` so its relative `arena_path`
(`Arenas/custom1.json`) resolves against the config directory.

## Pulls from (upstream)

- [core](core.md) — `protocol.Connection` / `WallLayout`, `config.EnvConfig`, `agent.Agent`, and
  `launch` (`build_launch_cmd` / `connect`).
- [env](env.md) — `TankEnv`.
- [agents](agents.md) — the selectable policies for both players.

## Pushes to (downstream)

Nothing — it's a leaf application. Output is a launched game window + a printed episode trace.

## Where it sits in the run

The "does it all work together?" smoke test. It is the one place in the documented slice that
launches a **live** Unity build and runs a full episode over a real socket. See the
[runbook](../runbook.md) for how to run it.

---
[← back to index](../README.md)
