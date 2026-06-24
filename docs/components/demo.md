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
  already-built env + `agent1` + `agent2` in, a [`DemoResult`](../../src/pop_trainer/demo.py) out).
  The env is a **pure transport that owns neither player**, so this loop drives **both**: each step
  it computes `a1 = agent1.act(state_vec)` (player1's own unflipped view) and
  `a2 = agent2.act(split_state_for_opponent(state_vec))` (player2's flipped first-person view,
  computed here), then feeds both to `env.step(a1, a2)` (`demo.py:167-169`). Unit-testable against a
  fake-backed `TankEnv` — no subprocess, no live socket.
- [`AGENT_SELECTORS`](../../src/pop_trainer/demo.py) / `make_agent` — the seed-factory selector
  surface. Four names (`demo.py:87`): `aggressive-coverage`, `wall-hugger`, `opponent-shadower`
  (the three [`CoverageAgent`](agents.md) presets), and `random` (the map-agnostic baseline).
  Default pairing: `player1 = aggressive-coverage` (sweeps the arena, aim-sweeps + fires) vs
  `player2 = opponent-shadower` (the same coverage movement but the aim tracks player1) — two
  visibly different policies (`demo.py:94-95`).
- **`--player1 human` / `--player2 human`** — keyboard-driven live play. `"human"` is in
  `PLAYER_CHOICES` (the `--player1` / `--player2` choices) but is NOT an `AGENT_SELECTORS` entry:
  it needs a SHARED listener + state a per-agent seed factory cannot express, so it is a SPECIAL
  PATH in `main`, not a seed-factory (`demo.py:97-101`). `AGENT_SELECTORS` membership is unchanged
  (still the four above).
- [`build_human_agents`](../../src/pop_trainer/demo.py) — wires the `(agent1, agent2)` pair: any
  `"human"` player becomes a [`HumanAgent`](agents.md#humanagent--keyboard-driven-live-play) over
  the player's keymap and the ONE shared [`KeyboardState`](agents.md), a non-human player is built
  via `make_agent`. Pure (takes an already-built `state`, never constructs a listener / touches
  `pynput`) and unit-tested (`demo.py:117-142`).

## Pixels must be on

`TankEnv.reset` / `step` always read a length-prefixed pixel frame after each state JSON, but the
shipped `Assets/StreamingAssets/config.json` has no `obs_pixels` key (the build defaults it off
and sends no frame). So the demo launches the build with
[`Assets/StreamingAssets/demo_config.json`](../../Assets/StreamingAssets/demo_config.json), which
sets `"obs_pixels": true` at 640×360, and constructs the env with `frame_shape=(360, 640, 3)` to
match. That config lives inside `StreamingAssets` so its relative `arena_path`
(`Arenas/custom1.json`) resolves against the config directory.

## Human play (`--player1 human` / `--player2 human`)

When either player is `human`, [`main`](../../src/pop_trainer/demo.py) builds ONE shared
[`KeyboardListener`](agents.md#humanagent--keyboard-driven-live-play) + `KeyboardState`, hands the
SAME state to both agents (via `build_human_agents`), starts the listener before the episode and
stops it in `finally` (`demo.py:300-348`). Two humans thus share one physical keyboard; one human
+ one bot is also valid (the bot player is built normally).

When a player is human the default config switches from `demo_config.json` to
[`human_config.json`](../../Assets/StreamingAssets/human_config.json) — real-time `timeScale: 1`
(vs the bots demo's `2`), `ai_actionFreq: 3`, `player_maxHealth: 4` — UNLESS `--config` is given
(`demo.py:302-308`). `pynput` is the optional `human` extra, imported lazily by the listener: a
missing extra fails in `main` at listener construction (BEFORE the build launches) with an
actionable message naming the extra.

## Pulls from (upstream)

- [core](core.md) — `protocol.Connection` / `WallLayout`, `config.EnvConfig`, `agent.Agent`, and
  `launch` (`build_launch_cmd` / `connect`).
- [env](env.md) — `TankEnv`.
- [agents](agents.md) — the selectable policies for both players, plus
  `HumanAgent` / `KeyboardListener` / `KeyboardState` / `player{1,2}_mapping` for the `human`
  selector (lazy `pynput`, the `human` extra).

## Pushes to (downstream)

Nothing — it's a leaf application. Output is a launched game window + a printed episode trace.

## Where it sits in the run

The "does it all work together?" smoke test. It is the one place in the documented slice that
launches a **live** Unity build and runs a full episode over a real socket. See the
[runbook](../runbook.md) for how to run it.

---
[← back to index](../README.md)
