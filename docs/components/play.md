# play

The runnable **entry point**: `python -m pop_trainer.play` launches the real Unity build windowed,
opens a TCP socket to it, wraps it in a [`Connection`](core.md), constructs a [`TankEnv`](env.md)
over it, and runs ONE episode with the two players named on `--player1` / `--player2` so a human can
watch — and play.

**Three player forms** on each slot, any matchup (rl-vs-human, rl-vs-rulebased, rl-vs-rl,
human-vs-rulebased, …):

- `human` — keyboard-driven live play (a shared listener; two humans share one keyboard).
- a rule-based selector — one of `aggressive-coverage`, `wall-hugger`, `opponent-shadower`,
  `random`, `noop` (the [`agents`](agents.md) registry).
- `rl:<checkpoint.zip>` — a trained Stable-Baselines3 PPO model loaded from that path (a bare
  `*.zip` path also works).

```bash
# default: two rule-based bots
uv run python -m pop_trainer.play

# play a human against a trained agent
uv run python -m pop_trainer.play --player1 human --player2 rl:runs/train/model.zip
```

**Boundary (an intentional, CTO-approved relaxation):** `play` is a **composition-root app**, like
[`rl/train.py`](rl.md). At module top it imports [`env`](env.md), [`agents`](agents.md), and
[`core`](core.md) **only**. It is a **leaf** entry point — imported by nothing. To load a trained PPO
checkpoint it reaches `stable_baselines3` **directly** (it does NOT import [`rl`](rl.md) at all);
that **direct sb3 dependency** is the deliberate relaxation of the former `demo`'s "torch-free" rule.
Still FORBIDDEN: `rl`, `models`, `data`, `pretraining`, anything from `tank_twin`. Crucially,
**`stable_baselines3` / `torch` are imported LAZILY** — only inside the RL-player factory
`make_rl_player` (`play.py:265`) — so human / rule-based play stays torch-free and fast. A unit test
asserts none of `stable_baselines3` / `torch` / `pop_trainer.rl` appears at module level.

## Key entry points

All in [`play.py`](../../src/pop_trainer/play.py):

- [`main`](../../src/pop_trainer/play.py) — the live path (not unit-tested): classifies each player
  spec, builds the `(player1, player2)` adapters (loading any `rl:` checkpoint **before** spawning
  the build, so a missing `*.zip` errors early), launches the build via `subprocess.Popen` over the
  shared [`core.launch.build_launch_cmd`](core.md) arg-list (windowed — `-batchmode` deliberately
  absent), connects with [`core.launch.connect`](core.md)'s bounded retry/backoff, builds the env,
  runs the episode, prints a trace, and tears down cleanly.
- [`run_play_episode`](../../src/pop_trainer/play.py) — the **pure** episode loop (an already-built
  env + the two player adapters in, a [`PlayResult`](../../src/pop_trainer/play.py) out). The env is
  a **pure transport that owns neither player**, so this loop drives **both**: each step it holds
  BOTH the current pixel **frame** and the current 52-float **state**, computes player1's and
  player2's actions from the SAME current `(frame, state)` (a simultaneous move), then feeds both to
  `env.step(a1, a2)`. Unit-testable against a fake-backed `TankEnv` — no subprocess, no live socket,
  no real SB3.
- **Per-player adapters** keyed by `obs_kind`:
  - `StateAdapter` (`obs_kind = "state"`) — drives a state-reading agent (rule-based OR human) on
    the 52-float state. player1 acts on the unflipped `info["state"]`; player2 acts on
    `split_state_for_opponent(state)` (its first-person view). A `HumanAgent` ignores the passed
    view (it polls its shared `KeyboardState`) but is handed one anyway — the same path.
  - `PixelsAdapter` (`obs_kind = "pixels"`) — drives a trained RL model on the rendered pixel
    **frame**. player1 acts on the frame as-is; player2 acts on
    [`TankEnv.player2_frame()`](env.md) (the R/B-swapped first-person frame). The action is
    `predict(frame, deterministic=True)[0]` — SB3's CnnPolicy does its own HWC→NCHW + normalize, so
    the RAW `(H, W, 3)` uint8 frame is fed unchanged (mirrors `rl.evaluate.evaluate_winrate`).
- [`make_rl_player`](../../src/pop_trainer/play.py) — the **RL-player factory** and the SOLE place
  `stable_baselines3` (`PPO`) is imported (lazily, inside the function). Validates the checkpoint
  path exists (actionable `FileNotFoundError` otherwise), loads it predict-only with `PPO.load`
  (no env needed for `.predict`), and returns a `PixelsAdapter`. An optional `model=` argument is
  the test seam: a fake object with a `.predict` can be injected with no filesystem / no SB3.
- [`parse_player_spec`](../../src/pop_trainer/play.py) — the `--player1` / `--player2` validator
  (argparse `type=`), replacing the former closed `choices=` set (which could not express
  `rl:<path>`). Classifies a string into a `PlayerSpec`: `human`, an `AGENT_SELECTORS` key, or
  `rl:<path>` / a bare `*.zip`. Rejects anything else with a message naming the valid forms.
- [`build_players`](../../src/pop_trainer/play.py) — wires the `(player1, player2)` adapter pair
  from the two specs and shares the ONE `KeyboardState` across any human players: a `human` becomes
  a [`HumanAgent`](agents.md#humanagent--keyboard-driven-live-play) in a `StateAdapter`, a selector
  becomes `make_agent(...)` in a `StateAdapter`, and an `rl:` spec becomes `make_rl_player(...)`.
- `AGENT_SELECTORS` / `make_agent` — re-exported from [`agents`](agents.md): the five rule-based
  selectors. Default pairing: `player1 = aggressive-coverage` (sweeps the arena, aim-sweeps + fires)
  vs `player2 = opponent-shadower` (the same coverage movement but the aim tracks player1) — two
  visibly different policies.

## Pixels must be on

`TankEnv.reset` / `step` always read a length-prefixed pixel frame after each state JSON, but the
shipped `unity/Assets/StreamingAssets/config.json` has no `obs_pixels` key (the build defaults it off
and sends no frame). So play launches the build with
[`unity/Assets/StreamingAssets/demo_config.json`](../../unity/Assets/StreamingAssets/demo_config.json), which
sets `"obs_pixels": true` plus `obs_pixels_width`/`obs_pixels_height` (640×360 today). The env's
`frame_shape` is then **DERIVED from that SAME launched config** via
[`core.obs.frame_shape_from_config`](core.md#the-observation-resolution-contract-coreobs)
(`play.py:489-491,517`), so the env always matches whatever resolution the build renders — no
hand-synced hardcoded shape. That config lives inside `StreamingAssets` so its relative `arena_path`
(`Arenas/custom1.json`) resolves against the config directory. An `rl:` player needs pixels too —
it acts on the frame — so it runs on the same obs_pixels config.

## Human play (`--player1 human` / `--player2 human`)

When either player is `human`, [`main`](../../src/pop_trainer/play.py) builds ONE shared
[`KeyboardListener`](agents.md#humanagent--keyboard-driven-live-play) + `KeyboardState`, hands the
SAME state to both agents (via `build_players`), starts the listener before the episode and stops it
in `finally`. Two humans thus share one physical keyboard; one human + one bot (rule-based OR `rl:`)
is also valid.

When a player is human the default config switches from `demo_config.json` to
[`human_config.json`](../../unity/Assets/StreamingAssets/human_config.json) — real-time `timeScale: 1`
(vs the bots config's `2`), `ai_actionFreq: 3`, `player_maxHealth: 4` — UNLESS `--config` is given.
An `rl:` player does **not** change the default cadence (it runs at the bots config unless a human is
also present or `--config` is passed). `pynput` is the optional `human` extra, imported lazily by the
listener: a missing extra fails in `main` at listener construction (BEFORE the build launches) with
an actionable message naming the extra.

## Pulls from (upstream)

- [core](core.md) — `protocol.Connection` / `WallLayout`, `config.EnvConfig`, `agent.Agent`,
  `state.split_state_for_opponent` / `flip_frame_perspective` (the perspective flips),
  `obs.frame_shape_from_config` (derive the env `frame_shape` from the launched config), and `launch`
  (`build_launch_cmd` / `connect`).
- [env](env.md) — `TankEnv` and its `player2_frame()` first-person view.
- [agents](agents.md) — the rule-based selectors for both players, plus
  `HumanAgent` / `KeyboardListener` / `KeyboardState` / `player{1,2}_mapping` for the `human` form
  (lazy `pynput`, the `human` extra).
- `stable_baselines3` (`PPO`) — **directly and lazily, inside `make_rl_player` only**
  (`play.py:265`): to load an `rl:` checkpoint predict-only. NOT imported at module level, and
  `pop_trainer.rl` is never imported at all.

## Pushes to (downstream)

Nothing — it's a leaf application. Output is a launched game window + a printed episode trace.

## Where it sits in the run

The "does it all work together?" smoke test, and the way to **watch a trained agent play**. It is
the one place in the documented slice that launches a **live** Unity build and runs a full episode
over a real socket with any pairing of player forms. See the [runbook](../runbook.md) for how to run
it.

---
[← back to index](../README.md)
