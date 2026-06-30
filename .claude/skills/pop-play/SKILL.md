---
name: pop-play
description: Contract + boundaries for pop_trainer.play — the unified human / rule-based / trained-RL play entry point. Load when building or reviewing play.py.
---

# Component: `pop_trainer.play` — the unified play entry point

**Responsibility:** ONE root-level play script that launches the live Unity build windowed and runs
ONE episode with any combination of player forms on `--player1` / `--player2`:
- `human` — keyboard-driven (shared listener; existing path, unchanged).
- a rule-based selector from `AGENT_SELECTORS` (`aggressive-coverage`, `wall-hugger`,
  `opponent-shadower`, `random`, `noop`) — unchanged.
- `rl:<checkpoint.zip>` — a trained SB3 PPO model loaded from that path.

Any matchup must work: rl-vs-human, rl-vs-rulebased, rl-vs-rl, human-vs-rulebased, etc.

This is the rename + generalization of the former `pop_trainer.demo` (run via
`python -m pop_trainer.play`). It preserves EVERYTHING demo did: windowed launch + reap, config
defaults (`demo_config.json` for bots, `human_config.json` when a player is human, `--config`
override), `--exe` / `--port` / `--max-steps` / `--seed`, the single-episode loop + printed trace.

**Boundary (INTENTIONAL RELAXATION — CTO-approved):** `play` is a COMPOSITION-ROOT APP, like
`rl/train.py`. It MAY import `env` + `agents` + `rl` + `core`. It is a LEAF entry point — imported
by NOTHING — so the `rl` import introduces NO cycle. This is a deliberate relaxation of demo's old
"never import `rl`" rule; the gate's repo-steward must NOT treat `play -> rl` as a violation (the
module docstring must state this so the steward sees the rationale). Still FORBIDDEN: `models`
(reach it only transitively through `rl`), `data`, `pretraining`, anything from `tank_twin`.

**Per-player obs + perspective (the load-bearing generalization):** the episode loop must give each
player the RIGHT view, by `obs_kind`:
- `state` (rule-based) → the 52-float state. player1 UNFLIPPED (`info["state"]`); player2 FLIPPED
  via `core.state.split_state_for_opponent`.
- `keyboard` (human) → the `HumanAgent` reads its shared `KeyboardState`; `act(obs)` ignores the
  passed view (it polls the listener). Treat as the `state` path for view selection (it is handed a
  view but does not use it) — do NOT special-case beyond building the agent.
- `pixels` (RL) → the pixel FRAME (the env's observation). player1 gets the frame as-is; player2
  gets the FLIPPED frame via `TankEnv.player2_frame()` (R/B swap, `core.state.flip_frame_perspective`).
  The action is `model.predict(frame, deterministic=True)[0]`. MIRROR how `rl.evaluate.evaluate_winrate`
  feeds the model: it calls `model.predict(obs, deterministic=True)` on the env's raw observation —
  SB3's CnnPolicy does its own HWC→NCHW + normalize, so do NOT transpose / normalize the frame here.

Generalize the old state-only `run_demo_episode` into a small per-player abstraction
(`obs_kind ∈ {state, pixels, keyboard}` → action). The loop must hold BOTH the current pixel frame
AND the current 52-float state each step (the old loop did `del obs` and drove off state only — the
new loop needs the frame too, for RL players). Keep it PURE: an already-built env + two player
adapters in, a result dataclass out — unit-testable against a fake transport, no subprocess, no
live socket, no real SB3.

**Lazy `rl` import (HARD requirement):** import `rl` / `stable_baselines3` / `torch` ONLY inside the
RL-player factory (invoked only when a `rl:` player is requested). human / rule-based play must stay
torch-free and fast. Eager (module-top) imports stay `env` + `agents` + `core` ONLY. A unit test
must assert the module's top-level imports contain none of `stable_baselines3` / `torch` / `rl`.

**`--player1` / `--player2` validation:** the old `choices=` (a closed set) cannot express
`rl:<path>`. Replace with a custom validator / `type=` that accepts: `human`, any `AGENT_SELECTORS`
key, or `rl:<path>` (and OPTIONALLY a bare `.zip` path if clean). Reject anything else with an
actionable error. Update the `--player1` / `--player2` help text + the `prog`/description to name
the three forms and give an `rl:` example.

**Config defaults unchanged:** `human` in either slot → `human_config.json` default; else
`demo_config.json`; `--config` always overrides. RL players do NOT change the config default (they
run at the bots cadence unless a human is also present or `--config` is given).

**Tests (pure, no live SB3 / Unity):**
- the pure episode loop against the fake transport (as demo was) — map forwarding, p2 flipped
  STATE view, winner/loss/draw/truncation, player1 actions on the wire.
- player-type DISPATCH: `human` → `HumanAgent`; a selector → `make_agent`; `rl:<path>` → the RL
  adapter. Bad inputs rejected.
- an RL player loads + acts with a STUBBED model — inject a FAKE `predict` (no real SB3 / no real
  `.zip`). Cover p1-RL (frame as-is) AND p2-RL (FLIPPED frame via `player2_frame`): assert the
  frame the fake `predict` saw is the right perspective.
- the lazy-import guard (top-level imports are torch-free).

**Do NOT touch the RL seam:** `DriverController`'s socket/`actions` path and the 52-float wire
layout are frozen. This is a pure rename + refactor + feature over the existing env/agents/rl/core
surface — no protocol or env-step wire change.

**Inspiration (do NOT copy):** none from legacy — `play` is the rename of the CURRENT `demo.py`
(git-mv it). Reuse the LIVE in-repo primitives: `rl.evaluate.evaluate_winrate` /
`rl.selfplay.SelfPlayWrapper` for the `model.predict` + perspective pattern; `TankEnv.player2_frame`
/ `player2_state` for the flips; `agents.make_agent` / `HumanAgent` for the non-RL players.
