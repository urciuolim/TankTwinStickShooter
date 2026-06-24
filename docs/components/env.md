# env

The **simulator interface** — the Gymnasium environment wrapping the Unity sim over the socket,
plus the budget-based shaped reward. This is the trainer's **swap point**: everything downstream
speaks the Gymnasium API, so a different simulator could implement the same interface unchanged.

**Boundary:** imports [`core`](core.md) only (plus `gymnasium` + `numpy`). No torch, no sb3,
nothing from `models` / `data` / `agents` / the Unity-side code.

## Key classes / entry points

- [`TankEnv`](../../src/pop_trainer/env/tank_env.py) — the single-agent, pixel-observation
  `gymnasium.Env` over the Unity socket. `reset()` runs the restart/start/first-state handshake;
  `step(action, opponent_action=None)` returns the Gymnasium 5-tuple
  `(obs, reward, terminated, truncated, info)`.
  - **Observation = the real rendered pixel frame** (`(H, W, 3)` uint8) read via
    `core.protocol.Connection.receive_state_and_frame`. The 52-float wire state is NOT in
    `observation_space` — it rides in `info["state"]` as the supervised-decode objective.
  - `ACTION_DIM = 5`: the action is `[move_x, move_y, aim_x, aim_y, fire]` in `[-1, 1]`.
  - **Injected transport** (`connection` or `connection_factory`), exactly like
    `core.protocol.Connection` — so the env is unit-testable against an in-process fake socket
    with no live Unity. `connection_factory` also enables reconnect after a dropped connection.
- [`shaped_step_reward` / `time_penalty_per_step`](../../src/pop_trainer/env/rewards.py) — the
  **pure** budget-based reward (no socket, no gym, no numpy): a per-step time penalty that
  accrues every step, plus the win/loss terminal *added* on the decided step. Survivor mode
  flips the terminal component.

## Pure symmetric transport (the self-play seam)

The env owns **neither** player — it is a pure symmetric transport. `step(action, opponent_action)`
takes **two** actions, BOTH from the **caller**: it sends the wire message `{1: a1, 2: a2}` where
`a1` is the player1 `action` (sent as `np.asarray(action, np.float32).tolist()`, the unchanged
player1 path) and `a2` is the caller's `opponent_action` coerced to the `[-1, 1]` length-5 wire
shape via `_coerce_action` (`tank_env.py:288-294`). When `opponent_action` is `None` the env sends
a **no-op zero action** `[0.0, 0.0, 0.0, 0.0, 0.0]`. Both wire actions are captured on the env
(`self.last_p1_action` / `self.last_p2_action`) and surfaced in `info["p1_action"]` /
`info["p2_action"]` (`tank_env.py:317-337`).

The **driver** ([data](data.md) collection / [demo](demo.md)) owns each agent and the player2
**perspective flip** — it computes player2's first-person view via
`core.state.split_state_for_opponent` and passes the resulting `a2` into `env.step(a1, a2)`. The
env exposes self-play perspective **helpers** the driver MAY use — `player2_frame()` (R/B channel
swap) and `player2_state()` (`split_state_for_opponent` on the latest raw state,
`tank_env.py:385-402`) — but the env does NOT call them itself during `step`.

> **Seam unchanged.** The wire shape is **byte-identical** to the frozen RL seam — only the
> *source* of `a2` moved from env-internal to the caller. Integer keys `1` / `2`, length-5 `[-1, 1]`
> actions, unchanged player1 path.

## Map tracking (`info["map"]`)

During the handshake the env reads the optional one-time `{"type": "walls", ...}` message
(routed by its `"type"` tag), parses it via `core.protocol.parse_walls_message`, and stores the
[`WallLayout`](core.md) on `self.current_map` — surfaced in both reset and step `info` as
`info["map"]` (`tank_env.py:258,338`). It stays `None` when no arena is configured; the env never
re-parses arena JSON. See the [core](core.md) seam writeup for the full Unity → Python path.

The env surfaces the `WallLayout` in `info["map"]` but does **not** itself notify any agent — both
players are driver-side, so the [data](data.md) collection and [demo](demo.md) loops hand the
layout to a map-aware [agent](agents.md) (e.g. a `CoverageAgent`) via the OPTIONAL `set_map` hook.
The env stays `agents`-free.

## Episode boundaries

- `terminated` — the game decided the round (a winner, or a bare `done`).
- `truncated` — `max_steps` reached on an undecided step, **OR** a lost connection. A dropped
  connection (`core.protocol` raises `ConnectionError`) becomes a `truncated` step with reward
  `0.0` and `info["lost_connection"] = True`, then the env reconnects via the factory.

## Pulls from (upstream)

- [core](core.md) — `state` (schema + `split_state_for_opponent` / `flip_frame_perspective`),
  `protocol` (`Connection`, `WallLayout`, `is_walls_message`, `parse_walls_message`), `config`
  (`EnvConfig` / `RewardConfig`), `agent.Agent`. Plus `gymnasium` + `numpy`.

## Pushes to (downstream)

- [data](data.md) — collection drives `TankEnv` (the same observation pipeline RL trains on).
- [demo](demo.md) — constructs a `TankEnv` over a live socket and runs one episode.
- (Future `rl` consumes `TankEnv` as its training env — not yet built.)

## Where it sits in the run

The hinge. The env IS the boundary between the trainer and the Unity simulator: it owns the
socket handshake, the per-step wire exchange, and the reward — but **neither player**. The driver
([data](data.md) / [demo](demo.md)) supplies both `a1` and `a2`; both `data` collection and the
`demo` run their episodes through it.

```mermaid
graph LR
    a1["caller's player1 action (a1)"] --> step["TankEnv.step(a1, a2)"]
    a2["caller's opponent_action (a2)<br/>(no-op zero when omitted)"] --> step
    step -->|"{1: a1, 2: a2}"| unity["Unity sim"]
    unity -->|"state JSON + pixel frame"| step
    step --> obs["obs = pixel frame"]
    step --> info["info: state, p1/p2_action, map, winner"]
    step --> rew["reward (shaped_step_reward)"]
```

---
[← back to index](../README.md)
