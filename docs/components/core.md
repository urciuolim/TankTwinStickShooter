# core

The dependency-free **contract layer** — the root of the `pop_trainer` dependency graph. It
imports nothing internal (no `models` / `env` / `data` / `agents`) and uses **stdlib + numpy
only** (no torch, no gymnasium, no sb3). Every other component reads its shared definitions
from here instead of re-hardcoding them. All JSON is strict.

## Key modules / entry points

- [`core.state`](../../src/pop_trainer/core/state.py) — the **52-float game-state schema**, the
  single source of truth for the wire state Unity emits (`GameController.UpdateState`, the
  frozen RL seam). Layout constants (`STATE_LEN = 52`, P1 = indices 0..25, P2 = 26..51),
  pure accessors (`position` / `velocity` / `aim` / `iter_bullets`), and the self-play
  **perspective transforms** `flip_frame_perspective` (R/B channel swap on a pixel frame) and
  `split_state_for_opponent` (swap the two 26-float halves). These are consumed **driver-side**:
  the [data](data.md) collection / [demo](demo.md) loops call `split_state_for_opponent` to build
  player2's flipped first-person view before passing its action to `env.step` (the env is a pure
  transport and does not flip perspective itself). The state is the supervised-decode *objective*,
  NOT the policy observation — the observation is the pixel frame.
- [`core.protocol`](../../src/pop_trainer/core/protocol.py) — the **strict TCP-JSON wire
  client** over an injected transport. `encode` / `decode` (pure, no socket), a frame-aware
  [`Connection`](../../src/pop_trainer/core/protocol.py) that reads exactly one top-level JSON
  object via a brace-depth scan and buffers trailing bytes, the length-prefixed binary
  **pixel-frame channel** (`parse_frame_header` / `Connection.receive_frame` →
  `(H, W, 3)` uint8), the **wall-layout message** seam (`is_walls_message` /
  `parse_walls_message` → an immutable [`WallLayout`](../../src/pop_trainer/core/protocol.py)),
  and the **switch-arena handshake** (`Connection.switch_arena` — the one additive outbound
  write; see below).
- [`core.config`](../../src/pop_trainer/core/config.py) — frozen, strict-JSON config
  dataclasses [`RunConfig`](../../src/pop_trainer/core/config.py) /
  [`EnvConfig`](../../src/pop_trainer/core/config.py) /
  [`RewardConfig`](../../src/pop_trainer/core/config.py) (reject unknown keys).
- [`core.maps`](../../src/pop_trainer/core/maps.py) — the **map-rotation resolution contract**:
  `resolve_map_rotation` turns a `--maps` flag value into a list of map-config paths.
- [`core.launch`](../../src/pop_trainer/core/launch.py) — the **shared live-launch seam**
  (stdlib only — `subprocess` / `socket` / `time` / `pathlib`, no internal imports, so `core`
  stays the leaf). [`build_launch_cmd`](../../src/pop_trainer/core/launch.py) is the **pure**
  `Popen` arg-list builder that launches the build **windowed** (`-screen-fullscreen 0` at
  1280×720; `-batchmode` deliberately absent), with the port positional + `--config` JSON;
  [`connect`](../../src/pop_trainer/core/launch.py) opens a TCP socket to `127.0.0.1:port` with a
  bounded retry/backoff (the build needs a moment to boot its listener). Shared by both the
  [demo](demo.md) (one local watch) and the [data](data.md) collection runner (one build per
  worker).
- [`core.agent`](../../src/pop_trainer/core/agent.py) — the **Agent Protocol** (`act(obs) →
  action`), type-only. [`Agent`](../../src/pop_trainer/core/agent.py) is `@runtime_checkable`
  (declares only `act`); [`StatefulAgent`](../../src/pop_trainer/core/agent.py) adds the OPTIONAL
  `reset(*, seed=None)` and the **`set_map(layout)` map hook** — how a map-aware agent receives the
  static `WallLayout` once per episode (`agent.py:80`). Those optional methods are consumed by
  probing with `getattr`/`hasattr` (never `isinstance`): folding either into the runtime-checkable
  surface would wrongly reject a valid `act`-only agent, so they live in the separate static-typing
  Protocol. It lives here so `env` can be
  typed against an agent without importing `agents` or torch. See [agents](agents.md) for who
  implements `set_map`.

## The wall-message / protocol seam (`WallLayout` → `info["map"]`)

Unity is the source of truth for the static map geometry. The end-to-end path:

1. On a map load (start handshake), `DriverController` writes `{"starting": true}` first, then a
   **separate** strict-JSON `{"type": "walls", ...}` message built by
   [`WallMessage.Build`](../../Assets/Scripts/WallMessage.cs) — skipped entirely when no arena is
   configured.
2. `core.protocol.is_walls_message` discriminates it by its `"type"` tag and
   `parse_walls_message` strict-parses it into a frozen `WallLayout` (column-x string keys →
   ints, a derived `occupied` frozenset of `(x, y)` cells).
3. [`env`](env.md) reads it during the handshake, stores it on `TankEnv.current_map`, and
   surfaces it as `info["map"]`. So Python tracks map-state without ever re-parsing the arena
   JSON.

## The switch-arena handshake seam (`Connection.switch_arena`)

The **one additive outbound write** on the otherwise-frozen wire — reset-time only. Between the
restart ack and the `{"start": True}` send (Unity's `!ingame` window) the caller MAY request a
map change, so a long-lived build can rotate arenas without relaunching. The per-step `{1: a1,
2: a2}` integer-keyed wire and the 52-float state are **unchanged**; a reset that does NOT switch
is **byte-identical** to today.

- [`Connection.switch_arena(arena_path)`](../../src/pop_trainer/core/protocol.py) sends
  `{"switch_arena": <arena_path>}` (the request key is the constant `SWITCH_ARENA_KEY`; the value
  is the arena-path string), then reads **exactly one** JSON object and validates the
  `{"arena_switched": true}` ack (key constant `ARENA_SWITCHED_KEY`). It raises **`ValueError`**
  if the ack is not a dict, is missing the `arena_switched` key, or that value is falsy
  (`protocol.py:261-268`). Returns the decoded ack dict.
- **It reads ONLY the ack.** Unity writes the `arena_switched` ack first, then — *only* when the
  NEW arena carries a "Walls" block — writes a `{"type": "walls", ...}` message as its own discrete
  write (identical ordering to the start branch). `switch_arena` deliberately leaves that OPTIONAL
  trailing walls message **on the wire** for the caller to route with the **same** `receive` +
  `is_walls_message` logic used after the start ack (`protocol.py:246-252`). Reading the ack alone
  is what guarantees a **walls-absent** switch never consumes the following `state`.
- **Window discipline.** Unity handles `switch_arena` only while `!ingame` — i.e. after the
  restart ack and before the `{"start": True}` send. Calling it outside that window desyncs the
  wire. A `socket.timeout` is translated to `ConnectionError` (consistent with the rest of the
  class).

The caller is [`env`](env.md): `TankEnv.reset(options={"switch_arena": <path>})` fires this in the
`!ingame` window and then drains the optional walls message. See the [env](env.md#map-rotation-resetoptionsswitch_arena) page for the reset hook.

## Pulls from (upstream)

Nothing internal. **`core` is the dependency-free root** — stdlib + numpy only.

## Pushes to (downstream)

Every other component depends on `core`:

- [models](models.md) — shares `core`'s leaf position (but imports nothing from it today).
- [env](env.md) — `state` (schema + transforms), `protocol` (`Connection` incl.
  `switch_arena`, `WallLayout`, `is_walls_message` / `parse_walls_message`), `config`,
  `agent.Agent`.
- [agents](agents.md) — `agent.Agent` Protocol + the `state` schema.
- [data](data.md) — `state` (`STATE_LEN` + `validate`), `config.EnvConfig`,
  `protocol.Connection`, `agent.Agent`, `launch` (the collection runner launches + connects one
  build per worker), and `maps.resolve_map_rotation` (the shared `--maps` rotation contract).
- [demo](demo.md) — `protocol.Connection` / `WallLayout`, `config.EnvConfig`, `agent.Agent`, and
  `launch` (`build_launch_cmd` / `connect`).

## Where it sits in the run

The bedrock. Before any episode, `core` defines what a state vector means, how bytes move on
the wire, and what an agent must look like. Everything else is built on these contracts.

---
[← back to index](../README.md)
