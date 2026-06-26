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
  - **Optional Unity log routing.** `build_launch_cmd` takes an optional keyword
    `unity_log_path` (`launch.py:94-132`): when given it appends Unity's standard
    `-logFile <path>` so that build writes its OWN C# log instead of clobbering the shared default
    `Player.log` — the Unity side of the per-process observability split (pair it with
    `logging_setup.unity_log_path` / the Python `env-<role>-<port>.log` by the shared port). When
    `None` (the default) **no** `-logFile` is added, so the arg-list is **byte-identical** to the
    no-log path and every existing call site / test is preserved.
- [`core.logging_setup`](../../src/pop_trainer/core/logging_setup.py) — the **per-process
  structured-JSONL observability seam** (the cross-stack diagnostic trail). **stdlib `logging`
  only** (no new dependency), and it lives in `core` (the dependency-free root) so every layer
  above — `train` / `env` / `protocol` — can route to it without a boundary violation
  (`logging_setup.py:1-7`).
  - **Per-process file routing.** A training run is many processes, and each writes to its OWN
    file so concurrent writers never contend on one handle. The **pure** path builders (no I/O,
    unit-tested for exact filenames): `system_log_path` → `training-system.log` (the main
    process), `env_log_path` → `env-<role>-<port>.log` (one per env connection, `role` =
    `train`/`eval`, `port` = the env's TCP port), and `unity_log_path` →
    `unity-<role>-<port>.log` (passed to `core.launch`'s `-logFile` so each Unity build writes its
    own C# log; pairs with `env_log_path` by the shared `(role, port)`) (`logging_setup.py:103-124`).
  - **Strict-JSON line schema.** [`JsonlFormatter`](../../src/pop_trainer/core/logging_setup.py)
    emits one strict-JSON object per record — keys `ts_wall` (`time.time`, the cross-file /
    cross-stack merge key), `ts_mono` (`time.monotonic`), `level`, `layer`, `role`, `port`,
    `event`, plus any per-call `detail` fields — with `role`/`port`/`layer` bound at setup so call
    sites pass only `event` + an optional `detail` dict (`logging_setup.py:28-46,139-182`).
  - **Eval isolation (load-bearing).** `setup_logger` / `setup_env_logger` /
    `setup_system_logger` open the per-process file handle and set **`propagate=False`** so a
    logger's records NEVER bubble up to the root/system handler — an eval env's records land ONLY
    in `env-eval-<port>.log` and never leak into `training-system.log`. Setup is **idempotent**
    (the handler it owns is tagged) so a re-entered factory never double-writes
    (`logging_setup.py:20-26,188-264`).
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
  `agent.Agent`, and `logging_setup.LAYER_ENV` (the optional observability logger's layer tag).
- [agents](agents.md) — `agent.Agent` Protocol + the `state` schema.
- [rl](rl.md) — `launch` (incl. the `build_launch_cmd` `unity_log_path` kwarg),
  `protocol.Connection`, `config`, and `logging_setup` (the integrator wires the per-process
  `system` / `env` loggers + the `--debug` / `POP_LOG_LEVEL` level switch via `level_from_env`).
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
