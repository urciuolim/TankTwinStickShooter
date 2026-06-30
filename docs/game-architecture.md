# Game architecture (as-built)

How the Unity build actually runs, traced from the C# source. Load-bearing for M0/M1 decisions.

## The game is a Python-clocked simulator — no standalone human mode

The build boots the **Driver** scene (scene 0); `DriverController` reads the resolved `config.json`,
opens a `TcpListener`, and **blocks on `AcceptTcpClient()`** (`DriverController.cs:286`) until a Python
client connects (`running` flips true after the accept). The socket is then the **clock**: `FixedUpdate`
holds `Time.timeScale = 0` between exchanges (`DriverController.cs:227,231`), so the sim only advances
when Python sends a step. Python is both the client and the launcher — the trainer ([env](components/env.md)
over [`core.protocol`](components/core.md#coreprotocol--the-strict-tcp-json-wire-client--pixel-channel)
+ [`core.launch`](components/core.md#corelaunch--the-shared-live-launch-seam)) `Popen`s the build.

**No config flag, CLI arg, or code branch bypasses the accept** — every mode (human-vs-human,
human-vs-AI, AI-vs-AI) needs Python connected, and a standalone double-click hangs at the listener. So
"two humans in a standalone build" is NOT achievable as-is. **CTO decision (2026-06-17):** keep Python
in the loop — a lightweight local-play host clocks the sim and stubs the human slots while humans
control locally in Unity; do NOT build a no-Python native mode. (Forward-looking: reuse this connection
to record human play for imitation learning — recording the human's *actual* action would need Unity to
expose it back over the protocol, a sign-off-gated RL-seam addition, not an M0 deliverable.)

## Where the players come from

- **Human input is read LOCALLY by Unity**, not piped through Python: when a tank's `AI` flag is false,
  `PlayerController.GetInput`'s human branch reads the New Input System (`PlayerController.cs:238,260+`).
  A local-play host sends a zero placeholder for the human slot, which Unity discards. Full wiring +
  config presets: [input-controls](input-controls.md).
- **AI opponents are Python, not Unity.** C# "AI" tanks have zero decision logic — they copy
  `DriverController.instance.actions[playerID]` off the socket (`PlayerController.cs:247-258`). The
  deciding policy is Python: a scripted [agents](components/agents.md) selector or a trained SB3 PPO
  policy ([rl](components/rl.md)).

> **Human-play tooling is currently un-wired.** The retired 2021 root-package local-play scripts
> went away with the old root package; a `pop_trainer` local-play entrypoint that supersedes
> them is not yet built. See [input-controls → two-human play-test](input-controls.md#two-human-play-test-owed-to-a-human--verification-lead).

## The Driver↔Arena coupling (and the NREs)

Flow: Driver boots → connects → `LoadScene("Arena")` → one match → round-over, waiting → next round.
**Arena depends on the Driver singleton.** Opening `Arena.unity` directly throws `NullReferenceException`
— `GameController.cs:60` and `PlayerController.cs:81` deref `DriverController.instance.config` with
`instance` null. This is **pre-existing scene coupling** (introduced in "First commit"), confirmed NOT a
regression from the Unity 6.5 upgrade or the startup-hygiene pre-fixes.

## The Python-driven round boundary + length-prefix framing

Two transport changes hardened the episode boundary. The **wire data contract is unchanged** — the
52-float state, the integer-keyed `{1: a1, 2: a2}` step, the pixel-frame header — only the *control
framing* and *who decides the round ends* changed. Unity side:
[`DriverProtocol.cs`](../unity/Assets/Scripts/DriverProtocol.cs),
[`DriverController.cs`](../unity/Assets/Scripts/DriverController.cs),
[`GameController.cs`](../unity/Assets/Scripts/GameController.cs); Python side
[`core.protocol`](components/core.md) + [`env`](components/env.md).

**The round boundary runs off Unity's OWN clock.** Previously a Python step cap drove restarts that
could fire mid-round and collapse into a synchronous scene reload inside the socket read. Now Unity ends
the round on its own clock (a winner OR the round-timer) and `GameController.EndGame` stamps
`done`/`winner` then sets `roundOver = true` (`GameController.cs:321-347`); `roundOver` **FREEZES** the
round — `FixedUpdate` early-returns so the stamped state is delivered to Python **exactly once**
(`GameController.cs:182-193`). `EndGame` **no longer `LoadScene`s** (`GameController.cs:339-348`); the
reload is **deferred** to the next `start`, and Unity goes `ingame = false` but **keeps servicing the
socket** while waiting (the `start` handler is the single reload point, `DriverController.cs:360-384,577`).
So Python's `done → terminated → reset()` sends `{"restart": True}` into a **clean waiting state, never
mid-round**. `max_steps` is no longer the boundary — just a safety cap (`DEFAULT_MAX_STEPS = 600`,
`tank_env.py:170`) above Unity's ~300-step round; if it ever fires, `step` truncates and sends no
restart, and the driver survives a restart on a safety-cap mid-round read without the old collapse
(`DriverController.cs:541-566`).

**Length-prefix framing (Python → Unity control/step only).** Unity's old single `nwStream.Read`
assumed one JSON object per `recv` — a dormant desync. Now every Python → Unity control/step message is
prefixed with a 4-byte BIG-ENDIAN uint32 = the UTF-8 JSON byte length, then exactly that many JSON
bytes — a **read-exactly frame** (`DriverController.ReadExactly`, `DriverController.cs:316-328`). The
pure, unit-tested helpers live in [`DriverProtocol.cs`](../unity/Assets/Scripts/DriverProtocol.cs):
`DecodeLengthPrefix` / `EncodeLengthPrefix` (symmetric big-endian, pinned by an EditMode test to stay in
lockstep with the Python `encode_framed`, `DriverProtocol.cs:40-59`) and `Classify` (maps an inbound
`JObject` to `Start`/`End`/`Restart`/`SwitchArena`/`Action`/`Unknown` so the waiting-state and
safety-cap reads share ONE tested dispatch, `DriverProtocol.cs:64-77`). The prefix counts ONLY the JSON
payload; **inbound (Unity → Python) JSON stays UNFRAMED** (Python's `receive()` brace-scans one object)
and the **pixel-frame header (10 bytes) is unchanged** — only the Python→Unity direction gained the
prefix.

**FrameCapture full-view supersampling (wire-identical).**
[`FrameCapture`](../unity/Assets/Scripts/FrameCapture.cs) renders the full 16:9 view into a 1280×720
intermediate (no crop — the WHOLE arena is in every frame, `FrameCapture.cs:62-63`), then area-averages
down to the obs dims via a progressive 2× bilinear-halving chain (each step a 2×2 box average, so thin
~1px sprites like the tank wireframe + barrel survive, `FrameCapture.cs:51-62,203-205`). Obs-quality
only: the frame-message wire contract (`BuildFrameMessage`, 10-byte big-endian header + RGB24 bottom-up
payload, `FrameCapture.cs:14-32`) is **byte-identical**.

## Observability logging (verbose-gated, ZERO wire impact)

The C# side has structured observability logging — added to diagnose the reset/restart-handshake hang —
gated **entirely** on the existing `verbose` config flag (`DriverController.cs:51`). When `verbose` is
false (the default) **none** of it runs; it never touches the socket, the 52-float state, message
ordering, or any control flow, and the lines go to `Debug.Log`, not the wire.

- **Pure helpers (EditMode-tested).** [`DriverLog`](../unity/Assets/Scripts/DriverLog.cs) formats one
  grep-able line (`[tag] wall=<ISO-8601 UtcNow 'o'> k=v …`); [`DeadZoneTracker`](../unity/Assets/Scripts/DeadZoneTracker.cs)
  is a pure state machine detecting ENTER/EXIT of the FixedUpdate **dead-zone** (`ingame && state == null`,
  the no-I/O window that is the prime hang suspect). Both read no clock / Unity state.
- **Verbose-gated `Debug.Log`** emits handshake events (`start_received` / `restart_received` /
  `ingame_flip` / `switch_arena_received` / `arena_switched`, `DriverController.cs:360-441,541-566`), the
  dead-zone + state null↔populated transitions (`DriverController.cs:239-254`), and per-step `ReadPixels`
  timing (`DriverController.cs:503-506`). Durations use `Stopwatch` / `DateTime.UtcNow`, **never**
  `Time.time` (frozen by `Time.timeScale = 0`, would report zero — `DriverController.cs:64-66`).
- **Paired to the Python logs by wall-clock.** `wall=<UtcNow>` correlates to the Python JSONL `ts_wall`.
  When a build's config sets `verbose: true`, its C# log routes to its own `unity-<role>-<port>.log` via
  `core.launch`'s `-logFile` (the [rl](components/rl.md#the-train_local-integrator) integrator does this).
  Operator guide: [runbook → Observability logs](runbook.md#observability-logs).

## Config / arena source-of-truth map

ONE canonical location for the shipped config + arenas; the rest are explicit fallbacks (a stale 2021
duplicate tree once shadowed this and was removed — see git history).
`DriverController.ResolveConfigPath` (`DriverController.cs:179-192`) precedence, highest first:

1. **`--config <path>` / `-config <path>` CLI arg** — explicit override (a local-play host passes this
   for a human preset; otherwise it omits the arg and the build falls back).
2. **`unity/Assets/StreamingAssets/config.json` — CANONICAL DEFAULT (= the M1 AI-vs-AI training
   topology).** Ships inside every build, so a player resolves here when **no `--config` is passed**. The
   trainer's [env](components/env.md) launches with *only* `[game_path, port]` (no `--config`, by design
   — the frozen RL seam), so this default MUST be AI-vs-AI (`player1_ai`+`player2_ai` true; `timeScale`
   5). The human presets `config_2p.json` / `config_2p_keyboard.json` live here too (selected via `--config`).
3. **`unity/Assets/config.json` — legacy last-resort fallback.** In-editor / when no StreamingAssets copy
   exists. KEPT intentionally; do not delete.

Arena resolution (`ResolveArenaPath`, `DriverController.cs:197-207`): `arena_path` is used as-is if
absolute or if it exists relative to the working dir (legacy `Assets/Arenas/...` form), else resolved
relative to the resolved config file's directory — so a config + its `Arenas/` travel together. Canonical
arenas live in `unity/Assets/StreamingAssets/Arenas/`; the format is on the
[arena-format reference](experiments.md). Edit configs/arenas there; `unity/Assets/config.json` is a
deliberate fallback, not a duplicate.

---
[← back to index](README.md)
