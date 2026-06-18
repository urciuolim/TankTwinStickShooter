# Game architecture (as-built)

How the existing game actually runs, traced from code (game-sim-engineer + training-engineer, 2026-06-17). Load-bearing for M0/M1 decisions.

## The game is a Python-clocked simulator — there is no standalone human mode
- The build boots the **Driver** scene (scene 0). `DriverController` (a `DontDestroyOnLoad` singleton, present only in Driver) reads the resolved `config.json` (see source-of-truth map below), opens a `TcpListener`, and **blocks on `AcceptTcpClient()`** (`DriverController.cs:169`) until a Python client connects. `running` flips true only after the accept returns.
- The socket is the **clock**: `FixedUpdate` holds `Time.timeScale = 0` between socket exchanges (`DriverController.cs:136,158`); the sim only advances on each Python send/receive round-trip. Python is the **client** (`tank_env.py:231`) and **launches** the Unity exe (`subprocess.Popen`, `tank_env.py:184`).
- **No config flag, CLI arg, or code branch bypasses the accept.** Every mode (human-vs-human, human-vs-AI, AI-vs-AI) requires Python connected. A standalone double-click hangs at the listener.

## Human play (the CTO's recollection — confirmed)
- Humans DID play, via the same Python harness + socket the AIs used: `human_matchmaking.py` is a human-vs-AI ELO ladder.
- But the human's **input is read locally by Unity** (`PlayerController.GetInput`, when the tank's `AI` flag is false) — keyboard+mouse or gamepad — NOT piped through Python. Python sends a **zero placeholder** for the human slot (`human_matchmaking.py:22`) which Unity discards; Python computes/sends only the AI opponent's action and runs matchmaking/ELO.

## AI opponents are Python, not Unity
- C# "AI" tanks have zero decision logic — they copy `DriverController.instance.actions[playerID]` off the socket (`PlayerController.cs:145-156`). The "hand-coded dumb agents" (README) are Python: `rand_opp` random actions (`tank_env.py:381`) or a trained SB3 PPO policy (`tank_env.py:384`).

## The Driver↔Arena coupling (and the NREs)
- Flow: Driver boots → connects → `LoadScene("Arena")` → one match → back to Driver. **Arena depends on the Driver singleton.** Opening `Arena.unity` directly throws `NullReferenceException` (`GameController.cs:39`, `PlayerController.cs:50` deref `DriverController.instance.config` with instance null). **Pre-existing scene coupling** (introduced in "First commit") — confirmed NOT a regression from the Unity 6.5 upgrade or the startup-hygiene pre-fixes.

## Implication for M0
"Two humans play in a standalone build" is NOT achievable with the current code as-is — it requires either a native no-Python local-play mode in Unity, or a helper host that clocks the game.

**CTO decision (2026-06-17):** keep Python in the loop for human play — a lightweight Python "local-play host" drives the clock and stubs the human action slots (humans control locally in Unity); do NOT build a no-Python native mode. Forward-looking: use this connection to **record human play for imitation learning / behavioral cloning**. Recording the state stream is natural (Python already sees it); recording the human's *actual* action needs Unity to expose it back over the protocol — a sign-off-gated addition to the RL seam, design-for-it but not an M0 deliverable.

Known latent bug for later: `human_matchmaking.get_human_stats` passes a path string to `json.load` (`human_matchmaking.py:43`), so the "existing human" branch is broken as written.

## Config / arena source-of-truth map (consolidated 2026-06-18, cleanup A4)

There is ONE canonical location for the shipped config + arenas; the rest are explicit fallbacks. The duplicate `PythonScripts/Assets/` tree (stale 2021 copy) was DELETED — its only consumer was a dead `os.path.exists("./Assets/config.json")` pre-flight in `preamble.py` that never read the file; that check is now removed.

`DriverController.ResolveConfigPath` (`DriverController.cs:124-137`) precedence, highest first:
1. **`--config <path>` / `-config <path>` CLI arg** — explicit override. The Python local-play host (`play_local.launch_build`) passes this when a config is selected; otherwise omits it and lets the build fall back.
2. **`Assets/StreamingAssets/config.json` — CANONICAL DEFAULT (= the M1 training topology, AI-vs-AI).** Ships inside every build (StreamingAssets is copied into the player), so a built player resolves here when **no `--config` is passed**. The M1 trainer's env (`src/tank_twin/env.py`) launches the build with *only* `[game_path, port]` (no `--config`, by design — that path is the frozen RL seam), so this default MUST be the AI-vs-AI training config (`player1_ai`+`player2_ai` true, both socket-driven; `timeScale` 5 to run the socket-clocked sim faster). M0 human play does **not** rely on this default: `play_local.py` / the input-controls docs always pass `--config config_2p.json` (or `config_2p_keyboard.json`) for the human presets. Sibling presets `config_2p.json` / `config_2p_keyboard.json` live here too (selected via the `--config` arg above).
3. **`Assets/config.json` — legacy last-resort fallback.** Used only in-editor / when no StreamingAssets copy exists (the working-dir relative path). KEPT intentionally; do not delete.

Arena resolution (`DriverController.ResolveArenaPath`, `DriverController.cs:142-152`): the config's `arena_path` is taken as-is if absolute or if it exists relative to the working dir (legacy `Assets/Arenas/...` form), else resolved **relative to the directory of the resolved config file** — so config + its `Arenas/` travel together. Canonical arenas live in `Assets/StreamingAssets/Arenas/` (`default.json`, `custom1.json`, `nowin_test.json`).

Net: edit configs/arenas under `Assets/StreamingAssets/`. `Assets/config.json` is a deliberate fallback, not a duplicate.
