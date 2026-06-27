# Input controls (M0 T4 — New Input System)

Human input for the two tanks runs on Unity's **New Input System** (`com.unity.inputsystem` 1.19.0).
Active Input Handling is set to **Both**, so the legacy `Input.*` path (used nowhere for human
actions now, and never for AI actions — those come from the socket) keeps working during the
transition. The AI / socket path is unchanged.

## What drives a tank

Each human tank (`playerID` 1 or 2, set in `Arena.unity`) reads input from its own clone of
`unity/Assets/Input/TankControls.inputactions` (action map `Tank`, actions `Move` / `Aim` / `Fire`).
The action vector applied locally is still `[velX, velY, aimX, aimY, trigger]` — only the source
of the values changed (`PlayerController.GetInput`, human branch).

Selection is by config flag + `playerID`:

| `player{ID}_keyboard` | Device for that tank                                    |
|-----------------------|---------------------------------------------------------|
| `false` (default)     | Gamepad: `playerID 1 -> first pad`, `playerID 2 -> second pad` |
| `true`                | Shared keyboard: `playerID 1 -> left cluster`, `playerID 2 -> right cluster` |

Pads are pinned by index (`Gamepad.all[playerID-1]`) so two identical Xbox pads never
cross-assign. Both keyboard players share ONE physical keyboard; they are kept disjoint by
**binding group** (`KeyboardLeft` vs `KeyboardRight`), not by device.

## Control mappings

### Xbox pad (per pad, `player{ID}_keyboard = false`)
- **Move:** left stick
- **Aim (barrel):** right stick
- **Fire:** right trigger (analog; fires past `player_triggerThreshold`, default 0.5)

### Shared keyboard — Player 1 (left cluster, `player1_keyboard = true`)
- **Move:** `W` up / `A` left / `S` down / `D` right
- **Aim (barrel):** `T` up / `F` left / `G` down / `H` right
- **Fire:** `Left Shift`

### Shared keyboard — Player 2 (right cluster, `player2_keyboard = true`)
- **Move:** Arrow keys (`Up` / `Left` / `Down` / `Right`)
- **Aim (barrel):** `I` up / `J` left / `K` down / `L` right
- **Fire:** `Right Shift`

All keyboard clusters are disjoint (WASD, TFGH, LeftShift vs Arrows, IJKL, RightShift). Aim is
key-based, not mouse-based, so the keyboard is genuinely shareable by two players.

## Config presets (`unity/Assets/StreamingAssets/`)
- `config_2p.json` — two humans, **both on pads** (default; most ergonomic for twin-stick).
- `config_2p_keyboard.json` — two humans, **shared keyboard** (P1 left cluster, P2 right cluster).
- To mix (pad + keyboard), copy a preset and set one player's `player{ID}_keyboard` to the opposite value.

## Two-human play-test (owed to a human + verification-lead)

Real device input cannot be verified headlessly. The game is a Python-clocked simulator, so a
local-play host must clock the sim while humans control locally in Unity (point it at one of the
two-human config presets above — `config_2p.json` / `config_2p_keyboard.json` — and at the
standalone build). The 2021 `play_local.py` host that did this was retired (it lives in git
history); a `pop_trainer` local-play entrypoint that supersedes it is not yet wired, so this
play-test is still owed.
