# Unity 2019.4 → Unity 6 upgrade runbook

Owner: Game & sim squad. Goal: move the project onto **Unity 6 (6000.x LTS)** — the engine target for M0 (New Input System) and the RL phases (ML-Agents 4.x requires Unity 6). Sequencing chosen by the CTO: **upgrade first**, then the M0 input work.

## What needs you (human, on the Windows + GPU box)

The actual editor upgrade is GUI/host-dependent and cannot be driven by an agent:

1. ~~Install Unity Hub + Unity 6 LTS~~ — **DONE**: Unity **6.5 (6000.5.0f1)** installed at `C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe`.
2. Open this project in Unity 6 and let it reimport.
3. Resolve any remaining compile/package errors (most are pre-fixed below).
4. Verify play, then produce a Windows build.

Everything around that — pre-fixes, package analysis, this runbook, and the M0 input work afterward — the squad does.

## Before opening in Unity 6

- [ ] **Git checkpoint** so the upgrade is reversible (the director can create a `pre-unity6-backup` branch / commit on request — say the word).
- [ ] Expect a slow first reimport: this is a 6-major-version jump and it WILL surface the package issues below.

## Package manifest issues to resolve (`Packages/manifest.json`)

- `com.unity.2d.tilemap.extras: "...2d-extras.git#master"` — an **unpinned git dependency**; replace with the registry package pinned to a Unity 6-compatible version.
- `com.unity.toolchain.win-x86_64-linux-x86_64: "0.1.19-preview"` — preview Linux-build toolchain; update to the Unity 6 equivalent (needed later for headless Linux builds).
- 2019-era packages (`2d.*`, `textmeshpro` 2.1.4, `timeline` 1.2.18, `ide.rider`/`ide.vscode`, `collab-proxy`) — the upgrade will bump these; let the editor resolve, then verify.
- **Not present yet (added later):** `com.unity.inputsystem` (M0 step 3) and `com.unity.ml-agents` (RL phase).

## Code pre-fixes (already applied by the squad — verify they compile on first open)

- [x] Removed build-blocking `using UnityEditor;` from `GameController.cs` (was line 9; confirmed unused). The editor-only namespace breaks standalone builds.
- [x] Fixed config-key typos in `GameController.cs`: `config["coutndownStart"]` → `config["game_countdownStart"]`; `config["healthBarMaxWidth"]` → `config["game_healthBarMaxWidth"]` (the guards already used the `game_`-prefixed keys, so these were silently falling back to defaults).
- [ ] Defensive default for `fixedDeltaTime` in `DriverController.cs` (if `ai_fixedDeltaTime` is absent it can be 0 → motion multiplies by zero). Shipped `config.json` sets `0.02`, so non-blocking; apply when next touching that file.

## Open + verify checklist (in Unity 6)

- [ ] Project opens; no missing-script / missing-package errors after reimport.
- [ ] `Driver` and `Arena` scenes load.
- [ ] Sprites, tilemaps, materials, prefabs reimport correctly (tank, bullet, HUD).
- [ ] Editor play: a local match runs (human vs AI from the shipped config).
- [ ] Produce a Windows standalone build; it launches and plays.
- [ ] Document any manual fixes Unity prompted for.

## Then: M0 input work (step 3)

On Unity 6, migrate input from the legacy Input Manager to the **New Input System** (`com.unity.inputsystem`): control schemes + device pairing for robust **two Xbox controllers AND a shared keyboard** (split bindings; key-aim so the mouse is not shared).

**M0 done =** two humans play a full match from documented steps, in a standalone build, via both input methods.
