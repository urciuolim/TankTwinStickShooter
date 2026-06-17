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

- [x] **Git checkpoint** so the upgrade is reversible. Checkpoint commit: `ace347d "Pre-Unity-6.5 upgrade checkpoint"` on `revival-2026`. A backup branch pointer `pre-unity6-backup` was created at that commit (`git update-ref refs/heads/pre-unity6-backup ace347d`) so the pre-upgrade state survives even if `revival-2026` advances.
- [ ] Expect a slow first reimport: this is a 6-major-version jump and it WILL surface the package issues below.

## Repeatable batchmode upgrade (headless)

This is the exact, repeatable invocation used to drive the reimport/API-update from the command line (git-bash on Windows). Run from any directory — `-projectPath` is absolute. It opens the project, lets the editor reimport assets, runs the API auto-updater (`-accept-apiupdate`), then quits.

```bash
"/c/Program Files/Unity/Hub/Editor/6000.5.0f1/Editor/Unity.exe" \
  -batchmode -quit -accept-apiupdate \
  -projectPath "C:/src/TankTwinStickShooter" \
  -logFile "C:/src/TankTwinStickShooter/Logs/unity6-upgrade.log"
```

- **Editor path:** `C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe`
- **Flags:** `-batchmode -quit -accept-apiupdate` (`-accept-apiupdate` lets the API updater rewrite obsolete API calls non-interactively; `-quit` exits after the import completes).
- **Log path:** `Logs/unity6-upgrade.log` (under the repo; `Logs/` is git-ignored). The eng-lead's reference invocation used `-logFile -` to stream to stdout; either works. Inspect the tail of the log for `error CS`, `Missing`, or licensing failures.
- **First-run note:** the first reimport is slow (full asset DB rebuild) and may appear to hang for several minutes — that is expected for a 6-major-version jump, not a failure. A second clean run (used for verification) is much faster.
- **If batchmode fails on licensing or hangs indefinitely:** STOP and open the project in the GUI editor instead; batchmode needs an activated license.

### Rollback (one line)

```bash
git checkout pre-unity6-backup    # or: git reset --hard ace347d   (then delete Library/ to force a clean reimport on next open)
```

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
