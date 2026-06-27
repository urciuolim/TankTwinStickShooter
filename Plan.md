# Tank Twin Stick Shooter 2026 Revival Plan

## Purpose

Revive this project as a maintainable local game and reinforcement-learning sandbox. The immediate target is a clean developer experience for:

- Local play: run the Unity game locally with human and/or scripted/AI players.
- Local training: run a small PPO training smoke test on Windows or Linux.
- Local inference: run trained agents on Windows or Linux for evaluation, demos, or human-vs-model play.
- Future scale-out: keep the architecture compatible with multi-process and cluster training, but do not build cluster scripts, deployment, or CI/CD yet.

## Non-Goals For Now

- No CI/CD pipeline.
- No cloud, cluster, or paid GPU execution.
- No deployment packaging beyond local Unity builds needed for development/training.
- No major gameplay redesign until the existing environment is reproducible.

## Guiding Principles

- Preserve the original research value: Unity as the simulator, Python as the training/evaluation driver, and agent populations/tournaments as a future extension.
- Target Unity 6.5 for the revived project. Treat Unity 2019.4 as the legacy import/migration source, not the long-term home.
- Target Python 3.12 with `pyproject.toml` as the Python project source of truth.
- Make Windows and Linux first-class runtime targets for training and inference. WSL can remain a useful compatibility path, but the code should not require it.
- Make local workflows boring and reliable before adding new ML features.
- Prefer small modular seams over large rewrites.
- Add tests around behavior before changing protocol, rewards, state layout, or training orchestration.
- Keep cluster-readiness as an interface design constraint, not an immediate implementation project.

## Current Readout

The repo is currently a Unity 2019.4.28f1 2D twin-stick tank game with a Python Gym environment around it. Unity communicates with Python over TCP JSON messages. Python contains training, evaluation, population, tournament, plotting, and matchmaking scripts built around Stable-Baselines3 PPO.

The revival target is Unity 6.5. If the project needs an intermediate safety step, use the latest patched Unity 2019.4 editor only as a migration bridge before moving to Unity 6.5.

Important revival risks already visible:

- Python dependency setup is implicit; there is no `pyproject.toml` or locked environment file.
- Some JSON files rely on permissive parsing, including trailing commas and non-standard numeric literals.
- The socket protocol sends raw JSON without explicit message framing, which can break when TCP packets are split or coalesced.
- Training scripts assume Unix-like tools and process behavior, such as `fork`, `forkserver`, `zip`, and `rm`.
- The historical WSL/supercomputer workflow means path handling, process launch, multiprocessing, and artifact cleanup need a portability pass for native Windows and Linux.
- The Unity 6.5 upgrade may require package updates, asset reimports, API fixes, and verification that simulator timing remains equivalent.
- Unity runtime scripts contain small correctness issues, such as config-key mismatches and a runtime `UnityEditor` import.
- Evaluation/training scripts are script-oriented rather than package-oriented, which makes unit testing and reuse harder.

## Target Architecture

### Unity

Target editor:

- Unity 6.5.

Keep Unity responsible for:

- Game simulation.
- Human controls.
- Rendering and local play.
- Arena loading.
- Exposing simulation state/actions through a stable local protocol for training and inference.
- Producing local simulator builds for Windows and Linux.

Refactor toward these modules over time:

- `Game`: player, bullet, health, game timer, win/loss state.
- `Arena`: arena JSON loading, validation, tile placement.
- `Config`: strongly named runtime config with defaults.
- `Protocol`: message serialization/deserialization and socket lifecycle.
- `AI Bridge`: maps external actions to players and publishes observations.
- `Build Profiles`: editor, local Windows simulator, local Linux simulator, and future headless simulator settings.

Testing target:

- EditMode tests for pure C# config/protocol/arena parsing logic.
- PlayMode smoke tests for scene loading, game reset, simple match completion, and bullet/player interactions.

### Python

Move from loose scripts toward a package, while keeping old script entry points during migration.

Proposed package layout (historical — realized instead as the `src/pop_trainer/` package; the
2021 tree this migrated from lives in git history):

```text
<2021 trainer root>/
  tank_twin/
    __init__.py
    config.py
    protocol.py
    env.py
    unity_process.py
    rewards.py
    opponents.py
    training.py
    evaluation.py
    arenas.py
  scripts/
    play_local.py
    train_local.py
    eval_local.py
  tests/
```

Python responsibilities:

- Launch or connect to Unity builds on Windows and Linux.
- Implement the Gym/Gymnasium-compatible environment.
- Own training, inference, and evaluation loops.
- Own model registry/population metadata.
- Provide deterministic local smoke tests using mocks where possible.

Testing target:

- `pytest` unit tests for protocol framing, JSON parsing, reward calculation, state transforms, opponent selection, ELO utilities, and model metadata.
- Unit tests for Windows/Linux path handling, process command construction, and artifact cleanup.
- Integration smoke test that runs without Unity by using a fake socket/server.
- Optional local integration test that launches a Unity build, marked separately so it does not run by default.

## Phase 0: Baseline And Reproducibility

Goal: make the current project understandable, runnable, and safe to modify.

Tasks:

- Document the legacy Unity version, the Unity 6.5 revival target, Python 3.12, expected build targets, and known WSL/Linux assumptions.
- Add `pyproject.toml` with Python 3.12 as the supported runtime and the initial dependency/test tooling configuration.
- Create a minimal local runbook in `README.md`:
  - Open Unity project.
  - Run local human play.
  - Build simulator executable.
  - Run local training smoke test.
  - Run local inference/evaluation with a model.
- Normalize JSON files to strict JSON where Python needs to parse them.
- Fix obvious startup/runtime correctness issues:
  - Config key mismatches in Unity.
  - Runtime `UnityEditor` import.
  - Default `fixedDeltaTime` behavior if config is missing.
  - Evaluation summary call mismatch.
- Add a `docs/` folder only if README becomes too large.

Acceptance criteria:

- A new contributor can identify the legacy Unity version, target Unity 6.5 version, and Python 3.12 requirement from the repo.
- Strict Python JSON parsing succeeds for config and arena files used by local training.
- Existing local play still works.
- At least one Python smoke command can run far enough to import core modules.

## Phase 1: Unity 6.5 Upgrade

Goal: move the project onto the supported Unity version that the revival will use long term.

Tasks:

- Back up current Unity project state through Git before opening in a newer editor.
- Try a direct upgrade from Unity 2019.4.28f1 to Unity 6.5.
- If direct upgrade is messy, use the latest patched Unity 2019.4 editor as an intermediate migration step.
- Update Unity packages to versions compatible with Unity 6.5.
- Verify scenes, prefabs, sprites, tilemaps, materials, and Json.NET assets after reimport.
- Verify local play in the editor.
- Produce Windows and Linux simulator builds.
- Document any manual upgrade steps and known asset/package changes.

Acceptance criteria:

- `ProjectSettings/ProjectVersion.txt` reflects Unity 6.5.
- The project opens cleanly in Unity 6.5.
- `Driver` and `Arena` scenes load without missing script or package errors.
- A local editor play session works.
- Windows and Linux simulator builds can be produced from the upgraded project.

## Phase 2: Local Play Path

Goal: make the game pleasant to run locally independent of training.

Tasks:

- Define local play config presets:
  - Human vs AI/scripted.
  - Human vs human, if input mappings are still usable.
  - AI vs AI visual/debug mode.
- Separate human-play config from training config.
- Add clear scene/run instructions.
- Make arena selection explicit and robust.
- Add a simple scripted opponent mode if the current AI path requires Python too early.
- Ensure local play does not require a Python socket unless the selected mode needs one.

Acceptance criteria:

- The Unity Editor can run a local match from documented steps.
- A human can play without starting the Python training stack.
- Config changes for local play are obvious and reversible.

## Phase 3: Local Training And Inference Smoke Tests

Goal: make small local training and inference loops reliable and cheap on Windows and Linux.

Tasks:

- Create a single `train_local` entry point with explicit arguments and sane defaults.
- Create a single `infer_local` or `eval_local` entry point for running a saved model without training.
- Support one Unity simulator instance first.
- Add a tiny training profile:
  - Small timestep count.
  - One arena.
  - One simple opponent mode.
  - Output into an ignored local artifact directory.
- Add a tiny inference profile:
  - Load one saved or fixture model.
  - Run a small number of matches.
  - Report wins/losses/ties and average steps.
- Add lifecycle cleanup so Unity subprocesses close when Python exits or errors.
- Make port assignment explicit and documented.
- Use platform-neutral process launch and path handling.
- Keep the old population-training scripts available but mark them as legacy until migrated.

Acceptance criteria:

- `train_local` can run a short local training smoke test on Windows and Linux.
- `infer_local` or `eval_local` can run model inference/evaluation on Windows and Linux.
- Logs and model artifacts land in predictable ignored folders.
- A failed run does not leave orphaned simulator processes under normal conditions.

## Phase 4: Protocol Hardening

Goal: make Unity/Python communication reliable enough for long runs.

Tasks:

- Replace raw JSON-over-TCP reads with explicit message framing.
- Prefer newline-delimited JSON or length-prefixed JSON; choose one and implement it on both sides.
- Add protocol versioning to messages.
- Define message types:
  - `start`
  - `restart`
  - `end`
  - `state`
  - `actions`
  - `ack`
  - `error`
- Add timeout and reconnect behavior in one place instead of scattered try/except blocks.
- Add tests with split/coalesced messages to prove framing works.

Acceptance criteria:

- Protocol tests pass for partial reads and multiple messages in one read.
- Unity and Python both reject malformed messages with useful errors.
- Training code no longer depends on `recv(1024)` returning exactly one full JSON object.
- Training and inference use the same protocol client.

## Phase 5: Python Modularization And Standards

Goal: make Python code testable without losing the working scripts.

Tasks:

- Introduce `tank_twin` package modules gradually.
- Move pure functions first:
  - ELO calculations.
  - Opponent selection.
  - State flipping/drawing.
  - Reward calculation.
  - Model path/stats helpers.
- Convert script globals into typed config objects.
- Use `pathlib` instead of manual string path concatenation in touched code.
- Replace shell-specific `os.system("zip ...")` and `os.system("rm ...")` with standard-library equivalents.
- Replace Unix-specific multiprocessing assumptions with a cross-platform worker strategy that supports Windows `spawn` and Linux process models.
- Add `pytest`.
- Add `ruff` or equivalent lightweight linting/formatting config.
- Add type hints where they clarify boundaries.

Acceptance criteria:

- Core utility tests run without Unity.
- Local training uses the package API rather than importing large script files as libraries.
- Local inference uses the same package API as local training.
- Existing legacy scripts either still work or have documented replacements.

## Phase 6: Cross-Platform Runtime Compatibility

Goal: make native Windows and native Linux equally supported for training and inference.

Tasks:

- Define platform-specific Unity simulator build paths through config, not hard-coded paths.
- Use `pathlib`, `subprocess` argument lists, and standard-library archive/file APIs.
- Avoid shell-only commands and shell-specific quoting.
- Add a process launcher abstraction with Windows and Linux coverage.
- Add tests for command construction, path normalization, artifact paths, and cleanup behavior.
- Ensure port allocation, socket timeouts, and subprocess shutdown behave consistently on Windows and Linux.
- Document any GPU-specific Python install notes separately from the core CPU/local smoke setup.

Acceptance criteria:

- Local training and inference commands have the same interface on Windows and Linux.
- No required code path depends on WSL-only behavior.
- Cross-platform unit tests cover launcher/config/path behavior.
- Linux remains cluster-friendly, but local Windows is not treated as a second-class target.

## Phase 7: Unity Modularization And Tests

Goal: reduce coupling in runtime scripts and make core behavior testable.

Tasks:

- Add Unity assembly definitions if useful for separating runtime and tests.
- Extract config loading into a dedicated class.
- Extract arena loading/validation from `GameController`.
- Extract protocol/socket behavior from `DriverController`.
- Keep MonoBehaviours thin where possible.
- Add EditMode tests for config and arena parsing.
- Add PlayMode tests for:
  - Scene boot.
  - Match timer ending.
  - Player death ending match.
  - Bullet collision/despawn behavior.

Acceptance criteria:

- Core C# parsing/protocol logic can be tested without loading full scenes.
- PlayMode tests cover the highest-risk gameplay state transitions.
- Runtime code no longer imports editor-only namespaces.

## Phase 8: Training, Inference, And Evaluation Cleanup

Goal: make experiments repeatable and understandable.

Tasks:

- Define a model/artifact directory structure:
  - `runs/`
  - `models/`
  - `logs/`
  - `datasets/`
- Keep generated artifacts ignored by Git.
- Standardize stats metadata with versioned JSON.
- Add deterministic seeds to local training/eval commands.
- Create a small baseline scripted opponent.
- Create a model evaluation command that can run a few matches and print a concise summary.
- Create a model inference command suitable for demos and human-vs-model play.
- Preserve population/tournament concepts, but move them behind cleaner APIs.

Acceptance criteria:

- A local model can be trained, saved, loaded, and evaluated from documented commands.
- A local model can be used for inference without invoking training code.
- Evaluation output includes wins/losses/ties, average reward, and average steps.
- Experiment metadata records config, seed, arena, code version, and model path.

## Phase 9: Future Cluster-Ready Design

Goal: avoid painting the project into a local-only corner.

Tasks to design, not execute yet:

- Keep trainer configuration serializable and runnable from CLI.
- Keep Unity simulator workers stateless except for their port/config.
- Keep port allocation configurable by base port plus worker index.
- Keep artifacts isolated by run ID.
- Make process launch strategy injectable:
  - Local Windows subprocess.
  - Local Linux subprocess.
  - Pre-launched simulator.
  - Future scheduler-managed worker.
- Document what a future SLURM or cluster adapter would need.

Acceptance criteria:

- Local training code does not assume a specific machine, username, or absolute path.
- Local inference code does not assume a specific machine, username, or absolute path.
- A future cluster runner can reuse the same training API and config objects.
- No paid cluster work is required for the revival branch.

## Suggested First Pull Requests

1. `docs/baseline-revival`
   - Add environment notes, Unity 6.5 target, Python 3.12 target, Windows/Linux runbook skeleton, and known issues.

2. `unity/upgrade-6-5`
   - Upgrade the project to Unity 6.5 and verify editor play plus Windows/Linux simulator builds.

3. `fix/startup-hygiene`
   - Fix Unity config key mismatches, remove runtime `UnityEditor` import, fix strict JSON files, and fix the `eval.py` summary call.

4. `python/test-foundation`
   - Add `pyproject.toml`, Python 3.12 metadata, `pytest`, and initial tests for ELO/opponent/state helpers.

5. `python/local-train-infer-entrypoints`
   - Add simple local training and inference CLIs with one simulator instance and predictable artifacts.

6. `python/cross-platform-runtime`
   - Remove WSL-only assumptions from process launch, paths, artifact cleanup, and multiprocessing.

7. `protocol/framing`
   - Add framed JSON protocol on both Python and Unity sides with tests.

## Definition Of Done For The Revival

The project can be considered revived when:

- Local play is documented and works from Unity.
- The project runs on Unity 6.5.
- Local training can run a cheap smoke test from a documented command on Windows and Linux.
- Local inference/evaluation can run from a documented command on Windows and Linux.
- Core Python utilities have unit tests.
- Core Unity parsing/protocol/game-end behavior has tests or clear test seams.
- The Unity/Python socket protocol is framed and versioned.
- Generated artifacts are ignored and organized.
- Future cluster training is an extension of the local training API, not a separate one-off script pile.
