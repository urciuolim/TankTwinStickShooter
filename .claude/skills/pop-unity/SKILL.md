---
name: pop-unity
description: Contract + boundaries for the Unity game component (Assets/, C#) of the target architecture. Load when building or reviewing the Unity simulator.
---

# Component: `Assets/` (Unity game) — the simulator

**Responsibility:** the C# game that IS the simulator behind the env interface — renders the frame, runs the deterministic game loop, and speaks the TCP-JSON protocol (emits the 52-float state + the pixel frame; consumes actions). It is a component of the target architecture like the Python packages, just engine-imposed under `Assets/`.

**Contains:** the game loop, the driver / socket server (the Unity side of the protocol), input handling, arena/map loading + rendering, and the build entry point (Windows + headless).

**Boundaries:** speaks ONLY the protocol contract that `pop_trainer.core` defines — the Python side never reaches into Unity internals, and vice versa. The protocol / 52-float state layout is the **RL seam** (behavior-stable; Director sign-off to change). No runtime `UnityEditor` imports in shipping code.

**Quality (same gates as Python — the platform GO/NO-GO applies):** asmdefs (runtime + test); **Unity Test Framework EditMode unit tests** for pure logic (extracted from MonoBehaviours); **coverage via the Unity Code Coverage package**; strict-JSON-compatible output.

**Future:** the one-time map/wall-layout message on load/map-change (Unity → Python), designed jointly with `core/protocol` at `env/` integration.

**Inspiration (do NOT copy):** the existing `Assets/` game; the 2021 Unity project.
