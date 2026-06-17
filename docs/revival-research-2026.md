# Revival research — 2026 (Board session 02 brief)

Synthesis of the four-lane research sprint (R1 RL/PBT stack, R2 engine/bridge + cluster constraint, R3 GCP/MLOps, plus the current-state code audit). Researched 2026-06-17; post-Jan-2026 facts verified against the live web by the analysts.

## TL;DR headlines

1. **The hard constraint is GO.** A Unity Linux build of a *vector-observation* game runs truly headless (`-batchmode -nographics`, Dedicated Server target) with **no GPU, no display, no Xvfb, and no runtime license**, and containerizes cleanly on GCP. The audit confirms our game uses **vector observations (52 floats)**, so we are on the easy path. Unity-as-trainer does **not** block cluster deployment.
2. **But naive engine-in-the-loop is the wrong workhorse.** Stepping Unity over IPC is 10²–10³× slower than a decoupled native/Python/JAX sim — it would dominate wall-clock and GCP cost for population-scale self-play. Recommended architecture is the **two-core "shared logic, two frontends"** pattern (details below).
3. **Modern stack:** **Stable-Baselines3 2.9 + PettingZoo 1.26 + a bespoke PFSP/ELO self-play league** (which is what our 2021 scripts already are, modernized). **Ray Tune PBT ≠ self-play** — it tunes hyperparameters, not opponent populations; don't confuse the two.
4. **GCP:** containerize → Artifact Registry → start on a single GPU job → scale to managed **Ray on Vertex AI** later. Topology: cheap **CPU Spot sim-workers + one GPU learner**. Default GPU quota is **0** — request early. Realistic part-time cost ≈ **$100–230/mo**.
5. **The 2021 code runs conceptually but is pinned to a dead era** (gym ~0.21, SB3 ~1.x, Linux/WSL assumptions). The self-play research IP is intact and worth preserving; the plumbing needs a rewrite.

## The hard constraint — verdict (R2)

GO, conditional on one thing we already satisfy:

- **Vector observations** (raycasts/positions/velocities — our 52-float state): headless build needs no graphics device at all. Drops into a slim Linux container on GKE/Compute Engine/Cloud Run. ✅ This is us.
- **Visual/camera observations**: would force Xvfb + a GL driver. We avoid this — the image-based CNN path in the old code is **deferred/optional**, not the training path.

Other confirmed facts: ML-Agents is **actively maintained** (package 4.0.3, 2026-04-17; commits today), exposes a low-level Python API + Gym/PettingZoo wrappers (so we can use our own trainer), and has built-in self-play + ELO. It requires **Unity 6** (6000.0+), so the 2019.4 → Unity 6 upgrade is on the critical path for the engine side. The custom TCP-JSON socket should be **retired** in favor of ML-Agents wherever Unity is in the loop.

## The architecture fork (decision needed)

**Option A — single-core, Unity-headless-only.** One codebase. Ship a headless ML-Agents build, train it on GCP (CPU sim-workers + GPU learner). Simpler to maintain; **slower and costlier training** at population scale; ties self-play to `mlagents-learn` or a PettingZoo+SB3 bridge over IPC.

**Option B — two-core, "shared logic, two frontends" (recommended).**
- **Unity (upgraded to 6)** stays the *playable, rendered* game — this is M0, and the trained policies run back inside it via ONNX/Sentis for demos and human-vs-model.
- A **fast portable sim** (pure-Python now, JAX/vectorized later) re-implements the exact tank logic as a headless gymnasium/PettingZoo env — **no engine, no socket** — and is the population-scale training workhorse. Trivially cluster-deployable, thousands of parallel matches per node.
- **Preserve the "RL in a real engine" signal** by *also* shipping a headless Unity + ML-Agents build on GCP, used to validate that fast-sim-trained policies transfer to the real engine. Cheap to stand up; it's the artifact that proves the signal.
- **Cost of B:** the game logic exists twice and must be kept in sync — mitigated by golden-trajectory parity tests (same seed + actions ⇒ identical state). This discipline is the main execution risk.

Why B fits us: our env is already a clean 52-float vector with a documented state layout, so the Python port is very feasible; B aligns with the SB3 + cheap-CPU-sim recommendations; and it gives fast/cheap local + cluster training **without** giving up Unity.

**Option C — defer:** proceed with M0 (identical under A or B) and lock the architecture at the M1 gate, once the game is running again.

## Recommended training stack (R1 + R3, tensions resolved)

- **Trainer:** Stable-Baselines3 2.9 (PPO). We already use SB3; it's the most readable, actively maintained path for a solo maintainer. Migration from the 2021 gym/SB3-1.x code is ~1–3 focused days for the env (gym→gymnasium 5-tuple `step`, `(obs,info)` `reset`), more for PettingZoo + league modernization.
- **Multi-agent API:** PettingZoo 1.26 (Farama, same family as Gymnasium). Adopt as the env interface now — cheap insurance that keeps RLlib/PufferLib/etc. open later.
- **Self-play / population:** keep it **bespoke** (our population + tournament + ELO is exactly the right architecture); the 2026 upgrade is **PFSP** opponent sampling (weight opponents by win-rate, `f(x)=(1−x)^p`) instead of uniform, plus optional exploiter agents. ELO/TrueSkill for matchmaking.
- **Do NOT** adopt Ray Tune PBT as "the modern population scheme" — it's a hyperparameter scheduler, not an opponent-league mechanism. Useful later, on top, to auto-tune PPO.
- **Scale-out:** first scale **population-wise** (many independent SB3 workers across GCP VMs — embarrassingly parallel, matches our existing design). Adopt **RLlib + Ray on Vertex AI** only if/when intra-run distributed sampling becomes the bottleneck — a deliberate later migration, not day one.
- **Throughput escape hatch:** PufferLib 3.0 if single-box SB3 `VecEnv` speed becomes the wall before cluster time.

## GCP target (R3)

- **Crawl:** containerize the trainer (+ headless sim) → push to **Artifact Registry** via Cloud Build → run as a **single-GPU job**: a Deep Learning VM, a **Cloud Run Job with an L4** (no quota request needed; scales to zero), or a Vertex AI CustomJob (T4/L4). Start with a 2–4 trial population in one process.
- **Run:** **Ray on Vertex AI** (managed Ray, no Kubernetes) when you outgrow one box. GKE + KubeRay only if you later want max control and accept the K8s ops.
- **Cost/quota:** default GPU quota is **0** — request T4/L4 days ahead. GPU Spot discount is small (~12%); CPU Spot is large (~55–72%) → keep the GPU off when idle, run a herd of cheap CPU Spot sim-workers.
- **Note:** "Vertex AI" was rebranded **Gemini Enterprise Agent Platform** at Cloud Next 2026 — branding/console only; REST APIs and `/vertex-ai/` doc URLs unchanged.
- **Azure ML** remains a viable fallback you know, but its Spot economics for ML clusters are currently weaker and its first-party RL tooling is retired — you'd BYO-Ray either way, so GCP is the better fit now.

## Current-state audit highlights (what we're working with)

- **Self-play IP (preserve):** population of PPO agents, each trained via self-play against **ELO-proximity-sampled** opponents, periodically round-robin tournamented to update Elo, with **PBT-style replace-and-mutate** of weak agents and optional **nemesis/survivor exploiters**. `elo.py` is pure and correct — first thing to unit-test.
- **Env:** action `Box(5)` = [velX, velY, aimX, aimY, trigger]; obs `Box(52)` vector (or optional image). Reward +1/−1/0. Opponent perspective via cheap half-swap of the state vector.
- **Protocol (rewrite):** unframed `recv(1024)` + one `json.loads`; mirror on the Unity side. Assumes one TCP read == one JSON object — fragile.
- **Version landmines (2026):** gym→gymnasium API drift; SB3 1.x→2.x attribute access; `start_method="fork"/"forkserver"` (neither exists on Windows — needs `spawn`); `os.system("zip"/"rm"/"cp"/"mv")`; latent missing `import json` in `train_pop.py`/`tournament.py`; `eval.py` summary-call arity bug; non-strict JSON (trailing commas, leading-dot floats).
- **M0 input reality (legacy Input Manager, no New Input System):**
  - **Two Xbox pads = lowest effort** — `Joy1`/`Joy2` axes are already wired; main work is re-mapping Xbox axis indices for Windows-2026 and enabling both players as human in config.
  - **Shared keyboard = net-new work** — only one keyboard+mouse scheme exists today, and it uses the mouse for aim (can't be shared). True split-keyboard two-player is new bindings.
  - **Build blocker:** `using UnityEditor;` in `GameController.cs:9` prevents standalone builds — must be removed.
  - The clean way to get robust two-pad **and** shared-keyboard is Unity 6's **New Input System** (a rewrite of `GetInput`), which we want for the RL phases anyway.

## The M0 plan (playable game, 2 humans, both pads + shared keyboard)

**Step 1 — Proof of life on Unity 2019.4 (fast win, de-risks "does it even open").**
Open the project in its native editor; remove `using UnityEditor;`; fix the config-key typos + default `fixedDeltaTime`; add a both-players-human config preset; verify **two-Xbox-pad** play (Joy1/Joy2 already wired, re-map axes). Deliverable: a recorded local 2-human match on pads. Establishes a working baseline before the upgrade.

**Step 2 — Upgrade to Unity 6** (on the critical path for ML-Agents, Dedicated Server headless builds, and the New Input System). Verify scenes/prefabs/tilemaps/materials reimport; editor play works; produce a Windows build.

**Step 3 — New Input System: deliver both input methods.** Re-implement input with `PlayerInput` + control schemes/device pairing: robust **two Xbox controllers** AND **shared keyboard** (split bindings, e.g. WASD+fire vs arrows/IJKL+fire; stick-aim for pads, key-aim for keyboard so the mouse isn't shared).

**M0 acceptance:** two humans play a full match from documented steps, in a standalone build, via **(a) two Xbox controllers** and **(b) a shared keyboard**.

Sequencing recommendation: **Step 1 first** (proof of life on 2019.4) is cheap, gives an early demo, and follows the right practice of verifying the project in its native editor before a six-version engine jump. Full M0 "done" (shared keyboard) lands after the Unity 6 upgrade.

## Open decisions for the board

1. **Target architecture:** A (single-core Unity-headless) · **B (two-core, recommended)** · C (defer to M1 gate).
2. **M0 sequencing:** proof-of-life-on-2019.4-then-upgrade (recommended) · upgrade-to-Unity-6-first.
3. **Approve the M0 plan** and start the Game & sim squad.

(Trainer stack — SB3 + PettingZoo + bespoke PFSP/ELO, RLlib deferred — is presented as the recommendation; flag any disagreement in notes.)

## Addendum (2026-06-17): board session 02 outcome + pixel-observation constraint

**Decisions:** architecture **deferred to the M1 gate**; M0 sequencing = **upgrade to Unity 6 first**; M0 plan **approved**, Game & sim squad started. See `unity6-upgrade-runbook.md`.

**New CTO constraint — agents observe pixels, not state.** A 2021 state-based MLP "didn't work"; a CNN over a small RGB grid did. Verified in code: this grid is **synthesized in Python from the 52-float state** (`tank_env.py` `draw_state`), NOT a Unity camera render (`GameController.cs` has no ScreenCapture/RenderTexture). The grid (~36×60×3 uint8) encodes self in the R channel, opponent in B, walls in G, with intensity distinguishing position/velocity/aim/bullets. Consequences:
- The headless-on-GCP **GO** verdict still holds — the R2 "visual observations force Xvfb" caveat was about *camera frames*, which this is not.
- Pixels are trivially fast/reproducible in a pure-Python/JAX sim (`draw_state` is already numpy) → this **strengthens** the deferred two-core option rather than threatening it.
- Default policy becomes SB3 **CnnPolicy**, not MlpPolicy. Leaning is "repro the old way first"; a modern real-rendered-frame CNN is a heavier path (reintroduces a render dependency) to weigh at the M1 gate.

**ML-Agents vs SB3 (CTO question).** Different layers: **ML-Agents** is the Unity↔Python *bridge* (modern replacement for the 2021 socket) and optionally bundles trainers; **SB3** is the *RL algorithm* (PPO). Under the recommended two-core path the training workhorse is a native Python sim, so SB3 trains directly and **ML-Agents is not a core dependency** — it only matters for the optional "RL running inside real Unity" validation/demo. Because pixels are synthesized from state (no Unity render needed), ML-Agents' role for M1/M2 is minimal/optional; it becomes relevant only if a future decision puts Unity itself in the training loop. **Is SB3 best?** Best *fit* for our constraints (revive existing SB3 code, solo maintainer, readable, maintained, PPO + CnnPolicy out of the box) — not the highest-throughput (PufferLib) or most-scalable-distributed (RLlib) in absolute terms. We hedge by keeping the env on the PettingZoo interface so the trainer stays swappable.

