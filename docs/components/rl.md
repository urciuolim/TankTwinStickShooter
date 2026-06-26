# rl

The **online RL** component — Stable-Baselines3 PPO with a `CnnPolicy` over the env's pixel
observations, plus the self-play machinery (opponent roster, ELO) layered as wrappers. **Three seams
ship today:** the policy↔encoder seam ([`EncoderExtractor`](../../src/pop_trainer/rl/extractor.py))
and the **self-play opponent seam** ([`selfplay.py`](../../src/pop_trainer/rl/selfplay.py) —
`OpponentProvider` + `SelfPlayWrapper` + `ScriptedOpponent`/`Opponent`), and now the **eval + ELO
seam** ([`evaluate.py`](../../src/pop_trainer/rl/evaluate.py) `evaluate_winrate` /
[`callbacks.py`](../../src/pop_trainer/rl/callbacks.py) `EvalWinRateCallback` /
[`elo.py`](../../src/pop_trainer/rl/elo.py) pure ELO math). What still lands in **later `rl`
tasks** is just the **PPO train loop** and the **live self-play smoke** — the package docstring
says so (`rl/__init__.py:22`).

**Boundary:** `rl` imports [`core`](core.md) (`core.state.split_state_for_opponent` — the
perspective flip), [`env`](env.md) (the symmetric `TankEnv` the wrapper wraps),
[`models`](models.md) (the encoder factory), and [`agents`](agents.md) (`make_agent` + the canonical
selector registry) — plus torch / gymnasium / numpy / stable-baselines3 / stdlib. It imports
**nothing** from [`data`](data.md) or `pretraining`: the pretrained encoder is consumed as a loaded
`state_dict` artifact (not via a `models.from_pretrained`, `extractor.py:16-17`), and the self-play
seam re-uses the opponent-driving **primitives** (`split_state_for_opponent`, `make_agent`) directly
rather than the `data` module (`selfplay.py:26-28`). The empirical boundary check leaks `[]`. No
cycles.

## Key classes / entry points

### The encoder seam

- [`EncoderExtractor`](../../src/pop_trainer/rl/extractor.py) — a Stable-Baselines3
  [`BaseFeaturesExtractor`](../../src/pop_trainer/rl/extractor.py) (from
  `stable_baselines3.common.torch_layers`, `extractor.py:26`) that **wraps the shared
  [`models.Encoder`](models.md)** so SB3's `CnnPolicy` reads the SAME flat embedding the supervised
  pretraining produces (`extractor.py:34`). It is re-exported from the package alongside the
  self-play symbols and the eval seam — `rl/__init__.py`'s `__all__` is `EncoderExtractor`,
  `EvalWinRateCallback`, `Opponent`, `OpponentProvider`, `ScriptedOpponent`, `SelfPlayWrapper`,
  `evaluate_winrate`, `win_rate` (`rl/__init__.py:40-49`). Note `elo` is NOT exported — it is reached
  as `from pop_trainer.rl.elo import ...`.
  - **Fixed architecture, no knobs.** The encoder is built `EncoderConfig(trunk="nature",
    pooling="flatten")` — a NatureCNN trunk + Flatten/FC head — at the canonical **360×640** frame
    (`extractor.py:61-63`). The extractor exposes NO architecture knobs; the trunk / pooling /
    resolution ablation lives in the [`models`](models.md) benchmark, NOT here (`extractor.py:5-8`).
    The constructor is `EncoderExtractor(observation_space, *, checkpoint=None, freeze=False)` —
    only those two keyword knobs (`extractor.py:51-57`).
  - **Derived, not hard-coded.** `in_channels` comes from the obs space `(C, H, W)`
    (`extractor.py:59,62`); `features_dim` is the encoder's flat embedding size probed at the obs
    `H × W` via `encoder.embedding_dim(input_hw=(H, W))` (`extractor.py:65`). It is NOT 84×84 — an
    84×84 frame would collapse the NatureCNN stem.
  - **Owns the pretrained-encoder load.** Given a `checkpoint`, it `torch.load`s a raw encoder
    `state_dict` and applies it via `self.encoder.load_state_dict` (`extractor.py:76-78`). There is
    NO `models.from_pretrained` — the extractor owns the load itself.
  - **The freeze contract.** `freeze=True` clears `requires_grad` on every encoder parameter
    (`extractor.py:80-82`). Freezing works through the **grad path** — SB3 builds its optimizer over
    ALL policy params, so a frozen param keeps a `None` grad under that optimizer and is never
    updated — NOT by excluding params from the optimizer. This is load-bearing: the old reference
    trainer got it wrong by excluding params (`extractor.py:11-14`).
  - **`forward` does not re-normalize.** SB3's preprocessing has already cast the uint8 frame to
    float and divided by 255, so observations arrive in `[0, 1]`; `forward` just returns
    `self.encoder(observations)` (`extractor.py:84-90`).

### The self-play opponent seam

The four symbols of [`selfplay.py`](../../src/pop_trainer/rl/selfplay.py) (all re-exported from the
package, `rl/__init__.py:33-38`). Self-play is a **WRAPPER over the trainer** (the ML-Agents
`ghost/` precedent), **never woven into the env or the PPO core** (`selfplay.py:7-8`).

- [`OpponentProvider`](../../src/pop_trainer/rl/selfplay.py) — a roster of opponents plus a
  **per-EPISODE** sampling strategy (`selfplay.py:116`). `sample()` is called once per episode at
  reset to pick that episode's opponent (`selfplay.py:167-174`).
  - **Two strategies.** `"round_robin"` cycles the roster in order with a modulo index
    (`selfplay.py:169-172`); `"uniform"` draws one uniformly from a **SEEDED**
    `np.random.default_rng` (`selfplay.py:146,174`) so a fixed `seed` reproduces the sequence of
    picks.
  - **Validated.** An unknown `strategy` raises `ValueError` listing the valid names
    (`selfplay.py:139-141`); an empty roster raises `ValueError` (`selfplay.py:143-144`). Valid
    strategies are `("round_robin", "uniform")` (`selfplay.py:54`).
  - **`from_roster` builds from `agents` selectors.** `from_roster(roster=DEFAULT_ROSTER, ...)`
    wraps each selector as `ScriptedOpponent(make_agent(sel, seed=seed))` (`selfplay.py:149-165`);
    `seed` threads into BOTH `make_agent` and the provider's sampling RNG. `DEFAULT_ROSTER` is the
    five canonical selectors — `noop`, `random`, `aggressive-coverage`, `wall-hugger`,
    `opponent-shadower` (`selfplay.py:46-52`) — the SAME selectors the collection runner resolves
    through the [`agents`](agents.md) registry.
- [`ScriptedOpponent`](../../src/pop_trainer/rl/selfplay.py) — wraps a scripted
  [`agents`](agents.md) `Agent` (`selfplay.py:74-87`). `obs_kind == "state"` — it consumes the
  52-float wire state (`selfplay.py:84`), and `act` delegates straight to the wrapped agent
  (`selfplay.py:89-91`). `set_map` / `reset` **getattr-PROBE** the wrapped agent and **no-op when
  the hook is absent** (`selfplay.py:93-113`) — mirroring `data.collect`'s `_maybe_set_map` /
  `_maybe_reset` but **re-implemented here**, not re-imported, to keep the `data` boundary.
- [`Opponent`](../../src/pop_trainer/rl/selfplay.py) — the runtime-checkable `Protocol` for the
  player2 side: `obs_kind` (which view the wrapper feeds) + `act` (returns the 5-float env action);
  `set_map` / `reset` are **OPTIONAL** on the protocol, never required (`selfplay.py:57-71`).
- [`SelfPlayWrapper`](../../src/pop_trainer/rl/selfplay.py) — a real `gymnasium.Wrapper` presenting
  the **symmetric pure-transport** `TankEnv` as a **1-action gym env** (`selfplay.py:177`). SB3
  supplies ONLY player1's action; the wrapper drives player2 behind the scenes. The action /
  observation spaces are player1's, inherited **unchanged** from the wrapped env (a
  `gymnasium.Wrapper` forwards both by default — `selfplay.py:180-182`).
  - **The mechanism.** At `reset` it samples ONE opponent for the episode, forwards
    `set_map(info["map"])` then `reset(seed)` to it (both no-ops unless the agent exposes the hook),
    and **CACHES** player2's flipped first-person view = `split_state_for_opponent(info["state"])` —
    the **PRE-step** view a simultaneous-move opponent sees (`selfplay.py:199-215`). At
    `step(action)` the opponent `act`s on that cached pre-step view, then `env.step(action, a2)`
    transports **BOTH** actions (a1 = caller, a2 = opponent), and the p2 view is **RE-cached** from
    the new `info["state"]` (`selfplay.py:217-229`). A new opponent is sampled **ONLY at reset,
    never mid-episode** (`selfplay.py:189-190`).
  - **It mirrors the collection driver.** This is exactly `data.collect.run_episode`'s driver loop
    (`a2 = player2.act(split_state_for_opponent(np.asarray(vec)))` → `env.step(a1, a2)`,
    [`collect.py:395-397`](../../src/pop_trainer/data/collect.py)) — but packaged as a `Wrapper`
    instead of a bespoke loop. The perspective flip (`core.state.split_state_for_opponent`, swapping
    the two 26-float halves, [`state.py:178-185`](../../src/pop_trainer/core/state.py)) lives in the
    wrapper, never reaching into env internals — so the flip stays visible and assertable.

### The eval + ELO seam

The M1 evaluation seam — three modules that score a trained policy by **win-rate** and log it during
training. The metric is win-rate (fraction of greedy episodes won), broken out **per opponent** so
the breakdown shows where the policy is strong/weak across the scripted family
(`evaluate.py:1-6`).

- [`evaluate_winrate(model, eval_env, opponents=DEFAULT_ROSTER, n_episodes=100, seed=None)`](../../src/pop_trainer/rl/evaluate.py)
  — the eval driver. For EACH selector in `opponents` it wraps the RAW `eval_env` in a fresh T3
  `SelfPlayWrapper(OpponentProvider.from_roster([selector], strategy="round_robin", seed=seed))` —
  a single-entry roster, so the whole batch is pinned to ONE fixed opponent — and plays `n_episodes`
  **GREEDY** (`model.predict(obs, deterministic=True)`) episodes to completion
  (`evaluate.py:112-126`). It returns a per-selector `dict[selector, win_rate]` whose insertion
  order matches the `opponents` argument (`evaluate.py:106,111,131-132`). Re-wrapping the same raw
  env across selectors is safe — `SelfPlayWrapper` holds no destructive state on the base env
  (`evaluate.py:92-94`).
  - **The win signal is `info["outcome"]`, NOT the reward sign.** The per-episode result is read
    from `info["outcome"]` on the `terminated` step ONLY, defaulting to `DRAW`
    (`evaluate.py:122,127-130`). A truncated-without-outcome episode (time-limit / lost connection)
    therefore stays a `"draw"` and can NEVER inflate the metric (`evaluate.py:88-90`). This is tied
    to the env source of truth: [`TankEnv.step`](../../src/pop_trainer/env/tank_env.py) sets
    `info["outcome"]="win"` iff `winner == S.PLAYER_1`, `"loss"` for a decided non-P1 winner, else
    `"draw"`, and ONLY on `terminated` (a truncation is not a decided game)
    (`tank_env.py:392-400`).
  - **Pure helpers (stdlib-only, unit-testable).** [`win_rate(outcomes)`](../../src/pop_trainer/rl/evaluate.py)
    is the pure counting helper — `wins / len(outcomes)`, draws/losses in the denominator only,
    `0.0` on empty input, unknown tokens tolerated (`evaluate.py:53-72`).
    [`overall_win_rate`](../../src/pop_trainer/rl/evaluate.py) is the unweighted mean of the
    per-opponent rates (the pooled rate when every opponent gets the same episode count)
    (`evaluate.py:149-161`) — it IS wired, used by the callback (`callbacks.py:114`).
    [`format_per_map_table`](../../src/pop_trainer/rl/evaluate.py) renders a one-line breakdown
    (`evaluate.py:164-183`) but is a **forward-looking display helper not yet wired to any caller**
    (only its own definition references it). All three avoid torch/env (type-only imports,
    `evaluate.py:21-23,33-35`).
- [`EvalWinRateCallback(BaseCallback)`](../../src/pop_trainer/rl/callbacks.py) — the SB3 callback
  that runs the eval periodically during training and logs it.
  - **Rollout-boundary ONLY — never mid-rollout.** `_on_step` is a no-op returning `True`
    (`callbacks.py:73-75`); the eval fires in `_on_rollout_end`, gated by `eval_freq` AND a timestep
    delta against the last eval (`callbacks.py:77-83`). This is load-bearing: running `env.step`
    inside a rollout would poison the PPO experience buffer (`callbacks.py:9-12`). `eval_freq <= 0`
    disables eval (`callbacks.py:49,79-80`).
  - **Evals against the RAW env.** It reaches `vec_env.envs[0].unwrapped` — the bare `TankEnv`
    behind SB3's `DummyVecEnv` / `Monitor` — because `evaluate_winrate` re-wraps it in its own
    per-opponent `SelfPlayWrapper`, so it must NOT be handed the vec wrapper
    (`callbacks.py:85-89`). It logs `eval/win_rate/<selector>` per opponent plus an overall
    `eval/win_rate` into the model's existing SB3 logger, then `dump`s — so the row lands in BOTH
    `progress.csv` and TensorBoard `tfevents` at this timestep (`callbacks.py:110-115`).
  - **THE CRUX — it repairs the rollout state afterward.** Eval drives `reset`/`step` on the SHARED
    training env, leaving `model._last_obs` / `model._last_episode_starts` stale. So in a `finally`
    (which ALWAYS runs, even if eval raises mid-batch) it does `obs = vec_env.reset()` then writes
    `model._last_obs = obs` and `model._last_episode_starts = all-True` so the NEXT rollout starts
    clean (`callbacks.py:116-125`).
  - **Rotation-desync handling — by omission, not a toggle.** It suppresses map rotation during eval
    NOT via a (nonexistent) `env.set_rotation_enabled` toggle but simply by NEVER passing
    `switch_arena` on any eval or repair reset: `TankEnv` rotates ONLY when the CALLER passes
    `reset(options={"switch_arena": ...})`, `evaluate_winrate` issues plain `reset()`s, and the
    repair is a plain `vec_env.reset()` — so the documented mid-eval arena-switch desync cannot occur
    (`callbacks.py:91-101,119-123`).
- [`elo.py`](../../src/pop_trainer/rl/elo.py) — pure ELO rating math, `import math` ONLY (no
  sb3/gym/torch/numpy — unit-testable on its own, `elo.py:1-8`). `elo_prob(elo1, elo2)` is the
  logistic expected-win probability (base 10, `/400`) (`elo.py:11-13`); `elo_change(elo_a, elo_b,
  K, a_win_rate)` returns the per-side ROUNDED deltas, which (being rounded per side) need NOT sum to
  zero (`elo.py:16-28`). It is **NOT exported from `rl/__init__.py`** — reach it as
  `from pop_trainer.rl.elo import elo_prob, elo_change`. It is for the **M2 ELO ladder over the
  population** and is **not wired into the M1 train loop** yet (`elo.py:2-5`).

## Pulls from (upstream)

- [models](models.md) — `EncoderConfig` / `build_encoder` and the [`Encoder`](models.md) it wraps
  (its `embed` flat embedding + `Encoder.embedding_dim` for the features dimension), for the
  extractor seam (`extractor.py:29,61-65`).
- [core](core.md) — `core.state.split_state_for_opponent`, the frozen, involutive perspective flip
  the wrapper applies to give the opponent its first-person 52-float view (`selfplay.py:40`;
  `state.py:178-185`).
- [env](env.md) — the symmetric pure-transport `TankEnv` that `SelfPlayWrapper` wraps and drives via
  `env.reset` / `env.step(a1, a2)` (`selfplay.py:177,207,227`); the eval seam ALSO reads its
  `info["outcome"]` win/loss/draw tag as the win signal (`tank_env.py:392-400`;
  `evaluate.py:127-130`).
- [agents](agents.md) — `make_agent` + the canonical selector registry; `OpponentProvider.from_roster`
  builds each roster opponent through it (`DEFAULT_ROSTER` = the five canonical selectors)
  (`selfplay.py:39,46-52,164`).
- Plus torch / gymnasium / numpy / stable-baselines3 — the `BaseFeaturesExtractor` base class +
  `gymnasium.Wrapper` (`extractor.py:24-27`; `selfplay.py:36-37`), the callback's `BaseCallback`
  base + the model's SB3 logger / `numpy` for the repair (`callbacks.py:29-30,123-125`). The pure
  `evaluate.py` / `elo.py` helpers stay import-light: `evaluate.py` keeps sb3/gym type-only
  (`evaluate.py:33-35`) and `elo.py` imports `math` only (`elo.py:8`).

## Pushes to (downstream)

The eval seam consumes the self-play seam internally (`evaluate_winrate` re-wraps the env in a
`SelfPlayWrapper`, `evaluate.py:109,114`); otherwise these seams are inputs the not-yet-built PPO
trainer wires up:

- `EncoderExtractor` plugs into the SB3 `CnnPolicy` as
  `policy_kwargs={"features_extractor_class": EncoderExtractor}`, so the policy / value heads read
  the standardized vision embedding.
- `SelfPlayWrapper` wraps the env the trainer learns over, presenting it as a 1-action gym so SB3
  trains player1 against the sampled scripted opponent.
- `EvalWinRateCallback` plugs into the SB3 `model.learn(callback=...)` list so the (not-yet-built)
  trainer logs greedy per-opponent `eval/win_rate` at rollout boundaries.
- **Still deferred (future work):** just the **PPO train loop** and the **live self-play smoke** —
  `rl/__init__.py:22`. (The opponent provider, `SelfPlayWrapper`, the eval seam — `evaluate_winrate`
  + `EvalWinRateCallback` — and the `elo` math are **now BUILT**, no longer deferred.)

## Where it sits in the run

Three seams of the online-RL phase, all consumed by the (not-yet-built) PPO trainer:

- **Encoder seam** — where the env's pixel observation meets the SB3 policy/value heads: it lets the
  trainer reuse the SAME standardized vision backbone the supervised pretraining produces, loaded
  from a checkpoint and optionally frozen. It consumes the pixel frames [env](env.md) produces,
  through the [models](models.md) `Encoder`.
- **Self-play seam** — where the symmetric `TankEnv` is packaged as a 1-action gym: at train time
  `SelfPlayWrapper` samples a scripted opponent per episode (from the [agents](agents.md) roster via
  [`OpponentProvider`](../../src/pop_trainer/rl/selfplay.py)) and drives player2 from its flipped
  [core](core.md) state view, so SB3 only ever supplies player1's action. This is the same
  opponent-driving path the [data](data.md) collection runner uses, repackaged as a wrapper over the
  trainer.
- **Eval + ELO seam** — where a trained policy is scored: `EvalWinRateCallback` fires at rollout
  boundaries and calls `evaluate_winrate`, which RE-USES the self-play seam (one pinned opponent per
  batch) to play greedy episodes and read the win/loss/draw straight from the env's `info["outcome"]`,
  logging per-opponent + overall `eval/win_rate`. The `elo` math is the M2 ladder primitive, not yet
  wired into the M1 loop.

**Encoder seam** — pixel obs → standardized embedding:

```mermaid
graph LR
    obs["pixel obs<br/>(C, H, W) @360×640"] -->|forward| ext["EncoderExtractor<br/>(BaseFeaturesExtractor)"]
    ckpt["checkpoint<br/>(raw state_dict)"] -.->|load_state_dict| ext
    ext -->|wraps| enc["models.Encoder<br/>NatureCNN × Flatten"]
    enc -->|embed| feat["flat embedding<br/>(N, features_dim)"]
    feat -.->|features_extractor_class| policy["SB3 CnnPolicy<br/>(future rl trainer)"]
```

**Self-play seam** — `SelfPlayWrapper` drives player2 behind a 1-action gym face:

```mermaid
graph LR
    sb3["SB3 (future trainer)<br/>supplies a1 only"] -->|"step(a1)"| wrap["SelfPlayWrapper<br/>(gymnasium.Wrapper)"]
    prov["OpponentProvider<br/>round_robin / uniform"] -.->|"sample() @reset"| wrap
    agents["agents.make_agent<br/>(DEFAULT_ROSTER selectors)"] -->|from_roster| prov
    wrap -->|"act(cached p2 view)"| opp["ScriptedOpponent → a2"]
    wrap -->|"split_state_for_opponent(info['state'])"| flip["flipped p2 view (cached)"]
    flip -.->|pre-step view| opp
    wrap -->|"env.step(a1, a2)"| env["symmetric TankEnv<br/>(pure transport)"]
    env -.->|"info['state'] → re-cache"| flip
```

**Eval + ELO seam** — `EvalWinRateCallback` scores the policy by win-rate at rollout boundaries:

```mermaid
graph LR
    sb3["SB3 trainer<br/>(future)"] -->|"_on_rollout_end (gated by eval_freq)"| cb["EvalWinRateCallback<br/>(BaseCallback)"]
    cb -->|"vec_env.envs[0].unwrapped"| raw["raw TankEnv"]
    cb -->|"evaluate_winrate(model, raw_env)"| eval["evaluate_winrate<br/>(greedy, 1 pinned opp/batch)"]
    eval -.->|"wraps per selector"| sp["SelfPlayWrapper<br/>(self-play seam)"]
    eval -->|"info['outcome'] on terminated"| wr["win_rate / overall_win_rate"]
    wr -->|"record + dump"| log["SB3 logger<br/>(progress.csv + tfevents)"]
    cb -.->|"finally: vec_env.reset() → _last_obs"| repair["rollout-state repair"]
    elo["elo.py (pure math)<br/>M2 ladder — not wired yet"]
```

---
[← back to index](../README.md)
