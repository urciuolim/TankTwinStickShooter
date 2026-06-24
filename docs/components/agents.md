# agents

The model-free **decision-makers** that fill the `player1` / `player2` slots. Every agent
implements the [`core.agent.Agent`](core.md) Protocol — `act(obs) → action`, returning the env's
5-float `[move_x, move_y, aim_x, aim_y, fire]`; stateful / seeded ones also implement
`reset(*, seed=None)`. No neural nets here — this package is model-free only.

**Boundary:** imports [`core`](core.md) only (the Agent Protocol + the state schema), plus numpy
and stdlib. No torch; nothing from `env` / `data` / `models`.

> **STUB — page intentionally brief.** The data-collection policies are being **redesigned into
> one map-aware family** right now. The detailed agent-by-agent page (the coverage policies, the
> rule policies, the self-play "act as PLAYER_1 on own view" convention) lands **post-build**,
> once the redesign is GO'd. Treat the per-class surface below as in flux; the responsibility and
> boundary above are stable.

## Pulls from (upstream)

- [core](core.md) — `agent.Agent` Protocol + the `state` schema (the accessors agents read).
- Plus numpy + stdlib.

## Pushes to (downstream)

- [env](env.md) — an agent is injected as `player2`.
- [data](data.md) — collection pairs a `player1` + `player2` agent to drive episodes.
- [demo](demo.md) — the `--player1` / `--player2` selectors build agents from here.

## Where it sits in the run

The opponents and data-collection drivers. Agents are what actually *play* — injected into the
env as player2 and driven directly as player1 — so a recorded trajectory (or a demo episode) is
the product of whichever pair of agents was chosen.

---
[← back to index](../README.md)
