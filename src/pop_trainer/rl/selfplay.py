"""The self-play opponent seam — packaging player2 as a reusable single-action gym Wrapper.

The symmetric :class:`~pop_trainer.env.tank_env.TankEnv` is PURE TRANSPORT: it owns neither
player and the CALLER supplies both actions to ``env.step(action, opponent_action)``. This module
packages the opponent (player2) side as a :class:`gymnasium.Wrapper` so SB3 sees a standard
1-agent env — the learner supplies player1's action, and the wrapper drives player2 from a
scripted agent behind the scenes. Self-play is a WRAPPER over the trainer (the ML-Agents ``ghost/``
precedent), NEVER woven into the env or the PPO core.

It mirrors the canonical opponent-driving pattern in ``data.collect.run_episode`` exactly, but as
a Wrapper instead of a bespoke loop:

* the opponent acts on player2's FLIPPED first-person 52-float state, computed via
  :func:`pop_trainer.core.state.split_state_for_opponent` (the frozen, involutive perspective flip
  — re-used here, never re-implemented);
* the wrapper caches that flipped view from ``info["state"]`` at reset (the PRE-step view) and
  RE-caches it after every step from the new ``info["state"]``;
* map / seed forwarding to the opponent reuses the OPTIONAL-hook ``getattr`` probe (a map-aware or
  stateful agent gets ``set_map`` / ``reset``; a map-agnostic / stateless one is left untouched).

The opponents are the canonical scripted agents resolved through
:func:`pop_trainer.agents.make_agent` (the SAME selector registry the collection runner uses — NOT
re-implemented here). Phase 1 is SCRIPTED only (coverage / random / noop); frozen-self opponents
are deferred.

Boundary: imports ``core`` / ``env`` / ``agents`` (+ gymnasium / numpy / stdlib). Imports NOTHING
from ``data`` or ``pretraining`` — the opponent-driving PRIMITIVES (``split_state_for_opponent``,
``make_agent``) are re-used directly, not the ``data`` module. No cycles.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import gymnasium
import numpy as np

from pop_trainer.agents import make_agent
from pop_trainer.core.state import split_state_for_opponent

__all__ = ["Opponent", "ScriptedOpponent", "OpponentProvider", "SelfPlayWrapper"]

# The default Phase-1 roster: the stationary floor, the map-agnostic baseline, and the three
# map-aware coverage presets. These are the canonical ``agents`` selectors (see the registry).
DEFAULT_ROSTER: tuple[str, ...] = (
    "noop",
    "random",
    "aggressive-coverage",
    "wall-hugger",
    "opponent-shadower",
)

_STRATEGIES = ("round_robin", "uniform")


@runtime_checkable
class Opponent(Protocol):
    """The player2 side of the self-play seam: what observation it consumes and how it acts.

    ``obs_kind`` declares which view the wrapper must feed ``act`` — Phase 1 is ``"state"`` (the
    52-float wire state, already perspective-flipped into player2's first-person frame). ``act``
    returns the env's 5-float action. ``set_map`` / ``reset`` are OPTIONAL (probed via ``getattr``,
    never required) and forwarded only when the wrapped agent exposes them.
    """

    obs_kind: str

    def act(self, obs):
        """Return player2's 5-float env action for ``obs`` (the view named by ``obs_kind``)."""
        ...


class ScriptedOpponent:
    """An :class:`Opponent` wrapping a scripted :class:`~pop_trainer.core.agent.Agent`.

    The scripted family (coverage / random / noop) is state/obs-agnostic and consumes the 52-float
    STATE, so ``obs_kind == "state"`` and the wrapper feeds it player2's flipped state. ``act``
    delegates straight to the wrapped agent; ``set_map`` / ``reset`` forward to the agent ONLY when
    it exposes the optional hook (mirroring ``data.collect``'s ``_maybe_set_map`` /
    ``_maybe_reset``), and are otherwise no-ops.
    """

    obs_kind = "state"

    def __init__(self, agent) -> None:
        self.agent = agent

    def act(self, obs):
        """Delegate to the wrapped agent's ``act`` (it already speaks the 52-float state)."""
        return self.agent.act(obs)

    def set_map(self, layout) -> None:
        """Forward the static layout to the agent IF it is map-aware; a ``None`` layout is a no-op.

        Mirrors ``data.collect._maybe_set_map``: probe the OPTIONAL ``set_map`` hook with
        ``getattr`` and call it only when present. Map-agnostic agents are left untouched.
        """
        if layout is None:
            return
        set_map = getattr(self.agent, "set_map", None)
        if callable(set_map):
            set_map(layout)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the agent IF it is stateful; a stateless agent is left untouched.

        Mirrors ``data.collect._maybe_reset``: probe the OPTIONAL ``reset`` hook with ``getattr``
        and call it as ``reset(seed=seed)`` (the keyword-only signature the agents expose).
        """
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset(seed=seed)


class OpponentProvider:
    """A roster of :class:`Opponent`s plus a per-episode sampling strategy.

    :meth:`sample` is called at EACH episode reset to pick the opponent for that episode:

    * ``"round_robin"`` cycles through the roster in order (wrapping with a modulo index);
    * ``"uniform"`` draws one uniformly at random from a SEEDED ``numpy.random.Generator`` so a
      fixed ``seed`` reproduces the sequence of picks.

    Args:
        opponents: a non-empty sequence of :class:`Opponent`s (held as a list).
        strategy: ``"round_robin"`` or ``"uniform"``; any other value raises ``ValueError``.
        seed: seeds the uniform-sampling RNG (ignored by round-robin). ``None`` is a
            nondeterministic draw.
    """

    def __init__(
        self,
        opponents: Sequence[Opponent],
        strategy: str = "round_robin",
        *,
        seed: int | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            valid = ", ".join(_STRATEGIES)
            raise ValueError(f"unknown strategy {strategy!r}; choose one of: {valid}")
        self.opponents = list(opponents)
        if not self.opponents:
            raise ValueError("OpponentProvider needs at least one opponent")
        self.strategy = strategy
        self._rng = np.random.default_rng(seed)
        self._index = 0

    @classmethod
    def from_roster(
        cls,
        roster: Sequence[str] = DEFAULT_ROSTER,
        strategy: str = "round_robin",
        *,
        seed: int | None = None,
    ) -> OpponentProvider:
        """Build a provider from a roster of ``agents`` selector strings.

        Each selector is resolved through :func:`pop_trainer.agents.make_agent` (the canonical
        registry — the SAME selectors the collection runner uses) and wrapped in a
        :class:`ScriptedOpponent`. ``seed`` threads into BOTH ``make_agent`` (so a seeded agent
        replays deterministically) and the provider's sampling RNG.
        """
        opponents = [ScriptedOpponent(make_agent(sel, seed=seed)) for sel in roster]
        return cls(opponents, strategy, seed=seed)

    def sample(self) -> Opponent:
        """Return the opponent for the NEXT episode per the configured strategy."""
        if self.strategy == "round_robin":
            opp = self.opponents[self._index % len(self.opponents)]
            self._index += 1
            return opp
        # uniform: a seeded draw from the roster.
        return self.opponents[int(self._rng.integers(len(self.opponents)))]


class SelfPlayWrapper(gymnasium.Wrapper):
    """Present a symmetric :class:`~pop_trainer.env.tank_env.TankEnv` as a 1-action gym env.

    SB3 supplies ONLY player1's action; this wrapper samples a player2 opponent per episode and
    drives it behind the scenes. The action / observation spaces are player1's, inherited unchanged
    from the wrapped env (a :class:`gymnasium.Wrapper` forwards both by default).

    The opponent acts on player2's FLIPPED first-person 52-float state. That view is computed in
    the wrapper (not by reaching into env internals) via
    :func:`pop_trainer.core.state.split_state_for_opponent` on ``info["state"]`` — exactly the
    driver path in ``data.collect.run_episode`` — so the flip stays visible and assertable. The
    flipped view is cached at reset (the PRE-step view a simultaneous-move opponent sees) and
    RE-cached after every step from the new ``info["state"]``. A NEW opponent is sampled ONLY at
    reset, never mid-episode.
    """

    def __init__(self, env, opponents: OpponentProvider) -> None:
        super().__init__(env)
        self.opponents = opponents
        self._opp: Opponent | None = None
        self._p2_obs = None

    def reset(self, *, seed=None, options=None):
        """Reset the env, sample + prime the opponent, and cache player2's flipped view.

        Mirrors ``data.collect``'s reset ordering: reset the env, then ``set_map`` then ``reset``
        the opponent (both no-ops unless the wrapped agent exposes the hook). The PRE-step player2
        view is cached from ``info["state"]`` through the perspective flip. Returns player1's
        ``(obs, info)`` unchanged.
        """
        obs_p1, info = self.env.reset(seed=seed, options=options)
        self._opp = self.opponents.sample()
        # Forward map then seed to the opponent (OPTIONAL hooks; no-op when absent / None map).
        self._opp.set_map(info.get("map"))
        self._opp.reset(seed)
        # Cache player2's PRE-step first-person view (the flip applied in the wrapper, like the
        # driver). The opponent acts on this at the first step (simultaneous move). reset ALWAYS
        # carries info["state"] (the env's post-reset priming contract), so this read is
        # unconditional — a missing key here is a real reset bug we want surfaced, not masked.
        self._p2_obs = split_state_for_opponent(np.asarray(info["state"]))
        return obs_p1, info

    def step(self, action):
        """Advance one simultaneous-move step: opponent acts on the cached p2 view, env transports.

        Player1's ``action`` comes from the caller (SB3); player2's action is the opponent acting
        on the PRE-step cached flipped view. Both go to ``env.step(action, a2)`` — the opponent
        action passes through ``TankEnv.step``'s ``opponent_action`` arg byte-unchanged (the env
        does its own coercion). The flipped p2 view is then RE-cached from the new
        ``info["state"]``. Returns the env's 5-tuple unchanged.
        """
        a2 = self._opp.act(self._p2_obs)
        obs_p1, reward, terminated, truncated, info = self.env.step(action, a2)
        # RE-cache the flipped p2 view ONLY when state is present. TankEnv.step's
        # ConnectionError/reconnect path returns info = {"lost_connection": True} with NO "state"
        # key, so an unconditional re-cache would KeyError on a transient socket drop. Keeping the
        # prior _p2_obs is safe: rewards.shaped_step_reward(lost_connection=True) -> (0.0, False,
        # True), so the step is TRUNCATED and the vec env auto-resets (reset re-primes _p2_obs);
        # in the non-done edge case, one stale opponent view is benign and the next step re-caches.
        if "state" in info:
            self._p2_obs = split_state_for_opponent(np.asarray(info["state"]))
        return obs_p1, reward, terminated, truncated, info
