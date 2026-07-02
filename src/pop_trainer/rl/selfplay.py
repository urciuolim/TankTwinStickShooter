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

Boundary: imports ``core`` / ``env`` / ``agents`` and the sibling ``rl.matchup`` pure-math module
(+ gymnasium / numpy / stdlib). Imports NOTHING from ``data`` or ``pretraining`` — the
opponent-driving PRIMITIVES (``split_state_for_opponent``, ``make_agent``) are re-used directly,
not the ``data`` module. No cycles.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import gymnasium
import numpy as np

from pop_trainer.agents import make_agent
from pop_trainer.core.maps import resolve_map_rotation
from pop_trainer.core.state import split_state_for_opponent
from pop_trainer.rl.matchup import matchup_cells, outcome_to_float

__all__ = [
    "Opponent",
    "ScriptedOpponent",
    "OpponentProvider",
    "MapProvider",
    "MatchupProvider",
    "SelfPlayWrapper",
]

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


class MapProvider:
    """A rotation of ARENA TARGETS plus a per-episode sampling strategy — the map sibling of
    :class:`OpponentProvider`.

    Each entry is an arena-target string (``Arenas/<name>.json`` — what Unity loads via
    ``switch_arena`` and echoes back as ``WallLayout.map_id``). :meth:`sample` is called at EACH
    episode reset to pick the arena for that episode, with the SAME two strategies the opponent
    rotation uses:

    * ``"round_robin"`` cycles through the rotation in order (wrapping with a modulo index);
    * ``"uniform"`` draws one uniformly at random from a SEEDED ``numpy.random.Generator`` so a
      fixed ``seed`` reproduces the sequence of picks.

    The provider takes an EXPLICIT, non-empty list of arena targets (not a flag value), so it stays
    pure + picklable — spawn-safe for a ``SubprocVecEnv`` worker exactly like
    :class:`OpponentProvider`. The flag-value resolution is :meth:`from_curated` (or the caller's
    own :func:`pop_trainer.core.maps.resolve_map_rotation`).

    Args:
        maps: a non-empty sequence of arena-target strings (held as a list).
        strategy: ``"round_robin"`` or ``"uniform"``; any other value raises ``ValueError``.
        seed: seeds the uniform-sampling RNG (ignored by round-robin). ``None`` is a
            nondeterministic draw.
    """

    def __init__(
        self,
        maps: Sequence[str],
        strategy: str = "round_robin",
        *,
        seed: int | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            valid = ", ".join(_STRATEGIES)
            raise ValueError(f"unknown strategy {strategy!r}; choose one of: {valid}")
        self.maps = list(maps)
        if not self.maps:
            raise ValueError("MapProvider needs at least one map")
        self.strategy = strategy
        self._rng = np.random.default_rng(seed)
        self._index = 0

    @classmethod
    def from_curated(
        cls,
        values: list[str] | None = None,
        strategy: str = "round_robin",
        *,
        seed: int | None = None,
    ) -> MapProvider:
        """Build a provider by resolving a ``--maps`` flag value through the rotation contract.

        Delegates to :func:`pop_trainer.core.maps.resolve_map_rotation` (the SINGLE source of how a
        rotation request maps to arena targets). ``values is None`` (the flag ABSENT) resolves to
        ``None`` — single-arena mode, with NO rotation — and is rejected here with the empty
        guard, because a :class:`MapProvider` only exists when there IS a rotation to play; the
        single-arena path passes ``maps=None`` to :class:`SelfPlayWrapper` instead of building one.
        """
        rotation = resolve_map_rotation(values)
        if not rotation:
            raise ValueError("MapProvider needs at least one map")
        return cls(rotation, strategy, seed=seed)

    def sample(self) -> str:
        """Return the arena target for the NEXT episode per the configured strategy."""
        if self.strategy == "round_robin":
            target = self.maps[self._index % len(self.maps)]
            self._index += 1
            return target
        # uniform: a seeded draw from the rotation.
        return self.maps[int(self._rng.integers(len(self.maps)))]


class MatchupProvider:
    """A JOINT per-episode sampler over cells = (opponent selector x arena target).

    The win-rate curriculum (``--matchup-sampling winrate``) replaces the two INDEPENDENT
    samplers' choice of matchup with ONE draw over the cross product: each :meth:`sample` picks a
    whole (opponent, arena) pair from :attr:`distribution`. Because a cell tag needs the SELECTOR
    string (a :class:`ScriptedOpponent` does not carry its selector name), the provider is built
    from explicit ``(selector, opponent)`` pairs and keeps that association.

    * :attr:`cells` — the fixed cell order from :func:`pop_trainer.rl.matchup.matchup_cells`
      (opponent-major), shared with the main-process aggregation callback so a broadcast
      distribution indexes the same cell everywhere.
    * :attr:`distribution` — the CURRENT sampling probabilities as PLAIN data (``list[float]``
      over :attr:`cells`). Initialized uniform (all win-rates start at the 0.5 prior -> equal
      deficits -> uniform). The aggregation callback replaces it by simple attribute assignment
      at rollout boundaries (via ``vec.set_attr`` on the wrapper — see
      :attr:`SelfPlayWrapper.matchup_distribution`); the seeded sampling RNG is NEVER replaced.
    * ``maps is None`` is the single-arena mode: every cell carries the ``None`` boot arena and
      the wrapper injects NO ``switch_arena`` (the reset handshake stays byte-identical;
      prioritization is over opponents only).

    Like its siblings the provider is plain-data picklable (spawn-safe for a ``SubprocVecEnv``
    worker): explicit non-empty inputs validated here, a seeded ``numpy.random.default_rng``, no
    live objects or lambdas in its state.

    Args:
        opponents: a non-empty sequence of ``(selector, opponent)`` pairs.
        maps: the arena rotation (non-empty), or ``None`` for the boot-arena mode.
        seed: seeds the cell-sampling RNG. ``None`` is a nondeterministic draw.
    """

    def __init__(
        self,
        opponents: Sequence[tuple[str, Opponent]],
        maps: Sequence[str] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        pairs = list(opponents)
        if not pairs:
            raise ValueError("MatchupProvider needs at least one opponent")
        if maps is not None and not list(maps):
            raise ValueError("MatchupProvider needs at least one map (or None for the boot arena)")
        self.maps = None if maps is None else list(maps)
        # cells + entries share one enumeration: cells[i] tags the (selector, opponent, arena)
        # entries[i] resolves. matchup_cells is the canonical order every consumer derives from.
        self.cells = matchup_cells([selector for selector, _opp in pairs], self.maps)
        arenas: tuple[str | None, ...] = (None,) if self.maps is None else tuple(self.maps)
        self._entries: list[tuple[str, Opponent, str | None]] = [
            (selector, opp, arena) for selector, opp in pairs for arena in arenas
        ]
        # The current sampling probabilities (PLAIN data — the broadcast target). All win-rates
        # start at the 0.5 prior, so the well-defined initial value is exactly uniform.
        self.distribution: list[float] = [1.0 / len(self.cells)] * len(self.cells)
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_roster(
        cls,
        roster: Sequence[str] = DEFAULT_ROSTER,
        maps: Sequence[str] | None = None,
        *,
        seed: int | None = None,
    ) -> MatchupProvider:
        """Build a provider from ``agents`` selector strings, keeping the selector association.

        Each selector resolves through :func:`pop_trainer.agents.make_agent` (the canonical
        registry, exactly like :meth:`OpponentProvider.from_roster`) into a
        :class:`ScriptedOpponent`, paired with its selector string so the sampled cell can be
        tagged. ``seed`` threads into BOTH ``make_agent`` and the cell-sampling RNG.
        """
        pairs = [(sel, ScriptedOpponent(make_agent(sel, seed=seed))) for sel in roster]
        return cls(pairs, maps, seed=seed)

    def sample(self) -> tuple[str, Opponent, str | None]:
        """Draw the NEXT episode's ``(selector, opponent, arena_target_or_None)`` cell.

        One seeded categorical draw over :attr:`cells` with the CURRENT :attr:`distribution`.
        A broadcast distribution of the wrong length is rejected here with a clear message
        (rather than a shape error inside numpy).
        """
        if len(self.distribution) != len(self.cells):
            raise ValueError(
                f"distribution length {len(self.distribution)} != cell count {len(self.cells)}"
            )
        index = int(self._rng.choice(len(self.cells), p=self.distribution))
        return self._entries[index]


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

    Map rotation (OPTIONAL): when a :class:`MapProvider` is supplied, the wrapper samples ONE arena
    target per episode at reset (alongside the opponent sample, never mid-episode) and merges it
    into the reset options as ``{"switch_arena": <target>}``, which :class:`TankEnv.reset` forwards
    to Unity. A caller-supplied ``switch_arena`` in ``options`` WINS (the provider only fills it in
    when absent). When ``maps is None`` (the default — the EVAL + backward-compat path) reset is
    BYTE-IDENTICAL to the no-rotation handshake: no options are injected and no ``switch_arena`` is
    ever sent.

    Matchup sampling (OPTIONAL): when a :class:`MatchupProvider` is supplied it is the SINGLE
    source of the episode's (opponent, arena) pair — one joint draw at reset overrides both
    ``self.opponents.sample()`` and ``self.maps`` for that episode. A cell carrying a real arena
    target merges ``switch_arena`` into the reset options exactly like the map path (caller still
    wins); a ``None`` boot-arena cell injects NOTHING (handshake byte-identical). On the step
    where the episode ends, the returned ``info`` is tagged with ``info["matchup"] =
    {"opponent": <selector>, "map": <target-or-None>, "outcome": <float>}`` — the plain-data
    sample the main-process aggregation callback folds into the per-cell win-rate EMA. With
    ``matchups is None`` (the default) NOTHING changes: no joint draw, no tag, no new info keys.
    """

    def __init__(
        self,
        env,
        opponents: OpponentProvider,
        *,
        maps: MapProvider | None = None,
        matchups: MatchupProvider | None = None,
    ) -> None:
        super().__init__(env)
        self.opponents = opponents
        self.maps = maps
        self.matchups = matchups
        self._opp: Opponent | None = None
        self._p2_obs = None
        # The active episode's cell tag (selector, arena-or-None); None when matchup sampling is
        # off, so the terminal tag is NEVER emitted on the default path.
        self._episode_cell: tuple[str, str | None] | None = None

    @property
    def matchup_distribution(self) -> list[float] | None:
        """The joint sampler's current distribution (``None`` when matchup sampling is off).

        The SETTER is the ``vec.set_attr`` broadcast target: the aggregation callback assigns the
        recomputed PLAIN-data distribution here at rollout boundaries and it lands INSIDE the
        held provider — the provider object (and its seeded RNG stream) is never replaced.
        """
        return None if self.matchups is None else self.matchups.distribution

    @matchup_distribution.setter
    def matchup_distribution(self, value: Sequence[float]) -> None:
        if self.matchups is None:
            raise ValueError(
                "matchup_distribution assigned on a SelfPlayWrapper without a MatchupProvider"
            )
        self.matchups.distribution = [float(p) for p in value]

    def reset(self, *, seed=None, options=None):
        """Reset the env, sample + prime the opponent, and cache player2's flipped view.

        Mirrors ``data.collect``'s reset ordering: reset the env, then ``set_map`` then ``reset``
        the opponent (both no-ops unless the wrapped agent exposes the hook). The PRE-step player2
        view is cached from ``info["state"]`` through the perspective flip. Returns player1's
        ``(obs, info)`` unchanged.

        When a :class:`MapProvider` is held, an arena target is sampled here (per-episode, at reset)
        and merged into ``options`` as ``{"switch_arena": <target>}`` BEFORE the env reset — a
        caller-supplied ``switch_arena`` is preserved (caller wins). With ``maps is None`` the
        options pass through untouched, so no ``switch_arena`` is sent (the eval / single-arena
        path).

        When a :class:`MatchupProvider` is held it OVERRIDES both samplers: one joint draw yields
        the episode's opponent AND arena. A ``None`` boot-arena cell injects nothing (options pass
        through untouched); a real target merges via the same caller-wins ``setdefault``. The
        drawn cell is remembered so the terminal step can tag its outcome.
        """
        self._episode_cell = None
        joint_opponent: Opponent | None = None
        if self.matchups is not None:
            # The joint draw is the SINGLE source of this episode's (opponent, arena) pair; the
            # sibling providers are not consulted. A None (boot) arena injects NO switch_arena so
            # the reset handshake stays byte-identical to the no-rotation path; a real target
            # merges with the same caller-wins setdefault the map path uses.
            selector, joint_opponent, arena = self.matchups.sample()
            self._episode_cell = (selector, arena)
            if arena is not None:
                options = dict(options or {})
                options.setdefault("switch_arena", arena)
        elif self.maps is not None:
            # Copy so we never mutate the caller's dict; fill switch_arena only if the caller did
            # not already pin one (caller wins). maps is None -> options pass through untouched, so
            # the no-rotation reset is byte-identical to before.
            options = dict(options or {})
            options.setdefault("switch_arena", self.maps.sample())
        obs_p1, info = self.env.reset(seed=seed, options=options)
        self._opp = joint_opponent if joint_opponent is not None else self.opponents.sample()
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
        ``info["state"]``. Returns the env's 5-tuple unchanged — except that with matchup
        sampling ACTIVE the episode-ending step's ``info`` gains the plain-data
        ``info["matchup"]`` cell + outcome tag (see the class docstring); with it off no key is
        ever added.
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
        if self._episode_cell is not None and (terminated or truncated):
            # Terminal tag (matchup sampling only): the episode's cell + its outcome as PLAIN
            # data. A done WITHOUT an outcome token (truncation / time-limit / lost connection)
            # maps to the 0.5 draw value — never a win — mirroring eval's convention.
            selector, arena = self._episode_cell
            info["matchup"] = {
                "opponent": selector,
                "map": arena,
                "outcome": outcome_to_float(info.get("outcome")),
            }
        return obs_p1, reward, terminated, truncated, info
