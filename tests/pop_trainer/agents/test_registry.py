"""Tests for the canonical agent-selector registry (``pop_trainer.agents.registry``).

The SINGLE source of truth for ``string -> agent`` that ``data.collect_runner`` and ``demo``
import. No socket / env / data import — selectors are built and type-checked directly.
"""

import pytest

import pop_trainer.core as core
from pop_trainer import agents
from pop_trainer.agents import registry

EXPECTED_SELECTORS = {
    "aggressive-coverage",
    "wall-hugger",
    "opponent-shadower",
    "random",
    "noop",
}


def test_registry_roster_is_the_family_plus_random_and_noop():
    assert set(registry.AGENT_SELECTORS) == EXPECTED_SELECTORS
    # Re-exported off the package surface (the import callers use).
    assert set(agents.AGENT_SELECTORS) == EXPECTED_SELECTORS
    assert agents.AGENT_SELECTORS is registry.AGENT_SELECTORS


def test_registry_is_exported():
    for name in ("AGENT_SELECTORS", "make_agent"):
        assert name in agents.__all__
    assert agents.make_agent is registry.make_agent


def test_make_agent_resolves_noop_to_noop_agent():
    assert isinstance(registry.make_agent("noop"), agents.NoOpAgent)


def test_make_agent_builds_the_right_types_for_every_selector():
    assert isinstance(registry.make_agent("aggressive-coverage"), agents.CoverageAgent)
    assert isinstance(registry.make_agent("wall-hugger"), agents.CoverageAgent)
    assert isinstance(registry.make_agent("opponent-shadower"), agents.CoverageAgent)
    assert isinstance(registry.make_agent("random"), agents.RandomAgent)
    assert isinstance(registry.make_agent("noop"), agents.NoOpAgent)


def test_every_selector_builds_an_agent_that_satisfies_the_protocol():
    obs = [0.0] * core.state.STATE_LEN
    for name in registry.AGENT_SELECTORS:
        agent = registry.make_agent(name, seed=0)
        assert isinstance(agent, core.Agent)
        out = agent.act(obs)
        assert len(out) == agents.ACTION_LEN


def test_make_agent_rejects_unknown_selector_and_lists_valid_names():
    with pytest.raises(ValueError, match="unknown agent selector") as exc:
        registry.make_agent("bogus")
    # The error lists every valid selector so the message is actionable.
    msg = str(exc.value)
    for name in EXPECTED_SELECTORS:
        assert name in msg


def test_seed_is_accepted_uniformly_even_by_stateless_selectors():
    # noop ignores the seed (stateless) but the factory signature is uniform: no TypeError.
    assert isinstance(registry.make_agent("noop", seed=123), agents.NoOpAgent)
