"""Contract tests for ``pop_trainer.core.agent`` (the Agent Protocol)."""

import pop_trainer.core as core
from pop_trainer.core import Agent, StatefulAgent
from pop_trainer.core.agent import Agent as AgentDirect


class StatelessAgent:
    """A pure agent: only ``act``. Should satisfy :class:`Agent`."""

    def __init__(self, action):
        self._action = action

    def act(self, obs):
        return self._action


class SeededAgent:
    """A stateful agent: ``act`` + ``reset``. Should satisfy both protocols."""

    def __init__(self, action):
        self._action = action
        self.last_seed = "unset"

    def act(self, obs):
        return self._action

    def reset(self, *, seed=None):
        self.last_seed = seed


class MapAwareAgent:
    """A map-aware agent: ``act`` + ``set_map``. Satisfies ``Agent`` (act present)."""

    def __init__(self, action):
        self._action = action
        self.got_map = "unset"

    def act(self, obs):
        return self._action

    def set_map(self, layout):
        self.got_map = layout


class NotAnAgent:
    """No ``act`` method — must fail the runtime-checkable ``Agent`` check."""

    def step(self, obs):
        return obs


def test_stateless_act_only_is_an_agent():
    assert isinstance(StatelessAgent([0.0, 0.0, 0.0, 0.0, 0.0]), Agent)


def test_missing_act_is_not_an_agent():
    assert not isinstance(NotAnAgent(), Agent)


def test_act_returns_the_built_action():
    action = [1.0, -1.0, 0.5, -0.5, 1.0]
    agent = StatelessAgent(action)
    result = agent.act(obs="any-observation")
    assert result == action
    assert len(result) == 5


def test_stateful_agent_satisfies_agent_and_resets():
    agent = SeededAgent([0.0, 0.0, 0.0, 0.0, 0.0])
    # A stateful agent is still an Agent (act is present).
    assert isinstance(agent, Agent)
    agent.reset(seed=7)
    assert agent.last_seed == 7


def test_stateless_agent_still_satisfies_agent_without_reset():
    # The optional reset story: an act-only agent is a valid Agent.
    stateless = StatelessAgent([0.0, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(stateless, Agent)
    assert not hasattr(stateless, "reset")


def test_stateful_agent_documents_optional_set_map():
    # The map hook lives on the static-only StatefulAgent protocol (NOT runtime-checkable Agent),
    # so adding it never breaks isinstance(x, Agent).
    assert hasattr(StatefulAgent, "set_map")


def test_map_aware_agent_satisfies_agent_and_receives_layout():
    # A map-aware agent (act + set_map) is still an Agent (act present), and set_map accepts a
    # layout — a lightweight runtime check of the optional hook.
    agent = MapAwareAgent([0.0, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(agent, Agent)
    sentinel = object()
    agent.set_map(sentinel)
    assert agent.got_map is sentinel


def test_map_agnostic_agent_need_not_implement_set_map():
    # An act-only agent is a valid Agent and is never required to expose set_map.
    stateless = StatelessAgent([0.0, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(stateless, Agent)
    assert not hasattr(stateless, "set_map")


def test_import_surface():
    assert "agent" in core.__all__
    assert "Agent" in core.__all__
    assert "StatefulAgent" in core.__all__
    # The re-exported symbol is the same class object as the module attribute.
    assert Agent is AgentDirect
    assert issubclass(StatefulAgent, Agent)
