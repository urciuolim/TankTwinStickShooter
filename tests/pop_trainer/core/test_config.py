"""Contract tests for ``pop_trainer.core.config`` (RewardConfig / EnvConfig / RunConfig)."""

import json

import pytest

from pop_trainer.core.config import EnvConfig, RewardConfig, RunConfig


def test_reward_defaults():
    r = RewardConfig()
    assert r.win_reward == 1.0
    assert r.loss_reward == -1.0
    assert r.time_total == -1.0
    assert r.action_total == -0.1
    assert r.action_norm == 5.0


def test_reward_round_trip_dict():
    r = RewardConfig(win_reward=2.0)
    assert RewardConfig.from_dict(r.to_dict()) == r


def test_reward_missing_keys_use_defaults():
    r = RewardConfig.from_dict({"win_reward": 3.0})
    assert r.win_reward == 3.0
    assert r.loss_reward == -1.0  # default


def test_reward_rejects_unknown_keys():
    with pytest.raises(TypeError):
        RewardConfig.from_dict({"win_reward": 1.0, "bogus": 5})


def test_reward_from_json_strict_rejects_trailing_comma():
    with pytest.raises(json.JSONDecodeError):
        RewardConfig.from_json('{"win_reward": 1.0,}')


def test_reward_from_json_rejects_non_object():
    with pytest.raises(TypeError):
        RewardConfig.from_json("[1, 2, 3]")


def test_env_defaults():
    e = EnvConfig()
    assert e.env_p == 3
    assert e.max_steps == 300
    assert e.game_ip == "127.0.0.1"
    assert e.game_port == 50000
    assert e.sock_timeout == 10.0


def test_env_round_trip_and_unknown_keys():
    e = EnvConfig(env_p=5, game_port=51000)
    assert EnvConfig.from_dict(e.to_dict()) == e
    with pytest.raises(TypeError):
        EnvConfig.from_dict({"env_p": 3, "nope": 1})


def test_run_config_nested_round_trip():
    run = RunConfig(
        run_name="exp42",
        seed=7,
        env=EnvConfig(env_p=4),
        reward=RewardConfig(action_total=-0.2),
    )
    d = run.to_dict()
    # Nested configs are expanded to dicts in the dict form.
    assert d["env"]["env_p"] == 4
    assert d["reward"]["action_total"] == -0.2
    rebuilt = RunConfig.from_dict(d)
    assert rebuilt == run
    assert isinstance(rebuilt.env, EnvConfig)
    assert isinstance(rebuilt.reward, RewardConfig)


def test_run_config_defaults_for_nested():
    run = RunConfig.from_dict({"run_name": "minimal"})
    assert run.run_name == "minimal"
    assert run.seed == 0
    assert run.env == EnvConfig()
    assert run.reward == RewardConfig()


def test_run_config_from_json_round_trip():
    run = RunConfig(run_name="json_run", seed=3)
    text = json.dumps(run.to_dict())
    assert RunConfig.from_json(text) == run


def test_run_config_rejects_unknown_top_and_nested_keys():
    with pytest.raises(TypeError):
        RunConfig.from_dict({"run_name": "x", "extra": 1})
    with pytest.raises(TypeError):
        RunConfig.from_dict({"run_name": "x", "env": {"bad_key": 1}})
