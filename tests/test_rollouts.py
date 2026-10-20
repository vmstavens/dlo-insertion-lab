from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from learning.evaluation.rollouts import evaluate_policy, rollout_episode, write_results
from learning.policies.checkpoints import load_policy, read_manifest
from tools.create_example_policy import main as create_example


class Environment:
    action_size = 1

    def __init__(self):
        self.actions = []

    def reset(self, key):
        return self.state(0)

    def state(self, index):
        return SimpleNamespace(
            obs=np.array([index]),
            reward=1.0,
            done=index == 3,
            metrics={"distance": 3 - index, "is_success": index == 2},
            info={},
        )

    def step(self, state, action):
        self.actions.append(action.copy())
        return self.state(int(state.obs[0]) + 1)


class StatefulPolicy:
    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.history = 0

    def act(self, observation):
        self.history += 1
        return np.array([self.history + self.rng.random()])


def test_observation_does_not_change_policy_actions_or_results(tmp_path):
    env = Environment()
    policy = StatefulPolicy()
    plain = rollout_episode(
        env, policy, seed=12, max_steps=10, compile_environment=False
    )
    actions = env.actions.copy()
    env.actions.clear()
    observed = []
    recorded = rollout_episode(
        env,
        policy,
        seed=12,
        max_steps=10,
        compile_environment=False,
        observer=lambda state, step: observed.append(step),
    )
    assert plain == recorded
    np.testing.assert_array_equal(actions, env.actions)
    assert observed == [0, 1, 2, 3]
    assert recorded.success is True
    assert recorded.done and not recorded.step_limit_reached
    assert recorded.total_reward == 3
    write_results([recorded], tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["success_rate"] == 1
    assert "metric/distance" in (tmp_path / "episodes.csv").read_text()


def test_step_limit_and_invalid_actions():
    result = rollout_episode(
        Environment(), StatefulPolicy(), seed=1, max_steps=1, compile_environment=False
    )
    assert result.step_limit_reached and not result.done
    policy = StatefulPolicy()
    policy.act = lambda observation: np.array([np.nan])
    with pytest.raises(ValueError, match="finite"):
        rollout_episode(
            Environment(), policy, seed=1, max_steps=1, compile_environment=False
        )


def test_checkpoint_example_and_dimension_validation(tmp_path):
    create_example([str(tmp_path)])
    path = tmp_path / "manifest.json"
    manifest = read_manifest(path)
    assert manifest["observation_dim"] == 5
    env = SimpleNamespace(observation_size=5, action_size=1)
    policy = load_policy(path, env)
    policy.reset(seed=0)
    np.testing.assert_allclose(policy.act(np.ones(5)), [np.tanh(0.5)])
    env.action_size = 2
    with pytest.raises(ValueError, match="action_dim"):
        load_policy(path, env)


def test_inspection_never_imports_loader(tmp_path):
    create_example([str(tmp_path)])
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["loader"] = "does_not_exist:factory"
    path.write_text(json.dumps(manifest))
    assert read_manifest(path)["loader"] == "does_not_exist:factory"


def test_evaluation_exports_requested_seeds_and_closes_environment(
    tmp_path, monkeypatch
):
    env = Environment()
    closed = []
    env.close = lambda: closed.append(True)
    monkeypatch.setattr(
        "learning.evaluation.rollouts.create_environment", lambda cfg: env
    )
    evaluate_policy(
        {
            "evaluation": {
                "policy": "zero",
                "seeds": [9, 11],
                "episodes": 20,
                "max_steps": 4,
                "compile_environment": False,
            }
        },
        tmp_path,
    )
    rows = json.loads((tmp_path / "episodes.json").read_text())
    assert [row["seed"] for row in rows] == [9, 11]
    assert closed == [True]


def test_failed_policy_load_closes_environment(tmp_path, monkeypatch):
    env = Environment()
    closed = []
    env.close = lambda: closed.append(True)
    monkeypatch.setattr(
        "learning.evaluation.rollouts.create_environment", lambda cfg: env
    )
    with pytest.raises(FileNotFoundError):
        evaluate_policy(
            {
                "evaluation": {
                    "policy_manifest": str(tmp_path / "missing.json"),
                    "episodes": 1,
                    "max_steps": 4,
                }
            },
            tmp_path,
        )
    assert closed == [True]


def test_protocol_success_any_and_final_use_same_transitions():
    rule = {"metric": "is_success", "operator": ">=", "threshold": 0.5, "mode": "any"}
    any_result = rollout_episode(
        Environment(),
        StatefulPolicy(),
        seed=0,
        max_steps=5,
        compile_environment=False,
        success_rule=rule,
    )
    final_result = rollout_episode(
        Environment(),
        StatefulPolicy(),
        seed=0,
        max_steps=5,
        compile_environment=False,
        success_rule={**rule, "mode": "final"},
    )
    assert any_result.success is True
    assert final_result.success is False
    assert any_result.total_reward == final_result.total_reward
