import torch
import pytest
from src.agent import Agent
from src.nn_architecture.network_config import EnvironmentConfig
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6
from itertools import permutations
from dataclasses import dataclass
from src.topp import TOPP
from src.nn_architecture.environment_config import EnvironmentConfig
from pydantic import BaseModel
from src.agents.random_agent import RandomAgent


def test_build_env():
    topp = TOPP()

    env_config = EnvironmentConfig(env_name="tictactoe", seed=42, render_mode="rgb_array")
    env = topp.build_env(env_config)

    assert env.__class__ == tictactoe_v3.env().__class__, f"Expected env of type {tictactoe_v3.env().__class__}, got {env.__class__}"

@pytest.mark.parametrize(
        "number_of_agents",
        [
            (2),
            (3),
            (4)
        ]
)
def test_run_topp(number_of_agents):
    agent_list = []
    for _ in range(number_of_agents):
        agent = RandomAgent(9)
        agent_list.append(agent)
    topp = TOPP()
    env_config = EnvironmentConfig(env_name="tictactoe", seed=42, render_mode="rgb_array")
    results = topp.run_topp(agent_list, env_config)

    expected = len(agent_list) * (len(agent_list) - 1)
    assert len(results.results) == expected, (
        f"Expected dictionary of length {expected}, got {len(results.results)}"
)

