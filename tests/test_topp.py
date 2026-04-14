import torch
import pytest
from itertools import permutations
from dataclasses import dataclass
from pydantic import BaseModel

from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6

from src.agents.random_agent import RandomAgent
from src.agent import Agent
from src.topp import TOPP, AgentMetrics
from src.nn_architecture.environment_config import EnvironmentConfig


def test_build_env():
    topp = TOPP()

    env_config = EnvironmentConfig(env_name="tictactoe", seed=42, render_mode="rgb_array")
    env = topp.build_env(env_config)

    assert env.__class__ == tictactoe_v3.env().__class__, f"Expected env of type {tictactoe_v3.env().__class__}, got {env.__class__}"

@pytest.mark.parametrize("number_of_agents", [2, 3, 4])
def test_run_topp(number_of_agents):
    agent_list = [RandomAgent(9) for _ in range(number_of_agents)]

    topp = TOPP()
    env_config = EnvironmentConfig(env_name="tictactoe", seed=42, render_mode="rgb_array")
    results = topp.run_topp(agent_list, env_config)

    expected_games = number_of_agents * (number_of_agents - 1)

    # Each ordered pair should produce one match row.
    assert len(results.matches) == expected_games

    # Metrics dict has one entry per policy name.
    assert len(results.results) == number_of_agents

    # In ordered permutations, each policy appears 2*(n-1) times.
    expected_games_per_policy = 2 * (number_of_agents - 1)
    for metrics in results.results.values():
        assert metrics.games == expected_games_per_policy
        assert metrics.wins + metrics.draws + metrics.losses == metrics.games

def test_draw_updates_wdl_and_score():
    outcome_a, outcome_b = TOPP._outcome_from_rewards(0.0, 0.0)
    assert outcome_a == 0.5
    assert outcome_b == 0.5

    a = AgentMetrics()
    b = AgentMetrics()

    TOPP._update_wdl(a, outcome_a)
    TOPP._update_wdl(b, outcome_b)

    assert a.draws == 1 and a.wins == 0 and a.losses == 0
    assert b.draws == 1 and b.wins == 0 and b.losses == 0
    assert a.score == 0.5
    assert b.score == 0.5

def test_elo_changes_after_non_draw_game():
    a = AgentMetrics()
    b = AgentMetrics()

    TOPP._update_elo(a, b, s_a=1.0, s_b=0.0, k=24.0)

    assert a.elo > 1000.0
    assert b.elo < 1000.0