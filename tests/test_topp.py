import torch
from src.agent import Agent
from src.nn_architecture.network_config import EnvironmentConfig
from pettingzoo.utils.env import AECEnv, connect_four_v3, tictactoe_v3, chess_v6
from itertools import permutations
from dataclasses import dataclass
from src.topp import build_env, TOPP
from src.nn_architecture.environment_config import EnvironmentConfig


def test_build_env(num_layers, input_shape, output_shape):
    topp = TOPP()

    env_config = EnvironmentConfig()
    env_config.env_name = "tictactoe"
    env_config.render_mode = None

    env = topp.build_env(env_config)

    assert env.__class__ == tictactoe_v3.env().__class__, f"Expected env of type {tictactoe_v3.env().__class__}, got {env.__class__}"
