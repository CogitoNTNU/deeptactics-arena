from src.agent import Agent
from src.nn_architecture.environment_config import EnvironmentConfig
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6
from itertools import permutations


def build_env(env_config: EnvironmentConfig)->AECEnv:
    match env_config.env_name:
        case "tictactoe":
            env = tictactoe_v3.env(render_mode=env_config.render_mode)
            env.reset(seed=42)
        case "connect_four":
            env = connect_four_v3.env(render_mode=env_config.render_mode)
        case "chess":
            env = chess_v6.env(render_mode=env_config.render_mode)
        case _:
            raise ValueError(f"Invalid env_name: {env_config.env_name}")


def run_topp(policies: list[Agent], env_config: EnvironmentConfig) -> TournamentResults:
    env = build_env(env_config)

    matchups = permutations(policies, 2)
    
    for current_matchup in matchups:
        env.reset(seed=env_config.seed)
        observation, reward, termination, truncation, info = env.last()
        while not termination or truncation:
            for player in current_matchup:
                observation, reward, termination, truncation, info = env.last()
                action = player.act(observation)
                env.step(action)
        env.close()







