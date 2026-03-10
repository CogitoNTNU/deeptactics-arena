from agent import Agent
from nn_architecture.network_config import EnvironmentConfig
from pettingzoo.utils.env import AECEnv, connect_four_v3, tictactoe_v3, chess_v6
from itertools import permutations
from dataclasses import dataclass, field
import torch.nn as nn

@dataclass
class TournamentResults:
        results: dict = field(default_factory=dict)

class TOPP:

    '''
    Tournament of Policies and Players (TOPP) is used to evaluate the
    performance of different agents in a tournament setting.
    It takes a list of policies and an environment configuration,
    runs a tournament, and returns the results.
    '''
    
    def __init__(self):
        super().__init__()

    def build_env(self, env_config: EnvironmentConfig)->AECEnv:
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
        return env


    def run_topp(self, policies: list[Agent], env_config: EnvironmentConfig) -> TournamentResults:
        env = self.build_env(env_config)
        matchups = permutations(policies, 2)
        tournament_results = TournamentResults()
        
        for current_matchup in matchups:
            env.reset(seed=env_config.seed)
            observation, reward, termination, truncation, info = env.last()
            while not termination or truncation:
                for player in current_matchup:
                    observation, reward, termination, truncation, info = env.last()
                    action = player.act(observation)
                    env.step(action)
                    if termination or truncation:
                        break
            tournament_results.results[current_matchup] = (env.agent_iter(), reward)
            env.close()

        return tournament_results

