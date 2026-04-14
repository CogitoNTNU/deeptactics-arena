from src.agent import Agent
from src.nn_architecture.network_config import EnvironmentConfig
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6
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

            env_agents = list(env.possible_agents)
            if len(env_agents) != 2:
                raise ValueError(f"TOPP currently expects 2 env agents, got {len(env_agents)}")

            policy_for_agent = {
                env_agents[0]: current_matchup[0],
                env_agents[1]: current_matchup[1],
            }

            for agent_name in env.agent_iter():
                obs_dict, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    action = None
                else:
                    observation = obs_dict["observation"]
                    legal_mask = obs_dict["action_mask"]
                    action = policy_for_agent[agent_name].act(observation, legal_mask)
                env.step(action)
            
            tournament_results.results[current_matchup] = dict(env.rewards)

        env.close()

        return tournament_results

