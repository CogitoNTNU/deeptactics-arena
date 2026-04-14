import csv
import json
from itertools import permutations
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3, chess_v6
import wandb

from src.agent import Agent
from src.nn_architecture.environment_config import EnvironmentConfig


@dataclass
class MatchResult:
    game_id: int
    player_a: str
    player_b: str
    reward_a: float
    reward_b: float
    outcome_a: float
    outcome_b: float

@dataclass
class AgentMetrics:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score: float = 0.0
    games: int = 0
    elo: float = 1000.0

@dataclass
class TournamentResults:
        matches: list[MatchResult] = field(default_factory=list)
        results: dict[str, AgentMetrics] = field(default_factory=dict)


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
            case "connect_four":
                env = connect_four_v3.env(render_mode=env_config.render_mode)
            case "chess":
                env = chess_v6.env(render_mode=env_config.render_mode)
            case _:
                raise ValueError(f"Invalid env_name: {env_config.env_name}")
        return env


    def run_topp(
            self,
            policies: list[Agent],
            env_config: EnvironmentConfig,
            policy_names: list[str] | None = None,
            use_elo: bool = True,
            elo_k: float = 24.0
            ) -> TournamentResults:
        """
        Runs a tournament of policies and players (TOPP) and returns the results.
        Args:
            policies: A list of Agent instances to be evaluated.
            env_config: Configuration for the environment.
            policy_names: A list of names for the policies.
            use_elo: Whether to use ELO rating system.
            elo_k: The K-factor for ELO rating updates.
        Returns:
            TournamentResults: The results of the tournament.
        """
        if policy_names is None:
            policy_names = [f"policy_{i}" for i in range(len(policies))]
        if len(policy_names) != len(policies):
            raise ValueError("policy_names must have same length as policies")

        policy_name_by_obj = {id(p): n for p, n in zip(policies, policy_names)}
        tournament_results = TournamentResults()

        env = self.build_env(env_config)
        game_id = 0

        try:
            for current_matchup in permutations(policies, 2):
                env.reset(seed=env_config.seed)

                env_agents = list(env.possible_agents)
                if len(env_agents) != 2:
                    raise ValueError(f"TOPP currently expects 2 env agents, got {len(env_agents)}")

                policy_for_agent = {
                    env_agents[0]: current_matchup[0],
                    env_agents[1]: current_matchup[1],
                }

                player_a = policy_name_by_obj[id(current_matchup[0])]
                player_b = policy_name_by_obj[id(current_matchup[1])]

                tournament_results.results.setdefault(player_a, AgentMetrics())
                tournament_results.results.setdefault(player_b, AgentMetrics())

                agent_rewards = {}

                for agent_name in env.agent_iter():
                    obs_dict, reward, termination, truncation, info = env.last()
                    
                    if termination or truncation:
                        action = None
                    else:
                        observation = obs_dict["observation"]
                        legal_mask = obs_dict["action_mask"]
                        action = policy_for_agent[agent_name].act(observation, legal_mask)
                    
                    env.step(action)
                    agent_rewards[agent_name] = reward

                reward_a = float(agent_rewards[env_agents[0]])
                reward_b = float(agent_rewards[env_agents[1]])
                
                outcome_a, outcome_b = self._outcome_from_rewards(reward_a, reward_b)

                self._update_wdl(tournament_results.results[player_a], outcome_a)
                self._update_wdl(tournament_results.results[player_b], outcome_b)

                if use_elo:
                    self._update_elo(
                        tournament_results.results[player_a],
                        tournament_results.results[player_b],
                        outcome_a,
                        outcome_b,
                        k=elo_k,
                    )

                tournament_results.matches.append(
                    MatchResult(
                        game_id=game_id,
                        player_a=player_a,
                        player_b=player_b,
                        reward_a=reward_a,
                        reward_b=reward_b,
                        outcome_a=outcome_a,
                        outcome_b=outcome_b,
                    )
                )
                game_id += 1
        finally:
            env.close()

        return tournament_results
    
    @staticmethod
    def _outcome_from_rewards(r_a: float, r_b: float) -> tuple[float, float]:
        if r_a > r_b:
            return 1.0, 0.0
        if r_b > r_a:
            return 0.0, 1.0
        return 0.5, 0.5
    
    @staticmethod
    def _update_wdl(m: AgentMetrics, outcome: float) -> None:
        m.games += 1
        m.score += outcome
        if outcome == 1.0:
            m.wins += 1
        elif outcome == 0.5:
            m.draws += 1
        else:
            m.losses += 1

    @staticmethod
    def _update_elo(a: AgentMetrics, b: AgentMetrics, s_a: float, s_b: float, k: float = 24.0) -> None:
        e_a = 1.0 / (1.0 + 10 ** ((b.elo - a.elo) / 400.0))
        e_b = 1.0 - e_a
        a.elo += k * (s_a - e_a)
        b.elo += k * (s_b - e_b)

    @staticmethod
    def log_to_wandb(results: TournamentResults, out_dir: str = "output/topp") -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        match_columns = [
            "game_id", "player_a", "player_b",
            "reward_a", "reward_b", "outcome_a", "outcome_b"
        ]
        match_rows = [
            [m.game_id, m.player_a, m.player_b, m.reward_a, m.reward_b, m.outcome_a, m.outcome_b]
            for m in results.matches
        ]
        wandb.log({"topp/matches_table": wandb.Table(columns=match_columns, data=match_rows)})

        metric_columns = ["agent", "wins", "draws", "losses", "score", "games", "win_rate", "elo"]
        metric_rows = []
        for name, m in results.results.items():
            win_rate = (m.wins / m.games) if m.games > 0 else 0.0
            metric_rows.append([name, m.wins, m.draws, m.losses, m.score, m.games, win_rate, m.elo])

        wandb.log({"topp/metrics_table": wandb.Table(columns=metric_columns, data=metric_rows)})
        wandb.log({"topp/num_games": len(results.matches)})

        matches_csv = out_path / "matches.csv"
        with matches_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(match_columns)
            writer.writerows(match_rows)

        metrics_json = out_path / "metrics.json"
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump({name: asdict(m) for name, m in results.results.items()}, f, indent=2)

        artifact = wandb.Artifact("topp_results", type="evaluation")
        artifact.add_file(str(matches_csv))
        artifact.add_file(str(metrics_json))
        wandb.log_artifact(artifact)