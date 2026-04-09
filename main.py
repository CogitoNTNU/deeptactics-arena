from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from torchrl.data import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
)
from torchrl.data import LazyTensorStorage
from torch.optim import AdamW

from src.training.vetle.mcts import MCTS

from src.configuration import Configuration
from src.training.trainer import train
from src.configuration import load_config
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet
from src.utils.record import record_episode, evaluate_vs_random
from src.environments.environment import build_environment
from tensordict import TensorDict
import torch
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_single_game(
    config: Configuration, model_state_dict: dict
) -> tuple[list[TensorDict], dict]:
    """Play a single self-play game and return trajectories + stats."""
    # Each worker builds its own model on CPU to avoid GPU contention
    model = AlphaZeroNet(config.network)
    model.load_state_dict(model_state_dict)
    model.eval()

    env = build_environment(config.env_name)
    env.reset()

    trajectories: list[TensorDict] = []
    mcts_entropies: list[float] = []

    with torch.no_grad():
        monte_carlo = MCTS(env=env, config=config, model=model, device="cpu")
        move_number = 0

        while True:
            observation = monte_carlo.root.obs
            policy_values = monte_carlo.run_simulations(config.mcts.num_simulations, move_number)

            entropy = -(policy_values * (policy_values + 1e-8).log()).sum().item()
            mcts_entropies.append(entropy)

            action = torch.multinomial(policy_values, num_samples=1).item()

            td = TensorDict(
                {
                    "observation": torch.tensor(
                        observation["observation"].copy(), dtype=torch.float32
                    ),
                    "action_mask": torch.tensor(
                        observation["action_mask"].copy(), dtype=torch.bool
                    ),
                    "policies": policy_values,
                },
                batch_size=[],
            )
            trajectories.append(td)

            monte_carlo.root = monte_carlo.root.children[action]
            monte_carlo.root.parent = None
            monte_carlo.root.pred_pol = monte_carlo.dirichlet(
                monte_carlo.root.pred_pol,
                monte_carlo.root.legal_actions,
                monte_carlo.config.mcts.epsilon,
            )

            move_number += 1
            current_agent = env.agent_selection
            env.step(action)
            _, _, terminated, truncated, _ = env.last()
            if terminated or truncated:
                reward = env.rewards[current_agent]
                break

    outcome = reward
    for i, td in enumerate(reversed(trajectories)):
        td["value"] = torch.tensor(outcome, dtype=torch.float32)
        outcome = -outcome

    exp_moves = min(len(trajectories), config.mcts.exploration_moves)
    stats = {
        "game_length": len(trajectories),
        "outcome": reward,
        "mcts_policy_entropy": sum(mcts_entropies) / len(mcts_entropies),
        "exploration_ratio": exp_moves / len(trajectories),
    }
    return trajectories, stats


def generate_games(
    config: Configuration, model: AlphaZeroNet, num_games: int
) -> tuple[list[list[TensorDict]], list[dict]]:
    """Generate multiple self-play games, in parallel if num_games > 1."""
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if num_games == 1:
        trajectories, stats = play_single_game(config, model_state_dict)
        return [trajectories], [stats]

    all_trajectories = []
    all_stats = []
    with ProcessPoolExecutor(max_workers=num_games) as executor:
        futures = [
            executor.submit(play_single_game, config, model_state_dict)
            for _ in range(num_games)
        ]
        for future in futures:
            trajectories, stats = future.result()
            all_trajectories.append(trajectories)
            all_stats.append(stats)

    return all_trajectories, all_stats


def training_loop(config: Configuration):
    replay_buffer: TensorDictPrioritizedReplayBuffer = (
        TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(max_size=config.train.max_replay_size),
            batch_size=config.train.batch_size,
        )
    )

    model = AlphaZeroNet(config.network).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_win_rate = -1.0
    num_parallel = config.train.num_parallel_games
    games_played = 0
    epoch_offset = 0

    for iteration in range(0, config.train.num_episodes, num_parallel):
        batch_size = min(num_parallel, config.train.num_episodes - iteration)
        all_trajectories, all_stats = generate_games(config, model, batch_size)

        for trajectories in all_trajectories:
            for td in trajectories:
                replay_buffer.add(td)

        for i, game_stats in enumerate(all_stats):
            games_played += 1
            wandb.log(
                {
                    "episode": games_played,
                    "episode/game_length": game_stats["game_length"],
                    "replay_buffer/size": len(replay_buffer),
                    "self_play/outcome": game_stats["outcome"],
                    "self_play/mcts_policy_entropy": game_stats["mcts_policy_entropy"],
                    "self_play/temperature": config.mcts.pi_temp,
                    "self_play/exploration_moves": config.mcts.exploration_moves,
                    "self_play/exploration_ratio": game_stats["exploration_ratio"],
                }
            )

        if len(replay_buffer) >= config.train.min_replay_size:
            epoch_offset = train(replay_buffer, model, optimizer, config.train, epoch_offset)
            record_episode(model, config.env_name, games_played, device)
            eval_metrics = evaluate_vs_random(model, config.env_name, device)
            wandb.log({"episode": games_played, **eval_metrics})

            win_rate = eval_metrics["eval/win_rate_vs_random"]
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), "models/best_model.pt")
                print(f"New best model saved (win rate: {win_rate:.0%})")


if __name__ == "__main__":
    # Get config
    parser = ArgumentParser()
    config_name = "config.yaml"
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=f"Config file to load (e.g. {config_name})",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help=f"Config file to load (e.g. {config_name})",
    )
    args = parser.parse_args()
    config_name = args.config_flag or args.config or config_name
    config = load_config(config_name)
    # Initialize wandb
    run = wandb.init(
        entity="iluddeludde",
        project="AlphaZero deeptactics",
        config=config.model_dump(),
        # mode="disabled",  # disabled offline online
        monitor_gym=True,
    )

    # Define x-axes: episode metrics use episode, training metrics use epoch
    wandb.define_metric("episode")
    wandb.define_metric("epoch")
    wandb.define_metric("episode/*", step_metric="episode")
    wandb.define_metric("self_play/*", step_metric="episode")
    wandb.define_metric("replay_buffer/*", step_metric="episode")
    wandb.define_metric("eval/*", step_metric="episode")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("batch/*", step_metric="epoch")

    # Start training loop
    training_loop(config)

    run.finish()
