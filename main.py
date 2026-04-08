from argparse import ArgumentParser
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer
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


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def generate_training_data(
    replay_buffer: ReplayBuffer, config: Configuration, model=None
) -> tuple[ReplayBuffer, dict]:
    env = build_environment(config.env_name)
    env.reset()
    monte_carlo = MCTS(env=env, config=config, model=model, device=device)

    trajectories: list[TensorDict] = []
    mcts_entropies: list[float] = []

    while True:
        observation = monte_carlo.root.obs
        policy_values = monte_carlo.run_simulations(1000)

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

        current_agent = env.agent_selection
        env.step(action)
        _, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            # reward from the perspective of the agent who made the last move
            reward = env.rewards[current_agent]
            break

    outcome = reward
    for i, td in enumerate(reversed(trajectories)):
        td["value"] = torch.tensor(outcome, dtype=torch.float32)
        outcome = -outcome

    for td in trajectories:
        replay_buffer.add(td)

    stats = {
        "game_length": len(trajectories),
        "outcome": reward,
        "mcts_policy_entropy": sum(mcts_entropies) / len(mcts_entropies),
    }
    return replay_buffer, stats


def training_loop(config: Configuration):
    replay_buffer: ReplayBuffer = PrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        storage=LazyTensorStorage(max_size=200_000),
        batch_size=config.train.batch_size,
    )

    model = AlphaZeroNet(config.network).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.weight_decay,
    )

    for episode in range(config.train.num_episodes):
        replay_buffer, game_stats = generate_training_data(replay_buffer, config, model)

        wandb.log(
            {
                "episode": episode,
                "episode/game_length": game_stats["game_length"],
                "replay_buffer/size": len(replay_buffer),
                "self_play/outcome": game_stats["outcome"],
                "self_play/mcts_policy_entropy": game_stats["mcts_policy_entropy"],
            }
        )

        if len(replay_buffer) >= config.train.min_replay_size:
            train(replay_buffer, model, optimizer, config.train)
            record_episode(model, config.env_name, episode, device)
            eval_metrics = evaluate_vs_random(model, config.env_name, device)
            wandb.log({"episode": episode, **eval_metrics})


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
        entity="deeptactics-arena",
        project="AlphaZero deeptactics",
        config=config.model_dump(),
        # mode="disabled",  # disabled offline online
        monitor_gym=True,
    )

    # Start training loop
    training_loop(config)

    run.finish()
