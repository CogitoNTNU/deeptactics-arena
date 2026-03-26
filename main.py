from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data import LazyTensorStorage
from torch.optim import AdamW

from src.training.vetle.mcts import MCTS

from src.configuration import Configuration
from src.training.trainer import train
from src.configuration import load_config
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet
from src.utils.record import record_episode
from src.environments.environment import build_environment
from tensordict import TensorDict
import torch
import wandb


def generate_training_data(
    replay_buffer: ReplayBuffer, config: Configuration, model=None
) -> ReplayBuffer:
    env = build_environment(config.env_name)
    env.reset()
    observation, reward, terminated, truncated, info = env.last()
    monte_carlo = MCTS(env=env, config=config, model=model)

    trajectories: list[TensorDict] = []

    while True:
        policy_values = monte_carlo.run_simulations(1000)
        action = torch.multinomial(policy_values, num_samples=1).item()

        td = TensorDict(
            {
                "observation": torch.tensor(observation["observation"].copy(), dtype=torch.float32),
                "policies": policy_values,
            },
            batch_size=[],
        )
        trajectories.append(td)

        env.step(action)
        observation, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            break

    outcome = reward
    for i, td in enumerate(reversed(trajectories)):
        td["value"] = torch.tensor(outcome, dtype=torch.float32)
        outcome = -outcome

    for td in trajectories:
        replay_buffer.add(td)

    return replay_buffer


def training_loop(config: Configuration):
    replay_buffer: ReplayBuffer = PrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        storage=LazyTensorStorage(max_size=200_000),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet(config.network).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.weight_decay,
    )

    for episode in range(config.train.num_episodes):
        replay_buffer = generate_training_data(replay_buffer, config, model)

        if len(replay_buffer) >= config.train.min_replay_size:
            train(replay_buffer, model, optimizer, config.train.num_epochs)
            record_episode(config.env_name, episode)

if __name__ == "__main__":
    # Get config
    config = load_config("config.yaml")

    # Initialize wandb
    run = wandb.init(
        entity="deeptactics-arena",
        project="AlphaZero deeptactics",
        config=config.model_dump(),
        mode="disabled",  # disabled offline online
        monitor_gym=True,
    )

    # Start training loop
    training_loop(config)

    run.finish()
